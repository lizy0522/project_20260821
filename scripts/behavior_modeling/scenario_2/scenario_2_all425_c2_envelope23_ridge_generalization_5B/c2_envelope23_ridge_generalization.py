"""C2-only Ridge tuning and C-to-B evaluation for frozen Envelope23-ilcEnd.

This module deliberately exposes two separate paths.  The C-only path builds
only C-segment design matrices and can be used before lambda is frozen.  The
final path is guarded and opens the B evaluator only after the global lambda
has been frozen.  No fingerprint, LUT, clustering, or DPD-replay code is
imported here.
"""

# ruff: noqa: E402,E501

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# These must be set before NumPy/SciPy imports in spawned workers.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np
import pandas as pd

MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from core.shared.metrics import nmse  # noqa: E402
from data_management.shared import build_state_table, load_by_id  # noqa: E402
from signal_segmentation.shared import (  # noqa: E402
    build_partition_from_xin,
    get_ilc_pair,
    preprocess_full_pair,
)

from behavior_modeling.shared.basis_function_selection.all425_ilcend_envelope23_evaluation import (  # noqa: E402
    verify_frozen_ilcend_support as _verify_frozen_ilcend_support,
)
from behavior_modeling.shared.basis_function_selection.envelope_dictionary import (  # noqa: E402
    DMAX,
    EnvelopeBasis,
    build_envelope_bank,
    build_envelope_dictionary,
)
from behavior_modeling.shared.basis_function_selection.hard20_envelope75_selection import (  # noqa: E402
    _raw_manifest,
)
from behavior_modeling.shared.basis_function_selection.model_solver import (  # noqa: E402
    fit_ols,
    fit_ridge,
)

TASK_NAME = "scenario_2_all425_c2_envelope23_ridge_generalization_5B"
RESULT_ROOT = PROJECT_ROOT / "results" / "behavior_modeling" / "scenario_2" /TASK_NAME
WORK_LOG = PROJECT_ROOT / "work_logs" / "behavior_modeling" / "scenario_2" /TASK_NAME / "execution_log.txt"
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"

STATE_COUNT = 425
FULL_LENGTH = 24_576
C2_COLUMN_INDEX = 1
C2_STAGE_NUMBER = C2_COLUMN_INDEX + 1
A_START = 0
B_START = 12_288
C_START = 17_203
B_RAW_LENGTH = 4_915
C_RAW_LENGTH = 7_373
B_VALID_LENGTH = B_RAW_LENGTH - DMAX
C_VALID_LENGTH = C_RAW_LENGTH - DMAX
CV_BLOCK_COUNT = 3
CV_BLOCK_RAW_LENGTHS = (2_458, 2_458, 2_457)
CV_BLOCK_VALID_LENGTHS = (2_456, 2_456, 2_455)
WORKER_COUNT = 10
BLAS_THREADS_PER_WORKER = 1
THRESHOLD_DB = -40.0

# This is the task-specific recommended grid.  It contains the exact OLS
# reference and spans the requested small-to-moderate raw-basis Ridge range.
RIDGE_LAMBDA_GRID = (
    0.0,
    1e-10,
    3e-10,
    1e-9,
    3e-9,
    1e-8,
    3e-8,
    1e-7,
    3e-7,
    1e-6,
    3e-6,
    1e-5,
)

EXPECTED_SUPPORT_IDS = (
    "LIN_d0",
    "LIN_d1",
    "LIN_d2",
    "ENV_p02_m0_q0",
    "ENV_p02_m0_q1",
    "ENV_p02_m0_q2",
    "ENV_p02_m1_q0",
    "ENV_p02_m1_q1",
    "ENV_p02_m2_q0",
    "ENV_p03_m0_q0",
    "ENV_p03_m0_q1",
    "ENV_p03_m1_q0",
    "ENV_p03_m1_q1",
    "ENV_p03_m2_q1",
    "ENV_p04_m0_q0",
    "ENV_p04_m0_q1",
    "ENV_p04_m1_q0",
    "ENV_p05_m0_q0",
    "ENV_p06_m0_q0",
    "ENV_p07_m0_q0",
    "ENV_p08_m0_q0",
    "ENV_p08_m0_q1",
    "ENV_p09_m0_q0",
)
EXPECTED_SUPPORT_HASH = "7c8348beaaa9db183c26bb8adfd29f226c11f36bf54ad587e9ac0baa6f14ea14"
EXPECTED_RAW_MANIFEST = {
    "sha256": "910b06e4be6b60b1f4dc63409a3083eb328765347fbec3b89a97a41d6633a1e0",
    "file_count": 429,
    "mat_count": 427,
    "bytes": 2_258_448_137,
}

PREVIOUS_C2_OLS_METRICS = (
    PROJECT_ROOT
    / "results"
    / "scenario_2_envelope23_ilcend_commonB_full_lut_retrieval_5B"
    / "03_envelope23_all425_retrieval_metrics.csv"
)

_WORKER_TERMS: tuple[EnvelopeBasis, ...] | None = None
_WORKER_SUPPORT: tuple[int, ...] | None = None
_WORKER_STATE_TABLE: dict[int, dict[str, int | float]] | None = None


@dataclass(frozen=True)
class PreparedC2State:
    """C2 canonical data and C-only design matrices for one state."""

    state_id: int
    state_info: dict[str, int | float]
    c2_column_index: int
    valid_ilc_count: int
    full_phi: np.ndarray
    full_target: np.ndarray
    block_phis: tuple[np.ndarray, np.ndarray, np.ndarray]
    block_targets: tuple[np.ndarray, np.ndarray, np.ndarray]
    fold_train_phis: tuple[np.ndarray, np.ndarray, np.ndarray]
    fold_train_targets: tuple[np.ndarray, np.ndarray, np.ndarray]
    full_rank: int
    full_condition: float
    fold_train_ranks: tuple[int, int, int]
    fold_train_conditions: tuple[float, float, float]
    b_input: np.ndarray | None
    b_target: np.ndarray | None


def support_digest(support_ids: Sequence[str]) -> str:
    """Hash support IDs in their frozen order."""

    return hashlib.sha256(
        json.dumps(list(support_ids), ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def raw_manifest_gate() -> dict[str, object]:
    """Require the immutable raw-data manifest used by the prior tasks."""

    value = _raw_manifest()
    if value != EXPECTED_RAW_MANIFEST:
        raise RuntimeError(f"raw manifest differs from frozen baseline: {value}")
    return value


def verify_frozen_support() -> tuple[tuple[EnvelopeBasis, ...], tuple[int, ...], str]:
    """Load the formal K=23 definition and verify the exact requested support."""

    terms, support, digest = _verify_frozen_ilcend_support()
    support_ids = tuple(terms[index].basis_id for index in support)
    if len(support) != 23 or support_ids != EXPECTED_SUPPORT_IDS:
        raise RuntimeError(
            "Frozen Envelope23-ilcEnd support differs from the formal requested support"
        )
    if digest != EXPECTED_SUPPORT_HASH or support_digest(support_ids) != EXPECTED_SUPPORT_HASH:
        raise RuntimeError(f"Frozen Envelope23-ilcEnd support hash mismatch: {digest}")
    return terms, support, digest


def build_c_block_slices() -> tuple[slice, slice, slice]:
    """Return the three contiguous raw C blocks with local history ownership."""

    blocks = np.array_split(np.arange(C_RAW_LENGTH), CV_BLOCK_COUNT)
    slices: list[slice] = []
    for block in blocks:
        if block.size == 0 or not np.array_equal(block, np.arange(block[0], block[-1] + 1)):
            raise RuntimeError("C CV block is not contiguous")
        slices.append(slice(int(block[0]), int(block[-1]) + 1))
    actual_raw = tuple(item.stop - item.start for item in slices)
    actual_valid = tuple(length - DMAX for length in actual_raw)
    if actual_raw != CV_BLOCK_RAW_LENGTHS or actual_valid != CV_BLOCK_VALID_LENGTHS:
        raise RuntimeError(f"C CV block lengths changed: raw={actual_raw}, valid={actual_valid}")
    return slices[0], slices[1], slices[2]


def _as_full_vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1 or array.size != FULL_LENGTH or not np.iscomplexobj(array):
        raise ValueError(f"{name} must be a complex vector of length {FULL_LENGTH}, got {array.shape}")
    array = np.asarray(array, dtype=np.complex128)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def _matrix_health(matrix: np.ndarray) -> tuple[int, float]:
    values = np.asarray(matrix, dtype=np.complex128)
    singular = np.linalg.svd(values, compute_uv=False)
    rank = int(np.linalg.matrix_rank(values))
    condition = (
        float(singular[0] / singular[-1])
        if singular.size and singular[-1] > 0.0
        else float("inf")
    )
    return rank, condition


def _resolve_worker_config(
    terms: Sequence[EnvelopeBasis] | None,
    support: Sequence[int] | None,
) -> tuple[tuple[EnvelopeBasis, ...], tuple[int, ...]]:
    if terms is None:
        if _WORKER_TERMS is None:
            raise RuntimeError("Envelope dictionary is not initialized")
        resolved_terms = _WORKER_TERMS
    else:
        resolved_terms = tuple(terms)
    if support is None:
        if _WORKER_SUPPORT is None:
            raise RuntimeError("Frozen support is not initialized")
        resolved_support = _WORKER_SUPPORT
    else:
        resolved_support = tuple(int(index) for index in support)
    if len(resolved_support) != 23 or len(set(resolved_support)) != 23:
        raise RuntimeError("Envelope23 support must contain 23 unique indices")
    return resolved_terms, resolved_support


def prepare_c2_state(
    state_id: int,
    *,
    terms: Sequence[EnvelopeBasis] | None = None,
    support: Sequence[int] | None = None,
    include_b: bool = False,
) -> PreparedC2State:
    """Prepare one C2 state; B arrays are omitted unless explicitly unlocked later."""

    resolved_terms, resolved_support = _resolve_worker_config(terms, support)
    normalized_id = int(state_id)
    if not 0 <= normalized_id < STATE_COUNT:
        raise IndexError(f"state_id must be in [0, {STATE_COUNT - 1}]")
    data = load_by_id(normalized_id)
    xin = _as_full_vector(data["xin"], "xin")
    input_history = np.asarray(data["xin_pd_ori_ilc"])
    output_history = np.asarray(data["yout_withdpd_ori_ilc"])
    if (
        input_history.ndim != 2
        or output_history.ndim != 2
        or input_history.shape != output_history.shape
        or input_history.shape[0] != FULL_LENGTH
        or input_history.shape[1] < C2_STAGE_NUMBER
        or not np.iscomplexobj(input_history)
        or not np.iscomplexobj(output_history)
    ):
        raise RuntimeError(f"state {normalized_id} C2 ILC history is invalid")
    if not np.all(np.isfinite(input_history)) or not np.all(np.isfinite(output_history)):
        raise RuntimeError(f"state {normalized_id} C2 ILC history contains NaN/Inf")
    partition = build_partition_from_xin(xin)
    if (partition.a_slice.start, partition.a_slice.stop) != (A_START, B_START):
        raise RuntimeError("A boundary changed")
    if (partition.b_slice.start, partition.b_slice.stop) != (B_START, C_START):
        raise RuntimeError("B boundary changed")
    if (partition.c_slice.start, partition.c_slice.stop) != (C_START, FULL_LENGTH):
        raise RuntimeError("C boundary changed")
    pair = get_ilc_pair(data, C2_COLUMN_INDEX)
    if pair.iteration_index != C2_COLUMN_INDEX:
        raise RuntimeError(f"state {normalized_id} did not select canonical C2")
    canonical = preprocess_full_pair(
        pair.input_full,
        pair.output_raw_full,
        partition,
        pair_type=pair.pair_type,
        iteration_index=pair.iteration_index,
        input_peak_normalization_factor=pair.input_peak_normalization_factor,
    )
    c_segment = canonical["C"]
    b_segment = canonical["B"]
    if c_segment.input.size != C_RAW_LENGTH or b_segment.input.size != B_RAW_LENGTH:
        raise RuntimeError(
            f"state {normalized_id} segment lengths changed: C={c_segment.input.size}, B={b_segment.input.size}"
        )

    c_bank_full = build_envelope_bank(c_segment.input, resolved_terms)
    full_phi = np.asarray(c_bank_full[:, resolved_support], dtype=np.complex128)
    full_target = np.asarray(c_segment.output[DMAX:], dtype=np.complex128)
    if full_phi.shape != (C_VALID_LENGTH, 23) or full_target.shape != (C_VALID_LENGTH,):
        raise RuntimeError(f"state {normalized_id} full C design shape is invalid")

    block_phis: list[np.ndarray] = []
    block_targets: list[np.ndarray] = []
    for block in build_c_block_slices():
        local_bank = build_envelope_bank(c_segment.input[block], resolved_terms)
        local_phi = np.asarray(local_bank[:, resolved_support], dtype=np.complex128)
        local_target = np.asarray(c_segment.output[block][DMAX:], dtype=np.complex128)
        block_phis.append(local_phi)
        block_targets.append(local_target)
    if tuple(phi.shape[0] for phi in block_phis) != CV_BLOCK_VALID_LENGTHS:
        raise RuntimeError(f"state {normalized_id} C block design lengths are invalid")

    fold_train_phis = tuple(
        np.concatenate([block_phis[index] for index in range(CV_BLOCK_COUNT) if index != validation], axis=0)
        for validation in range(CV_BLOCK_COUNT)
    )
    fold_train_targets = tuple(
        np.concatenate(
            [block_targets[index] for index in range(CV_BLOCK_COUNT) if index != validation],
            axis=0,
        )
        for validation in range(CV_BLOCK_COUNT)
    )
    full_rank, full_condition = _matrix_health(full_phi)
    fold_health = tuple(_matrix_health(matrix) for matrix in fold_train_phis)
    state_table = _WORKER_STATE_TABLE
    if state_table is None:
        state_table = {int(row["state_id"]): row for row in build_state_table()}
    info = state_table[normalized_id]
    b_input = None
    b_target = None
    if include_b:
        b_input = np.asarray(b_segment.input, dtype=np.complex128)
        b_target = np.asarray(b_segment.output[DMAX:], dtype=np.complex128)
        if b_input.shape != (B_RAW_LENGTH,) or b_target.shape != (B_VALID_LENGTH,):
            raise RuntimeError(f"state {normalized_id} B data shape is invalid")
    return PreparedC2State(
        state_id=normalized_id,
        state_info={
            "funMng": int(info["funMng"]),
            "funAng": int(info["funAng"]),
            "secMng": int(info["secMng"]),
            "secAng": int(info["secAng"]),
        },
        c2_column_index=C2_COLUMN_INDEX,
        valid_ilc_count=int(input_history.shape[1]),
        full_phi=full_phi,
        full_target=full_target,
        block_phis=(block_phis[0], block_phis[1], block_phis[2]),
        block_targets=(block_targets[0], block_targets[1], block_targets[2]),
        fold_train_phis=(fold_train_phis[0], fold_train_phis[1], fold_train_phis[2]),
        fold_train_targets=(fold_train_targets[0], fold_train_targets[1], fold_train_targets[2]),
        full_rank=full_rank,
        full_condition=full_condition,
        fold_train_ranks=tuple(item[0] for item in fold_health),
        fold_train_conditions=tuple(item[1] for item in fold_health),
        b_input=b_input,
        b_target=b_target,
    )


def _fit(phi: np.ndarray, target: np.ndarray, ridge_lambda: float):
    if float(ridge_lambda) == 0.0:
        return fit_ols(phi, target, scale_columns=True)
    return fit_ridge(phi, target, float(ridge_lambda))


def evaluate_c_only_lambda(prepared: PreparedC2State, ridge_lambda: float) -> dict[str, object]:
    """Evaluate one lambda using full-C diagnostics and strictly C-only CV."""

    normalized_lambda = float(ridge_lambda)
    if not np.isfinite(normalized_lambda) or normalized_lambda < 0.0:
        raise ValueError("ridge_lambda must be finite and nonnegative")
    full_fit = _fit(prepared.full_phi, prepared.full_target, normalized_lambda)
    cv_values: list[float] = []
    ranks = [int(full_fit.rank), *prepared.fold_train_ranks]
    conditions = [prepared.full_condition, *prepared.fold_train_conditions]
    finite = bool(
        np.all(np.isfinite(full_fit.theta))
        and np.all(np.isfinite(full_fit.prediction))
        and np.isfinite(full_fit.nmse_db)
    )
    for validation in range(CV_BLOCK_COUNT):
        fit = _fit(
            prepared.fold_train_phis[validation],
            prepared.fold_train_targets[validation],
            normalized_lambda,
        )
        prediction = prepared.block_phis[validation] @ fit.theta
        value = float(nmse(prepared.block_targets[validation], prediction))
        cv_values.append(value)
        finite = bool(
            finite
            and np.all(np.isfinite(fit.theta))
            and np.all(np.isfinite(prediction))
            and np.isfinite(value)
        )
    cv_w = float(max(cv_values))
    return {
        "State_ID": prepared.state_id,
        "lambda": normalized_lambda,
        "C_train_NMSE_dB": float(full_fit.nmse_db),
        "CV1_NMSE_dB": cv_values[0],
        "CV2_NMSE_dB": cv_values[1],
        "CV3_NMSE_dB": cv_values[2],
        "CV_W_NMSE_dB": cv_w,
        "CV_pass": bool(finite and cv_w < THRESHOLD_DB),
        "theta_norm": float(np.linalg.norm(full_fit.theta)),
        "rank": int(min(ranks)),
        "condition": float(max(conditions)),
        "finite": finite,
    }


def preflight_row(prepared: PreparedC2State) -> dict[str, object]:
    """Flatten the fixed C2 and C/CV length contract for one state."""

    finite = bool(
        np.all(np.isfinite(prepared.full_phi))
        and np.all(np.isfinite(prepared.full_target))
        and all(np.all(np.isfinite(value)) for value in prepared.block_phis)
        and all(np.all(np.isfinite(value)) for value in prepared.block_targets)
        and np.isfinite(prepared.full_condition)
        and all(np.isfinite(value) for value in prepared.fold_train_conditions)
    )
    return {
        "State_ID": prepared.state_id,
        "C2_column_index": prepared.c2_column_index,
        "C2_available": True,
        "valid_ilc_count": prepared.valid_ilc_count,
        "full_length": FULL_LENGTH,
        "B_raw_length": B_RAW_LENGTH,
        "B_valid_length": B_VALID_LENGTH,
        "C_raw_length": C_RAW_LENGTH,
        "C_valid_length": C_VALID_LENGTH,
        "C1_raw_length": CV_BLOCK_RAW_LENGTHS[0],
        "C1_valid_length": CV_BLOCK_VALID_LENGTHS[0],
        "C2block_raw_length": CV_BLOCK_RAW_LENGTHS[1],
        "C2block_valid_length": CV_BLOCK_VALID_LENGTHS[1],
        "C3_raw_length": CV_BLOCK_RAW_LENGTHS[2],
        "C3_valid_length": CV_BLOCK_VALID_LENGTHS[2],
        "finite": finite,
        "OLS_rank": prepared.full_rank,
        "OLS_condition": prepared.full_condition,
    }


def assert_b_evaluation_allowed(*, lambda_frozen: bool, b_evaluation_unlocked: bool) -> None:
    """Hard guard for the external B evaluator."""

    if not lambda_frozen:
        raise RuntimeError("B evaluation is forbidden before Ridge lambda is frozen.")
    if not b_evaluation_unlocked:
        raise RuntimeError("B evaluation is locked until the lambda-frozen phase.")


def evaluate_final_state(
    state_id: int,
    ridge_lambda: float,
    *,
    terms: Sequence[EnvelopeBasis] | None = None,
    support: Sequence[int] | None = None,
    lambda_frozen: bool,
    b_evaluation_unlocked: bool,
) -> dict[str, object]:
    """Fit final C OLS/Ridge and evaluate both on the independent B segment."""

    assert_b_evaluation_allowed(
        lambda_frozen=lambda_frozen,
        b_evaluation_unlocked=b_evaluation_unlocked,
    )
    resolved_terms, resolved_support = _resolve_worker_config(terms, support)
    prepared = prepare_c2_state(
        state_id,
        terms=resolved_terms,
        support=resolved_support,
        include_b=True,
    )
    if prepared.b_input is None or prepared.b_target is None:
        raise RuntimeError("B evaluator opened without B data")
    b_bank = build_envelope_bank(prepared.b_input, resolved_terms)
    b_phi = np.asarray(b_bank[:, resolved_support], dtype=np.complex128)
    if b_phi.shape != (B_VALID_LENGTH, 23):
        raise RuntimeError(f"state {state_id} B design shape is invalid: {b_phi.shape}")
    ols_fit = _fit(prepared.full_phi, prepared.full_target, 0.0)
    ridge_fit = _fit(prepared.full_phi, prepared.full_target, float(ridge_lambda))
    ols_b = float(nmse(prepared.b_target, b_phi @ ols_fit.theta))
    ridge_b = float(nmse(prepared.b_target, b_phi @ ridge_fit.theta))
    if not np.all(np.isfinite((ols_fit.nmse_db, ols_b, ridge_fit.nmse_db, ridge_b))):
        raise RuntimeError(f"state {state_id} final OLS/Ridge metric is non-finite")
    return {
        "State_ID": prepared.state_id,
        **prepared.state_info,
        "C2_column_index": C2_COLUMN_INDEX,
        "C_train_OLS_NMSE_dB": float(ols_fit.nmse_db),
        "C_to_B_OLS_NMSE_dB": ols_b,
        "C_train_Ridge_NMSE_dB": float(ridge_fit.nmse_db),
        "C_to_B_Ridge_NMSE_dB": ridge_b,
        "Delta_C_train_dB": float(ridge_fit.nmse_db - ols_fit.nmse_db),
        "Delta_C_to_B_dB": float(ridge_b - ols_b),
        "OLS_C_train_pass": bool(ols_fit.nmse_db < THRESHOLD_DB),
        "OLS_C_to_B_pass": bool(ols_b < THRESHOLD_DB),
        "Ridge_C_train_pass": bool(ridge_fit.nmse_db < THRESHOLD_DB),
        "Ridge_C_to_B_pass": bool(ridge_b < THRESHOLD_DB),
        "generalization_gap_OLS_dB": float(ols_b - ols_fit.nmse_db),
        "generalization_gap_Ridge_dB": float(ridge_b - ridge_fit.nmse_db),
        "rank_OLS": int(ols_fit.rank),
        "condition_OLS": float(prepared.full_condition),
        "rank_Ridge": int(ridge_fit.rank),
        "condition_Ridge": float(prepared.full_condition),
        "finite": True,
        "theta_ridge": np.asarray(ridge_fit.theta, dtype=np.complex128),
    }


def worker_init(support_indices: tuple[int, ...]) -> None:
    """Initialize only the frozen dictionary/support and state metadata."""

    global _WORKER_TERMS, _WORKER_SUPPORT, _WORKER_STATE_TABLE
    _WORKER_TERMS = tuple(build_envelope_dictionary())
    _WORKER_SUPPORT = tuple(int(index) for index in support_indices)
    _WORKER_STATE_TABLE = {int(row["state_id"]): row for row in build_state_table()}


def c_only_worker(state_id: int, lambda_grid: tuple[float, ...]) -> dict[str, object]:
    """One state job for the pre-freeze C-only lambda scan."""

    prepared = prepare_c2_state(int(state_id), include_b=False)
    return {
        "preflight": preflight_row(prepared),
        "cv_rows": [evaluate_c_only_lambda(prepared, value) for value in lambda_grid],
    }


def final_worker(state_id: int, ridge_lambda: float) -> dict[str, object]:
    """One post-freeze state job that is allowed to evaluate B."""

    return evaluate_final_state(
        int(state_id),
        float(ridge_lambda),
        lambda_frozen=True,
        b_evaluation_unlocked=True,
    )


def load_previous_ols_reference() -> pd.DataFrame:
    """Load only the prior per-state C2 OLS metrics used for regression."""

    if not PREVIOUS_C2_OLS_METRICS.is_file():
        raise FileNotFoundError(f"Previous C2 OLS reference is missing: {PREVIOUS_C2_OLS_METRICS}")
    frame = pd.read_csv(PREVIOUS_C2_OLS_METRICS).sort_values("State_n_R").reset_index(drop=True)
    required = {"State_n_R", "Y_C2_train_NMSE_dB", "Y_C2_B_NMSE_dB"}
    if len(frame) != STATE_COUNT or not required.issubset(frame.columns):
        raise RuntimeError("Previous C2 OLS reference is not the canonical 425-state result")
    if frame["State_n_R"].tolist() != list(range(STATE_COUNT)):
        raise RuntimeError("Previous C2 OLS reference state order is not canonical")
    return frame


def summary_stats(values: Sequence[float]) -> dict[str, float | int]:
    """Return the standard dB summary for a finite metric vector."""

    array = np.asarray(values, dtype=float)
    if array.size != STATE_COUNT or not np.all(np.isfinite(array)):
        raise ValueError("summary_stats requires 425 finite values")
    return {
        "pass_count": int(np.count_nonzero(array < THRESHOLD_DB)),
        "median_dB": float(np.median(array)),
        "mean_dB": float(np.mean(array)),
        "Q90_dB": float(np.quantile(array, 0.90)),
        "Q95_dB": float(np.quantile(array, 0.95)),
        "Q99_dB": float(np.quantile(array, 0.99)),
        "worst_dB": float(np.max(array)),
    }


def transition_label(ols_pass: bool, ridge_pass: bool) -> str:
    """Map C-to-B threshold outcomes to the four declared transition classes."""

    if ols_pass and ridge_pass:
        return "unchanged_pass"
    if not ols_pass and ridge_pass:
        return "OLS_fail_to_Ridge_pass"
    if ols_pass and not ridge_pass:
        return "OLS_pass_to_Ridge_fail"
    return "unchanged_fail"


__all__ = [
    "A_START",
    "B_RAW_LENGTH",
    "B_START",
    "B_VALID_LENGTH",
    "BLAS_THREADS_PER_WORKER",
    "C2_COLUMN_INDEX",
    "C2_STAGE_NUMBER",
    "C_RAW_LENGTH",
    "C_START",
    "C_VALID_LENGTH",
    "CV_BLOCK_COUNT",
    "CV_BLOCK_RAW_LENGTHS",
    "CV_BLOCK_VALID_LENGTHS",
    "DMAX",
    "EXPECTED_RAW_MANIFEST",
    "EXPECTED_SUPPORT_HASH",
    "EXPECTED_SUPPORT_IDS",
    "FULL_LENGTH",
    "HANDOFF_LOG",
    "PREVIOUS_C2_OLS_METRICS",
    "PROJECT_ROOT",
    "RESULT_ROOT",
    "RIDGE_LAMBDA_GRID",
    "STATE_COUNT",
    "TASK_NAME",
    "THRESHOLD_DB",
    "WORKER_COUNT",
    "WORK_LOG",
    "PreparedC2State",
    "assert_b_evaluation_allowed",
    "build_c_block_slices",
    "c_only_worker",
    "evaluate_c_only_lambda",
    "evaluate_final_state",
    "final_worker",
    "load_previous_ols_reference",
    "preflight_row",
    "prepare_c2_state",
    "raw_manifest_gate",
    "summary_stats",
    "support_digest",
    "transition_label",
    "verify_frozen_support",
    "worker_init",
]
