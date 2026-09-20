"""Scenario 2 Gate-B48 unified-support basis refinement.

This module is deliberately task-local.  It reuses the canonical
Frozen-centered dictionary and the existing ``data_management`` /
``signal_segmentation`` pipeline, but does not change the shared basis or model
solver modules.  The worker contract is state-major: a worker loads and
preprocesses one state once, builds all four Gate-B48 banks once, and returns
only scalar model metrics for the requested support candidates.
"""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# These must be set before importing NumPy/SciPy in spawned workers.
# ruff: noqa: E402
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

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from data_management.shared import build_state_table, load_by_id  # noqa: E402
from signal_segmentation.shared import (  # noqa: E402
    build_partition_from_xin,
    get_ilc_pair,
    preprocess_full_pair,
)

from behavior_modeling.shared.basis_function_selection.frozen_centered_dictionary import (  # noqa: E402
    FrozenCenteredBasis,
    build_frozen_centered_bank,
)

TASK_NAME = "scenario_2_gateb48_unified_basis_refinement_5B"
RESULT_ROOT = PROJECT_ROOT / "results" / "behavior_modeling" / "scenario_2" /TASK_NAME
WORK_LOG = PROJECT_ROOT / "work_logs" / "behavior_modeling" / "scenario_2" /TASK_NAME / "execution_log.txt"  # noqa: E501
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"

STATE_COUNT = 425
FULL_LENGTH = 24_576
ABC_BOUNDS = (0, 12_288, 17_203, 24_576)
VALID_LENGTHS = {"A": 12_286, "B": 4_913, "C": 7_371}
DMAX = 2
RIDGE_LAMBDA = 1e-8
THRESHOLD_DB = -40.0
DEVELOPMENT_COUNT = 340
VALIDATION_COUNT = 85
SPLIT_SEED = 20_260_827
K_SAFETY_CAP = 30
ACPR_UNUSED_MARKER = "not_used"

BASELINE_SUPPORT_IDS = (
    "LIN_d0",
    "LIN_d1",
    "LIN_d2",
    "ENV_p02_m0_q0",
    "ENV_p02_m0_q1",
    "ENV_p02_m0_q2",
    "ENV_p02_m1_q0",
    "ENV_p03_m0_q0",
    "ENV_p03_m0_q1",
    "ENV_p03_m1_q0",
    "ENV_p03_m1_q1",
    "ENV_p05_m0_q0",
    "ENV_p05_m1_q0",
    "ENV_p07_m0_q0",
    "ENV_p09_m0_q0",
    "ENV_p09_m0_q1",
    "ENV_p09_m1_q1",
)

RIDGE_GRID = (0.0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6)
EXTRA_RIDGE_LAMBDA = 1e-5
BENCHMARK_WORKER_COUNTS = (6, 8, 10, 12, 14, 16, 18)
BENCHMARK_SAMPLE_SIZE = 32
BENCHMARK_CANDIDATE_BATCH_SIZE = 4
BLAS_THREADS_PER_WORKER = 1

_WORKER_TERMS: tuple[FrozenCenteredBasis, ...] | None = None
_WORKER_GATE_B: tuple[int, ...] | None = None
_WORKER_STATE_CACHE: dict[int, PreparedState] = {}


@dataclass(frozen=True)
class PreparedState:
    """Full Gate-B48 matrices and valid targets for one state."""

    state_id: int
    a_train_bank: np.ndarray
    a_b_bank: np.ndarray
    a_train_target: np.ndarray
    a_b_target: np.ndarray
    c_train_bank: np.ndarray
    c_b_bank: np.ndarray
    c_train_target: np.ndarray
    c_b_target: np.ndarray


@dataclass(frozen=True)
class FitMetrics:
    train_nmse_db: float
    b_nmse_db: float
    rank: int
    condition_number: float


def _json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _as_vector(value: Any, name: str, length: int = FULL_LENGTH) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1 or array.size != length or not np.iscomplexobj(array):
        raise ValueError(f"{name} must be a complex vector of length {length}, got {array.shape}")
    array = np.asarray(array, dtype=np.complex128)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def _scalar(value: Any, name: str) -> float:
    array = np.asarray(value).reshape(-1)
    if array.size != 1 or not np.isfinite(array[0]):
        raise ValueError(f"{name} must be one finite scalar")
    return float(array[0])


def _nmse_db(target: np.ndarray, prediction: np.ndarray) -> float:
    target = np.asarray(target, dtype=np.complex128).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.complex128).reshape(-1)
    denominator = float(np.sum(np.abs(target) ** 2))
    if denominator <= 0.0 or not np.isfinite(denominator):
        raise ValueError("NMSE target energy is not positive and finite")
    error = float(np.sum(np.abs(target - prediction) ** 2))
    if error == 0.0:
        return float("-inf")
    return float(10.0 * np.log10(error / denominator))


def _fit_augmented_ridge(
    phi_train: np.ndarray,
    target_train: np.ndarray,
    phi_b: np.ndarray,
    target_b: np.ndarray,
    ridge_lambda: float,
    *,
    exact_condition: bool = False,
) -> FitMetrics:
    """Fit through augmented ``np.linalg.lstsq`` without normal equations."""

    phi_train = np.asarray(phi_train, dtype=np.complex128)
    target_train = np.asarray(target_train, dtype=np.complex128).reshape(-1)
    phi_b = np.asarray(phi_b, dtype=np.complex128)
    target_b = np.asarray(target_b, dtype=np.complex128).reshape(-1)
    if phi_train.ndim != 2 or phi_train.shape[0] != target_train.size:
        raise ValueError("training matrix/target shape mismatch")
    if phi_b.ndim != 2 or phi_b.shape[1] != phi_train.shape[1]:
        raise ValueError("B matrix/support shape mismatch")
    if phi_b.shape[0] != target_b.size or phi_train.shape[1] == 0:
        raise ValueError("B matrix/target shape mismatch")
    if not np.all(np.isfinite(phi_train)) or not np.all(np.isfinite(phi_b)):
        raise ValueError("design matrix contains NaN or Inf")
    ridge_lambda = float(ridge_lambda)
    if not np.isfinite(ridge_lambda) or ridge_lambda < 0.0:
        raise ValueError("ridge_lambda must be finite and nonnegative")

    n_samples, coefficient_count = phi_train.shape
    if ridge_lambda == 0.0:
        # Match the shared OLS solver's column reparameterization while still
        # solving with least squares rather than normal equations.
        scale = np.sqrt(np.mean(np.abs(phi_train) ** 2, axis=0))
        if np.any(scale <= 0.0) or not np.all(np.isfinite(scale)):
            raise RuntimeError("invalid OLS column scale")
        solve_matrix = phi_train / scale[None, :]
        beta, _, rank, singular = np.linalg.lstsq(solve_matrix, target_train, rcond=None)
        theta = beta / scale
    else:
        augmented = np.vstack(
            [
                phi_train,
                np.sqrt(n_samples * ridge_lambda) * np.eye(coefficient_count, dtype=np.complex128),
            ]
        )
        augmented_target = np.concatenate(
            [target_train, np.zeros(coefficient_count, dtype=np.complex128)]
        )
        theta, _, rank, singular = np.linalg.lstsq(augmented, augmented_target, rcond=None)
    prediction_train = phi_train @ theta
    prediction_b = phi_b @ theta
    if not np.all(np.isfinite(theta)) or not np.all(np.isfinite(prediction_b)):
        raise RuntimeError("ridge prediction contains NaN or Inf")
    if exact_condition:
        raw_singular = np.linalg.svd(phi_train, compute_uv=False)
        condition = (
            float(raw_singular[0] / raw_singular[-1])
            if raw_singular.size and raw_singular[-1] > 0.0
            else float("inf")
        )
    else:
        condition = (
            float(singular[0] / singular[-1])
            if singular.size and singular[-1] > 0.0
            else float("inf")
        )
    return FitMetrics(
        train_nmse_db=_nmse_db(target_train, prediction_train),
        b_nmse_db=_nmse_db(target_b, prediction_b),
        rank=int(rank),
        condition_number=condition,
    )


def _prepare_canonical_bank(
    canonical: Any,
    train_name: str,
    terms: Sequence[FrozenCenteredBasis],
    gate_b: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_segment = canonical[train_name]
    b_segment = canonical["B"]
    train_bank = build_frozen_centered_bank(train_segment.input, terms, gate_b)
    b_bank = build_frozen_centered_bank(b_segment.input, terms, gate_b)
    train_target = np.asarray(train_segment.output, dtype=np.complex128)[DMAX:]
    b_target = np.asarray(b_segment.output, dtype=np.complex128)[DMAX:]
    expected_train = VALID_LENGTHS[train_name]
    if train_bank.shape != (expected_train, len(gate_b)):
        raise RuntimeError(f"{train_name} Gate-B bank shape mismatch: {train_bank.shape}")
    if b_bank.shape != (VALID_LENGTHS["B"], len(gate_b)):
        raise RuntimeError(f"B Gate-B bank shape mismatch: {b_bank.shape}")
    if train_target.shape != (expected_train,) or b_target.shape != (VALID_LENGTHS["B"],):
        raise RuntimeError(f"{train_name}/B target shape mismatch")
    return train_bank, b_bank, train_target, b_target


def _prepare_state(
    state_id: int,
    terms: Sequence[FrozenCenteredBasis],
    gate_b: Sequence[int],
) -> PreparedState:
    data = load_by_id(int(state_id))
    input_history = np.asarray(data["xin_pd_ori_ilc"])
    output_history = np.asarray(data["yout_withdpd_ori_ilc"])
    if (
        input_history.ndim != 2
        or output_history.shape != input_history.shape
        or input_history.shape[0] != FULL_LENGTH
        or input_history.shape[1] < 2
        or input_history.shape[1] > 5
    ):
        raise RuntimeError(f"state {state_id} ILC history is invalid")
    xin = _as_vector(data["xin"], "xin")
    partition = build_partition_from_xin(xin)
    ilc_count = int(input_history.shape[1])
    a_pair = get_ilc_pair(data, ilc_count - 1)
    c_pair = get_ilc_pair(data, 1)
    a = preprocess_full_pair(
        a_pair.input_full,
        a_pair.output_raw_full,
        partition,
        pair_type="ILC",
        iteration_index=ilc_count - 1,
        input_peak_normalization_factor=a_pair.input_peak_normalization_factor,
    )
    c = preprocess_full_pair(
        c_pair.input_full,
        c_pair.output_raw_full,
        partition,
        pair_type="ILC",
        iteration_index=1,
        input_peak_normalization_factor=c_pair.input_peak_normalization_factor,
    )
    a_train, a_b, a_y, a_by = _prepare_canonical_bank(a, "A", terms, gate_b)
    c_train, c_b, c_y, c_by = _prepare_canonical_bank(c, "C", terms, gate_b)
    return PreparedState(
        state_id=int(state_id),
        a_train_bank=a_train,
        a_b_bank=a_b,
        a_train_target=a_y,
        a_b_target=a_by,
        c_train_bank=c_train,
        c_b_bank=c_b,
        c_train_target=c_y,
        c_b_target=c_by,
    )


def _evaluate_prepared_support(
    prepared: PreparedState,
    support: Sequence[int],
    ridge_lambda: float,
    *,
    exact_condition: bool = False,
) -> dict[str, Any]:
    indices = tuple(int(index) for index in support)
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("support must be nonempty and unique")
    a_fit = _fit_augmented_ridge(
        prepared.a_train_bank[:, indices],
        prepared.a_train_target,
        prepared.a_b_bank[:, indices],
        prepared.a_b_target,
        ridge_lambda,
        exact_condition=exact_condition,
    )
    c_fit = _fit_augmented_ridge(
        prepared.c_train_bank[:, indices],
        prepared.c_train_target,
        prepared.c_b_bank[:, indices],
        prepared.c_b_target,
        ridge_lambda,
        exact_condition=exact_condition,
    )
    joint_b = max(a_fit.b_nmse_db, c_fit.b_nmse_db)
    return {
        "state_id": prepared.state_id,
        "Aend_train_NMSE_dB": a_fit.train_nmse_db,
        "Aend_B_NMSE_dB": a_fit.b_nmse_db,
        "C2_train_NMSE_dB": c_fit.train_nmse_db,
        "C2_B_NMSE_dB": c_fit.b_nmse_db,
        "joint_B_NMSE_dB": joint_b,
        "pass_40dB": bool(joint_b < THRESHOLD_DB),
        "Aend_rank": a_fit.rank,
        "C2_rank": c_fit.rank,
        "Aend_condition_number": a_fit.condition_number,
        "C2_condition_number": c_fit.condition_number,
        "K": len(indices),
        "lambda": float(ridge_lambda),
    }


def _worker_init(terms: tuple[FrozenCenteredBasis, ...], gate_b: tuple[int, ...]) -> None:
    global _WORKER_TERMS, _WORKER_GATE_B, _WORKER_STATE_CACHE
    _WORKER_TERMS = terms
    _WORKER_GATE_B = gate_b
    _WORKER_STATE_CACHE = {}


def _state_candidate_worker(
    payload: tuple[int, tuple[tuple[str, tuple[int, ...]], ...], float, bool],
) -> list[dict[str, Any]]:
    state_id, support_payloads, ridge_lambda, exact_condition = payload
    if _WORKER_TERMS is None or _WORKER_GATE_B is None:
        raise RuntimeError("Gate-B worker is not initialized")
    prepared = _WORKER_STATE_CACHE.get(int(state_id))
    if prepared is None:
        prepared = _prepare_state(int(state_id), _WORKER_TERMS, _WORKER_GATE_B)
        _WORKER_STATE_CACHE[int(state_id)] = prepared
    rows: list[dict[str, Any]] = []
    for support_id, support in support_payloads:
        row = _evaluate_prepared_support(
            prepared,
            support,
            ridge_lambda,
            exact_condition=exact_condition,
        )
        row["support_id"] = support_id
        rows.append(row)
    return rows


def evaluate_supports(
    executor: ProcessPoolExecutor,
    state_ids: Iterable[int],
    supports: Sequence[tuple[str, tuple[int, ...]]],
    ridge_lambda: float,
    *,
    exact_condition: bool = False,
    progress_label: str = "evaluation",
) -> pd.DataFrame:
    """Evaluate a support batch with state-major worker tasks."""

    supports = tuple(
        (str(name), tuple(int(index) for index in support)) for name, support in supports
    )
    if not supports:
        raise ValueError("supports cannot be empty")
    state_ids = tuple(int(state_id) for state_id in state_ids)
    payload = (supports, float(ridge_lambda), bool(exact_condition))
    futures = {
        executor.submit(_state_candidate_worker, (state_id, *payload)): state_id
        for state_id in state_ids
    }
    rows: list[dict[str, Any]] = []
    for completed, future in enumerate(as_completed(futures), start=1):
        rows.extend(future.result())
        if completed % 25 == 0 or completed == len(futures):
            print(f"[{progress_label}] {completed}/{len(futures)} states", flush=True)
    frame = pd.DataFrame(rows)
    expected_rows = len(state_ids) * len(supports)
    if len(frame) != expected_rows:
        raise RuntimeError(f"{progress_label} returned {len(frame)} rows, expected {expected_rows}")
    frame = frame.sort_values(["support_id", "state_id"]).reset_index(drop=True)
    return frame


def make_dictionary_frame(
    terms: Sequence[FrozenCenteredBasis], gate_b: Sequence[int]
) -> pd.DataFrame:
    gate_b_set = set(int(index) for index in gate_b)
    rows = []
    for local_index, global_index in enumerate(gate_b):
        term = terms[int(global_index)]
        rows.append(
            {
                "gate": "GATE_B",
                "gate_local_index": local_index,
                "global_index": int(global_index),
                "basis_id": term.basis_id,
                "formula": term.formula,
                "family": term.family,
                "order": int(term.order),
                "signal_delay": term.signal_delay,
                "envelope_delay": term.envelope_delay,
                "in_baseline17": term.basis_id in BASELINE_SUPPORT_IDS,
                "in_gate_b": int(global_index) in gate_b_set,
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 48 or frame["basis_id"].nunique() != 48:
        raise RuntimeError("Gate-B48 dictionary contract failed")
    if int(frame["in_baseline17"].sum()) != 17:
        raise RuntimeError("Baseline17 is not a 17-term subset of Gate-B48")
    return frame


build_dictionary_frame = make_dictionary_frame


def build_split_definition() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Reproduce the historical weighted-fusion split builder exactly."""

    from behavior_fingerprint_ranking_consistency.shared.scenario2_y_lut_weighted_fusion_analysis import (  # noqa: E501
        build_query_split,
    )

    state_frame = pd.DataFrame(build_state_table())
    split, development, validation = build_query_split(state_frame, seed=SPLIT_SEED)
    if len(development) != DEVELOPMENT_COUNT or len(validation) != VALIDATION_COUNT:
        raise RuntimeError("historical split counts are not 340/85")
    if not np.array_equal(
        np.sort(np.concatenate([development, validation])), np.arange(STATE_COUNT)
    ):
        raise RuntimeError("historical split does not cover state_id 0...424")
    return split, development.astype(np.int64), validation.astype(np.int64)


def support_id_from_indices(terms: Sequence[FrozenCenteredBasis], support: Sequence[int]) -> str:
    return ";".join(terms[int(index)].basis_id for index in support)


def baseline_support_indices(
    terms: Sequence[FrozenCenteredBasis], gate_b: Sequence[int]
) -> tuple[int, ...]:
    by_id = {term.basis_id: term.index for term in terms}
    missing = [basis_id for basis_id in BASELINE_SUPPORT_IDS if basis_id not in by_id]
    if missing:
        raise RuntimeError(f"Baseline17 IDs missing from shared dictionary: {missing}")
    support = tuple(int(by_id[basis_id]) for basis_id in BASELINE_SUPPORT_IDS)
    gate_b_set = set(int(index) for index in gate_b)
    if not set(support) <= gate_b_set or len(set(support)) != 17:
        raise RuntimeError("Baseline17 is not a unique Gate-B subset")
    return support


def candidate_supports(
    terms: Sequence[FrozenCenteredBasis], gate_b: Sequence[int], current: Sequence[int]
) -> list[tuple[str, tuple[int, ...], int]]:
    current_set = set(int(index) for index in current)
    nonlinear = [
        int(index)
        for index in gate_b
        if int(index) not in current_set and terms[int(index)].order > 1
    ]
    return [
        (
            terms[index].basis_id,
            tuple([*current, index]),
            index,
        )
        for index in nonlinear
        if len(current) + 1 <= K_SAFETY_CAP
    ]


def aggregate_metrics(frame: pd.DataFrame, support_id: str) -> dict[str, Any]:
    selected = frame.loc[frame["support_id"] == support_id].copy()
    if selected.empty:
        raise ValueError(f"no metrics for support {support_id}")
    joint = selected["joint_B_NMSE_dB"].to_numpy(dtype=float)
    a_b = selected["Aend_B_NMSE_dB"].to_numpy(dtype=float)
    c_b = selected["C2_B_NMSE_dB"].to_numpy(dtype=float)
    a_train = selected["Aend_train_NMSE_dB"].to_numpy(dtype=float)
    c_train = selected["C2_train_NMSE_dB"].to_numpy(dtype=float)
    conditions = selected[["Aend_condition_number", "C2_condition_number"]].to_numpy(dtype=float)
    condition_values = conditions.reshape(-1)
    return {
        "support_id": support_id,
        "state_count": int(selected["state_id"].nunique()),
        "pass_count": int(np.count_nonzero(joint < THRESHOLD_DB)),
        "pass_rate": float(np.mean(joint < THRESHOLD_DB)),
        "joint_median_dB": float(np.median(joint)),
        "joint_mean_dB": float(np.mean(joint)),
        "joint_Q90_dB": float(np.quantile(joint, 0.90)),
        "joint_Q95_dB": float(np.quantile(joint, 0.95)),
        "joint_worst_dB": float(np.max(joint)),
        "Aend_B_median_dB": float(np.median(a_b)),
        "C2_B_median_dB": float(np.median(c_b)),
        "Aend_train_median_dB": float(np.median(a_train)),
        "C2_train_median_dB": float(np.median(c_train)),
        "condition_Q99": float(np.quantile(condition_values, 0.99)),
        "condition_max": float(np.max(condition_values)),
        "K": int(selected["K"].iloc[0]),
        "lambda": float(selected["lambda"].iloc[0]),
    }


def rank_summary(summary: dict[str, Any]) -> tuple[Any, ...]:
    return (
        -int(summary["pass_count"]),
        float(summary["joint_Q95_dB"]),
        float(summary["joint_worst_dB"]),
        float(summary["joint_median_dB"]),
        int(summary["K"]),
        str(summary["support_id"]),
    )


def acceptance_test(
    current: dict[str, Any], candidate: dict[str, Any], *, allow_equal: bool = False
) -> tuple[bool, str]:
    pass_increase = int(candidate["pass_count"]) > int(current["pass_count"])
    q95_gain = float(current["joint_Q95_dB"]) - float(candidate["joint_Q95_dB"])
    worst_gain = float(current["joint_worst_dB"]) - float(candidate["joint_worst_dB"])
    q95_condition = q95_gain >= 0.10
    q95_nonworse = q95_gain >= -0.02
    worst_condition = worst_gain >= 0.20
    side_ok = (
        float(candidate["Aend_B_median_dB"]) - float(current["Aend_B_median_dB"]) < 0.05
        and float(candidate["C2_B_median_dB"]) - float(current["C2_B_median_dB"]) < 0.05
    )
    if not side_ok:
        return False, "side_B_median_degradation_exceeds_0.05dB"
    if pass_increase:
        return True, "pass_count_increase"
    if q95_condition:
        return True, "joint_Q95_improved_by_at_least_0.10dB"
    if q95_nonworse and worst_condition:
        return True, "joint_Q95_nonworse_and_worst_improved_by_at_least_0.20dB"
    if allow_equal and int(candidate["pass_count"]) == int(current["pass_count"]):
        return True, "equal_metrics_allowed_for_local_rescue"
    return False, "not_accepted"


def prune_acceptance(current: dict[str, Any], candidate: dict[str, Any]) -> tuple[bool, str]:
    pass_ok = int(candidate["pass_count"]) >= int(current["pass_count"])
    q95_delta = float(candidate["joint_Q95_dB"]) - float(current["joint_Q95_dB"])
    worst_delta = float(candidate["joint_worst_dB"]) - float(current["joint_worst_dB"])
    side_ok = (
        float(candidate["Aend_B_median_dB"]) - float(current["Aend_B_median_dB"]) < 0.05
        and float(candidate["C2_B_median_dB"]) - float(current["C2_B_median_dB"]) < 0.05
    )
    accepted = pass_ok and q95_delta <= 0.03 and worst_delta <= 0.05 and side_ok
    return accepted, (
        f"pass_ok={pass_ok};q95_delta={q95_delta:.6g};"
        f"worst_delta={worst_delta:.6g};side_ok={side_ok}"
    )


def benchmark_worker_counts(
    terms: tuple[FrozenCenteredBasis, ...],
    gate_b: tuple[int, ...],
    sample_state_ids: Sequence[int],
    supports: Sequence[tuple[str, tuple[int, ...]]],
) -> tuple[pd.DataFrame, int]:
    """Benchmark bounded worker counts with a warm-up and two pass orders."""

    rows: list[dict[str, Any]] = []
    counts = tuple(count for count in BENCHMARK_WORKER_COUNTS if count <= int(os.cpu_count() or 1))
    if not counts:
        counts = (max(1, min(4, int(os.cpu_count() or 1))),)
    for worker_count in counts:
        for pass_name, ordered_ids in (
            ("ascending", tuple(sample_state_ids)),
            ("descending", tuple(reversed(sample_state_ids))),
        ):
            started = time.perf_counter()
            with ProcessPoolExecutor(
                max_workers=worker_count,
                mp_context=__import__("multiprocessing").get_context("spawn"),
                initializer=_worker_init,
                initargs=(terms, gate_b),
            ) as executor:
                warmup_started = time.perf_counter()
                evaluate_supports(
                    executor,
                    ordered_ids[:1],
                    supports,
                    RIDGE_LAMBDA,
                    progress_label=f"benchmark-{worker_count}-{pass_name}-warmup",
                )
                warmup_seconds = time.perf_counter() - warmup_started
                measured_started = time.perf_counter()
                evaluate_supports(
                    executor,
                    ordered_ids,
                    supports,
                    RIDGE_LAMBDA,
                    progress_label=f"benchmark-{worker_count}-{pass_name}",
                )
                measured_seconds = time.perf_counter() - measured_started
            rows.append(
                {
                    "worker_count": worker_count,
                    "pass_name": pass_name,
                    "sample_state_count": len(ordered_ids),
                    "candidate_batch_count": len(supports),
                    "warmup_seconds": warmup_seconds,
                    "measured_seconds": measured_seconds,
                    "throughput_states_per_second": len(ordered_ids) / measured_seconds,
                    "blas_threads_per_worker": BLAS_THREADS_PER_WORKER,
                    "wall_seconds_including_pool": time.perf_counter() - started,
                }
            )
    frame = pd.DataFrame(rows)
    medians = frame.groupby("worker_count")["throughput_states_per_second"].median()
    maximum = float(medians.max())
    eligible = [int(count) for count, value in medians.items() if float(value) >= 0.97 * maximum]
    selected = min(eligible) if eligible else int(medians.idxmax())
    frame["selected_worker_count"] = selected
    frame["median_throughput_states_per_second"] = frame["worker_count"].map(medians)
    frame["maximum_median_throughput_states_per_second"] = maximum
    return frame, selected


def metadata_frame() -> pd.DataFrame:
    return pd.DataFrame(build_state_table()).sort_values("state_id").reset_index(drop=True)


def exact_condition_metrics(
    frame: pd.DataFrame,
    support_id: str,
    terms: Sequence[FrozenCenteredBasis],
    support: Sequence[int],
    state_ids: Sequence[int],
    ridge_lambda: float,
    executor: ProcessPoolExecutor,
) -> dict[str, Any]:
    """Re-evaluate one support with raw-design condition numbers."""

    evaluated = evaluate_supports(
        executor,
        state_ids,
        ((support_id, tuple(support)),),
        ridge_lambda,
        exact_condition=True,
        progress_label=f"condition-{support_id[:30]}",
    )
    return aggregate_metrics(evaluated, support_id)


def dump_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8"
    )


def checkpoint(path: Path, **payload: Any) -> None:
    payload = {"timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"), **payload}
    dump_json(path, payload)


def build_summary_text(payload: dict[str, Any]) -> str:
    lines = [
        f"Task: {TASK_NAME}",
        "Scope: Scenario 2 / 425 states / Fs=100 MHz / B=20 MHz / 5B / N=24576",
        "ABC: A=[0,12288), B=[12288,17203), C=[17203,24576); valid A/B/C=12286/4913/7371 at dmax=2",
        f"Split: Development={DEVELOPMENT_COUNT}, Validation={VALIDATION_COUNT}, seed={SPLIT_SEED}",
        "Model roles: Y-Aend uses A->B; Y-C2 uses C->B; unified support and "
        "lambda; coefficients state-specific",
        f"Baseline support: K=17, lambda={RIDGE_LAMBDA:.12g}",
        "Gate-B dictionary: 3 linear + p in {2,3,5,7,9}, m,q in {0,1,2}; "
        "K=48; baseline subset=17; remaining=31",
        "Selection data boundary: Development only; Validation unlocked after "
        "support/lambda freeze",
        "Excluded from this task: LUT retrieval, common-B consistency, Type-III "
        "clustering, DPD, low-bandwidth and expanded p/dmax scans",
        "",
    ]
    for key, value in payload.items():
        lines.append(f"{key}: {json.dumps(value, ensure_ascii=False, default=str)}")
    return "\n".join(lines) + "\n"


__all__ = [
    "ABC_BOUNDS",
    "BASELINE_SUPPORT_IDS",
    "BENCHMARK_CANDIDATE_BATCH_SIZE",
    "BENCHMARK_SAMPLE_SIZE",
    "DEVELOPMENT_COUNT",
    "DMAX",
    "FULL_LENGTH",
    "K_SAFETY_CAP",
    "RESULT_ROOT",
    "RIDGE_GRID",
    "RIDGE_LAMBDA",
    "SPLIT_SEED",
    "STATE_COUNT",
    "TASK_NAME",
    "THRESHOLD_DB",
    "VALIDATION_COUNT",
    "VALID_LENGTHS",
    "WORK_LOG",
    "acceptance_test",
    "aggregate_metrics",
    "baseline_support_indices",
    "benchmark_worker_counts",
    "build_dictionary_frame",
    "build_split_definition",
    "build_summary_text",
    "candidate_supports",
    "checkpoint",
    "dump_json",
    "evaluate_supports",
    "make_dictionary_frame",
    "build_dictionary_frame",
    "metadata_frame",
    "prune_acceptance",
    "rank_summary",
    "support_id_from_indices",
]
