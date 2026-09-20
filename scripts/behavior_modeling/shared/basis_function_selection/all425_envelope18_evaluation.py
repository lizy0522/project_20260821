"""All-425 coverage evaluation for the frozen Envelope18 structure.

This module intentionally contains no candidate search, Hard-20 selection, or
Ridge tuning.  It verifies one already-frozen support on every canonical
Scenario 2 state using the same full-record alignment, independent Train/Test
gain calibration, local ``dmax=2`` history, and OLS solver as the preceding
Envelope75 task.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path

# Must be set before NumPy/SciPy is imported by spawned workers.
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
from data_management.shared import build_state_table  # noqa: E402

from behavior_modeling.shared.basis_function_selection.envelope_dictionary import (  # noqa: E402
    DMAX,
    EnvelopeBasis,
    build_envelope_bank,
    build_envelope_dictionary,
)
from behavior_modeling.shared.basis_function_selection.hard20_envelope75_selection import (  # noqa: E402
    EXPECTED_RAW_BYTES,
    EXPECTED_RAW_FILE_COUNT,
    EXPECTED_RAW_MANIFEST,
    EXPECTED_RAW_MAT_COUNT,
    OLS_CONDITION_HARD_LIMIT,
    _raw_manifest,
    prepare_forward_state,
)
from behavior_modeling.shared.basis_function_selection.model_solver import fit_ols  # noqa: E402

TASK_NAME = "scenario_2_all425_envelope18_coverage_evaluation_5B"
RESULT_ROOT = PROJECT_ROOT / "results" / "behavior_modeling" / "scenario_2" /TASK_NAME
WORK_LOG = PROJECT_ROOT / "work_logs" / "behavior_modeling" / "scenario_2" /TASK_NAME / "execution_log.txt"  # noqa: E501
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"

STATE_COUNT = 425
FULL_LENGTH = 24_576
TRAIN_START = 0
TRAIN_END = 16_384
TEST_START = 16_384
TEST_END = 24_576
TRAIN_LENGTH = TRAIN_END - TRAIN_START
TEST_LENGTH = TEST_END - TEST_START
TARGET_NMSE_DB = -40.0
WORKER_COUNT = 10
BLAS_THREADS_PER_WORKER = 1
GPU_USED = False

FINAL_SUPPORT_IDS = (
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
    "ENV_p04_m0_q0",
    "ENV_p04_m1_q0",
    "ENV_p05_m0_q0",
    "ENV_p05_m1_q0",
    "ENV_p07_m0_q0",
    "ENV_p08_m0_q0",
    "ENV_p09_m0_q0",
)

HARD20_IDS = frozenset(
    {
        325,
        332,
        333,
        334,
        336,
        353,
        324,
        338,
        350,
        326,
        349,
        339,
        351,
        342,
        328,
        323,
        355,
        331,
        327,
        356,
    }
)

FROZEN_MODEL_RESULT = (
    PROJECT_ROOT
    / "results"
    / "behavior_modeling"
    / "scenario_2" /"scenario_2_hard20_envelope75_basis_selection_5B"
    / "09_final_model_definition.csv"
)

_WORKER_TERMS: tuple[EnvelopeBasis, ...] | None = None
_WORKER_SUPPORT: tuple[int, ...] | None = None
_WORKER_STATE_TABLE: dict[int, dict[str, int | float]] | None = None


def support_hash(support_ids: Sequence[str] = FINAL_SUPPORT_IDS) -> str:
    """Return a stable hash of the frozen ordered support contract."""

    return hashlib.sha256(
        json.dumps(list(support_ids), ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def verify_frozen_support() -> tuple[tuple[EnvelopeBasis, ...], tuple[int, ...], str]:
    """Verify the source contract against the previous task's frozen definition."""

    terms = tuple(build_envelope_dictionary())
    if len(FINAL_SUPPORT_IDS) != 18 or len(set(FINAL_SUPPORT_IDS)) != 18:
        raise RuntimeError("Frozen Envelope18 support must contain 18 unique IDs")
    by_id = {term.basis_id: term for term in terms}
    missing = [basis_id for basis_id in FINAL_SUPPORT_IDS if basis_id not in by_id]
    if missing:
        raise RuntimeError(f"Frozen Envelope18 IDs missing from Envelope75 dictionary: {missing}")
    indices = tuple(by_id[basis_id].index for basis_id in FINAL_SUPPORT_IDS)
    if len(set(indices)) != 18:
        raise RuntimeError("Frozen Envelope18 global indices are not unique")
    if not all(terms[index].mandatory for index in indices[:3]):
        raise RuntimeError("Frozen Envelope18 lost one of the three mandatory linear terms")

    if not FROZEN_MODEL_RESULT.is_file():
        raise FileNotFoundError(
            f"Previous frozen model definition is missing: {FROZEN_MODEL_RESULT}"
        )
    frozen_frame = pd.read_csv(FROZEN_MODEL_RESULT)
    actual_ids = tuple(frozen_frame["basis_id"].astype(str).tolist())
    if actual_ids != FINAL_SUPPORT_IDS:
        raise RuntimeError(
            "Previous final model definition does not match the frozen Envelope18 contract: "
            f"{actual_ids}"
        )
    if len(frozen_frame) != 18 or set(frozen_frame["K"].astype(int)) != {18}:
        raise RuntimeError("Previous final model definition does not have K=18")
    if set(np.asarray(frozen_frame["lambda"], dtype=float)) != {0.0}:
        raise RuntimeError("Previous final model definition does not have lambda=0")
    if set(frozen_frame["dmax"].astype(int)) != {DMAX}:
        raise RuntimeError("Previous final model definition does not have dmax=2")
    expected_global_indices = tuple(by_id[basis_id].index for basis_id in FINAL_SUPPORT_IDS)
    actual_global_indices = tuple(frozen_frame["global_index"].astype(int).tolist())
    if actual_global_indices != expected_global_indices:
        raise RuntimeError("Previous final model global indices changed")
    return terms, indices, support_hash()


def load_type_from_state(state: dict[str, int | float]) -> str:
    """Classify the four canonical load-condition groups."""

    fundamental_mismatch = int(state["funMng"]) != 0
    second_harmonic_mismatch = int(state["secMng"]) != 0
    if not fundamental_mismatch and not second_harmonic_mismatch:
        return "matched"
    if fundamental_mismatch and not second_harmonic_mismatch:
        return "fundamental_only"
    if not fundamental_mismatch and second_harmonic_mismatch:
        return "second_harmonic_only"
    return "joint_mismatch"


def _selected_bank(
    signal: np.ndarray, terms: Sequence[EnvelopeBasis], indices: Sequence[int]
) -> np.ndarray:
    full_bank = build_envelope_bank(signal, terms)
    selected = np.asarray(full_bank[:, tuple(int(index) for index in indices)], dtype=np.complex128)
    return selected


def _worker_init(support_indices: tuple[int, ...]) -> None:
    global _WORKER_TERMS, _WORKER_SUPPORT, _WORKER_STATE_TABLE
    _WORKER_TERMS = tuple(build_envelope_dictionary())
    _WORKER_SUPPORT = tuple(int(index) for index in support_indices)
    _WORKER_STATE_TABLE = {int(row["state_id"]): row for row in build_state_table()}


def evaluate_state(state_id: int) -> dict[str, object]:
    """Evaluate one state with the frozen support and Train-fitted coefficients."""

    if _WORKER_TERMS is None or _WORKER_SUPPORT is None or _WORKER_STATE_TABLE is None:
        raise RuntimeError("All-425 worker was not initialized")
    normalized_id = int(state_id)
    prepared = prepare_forward_state(normalized_id)
    phi_train = _selected_bank(prepared.x_train, _WORKER_TERMS, _WORKER_SUPPORT)
    phi_test = _selected_bank(prepared.x_test, _WORKER_TERMS, _WORKER_SUPPORT)
    expected_train_shape = (TRAIN_LENGTH - DMAX, len(_WORKER_SUPPORT))
    expected_test_shape = (TEST_LENGTH - DMAX, len(_WORKER_SUPPORT))
    if phi_train.shape != expected_train_shape:
        raise RuntimeError(f"state {normalized_id} Train design shape is {phi_train.shape}")
    if phi_test.shape != expected_test_shape:
        raise RuntimeError(f"state {normalized_id} Test design shape is {phi_test.shape}")

    train_target = prepared.y_train_adjusted[DMAX:]
    test_target = prepared.y_test_adjusted[DMAX:]
    fitted = fit_ols(phi_train, train_target, scale_columns=True)
    train_prediction = fitted.prediction
    test_prediction = phi_test @ fitted.theta
    modeling_nmse = float(fitted.nmse_db)
    generalization_nmse = float(nmse(test_target, test_prediction))
    condition = float(fitted.condition_number)
    finite = bool(
        np.all(np.isfinite(phi_train))
        and np.all(np.isfinite(phi_test))
        and np.all(np.isfinite(fitted.theta))
        and np.all(np.isfinite(train_prediction))
        and np.all(np.isfinite(test_prediction))
        and np.isfinite(condition)
    )
    rank = int(fitted.rank)
    numerical_gate = bool(
        finite and rank == len(_WORKER_SUPPORT) and condition <= OLS_CONDITION_HARD_LIMIT
    )
    state = _WORKER_STATE_TABLE[normalized_id]
    modeling_pass = bool(modeling_nmse < TARGET_NMSE_DB)
    generalization_pass = bool(generalization_nmse < TARGET_NMSE_DB)
    return {
        "state_id": normalized_id,
        "funMng": int(state["funMng"]),
        "funAng": int(state["funAng"]),
        "secMng": int(state["secMng"]),
        "secAng": int(state["secAng"]),
        "Vm": float(state["Vm"]),
        "Pin": float(state["Pin"]),
        "load_type": load_type_from_state(state),
        "rough_delay": int(prepared.rough_delay),
        "fine_delay": float(prepared.fine_delay),
        "Train_gain_real": float(prepared.train_gain.real),
        "Train_gain_imag": float(prepared.train_gain.imag),
        "Test_gain_real": float(prepared.test_gain.real),
        "Test_gain_imag": float(prepared.test_gain.imag),
        "modeling_NMSE_dB": modeling_nmse,
        "generalization_NMSE_dB": generalization_nmse,
        "generalization_gap_dB": generalization_nmse - modeling_nmse,
        "modeling_pass": modeling_pass,
        "generalization_pass": generalization_pass,
        "both_pass": bool(modeling_pass and generalization_pass),
        "rank": rank,
        "condition_number": condition,
        "coefficient_norm": float(np.linalg.norm(fitted.theta)),
        "finite": finite,
        "numerical_gate_pass": numerical_gate,
        "is_hard20": normalized_id in HARD20_IDS,
    }


def _worker_entry(state_id: int) -> dict[str, object]:
    return evaluate_state(int(state_id))


def empty_metrics_frame() -> pd.DataFrame:
    """Return an empty frame with the stable all-425 result columns."""

    return pd.DataFrame(
        columns=[
            "state_id",
            "funMng",
            "funAng",
            "secMng",
            "secAng",
            "Vm",
            "Pin",
            "load_type",
            "rough_delay",
            "fine_delay",
            "Train_gain_real",
            "Train_gain_imag",
            "Test_gain_real",
            "Test_gain_imag",
            "modeling_NMSE_dB",
            "generalization_NMSE_dB",
            "generalization_gap_dB",
            "modeling_pass",
            "generalization_pass",
            "both_pass",
            "rank",
            "condition_number",
            "coefficient_norm",
            "finite",
            "numerical_gate_pass",
            "is_hard20",
        ]
    )


def distribution(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("distribution requires at least one value")
    return {
        "count": int(array.size),
        "median_dB": float(np.median(array)),
        "mean_dB": float(np.mean(array)),
        "Q90_dB": float(np.quantile(array, 0.90)),
        "Q95_dB": float(np.quantile(array, 0.95)),
        "Q99_dB": float(np.quantile(array, 0.99)),
        "worst_dB": float(np.max(array)),
    }


def summarize_subset(frame: pd.DataFrame) -> dict[str, object]:
    """Summarize modeling, generalization, both-pass, and Train/Test gap."""

    if frame.empty:
        raise ValueError("cannot summarize an empty state subset")
    modeling = frame["modeling_NMSE_dB"].to_numpy(dtype=float)
    generalization = frame["generalization_NMSE_dB"].to_numpy(dtype=float)
    gap = frame["generalization_gap_dB"].to_numpy(dtype=float)
    return {
        "N": int(len(frame)),
        "modeling_pass_count": int(frame["modeling_pass"].sum()),
        "modeling_pass_rate": float(frame["modeling_pass"].mean()),
        "generalization_pass_count": int(frame["generalization_pass"].sum()),
        "generalization_pass_rate": float(frame["generalization_pass"].mean()),
        "both_pass_count": int(frame["both_pass"].sum()),
        "both_pass_rate": float(frame["both_pass"].mean()),
        "modeling": distribution(modeling),
        "generalization": distribution(generalization),
        "gap": {
            "median_dB": float(np.median(gap)),
            "mean_dB": float(np.mean(gap)),
            "Q90_dB": float(np.quantile(gap, 0.90)),
            "Q95_dB": float(np.quantile(gap, 0.95)),
            "max_dB": float(np.max(gap)),
        },
    }


def add_failure_type(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()

    def classify(row: pd.Series) -> str:
        modeling_pass = bool(row["modeling_pass"])
        generalization_pass = bool(row["generalization_pass"])
        if modeling_pass and generalization_pass:
            return ""
        if not modeling_pass and generalization_pass:
            return "modeling_only_fail"
        if modeling_pass and not generalization_pass:
            return "generalization_only_fail"
        return "both_fail"

    result["failure_type"] = result.apply(classify, axis=1)
    return result


def load_type_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for load_type in (
        "matched",
        "fundamental_only",
        "second_harmonic_only",
        "joint_mismatch",
    ):
        selected = frame.loc[frame["load_type"] == load_type]
        summary = summarize_subset(selected)
        rows.append(
            {
                "load_type": load_type,
                "N": summary["N"],
                "modeling_pass_count": summary["modeling_pass_count"],
                "modeling_pass_rate": summary["modeling_pass_rate"],
                "generalization_pass_count": summary["generalization_pass_count"],
                "generalization_pass_rate": summary["generalization_pass_rate"],
                "both_pass_count": summary["both_pass_count"],
                "both_pass_rate": summary["both_pass_rate"],
                "modeling_median_dB": summary["modeling"]["median_dB"],
                "generalization_median_dB": summary["generalization"]["median_dB"],
                "generalization_Q95_dB": summary["generalization"]["Q95_dB"],
                "generalization_worst_dB": summary["generalization"]["worst_dB"],
            }
        )
    result = pd.DataFrame(rows)
    expected_counts = {
        "matched": 1,
        "fundamental_only": 24,
        "second_harmonic_only": 16,
        "joint_mismatch": 384,
    }
    if dict(zip(result["load_type"], result["N"], strict=True)) != expected_counts:
        raise RuntimeError(
            f"Load-type counts changed: {result[['load_type', 'N']].to_dict('records')}"
        )
    return result


def raw_manifest_gate() -> dict[str, object]:
    value = _raw_manifest()
    expected = {
        "sha256": EXPECTED_RAW_MANIFEST,
        "file_count": EXPECTED_RAW_FILE_COUNT,
        "mat_count": EXPECTED_RAW_MAT_COUNT,
        "bytes": EXPECTED_RAW_BYTES,
    }
    if value != expected:
        raise RuntimeError(f"data/raw manifest differs from frozen baseline: {value}")
    return value


def task_definition_text(support_ids: Sequence[str]) -> str:
    return (
        "\n".join(
            [
                f"Task: {TASK_NAME}",
                "Mode: coverage evaluation of an already-frozen Envelope18 model.",
                "Data: Scenario 2, all 425 canonical states, xin -> yout_withoutdpd_ori.",
                "Frozen model: K=18, lambda=0, OLS, dmax=2.",
                f"Support hash: {support_hash(support_ids)}",
                f"Support IDs: {list(support_ids)}",
                "Preprocess: full-record rough/fine alignment, fixed Train/Test split, "
                "independent gains.",
                "Acceptance: strict modeling_NMSE_dB < -40 and generalization_NMSE_dB < -40.",
                "Workers: 10 spawned CPU processes, one BLAS thread per worker, no GPU.",
                "Excluded: search, re-selection, Ridge scan, dmax/p-order expansion, "
                "LUT/DPD/clustering.",
            ]
        )
        + "\n"
    )


__all__ = [
    "BLAS_THREADS_PER_WORKER",
    "DMAX",
    "FINAL_SUPPORT_IDS",
    "FROZEN_MODEL_RESULT",
    "FULL_LENGTH",
    "HARD20_IDS",
    "OLS_CONDITION_HARD_LIMIT",
    "PROJECT_ROOT",
    "RESULT_ROOT",
    "STATE_COUNT",
    "TARGET_NMSE_DB",
    "TASK_NAME",
    "TEST_END",
    "TEST_LENGTH",
    "TEST_START",
    "TRAIN_END",
    "TRAIN_LENGTH",
    "TRAIN_START",
    "add_failure_type",
    "distribution",
    "empty_metrics_frame",
    "evaluate_state",
    "load_type_from_state",
    "load_type_summary",
    "raw_manifest_gate",
    "summarize_subset",
    "support_hash",
    "task_definition_text",
    "verify_frozen_support",
    "_worker_entry",
    "_worker_init",
]
