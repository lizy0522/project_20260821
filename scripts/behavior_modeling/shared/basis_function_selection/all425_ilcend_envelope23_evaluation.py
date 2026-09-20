"""Evaluation-only coverage of the frozen ilc_end Envelope23 model."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

# Must be set before NumPy/SciPy imports in spawned workers.
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
    OLS_CONDITION_HARD_LIMIT,
    _raw_manifest,
)
from behavior_modeling.shared.basis_function_selection.ilcend_behavior_dataset import (  # noqa: E402
    prepare_ilcend_state,
)
from behavior_modeling.shared.basis_function_selection.model_solver import fit_ols  # noqa: E402

TASK_NAME = "scenario_2_all425_ilcend_envelope23_coverage_evaluation_5B"
RESULT_ROOT = PROJECT_ROOT / "results" / "behavior_modeling" / "scenario_2" /TASK_NAME
PREVIOUS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_modeling"
    / "scenario_2" /"scenario_2_hard20_ilcend_envelope75_basis_selection_5B"
)
PREVIOUS_MODEL_DEFINITION = PREVIOUS_ROOT / "08_final_model_definition.csv"
PREVIOUS_MODEL_SUMMARY = PREVIOUS_ROOT / "12_final_result_summary.txt"
PREVIOUS_CHECKPOINT = PREVIOUS_ROOT / "13_checkpoint.json"
PREVIOUS_HARD20_FINAL = PREVIOUS_ROOT / "09_final_hard20_train_cv_test.csv"

STATE_COUNT = 425
FULL_LENGTH = 24_576
TRAIN_START = 0
TRAIN_END = 16_384
TEST_START = 16_384
TEST_END = 24_576
TRAIN_LENGTH = TRAIN_END - TRAIN_START
TEST_LENGTH = TEST_END - TEST_START
WORKER_COUNT = 10
TARGET_NMSE_DB = -40.0

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

_WORKER_TERMS: tuple[EnvelopeBasis, ...] | None = None
_WORKER_SUPPORT: tuple[int, ...] | None = None
_WORKER_STATE_TABLE: dict[int, dict[str, int | float]] | None = None


def support_digest(support_ids: tuple[str, ...]) -> str:
    return hashlib.sha256(
        json.dumps(list(support_ids), ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def verify_frozen_ilcend_support() -> tuple[tuple[EnvelopeBasis, ...], tuple[int, ...], str]:
    """Load and verify the previous formal K=23 model definition."""

    if not PREVIOUS_MODEL_DEFINITION.is_file():
        raise FileNotFoundError(
            f"Missing previous ilc_end model definition: {PREVIOUS_MODEL_DEFINITION}"
        )
    if not PREVIOUS_MODEL_SUMMARY.is_file() or not PREVIOUS_CHECKPOINT.is_file():
        raise FileNotFoundError("Previous ilc_end summary/checkpoint is missing")
    model = pd.read_csv(PREVIOUS_MODEL_DEFINITION)
    if len(model) != 23:
        raise RuntimeError(f"Frozen ilc_end model must contain 23 rows, got {len(model)}")
    support_ids = tuple(model["basis_id"].astype(str).tolist())
    if len(set(support_ids)) != 23:
        raise RuntimeError("Frozen ilc_end support IDs are not unique")
    if set(model["K"].astype(int)) != {23}:
        raise RuntimeError("Frozen ilc_end model K is not 23")
    if set(np.asarray(model["lambda"], dtype=float)) != {0.0}:
        raise RuntimeError("Frozen ilc_end lambda is not 0")
    if set(model["dmax"].astype(int)) != {DMAX}:
        raise RuntimeError("Frozen ilc_end dmax is not 2")
    checkpoint = json.loads(PREVIOUS_CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint.get("phase") != "completed" or checkpoint.get("model_frozen") is not True:
        raise RuntimeError("Previous ilc_end checkpoint is not a completed frozen model")
    if checkpoint.get("test_unlocked") is not True:
        raise RuntimeError("Previous ilc_end checkpoint did not complete Test evaluation")
    terms = tuple(build_envelope_dictionary())
    by_id = {term.basis_id: term for term in terms}
    missing = [basis_id for basis_id in support_ids if basis_id not in by_id]
    if missing:
        raise RuntimeError(f"Frozen ilc_end support IDs missing from Envelope75: {missing}")
    indices = tuple(by_id[basis_id].index for basis_id in support_ids)
    index_column = "global_index" if "global_index" in model else "term_index"
    if index_column not in model:
        raise RuntimeError("Previous final model definition lacks a basis index column")
    if tuple(model[index_column].astype(int).tolist()) != indices:
        raise RuntimeError("Frozen ilc_end global indices differ from canonical Envelope75")
    digest = support_digest(support_ids)
    if digest != str(model["support_hash"].iloc[0]):
        raise RuntimeError("Frozen ilc_end support hash is inconsistent with model definition")
    return terms, indices, digest


def load_type_from_state(state: dict[str, int | float]) -> str:
    fun = int(state["funMng"]) != 0
    sec = int(state["secMng"]) != 0
    if not fun and not sec:
        return "matched"
    if fun and not sec:
        return "fundamental_only"
    if not fun and sec:
        return "second_harmonic_only"
    return "joint_mismatch"


def _worker_init(support_indices: tuple[int, ...]) -> None:
    global _WORKER_TERMS, _WORKER_SUPPORT, _WORKER_STATE_TABLE
    _WORKER_TERMS = tuple(build_envelope_dictionary())
    _WORKER_SUPPORT = tuple(int(index) for index in support_indices)
    _WORKER_STATE_TABLE = {int(row["state_id"]): row for row in build_state_table()}


def evaluate_state(state_id: int) -> dict[str, object]:
    if _WORKER_TERMS is None or _WORKER_SUPPORT is None or _WORKER_STATE_TABLE is None:
        raise RuntimeError("ilc_end coverage worker is not initialized")
    prepared = prepare_ilcend_state(int(state_id))
    train_bank = build_envelope_bank(prepared.x_train, _WORKER_TERMS)
    test_bank = build_envelope_bank(prepared.x_test, _WORKER_TERMS)
    phi_train = np.asarray(train_bank[:, _WORKER_SUPPORT], dtype=np.complex128)
    phi_test = np.asarray(test_bank[:, _WORKER_SUPPORT], dtype=np.complex128)
    if phi_train.shape != (TRAIN_LENGTH - DMAX, 23):
        raise RuntimeError(f"State {state_id} Train Phi shape={phi_train.shape}")
    if phi_test.shape != (TEST_LENGTH - DMAX, 23):
        raise RuntimeError(f"State {state_id} Test Phi shape={phi_test.shape}")
    train_target = prepared.y_train_adjusted[DMAX:]
    test_target = prepared.y_test_adjusted[DMAX:]
    fit = fit_ols(phi_train, train_target, scale_columns=True)
    test_prediction = phi_test @ fit.theta
    train_nmse = float(fit.nmse_db)
    test_nmse = float(nmse(test_target, test_prediction))
    condition = float(fit.condition_number)
    finite = bool(
        np.all(np.isfinite(phi_train))
        and np.all(np.isfinite(phi_test))
        and np.all(np.isfinite(fit.theta))
        and np.all(np.isfinite(fit.prediction))
        and np.all(np.isfinite(test_prediction))
        and np.isfinite(condition)
    )
    state = _WORKER_STATE_TABLE[int(state_id)]
    return {
        "State_ID": int(state_id),
        "funMng": int(state["funMng"]),
        "funAng": int(state["funAng"]),
        "secMng": int(state["secMng"]),
        "secAng": int(state["secAng"]),
        "Vm": float(state["Vm"]),
        "Pin": float(state["Pin"]),
        "load_type": load_type_from_state(state),
        "NMSE_withoutdpd_dB": float(prepared.nodpd_nmse_db),
        "valid_ilc_count": int(prepared.valid_ilc_count),
        "ilc_end_index": int(prepared.ilc_end_index),
        "is_Hard20": int(state_id) in HARD20_IDS,
        "Train_NMSE_dB": train_nmse,
        "Test_NMSE_dB": test_nmse,
        "Generalization_gap_dB": test_nmse - train_nmse,
        "Train_pass": bool(train_nmse < TARGET_NMSE_DB),
        "Test_pass": bool(test_nmse < TARGET_NMSE_DB),
        "Both_pass": bool(train_nmse < TARGET_NMSE_DB and test_nmse < TARGET_NMSE_DB),
        "rank": int(fit.rank),
        "condition_number": condition,
        "finite": finite,
        "theta": np.asarray(fit.theta, dtype=np.complex128),
    }


def worker_entry(state_id: int) -> dict[str, object]:
    return evaluate_state(int(state_id))


def raw_manifest_gate() -> dict[str, object]:
    value = _raw_manifest()
    expected = {
        "sha256": "910b06e4be6b60b1f4dc63409a3083eb328765347fbec3b89a97a41d6633a1e0",
        "file_count": 429,
        "mat_count": 427,
        "bytes": 2_258_448_137,
    }
    if value != expected:
        raise RuntimeError(f"raw manifest differs from frozen baseline: {value}")
    return value


__all__ = [
    "DMAX",
    "HARD20_IDS",
    "OLS_CONDITION_HARD_LIMIT",
    "PREVIOUS_HARD20_FINAL",
    "PREVIOUS_MODEL_DEFINITION",
    "PREVIOUS_ROOT",
    "RESULT_ROOT",
    "STATE_COUNT",
    "TARGET_NMSE_DB",
    "TEST_END",
    "TEST_LENGTH",
    "TEST_START",
    "TRAIN_END",
    "TRAIN_LENGTH",
    "TRAIN_START",
    "evaluate_state",
    "load_type_from_state",
    "raw_manifest_gate",
    "support_digest",
    "verify_frozen_ilcend_support",
    "worker_entry",
    "_worker_init",
]
