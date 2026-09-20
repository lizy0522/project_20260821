"""All-425 evaluation of the frozen Envelope19-C2EndShared model."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

# ruff: noqa: E402,E501

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

from behavior_modeling.shared.basis_function_selection.dual_ilc_behavior_dataset import (  # noqa: E402
    BEHAVIORS,
    ILC_COL2,
    ILC_END,
    preflight_rows,
    prepare_dual_state,
)
from behavior_modeling.shared.basis_function_selection.envelope_dictionary import (  # noqa: E402
    DMAX,
    EnvelopeBasis,
    build_envelope_bank,
    build_envelope_dictionary,
)
from behavior_modeling.shared.basis_function_selection.hard20_envelope75_selection import (
    _raw_manifest,  # noqa: E402
)
from behavior_modeling.shared.basis_function_selection.model_solver import fit_ols  # noqa: E402

TASK_NAME = "scenario_2_all425_c2_ilcend_envelope19_shared_coverage_5B"
RESULT_ROOT = PROJECT_ROOT / "results" / "behavior_modeling" / "scenario_2" /TASK_NAME
WORK_LOG = PROJECT_ROOT / "work_logs" / "behavior_modeling" / "scenario_2" /TASK_NAME / "execution_log.txt"
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"
CHECKPOINT_PATH = RESULT_ROOT / "11_checkpoint.json"

PREVIOUS_ROOT = (
    PROJECT_ROOT / "results" / "behavior_modeling" / "scenario_2" /"scenario_2_hard20_c2_ilcend_shared_envelope75_basis_selection_5B"
)
FROZEN_MODEL_DEFINITION = PREVIOUS_ROOT / "07_final_model_definition.csv"
PREVIOUS_HARD20_RESULT = PREVIOUS_ROOT / "08_final_hard20_dual_behavior_train_cv_test.csv"
PREVIOUS_CHECKPOINT = PREVIOUS_ROOT / "12_checkpoint.json"

STATE_COUNT = 425
FULL_LENGTH = 24_576
TRAIN_START = 0
TRAIN_END = 16_384
TEST_START = 16_384
TEST_END = 24_576
TRAIN_LENGTH = TRAIN_END - TRAIN_START
TEST_LENGTH = TEST_END - TEST_START
WORKER_COUNT = 10
BLAS_THREADS_PER_WORKER = 1
TARGET_NMSE_DB = -40.0
EXPECTED_SUPPORT_HASH = "dda04e2fc270c9f5c2eb72b0b030059211ba0295934e3e3a4b2834172b11ac90"
EXPECTED_MODEL_NAME = "Envelope19-C2EndShared"
EXPECTED_RAW_MANIFEST = {
    "sha256": "910b06e4be6b60b1f4dc63409a3083eb328765347fbec3b89a97a41d6633a1e0",
    "file_count": 429,
    "mat_count": 427,
    "bytes": 2_258_448_137,
}
EXPECTED_HARD20_IDS = (
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
)

_WORKER_TERMS: tuple[EnvelopeBasis, ...] | None = None
_WORKER_SUPPORT: tuple[int, ...] | None = None
_WORKER_STATE_TABLE: dict[int, dict[str, int | float]] | None = None


def support_hash(ids: tuple[str, ...]) -> str:
    return hashlib.sha256(
        json.dumps(list(ids), ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def raw_manifest_gate() -> dict[str, object]:
    value = _raw_manifest()
    if value != EXPECTED_RAW_MANIFEST:
        raise RuntimeError(f"raw manifest differs from frozen baseline: {value}")
    return value


def load_frozen_model() -> tuple[tuple[EnvelopeBasis, ...], tuple[int, ...], tuple[str, ...], str, pd.DataFrame]:
    if not FROZEN_MODEL_DEFINITION.is_file():
        raise FileNotFoundError(f"frozen model definition is missing: {FROZEN_MODEL_DEFINITION}")
    if not PREVIOUS_CHECKPOINT.is_file():
        raise FileNotFoundError(f"previous checkpoint is missing: {PREVIOUS_CHECKPOINT}")
    frame = pd.read_csv(FROZEN_MODEL_DEFINITION)
    if len(frame) != 19 or set(frame["K"].astype(int)) != {19}:
        raise RuntimeError("frozen Envelope19 definition does not have K=19")
    if set(frame["dmax"].astype(int)) != {DMAX} or set(frame["lambda"].astype(float)) != {0.0}:
        raise RuntimeError("frozen Envelope19 dmax/lambda contract changed")
    if set(frame["solver"].astype(str)) != {"OLS"} or set(frame["model_name"].astype(str)) != {EXPECTED_MODEL_NAME}:
        raise RuntimeError("frozen Envelope19 model metadata changed")
    ids = tuple(frame["basis_id"].astype(str).tolist())
    digest = str(frame["support_hash"].iloc[0])
    if support_hash(ids) != EXPECTED_SUPPORT_HASH or digest != EXPECTED_SUPPORT_HASH:
        raise RuntimeError(f"frozen Envelope19 support hash mismatch: {digest}")
    terms = tuple(build_envelope_dictionary())
    by_id = {term.basis_id: term for term in terms}
    missing = [basis_id for basis_id in ids if basis_id not in by_id]
    if missing:
        raise RuntimeError(f"frozen support basis IDs missing from Envelope75: {missing}")
    indices = tuple(by_id[basis_id].index for basis_id in ids)
    index_column = "term_index" if "term_index" in frame else "global_index"
    if tuple(frame[index_column].astype(int).tolist()) != indices:
        raise RuntimeError("frozen Envelope19 term indices differ from canonical dictionary")
    checkpoint = json.loads(PREVIOUS_CHECKPOINT.read_text(encoding="utf-8"))
    if checkpoint.get("phase") != "completed" or checkpoint.get("model_frozen") is not True:
        raise RuntimeError("previous shared-support checkpoint is not completed/frozen")
    return terms, indices, ids, EXPECTED_SUPPORT_HASH, frame


def _worker_init(support_indices: tuple[int, ...]) -> None:
    global _WORKER_TERMS, _WORKER_SUPPORT, _WORKER_STATE_TABLE
    _WORKER_TERMS = tuple(build_envelope_dictionary())
    _WORKER_SUPPORT = tuple(int(index) for index in support_indices)
    _WORKER_STATE_TABLE = {int(row["state_id"]): row for row in build_state_table()}


def _evaluate_behavior(item: Any, support: tuple[int, ...], terms: tuple[EnvelopeBasis, ...]) -> dict[str, object]:
    train_phi = np.asarray(item.full_bank[:, support], dtype=np.complex128)
    fitted = fit_ols(train_phi, item.full_target, scale_columns=True)
    if item.x_test is None or item.y_test_adjusted is None or item.test_gain is None:
        raise RuntimeError("All425 evaluation requires Test data")
    test_bank = build_envelope_bank(item.x_test, terms)
    test_phi = np.asarray(test_bank[:, support], dtype=np.complex128)
    test_target = np.asarray(item.y_test_adjusted[DMAX:], dtype=np.complex128)
    test_prediction = test_phi @ fitted.theta
    train_value = float(fitted.nmse_db)
    test_value = float(nmse(test_target, test_prediction))
    finite = bool(
        np.all(np.isfinite(train_phi))
        and np.all(np.isfinite(test_phi))
        and np.all(np.isfinite(fitted.theta))
        and np.all(np.isfinite(fitted.prediction))
        and np.all(np.isfinite(test_prediction))
        and np.isfinite(fitted.condition_number)
    )
    return {
        "Train_NMSE_dB": train_value,
        "Test_NMSE_dB": test_value,
        "Generalization_gap_dB": test_value - train_value,
        "Train_pass": bool(train_value < TARGET_NMSE_DB),
        "Test_pass": bool(test_value < TARGET_NMSE_DB),
        "Both_pass": bool(train_value < TARGET_NMSE_DB and test_value < TARGET_NMSE_DB),
        "rank": int(fitted.rank),
        "condition_number": float(fitted.condition_number),
        "coefficient_norm": float(np.linalg.norm(fitted.theta)),
        "finite": finite,
        "theta": np.asarray(fitted.theta, dtype=np.complex128),
    }


def _worker_entry(state_id: int) -> dict[str, object]:
    if _WORKER_TERMS is None or _WORKER_SUPPORT is None or _WORKER_STATE_TABLE is None:
        raise RuntimeError("All425 worker was not initialized")
    prepared = prepare_dual_state(int(state_id), _WORKER_TERMS, include_test=True)
    state_info = _WORKER_STATE_TABLE[int(state_id)]
    behavior_results = {
        behavior: _evaluate_behavior(prepared.behavior(behavior), _WORKER_SUPPORT, _WORKER_TERMS)
        for behavior in BEHAVIORS
    }
    return {
        "State_ID": int(state_id),
        "funMng": int(state_info["funMng"]),
        "funAng": int(state_info["funAng"]),
        "secMng": int(state_info["secMng"]),
        "secAng": int(state_info["secAng"]),
        "preflight": preflight_rows(prepared),
        "behaviors": behavior_results,
    }


def summary_stats(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=float)
    if values.size != STATE_COUNT or not np.all(np.isfinite(values)):
        raise ValueError("All425 summary requires 425 finite values")
    return {
        "pass_count": int(np.count_nonzero(values < TARGET_NMSE_DB)),
        "pass_rate": float(np.mean(values < TARGET_NMSE_DB)),
        "median_dB": float(np.median(values)),
        "mean_dB": float(np.mean(values)),
        "Q90_dB": float(np.quantile(values, 0.90)),
        "Q95_dB": float(np.quantile(values, 0.95)),
        "Q99_dB": float(np.quantile(values, 0.99)),
        "worst_dB": float(np.max(values)),
    }


__all__ = [
    "BEHAVIORS",
    "EXPECTED_HARD20_IDS",
    "EXPECTED_MODEL_NAME",
    "EXPECTED_RAW_MANIFEST",
    "EXPECTED_SUPPORT_HASH",
    "FROZEN_MODEL_DEFINITION",
    "FULL_LENGTH",
    "HANDOFF_LOG",
    "ILC_COL2",
    "ILC_END",
    "PREVIOUS_HARD20_RESULT",
    "PREVIOUS_ROOT",
    "RESULT_ROOT",
    "STATE_COUNT",
    "TEST_LENGTH",
    "TASK_NAME",
    "TARGET_NMSE_DB",
    "TRAIN_LENGTH",
    "WORKER_COUNT",
    "WORK_LOG",
    "load_frozen_model",
    "raw_manifest_gate",
    "summary_stats",
    "support_hash",
    "_worker_entry",
    "_worker_init",
]
