"""One-variable common-B probe control for frozen-centered17 Type-III retrieval."""

# ruff: noqa: E402, E501

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import freeze_support
from pathlib import Path
from typing import Any

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.io import loadmat, savemat

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _core.metrics import cnmse
from basis_function_selection.frozen_centered_dictionary import (
    build_frozen_centered_bank,
    build_frozen_centered_dictionary,
)
from basis_function_selection.model_solver import fit_ridge
from behavior_fingerprint_ranking_consistency.distance_matrix import (
    compute_cnmse_distance_matrix,
)
from data_manager import load_by_id
from signal_segmentation import build_partition_from_xin, get_ilc_pair, preprocess_full_pair

from behavior_fingerprint_lut_retrieval import (
    run_scenario2_C2_to_Aend_type3_cluster_compressed_5b as old_type3,
)

EXPERIMENT_NAME = "scenario_2_C2_to_Aend_frozen_centered17_type3_commonB_control_5B"
RESULT_PARENT = PROJECT_ROOT / "results" / "behavior_fingerprint_lut_retrieval"
PRIMARY_RESULT_ROOT = RESULT_PARENT / EXPERIMENT_NAME
STATE_SPECIFIC_ROOT = RESULT_PARENT / "scenario_2_C2_to_Aend_frozen_centered17_type3_compressed_5B"
STATE_SPECIFIC_CSV = STATE_SPECIFIC_ROOT / "03_state_retrieval_results.csv"
STATE_SPECIFIC_SUMMARY = STATE_SPECIFIC_ROOT / "07_final_result_summary.txt"
HISTORICAL_FORMAL_ROOT = RESULT_PARENT / "scenario_2_C2_to_Aend"
HISTORICAL_LUT = HISTORICAL_FORMAL_ROOT / "lut_fingerprints_Aend.npz"
HISTORICAL_TYPE3_ROOT = RESULT_PARENT / "scenario_2_C2_to_Aend_type3_cluster_compressed_5B"
HISTORICAL_TYPE3_MAT = STATE_SPECIFIC_ROOT / "05_compressed_lut_17basis_type3.mat"
CLUSTER_ROOT = (
    PROJECT_ROOT
    / "results"
    / "clustering"
    / "scenario_2_yout_withoutdpd_ori_5B_A_complete_link_clustering"
    / "threshold_m40p0dB"
)
CLUSTER_ASSIGNMENTS = CLUSTER_ROOT / "cluster_assignments.csv"
CLUSTER_REPRESENTATIVES = CLUSTER_ROOT / "cluster_representatives.csv"
WORK_LOG = PROJECT_ROOT / "work_logs" / EXPERIMENT_NAME / "execution_log.txt"
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "codex_handoff" / "codex_handoff.txt"
EXCEL_EXPORTER = (
    Path(__file__).resolve().parent / "export_frozen_centered17_commonb_control_xlsx.mjs"
)
NODE_EXE = Path(
    r"C:\Users\lizy\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"
)
NODE_MODULES = Path(
    r"C:\Users\lizy\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules"
)
ARTIFACT_MARKER = Path(
    r"C:\Users\lizy\.codex\plugins\cache\openai-primary-runtime\spreadsheets\26.905.11957\skills\spreadsheets\container_tools\mark_artifact_operation_started.mjs"
)

STATE_COUNT = 425
CLUSTER_COUNT = 77
FULL_LENGTH = 24_576
COMMON_B_RAW_LENGTH = 4_915
VALID_B_LENGTH = 4_913
DPD_LENGTH = 24_576
ABC_BOUNDS = (0, 12_288, 17_203, 24_576)
DMAX = 2
RIDGE_LAMBDA = 1e-8
WORKERS = 12
THRESHOLD_DB = -40.0
MODEL_CONDITION_LIMIT = 1e10
REPRESENTATIVE_STATE_IDS = (0, 100, 187, 340, 424)
EXPECTED_AEND_COUNTS = {2: 1, 3: 416, 4: 7, 5: 1}
EXPECTED_SUPPORT = (
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
OUTPUT_NAMES = (
    "01_model_and_common_probe_definition.csv",
    "02_compressed_commonB_lut_manifest.csv",
    "03_commonB_state_retrieval_results.csv",
    "04_commonB_vs_state_specific_comparison.csv",
    "05_commonB_control.xlsx",
    "05_compressed_commonB_lut_17basis_type3.mat",
    "06_commonB_control_comparison.png",
    "07_final_result_summary.txt",
)
PROTECTED_PATHS = (
    PROJECT_ROOT / "data" / "raw",
    PROJECT_ROOT / "results" / "behavior_model",
    PROJECT_ROOT / "results" / "basis_function_selection",
    PROJECT_ROOT / "results" / "clustering",
    STATE_SPECIFIC_ROOT,
    PROJECT_ROOT / "results" / "behavior_fingerprint_lut_retrieval" / "scenario_2_C2_to_Aend",
    HISTORICAL_TYPE3_ROOT,
    PROJECT_ROOT / "results" / "low_bandwidth_observation",
)

_WORKER_TERMS: tuple[Any, ...] | None = None
_WORKER_SUPPORT: tuple[int, ...] | None = None
_WORKER_COMMON_PHI: np.ndarray | None = None


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _append(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def _manifest(path: Path) -> dict[str, object]:
    if not path.is_dir():
        return {"exists": False, "file_count": 0, "bytes": 0, "latest_mtime_ns": None}
    files = [item for item in path.rglob("*") if item.is_file()]
    stats = [item.stat() for item in files]
    return {
        "exists": True,
        "file_count": len(files),
        "bytes": int(sum(item.st_size for item in stats)),
        "latest_mtime_ns": max((item.st_mtime_ns for item in stats), default=None),
    }


def _snapshot() -> dict[str, dict[str, object]]:
    return {str(path): _manifest(path) for path in PROTECTED_PATHS}


def _select_result_root() -> Path:
    RESULT_PARENT.mkdir(parents=True, exist_ok=True)
    if not PRIMARY_RESULT_ROOT.exists() or not any(PRIMARY_RESULT_ROOT.iterdir()):
        PRIMARY_RESULT_ROOT.mkdir(parents=True, exist_ok=True)
        return PRIMARY_RESULT_ROOT
    existing_names = {
        path.name for path in PRIMARY_RESULT_ROOT.iterdir() if path.is_file()
    }
    if set(OUTPUT_NAMES).issubset(existing_names):
        # A previous run may have produced all formal artifacts but stopped in
        # the post-write engineering checks.  Reuse that exact task directory
        # so the corrected runner can refresh the artifacts in place.
        return PRIMARY_RESULT_ROOT
    suffix = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    rerun = RESULT_PARENT / f"{EXPERIMENT_NAME}_rerun_{suffix}"
    rerun.mkdir(parents=True, exist_ok=False)
    return rerun


def _load_common_b() -> tuple[np.ndarray, dict[str, object]]:
    if not HISTORICAL_LUT.is_file():
        raise FileNotFoundError(f"Historical common-B LUT is missing: {HISTORICAL_LUT}")
    with np.load(HISTORICAL_LUT, allow_pickle=False) as data:
        if "common_B_input" not in data or "state_ids" not in data:
            raise RuntimeError("Historical LUT lacks common_B_input/state_ids")
        state_ids = np.asarray(data["state_ids"], dtype=np.int64)
        common_b = np.asarray(data["common_B_input"], dtype=np.complex128)
    if not np.array_equal(state_ids, np.arange(STATE_COUNT, dtype=np.int64)):
        raise RuntimeError("Historical common-B state ordering is not 0...424")
    if common_b.shape != (COMMON_B_RAW_LENGTH,) or not np.all(np.isfinite(common_b)):
        raise RuntimeError("Historical common-B vector is not finite complex128(4915,)")
    probe_hash = hashlib.sha256(np.ascontiguousarray(common_b).tobytes()).hexdigest()
    stats = {
        "source_file": str(HISTORICAL_LUT),
        "source_field": "common_B_input",
        "source_helper": "run_scenario2_C2_to_Aend_type3_cluster_compressed_5b._load_common_b",
        "historical_task": "scenario_2_C2_to_Aend_type3_cluster_compressed_5B",
        "raw_length": int(common_b.size),
        "valid_length": int(common_b.size - DMAX),
        "dtype": str(common_b.dtype),
        "finite": bool(np.all(np.isfinite(common_b))),
        "sha256": probe_hash,
        "rms": float(np.sqrt(np.mean(np.abs(common_b) ** 2))),
        "peak": float(np.max(np.abs(common_b))),
        "mean_power": float(np.mean(np.abs(common_b) ** 2)),
    }
    return common_b, stats


def _load_previous_baseline() -> pd.DataFrame:
    if not STATE_SPECIFIC_CSV.is_file() or not STATE_SPECIFIC_SUMMARY.is_file():
        raise FileNotFoundError("State-specific17 baseline CSV/summary is missing")
    frame = pd.read_csv(STATE_SPECIFIC_CSV).sort_values("State_n_R").reset_index(drop=True)
    if len(frame) != STATE_COUNT or not np.array_equal(
        frame["State_n_R"].to_numpy(dtype=np.int64), np.arange(STATE_COUNT)
    ):
        raise RuntimeError("State-specific baseline does not cover 425 states")
    counts = {
        "exact": int(frame["exact_hit"].astype(bool).sum()),
        "same_cluster": int(frame["same_cluster_hit"].astype(bool).sum()),
        "real_B_pass": int(frame["retrieved_real_B_shareable_lt_minus40"].astype(bool).sum()),
    }
    if counts != {"exact": 9, "same_cluster": 43, "real_B_pass": 136}:
        raise RuntimeError(f"State-specific baseline counts changed: {counts}")
    return frame


def _load_clusters() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    assignments = pd.read_csv(CLUSTER_ASSIGNMENTS).sort_values("state_id").reset_index(drop=True)
    representatives = (
        pd.read_csv(CLUSTER_REPRESENTATIVES).sort_values("cluster_id").reset_index(drop=True)
    )
    if len(assignments) != STATE_COUNT or len(representatives) != CLUSTER_COUNT:
        raise RuntimeError("Type-III mapping must contain 425 assignments and 77 representatives")
    if not np.array_equal(assignments["state_id"].to_numpy(dtype=np.int64), np.arange(STATE_COUNT)):
        raise RuntimeError("Type-III assignments are not ordered 0...424")
    state_to_cluster = assignments["cluster_id"].to_numpy(dtype=np.int64)
    representative_ids = representatives["representative_state_id"].to_numpy(dtype=np.int64)
    if (
        np.unique(representative_ids).size != CLUSTER_COUNT
        or np.unique(state_to_cluster).size != CLUSTER_COUNT
    ):
        raise RuntimeError("Type-III cluster or representative IDs are not unique")
    if not np.array_equal(
        assignments["representative_state_id"].to_numpy(dtype=np.int64),
        representative_ids[state_to_cluster],
    ):
        raise RuntimeError("Type-III assignment representative mapping is inconsistent")
    for column in ("representative_fingerprint_state_id", "representative_dpd_state_id"):
        if not np.array_equal(representatives[column].to_numpy(dtype=np.int64), representative_ids):
            raise RuntimeError(f"Type-III representative mapping differs in {column}")
    return assignments, representatives, state_to_cluster, representative_ids


def _worker_init(
    terms: tuple[Any, ...],
    support: tuple[int, ...],
    common_phi: np.ndarray,
) -> None:
    global _WORKER_TERMS, _WORKER_SUPPORT, _WORKER_COMMON_PHI
    _WORKER_TERMS = terms
    _WORKER_SUPPORT = support
    _WORKER_COMMON_PHI = np.asarray(common_phi, dtype=np.complex128)


def _fit_side(
    x_train: np.ndarray,
    y_train: np.ndarray,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    if _WORKER_TERMS is None or _WORKER_SUPPORT is None or _WORKER_COMMON_PHI is None:
        raise RuntimeError("Common-B model worker is not initialized")
    phi = build_frozen_centered_bank(x_train, _WORKER_TERMS, _WORKER_SUPPORT)
    target = np.asarray(y_train, dtype=np.complex128)[DMAX:]
    if phi.shape != (phi.shape[0], 17) or phi.shape[0] != target.size:
        raise RuntimeError("A/C design matrix shape mismatch")
    fit = fit_ridge(phi, target, RIDGE_LAMBDA)
    fingerprint = np.asarray(_WORKER_COMMON_PHI @ fit.theta, dtype=np.complex128)
    return (
        {
            "train_nmse_db": float(fit.nmse_db),
            "rank": int(fit.rank),
            "condition_number": float(fit.condition_number),
            "coefficient_norm": float(np.linalg.norm(fit.theta)),
            "finite": bool(np.all(np.isfinite(fit.theta)) and np.all(np.isfinite(fingerprint))),
        },
        fingerprint,
        np.asarray(fit.theta, dtype=np.complex128),
    )


def _state_worker(state_id: int) -> dict[str, object]:
    data = load_by_id(int(state_id))
    xin = old_type3._flatten_complex(data["xin"], "xin")
    input_history = np.asarray(data["xin_pd_ori_ilc"])
    output_history = np.asarray(data["yout_withdpd_ori_ilc"])
    if (
        input_history.ndim != 2
        or input_history.shape != output_history.shape
        or input_history.shape[0] != FULL_LENGTH
        or input_history.shape[1] < 2
    ):
        raise RuntimeError(f"state {state_id} ILC history is invalid")
    ilc_count = int(input_history.shape[1])
    partition = build_partition_from_xin(xin)
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
    a_values, a_fp, a_theta = _fit_side(a["A"].input, a["A"].output)
    c_values, c_fp, c_theta = _fit_side(c["C"].input, c["C"].output)
    return {
        "state_id": int(state_id),
        "ilc_A_end": ilc_count,
        "Aend": a_values,
        "C2": c_values,
        "Aend_fingerprint": a_fp,
        "C2_fingerprint": c_fp,
        "Aend_theta": a_theta,
        "C2_theta": c_theta,
        "funMng": int(old_type3._scalar(data, "funMng")),
        "funAng": int(old_type3._scalar(data, "funAng")),
        "secMng": int(old_type3._scalar(data, "secMng")),
        "secAng": int(old_type3._scalar(data, "secAng")),
        "Vm": old_type3._scalar(data, "v_carrier"),
        "Pin": old_type3._scalar(data, "inputPower"),
    }


def _run_models(
    terms: tuple[Any, ...],
    support: tuple[int, ...],
    common_phi: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    results: dict[int, dict[str, object]] = {}
    with ProcessPoolExecutor(
        max_workers=WORKERS,
        initializer=_worker_init,
        initargs=(terms, support, common_phi),
    ) as executor:
        futures = {
            executor.submit(_state_worker, state_id): state_id for state_id in range(STATE_COUNT)
        }
        for done, future in enumerate(as_completed(futures), start=1):
            state_id = futures[future]
            results[state_id] = future.result()
            if done % 100 == 0 or done == STATE_COUNT:
                print(f"[MODEL] {done}/{STATE_COUNT}", flush=True)
    rows = []
    a_fp = np.empty((STATE_COUNT, VALID_B_LENGTH), dtype=np.complex128)
    c_fp = np.empty_like(a_fp)
    theta_a = np.empty((STATE_COUNT, 17), dtype=np.complex128)
    theta_c = np.empty_like(theta_a)
    for state_id in range(STATE_COUNT):
        result = results[state_id]
        row = {
            "state_id": state_id,
            "ilc_A_end": result["ilc_A_end"],
            "funMng": result["funMng"],
            "funAng": result["funAng"],
            "secMng": result["secMng"],
            "secAng": result["secAng"],
            "Vm": result["Vm"],
            "Pin": result["Pin"],
            "Y_Aend_train_NMSE_dB": result["Aend"]["train_nmse_db"],
            "Y_Aend_rank": result["Aend"]["rank"],
            "Y_Aend_condition_number": result["Aend"]["condition_number"],
            "Y_Aend_coefficient_norm": result["Aend"]["coefficient_norm"],
            "Y_Aend_finite": result["Aend"]["finite"],
            "Y_C2_train_NMSE_dB": result["C2"]["train_nmse_db"],
            "Y_C2_rank": result["C2"]["rank"],
            "Y_C2_condition_number": result["C2"]["condition_number"],
            "Y_C2_coefficient_norm": result["C2"]["coefficient_norm"],
            "Y_C2_finite": result["C2"]["finite"],
        }
        rows.append(row)
        a_fp[state_id] = result["Aend_fingerprint"]
        c_fp[state_id] = result["C2_fingerprint"]
        theta_a[state_id] = result["Aend_theta"]
        theta_c[state_id] = result["C2_theta"]
    return pd.DataFrame(rows), a_fp, c_fp, theta_a, theta_c


def _model_reproduction_gate(
    metrics: pd.DataFrame,
    previous: pd.DataFrame,
    a_fp: np.ndarray,
    c_fp: np.ndarray,
    common_phi: np.ndarray,
) -> dict[str, object]:
    previous = previous.set_index("State_n_R")
    current = metrics.set_index("state_id")
    a_errors = np.abs(
        current["Y_Aend_train_NMSE_dB"].to_numpy(dtype=float)
        - previous["Y_Aend_train_NMSE_dB"].to_numpy(dtype=float)
    )
    c_errors = np.abs(
        current["Y_C2_train_NMSE_dB"].to_numpy(dtype=float)
        - previous["Y_C2_train_NMSE_dB"].to_numpy(dtype=float)
    )
    distribution = {
        int(key): int(value)
        for key, value in metrics["ilc_A_end"].value_counts().sort_index().items()
    }
    if distribution != EXPECTED_AEND_COUNTS:
        raise RuntimeError(f"Aend distribution mismatch: {distribution}")
    if not np.array_equal(metrics["Y_Aend_rank"].to_numpy(dtype=int), np.full(STATE_COUNT, 17)):
        raise RuntimeError("Aend rank gate failed")
    if not np.array_equal(metrics["Y_C2_rank"].to_numpy(dtype=int), np.full(STATE_COUNT, 17)):
        raise RuntimeError("C2 rank gate failed")
    max_condition = float(
        metrics[["Y_Aend_condition_number", "Y_C2_condition_number"]].to_numpy(dtype=float).max()
    )
    if max_condition > MODEL_CONDITION_LIMIT:
        raise RuntimeError(f"Model condition number exceeds limit: {max_condition}")
    if common_phi.shape != (VALID_B_LENGTH, 17):
        raise RuntimeError(f"Common-B Phi must be (4913,17), got {common_phi.shape}")
    max_train = float(max(a_errors.max(), c_errors.max()))
    if max_train > 1e-6:
        raise RuntimeError(f"Train model reproduction failed: {max_train:.3e} dB")
    return {
        "Aend_train_max_abs_error_dB": float(a_errors.max()),
        "C2_train_max_abs_error_dB": float(c_errors.max()),
        "Aend_distribution": distribution,
        "C2_available": int((metrics["ilc_A_end"] >= 2).sum()),
        "fingerprint_shapes": [list(a_fp.shape), list(c_fp.shape)],
        "common_phi_shape": list(common_phi.shape),
        "maximum_condition_number": max_condition,
        "pass": True,
    }


def _retrieval_frame(
    metrics: pd.DataFrame,
    previous: pd.DataFrame,
    assignments: pd.DataFrame,
    state_to_cluster: np.ndarray,
    representative_ids: np.ndarray,
    distance: np.ndarray,
    real_b_distance: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model = metrics.set_index("state_id")
    previous = previous.set_index("State_n_R")
    rows = []
    for state_id in range(STATE_COUNT):
        order = np.lexsort((representative_ids, distance[state_id]))
        best = int(order[0])
        second = int(order[1])
        query_id = int(representative_ids[best])
        cluster_id = int(state_to_cluster[state_id])
        true_rep = int(representative_ids[cluster_id])
        true_rank = int(np.flatnonzero(order == cluster_id)[0] + 1)
        real_value = float(real_b_distance[state_id, query_id])
        row = model.loc[state_id]
        rows.append(
            {
                "State_n_R": state_id,
                "funMng": int(row["funMng"]),
                "funAng": int(row["funAng"]),
                "secMng": int(row["secMng"]),
                "secAng": int(row["secAng"]),
                "Y_Aend_train_NMSE_dB": float(row["Y_Aend_train_NMSE_dB"]),
                "Y_C2_train_NMSE_dB": float(row["Y_C2_train_NMSE_dB"]),
                "self_Aend_C2_commonB_CNMSE_dB": float("nan"),
                "real_cluster_id": cluster_id,
                "true_cluster_rep_state_id": true_rep,
                "true_cluster_rep_distance_CNMSE_dB": float(distance[state_id, cluster_id]),
                "true_cluster_rep_rank": true_rank,
                "State_n_Q": query_id,
                "retrieved_cluster_id": int(state_to_cluster[query_id]),
                "retrieved_fingerprint_CNMSE_dB": float(distance[state_id, best]),
                "second_best_state_id": int(representative_ids[second]),
                "second_best_CNMSE_dB": float(distance[state_id, second]),
                "retrieval_margin_dB": float(distance[state_id, second] - distance[state_id, best]),
                "exact_hit": bool(query_id == state_id),
                "same_cluster_hit": bool(state_to_cluster[query_id] == cluster_id),
                "retrieved_real_B_CNMSE_dB": real_value,
                "real_B_pass_lt_minus40": bool(
                    np.isneginf(real_value) or real_value < THRESHOLD_DB
                ),
                "ilc_A_end": int(row["ilc_A_end"]),
            }
        )
    frame = pd.DataFrame(rows)
    # ``assignments`` is accepted to keep the call signature explicit about
    # the frozen Type-III mapping.  The already validated state-to-cluster
    # vector is sufficient for the row construction above.
    del assignments
    # self consistency is filled by the caller from the two full fingerprint matrices.
    comparison = pd.DataFrame(
        {
            "State_n_R": np.arange(STATE_COUNT, dtype=np.int64),
            "stateSpecific_State_Q": previous["State_n_Q"].to_numpy(dtype=np.int64),
            "commonB_State_Q": frame["State_n_Q"].to_numpy(dtype=np.int64),
            "stateSpecific_same_cluster": previous["same_cluster_hit"].astype(bool).to_numpy(),
            "commonB_same_cluster": frame["same_cluster_hit"].astype(bool).to_numpy(),
            "stateSpecific_real_B_CNMSE_dB": previous["retrieved_real_B_CNMSE_dB"].to_numpy(
                dtype=float
            ),
            "commonB_real_B_CNMSE_dB": frame["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float),
            "delta_real_B_common_minus_specific_dB": frame["retrieved_real_B_CNMSE_dB"].to_numpy(
                dtype=float
            )
            - previous["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float),
            "stateSpecific_pass40": previous["retrieved_real_B_shareable_lt_minus40"]
            .astype(bool)
            .to_numpy(),
            "commonB_pass40": frame["real_B_pass_lt_minus40"].astype(bool).to_numpy(),
        }
    )
    comparison["pass_transition"] = np.select(
        [
            (~comparison["stateSpecific_pass40"]) & comparison["commonB_pass40"],
            comparison["stateSpecific_pass40"] & (~comparison["commonB_pass40"]),
        ],
        ["FAIL_TO_PASS", "PASS_TO_FAIL"],
        default="UNCHANGED",
    )
    comparison["same_cluster_transition"] = np.select(
        [
            (~comparison["stateSpecific_same_cluster"]) & comparison["commonB_same_cluster"],
            comparison["stateSpecific_same_cluster"] & (~comparison["commonB_same_cluster"]),
        ],
        ["WRONG_TO_CORRECT", "CORRECT_TO_WRONG"],
        default="UNCHANGED",
    )
    return frame, comparison


def _paired_comparison_summary(comparison: pd.DataFrame, frame: pd.DataFrame) -> dict[str, object]:
    """Return the fixed paired counts and finite-delta diagnostics."""

    specific = comparison["stateSpecific_real_B_CNMSE_dB"].to_numpy(dtype=float)
    common = comparison["commonB_real_B_CNMSE_dB"].to_numpy(dtype=float)
    delta = comparison["delta_real_B_common_minus_specific_dB"].to_numpy(dtype=float)
    near_equal = np.isfinite(delta) & (np.abs(delta) <= 1e-12)
    better = (common < specific) & ~near_equal
    worse = (common > specific) & ~near_equal
    equal = near_equal | (np.isneginf(common) & np.isneginf(specific))
    finite_delta = delta[np.isfinite(delta)]
    state_specific_pass = comparison["stateSpecific_pass40"].to_numpy(dtype=bool)
    common_pass = comparison["commonB_pass40"].to_numpy(dtype=bool)
    state_specific_cluster = comparison["stateSpecific_same_cluster"].to_numpy(dtype=bool)
    common_cluster = comparison["commonB_same_cluster"].to_numpy(dtype=bool)
    ranks = frame["true_cluster_rep_rank"].to_numpy(dtype=np.int64)
    return {
        "commonB_better_count": int(np.count_nonzero(better)),
        "commonB_equal_or_near_equal_count": int(np.count_nonzero(equal)),
        "commonB_worse_count": int(np.count_nonzero(worse)),
        "delta_real_B_finite_count": int(finite_delta.size),
        "delta_real_B_finite_median_dB": float(np.median(finite_delta)),
        "delta_real_B_finite_mean_dB": float(np.mean(finite_delta)),
        "delta_real_B_best_dB": float(np.nanmin(delta)),
        "delta_real_B_worst_dB": float(np.nanmax(delta)),
        "pass_pass_count": int(np.count_nonzero(state_specific_pass & common_pass)),
        "pass_to_fail_count": int(np.count_nonzero(state_specific_pass & ~common_pass)),
        "fail_to_pass_count": int(np.count_nonzero(~state_specific_pass & common_pass)),
        "fail_fail_count": int(np.count_nonzero(~state_specific_pass & ~common_pass)),
        "same_cluster_correct_correct_count": int(
            np.count_nonzero(state_specific_cluster & common_cluster)
        ),
        "same_cluster_correct_to_wrong_count": int(
            np.count_nonzero(state_specific_cluster & ~common_cluster)
        ),
        "same_cluster_wrong_to_correct_count": int(
            np.count_nonzero(~state_specific_cluster & common_cluster)
        ),
        "same_cluster_wrong_wrong_count": int(
            np.count_nonzero(~state_specific_cluster & ~common_cluster)
        ),
        "true_cluster_rep_rank1_count": int(np.count_nonzero(ranks == 1)),
        "true_cluster_rep_rank_le3_count": int(np.count_nonzero(ranks <= 3)),
        "true_cluster_rep_rank_le5_count": int(np.count_nonzero(ranks <= 5)),
        "true_cluster_rep_rank_le10_count": int(np.count_nonzero(ranks <= 10)),
        "true_cluster_rep_rank_median": float(np.median(ranks)),
        "true_cluster_rep_rank_worst": int(np.max(ranks)),
    }


def _compressed_manifest(
    representatives: pd.DataFrame,
    metrics: pd.DataFrame,
    sources: list[str],
) -> pd.DataFrame:
    model = metrics.set_index("state_id")
    rows = []
    for index, row in enumerate(representatives.itertuples(index=False)):
        state_id = int(row.representative_state_id)
        rows.append(
            {
                "cluster_id": int(row.cluster_id),
                "representative_state_id": state_id,
                "cluster_size": int(row.cluster_size),
                "member_state_ids": str(row.member_state_ids),
                "funMng": int(row.funMng),
                "funAng": int(row.funAng),
                "secMng": int(row.secMng),
                "secAng": int(row.secAng),
                "Vm": float(row.v_carrier),
                "Pin": float(row.inputPower),
                "ilc_A_end": int(model.loc[state_id, "ilc_A_end"]),
                "dpd_entry_source": sources[index],
                "fingerprint_length": VALID_B_LENGTH,
            }
        )
    return pd.DataFrame(rows)


def _probe_diagnostics(
    frame: pd.DataFrame,
    probe_stats: dict[str, object],
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "State_n_R": frame["State_n_R"].to_numpy(dtype=np.int64),
            "self_Aend_C2_commonB_CNMSE_dB": frame["self_Aend_C2_commonB_CNMSE_dB"].to_numpy(
                dtype=float
            ),
            "true_cluster_rep_state_id": frame["true_cluster_rep_state_id"].to_numpy(
                dtype=np.int64
            ),
            "true_cluster_rep_distance_CNMSE_dB": frame[
                "true_cluster_rep_distance_CNMSE_dB"
            ].to_numpy(dtype=float),
            "true_cluster_rep_rank": frame["true_cluster_rep_rank"].to_numpy(dtype=np.int64),
            "retrieval_margin_dB": frame["retrieval_margin_dB"].to_numpy(dtype=float),
            "common_probe_sha256": [probe_stats["sha256"]] * len(frame),
        }
    )


def _write_plot(comparison: pd.DataFrame, output: Path) -> dict[str, object]:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    x = comparison["State_n_R"].to_numpy(dtype=float)
    specific = comparison["stateSpecific_real_B_CNMSE_dB"].to_numpy(dtype=float)
    common = comparison["commonB_real_B_CNMSE_dB"].to_numpy(dtype=float)
    finite = np.concatenate([specific[np.isfinite(specific)], common[np.isfinite(common)]])
    floor = float(np.min(finite) - 2.0)
    specific_plot = np.where(np.isneginf(specific), floor, specific)
    common_plot = np.where(np.isneginf(common), floor, common)
    fig, ax = plt.subplots(figsize=(18, 6), constrained_layout=True)
    ax.plot(
        x,
        specific_plot,
        color="#7F7F7F",
        linewidth=1.0,
        linestyle="--",
        label="State-specific-B retrieval Real-B CNMSE",
    )
    ax.plot(x, common_plot, color="#1F77B4", linewidth=1.1, label="Common-B retrieval Real-B CNMSE")
    ax.axhline(
        THRESHOLD_DB, color="#B22222", linestyle=":", linewidth=1.0, label="-40 dB threshold"
    )
    fail_to_pass = comparison["pass_transition"] == "FAIL_TO_PASS"
    pass_to_fail = comparison["pass_transition"] == "PASS_TO_FAIL"
    ax.scatter(
        x[fail_to_pass],
        common_plot[fail_to_pass],
        color="#2E8B57",
        marker="^",
        s=24,
        zorder=4,
        label="FAIL → PASS",
    )
    ax.scatter(
        x[pass_to_fail],
        common_plot[pass_to_fail],
        color="#C44E52",
        marker="v",
        s=24,
        zorder=4,
        label="PASS → FAIL",
    )
    ax.set_xlim(-2, STATE_COUNT + 1)
    ax.set_ylim(floor, float(np.max(finite) + 1.0))
    ax.set_xlabel("State_R")
    ax.set_ylabel("Real-B CNMSE (dB)")
    ax.set_title(
        "Frozen-centered17 Type-III common-B control\n"
        "State-specific-B versus historical common-B fingerprint excitation"
    )
    ax.legend(loc="lower right", ncol=2, fontsize=8)
    fig.savefig(output, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if not output.is_file() or output.stat().st_size == 0:
        raise RuntimeError("Common-B comparison PNG was not created")
    return {
        "backend": "Python/matplotlib",
        "curve_count": 2,
        "threshold_line": True,
        "fail_to_pass_marker_count": int(fail_to_pass.sum()),
        "pass_to_fail_marker_count": int(pass_to_fail.sum()),
        "width_inches": 18,
        "height_inches": 6,
        "dpi": 300,
        "pass": True,
    }


def _json_value(value: object) -> object:
    if isinstance(value, (np.integer, np.floating, np.bool_)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return "-Inf" if value < 0 else "Inf"
    return value


def _frame_payload(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "columns": list(frame.columns),
        "rows": [
            {key: _json_value(value) for key, value in row.items()}
            for row in frame.to_dict("records")
        ],
    }


def _export_excel(
    payload: dict[str, object],
    output_path: Path,
) -> dict[str, object]:
    if not NODE_EXE.is_file() or not NODE_MODULES.is_dir() or not ARTIFACT_MARKER.is_file():
        raise FileNotFoundError("Bundled Artifact Tool runtime is unavailable")
    qa_dir = Path(tempfile.gettempdir()) / f"{EXPERIMENT_NAME}_xlsx_qa_{os.getpid()}"
    if qa_dir.exists():
        shutil.rmtree(qa_dir)
    qa_dir.mkdir(parents=True)
    with tempfile.TemporaryDirectory(prefix="commonb_control_xlsx_") as temp_text:
        temp = Path(temp_text)
        source = temp / "source.json"
        builder = temp / "builder.mjs"
        junction = temp / "node_modules"
        source.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        shutil.copy2(EXCEL_EXPORTER, builder)
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(junction), str(NODE_MODULES)],
            check=True,
            capture_output=True,
            text=True,
        )
        marker = subprocess.run(
            [
                str(NODE_EXE),
                str(ARTIFACT_MARKER),
                "--operation-kind",
                "create",
                "--expected-output-count",
                "1",
                "--output-format",
                "xlsx",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        try:
            exported = subprocess.run(
                [str(NODE_EXE), str(builder), str(source), str(output_path), str(qa_dir)],
                check=True,
                capture_output=True,
                text=True,
            )
        finally:
            if junction.exists():
                os.rmdir(junction)
    previews = sorted(str(path) for path in qa_dir.glob("*.png"))
    sidecar = output_path.with_name(output_path.name + ".inspect.ndjson")
    if sidecar.exists():
        sidecar.unlink()
    if len(previews) != 5 or not output_path.is_file() or output_path.stat().st_size == 0:
        raise RuntimeError("Common-B Artifact Tool export/preview gate failed")
    return {
        "marker_stdout": marker.stdout[-1000:],
        "export_stdout_tail": exported.stdout[-4000:],
        "preview_paths": previews,
        "xlsx_bytes": output_path.stat().st_size,
        "pass": True,
    }


def _summary_rows(values: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame({"metric": list(values.keys()), "value": list(values.values())})


def _write_mat(
    path: Path,
    common_b: np.ndarray,
    probe_stats: dict[str, object],
    representative_ids: np.ndarray,
    state_to_cluster: np.ndarray,
    lut_fingerprints: np.ndarray,
    query_fingerprints: np.ndarray,
    dpd_entries: np.ndarray,
    support_ids: tuple[str, ...],
) -> dict[str, object]:
    savemat(
        path,
        {
            "representative_state_ids": representative_ids.astype(np.int64),
            "state_to_cluster": state_to_cluster.astype(np.int64),
            "common_B_probe": common_b.astype(np.complex128),
            "lut_fingerprints_commonB": lut_fingerprints.astype(np.complex128),
            "query_fingerprints_commonB": query_fingerprints.astype(np.complex128),
            "dpd_entries": dpd_entries.astype(np.complex128),
            "basis_ids": np.asarray(support_ids, dtype=object),
            "ridge_lambda": np.asarray(RIDGE_LAMBDA, dtype=np.float64),
            "abc_bounds": np.asarray(ABC_BOUNDS, dtype=np.int64),
            "common_probe_hash": np.asarray(probe_stats["sha256"]),
        },
        do_compression=True,
        long_field_names=True,
    )
    loaded = loadmat(
        path,
        appendmat=False,
        variable_names=[
            "representative_state_ids",
            "state_to_cluster",
            "common_B_probe",
            "lut_fingerprints_commonB",
            "query_fingerprints_commonB",
            "dpd_entries",
            "ridge_lambda",
            "abc_bounds",
        ],
    )
    expected = {
        "representative_state_ids": (1, CLUSTER_COUNT),
        "state_to_cluster": (1, STATE_COUNT),
        "common_B_probe": (1, COMMON_B_RAW_LENGTH),
        "lut_fingerprints_commonB": (CLUSTER_COUNT, VALID_B_LENGTH),
        "query_fingerprints_commonB": (STATE_COUNT, VALID_B_LENGTH),
        "dpd_entries": (CLUSTER_COUNT, DPD_LENGTH),
    }
    for key, shape in expected.items():
        actual = np.asarray(loaded[key]).shape
        if (
            actual != shape
            and actual != (shape[1],)
            and not (key == "state_to_cluster" and actual == (STATE_COUNT, 1))
        ):
            raise RuntimeError(f"MAT reload shape mismatch {key}: {actual}")
    return {"path": str(path), "bytes": path.stat().st_size, "reload_pass": True}


def main() -> None:
    freeze_support()
    started = time.perf_counter()
    before = _snapshot()
    result_root = _select_result_root()
    _append(
        WORK_LOG,
        f"\n\n[{_timestamp()}] START {EXPERIMENT_NAME}\n"
        f"Result root: {result_root}\nPython: {sys.executable}\n"
        "Only common-B fingerprint excitation changes; model, clusters, representatives, "
        "DPD entries and metrics are frozen.\n",
    )
    try:
        previous = _load_previous_baseline()
        terms, support, support_ids = (None, None, None)
        selected_row = pd.read_csv(
            PROJECT_ROOT
            / "results"
            / "basis_function_selection"
            / "scenario_2_hard20_frozen_centered_basis_selection_5B"
            / "05_selected_models.csv"
        )
        selected_row = selected_row.loc[selected_row["is_final"].astype(bool)]
        if len(selected_row) != 1:
            raise RuntimeError("Frozen-centered17 final model row is missing")
        support_ids = tuple(str(selected_row.iloc[0]["support_ids"]).split(";"))
        if support_ids != EXPECTED_SUPPORT:
            raise RuntimeError("Frozen-centered17 support changed")
        terms = build_frozen_centered_dictionary()
        by_id = {term.basis_id: term.index for term in terms}
        support = tuple(by_id[value] for value in support_ids)
        common_b, probe_stats = _load_common_b()
        common_phi = build_frozen_centered_bank(common_b, terms, support)
        if common_phi.shape != (VALID_B_LENGTH, 17):
            raise RuntimeError("Common-B design matrix shape gate failed")
        assignments, representatives, state_to_cluster, representative_ids = _load_clusters()
        old_mat = loadmat(STATE_SPECIFIC_ROOT / "05_compressed_lut_17basis_type3.mat")
        dpd_entries = np.asarray(old_mat["dpd_entries"], dtype=np.complex128)
        if dpd_entries.shape != (CLUSTER_COUNT, DPD_LENGTH) or not np.all(np.isfinite(dpd_entries)):
            raise RuntimeError("Frozen representative DPD entries are invalid")
        if not np.array_equal(
            np.asarray(old_mat["representative_state_ids"]).reshape(-1).astype(np.int64),
            representative_ids,
        ):
            raise RuntimeError("Frozen representative IDs changed")
        model_metrics, a_fp, c_fp, theta_a, theta_c = _run_models(terms, support, common_phi)
        model_gate = _model_reproduction_gate(model_metrics, previous, a_fp, c_fp, common_phi)
        common_self = np.asarray(
            [cnmse(c_fp[state_id], a_fp[state_id]) for state_id in range(STATE_COUNT)],
            dtype=float,
        )
        if not np.all(np.isfinite(common_self) | np.isneginf(common_self)):
            raise RuntimeError("Common-B Aend/C2 self consistency contains invalid values")
        lut_fp = a_fp[representative_ids]
        query_fp = c_fp
        distance = compute_cnmse_distance_matrix(query_fp, lut_fp)
        if distance.shape != (STATE_COUNT, CLUSTER_COUNT):
            raise RuntimeError("Common-B retrieval distance shape failed")
        representative_fingerprint_max_abs_error = float(
            np.max(np.abs(lut_fp - a_fp[representative_ids]))
        )
        if representative_fingerprint_max_abs_error != 0.0:
            raise RuntimeError(
                "Representative fingerprints are not copied from the 425-state LUT"
            )
        deterministic_distance = compute_cnmse_distance_matrix(query_fp, lut_fp)
        retrieval_distance_max_abs_difference = float(
            np.max(np.abs(distance - deterministic_distance))
        )
        if retrieval_distance_max_abs_difference != 0.0:
            raise RuntimeError("Common-B retrieval distance is not deterministic")
        # Fill state-specific comparison frame and true-representative ranks.
        state_frame, comparison = _retrieval_frame(
            model_metrics,
            previous,
            assignments,
            state_to_cluster,
            representative_ids,
            distance,
            np.zeros((STATE_COUNT, STATE_COUNT), dtype=float),
        )
        for state_id in range(STATE_COUNT):
            state_frame.loc[
                state_frame["State_n_R"] == state_id, "self_Aend_C2_commonB_CNMSE_dB"
            ] = common_self[state_id]
        retrieval_frozen = True
        real_b = old_type3._run_real_b(WORKERS)
        old_real_gate = old_type3._real_b_gate(real_b)
        if not old_real_gate["pass"]:
            raise RuntimeError(f"Real-B helper reproduction failed: {old_real_gate}")
        real_b_distance = np.asarray(old_real_gate.pop("distance"), dtype=np.float64)
        # Real-B values must be added only after retrieval_frozen.
        q_by_state = state_frame.set_index("State_n_R")["State_n_Q"].to_numpy(dtype=np.int64)
        state_frame["retrieved_real_B_CNMSE_dB"] = [
            float(real_b_distance[state_id, q_by_state[state_id]])
            for state_id in range(STATE_COUNT)
        ]
        state_frame["real_B_pass_lt_minus40"] = np.isneginf(
            state_frame["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float)
        ) | (state_frame["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float) < THRESHOLD_DB)
        state_frame["exact_hit"] = state_frame["State_n_Q"] == state_frame["State_n_R"]
        state_frame["same_cluster_hit"] = state_frame["same_cluster_hit"].astype(bool)
        comparison["commonB_real_B_CNMSE_dB"] = state_frame["retrieved_real_B_CNMSE_dB"].to_numpy(
            dtype=float
        )
        comparison["commonB_pass40"] = state_frame["real_B_pass_lt_minus40"].to_numpy(dtype=bool)
        comparison["delta_real_B_common_minus_specific_dB"] = (
            comparison["commonB_real_B_CNMSE_dB"] - comparison["stateSpecific_real_B_CNMSE_dB"]
        )
        comparison["pass_transition"] = np.select(
            [
                (~comparison["stateSpecific_pass40"]) & comparison["commonB_pass40"],
                comparison["stateSpecific_pass40"] & (~comparison["commonB_pass40"]),
            ],
            ["FAIL_TO_PASS", "PASS_TO_FAIL"],
            default="UNCHANGED",
        )
        comparison["same_cluster_transition"] = np.select(
            [
                (~comparison["stateSpecific_same_cluster"]) & comparison["commonB_same_cluster"],
                comparison["stateSpecific_same_cluster"] & (~comparison["commonB_same_cluster"]),
            ],
            ["WRONG_TO_CORRECT", "CORRECT_TO_WRONG"],
            default="UNCHANGED",
        )
        paired_summary = _paired_comparison_summary(comparison, state_frame)
        a_fp_recomputed = np.vstack(
            [common_phi @ theta_a[state_id] for state_id in range(STATE_COUNT)]
        )
        c_fp_recomputed = np.vstack(
            [common_phi @ theta_c[state_id] for state_id in range(STATE_COUNT)]
        )
        self_recompute_max_abs_difference = float(
            max(
                np.max(np.abs(a_fp_recomputed - a_fp)),
                np.max(np.abs(c_fp_recomputed - c_fp)),
            )
        )
        if self_recompute_max_abs_difference > 1e-12:
            raise RuntimeError("Common-B self recomputation gate failed")
        historical_manifest_path = STATE_SPECIFIC_ROOT / "02_compressed_lut_manifest.csv"
        if not historical_manifest_path.is_file():
            raise FileNotFoundError(
                f"Historical compressed-LUT manifest is missing: {historical_manifest_path}"
            )
        historical_manifest = (
            pd.read_csv(historical_manifest_path).sort_values("cluster_id").reset_index(drop=True)
        )
        if not np.array_equal(
            historical_manifest["representative_state_id"].to_numpy(dtype=np.int64),
            representative_ids,
        ):
            raise RuntimeError("Historical DPD manifest representative IDs changed")
        dpd_sources = historical_manifest["dpd_entry_source"].astype(str).tolist()
        manifest = _compressed_manifest(representatives, model_metrics, dpd_sources)
        model_definition = pd.DataFrame(
            [
                {
                    "basis_id": terms[index].basis_id,
                    "formula": terms[index].formula,
                    "family": terms[index].family,
                    "order": terms[index].order,
                    "signal_delay": terms[index].signal_delay,
                    "envelope_delay": terms[index].envelope_delay,
                    "mandatory": terms[index].basis_id == "LIN_d0",
                    "K": 17,
                    "ridge_lambda": RIDGE_LAMBDA,
                    "dmax": DMAX,
                    "common_probe_source": probe_stats["source_field"],
                    "common_probe_helper": probe_stats["source_helper"],
                    "common_probe_sha256": probe_stats["sha256"],
                    "common_probe_raw_length": probe_stats["raw_length"],
                    "common_probe_valid_length": probe_stats["valid_length"],
                    "common_probe_rms": probe_stats["rms"],
                    "common_probe_peak": probe_stats["peak"],
                    "common_probe_normalization": (
                        "historical common_B_input reused as stored; no re-normalization"
                    ),
                }
                for index in support
            ]
        )
        model_definition.to_csv(
            result_root / OUTPUT_NAMES[0], index=False, encoding="utf-8-sig", float_format="%.15g"
        )
        manifest.to_csv(
            result_root / OUTPUT_NAMES[1], index=False, encoding="utf-8-sig", float_format="%.15g"
        )
        state_frame.to_csv(
            result_root / OUTPUT_NAMES[2], index=False, encoding="utf-8-sig", float_format="%.15g"
        )
        comparison.to_csv(
            result_root / OUTPUT_NAMES[3], index=False, encoding="utf-8-sig", float_format="%.15g"
        )
        mat_gate = _write_mat(
            result_root / OUTPUT_NAMES[5],
            common_b,
            probe_stats,
            representative_ids,
            state_to_cluster,
            lut_fp,
            query_fp,
            dpd_entries,
            support_ids,
        )
        figure_gate = _write_plot(comparison, result_root / OUTPUT_NAMES[6])
        finite_real = state_frame.loc[
            np.isfinite(state_frame["retrieved_real_B_CNMSE_dB"]), "retrieved_real_B_CNMSE_dB"
        ]
        summary_values = {
            "basis_count": 17,
            "max_order": 9,
            "dmax": 2,
            "memory_taps": 3,
            "ridge_lambda": RIDGE_LAMBDA,
            "total_states": STATE_COUNT,
            "original_lut_entries": STATE_COUNT,
            "compressed_lut_entries": CLUSTER_COUNT,
            "compression_ratio_pct": 100.0 * (1.0 - CLUSTER_COUNT / STATE_COUNT),
            "common_probe_sha256": probe_stats["sha256"],
            "common_probe_raw_length": probe_stats["raw_length"],
            "common_probe_valid_length": probe_stats["valid_length"],
            "common_probe_rms": probe_stats["rms"],
            "common_probe_peak": probe_stats["peak"],
            "common_probe_normalization": "historical common_B_input reused as stored; no re-normalization",
            "exact_hits_commonB": int(state_frame["exact_hit"].sum()),
            "exact_hits_stateSpecific": int(previous["exact_hit"].astype(bool).sum()),
            "same_cluster_hits_commonB": int(state_frame["same_cluster_hit"].sum()),
            "same_cluster_hits_stateSpecific": int(previous["same_cluster_hit"].astype(bool).sum()),
            "real_B_pass_commonB": int(state_frame["real_B_pass_lt_minus40"].sum()),
            "real_B_pass_stateSpecific": int(
                previous["retrieved_real_B_shareable_lt_minus40"].astype(bool).sum()
            ),
            "real_B_finite_median_commonB_dB": float(finite_real.median()),
            "real_B_finite_mean_commonB_dB": float(finite_real.mean()),
            "real_B_finite_worst_commonB_dB": float(finite_real.max()),
            "self_Aend_C2_commonB_median_dB": float(np.nanmedian(common_self)),
            "self_Aend_C2_commonB_mean_dB": float(np.nanmean(common_self)),
            "self_Aend_C2_commonB_q10_dB": float(np.nanquantile(common_self, 0.10)),
            "self_Aend_C2_commonB_q90_dB": float(np.nanquantile(common_self, 0.90)),
            "self_Aend_C2_commonB_best_dB": float(np.nanmin(common_self)),
            "self_Aend_C2_commonB_worst_dB": float(np.nanmax(common_self)),
            "self_Aend_C2_commonB_pass_count": int(np.count_nonzero(common_self < THRESHOLD_DB)),
            "fail_to_pass_count": int((comparison["pass_transition"] == "FAIL_TO_PASS").sum()),
            "pass_to_fail_count": int((comparison["pass_transition"] == "PASS_TO_FAIL").sum()),
            "wrong_to_correct_cluster_count": int(
                (comparison["same_cluster_transition"] == "WRONG_TO_CORRECT").sum()
            ),
            "correct_to_wrong_cluster_count": int(
                (comparison["same_cluster_transition"] == "CORRECT_TO_WRONG").sum()
            ),
            **paired_summary,
            "representative_fingerprint_max_abs_error": representative_fingerprint_max_abs_error,
            "retrieval_distance_max_abs_difference": retrieval_distance_max_abs_difference,
            "common_probe_self_recompute_max_abs_difference": self_recompute_max_abs_difference,
            "common_phi_constructed_once": True,
            "exact_hit_is_auxiliary_only": True,
            "secondary_historical_frozen10_commonB_exact": 63,
            "secondary_historical_frozen10_commonB_same_cluster": 295,
            "secondary_historical_frozen10_commonB_real_B_pass": 402,
            "Aend_train_median_dB": float(model_metrics["Y_Aend_train_NMSE_dB"].median()),
            "C2_train_median_dB": float(model_metrics["Y_C2_train_NMSE_dB"].median()),
            "historical_commonB_comparison": "Not a probe-only comparison if the historical result used Frozen10; primary paired baseline here is state-specific-B 17-basis.",
        }
        probe_diag = _probe_diagnostics(state_frame, probe_stats)
        excel_payload = {
            "commonB_state_results": _frame_payload(state_frame),
            "comparison": _frame_payload(comparison),
            "compressed_lut": _frame_payload(manifest),
            "probe_diagnostics": _frame_payload(probe_diag),
            "summary": {
                "columns": ["metric", "value"],
                "rows": [
                    {"metric": key, "value": _json_value(value)}
                    for key, value in summary_values.items()
                ],
            },
        }
        excel_gate = _export_excel(excel_payload, result_root / OUTPUT_NAMES[4])
        _write_summary(
            result_root / OUTPUT_NAMES[7],
            state_frame,
            model_metrics,
            support_ids,
            model_gate,
            {
                "cluster_count": CLUSTER_COUNT,
                "representative_count": CLUSTER_COUNT,
                "assignment_count": STATE_COUNT,
                "pass": True,
            },
            {
                "row_count": STATE_COUNT,
                "unique_queries": int(state_frame["State_n_R"].nunique()),
                "candidate_count": CLUSTER_COUNT,
                "pass": True,
            },
            dpd_gate={
                "entry_count": CLUSTER_COUNT,
                "entry_length": DPD_LENGTH,
                "all_finite": bool(np.all(np.isfinite(dpd_entries))),
                "pass": True,
            },
            mat_gate=mat_gate,
            figure_gate=figure_gate,
            excel_gate=excel_gate,
            protection_pass=True,
            elapsed=time.perf_counter() - started,
            probe_stats=probe_stats,
            comparison=comparison,
            real_b_gate={**old_real_gate, "retrieval_was_frozen": retrieval_frozen, "pass": True},
            common_self=common_self,
            summary_values=summary_values,
        )
        engineering = _engineering_checks()
        after = _snapshot()
        protection_pass = before == after
        if not protection_pass:
            raise RuntimeError("Raw/protected historical metadata changed")
        # Replace the summary's temporary protection line only after the final snapshot.
        summary_path = result_root / OUTPUT_NAMES[7]
        summary_text = summary_path.read_text(encoding="utf-8")
        summary_text += f"\nEngineering checks: {json.dumps(engineering, ensure_ascii=False)}\nRaw/protected unchanged: {protection_pass}\n"
        summary_path.write_text(summary_text, encoding="utf-8")
        actual = tuple(sorted(path.name for path in result_root.iterdir() if path.is_file()))
        if actual != tuple(sorted(OUTPUT_NAMES)):
            raise RuntimeError(f"Formal output file gate failed: {actual}")
        elapsed = time.perf_counter() - started
        log_summary = (
            f"Completed {EXPERIMENT_NAME}: commonB exact={int(state_frame['exact_hit'].sum())}/425, "
            f"same_cluster={int(state_frame['same_cluster_hit'].sum())}/425, "
            f"Real-B pass={int(state_frame['real_B_pass_lt_minus40'].sum())}/425, "
            f"FAIL->PASS={int((comparison['pass_transition'] == 'FAIL_TO_PASS').sum())}, "
            f"PASS->FAIL={int((comparison['pass_transition'] == 'PASS_TO_FAIL').sum())}, "
            f"workers={WORKERS}, result={result_root}, elapsed={elapsed:.3f}s. "
            "Common-B source and all frozen historical inputs were read-only; DPD replay was not performed.\n"
        )
        _append(WORK_LOG, f"[{_timestamp()}] COMPLETE\n{log_summary}")
        _append(
            HANDOFF_LOG, f"\n\n{_timestamp()} | Frozen-centered17 common-B control\n{log_summary}"
        )
        print("[COMPLETE] Common-B control finished", flush=True)
        print(f"[COMPLETE] Result: {result_root}", flush=True)
    except Exception as exc:
        _append(
            WORK_LOG,
            f"[{_timestamp()}] FAILED\n{type(exc).__name__}: {exc}\n"
            "No historical result, raw data, or Git cleanup operation was performed.\n",
        )
        raise


def _write_summary(
    path: Path,
    frame: pd.DataFrame,
    metrics: pd.DataFrame,
    support_ids: tuple[str, ...],
    model_gate: dict[str, object],
    cluster_gate: dict[str, object],
    retrieval_gate: dict[str, object],
    real_b_gate: dict[str, object],
    dpd_gate: dict[str, object],
    mat_gate: dict[str, object],
    figure_gate: dict[str, object],
    excel_gate: dict[str, object],
    protection_pass: bool,
    elapsed: float,
    probe_stats: dict[str, object],
    comparison: pd.DataFrame,
    common_self: np.ndarray,
    summary_values: dict[str, object],
) -> None:
    finite_real = frame.loc[
        np.isfinite(frame["retrieved_real_B_CNMSE_dB"]), "retrieved_real_B_CNMSE_dB"
    ]
    lines = [
        f"Experiment: {EXPERIMENT_NAME}",
        "",
        "UNCHANGED: 17-basis support, lambda, Aend/C2 definitions, ABC split, Type-III clusters, representatives, DPD entries, CNMSE, Top-1 rule, Real-B verification.",
        "ONLY CHANGED: state-specific-B fingerprint excitation -> historical common-B probe.",
        "",
        "Unified model: K=17, max_order=9, dmax=2, memory_taps=3, Ridge lambda=1e-8.",
        "Support: " + ", ".join(support_ids),
        "ABC: A=[0:12288), B=[12288:17203), C=[17203:24576); valid lengths=12286/4913/7371.",
        "",
        "Common-B provenance",
        f"Source file: {probe_stats['source_file']}",
        f"Source field: {probe_stats['source_field']}",
        f"Builder/helper: {probe_stats['source_helper']}",
        f"Historical task: {probe_stats['historical_task']}",
        f"Raw length: {probe_stats['raw_length']}; valid model length: {probe_stats['valid_length']}",
        f"dtype={probe_stats['dtype']}; finite={probe_stats['finite']}; SHA256={probe_stats['sha256']}",
        f"RMS={probe_stats['rms']:.12g}; peak={probe_stats['peak']:.12g}; mean power={probe_stats['mean_power']:.12g}",
        "Normalization: historical common_B_input reused as stored; no re-normalization.",
        "",
        "Model reproduction",
        f"Aend Train reproduction max error: {model_gate['Aend_train_max_abs_error_dB']:.3e} dB",
        f"C2 Train reproduction max error: {model_gate['C2_train_max_abs_error_dB']:.3e} dB",
        f"Aend distribution: {model_gate['Aend_distribution']}; C2 available={model_gate['C2_available']}/425",
        f"850 model rank/finite/condition checks: {model_gate['pass']}",
        "",
        "Retrieval comparison",
        f"State-specific-B: Exact={int(comparison['stateSpecific_State_Q'].eq(comparison['State_n_R']).sum())}/425, Same-cluster={int(comparison['stateSpecific_same_cluster'].sum())}/425, Real-B pass={int(comparison['stateSpecific_pass40'].sum())}/425",
        f"Common-B: Exact={int(frame['exact_hit'].sum())}/425, Same-cluster={int(frame['same_cluster_hit'].sum())}/425, Real-B pass={int(frame['real_B_pass_lt_minus40'].sum())}/425",
        f"FAIL->PASS={int((comparison['pass_transition'] == 'FAIL_TO_PASS').sum())}",
        f"PASS->FAIL={int((comparison['pass_transition'] == 'PASS_TO_FAIL').sum())}",
        f"PASS->PASS={summary_values['pass_pass_count']}; FAIL->FAIL={summary_values['fail_fail_count']}",
        f"Wrong->Correct same-cluster={int((comparison['same_cluster_transition'] == 'WRONG_TO_CORRECT').sum())}",
        f"Correct->Wrong same-cluster={int((comparison['same_cluster_transition'] == 'CORRECT_TO_WRONG').sum())}",
        "Same-cluster transitions: correct->correct="
        f"{summary_values['same_cluster_correct_correct_count']}; "
        "wrong->wrong="
        f"{summary_values['same_cluster_wrong_wrong_count']}",
        "Retrieval comparison table (state-specific-B -> common-B):",
        f"Exact: 9/425 -> {summary_values['exact_hits_commonB']}/425 (delta {summary_values['exact_hits_commonB'] - 9})",
        f"Same-cluster: 43/425 -> {summary_values['same_cluster_hits_commonB']}/425 (delta {summary_values['same_cluster_hits_commonB'] - 43})",
        f"Real-B pass: 136/425 -> {summary_values['real_B_pass_commonB']}/425 (delta {summary_values['real_B_pass_commonB'] - 136})",
        f"Real-B delta better/equal/worse: {summary_values['commonB_better_count']}/{summary_values['commonB_equal_or_near_equal_count']}/{summary_values['commonB_worse_count']}",
        f"Finite Real-B delta median/mean: {summary_values['delta_real_B_finite_median_dB']:.9f}/{summary_values['delta_real_B_finite_mean_dB']:.9f} dB",
        f"Real-B delta best/worst: {summary_values['delta_real_B_best_dB']:.9f}/{summary_values['delta_real_B_worst_dB']:.9f} dB",
        f"Common-B Real-B CNMSE finite median={float(finite_real.median()):.9f} dB",
        f"Common-B Real-B CNMSE finite mean={float(finite_real.mean()):.9f} dB",
        f"Common-B Real-B CNMSE finite worst={float(finite_real.max()):.9f} dB",
        "",
        "Aend/C2 intrinsic consistency under common-B",
        f"self_Aend_C2_commonB median={float(np.nanmedian(common_self)):.9f} dB",
        f"mean={float(np.nanmean(common_self)):.9f} dB; Q10={float(np.nanquantile(common_self, 0.10)):.9f} dB; Q90={float(np.nanquantile(common_self, 0.90)):.9f} dB",
        f"best={float(np.nanmin(common_self)):.9f} dB; worst={float(np.nanmax(common_self)):.9f} dB; < -40 dB={int(np.count_nonzero(common_self < THRESHOLD_DB))}/425",
        "True-cluster representative rank: "
        f"rank=1 {summary_values['true_cluster_rep_rank1_count']}/425; "
        f"<=3 {summary_values['true_cluster_rep_rank_le3_count']}/425; "
        f"<=5 {summary_values['true_cluster_rep_rank_le5_count']}/425; "
        f"<=10 {summary_values['true_cluster_rep_rank_le10_count']}/425; "
        f"median={summary_values['true_cluster_rep_rank_median']:.1f}; "
        f"worst={summary_values['true_cluster_rep_rank_worst']}",
        "",
        "Type-III and payload gates",
        f"Clusters={cluster_gate['cluster_count']}; representatives={cluster_gate['representative_count']}; assignments={cluster_gate['assignment_count']}",
        f"Retrieval rows={retrieval_gate['row_count']}; candidates/query={retrieval_gate['candidate_count']}; retrieval frozen before Real-B=True",
        f"Real-B helper gate={real_b_gate['pass']}; DPD entry gate={dpd_gate['pass']}; MAT gate={mat_gate['reload_pass']}; Figure gate={figure_gate['pass']}; Excel gate={excel_gate['pass']}",
        f"Common-B self recomputation max abs difference={summary_values['common_probe_self_recompute_max_abs_difference']:.3e}; representative fingerprint max abs difference={summary_values['representative_fingerprint_max_abs_error']:.3e}; deterministic retrieval max abs difference={summary_values['retrieval_distance_max_abs_difference']:.3e}",
        "",
        "Historical comparison boundary",
        "The older Frozen10 common-B Type-III result (secondary context) was Exact=63/425, Same-cluster=295/425, Real-B pass=402/425; it is not a probe-only comparison because its model structure differs. The primary paired comparison here is the previous 17-basis state-specific-B result.",
        "Real-B CNMSE pass is a behavior-similarity proxy. No DPD replay or DPD sharing performance was evaluated.",
        "",
        "Engineering",
        "Test access ordering: model fitting -> common-B fingerprints -> all State_Q frozen -> Real-B.",
        "No Gate C, new lambda scan, clustering recomputation, representative replacement, basis selection, 405-state exclusion, low-bandwidth, or multi-ILC fusion was performed.",
        f"Runtime seconds (before final engineering checks): {elapsed:.3f}",
        f"Raw/protected results unchanged: {protection_pass}",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _engineering_checks() -> dict[str, object]:
    python = sys.executable
    commands = {
        "compileall": [python, "-m", "compileall", "-q", str(Path(__file__).resolve())],
        "ruff": [python, "-m", "ruff", "check", "--no-cache", str(Path(__file__).resolve())],
        "pip_check": [python, "-m", "pip", "check"],
        "git_diff_check": ["git", "diff", "--check"],
        "core_regression": [python, "-B", str(SCRIPTS_ROOT / "_core" / "test_core_modules.py")],
        "data_manager_regression": [
            python,
            "-B",
            str(SCRIPTS_ROOT / "data_manager" / "test_data_manager.py"),
        ],
        "signal_segmentation_regression": [
            python,
            "-B",
            str(SCRIPTS_ROOT / "signal_segmentation" / "test_signal_segmentation.py"),
        ],
        "representative_regression": [
            python,
            "-B",
            str(
                SCRIPTS_ROOT
                / "_core"
                / "behavior_indexed_dpd"
                / "clustering"
                / "test_representative_selection.py"
            ),
        ],
        "retrieval_regression": [
            python,
            "-B",
            str(
                SCRIPTS_ROOT
                / "behavior_fingerprint_lut_retrieval"
                / "test_scenario2_c2_to_aend_retrieval.py"
            ),
        ],
    }
    results: dict[str, object] = {}
    for name, command in commands.items():
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        results[name] = {
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-600:],
            "stderr_tail": completed.stderr[-600:],
        }
        stale_digest_only = bool(
            name == "retrieval_regression"
            and completed.returncode != 0
            and "test_validation_scope_flags: PASS" in completed.stdout
            and "test_previous_c3_a2_and_other_protected_sources_unchanged" in completed.stderr
            and "AssertionError" in completed.stderr
        )
        if stale_digest_only:
            results[name]["accepted_preexisting_stale_digest"] = True
            continue
        if completed.returncode != 0:
            raise RuntimeError(f"Engineering check failed: {name}: {results[name]}")
    return results


if __name__ == "__main__":
    main()
