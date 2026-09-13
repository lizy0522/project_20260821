"""Scenario-2 Type-III cluster-compressed 5B C2-to-Aend retrieval.

The clustering matrix and representative mapping are read-only inputs.  The
Frozen MP model is fitted independently for all 425 states, but only the
minimax-medoid representative states enter the offline Aend LUT.  All 425 C2
queries are matched to that compressed LUT.  Real-B verification is started
only after every State_Q has been frozen.
"""

# ruff: noqa: E402,E501,I001

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any

for _thread_name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_name] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _core.behavior_indexed_dpd.clustering import cluster_behavior_states
from _core.signal import adjust_complex_gain, fine_align, rough_align
from behavior_fingerprint_ranking_consistency.distance_matrix import compute_cnmse_distance_matrix
from behavior_fingerprint_ranking_consistency.ranking import distance_to_ranks
from behavior_model import sparse_gmp as gmp
from data_manager import load_by_id
from signal_segmentation import build_partition_from_xin, get_ilc_pair, preprocess_full_pair


TASK_NAME = "scenario_2_C2_to_Aend_type3_cluster_compressed_5B"
RESULT_ROOT = PROJECT_ROOT / "results" / "behavior_fingerprint_lut_retrieval" / TASK_NAME
CLUSTER_ROOT = PROJECT_ROOT / "results" / "clustering" / "scenario_2_yout_withoutdpd_ori_5B_A_complete_link_clustering"
CLUSTER_MATRIX = CLUSTER_ROOT / "pairwise_cnmse_5B_A_preprocessed.npz"
CLUSTER_THRESHOLD_DEFAULT = -40.0
REAL_B_THRESHOLD_DEFAULT = -40.0
CLUSTER_ASSIGNMENTS = CLUSTER_ROOT / "threshold_m40p0dB" / "cluster_assignments.csv"
CLUSTER_REPRESENTATIVES = CLUSTER_ROOT / "threshold_m40p0dB" / "cluster_representatives.csv"
OLD_FORMAL_ROOT = PROJECT_ROOT / "results" / "behavior_fingerprint_lut_retrieval" / "scenario_2_C2_to_Aend"
OLD_LUT = OLD_FORMAL_ROOT / "lut_fingerprints_Aend.npz"
OLD_QUERY = OLD_FORMAL_ROOT / "query_fingerprints_C2.npz"
OLD_REAL_B = OLD_FORMAL_ROOT / "real_B_distance_matrix.npz"
ALL_ILC_METRICS = PROJECT_ROOT / "results" / "behavior_fingerprint_ranking_consistency" / "scenario_2_all_ilc" / "all_ilc_model_metrics.csv"
ALL_ILC_AVAILABILITY = PROJECT_ROOT / "results" / "behavior_fingerprint_ranking_consistency" / "scenario_2_all_ilc" / "ilc_availability.csv"
STATE_COUNT = 425
FULL_LENGTH = 24576
ABC_A_START = 0
ABC_A_END = 12288
ABC_B_START = 12288
ABC_B_END = 17203
ABC_C_START = 17203
ABC_C_END = 24576
FINGERPRINT_LENGTH = 4913
COMMON_B_LENGTH = 4915
FINE_ALIGN_SUBTIME = 256
ORDERS = (1, 2, 3, 5, 7, 9)
MEMORY = (3, 2, 2, 1, 1, 1)
RIDGE_LAMBDA = 1e-8
DMAX = 2
CPU_TARGET = 0.90
REAL_B_REPRO_TOL_DB = 1e-9
STATE_REPRODUCTION_IDS = (0, 100, 187, 340, 424)

PROTECTED_RESULT_DIRS = {
    "formal_retrieval": PROJECT_ROOT / "results" / "behavior_fingerprint_lut_retrieval" / "scenario_2_C2_to_Aend",
    "formal_g4_retrieval": PROJECT_ROOT / "results" / "behavior_fingerprint_lut_retrieval" / "scenario_2_C2_to_Aend_G4_sparse_gmp_5B",
    "all_ilc": PROJECT_ROOT / "results" / "behavior_fingerprint_ranking_consistency" / "scenario_2_all_ilc",
    "a123": PROJECT_ROOT / "results" / "behavior_fingerprint_ranking_consistency" / "scenario_2_y_lut_fusion_A123",
    "clustering": CLUSTER_ROOT,
    "behavior_model": PROJECT_ROOT / "results" / "behavior_model",
}

_WORKER_COMMON_B: np.ndarray | None = None
_WORKER_PHI_B: np.ndarray | None = None
_RETRIEVAL_QUERY: np.ndarray | None = None
_RETRIEVAL_LUT: np.ndarray | None = None
_RETRIEVAL_CANDIDATE_IDS: np.ndarray | None = None
_RETRIEVAL_QUERY_SHM: SharedMemory | None = None
_RETRIEVAL_LUT_SHM: SharedMemory | None = None
_REAL_B_SHM: SharedMemory | None = None


def _tree_digest(path: Path) -> str | None:
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    files = [
        file
        for file in path.rglob("*")
        if file.is_file() and "__pycache__" not in file.parts and file.suffix.lower() != ".pyc"
    ]
    for file in sorted(files, key=lambda item: str(item).lower()):
        content = file.read_bytes()
        digest.update(file.relative_to(path).as_posix().encode("utf-8") + b"\0")
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def _raw_manifest() -> dict[str, Any]:
    raw_root = PROJECT_ROOT / "data" / "raw"
    files = [file for file in raw_root.rglob("*") if file.is_file()]
    digest = hashlib.sha256()
    total = 0
    for file in sorted(files, key=lambda item: str(item).lower()):
        content = file.read_bytes()
        digest.update(file.relative_to(raw_root).as_posix().encode("utf-8") + b"\0")
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(hashlib.sha256(content).digest())
        total += len(content)
    return {"sha256": digest.hexdigest(), "file_count": len(files), "bytes": total}


def _snapshot() -> dict[str, Any]:
    return {"data/raw": _raw_manifest(), "result_dirs": {name: _tree_digest(path) for name, path in PROTECTED_RESULT_DIRS.items()}}


def _protection_check(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    raw_ok = before["data/raw"] == after["data/raw"]
    result = {
        name: {
            "sha256_unchanged": before["result_dirs"].get(name) == after["result_dirs"].get(name),
            "before": before["result_dirs"].get(name),
            "after": after["result_dirs"].get(name),
        }
        for name in PROTECTED_RESULT_DIRS
    }
    return {"data_raw_unchanged": raw_ok, "protected_results_unchanged": bool(all(item["sha256_unchanged"] for item in result.values())), "all_protected_unchanged": bool(raw_ok and all(item["sha256_unchanged"] for item in result.values())), "result_dirs": result}


def _flatten_complex(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if not np.iscomplexobj(array):
        raise ValueError(f"{name} must be complex")
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    array = np.asarray(array, dtype=np.complex128).reshape(-1)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite complex")
    return array


def _scalar(data: dict[str, Any], key: str) -> float:
    if key not in data:
        raise KeyError(f"missing scalar {key}")
    value = np.asarray(data[key]).reshape(-1)
    if value.size != 1:
        raise ValueError(f"{key} must be scalar")
    result = float(value[0])
    if not np.isfinite(result):
        raise ValueError(f"{key} must be finite")
    return result


def _fit_frozen_side(x: np.ndarray, y: np.ndarray, b_input: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    phi = gmp.build_frozen_mp_basis(x)
    y_valid = np.asarray(y, dtype=np.complex128).reshape(-1)[DMAX:]
    phi_b = gmp.build_frozen_mp_basis(b_input)
    if phi.shape[0] != y_valid.size or phi_b.shape != (FINGERPRINT_LENGTH, 10):
        raise RuntimeError("Frozen model support mismatch")
    theta, diagnostics = gmp.fit_ridge(phi, y_valid, RIDGE_LAMBDA)
    train_prediction = phi @ theta
    b_prediction = phi_b @ theta
    return {
        "train_NMSE_dB": float(10.0 * np.log10(np.sum(np.abs(y_valid - train_prediction) ** 2) / np.sum(np.abs(y_valid) ** 2))),
        "B_NMSE_dB": float("nan"),
        "theta_norm": float(diagnostics.theta_l2_norm),
        "rank": int(diagnostics.rank_phi),
        "rank_augmented": int(diagnostics.rank_augmented),
        "condition_number": float(diagnostics.condition_number_phi),
        "condition_number_augmented": float(diagnostics.condition_number_augmented),
        "n_train": int(phi.shape[0]),
        "n_B": int(phi_b.shape[0]),
        "theta": theta,
        "fingerprint": np.asarray(b_prediction, dtype=np.complex128),
    }, theta


def _state_model_worker_init(common_b: np.ndarray) -> None:
    global _WORKER_COMMON_B, _WORKER_PHI_B
    _WORKER_COMMON_B = np.asarray(common_b, dtype=np.complex128)
    _WORKER_PHI_B = gmp.build_frozen_mp_basis(_WORKER_COMMON_B)


def _state_model_worker(state_id: int) -> dict[str, Any]:
    if _WORKER_COMMON_B is None or _WORKER_PHI_B is None:
        raise RuntimeError("model worker is not initialized")
    data = load_by_id(int(state_id))
    xin = _flatten_complex(data["xin"], "xin")
    input_history = np.asarray(data["xin_pd_ori_ilc"])
    output_history = np.asarray(data["yout_withdpd_ori_ilc"])
    if input_history.ndim != 2 or output_history.shape != input_history.shape or input_history.shape[0] != FULL_LENGTH or input_history.shape[1] < 2:
        raise ValueError(f"state {state_id} ILC history is invalid")
    partition = build_partition_from_xin(xin)
    ilc_count = int(input_history.shape[1])
    a_pair = get_ilc_pair(data, ilc_count - 1)
    c_pair = get_ilc_pair(data, 1)
    a_canonical = preprocess_full_pair(a_pair.input_full, a_pair.output_raw_full, partition, pair_type="ILC", iteration_index=ilc_count - 1, input_peak_normalization_factor=a_pair.input_peak_normalization_factor)
    c_canonical = preprocess_full_pair(c_pair.input_full, c_pair.output_raw_full, partition, pair_type="ILC", iteration_index=1, input_peak_normalization_factor=c_pair.input_peak_normalization_factor)
    a_values, a_theta = _fit_frozen_side(a_canonical["A"].input, a_canonical["A"].output, _WORKER_COMMON_B)
    c_values, c_theta = _fit_frozen_side(c_canonical["C"].input, c_canonical["C"].output, _WORKER_COMMON_B)
    a_b_phi = gmp.build_frozen_mp_basis(a_canonical["B"].input)
    c_b_phi = gmp.build_frozen_mp_basis(c_canonical["B"].input)
    a_b_y = np.asarray(a_canonical["B"].output, dtype=np.complex128)[DMAX:]
    c_b_y = np.asarray(c_canonical["B"].output, dtype=np.complex128)[DMAX:]
    a_values["B_NMSE_dB"] = float(10.0 * np.log10(np.sum(np.abs(a_b_y - a_b_phi @ a_theta) ** 2) / np.sum(np.abs(a_b_y) ** 2)))
    c_values["B_NMSE_dB"] = float(10.0 * np.log10(np.sum(np.abs(c_b_y - c_b_phi @ c_theta) ** 2) / np.sum(np.abs(c_b_y) ** 2)))
    return {
        "state_id": int(state_id),
        "ilc_A_end": ilc_count,
        "funMng": int(_scalar(data, "funMng")),
        "funAng": int(_scalar(data, "funAng")),
        "secMng": int(_scalar(data, "secMng")),
        "secAng": int(_scalar(data, "secAng")),
        "v_carrier": _scalar(data, "v_carrier"),
        "inputPower": _scalar(data, "inputPower"),
        "nmse_withoutdpd_dB": _scalar(data, "nmse_withoutdpd"),
        "acpr_low_withoutdpd": _scalar(data, "acpr_low_withoutdpd"),
        "acpr_upper_withoutdpd": _scalar(data, "acpr_upper_withoutdpd"),
        "Y_Aend_train_NMSE_dB": a_values["train_NMSE_dB"],
        "Y_Aend_B_NMSE_dB": a_values["B_NMSE_dB"],
        "Y_C2_train_NMSE_dB": c_values["train_NMSE_dB"],
        "Y_C2_B_NMSE_dB": c_values["B_NMSE_dB"],
        "Aend_rank": a_values["rank"],
        "C2_rank": c_values["rank"],
        "Aend_n_train": a_values["n_train"],
        "Aend_n_B": a_values["n_B"],
        "C2_n_train": c_values["n_train"],
        "C2_n_B": c_values["n_B"],
        "Aend_fingerprint": np.asarray(a_values["fingerprint"], dtype=np.complex128),
        "C2_fingerprint": np.asarray(c_values["fingerprint"], dtype=np.complex128),
    }


def _run_state_models(common_b: np.ndarray, workers: int) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    results: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=workers, initializer=_state_model_worker_init, initargs=(common_b,)) as executor:
        futures = {executor.submit(_state_model_worker, state_id): state_id for state_id in range(STATE_COUNT)}
        checkpoints = {math.ceil(STATE_COUNT * fraction) for fraction in (0.25, 0.5, 0.75, 1.0)}
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results[int(result["state_id"])] = result
            if done in checkpoints:
                print(f"Behavior modeling: {done}/{STATE_COUNT} ({done / STATE_COUNT:.0%})", flush=True)
    if sorted(results) != list(range(STATE_COUNT)):
        raise RuntimeError("incomplete state modeling results")
    rows = []
    a_fp = np.empty((STATE_COUNT, FINGERPRINT_LENGTH), dtype=np.complex128)
    c_fp = np.empty_like(a_fp)
    for state_id in range(STATE_COUNT):
        result = results[state_id]
        result["acpr_withoutdpd_mean_dBc"] = (result["acpr_low_withoutdpd"] + result["acpr_upper_withoutdpd"]) / 2.0
        rows.append({key: value for key, value in result.items() if not isinstance(value, np.ndarray)})
        a_fp[state_id] = result["Aend_fingerprint"]
        c_fp[state_id] = result["C2_fingerprint"]
    return pd.DataFrame(rows).sort_values("state_id").reset_index(drop=True), a_fp, c_fp


def _real_b_worker(state_id: int) -> tuple[int, np.ndarray]:
    data = load_by_id(int(state_id))
    xin = _flatten_complex(data["xin"], "xin")
    y_raw = _flatten_complex(data["yout_withoutdpd_ori"], "yout_withoutdpd_ori")
    partition = build_partition_from_xin(xin)
    y_rough, _ = rough_align(xin, y_raw)
    y_fine, _ = fine_align(xin, y_rough, subtime=FINE_ALIGN_SUBTIME)
    x_b = xin[partition.b_slice]
    y_b = np.asarray(y_fine[partition.b_slice], dtype=np.complex128)
    y_b_adjusted, _ = adjust_complex_gain(x_b, y_b)
    if y_b_adjusted.shape != (ABC_B_END - ABC_B_START,) or not np.all(np.isfinite(y_b_adjusted)):
        raise RuntimeError(f"state {state_id} Real-B output invalid")
    return int(state_id), np.asarray(y_b_adjusted, dtype=np.complex128)


def _run_real_b(workers: int) -> np.ndarray:
    values: dict[int, np.ndarray] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_real_b_worker, state_id): state_id for state_id in range(STATE_COUNT)}
        checkpoints = {math.ceil(STATE_COUNT * fraction) for fraction in (0.25, 0.5, 0.75, 1.0)}
        for done, future in enumerate(as_completed(futures), start=1):
            state_id, waveform = future.result()
            values[int(state_id)] = waveform
            if done in checkpoints:
                print(f"Real-B preprocessing: {done}/{STATE_COUNT} ({done / STATE_COUNT:.0%})", flush=True)
    if sorted(values) != list(range(STATE_COUNT)):
        raise RuntimeError("incomplete Real-B results")
    return np.stack([values[state_id] for state_id in range(STATE_COUNT)]).astype(np.complex128, copy=False)


def _retrieval_worker_init(query_name: str, query_shape: tuple[int, int], lut_name: str, lut_shape: tuple[int, int], candidate_ids: np.ndarray) -> None:
    global _RETRIEVAL_QUERY, _RETRIEVAL_LUT, _RETRIEVAL_CANDIDATE_IDS, _RETRIEVAL_QUERY_SHM, _RETRIEVAL_LUT_SHM
    _RETRIEVAL_QUERY_SHM = SharedMemory(name=query_name)
    _RETRIEVAL_LUT_SHM = SharedMemory(name=lut_name)
    _RETRIEVAL_QUERY = np.ndarray(query_shape, dtype=np.complex128, buffer=_RETRIEVAL_QUERY_SHM.buf)
    _RETRIEVAL_LUT = np.ndarray(lut_shape, dtype=np.complex128, buffer=_RETRIEVAL_LUT_SHM.buf)
    _RETRIEVAL_CANDIDATE_IDS = np.asarray(candidate_ids, dtype=np.int64)


def _retrieval_worker(state_id: int) -> dict[str, Any]:
    if _RETRIEVAL_QUERY is None or _RETRIEVAL_LUT is None or _RETRIEVAL_CANDIDATE_IDS is None:
        raise RuntimeError("retrieval worker is not initialized")
    row = compute_cnmse_distance_matrix(_RETRIEVAL_QUERY[int(state_id) : int(state_id) + 1], _RETRIEVAL_LUT)[0]
    order = np.lexsort((_RETRIEVAL_CANDIDATE_IDS, row))
    best_index = int(order[0])
    return {"state_id_R": int(state_id), "retrieved_state_id_Q": int(_RETRIEVAL_CANDIDATE_IDS[best_index]), "retrieval_fingerprint_CNMSE_dB": float(row[best_index])}


def _run_retrieval(query: np.ndarray, lut: np.ndarray, candidate_ids: np.ndarray, workers: int) -> pd.DataFrame:
    query_shm = SharedMemory(create=True, size=query.nbytes)
    lut_shm = SharedMemory(create=True, size=lut.nbytes)
    try:
        query_shared = np.ndarray(query.shape, dtype=np.complex128, buffer=query_shm.buf)
        lut_shared = np.ndarray(lut.shape, dtype=np.complex128, buffer=lut_shm.buf)
        query_shared[:] = query
        lut_shared[:] = lut
        rows: list[dict[str, Any]] = []
        with ProcessPoolExecutor(max_workers=workers, initializer=_retrieval_worker_init, initargs=(query_shm.name, query.shape, lut_shm.name, lut.shape, candidate_ids)) as executor:
            futures = {executor.submit(_retrieval_worker, state_id): state_id for state_id in range(STATE_COUNT)}
            checkpoints = {math.ceil(STATE_COUNT * fraction) for fraction in (0.25, 0.5, 0.75, 1.0)}
            for done, future in enumerate(as_completed(futures), start=1):
                rows.append(future.result())
                if done in checkpoints:
                    print(f"Retrieval: {done}/{STATE_COUNT} ({done / STATE_COUNT:.0%})", flush=True)
        return pd.DataFrame(rows).sort_values("state_id_R").reset_index(drop=True)
    finally:
        query_shm.close()
        query_shm.unlink()
        lut_shm.close()
        lut_shm.unlink()


def _load_common_b() -> np.ndarray:
    with np.load(OLD_LUT, allow_pickle=False) as data:
        state_ids = np.asarray(data["state_ids"], dtype=np.int64)
        common_b = np.asarray(data["common_B_input"], dtype=np.complex128)
    if not np.array_equal(state_ids, np.arange(STATE_COUNT, dtype=np.int64)) or common_b.shape != (COMMON_B_LENGTH,) or not np.all(np.isfinite(common_b)):
        raise RuntimeError("formal common B probe is invalid")
    return common_b


def _load_cluster_inputs(threshold: float) -> tuple[np.ndarray, pd.DataFrame, Any, dict[str, Any]]:
    if not CLUSTER_MATRIX.is_file() or not CLUSTER_ASSIGNMENTS.is_file() or not CLUSTER_REPRESENTATIVES.is_file():
        raise FileNotFoundError("validated clustering matrix/representative inputs are missing")
    with np.load(CLUSTER_MATRIX, allow_pickle=False) as data:
        matrix = np.asarray(data["cnmse_db"], dtype=np.float64)
        metadata = {
            "signal_variable": str(np.asarray(data["signal_variable"]).reshape(-1)[0]),
            "bandwidth_mode": str(np.asarray(data["bandwidth_mode"]).reshape(-1)[0]),
            "preprocess_order": str(np.asarray(data["preprocess_order"]).reshape(-1)[0]),
            "symmetry_mode": str(np.asarray(data["cnmse_symmetry_mode"]).reshape(-1)[0]),
        }
    if metadata != {"signal_variable": "yout_withoutdpd_ori", "bandwidth_mode": "5B", "preprocess_order": "full_time_align_then_A_split_then_A_complex_gain", "symmetry_mode": "max_of_two_directions"}:
        raise RuntimeError(f"clustering matrix metadata mismatch: {metadata}")
    cluster_result = cluster_behavior_states(matrix, threshold)
    assignments = pd.read_csv(CLUSTER_ASSIGNMENTS)
    reps = pd.read_csv(CLUSTER_REPRESENTATIVES)
    representative_columns = {
        "representative_state_id",
        "representative_fingerprint_state_id",
        "representative_dpd_state_id",
    }
    if not representative_columns.issubset(reps.columns):
        raise RuntimeError("cluster representatives are missing representative mapping columns")
    reps_sorted = reps.sort_values("cluster_id").reset_index(drop=True)
    representative_ids = reps_sorted["representative_state_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(
        reps_sorted["representative_fingerprint_state_id"].to_numpy(dtype=np.int64),
        representative_ids,
    ) or not np.array_equal(
        reps_sorted["representative_dpd_state_id"].to_numpy(dtype=np.int64),
        representative_ids,
    ):
        raise RuntimeError("fingerprint/DPD representative mapping is not identical to state representative")
    assignment_columns = {
        "state_id",
        "cluster_id",
        "representative_state_id",
        "representative_fingerprint_state_id",
        "representative_dpd_state_id",
        "is_representative",
    }
    if not assignment_columns.issubset(assignments.columns):
        raise RuntimeError("cluster assignments are missing representative mapping columns")
    if not np.array_equal(
        assignments["representative_fingerprint_state_id"].to_numpy(dtype=np.int64),
        assignments["representative_state_id"].to_numpy(dtype=np.int64),
    ) or not np.array_equal(
        assignments["representative_dpd_state_id"].to_numpy(dtype=np.int64),
        assignments["representative_state_id"].to_numpy(dtype=np.int64),
    ):
        raise RuntimeError("assignment fingerprint/DPD representative mapping mismatch")
    expected_representative_flag = assignments["state_id"].to_numpy(dtype=np.int64) == assignments["representative_state_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(assignments["is_representative"].to_numpy(dtype=bool), expected_representative_flag):
        raise RuntimeError("assignment representative flags are inconsistent")
    old_sets = [tuple(sorted(group["state_id"].astype(int).tolist())) for _, group in assignments.groupby("cluster_id", sort=True)]
    new_sets = list(cluster_result.clusters)
    if abs(threshold - CLUSTER_THRESHOLD_DEFAULT) <= 1e-12:
        if sorted(old_sets, key=lambda members: (members[0], members)) != sorted(new_sets, key=lambda members: (members[0], members)):
            raise RuntimeError("-40 dB clustering member-set reproduction failed")
        if reps.shape[0] != len(new_sets) or not np.array_equal(representative_ids, cluster_result.representative_state_ids):
            raise RuntimeError("-40 dB representative reproduction failed")
    if matrix.shape != (STATE_COUNT, STATE_COUNT) or not np.all(np.isneginf(np.diag(matrix))) or not np.all(np.isfinite(matrix[~np.eye(STATE_COUNT, dtype=bool)])) or not np.array_equal(matrix, matrix.T):
        raise RuntimeError("clustering matrix gate failed")
    return matrix, assignments, cluster_result, metadata


def _frozen_reproduction_gate(metrics: pd.DataFrame, a_fp: np.ndarray, c_fp: np.ndarray) -> dict[str, Any]:
    if not ALL_ILC_METRICS.is_file() or not OLD_LUT.is_file() or not OLD_QUERY.is_file() or not ALL_ILC_AVAILABILITY.is_file():
        raise FileNotFoundError("formal Frozen references are missing")
    formal_metrics = pd.read_csv(ALL_ILC_METRICS)
    pd.read_csv(ALL_ILC_AVAILABILITY).sort_values("state_id").reset_index(drop=True)
    with np.load(OLD_LUT, allow_pickle=False) as data:
        formal_a_fp = np.asarray(data["Y_Aend_fingerprints"], dtype=np.complex128)
        formal_counts = np.asarray(data["N_ilc_available"], dtype=np.int64)
    with np.load(OLD_QUERY, allow_pickle=False) as data:
        formal_c_fp = np.asarray(data["Q_C2"], dtype=np.complex128)
    errors: list[float] = []
    rows: list[dict[str, Any]] = []
    for state_id in STATE_REPRODUCTION_IDS:
        current = metrics.loc[metrics["state_id"] == state_id].iloc[0]
        actual_n = int(current["ilc_A_end"])
        formal_a = formal_metrics.loc[(formal_metrics["state_id"] == state_id) & (formal_metrics["actual_ilc_n"] == actual_n) & (formal_metrics["model_role"] == "Y-A") & np.isclose(formal_metrics["ridge_lambda"].astype(float), RIDGE_LAMBDA)]
        formal_c = formal_metrics.loc[(formal_metrics["state_id"] == state_id) & (formal_metrics["actual_ilc_n"] == 2) & (formal_metrics["model_role"] == "Y-C") & np.isclose(formal_metrics["ridge_lambda"].astype(float), RIDGE_LAMBDA)]
        if formal_a.shape[0] != 1 or formal_c.shape[0] != 1:
            raise RuntimeError(f"formal metric mapping missing for state {state_id}")
        metric_error = max(abs(float(current["Y_Aend_train_NMSE_dB"]) - float(formal_a.iloc[0]["train_nmse_db"])), abs(float(current["Y_Aend_B_NMSE_dB"]) - float(formal_a.iloc[0]["B_generalization_nmse_db"])), abs(float(current["Y_C2_train_NMSE_dB"]) - float(formal_c.iloc[0]["train_nmse_db"])), abs(float(current["Y_C2_B_NMSE_dB"]) - float(formal_c.iloc[0]["B_generalization_nmse_db"])))
        fp_error = max(float(np.max(np.abs(a_fp[state_id] - formal_a_fp[state_id]))), float(np.max(np.abs(c_fp[state_id] - formal_c_fp[state_id]))))
        errors.extend([metric_error])
        rows.append({"state_id": state_id, "metric_max_abs_error_dB": metric_error, "fingerprint_max_abs_error": fp_error, "ilc_A_end_current": actual_n, "ilc_A_end_formal": int(formal_counts[state_id]), "pass": bool(metric_error <= 1e-6 and fp_error <= 1e-12)})
    return {"pass": bool(all(item["pass"] for item in rows)), "maximum_metric_abs_error_dB": max(item["metric_max_abs_error_dB"] for item in rows), "maximum_fingerprint_abs_error": max(item["fingerprint_max_abs_error"] for item in rows), "rows": rows}


def _real_b_gate(real_b: np.ndarray) -> dict[str, Any]:
    real_b = np.asarray(real_b, dtype=np.complex128)
    if real_b.shape != (STATE_COUNT, COMMON_B_LENGTH) or not np.all(np.isfinite(real_b)):
        raise ValueError(f"Real-B full B support must be {(STATE_COUNT, COMMON_B_LENGTH)}")
    # The canonical Frozen MP/Real-B contract evaluates the B segment after
    # discarding the first DMAX samples, exactly as the model basis does.
    real_b_valid = real_b[:, DMAX:]
    if real_b_valid.shape != (STATE_COUNT, FINGERPRINT_LENGTH):
        raise RuntimeError("Real-B valid support mismatch")
    current_distance = compute_cnmse_distance_matrix(real_b_valid, real_b_valid)
    with np.load(OLD_REAL_B, allow_pickle=False) as data:
        old_ids = np.asarray(data["state_ids"], dtype=np.int64)
        old_distance = np.asarray(data["D_B"], dtype=np.float64)
        old_ranking = np.asarray(data["R_B"], dtype=np.float64)
    current_ranking = distance_to_ranks(current_distance)
    finite = np.isfinite(current_distance) & np.isfinite(old_distance)
    max_error = float(np.max(np.abs(current_distance[finite] - old_distance[finite]))) if np.any(finite) else 0.0
    mismatch_inf = int(np.count_nonzero(np.isneginf(current_distance) != np.isneginf(old_distance)))
    ranking_error = float(np.max(np.abs(current_ranking - old_ranking)))
    return {"pass": bool(np.array_equal(old_ids, np.arange(STATE_COUNT, dtype=np.int64)) and current_distance.shape == (STATE_COUNT, STATE_COUNT) and mismatch_inf == 0 and max_error <= REAL_B_REPRO_TOL_DB and ranking_error == 0.0), "max_abs_error_dB": max_error, "ranking_max_abs_error": ranking_error, "infinity_pattern_mismatch": mismatch_inf, "reproduction_tolerance_dB": REAL_B_REPRO_TOL_DB, "distance": current_distance}


def _build_final_frame(metrics: pd.DataFrame, clustering: Any, retrieval: pd.DataFrame, real_b_distance: np.ndarray, real_b_threshold: float) -> pd.DataFrame:
    state_ids = np.arange(STATE_COUNT, dtype=np.int64)
    lookup = metrics.set_index("state_id")
    retrieval = retrieval.set_index("state_id_R").loc[state_ids].reset_index()
    rows: list[dict[str, Any]] = []
    for record in retrieval.itertuples(index=False):
        state_id = int(record.state_id_R)
        query_id = int(record.retrieved_state_id_Q)
        cluster_id = int(clustering.state_to_cluster[state_id])
        expected_rep = int(clustering.representative_state_ids[cluster_id])
        real_value = float(real_b_distance[state_id, query_id])
        state = lookup.loc[state_id]
        rows.append({
            "state_id_R": state_id,
            "funMng": int(state["funMng"]),
            "funAng": int(state["funAng"]),
            "secMng": int(state["secMng"]),
            "secAng": int(state["secAng"]),
            "nmse_withoutdpd_dB": float(state["nmse_withoutdpd_dB"]),
            "acpr_withoutdpd_mean_dBc": float(state["acpr_withoutdpd_mean_dBc"]),
            "Y_Aend_train_NMSE_dB": float(state["Y_Aend_train_NMSE_dB"]),
            "Y_Aend_B_NMSE_dB": float(state["Y_Aend_B_NMSE_dB"]),
            "Y_C2_train_NMSE_dB": float(state["Y_C2_train_NMSE_dB"]),
            "Y_C2_B_NMSE_dB": float(state["Y_C2_B_NMSE_dB"]),
            "true_cluster_id": cluster_id,
            "expected_representative_state_id": expected_rep,
            "retrieved_state_id_Q": query_id,
            "state_id_delta_Q_minus_R": query_id - state_id,
            "state_id_abs_delta": abs(query_id - state_id),
            "retrieval_fingerprint_CNMSE_dB": float(record.retrieval_fingerprint_CNMSE_dB),
            "retrieved_real_B_CNMSE_dB": real_value,
            "exact_state_hit": bool(query_id == state_id),
            "same_cluster_hit": bool(int(clustering.state_to_cluster[query_id]) == cluster_id),
            "real_B_below_threshold": bool(np.isneginf(real_value) or real_value < real_b_threshold),
        })
    frame = pd.DataFrame(rows).sort_values("state_id_R").reset_index(drop=True)
    if frame.shape[0] != STATE_COUNT or not np.array_equal(frame["state_id_R"].to_numpy(dtype=np.int64), state_ids):
        raise RuntimeError("final Type-III table must contain state_id_R 0...424 exactly once")
    return frame


def _write_summary(frame: pd.DataFrame, clustering: Any, cluster_threshold: float, real_b_threshold: float, matrix_metadata: dict[str, Any], model_gate: dict[str, Any], real_b_gate: dict[str, Any], protection: dict[str, Any], workers: int, matrix_action: str, threshold_root: Path, cluster_reproduction_pass: bool) -> None:
    b = len(clustering.representative_state_ids)
    finite_real_b = frame.loc[np.isfinite(frame["retrieved_real_B_CNMSE_dB"]), "retrieved_real_B_CNMSE_dB"]
    nonexact_finite = frame.loc[(~frame["exact_state_hit"]) & np.isfinite(frame["retrieved_real_B_CNMSE_dB"]), "retrieved_real_B_CNMSE_dB"]
    medians = frame[["Y_Aend_train_NMSE_dB", "Y_Aend_B_NMSE_dB", "Y_C2_train_NMSE_dB", "Y_C2_B_NMSE_dB"]].median()
    summary_lines = [
        f"Experiment: {TASK_NAME}",
        "Scenario: 2 C2-to-Aend Type-III cluster-compressed LUT retrieval",
        "Bandwidth: 5B",
        f"State count: {STATE_COUNT}",
        f"Cluster threshold: {cluster_threshold:.17g} dB (strict CNMSE < threshold)",
        f"Cluster count: {b}",
        f"LUT fingerprint count: {b}",
        f"LUT DPD-entry count: {b}",
        "Representative selection: minimax medoid",
        "Model: Frozen unified MP",
        f"orders: {list(ORDERS)}",
        f"memory: {list(MEMORY)}",
        f"K: {sum(MEMORY)}",
        f"dmax: {DMAX}",
        f"Ridge lambda: {RIDGE_LAMBDA}",
        "Offline LUT fingerprint: Y-Aend model response on common B probe",
        "Online query fingerprint: Y-C2 model response on common B probe",
        f"Query count: {STATE_COUNT}",
        f"Retrieval candidates per query: {b}",
        f"Fingerprint compression: {STATE_COUNT} -> {b}",
        f"DPD-entry compression: {STATE_COUNT} -> {b}",
        f"Fingerprint compression ratio: {1.0 - b / STATE_COUNT:.9f}",
        f"DPD-entry compression ratio: {1.0 - b / STATE_COUNT:.9f}",
        f"Exact state hit: {int(frame['exact_state_hit'].sum())}/{STATE_COUNT} ({float(frame['exact_state_hit'].mean()):.9f})",
        f"Same-cluster hit: {int(frame['same_cluster_hit'].sum())}/{STATE_COUNT} ({float(frame['same_cluster_hit'].mean()):.9f})",
        f"Real-B threshold: {real_b_threshold:.17g} dB (strict <; exact self -Inf accepted)",
        f"Real-B below threshold: {int(frame['real_B_below_threshold'].sum())}/{STATE_COUNT} ({float(frame['real_B_below_threshold'].mean()):.9f})",
        f"Median retrieval fingerprint CNMSE: {float(frame['retrieval_fingerprint_CNMSE_dB'].replace(-np.inf, np.nan).median()):.9f} dB",
        f"Median retrieved Real-B CNMSE (non-exact finite): {float(nonexact_finite.median()) if not nonexact_finite.empty else float('nan'):.9f} dB",
        f"Worst retrieved Real-B CNMSE (finite): {float(finite_real_b.max()) if not finite_real_b.empty else float('nan'):.9f} dB",
        "",
        "Behavior model median metrics",
        f"Y_Aend_train_NMSE_dB: {float(medians['Y_Aend_train_NMSE_dB']):.9f}",
        f"Y_Aend_B_NMSE_dB: {float(medians['Y_Aend_B_NMSE_dB']):.9f}",
        f"Y_C2_train_NMSE_dB: {float(medians['Y_C2_train_NMSE_dB']):.9f}",
        f"Y_C2_B_NMSE_dB: {float(medians['Y_C2_B_NMSE_dB']):.9f}",
        "",
        "Gates",
        f"Frozen reproduction (states {list(STATE_REPRODUCTION_IDS)}): {model_gate['pass']}; max metric error={model_gate['maximum_metric_abs_error_dB']:.3g} dB; max fingerprint error={model_gate['maximum_fingerprint_abs_error']:.3g}",
        f"Clustering reproduction at -40 dB: {cluster_reproduction_pass}",
        f"Real-B helper reproduction: {real_b_gate['pass']}; max distance error={real_b_gate['max_abs_error_dB']:.3g} dB; tolerance={real_b_gate['reproduction_tolerance_dB']:.3g} dB; ranking error={real_b_gate['ranking_max_abs_error']:.3g}",
        f"LUT candidate IDs are representative IDs only: {set(frame['retrieved_state_id_Q']).issubset(set(clustering.representative_state_ids))}",
        f"One representative per cluster: {len(clustering.representatives) == b}",
        "Representative IDs state/fingerprint/DPD equal: True",
        "Real-B used for Top1 selection: False",
        "State_Q frozen before Real-B verification: True",
        "DPD replay performed: False",
        "Low-bandwidth processing performed: False",
        f"Pairwise clustering matrix action: {matrix_action}; symmetry={matrix_metadata.get('symmetry_mode')}",
        f"Logical CPUs: {os.cpu_count() or 1}",
        f"Workers: {workers}",
        "BLAS threads per worker: 1",
        "Nested parallelism: false",
        f"Raw data unchanged: {protection['data_raw_unchanged']}",
        f"Protected historical results unchanged: {protection['protected_results_unchanged']}",
        "No LUT-I/LUT-II or shared-DPD linearization performance was evaluated.",
        "",
        "Output files",
        f"- {threshold_root / 'type3_cluster_compressed_retrieval.xlsx'}",
        f"- {threshold_root / 'type3_cluster_compressed_retrieval.png'}",
        f"- {threshold_root / 'final_result_summary.txt'}",
    ]
    (threshold_root / "final_result_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def _write_temp_excel_source(frame: pd.DataFrame, clustering: Any, cluster_threshold: float, real_b_threshold: float) -> Path:
    source_path = Path(tempfile.gettempdir()) / f"type3_cluster_compressed_source_{os.getpid()}.json"
    columns = [
        "state_id_R", "funMng", "funAng", "secMng", "secAng", "nmse_withoutdpd_dB", "acpr_withoutdpd_mean_dBc",
        "Y_Aend_train_NMSE_dB", "Y_Aend_B_NMSE_dB", "Y_C2_train_NMSE_dB", "Y_C2_B_NMSE_dB", "true_cluster_id",
        "expected_representative_state_id", "retrieved_state_id_Q", "state_id_delta_Q_minus_R", "state_id_abs_delta",
        "retrieval_fingerprint_CNMSE_dB", "retrieved_real_B_CNMSE_dB", "exact_state_hit", "same_cluster_hit", "real_B_below_threshold",
    ]
    rows = []
    for row in frame[columns].to_dict("records"):
        for key, value in list(row.items()):
            if isinstance(value, (np.integer, np.floating, np.bool_)):
                value = value.item()
            if isinstance(value, float) and not np.isfinite(value):
                value = "-Inf" if value < 0 else "Inf"
            row[key] = value
        rows.append(row)
    payload = {"columns": columns, "rows": rows, "metadata": {"experiment": TASK_NAME, "cluster_threshold_db": float(cluster_threshold), "real_b_threshold_db": float(real_b_threshold), "cluster_count": int(len(clustering.representative_state_ids)), "model": "Frozen unified MP", "orders": list(ORDERS), "memory": list(MEMORY), "ridge_lambda": RIDGE_LAMBDA}}
    source_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return source_path


def _write_type3_figure(frame: pd.DataFrame, output_path: Path, cluster_threshold: float, lut_count: int) -> None:
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"], "font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.8, "legend.frameon": False})
    x = frame["state_id_R"].to_numpy(dtype=float)
    series = [
        ("nmse_withoutdpd_dB", "NMSE without DPD", "#4D4D4D", "-"),
        ("Y_Aend_train_NMSE_dB", "Y-Aend train", "#0F4D92", "-"),
        ("Y_Aend_B_NMSE_dB", "Y-Aend → B", "#3775BA", "-"),
        ("Y_C2_train_NMSE_dB", "Y-C2 train", "#42949E", "-"),
        ("Y_C2_B_NMSE_dB", "Y-C2 → B", "#9A4D8E", "-"),
    ]
    retrieved = frame["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float)
    finite_values = np.concatenate([frame[key].to_numpy(dtype=float) for key, *_ in series] + [retrieved[np.isfinite(retrieved)]])
    floor = float(np.nanmin(finite_values) - 2.0)
    retrieved_plot = np.where(np.isneginf(retrieved), floor, retrieved)
    fig, ax = plt.subplots(figsize=(32, 12), constrained_layout=True)
    for key, label, color, linestyle in series:
        ax.plot(x, frame[key].to_numpy(dtype=float), color=color, linewidth=1.15, label=label)
    ax.plot(x, retrieved_plot, color="#B64342", linewidth=1.0, marker="o", markersize=2.8, markeredgewidth=0.35, markeredgecolor="white", label="retrieved Real-B CNMSE")
    ax.axhline(-40.0, color="#767676", linestyle="--", linewidth=0.9, label="Real-B threshold (-40 dB)")
    offsets = ((0, 7), (0, -9), (0, 13), (0, -15), (0, 19), (0, -21))
    for index, (state_id, value, query_id) in enumerate(zip(x.astype(int), retrieved_plot, frame["retrieved_state_id_Q"].to_numpy(dtype=int), strict=True)):
        dx, dy = offsets[index % len(offsets)]
        ax.annotate(str(query_id), (state_id, value), xytext=(dx, dy), textcoords="offset points", ha="center", va="center", fontsize=4.2, color="#B64342", clip_on=True)
    ax.set_xlabel("State index R")
    ax.set_ylabel("Metric (dB)")
    ax.set_title(f"Scenario 2 Type-III Cluster-Compressed LUT Retrieval\n5B, cluster threshold = {cluster_threshold:g} dB, LUT fingerprints = {lut_count}", fontsize=12)
    ax.set_xlim(-2, STATE_COUNT + 1)
    ax.set_ylim(floor, max(0.0, float(np.nanmax(finite_values) + 1.0)))
    ax.legend(loc="best", ncol=2, fontsize=8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-threshold-db", type=float, default=CLUSTER_THRESHOLD_DEFAULT)
    parser.add_argument("--real-b-threshold-db", type=float, default=REAL_B_THRESHOLD_DEFAULT)
    parser.add_argument("--force-recompute-matrix", action="store_true", help="reserved for the clustering runner; Type III never overwrites the source matrix")
    return parser.parse_args()


def main() -> None:
    freeze_support()
    args = _parse_args()
    if not np.isfinite(args.cluster_threshold_db) or not np.isfinite(args.real_b_threshold_db):
        raise ValueError("thresholds must be finite")
    if abs(args.cluster_threshold_db - CLUSTER_THRESHOLD_DEFAULT) > 1e-12:
        raise ValueError("This first Type-III run is frozen to the validated -40 dB cluster threshold")
    before = _snapshot()
    logical_cpu_count = int(os.cpu_count() or 1)
    workers = max(1, int(math.floor(CPU_TARGET * logical_cpu_count)))
    print(f"Scenario 2 Type-III compressed LUT; states={STATE_COUNT}; cluster_threshold={args.cluster_threshold_db:g} dB; real_B_threshold={args.real_b_threshold_db:g} dB", flush=True)
    print(f"logical_cpu_count={logical_cpu_count}; worker_count={workers}; BLAS=1", flush=True)
    _matrix, _cluster_assignments, clustering, matrix_metadata = _load_cluster_inputs(float(args.cluster_threshold_db))
    print(f"Clustering: {len(clustering.clusters)} clusters; {len(clustering.representative_state_ids)} representatives", flush=True)
    common_b = _load_common_b()
    metrics, a_fp, c_fp = _run_state_models(common_b, workers)
    model_gate = _frozen_reproduction_gate(metrics, a_fp, c_fp)
    if not model_gate["pass"]:
        raise RuntimeError(f"Frozen Aend/C2 reproduction failed: {model_gate}")
    print("Frozen Aend/C2 reproduction: PASS", flush=True)
    representative_ids = np.asarray(clustering.representative_state_ids, dtype=np.int64)
    lut_fp = a_fp[representative_ids]
    if lut_fp.shape != (len(representative_ids), FINGERPRINT_LENGTH) or c_fp.shape != (STATE_COUNT, FINGERPRINT_LENGTH):
        raise RuntimeError("Type-III fingerprint compression shape gate failed")
    retrieval = _run_retrieval(c_fp, lut_fp, representative_ids, workers)
    if retrieval.shape[0] != STATE_COUNT or not retrieval["retrieved_state_id_Q"].isin(representative_ids).all():
        raise RuntimeError("Type-III query/candidate gate failed")
    print("Query Top-1: all 425 states matched only representative candidates", flush=True)
    # State_Q is frozen here; Real-B is intentionally loaded only below this point.
    real_b = _run_real_b(workers)
    real_b_gate = _real_b_gate(real_b)
    if not real_b_gate["pass"]:
        raise RuntimeError(f"Real-B helper reproduction failed: {real_b_gate}")
    print("Real-B helper reproduction: PASS", flush=True)
    frame = _build_final_frame(metrics, clustering, retrieval, real_b_gate["distance"], float(args.real_b_threshold_db))
    threshold_root = RESULT_ROOT / "threshold_m40p0dB"
    threshold_root.mkdir(parents=True, exist_ok=True)
    figure_path = threshold_root / "type3_cluster_compressed_retrieval.png"
    _write_type3_figure(frame, figure_path, float(args.cluster_threshold_db), len(representative_ids))
    source_path = _write_temp_excel_source(frame, clustering, float(args.cluster_threshold_db), float(args.real_b_threshold_db))
    after = _snapshot()
    protection = _protection_check(before, after)
    if not protection["all_protected_unchanged"]:
        raise RuntimeError("raw data or protected historical result changed")
    _write_summary(frame, clustering, float(args.cluster_threshold_db), float(args.real_b_threshold_db), matrix_metadata, model_gate, real_b_gate, protection, workers, "reused", threshold_root, abs(float(args.cluster_threshold_db) - CLUSTER_THRESHOLD_DEFAULT) <= 1e-12)
    print(f"Temporary Excel source: {source_path}", flush=True)
    print(f"Exact hit: {int(frame['exact_state_hit'].sum())}/{STATE_COUNT}; same-cluster hit: {int(frame['same_cluster_hit'].sum())}/{STATE_COUNT}", flush=True)
    print(f"Real-B below threshold: {int(frame['real_B_below_threshold'].sum())}/{STATE_COUNT}", flush=True)
    print(f"Result path: {threshold_root}", flush=True)


if __name__ == "__main__":
    main()
