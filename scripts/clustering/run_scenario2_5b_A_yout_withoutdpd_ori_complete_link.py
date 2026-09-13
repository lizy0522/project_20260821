"""Cluster Scenario-2 5B-A raw without-DPD outputs by complete-link CNMSE.

The formal signal path is intentionally explicit:

    full xin/yout_withoutdpd_ori alignment -> fixed A split -> A-only gain
    adjustment -> pairwise CNMSE -> deterministic complete-link clustering.

The pairwise matrix is reusable across thresholds.  Only the matrix and the
threshold-specific assignment/summary/summary-text artifacts are persisted.
"""

# ruff: noqa: E402,E501,I001

from __future__ import annotations

import argparse
import hashlib
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
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

import numpy as np
import pandas as pd

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _core.behavior_indexed_dpd.clustering import (
    complete_link_threshold_cluster,
    summarize_clusters,
    validate_clusters,
)
from _core.metrics import cnmse
from _core.signal import adjust_complex_gain, fine_align, rough_align
from data_manager import load_by_id
from signal_segmentation import build_partition_from_xin, get_off_pair, preprocess_full_pair


TASK_NAME = "scenario_2_yout_withoutdpd_ori_5B_A_complete_link_clustering"
RESULT_ROOT = PROJECT_ROOT / "results" / "clustering" / TASK_NAME
WORK_LOG = PROJECT_ROOT / "work_logs" / "clustering" / "execution_log.txt"
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "codex_handoff" / "codex_handoff.txt"
MATRIX_FILENAME = "pairwise_cnmse_5B_A_preprocessed.npz"
STATE_COUNT = 425
FULL_LENGTH = 24576
A_START = 0
A_END = 12288
SAMPLE_RATE_HZ = 100_000_000
BASE_BANDWIDTH_HZ = 20_000_000
PAIR_CHUNK_SIZE = 256
CPU_TARGET = 0.90
FINE_ALIGN_SUBTIME = 256
SYMMETRY_PAIRS = ((0, 1), (0, 100), (10, 200), (187, 340), (300, 424))
PREPROCESS_ORDER = "full_time_align_then_A_split_then_A_complex_gain"

PROTECTED_RESULT_DIRS = {
    "formal_retrieval": PROJECT_ROOT / "results" / "behavior_fingerprint_lut_retrieval" / "scenario_2_C2_to_Aend",
    "all_ilc_frozen": PROJECT_ROOT / "results" / "behavior_fingerprint_ranking_consistency" / "scenario_2_all_ilc",
    "unified_capacity": PROJECT_ROOT / "results" / "behavior_model" / "scenario_2_unified_odd_order_mp_capacity_scan_5B",
    "odd_order_scan": PROJECT_ROOT / "results" / "behavior_model" / "scenario_2_failed14_odd_order_mp_order_dependent_memory_scan_5B",
    "p5_ridge": PROJECT_ROOT / "results" / "behavior_model" / "scenario_2_failed14_P5_variable_memory_ridge_scan_5B",
    "p5_order2": PROJECT_ROOT / "results" / "behavior_model" / "scenario_2_failed14_P5_with_order2_variable_memory_OLS_scan_5B",
    "frozen_neighborhood": PROJECT_ROOT / "results" / "behavior_model" / "scenario_2_failed14_frozen_neighborhood_memory_ridge_scan_5B",
    "sparse_gmp": PROJECT_ROOT / "results" / "behavior_model" / "scenario_2_failed14_frozen_mp_residual_sparse_gmp_5B",
    "multibranch": PROJECT_ROOT / "results" / "behavior_model" / "scenario_2_failed14_multibranch_independent_basis_OLS_scan_5B",
    "normalized_uniform": PROJECT_ROOT / "results" / "behavior_model" / "scenario_2_failed14_frozen_mp_normalized_uniform_ridge_5B",
    "normalized_dependent": PROJECT_ROOT / "results" / "behavior_model" / "scenario_2_failed14_frozen_mp_normalized_basis_dependent_ridge_5B",
}


@dataclass(frozen=True)
class PreparedA:
    state_id: int
    y_a: np.ndarray
    fun_mng: int
    fun_ang: int
    sec_mng: int
    sec_ang: int
    v_carrier: float
    input_power: float
    freq_sample_hz: float
    bandwidth_mhz: float
    rough_delay: int
    fine_delay: float
    complex_gain: complex


def _tree_digest(path: Path) -> str | None:
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    for file in sorted(path.rglob("*"), key=lambda item: str(item).lower()):
        if not file.is_file() or "__pycache__" in file.parts or file.suffix.lower() == ".pyc":
            continue
        content = file.read_bytes()
        digest.update(file.relative_to(path).as_posix().encode("utf-8") + b"\0")
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def _raw_manifest() -> dict[str, Any]:
    raw_root = PROJECT_ROOT / "data" / "raw"
    if not raw_root.is_dir():
        return {"exists": False}
    files = [file for file in raw_root.rglob("*") if file.is_file()]
    return {
        "exists": True,
        "file_count": len(files),
        "bytes": int(sum(file.stat().st_size for file in files)),
        "sha256": _tree_digest(raw_root),
    }


def _uniform_snapshot() -> dict[str, Any]:
    return {
        "data/raw": _raw_manifest(),
        "result_dirs": {name: _tree_digest(path) for name, path in PROTECTED_RESULT_DIRS.items()},
    }


def _uniform_protection_check(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    raw_ok = before["data/raw"] == after["data/raw"]
    result_checks = {
        name: {
            "sha256_unchanged": before["result_dirs"].get(name) == after["result_dirs"].get(name),
            "before": before["result_dirs"].get(name),
            "after": after["result_dirs"].get(name),
        }
        for name in PROTECTED_RESULT_DIRS
    }
    protected_ok = bool(all(item["sha256_unchanged"] for item in result_checks.values()))
    return {
        "data_raw_unchanged": raw_ok,
        "protected_result_dirs_unchanged": protected_ok,
        "all_protected_unchanged": bool(raw_ok and protected_ok),
        "data_raw": {"before": before["data/raw"], "after": after["data/raw"]},
        "result_dirs": result_checks,
    }


def _scalar(data: dict[str, Any], key: str) -> float:
    if key not in data:
        raise KeyError(f"state data missing {key!r}")
    value = np.asarray(data[key]).reshape(-1)
    if value.size != 1:
        raise ValueError(f"{key} must be scalar, got shape {np.asarray(data[key]).shape}")
    result = float(value[0])
    if not np.isfinite(result):
        raise ValueError(f"{key} is not finite")
    return result


def prepare_state_5b_A_behavior(state_id: int) -> PreparedA:
    """Prepare one raw without-DPD state through the frozen A-only path."""

    state_id = int(state_id)
    data = load_by_id(state_id)
    xin_raw = np.asarray(data["xin"])
    y_raw_input = np.asarray(data["yout_withoutdpd_ori"])
    if xin_raw.ndim == 2 and xin_raw.shape[1] == 1:
        xin_raw = xin_raw[:, 0]
    if y_raw_input.ndim == 2 and y_raw_input.shape[1] == 1:
        y_raw_input = y_raw_input[:, 0]
    if xin_raw.size != FULL_LENGTH or y_raw_input.size != FULL_LENGTH:
        raise ValueError(f"state {state_id} requires xin/yout length {FULL_LENGTH}")
    if not np.iscomplexobj(xin_raw) or not np.iscomplexobj(y_raw_input) or not np.all(np.isfinite(xin_raw)) or not np.all(np.isfinite(y_raw_input)):
        raise ValueError(f"state {state_id} xin/yout must be finite complex arrays")
    xin = np.asarray(xin_raw, dtype=np.complex128).reshape(-1)
    y_raw = np.asarray(y_raw_input, dtype=np.complex128).reshape(-1)
    freq_sample_hz = _scalar(data, "freqSample_Hz")
    bandwidth_mhz = _scalar(data, "bandWidth_MHz")
    if not np.isclose(freq_sample_hz, SAMPLE_RATE_HZ, rtol=0.0, atol=0.0) or not np.isclose(bandwidth_mhz, BASE_BANDWIDTH_HZ / 1e6, rtol=0.0, atol=0.0):
        raise ValueError(f"state {state_id} bandwidth metadata is not 100 MHz/20 MHz")
    if not np.isclose(freq_sample_hz / (bandwidth_mhz * 1e6), 5.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"state {state_id} is not a 5B record")
    y_rough, rough_delay = rough_align(xin, y_raw)
    y_fine, fine_delay = fine_align(xin, y_rough, subtime=FINE_ALIGN_SUBTIME)
    x_a = xin[A_START:A_END]
    y_a = np.asarray(y_fine[A_START:A_END], dtype=np.complex128)
    y_a_adjusted, complex_gain = adjust_complex_gain(x_a, y_a)
    if y_a_adjusted.shape != (A_END - A_START,) or not np.all(np.isfinite(y_a_adjusted)):
        raise ValueError(f"state {state_id} adjusted A output is invalid")
    return PreparedA(
        state_id=state_id,
        y_a=np.asarray(y_a_adjusted, dtype=np.complex128),
        fun_mng=int(_scalar(data, "funMng")),
        fun_ang=int(_scalar(data, "funAng")),
        sec_mng=int(_scalar(data, "secMng")),
        sec_ang=int(_scalar(data, "secAng")),
        v_carrier=_scalar(data, "v_carrier"),
        input_power=_scalar(data, "inputPower"),
        freq_sample_hz=freq_sample_hz,
        bandwidth_mhz=bandwidth_mhz,
        rough_delay=int(rough_delay),
        fine_delay=float(fine_delay),
        complex_gain=complex_gain,
    )


def _preprocess_worker(state_id: int) -> PreparedA:
    return prepare_state_5b_A_behavior(state_id)


def _preprocess_all(worker_count: int) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any]]:
    rows: dict[int, PreparedA] = {}
    errors: list[str] = []
    state_ids = list(range(STATE_COUNT))
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {executor.submit(_preprocess_worker, state_id): state_id for state_id in state_ids}
        checkpoints = {max(1, math.ceil(len(state_ids) * fraction)) for fraction in (0.25, 0.5, 0.75, 1.0)}
        for done, future in enumerate(as_completed(futures), start=1):
            state_id = futures[future]
            try:
                item = future.result()
                rows[item.state_id] = item
            except Exception as exc:  # pragma: no cover
                errors.append(f"state {state_id}: {type(exc).__name__}: {exc}")
            if done in checkpoints:
                print(f"Preprocessing: {done}/{len(state_ids)} ({done / len(state_ids):.0%})", flush=True)
    if errors:
        raise RuntimeError("preprocessing failure: " + " | ".join(errors[:5]))
    if sorted(rows) != state_ids:
        raise RuntimeError("preprocessing did not return all state IDs")
    y_matrix = np.stack([rows[state_id].y_a for state_id in state_ids]).astype(np.complex128, copy=False)
    metadata_rows = [
        {
            "state_id": item.state_id,
            "funMng": item.fun_mng,
            "funAng": item.fun_ang,
            "secMng": item.sec_mng,
            "secAng": item.sec_ang,
            "v_carrier": item.v_carrier,
            "inputPower": item.input_power,
        }
        for item in (rows[state_id] for state_id in state_ids)
    ]
    diagnostics = {
        "all_preprocessing_passed": True,
        "max_abs_rough_delay": int(max(abs(item.rough_delay) for item in rows.values())),
        "max_abs_fine_delay": float(max(abs(item.fine_delay) for item in rows.values())),
        "rough_delay_range": [int(min(item.rough_delay for item in rows.values())), int(max(item.rough_delay for item in rows.values()))],
        "fine_delay_range": [float(min(item.fine_delay for item in rows.values())), float(max(item.fine_delay for item in rows.values()))],
        "alignment_executed_before_A_split": True,
        "A_split_before_complex_gain": True,
        "A_segment_time_realignment": False,
    }
    return y_matrix, pd.DataFrame(metadata_rows), diagnostics


def _preprocessing_reproduction_gate() -> dict[str, Any]:
    checks = []
    for state_id in (0, 100, 187, 340, 424):
        wrapper = prepare_state_5b_A_behavior(state_id)
        data = load_by_id(state_id)
        xin = np.asarray(data["xin"])
        y_raw = np.asarray(data["yout_withoutdpd_ori"])
        if xin.ndim == 2 and xin.shape[1] == 1:
            xin = xin[:, 0]
        if y_raw.ndim == 2 and y_raw.shape[1] == 1:
            y_raw = y_raw[:, 0]
        partition = build_partition_from_xin(np.asarray(xin))
        off_pair = get_off_pair(data)
        canonical = preprocess_full_pair(off_pair.input_full, off_pair.output_raw_full, partition, pair_type="OFF")
        reference = np.asarray(canonical["A"].output, dtype=np.complex128)
        relative_error = float(np.linalg.norm(wrapper.y_a - reference) / max(np.linalg.norm(reference), 1e-30))
        checks.append({"state_id": state_id, "relative_l2_error": relative_error, "pass": relative_error < 1e-12})
    return {"pass": bool(all(item["pass"] for item in checks)), "checks": checks}


_PAIR_SHARED: np.ndarray | None = None
_PAIR_SHM: SharedMemory | None = None


def _pair_worker_init(shared_name: str, shape: tuple[int, int], dtype_name: str) -> None:
    global _PAIR_SHARED, _PAIR_SHM
    _PAIR_SHM = SharedMemory(name=shared_name)
    _PAIR_SHARED = np.ndarray(shape, dtype=np.dtype(dtype_name), buffer=_PAIR_SHM.buf)


def _pair_chunk_worker(payload: tuple[list[tuple[int, int]], str]) -> list[tuple[int, int, float]]:
    pairs, symmetry_mode = payload
    if _PAIR_SHARED is None:
        raise RuntimeError("pair worker shared array is not initialized")
    result: list[tuple[int, int, float]] = []
    for left, right in pairs:
        forward = float(cnmse(_PAIR_SHARED[left], _PAIR_SHARED[right]))
        if symmetry_mode == "max_of_two_directions":
            reverse = float(cnmse(_PAIR_SHARED[right], _PAIR_SHARED[left]))
            value = max(forward, reverse)
        else:
            value = forward
        if not np.isfinite(value):
            raise ValueError(f"non-finite pairwise CNMSE at ({left},{right})")
        result.append((left, right, value))
    return result


def _determine_symmetry(y_matrix: np.ndarray) -> tuple[str, list[dict[str, Any]]]:
    checks: list[dict[str, Any]] = []
    for left, right in SYMMETRY_PAIRS:
        forward = float(cnmse(y_matrix[left], y_matrix[right]))
        reverse = float(cnmse(y_matrix[right], y_matrix[left]))
        checks.append({"i": left, "j": right, "forward_dB": forward, "reverse_dB": reverse, "abs_difference_dB": abs(forward - reverse)})
    if all(item["abs_difference_dB"] < 1e-12 for item in checks):
        return "native_symmetric", checks
    return "max_of_two_directions", checks


def _pairwise_matrix(y_matrix: np.ndarray, worker_count: int, symmetry_mode: str) -> np.ndarray:
    pair_list = [(left, right) for left in range(STATE_COUNT) for right in range(left + 1, STATE_COUNT)]
    chunks = [pair_list[start : start + PAIR_CHUNK_SIZE] for start in range(0, len(pair_list), PAIR_CHUNK_SIZE)]
    matrix = np.full((STATE_COUNT, STATE_COUNT), -np.inf, dtype=np.float64)
    shared = SharedMemory(create=True, size=y_matrix.nbytes)
    try:
        shared_array = np.ndarray(y_matrix.shape, dtype=y_matrix.dtype, buffer=shared.buf)
        shared_array[:] = y_matrix
        with ProcessPoolExecutor(max_workers=worker_count, initializer=_pair_worker_init, initargs=(shared.name, y_matrix.shape, y_matrix.dtype.str)) as executor:
            futures = [executor.submit(_pair_chunk_worker, (chunk, symmetry_mode)) for chunk in chunks]
            checkpoints = {max(1, math.ceil(len(futures) * fraction)) for fraction in (0.25, 0.5, 0.75, 1.0)}
            for done, future in enumerate(as_completed(futures), start=1):
                for left, right, value in future.result():
                    matrix[left, right] = value
                    matrix[right, left] = value
                if done in checkpoints:
                    print(f"Pairwise CNMSE: {done}/{len(futures)} chunks ({done / len(futures):.0%})", flush=True)
    finally:
        shared.close()
        shared.unlink()
    return matrix


def _matrix_metadata_valid(data: Any) -> tuple[np.ndarray, dict[str, Any]]:
    matrix = np.asarray(data["cnmse_db"], dtype=np.float64)
    state_ids = np.asarray(data["state_ids"], dtype=np.int64)
    if matrix.shape != (STATE_COUNT, STATE_COUNT) or not np.array_equal(state_ids, np.arange(STATE_COUNT, dtype=np.int64)):
        raise ValueError("pairwise matrix shape/state order mismatch")
    if not np.all(np.isneginf(np.diag(matrix))) or not np.allclose(matrix, matrix.T, rtol=0.0, atol=0.0) or not np.all(np.isfinite(matrix[~np.eye(STATE_COUNT, dtype=bool)])):
        raise ValueError("pairwise matrix symmetry/finite/diagonal gate failed")
    def text(name: str) -> str:
        value = np.asarray(data[name]).reshape(-1)
        return str(value[0])
    metadata = {
        "signal_variable": text("signal_variable"),
        "bandwidth_mode": text("bandwidth_mode"),
        "preprocess_order": text("preprocess_order"),
        "cnmse_symmetry_mode": text("cnmse_symmetry_mode"),
        "freq_sample_hz": float(np.asarray(data["freq_sample_hz"]).reshape(-1)[0]),
        "base_bandwidth_hz": float(np.asarray(data["base_bandwidth_hz"]).reshape(-1)[0]),
        "full_record_length": int(np.asarray(data["full_record_length"]).reshape(-1)[0]),
        "A_start": int(np.asarray(data["A_start"]).reshape(-1)[0]),
        "A_end": int(np.asarray(data["A_end"]).reshape(-1)[0]),
    }
    expected = {
        "signal_variable": "yout_withoutdpd_ori",
        "bandwidth_mode": "5B",
        "preprocess_order": PREPROCESS_ORDER,
        "freq_sample_hz": SAMPLE_RATE_HZ,
        "base_bandwidth_hz": BASE_BANDWIDTH_HZ,
        "full_record_length": FULL_LENGTH,
        "A_start": A_START,
        "A_end": A_END,
    }
    for key, value in expected.items():
        if metadata[key] != value:
            raise ValueError(f"pairwise matrix metadata mismatch at {key}: {metadata[key]!r} != {value!r}")
    return matrix, metadata


def _load_or_build_matrix(matrix_path: Path, worker_count: int, force_recompute: bool) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    if matrix_path.is_file() and not force_recompute:
        with np.load(matrix_path, allow_pickle=False) as data:
            matrix, metadata = _matrix_metadata_valid(data)
            state_frame = pd.DataFrame({
                "state_id": np.asarray(data["state_ids"], dtype=np.int64),
                "funMng": np.asarray(data["funMng"], dtype=np.int64),
                "funAng": np.asarray(data["funAng"], dtype=np.int64),
                "secMng": np.asarray(data["secMng"], dtype=np.int64),
                "secAng": np.asarray(data["secAng"], dtype=np.int64),
                "v_carrier": np.asarray(data["v_carrier"], dtype=np.float64),
                "inputPower": np.asarray(data["inputPower"], dtype=np.float64),
            })
            diagnostics = {
                "all_preprocessing_passed": bool(np.asarray(data["all_preprocessing_passed"]).reshape(-1)[0]),
                "max_abs_rough_delay": int(np.asarray(data["max_abs_rough_delay"]).reshape(-1)[0]),
                "max_abs_fine_delay": float(np.asarray(data["max_abs_fine_delay"]).reshape(-1)[0]),
                "alignment_executed_before_A_split": True,
                "A_split_before_complex_gain": True,
                "A_segment_time_realignment": False,
            }
        print("Pairwise matrix: reused", flush=True)
        return matrix, state_frame, metadata, diagnostics

    y_matrix, state_frame, diagnostics = _preprocess_all(worker_count)
    symmetry_mode, symmetry_checks = _determine_symmetry(y_matrix)
    matrix = _pairwise_matrix(y_matrix, worker_count, symmetry_mode)
    if matrix.shape != (STATE_COUNT, STATE_COUNT) or not np.all(np.isneginf(np.diag(matrix))) or not np.allclose(matrix, matrix.T, rtol=0.0, atol=0.0) or not np.all(np.isfinite(matrix[~np.eye(STATE_COUNT, dtype=bool)])):
        raise RuntimeError("pairwise matrix gate failed")
    metadata = {"signal_variable": "yout_withoutdpd_ori", "bandwidth_mode": "5B", "preprocess_order": PREPROCESS_ORDER, "cnmse_symmetry_mode": symmetry_mode, "freq_sample_hz": SAMPLE_RATE_HZ, "base_bandwidth_hz": BASE_BANDWIDTH_HZ, "full_record_length": FULL_LENGTH, "A_start": A_START, "A_end": A_END}
    matrix_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        matrix_path,
        cnmse_db=matrix,
        state_ids=state_frame["state_id"].to_numpy(dtype=np.int64),
        funMng=state_frame["funMng"].to_numpy(dtype=np.int64),
        funAng=state_frame["funAng"].to_numpy(dtype=np.int64),
        secMng=state_frame["secMng"].to_numpy(dtype=np.int64),
        secAng=state_frame["secAng"].to_numpy(dtype=np.int64),
        v_carrier=state_frame["v_carrier"].to_numpy(dtype=np.float64),
        inputPower=state_frame["inputPower"].to_numpy(dtype=np.float64),
        signal_variable=np.asarray(metadata["signal_variable"]),
        bandwidth_mode=np.asarray(metadata["bandwidth_mode"]),
        preprocess_order=np.asarray(metadata["preprocess_order"]),
        cnmse_symmetry_mode=np.asarray(metadata["cnmse_symmetry_mode"]),
        freq_sample_hz=np.asarray(metadata["freq_sample_hz"], dtype=np.int64),
        base_bandwidth_hz=np.asarray(metadata["base_bandwidth_hz"], dtype=np.int64),
        full_record_length=np.asarray(metadata["full_record_length"], dtype=np.int64),
        A_start=np.asarray(metadata["A_start"], dtype=np.int64),
        A_end=np.asarray(metadata["A_end"], dtype=np.int64),
        all_preprocessing_passed=np.asarray(diagnostics["all_preprocessing_passed"]),
        max_abs_rough_delay=np.asarray(diagnostics["max_abs_rough_delay"], dtype=np.int64),
        max_abs_fine_delay=np.asarray(diagnostics["max_abs_fine_delay"], dtype=np.float64),
    )
    metadata["symmetry_checks"] = symmetry_checks
    print("Pairwise matrix: computed and saved", flush=True)
    return matrix, state_frame, metadata, diagnostics


def _threshold_directory(threshold: float) -> Path:
    sign = "m" if threshold < 0 else "p"
    token = f"{abs(float(threshold)):.1f}".replace(".", "p")
    return RESULT_ROOT / f"threshold_{sign}{token}dB"


def _write_cluster_results(matrix: np.ndarray, state_frame: pd.DataFrame, threshold: float, matrix_metadata: dict[str, Any], diagnostics: dict[str, Any], worker_count: int, matrix_action: str, preprocessing_gate: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    clusters = complete_link_threshold_cluster(matrix, threshold)
    validation = validate_clusters(matrix, clusters, threshold)
    if not validation["all_gates_pass"]:
        raise RuntimeError(f"clustering gate failed: {validation}")
    summaries = summarize_clusters(matrix, clusters)
    assignments: list[dict[str, Any]] = []
    for cluster_id, members in enumerate(clusters):
        for state_id in members:
            row = state_frame.loc[state_frame["state_id"] == state_id].iloc[0]
            assignments.append({"state_id": int(state_id), "cluster_id": int(cluster_id), "cluster_size": len(members), "funMng": int(row["funMng"]), "funAng": int(row["funAng"]), "secMng": int(row["secMng"]), "secAng": int(row["secAng"]), "v_carrier": float(row["v_carrier"]), "inputPower": float(row["inputPower"])})
    assignment_frame = pd.DataFrame(assignments).sort_values("state_id").reset_index(drop=True)
    summary_frame = pd.DataFrame(summaries).sort_values("cluster_id").reset_index(drop=True)
    threshold_root = _threshold_directory(threshold)
    threshold_root.mkdir(parents=True, exist_ok=True)
    assignment_frame.to_csv(threshold_root / "cluster_assignments.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    summary_frame.to_csv(threshold_root / "cluster_summary.csv", index=False, encoding="utf-8-sig", float_format="%.17g")
    within_worst = float(summary_frame["worst_pair_cnmse_db"].max()) if not summary_frame.empty else float("-inf")
    pair_clusters = summary_frame.loc[summary_frame["pair_count"] > 0]
    within_best = float(pair_clusters["best_pair_cnmse_db"].min()) if not pair_clusters.empty else float("-inf")
    singleton_count = int((summary_frame["cluster_size"] == 1).sum()) if not summary_frame.empty else 0
    summary_lines = [
        f"Experiment: {TASK_NAME}",
        "Scenario: 2",
        f"State count: {STATE_COUNT}",
        "Signal: yout_withoutdpd_ori",
        "Bandwidth: 5B",
        "Sample rate: 100 MHz",
        "Base bandwidth: 20 MHz",
        "Preprocessing:",
        "1. Full-record rough alignment of yout_withoutdpd_ori to xin",
        "2. Full-record fine alignment to xin",
        "3. Fixed A split [0:12288)",
        "4. A-segment-only complex-gain adjustment relative to xin_A",
        "5. No A-segment time realignment after gain adjustment",
        f"Preprocessing order: {PREPROCESS_ORDER}",
        f"Max absolute rough delay: {diagnostics.get('max_abs_rough_delay', 'reused-metadata')}",
        f"Max absolute fine delay: {diagnostics.get('max_abs_fine_delay', 'reused-metadata')}",
        f"All preprocessing passed: {diagnostics.get('all_preprocessing_passed', False)}",
        "CNMSE definition: existing project _core.metrics.cnmse",
        f"CNMSE symmetry mode: {matrix_metadata.get('cnmse_symmetry_mode', 'unknown')}",
        f"Threshold: {threshold:.17g} dB",
        "Threshold rule: strict D_ij < threshold",
        f"Pairwise matrix: {STATE_COUNT} x {STATE_COUNT}",
        "Unique state pairs: 90100",
        f"Pairwise matrix action: {matrix_action}",
        "Clustering: deterministic complete-link threshold clustering",
        f"Cluster count: {len(clusters)}",
        f"Largest cluster size: {max((len(cluster) for cluster in clusters), default=0)}",
        f"Smallest cluster size: {min((len(cluster) for cluster in clusters), default=0)}",
        f"Median cluster size: {float(np.median([len(cluster) for cluster in clusters])) if clusters else 0.0:.6g}",
        f"Singleton cluster count: {singleton_count}",
        f"Global worst within-cluster CNMSE: {within_worst:.9f} dB",
        f"Global best within-cluster CNMSE: {within_best:.9f} dB",
        f"All within-cluster pairs satisfy threshold: {validation['within_cluster_pass']}",
        f"All 425 states assigned exactly once: {validation['coverage_pass']}",
        f"Any two final clusters still mergeable: {not validation['non_mergeability_pass']}",
        f"Logical CPUs: {os.cpu_count() or 1}",
        f"Workers: {worker_count}",
        "BLAS threads per worker: 1",
        "Nested parallelism: false",
        f"Preprocessing reproduction gate: {preprocessing_gate['pass']}",
        f"All gates PASS: {validation['all_gates_pass'] and preprocessing_gate['pass']}",
        "No model/Ridge/GMP/LUT/retrieval/low-bandwidth processing was performed.",
    ]
    summary_path = threshold_root / "final_result_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    return threshold_root, {"clusters": clusters, "validation": validation, "summary_frame": summary_frame, "assignment_frame": assignment_frame, "within_worst": within_worst, "within_best": within_best, "singleton_count": singleton_count}


def _synthetic_gates() -> dict[str, Any]:
    chain = np.asarray([[-np.inf, -45.0, -37.0], [-45.0, -np.inf, -43.0], [-37.0, -43.0, -np.inf]])
    compatible = np.asarray([[-np.inf, -45.0, -42.0], [-45.0, -np.inf, -43.0], [-42.0, -43.0, -np.inf]])
    strict = np.asarray([[-np.inf, -40.0], [-40.0, -np.inf]])
    chain_clusters = complete_link_threshold_cluster(chain, -40.0)
    compatible_clusters = complete_link_threshold_cluster(compatible, -40.0)
    strict_clusters = complete_link_threshold_cluster(strict, -40.0)
    return {
        "chain_gate": bool(tuple(sorted(chain_clusters)) != ((0, 1, 2),)),
        "compatible_gate": bool(tuple(sorted(compatible_clusters)) == ((0, 1, 2),)),
        "strict_less_gate": bool(tuple(sorted(strict_clusters)) == ((0,), (1,))),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cnmse-threshold-db", type=float, default=-40.0, dest="threshold")
    parser.add_argument("--force-recompute-matrix", action="store_true")
    return parser.parse_args()


def main() -> None:
    freeze_support()
    args = _parse_args()
    if not np.isfinite(args.threshold):
        raise ValueError("--cnmse-threshold-db must be finite")
    if not all(0 <= left < STATE_COUNT and 0 <= right < STATE_COUNT for left, right in SYMMETRY_PAIRS):
        raise RuntimeError("symmetry test pair outside state range")
    synthetic = _synthetic_gates()
    if not all(synthetic.values()):
        raise RuntimeError(f"synthetic clustering gate failed: {synthetic}")
    before = _uniform_snapshot()
    logical_cpu_count = int(os.cpu_count() or 1)
    worker_count = max(1, int(math.floor(CPU_TARGET * logical_cpu_count)))
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    matrix_path = RESULT_ROOT / MATRIX_FILENAME
    matrix_exists_before = matrix_path.is_file()
    preprocessing_gate = _preprocessing_reproduction_gate() if (args.force_recompute_matrix or not matrix_exists_before) else {"pass": True, "checks": [], "reused_matrix": True}
    if not preprocessing_gate["pass"]:
        raise RuntimeError(f"preprocessing reproduction gate failed: {preprocessing_gate}")
    matrix, state_frame, matrix_metadata, diagnostics = _load_or_build_matrix(matrix_path, worker_count, args.force_recompute_matrix)
    matrix_action = "recomputed" if args.force_recompute_matrix or not matrix_exists_before else "reused"
    threshold_root, clustering = _write_cluster_results(matrix, state_frame, float(args.threshold), matrix_metadata, diagnostics, worker_count, matrix_action, preprocessing_gate)
    after = _uniform_snapshot()
    protection = _uniform_protection_check(before, after)
    if not protection["all_protected_unchanged"]:
        raise RuntimeError("raw data or protected historical result changed")
    log = (
        f"{TASK_NAME} completed: threshold={args.threshold:.17g} dB, states={STATE_COUNT}, clusters={len(clustering['clusters'])}, "
        f"matrix={matrix_action}, workers={worker_count}, "
        f"synthetic_gates={synthetic}, preprocessing_gate={preprocessing_gate['pass']}, clustering_gates={clustering['validation']['all_gates_pass']}, "
        f"raw/protected unchanged={protection['all_protected_unchanged']}. Result: {threshold_root}"
    )
    WORK_LOG.parent.mkdir(parents=True, exist_ok=True)
    HANDOFF_LOG.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    with WORK_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n[{timestamp}] {TASK_NAME}\n{log}\n")
    with HANDOFF_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n{timestamp} | clustering 新派生任务：Scenario 2 5B-A yout_withoutdpd_ori complete-link\n{log}\n")
    print(f"Threshold = {args.threshold:.17g} dB", flush=True)
    print(f"Cluster count = {len(clustering['clusters'])}", flush=True)
    print(f"Largest cluster = {max((len(cluster) for cluster in clustering['clusters']), default=0)}", flush=True)
    print(f"Median cluster size = {float(np.median([len(cluster) for cluster in clustering['clusters']])) if clustering['clusters'] else 0.0:.6g}", flush=True)
    print(f"Singleton count = {clustering['singleton_count']}", flush=True)
    print(f"Global worst within-cluster CNMSE = {clustering['within_worst']:.9f} dB", flush=True)
    print("All gates PASS", flush=True)
    print(f"Result path: {threshold_root}", flush=True)


if __name__ == "__main__":
    main()
