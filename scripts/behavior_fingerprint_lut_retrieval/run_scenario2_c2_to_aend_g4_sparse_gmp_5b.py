"""Formal 5B C2→Aend retrieval using the frozen G4 Sparse-GMP model.

G4 is loaded from the completed Sparse-GMP discovery result and is never
reselected here.  Every state is prepared with the canonical full-record
pipeline, fitted independently for Aend and C2, projected on the frozen common
B probe, and then compared in a 425×425 behavior-fingerprint matrix.  The
historical Frozen retrieval and Real-B matrix are read-only references.
"""

# ruff: noqa: E402,I001,E501

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import freeze_support
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

from behavior_fingerprint_lut_retrieval.plot_scenario2_c2_to_aend_g4_sparse_gmp_5b import (  # noqa: E402
    generate_figures,
)
from behavior_fingerprint_ranking_consistency.distance_matrix import (  # noqa: E402
    compute_cnmse_distance_matrix,
)
from behavior_fingerprint_ranking_consistency.scenario2_all_ilc_analysis import (  # noqa: E402
    load_frozen_scenario2_artifacts,
)
from behavior_model import frozen_neighborhood_memory_ridge_scan as frozen_model  # noqa: E402
from behavior_model import sparse_gmp as gmp  # noqa: E402
from behavior_model.evaluation import calculate_nmse  # noqa: E402
from data_manager import build_state_table, load_variable_by_id  # noqa: E402

STATE_COUNT = 425
STATE_IDS = tuple(range(STATE_COUNT))
ORDERS = gmp.MP_ORDERS
COMMON_B_LENGTH = 4915
FINGERPRINT_LENGTH = 4913
THRESHOLD_DB = -40.0
WORKER_TARGET = 0.80
ABC_A_START, ABC_A_END = 0, 12288
ABC_B_START, ABC_B_END = 12288, 17203
ABC_C_START, ABC_C_END = 17203, 24576

TASK_NAME = "behavior_fingerprint_lut_retrieval"
EXPERIMENT_NAME = "scenario_2_C2_to_Aend_G4_sparse_gmp_5B"
RESULT_ROOT = PROJECT_ROOT / "results" / TASK_NAME / EXPERIMENT_NAME
CONFIG_ROOT = RESULT_ROOT / "config"
CACHE_ROOT = RESULT_ROOT / "cache"
STATE_CACHE_ROOT = CACHE_ROOT / "state_chunks"
TABLE_ROOT = RESULT_ROOT / "tables"
MODEL_ROOT = RESULT_ROOT / "models"
FIGURE_ROOT = RESULT_ROOT / "figures"
VALIDATION_ROOT = RESULT_ROOT / "validation"
MODEL_LOG = PROJECT_ROOT / "work_logs" / TASK_NAME / "execution_log.txt"
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "codex_handoff" / "codex_handoff.txt"

G4_ROOT = PROJECT_ROOT / "results" / "behavior_model" / "scenario_2_failed14_frozen_mp_residual_sparse_gmp_5B"
G4_STRUCTURE_PATH = G4_ROOT / "models" / "selected_sparse_gmp_structure.json"
OLD_RETRIEVAL_ROOT = PROJECT_ROOT / "results" / TASK_NAME / "scenario_2_C2_to_Aend"
OLD_SCENARIO2_ROOT = PROJECT_ROOT / "results" / "behavior_fingerprint_ranking_consistency" / "scenario_2"
OLD_ALL_ILC_ROOT = PROJECT_ROOT / "results" / "behavior_fingerprint_ranking_consistency" / "scenario_2_all_ilc"
OLD_FORMAL_METRICS = OLD_ALL_ILC_ROOT / "all_ilc_model_metrics.csv"
FAILURE_SOURCE = OLD_RETRIEVAL_ROOT / "failed_retrieval_states.csv"
OLD_REAL_B_MATRIX = OLD_RETRIEVAL_ROOT / "real_B_distance_matrix.npz"
OLD_RETRIEVAL_RESULTS = OLD_RETRIEVAL_ROOT / "retrieval_results.csv"
OLD_REAL_B_DIAGNOSTICS = OLD_RETRIEVAL_ROOT / "retrieved_real_B_cnmse.csv"
OLD_LUT_FINGERPRINTS = OLD_RETRIEVAL_ROOT / "lut_fingerprints_Aend.npz"
OLD_QUERY_FINGERPRINTS = OLD_RETRIEVAL_ROOT / "query_fingerprints_C2.npz"
OLD_ALL_ILC_THETA = OLD_ALL_ILC_ROOT / "all_ilc_theta.npz"

PROTECTED_RESULT_DIRS = {
    "formal_frozen_retrieval": OLD_RETRIEVAL_ROOT,
    "all_ilc_frozen_metrics": OLD_ALL_ILC_ROOT,
    "g4_sparse_gmp_discovery": G4_ROOT,
    "unified_capacity_scan": PROJECT_ROOT / "results" / "behavior_model" / "scenario_2_unified_odd_order_mp_capacity_scan_5B",
    "odd_only_failed14_scan": PROJECT_ROOT / "results" / "behavior_model" / "scenario_2_failed14_odd_order_mp_order_dependent_memory_scan_5B",
    "p5_ridge_scan": PROJECT_ROOT / "results" / "behavior_model" / "scenario_2_failed14_P5_variable_memory_ridge_scan_5B",
    "p5_order2_scan": PROJECT_ROOT / "results" / "behavior_model" / "scenario_2_failed14_P5_with_order2_variable_memory_OLS_scan_5B",
}

_WORKER_COMMON_B: np.ndarray | None = None
_WORKER_TERMS: tuple[gmp.GMPTerm, ...] = ()
_WORKER_STRUCTURE_FILE: str | None = None


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, pd.DataFrame):
        return _json_safe(value.where(pd.notna(value), None).to_dict("records"))
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_frame(frame: pd.DataFrame, path: Path, *, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", na_rep="NaN", float_format="%.17g", compression=compression)


def _write_npz(path: Path, arrays: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


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
    digest = hashlib.sha256()
    files = sorted((file for file in raw_root.rglob("*") if file.is_file()), key=lambda item: str(item).lower())
    total = 0
    for file in files:
        size = int(file.stat().st_size)
        digest.update(str(file.relative_to(raw_root)).replace("\\", "/").encode("utf-8") + b"\0")
        digest.update(size.to_bytes(8, "little"))
        total += size
    return {"sha256": digest.hexdigest(), "file_count": len(files), "bytes": total}


def _snapshot() -> dict[str, Any]:
    return {"data/raw": _raw_manifest(), "result_dirs": {name: {"path": str(path), "exists": path.is_dir(), "sha256": _tree_digest(path)} for name, path in PROTECTED_RESULT_DIRS.items()}}


def _verify_protection(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    rb, ra = before["data/raw"], after["data/raw"]
    raw = {"sha256_unchanged": rb["sha256"] == ra["sha256"], "file_count_unchanged": rb["file_count"] == ra["file_count"], "bytes_unchanged": rb["bytes"] == ra["bytes"]}
    result_dirs = {name: {"sha256_unchanged": item["sha256"] == after["result_dirs"][name]["sha256"], "before": item["sha256"], "after": after["result_dirs"][name]["sha256"]} for name, item in before["result_dirs"].items()}
    return {"data/raw": raw, "result_dirs": result_dirs, "all_protected_unchanged": bool(all(raw.values()) and all(item["sha256_unchanged"] for item in result_dirs.values()))}


def _worker_count() -> tuple[int, int]:
    logical = int(os.cpu_count() or 1)
    return logical, max(1, int(np.floor(WORKER_TARGET * logical)))


def _load_g4_structure() -> tuple[dict[str, Any], tuple[gmp.GMPTerm, ...]]:
    if not G4_STRUCTURE_PATH.is_file():
        raise FileNotFoundError(f"missing frozen G4 structure: {G4_STRUCTURE_PATH}")
    payload = json.loads(G4_STRUCTURE_PATH.read_text(encoding="utf-8"))
    expected = ["GMP005", "GMP011", "GMP001", "GMP017"]
    ids = payload.get("selected_sparse_term_ids")
    if ids != expected or int(payload.get("selected_sparse_term_count", -1)) != 4:
        raise RuntimeError(f"G4 structure mismatch: {ids}")
    frozen = payload.get("frozen_mp", {})
    if frozen.get("orders") != list(ORDERS) or frozen.get("memory") != [3, 2, 2, 1, 1, 1] or float(frozen.get("lambda")) != gmp.RIDGE_LAMBDA or int(frozen.get("d_max")) != 2 or int(payload.get("K_total")) != 14:
        raise RuntimeError("G4 frozen MP/lambda/K/dmax definition mismatch")
    terms = tuple(gmp.term_by_id(term_id) for term_id in ids)
    return payload, terms


def _validate_state_order() -> pd.DataFrame:
    frame = pd.DataFrame(build_state_table()).sort_values("state_id").reset_index(drop=True)
    if frame.shape[0] != STATE_COUNT or not np.array_equal(frame["state_id"].to_numpy(int), np.arange(STATE_COUNT)):
        raise RuntimeError("state order is not 0...424")
    return frame


def _load_common_b_probe() -> np.ndarray:
    if not OLD_LUT_FINGERPRINTS.is_file():
        raise FileNotFoundError(OLD_LUT_FINGERPRINTS)
    with np.load(OLD_LUT_FINGERPRINTS, allow_pickle=False) as data:
        state_ids = np.asarray(data["state_ids"], dtype=np.int64)
        common = np.asarray(data["common_B_input"], dtype=np.complex128)
    if not np.array_equal(state_ids, np.arange(STATE_COUNT)) or common.shape != (COMMON_B_LENGTH,) or not np.all(np.isfinite(common)):
        raise RuntimeError("formal common B probe/state order is invalid")
    return common


def _frozen_g0_gate(common_b: np.ndarray) -> dict[str, Any]:
    with np.load(OLD_ALL_ILC_THETA, allow_pickle=False) as data:
        theta_a = np.asarray(data["theta_Y_A_actual"], dtype=np.complex128)
        theta_c = np.asarray(data["theta_Y_C_actual"], dtype=np.complex128)
        counts = np.asarray(data["ilc_column_counts"], dtype=np.int64)
    with np.load(OLD_LUT_FINGERPRINTS, allow_pickle=False) as data:
        old_a = np.asarray(data["Y_Aend_fingerprints"], dtype=np.complex128)
    with np.load(OLD_QUERY_FINGERPRINTS, allow_pickle=False) as data:
        old_c = np.asarray(data["Q_C2"], dtype=np.complex128)
    phi_b = gmp.build_frozen_mp_basis(common_b)
    current_a = (phi_b @ theta_a[np.arange(STATE_COUNT), counts - 1, :].T).T
    current_c = (phi_b @ theta_c[:, 1, :].T).T
    a_error = float(np.max(np.abs(current_a - old_a)))
    c_error = float(np.max(np.abs(current_c - old_c)))
    return {"pass": bool(a_error <= 1e-12 and c_error <= 1e-12), "Aend_max_abs_error": a_error, "C2_max_abs_error": c_error, "support": list(phi_b.shape), "state_count": STATE_COUNT}


def _real_b_gate() -> tuple[np.ndarray, dict[str, Any]]:
    if not OLD_REAL_B_MATRIX.is_file():
        raise FileNotFoundError(OLD_REAL_B_MATRIX)
    with np.load(OLD_REAL_B_MATRIX, allow_pickle=False) as data:
        state_ids = np.asarray(data["state_ids"], dtype=np.int64)
        matrix = np.asarray(data["D_B"], dtype=np.float64)
    if not np.array_equal(state_ids, np.arange(STATE_COUNT)) or matrix.shape != (STATE_COUNT, STATE_COUNT) or np.isnan(matrix).any() or np.isposinf(matrix).any():
        raise RuntimeError("formal Real-B matrix/state order invalid")
    # The formal Real-B helper includes the canonical OFF preprocessing and
    # gain adjustment.  Reuse its frozen D_RR result instead of comparing raw
    # unprocessed MAT slices, which are not the formal Real-B definition.
    frozen = load_frozen_scenario2_artifacts(OLD_SCENARIO2_ROOT)
    if not np.array_equal(frozen.state_ids, np.arange(STATE_COUNT)):
        raise RuntimeError("formal Real-B helper state order mismatch")
    finite = np.isfinite(matrix) & np.isfinite(frozen.D_RR)
    mismatch_inf = np.count_nonzero(np.isneginf(matrix) != np.isneginf(frozen.D_RR))
    max_error = float(np.max(np.abs(matrix[finite] - frozen.D_RR[finite]))) if np.any(finite) else 0.0
    max_error = max(max_error, 0.0 if mismatch_inf == 0 else float("inf"))
    return matrix, {"pass": bool(max_error <= 1e-12), "comparison": "formal scenario_2 D_RR helper", "reference_root": str(OLD_SCENARIO2_ROOT), "max_abs_error_dB": max_error, "diagonal_all_negative_infinity": bool(np.all(np.isneginf(np.diag(matrix))))}


def _fit_pair(x: np.ndarray, y: np.ndarray, b_input: np.ndarray, terms: Sequence[gmp.GMPTerm]) -> tuple[dict[str, Any], np.ndarray]:
    phi = gmp.build_combined_basis(x, terms)
    phi_b = gmp.build_combined_basis(b_input, terms)
    y = np.asarray(y, dtype=np.complex128).reshape(-1)
    if y.size == phi.shape[0] + gmp.COMMON_MAX_DELAY:
        y = y[gmp.COMMON_MAX_DELAY :]
    if y.size != phi.shape[0] or phi_b.shape[0] != FINGERPRINT_LENGTH:
        raise RuntimeError("G4 support mismatch")
    theta, diagnostics = gmp.fit_ridge(phi, y, gmp.RIDGE_LAMBDA)
    train_prediction = phi @ theta
    b_prediction = phi_b @ theta
    return {"train_NMSE_dB": float(calculate_nmse(y, train_prediction)), "B_NMSE_dB": float("nan"), "generalization_gap_dB": float("nan"), "rank": diagnostics.rank_phi, "rank_augmented": diagnostics.rank_augmented, "column_count": diagnostics.column_count, "rank_ratio": diagnostics.rank_phi / diagnostics.column_count, "sigma_max": diagnostics.sigma_max_phi, "sigma_min": diagnostics.sigma_min_phi, "condition_number": diagnostics.condition_number_phi, "condition_number_augmented": diagnostics.condition_number_augmented, "theta_l2_norm": diagnostics.theta_l2_norm, "theta_max_abs": float(np.max(np.abs(theta))), "residual_norm": diagnostics.residual_norm, "ridge_penalty": diagnostics.ridge_penalty, "MP_theta_l2_norm": float(np.linalg.norm(theta[:gmp.FROZEN_MP_K])), "GMP_theta_l2_norm": float(np.linalg.norm(theta[gmp.FROZEN_MP_K:])), "GMP_response_energy_ratio": float(np.linalg.norm(phi[:, gmp.FROZEN_MP_K:] @ theta[gmp.FROZEN_MP_K:]) ** 2 / np.linalg.norm(train_prediction) ** 2) if len(terms) else 0.0, "n_train": int(phi.shape[0]), "n_B": int(phi_b.shape[0]), "fingerprint": b_prediction.astype(np.complex128, copy=False)}, theta


def _worker_initializer(common_b: np.ndarray, terms: tuple[gmp.GMPTerm, ...], structure_file: str) -> None:
    global _WORKER_COMMON_B, _WORKER_TERMS, _WORKER_STRUCTURE_FILE
    _WORKER_COMMON_B = np.asarray(common_b, dtype=np.complex128)
    _WORKER_TERMS = tuple(terms)
    _WORKER_STRUCTURE_FILE = structure_file
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"


def _state_worker(state_id: int) -> dict[str, Any]:
    if _WORKER_COMMON_B is None or not _WORKER_TERMS or _WORKER_STRUCTURE_FILE is None or not Path(_WORKER_STRUCTURE_FILE).is_file():
        raise RuntimeError("G4 worker not initialized after structure freeze")
    prepared = frozen_model.prepare_state_any(int(state_id))
    a, theta_a = _fit_pair(prepared.aend_A_input, prepared.aend_A_output, _WORKER_COMMON_B, _WORKER_TERMS)
    c, theta_c = _fit_pair(prepared.c2_C_input, prepared.c2_C_output, _WORKER_COMMON_B, _WORKER_TERMS)
    a_b_phi = gmp.build_combined_basis(prepared.aend_B_input, _WORKER_TERMS)
    c_b_phi = gmp.build_combined_basis(prepared.c2_B_input, _WORKER_TERMS)
    a_b_y = prepared.aend_B_output[2:]
    c_b_y = prepared.c2_B_output[2:]
    a["B_NMSE_dB"] = float(calculate_nmse(a_b_y, a_b_phi @ theta_a))
    c["B_NMSE_dB"] = float(calculate_nmse(c_b_y, c_b_phi @ theta_c))
    a["generalization_gap_dB"] = a["B_NMSE_dB"] - a["train_NMSE_dB"]
    c["generalization_gap_dB"] = c["B_NMSE_dB"] - c["train_NMSE_dB"]
    return {"state_id": int(state_id), "ilc_A_end": int(prepared.ilc_A_end), "Aend": {key: value for key, value in a.items() if key != "fingerprint"}, "C2": {key: value for key, value in c.items() if key != "fingerprint"}, "Aend_fingerprint": a["fingerprint"], "C2_fingerprint": c["fingerprint"], "theta_Aend": theta_a, "theta_C2": theta_c, "support": {"A_train": int(a["n_train"]), "A_B": int(a["n_B"]), "C_train": int(c["n_train"]), "C_B": int(c["n_B"])}}


def _run_state_fits(common_b: np.ndarray, terms: tuple[gmp.GMPTerm, ...], workers: int, structure_file: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit G4 for all states with one state per ProcessPool task."""

    started = time.perf_counter()
    results: dict[int, dict[str, Any]] = {}
    STATE_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_initializer, initargs=(common_b, terms, str(structure_file))) as executor:
        futures = {executor.submit(_state_worker, state_id): state_id for state_id in STATE_IDS}
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            state_id = int(result["state_id"])
            results[state_id] = result
            _write_npz(STATE_CACHE_ROOT / f"state_{state_id:03d}.npz", {"Aend_fingerprint": result["Aend_fingerprint"], "C2_fingerprint": result["C2_fingerprint"], "theta_Aend": result["theta_Aend"], "theta_C2": result["theta_C2"]})
            if done % 50 == 0 or done == STATE_COUNT:
                print(f"G4 state fits: {done}/{STATE_COUNT}", flush=True)
    if set(results) != set(STATE_IDS):
        raise RuntimeError("G4 state fit result set is incomplete")
    rows: list[dict[str, Any]] = []
    a_fingerprints = np.empty((STATE_COUNT, FINGERPRINT_LENGTH), dtype=np.complex128)
    c_fingerprints = np.empty_like(a_fingerprints)
    theta_a: list[np.ndarray] = []
    theta_c: list[np.ndarray] = []
    for state_id in STATE_IDS:
        result = results[state_id]
        a, c = result["Aend"], result["C2"]
        rows.append({"state_id": state_id, "ilc_A_end": result["ilc_A_end"], "Y_Aend_train_NMSE_dB": a["train_NMSE_dB"], "Y_Aend_B_NMSE_dB": a["B_NMSE_dB"], "Y_Aend_generalization_gap_dB": a["generalization_gap_dB"], "Y_C2_train_NMSE_dB": c["train_NMSE_dB"], "Y_C2_B_NMSE_dB": c["B_NMSE_dB"], "Y_C2_generalization_gap_dB": c["generalization_gap_dB"], "Aend_rank": a["rank"], "C2_rank": c["rank"], "Aend_condition_number": a["condition_number"], "C2_condition_number": c["condition_number"], "Aend_condition_number_augmented": a["condition_number_augmented"], "C2_condition_number_augmented": c["condition_number_augmented"], "Aend_theta_l2_norm": a["theta_l2_norm"], "C2_theta_l2_norm": c["theta_l2_norm"], "Aend_GMP_theta_l2_norm": a["GMP_theta_l2_norm"], "C2_GMP_theta_l2_norm": c["GMP_theta_l2_norm"], "Aend_GMP_response_energy_ratio": a["GMP_response_energy_ratio"], "C2_GMP_response_energy_ratio": c["GMP_response_energy_ratio"], "Aend_n_train": a["n_train"], "Aend_n_B": a["n_B"], "C2_n_train": c["n_train"], "C2_n_B": c["n_B"]})
        a_fingerprints[state_id, :] = result["Aend_fingerprint"]
        c_fingerprints[state_id, :] = result["C2_fingerprint"]
        theta_a.append(result["theta_Aend"])
        theta_c.append(result["theta_C2"])
    frame = pd.DataFrame(rows).sort_values("state_id").reset_index(drop=True)
    return frame, a_fingerprints, c_fingerprints, np.stack(theta_a), np.stack(theta_c), {"elapsed_seconds": time.perf_counter() - started, "state_count": STATE_COUNT, "model_fit_count": STATE_COUNT * 2, "worker_count": workers}


def _load_old_retrieval_reference() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, dict[str, Any]]:
    """Load Frozen retrieval, Real-B matrix and historical state metrics read-only."""

    old_results = pd.read_csv(OLD_RETRIEVAL_RESULTS)
    old_diag = pd.read_csv(OLD_REAL_B_DIAGNOSTICS)
    required_result = {"State_n_R", "State_n_Q", "exact_hit"}
    required_diag = {"State_n_R", "State_n_Q", "retrieved_real_B_CNMSE_dB", "dpd_shareable", "failure"}
    if not required_result.issubset(old_results.columns) or not required_diag.issubset(old_diag.columns) or old_results.shape[0] != STATE_COUNT or old_diag.shape[0] != STATE_COUNT:
        raise RuntimeError("formal Frozen retrieval tables are incomplete")
    with np.load(OLD_REAL_B_MATRIX, allow_pickle=False) as data:
        old_distance = np.asarray(data["D_B"], dtype=np.float64)
        old_ids = np.asarray(data["state_ids"], dtype=np.int64)
    if not np.array_equal(old_ids, np.arange(STATE_COUNT)) or old_distance.shape != (STATE_COUNT, STATE_COUNT):
        raise RuntimeError("formal Real-B matrix state order mismatch")
    old_metrics = pd.read_csv(OLD_FORMAL_METRICS)
    return old_results, old_diag, old_distance, {"old_metrics": old_metrics, "old_result_path": str(OLD_RETRIEVAL_RESULTS), "old_real_b_matrix": str(OLD_REAL_B_MATRIX)}


def _top1_retrieval(distance: np.ndarray) -> tuple[pd.DataFrame, np.ndarray]:
    if distance.shape != (STATE_COUNT, STATE_COUNT) or np.isnan(distance).any() or np.isposinf(distance).any():
        raise RuntimeError("G4 behavior distance matrix has invalid values")
    rows: list[dict[str, Any]] = []
    ranking = np.empty_like(distance, dtype=np.float64)
    order_matrix = np.empty((STATE_COUNT, STATE_COUNT), dtype=np.int64)
    for state_id in STATE_IDS:
        order = np.lexsort((np.arange(STATE_COUNT, dtype=np.int64), distance[state_id]))
        order_matrix[state_id] = order
        ranks = np.empty(STATE_COUNT, dtype=np.int64)
        ranks[order] = np.arange(1, STATE_COUNT + 1, dtype=np.int64)
        ranking[state_id] = ranks
        q = int(order[0])
        d1 = float(distance[state_id, q])
        d2 = float(distance[state_id, order[1]])
        minimum_tie_count = int(np.count_nonzero(distance[state_id] == d1))
        rows.append({"state_id_R": state_id, "retrieved_state_id_Q": q, "retrieved_fingerprint_CNMSE_dB": d1, "second_best_state_id": int(order[1]), "second_best_fingerprint_CNMSE_dB": d2, "retrieval_margin_dB": float(d2 - d1), "true_state_rank": int(ranks[state_id]), "minimum_tie_count": minimum_tie_count, "exact_hit": bool(q == state_id)})
    return pd.DataFrame(rows), order_matrix


def _real_b_results(g4_retrieval: pd.DataFrame, real_b_distance: np.ndarray, state_frame: pd.DataFrame, g4_metrics: pd.DataFrame, old_results: pd.DataFrame, old_diag: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    lookup = state_frame.set_index("state_id")
    metric_lookup = g4_metrics.set_index("state_id")
    old_r = old_results.set_index("State_n_R")
    old_d = old_diag.set_index("State_n_R")
    rows: list[dict[str, Any]] = []
    for record in g4_retrieval.itertuples(index=False):
        r, q = int(record.state_id_R), int(record.retrieved_state_id_Q)
        real_value = float(real_b_distance[r, q])
        old_q = int(old_r.loc[r, "State_n_Q"])
        old_value = float(old_d.loc[r, "retrieved_real_B_CNMSE_dB"])
        g4_shareable = bool(real_value < THRESHOLD_DB)
        old_shareable = bool(old_value < THRESHOLD_DB)
        real_config, q_config = lookup.loc[r], lookup.loc[q]
        metric = metric_lookup.loc[r]
        row = {"state_id": r, "funMng": real_config["funMng"], "funAng_deg": real_config["funAng"], "secMng": real_config["secMng"], "secAng_deg": real_config["secAng"], "retrieved_state_id": q, "retrieved_real_B_CNMSE_dB": real_value, "retrieved_fingerprint_CNMSE_dB": float(record.retrieved_fingerprint_CNMSE_dB), "second_best_state_id": int(record.second_best_state_id), "second_best_fingerprint_CNMSE_dB": float(record.second_best_fingerprint_CNMSE_dB), "retrieval_margin_dB": float(record.retrieval_margin_dB), "true_state_rank": int(record.true_state_rank), "minimum_tie_count": int(record.minimum_tie_count), "exact_hit": bool(record.exact_hit), "shareable_under_minus40dB": g4_shareable, "failure": not g4_shareable, "retrieved_state_signed_diff": q - r, "retrieved_state_abs_diff": abs(q - r), "retrieved_funMng": q_config["funMng"], "retrieved_funAng_deg": q_config["funAng"], "retrieved_secMng": q_config["secMng"], "retrieved_secAng_deg": q_config["secAng"], "Y_Aend_train_NMSE_dB": metric["Y_Aend_train_NMSE_dB"], "Y_Aend_B_NMSE_dB": metric["Y_Aend_B_NMSE_dB"], "Y_C2_train_NMSE_dB": metric["Y_C2_train_NMSE_dB"], "Y_C2_B_NMSE_dB": metric["Y_C2_B_NMSE_dB"], "Y_Aend_generalization_gap_dB": metric["Y_Aend_generalization_gap_dB"], "Y_C2_generalization_gap_dB": metric["Y_C2_generalization_gap_dB"], "Frozen_retrieved_state_id": old_q, "Frozen_exact": bool(old_r.loc[r, "exact_hit"]), "Frozen_real_B_CNMSE_dB": old_value, "Frozen_shareable": old_shareable, "delta_real_B_CNMSE_dB": real_value - old_value, "rescued": bool((not old_shareable) and g4_shareable), "new_failure": bool(old_shareable and (not g4_shareable))}
        rows.append(row)
    summary = pd.DataFrame(rows).sort_values("state_id").reset_index(drop=True)
    failure14_ids = (187, 189, 195, 196, 199, 206, 323, 327, 330, 335, 340, 344, 346, 354)
    failure14 = summary.loc[summary["state_id"].isin(failure14_ids)].copy()
    return summary, failure14, lookup


def _retrieval_statistics(summary: pd.DataFrame, failure14: pd.DataFrame) -> pd.DataFrame:
    exact = summary["exact_hit"].astype(bool)
    shareable = summary["shareable_under_minus40dB"].astype(bool)
    rows = [
        ("state_count", len(summary)),
        ("exact_count", int(exact.sum())),
        ("exact_rate", float(exact.mean())),
        ("nonexact_count", int((~exact).sum())),
        ("shareable_count", int(shareable.sum())),
        ("shareable_rate", float(shareable.mean())),
        ("nonexact_shareable_count", int((shareable & ~exact).sum())),
        ("nonexact_shareable_rate", float((shareable & ~exact).sum() / max((~exact).sum(), 1))),
        ("failure_count", int((~shareable).sum())),
        ("failure_state_ids", ",".join(str(value) for value in summary.loc[~shareable, "state_id"].tolist())),
        ("rescued_original_failure14_count", int(failure14["rescued"].sum())),
        ("new_failure_success411_count", int(summary.loc[~summary["state_id"].isin(failure14["state_id"]), "new_failure"].sum())),
        ("g4_top1_margin_median_dB", float(summary["retrieval_margin_dB"].median())),
        ("g4_top1_margin_q05_dB", float(summary["retrieval_margin_dB"].quantile(0.05))),
        ("g4_top1_margin_min_dB", float(summary["retrieval_margin_dB"].min())),
        ("delta_real_B_CNMSE_median_dB", float(summary["delta_real_B_CNMSE_dB"].replace(-np.inf, np.nan).median())),
        ("delta_real_B_CNMSE_q95_dB", float(summary["delta_real_B_CNMSE_dB"].replace(-np.inf, np.nan).quantile(0.95))),
    ]
    return pd.DataFrame(rows, columns=["metric", "value"])


def _load_raw_scalar_metrics() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for position, state_id in enumerate(STATE_IDS, start=1):
        values: dict[str, float | int] = {"state_id": state_id}
        for variable in ("nmse_withoutdpd", "acpr_low_withoutdpd", "acpr_upper_withoutdpd"):
            value = np.asarray(load_variable_by_id(state_id, variable), dtype=float).reshape(-1)
            if value.size != 1 or not np.isfinite(value[0]):
                raise RuntimeError(f"state {state_id} {variable} is not a finite scalar")
            values[variable] = float(value[0])
        values["nmse_withoutdpd_dB"] = float(values["nmse_withoutdpd"])
        values["acpr_withoutdpd_avg_dBc"] = (float(values["acpr_low_withoutdpd"]) + float(values["acpr_upper_withoutdpd"])) / 2.0
        rows.append(values)
        if position % 50 == 0 or position == STATE_COUNT:
            print(f"raw PA scalar metrics: {position}/{STATE_COUNT}", flush=True)
    return pd.DataFrame(rows).sort_values("state_id").reset_index(drop=True)


def _distance_frame(distance: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(distance, columns=[f"Q_{state_id:03d}" for state_id in STATE_IDS])
    frame.insert(0, "state_id_R", np.asarray(STATE_IDS, dtype=np.int64))
    return frame


def _frozen_vs_g4(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "state_id", "Frozen_retrieved_state_id", "retrieved_state_id", "Frozen_exact", "exact_hit",
        "Frozen_real_B_CNMSE_dB", "retrieved_real_B_CNMSE_dB", "delta_real_B_CNMSE_dB",
        "Frozen_shareable", "shareable_under_minus40dB", "rescued", "new_failure",
        "retrieved_state_signed_diff", "retrieved_state_abs_diff", "retrieval_margin_dB",
    ]
    return summary.loc[:, columns].rename(columns={"retrieved_state_id": "G4_retrieved_state_id", "exact_hit": "G4_exact", "retrieved_real_B_CNMSE_dB": "G4_real_B_CNMSE_dB", "shareable_under_minus40dB": "G4_shareable"}).sort_values("state_id").reset_index(drop=True)


def _retrieval_summary_frame(summary: pd.DataFrame) -> pd.DataFrame:
    exact = summary["exact_hit"].astype(bool)
    shareable = summary["shareable_under_minus40dB"].astype(bool)
    nonexact = ~exact
    rows = [
        {"metric": "state_count", "value": int(summary.shape[0])},
        {"metric": "exact_count", "value": int(exact.sum())},
        {"metric": "exact_rate", "value": float(exact.mean())},
        {"metric": "nonexact_count", "value": int(nonexact.sum())},
        {"metric": "shareable_count", "value": int(shareable.sum())},
        {"metric": "shareable_rate", "value": float(shareable.mean())},
        {"metric": "nonexact_shareable_count", "value": int((shareable & nonexact).sum())},
        {"metric": "nonexact_shareable_rate", "value": float((shareable & nonexact).sum() / max(nonexact.sum(), 1))},
        {"metric": "failure_count", "value": int((~shareable).sum())},
        {"metric": "failure_state_ids", "value": ",".join(str(int(value)) for value in summary.loc[~shareable, "state_id"])},
        {"metric": "top1_fingerprint_margin_median_dB", "value": float(summary["retrieval_margin_dB"].median())},
        {"metric": "top1_fingerprint_margin_min_dB", "value": float(summary["retrieval_margin_dB"].min())},
        {"metric": "rescued_original_failure14_count", "value": int(summary.loc[summary["state_id"].isin((187,189,195,196,199,206,323,327,330,335,340,344,346,354)), "rescued"].sum())},
        {"metric": "new_failure_success411_count", "value": int(summary.loc[~summary["state_id"].isin((187,189,195,196,199,206,323,327,330,335,340,344,346,354)), "new_failure"].sum())},
    ]
    return pd.DataFrame(rows)


def _distance_margin_summary(path: Path, field: str) -> dict[str, float]:
    if not path.is_file():
        return {"median": float("nan"), "minimum": float("nan")}
    with np.load(path, allow_pickle=False) as data:
        distance = np.asarray(data[field], dtype=np.float64)
    if distance.shape != (STATE_COUNT, STATE_COUNT):
        raise RuntimeError(f"historical behavior distance shape mismatch: {distance.shape}")
    margins = []
    for state_id in STATE_IDS:
        order = np.lexsort((np.arange(STATE_COUNT, dtype=np.int64), distance[state_id]))
        margins.append(float(distance[state_id, order[1]] - distance[state_id, order[0]]))
    return {"median": float(np.median(margins)), "minimum": float(np.min(margins))}


def _config_rows(g4_payload: Mapping[str, Any], logical_cpu_count: int, workers: int, stats: pd.DataFrame) -> list[dict[str, Any]]:
    values = {str(row["metric"]): row["value"] for _, row in stats.iterrows()}
    return [
        {"field": "experiment", "value": EXPERIMENT_NAME},
        {"field": "state_count", "value": STATE_COUNT},
        {"field": "G4_structure_file", "value": str(G4_STRUCTURE_PATH)},
        {"field": "G4_selected_sparse_term_ids", "value": g4_payload["selected_sparse_term_ids"]},
        {"field": "G4_K_total", "value": g4_payload["K_total"]},
        {"field": "lambda", "value": gmp.RIDGE_LAMBDA},
        {"field": "common_B_length", "value": COMMON_B_LENGTH},
        {"field": "fingerprint_length", "value": FINGERPRINT_LENGTH},
        {"field": "self_match_included", "value": True},
        {"field": "exact_count", "value": values["exact_count"]},
        {"field": "shareable_count", "value": values["shareable_count"]},
        {"field": "failure_count", "value": values["failure_count"]},
        {"field": "threshold", "value": "retrieved Real-B CNMSE < -40 dB"},
        {"field": "logical_cpu_count", "value": logical_cpu_count},
        {"field": "worker_count_effective", "value": workers},
        {"field": "parallelization_axis", "value": "state"},
        {"field": "retrieval_used", "value": True},
        {"field": "real_B_used_after_top1", "value": True},
    ]


def _write_retrieval_outputs(summary: pd.DataFrame, g4_metrics: pd.DataFrame, a_fp: np.ndarray, c_fp: np.ndarray, distance: np.ndarray, real_b_distance: np.ndarray, common_b: np.ndarray, theta_a: np.ndarray, theta_c: np.ndarray, state_frame: pd.DataFrame, g4_payload: Mapping[str, Any], old_results: pd.DataFrame, old_diag: pd.DataFrame, old_frozen_gate: Mapping[str, Any], real_b_gate: Mapping[str, Any], state_fit_info: Mapping[str, Any], gates: Mapping[str, Any], before: Mapping[str, Any], after: Mapping[str, Any], protection: Mapping[str, Any], logical_cpu_count: int, workers: int) -> None:
    summary = summary.sort_values("state_id").reset_index(drop=True)
    failure14_ids = (187, 189, 195, 196, 199, 206, 323, 327, 330, 335, 340, 344, 346, 354)
    failure14 = summary.loc[summary["state_id"].isin(failure14_ids)].copy()
    raw_scalars = _load_raw_scalar_metrics()
    merged = summary.merge(state_frame.loc[:, ["state_id", "Vm", "Pin"]], on="state_id", how="left", validate="one_to_one").merge(raw_scalars.loc[:, ["state_id", "nmse_withoutdpd_dB", "acpr_withoutdpd_avg_dBc"]], on="state_id", how="left", validate="one_to_one")
    state_summary = merged.loc[:, ["state_id", "funMng", "funAng_deg", "secMng", "secAng_deg", "nmse_withoutdpd_dB", "acpr_withoutdpd_avg_dBc", "Y_Aend_train_NMSE_dB", "Y_Aend_B_NMSE_dB", "Y_C2_train_NMSE_dB", "Y_C2_B_NMSE_dB", "retrieved_state_id", "retrieved_real_B_CNMSE_dB", "retrieved_state_signed_diff", "retrieved_state_abs_diff", "exact_hit", "shareable_under_minus40dB", "retrieved_funMng", "retrieved_funAng_deg", "retrieved_secMng", "retrieved_secAng_deg", "retrieved_fingerprint_CNMSE_dB", "second_best_state_id", "second_best_fingerprint_CNMSE_dB", "retrieval_margin_dB", "true_state_rank", "minimum_tie_count", "failure", "rescued", "new_failure", "Frozen_retrieved_state_id", "Frozen_exact", "Frozen_real_B_CNMSE_dB", "Frozen_shareable"]].copy()
    state_summary["shareable"] = state_summary["shareable_under_minus40dB"]
    frozen_vs = _frozen_vs_g4(summary)
    stats = _retrieval_summary_frame(summary)
    excel_state_summary = state_summary.copy()
    excel_frozen_vs = frozen_vs.copy()
    for frame, columns in ((excel_state_summary, ["retrieved_real_B_CNMSE_dB", "Frozen_real_B_CNMSE_dB"]), (excel_frozen_vs, ["G4_real_B_CNMSE_dB", "Frozen_real_B_CNMSE_dB"])):
        for column in columns:
            frame[column] = frame[column].astype(object)
            values = frame[column].to_numpy(object)
            frame.loc[np.isneginf(pd.to_numeric(values, errors="coerce")), column] = "-Inf"
    frozen_exact_count = int(summary["Frozen_exact"].astype(bool).sum())
    g4_exact_count = int(summary["exact_hit"].astype(bool).sum())
    frozen_shareable_count = int(summary["Frozen_shareable"].astype(bool).sum())
    g4_shareable_count = int(summary["shareable_under_minus40dB"].astype(bool).sum())
    changed_top1_count = int((summary["Frozen_retrieved_state_id"] != summary["retrieved_state_id"]).sum())
    failure14_ids = (187, 189, 195, 196, 199, 206, 323, 327, 330, 335, 340, 344, 346, 354)
    failure14_top1_changed_count = int(summary.loc[summary["state_id"].isin(failure14_ids), "Frozen_retrieved_state_id"].ne(summary.loc[summary["state_id"].isin(failure14_ids), "retrieved_state_id"]).sum())
    frozen_margin = _distance_margin_summary(OLD_RETRIEVAL_ROOT / "retrieval_distance_matrix.npz", "D_C2_Aend")
    g4_margin = {"median": float(summary["retrieval_margin_dB"].median()), "minimum": float(summary["retrieval_margin_dB"].min())}
    _write_frame(g4_metrics, TABLE_ROOT / "g4_state_model_metrics.csv")
    _write_frame(state_summary, TABLE_ROOT / "retrieval_state_summary.csv")
    _write_frame(_distance_frame(distance), TABLE_ROOT / "retrieval_distance_matrix.csv.gz", compression="gzip")
    _write_frame(stats, TABLE_ROOT / "retrieval_statistics.csv")
    _write_frame(failure14, TABLE_ROOT / "original_failure14.csv")
    _write_frame(frozen_vs, TABLE_ROOT / "frozen_vs_G4.csv")
    _write_frame(summary.loc[summary["failure"].astype(bool)], TABLE_ROOT / "failure_states.csv")
    _write_npz(CACHE_ROOT / "Aend_fingerprints.npz", {"state_ids": np.asarray(STATE_IDS), "common_B_input": common_b, "Aend_LUT_fingerprints": a_fp})
    _write_npz(CACHE_ROOT / "C2_Query_fingerprints.npz", {"state_ids": np.asarray(STATE_IDS), "C2_Query_fingerprints": c_fp})
    _write_npz(CACHE_ROOT / "fingerprint_distance_matrix.npz", {"state_ids": np.asarray(STATE_IDS), "D_C2_Aend": distance})
    _write_npz(CACHE_ROOT / "real_B_CNMSE_matrix.npz", {"state_ids": np.asarray(STATE_IDS), "D_B": real_b_distance})
    _write_npz(MODEL_ROOT / "Aend_coefficients.npz", {"state_ids": np.asarray(STATE_IDS), "theta_Aend": theta_a})
    _write_npz(MODEL_ROOT / "C2_coefficients.npz", {"state_ids": np.asarray(STATE_IDS), "theta_C2": theta_c})
    _write_frame(pd.DataFrame({"state_id": STATE_IDS, "funMng": state_frame["funMng"], "funAng_deg": state_frame["funAng"], "secMng": state_frame["secMng"], "secAng_deg": state_frame["secAng"], "Vm": state_frame["Vm"], "Pin": state_frame["Pin"]}), CONFIG_ROOT / "state_order.csv")
    figure_details = generate_figures(state_summary, FIGURE_ROOT)
    validation = {"experiment": EXPERIMENT_NAME, "state_count": STATE_COUNT, "G4_model_definition_pass": True, "G4_structure_file": str(G4_STRUCTURE_PATH), "G4_selected_sparse_term_ids": list(g4_payload["selected_sparse_term_ids"]), "G4_K_total": int(g4_payload["K_total"]), "orders": list(ORDERS), "memory_MP": [gmp.MP_MEMORY[order] for order in ORDERS], "lambda": gmp.RIDGE_LAMBDA, "dmax": 2, "ABC": {"A": [ABC_A_START, ABC_A_END], "B": [ABC_B_START, ABC_B_END], "C": [ABC_C_START, ABC_C_END]}, "A_valid_length": ABC_A_END - 2, "B_valid_length": ABC_B_END - ABC_B_START - 2, "C_valid_length": ABC_C_END - ABC_B_END - 2, "Aend_fit_count": STATE_COUNT, "C2_fit_count": STATE_COUNT, "model_fit_count": STATE_COUNT * 2, "Aend_fingerprint_shape": list(a_fp.shape), "C2_fingerprint_shape": list(c_fp.shape), "behavior_distance_shape": list(distance.shape), "real_B_distance_shape": list(real_b_distance.shape), "self_match_included": True, "stable_tie_break_state_id_ascending": True, "real_B_helper_reused": True, "real_B_helper_gate": real_b_gate, "frozen_g0_gate": old_frozen_gate, "gates": gates, "exact_count": g4_exact_count, "exact_rate": float(summary["exact_hit"].mean()), "frozen_exact_count": frozen_exact_count, "exact_delta_count_G4_minus_Frozen": g4_exact_count - frozen_exact_count, "shareable_count": g4_shareable_count, "shareable_rate": float(summary["shareable_under_minus40dB"].mean()), "frozen_shareable_count": frozen_shareable_count, "shareable_delta_count_G4_minus_Frozen": g4_shareable_count - frozen_shareable_count, "failure_count": int(summary["failure"].sum()), "failure_state_ids": [int(value) for value in summary.loc[summary["failure"], "state_id"]], "original_failure14_rescued_count": int(failure14["rescued"].sum()), "success411_new_failure_count": int(summary.loc[~summary["state_id"].isin(failure14_ids), "new_failure"].sum()), "top1_changed_count": changed_top1_count, "failure14_top1_changed_count": failure14_top1_changed_count, "G4_margin_median_dB": g4_margin["median"], "G4_margin_min_dB": g4_margin["minimum"], "Frozen_margin_median_dB": frozen_margin["median"], "Frozen_margin_min_dB": frozen_margin["minimum"], "state_level_parallel": True, "logical_cpu_count": logical_cpu_count, "worker_count_requested": workers, "worker_count_effective": workers, "cpu_target_fraction": WORKER_TARGET, "parallelization_axis": "state", "blas_threads_per_worker": 1, "nested_parallelism": False, "raw_data_modified": False, "protected_results_modified": False, "figures": figure_details, "runtime": state_fit_info, "protection_verification": protection}
    _write_json(CONFIG_ROOT / "G4_model_definition.json", g4_payload)
    _write_json(CONFIG_ROOT / "experiment_config.json", {**validation, "selection_or_training": "G4 loaded from frozen structure; no term/lambda selection", "real_B_threshold_rule": "retrieved Real-B CNMSE < -40 dB"})
    _write_json(RESULT_ROOT / "validation.json", validation)
    _write_json(VALIDATION_ROOT / "validation.json", validation)
    _write_json(RESULT_ROOT / "search_progress.json", {"completed": True, "state_count": STATE_COUNT, "model_fit_count": STATE_COUNT * 2, "fingerprint_distance_shape": list(distance.shape), "selected_g4": list(g4_payload["selected_sparse_term_ids"])})
    med = {key: float(summary[key].median()) for key in ("Y_Aend_train_NMSE_dB", "Y_Aend_B_NMSE_dB", "Y_C2_train_NMSE_dB", "Y_C2_B_NMSE_dB", "retrieved_real_B_CNMSE_dB") if np.isfinite(summary[key]).any()}
    text = "\n".join([EXPERIMENT_NAME, "", "G4 frozen model was loaded from the prior Sparse-GMP structure and used for C2→Aend retrieval.", f"G4 model fit count: Aend={STATE_COUNT}, C2={STATE_COUNT}, total={STATE_COUNT*2}.", f"Behavior fingerprint matrix: {distance.shape}; Real-B matrix: {real_b_distance.shape}.", f"G4 Exact={g4_exact_count}/{STATE_COUNT} ({g4_exact_count/STATE_COUNT:.6f}); Frozen Exact={frozen_exact_count}/{STATE_COUNT}; delta={g4_exact_count-frozen_exact_count}.", f"G4 Shareable={g4_shareable_count}/{STATE_COUNT} ({g4_shareable_count/STATE_COUNT:.6f}); Frozen Shareable={frozen_shareable_count}/{STATE_COUNT}; delta={g4_shareable_count-frozen_shareable_count}; G4 Failure={int(summary['failure'].sum())}.", f"Non-exact G4 shareable={int((summary['shareable_under_minus40dB'] & ~summary['exact_hit']).sum())}/{int((~summary['exact_hit']).sum())}.", f"Original failure14 rescued={int(failure14['rescued'].sum())}; success411 new failure={int(summary.loc[~summary['state_id'].isin(failure14_ids), 'new_failure'].sum())}; Top-1 changed={changed_top1_count}/{STATE_COUNT}; failure14 Top-1 changed={failure14_top1_changed_count}/{len(failure14_ids)}.", f"Top-1 margin median/min: G4={g4_margin['median']:.6f}/{g4_margin['minimum']:.6f} dB; Frozen={frozen_margin['median']:.6f}/{frozen_margin['minimum']:.6f} dB.", f"G4 retrieval metric medians: fingerprint={float(summary['retrieved_fingerprint_CNMSE_dB'].median()):.6f} dB; Real-B={float(summary['retrieved_real_B_CNMSE_dB'].replace(-np.inf, np.nan).median()):.6f} dB.", f"G4 model metric medians: Aend train={med['Y_Aend_train_NMSE_dB']:.6f}; Aend→B={med['Y_Aend_B_NMSE_dB']:.6f}; C2 train={med['Y_C2_train_NMSE_dB']:.6f}; C2→B={med['Y_C2_B_NMSE_dB']:.6f} dB.", "B probe is the same formal common B input for all 425 states; Real-B uses the prior formal yout_withoutdpd_ori B matrix/helper.", "No retrieval result or old Frozen file was overwritten; this directory contains a new G4 comparison only.", f"Gates all pass={all(bool(value['pass']) if isinstance(value, dict) and 'pass' in value else bool(value) for value in gates.values())}; raw/protected unchanged={protection['all_protected_unchanged']}.", "G4 behavior improvement is sufficient to justify a later retrieval comparison, but this task stops before any follow-up retrieval variant.", f"Results: {RESULT_ROOT}"])
    (RESULT_ROOT / "final_result_summary.txt").write_text(text + "\n", encoding="utf-8")
    _write_json(RESULT_ROOT / "excel_source.json", {"sheets": {"experiment_config": {"columns": ["field", "value"], "rows": [[row["field"], json.dumps(row["value"], ensure_ascii=False) if isinstance(row["value"], (list, dict)) else row["value"]] for row in _config_rows(g4_payload, logical_cpu_count, workers, stats)]}, "G4_model_definition": {"columns": list(g4_payload.keys()), "rows": [g4_payload]}, "retrieval_state_summary": {"columns": excel_state_summary.columns.tolist(), "rows": excel_state_summary.where(pd.notna(excel_state_summary), None).to_dict("records")}, "retrieval_distance_matrix": {"columns": list(_distance_frame(distance).columns), "rows": _distance_frame(distance).where(pd.notna(_distance_frame(distance)), None).to_dict("records")}, "retrieval_statistics": {"columns": stats.columns.tolist(), "rows": stats.where(pd.notna(stats), None).to_dict("records")}, "failure_states": {"columns": summary.loc[summary["failure"].astype(bool)].columns.tolist(), "rows": summary.loc[summary["failure"].astype(bool)].where(pd.notna(summary.loc[summary["failure"].astype(bool)]), None).to_dict("records")}, "frozen_vs_G4": {"columns": excel_frozen_vs.columns.tolist(), "rows": excel_frozen_vs.where(pd.notna(excel_frozen_vs), None).to_dict("records")}}, "metadata": {"experiment": EXPERIMENT_NAME, "state_order": "funMng→funAng→secMng→secAng→Vm→Pin", "real_B_used_after_top1": True}})


def _self_and_tie_gates() -> dict[str, Any]:
    synthetic = np.zeros((STATE_COUNT, STATE_COUNT), dtype=np.float64)
    selected, _ = _top1_retrieval(synthetic)
    tie_pass = bool(np.all(selected["retrieved_state_id_Q"].to_numpy(dtype=np.int64) == 0))
    return {"self_retrieval_allowed": True, "stable_tie_break_state_id_ascending": tie_pass, "pass": tie_pass}


def _serial_parallel_consistency(common_b: np.ndarray, terms: tuple[gmp.GMPTerm, ...], structure_file: Path, workers: int, parallel_metrics: pd.DataFrame, parallel_a: np.ndarray, parallel_c: np.ndarray) -> dict[str, Any]:
    global _WORKER_COMMON_B, _WORKER_TERMS, _WORKER_STRUCTURE_FILE
    _WORKER_COMMON_B, _WORKER_TERMS, _WORKER_STRUCTURE_FILE = common_b, terms, str(structure_file)
    errors: list[float] = []
    fp_errors: list[float] = []
    for state_id in (187, 195, 340, 354):
        serial = _state_worker(state_id)
        row = parallel_metrics.loc[parallel_metrics["state_id"] == state_id].iloc[0]
        errors.extend([abs(float(serial["Aend"]["train_NMSE_dB"]) - float(row["Y_Aend_train_NMSE_dB"])), abs(float(serial["Aend"]["B_NMSE_dB"]) - float(row["Y_Aend_B_NMSE_dB"])), abs(float(serial["C2"]["train_NMSE_dB"]) - float(row["Y_C2_train_NMSE_dB"])), abs(float(serial["C2"]["B_NMSE_dB"]) - float(row["Y_C2_B_NMSE_dB"]))])
        fp_errors.extend([float(np.max(np.abs(serial["Aend_fingerprint"] - parallel_a[state_id]))), float(np.max(np.abs(serial["C2_fingerprint"] - parallel_c[state_id])))])
    max_metric, max_fp = max(errors, default=0.0), max(fp_errors, default=0.0)
    return {"pass": bool(max_metric <= 1e-10 and max_fp <= 1e-10), "state_ids": [187, 195, 340, 354], "metric_max_abs_error_dB": max_metric, "fingerprint_max_abs_error": max_fp, "parallelization_axis": "state"}


def main() -> None:
    freeze_support()
    print(f"Starting {EXPERIMENT_NAME}", flush=True)
    before = _snapshot()
    g4_payload, terms = _load_g4_structure()
    state_frame = _validate_state_order()
    common_b = _load_common_b_probe()
    old_results, old_diag, real_b_distance, old_reference_info = _load_old_retrieval_reference()
    old_frozen_gate = _frozen_g0_gate(common_b)
    real_b_distance, real_b_gate = _real_b_gate()
    logical_cpu_count, workers = _worker_count()
    for path in (CONFIG_ROOT, CACHE_ROOT, STATE_CACHE_ROOT, TABLE_ROOT, MODEL_ROOT, FIGURE_ROOT, VALIDATION_ROOT):
        path.mkdir(parents=True, exist_ok=True)
    print(f"logical_cpu_count={logical_cpu_count}; worker_count={workers}; target=0.80", flush=True)
    if not old_frozen_gate["pass"] or not real_b_gate["pass"]:
        raise RuntimeError(f"pre-retrieval frozen/Real-B gate failed: {old_frozen_gate}/{real_b_gate}")
    parallel_metrics, a_fp, c_fp, theta_a, theta_c, state_fit_info = _run_state_fits(common_b, terms, workers, G4_STRUCTURE_PATH)
    if parallel_metrics.shape[0] != STATE_COUNT or a_fp.shape != (STATE_COUNT, FINGERPRINT_LENGTH) or c_fp.shape != a_fp.shape or not np.all(np.isfinite(a_fp)) or not np.all(np.isfinite(c_fp)):
        raise RuntimeError("G4 state fit/fingerprint output is incomplete")
    b_support_pass = bool((parallel_metrics[["Aend_n_B", "C2_n_B"]] == 4913).all().all() and (parallel_metrics[["Aend_n_train", "C2_n_train"]] == [12286, 7371]).all(axis=1).all())
    if not b_support_pass:
        raise RuntimeError("G4 support gate failed")
    serial_parallel = _serial_parallel_consistency(common_b, terms, G4_STRUCTURE_PATH, workers, parallel_metrics, a_fp, c_fp)
    if not serial_parallel["pass"]:
        raise RuntimeError(f"G4 serial/parallel gate failed: {serial_parallel}")
    distance = compute_cnmse_distance_matrix(c_fp, a_fp)
    retrieval, _ = _top1_retrieval(distance)
    summary, failure14, _ = _real_b_results(retrieval, real_b_distance, state_frame, parallel_metrics, old_results, old_diag)
    tie_gate = _self_and_tie_gates()
    gates = {"G4_definition": {"pass": True, "selected_sparse_term_ids": list(g4_payload["selected_sparse_term_ids"]), "K_total": int(g4_payload["K_total"]), "dmax": 2, "lambda": gmp.RIDGE_LAMBDA}, "state_order": {"pass": True, "order": "funMng→funAng→secMng→secAng→Vm→Pin"}, "ABC": {"pass": True, "A": [ABC_A_START, ABC_A_END], "B": [ABC_B_START, ABC_B_END], "C": [ABC_C_START, ABC_C_END]}, "B_support": {"pass": b_support_pass, "A_valid": 12286, "B_valid": 4913, "C_valid": 7371}, "frozen_g0": old_frozen_gate, "real_B_helper": real_b_gate, "serial_parallel": serial_parallel, "self_and_tie": tie_gate}
    if not tie_gate["pass"]:
        raise RuntimeError("self/tie gate failed")
    # Keep the caller's state-fit DataFrame as the source for model columns;
    # attach retrieval information in the same stable State_R order.
    summary = summary.sort_values("state_id").reset_index(drop=True)
    after = _snapshot()
    protection = _verify_protection(before, after)
    if not protection["all_protected_unchanged"]:
        raise RuntimeError("raw or protected historical result changed before output")
    _write_retrieval_outputs(summary, parallel_metrics, a_fp, c_fp, distance, real_b_distance, common_b, theta_a, theta_c, state_frame, g4_payload, old_results, old_diag, old_frozen_gate, real_b_gate, state_fit_info, gates, before, after, protection, logical_cpu_count, workers)
    print(f"Completed {EXPERIMENT_NAME}: exact={int(summary['exact_hit'].sum())}, shareable={int(summary['shareable_under_minus40dB'].sum())}, failure={int(summary['failure'].sum())}", flush=True)
    print(f"Results: {RESULT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
