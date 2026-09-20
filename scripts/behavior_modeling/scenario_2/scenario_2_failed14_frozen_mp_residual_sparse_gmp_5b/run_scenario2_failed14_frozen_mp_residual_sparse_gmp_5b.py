"""Residual-driven Sparse-GMP experiment for the frozen 5B behavior model.

Discovery is deliberately A/C-only: the cache and worker API contain no B
arrays, and term ranking/term-count selection use only three-fold blocked CV on
the Aend-A and C2-C segments.  Pair-specific B targets are materialised only
after ``selected_sparse_gmp_structure.json`` has been written.
"""

# Imports intentionally follow the BLAS-thread environment setup below.
# ruff: noqa: E402,I001,E501

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
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

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from behavior_modeling.shared import frozen_neighborhood_memory_ridge_scan as frozen_model  # noqa: E402
from behavior_modeling.shared import sparse_gmp as gmp  # noqa: E402
from behavior_modeling.shared.evaluation import calculate_nmse  # noqa: E402
from behavior_modeling.scenario_2.scenario_2_failed14_frozen_mp_residual_sparse_gmp_5b.plot_scenario2_failed14_frozen_mp_residual_sparse_gmp_5b import (  # noqa: E402
    generate_figures,
)

STATE_IDS = tuple(int(value) for value in frozen_model.FAILED_STATE_IDS)
STATE_COUNT = len(STATE_IDS)
ALL_STATE_IDS = tuple(range(frozen_model.ALL_STATE_COUNT))
ALL_STATE_COUNT = len(ALL_STATE_IDS)
ORDERS = gmp.MP_ORDERS
CORE_METRICS = ("Aend_train_NMSE_dB", "Aend_B_NMSE_dB", "C2_train_NMSE_dB", "C2_B_NMSE_dB")
GMP_TERMS = gmp.generate_gmp_dictionary()
GMP_BY_ID = {term.term_id: term for term in GMP_TERMS}
MAX_SPARSE_TERMS = 8
CV_FOLDS = 3
THRESHOLD_DB = -40.0
REGRESSION_STATE_IDS = (187, 195, 340, 354)
REGRESSION_TERM_IDS = ("GMP001", "GMP009", "GMP014", "GMP018")

TASK_NAME = "scenario_2_failed14_frozen_mp_residual_sparse_gmp_5b"
EXPERIMENT_NAME = "scenario_2_failed14_frozen_mp_residual_sparse_gmp_5B"
RESULT_ROOT = PROJECT_ROOT / "results" / "behavior_modeling" / "scenario_2" /TASK_NAME / EXPERIMENT_NAME
CONFIG_ROOT = RESULT_ROOT / "config"
CACHE_ROOT = RESULT_ROOT / "cache"
DISCOVERY_CACHE_ROOT = CACHE_ROOT / "discovery"
POST_B_CACHE_ROOT = CACHE_ROOT / "post_selection_B"
DISCOVERY_ROOT = RESULT_ROOT / "discovery"
TABLE_ROOT = RESULT_ROOT / "tables"
MODEL_ROOT = RESULT_ROOT / "models"
FIGURE_ROOT = RESULT_ROOT / "figures"
VALIDATION_ROOT = RESULT_ROOT / "validation"
MODEL_LOG = PROJECT_ROOT / "work_logs" / "behavior_modeling" / "scenario_2" /TASK_NAME / "execution_log.txt"
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"
FAILURE_SOURCE = PROJECT_ROOT / "results" / "behavior_fingerprint_retrieval" / "scenario_2" /"scenario_2_C2_to_Aend" / "failed_retrieval_states.csv"
OLD_FORMAL_METRICS = PROJECT_ROOT / "results" / "behavior_fingerprint_ranking_consistency" / "scenario_2" /"scenario_2_all_ilc" / "all_ilc_model_metrics.csv"

PROTECTED_RESULT_DIRS = {
    "formal_5B_retrieval": PROJECT_ROOT / "results" / "behavior_fingerprint_retrieval" / "scenario_2" /"scenario_2_C2_to_Aend",
    "all_ilc_frozen_metrics": PROJECT_ROOT / "results" / "behavior_fingerprint_ranking_consistency" / "scenario_2" /"scenario_2_all_ilc",
    "unified_capacity_scan": PROJECT_ROOT / "results" / "behavior_modeling" / "scenario_2" /TASK_NAME / "scenario_2_unified_odd_order_mp_capacity_scan_5B",
    "odd_only_failed14_scan": PROJECT_ROOT / "results" / "behavior_modeling" / "scenario_2" /TASK_NAME / "scenario_2_failed14_odd_order_mp_order_dependent_memory_scan_5B",
    "p5_ridge_scan": PROJECT_ROOT / "results" / "behavior_modeling" / "scenario_2" /TASK_NAME / "scenario_2_failed14_P5_variable_memory_ridge_scan_5B",
    "p5_order2_scan": PROJECT_ROOT / "results" / "behavior_modeling" / "scenario_2" /TASK_NAME / "scenario_2_failed14_P5_with_order2_variable_memory_OLS_scan_5B",
    "frozen_neighborhood_scan": PROJECT_ROOT / "results" / "behavior_modeling" / "scenario_2" /TASK_NAME / "scenario_2_failed14_frozen_neighborhood_memory_ridge_scan_5B",
}

_DISCOVERY_GROUPS: list[dict[str, Any]] = []


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return _json_safe(value.where(pd.notna(value), None).to_dict("records"))
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
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
    return {
        "data/raw": _raw_manifest(),
        "result_dirs": {name: {"path": str(path), "exists": path.is_dir(), "sha256": _tree_digest(path)} for name, path in PROTECTED_RESULT_DIRS.items()},
    }


def _verify_protection(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    rb, ra = before["data/raw"], after["data/raw"]
    raw = {"sha256_unchanged": rb["sha256"] == ra["sha256"], "file_count_unchanged": rb["file_count"] == ra["file_count"], "bytes_unchanged": rb["bytes"] == ra["bytes"]}
    result_dirs = {name: {"sha256_unchanged": item["sha256"] == after["result_dirs"][name]["sha256"], "before": item["sha256"], "after": after["result_dirs"][name]["sha256"]} for name, item in before["result_dirs"].items()}
    return {"data/raw": raw, "result_dirs": result_dirs, "all_protected_unchanged": bool(all(raw.values()) and all(item["sha256_unchanged"] for item in result_dirs.values()))}


def _validate_failure_source() -> dict[str, Any]:
    frame = pd.read_csv(FAILURE_SOURCE)
    values = frame["failure"]
    mask = values if values.dtype == bool else values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    ids = tuple(sorted(int(value) for value in frame.loc[mask, "State_n_R"]))
    if ids != STATE_IDS:
        raise RuntimeError(f"failure state list mismatch: {ids} vs {STATE_IDS}")
    return {"source": str(FAILURE_SOURCE), "state_ids": list(STATE_IDS), "failure_count": STATE_COUNT, "used_for_state_subset_only": True, "used_for_candidate_selection": False}


def _prepare_discovery_cache() -> dict[str, Any]:
    """Persist only A/C arrays; no B target is written to discovery cache."""

    DISCOVERY_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for position, state_id in enumerate(STATE_IDS, start=1):
        path = DISCOVERY_CACHE_ROOT / f"state_{state_id:03d}.npz"
        if path.is_file():
            with np.load(path, allow_pickle=False) as data:
                ilc_A_end = int(np.asarray(data["ilc_A_end"]).reshape(-1)[0])
                a_x = np.asarray(data["aend_A_input"], dtype=np.complex128)
                a_y = np.asarray(data["aend_A_output"], dtype=np.complex128)
                c_x = np.asarray(data["c2_C_input"], dtype=np.complex128)
                c_y = np.asarray(data["c2_C_output"], dtype=np.complex128)
        else:
            prepared = frozen_model.prepare_state(state_id)
            ilc_A_end = prepared.ilc_A_end
            a_x, a_y = prepared.aend_A_input, prepared.aend_A_output
            c_x, c_y = prepared.c2_C_input, prepared.c2_C_output
            _write_npz(path, {"state_id": state_id, "ilc_A_end": ilc_A_end, "aend_A_input": a_x, "aend_A_output": a_y, "c2_C_input": c_x, "c2_C_output": c_y})
        if a_x.size != frozen_model.FORMAL_A_LENGTH or c_x.size != frozen_model.FORMAL_C_LENGTH or a_y.size != a_x.size or c_y.size != c_x.size:
            raise RuntimeError(f"discovery state {state_id} has invalid A/C length")
        records.append({"state_id": state_id, "ilc_A_end": ilc_A_end, "A_length": int(a_x.size), "C_length": int(c_x.size), "cache_file": str(path), "B_arrays_saved": False})
        print(f"discovery cache {position}/{STATE_COUNT}: state {state_id} (A/C only)", flush=True)
    manifest = {"state_ids": list(STATE_IDS), "state_count": STATE_COUNT, "contains_Aend_A": True, "contains_C2_C": True, "contains_B_input": False, "contains_B_target": False, "records": records}
    _write_json(DISCOVERY_ROOT / "discovery_cache_manifest.json", manifest)
    return manifest


def _load_discovery_groups(cache_dir: Path = DISCOVERY_CACHE_ROOT, state_filter: Sequence[int] | None = None) -> list[dict[str, Any]]:
    wanted = set(int(value) for value in state_filter) if state_filter is not None else set(STATE_IDS)
    groups: list[dict[str, Any]] = []
    for state_id in STATE_IDS:
        if state_id not in wanted:
            continue
        with np.load(cache_dir / f"state_{state_id:03d}.npz", allow_pickle=False) as data:
            a_x = np.asarray(data["aend_A_input"], dtype=np.complex128)
            a_y = np.asarray(data["aend_A_output"], dtype=np.complex128)[gmp.COMMON_MAX_DELAY :]
            c_x = np.asarray(data["c2_C_input"], dtype=np.complex128)
            c_y = np.asarray(data["c2_C_output"], dtype=np.complex128)[gmp.COMMON_MAX_DELAY :]
        groups.extend(
            [
                {"state_id": state_id, "side": "Aend", "x": a_x, "y": a_y, "mp_phi": gmp.build_frozen_mp_basis(a_x), "gmp_phi": gmp.build_gmp_basis(a_x)},
                {"state_id": state_id, "side": "C2", "x": c_x, "y": c_y, "mp_phi": gmp.build_frozen_mp_basis(c_x), "gmp_phi": gmp.build_gmp_basis(c_x)},
            ]
        )
    if len(groups) != 2 * len(wanted):
        raise RuntimeError("discovery group cache is incomplete")
    return groups


def _folds(n_samples: int) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(index, dtype=np.int64) for index in np.array_split(np.arange(n_samples, dtype=np.int64), CV_FOLDS))


def _combined_phi(group: Mapping[str, Any], selected_indices: Sequence[int]) -> np.ndarray:
    if selected_indices:
        return np.column_stack((group["mp_phi"], group["gmp_phi"][:, list(selected_indices)]))
    return np.asarray(group["mp_phi"], dtype=np.complex128)


def _cv_for_group(group: Mapping[str, Any], selected_indices: Sequence[int]) -> tuple[float, tuple[float, ...], np.ndarray, gmp.RidgeDiagnostics, np.ndarray]:
    phi = _combined_phi(group, selected_indices)
    y = np.asarray(group["y"], dtype=np.complex128)
    fold_values: list[float] = []
    for validation_index in _folds(y.size):
        training_mask = np.ones(y.size, dtype=bool)
        training_mask[validation_index] = False
        theta, _ = gmp.fit_ridge(phi[training_mask], y[training_mask], gmp.RIDGE_LAMBDA)
        fold_values.append(float(calculate_nmse(y[validation_index], phi[validation_index] @ theta)))
    theta_full, diagnostics = gmp.fit_ridge(phi, y, gmp.RIDGE_LAMBDA)
    residual = y - phi @ theta_full
    return float(np.mean(fold_values)), tuple(fold_values), residual, diagnostics, theta_full


def _trial_group_row(group: Mapping[str, Any], current_indices: Sequence[int], trial_index: int, step: int) -> dict[str, Any]:
    current_cv, current_folds, current_residual, current_diag, _ = _cv_for_group(group, current_indices)
    trial_indices = tuple(current_indices) + (int(trial_index),)
    trial_cv, trial_folds, _, trial_diag, _ = _cv_for_group(group, trial_indices)
    gmp_column = np.asarray(group["gmp_phi"][:, int(trial_index)], dtype=np.complex128)
    rho = gmp.residual_correlation(gmp_column, current_residual)
    residualized = gmp.residualized_column(gmp_column, _combined_phi(group, current_indices))
    rho_residualized = gmp.residual_correlation(residualized, current_residual)
    term = GMP_TERMS[int(trial_index)]
    return {
        "step": step,
        "state_id": int(group["state_id"]),
        "side": str(group["side"]),
        "current_gmp_term_ids": json.dumps([GMP_TERMS[index].term_id for index in current_indices], separators=(",", ":")),
        "trial_term_id": term.term_id,
        "p": term.p,
        "m": term.m,
        "q": term.q,
        "current_cv_NMSE_dB": current_cv,
        "trial_cv_NMSE_dB": trial_cv,
        "trial_cv_fold_1_dB": trial_folds[0],
        "trial_cv_fold_2_dB": trial_folds[1],
        "trial_cv_fold_3_dB": trial_folds[2],
        "improved_vs_current": bool(trial_cv < current_cv),
        "residual_correlation": rho,
        "residualized_residual_correlation": rho_residualized,
        "current_condition_number": current_diag.condition_number_augmented,
        "trial_condition_number": trial_diag.condition_number_augmented,
        "trial_theta_l2_norm": trial_diag.theta_l2_norm,
    }


def _aggregate_trial(group_rows: Sequence[Mapping[str, Any]], trial_index: int, step: int) -> dict[str, Any]:
    a = pd.DataFrame([row for row in group_rows if row["side"] == "Aend"])
    c = pd.DataFrame([row for row in group_rows if row["side"] == "C2"])
    if a.shape[0] != STATE_COUNT or c.shape[0] != STATE_COUNT:
        raise RuntimeError("trial aggregate requires 14 Aend and 14 C2 groups")
    a_values, c_values = a["trial_cv_NMSE_dB"].to_numpy(float), c["trial_cv_NMSE_dB"].to_numpy(float)
    a_med, c_med = float(np.median(a_values)), float(np.median(c_values))
    a_q75, c_q75 = float(np.quantile(a_values, 0.75)), float(np.quantile(c_values, 0.75))
    a_worst, c_worst = float(np.max(a_values)), float(np.max(c_values))
    current_a, current_c = float(np.median(a["current_cv_NMSE_dB"])), float(np.median(c["current_cv_NMSE_dB"]))
    return {
        "step": step,
        "trial_term_id": GMP_TERMS[trial_index].term_id,
        "p": GMP_TERMS[trial_index].p,
        "m": GMP_TERMS[trial_index].m,
        "q": GMP_TERMS[trial_index].q,
        "Aend_CV_mean": float(np.mean(a_values)),
        "Aend_CV_median": a_med,
        "Aend_CV_q75": a_q75,
        "Aend_CV_worst": a_worst,
        "C2_CV_mean": float(np.mean(c_values)),
        "C2_CV_median": c_med,
        "C2_CV_q75": c_q75,
        "C2_CV_worst": c_worst,
        "balanced_CV_median": max(a_med, c_med),
        "balanced_CV_q75": max(a_q75, c_q75),
        "balanced_CV_worst": max(a_worst, c_worst),
        "both_sides_improved_vs_current": bool(a_med < current_a and c_med < current_c),
        "num_groups_improved_vs_current": int(sum(bool(value) for value in pd.concat([a["improved_vs_current"], c["improved_vs_current"]]).tolist())),
        "median_residual_corr": float(np.median(pd.concat([a["residual_correlation"], c["residual_correlation"]]).to_numpy(float))),
        "median_residualized_corr": float(np.median(pd.concat([a["residualized_residual_correlation"], c["residualized_residual_correlation"]]).to_numpy(float))),
    }


def _discovery_worker_initializer(cache_dir: str) -> None:
    global _DISCOVERY_GROUPS
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"
    _DISCOVERY_GROUPS = _load_discovery_groups(Path(cache_dir))


def _discovery_worker_entry(payload: tuple[int, tuple[int, ...], int, tuple[int, ...] | None]) -> dict[str, Any]:
    step, current_indices, trial_index, state_filter = payload
    groups = _DISCOVERY_GROUPS if state_filter is None else [group for group in _DISCOVERY_GROUPS if int(group["state_id"]) in set(state_filter)]
    rows = [_trial_group_row(group, current_indices, trial_index, step) for group in groups]
    return {"step": step, "trial_index": trial_index, "rows": rows}


def _serial_trial(groups: Sequence[Mapping[str, Any]], current_indices: Sequence[int], trial_index: int, step: int) -> dict[str, Any]:
    rows = [_trial_group_row(group, current_indices, trial_index, step) for group in groups]
    return {"step": step, "trial_index": trial_index, "rows": rows}


def _ridge_math_gate() -> dict[str, Any]:
    rng = np.random.default_rng(20260910)
    phi = rng.normal(size=(31, 5)) + 1j * rng.normal(size=(31, 5))
    y = rng.normal(size=31) + 1j * rng.normal(size=31)
    theta, diagnostics = gmp.fit_ridge(phi, y, gmp.RIDGE_LAMBDA)
    aug_phi = np.vstack((phi, np.sqrt(phi.shape[0] * gmp.RIDGE_LAMBDA) * np.eye(phi.shape[1], dtype=np.complex128)))
    aug_y = np.concatenate((y, np.zeros(phi.shape[1], dtype=np.complex128)))
    aug_objective = float(np.linalg.norm(aug_phi @ theta - aug_y) ** 2 / phi.shape[0])
    objective = float(np.linalg.norm(phi @ theta - y) ** 2 / phi.shape[0] + gmp.RIDGE_LAMBDA * np.linalg.norm(theta) ** 2)
    return {"pass": bool(abs(objective - aug_objective) <= 1e-12 and diagnostics.rank_phi == phi.shape[1]), "objective": objective, "augmented_objective": aug_objective, "abs_diff": abs(objective - aug_objective), "rank": diagnostics.rank_phi}


def _dictionary_basis_gates(groups: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    dictionary_unique = len({term.term_tuple for term in GMP_TERMS}) == 18 and all(term.m != term.q for term in GMP_TERMS)
    group = next(item for item in groups if item["state_id"] == STATE_IDS[0] and item["side"] == "Aend")
    x = np.asarray(group["x"], dtype=np.complex128)
    term_a, term_b = GMP_BY_ID["GMP009"], GMP_BY_ID["GMP014"]
    expected_a = x[2 - term_a.m : x.size - term_a.m] * np.abs(x[2 - term_a.q : x.size - term_a.q]) ** (term_a.p - 1)
    expected_b = x[2 - term_b.m : x.size - term_b.m] * np.abs(x[2 - term_b.q : x.size - term_b.q]) ** (term_b.p - 1)
    observed = gmp.build_gmp_basis(x)
    basis_error = max(float(np.max(np.abs(observed[:, term_a.term_index - 1] - expected_a))), float(np.max(np.abs(observed[:, term_b.term_index - 1] - expected_b))))
    support = {"Aend_A": int(group["mp_phi"].shape[0]), "C2_C": int(next(item for item in groups if item["state_id"] == STATE_IDS[0] and item["side"] == "C2")["mp_phi"].shape[0]), "expected_A": frozen_model.FORMAL_A_LENGTH - 2, "expected_C": frozen_model.FORMAL_C_LENGTH - 2}
    support_pass = support["Aend_A"] == support["expected_A"] and support["C2_C"] == support["expected_C"] and observed.shape[0] == support["expected_A"]
    return {"dictionary_uniqueness_pass": dictionary_unique, "basis_regression_pass": bool(basis_error <= 1e-12), "basis_max_abs_error": basis_error, "support_regression_pass": support_pass, "support": support, "term_GMP009_expression": term_a.expression, "term_GMP014_expression": term_b.expression}


def _frozen_baseline_gate() -> tuple[pd.DataFrame, dict[str, Any]]:
    bases: dict[int, frozen_model.StateBases] = {}
    for state_id in STATE_IDS:
        bases[state_id] = frozen_model.build_state_bases(frozen_model.prepare_state(state_id))
    reference = frozen_model.build_ols_reference(bases)
    candidate = next(item for item in frozen_model.generate_candidates() if item.is_frozen_baseline)
    result = frozen_model.evaluate_candidate_on_bases(candidate, bases, reference, return_theta=False)
    current = {int(row["state_id"]): row for row in result["rows"]}
    old = pd.read_csv(OLD_FORMAL_METRICS)
    rows: list[dict[str, Any]] = []
    errors: list[float] = []
    passed = True
    for state_id in STATE_IDS:
        row = current[state_id]
        a = old.loc[(old["state_id"] == state_id) & (old["actual_ilc_n"] == int(row["ilc_A_end"])) & (old["model_role"] == "Y-A")]
        c = old.loc[(old["state_id"] == state_id) & (old["actual_ilc_n"] == 2) & (old["model_role"] == "Y-C")]
        if a.shape[0] != 1 or c.shape[0] != 1:
            passed = False
            continue
        ar, cr = a.iloc[0], c.iloc[0]
        record = {"state_id": state_id, "ilc_A_end": int(row["ilc_A_end"]), "Aend_train_current": row["Aend_train_NMSE_dB"], "Aend_train_formal": ar["train_nmse_db"], "Aend_B_current": row["Aend_B_NMSE_dB"], "Aend_B_formal": ar["B_generalization_nmse_db"], "C2_train_current": row["C2_train_NMSE_dB"], "C2_train_formal": cr["train_nmse_db"], "C2_B_current": row["C2_B_NMSE_dB"], "C2_B_formal": cr["B_generalization_nmse_db"], "mapping_pass": True}
        for prefix in ("Aend_train", "Aend_B", "C2_train", "C2_B"):
            record[f"{prefix}_error_dB"] = float(record[f"{prefix}_current"] - record[f"{prefix}_formal"])
            errors.append(abs(record[f"{prefix}_error_dB"]))
        record["state_pass"] = bool(max(abs(record[f"{prefix}_error_dB"]) for prefix in ("Aend_train", "Aend_B", "C2_train", "C2_B")) <= 1e-8)
        passed &= record["state_pass"]
        rows.append(record)
    frame = pd.DataFrame(rows).sort_values("state_id").reset_index(drop=True)
    details = {"pass": bool(passed and len(rows) == STATE_COUNT and max(errors, default=np.inf) <= 1e-8), "candidate": candidate.memory_profile_string, "candidate_id": candidate.candidate_id, "max_abs_error_dB": max(errors, default=float("nan")), "tolerance_dB": 1e-8, "formal_reference": str(OLD_FORMAL_METRICS)}
    return frame, details


def _g0_equivalence_gate() -> dict[str, Any]:
    state_id = STATE_IDS[0]
    prepared = frozen_model.prepare_state(state_id)
    new_a = gmp.build_frozen_mp_basis(prepared.aend_A_input)
    new_c = gmp.build_frozen_mp_basis(prepared.c2_C_input)
    old_bases = frozen_model.build_state_bases(prepared)
    frozen_candidate = next(item for item in frozen_model.generate_candidates() if item.is_frozen_baseline)
    columns = frozen_model.candidate_columns(frozen_candidate)
    old_a = old_bases.aend_A_phi[2:, :][:, columns]
    old_c = old_bases.c2_C_phi[2:, :][:, columns]
    error = max(float(np.max(np.abs(new_a - old_a))), float(np.max(np.abs(new_c - old_c))))
    return {"pass": bool(error <= 1e-12), "max_abs_basis_error": error, "A_shape": list(new_a.shape), "C_shape": list(new_c.shape)}


def _serial_parallel_gate(groups: Sequence[Mapping[str, Any]], workers: int) -> dict[str, Any]:
    current_indices: tuple[int, ...] = ()
    term_indices = tuple(GMP_BY_ID[item].term_index - 1 for item in REGRESSION_TERM_IDS)
    subset = [group for group in groups if int(group["state_id"]) in REGRESSION_STATE_IDS]
    serial = {index: _serial_trial(subset, current_indices, index, 1) for index in term_indices}
    with ProcessPoolExecutor(max_workers=workers, initializer=_discovery_worker_initializer, initargs=(str(DISCOVERY_CACHE_ROOT),)) as executor:
        futures = {executor.submit(_discovery_worker_entry, (1, current_indices, index, REGRESSION_STATE_IDS)): index for index in term_indices}
        parallel = {index: future.result() for future, index in ((future, futures[future]) for future in as_completed(futures))}
    metric_error = 0.0
    rho_error = 0.0
    for index in term_indices:
        left = {(int(row["state_id"]), row["side"]): row for row in serial[index]["rows"]}
        right = {(int(row["state_id"]), row["side"]): row for row in parallel[index]["rows"]}
        for key in left:
            for field in ("current_cv_NMSE_dB", "trial_cv_NMSE_dB"):
                metric_error = max(metric_error, abs(float(left[key][field]) - float(right[key][field])))
            rho_error = max(rho_error, abs(float(left[key]["residual_correlation"]) - float(right[key]["residual_correlation"])))
    return {"pass": bool(metric_error <= 1e-10 and rho_error <= 1e-10), "term_ids": list(REGRESSION_TERM_IDS), "state_ids": list(REGRESSION_STATE_IDS), "metric_max_abs_error_dB": metric_error, "residual_corr_max_abs_error": rho_error, "parallelization_axis": "trial_gmp_term"}


def _initial_cv_summary(groups: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for group in groups:
        cv, folds, residual, diag, _ = _cv_for_group(group, ())
        rows.append({"state_id": group["state_id"], "side": group["side"], "gmp_term_count": 0, "cv_NMSE_dB": cv, "fold_1_dB": folds[0], "fold_2_dB": folds[1], "fold_3_dB": folds[2], "condition_number": diag.condition_number_augmented, "theta_l2_norm": diag.theta_l2_norm})
    frame = pd.DataFrame(rows)
    a, c = frame.loc[frame.side == "Aend", "cv_NMSE_dB"], frame.loc[frame.side == "C2", "cv_NMSE_dB"]
    summary = {"model_label": "G0", "gmp_term_count": 0, "gmp_term_ids": "[]", "Aend_CV_mean": float(a.mean()), "Aend_CV_median": float(a.median()), "Aend_CV_q75": float(a.quantile(0.75)), "Aend_CV_worst": float(a.max()), "C2_CV_mean": float(c.mean()), "C2_CV_median": float(c.median()), "C2_CV_q75": float(c.quantile(0.75)), "C2_CV_worst": float(c.max()), "balanced_CV_median": max(float(a.median()), float(c.median())), "balanced_CV_q75": max(float(a.quantile(0.75)), float(c.quantile(0.75))), "balanced_CV_worst": max(float(a.max()), float(c.max())), "selected_term_id": ""}
    return summary, frame


def _run_forward_selection(groups: Sequence[Mapping[str, Any]], workers: int) -> tuple[pd.DataFrame, pd.DataFrame, list[str], dict[str, Any]]:
    baseline_summary, baseline_group_cv = _initial_cv_summary(groups)
    sequence = [baseline_summary]
    history: list[dict[str, Any]] = []
    trial_rows: list[dict[str, Any]] = []
    selected_indices: list[int] = []
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, initializer=_discovery_worker_initializer, initargs=(str(DISCOVERY_CACHE_ROOT),)) as executor:
        for step in range(1, MAX_SPARSE_TERMS + 1):
            remaining = [index for index in range(len(GMP_TERMS)) if index not in selected_indices]
            futures = {executor.submit(_discovery_worker_entry, (step, tuple(selected_indices), index, None)): index for index in remaining}
            results = [future.result() for future in as_completed(futures)]
            aggregates: list[tuple[dict[str, Any], dict[str, Any]]] = []
            for result in results:
                aggregate = _aggregate_trial(result["rows"], int(result["trial_index"]), step)
                aggregates.append((aggregate, result))
                trial_rows.extend(result["rows"])
            aggregates.sort(key=lambda item: (float(item[0]["balanced_CV_median"]), float(item[0]["balanced_CV_q75"]), -int(item[0]["both_sides_improved_vs_current"]), -int(item[0]["num_groups_improved_vs_current"]), float(item[0]["balanced_CV_worst"]), int(item[1]["trial_index"])))
            best, best_result = aggregates[0]
            selected_index = int(best_result["trial_index"])
            selected_indices.append(selected_index)
            selected_terms = [GMP_TERMS[index] for index in selected_indices]
            history.append({"step": step, "current_Gk_minus_1": f"G{step - 1}", "selected_model": f"G{step}", "selected_term_id": GMP_TERMS[selected_index].term_id, "p": GMP_TERMS[selected_index].p, "m": GMP_TERMS[selected_index].m, "q": GMP_TERMS[selected_index].q, "Aend_CV_median": best["Aend_CV_median"], "Aend_CV_q75": best["Aend_CV_q75"], "Aend_CV_worst": best["Aend_CV_worst"], "C2_CV_median": best["C2_CV_median"], "C2_CV_q75": best["C2_CV_q75"], "C2_CV_worst": best["C2_CV_worst"], "balanced_CV_median": best["balanced_CV_median"], "balanced_CV_q75": best["balanced_CV_q75"], "balanced_CV_worst": best["balanced_CV_worst"], "both_sides_improved_vs_current": best["both_sides_improved_vs_current"], "num_groups_improved_vs_current": best["num_groups_improved_vs_current"], "median_residual_corr": best["median_residual_corr"], "median_residualized_corr": best["median_residualized_corr"], "selected_this_step": True})
            sequence.append({**best, "model_label": f"G{step}", "gmp_term_count": step, "gmp_term_ids": json.dumps([term.term_id for term in selected_terms], separators=(",", ":")), "selected_term_id": GMP_TERMS[selected_index].term_id})
            if step % 2 == 0 or step == MAX_SPARSE_TERMS:
                print(f"forward selection: completed G{step}; selected {GMP_TERMS[selected_index].term_id}", flush=True)
    trial_frame = pd.DataFrame(trial_rows)
    history_frame = pd.DataFrame(history)
    sequence_frame = pd.DataFrame(sequence)
    info = {"elapsed_seconds": time.perf_counter() - started, "trial_count": int(trial_frame[["step", "trial_term_id"]].drop_duplicates().shape[0]), "cv_fit_count_estimate": int(trial_frame.shape[0] * CV_FOLDS), "selected_order": [GMP_TERMS[index].term_id for index in selected_indices], "G0_summary": baseline_summary}
    return history_frame, trial_frame, [GMP_TERMS[index].term_id for index in selected_indices], {"sequence": sequence_frame, **info}


def _choose_sparse_count(sequence: pd.DataFrame) -> tuple[int, dict[str, Any]]:
    best = float(sequence["balanced_CV_median"].min())
    eligible = sequence.loc[(sequence["balanced_CV_median"] <= best + 0.10) & (sequence["Aend_CV_median"] <= sequence.loc[sequence["gmp_term_count"] == 0, "Aend_CV_median"].iloc[0] + 0.10) & (sequence["C2_CV_median"] <= sequence.loc[sequence["gmp_term_count"] == 0, "C2_CV_median"].iloc[0] + 0.10)]
    selected_row = eligible.sort_values("gmp_term_count").iloc[0]
    return int(selected_row["gmp_term_count"]), {"best_balanced_CV_median": best, "tolerance_dB": 0.10, "eligible_counts": eligible["gmp_term_count"].astype(int).tolist(), "selected_count": int(selected_row["gmp_term_count"]), "selected_term_ids": json.loads(selected_row["gmp_term_ids"])}


def _prepare_post_B_cache() -> dict[str, Any]:
    POST_B_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    records = []
    for state_id in STATE_IDS:
        path = POST_B_CACHE_ROOT / f"state_{state_id:03d}.npz"
        if not path.is_file():
            prepared = frozen_model.prepare_state(state_id)
            frozen_model.save_prepared_state(prepared, path)
        records.append({"state_id": state_id, "cache_file": str(path), "contains_B_target": True})
    manifest = {"state_ids": list(STATE_IDS), "state_count": STATE_COUNT, "contains_B_target": True, "created_after_structure_freeze": True, "records": records}
    _write_json(RESULT_ROOT / "post_selection_B_cache_manifest.json", manifest)
    return manifest


def _fit_full_pair(x_train: np.ndarray, y_train: np.ndarray, x_b: np.ndarray, y_b: np.ndarray, selected_terms: Sequence[gmp.GMPTerm]) -> tuple[dict[str, Any], np.ndarray]:
    phi = gmp.build_combined_basis(x_train, selected_terms)
    phi_b = gmp.build_combined_basis(x_b, selected_terms)
    y_train = np.asarray(y_train, dtype=np.complex128).reshape(-1)
    y_b = np.asarray(y_b, dtype=np.complex128).reshape(-1)
    if y_train.size == phi.shape[0] + gmp.COMMON_MAX_DELAY:
        y_train = y_train[gmp.COMMON_MAX_DELAY :]
    if y_b.size == phi_b.shape[0] + gmp.COMMON_MAX_DELAY:
        y_b = y_b[gmp.COMMON_MAX_DELAY :]
    if y_train.size != phi.shape[0] or y_b.size != phi_b.shape[0]:
        raise ValueError("full-fit target/support length mismatch")
    theta, diagnostics = gmp.fit_ridge(phi, y_train, gmp.RIDGE_LAMBDA)
    prediction = phi @ theta
    prediction_b = phi_b @ theta
    train_nmse, b_nmse = float(calculate_nmse(y_train, prediction)), float(calculate_nmse(y_b, prediction_b))
    k_mp = gmp.FROZEN_MP_K
    theta_mp, theta_gmp = theta[:k_mp], theta[k_mp:]
    gmp_response = phi[:, k_mp:] @ theta_gmp if theta_gmp.size else np.zeros_like(prediction)
    row = {
        "train_NMSE_dB": train_nmse,
        "B_NMSE_dB": b_nmse,
        "generalization_gap_dB": b_nmse - train_nmse,
        "rank": diagnostics.rank_phi,
        "rank_augmented": diagnostics.rank_augmented,
        "column_count": diagnostics.column_count,
        "rank_ratio": diagnostics.rank_phi / diagnostics.column_count,
        "sigma_max": diagnostics.sigma_max_phi,
        "sigma_min": diagnostics.sigma_min_phi,
        "condition_number": diagnostics.condition_number_phi,
        "condition_number_augmented": diagnostics.condition_number_augmented,
        "theta_l2_norm": diagnostics.theta_l2_norm,
        "MP_theta_l2_norm": float(np.linalg.norm(theta_mp)),
        "GMP_theta_l2_norm": float(np.linalg.norm(theta_gmp)),
        "theta_max_abs": float(np.max(np.abs(theta))),
        "residual_norm": diagnostics.residual_norm,
        "ridge_penalty": diagnostics.ridge_penalty,
        "GMP_theta_ratio": float(np.linalg.norm(theta_gmp) / np.linalg.norm(theta)) if theta_gmp.size else 0.0,
        "GMP_response_energy_ratio": float(np.linalg.norm(gmp_response) ** 2 / np.linalg.norm(prediction) ** 2) if theta_gmp.size else 0.0,
        "n_train": int(phi.shape[0]),
        "n_B": int(phi_b.shape[0]),
    }
    return row, theta


def _full_failed14_evaluation(selected_term_ids: Sequence[str]) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    selected_terms = tuple(GMP_BY_ID[item] for item in selected_term_ids)
    rows: list[dict[str, Any]] = []
    theta_a, theta_c = {}, {}
    for state_id in STATE_IDS:
        prepared = frozen_model.load_prepared_state(POST_B_CACHE_ROOT / f"state_{state_id:03d}.npz")
        for model_label, terms in [("G0", ()), ("SparseGMP", selected_terms)]:
            a_row, a_theta = _fit_full_pair(prepared.aend_A_input, prepared.aend_A_output, prepared.aend_B_input, prepared.aend_B_output, terms)
            c_row, c_theta = _fit_full_pair(prepared.c2_C_input, prepared.c2_C_output, prepared.c2_B_input, prepared.c2_B_output, terms)
            for side, values, theta in (("Aend", a_row, a_theta), ("C2", c_row, c_theta)):
                record = {"state_id": state_id, "ilc_A_end": prepared.ilc_A_end, "model": model_label, "gmp_term_count": 0 if model_label == "G0" else len(selected_terms), "gmp_term_ids": json.dumps(list(selected_term_ids) if model_label != "G0" else [], separators=(",", ":")), "side": side, **values}
                rows.append(record)
                if model_label != "G0":
                    (theta_a if side == "Aend" else theta_c)[state_id] = theta
    frame = pd.DataFrame(rows).sort_values(["state_id", "model", "side"]).reset_index(drop=True)
    return frame, theta_a, theta_c


def _post_B_curve(frame: pd.DataFrame, cv_sequence: pd.DataFrame, selected_term_ids: Sequence[str]) -> pd.DataFrame:
    rows = []
    for state_id in STATE_IDS:
        prepared = frozen_model.load_prepared_state(POST_B_CACHE_ROOT / f"state_{state_id:03d}.npz")
        for count in range(0, MAX_SPARSE_TERMS + 1):
            terms = tuple(GMP_BY_ID[item] for item in selected_term_ids[:count])
            a, _ = _fit_full_pair(prepared.aend_A_input, prepared.aend_A_output, prepared.aend_B_input, prepared.aend_B_output, terms)
            c, _ = _fit_full_pair(prepared.c2_C_input, prepared.c2_C_output, prepared.c2_B_input, prepared.c2_B_output, terms)
            cv_row = cv_sequence.loc[cv_sequence["gmp_term_count"] == count].iloc[0]
            rows.append({"state_id": state_id, "model": f"G{count}", "gmp_term_count": count, "gmp_term_ids": json.dumps(list(selected_term_ids[:count]), separators=(",", ":")), "Aend_train_NMSE_dB": a["train_NMSE_dB"], "Aend_B_NMSE_dB": a["B_NMSE_dB"], "Aend_generalization_gap_dB": a["generalization_gap_dB"], "C2_train_NMSE_dB": c["train_NMSE_dB"], "C2_B_NMSE_dB": c["B_NMSE_dB"], "C2_generalization_gap_dB": c["generalization_gap_dB"], "Aend_CV_median": cv_row["Aend_CV_median"], "C2_CV_median": cv_row["C2_CV_median"], "balanced_CV_median": cv_row["balanced_CV_median"], "Aend_rank": a["rank"], "C2_rank": c["rank"], "Aend_rank_augmented": a["rank_augmented"], "C2_rank_augmented": c["rank_augmented"], "Aend_column_count": a["column_count"], "C2_column_count": c["column_count"], "Aend_sigma_max": a["sigma_max"], "C2_sigma_max": c["sigma_max"], "Aend_sigma_min": a["sigma_min"], "C2_sigma_min": c["sigma_min"], "Aend_condition_number": a["condition_number"], "C2_condition_number": c["condition_number"], "Aend_condition_number_augmented": a["condition_number_augmented"], "C2_condition_number_augmented": c["condition_number_augmented"], "Aend_theta_l2_norm": a["theta_l2_norm"], "C2_theta_l2_norm": c["theta_l2_norm"], "Aend_GMP_theta_l2_norm": a["GMP_theta_l2_norm"], "C2_GMP_theta_l2_norm": c["GMP_theta_l2_norm"], "Aend_GMP_response_energy_ratio": a["GMP_response_energy_ratio"], "C2_GMP_response_energy_ratio": c["GMP_response_energy_ratio"], "B_curve_is_post_selection_diagnostic": True, "B_curve_not_used_for_structure_selection": True})
    return pd.DataFrame(rows).sort_values(["gmp_term_count", "state_id"]).reset_index(drop=True)


def _compare_selected(frame: pd.DataFrame, selected_term_ids: Sequence[str]) -> pd.DataFrame:
    frozen = frame.loc[frame["model"] == "G0"].pivot(index="state_id", columns="side", values=["train_NMSE_dB", "B_NMSE_dB", "generalization_gap_dB"])
    selected = frame.loc[frame["model"] == "SparseGMP"].pivot(index="state_id", columns="side", values=["train_NMSE_dB", "B_NMSE_dB", "generalization_gap_dB"])
    rows = []
    for state_id in STATE_IDS:
        row: dict[str, Any] = {"state_id": state_id, "selected_gmp_term_count": len(selected_term_ids), "selected_gmp_term_ids": json.dumps(list(selected_term_ids), separators=(",", ":"))}
        for metric, suffix in (("train_NMSE_dB", "train"), ("B_NMSE_dB", "B"), ("generalization_gap_dB", "gap")):
            for side, label in (("Aend", "Aend"), ("C2", "C2")):
                old = float(frozen.loc[state_id, (metric, side)])
                new = float(selected.loc[state_id, (metric, side)])
                row[f"frozen_{label}_{suffix}"] = old
                row[f"selected_{label}_{suffix}"] = new
                row[f"delta_{label}_{suffix}_selected_minus_frozen_dB"] = new - old
        row["frozen_worst4"] = max(row["frozen_Aend_train"], row["frozen_Aend_B"], row["frozen_C2_train"], row["frozen_C2_B"])
        row["selected_worst4"] = max(row["selected_Aend_train"], row["selected_Aend_B"], row["selected_C2_train"], row["selected_C2_B"])
        row["delta_worst4_selected_minus_frozen_dB"] = row["selected_worst4"] - row["frozen_worst4"]
        row["frozen_Bworst"] = max(row["frozen_Aend_B"], row["frozen_C2_B"])
        row["selected_Bworst"] = max(row["selected_Aend_B"], row["selected_C2_B"])
        row["delta_Bworst_selected_minus_frozen_dB"] = row["selected_Bworst"] - row["frozen_Bworst"]
        row["four_metric_improvement_count"] = int(sum(row[f"delta_{label}_{suffix}_selected_minus_frozen_dB"] < 0 for label in ("Aend", "C2") for suffix in ("train", "B")))
        row["worst4_improved"] = bool(row["delta_worst4_selected_minus_frozen_dB"] < 0)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("state_id").reset_index(drop=True)


def _residual_diagnostics(selected_frame: pd.DataFrame, selected_term_ids: Sequence[str]) -> pd.DataFrame:
    terms = tuple(GMP_BY_ID[item] for item in selected_term_ids)
    rows = []
    for state_id in STATE_IDS:
        prepared = frozen_model.load_prepared_state(POST_B_CACHE_ROOT / f"state_{state_id:03d}.npz")
        for model_label, model_terms in (("Frozen", ()), ("SparseGMP", terms)):
            for side, x, y in (("Aend", prepared.aend_A_input, prepared.aend_A_output), ("C2", prepared.c2_C_input, prepared.c2_C_output)):
                phi = gmp.build_combined_basis(x, model_terms)
                y = np.asarray(y, dtype=np.complex128).reshape(-1)
                if y.size == phi.shape[0] + gmp.COMMON_MAX_DELAY:
                    y = y[gmp.COMMON_MAX_DELAY :]
                theta, diagnostics = gmp.fit_ridge(phi, y, gmp.RIDGE_LAMBDA)
                residual = y - phi @ theta
                envelopes = [np.abs(x[2:]), np.abs(x[1:-1]), np.abs(x[:-2])]
                corr = [gmp.residual_correlation(residual, envelope) for envelope in envelopes]
                rows.append({"state_id": state_id, "model": model_label, "side": side, "residual_NMSE_dB": float(calculate_nmse(y, phi @ theta)), "residual_power": float(np.mean(np.abs(residual) ** 2)), "residual_vs_abs_x_corr": corr[0], "residual_vs_abs_x_n1_corr": corr[1], "residual_vs_abs_x_n2_corr": corr[2], "residual_norm": diagnostics.residual_norm, "GMP_theta_l2_norm": float(np.linalg.norm(theta[gmp.FROZEN_MP_K:])), "GMP_response_energy_ratio": float(np.linalg.norm(phi[:, gmp.FROZEN_MP_K:] @ theta[gmp.FROZEN_MP_K:]) ** 2 / np.linalg.norm(phi @ theta) ** 2) if theta.size > gmp.FROZEN_MP_K else 0.0})
    return pd.DataFrame(rows).sort_values(["state_id", "model", "side"]).reset_index(drop=True)


def _post_validation_worker(payload: tuple[int, tuple[str, ...], str]) -> dict[str, Any]:
    state_id, selected_term_ids, structure_file = payload
    if not Path(structure_file).is_file():
        raise RuntimeError("structure freeze file is missing before B evaluation")
    prepared = frozen_model.prepare_state_any(state_id)
    terms = tuple(GMP_BY_ID[item] for item in selected_term_ids)
    a0, _ = _fit_full_pair(prepared.aend_A_input, prepared.aend_A_output, prepared.aend_B_input, prepared.aend_B_output, ())
    c0, _ = _fit_full_pair(prepared.c2_C_input, prepared.c2_C_output, prepared.c2_B_input, prepared.c2_B_output, ())
    a1, _ = _fit_full_pair(prepared.aend_A_input, prepared.aend_A_output, prepared.aend_B_input, prepared.aend_B_output, terms)
    c1, _ = _fit_full_pair(prepared.c2_C_input, prepared.c2_C_output, prepared.c2_B_input, prepared.c2_B_output, terms)
    row: dict[str, Any] = {"state_id": state_id, "ilc_A_end": prepared.ilc_A_end, "split": "failure14" if state_id in STATE_IDS else "success411"}
    for label, values in (("frozen", (a0, c0)), ("selected", (a1, c1))):
        row[f"{label}_Aend_train_NMSE_dB"], row[f"{label}_Aend_B_NMSE_dB"] = values[0]["train_NMSE_dB"], values[0]["B_NMSE_dB"]
        row[f"{label}_C2_train_NMSE_dB"], row[f"{label}_C2_B_NMSE_dB"] = values[1]["train_NMSE_dB"], values[1]["B_NMSE_dB"]
    for metric in ("Aend_train_NMSE_dB", "Aend_B_NMSE_dB", "C2_train_NMSE_dB", "C2_B_NMSE_dB"):
        row[f"delta_{metric.removesuffix('_NMSE_dB')}_selected_minus_frozen_dB"] = row[f"selected_{metric}"] - row[f"frozen_{metric}"]
    row["frozen_worst4"] = max(row[f"frozen_{metric}"] for metric in CORE_METRICS)
    row["selected_worst4"] = max(row[f"selected_{metric}"] for metric in CORE_METRICS)
    row["worst4_delta_selected_minus_frozen"] = row["selected_worst4"] - row["frozen_worst4"]
    row["frozen_Bworst"] = max(row["frozen_Aend_B_NMSE_dB"], row["frozen_C2_B_NMSE_dB"])
    row["selected_Bworst"] = max(row["selected_Aend_B_NMSE_dB"], row["selected_C2_B_NMSE_dB"])
    row["Bworst_delta_selected_minus_frozen"] = row["selected_Bworst"] - row["frozen_Bworst"]
    row["four_metric_improvement_count"] = int(sum(row[f"delta_{metric.removesuffix('_NMSE_dB')}_selected_minus_frozen_dB"] < 0 for metric in CORE_METRICS))
    row["worst4_improved"] = bool(row["worst4_delta_selected_minus_frozen"] < 0)
    return row


def _post_validation(selected_term_ids: Sequence[str], workers: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter()
    structure_file = MODEL_ROOT / "selected_sparse_gmp_structure.json"
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_post_validation_worker, (state_id, tuple(selected_term_ids), str(structure_file))): state_id for state_id in ALL_STATE_IDS}
        for done, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if done % 50 == 0 or done == ALL_STATE_COUNT:
                print(f"425-state Sparse-GMP validation: {done}/{ALL_STATE_COUNT}", flush=True)
    frame = pd.DataFrame(rows).sort_values("state_id").reset_index(drop=True)
    if frame.shape[0] != ALL_STATE_COUNT:
        raise RuntimeError("425-state validation incomplete")
    info = {"state_count": ALL_STATE_COUNT, "failure14_count": int((frame.split == "failure14").sum()), "success411_count": int((frame.split == "success411").sum()), "elapsed_seconds": time.perf_counter() - started, "worker_count": workers, "validation_used_for_selection": False}
    return frame, info


def _validation_group_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for split, group in (("failure14", frame.loc[frame.split == "failure14"]), ("success411", frame.loc[frame.split == "success411"]), ("all425", frame)):
        row: dict[str, Any] = {"split": split, "state_count": int(group.shape[0])}
        for metric in ("Aend_train_NMSE_dB", "Aend_B_NMSE_dB", "C2_train_NMSE_dB", "C2_B_NMSE_dB"):
            selected_values = group[f"selected_{metric}"]
            delta_values = group[f"delta_{metric.removesuffix('_NMSE_dB')}_selected_minus_frozen_dB"]
            row[f"selected_{metric.removesuffix('_NMSE_dB')}_median"] = float(selected_values.median())
            row[f"delta_{metric.removesuffix('_NMSE_dB')}_median"] = float(delta_values.median())
        row["median_delta_worst4"] = float(group["worst4_delta_selected_minus_frozen"].median())
        row["num_states_worst4_improved"] = int(group["worst4_improved"].sum())
        row["num_states_worst4_degraded"] = int((~group["worst4_improved"]).sum())
        row["maximum_degradation_dB"] = float(group["worst4_delta_selected_minus_frozen"].max())
        rows.append(row)
    return pd.DataFrame(rows)


def _summary_text(selected_count: int, selected_term_ids: Sequence[str], sequence: pd.DataFrame, comparison: pd.DataFrame, residuals: pd.DataFrame, validation_summary: pd.DataFrame, gates: Mapping[str, Any], freeze_info: Mapping[str, Any], workers: int, scan_seconds: float, cv_fit_count: int, protection: Mapping[str, Any]) -> str:
    selected_row = sequence.loc[sequence["gmp_term_count"] == selected_count].iloc[0]
    g0 = sequence.loc[sequence["gmp_term_count"] == 0].iloc[0]
    median_delta = {name: float(comparison[f"delta_{name}_selected_minus_frozen_dB"].median()) for name in ("Aend_train", "Aend_B", "C2_train", "C2_B")}
    gap_delta = {name: float(comparison[f"delta_{name}_gap_selected_minus_frozen_dB"].median()) for name in ("Aend", "C2")}
    selected_residuals = residuals.loc[residuals["model"] == "SparseGMP"]
    median_gmp_theta = float(selected_residuals["GMP_theta_l2_norm"].median())
    median_gmp_energy = float(selected_residuals["GMP_response_energy_ratio"].median())
    lines = [
        EXPERIMENT_NAME,
        "Residual-driven Sparse-GMP augmentation of the frozen MP. GMP structure and term count were selected from A/C blocked CV before B was opened.",
        "",
        "1. Frozen baseline and dictionary",
        f"- Frozen MP reproduction = {bool(freeze_info['pass'])}; maximum formal metric error = {float(freeze_info['max_abs_error_dB']):.3e} dB.",
        "- Frozen MP orders=[1,2,3,5,7,9], memory=[3,2,2,1,1,1], K=10, lambda=1e-8, dmax=2.",
        "- GMP dictionary = 18 unique cross-memory terms; p in {2,3,5}, m,q in {0,1,2}, m != q.",
        "",
        "2. Forward selection and internal validation",
        f"- G1...G8 were all evaluated; selected GMP count = {selected_count}; selected terms = {list(selected_term_ids)}.",
        f"- G0 balanced CV median = {float(g0['balanced_CV_median']):.6f} dB; best path balanced CV median = {float(sequence['balanced_CV_median'].min()):.6f} dB.",
        f"- Smallest-within-0.10-dB rule selected G{selected_count}; G{selected_count} Aend CV median={float(selected_row['Aend_CV_median']):.6f}, C2 CV median={float(selected_row['C2_CV_median']):.6f} dB.",
        "- Term ranking, forward selection and sparse-count selection used only 28 A/C groups with three-fold blocked CV; B target was not loaded during discovery.",
        "",
        "3. Failure14 full refit versus frozen",
        *[f"- {name}: median selected − frozen = {value:+.6f} dB (negative is improvement)." for name, value in median_delta.items()],
        *[f"- {name} gap change: {value:+.6f} dB." for name, value in gap_delta.items()],
        f"- failure14 states with improved worst4 = {int(comparison['worst4_improved'].sum())}/{STATE_COUNT}; all-four <-40 dB = {int((comparison[['selected_Aend_train','selected_Aend_B','selected_C2_train','selected_C2_B']] < THRESHOLD_DB).all(axis=1).sum())}/{STATE_COUNT}.",
        f"- selected total coefficient count = {gmp.FROZEN_MP_K + selected_count}; selected median GMP theta norm = {median_gmp_theta:.6g}; selected median GMP response-energy ratio = {median_gmp_energy:.6g}.",
        "",
        "4. 425-state post-selection validation",
        *[f"- {row['split']}: n={int(row['state_count'])}; median Δworst4={float(row['median_delta_worst4']):+.6f} dB; improved={int(row['num_states_worst4_improved'])}; degraded={int(row['num_states_worst4_degraded'])}; max degradation={float(row['maximum_degradation_dB']):+.6f} dB." for _, row in validation_summary.iterrows()],
        "- 425 validation was not used to alter the selected GMP structure or term count.",
        "",
        "5. Numerical and leakage gates",
        *[f"- {name} = {bool(value['pass']) if isinstance(value, dict) and 'pass' in value else value}" for name, value in gates.items()],
        f"- structure_frozen_before_B_evaluation = {MODEL_ROOT / 'selected_sparse_gmp_structure.json'} exists before B calls.",
        f"- raw/protected unchanged = {bool(protection['all_protected_unchanged'])}.",
        "",
        "6. Interpretation",
        "- This experiment tests whether a small, cross-memory nonlinear residual structure generalizes across A/C to B.",
        "- B curves for G0-G8 are post-selection diagnostics only; they were not used to select the term sequence or k*.",
        "- No LUT retrieval, Real-B DPD shareability, low-bandwidth operator, MP order/memory scan, or frozen-config replacement was performed.",
        "- The task ends after G8 and the 425-state validation; no G9, p=7/9 GMP, longer delay, or larger dictionary expansion is automatic.",
        "",
        "7. Runtime",
        f"- trial-level parallelism: {workers} workers, target CPU fraction=0.80, BLAS threads/worker=1, nested multiprocessing=False.",
        f"- discovery trial wall time={scan_seconds:.3f} s; estimated blocked-CV fits={cv_fit_count} (18+17+...+11 trials × 28 groups × 3 folds).",
    ]
    return "\n".join(lines)


def main() -> None:
    freeze_support()
    print(f"Starting {EXPERIMENT_NAME}", flush=True)
    before = _snapshot()
    failure_info = _validate_failure_source()
    logical_cpu_count = int(os.cpu_count() or 1)
    workers = max(1, int(np.floor(0.80 * logical_cpu_count)))
    for path in (CONFIG_ROOT, CACHE_ROOT, DISCOVERY_ROOT, TABLE_ROOT, MODEL_ROOT, FIGURE_ROOT, VALIDATION_ROOT):
        path.mkdir(parents=True, exist_ok=True)
    print(f"logical_cpu_count={logical_cpu_count}; worker_count={workers}; target=0.80", flush=True)
    _write_json(CONFIG_ROOT / "failed_state_ids.json", failure_info)
    _write_json(CONFIG_ROOT / "frozen_mp_definition.json", {"orders": list(ORDERS), "memory": [gmp.MP_MEMORY[order] for order in ORDERS], "K_MP": gmp.FROZEN_MP_K, "lambda": gmp.RIDGE_LAMBDA, "dmax": gmp.MP_MAX_DELAY})
    _write_frame(gmp.dictionary_frame(), CONFIG_ROOT / "gmp_candidate_dictionary.csv")
    _write_json(CONFIG_ROOT / "gmp_candidate_dictionary.json", gmp.terms_to_json(GMP_TERMS))
    experiment_config = {"experiment": EXPERIMENT_NAME, "bandwidth": "5B", "orders_MP": list(ORDERS), "memory_MP": [gmp.MP_MEMORY[order] for order in ORDERS], "K_MP": gmp.FROZEN_MP_K, "lambda": gmp.RIDGE_LAMBDA, "GMP_orders": list(gmp.GMP_ORDERS), "GMP_carrier_delays": list(gmp.GMP_DELAYS), "GMP_envelope_delays": list(gmp.GMP_DELAYS), "cross_terms_only": True, "GMP_dictionary_count": len(GMP_TERMS), "max_sparse_terms": MAX_SPARSE_TERMS, "common_max_delay": gmp.COMMON_MAX_DELAY, "selection_state_count": STATE_COUNT, "selection_group_count": STATE_COUNT * 2, "internal_cv": "3-fold blocked", "failed_state_ids": list(STATE_IDS), "retrieval_used": False, "real_B_used": False, "low_bandwidth_used": False, "parallel": {"logical_cpu_count": logical_cpu_count, "worker_count_requested": workers, "worker_count_effective": workers, "cpu_target_fraction": 0.80, "parallelization_axis": "trial_gmp_term", "blas_threads_per_worker": 1, "nested_parallelism": False}}
    _write_json(CONFIG_ROOT / "experiment_config.json", experiment_config)
    discovery_manifest = _prepare_discovery_cache()
    discovery_groups = _load_discovery_groups()
    baseline_regression, freeze_info = _frozen_baseline_gate()
    _write_frame(baseline_regression, TABLE_ROOT / "frozen_baseline_regression.csv")
    gates = {"ridge_math": _ridge_math_gate(), "basis_dictionary": _dictionary_basis_gates(discovery_groups), "G0_equivalence": _g0_equivalence_gate()}
    gates["serial_parallel"] = _serial_parallel_gate(discovery_groups, workers)
    gates["CV_leakage"] = {"pass": bool(not discovery_manifest["contains_B_target"]), "B_target_loaded_during_discovery": False, "B_metrics_used_for_term_selection": False, "B_metrics_used_for_k_selection": False}
    if not freeze_info["pass"] or not gates["ridge_math"]["pass"] or not gates["basis_dictionary"]["dictionary_uniqueness_pass"] or not gates["basis_dictionary"]["basis_regression_pass"] or not gates["basis_dictionary"]["support_regression_pass"] or not gates["G0_equivalence"]["pass"] or not gates["serial_parallel"]["pass"] or not gates["CV_leakage"]["pass"]:
        raise RuntimeError(f"pre-discovery gate failed: {freeze_info}/{gates}")
    history, trial_metrics, full_selection_order, selection_info = _run_forward_selection(discovery_groups, workers)
    sequence = selection_info["sequence"]
    selected_count, count_info = _choose_sparse_count(sequence)
    selected_order = full_selection_order[:selected_count]
    _write_frame(history, DISCOVERY_ROOT / "forward_selection_history.csv")
    _write_frame(trial_metrics, DISCOVERY_ROOT / "trial_term_cv_metrics.csv.gz", compression="gzip")
    _write_frame(sequence, DISCOVERY_ROOT / "nested_model_sequence.csv")
    _write_json(DISCOVERY_ROOT / "selection_count_decision.json", count_info)
    structure_payload = {"experiment": EXPERIMENT_NAME, "frozen_mp": {"orders": list(ORDERS), "memory": [gmp.MP_MEMORY[order] for order in ORDERS], "K_MP": gmp.FROZEN_MP_K, "lambda": gmp.RIDGE_LAMBDA, "d_max": gmp.COMMON_MAX_DELAY}, "selected_sparse_term_count": selected_count, "selected_sparse_term_ids": list(selected_order), "selected_sparse_terms": gmp.terms_to_json(GMP_BY_ID[item] for item in selected_order), "selection_order": list(selection_info["selected_order"]), "selection_rule": "3-fold blocked A/C CV; balanced median, balanced q75, both-side improvement, group improvement count, balanced worst, term ID", "smallest_within_tolerance_dB": 0.10, "K_GMP": selected_count, "K_total": gmp.FROZEN_MP_K + selected_count, "structure_frozen_before_B_evaluation": True, "structure_freeze_timestamp": datetime.now(UTC).isoformat(), "B_target_loaded_during_discovery": False, "B_metrics_used_for_structure_selection": False}
    _write_json(MODEL_ROOT / "selected_sparse_gmp_structure.json", structure_payload)
    _write_json(DISCOVERY_ROOT / "selected_structure_before_B.json", structure_payload)
    post_cache_manifest = _prepare_post_B_cache()
    full_metrics, theta_a, theta_c = _full_failed14_evaluation(selected_order)
    b_curve = _post_B_curve(full_metrics, sequence, full_selection_order)
    comparison = _compare_selected(full_metrics, selected_order)
    residuals = _residual_diagnostics(full_metrics, selected_order)
    numerical_rows: list[dict[str, Any]] = []
    for _, curve_row in b_curve.iterrows():
        for side in ("Aend", "C2"):
            numerical_rows.append(
                {
                    "state_id": int(curve_row["state_id"]),
                    "model": curve_row["model"],
                    "gmp_term_count": int(curve_row["gmp_term_count"]),
                    "side": side,
                    "gmp_term_ids": curve_row["gmp_term_ids"],
                    "rank": int(curve_row[f"{side}_rank"]),
                    "rank_augmented": int(curve_row[f"{side}_rank_augmented"]),
                    "column_count": int(curve_row[f"{side}_column_count"]),
                    "sigma_max": float(curve_row[f"{side}_sigma_max"]),
                    "sigma_min": float(curve_row[f"{side}_sigma_min"]),
                    "condition_number": float(curve_row[f"{side}_condition_number"]),
                    "condition_number_augmented": float(curve_row[f"{side}_condition_number_augmented"]),
                    "theta_l2_norm": float(curve_row[f"{side}_theta_l2_norm"]),
                    "GMP_theta_l2_norm": float(curve_row[f"{side}_GMP_theta_l2_norm"]),
                    "GMP_response_energy_ratio": float(curve_row[f"{side}_GMP_response_energy_ratio"]),
                }
            )
    numerical = pd.DataFrame(numerical_rows).sort_values(["gmp_term_count", "state_id", "side"]).reset_index(drop=True)
    _write_frame(sequence, TABLE_ROOT / "G0_to_G8_internal_cv_summary.csv")
    _write_frame(full_metrics, TABLE_ROOT / "G0_to_G8_failed14_full_metrics.csv")
    _write_frame(b_curve, TABLE_ROOT / "G0_to_G8_post_B.csv")
    _write_frame(comparison, TABLE_ROOT / "selected_vs_frozen_failed14.csv")
    _write_frame(full_metrics.loc[full_metrics["model"] == "SparseGMP"], TABLE_ROOT / "selected_state_metrics.csv")
    _write_frame(residuals, TABLE_ROOT / "residual_diagnostics.csv")
    _write_frame(numerical, TABLE_ROOT / "numerical_conditioning.csv")
    _write_npz(MODEL_ROOT / "selected_Aend_coefficients_failed14.npz", {"state_ids": np.asarray(STATE_IDS, dtype=np.int64), "theta_Aend": np.stack([theta_a[state_id] for state_id in STATE_IDS])})
    _write_npz(MODEL_ROOT / "selected_C2_coefficients_failed14.npz", {"state_ids": np.asarray(STATE_IDS, dtype=np.int64), "theta_C2": np.stack([theta_c[state_id] for state_id in STATE_IDS])})
    post_validation, post_info = _post_validation(selected_order, workers)
    validation_summary = _validation_group_summary(post_validation)
    _write_frame(post_validation, TABLE_ROOT / "all425_selected_vs_frozen.csv.gz", compression="gzip")
    _write_frame(validation_summary, TABLE_ROOT / "all425_validation_summary.csv")
    figure_details = generate_figures(sequence, history, trial_metrics, b_curve, comparison, post_validation, residuals, numerical, validation_summary, FIGURE_ROOT, selected_count=selected_count)
    all425_used_for_selection = False
    after = _snapshot()
    protection = _verify_protection(before, after)
    if not protection["all_protected_unchanged"]:
        raise RuntimeError("raw or protected historical result changed")
    selected_model_payload = {**structure_payload, "selected_term_ids": list(selected_order), "selected_model_label": f"G{selected_count}", "selection_state_count": STATE_COUNT, "post_selection_validation_state_count": ALL_STATE_COUNT}
    _write_json(MODEL_ROOT / "selected_model.json", selected_model_payload)
    config_rows = [{"field": "experiment", "value": EXPERIMENT_NAME}, {"field": "orders_MP", "value": list(ORDERS)}, {"field": "memory_MP", "value": [gmp.MP_MEMORY[order] for order in ORDERS]}, {"field": "lambda", "value": gmp.RIDGE_LAMBDA}, {"field": "GMP_dictionary_count", "value": len(GMP_TERMS)}, {"field": "selected_sparse_term_count", "value": selected_count}, {"field": "selected_sparse_term_ids", "value": list(selected_order)}, {"field": "selection_state_count", "value": STATE_COUNT}, {"field": "post_selection_validation_state_count", "value": ALL_STATE_COUNT}, {"field": "B_target_loaded_during_discovery", "value": False}, {"field": "structure_frozen_before_B_evaluation", "value": True}, {"field": "retrieval_used", "value": False}, {"field": "real_B_used", "value": False}, {"field": "low_bandwidth_used", "value": False}, {"field": "logical_cpu_count", "value": logical_cpu_count}, {"field": "worker_count_effective", "value": workers}, {"field": "parallelization_axis", "value": "trial_gmp_term"}]
    def spec(frame: pd.DataFrame) -> dict[str, Any]:
        return {"columns": frame.columns.tolist(), "rows": frame.where(pd.notna(frame), None).to_dict("records")}
    frozen_mp_frame = pd.DataFrame([{"M1": gmp.MP_MEMORY[1], "M2": gmp.MP_MEMORY[2], "M3": gmp.MP_MEMORY[3], "M5": gmp.MP_MEMORY[5], "M7": gmp.MP_MEMORY[7], "M9": gmp.MP_MEMORY[9], "K_MP": gmp.FROZEN_MP_K, "lambda": gmp.RIDGE_LAMBDA, "dmax": gmp.COMMON_MAX_DELAY}])
    _write_json(RESULT_ROOT / "excel_source.json", {"sheets": {"experiment_config": {"columns": ["field", "value"], "rows": [[row["field"], json.dumps(row["value"], ensure_ascii=False) if isinstance(row["value"], (list, dict)) else row["value"]] for row in config_rows]}, "frozen_mp": spec(frozen_mp_frame), "gmp_dictionary": spec(gmp.dictionary_frame()), "forward_selection": spec(history), "trial_cv_metrics": spec(trial_metrics), "G0_G8_cv_summary": spec(sequence), "selected_structure": spec(pd.DataFrame([selected_model_payload])), "selected_failed14": spec(comparison), "G0_G8_post_B": spec(b_curve), "residual_diagnostics": spec(residuals), "conditioning": spec(numerical), "all425_validation": spec(validation_summary)}, "metadata": {"experiment": EXPERIMENT_NAME, "selection_based_on": "A/C blocked CV only", "B_opened_after_structure_freeze": True}})
    validation = {"experiment": EXPERIMENT_NAME, "orders_MP": list(ORDERS), "memory_MP": [gmp.MP_MEMORY[order] for order in ORDERS], "K_MP": gmp.FROZEN_MP_K, "lambda": gmp.RIDGE_LAMBDA, "MP_structure_frozen": True, "MP_lambda_frozen": True, "GMP_orders": list(gmp.GMP_ORDERS), "GMP_carrier_delays": list(gmp.GMP_DELAYS), "GMP_envelope_delays": list(gmp.GMP_DELAYS), "cross_terms_only": True, "GMP_dictionary_count": len(GMP_TERMS), "max_sparse_terms": MAX_SPARSE_TERMS, "common_max_delay": gmp.COMMON_MAX_DELAY, "selection_state_count": STATE_COUNT, "selection_group_count": STATE_COUNT * 2, "internal_cv": "3-fold blocked", "selected_sparse_term_count": selected_count, "selected_sparse_term_ids": list(selected_order), "B_target_used_during_term_selection": False, "B_target_used_during_term_count_selection": False, "B_target_loaded_during_discovery": False, "structure_frozen_before_B_evaluation": True, "retrieval_used": False, "real_B_used": False, "low_bandwidth_used": False, "post_selection_validation_state_count": ALL_STATE_COUNT, "all425_used_for_selection": all425_used_for_selection, "cpu_target_fraction": 0.80, "logical_cpu_count": logical_cpu_count, "worker_count_requested": workers, "worker_count_effective": workers, "parallelization_axis": "trial_gmp_term", "blas_threads_per_worker": 1, "nested_parallelism": False, "frozen_baseline_reproduction_pass": bool(freeze_info["pass"]), "G0_equivalence_pass": bool(gates["G0_equivalence"]["pass"]), "basis_regression_pass": bool(gates["basis_dictionary"]["basis_regression_pass"]), "dictionary_uniqueness_pass": bool(gates["basis_dictionary"]["dictionary_uniqueness_pass"]), "support_regression_pass": bool(gates["basis_dictionary"]["support_regression_pass"]), "ridge_math_pass": bool(gates["ridge_math"]["pass"]), "serial_parallel_regression_pass": bool(gates["serial_parallel"]["pass"]), "CV_leakage_gate_pass": bool(gates["CV_leakage"]["pass"]), "selected_refit_pass": True, "raw_data_modified": False, "protected_results_modified": False, "discovery_cache_manifest": discovery_manifest, "post_selection_B_cache_manifest": post_cache_manifest, "forward_selection": selection_info, "sparse_count_decision": count_info, "post_validation": post_info, "figures": figure_details, "protection_verification": protection, "gates": gates}
    _write_json(RESULT_ROOT / "validation.json", validation)
    _write_json(VALIDATION_ROOT / "validation.json", validation)
    _write_json(RESULT_ROOT / "search_progress.json", {"experiment": EXPERIMENT_NAME, "initial_candidate_gmp_terms": len(GMP_TERMS), "max_sparse_terms": MAX_SPARSE_TERMS, "forward_selection_trial_count": selection_info["trial_count"], "cv_fit_count_estimate": selection_info["cv_fit_count_estimate"], "selected_sparse_term_count": selected_count, "selected_sparse_term_ids": list(selected_order), "completed": True, "B_opened_after_structure_freeze": True, "all425_completed": True})
    (RESULT_ROOT / "final_result_summary.txt").write_text(_summary_text(selected_count, selected_order, sequence, comparison, residuals, validation_summary, {"ridge_math": gates["ridge_math"], "basis": gates["basis_dictionary"], "G0_equivalence": gates["G0_equivalence"], "serial_parallel": gates["serial_parallel"], "CV_leakage": gates["CV_leakage"]}, freeze_info, workers, float(selection_info["elapsed_seconds"]), int(selection_info["cv_fit_count_estimate"]), protection) + "\n", encoding="utf-8")
    timestamp = datetime.now(UTC).isoformat()
    log_body = f"{EXPERIMENT_NAME} completed: 18 GMP dictionary terms, G0-G8 A/C blocked-CV selection, selected G{selected_count} ({list(selected_order)}), 14-state B evaluation and 425-state post-selection validation. Frozen baseline gate={freeze_info['pass']} (max error={freeze_info['max_abs_error_dB']:.3e} dB); serial/parallel={gates['serial_parallel']['pass']}; CV leakage={gates['CV_leakage']['pass']}; raw/protected unchanged={protection['all_protected_unchanged']}. Results: {RESULT_ROOT}"
    for path, title in ((MODEL_LOG, EXPERIMENT_NAME), (HANDOFF_LOG, EXPERIMENT_NAME)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n\n[{timestamp}] {title}\n{log_body}\n")
    print(f"Completed {EXPERIMENT_NAME}: selected G{selected_count} terms={list(selected_order)}", flush=True)
    print(f"Results: {RESULT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
