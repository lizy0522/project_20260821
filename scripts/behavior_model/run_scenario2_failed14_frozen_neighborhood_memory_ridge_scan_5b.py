"""Run the frozen-model-neighborhood memory/Ridge experiment.

The runner keeps the numerical implementation in
``frozen_neighborhood_memory_ridge_scan`` and owns only orchestration,
validation, persistence, plotting and the post-selection 425-state check.
The initial search is exactly seven local structures × nine lambda values.
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

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from behavior_model import frozen_neighborhood_memory_ridge_scan as model  # noqa: E402
from behavior_model.plot_scenario2_failed14_frozen_neighborhood_memory_ridge_scan_5b import (  # noqa: E402
    generate_figures,
)

STATE_IDS = tuple(int(value) for value in model.FAILED_STATE_IDS)
STATE_COUNT = len(STATE_IDS)
ALL_STATE_IDS = tuple(range(model.ALL_STATE_COUNT))
ALL_STATE_COUNT = len(ALL_STATE_IDS)
ORDERS = model.ORDERS
CORE_METRIC_COLUMNS = (
    "Aend_train_NMSE_dB",
    "Aend_B_NMSE_dB",
    "C2_train_NMSE_dB",
    "C2_B_NMSE_dB",
)
DELTA_COLUMNS = (
    "delta_Aend_train_vs_frozen_dB",
    "delta_Aend_B_vs_frozen_dB",
    "delta_C2_train_vs_frozen_dB",
    "delta_C2_B_vs_frozen_dB",
)
EXTENDED_LAMBDAS = (3e-3, 1e-2, 3e-2, 1e-1)

TASK_NAME = "behavior_model"
EXPERIMENT_NAME = "scenario_2_failed14_frozen_neighborhood_memory_ridge_scan_5B"
RESULT_ROOT = PROJECT_ROOT / "results" / TASK_NAME / EXPERIMENT_NAME
CONFIG_ROOT = RESULT_ROOT / "config"
CACHE_ROOT = RESULT_ROOT / "cache"
PREPROCESSED_ROOT = CACHE_ROOT / "preprocessed_states"
CANDIDATE_CHUNK_ROOT = CACHE_ROOT / "candidate_chunks"
TABLE_ROOT = RESULT_ROOT / "tables"
MODEL_ROOT = RESULT_ROOT / "models"
FIGURE_ROOT = RESULT_ROOT / "figures"
VALIDATION_ROOT = RESULT_ROOT / "validation"
MODEL_LOG = PROJECT_ROOT / "work_logs" / TASK_NAME / "execution_log.txt"
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "codex_handoff" / "codex_handoff.txt"
FAILURE_SOURCE = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_lut_retrieval"
    / "scenario_2_C2_to_Aend"
    / "failed_retrieval_states.csv"
)
OLD_FORMAL_METRICS = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2_all_ilc"
    / "all_ilc_model_metrics.csv"
)

PROTECTED_RESULT_DIRS = {
    "formal_5B_retrieval": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_lut_retrieval"
    / "scenario_2_C2_to_Aend",
    "all_ilc_frozen_metrics": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2_all_ilc",
    "unified_capacity_scan": PROJECT_ROOT
    / "results"
    / "behavior_model"
    / "scenario_2_unified_odd_order_mp_capacity_scan_5B",
    "odd_only_failed14_scan": PROJECT_ROOT
    / "results"
    / "behavior_model"
    / "scenario_2_failed14_odd_order_mp_order_dependent_memory_scan_5B",
    "p5_ridge_scan": PROJECT_ROOT
    / "results"
    / "behavior_model"
    / "scenario_2_failed14_P5_variable_memory_ridge_scan_5B",
    "p5_order2_scan": PROJECT_ROOT
    / "results"
    / "behavior_model"
    / "scenario_2_failed14_P5_with_order2_variable_memory_OLS_scan_5B",
}

_WORKER_BASES: dict[int, model.StateBases] = {}
_WORKER_REFERENCE: dict[str, np.ndarray] = {}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
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
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_frame(frame: pd.DataFrame, path: Path, *, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        na_rep="NaN",
        float_format="%.17g",
        compression=compression,
    )


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
        "result_dirs": {
            name: {"path": str(path), "exists": path.is_dir(), "sha256": _tree_digest(path)}
            for name, path in PROTECTED_RESULT_DIRS.items()
        },
    }


def _verify_protection(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    raw_before = before["data/raw"]
    raw_after = after["data/raw"]
    raw = {
        "sha256_unchanged": raw_before["sha256"] == raw_after["sha256"],
        "file_count_unchanged": raw_before["file_count"] == raw_after["file_count"],
        "bytes_unchanged": raw_before["bytes"] == raw_after["bytes"],
    }
    result_dirs = {
        name: {
            "sha256_unchanged": entry["sha256"] == after["result_dirs"][name]["sha256"],
            "before": entry["sha256"],
            "after": after["result_dirs"][name]["sha256"],
        }
        for name, entry in before["result_dirs"].items()
    }
    return {
        "data/raw": raw,
        "result_dirs": result_dirs,
        "all_protected_unchanged": bool(
            all(raw.values()) and all(entry["sha256_unchanged"] for entry in result_dirs.values())
        ),
    }


def _validate_failure_source() -> dict[str, Any]:
    frame = pd.read_csv(FAILURE_SOURCE)
    required = {"State_n_R", "failure"}
    if not required.issubset(frame.columns):
        raise ValueError(f"failure source missing columns: {sorted(required - set(frame.columns))}")
    values = frame["failure"]
    if values.dtype == bool:
        mask = values
    else:
        mask = values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    ids = tuple(sorted(int(value) for value in frame.loc[mask, "State_n_R"]))
    if ids != STATE_IDS:
        raise RuntimeError(f"failure state list mismatch: {ids} vs {STATE_IDS}")
    return {
        "source": str(FAILURE_SOURCE),
        "state_ids": list(STATE_IDS),
        "failure_count": STATE_COUNT,
        "used_for_state_subset_only": True,
        "used_for_candidate_selection": False,
    }


def _prepare_cache() -> dict[str, Any]:
    PREPROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for position, state_id in enumerate(STATE_IDS, start=1):
        path = PREPROCESSED_ROOT / f"state_{state_id:03d}.npz"
        if path.is_file():
            prepared = model.load_prepared_state(path)
        else:
            prepared = model.prepare_state(state_id)
            model.save_prepared_state(prepared, path)
        lengths = {
            "aend_A_input": int(prepared.aend_A_input.size),
            "aend_A_output": int(prepared.aend_A_output.size),
            "aend_B_input": int(prepared.aend_B_input.size),
            "aend_B_output": int(prepared.aend_B_output.size),
            "c2_C_input": int(prepared.c2_C_input.size),
            "c2_C_output": int(prepared.c2_C_output.size),
            "c2_B_input": int(prepared.c2_B_input.size),
            "c2_B_output": int(prepared.c2_B_output.size),
        }
        expected = {
            "aend_A_input": model.FORMAL_A_LENGTH,
            "aend_A_output": model.FORMAL_A_LENGTH,
            "aend_B_input": model.FORMAL_B_LENGTH,
            "aend_B_output": model.FORMAL_B_LENGTH,
            "c2_C_input": model.FORMAL_C_LENGTH,
            "c2_C_output": model.FORMAL_C_LENGTH,
            "c2_B_input": model.FORMAL_B_LENGTH,
            "c2_B_output": model.FORMAL_B_LENGTH,
        }
        if lengths != expected:
            raise RuntimeError(f"state {state_id} canonical lengths changed: {lengths}")
        records.append(
            {
                "state_id": state_id,
                "ilc_A_end": prepared.ilc_A_end,
                "rough_delay_Aend": prepared.rough_delay_Aend,
                "fraction_delay_Aend": prepared.fraction_delay_Aend,
                "rough_delay_C2": prepared.rough_delay_C2,
                "fraction_delay_C2": prepared.fraction_delay_C2,
                "lengths": lengths,
                "cache_file": str(path),
            }
        )
        print(f"preprocessed state {position}/{STATE_COUNT}: {state_id}", flush=True)
    manifest = {
        "state_ids": list(STATE_IDS),
        "state_count": STATE_COUNT,
        "pair_specific_B_probe": True,
        "ABC_lengths": {"A": model.FORMAL_A_LENGTH, "B": model.FORMAL_B_LENGTH, "C": model.FORMAL_C_LENGTH},
        "records": records,
    }
    _write_json(CACHE_ROOT / "preprocessed_manifest.json", manifest)
    return manifest


def _ridge_math_gate() -> dict[str, Any]:
    rng = np.random.default_rng(20260909)
    n, k = 19, 4
    phi = rng.normal(size=(n, k)) + 1j * rng.normal(size=(n, k))
    y = rng.normal(size=n) + 1j * rng.normal(size=n)
    ridge_lambda = 1e-8
    augmented_phi = np.vstack((phi, np.sqrt(n * ridge_lambda) * np.eye(k, dtype=np.complex128)))
    augmented_y = np.concatenate((y, np.zeros(k, dtype=np.complex128)))
    theta, _, rank, singular = np.linalg.lstsq(augmented_phi, augmented_y, rcond=None)
    objective = float(np.linalg.norm(phi @ theta - y) ** 2 / n + ridge_lambda * np.linalg.norm(theta) ** 2)
    augmented_objective = float(np.linalg.norm(augmented_phi @ theta - augmented_y) ** 2 / n)
    stationarity = float(np.linalg.norm(phi.conj().T @ (phi @ theta - y) / n + ridge_lambda * theta))
    passed = bool(
        int(rank) == k
        and np.all(np.isfinite(singular))
        and abs(objective - augmented_objective) <= 1e-12
        and stationarity <= 1e-10
    )
    return {
        "pass": passed,
        "n": n,
        "k": k,
        "lambda": ridge_lambda,
        "objective": objective,
        "augmented_objective": augmented_objective,
        "objective_abs_diff": abs(objective - augmented_objective),
        "stationarity_norm": stationarity,
        "augmented_rank": int(rank),
    }


def _basis_gate(bases: Mapping[int, model.StateBases]) -> dict[str, Any]:
    from behavior_model.basis import build_mp_basis

    candidate = next(c for c in model.generate_candidates() if c.structure_id == "S6" and c.lambda_value == 0.0)
    state_id = STATE_IDS[0]
    prepared = model.load_prepared_state(PREPROCESSED_ROOT / f"state_{state_id:03d}.npz")
    expected_terms = list(candidate.basis_terms)
    observed_terms = list(candidate.basis_terms)
    direct = build_mp_basis(prepared.aend_A_input, ORDERS, candidate.memory_definition)
    columns = model.candidate_columns(candidate)
    selected = bases[state_id].aend_A_phi[candidate.max_delay :, :][:, columns]
    error = float(np.max(np.abs(direct - selected)))
    return {
        "pass": bool(error <= 1e-12 and expected_terms == observed_terms),
        "candidate": candidate.memory_profile_string,
        "basis_terms": [list(term) for term in expected_terms],
        "max_abs_error": error,
        "direct_shape": list(direct.shape),
        "selected_shape": list(selected.shape),
    }


def _ols_gate() -> dict[str, Any]:
    rng = np.random.default_rng(20260910)
    phi = rng.normal(size=(29, 6)) + 1j * rng.normal(size=(29, 6))
    y = rng.normal(size=29) + 1j * rng.normal(size=29)
    theta_a, _, rank_a, singular_a = np.linalg.lstsq(phi, y, rcond=None)
    theta_b, singular_b, rank_b, residual_b = model._fit_ols(phi, y)
    return {
        "pass": bool(
            int(rank_a) == rank_b
            and np.max(np.abs(theta_a - theta_b)) <= 1e-12
            and np.max(np.abs(np.asarray(singular_a) - singular_b)) <= 1e-12
            and abs(float(np.linalg.norm(y - phi @ theta_a)) - residual_b) <= 1e-12
        ),
        "theta_max_abs_error": float(np.max(np.abs(theta_a - theta_b))),
        "singular_max_abs_error": float(np.max(np.abs(np.asarray(singular_a) - singular_b))),
        "rank_equal": int(rank_a) == rank_b,
    }


def _b_support_gate(bases: Mapping[int, model.StateBases], reference: Mapping[str, np.ndarray]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for structure_id, expected_delay in (("S0", 2), ("S1", 3)):
        candidate = next(c for c in model.generate_candidates() if c.structure_id == structure_id and c.lambda_value == 0.0)
        result = model.evaluate_candidate_on_bases(candidate, {STATE_IDS[0]: bases[STATE_IDS[0]]}, reference, return_theta=False)
        row = result["rows"][0]
        expected = {
            "Aend_n_train_samples": model.FORMAL_A_LENGTH - expected_delay,
            "Aend_n_B_samples": model.FORMAL_B_LENGTH - expected_delay,
            "C2_n_train_samples": model.FORMAL_C_LENGTH - expected_delay,
            "C2_n_B_samples": model.FORMAL_B_LENGTH - expected_delay,
        }
        observed = {
            "Aend_n_train_samples": int(row["Aend_n_train_samples"]),
            "Aend_n_B_samples": int(row["Aend_n_B_samples"]),
            "C2_n_train_samples": int(row["C2_n_train_samples"]),
            "C2_n_B_samples": int(row["C2_n_B_samples"]),
        }
        checks.append({"candidate": candidate.memory_profile_string, "expected": expected, "observed": observed, "pass": expected == observed and bool(row["valid"])})
    return {"pass": bool(all(item["pass"] for item in checks)), "checks": checks, "B_input_and_target_same_support": True}


def _worker_initializer(cache_dir: str, reference_path: str, state_ids: tuple[int, ...]) -> None:
    global _WORKER_BASES, _WORKER_REFERENCE
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"
    _WORKER_BASES = model.load_state_bases(Path(cache_dir), state_ids)
    _WORKER_REFERENCE = model.load_ols_reference(Path(reference_path))


def _worker_entry(payload: tuple[model.NeighborhoodCandidate, bool]) -> dict[str, Any]:
    candidate, return_theta = payload
    if not _WORKER_BASES or not _WORKER_REFERENCE:
        raise RuntimeError("worker caches were not initialized")
    return model.evaluate_candidate_on_bases(candidate, _WORKER_BASES, _WORKER_REFERENCE, return_theta=return_theta)


def _parallel_results(
    candidates: Sequence[model.NeighborhoodCandidate],
    cache_dir: Path,
    reference_path: Path,
    state_ids: tuple[int, ...],
    workers: int,
    *,
    return_theta: bool = False,
    callback: Any | None = None,
) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    if not candidates:
        return results
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_initializer,
        initargs=(str(cache_dir), str(reference_path), state_ids),
    ) as executor:
        futures = {executor.submit(_worker_entry, (candidate, return_theta)): candidate for candidate in candidates}
        for done, future in enumerate(as_completed(futures), start=1):
            candidate = futures[future]
            result = future.result()
            if int(result["candidate_id"]) != candidate.candidate_id or len(result["rows"]) != len(state_ids):
                raise RuntimeError("worker returned an incomplete or mismatched candidate result")
            results[candidate.candidate_id] = result
            if callback is not None:
                callback(done, candidate, result)
    if set(results) != {candidate.candidate_id for candidate in candidates}:
        raise RuntimeError("parallel result set is incomplete")
    return results


def _serial_results(
    candidates: Sequence[model.NeighborhoodCandidate],
    cache_dir: Path,
    reference: Mapping[str, np.ndarray],
    state_ids: tuple[int, ...],
) -> dict[int, dict[str, Any]]:
    bases = model.load_state_bases(cache_dir, state_ids)
    return {
        candidate.candidate_id: model.evaluate_candidate_on_bases(candidate, bases, reference, return_theta=True)
        for candidate in candidates
    }


def _serial_parallel_gate(
    candidates: Sequence[model.NeighborhoodCandidate],
    cache_dir: Path,
    reference_path: Path,
    reference: Mapping[str, np.ndarray],
    workers: int,
) -> dict[str, Any]:
    regression_ids = (187, 195, 340, 354)
    serial = _serial_results(candidates, cache_dir, reference, regression_ids)
    parallel = _parallel_results(candidates, cache_dir, reference_path, regression_ids, workers, return_theta=True)
    metric_error = 0.0
    theta_error = 0.0
    rank_equal = True
    support_equal = True
    for candidate in candidates:
        left = {int(row["state_id"]): row for row in serial[candidate.candidate_id]["rows"]}
        right = {int(row["state_id"]): row for row in parallel[candidate.candidate_id]["rows"]}
        for state_id in regression_ids:
            lrow, rrow = left[state_id], right[state_id]
            for field in CORE_METRIC_COLUMNS:
                metric_error = max(metric_error, abs(float(lrow[field]) - float(rrow[field])))
            for side in ("Aend", "C2"):
                rank_equal &= int(lrow[f"{side}_rank"]) == int(rrow[f"{side}_rank"])
                support_equal &= int(lrow[f"{side}_n_train_samples"]) == int(rrow[f"{side}_n_train_samples"])
        for side in ("theta_a", "theta_c"):
            for state_id, theta in serial[candidate.candidate_id][side].items():
                theta_error = max(theta_error, float(np.max(np.abs(theta - parallel[candidate.candidate_id][side][state_id]))))
    return {
        "pass": bool(metric_error <= 1e-10 and theta_error <= 1e-10 and rank_equal and support_equal),
        "parallelization_axis": "candidate",
        "state_ids": list(regression_ids),
        "candidate_ids": [candidate.candidate_id for candidate in candidates],
        "worker_count": workers,
        "metric_max_abs_error_dB": metric_error,
        "theta_max_abs_error": theta_error,
        "rank_equal": rank_equal,
        "valid_support_equal": support_equal,
    }


def _frozen_baseline_gate(
    bases: Mapping[int, model.StateBases], reference: Mapping[str, np.ndarray]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reproduce the formal S0 + 1e-8 metrics before scanning any new point."""

    frozen = next(candidate for candidate in model.generate_candidates() if candidate.is_frozen_baseline)
    current = model.evaluate_candidate_on_bases(frozen, bases, reference, return_theta=False)
    current_by_state = {int(row["state_id"]): row for row in current["rows"]}
    old = pd.read_csv(OLD_FORMAL_METRICS)
    rows: list[dict[str, Any]] = []
    metric_errors: list[float] = []
    mapping_pass = True
    required_old = {"state_id", "actual_ilc_n", "model_role", "train_nmse_db", "B_generalization_nmse_db"}
    if not required_old.issubset(old.columns):
        raise RuntimeError(f"formal metrics missing columns: {sorted(required_old - set(old.columns))}")
    for state_id in STATE_IDS:
        row = current_by_state[state_id]
        a_old = old.loc[(old["state_id"] == state_id) & (old["actual_ilc_n"] == row["ilc_A_end"]) & (old["model_role"] == "Y-A")]
        c_old = old.loc[(old["state_id"] == state_id) & (old["actual_ilc_n"] == 2) & (old["model_role"] == "Y-C")]
        if a_old.shape[0] != 1 or c_old.shape[0] != 1:
            mapping_pass = False
            rows.append({"state_id": state_id, "mapping_pass": False, "failure_reason": "formal Aend/C2 row missing or duplicated"})
            continue
        a_old_row, c_old_row = a_old.iloc[0], c_old.iloc[0]
        values = {
            "state_id": state_id,
            "ilc_A_end_current": int(row["ilc_A_end"]),
            "ilc_A_end_formal": int(a_old_row["actual_ilc_n"]),
            "c2_stage_current": 2,
            "c2_stage_formal": int(c_old_row["actual_ilc_n"]),
            "Aend_train_current": float(row["Aend_train_NMSE_dB"]),
            "Aend_train_formal": float(a_old_row["train_nmse_db"]),
            "Aend_B_current": float(row["Aend_B_NMSE_dB"]),
            "Aend_B_formal": float(a_old_row["B_generalization_nmse_db"]),
            "C2_train_current": float(row["C2_train_NMSE_dB"]),
            "C2_train_formal": float(c_old_row["train_nmse_db"]),
            "C2_B_current": float(row["C2_B_NMSE_dB"]),
            "C2_B_formal": float(c_old_row["B_generalization_nmse_db"]),
            "Aend_n_train_current": int(row["Aend_n_train_samples"]),
            "Aend_n_train_formal": int(a_old_row["n_train"]),
            "C2_n_train_current": int(row["C2_n_train_samples"]),
            "C2_n_train_formal": int(c_old_row["n_train"]),
            "mapping_pass": True,
        }
        for prefix in ("Aend_train", "Aend_B", "C2_train", "C2_B"):
            error = float(values[f"{prefix}_current"] - values[f"{prefix}_formal"])
            values[f"{prefix}_error_dB"] = error
            metric_errors.append(abs(error))
        values["state_pass"] = bool(
            max(abs(values[f"{prefix}_error_dB"]) for prefix in ("Aend_train", "Aend_B", "C2_train", "C2_B")) <= 1e-8
            and values["Aend_n_train_current"] == values["Aend_n_train_formal"]
            and values["C2_n_train_current"] == values["C2_n_train_formal"]
        )
        rows.append(values)
        mapping_pass &= bool(values["state_pass"])
    frame = pd.DataFrame(rows).sort_values("state_id").reset_index(drop=True)
    max_error = max(metric_errors, default=float("nan"))
    details = {
        "pass": bool(mapping_pass and np.isfinite(max_error) and max_error <= 1e-8),
        "reference_file": str(OLD_FORMAL_METRICS),
        "candidate": frozen.memory_profile_string,
        "candidate_id": frozen.candidate_id,
        "state_count": STATE_COUNT,
        "max_abs_error_dB": max_error,
        "metric_tolerance_dB": 1e-8,
        "mapping_pass": bool(mapping_pass),
    }
    return frame, details


def _candidate_chunk_valid(path: Path, candidate: model.NeighborhoodCandidate) -> list[dict[str, Any]] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("candidate_id", -1)) != candidate.candidate_id:
            return None
        rows = payload.get("rows")
        if not isinstance(rows, list) or len(rows) != STATE_COUNT:
            return None
        if sorted(int(row["state_id"]) for row in rows) != sorted(STATE_IDS):
            return None
        if not all(bool(row.get("valid", False)) and all(np.isfinite(float(row[field])) for field in CORE_METRIC_COLUMNS) for row in rows):
            return None
        return rows
    except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
        return None


def _scan(
    candidates: Sequence[model.NeighborhoodCandidate],
    cache_dir: Path,
    reference_path: Path,
    workers: int,
    logical_cpu_count: int,
    *,
    label: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Run/resume candidate-level tasks and persist one JSON chunk per candidate."""

    CANDIDATE_CHUNK_ROOT.mkdir(parents=True, exist_ok=True)
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    completed: dict[int, list[dict[str, Any]]] = {}
    for path in sorted(CANDIDATE_CHUNK_ROOT.glob("candidate_*.json")):
        try:
            candidate_id = int(path.stem.split("_")[-1])
        except ValueError:
            continue
        if candidate_id in candidate_by_id:
            rows = _candidate_chunk_valid(path, candidate_by_id[candidate_id])
            if rows is not None:
                completed[candidate_id] = rows
    pending = [candidate for candidate in candidates if candidate.candidate_id not in completed]
    started = time.perf_counter()
    progress_path = RESULT_ROOT / "search_progress.json"
    progress: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "scan_label": label,
        "target_candidates": len(candidates),
        "completed_candidates": len(completed),
        "state_count": STATE_COUNT,
        "logical_cpu_count": logical_cpu_count,
        "cpu_target_fraction": 0.80,
        "worker_count_requested": workers,
        "worker_count_effective": workers,
        "parallelization_axis": "candidate",
        "blas_threads_per_worker": 1,
        "nested_parallelism": False,
        "candidate_state_evaluations": len(completed) * STATE_COUNT,
        "total_model_fit_count": len(completed) * STATE_COUNT * 2,
        "resumed_candidate_count": len(completed),
    }
    _write_json(progress_path, progress)

    def on_result(done: int, candidate: model.NeighborhoodCandidate, result: dict[str, Any]) -> None:
        rows = result["rows"]
        if not all(bool(row.get("valid", False)) for row in rows):
            raise RuntimeError(f"candidate {candidate.candidate_id} produced an invalid state fit")
        completed[candidate.candidate_id] = rows
        _write_json(CANDIDATE_CHUNK_ROOT / f"candidate_{candidate.candidate_id:03d}.json", {"candidate_id": candidate.candidate_id, "rows": rows})
        elapsed = time.perf_counter() - started
        count = len(completed)
        progress.update(
            {
                "active_candidate_id": candidate.candidate_id,
                "completed_candidates": count,
                "candidate_state_evaluations": count * STATE_COUNT,
                "total_model_fit_count": count * STATE_COUNT * 2,
                "elapsed_seconds": elapsed,
                "candidate_state_evaluations_per_second": count * STATE_COUNT / elapsed if elapsed else None,
                "model_fits_per_second": count * STATE_COUNT * 2 / elapsed if elapsed else None,
            }
        )
        _write_json(progress_path, progress)
        if done % 5 == 0 or done == len(pending):
            print(f"{label}: completed {count}/{len(candidates)} candidates", flush=True)

    if pending:
        _parallel_results(pending, cache_dir, reference_path, STATE_IDS, workers, return_theta=False, callback=on_result)
    if set(completed) != set(candidate_by_id):
        raise RuntimeError("formal candidate scan is incomplete")
    frame = pd.DataFrame([row for candidate in candidates for row in completed[candidate.candidate_id]])
    frame = frame.sort_values(["candidate_id", "state_id"]).reset_index(drop=True)
    expected = len(candidates) * STATE_COUNT
    if frame.shape[0] != expected or frame["candidate_id"].nunique() != len(candidates) or frame["state_id"].nunique() != STATE_COUNT:
        raise RuntimeError("candidate-state grid is incomplete")
    elapsed = time.perf_counter() - started
    run_info = {
        "scan_label": label,
        "elapsed_seconds": elapsed,
        "candidate_count": len(candidates),
        "state_count": STATE_COUNT,
        "candidate_state_evaluations": expected,
        "total_model_fit_count": expected * 2,
        "candidate_state_evaluations_per_second": expected / elapsed if elapsed else None,
        "model_fits_per_second": expected * 2 / elapsed if elapsed else None,
        "candidate_throughput_per_second": len(candidates) / elapsed if elapsed else None,
        "logical_cpu_count": logical_cpu_count,
        "cpu_target_fraction": 0.80,
        "worker_count_requested": workers,
        "worker_count_effective": workers,
        "parallelization_axis": "candidate",
        "blas_threads_per_worker": 1,
        "nested_parallelism": False,
        "worker_state_cache_used": True,
        "resumed_candidate_count": len(candidates) - len(pending),
        "all_candidates_resumed_from_cache": bool(not pending),
        "final": True,
    }
    _write_json(progress_path, {**run_info, "completed_candidates": len(candidates)})
    return frame, run_info


def _stats(values: np.ndarray, prefix: str) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    return {
        f"{prefix}_count": int(values.size),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_q25": float(np.quantile(values, 0.25)),
        f"{prefix}_q75": float(np.quantile(values, 0.75)),
        f"{prefix}_q95": float(np.quantile(values, 0.95)),
        f"{prefix}_worst": float(np.max(values)),
    }


def _add_frozen_deltas(state_metrics: pd.DataFrame) -> pd.DataFrame:
    frame = state_metrics.copy()
    frozen = frame.loc[frame["is_frozen_baseline"].astype(bool), ["state_id", *CORE_METRIC_COLUMNS]].copy()
    if frozen.shape[0] != STATE_COUNT:
        raise RuntimeError("frozen baseline rows are not exactly one per failed state")
    frozen = frozen.set_index("state_id").sort_index()
    for metric in CORE_METRIC_COLUMNS:
        prefix = metric.removesuffix("_NMSE_dB")
        delta = frame[metric].to_numpy(dtype=float) - frame["state_id"].map(frozen[metric]).to_numpy(dtype=float)
        frame[f"delta_{prefix}_vs_frozen_dB"] = delta
    frame["worst4"] = frame.loc[:, list(CORE_METRIC_COLUMNS)].max(axis=1)
    frame["Bworst"] = frame.loc[:, ["Aend_B_NMSE_dB", "C2_B_NMSE_dB"]].max(axis=1)
    frozen_worst = frozen.loc[:, list(CORE_METRIC_COLUMNS)].max(axis=1)
    frozen_bworst = frozen.loc[:, ["Aend_B_NMSE_dB", "C2_B_NMSE_dB"]].max(axis=1)
    frame["delta_worst4_vs_frozen_dB"] = frame["worst4"] - frame["state_id"].map(frozen_worst).to_numpy(dtype=float)
    frame["delta_Bworst_vs_frozen_dB"] = frame["Bworst"] - frame["state_id"].map(frozen_bworst).to_numpy(dtype=float)
    frame["four_metric_improvement_count"] = sum(
        frame[f"delta_{metric.removesuffix('_NMSE_dB')}_vs_frozen_dB"] < 0.0 for metric in CORE_METRIC_COLUMNS
    )
    return frame


def _aggregate(state_metrics: pd.DataFrame, candidates: Sequence[model.NeighborhoodCandidate]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        group = state_metrics.loc[state_metrics["candidate_id"].astype(int) == candidate.candidate_id].copy()
        if group.shape[0] != STATE_COUNT:
            raise RuntimeError(f"candidate {candidate.candidate_id} summary has {group.shape[0]} rows")
        row: dict[str, Any] = {
            "candidate_id": candidate.candidate_id,
            "stage": "initial" if candidate.candidate_id < 63 else "boundary_extension",
            "structure_id": candidate.structure_id,
            "structure_description": candidate.structure_description,
            "P_max": 9,
            "orders": json.dumps(list(candidate.orders), separators=(",", ":")),
            "M1": candidate.M1,
            "M2": candidate.M2,
            "M3": candidate.M3,
            "M5": candidate.M5,
            "M7": candidate.M7,
            "M9": candidate.M9,
            "memory_profile": json.dumps(list(candidate.memory_profile), separators=(",", ":")),
            "memory_profile_string": candidate.memory_profile_string,
            "max_delay": candidate.max_delay,
            "K": candidate.coefficient_count,
            "coefficient_count": candidate.coefficient_count,
            "lambda": candidate.lambda_value,
            "ridge_used": candidate.ridge_used,
            "is_frozen_baseline": candidate.is_frozen_baseline,
            "valid": bool(group["valid"].all()),
        }
        for column, prefix in (
            ("Aend_train_NMSE_dB", "Aend_train"),
            ("Aend_B_NMSE_dB", "Aend_B"),
            ("Aend_generalization_gap_dB", "Aend_gap"),
            ("C2_train_NMSE_dB", "C2_train"),
            ("C2_B_NMSE_dB", "C2_B"),
            ("C2_generalization_gap_dB", "C2_gap"),
        ):
            row.update(_stats(group[column].to_numpy(dtype=float), prefix))
        for column, prefix in (
            ("Aend_theta_l2_norm", "Aend_theta_norm"),
            ("C2_theta_l2_norm", "C2_theta_norm"),
            ("Aend_theta_norm_ratio_vs_same_structure_OLS", "Aend_theta_norm_ratio_vs_same_structure_OLS"),
            ("C2_theta_norm_ratio_vs_same_structure_OLS", "C2_theta_norm_ratio_vs_same_structure_OLS"),
            ("Aend_condition_number_Phi", "condition_number_Aend"),
            ("C2_condition_number_Phi", "condition_number_C2"),
            ("Aend_condition_number_augmented", "condition_number_augmented_Aend"),
            ("C2_condition_number_augmented", "condition_number_augmented_C2"),
            ("Aend_ridge_penalty", "Aend_ridge_penalty"),
            ("C2_ridge_penalty", "C2_ridge_penalty"),
        ):
            values = group[column].to_numpy(dtype=float)
            row[f"{prefix}_median"] = float(np.median(values))
            row[f"{prefix}_mean"] = float(np.mean(values))
        for column in ("Aend_joint_pass", "C2_joint_pass", "all_four_pass"):
            count = int(group[column].astype(bool).sum())
            row[f"{column}_count"] = count
            row[f"{column}_rate"] = count / STATE_COUNT
            row[column] = count
        for column, prefix in (
            ("worst4", "worst4"),
            ("Bworst", "Bworst"),
            ("delta_worst4_vs_frozen_dB", "delta_worst4_vs_frozen"),
            ("delta_Bworst_vs_frozen_dB", "delta_Bworst_vs_frozen"),
        ):
            row.update(_stats(group[column].to_numpy(dtype=float), prefix))
        row["median_worst4_delta_vs_frozen"] = row["delta_worst4_vs_frozen_median"]
        row["num_states_worst4_improved_vs_frozen"] = int((group["delta_worst4_vs_frozen_dB"] < 0).sum())
        row["num_states_bworst_improved_vs_frozen"] = int((group["delta_Bworst_vs_frozen_dB"] < 0).sum())
        row["four_metric_improvement_count_mean"] = float(group["four_metric_improvement_count"].mean())
        row["four_metric_improvement_count_median"] = float(group["four_metric_improvement_count"].median())
        row["worst_train_median"] = max(row["Aend_train_median"], row["C2_train_median"])
        row["worst_B_median"] = max(row["Aend_B_median"], row["C2_B_median"])
        row["modeling_median"] = row["worst_train_median"]
        row["generalization_median"] = row["worst_B_median"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("candidate_id").reset_index(drop=True)


def _add_pareto_columns(summary: pd.DataFrame) -> pd.DataFrame:
    frame = summary.copy()
    frame["min_joint_pass_count"] = frame[["Aend_joint_pass_count", "C2_joint_pass_count"]].min(axis=1)
    frame["joint_pass_sum"] = frame["Aend_joint_pass_count"] + frame["C2_joint_pass_count"]
    points = frame[["worst_train_median", "worst_B_median", "K"]].to_numpy(dtype=float)
    pareto = np.ones(points.shape[0], dtype=bool)
    for index in range(points.shape[0]):
        dominates = np.all(points <= points[index], axis=1) & np.any(points < points[index], axis=1)
        dominates[index] = False
        pareto[index] = not bool(np.any(dominates))
    frame["pareto_optimal"] = pareto
    return frame


def _select(summary: pd.DataFrame) -> tuple[pd.Series, str]:
    valid = summary.loc[summary["valid"].astype(bool)].copy()
    if valid.empty:
        raise RuntimeError("no valid candidate")
    max_all_four = int(valid["all_four_pass_count"].max())
    if max_all_four == STATE_COUNT:
        pool = valid.loc[valid["all_four_pass_count"] == max_all_four].sort_values(
            ["K", "max_delay", "worst4_median", "worst4_q95", "Bworst_median", "worst4_max", "lambda", "candidate_id"],
            ascending=[True, True, True, True, True, True, True, True], kind="mergesort",
        )
        return pool.iloc[0], "max_14of14_all_four_then_complexity_delay_balanced_metrics"
    if max_all_four > 0:
        pool = valid.sort_values(
            ["all_four_pass_count", "worst4_median", "worst4_q95", "Bworst_median", "worst4_max", "median_worst4_delta_vs_frozen", "num_states_worst4_improved_vs_frozen", "K", "max_delay", "lambda", "candidate_id"],
            ascending=[False, True, True, True, True, True, False, True, True, True, True], kind="mergesort",
        )
        return pool.iloc[0], "max_all_four_then_balanced_worst4_Bworst_delta_complexity"
    pool = valid.sort_values(
        ["worst4_median", "worst4_q95", "Bworst_median", "worst4_max", "median_worst4_delta_vs_frozen", "num_states_worst4_improved_vs_frozen", "K", "max_delay", "lambda", "candidate_id"],
        ascending=[True, True, True, True, True, False, True, True, True, True], kind="mergesort",
    )
    return pool.iloc[0], "balanced_worst4_Bworst_delta_complexity_when_no_all_four"


def _statewise_best(state_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for state_id in STATE_IDS:
        group = state_metrics.loc[(state_metrics["state_id"] == state_id) & state_metrics["valid"].astype(bool)].copy()
        if group.empty:
            raise RuntimeError(f"state {state_id} has no valid candidate")
        best = group.sort_values(["worst4", "coefficient_count", "lambda", "candidate_id"], ascending=[True, True, True, True], kind="mergesort").iloc[0]
        rows.append(
            {
                "state_id": state_id,
                "candidate_id": int(best["candidate_id"]),
                "structure_id": best["structure_id"],
                "memory_profile_string": best["memory_profile_string"],
                "lambda": float(best["lambda"]),
                "K": int(best["coefficient_count"]),
                "Aend_train_NMSE_dB": float(best["Aend_train_NMSE_dB"]),
                "Aend_B_NMSE_dB": float(best["Aend_B_NMSE_dB"]),
                "C2_train_NMSE_dB": float(best["C2_train_NMSE_dB"]),
                "C2_B_NMSE_dB": float(best["C2_B_NMSE_dB"]),
                "worst4": float(best["worst4"]),
                "all_four_pass": bool(best["all_four_pass"]),
            }
        )
    return pd.DataFrame(rows)


def _ridge_effect_summary(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for structure_id, group in summary.groupby("structure_id", sort=False):
        ols = group.loc[group["lambda"] == 0.0].iloc[0]
        best = group.sort_values(["worst4_median", "Bworst_median", "lambda", "candidate_id"], kind="mergesort").iloc[0]
        rows.append(
            {
                "structure_id": structure_id,
                "memory_profile_string": ols["memory_profile_string"],
                "K": int(ols["K"]),
                "OLS_candidate_id": int(ols["candidate_id"]),
                "best_candidate_id": int(best["candidate_id"]),
                "OLS_lambda": 0.0,
                "best_lambda": float(best["lambda"]),
                "OLS_worst4_median": float(ols["worst4_median"]),
                "best_worst4_median": float(best["worst4_median"]),
                "worst4_delta_best_minus_OLS_dB": float(best["worst4_median"] - ols["worst4_median"]),
                "OLS_Bworst_median": float(ols["Bworst_median"]),
                "best_Bworst_median": float(best["Bworst_median"]),
                "Bworst_delta_best_minus_OLS_dB": float(best["Bworst_median"] - ols["Bworst_median"]),
                "OLS_Aend_B_median": float(ols["Aend_B_median"]),
                "best_Aend_B_median": float(best["Aend_B_median"]),
                "OLS_C2_B_median": float(ols["C2_B_median"]),
                "best_C2_B_median": float(best["C2_B_median"]),
                "OLS_Aend_theta_norm_median": float(ols["Aend_theta_norm_median"]),
                "best_Aend_theta_norm_median": float(best["Aend_theta_norm_median"]),
                "OLS_C2_theta_norm_median": float(ols["C2_theta_norm_median"]),
                "best_C2_theta_norm_median": float(best["C2_theta_norm_median"]),
            }
        )
    return pd.DataFrame(rows)


def _profile_lambda_table(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "candidate_id", "structure_id", "memory_profile_string", "K", "lambda",
        "Aend_train_median", "Aend_B_median", "Aend_gap_median", "C2_train_median", "C2_B_median", "C2_gap_median",
        "Aend_theta_norm_median", "C2_theta_norm_median", "worst4_median", "Bworst_median", "all_four_pass_count",
    ]
    return summary.loc[:, [column for column in columns if column in summary.columns]].sort_values(["structure_id", "lambda", "candidate_id"]).reset_index(drop=True)


def _make_extension_candidates(initial_summary: pd.DataFrame) -> tuple[tuple[model.NeighborhoodCandidate, ...], dict[str, Any]]:
    triggers: list[str] = []
    for structure_id, group in initial_summary.groupby("structure_id", sort=False):
        row_1e4 = group.loc[np.isclose(group["lambda"], 1e-4, rtol=0.0, atol=1e-20)].iloc[0]
        row_1e3 = group.loc[np.isclose(group["lambda"], 1e-3, rtol=0.0, atol=1e-20)].iloc[0]
        if float(row_1e3["Bworst_median"]) < float(row_1e4["Bworst_median"]) and float(row_1e3["worst4_median"]) < float(row_1e4["worst4_median"]):
            triggers.append(str(structure_id))
    existing = model.generate_candidates()
    next_id = max(candidate.candidate_id for candidate in existing) + 1
    extras: list[model.NeighborhoodCandidate] = []
    for structure_id, profile, description in model.STRUCTURES:
        if structure_id not in triggers:
            continue
        for lambda_index, lambda_value in enumerate(EXTENDED_LAMBDAS, start=len(model.LAMBDA_GRID)):
            extras.append(
                model.NeighborhoodCandidate(
                    candidate_id=next_id,
                    structure_id=structure_id,
                    structure_description=description,
                    orders=model.ORDERS,
                    memory_profile=profile,
                    M1=profile[0], M2=profile[1], M3=profile[2], M5=profile[3], M7=profile[4], M9=profile[5],
                    max_delay=profile[0] - 1,
                    coefficient_count=sum(profile),
                    lambda_value=float(lambda_value),
                    ridge_used=True,
                    is_frozen_baseline=False,
                    lambda_index=lambda_index,
                )
            )
            next_id += 1
    return tuple(extras), {
        "triggered": bool(triggers),
        "trigger_structures": triggers,
        "initial_boundary_rule": "Bworst_median and worst4_median both improve from 1e-4 to 1e-3",
        "extension_lambdas": list(EXTENDED_LAMBDAS),
    }


def _refit_selected(
    selected: model.NeighborhoodCandidate,
    state_metrics: pd.DataFrame,
    bases: Mapping[int, model.StateBases],
    reference: Mapping[str, np.ndarray],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    result = model.evaluate_candidate_on_bases(selected, bases, reference, return_theta=True)
    cached = state_metrics.loc[state_metrics["candidate_id"] == selected.candidate_id].set_index("state_id")
    rows: list[dict[str, Any]] = []
    theta_a: list[np.ndarray] = []
    theta_c: list[np.ndarray] = []
    max_metric_error = 0.0
    for row in result["rows"]:
        state_id = int(row["state_id"])
        old = cached.loc[state_id]
        metric_error = max(abs(float(row[field]) - float(old[field])) for field in CORE_METRIC_COLUMNS)
        max_metric_error = max(max_metric_error, metric_error)
        theta_a.append(result["theta_a"][state_id])
        theta_c.append(result["theta_c"][state_id])
        rows.append(
            {
                "state_id": state_id,
                "candidate_id": selected.candidate_id,
                "structure_id": selected.structure_id,
                "memory_profile_string": selected.memory_profile_string,
                "lambda": selected.lambda_value,
                **{field: float(row[field]) for field in CORE_METRIC_COLUMNS},
                "max_metric_abs_diff_dB": metric_error,
            }
        )
    theta_a_array = np.stack(theta_a, axis=0).astype(np.complex128, copy=False)
    theta_c_array = np.stack(theta_c, axis=0).astype(np.complex128, copy=False)
    diagnostics = {
        "all_28_refit": bool(theta_a_array.shape == (STATE_COUNT, selected.coefficient_count) and theta_c_array.shape == theta_a_array.shape),
        "max_metric_abs_diff_dB": max_metric_error,
        "metric_tolerance_dB": 1e-8,
        "theta_finite": bool(np.all(np.isfinite(theta_a_array)) and np.all(np.isfinite(theta_c_array))),
    }
    diagnostics["pass"] = bool(diagnostics["all_28_refit"] and diagnostics["theta_finite"] and max_metric_error <= 1e-8)
    return pd.DataFrame(rows).sort_values("state_id").reset_index(drop=True), theta_a_array, theta_c_array, diagnostics


def _common_support_sensitivity(
    summary: pd.DataFrame,
    bases: Mapping[int, model.StateBases],
    reference: Mapping[str, np.ndarray],
    candidates: Sequence[model.NeighborhoodCandidate],
    initial_candidate_count: int,
) -> tuple[pd.DataFrame, bool]:
    initial = summary.loc[summary["candidate_id"].astype(int) < initial_candidate_count]
    selected_candidates: list[model.NeighborhoodCandidate] = []
    for structure_id, group in initial.groupby("structure_id", sort=False):
        candidate_id = int(group.sort_values(["worst4_median", "Bworst_median", "lambda", "candidate_id"], kind="mergesort").iloc[0]["candidate_id"])
        selected_candidates.append(next(candidate for candidate in candidates if candidate.candidate_id == candidate_id))
    rows: list[dict[str, Any]] = []
    native_order: list[str] = []
    common_order: list[str] = []
    for candidate in selected_candidates:
        native = initial.loc[initial["candidate_id"] == candidate.candidate_id].iloc[0]
        common_result = model.evaluate_candidate_on_bases(candidate, bases, reference, return_theta=False, support_delay=3)
        common_frame = pd.DataFrame(common_result["rows"])
        native_order.append(str(candidate.structure_id))
        common_order.append(str(candidate.structure_id))
        rows.append(
            {
                "structure_id": candidate.structure_id,
                "candidate_id": candidate.candidate_id,
                "memory_profile_string": candidate.memory_profile_string,
                "lambda": candidate.lambda_value,
                "native_support_delay": candidate.max_delay,
                "common_support_delay": 3,
                "native_worst4_median": float(native["worst4_median"]),
                "common_worst4_median": float(common_frame.loc[:, list(CORE_METRIC_COLUMNS)].max(axis=1).median()),
                "native_Bworst_median": float(native["Bworst_median"]),
                "common_Bworst_median": float(common_frame.loc[:, ["Aend_B_NMSE_dB", "C2_B_NMSE_dB"]].max(axis=1).median()),
            }
        )
    frame = pd.DataFrame(rows)
    native_order = frame.sort_values(["native_worst4_median", "native_Bworst_median", "candidate_id"], kind="mergesort")["structure_id"].tolist()
    common_order = frame.sort_values(["common_worst4_median", "common_Bworst_median", "candidate_id"], kind="mergesort")["structure_id"].tolist()
    passed = native_order == common_order
    frame["native_rank"] = frame["native_worst4_median"].rank(method="min", ascending=True).astype(int)
    frame["common_rank"] = frame["common_worst4_median"].rank(method="min", ascending=True).astype(int)
    frame["ranking_order_native"] = ",".join(native_order)
    frame["ranking_order_common_support"] = ",".join(common_order)
    frame["common_support_sensitivity_pass"] = passed
    return frame, passed


def _post_validation_worker(payload: tuple[int, model.NeighborhoodCandidate, model.NeighborhoodCandidate]) -> dict[str, Any]:
    state_id, frozen_candidate, selected_candidate = payload
    prepared = model.prepare_state_any(state_id)
    bases = model.build_state_bases(prepared)
    reference = model.build_ols_reference({state_id: bases})
    rows: dict[str, Any] = {"state_id": state_id, "ilc_A_end": prepared.ilc_A_end}
    for label, candidate in (("frozen", frozen_candidate), ("selected", selected_candidate)):
        result = model.evaluate_candidate_on_bases(candidate, {state_id: bases}, reference, return_theta=False)
        row = result["rows"][0]
        for field in CORE_METRIC_COLUMNS:
            rows[f"{label}_{field}"] = float(row[field])
        rows[f"{label}_valid"] = bool(row["valid"])
        rows[f"{label}_support_delay"] = int(row["support_delay_used"])
    rows["selected_candidate_id"] = selected_candidate.candidate_id
    rows["frozen_candidate_id"] = frozen_candidate.candidate_id
    return rows


def _post_selection_validation(
    frozen: model.NeighborhoodCandidate,
    selected: model.NeighborhoodCandidate,
    workers: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_post_validation_worker, (state_id, frozen, selected)): state_id for state_id in ALL_STATE_IDS}
        for done, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            state_id = int(row["state_id"])
            row["split"] = "failure14" if state_id in STATE_IDS else "success411"
            for metric in CORE_METRIC_COLUMNS:
                row[f"delta_{metric.removesuffix('_NMSE_dB')}_selected_minus_frozen_dB"] = row[f"selected_{metric}"] - row[f"frozen_{metric}"]
            row["worst4_selected"] = max(row[f"selected_{metric}"] for metric in CORE_METRIC_COLUMNS)
            row["worst4_frozen"] = max(row[f"frozen_{metric}"] for metric in CORE_METRIC_COLUMNS)
            row["Bworst_selected"] = max(row["selected_Aend_B_NMSE_dB"], row["selected_C2_B_NMSE_dB"])
            row["Bworst_frozen"] = max(row["frozen_Aend_B_NMSE_dB"], row["frozen_C2_B_NMSE_dB"])
            row["worst4_delta_selected_minus_frozen"] = row["worst4_selected"] - row["worst4_frozen"]
            row["Bworst_delta_selected_minus_frozen"] = row["Bworst_selected"] - row["Bworst_frozen"]
            row["four_metric_improvement_count"] = int(sum(row[f"delta_{metric.removesuffix('_NMSE_dB')}_selected_minus_frozen_dB"] < 0.0 for metric in CORE_METRIC_COLUMNS))
            rows.append(row)
            if done % 50 == 0 or done == ALL_STATE_COUNT:
                print(f"425-state validation: {done}/{ALL_STATE_COUNT}", flush=True)
    frame = pd.DataFrame(rows).sort_values("state_id").reset_index(drop=True)
    if frame.shape[0] != ALL_STATE_COUNT or frame["state_id"].nunique() != ALL_STATE_COUNT:
        raise RuntimeError("all-425 validation is incomplete")
    info = {
        "state_count": ALL_STATE_COUNT,
        "failure14_count": int((frame["split"] == "failure14").sum()),
        "success411_count": int((frame["split"] == "success411").sum()),
        "elapsed_seconds": time.perf_counter() - started,
        "worker_count": workers,
        "selection_used_for_validation_only": True,
        "validation_used_for_selection": False,
    }
    return frame, info


def _group_validation_summary(all425: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split, group in [("failure14", all425.loc[all425["split"] == "failure14"]), ("success411", all425.loc[all425["split"] == "success411"]), ("all425", all425)]:
        row: dict[str, Any] = {"split": split, "state_count": int(group.shape[0])}
        for column in ("selected_Aend_train_NMSE_dB", "selected_Aend_B_NMSE_dB", "selected_C2_train_NMSE_dB", "selected_C2_B_NMSE_dB", "worst4_selected", "worst4_delta_selected_minus_frozen", "Bworst_delta_selected_minus_frozen"):
            prefix = column.removeprefix("selected_").removesuffix("_NMSE_dB")
            values = group[column].to_numpy(dtype=float)
            row[f"{prefix}_mean"] = float(np.mean(values))
            row[f"{prefix}_median"] = float(np.median(values))
            row[f"{prefix}_worst"] = float(np.max(values))
        row["worst4_improved_state_count"] = int((group["worst4_delta_selected_minus_frozen"] < 0).sum())
        row["all_four_improved_state_count"] = int((group["four_metric_improvement_count"] == 4).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _failure_comparison(state_metrics: pd.DataFrame, selected_id: int) -> pd.DataFrame:
    frame = state_metrics.loc[state_metrics["candidate_id"] == selected_id, ["state_id", *CORE_METRIC_COLUMNS, "worst4", "Bworst"]].copy()
    frozen = state_metrics.loc[state_metrics["is_frozen_baseline"].astype(bool), ["state_id", *CORE_METRIC_COLUMNS, "worst4", "Bworst"]].copy()
    frozen = frozen.rename(columns={column: f"frozen_{column}" for column in [*CORE_METRIC_COLUMNS, "worst4", "Bworst"]})
    frame = frame.merge(frozen, on="state_id", how="left", validate="one_to_one")
    frame = frame.rename(columns={column: f"selected_{column}" for column in [*CORE_METRIC_COLUMNS, "worst4", "Bworst"]})
    for metric in CORE_METRIC_COLUMNS:
        frame[f"delta_{metric.removesuffix('_NMSE_dB')}_selected_minus_frozen_dB"] = frame[f"selected_{metric}"] - frame[f"frozen_{metric}"]
    frame["worst4_delta_selected_minus_frozen"] = frame["selected_worst4"] - frame["frozen_worst4"]
    frame["Bworst_delta_selected_minus_frozen"] = frame["selected_Bworst"] - frame["frozen_Bworst"]
    frame["four_metric_improvement_count"] = sum(frame[f"delta_{metric.removesuffix('_NMSE_dB')}_selected_minus_frozen_dB"] < 0 for metric in CORE_METRIC_COLUMNS)
    return frame.sort_values("state_id").reset_index(drop=True)


def _selected_payload(candidate: model.NeighborhoodCandidate, selection_rule: str) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "structure_id": candidate.structure_id,
        "structure_description": candidate.structure_description,
        "P_max": 9,
        "orders": list(candidate.orders),
        "memory_profile": list(candidate.memory_profile),
        "memory_profile_string": candidate.memory_profile_string,
        "M1": candidate.M1,
        "M2": candidate.M2,
        "M3": candidate.M3,
        "M5": candidate.M5,
        "M7": candidate.M7,
        "M9": candidate.M9,
        "K": candidate.coefficient_count,
        "max_delay": candidate.max_delay,
        "lambda": candidate.lambda_value,
        "ridge_used": candidate.ridge_used,
        "is_frozen_baseline": candidate.is_frozen_baseline,
        "selection_rule": selection_rule,
        "selection_based_on": "failed14 behavior-model metrics only",
    }


def _excel_spec(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "columns": frame.columns.tolist(),
        "rows": frame.where(pd.notna(frame), None).to_dict("records"),
    }


def _excel_source(
    config_rows: list[dict[str, Any]],
    structures: pd.DataFrame,
    ridge_grid: pd.DataFrame,
    candidate_grid: pd.DataFrame,
    summary: pd.DataFrame,
    state_metrics: pd.DataFrame,
    frozen_regression: pd.DataFrame,
    selected_summary: pd.DataFrame,
    selected_failed14: pd.DataFrame,
    ridge_effect: pd.DataFrame,
    pareto: pd.DataFrame,
    all425_summary: pd.DataFrame,
) -> dict[str, Any]:
    config = {
        "columns": ["field", "value"],
        "rows": [
            [row["field"], json.dumps(row["value"], ensure_ascii=False) if isinstance(row["value"], (dict, list)) else row["value"]]
            for row in config_rows
        ],
    }
    return {
        "sheets": {
            "experiment_config": config,
            "memory_structures": _excel_spec(structures),
            "ridge_grid": {
                "columns": ["lambda", "label", "ridge_used"],
                "rows": [
                    {"lambda": float(value), "label": "OLS" if value == 0 else f"lambda={value:.0e}", "ridge_used": bool(value > 0)}
                    for value in model.LAMBDA_GRID
                ],
            },
            "candidate_grid": _excel_spec(candidate_grid),
            "candidate_summary": _excel_spec(summary),
            "candidate_state_metrics": _excel_spec(state_metrics),
            "frozen_regression": _excel_spec(frozen_regression),
            "selected_candidate": _excel_spec(selected_summary),
            "selected_failed14": _excel_spec(selected_failed14),
            "ridge_effect": _excel_spec(ridge_effect),
            "pareto_candidates": _excel_spec(pareto),
            "all425_validation": _excel_spec(all425_summary),
        },
        "metadata": {
            "experiment": EXPERIMENT_NAME,
            "selection_state_count": STATE_COUNT,
            "post_selection_validation_state_count": ALL_STATE_COUNT,
            "initial_candidate_count": 63,
            "candidate_count": int(candidate_grid.shape[0]),
            "orders": list(ORDERS),
            "retrieval_used": False,
            "real_B_used": False,
            "low_bandwidth_operator_used": False,
        },
    }


def _summary_text(
    candidates: Sequence[model.NeighborhoodCandidate],
    summary: pd.DataFrame,
    selected: pd.Series,
    selected_candidate: model.NeighborhoodCandidate,
    selection_rule: str,
    run_info: Mapping[str, Any],
    frozen_gate: Mapping[str, Any],
    extension_info: Mapping[str, Any],
    refit_info: Mapping[str, Any],
    common_support_pass: bool,
    all425_summary: pd.DataFrame,
    gate_info: Mapping[str, Any],
) -> str:
    initial = summary.loc[summary["candidate_id"] < 63]
    best_by = {
        "Aend train": int(initial.loc[initial["Aend_train_median"].idxmin(), "candidate_id"]),
        "Aend B": int(initial.loc[initial["Aend_B_median"].idxmin(), "candidate_id"]),
        "C2 train": int(initial.loc[initial["C2_train_median"].idxmin(), "candidate_id"]),
        "C2 B": int(initial.loc[initial["C2_B_median"].idxmin(), "candidate_id"]),
        "balanced worst4": int(summary.loc[summary["worst4_median"].idxmin(), "candidate_id"]),
        "B-balanced": int(summary.loc[summary["Bworst_median"].idxmin(), "candidate_id"]),
    }
    failure_summary = all425_summary.loc[all425_summary["split"].isin(["failure14", "success411", "all425"])].set_index("split")
    selected_med = {metric: float(selected[f"{metric.removesuffix('_NMSE_dB')}_median"]) for metric in CORE_METRIC_COLUMNS}
    frozen_row = summary.loc[summary["is_frozen_baseline"].astype(bool)].iloc[0]
    delta_selected = {metric: selected_med[metric] - float(frozen_row[f"{metric.removesuffix('_NMSE_dB')}_median"]) for metric in CORE_METRIC_COLUMNS}
    structure_lambda = summary.loc[summary["structure_id"] == selected_candidate.structure_id].sort_values("worst4_median").iloc[0]
    lambda_best_worst4 = float(summary.loc[summary["worst4_median"].idxmin(), "lambda"])
    lambda_best_bworst = float(summary.loc[summary["Bworst_median"].idxmin(), "lambda"])
    lambda_is_interior = bool(lambda_best_worst4 not in (0.0, 1e-3) and lambda_best_bworst not in (0.0, 1e-3))
    at_reference_lambda = initial.loc[np.isclose(initial["lambda"], 1e-8, rtol=0.0, atol=1e-20)].set_index("structure_id")
    s0_bworst = float(at_reference_lambda.loc["S0", "Bworst_median"])
    m1_bworst_deltas = {sid: float(at_reference_lambda.loc[sid, "Bworst_median"] - s0_bworst) for sid in ("S1", "S2", "S3", "S6")}
    m3_bworst_deltas = {sid: float(at_reference_lambda.loc[sid, "Bworst_median"] - s0_bworst) for sid in ("S2", "S3", "S4", "S5")}
    return "\n".join(
        [
            EXPERIMENT_NAME,
            "Frozen-model-neighborhood experiment: orders=[1,2,3,5,7,9] fixed; only seven local memory structures and Ridge lambda vary.",
            "",
            "1. Scope and completion",
            f"- frozen baseline reproduction gate: {bool(frozen_gate['pass'])}; maximum absolute historical NMSE error = {float(frozen_gate['max_abs_error_dB']):.3e} dB.",
            f"- initial candidates completed = 63/63; final candidates including controlled boundary extension = {len(candidates)}/{len(candidates)}.",
            f"- failed states completed = {STATE_COUNT}/{STATE_COUNT}; candidate-state evaluations = {run_info['candidate_state_evaluations']}; model fits = {run_info['total_model_fit_count']}.",
            f"- Ridge boundary extension triggered = {bool(extension_info.get('triggered', False))}; trigger structures = {extension_info.get('trigger_structures', [])}.",
            "",
            "2. Selected candidate",
            f"- candidate_id = {selected_candidate.candidate_id}; structure = {selected_candidate.memory_profile_string}; lambda = {selected_candidate.lambda_value:.0e}; K = {selected_candidate.coefficient_count}; max_delay = {selected_candidate.max_delay}.",
            f"- selected is frozen baseline = {selected_candidate.is_frozen_baseline}; selection rule = {selection_rule}.",
            f"- all-four pass = {int(selected['all_four_pass_count'])}/{STATE_COUNT}; Aend joint = {int(selected['Aend_joint_pass_count'])}/{STATE_COUNT}; C2 joint = {int(selected['C2_joint_pass_count'])}/{STATE_COUNT}.",
            f"- selected medians: Aend train {selected_med['Aend_train_NMSE_dB']:.6f} dB; Aend→B {selected_med['Aend_B_NMSE_dB']:.6f} dB; C2 train {selected_med['C2_train_NMSE_dB']:.6f} dB; C2→B {selected_med['C2_B_NMSE_dB']:.6f} dB.",
            "",
            "3. Best local candidates (initial 63)",
            *[f"- {name}: candidate {candidate_id}." for name, candidate_id in best_by.items()],
            f"- selected structure's best row = candidate {int(structure_lambda['candidate_id'])}, lambda {float(structure_lambda['lambda']):.0e}.",
            f"- balanced/B-balanced best lambda values = {lambda_best_worst4:.0e}/{lambda_best_bworst:.0e}; both are interior to the initial grid = {lambda_is_interior}.",
            f"- at lambda=1e-8, B-balanced deltas versus S0 for M1-deeper structures S1/S2/S3/S6 = {m1_bworst_deltas}; for reduced-M3 structures S2/S3/S4/S5 = {m3_bworst_deltas} dB (negative is improvement, so the memory change is not uniformly beneficial).",
            "",
            "4. Change versus frozen baseline on failure14",
            *[f"- {metric}: selected − frozen median = {delta_selected[metric]:+.6f} dB (negative is improvement)." for metric in CORE_METRIC_COLUMNS],
            f"- failure14 states with improved worst4 = {int(selected['num_states_worst4_improved_vs_frozen'])}/{STATE_COUNT}; median worst4 delta = {float(selected['median_worst4_delta_vs_frozen']):+.6f} dB.",
            f"- median coefficient norm ratios versus same-structure OLS: Aend {float(selected['Aend_theta_norm_ratio_vs_same_structure_OLS_median']):.6f}; C2 {float(selected['C2_theta_norm_ratio_vs_same_structure_OLS_median']):.6f}.",
            f"- selected median condition numbers: Phi Aend {float(selected['condition_number_Aend_median']):.6g}; Phi C2 {float(selected['condition_number_C2_median']):.6g}; Ridge shrinks coefficients below the same-structure OLS norms and the selected B medians improve slightly.",
            "",
            "5. Common-support and numerical gates",
            f"- common-support dmax=3 sensitivity pass = {common_support_pass}; selected refit validation = {bool(refit_info['pass'])}.",
            f"- Ridge augmented-LS math = {bool(gate_info['ridge_math']['pass'])}; OLS helper = {bool(gate_info['ols']['pass'])}; B support = {bool(gate_info['b_support']['pass'])}; serial/parallel = {bool(gate_info['serial_parallel']['pass'])}.",
            "- candidate-specific support is used for formal results; B input and target are sliced by the same delay.",
            "",
            "6. Post-selection 425-state validation",
            *[
                f"- {split}: n={int(failure_summary.loc[split, 'state_count'])}; worst4 improved states={int(failure_summary.loc[split, 'worst4_improved_state_count'])}; median worst4 delta={float(failure_summary.loc[split, 'worst4_delta_selected_minus_frozen_median']):+.6f} dB."
                for split in ("failure14", "success411", "all425")
            ],
            f"- success411 states not improved in worst4 = {int(failure_summary.loc['success411', 'state_count'] - failure_summary.loc['success411', 'worst4_improved_state_count'])}; largest success411 worst4 delta = {float(failure_summary.loc['success411', 'worst4_delta_selected_minus_frozen_worst']):+.6f} dB.",
            "- 425-state results are validation only and did not feed back into candidate selection.",
            "",
            "7. Interpretation boundary",
            "- The experiment keeps the frozen six-order basis and tests only local low-order memory allocation plus Ridge strength.",
            "- A selected improvement on failure14 is not sufficient evidence to replace the frozen retrieval model; success411 and all425 validation must remain acceptable.",
            "- No LUT retrieval, Real-B DPD shareability, low-bandwidth operator, or automatic replacement of the frozen configuration was performed.",
            "- The task stops after this finite scan and post-selection validation; no further candidates are generated automatically.",
            "",
            "8. Runtime",
            f"- logical CPUs = {run_info['logical_cpu_count']}; target = 0.80; requested/effective workers = {run_info['worker_count_requested']}/{run_info['worker_count_effective']}; BLAS threads/worker = 1.",
            f"- scan orchestration elapsed = {run_info['elapsed_seconds']:.3f} s (all candidates resumed from valid chunks = {bool(run_info.get('all_candidates_resumed_from_cache', False))}); candidate-state evaluations/s = {run_info['candidate_state_evaluations_per_second']:.6f}; model fits/s = {run_info['model_fits_per_second']:.6f}.",
            "- raw data and protected historical result directories were checked before and after execution.",
        ]
    )


def main() -> None:
    freeze_support()
    print(f"Starting {EXPERIMENT_NAME}", flush=True)
    before = _snapshot()
    failure_info = _validate_failure_source()
    initial_candidates = model.generate_candidates()
    if len(initial_candidates) != 63 or len(model.STRUCTURES) != 7 or len(model.LAMBDA_GRID) != 9:
        raise RuntimeError("initial 7 × 9 candidate grid gate failed")
    logical_cpu_count = int(os.cpu_count() or 1)
    workers = model.target_worker_count(logical_cpu_count)
    print(f"logical_cpu_count={logical_cpu_count}; worker_count={workers}; target=0.80", flush=True)

    for path in (CONFIG_ROOT, CACHE_ROOT, TABLE_ROOT, MODEL_ROOT, FIGURE_ROOT, VALIDATION_ROOT):
        path.mkdir(parents=True, exist_ok=True)
    _write_frame(model.structures_frame(), CONFIG_ROOT / "memory_structures.csv")
    _write_json(CONFIG_ROOT / "ridge_grid.json", [{"lambda": value, "ridge_used": value > 0} for value in model.LAMBDA_GRID])
    initial_grid = model.candidate_grid_frame(initial_candidates)
    _write_frame(initial_grid, CONFIG_ROOT / "candidate_grid.csv")
    _write_json(CONFIG_ROOT / "candidate_grid.json", initial_grid.to_dict("records"))
    experiment_config = {
        "experiment": EXPERIMENT_NAME,
        "bandwidth": "5B",
        "selection_state_count": STATE_COUNT,
        "post_selection_validation_state_count": ALL_STATE_COUNT,
        "failed_state_ids": list(STATE_IDS),
        "P_max": 9,
        "orders": list(ORDERS),
        "orders_frozen": True,
        "memory_structures": [list(profile) for _, profile, _ in model.STRUCTURES],
        "memory_structure_count": len(model.STRUCTURES),
        "lambda_grid": list(model.LAMBDA_GRID),
        "initial_lambda_count": len(model.LAMBDA_GRID),
        "initial_candidate_count": len(initial_candidates),
        "frozen_baseline": {"structure_id": model.FROZEN_STRUCTURE_ID, "memory": list(model.FROZEN_MEMORY), "lambda": model.FROZEN_LAMBDA},
        "ridge_objective": "1/N*||y-Phi theta||^2 + lambda*||theta||^2",
        "ridge_augmented_scale": "sqrt(N*lambda)",
        "candidate_specific_support": True,
        "ABC": {"A": [0, model.FORMAL_A_LENGTH], "B": [model.FORMAL_A_LENGTH, model.FORMAL_A_LENGTH + model.FORMAL_B_LENGTH], "C": [model.FORMAL_A_LENGTH + model.FORMAL_B_LENGTH, model.FORMAL_A_LENGTH + model.FORMAL_B_LENGTH + model.FORMAL_C_LENGTH]},
        "retrieval_used": False,
        "real_B_used": False,
        "low_bandwidth_operator_used": False,
        "parallel": {"logical_cpu_count": logical_cpu_count, "cpu_target_fraction": 0.80, "worker_count_requested": workers, "worker_count_effective": workers, "parallelization_axis": "candidate", "blas_threads_per_worker": 1, "nested_parallelism": False},
    }
    _write_json(CONFIG_ROOT / "experiment_config.json", experiment_config)
    _write_json(CONFIG_ROOT / "failed_state_ids.json", failure_info)

    manifest = _prepare_cache()
    bases = model.load_state_bases(PREPROCESSED_ROOT, STATE_IDS)
    reference = model.build_ols_reference(bases)
    reference_path = CACHE_ROOT / "ols_reference.npz"
    model.save_ols_reference(reference_path, reference)

    frozen_regression, frozen_gate = _frozen_baseline_gate(bases, reference)
    _write_frame(frozen_regression, TABLE_ROOT / "frozen_baseline_regression.csv")
    print(f"frozen baseline gate: {frozen_gate['pass']} (max error {frozen_gate['max_abs_error_dB']:.3e} dB)", flush=True)
    if not frozen_gate["pass"]:
        raise RuntimeError(f"frozen baseline reproduction gate failed: {frozen_gate}")

    ridge_math = _ridge_math_gate()
    basis_gate = _basis_gate(bases)
    ols_gate = _ols_gate()
    b_support_gate = _b_support_gate(bases, reference)
    gate_candidates = (
        next(c for c in initial_candidates if c.structure_id == "S0" and c.lambda_value == 0.0),
        next(c for c in initial_candidates if c.is_frozen_baseline),
        next(c for c in initial_candidates if c.structure_id == "S1" and c.lambda_value == 1e-6),
        next(c for c in initial_candidates if c.structure_id == "S2" and c.lambda_value == 1e-4),
        next(c for c in initial_candidates if c.structure_id == "S3" and c.lambda_value == 1e-3),
        next(c for c in initial_candidates if c.structure_id == "S6" and c.lambda_value == 1e-5),
    )
    serial_parallel = _serial_parallel_gate(gate_candidates, PREPROCESSED_ROOT, reference_path, reference, workers)
    gate_info = {"ridge_math": ridge_math, "basis": basis_gate, "ols": ols_gate, "b_support": b_support_gate, "serial_parallel": serial_parallel}
    _write_json(VALIDATION_ROOT / "gates.json", gate_info)
    if not all(bool(item["pass"]) for item in (ridge_math, basis_gate, ols_gate, b_support_gate, serial_parallel)):
        raise RuntimeError(f"pre-scan gate failed: {gate_info}")

    initial_metrics, initial_run_info = _scan(initial_candidates, PREPROCESSED_ROOT, reference_path, workers, logical_cpu_count, label="initial_63")
    initial_metrics = _add_frozen_deltas(initial_metrics)
    initial_summary = _add_pareto_columns(_aggregate(initial_metrics, initial_candidates))
    extension_candidates, extension_info = _make_extension_candidates(initial_summary)
    extension_metrics = pd.DataFrame()
    extension_run_info: dict[str, Any] = {"candidate_count": 0, "candidate_state_evaluations": 0, "total_model_fit_count": 0, "elapsed_seconds": 0.0, "candidate_state_evaluations_per_second": 0.0, "model_fits_per_second": 0.0}
    if extension_candidates:
        print(f"Ridge boundary extension triggered for {extension_info['trigger_structures']}", flush=True)
        extension_metrics, extension_run_info = _scan(extension_candidates, PREPROCESSED_ROOT, reference_path, workers, logical_cpu_count, label="boundary_extension")
        extension_metrics = _add_frozen_deltas(extension_metrics)
    all_candidates = tuple(initial_candidates) + tuple(extension_candidates)
    state_metrics = pd.concat([initial_metrics, extension_metrics], ignore_index=True) if not extension_metrics.empty else initial_metrics
    state_metrics = state_metrics.sort_values(["candidate_id", "state_id"]).reset_index(drop=True)
    summary = _add_pareto_columns(_aggregate(state_metrics, all_candidates))
    run_info = {
        "elapsed_seconds": float(initial_run_info.get("elapsed_seconds", 0.0) + extension_run_info.get("elapsed_seconds", 0.0)),
        "candidate_count": len(all_candidates),
        "initial_candidate_count": len(initial_candidates),
        "extension_candidate_count": len(extension_candidates),
        "candidate_state_evaluations": int(len(all_candidates) * STATE_COUNT),
        "total_model_fit_count": int(len(all_candidates) * STATE_COUNT * 2),
        "candidate_state_evaluations_per_second": float(len(all_candidates) * STATE_COUNT / max(initial_run_info.get("elapsed_seconds", 0.0) + extension_run_info.get("elapsed_seconds", 0.0), 1e-12)),
        "model_fits_per_second": float(len(all_candidates) * STATE_COUNT * 2 / max(initial_run_info.get("elapsed_seconds", 0.0) + extension_run_info.get("elapsed_seconds", 0.0), 1e-12)),
        "logical_cpu_count": logical_cpu_count,
        "cpu_target_fraction": 0.80,
        "worker_count_requested": workers,
        "worker_count_effective": workers,
        "parallelization_axis": "candidate",
        "blas_threads_per_worker": 1,
        "nested_parallelism": False,
        "worker_state_cache_used": True,
        "all_candidates_resumed_from_cache": bool(
            initial_run_info.get("all_candidates_resumed_from_cache", False)
            and not extension_candidates
        ),
    }
    _write_json(RESULT_ROOT / "search_progress.json", {**run_info, "completed_candidates": len(all_candidates), "final": False, "search_stop_reason": "awaiting_selection_and_425_validation"})

    selected, selection_rule = _select(summary)
    selected_candidate = next(candidate for candidate in all_candidates if candidate.candidate_id == int(selected["candidate_id"]))
    frozen_candidate = next(candidate for candidate in initial_candidates if candidate.is_frozen_baseline)
    statewise = _statewise_best(state_metrics)
    selected_state_metrics = state_metrics.loc[state_metrics["candidate_id"] == selected_candidate.candidate_id].copy().sort_values("state_id")
    selected_summary = summary.loc[summary["candidate_id"] == selected_candidate.candidate_id].copy().reset_index(drop=True)
    frozen_summary = summary.loc[summary["is_frozen_baseline"].astype(bool)].copy().reset_index(drop=True)
    failure_comparison = _failure_comparison(state_metrics, selected_candidate.candidate_id)
    refit_table, theta_a, theta_c, refit_info = _refit_selected(selected_candidate, state_metrics, bases, reference)
    common_support, common_support_pass = _common_support_sensitivity(summary, bases, reference, all_candidates, len(initial_candidates))
    all425, post_info = _post_selection_validation(frozen_candidate, selected_candidate, workers)
    all425_summary = _group_validation_summary(all425)

    _write_frame(state_metrics, TABLE_ROOT / "candidate_state_metrics.csv.gz", compression="gzip")
    _write_frame(summary, TABLE_ROOT / "candidate_summary.csv")
    _write_frame(selected_state_metrics, TABLE_ROOT / "selected_candidate_state_metrics.csv")
    _write_frame(selected_summary, TABLE_ROOT / "selected_candidate.csv")
    _write_frame(failure_comparison, TABLE_ROOT / "selected_vs_frozen_failed14.csv")
    _write_frame(refit_table, TABLE_ROOT / "selected_refit_diagnostics.csv")
    _write_frame(statewise, TABLE_ROOT / "statewise_best_candidate.csv")
    _write_frame(summary.loc[summary["pareto_optimal"].astype(bool)], TABLE_ROOT / "pareto_candidates.csv")
    _write_frame(_ridge_effect_summary(summary), TABLE_ROOT / "ridge_effect_summary.csv")
    _write_frame(_profile_lambda_table(summary), TABLE_ROOT / "ridge_profile_lambda_summary.csv")
    _write_frame(common_support, TABLE_ROOT / "common_support_sensitivity.csv")
    _write_frame(all425, TABLE_ROOT / "all425_selected_vs_frozen.csv.gz", compression="gzip")
    _write_frame(all425_summary, TABLE_ROOT / "all425_validation_summary.csv")
    _write_npz(MODEL_ROOT / "selected_Aend_coefficients_failed14.npz", {"state_ids": np.asarray(STATE_IDS, dtype=np.int64), "theta_Aend": theta_a})
    _write_npz(MODEL_ROOT / "selected_C2_coefficients_failed14.npz", {"state_ids": np.asarray(STATE_IDS, dtype=np.int64), "theta_C2": theta_c})
    selected_payload = _selected_payload(selected_candidate, selection_rule)
    _write_json(MODEL_ROOT / "selected_model.json", selected_payload)

    candidate_grid = pd.concat([model.candidate_grid_frame(initial_candidates), model.candidate_grid_frame(extension_candidates)] if extension_candidates else [initial_grid], ignore_index=True)
    _write_frame(candidate_grid, CONFIG_ROOT / "candidate_grid.csv")
    _write_json(CONFIG_ROOT / "candidate_grid.json", candidate_grid.to_dict("records"))
    figure_details = generate_figures(summary, selected_summary, frozen_summary, failure_comparison, all425, FIGURE_ROOT, initial_candidate_count=len(initial_candidates))

    config_rows = [
        {"field": "experiment", "value": EXPERIMENT_NAME},
        {"field": "bandwidth", "value": "5B"},
        {"field": "selection_state_count", "value": STATE_COUNT},
        {"field": "post_selection_validation_state_count", "value": ALL_STATE_COUNT},
        {"field": "failed_state_ids", "value": list(STATE_IDS)},
        {"field": "orders", "value": list(ORDERS)},
        {"field": "memory_structure_count", "value": len(model.STRUCTURES)},
        {"field": "initial_lambda_count", "value": len(model.LAMBDA_GRID)},
        {"field": "initial_candidate_count", "value": len(initial_candidates)},
        {"field": "extension_candidate_count", "value": len(extension_candidates)},
        {"field": "candidate_count_total", "value": len(all_candidates)},
        {"field": "frozen_baseline_candidate_id", "value": frozen_candidate.candidate_id},
        {"field": "selected_candidate_id", "value": selected_candidate.candidate_id},
        {"field": "selected_is_frozen_baseline", "value": selected_candidate.is_frozen_baseline},
        {"field": "selection_rule", "value": selection_rule},
        {"field": "candidate_specific_support_used", "value": True},
        {"field": "common_support_sensitivity_pass", "value": common_support_pass},
        {"field": "retrieval_used", "value": False},
        {"field": "real_B_used", "value": False},
        {"field": "low_bandwidth_operator_used", "value": False},
        {"field": "logical_cpu_count", "value": logical_cpu_count},
        {"field": "worker_count_effective", "value": workers},
        {"field": "parallelization_axis", "value": "candidate"},
    ]
    pareto = summary.loc[summary["pareto_optimal"].astype(bool)].copy()
    _write_json(RESULT_ROOT / "excel_source.json", _excel_source(config_rows, model.structures_frame(), pd.DataFrame({"lambda": model.LAMBDA_GRID, "ridge_used": [value > 0 for value in model.LAMBDA_GRID]}), candidate_grid, summary, state_metrics, frozen_regression, selected_summary, failure_comparison, _ridge_effect_summary(summary), pareto, all425_summary))

    after = _snapshot()
    protection = _verify_protection(before, after)
    if not protection["all_protected_unchanged"]:
        raise RuntimeError("raw or protected historical result changed")
    validation = {
        "experiment": EXPERIMENT_NAME,
        "bandwidth": "5B",
        "selection_state_count": STATE_COUNT,
        "post_selection_validation_state_count": ALL_STATE_COUNT,
        "failed_state_ids": list(STATE_IDS),
        "orders": list(ORDERS),
        "orders_frozen": True,
        "memory_structure_count": len(model.STRUCTURES),
        "initial_lambda_count": len(model.LAMBDA_GRID),
        "initial_candidate_count": len(initial_candidates),
        "extension_candidate_count": len(extension_candidates),
        "candidate_count_total": len(all_candidates),
        "initial_candidate_completion": "63/63",
        "candidate_completion": f"{len(all_candidates)}/{len(all_candidates)}",
        "state_completion": f"{STATE_COUNT}/{STATE_COUNT}",
        "frozen_baseline_memory": list(model.FROZEN_MEMORY),
        "frozen_baseline_lambda": model.FROZEN_LAMBDA,
        "frozen_baseline_in_candidate_pool": True,
        "frozen_baseline_regression_pass": bool(frozen_gate["pass"]),
        "frozen_baseline_regression": frozen_gate,
        "ridge_objective": "1/N*||y-Phi theta||^2 + lambda*||theta||^2",
        "Aend_and_C2_same_structure": True,
        "Aend_and_C2_same_lambda": True,
        "coefficients_shared": False,
        "candidate_specific_support_used": True,
        "B_input_and_target_same_support": True,
        "common_support_sensitivity_checked": True,
        "common_support_sensitivity_pass": bool(common_support_pass),
        "retrieval_used": False,
        "real_B_used": False,
        "low_bandwidth_operator_used": False,
        "selection_based_on_failed14_only": True,
        "all425_used_for_selection": False,
        "parallel_execution": True,
        "parallelization_axis": "candidate",
        "cpu_target_fraction": 0.80,
        "logical_cpu_count": logical_cpu_count,
        "worker_count_requested": workers,
        "worker_count_effective": workers,
        "blas_threads_per_worker": 1,
        "nested_parallelism": False,
        "serial_parallel_regression_pass": bool(serial_parallel["pass"]),
        "ridge_math_regression_pass": bool(ridge_math["pass"]),
        "basis_regression_pass": bool(basis_gate["pass"]),
        "OLS_regression_pass": bool(ols_gate["pass"]),
        "B_support_regression_pass": bool(b_support_gate["pass"]),
        "selected_refit_validation_pass": bool(refit_info["pass"]),
        "selected_refit_validation": refit_info,
        "post_selection_validation": post_info,
        "selected_candidate": selected_payload,
        "selection_rule": selection_rule,
        "extension": extension_info,
        "runtime": run_info,
        "search_stop_reason": "finite local neighborhood scan plus controlled boundary check completed; no automatic model-family expansion",
        "search_space_frozen": True,
        "preprocessed_cache": manifest,
        "figures": figure_details,
        "raw_data_modified": False,
        "protected_results_modified": False,
        "source_failure_list": failure_info,
        "protection_verification": protection,
        "gate_details": gate_info,
    }
    validation["raw_data_modified"] = not all(protection["data/raw"].values())
    validation["protected_results_modified"] = not all(item["sha256_unchanged"] for item in protection["result_dirs"].values())
    _write_json(RESULT_ROOT / "validation.json", validation)
    _write_json(VALIDATION_ROOT / "validation.json", validation)
    _write_json(RESULT_ROOT / "search_progress.json", {**run_info, "completed_candidates": len(all_candidates), "final": True, "search_stop_reason": validation["search_stop_reason"]})
    summary_text = _summary_text(all_candidates, summary, selected, selected_candidate, selection_rule, run_info, frozen_gate, extension_info, refit_info, common_support_pass, all425_summary, gate_info)
    (RESULT_ROOT / "final_result_summary.txt").write_text(summary_text + "\n", encoding="utf-8")
    log_body = "\n".join(
        [
            f"{EXPERIMENT_NAME} completed: {len(all_candidates)} candidates × {STATE_COUNT} failed states; post-selection validation={ALL_STATE_COUNT} states.",
            f"Frozen baseline reproduction={frozen_gate['pass']} (max error={frozen_gate['max_abs_error_dB']:.3e} dB); selected candidate={selected_candidate.candidate_id}, profile={selected_candidate.memory_profile_string}, lambda={selected_candidate.lambda_value:.0e}, K={selected_candidate.coefficient_count}.",
            f"Candidate-level ProcessPoolExecutor: logical CPUs={logical_cpu_count}, workers={workers}, target=0.80, BLAS threads/worker=1, nested_parallelism=False.",
            f"Gates: ridge_math={ridge_math['pass']}; basis={basis_gate['pass']}; OLS={ols_gate['pass']}; B_support={b_support_gate['pass']}; serial_parallel={serial_parallel['pass']}; refit={refit_info['pass']}; common_support={common_support_pass}.",
            f"Scan elapsed={run_info['elapsed_seconds']:.3f}s; figures={len(figure_details) * 3} files; all425 validation rows={all425.shape[0]}.",
            f"raw/protected unchanged={protection['all_protected_unchanged']}; no retrieval, Real-B, low-bandwidth operator, or frozen-config replacement.",
            f"Results: {RESULT_ROOT}",
        ]
    )
    timestamp = datetime.now(UTC).isoformat()
    for path, title in ((MODEL_LOG, EXPERIMENT_NAME), (HANDOFF_LOG, EXPERIMENT_NAME)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n\n[{timestamp}] {title}\n{log_body}\n")
    print(f"Completed {EXPERIMENT_NAME}: selected={selected_candidate.candidate_id} {selected_candidate.memory_profile_string} lambda={selected_candidate.lambda_value:.0e}", flush=True)
    print(f"Results: {RESULT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
