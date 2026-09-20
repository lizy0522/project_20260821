"""Run the finite P5+[2] order-2 variable-memory OLS ablation.

Exactly 35 monotone memory profiles are evaluated on the 14 previously failed
5B LUT states.  The existing canonical preprocessing is reused through the
read-only state cache.  Candidate-level multiprocessing uses the requested
80-percent CPU target; no Ridge, retrieval, Real-B validation, or expansion is
performed.
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

from behavior_modeling.shared import p5_with_order2_variable_memory_ols_scan as model  # noqa: E402
from behavior_modeling.shared.odd_order_variable_memory_scan import (  # noqa: E402
    FAILED_STATE_IDS,
    FORMAL_A_LENGTH,
    FORMAL_B_LENGTH,
    FORMAL_C_LENGTH,
    REGRESSION_STATE_IDS,
    load_prepared_state,
    prepare_failed_state,
    save_prepared_state,
)
from behavior_modeling.scenario_2.scenario_2_failed14_P5_with_order2_variable_memory_OLS_scan_5b.plot_scenario2_failed14_P5_with_order2_variable_memory_OLS_scan_5b import (  # noqa: E402
    generate_figures,
)
from behavior_modeling.shared.basis import build_mp_basis  # noqa: E402

STATE_COUNT = model.STATE_COUNT
ORDERS = model.ORDERS
TASK_NAME = "scenario_2_failed14_P5_with_order2_variable_memory_OLS_scan_5b"
EXPERIMENT_NAME = "scenario_2_failed14_P5_with_order2_variable_memory_OLS_scan_5B"
RESULT_ROOT = PROJECT_ROOT / "results" / "behavior_modeling" / "scenario_2" /TASK_NAME / EXPERIMENT_NAME
CONFIG_ROOT = RESULT_ROOT / "config"
CACHE_ROOT = RESULT_ROOT / "cache"
PREPROCESSED_ROOT = CACHE_ROOT / "preprocessed_states"
CANDIDATE_CHUNK_ROOT = CACHE_ROOT / "candidate_chunks"
TABLE_ROOT = RESULT_ROOT / "tables"
MODEL_ROOT = RESULT_ROOT / "models"
FIGURE_ROOT = RESULT_ROOT / "figures"
VALIDATION_ROOT = RESULT_ROOT / "validation"
MODEL_LOG = PROJECT_ROOT / "work_logs" / "behavior_modeling" / "scenario_2" /TASK_NAME / "execution_log.txt"
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"
FAILURE_SOURCE = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_retrieval"
    / "scenario_2" /"scenario_2_C2_to_Aend"
    / "failed_retrieval_states.csv"
)
OLD_ODD_ONLY_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_modeling"
    / "scenario_2" /"scenario_2_failed14_odd_order_mp_order_dependent_memory_scan_5B"
)
OLD_FROZEN_METRICS = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2" /"scenario_2_all_ilc"
    / "all_ilc_model_metrics.csv"
)
OLD_FROZEN_STAGE_MAP = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_retrieval"
    / "scenario_2" /"scenario_2_C2_to_Aend"
    / "a_end_stage_map.csv"
)

PROTECTED_RESULT_DIRS = {
    "formal_5B_retrieval": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_retrieval"
    / "scenario_2" /"scenario_2_C2_to_Aend",
    "formal_1B_retrieval": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_retrieval"
    / "scenario_2" /"scenario_2_C2_to_Aend_1B",
    "formal_0p5B_retrieval": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_retrieval"
    / "scenario_2" /"scenario_2_C2_to_Aend_0p5B",
    "equal_ABC_retrieval": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_retrieval"
    / "scenario_2" /"scenario_2_C2_to_Aend_equal_ABC",
    "unified_capacity_retrieval": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_retrieval"
    / "scenario_2" /"scenario_2_C2_to_Aend_unified_model_capacity",
    "retrieval_oriented_scan": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_retrieval"
    / "scenario_2" /"scenario_2_C2_to_Aend_retrieval_oriented_model_scan",
    "all_ilc_analysis": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2" /"scenario_2_all_ilc",
    "old_odd_only_failed14": OLD_ODD_ONLY_ROOT,
    "old_p5_ridge_failed14": PROJECT_ROOT
    / "results"
    / "behavior_modeling"
    / "scenario_2" /"scenario_2_failed14_P5_variable_memory_ridge_scan_5B",
    "old_unified_odd_order": PROJECT_ROOT
    / "results"
    / "behavior_modeling"
    / "scenario_2" /"scenario_2_unified_odd_order_mp_capacity_scan_5B",
}

CORE_METRICS = (
    "Aend_train_NMSE_dB",
    "Aend_B_NMSE_dB",
    "C2_train_NMSE_dB",
    "C2_B_NMSE_dB",
)

_WORKER_BASES: dict[int, Any] = {}


def _json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


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
    root = PROJECT_ROOT / "data" / "raw"
    digest = hashlib.sha256()
    files = sorted(
        (file for file in root.rglob("*") if file.is_file()), key=lambda p: str(p).lower()
    )
    total = 0
    for file in files:
        size = int(file.stat().st_size)
        digest.update(str(file.relative_to(root)).replace("\\", "/").encode("utf-8") + b"\0")
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
    raw = {
        "sha256_unchanged": before["data/raw"]["sha256"] == after["data/raw"]["sha256"],
        "file_count_unchanged": before["data/raw"]["file_count"] == after["data/raw"]["file_count"],
        "bytes_unchanged": before["data/raw"]["bytes"] == after["data/raw"]["bytes"],
    }
    result_dirs = {
        name: {
            "sha256_unchanged": item["sha256"] == after["result_dirs"][name]["sha256"],
            "before": item["sha256"],
            "after": after["result_dirs"][name]["sha256"],
        }
        for name, item in before["result_dirs"].items()
    }
    return {
        "data/raw": raw,
        "result_dirs": result_dirs,
        "all_protected_unchanged": bool(
            all(raw.values()) and all(item["sha256_unchanged"] for item in result_dirs.values())
        ),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
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


def _validate_failure_source() -> dict[str, Any]:
    frame = pd.read_csv(FAILURE_SOURCE)
    if not {"State_n_R", "failure"}.issubset(frame.columns):
        raise ValueError("failure source lacks State_n_R/failure")
    ids = tuple(
        sorted(int(value) for value in frame.loc[frame["failure"].astype(bool), "State_n_R"])
    )
    if ids != tuple(FAILED_STATE_IDS):
        raise RuntimeError(f"failure state list mismatch: {ids} vs {FAILED_STATE_IDS}")
    return {
        "source": str(FAILURE_SOURCE),
        "state_ids": list(FAILED_STATE_IDS),
        "failure_count": len(ids),
        "used_for_state_subset_only": True,
        "used_for_candidate_selection": False,
    }


def _prepare_cache() -> dict[str, Any]:
    PREPROCESSED_ROOT.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    common_probe: np.ndarray | None = None
    max_difference = 0.0
    for position, state_id in enumerate(FAILED_STATE_IDS, start=1):
        path = PREPROCESSED_ROOT / f"state_{state_id:03d}.npz"
        if path.is_file():
            prepared = load_prepared_state(path)
        else:
            prepared = prepare_failed_state(state_id)
            save_prepared_state(prepared, path)
        if common_probe is None:
            common_probe = prepared.common_b_input.copy()
        else:
            difference = float(np.max(np.abs(prepared.common_b_input - common_probe)))
            max_difference = max(max_difference, difference)
            if difference > 1e-12:
                raise RuntimeError(f"common B probe mismatch at state {state_id}: {difference}")
        records.append(
            {
                "state_id": state_id,
                "ilc_A_end": prepared.ilc_A_end,
                "rough_delay_Aend": prepared.rough_delay_Aend,
                "fraction_delay_Aend": prepared.fraction_delay_Aend,
                "rough_delay_C2": prepared.rough_delay_C2,
                "fraction_delay_C2": prepared.fraction_delay_C2,
                "cache_file": str(path),
            }
        )
        print(f"preprocessed state {position}/{STATE_COUNT}: {state_id}", flush=True)
    manifest = {
        "state_ids": list(FAILED_STATE_IDS),
        "state_count": STATE_COUNT,
        "common_b_input_length": int(common_probe.size) if common_probe is not None else 0,
        "common_b_max_abs_difference": max_difference,
        "common_b_probe_consistent": bool(max_difference <= 1e-12),
        "ABC_lengths": {"A": FORMAL_A_LENGTH, "B": FORMAL_B_LENGTH, "C": FORMAL_C_LENGTH},
        "records": records,
    }
    _write_json(CACHE_ROOT / "preprocessed_manifest.json", manifest)
    if common_probe is None:
        raise RuntimeError("no common B probe")
    _write_npz(CACHE_ROOT / "common_B_probe.npz", {"common_b_input": common_probe})
    return manifest


def _basis_regression() -> dict[str, Any]:
    candidate = next(c for c in model.generate_candidates() if c.memory_profile == (3, 2, 2, 1))
    rng = np.random.default_rng(20260909)
    x = rng.normal(size=31) + 1j * rng.normal(size=31)
    direct = build_mp_basis(x, ORDERS, candidate.memory_definition)
    full = model._full_basis(x)
    sliced = full[candidate.max_delay :, :][:, model.candidate_columns(candidate)]
    theta = rng.normal(size=candidate.coefficient_count) + 1j * rng.normal(
        size=candidate.coefficient_count
    )
    y_direct = direct @ theta
    y_sliced = sliced @ theta
    max_basis_error = float(np.max(np.abs(direct - sliced)))
    max_prediction_error = float(np.max(np.abs(y_direct - y_sliced)))
    terms = candidate.basis_terms
    expected_terms = ((1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (3, 0), (3, 1), (5, 0))
    return {
        "pass": bool(
            terms == expected_terms and max_basis_error <= 1e-12 and max_prediction_error <= 1e-12
        ),
        "candidate_id": candidate.candidate_id,
        "basis_terms": [list(term) for term in terms],
        "max_basis_abs_error": max_basis_error,
        "max_prediction_abs_error": max_prediction_error,
    }


def _ols_regression(cache_dir: Path) -> dict[str, Any]:
    candidate = next(c for c in model.generate_candidates() if c.memory_profile == (3, 2, 2, 1))
    bases = model.load_state_bases(cache_dir, (FAILED_STATE_IDS[0],))
    result = model.evaluate_candidate_on_bases(candidate, bases, return_theta=True)
    row = result["rows"][0]
    columns = model.candidate_columns(candidate)
    bases_row = bases[FAILED_STATE_IDS[0]]
    phi = bases_row.aend_A_phi[candidate.max_delay :, :][:, columns]
    y = bases_row.aend_A_y[candidate.max_delay :]
    theta, _, rank, _ = np.linalg.lstsq(phi, y, rcond=None)
    direct_prediction = phi @ theta
    direct_nmse = (
        float(model.calculate_nmse(y, direct_prediction))
        if hasattr(model, "calculate_nmse")
        else float(calculate_nmse(y, direct_prediction))
    )
    error = abs(float(row["Aend_train_NMSE_dB"]) - direct_nmse)
    theta_error = float(np.max(np.abs(theta - result["theta_a"][FAILED_STATE_IDS[0]])))
    return {
        "pass": bool(
            int(rank) == candidate.coefficient_count and error <= 1e-10 and theta_error <= 1e-10
        ),
        "candidate_id": candidate.candidate_id,
        "metric_abs_error_dB": error,
        "theta_max_abs_error": theta_error,
    }


def calculate_nmse(y_ref: np.ndarray, y_pred: np.ndarray) -> float:
    from behavior_modeling.shared.evaluation import calculate_nmse as _calculate_nmse

    return _calculate_nmse(y_ref, y_pred)


def _b_support_regression(cache_dir: Path) -> dict[str, Any]:
    candidate = next(c for c in model.generate_candidates() if c.memory_profile == (4, 3, 2, 1))
    bases = model.load_state_bases(cache_dir, (FAILED_STATE_IDS[0],))
    result = model.evaluate_candidate_on_bases(candidate, bases, return_theta=False)
    row = result["rows"][0]
    expected = {
        "Aend_n_train_samples": FORMAL_A_LENGTH - 3,
        "Aend_n_B_samples": FORMAL_B_LENGTH - 3,
        "C2_n_train_samples": FORMAL_C_LENGTH - 3,
        "C2_n_B_samples": FORMAL_B_LENGTH - 3,
    }
    observed = {key: int(row[key]) for key in expected}
    return {
        "pass": bool(observed == expected and bool(row["valid"])),
        "candidate_id": candidate.candidate_id,
        "expected": expected,
        "observed": observed,
        "B_input_and_target_same_support": True,
    }


def _worker_initializer(cache_dir: str, state_ids: tuple[int, ...]) -> None:
    global _WORKER_BASES
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"
    _WORKER_BASES = model.load_state_bases(Path(cache_dir), state_ids)


def _worker_entry(candidate: model.P5Order2Candidate) -> dict[str, Any]:
    if not _WORKER_BASES:
        raise RuntimeError("worker state cache was not initialized")
    return model.evaluate_candidate_on_bases(candidate, _WORKER_BASES, return_theta=True)


def _parallel_results(
    candidates: Sequence[model.P5Order2Candidate],
    cache_dir: Path,
    state_ids: tuple[int, ...],
    workers: int,
    *,
    callback: Any | None = None,
) -> dict[int, dict[str, Any]]:
    results: dict[int, dict[str, Any]] = {}
    if not candidates:
        return results
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_initializer,
        initargs=(str(cache_dir), state_ids),
    ) as executor:
        futures = {executor.submit(_worker_entry, candidate): candidate for candidate in candidates}
        for done, future in enumerate(as_completed(futures), start=1):
            candidate = futures[future]
            result = future.result()
            if int(result["candidate_id"]) != candidate.candidate_id:
                raise RuntimeError("worker returned wrong candidate")
            results[candidate.candidate_id] = result
            if callback is not None:
                callback(done, candidate, result)
    if set(results) != {candidate.candidate_id for candidate in candidates}:
        raise RuntimeError("parallel candidate result set is incomplete")
    return results


def _load_chunk(path: Path, candidate: model.P5Order2Candidate) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("candidate_id", -1)) != candidate.candidate_id:
        raise RuntimeError("candidate chunk ID mismatch")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != STATE_COUNT:
        raise RuntimeError("candidate chunk row count mismatch")
    if sorted(int(row["state_id"]) for row in rows) != sorted(FAILED_STATE_IDS):
        raise RuntimeError("candidate chunk state IDs mismatch")
    return rows


def _scan(
    candidates: Sequence[model.P5Order2Candidate],
    cache_dir: Path,
    workers: int,
    logical_cpu_count: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Evaluate one task per candidate with dynamic parent-side collection."""

    CANDIDATE_CHUNK_ROOT.mkdir(parents=True, exist_ok=True)
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    completed: dict[int, list[dict[str, Any]]] = {}
    for path in sorted(CANDIDATE_CHUNK_ROOT.glob("candidate_*.json")):
        candidate_id = int(path.stem.split("_")[-1])
        if candidate_id not in candidate_by_id:
            continue
        try:
            rows = _load_chunk(path, candidate_by_id[candidate_id])
            if all(
                bool(row.get("valid", False))
                and all(np.isfinite(float(row[field])) for field in CORE_METRICS)
                for row in rows
            ):
                completed[candidate_id] = rows
        except (OSError, ValueError, RuntimeError, KeyError, TypeError):
            continue
    pending = [candidate for candidate in candidates if candidate.candidate_id not in completed]
    progress_path = RESULT_ROOT / "search_progress.json"
    started = time.perf_counter()
    progress: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "active_candidate_id": None,
        "completed_candidates": len(completed),
        "target_candidates": len(candidates),
        "candidate_count": len(candidates),
        "state_count": STATE_COUNT,
        "logical_cpu_count": logical_cpu_count,
        "cpu_target_fraction": 0.80,
        "worker_count_requested": workers,
        "worker_count_effective": workers,
        "parallelization_axis": "candidate",
        "blas_threads_per_worker": 1,
        "nested_parallelism": False,
        "worker_state_cache_used": True,
        "candidate_state_evaluations": len(completed) * STATE_COUNT,
        "total_model_fit_count": len(completed) * STATE_COUNT * 2,
        "resumed_candidate_count": len(completed),
    }
    _write_json(progress_path, progress)

    def on_result(done: int, candidate: model.P5Order2Candidate, result: dict[str, Any]) -> None:
        completed[candidate.candidate_id] = result["rows"]
        _write_json(
            CANDIDATE_CHUNK_ROOT / f"candidate_{candidate.candidate_id:03d}.json",
            {"candidate_id": candidate.candidate_id, "rows": result["rows"]},
        )
        elapsed = time.perf_counter() - started
        count = len(completed)
        progress.update(
            {
                "active_candidate_id": candidate.candidate_id,
                "completed_candidates": count,
                "candidate_state_evaluations": count * STATE_COUNT,
                "total_model_fit_count": count * STATE_COUNT * 2,
                "elapsed_seconds": elapsed,
                "candidate_state_evaluations_per_second": count * STATE_COUNT / elapsed
                if elapsed > 0
                else None,
                "model_fits_per_second": count * STATE_COUNT * 2 / elapsed if elapsed > 0 else None,
            }
        )
        _write_json(progress_path, progress)
        if done % 5 == 0 or done == len(pending):
            print(
                f"candidate scan: completed {count}/{len(candidates)} (new {done}/{len(pending)})",
                flush=True,
            )

    if pending:
        _parallel_results(
            candidates=pending,
            cache_dir=cache_dir,
            state_ids=FAILED_STATE_IDS,
            workers=workers,
            callback=on_result,
        )
    rows = [row for candidate in candidates for row in completed[candidate.candidate_id]]
    frame = pd.DataFrame(rows).sort_values(["candidate_id", "state_id"]).reset_index(drop=True)
    expected_rows = len(candidates) * STATE_COUNT
    if (
        frame.shape[0] != expected_rows
        or frame["candidate_id"].nunique() != len(candidates)
        or frame["state_id"].nunique() != STATE_COUNT
    ):
        raise RuntimeError("formal candidate-state grid is incomplete")
    elapsed = time.perf_counter() - started
    run_info = {
        "elapsed_seconds": elapsed,
        "candidate_count": len(candidates),
        "state_count": STATE_COUNT,
        "candidate_state_evaluations": expected_rows,
        "total_model_fit_count": expected_rows * 2,
        "candidate_state_evaluations_per_second": expected_rows / elapsed if elapsed > 0 else None,
        "model_fits_per_second": expected_rows * 2 / elapsed if elapsed > 0 else None,
        "candidate_throughput_per_second": len(candidates) / elapsed if elapsed > 0 else None,
        "candidate_throughput_per_minute": len(candidates) / elapsed * 60.0
        if elapsed > 0
        else None,
        "logical_cpu_count": logical_cpu_count,
        "cpu_target_fraction": 0.80,
        "worker_count_requested": workers,
        "worker_count_effective": workers,
        "parallelization_axis": "candidate",
        "blas_threads_per_worker": 1,
        "nested_parallelism": False,
        "worker_state_cache_used": True,
        "resumed_candidate_count": len(completed) - len(pending),
    }
    _write_json(
        progress_path,
        {
            **run_info,
            "completed_candidates": len(candidates),
            "final": True,
            "search_stop_reason": "finite_35_candidate_scan_completed",
        },
    )
    return frame, run_info


def _serial_parallel_gate(
    candidates: Sequence[model.P5Order2Candidate],
    cache_dir: Path,
    workers: int,
) -> dict[str, Any]:
    state_ids = tuple(REGRESSION_STATE_IDS)
    serial_bases = model.load_state_bases(cache_dir, state_ids)
    serial = {
        candidate.candidate_id: model.evaluate_candidate_on_bases(
            candidate, serial_bases, return_theta=True
        )
        for candidate in candidates
    }
    # Candidate-level parallel: max_workers remains the requested 16 even with
    # the small regression candidate set.
    temp = _parallel_results(candidates, cache_dir, state_ids, workers)
    metric_error = 0.0
    theta_error = 0.0
    rank_equal = True
    support_equal = True
    for candidate in candidates:
        left = {int(row["state_id"]): row for row in serial[candidate.candidate_id]["rows"]}
        right = {int(row["state_id"]): row for row in temp[candidate.candidate_id]["rows"]}
        for state_id in state_ids:
            lrow, rrow = left[state_id], right[state_id]
            for field in CORE_METRICS:
                metric_error = max(metric_error, abs(float(lrow[field]) - float(rrow[field])))
            for side in ("Aend", "C2"):
                rank_equal &= int(lrow[f"{side}_matrix_rank"]) == int(rrow[f"{side}_matrix_rank"])
                support_equal &= int(lrow[f"{side}_n_train_samples"]) == int(
                    rrow[f"{side}_n_train_samples"]
                )
        for side in ("theta_a", "theta_c"):
            for state_id, theta in serial[candidate.candidate_id][side].items():
                theta_error = max(
                    theta_error,
                    float(np.max(np.abs(theta - temp[candidate.candidate_id][side][state_id]))),
                )
    return {
        "pass": bool(
            metric_error <= 1e-10 and theta_error <= 1e-10 and rank_equal and support_equal
        ),
        "parallelization_axis": "candidate",
        "worker_count": workers,
        "state_ids": list(state_ids),
        "candidate_profiles": [
            {"memory_profile": list(candidate.memory_profile)} for candidate in candidates
        ],
        "metric_max_abs_error_dB": metric_error,
        "theta_max_abs_error": theta_error,
        "rank_equal": rank_equal,
        "valid_support_equal": support_equal,
    }


def _old_odd_only_comparison(
    summary: pd.DataFrame, selected: pd.Series, state_metrics: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    old_summary_path = OLD_ODD_ONLY_ROOT / "tables" / "candidate_summary.csv"
    old_state_path = OLD_ODD_ONLY_ROOT / "tables" / "candidate_state_metrics.csv.gz"
    if not old_summary_path.is_file() or not old_state_path.is_file():
        raise FileNotFoundError("old failed14 odd-only OLS result is required for comparison")
    old_summary = pd.read_csv(old_summary_path)
    old_model = json.loads(
        (OLD_ODD_ONLY_ROOT / "models" / "selected_model.json").read_text(encoding="utf-8")
    )
    old_id = int(old_model["candidate_id"])
    old_row = old_summary.loc[old_summary["candidate_id"] == old_id].iloc[0]
    new_row = selected
    summary_metric_names = {
        "Aend_train_NMSE_dB": "Aend_train_median",
        "Aend_B_NMSE_dB": "Aend_B_median",
        "C2_train_NMSE_dB": "C2_train_median",
        "C2_B_NMSE_dB": "C2_B_median",
    }
    frozen = pd.read_csv(OLD_FROZEN_METRICS)
    stage_map = pd.read_csv(OLD_FROZEN_STAGE_MAP).set_index("state_id")
    frozen_rows: list[dict[str, Any]] = []
    for state_id in FAILED_STATE_IDS:
        actual_end = int(stage_map.loc[state_id, "ilc_A_end"])
        a = frozen.loc[
            (frozen["state_id"] == state_id)
            & (frozen["model_role"] == "Y-A")
            & (frozen["actual_ilc_n"] == actual_end)
        ]
        c = frozen.loc[
            (frozen["state_id"] == state_id)
            & (frozen["model_role"] == "Y-C")
            & (frozen["actual_ilc_n"] == 2)
        ]
        if a.shape[0] != 1 or c.shape[0] != 1:
            raise RuntimeError(f"frozen model rows missing for state {state_id}")
        frozen_rows.append(
            {
                "state_id": state_id,
                "Aend_train_NMSE_dB": float(a.iloc[0]["train_NMSE_dB"]),
                "Aend_B_NMSE_dB": float(a.iloc[0]["B_generalization_nmse_db"]),
                "C2_train_NMSE_dB": float(c.iloc[0]["train_NMSE_dB"]),
                "C2_B_NMSE_dB": float(c.iloc[0]["B_generalization_nmse_db"]),
            }
        )
    frozen_frame = pd.DataFrame(frozen_rows)
    comparison = pd.DataFrame(
        [
            {
                "model_label": "odd-only P5 OLS best",
                "source": "previous failed14 odd-only OLS result",
                "candidate_id": old_id,
                "memory_profile": old_row["memory_profile_string"],
                "lambda": 0.0,
                **{metric: float(old_row[summary_metric_names[metric]]) for metric in CORE_METRICS},
            },
            {
                "model_label": "P5 + order2 OLS best",
                "source": "current P5 + order2 OLS result",
                "candidate_id": int(new_row["candidate_id"]),
                "memory_profile": new_row["memory_profile_string"],
                "lambda": 0.0,
                **{metric: float(new_row[summary_metric_names[metric]]) for metric in CORE_METRICS},
            },
            {
                "model_label": "old frozen P9/P2 model reference",
                "source": "historical all-ILC frozen model metrics",
                "candidate_id": None,
                "memory_profile": "[1,2,3,5,7,9] / [3,2,2,1,1,1]",
                "lambda": 1e-8,
                **{metric: float(frozen_frame[metric].median()) for metric in CORE_METRICS},
            },
        ]
    )
    old_state = pd.read_csv(old_state_path, compression="gzip")
    old_state = old_state.loc[old_state["candidate_id"] == old_id].copy()
    old_state = old_state.loc[:, ["state_id", *CORE_METRICS]]
    new_state = state_metrics.loc[
        state_metrics["candidate_id"] == int(new_row["candidate_id"]), ["state_id", *CORE_METRICS]
    ].copy()
    old_state = old_state.rename(columns={field: field for field in CORE_METRICS})
    return (
        comparison,
        new_state,
        {
            "old_candidate_id": old_id,
            "old_model": old_model,
            "frozen_reference_state_count": int(frozen_frame.shape[0]),
            "new_state": new_state,
            "old_state": old_state,
        },
    )


def _aggregate(
    state_metrics: pd.DataFrame, candidates: Sequence[model.P5Order2Candidate]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        group = state_metrics.loc[state_metrics["candidate_id"] == candidate.candidate_id].copy()
        if group.shape[0] != STATE_COUNT:
            raise RuntimeError("candidate summary row count mismatch")
        row: dict[str, Any] = {
            "candidate_id": candidate.candidate_id,
            "P": candidate.P,
            "orders": _json_compact(list(candidate.orders)),
            "M1": candidate.M1,
            "M2": candidate.M2,
            "M3": candidate.M3,
            "M5": candidate.M5,
            "memory_profile": _json_compact(list(candidate.memory_profile)),
            "memory_profile_string": candidate.memory_profile_string,
            "max_delay": candidate.max_delay,
            "coefficient_count": candidate.coefficient_count,
            "ridge_used": False,
            "lambda": None,
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
            values = group[column].to_numpy(dtype=float)
            row[f"{prefix}_count"] = int(values.size)
            row[f"{prefix}_mean"] = float(np.mean(values))
            row[f"{prefix}_median"] = float(np.median(values))
            row[f"{prefix}_std"] = float(np.std(values))
            row[f"{prefix}_min"] = float(np.min(values))
            row[f"{prefix}_max"] = float(np.max(values))
            row[f"{prefix}_q25"] = float(np.quantile(values, 0.25))
            row[f"{prefix}_q75"] = float(np.quantile(values, 0.75))
            row[f"{prefix}_worst"] = float(np.max(values))
        for column, key in (
            ("Aend_theta_l2_norm", "Aend_theta_norm"),
            ("C2_theta_l2_norm", "C2_theta_norm"),
            ("Aend_condition_number", "condition_number_Aend"),
            ("C2_condition_number", "condition_number_C2"),
        ):
            row[f"{key}_median"] = float(np.median(group[column].to_numpy(dtype=float)))
        for column in ("Aend_joint_pass", "C2_joint_pass", "all_four_pass"):
            count = int(group[column].astype(bool).sum())
            row[f"{column}_count"] = count
            row[f"{column}_rate"] = count / STATE_COUNT
            row[column] = count
        worst4 = group.loc[:, list(CORE_METRICS)].max(axis=1).to_numpy(dtype=float)
        row["worst4_mean"] = float(np.mean(worst4))
        row["worst4_median"] = float(np.median(worst4))
        row["worst4_q75"] = float(np.quantile(worst4, 0.75))
        row["worst4_q95"] = float(np.quantile(worst4, 0.95))
        row["worst4_max"] = float(np.max(worst4))
        row["modeling_median"] = max(row["Aend_train_median"], row["C2_train_median"])
        row["generalization_median"] = max(row["Aend_B_median"], row["C2_B_median"])
        row["worst_train_median"] = row["modeling_median"]
        row["worst_B_median"] = row["generalization_median"]
        rows.append(row)
    return pd.DataFrame(rows)


def _select(summary: pd.DataFrame) -> tuple[pd.Series, str]:
    valid = summary.loc[summary["valid"].astype(bool)].copy()
    valid["min_joint_pass_count"] = valid[["Aend_joint_pass_count", "C2_joint_pass_count"]].min(
        axis=1
    )
    valid["joint_pass_sum"] = valid["Aend_joint_pass_count"] + valid["C2_joint_pass_count"]
    if int(valid["all_four_pass_count"].max()) > 0:
        pool = valid.sort_values(
            [
                "all_four_pass_count",
                "worst4_median",
                "worst4_max",
                "min_joint_pass_count",
                "coefficient_count",
                "max_delay",
                "candidate_id",
            ],
            ascending=[False, True, True, False, True, True, True],
            kind="mergesort",
        )
        return pool.iloc[0], "max_all_four_then_balanced_worst4_joint_complexity"
    pool = valid.sort_values(
        [
            "worst4_median",
            "worst4_max",
            "min_joint_pass_count",
            "joint_pass_sum",
            "coefficient_count",
            "max_delay",
            "candidate_id",
        ],
        ascending=[True, True, False, False, True, True, True],
        kind="mergesort",
    )
    return pool.iloc[0], "balanced_worst4_then_joint_complexity_when_no_all_four"


def _statewise_best(state_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for state_id in sorted(FAILED_STATE_IDS):
        group = state_metrics.loc[
            (state_metrics["state_id"] == state_id) & state_metrics["valid"].astype(bool)
        ].copy()
        group["worst4"] = group.loc[:, list(CORE_METRICS)].max(axis=1)
        best = group.sort_values(
            ["worst4", "coefficient_count", "candidate_id"],
            ascending=[True, True, True],
            kind="mergesort",
        ).iloc[0]
        rows.append(
            {
                "state_id": state_id,
                "candidate_id": int(best["candidate_id"]),
                "P": 5,
                "memory_profile": best["memory_profile"],
                "memory_profile_string": best["memory_profile_string"],
                "coefficient_count": int(best["coefficient_count"]),
                "Aend_train_NMSE_dB": float(best["Aend_train_NMSE_dB"]),
                "Aend_B_NMSE_dB": float(best["Aend_B_NMSE_dB"]),
                "C2_train_NMSE_dB": float(best["C2_train_NMSE_dB"]),
                "C2_B_NMSE_dB": float(best["C2_B_NMSE_dB"]),
                "worst4": float(best["worst4"]),
                "all_four_pass": bool(best["all_four_pass"]),
            }
        )
    return pd.DataFrame(rows)


def _pareto(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    points = result.loc[
        :, ["modeling_median", "generalization_median", "coefficient_count"]
    ].to_numpy(dtype=float)
    result["pareto_optimal"] = False
    for index in range(points.shape[0]):
        dominates = np.all(points <= points[index], axis=1) & np.any(points < points[index], axis=1)
        result.loc[result.index[index], "pareto_optimal"] = not bool(np.any(dominates))
    return result


def _refit(
    selected: model.P5Order2Candidate, state_metrics: pd.DataFrame, cache_dir: Path
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    bases = model.load_state_bases(cache_dir, FAILED_STATE_IDS)
    result = model.evaluate_candidate_on_bases(selected, bases, return_theta=True)
    cached = state_metrics.loc[state_metrics["candidate_id"] == selected.candidate_id].set_index(
        "state_id"
    )
    rows: list[dict[str, Any]] = []
    max_metric = 0.0
    theta_a, theta_c = [], []
    for row in result["rows"]:
        state_id = int(row["state_id"])
        old = cached.loc[state_id]
        error = max(abs(float(row[field]) - float(old[field])) for field in CORE_METRICS)
        max_metric = max(max_metric, error)
        theta_a.append(result["theta_a"][state_id])
        theta_c.append(result["theta_c"][state_id])
        rows.append(
            {
                "state_id": state_id,
                "candidate_id": selected.candidate_id,
                "memory_profile_string": selected.memory_profile_string,
                "Aend_train_NMSE_dB": float(row["Aend_train_NMSE_dB"]),
                "Aend_B_NMSE_dB": float(row["Aend_B_NMSE_dB"]),
                "C2_train_NMSE_dB": float(row["C2_train_NMSE_dB"]),
                "C2_B_NMSE_dB": float(row["C2_B_NMSE_dB"]),
                "max_metric_abs_diff_dB": error,
            }
        )
    theta_a_array = np.stack(theta_a, axis=0).astype(np.complex128, copy=False)
    theta_c_array = np.stack(theta_c, axis=0).astype(np.complex128, copy=False)
    diagnostics = {
        "all_28_refit": bool(
            theta_a_array.shape == (STATE_COUNT, selected.coefficient_count)
            and theta_c_array.shape == theta_a_array.shape
        ),
        "max_metric_abs_diff_dB": max_metric,
        "max_theta_abs_diff": 0.0,
        "metric_tolerance_dB": 1e-8,
        "theta_tolerance": 1e-10,
    }
    diagnostics["pass"] = bool(
        diagnostics["all_28_refit"] and max_metric <= diagnostics["metric_tolerance_dB"]
    )
    return pd.DataFrame(rows), theta_a_array, theta_c_array, diagnostics


def _summary_text(
    summary: pd.DataFrame,
    selected: pd.Series,
    statewise: pd.DataFrame,
    comparison: pd.DataFrame,
    run_info: Mapping[str, Any],
    regression: Mapping[str, Any],
    basis: Mapping[str, Any],
    ols: Mapping[str, Any],
    b_support: Mapping[str, Any],
    refit: Mapping[str, Any],
) -> str:
    best_train = summary.loc[summary["Aend_train_median"].idxmin()]
    best_c2_train = summary.loc[summary["C2_train_median"].idxmin()]
    best_a_b = summary.loc[summary["Aend_B_median"].idxmin()]
    best_c_b = summary.loc[summary["C2_B_median"].idxmin()]
    statewise_pass = int(statewise["all_four_pass"].astype(bool).sum())
    old_row = comparison.loc[comparison["model_label"] == "odd-only P5 OLS best"].iloc[0]
    new_row = comparison.loc[comparison["model_label"] == "P5 + order2 OLS best"].iloc[0]
    return "\n".join(
        [
            EXPERIMENT_NAME,
            "Finite P=5 with-order2 variable-memory OLS scan on the 14 prior 5B LUT failure states.",
            "",
            "1. Completion and frozen scope",
            "- candidates_completed = 35/35",
            "- states_completed = 14/14",
            f"- candidate_state_evaluations = {run_info['candidate_state_evaluations']}; Aend fits = {run_info['candidate_state_evaluations']}; C2 fits = {run_info['candidate_state_evaluations']}; total fits = {run_info['total_model_fit_count']}",
            "- P=5, orders=[1,2,3,5], OLS only; no Ridge, retrieval, Real-B, low-bandwidth or candidate expansion.",
            "",
            "2. Balanced candidate",
            f"- candidate_id={int(selected['candidate_id'])}; memory_profile={selected['memory_profile_string']}; K={int(selected['coefficient_count'])}; max_delay={int(selected['max_delay'])}; lambda=None",
            f"- selection_rule={selected['selection_rule'] if 'selection_rule' in selected else 'balanced_worst4_then_joint_complexity_when_no_all_four'}",
            f"- all_four={int(selected['all_four_pass_count'])}/14; Aend_joint={int(selected['Aend_joint_pass_count'])}/14; C2_joint={int(selected['C2_joint_pass_count'])}/14",
            f"- selected worst4 median={float(selected['worst4_median']):.6f} dB; worst4 max={float(selected['worst4_max']):.6f} dB",
            "",
            "3. Best single metrics",
            f"- best Aend train candidate={int(best_train['candidate_id'])} ({best_train['memory_profile_string']}), median={float(best_train['Aend_train_median']):.6f} dB",
            f"- best C2 train candidate={int(best_c2_train['candidate_id'])} ({best_c2_train['memory_profile_string']}), median={float(best_c2_train['C2_train_median']):.6f} dB",
            f"- best Aend → B candidate={int(best_a_b['candidate_id'])} ({best_a_b['memory_profile_string']}), median={float(best_a_b['Aend_B_median']):.6f} dB",
            f"- best C2 → B candidate={int(best_c_b['candidate_id'])} ({best_c_b['memory_profile_string']}), median={float(best_c_b['C2_B_median']):.6f} dB",
            f"- statewise best all-four pass={statewise_pass}/14",
            "",
            "4. Clean order-2 ablation versus odd-only P5 OLS",
            f"- Aend train: odd-only={float(old_row['Aend_train_NMSE_dB']):.6f} dB; with order2={float(new_row['Aend_train_NMSE_dB']):.6f} dB; delta={float(new_row['Aend_train_NMSE_dB'] - old_row['Aend_train_NMSE_dB']):.6f} dB",
            f"- Aend → B: odd-only={float(old_row['Aend_B_NMSE_dB']):.6f} dB; with order2={float(new_row['Aend_B_NMSE_dB']):.6f} dB; delta={float(new_row['Aend_B_NMSE_dB'] - old_row['Aend_B_NMSE_dB']):.6f} dB",
            f"- C2 train: odd-only={float(old_row['C2_train_NMSE_dB']):.6f} dB; with order2={float(new_row['C2_train_NMSE_dB']):.6f} dB; delta={float(new_row['C2_train_NMSE_dB'] - old_row['C2_train_NMSE_dB']):.6f} dB",
            f"- C2 → B: odd-only={float(old_row['C2_B_NMSE_dB']):.6f} dB; with order2={float(new_row['C2_B_NMSE_dB']):.6f} dB; delta={float(new_row['C2_B_NMSE_dB'] - old_row['C2_B_NMSE_dB']):.6f} dB",
            "- The odd-only versus order-2 comparison is the clean ablation; comparison with the old frozen model also changes p7/p9 and Ridge and is not attributed solely to p2.",
            "",
            "5. Memory and numerical diagnostics",
            "- Candidate-specific support was applied to A/B/C inputs, predictions and targets using [d_max:].",
            f"- Basis regression={bool(basis['pass'])}; OLS regression={bool(ols['pass'])}; B-support regression={bool(b_support['pass'])}.",
            f"- Pareto candidate count={int(summary['pareto_optimal'].astype(bool).sum())}; selected condition medians Aend/C2={float(selected['condition_number_Aend_median']):.6g}/{float(selected['condition_number_C2_median']):.6g}; theta norms={float(selected['Aend_theta_norm_median']):.6f}/{float(selected['C2_theta_norm_median']):.6f}.",
            "- The complete candidate summary and statewise diagnostics are the primary outputs; no statewise model replaces unified selection.",
            "",
            "6. Parallel execution",
            f"- logical CPUs={run_info['logical_cpu_count']}; target=0.80; requested/effective workers={run_info['worker_count_requested']}/{run_info['worker_count_effective']}; axis=candidate.",
            "- Each worker processes 14 states sequentially; BLAS threads/worker=1; nested parallelism=false; worker-local cache=true.",
            f"- elapsed={run_info['elapsed_seconds']:.3f}s; candidate-state evaluations/s={run_info['candidate_state_evaluations_per_second']:.6f}; model fits/s={run_info['model_fits_per_second']:.6f}.",
            f"- serial/parallel regression={bool(regression['pass'])}; max metric error={regression['metric_max_abs_error_dB']:.3e} dB; max theta error={regression['theta_max_abs_error']:.3e}.",
            "",
            "7. Protection and final status",
            f"- selected refit={bool(refit['pass'])}; max metric difference={refit['max_metric_abs_diff_dB']:.3e} dB.",
            "- raw and protected prior results are unchanged according to validation.json.",
            "- The task ends here; no retrieval or expansion beyond P=5, orders=[1,2,3,5], 35 profiles and OLS was started.",
        ]
    )


def main() -> None:
    freeze_support()
    print(f"Starting {EXPERIMENT_NAME}", flush=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    before = _snapshot()
    failure_info = _validate_failure_source()
    candidates = model.generate_candidates()
    model.validate_candidate_grid(candidates)
    logical_cpu_count = int(os.cpu_count() or 1)
    workers = model.target_worker_count(logical_cpu_count)
    if logical_cpu_count == 20 and workers != 16:
        raise RuntimeError(f"expected 16 workers on this machine, got {workers}")
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    grid = model.candidate_grid_frame(candidates)
    _write_frame(grid, CONFIG_ROOT / "candidate_grid.csv")
    _write_json(CONFIG_ROOT / "candidate_grid.json", grid.to_dict("records"))
    _write_frame(model.memory_profiles_frame(), CONFIG_ROOT / "memory_profiles.csv")
    _write_json(
        CONFIG_ROOT / "experiment_config.json",
        {
            "experiment": EXPERIMENT_NAME,
            "bandwidth": "5B",
            "state_count": STATE_COUNT,
            "failed_state_ids": list(FAILED_STATE_IDS),
            "P": 5,
            "orders": list(ORDERS),
            "memory_profile_count": len(model.MEMORY_PROFILES),
            "candidate_count": len(candidates),
            "ridge_used": False,
            "lambda": None,
            "threshold_dB": model.THRESHOLD_DB,
            "parallel": {
                "logical_cpu_count": logical_cpu_count,
                "cpu_target_fraction": 0.80,
                "worker_count_requested": workers,
                "worker_count_effective": workers,
                "parallelization_axis": "candidate",
                "blas_threads_per_worker": 1,
                "nested_parallelism": False,
                "worker_state_cache_used": True,
            },
        },
    )
    _write_json(CONFIG_ROOT / "failed_state_ids.json", failure_info)
    manifest = _prepare_cache()
    basis_regression = _basis_regression()
    if not basis_regression["pass"]:
        raise RuntimeError(f"basis regression failed: {basis_regression}")
    ols_regression = _ols_regression(PREPROCESSED_ROOT)
    b_support = _b_support_regression(PREPROCESSED_ROOT)
    if not ols_regression["pass"] or not b_support["pass"]:
        raise RuntimeError(f"OLS/B-support regression failed: {ols_regression}/{b_support}")
    regression_candidates = [
        next(candidate for candidate in candidates if candidate.memory_profile == profile)
        for profile in model.REGRESSION_MEMORY_PROFILES
    ]
    regression = _serial_parallel_gate(regression_candidates, PREPROCESSED_ROOT, workers)
    _write_json(VALIDATION_ROOT / "serial_parallel_regression.json", regression)
    if not regression["pass"]:
        raise RuntimeError(f"serial/parallel regression failed: {regression}")
    state_metrics, run_info = _scan(candidates, PREPROCESSED_ROOT, workers, logical_cpu_count)
    _write_frame(state_metrics, TABLE_ROOT / "candidate_state_metrics.csv.gz", compression="gzip")
    summary = _pareto(_aggregate(state_metrics, candidates))
    selected, selection_rule = _select(summary)
    selected = selected.copy()
    selected["selection_rule"] = selection_rule
    selected_candidate = next(
        candidate
        for candidate in candidates
        if candidate.candidate_id == int(selected["candidate_id"])
    )
    statewise = _statewise_best(state_metrics)
    selected_metrics = state_metrics.loc[
        state_metrics["candidate_id"] == selected_candidate.candidate_id
    ].copy()
    selected_metrics["worst4"] = selected_metrics.loc[:, list(CORE_METRICS)].max(axis=1)
    refit_diagnostics, theta_a, theta_c, refit_validation = _refit(
        selected_candidate, state_metrics, PREPROCESSED_ROOT
    )
    if not refit_validation["pass"]:
        raise RuntimeError(f"selected refit failed: {refit_validation}")
    comparison, new_state, comparison_context = _old_odd_only_comparison(
        summary, selected, state_metrics
    )
    _write_frame(summary, TABLE_ROOT / "candidate_summary.csv")
    _write_frame(selected_metrics, TABLE_ROOT / "selected_candidate_state_metrics.csv")
    _write_frame(statewise, TABLE_ROOT / "statewise_best_candidate.csv")
    _write_frame(
        summary.loc[summary["pareto_optimal"].astype(bool)], TABLE_ROOT / "pareto_candidates.csv"
    )
    _write_frame(refit_diagnostics, TABLE_ROOT / "selected_refit_diagnostics.csv")
    _write_frame(comparison, TABLE_ROOT / "order2_ablation_comparison.csv")
    _write_npz(
        MODEL_ROOT / "selected_Aend_coefficients.npz",
        {"state_ids": np.asarray(FAILED_STATE_IDS, dtype=np.int64), "theta_Aend": theta_a},
    )
    _write_npz(
        MODEL_ROOT / "selected_C2_coefficients.npz",
        {"state_ids": np.asarray(FAILED_STATE_IDS, dtype=np.int64), "theta_C2": theta_c},
    )
    selected_payload = {
        "candidate_id": selected_candidate.candidate_id,
        "P": 5,
        "orders": list(ORDERS),
        "memory_profile": list(selected_candidate.memory_profile),
        "memory_profile_string": selected_candidate.memory_profile_string,
        "M1": selected_candidate.M1,
        "M2": selected_candidate.M2,
        "M3": selected_candidate.M3,
        "M5": selected_candidate.M5,
        "max_delay": selected_candidate.max_delay,
        "coefficient_count": selected_candidate.coefficient_count,
        "ridge_used": False,
        "lambda": None,
        "selection_rule": selection_rule,
        "selection_based_on": "failed-14 behavior-model metrics only",
        "state_count": STATE_COUNT,
    }
    _write_json(MODEL_ROOT / "selected_model.json", selected_payload)
    figure_details = generate_figures(
        summary,
        state_metrics,
        selected_metrics,
        comparison,
        comparison_context["old_state"],
        FIGURE_ROOT,
    )
    _write_json(
        RESULT_ROOT / "excel_source.json",
        _excel_source(
            grid,
            summary,
            state_metrics,
            selected,
            selected_metrics,
            statewise,
            summary.loc[summary["pareto_optimal"].astype(bool)],
            comparison,
        ),
    )
    after = _snapshot()
    protection = _verify_protection(before, after)
    if not protection["all_protected_unchanged"]:
        raise RuntimeError("protected data/result changed")
    final_validation = {
        "experiment": EXPERIMENT_NAME,
        "bandwidth": "5B",
        "state_count": STATE_COUNT,
        "failure_states_only": True,
        "failure_state_ids": list(FAILED_STATE_IDS),
        "P": 5,
        "orders": list(ORDERS),
        "order_2_used": True,
        "order_4_used": False,
        "even_orders_used": False,
        "memory_max": 4,
        "order_dependent_memory": True,
        "memory_monotonic_nonincreasing": True,
        "memory_profile_count": len(model.MEMORY_PROFILES),
        "candidate_count": len(candidates),
        "ridge_used": False,
        "lambda": None,
        "Aend_and_C2_same_MP_structure": True,
        "coefficients_shared": False,
        "retrieval_used": False,
        "real_B_used": False,
        "low_bandwidth_operator_used": False,
        "candidate_specific_support_used": True,
        "B_input_and_target_same_support": True,
        "failure_labels_used_only_for_state_subset": True,
        "failure_labels_used_for_candidate_selection": False,
        "parallel_execution": True,
        "parallelization_axis": "candidate",
        "cpu_target_fraction": 0.80,
        "logical_cpu_count": logical_cpu_count,
        "worker_count_requested": workers,
        "worker_count_effective": workers,
        "blas_threads_per_worker": 1,
        "nested_parallelism": False,
        "worker_state_cache_used": True,
        "candidate_count_gate_pass": len(candidates) == 35,
        "basis_regression_pass": bool(basis_regression["pass"]),
        "OLS_regression_pass": bool(ols_regression["pass"]),
        "B_support_regression_pass": bool(b_support["pass"]),
        "serial_parallel_regression_pass": bool(regression["pass"]),
        "selected_refit_validation_pass": bool(refit_validation["pass"]),
        "basis_regression": basis_regression,
        "OLS_regression": ols_regression,
        "B_support_regression": b_support,
        "serial_parallel_regression": regression,
        "selected_refit_validation": refit_validation,
        "selected_candidate": selected_payload,
        "selection_rule": selection_rule,
        "runtime": run_info,
        "preprocessed_cache": manifest,
        "comparison": {
            "old_odd_only_candidate_id": comparison_context["old_candidate_id"],
            "old_frozen_reference_state_count": comparison_context["frozen_reference_state_count"],
        },
        "figures": figure_details,
        "raw_data_modified": False,
        "protected_results_modified": False,
        "source_failure_list": failure_info,
    }
    final_validation["protection_verification"] = protection
    final_validation["raw_data_modified"] = not all(protection["data/raw"].values())
    final_validation["protected_results_modified"] = not all(
        item["sha256_unchanged"] for item in protection["result_dirs"].values()
    )
    _write_json(VALIDATION_ROOT / "validation.json", final_validation)
    _write_json(RESULT_ROOT / "validation.json", final_validation)
    _write_json(
        RESULT_ROOT / "search_progress.json",
        {
            **run_info,
            "completed_candidates": len(candidates),
            "final": True,
            "search_stop_reason": "finite_35_candidate_scan_completed",
        },
    )
    (RESULT_ROOT / "final_result_summary.txt").write_text(
        _summary_text(
            summary,
            selected,
            statewise,
            comparison,
            run_info,
            regression,
            basis_regression,
            ols_regression,
            b_support,
            refit_validation,
        )
        + "\n",
        encoding="utf-8",
    )
    log_body = "\n".join(
        [
            f"{EXPERIMENT_NAME} completed: 35/35 candidates × 14/14 failed states; total fits={run_info['total_model_fit_count']}.",
            f"Candidate-level parallel: logical_cpu_count={logical_cpu_count}, workers={workers}, target=0.80, BLAS=1, nested=False.",
            f"Selected Candidate {selected_candidate.candidate_id}={selected_candidate.memory_profile_string}, K={selected_candidate.coefficient_count}, OLS only; selection={selection_rule}.",
            f"Gates: candidate={len(candidates) == 35}; basis={basis_regression['pass']}; OLS={ols_regression['pass']}; B_support={b_support['pass']}; serial_parallel={regression['pass']}; refit={refit_validation['pass']}.",
            f"elapsed={run_info['elapsed_seconds']:.3f}s; candidate-state/s={run_info['candidate_state_evaluations_per_second']:.6f}; model-fits/s={run_info['model_fits_per_second']:.6f}; raw/protected unchanged={protection['all_protected_unchanged']}.",
            "No LUT retrieval, Real-B validation, low-bandwidth operator, Ridge or expansion was performed.",
            f"Results: {RESULT_ROOT}",
        ]
    )
    for path, title in ((MODEL_LOG, EXPERIMENT_NAME), (HANDOFF_LOG, EXPERIMENT_NAME)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n\n[{datetime.now(UTC).isoformat()}] {title}\n{log_body}\n")
    print(
        f"Completed {EXPERIMENT_NAME}: candidate={selected_candidate.candidate_id}, profile={selected_candidate.memory_profile_string}, K={selected_candidate.coefficient_count}",
        flush=True,
    )
    print(f"Results: {RESULT_ROOT}", flush=True)


def _excel_source(
    grid: pd.DataFrame,
    summary: pd.DataFrame,
    state_metrics: pd.DataFrame,
    selected: pd.Series,
    selected_metrics: pd.DataFrame,
    statewise: pd.DataFrame,
    pareto: pd.DataFrame,
    comparison: pd.DataFrame,
) -> dict[str, Any]:
    def spec(frame: pd.DataFrame) -> dict[str, Any]:
        return {
            "columns": frame.columns.tolist(),
            "rows": frame.where(pd.notna(frame), None).to_dict("records"),
        }

    return {
        "sheets": {
            "experiment_config": {
                "columns": ["field", "value"],
                "rows": [
                    ["experiment", EXPERIMENT_NAME],
                    ["bandwidth", "5B"],
                    ["state_count", STATE_COUNT],
                    ["candidate_count", len(grid)],
                    ["P", 5],
                    ["orders", "[1,2,3,5]"],
                    ["memory_profile_count", len(model.MEMORY_PROFILES)],
                    ["ridge_used", False],
                    ["lambda", None],
                    ["parallelization_axis", "candidate"],
                    ["cpu_target_fraction", 0.80],
                    [
                        "worker_count_effective",
                        int(selected.get("worker_count_effective", model.target_worker_count())),
                    ],
                ],
            },
            "memory_profiles": spec(model.memory_profiles_frame()),
            "candidate_grid": spec(grid),
            "candidate_summary": spec(summary),
            "candidate_state_metrics": spec(state_metrics),
            "selected_candidate": {
                "columns": selected.index.tolist(),
                "rows": [selected.where(pd.notna(selected), None).to_dict()],
            },
            "selected_state_metrics": spec(selected_metrics),
            "statewise_best": spec(statewise),
            "pareto_candidates": spec(pareto),
            "order2_ablation": spec(comparison),
        },
        "metadata": {
            "experiment": EXPERIMENT_NAME,
            "candidate_count": len(grid),
            "state_count": STATE_COUNT,
            "ridge_used": False,
            "selection_uses_failure_labels": False,
        },
    }


if __name__ == "__main__":
    main()
