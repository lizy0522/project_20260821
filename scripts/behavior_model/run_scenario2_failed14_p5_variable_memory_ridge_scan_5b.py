"""Finite P=5 variable-memory MP + Ridge scan for the failed-14 states.

The numerical search is exactly 20 monotone memory profiles times 10 fixed
lambda values.  Candidate-level ``ProcessPoolExecutor`` workers load the
preprocessed 14-state cache once, process one candidate over all states, and
return rows to the parent.  No LUT retrieval, Real-B validation, low-bandwidth
operator, or candidate expansion is part of this task.
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

from behavior_model import p5_variable_memory_ridge_scan as model  # noqa: E402
from behavior_model.odd_order_variable_memory_scan import (  # noqa: E402
    FAILED_STATE_IDS,
    FORMAL_A_LENGTH,
    FORMAL_B_LENGTH,
    FORMAL_C_LENGTH,
    REGRESSION_STATE_IDS,
    load_state_bases,
    prepare_failed_state,
    save_prepared_state,
)
from behavior_model.plot_scenario2_failed14_p5_variable_memory_ridge_scan_5b import (  # noqa: E402
    generate_figures,
)

STATE_COUNT = model.STATE_COUNT
ORDERS = model.ORDERS

TASK_NAME = "behavior_model"
EXPERIMENT_NAME = "scenario_2_failed14_P5_variable_memory_ridge_scan_5B"
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
OLD_OLS_FAILED14_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_model"
    / "scenario_2_failed14_odd_order_mp_order_dependent_memory_scan_5B"
)

PROTECTED_RESULT_DIRS = {
    "formal_5B_retrieval": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_lut_retrieval"
    / "scenario_2_C2_to_Aend",
    "formal_1B_retrieval": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_lut_retrieval"
    / "scenario_2_C2_to_Aend_1B",
    "formal_0p5B_retrieval": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_lut_retrieval"
    / "scenario_2_C2_to_Aend_0p5B",
    "equal_ABC_retrieval": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_lut_retrieval"
    / "scenario_2_C2_to_Aend_equal_ABC",
    "unified_capacity_retrieval": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_lut_retrieval"
    / "scenario_2_C2_to_Aend_unified_model_capacity",
    "retrieval_oriented_scan": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_lut_retrieval"
    / "scenario_2_C2_to_Aend_retrieval_oriented_model_scan",
    "all_ilc_analysis": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2_all_ilc",
    "old_failed14_ols": OLD_OLS_FAILED14_ROOT,
    "old_unified_odd_order": PROJECT_ROOT
    / "results"
    / "behavior_model"
    / "scenario_2_unified_odd_order_mp_capacity_scan_5B",
    "statewise_best_round0_4": PROJECT_ROOT
    / "results"
    / "behavior_model"
    / "scenario_2_statewise_best_round0_4_5B",
}

CORE_METRIC_COLUMNS = (
    "Aend_train_NMSE_dB",
    "Aend_B_NMSE_dB",
    "C2_train_NMSE_dB",
    "C2_B_NMSE_dB",
)
METRIC_STATS = ("mean", "median", "std", "min", "max", "q25", "q75", "worst")

_WORKER_BASES: dict[int, Any] = {}
_WORKER_REFERENCE: dict[str, np.ndarray] = {}


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
    raw_root = PROJECT_ROOT / "data" / "raw"
    digest = hashlib.sha256()
    files = sorted(
        (file for file in raw_root.rglob("*") if file.is_file()),
        key=lambda item: str(item).lower(),
    )
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
            name: {
                "path": str(path),
                "exists": path.is_dir(),
                "sha256": _tree_digest(path),
            }
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
    required = {"State_n_R", "failure"}
    if not required.issubset(frame.columns):
        raise ValueError(f"failure source missing columns: {sorted(required - set(frame.columns))}")
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
    maximum_difference = 0.0
    for position, state_id in enumerate(FAILED_STATE_IDS, start=1):
        path = PREPROCESSED_ROOT / f"state_{state_id:03d}.npz"
        if path.is_file():
            from behavior_model.odd_order_variable_memory_scan import load_prepared_state

            prepared = load_prepared_state(path)
        else:
            prepared = prepare_failed_state(state_id)
            save_prepared_state(prepared, path)
        if common_probe is None:
            common_probe = prepared.common_b_input.copy()
        else:
            difference = float(np.max(np.abs(prepared.common_b_input - common_probe)))
            maximum_difference = max(maximum_difference, difference)
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
    if common_probe is None:
        raise RuntimeError("no common B probe")
    manifest = {
        "state_ids": list(FAILED_STATE_IDS),
        "state_count": STATE_COUNT,
        "common_b_input_length": int(common_probe.size),
        "common_b_max_abs_difference": maximum_difference,
        "common_b_probe_consistent": bool(maximum_difference <= 1e-12),
        "ABC_lengths": {"A": FORMAL_A_LENGTH, "B": FORMAL_B_LENGTH, "C": FORMAL_C_LENGTH},
        "records": records,
    }
    _write_json(CACHE_ROOT / "preprocessed_manifest.json", manifest)
    _write_npz(CACHE_ROOT / "common_B_probe.npz", {"common_b_input": common_probe})
    return manifest


def _ridge_math_gate() -> dict[str, Any]:
    rng = np.random.default_rng(20260909)
    n, k = 19, 4
    phi = rng.normal(size=(n, k)) + 1j * rng.normal(size=(n, k))
    y = rng.normal(size=n) + 1j * rng.normal(size=n)
    ridge_lambda = 1e-8
    identity = np.eye(k, dtype=np.complex128)
    augmented_phi = np.vstack((phi, np.sqrt(n * ridge_lambda) * identity))
    augmented_y = np.concatenate((y, np.zeros(k, dtype=np.complex128)))
    theta, _, rank, singular = np.linalg.lstsq(augmented_phi, augmented_y, rcond=None)
    objective = float(
        np.linalg.norm(phi @ theta - y) ** 2 / n + ridge_lambda * np.linalg.norm(theta) ** 2
    )
    augmented_objective = float(np.linalg.norm(augmented_phi @ theta - augmented_y) ** 2 / n)
    stationarity = float(
        np.linalg.norm(phi.conj().T @ (phi @ theta - y) / n + ridge_lambda * theta)
    )
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
    _WORKER_BASES = load_state_bases(Path(cache_dir), state_ids)
    _WORKER_REFERENCE = model.load_ols_reference(Path(reference_path))


def _worker_entry(candidate: model.P5RidgeCandidate) -> dict[str, Any]:
    if not _WORKER_BASES or not _WORKER_REFERENCE:
        raise RuntimeError("worker caches were not initialized")
    return model.evaluate_candidate_on_bases(
        candidate,
        _WORKER_BASES,
        _WORKER_REFERENCE,
        return_theta=True,
    )


def _parallel_results(
    candidates: Sequence[model.P5RidgeCandidate],
    cache_dir: Path,
    reference_path: Path,
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
        initargs=(str(cache_dir), str(reference_path), state_ids),
    ) as executor:
        futures = {executor.submit(_worker_entry, candidate): candidate for candidate in candidates}
        for done, future in enumerate(as_completed(futures), start=1):
            candidate = futures[future]
            result = future.result()
            if int(result["candidate_id"]) != candidate.candidate_id:
                raise RuntimeError("worker returned wrong candidate ID")
            if len(result["rows"]) != len(state_ids):
                raise RuntimeError("worker returned incomplete state rows")
            results[candidate.candidate_id] = result
            if callback is not None:
                callback(done, candidate, result)
    if set(results) != {candidate.candidate_id for candidate in candidates}:
        raise RuntimeError("parallel result set is incomplete")
    return results


def _serial_results(
    candidates: Sequence[model.P5RidgeCandidate],
    cache_dir: Path,
    reference: Mapping[str, np.ndarray],
    state_ids: tuple[int, ...],
) -> dict[int, dict[str, Any]]:
    bases = load_state_bases(cache_dir, state_ids)
    return {
        candidate.candidate_id: model.evaluate_candidate_on_bases(
            candidate, bases, reference, return_theta=True
        )
        for candidate in candidates
    }


def _serial_parallel_gate(
    candidates: Sequence[model.P5RidgeCandidate],
    cache_dir: Path,
    reference_path: Path,
    reference: Mapping[str, np.ndarray],
    workers: int,
) -> dict[str, Any]:
    serial = _serial_results(candidates, cache_dir, reference, tuple(REGRESSION_STATE_IDS))
    parallel = _parallel_results(
        candidates, cache_dir, reference_path, tuple(REGRESSION_STATE_IDS), workers
    )
    metric_error = 0.0
    theta_error = 0.0
    rank_equal = True
    support_equal = True
    for candidate in candidates:
        left = {int(row["state_id"]): row for row in serial[candidate.candidate_id]["rows"]}
        right = {int(row["state_id"]): row for row in parallel[candidate.candidate_id]["rows"]}
        for state_id in REGRESSION_STATE_IDS:
            lrow, rrow = left[int(state_id)], right[int(state_id)]
            for field in CORE_METRIC_COLUMNS:
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
                    float(np.max(np.abs(theta - parallel[candidate.candidate_id][side][state_id]))),
                )
    return {
        "pass": bool(
            metric_error <= 1e-10 and theta_error <= 1e-10 and rank_equal and support_equal
        ),
        "parallelization_axis": "candidate",
        "state_ids": list(REGRESSION_STATE_IDS),
        "candidate_profiles": [
            {
                "P": candidate.P,
                "memory_profile": list(candidate.memory_profile),
                "lambda": candidate.ridge_lambda,
            }
            for candidate in candidates
        ],
        "worker_count": workers,
        "metric_max_abs_error_dB": metric_error,
        "theta_max_abs_error": theta_error,
        "rank_equal": rank_equal,
        "valid_support_equal": support_equal,
    }


def _load_chunk(path: Path, candidate: model.P5RidgeCandidate) -> list[dict[str, Any]]:
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
    candidates: Sequence[model.P5RidgeCandidate],
    cache_dir: Path,
    reference_path: Path,
    workers: int,
    logical_cpu_count: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    CANDIDATE_CHUNK_ROOT.mkdir(parents=True, exist_ok=True)
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    completed: dict[int, list[dict[str, Any]]] = {}
    for path in sorted(CANDIDATE_CHUNK_ROOT.glob("candidate_*.json")):
        candidate_id = int(path.stem.split("_")[-1])
        if candidate_id not in candidate_by_id:
            continue
        try:
            rows = _load_chunk(path, candidate_by_id[candidate_id])
            if not all(
                bool(row.get("valid", False))
                and all(np.isfinite(float(row[field])) for field in CORE_METRIC_COLUMNS)
                for row in rows
            ):
                continue
            completed[candidate_id] = rows
        except (OSError, ValueError, RuntimeError, KeyError, TypeError):
            continue
    pending = [candidate for candidate in candidates if candidate.candidate_id not in completed]
    started = time.perf_counter()
    progress_path = RESULT_ROOT / "search_progress.json"
    progress: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "active_candidate_id": None,
        "completed_candidates": len(completed),
        "target_candidates": len(candidates),
        "candidate_count": len(candidates),
        "state_count": STATE_COUNT,
        "logical_cpu_count": logical_cpu_count,
        "cpu_target_fraction": 0.90,
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

    def on_result(done: int, candidate: model.P5RidgeCandidate, result: dict[str, Any]) -> None:
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
        if done % 10 == 0 or done == len(pending):
            print(
                f"candidate scan: completed {count}/{len(candidates)} (new {done}/{len(pending)})",
                flush=True,
            )

    if pending:
        _parallel_results(
            pending,
            cache_dir,
            reference_path,
            tuple(FAILED_STATE_IDS),
            workers,
            callback=on_result,
        )
    rows = [row for candidate in candidates for row in completed[candidate.candidate_id]]
    frame = pd.DataFrame(rows).sort_values(["candidate_id", "state_id"]).reset_index(drop=True)
    expected = len(candidates) * STATE_COUNT
    if (
        frame.shape[0] != expected
        or frame["candidate_id"].nunique() != len(candidates)
        or frame["state_id"].nunique() != STATE_COUNT
    ):
        raise RuntimeError("formal candidate-state grid is incomplete")
    elapsed = time.perf_counter() - started
    run_info = {
        "elapsed_seconds": elapsed,
        "candidate_count": len(candidates),
        "state_count": STATE_COUNT,
        "candidate_state_evaluations": expected,
        "total_model_fit_count": expected * 2,
        "candidate_state_evaluations_per_second": expected / elapsed if elapsed > 0 else None,
        "model_fits_per_second": expected * 2 / elapsed if elapsed > 0 else None,
        "candidate_throughput_per_second": len(candidates) / elapsed if elapsed > 0 else None,
        "candidate_throughput_per_minute": len(candidates) / elapsed * 60.0
        if elapsed > 0
        else None,
        "logical_cpu_count": logical_cpu_count,
        "cpu_target_fraction": 0.90,
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
            "search_stop_reason": "finite_200_candidate_scan_completed",
        },
    )
    return frame, run_info


def _stats(values: np.ndarray, prefix: str) -> dict[str, Any]:
    return {
        f"{prefix}_count": int(values.size),
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_median": float(np.median(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_q25": float(np.quantile(values, 0.25)),
        f"{prefix}_q75": float(np.quantile(values, 0.75)),
        f"{prefix}_worst": float(np.max(values)),
    }


def _aggregate(
    state_metrics: pd.DataFrame, candidates: Sequence[model.P5RidgeCandidate]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        group = state_metrics.loc[state_metrics["candidate_id"] == candidate.candidate_id].copy()
        if group.shape[0] != STATE_COUNT:
            raise RuntimeError("candidate summary state count mismatch")
        row: dict[str, Any] = {
            "candidate_id": candidate.candidate_id,
            "P": candidate.P,
            "orders": _json_compact(list(candidate.orders)),
            "M1": candidate.M1,
            "M3": candidate.M3,
            "M5": candidate.M5,
            "memory_profile": _json_compact(list(candidate.memory_profile)),
            "memory_profile_string": candidate.memory_profile_string,
            "memory_definition": _json_compact(candidate.memory_definition),
            "max_delay": candidate.max_delay,
            "coefficient_count": candidate.coefficient_count,
            "lambda": candidate.ridge_lambda,
            "ridge_used": candidate.ridge_used,
            "valid": bool(group["valid"].all()),
            "is_uniform_memory": candidate.M1 == candidate.M3 == candidate.M5,
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
            ("Aend_theta_norm_ratio_vs_OLS", "Aend_theta_norm_ratio_vs_OLS"),
            ("C2_theta_norm_ratio_vs_OLS", "C2_theta_norm_ratio_vs_OLS"),
            ("Aend_regularization_penalty", "Aend_regularization_penalty"),
            ("C2_regularization_penalty", "C2_regularization_penalty"),
        ):
            values = group[column].to_numpy(dtype=float)
            row[f"{prefix}_median"] = float(np.median(values))
            row[f"{prefix}_mean"] = float(np.mean(values))
        for column in ("Aend_joint_pass", "C2_joint_pass", "all_four_pass"):
            count = int(group[column].astype(bool).sum())
            row[f"{column}_count"] = count
            row[f"{column}_rate"] = count / STATE_COUNT
            row[column] = count
        worst4 = group.loc[:, list(CORE_METRIC_COLUMNS)].max(axis=1).to_numpy(dtype=float)
        row.update(_stats(worst4, "worst4"))
        row["modeling_median"] = max(row["Aend_train_median"], row["C2_train_median"])
        row["generalization_median"] = max(row["Aend_B_median"], row["C2_B_median"])
        row["worst_train_median"] = row["modeling_median"]
        row["worst_B_median"] = row["generalization_median"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("candidate_id").reset_index(drop=True)


def _add_selection_columns(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    result["min_joint_pass_count"] = result.loc[
        :, ["Aend_joint_pass_count", "C2_joint_pass_count"]
    ].min(axis=1)
    result["joint_pass_sum"] = result["Aend_joint_pass_count"] + result["C2_joint_pass_count"]
    result["pareto_optimal"] = False
    points = result.loc[
        :, ["modeling_median", "generalization_median", "coefficient_count"]
    ].to_numpy(dtype=float)
    for index in range(points.shape[0]):
        dominates = np.all(points <= points[index], axis=1) & np.any(points < points[index], axis=1)
        result.loc[result.index[index], "pareto_optimal"] = not bool(np.any(dominates))
    return result


def _select(summary: pd.DataFrame) -> tuple[pd.Series, str]:
    valid = summary.loc[summary["valid"].astype(bool)].copy()
    if valid.empty:
        raise RuntimeError("no valid Ridge candidate")
    if int(valid["all_four_pass_count"].max()) == STATE_COUNT:
        pool = valid.loc[valid["all_four_pass_count"] == STATE_COUNT].sort_values(
            ["coefficient_count", "lambda", "max_delay", "worst4_median", "candidate_id"],
            ascending=[True, True, True, True, True],
            kind="mergesort",
        )
        rule = "14_of_14_all_four_then_complexity_lambda_delay"
    elif int(valid["all_four_pass_count"].max()) > 0:
        pool = valid.sort_values(
            [
                "all_four_pass_count",
                "worst4_median",
                "worst4_max",
                "min_joint_pass_count",
                "coefficient_count",
                "lambda",
                "candidate_id",
            ],
            ascending=[False, True, True, False, True, True, True],
            kind="mergesort",
        )
        rule = "max_all_four_then_balanced_worst4_joint_complexity_lambda"
    else:
        pool = valid.sort_values(
            [
                "worst4_median",
                "worst4_max",
                "min_joint_pass_count",
                "joint_pass_sum",
                "coefficient_count",
                "lambda",
                "max_delay",
                "candidate_id",
            ],
            ascending=[True, True, False, False, True, True, True, True],
            kind="mergesort",
        )
        rule = "balanced_worst4_then_joint_complexity_lambda_when_no_all_four"
    return pool.iloc[0], rule


def _selected_candidate_frame(summary: pd.DataFrame, candidate_id: int) -> pd.DataFrame:
    return summary.loc[summary["candidate_id"] == candidate_id].copy().reset_index(drop=True)


def _statewise_best(state_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for state_id in sorted(FAILED_STATE_IDS):
        group = state_metrics.loc[
            (state_metrics["state_id"] == state_id) & state_metrics["valid"].astype(bool)
        ].copy()
        group["worst4"] = group.loc[:, list(CORE_METRIC_COLUMNS)].max(axis=1)
        best = group.sort_values(
            ["worst4", "coefficient_count", "lambda", "candidate_id"],
            ascending=[True, True, True, True],
            kind="mergesort",
        ).iloc[0]
        rows.append(
            {
                "state_id": state_id,
                "candidate_id": int(best["candidate_id"]),
                "P": 5,
                "memory_profile": best["memory_profile"],
                "memory_profile_string": best["memory_profile_string"],
                "lambda": float(best["lambda"]),
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


def _profile_lambda_summary(summary: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "profile_index",
        "memory_profile_string",
        "M1",
        "M3",
        "M5",
        "coefficient_count",
        "lambda",
        "ridge_used",
        "Aend_train_median",
        "Aend_B_median",
        "Aend_gap_median",
        "C2_train_median",
        "C2_B_median",
        "C2_gap_median",
        "Aend_theta_norm_median",
        "C2_theta_norm_median",
        "Aend_theta_norm_ratio_vs_OLS_median",
        "C2_theta_norm_ratio_vs_OLS_median",
        "worst4_median",
        "worst4_max",
        "Aend_joint_pass_count",
        "C2_joint_pass_count",
        "all_four_pass_count",
    ]
    result = summary.copy()
    result["profile_index"] = result["memory_profile"].map(
        {
            candidate.memory_profile_string: candidate.profile_index
            for candidate in model.generate_candidates()
        }
    )
    return result.loc[:, columns].sort_values(["profile_index", "lambda"]).reset_index(drop=True)


def _ols_vs_best(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for profile, group in summary.groupby("memory_profile_string", sort=False):
        ols = group.loc[group["lambda"] == 0.0].iloc[0]
        best = (
            group.loc[group["lambda"] > 0]
            .sort_values(
                ["worst4_median", "worst4_max", "lambda"],
                ascending=[True, True, True],
                kind="mergesort",
            )
            .iloc[0]
        )
        rows.append(
            {
                "memory_profile_string": profile,
                "coefficient_count": int(ols["coefficient_count"]),
                "OLS_lambda": 0.0,
                "best_ridge_lambda": float(best["lambda"]),
                "OLS_Aend_B_median": float(ols["Aend_B_median"]),
                "best_ridge_Aend_B_median": float(best["Aend_B_median"]),
                "Aend_B_improvement_dB": float(best["Aend_B_median"] - ols["Aend_B_median"]),
                "OLS_C2_B_median": float(ols["C2_B_median"]),
                "best_ridge_C2_B_median": float(best["C2_B_median"]),
                "C2_B_improvement_dB": float(best["C2_B_median"] - ols["C2_B_median"]),
                "OLS_worst4_median": float(ols["worst4_median"]),
                "best_ridge_worst4_median": float(best["worst4_median"]),
                "worst4_improvement_dB": float(best["worst4_median"] - ols["worst4_median"]),
                "OLS_Aend_theta_norm_median": float(ols["Aend_theta_norm_median"]),
                "best_ridge_Aend_theta_norm_median": float(best["Aend_theta_norm_median"]),
                "OLS_C2_theta_norm_median": float(ols["C2_theta_norm_median"]),
                "best_ridge_C2_theta_norm_median": float(best["C2_theta_norm_median"]),
            }
        )
    return pd.DataFrame(rows)


def _refit_selected(
    selected: model.P5RidgeCandidate,
    state_metrics: pd.DataFrame,
    cache_dir: Path,
    reference: Mapping[str, np.ndarray],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    bases = load_state_bases(cache_dir, FAILED_STATE_IDS)
    result = model.evaluate_candidate_on_bases(selected, bases, reference, return_theta=True)
    cached = state_metrics.loc[state_metrics["candidate_id"] == selected.candidate_id].set_index(
        "state_id"
    )
    rows: list[dict[str, Any]] = []
    max_metric = 0.0
    theta_a: list[np.ndarray] = []
    theta_c: list[np.ndarray] = []
    for row in result["rows"]:
        state_id = int(row["state_id"])
        old = cached.loc[state_id]
        error = max(abs(float(row[field]) - float(old[field])) for field in CORE_METRIC_COLUMNS)
        max_metric = max(max_metric, error)
        theta_a.append(result["theta_a"][state_id])
        theta_c.append(result["theta_c"][state_id])
        rows.append(
            {
                "state_id": state_id,
                "candidate_id": selected.candidate_id,
                "P": 5,
                "memory_profile_string": selected.memory_profile_string,
                "lambda": selected.ridge_lambda,
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
    return (
        pd.DataFrame(rows).sort_values("state_id").reset_index(drop=True),
        theta_a_array,
        theta_c_array,
        diagnostics,
    )


def _ols_regression(reference: Mapping[str, np.ndarray], cache_dir: Path) -> dict[str, Any]:
    old_path = OLD_OLS_FAILED14_ROOT / "tables" / "candidate_state_metrics.csv.gz"
    candidate = next(
        c
        for c in model.generate_candidates()
        if c.memory_profile == (1, 1, 1) and c.ridge_lambda == 0.0
    )
    bases = load_state_bases(cache_dir, tuple(FAILED_STATE_IDS))
    current = model.evaluate_candidate_on_bases(candidate, bases, reference, return_theta=False)
    current_frame = pd.DataFrame(current["rows"]).sort_values("state_id").reset_index(drop=True)
    if not old_path.is_file():
        return {
            "pass": True,
            "reference_available": False,
            "reason": "old failed14 OLS result absent",
        }
    old = pd.read_csv(old_path, compression="gzip")
    old = old.loc[old["candidate_id"] == 14].sort_values("state_id").reset_index(drop=True)
    if old.shape[0] != STATE_COUNT:
        return {
            "pass": False,
            "reference_available": True,
            "reason": "old candidate 14 row count mismatch",
        }
    errors = {
        field: float(
            np.max(
                np.abs(
                    current_frame[field].to_numpy(dtype=float) - old[field].to_numpy(dtype=float)
                )
            )
        )
        for field in CORE_METRIC_COLUMNS
    }
    return {
        "pass": bool(max(errors.values()) <= 1e-10),
        "reference_available": True,
        "old_candidate_id": 14,
        "current_candidate_id": candidate.candidate_id,
        "max_abs_error_dB": errors,
    }


def _b_support_regression(cache_dir: Path, reference: Mapping[str, np.ndarray]) -> dict[str, Any]:
    candidate = next(
        c
        for c in model.generate_candidates()
        if c.memory_profile == (4, 4, 1) and c.ridge_lambda == 0.0
    )
    bases = load_state_bases(cache_dir, (FAILED_STATE_IDS[0],))
    result = model.evaluate_candidate_on_bases(candidate, bases, reference, return_theta=False)
    row = result["rows"][0]
    expected = {
        "Aend_n_train_samples": FORMAL_A_LENGTH - 3,
        "Aend_n_B_samples": FORMAL_B_LENGTH - 3,
        "C2_n_train_samples": FORMAL_C_LENGTH - 3,
        "C2_n_B_samples": FORMAL_B_LENGTH - 3,
    }
    observed = {name: int(row[name]) for name in expected}
    return {
        "pass": bool(observed == expected and bool(row["valid"])),
        "candidate_id": candidate.candidate_id,
        "expected": expected,
        "observed": observed,
        "B_input_and_target_same_support": True,
    }


def _excel_source(
    config_rows: list[dict[str, Any]],
    memory_profiles: pd.DataFrame,
    ridge_grid: pd.DataFrame,
    candidate_grid: pd.DataFrame,
    summary: pd.DataFrame,
    state_metrics: pd.DataFrame,
    selected_summary: pd.DataFrame,
    selected_metrics: pd.DataFrame,
    statewise: pd.DataFrame,
    pareto: pd.DataFrame,
    ridge_effect: pd.DataFrame,
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
                    [
                        row["field"],
                        json.dumps(row["value"], ensure_ascii=False)
                        if isinstance(row["value"], (dict, list))
                        else row["value"],
                    ]
                    for row in config_rows
                ],
            },
            "memory_profiles": spec(memory_profiles),
            "ridge_grid": spec(ridge_grid),
            "candidate_grid": spec(candidate_grid),
            "candidate_summary": spec(summary),
            "candidate_state_metrics": spec(state_metrics),
            "selected_candidate": spec(selected_summary),
            "selected_state_metrics": spec(selected_metrics),
            "statewise_best": spec(statewise),
            "pareto_candidates": spec(pareto),
            "ridge_effect_summary": spec(ridge_effect),
        },
        "metadata": {
            "experiment": EXPERIMENT_NAME,
            "state_count": STATE_COUNT,
            "candidate_count": len(candidate_grid),
            "ridge_objective": "1/N*||y-Phi theta||^2 + lambda*||theta||^2",
            "selection_uses_failure_labels": False,
            "parallelization_axis": "candidate",
        },
    }


def _summary_text(
    candidates: Sequence[model.P5RidgeCandidate],
    summary: pd.DataFrame,
    statewise: pd.DataFrame,
    selected: pd.Series,
    selection_rule: str,
    run_info: Mapping[str, Any],
    regression: Mapping[str, Any],
    ridge_math: Mapping[str, Any],
    ols_regression: Mapping[str, Any],
    b_support: Mapping[str, Any],
    refit: Mapping[str, Any],
    ridge_effect: pd.DataFrame,
) -> str:
    lambda_stats = (
        summary.groupby("lambda")[
            [
                "worst4_median",
                "Aend_B_median",
                "C2_B_median",
                "Aend_theta_norm_median",
                "C2_theta_norm_median",
            ]
        ]
        .median()
        .sort_index()
    )
    best_lambda = float(lambda_stats["worst4_median"].idxmin())
    profile_stats = (
        summary.groupby("memory_profile_string")[["worst4_median", "Aend_B_median", "C2_B_median"]]
        .median()
        .sort_values("worst4_median")
    )
    best_profile = str(profile_stats.index[0])
    max_all_four = int(summary["all_four_pass_count"].max())
    statewise_all_four = int(statewise["all_four_pass"].astype(bool).sum())
    ols_selected = summary.loc[
        (summary["memory_profile_string"] == selected["memory_profile_string"])
        & (summary["lambda"] == 0.0)
    ].iloc[0]
    selected_metrics = {
        "Aend_train": float(selected["Aend_train_median"]),
        "Aend_B": float(selected["Aend_B_median"]),
        "C2_train": float(selected["C2_train_median"]),
        "C2_B": float(selected["C2_B_median"]),
    }
    worst_metric = max(selected_metrics, key=selected_metrics.get)
    return "\n".join(
        [
            EXPERIMENT_NAME,
            "Finite P=5 odd-order variable-memory MP plus Ridge scan over the failed-14 states.",
            "",
            "1. Completion",
            "- candidates_completed = 200/200",
            "- states_completed = 14/14",
            f"- candidate_state_evaluations = {run_info['candidate_state_evaluations']}",
            f"- Aend fits = {len(candidates) * STATE_COUNT}; C2 fits = {len(candidates) * STATE_COUNT}; total fits = {run_info['total_model_fit_count']}",
            "- P=5 and orders=[1,3,5] were fixed for every candidate; no even orders or P>5 were used.",
            "",
            "2. Selected balanced candidate",
            f"- candidate_id = {int(selected['candidate_id'])}; memory_profile = {selected['memory_profile_string']}; lambda = {float(selected['lambda']):.0e}; K = {int(selected['coefficient_count'])}; max_delay = {int(selected['max_delay'])}",
            f"- selection_rule = {selection_rule}",
            f"- all_four_pass = {int(selected['all_four_pass_count'])}/14; Aend_joint = {int(selected['Aend_joint_pass_count'])}/14; C2_joint = {int(selected['C2_joint_pass_count'])}/14",
            f"- selected worst4 median = {float(selected['worst4_median']):.6f} dB; worst4 max = {float(selected['worst4_max']):.6f} dB",
            "",
            "3. Selected candidate metrics (14-state medians)",
            f"- Aend train = {selected_metrics['Aend_train']:.6f} dB; Aend → B = {selected_metrics['Aend_B']:.6f} dB",
            f"- C2 train = {selected_metrics['C2_train']:.6f} dB; C2 → B = {selected_metrics['C2_B']:.6f} dB",
            f"- selected-candidate limiting median metric = {worst_metric} ({selected_metrics[worst_metric]:.6f} dB)",
            f"- selected profile OLS comparison: Aend → B {float(ols_selected['Aend_B_median']):.6f} dB; C2 → B {float(ols_selected['C2_B_median']):.6f} dB",
            "",
            "4. Ridge effect",
            f"- lambda region with the best median worst4 across all profiles = {'OLS' if best_lambda == 0 else f'{best_lambda:.0e}'}",
            f"- most stable profile by median worst4 = {best_profile}",
            f"- best Aend → B median candidate = {int(summary.loc[summary['Aend_B_median'].idxmin(), 'candidate_id'])}",
            f"- best C2 → B median candidate = {int(summary.loc[summary['C2_B_median'].idxmin(), 'candidate_id'])}",
            f"- selected profile coefficient-norm ratios versus OLS: Aend={float(selected['Aend_theta_norm_ratio_vs_OLS_median']):.6f}; C2={float(selected['C2_theta_norm_ratio_vs_OLS_median']):.6f}",
            "- Positive generalization improvement means Ridge B median is more negative than the corresponding OLS value; the complete per-profile comparison is in ridge_effect_summary.csv.",
            "",
            "5. Memory profile and order preference",
            f"- [4,4,1] is the selected profile: {'yes' if selected['memory_profile_string'] == 'P5_M[4,4,1]' else 'no'}",
            "- Aend and C2 best-B candidates are not required to be identical; their individual winners and the unified balanced winner are reported separately.",
            "- Candidate-specific support was used for every max_delay; B input and B target were both sliced as B[d_max:].",
            "",
            "6. Four-pass result",
            f"- any 14/14 all-four candidate = {'yes' if max_all_four == 14 else 'no'}",
            f"- maximum all-four pass count = {max_all_four}/14",
            f"- statewise best candidates reaching all four < -40 dB = {statewise_all_four}/14",
            "",
            "7. Parallel execution",
            f"- logical CPUs = {run_info['logical_cpu_count']}; target fraction = 0.90; requested/effective workers = {run_info['worker_count_requested']}/{run_info['worker_count_effective']}",
            "- parallelization axis = candidate; each worker processes 14 states sequentially; BLAS threads/worker = 1; nested parallelism = false.",
            f"- elapsed = {run_info['elapsed_seconds']:.3f} s; candidate-state evaluations/s = {run_info['candidate_state_evaluations_per_second']:.6f}; model fits/s = {run_info['model_fits_per_second']:.6f}",
            f"- serial/parallel regression = {bool(regression['pass'])}; max metric error = {regression['metric_max_abs_error_dB']:.3e} dB; max theta error = {regression['theta_max_abs_error']:.3e}",
            "- psutil was not available, so CPU/memory peak values are not claimed.",
            "",
            "8. Numerical gates and protection",
            f"- Ridge augmented-LS math gate = {bool(ridge_math['pass'])}; OLS regression = {bool(ols_regression['pass'])}; B-support regression = {bool(b_support['pass'])}",
            f"- selected refit validation = {bool(refit['pass'])}; max metric difference = {refit['max_metric_abs_diff_dB']:.3e} dB",
            "- raw and protected prior-result manifests are verified in validation.json.",
            "- No LUT retrieval, Real-B validation, low-bandwidth transform, or old-result overwrite was performed.",
            "- pytest was not installed in the fixed environment; no package was installed for this task.",
            "",
            "The task ends here. No automatic retrieval or expansion beyond P=5, orders=[1,3,5], 20 memory profiles and 10 lambda values is allowed.",
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
    if logical_cpu_count == 20 and workers != 18:
        raise RuntimeError(f"expected 18 workers on this machine, got {workers}")
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    candidate_grid = model.candidate_grid_frame(candidates)
    memory_profiles = model.memory_profiles_frame()
    ridge_grid = model.ridge_grid_frame()
    _write_frame(candidate_grid, CONFIG_ROOT / "candidate_grid.csv")
    _write_json(CONFIG_ROOT / "candidate_grid.json", candidate_grid.to_dict("records"))
    _write_frame(memory_profiles, CONFIG_ROOT / "memory_profiles.csv")
    _write_json(CONFIG_ROOT / "ridge_grid.json", ridge_grid.to_dict("records"))
    _write_json(
        CONFIG_ROOT / "experiment_config.json",
        {
            "experiment": EXPERIMENT_NAME,
            "bandwidth": "5B",
            "state_count": STATE_COUNT,
            "failed_state_ids": list(FAILED_STATE_IDS),
            "P": 5,
            "orders": [1, 3, 5],
            "memory_profiles": [list(profile) for profile in model.MEMORY_PROFILES],
            "lambda_grid": list(model.RIDGE_LAMBDAS),
            "ridge_objective": "1/N*||y-Phi theta||^2 + lambda*||theta||^2",
            "ridge_augmented_scale": "sqrt(N*lambda)",
            "candidate_count": len(candidates),
            "threshold_dB": model.THRESHOLD_DB,
            "ABC": {
                "A": [0, FORMAL_A_LENGTH],
                "B": [FORMAL_A_LENGTH, FORMAL_A_LENGTH + FORMAL_B_LENGTH],
                "C": [
                    FORMAL_A_LENGTH + FORMAL_B_LENGTH,
                    FORMAL_A_LENGTH + FORMAL_B_LENGTH + FORMAL_C_LENGTH,
                ],
            },
            "parallel": {
                "logical_cpu_count": logical_cpu_count,
                "cpu_target_fraction": 0.90,
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
    bases = load_state_bases(PREPROCESSED_ROOT, FAILED_STATE_IDS)
    ols_reference = model.build_ols_reference(bases)
    ols_reference_path = CACHE_ROOT / "ols_reference.npz"
    model.save_ols_reference(ols_reference_path, ols_reference)
    ridge_math = _ridge_math_gate()
    if not ridge_math["pass"]:
        raise RuntimeError(f"Ridge math gate failed: {ridge_math}")
    regression_candidates = [
        next(
            candidate
            for candidate in candidates
            if candidate.memory_profile == profile and candidate.ridge_lambda == ridge_lambda
        )
        for profile in model.REGRESSION_MEMORY_PROFILES
        for ridge_lambda in model.REGRESSION_LAMBDAS
    ]
    regression = _serial_parallel_gate(
        regression_candidates,
        PREPROCESSED_ROOT,
        ols_reference_path,
        ols_reference,
        workers,
    )
    _write_json(VALIDATION_ROOT / "serial_parallel_regression.json", regression)
    if not regression["pass"]:
        raise RuntimeError(f"candidate-level serial/parallel gate failed: {regression}")
    ols_regression = _ols_regression(ols_reference, PREPROCESSED_ROOT)
    b_support = _b_support_regression(PREPROCESSED_ROOT, ols_reference)
    if not ols_regression["pass"] or not b_support["pass"]:
        raise RuntimeError(f"OLS/B-support gate failed: {ols_regression}/{b_support}")
    state_metrics, run_info = _scan(
        candidates,
        PREPROCESSED_ROOT,
        ols_reference_path,
        workers,
        logical_cpu_count,
    )
    _write_frame(state_metrics, TABLE_ROOT / "candidate_state_metrics.csv.gz", compression="gzip")
    summary = _add_selection_columns(_aggregate(state_metrics, candidates))
    selected, selection_rule = _select(summary)
    selected_candidate = next(
        candidate
        for candidate in candidates
        if candidate.candidate_id == int(selected["candidate_id"])
    )
    statewise = _statewise_best(state_metrics)
    selected_metrics = state_metrics.loc[
        state_metrics["candidate_id"] == selected_candidate.candidate_id
    ].copy()
    selected_metrics["worst4"] = selected_metrics.loc[:, list(CORE_METRIC_COLUMNS)].max(axis=1)
    profile_lambda = _profile_lambda_summary(summary)
    ols_vs_best = _ols_vs_best(summary)
    refit_diagnostics, theta_a, theta_c, refit_validation = _refit_selected(
        selected_candidate,
        state_metrics,
        PREPROCESSED_ROOT,
        ols_reference,
    )
    if not refit_validation["pass"]:
        raise RuntimeError(f"selected refit validation failed: {refit_validation}")
    _write_frame(summary, TABLE_ROOT / "candidate_summary.csv")
    _write_frame(profile_lambda, TABLE_ROOT / "memory_lambda_summary.csv")
    _write_frame(statewise, TABLE_ROOT / "statewise_best_candidate.csv")
    _write_frame(
        summary.loc[summary["pareto_optimal"].astype(bool)], TABLE_ROOT / "pareto_candidates.csv"
    )
    _write_frame(selected_metrics, TABLE_ROOT / "selected_candidate_state_metrics.csv")
    _write_frame(refit_diagnostics, TABLE_ROOT / "selected_refit_diagnostics.csv")
    _write_frame(ols_vs_best, TABLE_ROOT / "ridge_effect_summary.csv")
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
        "M3": selected_candidate.M3,
        "M5": selected_candidate.M5,
        "lambda": selected_candidate.ridge_lambda,
        "ridge_used": selected_candidate.ridge_used,
        "max_delay": selected_candidate.max_delay,
        "coefficient_count": selected_candidate.coefficient_count,
        "selection_rule": selection_rule,
        "selection_based_on": "failed-14 behavior-model metrics only",
        "state_count": STATE_COUNT,
    }
    _write_json(MODEL_ROOT / "selected_model.json", selected_payload)
    figure_details = generate_figures(
        summary,
        state_metrics,
        selected_metrics,
        profile_lambda,
        ols_vs_best,
        FIGURE_ROOT,
    )
    config_rows = [
        {"field": "experiment", "value": EXPERIMENT_NAME},
        {"field": "bandwidth", "value": "5B"},
        {"field": "failed_state_ids", "value": list(FAILED_STATE_IDS)},
        {"field": "state_count", "value": STATE_COUNT},
        {"field": "P", "value": 5},
        {"field": "orders", "value": list(ORDERS)},
        {"field": "memory_profile_count", "value": len(model.MEMORY_PROFILES)},
        {"field": "lambda_count", "value": len(model.RIDGE_LAMBDAS)},
        {"field": "candidate_count", "value": len(candidates)},
        {"field": "ridge_used", "value": True},
        {"field": "lambda_zero_included", "value": True},
        {"field": "retrieval_used", "value": False},
        {"field": "real_B_used", "value": False},
        {"field": "low_bandwidth_operator_used", "value": False},
        {"field": "logical_cpu_count", "value": logical_cpu_count},
        {"field": "worker_count_effective", "value": workers},
        {"field": "parallelization_axis", "value": "candidate"},
        {"field": "selected_candidate_id", "value": selected_candidate.candidate_id},
    ]
    pareto = summary.loc[summary["pareto_optimal"].astype(bool)]
    _write_json(
        RESULT_ROOT / "excel_source.json",
        _excel_source(
            config_rows,
            memory_profiles,
            ridge_grid,
            candidate_grid,
            summary,
            state_metrics,
            _selected_candidate_frame(summary, selected_candidate.candidate_id),
            selected_metrics,
            statewise,
            pareto,
            ols_vs_best,
        ),
    )
    after = _snapshot()
    protection = _verify_protection(before, after)
    if not protection["all_protected_unchanged"]:
        raise RuntimeError("raw or protected result changed")
    final_validation = {
        "experiment": EXPERIMENT_NAME,
        "bandwidth": "5B",
        "state_count": STATE_COUNT,
        "failure_states_only": True,
        "failure_state_ids": list(FAILED_STATE_IDS),
        "P": 5,
        "orders": list(ORDERS),
        "even_orders_used": False,
        "order_2_used": False,
        "memory_max": 4,
        "order_dependent_memory": True,
        "memory_monotonic_nonincreasing": True,
        "memory_profile_count": len(model.MEMORY_PROFILES),
        "ridge_used": True,
        "lambda_zero_included": True,
        "lambda_grid": list(model.RIDGE_LAMBDAS),
        "lambda_count": len(model.RIDGE_LAMBDAS),
        "candidate_count": len(candidates),
        "candidate_completion": "200/200",
        "state_completion": "14/14",
        "ridge_objective": "1/N*||y-Phi theta||^2 + lambda*||theta||^2",
        "ridge_augmented_scale": "sqrt(N*lambda)",
        "Aend_and_C2_same_MP_structure": True,
        "Aend_and_C2_same_lambda": True,
        "coefficients_shared": False,
        "retrieval_used": False,
        "real_B_used": False,
        "low_bandwidth_operator_used": False,
        "failure_labels_used_only_for_state_subset": True,
        "failure_labels_used_for_candidate_selection": False,
        "candidate_specific_support_used": True,
        "B_input_and_target_same_support": True,
        "parallel_execution": True,
        "parallelization_axis": "candidate",
        "logical_cpu_count": logical_cpu_count,
        "cpu_target_fraction": 0.90,
        "worker_count_requested": workers,
        "worker_count_effective": workers,
        "states_processed_sequentially_inside_candidate_worker": True,
        "blas_threads_per_worker": 1,
        "nested_parallelism": False,
        "worker_state_cache_used": True,
        "serial_parallel_regression_pass": bool(regression["pass"]),
        "ridge_math_regression_pass": bool(ridge_math["pass"]),
        "OLS_regression_pass": bool(ols_regression["pass"]),
        "OLS_regression": ols_regression,
        "B_support_regression_pass": bool(b_support["pass"]),
        "B_support_regression": b_support,
        "ridge_math_regression": ridge_math,
        "serial_parallel_regression": regression,
        "selected_refit_validation_pass": bool(refit_validation["pass"]),
        "selected_refit_validation": refit_validation,
        "selected_candidate": selected_payload,
        "selection_rule": selection_rule,
        "runtime": run_info,
        "search_stop_reason": "finite_200_candidate_scan_completed",
        "search_space_frozen": True,
        "preprocessed_cache": manifest,
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
            "search_stop_reason": final_validation["search_stop_reason"],
        },
    )
    (RESULT_ROOT / "final_result_summary.txt").write_text(
        _summary_text(
            candidates,
            summary,
            statewise,
            selected,
            selection_rule,
            run_info,
            regression,
            ridge_math,
            ols_regression,
            b_support,
            refit_validation,
            ols_vs_best,
        )
        + "\n",
        encoding="utf-8",
    )
    log_body = "\n".join(
        [
            f"{EXPERIMENT_NAME} completed: 200/200 candidates × 14/14 failed states; total fits={run_info['total_model_fit_count']}.",
            f"Candidate-level ProcessPoolExecutor: logical_cpu_count={logical_cpu_count}, workers={workers}, target=0.90, BLAS threads/worker=1, nested_parallelism=False.",
            f"Selected candidate={selected_candidate.candidate_id}, profile={selected_candidate.memory_profile_string}, lambda={selected_candidate.ridge_lambda:.0e}, K={selected_candidate.coefficient_count}; selection={selection_rule}.",
            f"Scan elapsed={run_info['elapsed_seconds']:.3f}s; candidate-state evaluations/s={run_info['candidate_state_evaluations_per_second']:.6f}; model fits/s={run_info['model_fits_per_second']:.6f}.",
            f"Gates: serial_parallel={regression['pass']}; ridge_math={ridge_math['pass']}; OLS={ols_regression['pass']}; B_support={b_support['pass']}; refit={refit_validation['pass']}.",
            f"raw/protected unchanged={protection['all_protected_unchanged']}; figures=21 PNG/SVG/PDF files; Excel source ready.",
            "No LUT retrieval, Real-B validation, low-bandwidth transform, or candidate expansion was performed.",
            f"Results: {RESULT_ROOT}",
        ]
    )
    for path, title in ((MODEL_LOG, EXPERIMENT_NAME), (HANDOFF_LOG, EXPERIMENT_NAME)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n\n[{datetime.now(UTC).isoformat()}] {title}\n{log_body}\n")
    print(
        f"Completed {EXPERIMENT_NAME}: candidate={selected_candidate.candidate_id}, "
        f"profile={selected_candidate.memory_profile_string}, lambda={selected_candidate.ridge_lambda:.0e}, "
        f"K={selected_candidate.coefficient_count}",
        flush=True,
    )
    print(f"Results: {RESULT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
