"""Run the frozen failed-14 order-dependent-memory MP experiment.

The experiment is intentionally finite: 14 states and exactly 125 compact
odd-order candidates.  The numerical axis is Candidate-level multiprocessing;
each worker loads the read-only preprocessed state cache once and evaluates one
candidate across all 14 states.  No retrieval, Real-B validation, low-bandwidth
operator, or Ridge path is imported or executed.
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

from behavior_modeling.shared.odd_order_variable_memory_scan import (  # noqa: E402
    ALL_STATE_COUNT,
    FAILED_STATE_IDS,
    FORMAL_A_LENGTH,
    FORMAL_B_LENGTH,
    FORMAL_C_LENGTH,
    REGRESSION_PROFILES,
    REGRESSION_STATE_IDS,
    STATE_COUNT,
    THRESHOLD_DB,
    VariableMemoryCandidate,
    candidate_grid_frame,
    evaluate_candidate_on_bases,
    generate_candidates,
    load_prepared_state,
    load_state_bases,
    prepare_failed_state,
    save_prepared_state,
    target_worker_count,
    validate_candidate_grid,
)
from behavior_modeling.scenario_2.scenario_2_failed14_odd_order_mp_order_dependent_memory_scan_5b.plot_scenario2_failed14_odd_order_mp_order_dependent_memory_scan_5b import (  # noqa: E402
    generate_figures,
)

TASK_NAME = "scenario_2_failed14_odd_order_mp_order_dependent_memory_scan_5b"
EXPERIMENT_NAME = "scenario_2_failed14_odd_order_mp_order_dependent_memory_scan_5B"
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
    "statewise_best_round0_4": PROJECT_ROOT
    / "results"
    / "behavior_modeling"
    / "scenario_2" /"scenario_2_statewise_best_round0_4_5B",
    "old_unified_odd_order_scan": PROJECT_ROOT
    / "results"
    / "behavior_modeling"
    / "scenario_2" /"scenario_2_unified_odd_order_mp_capacity_scan_5B",
}

METRIC_NAMES = (
    "Aend_train",
    "Aend_B",
    "Aend_gap",
    "C2_train",
    "C2_B",
    "C2_gap",
)
CORE_METRIC_COLUMNS = (
    "Aend_train_NMSE_dB",
    "Aend_B_NMSE_dB",
    "C2_train_NMSE_dB",
    "C2_B_NMSE_dB",
)


def _json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


_WORKER_BASES: dict[int, Any] = {}


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
    total_bytes = 0
    for file in files:
        size = int(file.stat().st_size)
        digest.update(str(file.relative_to(raw_root)).replace("\\", "/").encode("utf-8") + b"\0")
        digest.update(size.to_bytes(8, "little"))
        total_bytes += size
    return {"sha256": digest.hexdigest(), "file_count": len(files), "bytes": total_bytes}


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
    raw_before = before["data/raw"]
    raw_after = after["data/raw"]
    raw = {
        "sha256_unchanged": raw_before["sha256"] == raw_after["sha256"],
        "file_count_unchanged": raw_before["file_count"] == raw_after["file_count"],
        "bytes_unchanged": raw_before["bytes"] == raw_after["bytes"],
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
    if not FAILURE_SOURCE.is_file():
        raise FileNotFoundError(f"formal failure-state list is missing: {FAILURE_SOURCE}")
    frame = pd.read_csv(FAILURE_SOURCE)
    if "State_n_R" not in frame.columns or "failure" not in frame.columns:
        raise ValueError("formal failure-state list lacks State_n_R/failure")
    ids = tuple(
        sorted(int(value) for value in frame.loc[frame["failure"].astype(bool), "State_n_R"])
    )
    if ids != tuple(FAILED_STATE_IDS):
        raise RuntimeError(f"failure-state list mismatch: {ids} vs {FAILED_STATE_IDS}")
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
    maximum_probe_difference = 0.0
    for position, state_id in enumerate(FAILED_STATE_IDS, start=1):
        path = PREPROCESSED_ROOT / f"state_{state_id:03d}.npz"
        if path.is_file():
            prepared = load_prepared_state(path)
        else:
            prepared = prepare_failed_state(state_id)
            save_prepared_state(prepared, path)
        if prepared.state_id != state_id:
            raise RuntimeError(f"prepared cache state mismatch for {state_id}")
        if common_probe is None:
            common_probe = prepared.common_b_input.copy()
        else:
            difference = float(np.max(np.abs(prepared.common_b_input - common_probe)))
            maximum_probe_difference = max(maximum_probe_difference, difference)
            if difference > 1e-12:
                raise RuntimeError(f"common B probe differs for state {state_id}: {difference}")
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
        print(f"preprocessed failed state {position}/{STATE_COUNT}: state={state_id}", flush=True)
    if common_probe is None:
        raise RuntimeError("no common B probe was prepared")
    manifest = {
        "state_ids": list(FAILED_STATE_IDS),
        "state_count": STATE_COUNT,
        "common_b_input_length": int(common_probe.size),
        "common_b_max_abs_difference": maximum_probe_difference,
        "common_b_probe_consistent": bool(maximum_probe_difference <= 1e-12),
        "ABC_lengths": {
            "A": FORMAL_A_LENGTH,
            "B": FORMAL_B_LENGTH,
            "C": FORMAL_C_LENGTH,
        },
        "records": records,
    }
    _write_json(CACHE_ROOT / "preprocessed_manifest.json", manifest)
    _write_npz(CACHE_ROOT / "common_B_probe.npz", {"common_b_input": common_probe})
    return manifest


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
    _WORKER_BASES = load_state_bases(Path(cache_dir), state_ids)


def _worker_entry(candidate: VariableMemoryCandidate) -> dict[str, Any]:
    if not _WORKER_BASES:
        raise RuntimeError("worker state cache was not initialized")
    return evaluate_candidate_on_bases(candidate, _WORKER_BASES, return_theta=True)


def _parallel_candidate_results(
    candidates: Sequence[VariableMemoryCandidate],
    cache_dir: Path,
    state_ids: tuple[int, ...],
    workers: int,
    *,
    progress_callback: Any | None = None,
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
        for done_index, future in enumerate(as_completed(futures), start=1):
            candidate = futures[future]
            result = future.result()
            if int(result["candidate_id"]) != candidate.candidate_id:
                raise RuntimeError("worker returned the wrong candidate ID")
            if len(result["rows"]) != len(state_ids):
                raise RuntimeError(
                    f"candidate={candidate.candidate_id} returned {len(result['rows'])} rows"
                )
            results[candidate.candidate_id] = result
            if progress_callback is not None:
                progress_callback(done_index, candidate, result)
    if set(results) != {candidate.candidate_id for candidate in candidates}:
        raise RuntimeError("parallel candidate result set is incomplete")
    return results


def _serial_candidate_results(
    candidates: Sequence[VariableMemoryCandidate], cache_dir: Path, state_ids: tuple[int, ...]
) -> dict[int, dict[str, Any]]:
    bases = load_state_bases(cache_dir, state_ids)
    return {
        candidate.candidate_id: evaluate_candidate_on_bases(candidate, bases, return_theta=True)
        for candidate in candidates
    }


def _regression_gate(
    candidates: Sequence[VariableMemoryCandidate], cache_dir: Path, workers: int
) -> dict[str, Any]:
    serial = _serial_candidate_results(candidates, cache_dir, tuple(REGRESSION_STATE_IDS))
    parallel = _parallel_candidate_results(
        candidates, cache_dir, tuple(REGRESSION_STATE_IDS), workers
    )
    metric_fields = (
        "Aend_train_NMSE_dB",
        "Aend_B_NMSE_dB",
        "C2_train_NMSE_dB",
        "C2_B_NMSE_dB",
    )
    max_metric_error = 0.0
    max_theta_error = 0.0
    rank_equal = True
    support_equal = True
    for candidate in candidates:
        serial_rows = {int(row["state_id"]): row for row in serial[candidate.candidate_id]["rows"]}
        parallel_rows = {
            int(row["state_id"]): row for row in parallel[candidate.candidate_id]["rows"]
        }
        for state_id in REGRESSION_STATE_IDS:
            left = serial_rows[int(state_id)]
            right = parallel_rows[int(state_id)]
            for field in metric_fields:
                max_metric_error = max(
                    max_metric_error,
                    abs(float(left[field]) - float(right[field])),
                )
            for side in ("Aend", "C2"):
                rank_equal &= int(left[f"{side}_matrix_rank"]) == int(right[f"{side}_matrix_rank"])
                support_equal &= int(left[f"{side}_n_train_samples"]) == int(
                    right[f"{side}_n_train_samples"]
                )
        for side in ("theta_a", "theta_c"):
            for state_id, theta in serial[candidate.candidate_id][side].items():
                max_theta_error = max(
                    max_theta_error,
                    float(np.max(np.abs(theta - parallel[candidate.candidate_id][side][state_id]))),
                )
    passed = bool(
        max_metric_error <= 1e-10 and max_theta_error <= 1e-10 and rank_equal and support_equal
    )
    return {
        "pass": passed,
        "parallelization_axis": "candidate",
        "state_ids": list(REGRESSION_STATE_IDS),
        "candidate_profiles": [
            {"P": candidate.P, "memory_profile": list(candidate.memory_profile)}
            for candidate in candidates
        ],
        "worker_count": workers,
        "metric_max_abs_error_dB": max_metric_error,
        "theta_max_abs_error": max_theta_error,
        "rank_equal": rank_equal,
        "valid_support_equal": support_equal,
        "metric_fields": list(metric_fields),
    }


def _load_chunk(path: Path, candidate: VariableMemoryCandidate) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("candidate_id", -1)) != candidate.candidate_id:
        raise RuntimeError(f"candidate chunk ID mismatch: {path}")
    rows = payload.get("rows")
    if not isinstance(rows, list) or len(rows) != STATE_COUNT:
        raise RuntimeError(f"candidate chunk row count mismatch: {path}")
    state_ids = sorted(int(row["state_id"]) for row in rows)
    if state_ids != sorted(FAILED_STATE_IDS):
        raise RuntimeError(f"candidate chunk state IDs mismatch: {path}")
    return rows


def _scan_candidates(
    candidates: Sequence[VariableMemoryCandidate],
    cache_dir: Path,
    workers: int,
    logical_cpu_count: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    CANDIDATE_CHUNK_ROOT.mkdir(parents=True, exist_ok=True)
    completed_rows: dict[int, list[dict[str, Any]]] = {}
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    for path in sorted(CANDIDATE_CHUNK_ROOT.glob("candidate_*.json")):
        candidate_id = int(path.stem.split("_")[-1])
        if candidate_id in candidate_by_id:
            try:
                rows = _load_chunk(path, candidate_by_id[candidate_id])
                # A previous interrupted/buggy run may have left a chunk with
                # NaN metrics.  Treat it as pending so it is recomputed rather
                # than silently carried into the formal aggregate.
                if not all(
                    bool(row.get("valid", False))
                    and all(np.isfinite(float(row[field])) for field in CORE_METRIC_COLUMNS)
                    for row in rows
                ):
                    continue
                completed_rows[candidate_id] = rows
            except (OSError, ValueError, RuntimeError, KeyError, TypeError):
                continue
    pending = [
        candidate for candidate in candidates if candidate.candidate_id not in completed_rows
    ]
    progress_path = RESULT_ROOT / "search_progress.json"
    started = time.perf_counter()
    progress: dict[str, Any] = {
        "experiment": EXPERIMENT_NAME,
        "active_candidate_id": None,
        "completed_candidates": len(completed_rows),
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
        "candidate_state_evaluations": len(completed_rows) * STATE_COUNT,
        "total_model_fit_count": len(completed_rows) * STATE_COUNT * 2,
        "search_stop_reason": None,
        "resumed_from_candidate_chunks": bool(completed_rows),
    }
    _write_json(progress_path, progress)

    def on_result(
        done_index: int, candidate: VariableMemoryCandidate, result: dict[str, Any]
    ) -> None:
        completed_rows[candidate.candidate_id] = result["rows"]
        chunk_path = CANDIDATE_CHUNK_ROOT / f"candidate_{candidate.candidate_id:03d}.json"
        _write_json(chunk_path, {"candidate_id": candidate.candidate_id, "rows": result["rows"]})
        elapsed = time.perf_counter() - started
        done_count = len(completed_rows)
        progress.update(
            {
                "active_candidate_id": candidate.candidate_id,
                "completed_candidates": done_count,
                "candidate_state_evaluations": done_count * STATE_COUNT,
                "total_model_fit_count": done_count * STATE_COUNT * 2,
                "elapsed_seconds": elapsed,
                "candidate_throughput_per_second": done_count / elapsed if elapsed > 0 else None,
                "candidate_state_evaluations_per_second": (
                    done_count * STATE_COUNT / elapsed if elapsed > 0 else None
                ),
            }
        )
        _write_json(progress_path, progress)
        if done_index % 5 == 0 or done_index == len(pending):
            print(
                f"candidate scan: completed {done_count}/{len(candidates)} "
                f"(new {done_index}/{len(pending)})",
                flush=True,
            )

    if pending:
        _parallel_candidate_results(
            candidates=pending,
            cache_dir=cache_dir,
            state_ids=FAILED_STATE_IDS,
            workers=workers,
            progress_callback=on_result,
        )
    rows = [row for candidate in candidates for row in completed_rows[candidate.candidate_id]]
    frame = pd.DataFrame(rows).sort_values(["candidate_id", "state_id"]).reset_index(drop=True)
    expected_rows = len(candidates) * STATE_COUNT
    if frame.shape[0] != expected_rows:
        raise RuntimeError(f"candidate-state rows mismatch: {frame.shape[0]} vs {expected_rows}")
    if (
        frame["candidate_id"].nunique() != len(candidates)
        or frame["state_id"].nunique() != STATE_COUNT
    ):
        raise RuntimeError("candidate-state scan does not cover the complete grid")
    elapsed = time.perf_counter() - started
    run_info = {
        "elapsed_seconds": elapsed,
        "candidate_count": len(candidates),
        "state_count": STATE_COUNT,
        "candidate_state_evaluations": len(candidates) * STATE_COUNT,
        "total_model_fit_count": len(candidates) * STATE_COUNT * 2,
        "candidate_state_evaluations_per_second": (
            len(candidates) * STATE_COUNT / elapsed if elapsed > 0 else None
        ),
        "model_fits_per_second": (
            len(candidates) * STATE_COUNT * 2 / elapsed if elapsed > 0 else None
        ),
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
        "resumed_candidate_count": len(completed_rows) - len(pending),
    }
    progress.update({**run_info, "completed_candidates": len(candidates), "final": True})
    _write_json(progress_path, progress)
    return frame, run_info


def _stats(values: np.ndarray, prefix: str) -> dict[str, Any]:
    if values.size == 0 or not np.all(np.isfinite(values)):
        return {
            f"{prefix}_{name}": np.nan
            for name in ("count", "mean", "median", "std", "min", "max", "q25", "q75", "worst")
        }
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


def _aggregate_summary(
    state_metrics: pd.DataFrame, candidates: Sequence[VariableMemoryCandidate]
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        group = state_metrics.loc[state_metrics["candidate_id"] == candidate.candidate_id].copy()
        if group.shape[0] != STATE_COUNT:
            raise RuntimeError(f"candidate={candidate.candidate_id} does not have 14 state rows")
        row: dict[str, Any] = {
            "candidate_id": candidate.candidate_id,
            "P": candidate.P,
            "orders": _json_compact(list(candidate.orders)),
            "memory_profile": _json_compact(list(candidate.memory_profile)),
            "memory_profile_string": candidate.memory_profile_string,
            "memory_definition": _json_compact(candidate.memory_definition),
            "max_delay": candidate.max_delay,
            "coefficient_count": candidate.coefficient_count,
            "valid": bool(group["valid"].all()),
            "is_uniform_memory": len(set(candidate.memory_profile)) == 1,
        }
        for column, short in (
            ("Aend_train_NMSE_dB", "Aend_train"),
            ("Aend_B_NMSE_dB", "Aend_B"),
            ("Aend_generalization_gap_dB", "Aend_gap"),
            ("C2_train_NMSE_dB", "C2_train"),
            ("C2_B_NMSE_dB", "C2_B"),
            ("C2_generalization_gap_dB", "C2_gap"),
        ):
            row.update(_stats(group[column].to_numpy(dtype=float), short))
        for column, name in (
            ("Aend_joint_pass", "Aend_joint_pass"),
            ("C2_joint_pass", "C2_joint_pass"),
            ("all_four_pass", "all_four_pass"),
        ):
            count = int(group[column].astype(bool).sum())
            row[f"{name}_count"] = count
            row[f"{name}_rate"] = float(count / STATE_COUNT)
            row[name] = count
        worst4 = group.loc[:, CORE_METRIC_COLUMNS].max(axis=1).to_numpy(dtype=float)
        row.update(_stats(worst4, "worst4"))
        row["modeling_median"] = float(max(row["Aend_train_median"], row["C2_train_median"]))
        row["generalization_median"] = float(max(row["Aend_B_median"], row["C2_B_median"]))
        row["worst_train_median"] = row["modeling_median"]
        row["worst_B_median"] = row["generalization_median"]
        rows.append(row)
    return pd.DataFrame(rows).sort_values("candidate_id").reset_index(drop=True)


def _pareto_flags(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    result["pareto_optimal"] = False
    points = result.loc[
        :, ["modeling_median", "generalization_median", "coefficient_count"]
    ].to_numpy(dtype=float)
    for i in range(points.shape[0]):
        dominates = np.all(points <= points[i], axis=1) & np.any(points < points[i], axis=1)
        result.loc[result.index[i], "pareto_optimal"] = not bool(np.any(dominates))
    return result


def _select_candidate(summary: pd.DataFrame) -> tuple[pd.Series, str]:
    valid = summary.loc[summary["valid"].astype(bool)].copy()
    if valid.empty:
        raise RuntimeError("no valid candidate in failed-14 scan")
    if int(valid["all_four_pass_count"].max()) == STATE_COUNT:
        pool = valid.loc[valid["all_four_pass_count"] == STATE_COUNT].sort_values(
            ["coefficient_count", "max_delay", "P", "worst4_median", "candidate_id"],
            ascending=[True, True, True, True, True],
            kind="mergesort",
        )
        rule = "14_of_14_all_four_then_complexity"
    elif int(valid["all_four_pass_count"].max()) > 0:
        pool = valid.sort_values(
            [
                "all_four_pass_count",
                "worst4_median",
                "worst4_max",
                "min_joint_pass_count",
                "coefficient_count",
                "max_delay",
                "P",
                "candidate_id",
            ],
            ascending=[False, True, True, False, True, True, True, True],
            kind="mergesort",
        )
        rule = "max_all_four_then_balanced_worst4_then_joint_then_complexity"
    else:
        pool = valid.sort_values(
            [
                "worst4_median",
                "worst4_max",
                "min_joint_pass_count",
                "joint_pass_sum",
                "coefficient_count",
                "max_delay",
                "P",
                "candidate_id",
            ],
            ascending=[True, True, False, False, True, True, True, True],
            kind="mergesort",
        )
        rule = "balanced_worst4_then_joint_then_complexity_when_no_all_four_pass"
    return pool.iloc[0], rule


def _add_selection_columns(summary: pd.DataFrame) -> pd.DataFrame:
    result = summary.copy()
    result["min_joint_pass_count"] = result.loc[
        :, ["Aend_joint_pass_count", "C2_joint_pass_count"]
    ].min(axis=1)
    result["joint_pass_sum"] = result["Aend_joint_pass_count"] + result["C2_joint_pass_count"]
    return result


def _statewise_best(state_metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for state_id in sorted(FAILED_STATE_IDS):
        group = state_metrics.loc[
            (state_metrics["state_id"] == state_id) & state_metrics["valid"].astype(bool)
        ].copy()
        if group.empty:
            raise RuntimeError(f"state={state_id} has no valid candidate")
        group["worst4"] = group.loc[:, CORE_METRIC_COLUMNS].max(axis=1)
        best = group.sort_values(
            ["worst4", "coefficient_count", "max_delay", "P", "candidate_id"],
            ascending=[True, True, True, True, True],
            kind="mergesort",
        ).iloc[0]
        rows.append(
            {
                "state_id": state_id,
                "best_candidate_id": int(best["candidate_id"]),
                "P": int(best["P"]),
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


def _refit_selected(
    selected: VariableMemoryCandidate,
    state_metrics: pd.DataFrame,
    cache_dir: Path,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, dict[str, Any]]:
    bases = load_state_bases(cache_dir, FAILED_STATE_IDS)
    result = evaluate_candidate_on_bases(selected, bases, return_theta=True)
    cached = state_metrics.loc[state_metrics["candidate_id"] == selected.candidate_id].set_index(
        "state_id"
    )
    rows: list[dict[str, Any]] = []
    max_metric_error = 0.0
    max_theta_error = 0.0
    theta_a_rows: list[np.ndarray] = []
    theta_c_rows: list[np.ndarray] = []
    for row in result["rows"]:
        state_id = int(row["state_id"])
        old = cached.loc[state_id]
        metric_error = max(
            abs(float(row[field]) - float(old[field])) for field in CORE_METRIC_COLUMNS
        )
        max_metric_error = max(max_metric_error, metric_error)
        theta_a_rows.append(result["theta_a"][state_id])
        theta_c_rows.append(result["theta_c"][state_id])
        rows.append(
            {
                "state_id": state_id,
                "candidate_id": selected.candidate_id,
                "P": selected.P,
                "memory_profile": _json_compact(list(selected.memory_profile)),
                "Aend_train_NMSE_dB": float(row["Aend_train_NMSE_dB"]),
                "Aend_B_NMSE_dB": float(row["Aend_B_NMSE_dB"]),
                "C2_train_NMSE_dB": float(row["C2_train_NMSE_dB"]),
                "C2_B_NMSE_dB": float(row["C2_B_NMSE_dB"]),
                "max_metric_abs_diff_dB": float(metric_error),
            }
        )
    theta_a = np.stack(theta_a_rows, axis=0).astype(np.complex128, copy=False)
    theta_c = np.stack(theta_c_rows, axis=0).astype(np.complex128, copy=False)
    diagnostics = {
        "all_28_refit": bool(
            theta_a.shape == (STATE_COUNT, selected.coefficient_count)
            and theta_c.shape == theta_a.shape
        ),
        "max_metric_abs_diff_dB": max_metric_error,
        "max_theta_abs_diff": max_theta_error,
        "metric_tolerance_dB": 1e-10,
        "theta_tolerance": 1e-10,
    }
    diagnostics["pass"] = bool(
        diagnostics["all_28_refit"]
        and diagnostics["max_metric_abs_diff_dB"] <= diagnostics["metric_tolerance_dB"]
    )
    return (
        pd.DataFrame(rows).sort_values("state_id").reset_index(drop=True),
        theta_a,
        theta_c,
        diagnostics,
    )


def _selected_candidate_frame(summary: pd.DataFrame, selected_id: int) -> pd.DataFrame:
    return summary.loc[summary["candidate_id"] == selected_id].copy().reset_index(drop=True)


def _excel_source(
    config_rows: list[dict[str, Any]],
    candidate_grid: pd.DataFrame,
    summary: pd.DataFrame,
    state_metrics: pd.DataFrame,
    selected_summary: pd.DataFrame,
    selected_metrics: pd.DataFrame,
    statewise: pd.DataFrame,
    pareto: pd.DataFrame,
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
            "candidate_grid": spec(candidate_grid),
            "candidate_summary": spec(summary),
            "candidate_state_metrics": spec(state_metrics),
            "selected_candidate": spec(selected_summary),
            "selected_state_metrics": spec(selected_metrics),
            "statewise_best": spec(statewise),
            "pareto_candidates": spec(pareto),
        },
        "metadata": {
            "experiment": EXPERIMENT_NAME,
            "state_count": STATE_COUNT,
            "candidate_count": len(candidate_grid),
            "threshold_dB": THRESHOLD_DB,
            "selection_uses_failure_labels": False,
            "parallelization_axis": "candidate",
        },
    }


def _final_summary_text(
    candidates: Sequence[VariableMemoryCandidate],
    summary: pd.DataFrame,
    statewise: pd.DataFrame,
    selected: pd.Series,
    selection_rule: str,
    run_info: Mapping[str, Any],
    regression: Mapping[str, Any],
    refit: Mapping[str, Any],
) -> str:
    p_stats = (
        summary.groupby("P")[
            ["modeling_median", "generalization_median", "Aend_gap_median", "C2_gap_median"]
        ]
        .median()
        .sort_index()
    )
    best_model_p = int(p_stats["modeling_median"].idxmin())
    best_generalization_p = int(p_stats["generalization_median"].idxmin())
    delay_stats = (
        summary.groupby("max_delay")[["worst4_median", "generalization_median"]]
        .median()
        .sort_index()
    )
    uniform = summary.loc[summary["is_uniform_memory"].astype(bool), "worst4_median"]
    nonuniform = summary.loc[~summary["is_uniform_memory"].astype(bool), "worst4_median"]
    statewise_high_precision = int((statewise["worst4"] < THRESHOLD_DB).sum())
    selected_metrics = {
        "Aend_train": float(selected["Aend_train_median"]),
        "Aend_B": float(selected["Aend_B_median"]),
        "C2_train": float(selected["C2_train_median"]),
        "C2_B": float(selected["C2_B_median"]),
    }
    selected_failure_driver = max(selected_metrics, key=selected_metrics.get)
    p_lines = [
        f"P={int(p)}: median(candidate modeling_median)={row['modeling_median']:.6f} dB; "
        f"median(candidate generalization_median)={row['generalization_median']:.6f} dB; "
        f"median Aend gap={row['Aend_gap_median']:.6f} dB; median C2 gap={row['C2_gap_median']:.6f} dB"
        for p, row in p_stats.iterrows()
    ]
    delay_lines = [
        f"max_delay={int(delay)}: median worst4={row['worst4_median']:.6f} dB; "
        f"median generalization={row['generalization_median']:.6f} dB"
        for delay, row in delay_stats.iterrows()
    ]
    pareto_count = int(summary["pareto_optimal"].astype(bool).sum())
    return "\n".join(
        [
            f"{EXPERIMENT_NAME}",
            "Final finite scan: 125 compact odd-order candidates × 14 failed states.",
            "",
            "1. Completion and scientific boundary",
            f"- candidates_completed = {len(candidates)}/125",
            f"- states_completed = {STATE_COUNT}/14",
            f"- candidate_state_evaluations = {run_info['candidate_state_evaluations']}",
            f"- Aend fits = {len(candidates) * STATE_COUNT}; C2 fits = {len(candidates) * STATE_COUNT}; total fits = {run_info['total_model_fit_count']}",
            "- failure labels only define the state subset; they are not used for candidate selection.",
            "- No LUT retrieval, Real-B validation, low-bandwidth operator, Ridge, or old-result overwrite was performed.",
            "",
            "2. Candidate selection",
            f"- selected candidate_id = {int(selected['candidate_id'])}",
            f"- P = {int(selected['P'])}; memory_profile = {selected['memory_profile_string']}; K = {int(selected['coefficient_count'])}; max_delay = {int(selected['max_delay'])}",
            f"- selection_rule = {selection_rule}",
            f"- all_four_pass = {int(selected['all_four_pass_count'])}/{STATE_COUNT}; Aend_joint = {int(selected['Aend_joint_pass_count'])}/{STATE_COUNT}; C2_joint = {int(selected['C2_joint_pass_count'])}/{STATE_COUNT}",
            f"- balanced worst4 median = {float(selected['worst4_median']):.6f} dB; balanced worst4 max = {float(selected['worst4_max']):.6f} dB",
            "",
            "3. Which P is best",
            f"- best P by median candidate modeling_median across its profiles: P={best_model_p}",
            f"- best P by median candidate generalization_median across its profiles: P={best_generalization_p}",
            *[f"- {line}" for line in p_lines],
            "",
            "4. Memory-depth and profile observations",
            *[f"- {line}" for line in delay_lines],
            f"- uniform-memory candidate worst4 median: median={float(np.median(uniform)):.6f} dB; best={float(np.min(uniform)):.6f} dB",
            f"- order-dependent nonuniform candidate worst4 median: median={float(np.median(nonuniform)):.6f} dB; best={float(np.min(nonuniform)):.6f} dB",
            "- Low-order long-memory versus uniform-memory is reported as a diagnostic comparison only; no separate model family is introduced.",
            "",
            "5. Aend versus C2 preference and failure driver",
            f"- best Aend_B median candidate = {int(summary.loc[summary['Aend_B_median'].idxmin(), 'candidate_id'])}",
            f"- best C2_B median candidate = {int(summary.loc[summary['C2_B_median'].idxmin(), 'candidate_id'])}",
            f"- selected-candidate worst median metric is {selected_failure_driver} = {selected_metrics[selected_failure_driver]:.6f} dB",
            f"- statewise best candidates below -40 dB on all four metrics: {statewise_high_precision}/{STATE_COUNT}",
            f"- Pareto candidates (modeling median, generalization median, K): {pareto_count}",
            "- If statewise best is substantially better than the selected unified candidate, the unified-structure constraint is the likely bottleneck; this is a diagnostic inference, not a retrieval claim.",
            "",
            "6. 14/14 all-four result",
            f"- any 14/14 all-four candidate: {'yes' if int(summary['all_four_pass_count'].max()) == STATE_COUNT else 'no'}",
            f"- maximum all-four pass count: {int(summary['all_four_pass_count'].max())}/{STATE_COUNT}",
            "",
            "7. Parallel execution",
            f"- logical CPUs = {run_info['logical_cpu_count']}; target fraction = {run_info['cpu_target_fraction']}; requested/effective workers = {run_info['worker_count_requested']}/{run_info['worker_count_effective']}",
            "- parallelization axis = candidate; states are sequential inside each candidate worker; BLAS threads/worker = 1; nested parallelism = false.",
            f"- elapsed = {run_info['elapsed_seconds']:.3f} s; candidate-state evaluations/s = {run_info['candidate_state_evaluations_per_second']:.6f}; model fits/s = {run_info['model_fits_per_second']:.6f}",
            f"- serial/parallel regression pass = {bool(regression['pass'])}; max metric error = {regression['metric_max_abs_error_dB']:.3e} dB; max theta error = {regression['theta_max_abs_error']:.3e}",
            "- psutil was not available; CPU and memory peak statistics are not claimed.",
            "",
            "8. Refit, protection, and limitations",
            f"- selected refit validation = {bool(refit['pass'])}; max metric error = {refit['max_metric_abs_diff_dB']:.3e} dB",
            "- raw and protected prior-result manifests are checked in validation.json.",
            "- pytest was not installed in the fixed environment; no package was installed for this task.",
            "",
            "The task ends here. Do not automatically enter LUT retrieval or add candidates beyond P<=9 and M_p<=4.",
        ]
    )


def main() -> None:
    freeze_support()
    print(f"Starting {EXPERIMENT_NAME}", flush=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    before = _snapshot()
    failure_info = _validate_failure_source()
    candidates = generate_candidates()
    validate_candidate_grid(candidates)
    logical_cpu_count = int(os.cpu_count() or 1)
    workers = target_worker_count(logical_cpu_count)
    CONFIG_ROOT.mkdir(parents=True, exist_ok=True)
    _write_json(
        CONFIG_ROOT / "experiment_config.json",
        {
            "experiment": EXPERIMENT_NAME,
            "bandwidth": "5B",
            "state_count": STATE_COUNT,
            "all_state_count": ALL_STATE_COUNT,
            "failure_states_only": True,
            "failed_state_ids": list(FAILED_STATE_IDS),
            "P_max": 9,
            "allowed_orders": [1, 3, 5, 7, 9],
            "memory_max": 4,
            "memory_monotonic_nonincreasing": True,
            "candidate_count": len(candidates),
            "ridge_used": False,
            "threshold_dB": THRESHOLD_DB,
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
                "states_processed_sequentially_inside_candidate_worker": True,
                "blas_threads_per_worker": 1,
                "nested_parallelism": False,
                "worker_state_cache_used": True,
            },
        },
    )
    _write_json(CONFIG_ROOT / "failed_state_ids.json", failure_info)
    grid = candidate_grid_frame(candidates)
    _write_frame(grid, CONFIG_ROOT / "candidate_grid.csv")
    _write_json(
        CONFIG_ROOT / "candidate_grid.json",
        grid.to_dict("records"),
    )
    manifest = _prepare_cache()

    regression_candidates = []
    for P, profile in REGRESSION_PROFILES:
        match = next(
            candidate
            for candidate in candidates
            if candidate.P == P and candidate.memory_profile == tuple(profile)
        )
        regression_candidates.append(match)
    regression = _regression_gate(regression_candidates, PREPROCESSED_ROOT, workers)
    _write_json(VALIDATION_ROOT / "serial_parallel_regression.json", regression)
    if not regression["pass"]:
        raise RuntimeError(f"candidate-level serial/parallel regression failed: {regression}")

    state_metrics, run_info = _scan_candidates(
        candidates,
        PREPROCESSED_ROOT,
        workers,
        logical_cpu_count,
    )
    _write_frame(state_metrics, TABLE_ROOT / "candidate_state_metrics.csv.gz", compression="gzip")
    summary = _add_selection_columns(_aggregate_summary(state_metrics, candidates))
    summary = _pareto_flags(summary)
    selected, selection_rule = _select_candidate(summary)
    selected_candidate = next(
        candidate
        for candidate in candidates
        if candidate.candidate_id == int(selected["candidate_id"])
    )
    statewise = _statewise_best(state_metrics)
    selected_metrics = state_metrics.loc[
        state_metrics["candidate_id"] == selected_candidate.candidate_id
    ].copy()
    selected_metrics["worst4"] = selected_metrics.loc[:, CORE_METRIC_COLUMNS].max(axis=1)
    refit_diagnostics, theta_a, theta_c, refit_validation = _refit_selected(
        selected_candidate, state_metrics, PREPROCESSED_ROOT
    )
    if not refit_validation["pass"]:
        raise RuntimeError(f"selected refit validation failed: {refit_validation}")

    _write_frame(summary, TABLE_ROOT / "candidate_summary.csv")
    _write_frame(statewise, TABLE_ROOT / "statewise_best_candidate.csv")
    _write_frame(
        summary.loc[summary["pareto_optimal"].astype(bool)], TABLE_ROOT / "pareto_candidates.csv"
    )
    _write_frame(selected_metrics, TABLE_ROOT / "selected_candidate_state_metrics.csv")
    _write_frame(refit_diagnostics, TABLE_ROOT / "selected_refit_diagnostics.csv")
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
        "P": selected_candidate.P,
        "orders": list(selected_candidate.orders),
        "memory_profile": list(selected_candidate.memory_profile),
        "memory_profile_string": selected_candidate.memory_profile_string,
        "memory_definition": selected_candidate.memory_definition,
        "max_delay": selected_candidate.max_delay,
        "coefficient_count": selected_candidate.coefficient_count,
        "ridge_used": False,
        "selection_rule": selection_rule,
        "selection_based_on": "failed-14 behavior-model metrics only",
        "state_count": STATE_COUNT,
        "search_space_frozen": True,
    }
    _write_json(MODEL_ROOT / "selected_model.json", selected_payload)
    figure_details = generate_figures(summary, state_metrics, selected_metrics, FIGURE_ROOT)

    config_rows = [
        {"field": "experiment", "value": EXPERIMENT_NAME},
        {"field": "bandwidth", "value": "5B"},
        {"field": "failed_state_ids", "value": list(FAILED_STATE_IDS)},
        {"field": "candidate_count", "value": len(candidates)},
        {"field": "state_count", "value": STATE_COUNT},
        {"field": "P_max", "value": 9},
        {"field": "memory_max", "value": 4},
        {"field": "threshold_dB", "value": THRESHOLD_DB},
        {"field": "ridge_used", "value": False},
        {"field": "retrieval_used", "value": False},
        {"field": "real_B_used", "value": False},
        {"field": "low_bandwidth_operator_used", "value": False},
        {"field": "parallelization_axis", "value": "candidate"},
        {"field": "logical_cpu_count", "value": logical_cpu_count},
        {"field": "worker_count_effective", "value": workers},
        {"field": "blas_threads_per_worker", "value": 1},
        {"field": "nested_parallelism", "value": False},
        {"field": "search_stop_reason", "value": "finite_125_candidate_scan_completed"},
        {"field": "selected_candidate_id", "value": selected_candidate.candidate_id},
    ]
    _write_json(
        RESULT_ROOT / "excel_source.json",
        _excel_source(
            config_rows,
            grid,
            summary,
            state_metrics,
            _selected_candidate_frame(summary, selected_candidate.candidate_id),
            selected_metrics,
            statewise,
            summary.loc[summary["pareto_optimal"].astype(bool)],
        ),
    )
    final_validation = {
        "experiment": EXPERIMENT_NAME,
        "bandwidth": "5B",
        "state_count": STATE_COUNT,
        "all_state_count": ALL_STATE_COUNT,
        "failure_states_only": True,
        "failure_state_ids": list(FAILED_STATE_IDS),
        "P_max": 9,
        "allowed_orders": [1, 3, 5, 7, 9],
        "even_orders_used": False,
        "order_2_used": False,
        "memory_max": 4,
        "order_dependent_memory": True,
        "memory_monotonic_nonincreasing": True,
        "candidate_count": len(candidates),
        "candidate_completion": f"{len(candidates)}/125",
        "state_completion": f"{STATE_COUNT}/14",
        "ridge_used": False,
        "Aend_and_C2_same_MP_structure": True,
        "coefficients_shared": False,
        "retrieval_used": False,
        "real_B_used": False,
        "low_bandwidth_operator_used": False,
        "failure_labels_used_only_for_state_subset": True,
        "failure_labels_used_for_candidate_selection": False,
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
        "serial_parallel_regression": regression,
        "selected_refit_validation_pass": bool(refit_validation["pass"]),
        "selected_refit_validation": refit_validation,
        "candidate_state_evaluations": run_info["candidate_state_evaluations"],
        "total_model_fit_count": run_info["total_model_fit_count"],
        "runtime": run_info,
        "selected_candidate": selected_payload,
        "selection_rule": selection_rule,
        "search_stop_reason": "finite_125_candidate_scan_completed",
        "search_space_frozen_after_completion": True,
        "preprocessed_cache": manifest,
        "figures": figure_details,
        "raw_data_modified": False,
        "protected_results_modified": False,
        "source_failure_list": failure_info,
    }
    after = _snapshot()
    protection = _verify_protection(before, after)
    if not protection["all_protected_unchanged"]:
        raise RuntimeError("raw data or protected prior results changed")
    final_validation["protection_verification"] = protection
    final_validation["raw_data_modified"] = not all(protection["data/raw"].values())
    final_validation["protected_results_modified"] = not all(
        item["sha256_unchanged"] for item in protection["result_dirs"].values()
    )
    _write_json(VALIDATION_ROOT / "validation.json", final_validation)
    _write_json(RESULT_ROOT / "validation.json", final_validation)
    _write_json(
        RESULT_ROOT / "search_progress.json",
        {**run_info, "final": True, "search_stop_reason": final_validation["search_stop_reason"]},
    )
    summary_text = _final_summary_text(
        candidates,
        summary,
        statewise,
        selected,
        selection_rule,
        run_info,
        regression,
        refit_validation,
    )
    (RESULT_ROOT / "final_result_summary.txt").write_text(summary_text + "\n", encoding="utf-8")
    log_body = "\n".join(
        [
            f"{EXPERIMENT_NAME} completed: 125/125 candidates × 14/14 failed states.",
            f"Candidate-level ProcessPoolExecutor: logical_cpu_count={logical_cpu_count}, workers={workers}, target=0.90, BLAS threads/worker=1, nested_parallelism=False.",
            f"Selected candidate={selected_candidate.candidate_id}, P={selected_candidate.P}, memory_profile={selected_candidate.memory_profile_string}, K={selected_candidate.coefficient_count}; selection={selection_rule}.",
            f"Total fits={run_info['total_model_fit_count']}; elapsed={run_info['elapsed_seconds']:.3f}s; candidate-state evaluations/s={run_info['candidate_state_evaluations_per_second']:.6f}; model fits/s={run_info['model_fits_per_second']:.6f}.",
            f"Serial/parallel regression={regression['pass']}; selected refit={refit_validation['pass']}; raw/protected unchanged={protection['all_protected_unchanged']}.",
            "No LUT retrieval, Real-B validation, low-bandwidth transform, Ridge, or candidate expansion was performed after this finite scan.",
            f"Results: {RESULT_ROOT}",
        ]
    )
    for path, title in (
        (MODEL_LOG, f"{EXPERIMENT_NAME}"),
        (HANDOFF_LOG, f"{EXPERIMENT_NAME}"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n\n[{datetime.now(UTC).isoformat()}] {title}\n{log_body}\n")
    print(
        f"Completed {EXPERIMENT_NAME}: selected candidate={selected_candidate.candidate_id}, "
        f"P={selected_candidate.P}, profile={selected_candidate.memory_profile_string}, "
        f"K={selected_candidate.coefficient_count}",
        flush=True,
    )
    print(f"Results: {RESULT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
