"""Small read-only runtime benchmarks; this is never a formal search entry point."""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from behavior_modeling.shared.basis_function_selection.model_solver import fit_ridge
from behavior_modeling.shared.volterra_terms import build_basis_columns

from retrieval_oriented_model_selection.shared.shareability_optimized_evaluator import (
    evaluate_partition,
    rank_and_score,
)
from retrieval_oriented_model_selection.shared.shareability_persistence import (
    load_screening_artifact,
    save_screening_artifact,
)
from retrieval_oriented_model_selection.shared.shareability_screening import (
    build_shortlist,
    candidate_pool_fingerprint,
    descriptor_stratum,
    merge_screening_partitions,
    screen_candidate_pool,
)
from retrieval_oriented_model_selection.shared.shareability_structure import CandidateScoreCache
from retrieval_oriented_model_selection.shared.shareability_sufficient_stats import support_id
from retrieval_oriented_model_selection.shared.shareability_types import CandidateScreeningSpec

from .run_search import (
    DATASET_FILES,
    persist_screening_scores,
    validated_science_fingerprint,
    validated_search_policy_fingerprint,
)
from .task_config import (
    DICTIONARY,
    LOG_ROOT,
    PARALLEL,
    REAL_B_SOURCE,
    SCREENING,
    SCREENING_BLOCK_SIZE,
    SEED_TERM,
    STRUCTURE,
)

_THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)

PRE_SEARCH_ROOT = LOG_ROOT / "pre_search_validation"
FULL_K2_CHECKPOINT = PRE_SEARCH_ROOT / "k2_exhaustive_checkpoint.json"
FULL_K2_TRACE = PRE_SEARCH_ROOT / "k2_exhaustive_results.jsonl"
FULL_K2_CACHE = PRE_SEARCH_ROOT / "candidate_score_cache"


def _partition_job(ids: tuple[str, ...], block_size: int) -> dict[str, object]:
    if any(os.environ.get(name) != "1" for name in _THREAD_VARIABLES):
        raise RuntimeError("one BLAS thread per spawned worker required")
    started = time.monotonic()
    arrays = {name: np.load(path, mmap_mode="r") for name, path in DATASET_FILES.items()}
    scores, diagnostics = evaluate_partition(
        ((SEED_TERM.basis_id,),),
        ids,
        x_a=arrays["x_aend"],
        y_a=arrays["y_aend"],
        x_c=arrays["x_c2"],
        y_c=arrays["y_c2"],
        common_b=arrays["common_b"],
        real_b=np.load(REAL_B_SOURCE, mmap_mode="r"),
        dictionary=DICTIONARY,
        ridge_lambda=STRUCTURE.ridge_lambda,
        block_size=block_size,
    )
    return {
        "pid": os.getpid(),
        "elapsed_seconds": time.monotonic() - started,
        "max_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "count": len(scores),
        "diagnostics": diagnostics,
        "scores": {"|".join(key): score for key, score in scores.items()},
    }


def _screen_partition_job(
    ids: tuple[str, ...], parents: tuple[tuple[str, ...], ...], block_size: int
) -> dict[str, object]:
    if any(os.environ.get(name) != "1" for name in _THREAD_VARIABLES):
        raise RuntimeError("one BLAS thread per spawned worker required")
    started = time.monotonic()
    arrays = {name: np.load(path, mmap_mode="r") for name, path in DATASET_FILES.items()}
    scores = screen_candidate_pool(
        parents,
        ids,
        x_a=arrays["x_aend"],
        y_a=arrays["y_aend"],
        x_c=arrays["x_c2"],
        y_c=arrays["y_c2"],
        dictionary=DICTIONARY,
        ridge_lambda=STRUCTURE.ridge_lambda,
        spec=SCREENING,
        block_size=block_size,
    )
    return {
        "pid": os.getpid(),
        "elapsed_seconds": time.monotonic() - started,
        "max_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "scores": scores,
    }


def representative_candidate_ids(count: int) -> tuple[str, ...]:
    """Deterministic proportional sample with every canonical stratum represented."""
    available = tuple(sorted(basis for basis in DICTIONARY if basis != SEED_TERM.basis_id))
    if not 1 <= count <= len(available):
        raise ValueError("representative candidate count is outside the complete K2 pool")
    if count == len(available):
        return available
    groups: dict[str, list[str]] = {}
    for basis in available:
        groups.setdefault(descriptor_stratum(DICTIONARY[basis]), []).append(basis)
    strata = tuple(sorted(groups))
    allocations = {stratum: 0 for stratum in strata}
    if count >= len(strata):
        for stratum in strata:
            allocations[stratum] = 1
    remaining = count - sum(allocations.values())
    capacities = {stratum: len(groups[stratum]) - allocations[stratum] for stratum in strata}
    total_capacity = sum(capacities.values())
    exact = {
        stratum: remaining * capacities[stratum] / total_capacity if total_capacity else 0.0
        for stratum in strata
    }
    for stratum in strata:
        allocations[stratum] += min(capacities[stratum], int(exact[stratum]))
    residual = count - sum(allocations.values())
    for stratum in sorted(strata, key=lambda item: (-(exact[item] % 1), item)):
        if residual == 0:
            break
        if allocations[stratum] < len(groups[stratum]):
            allocations[stratum] += 1
            residual -= 1
    selected: list[str] = []
    for stratum in strata:
        group = groups[stratum]
        take = allocations[stratum]
        if take:
            selected.extend(group[index * len(group) // take] for index in range(take))
    if len(selected) != count or len(set(selected)) != count:
        raise RuntimeError("representative candidate allocation failed")
    return tuple(sorted(selected))


def benchmark(count: int, workers: int, block_size: int) -> dict[str, object]:
    if not 1 <= count <= 1000 or not 1 <= workers <= PARALLEL.max_workers:
        raise ValueError("only bounded maintenance benchmarks are permitted")
    fingerprint = validated_science_fingerprint()["sha256"]
    ids = tuple(b for b in sorted(DICTIONARY) if b != SEED_TERM.basis_id)[:count]
    partitions = tuple(tuple(ids[i::workers]) for i in range(workers))
    previous = {name: os.environ.get(name) for name in _THREAD_VARIABLES}
    try:
        for name in _THREAD_VARIABLES:
            os.environ[name] = "1"
        started = time.monotonic()
        if workers == 1:
            rows = [_partition_job(ids, block_size)]
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
                futures = [
                    pool.submit(_partition_job, part, block_size) for part in partitions if part
                ]
                rows = [future.result() for future in futures]
        elapsed = time.monotonic() - started
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    all_scores = {}
    for row in rows:
        if set(all_scores) & set(row["scores"]):
            raise RuntimeError("overlapping worker partitions")
        all_scores.update(row.pop("scores"))
    if len(all_scores) != count:
        raise RuntimeError("incomplete benchmark partition")
    return {
        "science_fingerprint": fingerprint,
        "count": count,
        "workers": workers,
        "block_size": block_size,
        "wall_seconds": elapsed,
        "supports_per_second": count / elapsed,
        "worker_rows": rows,
        "scores_in_canonical_order": [
            all_scores["|".join(sorted((SEED_TERM.basis_id, b)))] for b in ids
        ],
    }


def _run_screening_pool(
    candidate_ids: tuple[str, ...],
    parents: tuple[tuple[str, ...], ...],
    *,
    workers: int,
    block_size: int,
) -> tuple[dict[tuple[str, ...], object], list[dict[str, object]], float]:
    if not 1 <= workers <= PARALLEL.max_workers:
        raise ValueError("worker count is outside the formal parallel policy")
    partitions = tuple(tuple(candidate_ids[index::workers]) for index in range(workers))
    previous = {name: os.environ.get(name) for name in _THREAD_VARIABLES}
    try:
        for name in _THREAD_VARIABLES:
            os.environ[name] = "1"
        started = time.monotonic()
        if workers == 1:
            rows = [_screen_partition_job(candidate_ids, parents, block_size)]
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
                futures = [
                    pool.submit(_screen_partition_job, partition, parents, block_size)
                    for partition in partitions
                    if partition
                ]
                rows = [future.result() for future in futures]
        elapsed = time.monotonic() - started
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    merged = merge_screening_partitions(
        tuple(row.pop("scores") for row in rows),
        candidate_ids,  # type: ignore[arg-type]
    )
    return merged, rows, elapsed


def _run_exact_pool(
    candidate_ids: tuple[str, ...], *, workers: int, block_size: int
) -> tuple[dict[str, dict[str, object]], list[dict[str, object]], float]:
    if not 1 <= workers <= PARALLEL.max_workers:
        raise ValueError("worker count is outside the formal parallel policy")
    partitions = tuple(tuple(candidate_ids[index::workers]) for index in range(workers))
    previous = {name: os.environ.get(name) for name in _THREAD_VARIABLES}
    try:
        for name in _THREAD_VARIABLES:
            os.environ[name] = "1"
        started = time.monotonic()
        if workers == 1:
            rows = [_partition_job(candidate_ids, block_size)]
        else:
            with ProcessPoolExecutor(max_workers=workers, mp_context=get_context("spawn")) as pool:
                futures = [
                    pool.submit(_partition_job, partition, block_size)
                    for partition in partitions
                    if partition
                ]
                rows = [future.result() for future in futures]
        elapsed = time.monotonic() - started
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    combined: dict[str, dict[str, object]] = {}
    for row in rows:
        for support, score in row.pop("scores").items():
            if support in combined:
                raise RuntimeError("overlapping exact candidate partition")
            combined[support] = score
    if len(combined) != len(candidate_ids):
        raise RuntimeError("incomplete exact candidate partition")
    return combined, rows, elapsed


def screening_benchmark(
    count: int,
    workers: int,
    block_size: int = SCREENING_BLOCK_SIZE,
) -> dict[str, object]:
    """Bounded real-data screening benchmark; it performs no exact evaluation."""
    if not 1 <= count <= 1000:
        raise ValueError("maintenance screening benchmark is limited to 1..1000 candidates")
    candidate_ids = representative_candidate_ids(count)
    parents = ((SEED_TERM.basis_id,),)
    scores, rows, elapsed = _run_screening_pool(
        candidate_ids, parents, workers=workers, block_size=block_size
    )
    parent_scores = scores[parents[0]]
    persist_screening_scores(parent_scores)  # type: ignore[arg-type]
    shortlist = build_shortlist(parent_scores, SCREENING, DICTIONARY)  # type: ignore[arg-type]
    policy = validated_search_policy_fingerprint()
    return {
        **policy,
        "candidate_count": count,
        "workers": workers,
        "block_size": block_size,
        "wall_seconds": elapsed,
        "candidates_per_second": count / elapsed,
        "shortlist_size": len(shortlist),
        "shortlist": [asdict(entry) for entry in shortlist],
        "worker_rows": rows,
    }


def _recall(shortlist: tuple[str, ...], exact_ranked: tuple[str, ...]) -> dict[str, object]:
    chosen = set(shortlist)
    return {
        "exact_top1_in_shortlist": exact_ranked[0] in chosen,
        "exact_top3_recalled": int(sum(item in chosen for item in exact_ranked[:3])),
        "exact_top10_recalled": int(sum(item in chosen for item in exact_ranked[:10])),
    }


def screening_quality_validation(
    count: int = 1000,
    workers: int = 10,
    block_size: int = SCREENING_BLOCK_SIZE,
    *,
    allow_large: bool = False,
) -> dict[str, object]:
    """Bounded 425-state SOMP recall validation against exact K2 evaluation."""
    upper = len(DICTIONARY) - 1 if allow_large else 1000
    if not 10 <= count <= upper:
        raise ValueError(f"quality validation is limited to 10..{upper} candidates")
    candidate_ids = representative_candidate_ids(count)
    parent = (SEED_TERM.basis_id,)
    scores, screen_rows, screen_elapsed = _run_screening_pool(
        candidate_ids, (parent,), workers=workers, block_size=block_size
    )
    parent_scores = scores[parent]
    exact, exact_rows, exact_elapsed = _run_exact_pool(
        candidate_ids, workers=workers, block_size=block_size
    )
    ranked = tuple(
        next(basis for basis in support.split("|") if basis != SEED_TERM.basis_id)
        for support, _ in sorted(
            exact.items(), key=lambda item: (-int(item[1]["N_shareable"]), item[0])
        )
    )
    variants: dict[str, CandidateScreeningSpec] = {}
    for size in sorted(set(size for size in (100, 250, 500, 1000, 2000, count) if size <= count)):
        explore = min(100, size)
        variants[f"mixed_{size}"] = CandidateScreeningSpec(
            shortlist_size=size, exploit_size=size - explore, explore_size=explore
        )
    if count >= 500:
        variants["top500_only"] = CandidateScreeningSpec(
            shortlist_size=500, exploit_size=500, explore_size=0
        )
        variants["mixed_400_plus_100"] = SCREENING
    quality: dict[str, object] = {}
    for name, spec in variants.items():
        shortlist = parent_scores.shortlist(spec, DICTIONARY)
        ids = tuple(entry.basis_id for entry in shortlist)
        quality[name] = {"shortlist_size": len(ids), **_recall(ids, ranked)}
    full_spec = CandidateScreeningSpec(shortlist_size=count, exploit_size=count, explore_size=0)
    full_shortlist = tuple(
        entry.basis_id for entry in parent_scores.shortlist(full_spec, DICTIONARY)
    )
    if set(full_shortlist) != set(candidate_ids):
        raise RuntimeError("full-pool screening did not preserve every exact candidate")
    persist_screening_scores(parent_scores)  # type: ignore[arg-type]
    return {
        **validated_search_policy_fingerprint(),
        "candidate_count": count,
        "candidate_pool": list(candidate_ids),
        "workers": workers,
        "block_size": block_size,
        "screening_wall_seconds": screen_elapsed,
        "exact_wall_seconds": exact_elapsed,
        "screening_candidates_per_second": count / screen_elapsed,
        "exact_candidates_per_second": count / exact_elapsed,
        "exact_ranking_by_N_shareable_then_support": list(ranked),
        "quality": quality,
        "full_pool_shortlist_equals_candidate_pool": True,
        "screen_worker_rows": screen_rows,
        "exact_worker_rows": exact_rows,
    }


def screening_parallel_determinism_validation(
    count: int = 200, block_size: int = SCREENING_BLOCK_SIZE
) -> dict[str, object]:
    """Compare real-data K2 screening shortlists from one and ten spawn workers."""
    if not 10 <= count <= 1000:
        raise ValueError("determinism validation is limited to 10..1000 candidates")
    candidate_ids = representative_candidate_ids(count)
    parent = (SEED_TERM.basis_id,)
    serial, serial_rows, serial_elapsed = _run_screening_pool(
        candidate_ids, (parent,), workers=1, block_size=block_size
    )
    parallel, parallel_rows, parallel_elapsed = _run_screening_pool(
        candidate_ids, (parent,), workers=PARALLEL.max_workers, block_size=block_size
    )
    serial_shortlist = serial[parent].shortlist(SCREENING, DICTIONARY)
    parallel_shortlist = parallel[parent].shortlist(SCREENING, DICTIONARY)
    serial_ids = tuple(entry.basis_id for entry in serial_shortlist)
    parallel_ids = tuple(entry.basis_id for entry in parallel_shortlist)
    if serial_ids != parallel_ids:
        raise RuntimeError("one-worker and ten-worker screening shortlist differ")
    np.testing.assert_allclose(serial[parent].scores, parallel[parent].scores, equal_nan=True)
    return {
        **validated_search_policy_fingerprint(),
        "candidate_count": count,
        "shortlist_identical": True,
        "shortlist": list(serial_ids),
        "one_worker_seconds": serial_elapsed,
        "ten_worker_seconds": parallel_elapsed,
        "one_worker_rows": serial_rows,
        "ten_worker_rows": parallel_rows,
    }


def reference_optimized_gate(count: int = 20) -> dict[str, object]:
    """Compare optimized vs full-rebuild exact evaluation before full K2."""
    if not 20 <= count <= 20:
        raise ValueError("reference gate is fixed at exactly 20 K2 supports")
    candidate_ids = representative_candidate_ids(count)
    arrays = {name: np.load(path, mmap_mode="r") for name, path in DATASET_FILES.items()}
    real_b = np.load(REAL_B_SOURCE, mmap_mode="r")
    parent = (SEED_TERM.basis_id,)
    rows = []
    for basis_id in candidate_ids:
        child = tuple(sorted((*parent, basis_id)))
        optimized_scores, diagnostics = evaluate_partition(
            (parent,),
            (basis_id,),
            x_a=arrays["x_aend"],
            y_a=arrays["y_aend"],
            x_c=arrays["x_c2"],
            y_c=arrays["y_c2"],
            common_b=arrays["common_b"],
            real_b=real_b,
            dictionary=DICTIONARY,
            ridge_lambda=STRUCTURE.ridge_lambda,
            include_details=True,
        )
        optimized = optimized_scores[child]
        theta_a = np.empty((425, 2), complex)
        theta_c = np.empty_like(theta_a)
        for state in range(425):
            phi_a = build_basis_columns(arrays["x_aend"][state], child, DICTIONARY, dmax=4)
            phi_c = build_basis_columns(arrays["x_c2"][state], child, DICTIONARY, dmax=4)
            theta_a[state] = fit_ridge(phi_a, arrays["y_aend"][state, 4:], 1e-8).theta
            theta_c[state] = fit_ridge(phi_c, arrays["y_c2"][state, 4:], 1e-8).theta
        reference = rank_and_score(
            theta_a,
            theta_c,
            build_basis_columns(arrays["common_b"], child, DICTIONARY, dmax=4),
            real_b,
            include_details=True,
        )
        for field in (
            "N_shareable",
            "N_self",
            "N_top3",
            "N_top5",
            "N_top10",
            "Q1",
            "top10",
            "true_state_rank",
        ):
            if optimized[field] != reference[field]:
                raise RuntimeError(f"reference gate mismatch field={field} basis={basis_id}")
        finite = np.isfinite(reference["distance"])
        distance_error = float(
            np.max(np.abs(optimized["distance"][finite] - reference["distance"][finite]))
        )
        if distance_error > 1e-5:
            raise RuntimeError(
                f"reference gate distance mismatch basis={basis_id}: {distance_error}"
            )
        rows.append(
            {
                "basis_id": basis_id,
                "support": list(child),
                "max_distance_error_db": distance_error,
                "fallbacks": diagnostics["fallbacks"],
            }
        )
    return {
        **validated_search_policy_fingerprint("exhaustive"),
        "stage": "reference_vs_optimized_k2_gate",
        "candidate_count": count,
        "all_metrics_identical": True,
        "rows": rows,
    }


def _atomic_json_write(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _load_cached_candidate(
    cache: CandidateScoreCache,
    basis_id: str,
) -> dict[str, int] | None:
    try:
        return cache.load((SEED_TERM.basis_id, basis_id), STRUCTURE.ridge_lambda)
    except FileNotFoundError:
        return None


def full_k2_exhaustive_validation(
    *,
    confirm_large_compute: bool,
    batch_size: int = 1000,
    block_size: int = 64,
) -> dict[str, object]:
    """Run/resume all 38,334 K2 exact scores under a separate validation root."""
    if not confirm_large_compute:
        raise PermissionError("full K2 validation requires --confirm-large-compute")
    if batch_size < 1 or batch_size > 2000:
        raise ValueError("full K2 batch size must be in 1..2000")
    exact = validated_science_fingerprint()
    exact_fp = exact["sha256"]
    candidate_ids = representative_candidate_ids(len(DICTIONARY) - 1)
    pool_fp = candidate_pool_fingerprint(candidate_ids)
    cache = CandidateScoreCache(FULL_K2_CACHE, exact_fp, "5B")
    checkpoint: dict[str, object] = {}
    if FULL_K2_CHECKPOINT.exists():
        checkpoint = json.loads(FULL_K2_CHECKPOINT.read_text(encoding="utf-8"))
        expected = {
            "stage": "exhaustive_k2_validation",
            "parent_support_id": support_id((SEED_TERM.basis_id,)),
            "candidate_pool_hash": pool_fp,
            "exact_evaluation_fingerprint": exact_fp,
            "effective_workers": PARALLEL.max_workers,
        }
        if any(checkpoint.get(key) != value for key, value in expected.items()):
            raise ValueError("full K2 checkpoint fingerprint/config mismatch")
    start = int(checkpoint.get("next_candidate_index", 0))
    if start < 0 or start > len(candidate_ids):
        raise ValueError("full K2 checkpoint index is outside candidate pool")
    PRE_SEARCH_ROOT.mkdir(parents=True, exist_ok=True)
    total_started = time.monotonic()
    total_cache_hits = int(checkpoint.get("cache_hits", 0))
    total_cache_misses = int(checkpoint.get("cache_misses", 0))
    total_fallbacks = int(checkpoint.get("fallback_count", 0))
    if start == 0 and not FULL_K2_TRACE.exists():
        FULL_K2_TRACE.touch()
    logged_candidates: set[str] = set()
    if FULL_K2_TRACE.exists():
        with FULL_K2_TRACE.open(encoding="utf-8") as trace:
            for line in trace:
                if line.strip():
                    logged_candidates.add(str(json.loads(line)["basis_id"]))
    for begin in range(start, len(candidate_ids), batch_size):
        batch = candidate_ids[begin : begin + batch_size]
        missing: list[str] = []
        hit_rows: dict[str, dict[str, int]] = {}
        for basis_id in batch:
            cached = _load_cached_candidate(cache, basis_id)
            if cached is None:
                missing.append(basis_id)
            else:
                hit_rows[basis_id] = cached
        total_cache_hits += len(hit_rows)
        total_cache_misses += len(missing)
        computed: dict[str, dict[str, object]] = {}
        batch_rows: list[dict[str, object]] = []
        batch_started = time.monotonic()
        if missing:
            exact_scores, worker_rows, _ = _run_exact_pool(
                tuple(missing), workers=PARALLEL.max_workers, block_size=block_size
            )
            for support_key, score in exact_scores.items():
                parts = tuple(support_key.split("|"))
                basis_id = next(item for item in parts if item != SEED_TERM.basis_id)
                computed[basis_id] = score
            if len(computed) != len(missing):
                raise RuntimeError("full K2 exact batch returned incomplete candidate scores")
            for basis_id, score in computed.items():
                cache.save((SEED_TERM.basis_id, basis_id), STRUCTURE.ridge_lambda, score)
                total_fallbacks += int(score.get("fallbacks", 0))
        with FULL_K2_TRACE.open("a", encoding="utf-8") as trace:
            for basis_id in batch:
                score = hit_rows.get(basis_id, computed[basis_id])
                row = {
                    "stage": "exhaustive_k2_validation",
                    "candidate_index": begin + batch.index(basis_id),
                    "basis_id": basis_id,
                    "support": [SEED_TERM.basis_id, basis_id],
                    "candidate_score_source": "cache_hit"
                    if basis_id in hit_rows
                    else "exact_evaluation",
                    **score,
                }
                if basis_id not in logged_candidates:
                    trace.write(json.dumps(row, ensure_ascii=False) + "\n")
                    logged_candidates.add(basis_id)
                batch_rows.append(row)
            trace.flush()
        completed = begin + len(batch)
        elapsed = time.monotonic() - total_started
        throughput = completed / elapsed if elapsed > 0 else 0.0
        remaining = len(candidate_ids) - completed
        eta = remaining / throughput if throughput > 0 else None
        checkpoint = {
            "stage": "exhaustive_k2_validation",
            "parent_support_id": support_id((SEED_TERM.basis_id,)),
            "candidate_pool_hash": pool_fp,
            "exact_evaluation_fingerprint": exact_fp,
            "effective_workers": PARALLEL.max_workers,
            "next_candidate_index": completed,
            "completed_candidate_count": completed,
            "total_candidate_count": len(candidate_ids),
            "cache_hits": total_cache_hits,
            "cache_misses": total_cache_misses,
            "fallback_count": total_fallbacks,
            "last_batch_seconds": time.monotonic() - batch_started,
            "throughput_candidates_per_second": throughput,
            "eta_seconds": eta,
            "checkpoint_status": "valid_after_atomic_replace",
        }
        _atomic_json_write(FULL_K2_CHECKPOINT, checkpoint)
        print(
            json.dumps(
                {
                    "stage": checkpoint["stage"],
                    "completed": completed,
                    "total": len(candidate_ids),
                    "progress_percent": 100.0 * completed / len(candidate_ids),
                    "effective_workers": PARALLEL.max_workers,
                    "throughput": throughput,
                    "eta_seconds": eta,
                    "cache_hits": total_cache_hits,
                    "cache_misses": total_cache_misses,
                    "fallback_count": total_fallbacks,
                    "checkpoint": str(FULL_K2_CHECKPOINT),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
    summary = {
        **validated_search_policy_fingerprint("exhaustive"),
        "stage": "exhaustive_k2_validation",
        "candidate_count": len(candidate_ids),
        "cache_hits": total_cache_hits,
        "cache_misses": total_cache_misses,
        "fallback_count": total_fallbacks,
        "trace": str(FULL_K2_TRACE),
        "checkpoint": str(FULL_K2_CHECKPOINT),
    }
    _atomic_json_write(PRE_SEARCH_ROOT / "01_full_k2_exhaustive_validation.json", summary)
    return summary


def full_k2_screening_and_recall(
    *,
    confirm_large_compute: bool,
    block_size: int = SCREENING_BLOCK_SIZE,
    partition_count: int = 40,
) -> dict[str, object]:
    """Screen the complete K2 pool with resumable partition artefacts and score recall."""
    if not confirm_large_compute:
        raise PermissionError("full K2 screening requires --confirm-large-compute")
    if partition_count < PARALLEL.max_workers or partition_count % PARALLEL.max_workers:
        raise ValueError("partition_count must be a positive multiple of ten")
    exact = validated_science_fingerprint()
    policy = validated_search_policy_fingerprint("screened")
    candidate_ids = representative_candidate_ids(len(DICTIONARY) - 1)
    pool_fp = candidate_pool_fingerprint(candidate_ids)
    parent = (SEED_TERM.basis_id,)
    partitions = tuple(
        tuple(candidate_ids[index::partition_count]) for index in range(partition_count)
    )
    root = PRE_SEARCH_ROOT / "screening_partitions"
    checkpoint_path = PRE_SEARCH_ROOT / "k2_screening_checkpoint.json"
    checkpoint: dict[str, object] = {}
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        expected = {
            "stage": "screening_k2_validation",
            "candidate_pool_hash": pool_fp,
            "exact_evaluation_fingerprint": exact["sha256"],
            "search_policy_fingerprint": policy["search_policy_fingerprint"],
            "effective_workers": PARALLEL.max_workers,
            "partition_count": partition_count,
        }
        if any(checkpoint.get(key) != value for key, value in expected.items()):
            raise ValueError("K2 screening checkpoint fingerprint/config mismatch")
    completed = {int(value) for value in checkpoint.get("completed_blocks", [])}
    partition_rows: dict[int, dict[tuple[str, ...], object]] = {}
    for index in sorted(completed):
        artifact_stem = f"partition_{index:04d}"
        arrays = load_screening_artifact(
            root,
            artifact_stem,
            {
                "stage": "screening_k2_validation",
                "partition_index": index,
                "candidate_pool_hash": pool_fp,
                "exact_evaluation_fingerprint": exact["sha256"],
                "search_policy_fingerprint": policy["search_policy_fingerprint"],
                "parent_support_id": support_id(parent),
            },
        )
        from retrieval_oriented_model_selection.shared.shareability_screening import (
            ParentScreeningScores,
        )

        partition_rows[index] = {
            parent: ParentScreeningScores(
                parent,
                tuple(str(value) for value in arrays["basis_ids"]),
                arrays["scores"],
                arrays["aend_scores"],
                arrays["c2_scores"],
                tuple(str(value) for value in arrays["strata"]),
                {},
            )
        }
    pending = [index for index in range(partition_count) if index not in completed]
    started = time.monotonic()
    previous = {name: os.environ.get(name) for name in _THREAD_VARIABLES}
    try:
        for name in _THREAD_VARIABLES:
            os.environ[name] = "1"
        with ProcessPoolExecutor(
            max_workers=PARALLEL.max_workers, mp_context=get_context("spawn")
        ) as pool:
            futures = {
                pool.submit(_screen_partition_job, partitions[index], (parent,), block_size): index
                for index in pending
            }
            for future in as_completed(futures):
                index = futures[future]
                row = future.result()
                screen_scores = row["scores"][parent]
                artifact_stem = f"partition_{index:04d}"
                metadata = {
                    "stage": "screening_k2_validation",
                    "partition_index": index,
                    "candidate_count": len(partitions[index]),
                    "candidate_pool_hash": pool_fp,
                    "exact_evaluation_fingerprint": exact["sha256"],
                    "search_policy_fingerprint": policy["search_policy_fingerprint"],
                    "parent_support_id": support_id(parent),
                }
                save_screening_artifact(
                    root,
                    artifact_stem,
                    metadata,
                    {
                        "basis_ids": np.asarray(screen_scores.candidate_ids),
                        "scores": screen_scores.scores,
                        "aend_scores": screen_scores.aend_scores,
                        "c2_scores": screen_scores.c2_scores,
                        "strata": np.asarray(screen_scores.strata),
                    },
                )
                partition_rows[index] = {parent: screen_scores}
                completed.add(index)
                elapsed = time.monotonic() - started
                candidate_done = sum(len(partitions[item]) for item in completed)
                throughput = candidate_done / elapsed if elapsed else 0.0
                eta = (len(candidate_ids) - candidate_done) / throughput if throughput else None
                checkpoint = {
                    "stage": "screening_k2_validation",
                    "candidate_pool_hash": pool_fp,
                    "exact_evaluation_fingerprint": exact["sha256"],
                    "search_policy_fingerprint": policy["search_policy_fingerprint"],
                    "effective_workers": PARALLEL.max_workers,
                    "partition_count": partition_count,
                    "completed_blocks": sorted(completed),
                    "completed_candidate_count": candidate_done,
                    "total_candidate_count": len(candidate_ids),
                    "throughput_candidates_per_second": throughput,
                    "eta_seconds": eta,
                    "checkpoint_status": "valid_after_atomic_replace",
                }
                _atomic_json_write(checkpoint_path, checkpoint)
                print(
                    json.dumps(
                        {
                            "stage": checkpoint["stage"],
                            "completed_blocks": len(completed),
                            "total_blocks": partition_count,
                            "completed_candidates": candidate_done,
                            "total_candidates": len(candidate_ids),
                            "progress_percent": 100.0 * candidate_done / len(candidate_ids),
                            "effective_workers": PARALLEL.max_workers,
                            "throughput": throughput,
                            "eta_seconds": eta,
                            "checkpoint": str(checkpoint_path),
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    from retrieval_oriented_model_selection.shared.shareability_screening import (
        merge_screening_partitions,
    )

    merged = merge_screening_partitions(
        tuple(partition_rows[index] for index in range(partition_count)), candidate_ids
    )[parent]
    exact_rows = [
        json.loads(line)
        for line in FULL_K2_TRACE.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    exact_rows.sort(key=lambda row: (-int(row["N_shareable"]), str(row["basis_id"])))
    exact_ranked = tuple(row["basis_id"] for row in exact_rows)
    shortlist = tuple(entry.basis_id for entry in merged.shortlist(SCREENING, DICTIONARY))
    pure_spec = CandidateScreeningSpec(shortlist_size=500, exploit_size=500, explore_size=0)
    pure_top500 = tuple(entry.basis_id for entry in merged.shortlist(pure_spec, DICTIONARY))

    def recall(ids: tuple[str, ...]) -> dict[str, object]:
        chosen = set(ids)
        return {
            "top1": exact_ranked[0] in chosen,
            "top3_count": sum(item in chosen for item in exact_ranked[:3]),
            "top10_count": sum(item in chosen for item in exact_ranked[:10]),
            "top50_count": sum(item in chosen for item in exact_ranked[:50]),
            "top100_count": sum(item in chosen for item in exact_ranked[:100]),
        }

    summary = {
        **policy,
        "stage": "screening_k2_validation",
        "candidate_count": len(candidate_ids),
        "exact_top": {
            f"top{size}": [
                {"basis_id": row["basis_id"], "N_shareable": row["N_shareable"]}
                for row in exact_rows[:size]
            ]
            for size in (1, 3, 10, 50, 100)
        },
        "mixed_400_plus_100": recall(shortlist),
        "pure_top500": recall(pure_top500),
        "shortlist_size": len(shortlist),
        "screening_checkpoint": str(checkpoint_path),
        "screening_partition_root": str(root),
        "exact_trace": str(FULL_K2_TRACE),
    }
    _atomic_json_write(PRE_SEARCH_ROOT / "02_k2_screening_recall.json", summary)
    return summary


def validate_screening_against_exhaustive_k2(
    *, confirm_large_compute: bool, block_size: int = SCREENING_BLOCK_SIZE
) -> dict[str, object]:
    """Explicit full K2 quality check; never called by normal maintenance modes."""
    if not confirm_large_compute:
        raise PermissionError(
            "full 38,334-candidate K2 validation requires --confirm-large-compute"
        )
    candidates = representative_candidate_ids(len(DICTIONARY) - 1)
    return screening_quality_validation(
        len(candidates), PARALLEL.max_workers, block_size, allow_large=True
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=(
            "reference-gate",
            "full-k2-exhaustive",
            "full-k2-screening",
            "exact-k2",
            "screen-k2",
            "quality-validation",
            "parallel-determinism",
        ),
        default="exact-k2",
    )
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=1000)
    parser.add_argument("--partition-count", type=int, default=40)
    parser.add_argument("--validate-screening-against-exhaustive-k2", action="store_true")
    parser.add_argument("--confirm-large-compute", action="store_true")
    args = parser.parse_args()
    if args.validate_screening_against_exhaustive_k2:
        result = validate_screening_against_exhaustive_k2(
            confirm_large_compute=args.confirm_large_compute, block_size=args.block_size
        )
    elif args.mode == "reference-gate":
        result = reference_optimized_gate(args.count)
    elif args.mode == "full-k2-exhaustive":
        result = full_k2_exhaustive_validation(
            confirm_large_compute=args.confirm_large_compute,
            batch_size=args.batch_size,
            block_size=args.block_size,
        )
    elif args.mode == "full-k2-screening":
        result = full_k2_screening_and_recall(
            confirm_large_compute=args.confirm_large_compute,
            block_size=args.block_size,
            partition_count=args.partition_count,
        )
    elif args.mode == "exact-k2":
        result = benchmark(args.count, args.workers, args.block_size)
    elif args.mode == "screen-k2":
        result = screening_benchmark(args.count, args.workers, args.block_size)
    elif args.mode == "quality-validation":
        result = screening_quality_validation(args.count, args.workers, args.block_size)
    else:
        result = screening_parallel_determinism_validation(args.count, args.block_size)
    fingerprint = str(
        result.get(
            "search_policy_fingerprint",
            result.get("science_fingerprint", "unfingerprinted"),
        )
    )
    recorded_count = int(result.get("candidate_count", args.count))
    path = (
        LOG_ROOT
        / "optimization"
        / (
            f"{args.mode}_k2_n{recorded_count}_w{args.workers}_b{args.block_size}_"
            f"{fingerprint[:12]}.json"
        )
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
