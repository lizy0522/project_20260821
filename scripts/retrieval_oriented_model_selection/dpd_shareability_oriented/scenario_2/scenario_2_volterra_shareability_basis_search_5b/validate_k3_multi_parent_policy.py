"""K3 multi-parent policy validation; never starts K4+ or formal search."""

# Validation field payloads are intentionally verbose.
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from behavior_modeling.shared.basis_function_selection.model_solver import fit_ridge
from behavior_modeling.shared.volterra_terms import build_basis_columns

from retrieval_oriented_model_selection.shared.shareability_optimized_evaluator import (
    evaluate_partition,
    rank_and_score,
)
from retrieval_oriented_model_selection.shared.shareability_structure import CandidateScoreCache
from retrieval_oriented_model_selection.shared.shareability_sufficient_stats import (
    canonical_support,
    support_id,
)

from .benchmark_optimization import (
    _THREAD_VARIABLES,
    DATASET_FILES,
    PARALLEL,
    REAL_B_SOURCE,
    _atomic_json_write,
)
from .run_search import validated_science_fingerprint
from .task_config import DICTIONARY, SEED_TERM, STRUCTURE

ROOT = Path(__file__).resolve().parents[4]
TASK_LOG_ROOT = (
    ROOT
    / "work_logs"
    / "retrieval_oriented_model_selection"
    / "dpd_shareability_oriented"
    / "scenario_2" /"scenario_2_volterra_shareability_basis_search_5b"
)
K3_ROOT = TASK_LOG_ROOT / "pre_search_validation" / "k3_multi_parent_policy_validation"
K3_CACHE_ROOT = K3_ROOT / "candidate_score_cache"
K3_TRACE = K3_ROOT / "k3_exhaustive_results.jsonl"
K3_CHECKPOINT = K3_ROOT / "k3_exhaustive_checkpoint.json"


def _load_k2_parents() -> tuple[tuple[str, ...], ...]:
    trace = TASK_LOG_ROOT / "pre_search_validation" / "k2_exhaustive_results.jsonl"
    rows = [
        json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    rows.sort(key=lambda row: (-int(row["N_shareable"]), row["basis_id"]))
    if len(rows) != 38334:
        raise RuntimeError("K2 exact ground truth row count mismatch")
    expected = ("V3_u00_u01_c00", "V5_u01_u02_u02_c01_c02", "V3_u01_u01_c00")
    actual = tuple(row["basis_id"] for row in rows[:3])
    values = tuple(int(row["N_shareable"]) for row in rows[:3])
    if actual != expected or values != (403, 399, 393):
        raise RuntimeError(f"K2 Exact Top3 mismatch: {actual} {values}")
    return tuple((SEED_TERM.basis_id, basis) for basis in actual)


def k3_validation_fingerprint(parents: tuple[tuple[str, ...], ...]) -> str:
    payload = {
        "exact_evaluation_fingerprint": validated_science_fingerprint()["sha256"],
        "parents": [list(parent) for parent in parents],
        "dictionary_size": len(DICTIONARY),
        "dmax": 4,
        "ridge_lambda": STRUCTURE.ridge_lambda,
        "state_count": 425,
        "task": "k3_multi_parent_policy_validation",
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_k3_manifest(parents: tuple[tuple[str, ...], ...]) -> dict[str, object]:
    primary: dict[tuple[str, ...], tuple[int, str]] = {}
    provenance: dict[tuple[str, ...], list[dict[str, object]]] = {}
    raw = 0
    pool = tuple(sorted(basis for basis in DICTIONARY if basis != SEED_TERM.basis_id))
    for parent_index, parent in enumerate(parents):
        for added in pool:
            if added in parent:
                continue
            raw += 1
            child = canonical_support((*parent, added))
            provenance.setdefault(child, []).append(
                {"parent_index": parent_index, "parent": list(parent), "added_basis_id": added}
            )
            primary.setdefault(child, (parent_index, added))
    unique = tuple(sorted(provenance))
    if raw != 114999 or len(unique) != 114996:
        raise RuntimeError(f"unexpected K3 proposal counts raw={raw} unique={len(unique)}")
    manifest = {
        "raw_proposals": raw,
        "unique_supports": len(unique),
        "duplicate_supports": raw - len(unique),
        "parents": [list(parent) for parent in parents],
        "support_ids": [support_id(child) for child in unique],
        "supports": [list(child) for child in unique],
        "primary_parent_index": [primary[child][0] for child in unique],
        "primary_added_basis_id": [primary[child][1] for child in unique],
        "provenance": {
            support_id(child): provenance[child] for child in unique if len(provenance[child]) > 1
        },
    }
    K3_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        K3_ROOT / "04_k3_candidate_pool_manifest.npz",
        support_ids=np.asarray(manifest["support_ids"]),
        supports=np.asarray(["|".join(child) for child in unique]),
        primary_parent_index=np.asarray(manifest["primary_parent_index"], dtype=np.int64),
        primary_added_basis_id=np.asarray(manifest["primary_added_basis_id"]),
    )
    _atomic_json_write(K3_ROOT / "04_k3_candidate_pool_manifest.json", manifest)
    return {"supports": unique, "primary": primary, "provenance": provenance, **manifest}


def _reference_score(
    child: tuple[str, ...], arrays: dict[str, np.ndarray], real_b: np.ndarray
) -> dict[str, object]:
    theta_a = np.empty((425, len(child)), complex)
    theta_c = np.empty_like(theta_a)
    for state in range(425):
        phi_a = build_basis_columns(arrays["x_aend"][state], child, DICTIONARY, dmax=4)
        phi_c = build_basis_columns(arrays["x_c2"][state], child, DICTIONARY, dmax=4)
        theta_a[state] = fit_ridge(phi_a, arrays["y_aend"][state, 4:], 1e-8).theta
        theta_c[state] = fit_ridge(phi_c, arrays["y_c2"][state, 4:], 1e-8).theta
    return rank_and_score(
        theta_a,
        theta_c,
        build_basis_columns(arrays["common_b"], child, DICTIONARY, dmax=4),
        real_b,
        include_details=True,
    )


def reference_gate(
    manifest: dict[str, object], parents: tuple[tuple[str, ...], ...]
) -> dict[str, object]:
    arrays = {name: np.load(path, mmap_mode="r") for name, path in DATASET_FILES.items()}
    real_b = np.load(REAL_B_SOURCE, mmap_mode="r")
    supports = tuple(manifest["supports"])
    selected: list[tuple[int, tuple[str, ...]]] = []
    for parent_index, parent in enumerate(parents):
        candidates = [
            child
            for child in supports
            if tuple(child[:2]) == tuple(sorted(parent)) and len(child) == 3
        ]
        chosen = [candidates[index * len(candidates) // 10] for index in range(10)]
        selected.extend((parent_index, tuple(child)) for child in chosen)
    rows = []
    for parent_index, child in selected:
        parent = parents[parent_index]
        added = next(item for item in child if item not in parent)
        optimized = evaluate_partition(
            (parent,),
            (added,),
            x_a=arrays["x_aend"],
            y_a=arrays["y_aend"],
            x_c=arrays["x_c2"],
            y_c=arrays["y_c2"],
            common_b=arrays["common_b"],
            real_b=real_b,
            dictionary=DICTIONARY,
            ridge_lambda=1e-8,
            include_details=True,
        )[0][child]
        reference = _reference_score(child, arrays, real_b)
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
                raise RuntimeError(
                    f"K3 reference mismatch parent={parent_index} child={child} field={field}"
                )
        finite = np.isfinite(reference["distance"])
        error = float(np.max(np.abs(optimized["distance"][finite] - reference["distance"][finite])))
        rows.append(
            {
                "parent_index": parent_index,
                "support": list(child),
                "max_distance_error_db": error,
                "N_shareable": reference["N_shareable"],
            }
        )
    result = {
        "stage": "k3_reference_gate",
        "supports": len(rows),
        "all_metrics_identical": True,
        "max_distance_error_db": max(row["max_distance_error_db"] for row in rows),
        "rows": rows,
    }
    _atomic_json_write(K3_ROOT / "03_k3_reference_gate.json", result)
    return result


def _k3_partition_job(
    parent: tuple[str, ...], candidates: tuple[str, ...], block_size: int
) -> dict[str, object]:
    if any(os.environ.get(name) != "1" for name in _THREAD_VARIABLES):
        raise RuntimeError("K3 worker BLAS environment is not fixed to one thread")
    arrays = {name: np.load(path, mmap_mode="r") for name, path in DATASET_FILES.items()}
    real_b = np.load(REAL_B_SOURCE, mmap_mode="r")
    scores, diagnostics = evaluate_partition(
        (parent,),
        candidates,
        x_a=arrays["x_aend"],
        y_a=arrays["y_aend"],
        x_c=arrays["x_c2"],
        y_c=arrays["y_c2"],
        common_b=arrays["common_b"],
        real_b=real_b,
        dictionary=DICTIONARY,
        ridge_lambda=1e-8,
        block_size=block_size,
    )
    return {
        "scores": {"|".join(key): score for key, score in scores.items()},
        "diagnostics": diagnostics,
    }


def _load_k3_cached(cache: CandidateScoreCache, child: tuple[str, ...]) -> dict[str, int] | None:
    try:
        return cache.load(child, 1e-8)
    except FileNotFoundError:
        return None


def full_exact_validation(
    manifest: dict[str, object], parents: tuple[tuple[str, ...], ...], *, batch_size: int = 1000
) -> dict[str, object]:
    validation_fp = k3_validation_fingerprint(parents)
    exact_fp = validated_science_fingerprint()["sha256"]
    supports = tuple(tuple(child) for child in manifest["supports"])
    primary = manifest["primary"]
    cache = CandidateScoreCache(K3_CACHE_ROOT, exact_fp, "5B")
    trace = K3_TRACE
    checkpoint: dict[str, object] = (
        json.loads(K3_CHECKPOINT.read_text()) if K3_CHECKPOINT.exists() else {}
    )
    if checkpoint and (
        checkpoint.get("validation_fingerprint") != validation_fp
        or checkpoint.get("total_unique_supports") != len(supports)
    ):
        raise ValueError("K3 exact checkpoint fingerprint/size mismatch")
    logged = set()
    if trace.exists():
        logged = {
            json.loads(line)["support_id"]
            for line in trace.read_text().splitlines()
            if line.strip()
        }
    work: list[tuple[int, tuple[str, ...]]] = []
    for parent_index, parent in enumerate(parents):
        candidates = [added for child, (owner, added) in primary.items() if owner == parent_index]
        for begin in range(0, len(candidates), batch_size):
            work.append((parent_index, tuple(candidates[begin : begin + batch_size])))
    completed = int(checkpoint.get("completed_supports", 0))
    cache_hits = int(checkpoint.get("cache_hits", 0))
    cache_misses = int(checkpoint.get("cache_misses", 0))
    completed_work = {int(value) for value in checkpoint.get("completed_work_indices", [])}
    started = time.monotonic()
    trace.parent.mkdir(parents=True, exist_ok=True)

    def publish_block(
        work_index: int,
        parent_index: int,
        candidates: tuple[str, ...],
        cached_rows: dict[tuple[str, ...], dict[str, int]],
        computed: dict[tuple[str, ...], dict[str, object]],
    ) -> None:
        nonlocal completed, cache_hits, cache_misses
        parent = parents[parent_index]
        for child, score in computed.items():
            cache.save(child, 1e-8, score)
        with trace.open("a", encoding="utf-8") as handle:
            for added in candidates:
                child = canonical_support((*parent, added))
                sid = support_id(child)
                if sid in logged:
                    continue
                score = cached_rows.get(child, computed[child])
                handle.write(
                    json.dumps(
                        {"support_id": sid, "support": list(child), "parent_index": parent_index, **score},
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                logged.add(sid)
        completed += len(candidates)
        completed_work.add(work_index)
        elapsed = time.monotonic() - started
        rate = completed / elapsed if elapsed else 0.0
        checkpoint = {
            "stage": "k3_multi_parent_exhaustive_exact",
            "validation_fingerprint": validation_fp,
            "exact_evaluation_fingerprint": exact_fp,
            "total_unique_supports": len(supports),
            "completed_supports": completed,
            "completed_work_indices": sorted(completed_work),
            "total_work_blocks": len(work),
            "cache_hits": cache_hits,
            "cache_misses": cache_misses,
            "effective_workers": PARALLEL.max_workers,
            "throughput_candidates_per_second": rate,
            "eta_seconds": (len(supports) - completed) / rate if rate else None,
            "checkpoint_status": "valid_after_atomic_replace",
        }
        _atomic_json_write(K3_CHECKPOINT, checkpoint)
        print(
            json.dumps(
                {
                    "stage": checkpoint["stage"],
                    "completed": completed,
                    "total": len(supports),
                    "progress_percent": 100 * completed / len(supports),
                    "throughput": rate,
                    "eta_seconds": checkpoint["eta_seconds"],
                    "cache_hits": cache_hits,
                    "cache_misses": cache_misses,
                    "checkpoint": str(K3_CHECKPOINT),
                }
            ),
            flush=True,
        )

    pending_indices = [index for index in range(len(work)) if index not in completed_work]
    previous_env = {name: os.environ.get(name) for name in _THREAD_VARIABLES}
    try:
        for name in _THREAD_VARIABLES:
            os.environ[name] = "1"
        with ProcessPoolExecutor(max_workers=PARALLEL.max_workers) as pool:
            for wave_start in range(0, len(pending_indices), PARALLEL.max_workers):
                futures = {}
                immediate = []
                for work_index in pending_indices[wave_start : wave_start + PARALLEL.max_workers]:
                    parent_index, candidates = work[work_index]
                    parent = parents[parent_index]
                    missing = []
                    cached_rows = {}
                    for added in candidates:
                        child = canonical_support((*parent, added))
                        value = _load_k3_cached(cache, child)
                        if value is None:
                            missing.append(added)
                        else:
                            cached_rows[child] = value
                    cache_hits += len(cached_rows)
                    cache_misses += len(missing)
                    if missing:
                        future = pool.submit(_k3_partition_job, parent, tuple(missing), 64)
                        futures[future] = (work_index, parent_index, candidates, cached_rows)
                    else:
                        immediate.append((work_index, parent_index, candidates, cached_rows, {}))
                for item in immediate:
                    publish_block(*item)
                for future in as_completed(futures):
                    work_index, parent_index, candidates, cached_rows = futures[future]
                    result = future.result()
                    computed = {tuple(key.split("|")): score for key, score in result["scores"].items()}
                    missing_count = len(candidates) - len(cached_rows)
                    if len(computed) != missing_count:
                        raise RuntimeError("K3 exact batch incomplete")
                    publish_block(work_index, parent_index, candidates, cached_rows, computed)
    finally:
        for name, value in previous_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return json.loads(K3_CHECKPOINT.read_text(encoding="utf-8"))


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("audit", "reference-gate", "exact"), default="audit")
    parser.add_argument("--confirm-large-compute", action="store_true")
    args = parser.parse_args()
    parents = _load_k2_parents()
    if args.stage == "audit":
        print(
            json.dumps(
                {
                    "parents": [list(parent) for parent in parents],
                    "validation_fingerprint": k3_validation_fingerprint(parents),
                },
                indent=2,
            )
        )
        return
    manifest = build_k3_manifest(parents)
    if args.stage == "reference-gate":
        if not args.confirm_large_compute:
            raise PermissionError("K3 reference gate requires explicit confirmation")
        print(json.dumps(reference_gate(manifest, parents), indent=2))
        return
    if not args.confirm_large_compute:
        raise PermissionError("K3 exact validation requires explicit confirmation")
    print(json.dumps(full_exact_validation(manifest, parents), indent=2))


if __name__ == "__main__":
    main()
