"""Validate K3 screening policies against the completed exhaustive exact trace.

This task-specific validation computes only SOMP residual-correlation scores for
the three fixed K2 Exact parents.  It never calls the exact evaluator and never
modifies task_config.py or starts the formal K2-K20 search.
"""

# Validation payloads intentionally contain long provenance fields.
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np

from retrieval_oriented_model_selection.shared.shareability_persistence import (
    load_screening_artifact,
    save_screening_artifact,
)
from retrieval_oriented_model_selection.shared.shareability_screening import (
    ParentScreeningScores,
    candidate_pool_fingerprint,
    screen_candidate_pool,
)
from retrieval_oriented_model_selection.shared.shareability_sufficient_stats import (
    canonical_support,
    support_id,
)
from retrieval_oriented_model_selection.shared.shareability_types import CandidateScreeningSpec

from .benchmark_optimization import _THREAD_VARIABLES, DATASET_FILES
from .task_config import DICTIONARY, PARALLEL, SCREENING_BLOCK_SIZE, SEED_TERM, STRUCTURE
from .validate_k3_multi_parent_policy import K3_ROOT, _load_k2_parents

SCREEN_ROOT = K3_ROOT / "screening_score_partitions"
CHECKPOINT_PATH = K3_ROOT / "k3_screening_policy_checkpoint.json"
SUMMARY_PATH = K3_ROOT / "08_k3_screening_policy_validation.json"
PARTITION_COUNT = 40
SCORE_SPEC = CandidateScreeningSpec(
    shortlist_size=len(DICTIONARY) - 1,
    exploit_size=len(DICTIONARY) - 1,
    explore_size=0,
    aend_weight=0.5,
    c2_weight=0.5,
)
POLICIES = {
    "policy_a": CandidateScreeningSpec(
        shortlist_size=3000,
        exploit_size=2100,
        explore_size=900,
        aend_weight=0.5,
        c2_weight=0.5,
    ),
    "policy_b": CandidateScreeningSpec(
        shortlist_size=5000,
        exploit_size=3000,
        explore_size=2000,
        aend_weight=0.5,
        c2_weight=0.5,
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _completed_k3_checkpoint() -> dict[str, object]:
    if not (K3_ROOT / "k3_exhaustive_checkpoint.json").is_file():
        raise RuntimeError("completed K3 exact checkpoint is missing")
    checkpoint = json.loads((K3_ROOT / "k3_exhaustive_checkpoint.json").read_text(encoding="utf-8"))
    if checkpoint.get("completed_supports") != 114996 or checkpoint.get("total_unique_supports") != 114996:
        raise RuntimeError("K3 exact checkpoint is not complete")
    exact_fingerprint = checkpoint.get("exact_evaluation_fingerprint")
    validation_fingerprint = checkpoint.get("validation_fingerprint")
    if not isinstance(exact_fingerprint, str) or len(exact_fingerprint) != 64:
        raise RuntimeError("K3 exact checkpoint has no valid exact-evaluation fingerprint")
    if not isinstance(validation_fingerprint, str) or len(validation_fingerprint) != 64:
        raise RuntimeError("K3 exact checkpoint has no valid validation fingerprint")
    return checkpoint


def _screening_score_fingerprint(
    parents: tuple[tuple[str, ...], ...], candidate_ids: tuple[str, ...], checkpoint: dict[str, object]
) -> str:
    implementation = Path(__file__).resolve().parents[2] / "shared" / "shareability_screening.py"
    payload = {
        "stage": "k3_multi_parent_screening_scores",
        "exact_evaluation_fingerprint": checkpoint["exact_evaluation_fingerprint"],
        "k3_validation_fingerprint": checkpoint["validation_fingerprint"],
        "parents": [list(parent) for parent in parents],
        "candidate_pool_fingerprint": candidate_pool_fingerprint(candidate_ids),
        "score_spec": SCORE_SPEC.to_dict(),
        "implementation_sha256": _sha256(implementation),
        "dmax": 4,
        "ridge_lambda": STRUCTURE.ridge_lambda,
        "state_count": 425,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _screen_partition_job(
    candidate_ids: tuple[str, ...], parents: tuple[tuple[str, ...], ...], block_size: int
) -> dict[str, object]:
    if any(os.environ.get(name) != "1" for name in _THREAD_VARIABLES):
        raise RuntimeError("one BLAS thread per spawned worker required")
    started = time.monotonic()
    arrays = {name: np.load(path, mmap_mode="r") for name, path in DATASET_FILES.items()}
    scores = screen_candidate_pool(
        parents,
        candidate_ids,
        x_a=arrays["x_aend"],
        y_a=arrays["y_aend"],
        x_c=arrays["x_c2"],
        y_c=arrays["y_c2"],
        dictionary=DICTIONARY,
        ridge_lambda=STRUCTURE.ridge_lambda,
        spec=SCORE_SPEC,
        block_size=block_size,
    )
    return {
        "pid": os.getpid(),
        "elapsed_seconds": time.monotonic() - started,
        "max_rss_bytes_macos": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "scores": scores,
    }


def _artifact_arrays(
    rows: dict[tuple[str, ...], ParentScreeningScores],
    parents: tuple[tuple[str, ...], ...],
) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    for index, parent in enumerate(parents):
        row = rows[parent]
        valid = np.isfinite(row.scores) & np.asarray(
            [basis not in parent for basis in row.candidate_ids], dtype=bool
        )
        arrays[f"parent_{index}_basis_ids"] = np.asarray(
            [basis for basis, keep in zip(row.candidate_ids, valid, strict=True) if keep]
        )
        arrays[f"parent_{index}_scores"] = np.asarray(row.scores[valid], dtype=np.float64)
        arrays[f"parent_{index}_aend_scores"] = np.asarray(
            row.aend_scores[valid], dtype=np.float64
        )
        arrays[f"parent_{index}_c2_scores"] = np.asarray(row.c2_scores[valid], dtype=np.float64)
        arrays[f"parent_{index}_strata"] = np.asarray(
            [stratum for stratum, keep in zip(row.strata, valid, strict=True) if keep]
        )
    return arrays


def _load_partition(
    index: int,
    parents: tuple[tuple[str, ...], ...],
    candidate_ids: tuple[str, ...],
    score_fingerprint: str,
) -> dict[tuple[str, ...], ParentScreeningScores]:
    metadata = {
        "stage": "k3_multi_parent_screening_scores",
        "partition_index": index,
        "candidate_pool_fingerprint": candidate_pool_fingerprint(candidate_ids),
        "score_fingerprint": score_fingerprint,
        "parent_support_ids": [support_id(parent) for parent in parents],
        "candidate_count": len(tuple(candidate_ids[index::PARTITION_COUNT])),
    }
    arrays = load_screening_artifact(SCREEN_ROOT, f"partition_{index:04d}", metadata)
    result: dict[tuple[str, ...], ParentScreeningScores] = {}
    for parent_index, parent in enumerate(parents):
        prefix = f"parent_{parent_index}_"
        result[parent] = ParentScreeningScores(
            parent,
            tuple(str(value) for value in arrays[prefix + "basis_ids"]),
            arrays[prefix + "scores"],
            arrays[prefix + "aend_scores"],
            arrays[prefix + "c2_scores"],
            tuple(str(value) for value in arrays[prefix + "strata"]),
            {},
        )
    return result


def _save_partition(
    index: int,
    rows: dict[tuple[str, ...], ParentScreeningScores],
    parents: tuple[tuple[str, ...], ...],
    candidate_ids: tuple[str, ...],
    score_fingerprint: str,
) -> None:
    metadata = {
        "stage": "k3_multi_parent_screening_scores",
        "partition_index": index,
        "candidate_pool_fingerprint": candidate_pool_fingerprint(candidate_ids),
        "score_fingerprint": score_fingerprint,
        "parent_support_ids": [support_id(parent) for parent in parents],
        "candidate_count": len(tuple(candidate_ids[index::PARTITION_COUNT])),
    }
    save_screening_artifact(
        SCREEN_ROOT,
        f"partition_{index:04d}",
        metadata,
        _artifact_arrays(rows, parents),
    )


def _merge_partitions(
    partitions: tuple[dict[tuple[str, ...], ParentScreeningScores], ...],
    parents: tuple[tuple[str, ...], ...],
    candidate_ids: tuple[str, ...],
) -> dict[tuple[str, ...], ParentScreeningScores]:
    if len(partitions) != PARTITION_COUNT:
        raise RuntimeError("K3 screening partition count mismatch")
    merged: dict[tuple[str, ...], ParentScreeningScores] = {}
    for parent in parents:
        expected = tuple(basis for basis in candidate_ids if basis not in parent)
        by_id: dict[str, tuple[float, float, float, str]] = {}
        diagnostics: dict[str, int | float] = {}
        for partition in partitions:
            row = partition[parent]
            for basis, score, aend, c2, stratum in zip(
                row.candidate_ids,
                row.scores,
                row.aend_scores,
                row.c2_scores,
                row.strata,
                strict=True,
            ):
                if basis in by_id:
                    raise RuntimeError(f"duplicate K3 screening basis {basis}")
                by_id[basis] = (float(score), float(aend), float(c2), stratum)
            for key, value in row.diagnostics.items():
                diagnostics[key] = diagnostics.get(key, 0) + value
        if tuple(sorted(by_id)) != tuple(sorted(expected)):
            raise RuntimeError(
                f"K3 screening pool mismatch for parent {parent}: "
                f"got={len(by_id)} expected={len(expected)}"
            )
        values = [by_id[basis] for basis in expected]
        merged[parent] = ParentScreeningScores(
            parent,
            expected,
            np.asarray([value[0] for value in values], dtype=np.float64),
            np.asarray([value[1] for value in values], dtype=np.float64),
            np.asarray([value[2] for value in values], dtype=np.float64),
            tuple(value[3] for value in values),
            diagnostics,
        )
    return merged


def _exact_rankings(parents: tuple[tuple[str, ...], ...]) -> tuple[dict[int, list[dict[str, Any]]], list[dict[str, Any]]]:
    trace = K3_ROOT / "k3_exhaustive_results.jsonl"
    rows = [json.loads(line) for line in trace.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != 114996 or len({row["support_id"] for row in rows}) != 114996:
        raise RuntimeError("K3 exact trace is incomplete or duplicated")
    manifest = json.loads((K3_ROOT / "04_k3_candidate_pool_manifest.json").read_text(encoding="utf-8"))
    sid_to_parents: dict[str, set[int]] = {
        support_id: {int(parent_index)}
        for support_id, parent_index in zip(
            manifest["support_ids"], manifest["primary_parent_index"], strict=True
        )
    }
    for sid, origins in manifest["provenance"].items():
        sid_to_parents.setdefault(sid, set()).update(int(origin["parent_index"]) for origin in origins)
    by_parent: dict[int, list[dict[str, Any]]] = {}
    for parent_index in range(len(parents)):
        by_parent[parent_index] = sorted(
            [row for row in rows if parent_index in sid_to_parents[row["support_id"]]],
            key=lambda row: (-int(row["N_shareable"]), row["support_id"]),
        )
    return by_parent, sorted(rows, key=lambda row: (-int(row["N_shareable"]), row["support_id"]))


def _recall(chosen: set[str], ranked: list[dict[str, Any]]) -> dict[str, int | bool]:
    return {
        "top1": ranked[0]["support_id"] in chosen,
        "top3": sum(row["support_id"] in chosen for row in ranked[:3]),
        "top10": sum(row["support_id"] in chosen for row in ranked[:10]),
        "top20": sum(row["support_id"] in chosen for row in ranked[:20]),
        "top50": sum(row["support_id"] in chosen for row in ranked[:50]),
        "top100": sum(row["support_id"] in chosen for row in ranked[:100]),
    }


def _policy_result(
    name: str,
    spec: CandidateScreeningSpec,
    scores: dict[tuple[str, ...], ParentScreeningScores],
    parents: tuple[tuple[str, ...], ...],
    exact_by_parent: dict[int, list[dict[str, Any]]],
    exact_global: list[dict[str, Any]],
) -> dict[str, object]:
    selected_by_parent: dict[int, set[str]] = {}
    shortlist_details: dict[str, list[dict[str, object]]] = {}
    parent_payload: dict[str, object] = {}
    for parent_index, parent in enumerate(parents):
        entries = scores[parent].shortlist(spec, DICTIONARY)
        selected: set[str] = set()
        details: list[dict[str, object]] = []
        for entry in entries:
            child = canonical_support((*parent, entry.basis_id))
            sid = support_id(child)
            selected.add(sid)
            details.append(
                {
                    "support_id": sid,
                    "added_basis_id": entry.basis_id,
                    "screen_rank": entry.rank,
                    "screen_score": entry.score,
                    "proposal_source": entry.proposal_source,
                }
            )
        selected_by_parent[parent_index] = selected
        shortlist_details[str(parent_index)] = details
        parent_payload[str(parent_index)] = {
            "parent_support": list(parent),
            "screened_candidates": len(entries),
            "selected_unique_supports": len(selected),
            "recall": _recall(selected, exact_by_parent[parent_index]),
            "exact_top10": [
                {"support_id": row["support_id"], "N_shareable": row["N_shareable"], "selected": row["support_id"] in selected}
                for row in exact_by_parent[parent_index][:10]
            ],
        }
    union = set().union(*selected_by_parent.values())
    global_payload = {
        "screened_candidate_union": len(union),
        "duplicate_selection_across_parents": sum(len(values) for values in selected_by_parent.values()) - len(union),
        "recall": _recall(union, exact_global),
        "exact_top10": [
            {"support_id": row["support_id"], "N_shareable": row["N_shareable"], "selected": row["support_id"] in union}
            for row in exact_global[:10]
        ],
    }
    return {
        "policy": name,
        "spec": spec.to_dict(),
        "parents": parent_payload,
        "global_union": global_payload,
        "shortlist_details": shortlist_details,
    }


def run() -> dict[str, object]:
    parents = _load_k2_parents()
    exact_checkpoint = _completed_k3_checkpoint()
    candidate_ids = tuple(sorted(basis for basis in DICTIONARY if basis != SEED_TERM.basis_id))
    if len(candidate_ids) != 38334:
        raise RuntimeError(f"unexpected K3 screening candidate pool size: {len(candidate_ids)}")
    pool_fp = candidate_pool_fingerprint(candidate_ids)
    score_fp = _screening_score_fingerprint(parents, candidate_ids, exact_checkpoint)
    SCREEN_ROOT.mkdir(parents=True, exist_ok=True)
    expected_checkpoint = {
        "stage": "k3_multi_parent_screening_scores",
        "candidate_pool_fingerprint": pool_fp,
        "score_fingerprint": score_fp,
        "partition_count": PARTITION_COUNT,
        "effective_workers": PARALLEL.max_workers,
        "parent_support_ids": [support_id(parent) for parent in parents],
        "total_candidate_count": len(candidate_ids),
    }
    checkpoint: dict[str, object] = {}
    if CHECKPOINT_PATH.exists():
        checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        if any(checkpoint.get(key) != value for key, value in expected_checkpoint.items()):
            raise RuntimeError("K3 screening checkpoint fingerprint/config mismatch")
    partitions = tuple(tuple(candidate_ids[index::PARTITION_COUNT]) for index in range(PARTITION_COUNT))
    completed = {int(value) for value in checkpoint.get("completed_blocks", [])}
    partition_rows: dict[int, dict[tuple[str, ...], ParentScreeningScores]] = {}
    for index in sorted(completed):
        partition_rows[index] = _load_partition(index, parents, candidate_ids, score_fp)
    pending = [index for index in range(PARTITION_COUNT) if index not in completed]
    started = time.monotonic()
    previous = {name: os.environ.get(name) for name in _THREAD_VARIABLES}
    try:
        for name in _THREAD_VARIABLES:
            os.environ[name] = "1"
        if pending:
            with ProcessPoolExecutor(
                max_workers=PARALLEL.max_workers, mp_context=get_context("spawn")
            ) as pool:
                futures = {
                    pool.submit(_screen_partition_job, partitions[index], parents, SCREENING_BLOCK_SIZE): index
                    for index in pending
                }
                for future in as_completed(futures):
                    index = futures[future]
                    result = future.result()
                    rows = result["scores"]
                    if not isinstance(rows, dict):
                        raise RuntimeError("K3 screening worker returned invalid score mapping")
                    _save_partition(index, rows, parents, candidate_ids, score_fp)
                    # Re-read the committed artifact so the in-memory merge uses
                    # the same finite, parent-excluded rows that were persisted.
                    partition_rows[index] = _load_partition(
                        index, parents, candidate_ids, score_fp
                    )
                    completed.add(index)
                    elapsed = time.monotonic() - started
                    done = sum(len(partitions[item]) for item in completed)
                    throughput = done / elapsed if elapsed else 0.0
                    eta = (len(candidate_ids) - done) / throughput if throughput else None
                    checkpoint = {
                        **expected_checkpoint,
                        "completed_blocks": sorted(completed),
                        "completed_candidate_count": done,
                        "throughput_candidates_per_second": throughput,
                        "eta_seconds": eta,
                        "checkpoint_status": "valid_after_atomic_replace",
                        "last_worker_pid": result["pid"],
                        "last_worker_elapsed_seconds": result["elapsed_seconds"],
                        "last_worker_rss_bytes_macos": result["max_rss_bytes_macos"],
                    }
                    temporary = CHECKPOINT_PATH.with_suffix(".tmp")
                    temporary.write_text(
                        json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
                    )
                    os.replace(temporary, CHECKPOINT_PATH)
                    print(
                        json.dumps(
                            {
                                "stage": "k3_multi_parent_screening_scores",
                                "completed_blocks": len(completed),
                                "total_blocks": PARTITION_COUNT,
                                "completed_candidates": done,
                                "total_candidates": len(candidate_ids),
                                "progress_percent": 100.0 * done / len(candidate_ids),
                                "effective_workers": PARALLEL.max_workers,
                                "throughput": throughput,
                                "eta_seconds": eta,
                                "worker_pid": result["pid"],
                                "worker_rss_bytes_macos": result["max_rss_bytes_macos"],
                                "checkpoint": str(CHECKPOINT_PATH),
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
    if len(partition_rows) != PARTITION_COUNT:
        raise RuntimeError("K3 screening did not complete all partitions")
    merged = _merge_partitions(
        tuple(partition_rows[index] for index in range(PARTITION_COUNT)), parents, candidate_ids
    )
    exact_by_parent, exact_global = _exact_rankings(parents)
    policy_results = {
        name: _policy_result(name, spec, merged, parents, exact_by_parent, exact_global)
        for name, spec in POLICIES.items()
    }
    final = {
        **expected_checkpoint,
        "stage": "k3_screening_policy_validation",
        "completed_blocks": sorted(completed),
        "completed_candidate_count": len(candidate_ids),
        "screening_score_fingerprint": score_fp,
        "exact_trace": str(K3_ROOT / "k3_exhaustive_results.jsonl"),
        "candidate_score_cache_reused": True,
        "exact_evaluator_called": False,
        "policies": policy_results,
    }
    temporary = SUMMARY_PATH.with_suffix(".tmp")
    temporary.write_text(json.dumps(final, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, SUMMARY_PATH)
    return final


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2))
