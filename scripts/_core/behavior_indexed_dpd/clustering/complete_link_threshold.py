"""Deterministic complete-link threshold clustering primitives.

This module deliberately knows nothing about Scenario 2, MAT files, state
metadata, or a particular signal.  It accepts a precomputed symmetric
distance/similarity matrix whose *larger* values are worse, and enforces the
strict within-cluster rule ``distance < threshold``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from itertools import combinations
from typing import Any

import numpy as np


def build_pairwise_cnmse_matrix(
    signals: np.ndarray,
    cnmse_fn: Callable[[np.ndarray, np.ndarray], float],
    *,
    symmetry_mode: str = "native_symmetric",
) -> np.ndarray:
    """Build a square pairwise matrix from row-wise signals.

    ``cnmse_fn`` is called only for the upper triangle unless
    ``symmetry_mode='max_of_two_directions'``.  The diagonal is always
    ``-inf``.  The function is intentionally serial; scenario-specific
    runners can use shared-memory workers for expensive pair evaluation.
    """

    values = np.asarray(signals)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("signals must be a non-empty 2-D array")
    if symmetry_mode not in {"native_symmetric", "max_of_two_directions"}:
        raise ValueError("unknown symmetry_mode")
    state_count = int(values.shape[0])
    matrix = np.full((state_count, state_count), -np.inf, dtype=np.float64)
    for i, j in combinations(range(state_count), 2):
        forward = float(cnmse_fn(values[i], values[j]))
        if symmetry_mode == "max_of_two_directions":
            reverse = float(cnmse_fn(values[j], values[i]))
            score = max(forward, reverse)
        else:
            score = forward
        if not np.isfinite(score):
            raise ValueError(f"non-finite off-diagonal distance at ({i},{j})")
        matrix[i, j] = score
        matrix[j, i] = score
    return matrix


def _cross_score(distance_matrix: np.ndarray, left: Sequence[int], right: Sequence[int]) -> float:
    """Return complete-link score between two member-index sequences."""

    return float(np.max(distance_matrix[np.ix_(list(left), list(right))]))


def complete_link_threshold_cluster(
    distance_matrix: np.ndarray,
    threshold: float,
) -> list[tuple[int, ...]]:
    """Agglomerate singleton clusters under a strict complete-link threshold.

    At each iteration the pair with minimum complete-link score is selected.
    Ties are resolved deterministically by larger merged size, then the
    smaller and larger minimum member index.  Merging stops when the best
    score is greater than or equal to ``threshold``.
    """

    matrix = np.asarray(distance_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("distance_matrix must be a non-empty square matrix")
    if not np.allclose(matrix, matrix.T, rtol=0.0, atol=0.0, equal_nan=False):
        raise ValueError("distance_matrix must be exactly symmetric")
    if not np.all(np.isneginf(np.diag(matrix))):
        raise ValueError("distance_matrix diagonal must be -inf")
    off_diagonal = matrix[~np.eye(matrix.shape[0], dtype=bool)]
    if not np.all(np.isfinite(off_diagonal)):
        raise ValueError("distance_matrix off-diagonal values must be finite")
    threshold = float(threshold)
    if not np.isfinite(threshold):
        raise ValueError("threshold must be finite")

    clusters: dict[int, tuple[int, ...]] = {index: (index,) for index in range(matrix.shape[0])}
    active = list(clusters)
    next_cluster_id = matrix.shape[0]
    distances: dict[tuple[int, int], float] = {
        (left, right): float(matrix[left, right])
        for left, right in combinations(active, 2)
    }

    while len(active) > 1:
        best_key: tuple[float, int, int, int, int, int] | None = None
        for position, left in enumerate(active[:-1]):
            for right in active[position + 1 :]:
                key_pair = (left, right) if left < right else (right, left)
                score = distances[key_pair]
                members = clusters[left] + clusters[right]
                ordered_members = tuple(sorted(members))
                ordered_cluster_mins = tuple(sorted((clusters[left][0], clusters[right][0])))
                candidate_key = (
                    score,
                    -len(ordered_members),
                    ordered_cluster_mins[0],
                    ordered_cluster_mins[1],
                    key_pair[0],
                    key_pair[1],
                )
                if best_key is None or candidate_key < best_key:
                    best_key = candidate_key
        if best_key is None or best_key[0] >= threshold:
            break

        _, _, _, _, left, right = best_key
        new_members = tuple(sorted(clusters[left] + clusters[right]))
        new_id = next_cluster_id
        next_cluster_id += 1
        for other in active:
            if other in {left, right}:
                continue
            old_left = distances[(min(left, other), max(left, other))]
            old_right = distances[(min(right, other), max(right, other))]
            distances[(min(new_id, other), max(new_id, other))] = max(old_left, old_right)
        for other in active:
            if other != left:
                distances.pop((min(left, other), max(left, other)), None)
            if other != right:
                distances.pop((min(right, other), max(right, other)), None)
        clusters[new_id] = new_members
        active = [item for item in active if item not in {left, right}]
        active.append(new_id)

    return sorted((clusters[item] for item in active), key=lambda members: (members[0], members))


def validate_clusters(
    distance_matrix: np.ndarray,
    clusters: Sequence[Sequence[int]],
    threshold: float,
) -> dict[str, Any]:
    """Validate coverage, strict all-pair membership, and non-mergeability."""

    matrix = np.asarray(distance_matrix, dtype=np.float64)
    threshold = float(threshold)
    normalized = [tuple(sorted(int(member) for member in cluster)) for cluster in clusters]
    flattened = [member for cluster in normalized for member in cluster]
    expected = list(range(matrix.shape[0]))
    missing = sorted(set(expected) - set(flattened))
    duplicates = sorted(member for member in set(flattened) if flattened.count(member) > 1)
    within_failures: list[dict[str, Any]] = []
    for cluster_id, members in enumerate(normalized):
        for left, right in combinations(members, 2):
            value = float(matrix[left, right])
            if not value < threshold:
                within_failures.append(
                    {"cluster_id": cluster_id, "state_i": left, "state_j": right, "distance": value}
                )

    mergeable_pairs: list[dict[str, Any]] = []
    for left_index, right_index in combinations(range(len(normalized)), 2):
        score = _cross_score(matrix, normalized[left_index], normalized[right_index])
        if score < threshold:
            mergeable_pairs.append(
                {"cluster_a": left_index, "cluster_b": right_index, "complete_link_score": score}
            )

    return {
        "coverage_pass": sorted(flattened) == expected,
        "missing_state_ids": missing,
        "duplicate_state_ids": duplicates,
        "within_cluster_pass": not within_failures,
        "within_cluster_failures": within_failures,
        "non_mergeability_pass": not mergeable_pairs,
        "mergeable_cluster_pairs": mergeable_pairs,
        "all_gates_pass": bool(
            not missing and not duplicates and not within_failures and not mergeable_pairs
        ),
    }


def summarize_clusters(
    distance_matrix: np.ndarray,
    clusters: Sequence[Sequence[int]],
) -> list[dict[str, Any]]:
    """Return one compact summary dictionary per cluster."""

    matrix = np.asarray(distance_matrix, dtype=np.float64)
    normalized = sorted(
        (tuple(sorted(int(member) for member in cluster)) for cluster in clusters),
        key=lambda members: (members[0], members),
    )
    rows: list[dict[str, Any]] = []
    for cluster_id, members in enumerate(normalized):
        pairs = [
            (left, right, float(matrix[left, right]))
            for left, right in combinations(members, 2)
        ]
        if pairs:
            values = np.asarray([item[2] for item in pairs], dtype=np.float64)
            worst = max(pairs, key=lambda item: (item[2], item[0], item[1]))
            best = min(pairs, key=lambda item: (item[2], item[0], item[1]))
            worst_pair_i, worst_pair_j = int(worst[0]), int(worst[1])
        else:
            values = np.asarray([], dtype=np.float64)
            worst = (None, None, float("-inf"))
            best = (None, None, float("-inf"))
            worst_pair_i = worst_pair_j = None
        rows.append(
            {
                "cluster_id": cluster_id,
                "cluster_size": len(members),
                "pair_count": len(pairs),
                "worst_pair_cnmse_db": float(worst[2]),
                "best_pair_cnmse_db": float(best[2]),
                "median_pair_cnmse_db": float(np.median(values)) if values.size else float("-inf"),
                "mean_pair_cnmse_db": float(np.mean(values)) if values.size else float("-inf"),
                "min_state_id": int(members[0]),
                "member_state_ids": ",".join(str(member) for member in members),
                "worst_pair_state_i": worst_pair_i,
                "worst_pair_state_j": worst_pair_j,
            }
        )
    return rows


__all__ = [
    "build_pairwise_cnmse_matrix",
    "complete_link_threshold_cluster",
    "summarize_clusters",
    "validate_clusters",
]
