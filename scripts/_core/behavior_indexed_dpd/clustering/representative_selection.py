"""Generic minimax-medoid representative selection for distance clusters."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RepresentativeResult:
    """Representative state and its within-cluster similarity diagnostics."""

    cluster_id: int
    representative_state_id: int
    cluster_size: int
    worst_cnmse_db: float
    median_cnmse_db: float
    mean_cnmse_db: float


def _validate_members(cluster_members: Sequence[int], state_count: int) -> tuple[int, ...]:
    members = tuple(sorted(int(member) for member in cluster_members))
    if not members:
        raise ValueError("cluster_members must not be empty")
    if len(set(members)) != len(members):
        raise ValueError("cluster_members must not contain duplicates")
    if members[0] < 0 or members[-1] >= state_count:
        raise ValueError("cluster member is outside distance matrix range")
    return members


def select_cluster_representative(
    cluster_members: Sequence[int],
    pairwise_cnmse_db: np.ndarray,
    *,
    cluster_id: int = 0,
) -> RepresentativeResult:
    """Select a deterministic minimax medoid from one cluster.

    For each candidate member ``i`` the primary score is the worst CNMSE to
    any other member.  Ties are resolved by the median score, mean score, and
    finally the state ID, all ascending because more-negative CNMSE is better.
    Singleton clusters use ``-inf`` for all three scores.
    """

    matrix = np.asarray(pairwise_cnmse_db, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError("pairwise_cnmse_db must be a non-empty square matrix")
    members = _validate_members(cluster_members, matrix.shape[0])
    if len(members) == 1:
        return RepresentativeResult(int(cluster_id), members[0], 1, -np.inf, -np.inf, -np.inf)

    candidates: list[tuple[tuple[float, float, float, int], int]] = []
    for state_id in members:
        values = np.asarray(
            [matrix[state_id, other] for other in members if other != state_id],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("within-cluster CNMSE values must be finite")
        worst = float(np.max(values))
        median = float(np.median(values))
        mean = float(np.mean(values))
        candidates.append(((worst, median, mean, state_id), state_id))
    selected_key, representative = min(candidates, key=lambda item: item[0])
    return RepresentativeResult(
        int(cluster_id),
        int(representative),
        len(members),
        float(selected_key[0]),
        float(selected_key[1]),
        float(selected_key[2]),
    )


def select_all_cluster_representatives(
    clusters: Sequence[Sequence[int]],
    pairwise_cnmse_db: np.ndarray,
) -> tuple[RepresentativeResult, ...]:
    """Select one representative per cluster after stable min-state sorting."""

    matrix = np.asarray(pairwise_cnmse_db, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("pairwise_cnmse_db must be square")
    normalized = sorted(
        (_validate_members(cluster, matrix.shape[0]) for cluster in clusters),
        key=lambda members: (members[0], members),
    )
    return tuple(
        select_cluster_representative(members, matrix, cluster_id=cluster_id)
        for cluster_id, members in enumerate(normalized)
    )


__all__ = [
    "RepresentativeResult",
    "select_all_cluster_representatives",
    "select_cluster_representative",
]
