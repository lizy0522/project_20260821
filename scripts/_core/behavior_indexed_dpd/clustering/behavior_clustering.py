"""High-level clustering result combining Complete-Link and representatives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .complete_link_threshold import complete_link_threshold_cluster, validate_clusters
from .representative_selection import (
    RepresentativeResult,
    select_all_cluster_representatives,
)


@dataclass(frozen=True)
class BehaviorClusteringResult:
    """Stable clusters, state-to-cluster map, and one representative per class."""

    threshold_db: float
    clusters: tuple[tuple[int, ...], ...]
    state_to_cluster: np.ndarray
    representative_state_ids: np.ndarray
    representatives: tuple[RepresentativeResult, ...]

    def representative_for_state(self, state_id: int) -> int:
        """Return the representative state ID for a member state."""

        state_id = int(state_id)
        if state_id < 0 or state_id >= self.state_to_cluster.size:
            raise IndexError("state_id outside clustering result")
        return int(self.representative_state_ids[int(self.state_to_cluster[state_id])])


def cluster_behavior_states(
    pairwise_cnmse_db: np.ndarray,
    threshold_db: float,
) -> BehaviorClusteringResult:
    """Run Complete-Link clustering, validate it, and select representatives."""

    matrix = np.asarray(pairwise_cnmse_db, dtype=np.float64)
    threshold_db = float(threshold_db)
    clusters = tuple(complete_link_threshold_cluster(matrix, threshold_db))
    validation = validate_clusters(matrix, clusters, threshold_db)
    if not validation["all_gates_pass"]:
        raise ValueError(f"cluster validation failed: {validation}")
    representatives = select_all_cluster_representatives(clusters, matrix)
    state_to_cluster = np.full(matrix.shape[0], -1, dtype=np.int64)
    for cluster_id, members in enumerate(clusters):
        for state_id in members:
            if state_to_cluster[state_id] != -1:
                raise ValueError("state assigned to multiple clusters")
            state_to_cluster[state_id] = cluster_id
    if np.any(state_to_cluster < 0):
        raise ValueError("some state is missing from clustering result")
    representative_state_ids = np.asarray(
        [item.representative_state_id for item in representatives], dtype=np.int64
    )
    return BehaviorClusteringResult(
        threshold_db=threshold_db,
        clusters=clusters,
        state_to_cluster=state_to_cluster,
        representative_state_ids=representative_state_ids,
        representatives=representatives,
    )


__all__ = ["BehaviorClusteringResult", "cluster_behavior_states"]
