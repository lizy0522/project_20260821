"""Reusable clustering primitives for behavior-distance matrices."""

from .behavior_clustering import BehaviorClusteringResult, cluster_behavior_states
from .complete_link_threshold import (
    build_pairwise_cnmse_matrix,
    complete_link_threshold_cluster,
    summarize_clusters,
    validate_clusters,
)
from .representative_selection import (
    RepresentativeResult,
    select_all_cluster_representatives,
    select_cluster_representative,
)

__all__ = [
    "build_pairwise_cnmse_matrix",
    "BehaviorClusteringResult",
    "cluster_behavior_states",
    "complete_link_threshold_cluster",
    "RepresentativeResult",
    "select_all_cluster_representatives",
    "select_cluster_representative",
    "summarize_clusters",
    "validate_clusters",
]
