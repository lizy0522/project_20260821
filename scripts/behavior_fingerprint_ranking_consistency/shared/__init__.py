"""
Scenario 2 行为指纹逐状态排序一致性分析。

本包只使用已经冻结的 Scenario 2 Ridge 正向模型和公共 B probe，生成 X-A、Y-A、
Y-C Query、Real-B 四类指纹，并以 Real-B→Real-B 排序为参考计算三类 Spearman。
"""

from .distance_matrix import build_distance_matrices, compute_cnmse_distance_matrix
from .fingerprint_builder import (
    EXPECTED_MODEL_IDS,
    FINGERPRINT_LENGTH,
    FingerprintResult,
    FrozenRidgeModels,
    build_scenario2_fingerprints,
    load_frozen_ridge_models,
)
from .plotting import (
    plot_all_ilc_spearman_2x5,
    plot_all_ilc_spearman_2x5_y_a_only,
    plot_all_ilc_spearman_boxplot,
    plot_all_ilc_spearman_by_state,
)
from .ranking import (
    build_ranking_matrices,
    compute_spearman_by_state,
    distance_to_ranks,
    spearman_statistic,
)
from .scenario2_analysis import (
    LUT_LABELS,
    Scenario2RankingResult,
    build_scenario2_ranking_analysis,
    build_spearman_by_state_table,
    build_spearman_summary,
)

__all__ = [
    "EXPECTED_MODEL_IDS",
    "FINGERPRINT_LENGTH",
    "FrozenRidgeModels",
    "FingerprintResult",
    "LUT_LABELS",
    "Scenario2RankingResult",
    "build_distance_matrices",
    "build_ranking_matrices",
    "build_scenario2_fingerprints",
    "build_scenario2_ranking_analysis",
    "build_spearman_by_state_table",
    "build_spearman_summary",
    "compute_cnmse_distance_matrix",
    "compute_spearman_by_state",
    "distance_to_ranks",
    "load_frozen_ridge_models",
    "plot_all_ilc_spearman_2x5",
    "plot_all_ilc_spearman_2x5_y_a_only",
    "plot_all_ilc_spearman_boxplot",
    "plot_all_ilc_spearman_by_state",
    "spearman_statistic",
]
