"""将 CNMSE 距离行转换为平均排名，并计算逐状态 Spearman。"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from scipy.stats import rankdata, spearmanr

STATE_COUNT = 425


def distance_to_ranks(distance_matrix: np.ndarray) -> np.ndarray:
    """按行对距离升序排名，保留 ``-inf``，并列使用平均排名。"""

    distance = np.asarray(distance_matrix)
    if distance.ndim != 2 or distance.shape != (STATE_COUNT, STATE_COUNT):
        raise ValueError(f"distance_matrix必须是(425,425)，实际为{distance.shape}")
    if np.isnan(distance).any() or np.isposinf(distance).any():
        raise ValueError("distance_matrix不允许NaN或+Inf")
    ranks = rankdata(distance, axis=1, method="average")
    if ranks.shape != distance.shape or not np.all(np.isfinite(ranks)):
        raise RuntimeError("排名矩阵无效")
    return ranks.astype(np.float64, copy=False)


def build_ranking_matrices(
    distance_matrices: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """对四个命名距离矩阵逐行执行 ``rankdata(method='average')``。"""

    expected = ("D_RR", "D_CX", "D_CY", "D_CR")
    missing = [name for name in expected if name not in distance_matrices]
    if missing:
        raise ValueError(f"缺少距离矩阵：{missing}")
    return {
        name.replace("D_", "R_", 1): distance_to_ranks(distance_matrices[name])
        for name in expected
    }


def spearman_statistic(reference_rank: np.ndarray, candidate_rank: np.ndarray) -> float:
    """返回两个排名向量的 Spearman statistic，不返回 p-value。"""

    reference = np.asarray(reference_rank, dtype=float)
    candidate = np.asarray(candidate_rank, dtype=float)
    if reference.ndim != 1 or candidate.ndim != 1 or reference.shape != candidate.shape:
        raise ValueError("Spearman输入必须是等长一维向量")
    if (
        reference.size < 2
        or not np.all(np.isfinite(reference))
        or not np.all(np.isfinite(candidate))
    ):
        raise ValueError("Spearman输入必须是至少2个有限值")
    statistic = float(spearmanr(reference, candidate).statistic)
    if not np.isfinite(statistic) or statistic < -1.0 - 1e-12 or statistic > 1.0 + 1e-12:
        raise RuntimeError(f"Spearman结果无效：{statistic}")
    return float(np.clip(statistic, -1.0, 1.0))


def compute_spearman_by_state(
    ranking_matrices: Mapping[str, np.ndarray],
) -> np.ndarray:
    """以 ``R_RR[i,:]`` 为唯一参考，计算三类 LUT 的逐状态 Spearman。"""

    expected = ("R_RR", "R_CX", "R_CY", "R_CR")
    missing = [name for name in expected if name not in ranking_matrices]
    if missing:
        raise ValueError(f"缺少排名矩阵：{missing}")
    reference = np.asarray(ranking_matrices["R_RR"])
    if reference.shape != (STATE_COUNT, STATE_COUNT):
        raise ValueError("R_RR必须是(425,425)")
    values = np.empty((STATE_COUNT, 3), dtype=np.float64)
    for state_id in range(STATE_COUNT):
        reference_row = reference[state_id, :]
        values[state_id, 0] = spearman_statistic(
            reference_row,
            np.asarray(ranking_matrices["R_CX"])[state_id, :],
        )
        values[state_id, 1] = spearman_statistic(
            reference_row,
            np.asarray(ranking_matrices["R_CY"])[state_id, :],
        )
        values[state_id, 2] = spearman_statistic(
            reference_row,
            np.asarray(ranking_matrices["R_CR"])[state_id, :],
        )
    if not np.all(np.isfinite(values)) or np.any(values < -1) or np.any(values > 1):
        raise RuntimeError("逐状态Spearman存在非有限值或超出[-1,1]")
    return values


__all__ = [
    "build_ranking_matrices",
    "compute_spearman_by_state",
    "distance_to_ranks",
    "spearman_statistic",
]
