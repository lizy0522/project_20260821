"""整合 Scenario 2 指纹、距离、排名和三类逐状态 Spearman。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from data_manager import build_state_table

from .distance_matrix import build_distance_matrices
from .fingerprint_builder import FingerprintResult, FrozenRidgeModels
from .ranking import build_ranking_matrices, compute_spearman_by_state

STATE_COUNT = 425
LUT_LABELS = ("X-A LUT", "Y-A LUT", "Real-B LUT")
SPEARMAN_COLUMNS = (
    "spearman_X_A_LUT",
    "spearman_Y_A_LUT",
    "spearman_Real_B_LUT",
)


@dataclass(frozen=True)
class Scenario2RankingResult:
    """完整排名一致性分析结果。"""

    frozen_models: FrozenRidgeModels
    fingerprints: FingerprintResult
    distance_matrices: dict[str, np.ndarray]
    ranking_matrices: dict[str, np.ndarray]
    spearman_values: np.ndarray
    spearman_by_state: pd.DataFrame
    spearman_summary: pd.DataFrame
    validation: dict[str, Any]


def _stats(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("Spearman统计不能没有有限值")
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "q25": float(np.quantile(finite, 0.25)),
        "q75": float(np.quantile(finite, 0.75)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def build_spearman_by_state_table(spearman_values: np.ndarray) -> pd.DataFrame:
    """生成425行、保持state_id固定顺序的Spearman长表。"""

    values = np.asarray(spearman_values, dtype=float)
    if values.shape != (STATE_COUNT, 3):
        raise ValueError(f"spearman_values shape错误：{values.shape}")
    if not np.all(np.isfinite(values)) or np.any(values < -1) or np.any(values > 1):
        raise ValueError("Spearman值必须全部finite且位于[-1,1]")
    state_frame = pd.DataFrame(build_state_table())
    table = state_frame.loc[
        :, ["state_id", "funMng", "funAng", "secMng", "secAng", "Vm", "Pin"]
    ].copy()
    for index, column in enumerate(SPEARMAN_COLUMNS):
        table[column] = values[:, index]
    return table


def build_spearman_summary(spearman_values: np.ndarray) -> pd.DataFrame:
    """生成三行 LUT 聚合统计表。"""

    values = np.asarray(spearman_values, dtype=float)
    if values.shape != (STATE_COUNT, 3):
        raise ValueError(f"spearman_values shape错误：{values.shape}")
    rows: list[dict[str, Any]] = []
    for index, label in enumerate(LUT_LABELS):
        row: dict[str, Any] = {"lut_type": label}
        row.update(_stats(values[:, index]))
        rows.append(row)
    columns = ["lut_type", "count", "mean", "median", "std", "q25", "q75", "min", "max"]
    return pd.DataFrame(rows, columns=columns)


def _matrix_validation(
    distance_matrices: dict[str, np.ndarray],
    ranking_matrices: dict[str, np.ndarray],
) -> dict[str, Any]:
    distance_shapes = {name: list(matrix.shape) for name, matrix in distance_matrices.items()}
    ranking_shapes = {name: list(matrix.shape) for name, matrix in ranking_matrices.items()}
    for name, matrix in distance_matrices.items():
        if matrix.shape != (STATE_COUNT, STATE_COUNT):
            raise RuntimeError(f"{name} shape错误：{matrix.shape}")
        if np.isnan(matrix).any() or np.isposinf(matrix).any():
            raise RuntimeError(f"{name}出现NaN或+Inf")
    for name, matrix in ranking_matrices.items():
        if matrix.shape != (STATE_COUNT, STATE_COUNT) or not np.all(np.isfinite(matrix)):
            raise RuntimeError(f"{name} shape或有限性错误")

    d_rr = distance_matrices["D_RR"]
    row_min = np.min(d_rr, axis=1)
    diagonal = d_rr[np.arange(STATE_COUNT), np.arange(STATE_COUNT)]
    self_row_min_count = int(np.count_nonzero(diagonal <= row_min))
    if self_row_min_count != STATE_COUNT:
        raise RuntimeError("D_RR存在self不是每行最相似状态的行")
    r_rr = ranking_matrices["R_RR"]
    self_rank = r_rr[np.arange(STATE_COUNT), np.arange(STATE_COUNT)]
    self_min_tied_count = int(np.count_nonzero(self_rank <= np.min(r_rr, axis=1)))
    if self_min_tied_count != STATE_COUNT:
        raise RuntimeError("R_RR存在self不在最小并列排名的行")
    return {
        "D_RR_shape": distance_shapes["D_RR"],
        "D_CX_shape": distance_shapes["D_CX"],
        "D_CY_shape": distance_shapes["D_CY"],
        "D_CR_shape": distance_shapes["D_CR"],
        "R_RR_shape": ranking_shapes["R_RR"],
        "R_CX_shape": ranking_shapes["R_CX"],
        "R_CY_shape": ranking_shapes["R_CY"],
        "R_CR_shape": ranking_shapes["R_CR"],
        "D_RR_negative_infinity_count": int(np.count_nonzero(np.isneginf(d_rr))),
        "D_CX_negative_infinity_count": int(
            np.count_nonzero(np.isneginf(distance_matrices["D_CX"]))
        ),
        "D_CY_negative_infinity_count": int(
            np.count_nonzero(np.isneginf(distance_matrices["D_CY"]))
        ),
        "D_CR_negative_infinity_count": int(
            np.count_nonzero(np.isneginf(distance_matrices["D_CR"]))
        ),
        "D_RR_self_row_min_count": self_row_min_count,
        "R_RR_self_min_tied_rank_count": self_min_tied_count,
        "R_RR_self_rank_mean": float(np.mean(self_rank)),
        "candidate_count_per_query": STATE_COUNT,
        "self_match_included": True,
    }


def build_scenario2_ranking_analysis(
    frozen_models: FrozenRidgeModels,
    fingerprints: FingerprintResult,
) -> Scenario2RankingResult:
    """完成距离→排名→Spearman整合分析。"""

    distance_matrices = build_distance_matrices(fingerprints)
    ranking_matrices = build_ranking_matrices(distance_matrices)
    spearman_values = compute_spearman_by_state(ranking_matrices)
    by_state = build_spearman_by_state_table(spearman_values)
    summary = build_spearman_summary(spearman_values)
    matrix_validation = _matrix_validation(distance_matrices, ranking_matrices)
    validation: dict[str, Any] = {
        "scenario": 2,
        "state_count": STATE_COUNT,
        "candidate_count_per_query": STATE_COUNT,
        "self_match_included": True,
        "ridge_lambda": frozen_models.ridge_lambda,
        "frozen_theta_source": str(frozen_models.source_path),
        "model_ids": list(frozen_models.model_ids),
        "fingerprint_length": fingerprints.X_A_fingerprints.shape[1],
        "X_A_fingerprint_shape": list(fingerprints.X_A_fingerprints.shape),
        "Y_A_fingerprint_shape": list(fingerprints.Y_A_fingerprints.shape),
        "Y_C_query_shape": list(fingerprints.Y_C_query_fingerprints.shape),
        "real_B_shape": list(fingerprints.real_B_fingerprints.shape),
        "common_B_input_shape": list(fingerprints.common_B_input.shape),
        "common_B_input_consistent": fingerprints.common_B_consistent,
        "common_B_max_abs_difference": fingerprints.common_B_max_abs_difference,
        "spearman_X_count": int(spearman_values[:, 0].size),
        "spearman_Y_count": int(spearman_values[:, 1].size),
        "spearman_real_count": int(spearman_values[:, 2].size),
        "all_spearman_finite": bool(np.all(np.isfinite(spearman_values))),
        "spearman_range_min": float(np.min(spearman_values)),
        "spearman_range_max": float(np.max(spearman_values)),
        "only_ilc_iteration_1_used": True,
        "used_frozen_ridge_models": True,
        "retrained_behavior_models": False,
        "distance_direction": (
            "CNMSE(Query, Candidate); more negative is more similar; ascending rank"
        ),
        "reference_ranking": "R_RR = Real-B to Real-B",
        "matrix_validation": matrix_validation,
    }
    validation.update(matrix_validation)
    if fingerprints.common_B_input.size != 4915:
        raise RuntimeError("公共B输入长度不是4915")
    return Scenario2RankingResult(
        frozen_models=frozen_models,
        fingerprints=fingerprints,
        distance_matrices=distance_matrices,
        ranking_matrices=ranking_matrices,
        spearman_values=spearman_values,
        spearman_by_state=by_state,
        spearman_summary=summary,
        validation=validation,
    )


__all__ = [
    "LUT_LABELS",
    "SPEARMAN_COLUMNS",
    "Scenario2RankingResult",
    "build_scenario2_ranking_analysis",
    "build_spearman_by_state_table",
    "build_spearman_summary",
]
