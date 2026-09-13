"""
功能说明：使用无正则复数最小二乘提取正向Memory Polynomial行为系数。
输入：等长一维复信号x、y和冻结模型配置。
输出：10个复数系数；内部同时生成秩、奇异值和残差诊断。
用途：禁止Ridge、矩阵求逆和正规方程，只调用numpy.linalg.lstsq。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from _core.utils import validate_pair

from .basis import build_mp_basis
from .config import MAX_DELAY, MP_CONFIG, NUM_COEFFICIENTS, validate_mp_config


@dataclass(frozen=True)
class LeastSquaresDiagnostics:
    """一次复数最小二乘拟合的数值诊断。"""

    sample_count: int
    coefficient_count: int
    rank: int
    residual_l2: float
    singular_value_max: float
    singular_value_min: float
    condition_number: float

    def as_dict(self) -> dict[str, int | float]:
        return {
            "sample_count": self.sample_count,
            "coefficient_count": self.coefficient_count,
            "rank": self.rank,
            "residual_l2": self.residual_l2,
            "singular_value_max": self.singular_value_max,
            "singular_value_min": self.singular_value_min,
            "condition_number": self.condition_number,
        }


def _solve_basis_ols(
    basis: np.ndarray,
    y_valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """冻结OLS数值路径；Ridge的lambda=0必须直接调用本函数。"""
    return np.linalg.lstsq(basis, y_valid, rcond=None)


def _solve_behavior_coefficient(
    x: np.ndarray,
    y: np.ndarray,
    config: Mapping[str, Any] = MP_CONFIG,
) -> tuple[np.ndarray, LeastSquaresDiagnostics]:
    x_checked, y_checked = validate_pair(x, y, "x", "y", require_complex=True)
    terms = validate_mp_config(config)
    basis = build_mp_basis(x_checked, config["orders"], config["memory_depth"])
    y_valid = y_checked[MAX_DELAY:]
    theta, _, rank, singular_values = _solve_basis_ols(basis, y_valid)

    if theta.shape != (NUM_COEFFICIENTS,):
        raise RuntimeError(f"theta形状错误：{theta.shape}")
    if rank != len(terms):
        raise RuntimeError(f"设计矩阵秩不足：rank={rank}, expected={len(terms)}")
    if not np.all(np.isfinite(theta)):
        raise RuntimeError("最小二乘得到非有限系数")

    residual_l2 = float(np.linalg.norm(basis @ theta - y_valid))
    singular_max = float(singular_values[0])
    singular_min = float(singular_values[-1])
    diagnostics = LeastSquaresDiagnostics(
        sample_count=basis.shape[0],
        coefficient_count=basis.shape[1],
        rank=int(rank),
        residual_l2=residual_l2,
        singular_value_max=singular_max,
        singular_value_min=singular_min,
        condition_number=float(singular_max / singular_min),
    )
    return theta, diagnostics


def extract_behavior_coefficient(
    x: np.ndarray,
    y: np.ndarray,
    config: Mapping[str, Any] = MP_CONFIG,
) -> np.ndarray:
    """返回形状严格为(10,)的复数行为系数。"""
    theta, _ = _solve_behavior_coefficient(x, y, config)
    return theta


__all__ = ["LeastSquaresDiagnostics", "extract_behavior_coefficient"]
