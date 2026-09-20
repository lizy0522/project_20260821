"""
功能说明：使用冻结MP结构和已拟合系数预测有效区间的正向行为输出。
输入：一维复信号x、形状为(10,)的theta和模型配置。
输出：长度为N-2的复数预测波形。
用途：保证预测使用与系数提取完全相同的基函数定义和列顺序。
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .basis import build_mp_basis
from .config import MP_CONFIG, NUM_COEFFICIENTS, validate_mp_config


def predict_behavior(
    x: np.ndarray,
    theta: np.ndarray,
    config: Mapping[str, Any] = MP_CONFIG,
) -> np.ndarray:
    """计算 ``Phi @ theta``，不补回最大延迟对应的前置样本。"""
    validate_mp_config(config)
    if not isinstance(theta, np.ndarray):
        raise TypeError("theta必须是numpy.ndarray")
    if theta.shape != (NUM_COEFFICIENTS,):
        raise ValueError(f"theta形状必须为({NUM_COEFFICIENTS},)，实际为{theta.shape}")
    if not np.iscomplexobj(theta) or not np.all(np.isfinite(theta)):
        raise ValueError("theta必须是有限复数数组")
    basis = build_mp_basis(x, config["orders"], config["memory_depth"])
    return basis @ theta


__all__ = ["predict_behavior"]
