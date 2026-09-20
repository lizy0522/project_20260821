"""
功能说明：计算参考信号 x 与输出信号 y 之间的归一化均方误差。
输入：等长的一维数值 NumPy 数组 x、y。
输出：以 dB 表示的 NMSE 标量；完全相同时返回 -Inf。
数学定义：NMSE=10*log10(sum(|x-y|^2)/sum(|x|^2))。
"""

from __future__ import annotations

import numpy as np

from ..utils import validate_pair


def nmse(x: np.ndarray, y: np.ndarray) -> float:
    """只计算 NMSE 公式，不执行对齐、复增益调整或归一化。"""
    x_checked, y_checked = validate_pair(x, y, "x", "y")
    denominator = float(np.sum(np.abs(x_checked) ** 2))
    if denominator == 0:
        raise ValueError("x 的能量为零，NMSE 分母无定义")
    numerator = float(np.sum(np.abs(x_checked - y_checked) ** 2))
    if numerator == 0:
        return float("-inf")
    return float(10 * np.log10(numerator / denominator))


__all__ = ["nmse"]
