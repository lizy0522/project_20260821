"""
功能说明：计算两个已预处理输出波形之间的复波形归一化均方误差。
输入：等长的一维数值 NumPy 数组 y_ref、y_test。
输出：以 dB 表示的 CNMSE 标量；完全相同时返回 -Inf。
数学定义：CNMSE=10*log10(sum(|y_ref-y_test|^2)/sum(|y_ref|^2))。
"""

from __future__ import annotations

import numpy as np

from ..utils import validate_pair


def cnmse(y_ref: np.ndarray, y_test: np.ndarray) -> float:
    """只计算 CNMSE 公式，不执行对齐、复增益调整或归一化。"""
    reference, test = validate_pair(y_ref, y_test, "y_ref", "y_test")
    denominator = float(np.sum(np.abs(reference) ** 2))
    if denominator == 0:
        raise ValueError("y_ref 的能量为零，CNMSE 分母无定义")
    numerator = float(np.sum(np.abs(reference - test) ** 2))
    if numerator == 0:
        return float("-inf")
    return float(10 * np.log10(numerator / denominator))


__all__ = ["cnmse"]
