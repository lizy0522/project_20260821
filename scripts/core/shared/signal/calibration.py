"""
功能说明：迁移 MATLAB adjust.m，用单一复增益使 y 最小二乘匹配参考信号 x。
输入：等长的一维 complex NumPy 数组 x、y。
输出：复增益调整后的 y_adjusted，以及实际施加到 y 的 complex gain。
数学定义：gain=(y^H x)/(y^H y)，y_adjusted=gain*y。
"""

from __future__ import annotations

import numpy as np

from ..utils import validate_pair


def adjust_complex_gain(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, complex]:
    """只执行复增益最小二乘匹配，不做峰值或功率归一化。"""
    x_checked, y_checked = validate_pair(
        x,
        y,
        "x",
        "y",
        require_complex=True,
    )
    denominator = np.vdot(y_checked, y_checked)
    if denominator == 0:
        raise ValueError("y 的能量为零，无法计算复增益")
    gain = np.vdot(y_checked, x_checked) / denominator
    return y_checked * gain, complex(gain)


__all__ = ["adjust_complex_gain"]
