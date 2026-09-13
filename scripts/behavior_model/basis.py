"""
功能说明：生成包含偶数二阶项的阶数相关记忆深度复基带MP设计矩阵。
输入：一维复信号x、非线性阶数序列和各阶记忆深度。
输出：形状为(N-max_delay, num_terms)的复数设计矩阵Phi。
用途：为正向行为系数最小二乘拟合和模型预测提供完全一致的基函数列。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from _core.utils import as_numeric_vector

from .config import get_basis_terms


def build_mp_basis(
    x: np.ndarray,
    orders: Sequence[int],
    memory_depth: Mapping[int, int],
) -> np.ndarray:
    """按 ``x[n-m]*abs(x[n-m])**(p-1)`` 构造order-major设计矩阵。"""
    x_checked = as_numeric_vector(x, "x", require_complex=True)
    terms = get_basis_terms(orders, memory_depth)
    max_delay = max(memory for _, memory in terms)
    if x_checked.size <= max_delay:
        raise ValueError(f"x长度必须大于最大延迟{max_delay}")

    valid_length = x_checked.size - max_delay
    basis = np.empty((valid_length, len(terms)), dtype=np.complex128)
    for column, (order, memory) in enumerate(terms):
        start = max_delay - memory
        stop = x_checked.size - memory
        delayed = x_checked[start:stop]
        basis[:, column] = delayed * np.abs(delayed) ** (order - 1)
    return basis


__all__ = ["build_mp_basis"]
