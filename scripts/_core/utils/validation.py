"""
功能说明：为共享信号处理与指标函数提供统一的一维数值数组验证。
输入：NumPy 数值数组及参数名称。
输出：通过验证的原数组，不复制、不修改输入。
数学定义：不执行数学变换，只检查类型、维度、长度和有限性。
"""

from __future__ import annotations

import numpy as np


def as_numeric_vector(
    value: np.ndarray,
    name: str,
    *,
    require_complex: bool = False,
) -> np.ndarray:
    """验证并返回非空的一维有限数值数组。"""
    if not isinstance(value, np.ndarray):
        raise TypeError(f"{name} 必须是 numpy.ndarray")
    if value.ndim != 1:
        raise ValueError(f"{name} 必须是一维数组，实际 shape={value.shape}")
    if value.size == 0:
        raise ValueError(f"{name} 不能为空")
    if value.dtype.kind not in "iufc":
        raise TypeError(f"{name} 必须是数值数组，实际 dtype={value.dtype}")
    if require_complex and not np.iscomplexobj(value):
        raise TypeError(f"{name} 必须是复数数组")
    if not np.all(np.isfinite(value)):
        raise ValueError(f"{name} 包含 NaN 或 Inf")
    return value


def validate_pair(
    first: np.ndarray,
    second: np.ndarray,
    first_name: str,
    second_name: str,
    *,
    require_complex: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """验证两个数组均合规且 shape 完全一致。"""
    first_checked = as_numeric_vector(
        first,
        first_name,
        require_complex=require_complex,
    )
    second_checked = as_numeric_vector(
        second,
        second_name,
        require_complex=require_complex,
    )
    if first_checked.shape != second_checked.shape:
        raise ValueError(
            f"{first_name} 与 {second_name} 的 shape 必须一致，"
            f"实际为 {first_checked.shape} 和 {second_checked.shape}"
        )
    return first_checked, second_checked


__all__ = ["as_numeric_vector", "validate_pair"]
