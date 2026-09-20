"""
功能说明：定义并验证state0正向Memory Polynomial模型的冻结结构。
输入：可选的模型配置字典。
输出：阶数、阶数相关记忆深度、基函数清单和固定规模常量。
用途：保证X类和所有Y类模型使用完全相同的10系数结构。
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

MP_CONFIG: dict[str, Any] = {
    "orders": [1, 2, 3, 5, 7, 9],
    "memory_depth": {1: 3, 2: 2, 3: 2, 5: 1, 7: 1, 9: 1},
}

NUM_COEFFICIENTS = 10


def get_basis_terms(
    orders: Sequence[int],
    memory_depth: Mapping[int, int],
) -> tuple[tuple[int, int], ...]:
    """按 order优先、memory次优先返回 ``(order, memory)`` 基函数顺序。"""
    terms: list[tuple[int, int]] = []
    seen_orders: set[int] = set()
    for order in orders:
        if not isinstance(order, Integral) or isinstance(order, bool) or order <= 0:
            raise ValueError(f"非线性阶数必须是正整数，实际为{order!r}")
        normalized_order = int(order)
        if normalized_order in seen_orders:
            raise ValueError(f"非线性阶数重复：{normalized_order}")
        seen_orders.add(normalized_order)
        if normalized_order not in memory_depth:
            raise KeyError(f"memory_depth缺少order={normalized_order}")
        depth = memory_depth[normalized_order]
        if not isinstance(depth, Integral) or isinstance(depth, bool) or depth <= 0:
            raise ValueError(f"order={normalized_order}的memory depth必须是正整数")
        terms.extend((normalized_order, memory) for memory in range(int(depth)))
    unexpected = set(memory_depth) - seen_orders
    if unexpected:
        raise ValueError(f"memory_depth包含未使用阶数：{sorted(unexpected)}")
    return tuple(terms)


def validate_mp_config(config: Mapping[str, Any]) -> tuple[tuple[int, int], ...]:
    """验证配置并返回冻结列顺序；当前任务必须严格得到10个基函数。"""
    try:
        orders = config["orders"]
        memory_depth = config["memory_depth"]
    except KeyError as exc:
        raise KeyError("模型配置必须包含orders和memory_depth") from exc
    if not isinstance(orders, Sequence) or isinstance(orders, (str, bytes)):
        raise TypeError("orders必须是整数序列")
    if not isinstance(memory_depth, Mapping):
        raise TypeError("memory_depth必须是映射")
    terms = get_basis_terms(orders, memory_depth)
    if len(terms) != NUM_COEFFICIENTS:
        raise ValueError(f"当前冻结模型必须有{NUM_COEFFICIENTS}个基函数，实际为{len(terms)}")
    return terms


BASIS_TERMS = validate_mp_config(MP_CONFIG)
MAX_DELAY = max(memory for _, memory in BASIS_TERMS)

__all__ = [
    "BASIS_TERMS",
    "MAX_DELAY",
    "MP_CONFIG",
    "NUM_COEFFICIENTS",
    "get_basis_terms",
    "validate_mp_config",
]
