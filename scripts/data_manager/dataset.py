"""
功能说明：提供425个PA状态数据的长度、按state_id访问和顺序遍历接口。
输入：构造时无参数；索引访问时输入 state_id。
输出：指定状态或迭代状态的完整MAT业务变量字典。
用途：为行为模型、LUT、聚类等后续任务提供统一数据集入口。
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

from .file_index import STATE_FILE_MAP
from .loader import load_by_id


class PAStateDataset:
    """按 d0 state_id 顺序访问425个状态；不缓存MAT内容。"""

    def __len__(self) -> int:
        return len(STATE_FILE_MAP)

    def __getitem__(self, state_id: int) -> dict[str, Any]:
        return load_by_id(state_id)

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for state_id in range(len(self)):
            yield load_by_id(state_id)


__all__ = ["PAStateDataset"]
