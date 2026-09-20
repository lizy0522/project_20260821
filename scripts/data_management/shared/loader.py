"""
功能说明：通过 state_id、六状态参数或变量名统一只读加载MAT数据。
输入：state_id，或 funMng、funAng、secMng、secAng、Vm、Pin，以及可选变量名。
输出：去除MAT元数据项的变量字典，或指定的 numpy.ndarray。
用途：屏蔽文件路径、深层Windows路径和 scipy.io.loadmat 调用细节。
"""

from __future__ import annotations

import os
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
from scipy.io import loadmat

from .file_index import get_file_by_id
from .state_index import get_state_id


def _as_io_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return f"\\\\?\\{resolved}"
    return resolved


def _clean_mat_dict(data: dict[str, Any]) -> dict[str, Any]:
    return {name: value for name, value in data.items() if not name.startswith("__")}


def load_by_id(state_id: int) -> dict[str, Any]:
    """根据 state_id 读取对应MAT的全部业务变量，不压缩数组维度。"""
    path = get_file_by_id(state_id)
    data = loadmat(
        _as_io_path(path),
        appendmat=False,
        verify_compressed_data_integrity=True,
    )
    return _clean_mat_dict(data)


def load_by_condition(
    funMng: Real,
    funAng: Real,
    secMng: Real,
    secAng: Real,
    Vm: Real,
    Pin: Real,
) -> dict[str, Any]:
    """根据六个状态参数定位 state_id 并读取全部业务变量。"""
    state_id = get_state_id(funMng, funAng, secMng, secAng, Vm, Pin)
    return load_by_id(state_id)


def load_variable_by_id(state_id: int, variable_name: str) -> np.ndarray:
    """只从指定 state_id 的MAT中读取一个变量。"""
    if not isinstance(variable_name, str) or not variable_name:
        raise ValueError("variable_name 必须是非空字符串")
    path = get_file_by_id(state_id)
    data = loadmat(
        _as_io_path(path),
        appendmat=False,
        variable_names=[variable_name],
        verify_compressed_data_integrity=True,
    )
    if variable_name not in data:
        raise KeyError(f"state_id={state_id}的MAT中不存在变量{variable_name!r}")
    value = data[variable_name]
    if not isinstance(value, np.ndarray):
        raise TypeError(f"变量{variable_name!r}不是 numpy.ndarray")
    return value


__all__ = ["load_by_condition", "load_by_id", "load_variable_by_id"]
