"""
功能说明：从data_manager返回的状态字典中提取OFF、ILC和STALE三类唯一合法原始数据对。
输入：包含MAT业务变量的state_data、ILC零基列索引或唯一Partition。
输出：一维完整RawBehaviorPair，或固定公共probe xin_B。
用途：冻结字段关系和ILC峰值归一化顺序，不允许调用方任意组合输入输出字段。
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import Any

import numpy as np

from .partition import SegmentPartition
from .splitter import split_signal
from .validation import validate_ilc_histories, validate_peak_normalized


@dataclass(frozen=True)
class RawBehaviorPair:
    """进入完整记录同步前的实际输入与原始PA输出。"""

    pair_type: str
    input_full: np.ndarray
    output_raw_full: np.ndarray
    iteration_index: int | None = None
    input_peak_normalization_factor: float | None = None


def _require_field(state_data: dict[str, Any], variable_name: str) -> np.ndarray:
    if variable_name not in state_data:
        raise KeyError(f"state_data缺少变量{variable_name!r}")
    return np.asarray(state_data[variable_name])


def _as_complex_vector(value: np.ndarray, variable_name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or not np.iscomplexobj(array):
        raise ValueError(f"{variable_name}必须是一维复数波形或(N,1)复数列")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{variable_name}包含NaN或Inf")
    return array


def get_off_pair(state_data: dict[str, Any]) -> RawBehaviorPair:
    """严格返回 ``xin -> yout_withoutdpd_ori`` 原始OFF数据对。"""
    return RawBehaviorPair(
        pair_type="OFF",
        input_full=_as_complex_vector(_require_field(state_data, "xin"), "xin"),
        output_raw_full=_as_complex_vector(
            _require_field(state_data, "yout_withoutdpd_ori"),
            "yout_withoutdpd_ori",
        ),
    )


def get_ilc_pair(state_data: dict[str, Any], iteration_index: int) -> RawBehaviorPair:
    """逐列峰值归一化ILC输入，并匹配同列原始输出。"""
    if not isinstance(iteration_index, Integral) or isinstance(iteration_index, bool):
        raise TypeError("iteration_index必须是零基整数")
    normalized_index = int(iteration_index)
    input_history = _require_field(state_data, "xin_pd_ori_ilc")
    output_history = _require_field(state_data, "yout_withdpd_ori_ilc")
    nth_inter_value = state_data.get("nth_inter")
    nth_inter = None
    if nth_inter_value is not None:
        nth_inter = int(np.asarray(nth_inter_value).reshape(-1)[0])
    iterations = validate_ilc_histories(input_history, output_history, nth_inter)
    if not 0 <= normalized_index < iterations:
        raise IndexError(f"iteration_index必须位于[0, {iterations - 1}]")

    x_raw = input_history[:, normalized_index]
    peak = float(np.max(np.abs(x_raw)))
    if peak <= 0:
        raise ValueError(f"第{normalized_index + 1}列ILC输入峰值必须大于0")
    x_actual = x_raw / peak
    validate_peak_normalized(x_actual)
    return RawBehaviorPair(
        pair_type="ILC",
        input_full=x_actual,
        output_raw_full=output_history[:, normalized_index],
        iteration_index=normalized_index,
        input_peak_normalization_factor=peak,
    )


def get_stale_pair(state_data: dict[str, Any]) -> RawBehaviorPair:
    """严格返回 ``xin_pd_nominal -> yout_withdpd_stale_ori`` 原始STALE数据对。"""
    return RawBehaviorPair(
        pair_type="STALE",
        input_full=_as_complex_vector(
            _require_field(state_data, "xin_pd_nominal"),
            "xin_pd_nominal",
        ),
        output_raw_full=_as_complex_vector(
            _require_field(state_data, "yout_withdpd_stale_ori"),
            "yout_withdpd_stale_ori",
        ),
    )


def get_common_probe(
    state_data: dict[str, Any],
    partition: SegmentPartition,
) -> np.ndarray:
    """返回唯一Partition对应的公共探测输入xin_B，不做额外处理。"""
    xin = _as_complex_vector(_require_field(state_data, "xin"), "xin")
    return split_signal(xin, partition)["B"]


__all__ = [
    "RawBehaviorPair",
    "get_common_probe",
    "get_ilc_pair",
    "get_off_pair",
    "get_stale_pair",
]
