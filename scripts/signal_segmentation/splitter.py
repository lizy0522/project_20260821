"""
功能说明：沿axis=0使用既有Partition切分一个或多个同步波形。
输入：时间长度等于partition.total_length的NumPy数组或同长度数组映射。
输出：A/B/C数组视图；二维ILC矩阵的第二维保持不变。
用途：只执行固定样点所有权切分，不重建边界、不对齐、不调整增益。
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .config import SEGMENT_NAMES, TIME_AXIS
from .partition import SegmentPartition


def _validate_time_axis(signal: np.ndarray, partition: SegmentPartition, name: str) -> None:
    if not isinstance(signal, np.ndarray):
        raise TypeError(f"{name}必须是numpy.ndarray")
    if signal.ndim < 1:
        raise ValueError(f"{name}必须包含axis=0时间轴")
    if signal.shape[TIME_AXIS] != partition.total_length:
        raise ValueError(
            f"{name}时间长度{signal.shape[TIME_AXIS]}与Partition长度{partition.total_length}不一致"
        )


def split_signal(
    signal: np.ndarray,
    partition: SegmentPartition,
) -> dict[str, np.ndarray]:
    """返回共享原数组存储的A/B/C axis=0视图。"""
    _validate_time_axis(signal, partition, "signal")
    return {
        segment_name: signal[partition.slice_for(segment_name)] for segment_name in SEGMENT_NAMES
    }


def split_signals(
    signals: Mapping[str, np.ndarray],
    partition: SegmentPartition,
) -> dict[str, dict[str, np.ndarray]]:
    """同步切分多个波形，返回 ``signal_name -> A/B/C`` 映射。"""
    if not isinstance(signals, Mapping) or not signals:
        raise ValueError("signals必须是非空映射")
    result: dict[str, dict[str, np.ndarray]] = {}
    for name, signal in signals.items():
        _validate_time_axis(signal, partition, name)
        result[name] = split_signal(signal, partition)
    return result


__all__ = ["split_signal", "split_signals"]
