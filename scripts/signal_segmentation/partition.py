"""
功能说明：根据完整xin时间长度生成唯一、连续、无重叠的A/B/C样点所有权。
输入：正整数signal_length，或时间轴位于axis=0的xin数组。
输出：不可变SegmentPartition，包含三段slice、边界和ownership长度。
用途：只定义样点归属；不执行对齐、增益调整、滤波、重采样或MP有效区裁剪。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Integral

import numpy as np

from .config import A_RATIO, B_RATIO, SEGMENT_NAMES


@dataclass(frozen=True)
class SegmentPartition:
    """完整时间轴上的唯一A/B/C连续slice。"""

    total_length: int
    a_slice: slice
    b_slice: slice
    c_slice: slice

    @property
    def n_a(self) -> int:
        return self.a_slice.stop - self.a_slice.start

    @property
    def n_b(self) -> int:
        return self.b_slice.stop - self.b_slice.start

    @property
    def n_c(self) -> int:
        return self.c_slice.stop - self.c_slice.start

    def slice_for(self, segment_name: str) -> slice:
        """返回A、B或C对应slice。"""
        slices = {"A": self.a_slice, "B": self.b_slice, "C": self.c_slice}
        try:
            return slices[segment_name]
        except KeyError as exc:
            raise KeyError(f"segment_name必须属于{SEGMENT_NAMES}") from exc

    def as_dict(self) -> dict[str, int]:
        """返回便于JSON/CSV记录的边界和长度。"""
        return {
            "total_length": self.total_length,
            "a_start": self.a_slice.start,
            "a_stop": self.a_slice.stop,
            "n_a": self.n_a,
            "b_start": self.b_slice.start,
            "b_stop": self.b_slice.stop,
            "n_b": self.n_b,
            "c_start": self.c_slice.start,
            "c_stop": self.c_slice.stop,
            "n_c": self.n_c,
        }


def build_partition(signal_length: int) -> SegmentPartition:
    """使用floor(0.5N)、floor(0.2N)、余数归C构建Partition。"""
    if not isinstance(signal_length, Integral) or isinstance(signal_length, bool):
        raise TypeError("signal_length必须是整数")
    normalized_length = int(signal_length)
    if normalized_length <= 0:
        raise ValueError("signal_length必须为正数")

    n_a = math.floor(A_RATIO * normalized_length)
    n_b = math.floor(B_RATIO * normalized_length)
    n_c = normalized_length - n_a - n_b
    if min(n_a, n_b, n_c) <= 0:
        raise ValueError("signal_length过短，A/B/C必须均含至少一个样点")

    a_stop = n_a
    b_stop = n_a + n_b
    return SegmentPartition(
        total_length=normalized_length,
        a_slice=slice(0, a_stop),
        b_slice=slice(a_stop, b_stop),
        c_slice=slice(b_stop, normalized_length),
    )


def build_partition_from_xin(xin: np.ndarray) -> SegmentPartition:
    """正式科研入口：仅使用xin的axis=0长度建立唯一Partition。"""
    if not isinstance(xin, np.ndarray):
        raise TypeError("xin必须是numpy.ndarray")
    if xin.ndim < 1:
        raise ValueError("xin必须至少包含时间轴")
    return build_partition(xin.shape[0])


__all__ = ["SegmentPartition", "build_partition", "build_partition_from_xin"]
