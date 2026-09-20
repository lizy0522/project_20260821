"""
功能说明：执行actual input→FULL Rough→FULL Fine→ABC→逐段复增益调整的canonical流水线。
输入：完整实际PA输入、完整原始PA输出、由xin生成的唯一Partition及可选元数据。
输出：包含A/B/C canonical输入输出、边界、增益和完整同步延迟的结构化结果。
用途：统一OFF、逐列ILC和STALE行为数据；禁止全记录先adjust或每段重新同步。
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import numpy as np
from core.shared.signal import adjust_complex_gain, fine_align, rough_align

from .behavior_pairs import get_ilc_pair, get_off_pair, get_stale_pair
from .config import FINE_ALIGN_SUBTIME, SEGMENT_NAMES
from .partition import SegmentPartition
from .splitter import split_signals
from .validation import (
    validate_canonical_segments,
    validate_full_pair,
    validate_ilc_histories,
)


@dataclass(frozen=True)
class CanonicalSegment:
    """单个ownership段的canonical输入和段内独立调整输出。"""

    name: str
    input: np.ndarray
    output: np.ndarray
    start: int
    stop: int
    ownership_length: int
    output_adjust_gain: complex


@dataclass(frozen=True)
class CanonicalSegments:
    """一次完整数据对预处理产生的A/B/C canonical集合与同步元数据。"""

    pair_type: str
    partition: SegmentPartition
    segments: Mapping[str, CanonicalSegment]
    rough_delay: int
    fraction_delay: float
    iteration_index: int | None = None
    input_peak_normalization_factor: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "segments", MappingProxyType(dict(self.segments)))

    def __getitem__(self, segment_name: str) -> CanonicalSegment:
        return self.segments[segment_name]


def preprocess_full_pair(
    input_full: np.ndarray,
    output_raw_full: np.ndarray,
    partition: SegmentPartition,
    *,
    pair_type: str = "UNSPECIFIED",
    iteration_index: int | None = None,
    input_peak_normalization_factor: float | None = None,
    subtime: int = FINE_ALIGN_SUBTIME,
) -> CanonicalSegments:
    """严格执行完整同步、统一切分、逐段adjust，不调用_core.preprocess_pair。"""
    validate_full_pair(input_full, output_raw_full, partition)
    output_rough, rough_delay = rough_align(input_full, output_raw_full)
    output_aligned, fraction_delay = fine_align(
        input_full,
        output_rough,
        subtime=subtime,
    )
    split = split_signals(
        {"input": input_full, "output_aligned": output_aligned},
        partition,
    )

    canonical_segments: dict[str, CanonicalSegment] = {}
    for segment_name in SEGMENT_NAMES:
        segment_slice = partition.slice_for(segment_name)
        input_segment = split["input"][segment_name]
        output_segment = split["output_aligned"][segment_name]
        output_adjusted, gain = adjust_complex_gain(input_segment, output_segment)
        canonical_segments[segment_name] = CanonicalSegment(
            name=segment_name,
            input=input_segment,
            output=output_adjusted,
            start=segment_slice.start,
            stop=segment_slice.stop,
            ownership_length=segment_slice.stop - segment_slice.start,
            output_adjust_gain=gain,
        )

    result = CanonicalSegments(
        pair_type=pair_type,
        partition=partition,
        segments=canonical_segments,
        rough_delay=rough_delay,
        fraction_delay=fraction_delay,
        iteration_index=iteration_index,
        input_peak_normalization_factor=input_peak_normalization_factor,
    )
    validate_canonical_segments(result)
    return result


def build_off_segments(
    state_data: dict[str, Any],
    partition: SegmentPartition,
) -> CanonicalSegments:
    """生成 ``xin + yout_withoutdpd_ori`` 的canonical OFF A/B/C。"""
    pair = get_off_pair(state_data)
    return preprocess_full_pair(
        pair.input_full,
        pair.output_raw_full,
        partition,
        pair_type=pair.pair_type,
    )


def build_ilc_segments(
    state_data: dict[str, Any],
    partition: SegmentPartition,
) -> tuple[CanonicalSegments, ...]:
    """逐列独立归一化、完整同步和分段调整全部ILC迭代。"""
    input_history = np.asarray(state_data["xin_pd_ori_ilc"])
    output_history = np.asarray(state_data["yout_withdpd_ori_ilc"])
    nth_inter_value = state_data.get("nth_inter")
    nth_inter = None
    if nth_inter_value is not None:
        nth_inter = int(np.asarray(nth_inter_value).reshape(-1)[0])
    iterations = validate_ilc_histories(input_history, output_history, nth_inter)
    results: list[CanonicalSegments] = []
    for iteration_index in range(iterations):
        pair = get_ilc_pair(state_data, iteration_index)
        results.append(
            preprocess_full_pair(
                pair.input_full,
                pair.output_raw_full,
                partition,
                pair_type=pair.pair_type,
                iteration_index=iteration_index,
                input_peak_normalization_factor=pair.input_peak_normalization_factor,
            )
        )
    return tuple(results)


def build_stale_segments(
    state_data: dict[str, Any],
    partition: SegmentPartition,
) -> CanonicalSegments:
    """生成 ``xin_pd_nominal + yout_withdpd_stale_ori`` 的canonical STALE A/B/C。"""
    pair = get_stale_pair(state_data)
    return preprocess_full_pair(
        pair.input_full,
        pair.output_raw_full,
        partition,
        pair_type=pair.pair_type,
    )


__all__ = [
    "CanonicalSegment",
    "CanonicalSegments",
    "build_ilc_segments",
    "build_off_segments",
    "build_stale_segments",
    "preprocess_full_pair",
]
