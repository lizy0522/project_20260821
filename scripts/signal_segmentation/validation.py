"""
功能说明：集中验证Partition、完整数据对、ILC历史、峰值归一化和canonical段约束。
输入：Partition、NumPy数组或canonical段对象。
输出：验证通过时无返回；违反科研约束时抛出明确异常。
用途：防止重叠/遗漏、独立边界、隐式裁剪、零峰值和ILC列语义错误。
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import EXPECTED_SIGNAL_LENGTH, SEGMENT_NAMES
from .partition import SegmentPartition


def validate_partition(partition: SegmentPartition) -> None:
    """验证A/B/C连续、无重叠、完整覆盖且长度总和等于N。"""
    if not isinstance(partition, SegmentPartition):
        raise TypeError("partition必须是SegmentPartition")
    if partition.a_slice.start != 0:
        raise ValueError("A必须从0开始")
    if partition.a_slice.stop != partition.b_slice.start:
        raise ValueError("A.stop必须等于B.start")
    if partition.b_slice.stop != partition.c_slice.start:
        raise ValueError("B.stop必须等于C.start")
    if partition.c_slice.stop != partition.total_length:
        raise ValueError("C.stop必须等于总长度")
    if partition.n_a + partition.n_b + partition.n_c != partition.total_length:
        raise ValueError("A/B/C长度总和与N不一致")
    if min(partition.n_a, partition.n_b, partition.n_c) <= 0:
        raise ValueError("A/B/C均必须非空")


def validate_current_experiment_partition(partition: SegmentPartition) -> None:
    """验收24576点当前实验唯一边界12288/4915/7373。"""
    validate_partition(partition)
    expected = (EXPECTED_SIGNAL_LENGTH, 12288, 4915, 7373)
    actual = (partition.total_length, partition.n_a, partition.n_b, partition.n_c)
    if actual != expected:
        raise ValueError(f"当前实验Partition错误：expected={expected}, actual={actual}")


def validate_full_pair(
    input_full: np.ndarray,
    output_raw_full: np.ndarray,
    partition: SegmentPartition,
) -> None:
    """验证完整输入/raw输出均是一维等长有限复波形并匹配唯一Partition。"""
    validate_partition(partition)
    for name, value in (("input_full", input_full), ("output_raw_full", output_raw_full)):
        if not isinstance(value, np.ndarray):
            raise TypeError(f"{name}必须是numpy.ndarray")
        if value.ndim != 1 or not np.iscomplexobj(value):
            raise ValueError(f"{name}必须是一维复数波形，实际shape={value.shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name}包含NaN或Inf")
    if input_full.shape[0] != output_raw_full.shape[0]:
        raise ValueError("完整输入与raw输出长度不一致，禁止自动裁剪或补零")
    if input_full.shape[0] != partition.total_length:
        raise ValueError("完整数据对长度与xin定义的Partition不一致")


def validate_ilc_histories(
    input_history: np.ndarray,
    output_history: np.ndarray,
    nth_inter: int | None = None,
) -> int:
    """验证ILC输入输出均为同shape二维有限复矩阵，并返回列数。"""
    for name, value in (
        ("xin_pd_ori_ilc", input_history),
        ("yout_withdpd_ori_ilc", output_history),
    ):
        if not isinstance(value, np.ndarray) or value.ndim != 2:
            raise ValueError(f"{name}必须是二维numpy.ndarray")
        if not np.iscomplexobj(value) or not np.all(np.isfinite(value)):
            raise ValueError(f"{name}必须是有限复数矩阵")
    if input_history.shape != output_history.shape:
        raise ValueError("ILC输入与输出历史shape不一致")
    iterations = input_history.shape[1]
    if iterations <= 0:
        raise ValueError("ILC历史必须至少包含一列")
    if nth_inter is not None and iterations != int(nth_inter):
        raise ValueError(f"ILC列数{iterations}与nth_inter={nth_inter}不一致")
    return iterations


def validate_peak_normalized(signal: np.ndarray, *, atol: float = 1e-12) -> None:
    """验证一维复波形峰值为1。"""
    if not isinstance(signal, np.ndarray) or signal.ndim != 1:
        raise ValueError("归一化信号必须是一维numpy.ndarray")
    peak = float(np.max(np.abs(signal)))
    if not np.isclose(peak, 1.0, rtol=0.0, atol=atol):
        raise ValueError(f"峰值归一化失败：peak={peak}")


def validate_canonical_segments(canonical: Any) -> None:
    """以公开属性验证canonical A/B/C尺寸、有限性和元数据。"""
    partition = canonical.partition
    validate_partition(partition)
    expected_lengths = {"A": partition.n_a, "B": partition.n_b, "C": partition.n_c}
    for segment_name in SEGMENT_NAMES:
        segment = canonical[segment_name]
        expected_length = expected_lengths[segment_name]
        if segment.input.shape != (expected_length,):
            raise ValueError(f"{segment_name}输入shape错误：{segment.input.shape}")
        if segment.output.shape != (expected_length,):
            raise ValueError(f"{segment_name}输出shape错误：{segment.output.shape}")
        if segment.ownership_length != expected_length:
            raise ValueError(f"{segment_name}ownership_length错误")
        if not np.all(np.isfinite(segment.input)) or not np.all(np.isfinite(segment.output)):
            raise ValueError(f"{segment_name}包含NaN或Inf")


__all__ = [
    "validate_canonical_segments",
    "validate_current_experiment_partition",
    "validate_full_pair",
    "validate_ilc_histories",
    "validate_partition",
    "validate_peak_normalized",
]
