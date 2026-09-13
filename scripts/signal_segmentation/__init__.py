"""
功能说明：统一导出canonical A/B/C Partition、切分、原始数据对、预处理和验证接口。
输入：完整PA状态数据字典、完整波形或由xin生成的唯一Partition。
输出：OFF、ILC、STALE三类结构化canonical A/B/C输入输出对。
用途：作为data_manager与后续行为建模之间的唯一canonical信号入口。
"""

from .behavior_pairs import (
    RawBehaviorPair,
    get_common_probe,
    get_ilc_pair,
    get_off_pair,
    get_stale_pair,
)
from .config import (
    A_RATIO,
    B_RATIO,
    C_RATIO,
    EXPECTED_SIGNAL_LENGTH,
    FINE_ALIGN_SUBTIME,
    SEGMENT_NAMES,
    TIME_AXIS,
)
from .partition import SegmentPartition, build_partition, build_partition_from_xin
from .preprocessing import (
    CanonicalSegment,
    CanonicalSegments,
    build_ilc_segments,
    build_off_segments,
    build_stale_segments,
    preprocess_full_pair,
)
from .splitter import split_signal, split_signals
from .validation import (
    validate_canonical_segments,
    validate_current_experiment_partition,
    validate_full_pair,
    validate_ilc_histories,
    validate_partition,
    validate_peak_normalized,
)

__all__ = [
    "A_RATIO",
    "B_RATIO",
    "C_RATIO",
    "EXPECTED_SIGNAL_LENGTH",
    "FINE_ALIGN_SUBTIME",
    "SEGMENT_NAMES",
    "TIME_AXIS",
    "CanonicalSegment",
    "CanonicalSegments",
    "RawBehaviorPair",
    "SegmentPartition",
    "build_ilc_segments",
    "build_off_segments",
    "build_partition",
    "build_partition_from_xin",
    "build_stale_segments",
    "get_common_probe",
    "get_ilc_pair",
    "get_off_pair",
    "get_stale_pair",
    "preprocess_full_pair",
    "split_signal",
    "split_signals",
    "validate_canonical_segments",
    "validate_current_experiment_partition",
    "validate_full_pair",
    "validate_ilc_histories",
    "validate_partition",
    "validate_peak_normalized",
]
