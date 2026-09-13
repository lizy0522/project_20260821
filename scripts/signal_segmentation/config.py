"""
功能说明：冻结canonical A/B/C样点所有权比例、名称和Fine对齐分辨率。
输入：无。
输出：A/B/C比例、段名、时间轴、当前实验验收长度和Fine参数常量。
用途：所有波形共享由完整xin时间轴生成的唯一Partition，不包含任何MP模型参数。
"""

A_RATIO = 0.5
B_RATIO = 0.2
C_RATIO = 0.3
SEGMENT_NAMES = ("A", "B", "C")
TIME_AXIS = 0
FINE_ALIGN_SUBTIME = 256
EXPECTED_SIGNAL_LENGTH = 24576

if abs(A_RATIO + B_RATIO + C_RATIO - 1.0) > 1e-15:
    raise RuntimeError("A/B/C比例之和必须为1")

__all__ = [
    "A_RATIO",
    "B_RATIO",
    "C_RATIO",
    "EXPECTED_SIGNAL_LENGTH",
    "FINE_ALIGN_SUBTIME",
    "SEGMENT_NAMES",
    "TIME_AXIS",
]
