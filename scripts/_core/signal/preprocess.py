"""
功能说明：封装输入输出信号对的固定预处理流水线。
输入：参考信号 x、待处理输出 y，以及可选的分数延迟搜索参数。
输出：依次完成整数对齐、分数对齐和复增益调整后的 y_processed。
数学定义：y_processed=adjust(fine_align(rough_align(y)))，所有步骤均以 x 为参考。
"""

from __future__ import annotations

import numpy as np

from .alignment import fine_align, rough_align
from .calibration import adjust_complex_gain


def preprocess_pair(
    x: np.ndarray,
    y: np.ndarray,
    subtime: int = 256,
    ns: int | None = None,
) -> np.ndarray:
    """按固定顺序处理 y，并且不修改 x、y 输入数组。"""
    y_rough, _ = rough_align(x, y)
    y_fine, _ = fine_align(x, y_rough, subtime=subtime, ns=ns)
    y_processed, _ = adjust_complex_gain(x, y_fine)
    return y_processed


__all__ = ["preprocess_pair"]
