"""
功能说明：提供前向模型NMSE和已处理输出波形CNMSE评价接口。
输入：等长的一维参考波形和预测/测试波形。
输出：以dB表示的NMSE或CNMSE。
用途：复用_core.metrics的冻结公式，不在评价函数内部执行对齐、复增益或归一化。
"""

from __future__ import annotations

import numpy as np
from _core.metrics import cnmse, nmse


def calculate_nmse(y_ref: np.ndarray, y_pred: np.ndarray) -> float:
    """计算模型预测NMSE，不执行任何额外预处理。"""
    return nmse(y_ref, y_pred)


def calculate_cnmse(y_ref: np.ndarray, y_test: np.ndarray) -> float:
    """计算两个已时间/复增益对齐输出的CNMSE。"""
    return cnmse(y_ref, y_test)


__all__ = ["calculate_cnmse", "calculate_nmse"]
