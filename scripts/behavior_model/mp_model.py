"""
功能说明：封装冻结结构Memory Polynomial模型的拟合、预测和数值诊断状态。
输入：构造时可传配置；fit输入x、y，predict输入x。
输出：保存于self.theta的10系数模型、预测波形及fit_diagnostics。
用途：为X模型和每次Y迭代提供相同的面向对象接口。
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

import numpy as np

from .coefficient import LeastSquaresDiagnostics, _solve_behavior_coefficient
from .config import MP_CONFIG, validate_mp_config
from .prediction import predict_behavior


class MemoryPolynomialModel:
    """包含偶数二阶项的阶数相关记忆深度复基带MP模型。"""

    def __init__(self, config: Mapping[str, Any] = MP_CONFIG) -> None:
        validate_mp_config(config)
        self.config = deepcopy(dict(config))
        self.theta: np.ndarray | None = None
        self.fit_diagnostics: LeastSquaresDiagnostics | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> MemoryPolynomialModel:
        """使用numpy.linalg.lstsq拟合并保存系数与数值诊断。"""
        self.theta, self.fit_diagnostics = _solve_behavior_coefficient(x, y, self.config)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        """预测有效区间输出；尚未fit时显式报错。"""
        if self.theta is None:
            raise RuntimeError("模型尚未fit")
        return predict_behavior(x, self.theta, self.config)


__all__ = ["MemoryPolynomialModel"]
