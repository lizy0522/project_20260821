"""
功能说明：统一导出state0正向Memory Polynomial行为建模公开接口。
输入：具体输入由基函数、系数、模型、预测和评价函数定义。
输出：冻结配置、10基函数构造、LS系数、模型类、预测和评价函数。
用途：允许任务入口和后续验证从behavior_model单一包入口调用。
"""

from .basis import build_mp_basis
from .coefficient import LeastSquaresDiagnostics, extract_behavior_coefficient
from .config import BASIS_TERMS, MAX_DELAY, MP_CONFIG, NUM_COEFFICIENTS
from .evaluation import calculate_cnmse, calculate_nmse
from .mp_model import MemoryPolynomialModel
from .prediction import predict_behavior
from .ridge import RidgeFitDiagnostics, fit_coefficients_ridge, validate_ridge_lambda
from .ridge_scan import RIDGE_LAMBDAS, RidgeScanResult, run_state0_ridge_scan
from .scenario2_xy_analysis import (
    MODEL_IDS,
    Scenario2XYAnalysisResult,
    ScenarioModelResult,
    StateXYAnalysisResult,
    analyze_state_xy,
    collect_scenario2_xy_analysis,
    validate_state0_regression,
)
from .xy_equivalence import (
    ModelEvaluation,
    State0XYEquivalenceResult,
    analyze_state0_xy_equivalence,
)

__all__ = [
    "BASIS_TERMS",
    "MAX_DELAY",
    "MODEL_IDS",
    "MP_CONFIG",
    "NUM_COEFFICIENTS",
    "LeastSquaresDiagnostics",
    "MemoryPolynomialModel",
    "ModelEvaluation",
    "State0XYEquivalenceResult",
    "RIDGE_LAMBDAS",
    "RidgeFitDiagnostics",
    "RidgeScanResult",
    "Scenario2XYAnalysisResult",
    "ScenarioModelResult",
    "StateXYAnalysisResult",
    "analyze_state0_xy_equivalence",
    "analyze_state_xy",
    "build_mp_basis",
    "calculate_cnmse",
    "calculate_nmse",
    "collect_scenario2_xy_analysis",
    "extract_behavior_coefficient",
    "fit_coefficients_ridge",
    "predict_behavior",
    "run_state0_ridge_scan",
    "validate_ridge_lambda",
    "validate_state0_regression",
]
