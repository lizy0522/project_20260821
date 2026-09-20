"""
功能说明：实现仅改变lambda的复数Ridge系数求解，并保留lambda=0的原OLS数值路径。
输入：复数设计矩阵Phi、有效输出y和有限非负ridge_lambda。
输出：10个复数系数及原始/增广矩阵数值诊断。
用途：生产代码使用增广lstsq，不显式求逆、不使用正规方程、不缩放特征或改变MP结构。
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real

import numpy as np

from .coefficient import _solve_basis_ols


@dataclass(frozen=True)
class RidgeFitDiagnostics:
    """一次OLS/Ridge拟合的原始Phi和增广系统诊断。"""

    ridge_lambda: float
    n_train_samples: int
    coefficient_count: int
    rank_phi: int
    rank_augmented: int
    condition_number_phi: float
    condition_number_augmented: float
    theta_l2_norm: float
    residual_l2: float
    solver_path: str


def validate_ridge_lambda(ridge_lambda: Real) -> float:
    """返回规范化lambda；负数、NaN和Inf均拒绝。"""
    if not isinstance(ridge_lambda, Real) or isinstance(ridge_lambda, bool):
        raise TypeError("ridge_lambda必须是实数")
    normalized = float(ridge_lambda)
    if not np.isfinite(normalized):
        raise ValueError("ridge_lambda必须是有限值")
    if normalized < 0:
        raise ValueError("ridge_lambda必须大于或等于0")
    return normalized


def _validate_system(phi: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(phi, np.ndarray) or phi.ndim != 2:
        raise ValueError("phi必须是二维numpy.ndarray")
    if not isinstance(y, np.ndarray) or y.ndim != 1:
        raise ValueError("y必须是一维numpy.ndarray")
    if phi.shape[0] != y.shape[0]:
        raise ValueError("phi行数必须等于y长度")
    if phi.shape[1] <= 0 or phi.shape[0] <= phi.shape[1]:
        raise ValueError("phi必须是样本数大于系数数的非空设计矩阵")
    if not np.iscomplexobj(phi) or not np.iscomplexobj(y):
        raise ValueError("phi和y必须是复数数组")
    if not np.all(np.isfinite(phi)) or not np.all(np.isfinite(y)):
        raise ValueError("phi或y包含NaN/Inf")
    return phi, y


def fit_coefficients_ridge(
    phi: np.ndarray,
    y: np.ndarray,
    ridge_lambda: Real = 0.0,
    *,
    ols_solution: tuple[np.ndarray, int, np.ndarray] | None = None,
) -> tuple[np.ndarray, RidgeFitDiagnostics]:
    """求解 ``||e||²/N + lambda||theta||²``，lambda=0直接走冻结OLS函数。

    ``ols_solution`` 是扫描阶段可选的已缓存 OLS 解 ``(theta, rank, singular_values)``。
    它只避免重复计算原始设计矩阵的 OLS 诊断；非零 lambda 仍然严格通过增广
    复数 ``lstsq`` 求解，默认调用方式和既有 State0 结果保持不变。
    """
    phi_checked, y_checked = _validate_system(phi, y)
    normalized_lambda = validate_ridge_lambda(ridge_lambda)
    n_samples, n_coefficients = phi_checked.shape

    if ols_solution is None:
        ols_theta, _, rank_phi, singular_phi = _solve_basis_ols(phi_checked, y_checked)
    else:
        ols_theta, rank_phi, singular_phi = ols_solution
        if (
            not isinstance(ols_theta, np.ndarray)
            or ols_theta.shape != (n_coefficients,)
            or not isinstance(singular_phi, np.ndarray)
            or singular_phi.ndim != 1
            or singular_phi.size != n_coefficients
        ):
            raise ValueError("ols_solution形状必须与phi系数列数一致")
        if not np.all(np.isfinite(ols_theta)) or not np.all(np.isfinite(singular_phi)):
            raise ValueError("ols_solution不能包含NaN或Inf")
    condition_phi = float(singular_phi[0] / singular_phi[-1])
    if normalized_lambda == 0.0:
        theta = ols_theta
        rank_augmented = int(rank_phi)
        condition_augmented = condition_phi
        solver_path = "existing_ols"
    else:
        penalty = np.sqrt(n_samples * normalized_lambda)
        phi_augmented = np.vstack(
            [
                phi_checked,
                penalty * np.eye(n_coefficients, dtype=np.complex128),
            ]
        )
        y_augmented = np.concatenate([y_checked, np.zeros(n_coefficients, dtype=np.complex128)])
        theta, _, rank_augmented, singular_augmented = np.linalg.lstsq(
            phi_augmented,
            y_augmented,
            rcond=None,
        )
        condition_augmented = float(singular_augmented[0] / singular_augmented[-1])
        solver_path = "augmented_lstsq"

    if theta.shape != (n_coefficients,) or not np.all(np.isfinite(theta)):
        raise RuntimeError("Ridge求解未得到有限的一维系数向量")
    diagnostics = RidgeFitDiagnostics(
        ridge_lambda=normalized_lambda,
        n_train_samples=n_samples,
        coefficient_count=n_coefficients,
        rank_phi=int(rank_phi),
        rank_augmented=int(rank_augmented),
        condition_number_phi=condition_phi,
        condition_number_augmented=condition_augmented,
        theta_l2_norm=float(np.linalg.norm(theta)),
        residual_l2=float(np.linalg.norm(phi_checked @ theta - y_checked)),
        solver_path=solver_path,
    )
    return theta, diagnostics


__all__ = ["RidgeFitDiagnostics", "fit_coefficients_ridge", "validate_ridge_lambda"]
