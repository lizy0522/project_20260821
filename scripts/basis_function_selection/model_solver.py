"""Complex OLS/raw-basis Ridge solvers and support-level evaluation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from _core.metrics import nmse


@dataclass(frozen=True)
class FitResult:
    theta: np.ndarray
    rank: int
    condition_number: float
    prediction: np.ndarray
    nmse_db: float


@dataclass(frozen=True)
class StateModelMetrics:
    state_id: int
    full_train_nmse_db: float
    cv_nmse_db: tuple[float, float, float]
    worst_nmse_db: float
    max_condition_number: float
    min_rank_ratio: float


def _validate(phi: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    matrix = np.asarray(phi, dtype=np.complex128)
    target = np.asarray(y, dtype=np.complex128).reshape(-1)
    if matrix.ndim != 2 or matrix.shape[0] != target.size or matrix.shape[0] <= matrix.shape[1]:
        raise ValueError("Invalid design matrix/target dimensions")
    if matrix.shape[1] == 0 or not np.all(np.isfinite(matrix)) or not np.all(np.isfinite(target)):
        raise ValueError("Design matrix/target must be finite and nonempty")
    return matrix, target


def fit_ols(phi: np.ndarray, y: np.ndarray, *, scale_columns: bool = True) -> FitResult:
    """Fit OLS through ``lstsq``; optional RMS scaling is reparameterization only."""

    matrix, target = _validate(phi, y)
    if scale_columns:
        scale = np.sqrt(np.mean(np.abs(matrix) ** 2, axis=0))
        if np.any(scale <= 0) or not np.all(np.isfinite(scale)):
            raise RuntimeError("Invalid OLS RMS column scale")
        solve_matrix = matrix / scale[None, :]
        beta, _, rank, singular = np.linalg.lstsq(solve_matrix, target, rcond=None)
        theta = beta / scale
    else:
        theta, _, rank, singular = np.linalg.lstsq(matrix, target, rcond=None)
    prediction = matrix @ theta
    condition = (
        float(singular[0] / singular[-1]) if singular.size and singular[-1] > 0 else float("inf")
    )
    return FitResult(
        theta=np.asarray(theta, dtype=np.complex128),
        rank=int(rank),
        condition_number=condition,
        prediction=np.asarray(prediction, dtype=np.complex128),
        nmse_db=float(nmse(target, prediction)),
    )


def fit_ridge(phi: np.ndarray, y: np.ndarray, ridge_lambda: float) -> FitResult:
    """Fit raw-basis ``||e||²/N + lambda||theta||²`` by augmented LS."""

    matrix, target = _validate(phi, y)
    ridge_lambda = float(ridge_lambda)
    if not np.isfinite(ridge_lambda) or ridge_lambda < 0:
        raise ValueError("ridge_lambda must be finite and nonnegative")
    if ridge_lambda == 0.0:
        return fit_ols(matrix, target, scale_columns=True)
    n_samples, coefficient_count = matrix.shape
    augmented = np.vstack(
        [matrix, np.sqrt(n_samples * ridge_lambda) * np.eye(coefficient_count, dtype=np.complex128)]
    )
    augmented_target = np.concatenate([target, np.zeros(coefficient_count, dtype=np.complex128)])
    theta, _, rank_augmented, singular_augmented = np.linalg.lstsq(
        augmented, augmented_target, rcond=None
    )
    raw_singular = np.linalg.svd(matrix, compute_uv=False)
    raw_rank = int(np.linalg.matrix_rank(matrix))
    condition = float(raw_singular[0] / raw_singular[-1]) if raw_singular[-1] > 0 else float("inf")
    prediction = matrix @ theta
    if int(rank_augmented) != coefficient_count or singular_augmented.size != coefficient_count:
        raise RuntimeError("Augmented Ridge system is not full column rank")
    return FitResult(
        theta=np.asarray(theta, dtype=np.complex128),
        rank=raw_rank,
        condition_number=condition,
        prediction=np.asarray(prediction, dtype=np.complex128),
        nmse_db=float(nmse(target, prediction)),
    )


def evaluate_state_support(
    state_id: int,
    full_bank: np.ndarray,
    full_target: np.ndarray,
    block_banks: Sequence[np.ndarray],
    block_targets: Sequence[np.ndarray],
    support: Sequence[int],
    *,
    ridge_lambda: float = 0.0,
) -> StateModelMetrics:
    """Evaluate full Train and three strict blocked folds for one state/support."""

    indices = np.asarray(tuple(int(index) for index in support), dtype=np.int64)
    if indices.size == 0 or np.unique(indices).size != indices.size:
        raise ValueError("support must contain unique basis indices")
    solver = fit_ols if ridge_lambda == 0.0 else fit_ridge

    full_phi = np.asarray(full_bank[:, indices], dtype=np.complex128)
    if ridge_lambda == 0.0:
        full_fit = solver(full_phi, full_target, scale_columns=True)
    else:
        full_fit = solver(full_phi, full_target, ridge_lambda)
    nmse_values: list[float] = []
    conditions = [full_fit.condition_number]
    rank_ratios = [full_fit.rank / indices.size]
    for validation_index in range(3):
        train_matrices = [
            block_banks[index][:, indices] for index in range(3) if index != validation_index
        ]
        train_targets = [block_targets[index] for index in range(3) if index != validation_index]
        train_phi = np.concatenate(train_matrices, axis=0)
        train_target = np.concatenate(train_targets)
        validation_phi = np.asarray(block_banks[validation_index][:, indices], dtype=np.complex128)
        validation_target = np.asarray(block_targets[validation_index], dtype=np.complex128)
        if ridge_lambda == 0.0:
            fit = solver(train_phi, train_target, scale_columns=True)
        else:
            fit = solver(train_phi, train_target, ridge_lambda)
        validation_prediction = validation_phi @ fit.theta
        nmse_values.append(float(nmse(validation_target, validation_prediction)))
        conditions.append(fit.condition_number)
        rank_ratios.append(fit.rank / indices.size)
    worst = float(max([full_fit.nmse_db, *nmse_values]))
    return StateModelMetrics(
        state_id=int(state_id),
        full_train_nmse_db=full_fit.nmse_db,
        cv_nmse_db=(nmse_values[0], nmse_values[1], nmse_values[2]),
        worst_nmse_db=worst,
        max_condition_number=float(max(conditions)),
        min_rank_ratio=float(min(rank_ratios)),
    )


def ols_scaling_equivalence_gate(phi: np.ndarray, y: np.ndarray) -> dict[str, float | bool]:
    """Verify raw and RMS-reparameterized OLS predictions agree."""

    raw = fit_ols(phi, y, scale_columns=False)
    scaled = fit_ols(phi, y, scale_columns=True)
    difference = abs(raw.nmse_db - scaled.nmse_db)
    prediction_relative_l2 = float(
        np.linalg.norm(raw.prediction - scaled.prediction) / np.linalg.norm(raw.prediction)
    )
    passed = bool(difference < 1e-9)
    if not passed:
        raise RuntimeError(f"OLS scaling equivalence failed: {difference:.3e} dB")
    return {
        "nmse_abs_difference_db": float(difference),
        "prediction_relative_l2": prediction_relative_l2,
        "pass": passed,
    }


__all__ = [
    "FitResult",
    "StateModelMetrics",
    "evaluate_state_support",
    "fit_ols",
    "fit_ridge",
    "ols_scaling_equivalence_gate",
]
