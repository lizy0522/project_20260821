"""Frozen10 reference models and Train-only evaluation for the Hard-20 gate."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from core.shared.metrics import nmse

from .config import DMAX, TARGET_NMSE_DB, TRAIN_LENGTH
from .data_preparation import prepare_train_only_state
from .model_solver import evaluate_state_support, fit_ols, fit_ridge
from .volterra_dictionary import build_basis_bank, build_dictionary


@dataclass(frozen=True)
class FrozenBasis:
    basis_id: str
    order: int
    memory_delay: int
    formula: str


@dataclass(frozen=True)
class ReferenceModel:
    model_id: str
    model_name: str
    solver: str
    ridge_lambda: float
    support: tuple[int, ...]
    purpose: str

    @property
    def k(self) -> int:
        return len(self.support)


FROZEN_BASES = (
    FrozenBasis("F01", 1, 0, "x[n]"),
    FrozenBasis("F02", 1, 1, "x[n-1]"),
    FrozenBasis("F03", 1, 2, "x[n-2]"),
    FrozenBasis("F04", 2, 0, "x[n]|x[n]|"),
    FrozenBasis("F05", 2, 1, "x[n-1]|x[n-1]|"),
    FrozenBasis("F06", 3, 0, "x[n]|x[n]|^2"),
    FrozenBasis("F07", 3, 1, "x[n-1]|x[n-1]|^2"),
    FrozenBasis("F08", 5, 0, "x[n]|x[n]|^4"),
    FrozenBasis("F09", 7, 0, "x[n]|x[n]|^6"),
    FrozenBasis("F10", 9, 0, "x[n]|x[n]|^8"),
)

REFERENCE_MODELS = (
    ReferenceModel(
        "M1",
        "Frozen135_OLS",
        "OLS",
        0.0,
        (0, 1, 2, 5, 6, 7),
        "Frozen 1/3/5-order subspace reference",
    ),
    ReferenceModel(
        "M2",
        "Frozen135_plus_P2_OLS",
        "OLS",
        0.0,
        (0, 1, 2, 3, 4, 5, 6, 7),
        "Isolate the two aligned P2 envelope terms",
    ),
    ReferenceModel(
        "M3",
        "Frozen135_plus_P79_OLS",
        "OLS",
        0.0,
        (0, 1, 2, 5, 6, 7, 8, 9),
        "Isolate the aligned P7/P9 terms",
    ),
    ReferenceModel(
        "M4",
        "Frozen10_OLS",
        "OLS",
        0.0,
        tuple(range(10)),
        "Pure Frozen10 function-space capacity",
    ),
    ReferenceModel(
        "M5",
        "Frozen10_Ridge1e-8",
        "raw-basis Ridge",
        1e-8,
        tuple(range(10)),
        "Formal Frozen10 raw-basis Ridge reference",
    ),
)


def validate_reference_definitions() -> dict[str, object]:
    """Validate exact Frozen10 bases, models, support sizes, and max delay."""

    if len(FROZEN_BASES) != 10 or len({item.basis_id for item in FROZEN_BASES}) != 10:
        raise RuntimeError("Frozen10 must contain exactly 10 unique bases")
    if max(item.memory_delay for item in FROZEN_BASES) != DMAX:
        raise RuntimeError("Frozen10 max delay must be 2")
    expected_sizes = {"M1": 6, "M2": 8, "M3": 8, "M4": 10, "M5": 10}
    if {model.model_id: model.k for model in REFERENCE_MODELS} != expected_sizes:
        raise RuntimeError("Reference model support sizes are incorrect")
    if any(len(set(model.support)) != model.k for model in REFERENCE_MODELS):
        raise RuntimeError("Reference support contains duplicates")
    if REFERENCE_MODELS[-1].ridge_lambda != 1e-8:
        raise RuntimeError("Formal Frozen10 Ridge lambda must be 1e-8")
    return {
        "frozen_basis_count": len(FROZEN_BASES),
        "model_count": len(REFERENCE_MODELS),
        "model_sizes": expected_sizes,
        "max_delay": DMAX,
        "pass": True,
    }


def build_frozen_bank(x: np.ndarray) -> np.ndarray:
    """Build the raw Frozen10 MP basis with support local to ``x``."""

    signal = np.asarray(x, dtype=np.complex128).reshape(-1)
    if signal.size <= DMAX or not np.all(np.isfinite(signal)):
        raise ValueError("x must be a finite complex vector longer than dmax")
    rows = signal.size - DMAX
    bank = np.empty((rows, len(FROZEN_BASES)), dtype=np.complex128)
    for index, basis in enumerate(FROZEN_BASES):
        delayed = signal[DMAX - basis.memory_delay : signal.size - basis.memory_delay]
        bank[:, index] = delayed * np.abs(delayed) ** (basis.order - 1)
    if not np.all(np.isfinite(bank)):
        raise RuntimeError("Frozen10 bank contains NaN/Inf")
    return bank


def frozen135_subset_gate() -> dict[str, object]:
    """Map all six Frozen135 columns exactly into the strict Full81 dictionary."""

    strict_terms = build_dictionary()
    signature_to_index = {term.signature: term.index for term in strict_terms}
    signatures = (
        (1, (0,), ()),
        (1, (1,), ()),
        (1, (2,), ()),
        (3, (0, 0), (0,)),
        (3, (1, 1), (1,)),
        (5, (0, 0, 0), (0, 0)),
    )
    mapped = tuple(signature_to_index[signature] for signature in signatures)
    rng = np.random.default_rng(20260912)
    x = rng.normal(size=512) + 1j * rng.normal(size=512)
    frozen_bank = build_frozen_bank(x)[:, REFERENCE_MODELS[0].support]
    full81_bank = build_basis_bank(x, strict_terms)[:, mapped]
    max_error = float(np.max(np.abs(frozen_bank - full81_bank)))
    if max_error > 1e-12:
        raise RuntimeError(f"Frozen135 subset mapping failed: {max_error:.3e}")
    return {
        "full81_indices": mapped,
        "max_abs_column_error": max_error,
        "pass": True,
    }


def _block_slices() -> tuple[slice, slice, slice]:
    indices = np.array_split(np.arange(TRAIN_LENGTH), 3)
    return tuple(slice(int(index[0]), int(index[-1]) + 1) for index in indices)  # type: ignore[return-value]


def _reference_metrics(
    state_id: int,
    bank: np.ndarray,
    target: np.ndarray,
    block_banks: Sequence[np.ndarray],
    block_targets: Sequence[np.ndarray],
    model: ReferenceModel,
) -> dict[str, object]:
    metrics = evaluate_state_support(
        state_id,
        bank,
        target,
        block_banks,
        block_targets,
        model.support,
        ridge_lambda=model.ridge_lambda,
    )
    minimum_rank = int(round(metrics.min_rank_ratio * model.k))
    return {
        "state_id": int(state_id),
        "model_id": model.model_id,
        "K": model.k,
        "lambda": model.ridge_lambda,
        "train_nmse_db": metrics.full_train_nmse_db,
        "cv1_nmse_db": metrics.cv_nmse_db[0],
        "cv2_nmse_db": metrics.cv_nmse_db[1],
        "cv3_nmse_db": metrics.cv_nmse_db[2],
        "W_db": metrics.worst_nmse_db,
        "pass_train40": bool(metrics.full_train_nmse_db < TARGET_NMSE_DB),
        "pass_W40": bool(metrics.worst_nmse_db < TARGET_NMSE_DB),
        "rank": minimum_rank,
        "condition_number": metrics.max_condition_number,
    }


def evaluate_reference_state(
    state_id: int,
    hard20_rank: int,
    saved_ranking_nmse_db: float,
    saved_full81: Mapping[str, object],
) -> dict[str, object]:
    """Reprocess and evaluate all five references plus Full81 for one state."""

    prepared = prepare_train_only_state(state_id)
    ranking_nmse = float(nmse(prepared.x_train, prepared.y_train_adjusted))
    ranking_delta = abs(ranking_nmse - float(saved_ranking_nmse_db))
    if ranking_delta > 1e-9:
        raise RuntimeError(f"state {state_id} Train definition changed by {ranking_delta:.3e} dB")

    frozen_bank = build_frozen_bank(prepared.x_train)
    full_target = prepared.y_train_adjusted[DMAX:]
    frozen_block_banks = []
    strict_block_banks = []
    block_targets = []
    strict_terms = build_dictionary()
    for block in _block_slices():
        x_block = prepared.x_train[block]
        y_block = prepared.y_train_adjusted[block]
        frozen_block_banks.append(build_frozen_bank(x_block))
        strict_block_banks.append(build_basis_bank(x_block, strict_terms))
        block_targets.append(y_block[DMAX:])

    rows = [
        _reference_metrics(
            state_id,
            frozen_bank,
            full_target,
            frozen_block_banks,
            block_targets,
            model,
        )
        for model in REFERENCE_MODELS
    ]

    strict_bank = build_basis_bank(prepared.x_train, strict_terms)
    full81_metrics = evaluate_state_support(
        state_id,
        strict_bank,
        full_target,
        strict_block_banks,
        block_targets,
        tuple(range(81)),
        ridge_lambda=0.0,
    )
    saved_values = (
        float(saved_full81["train_nmse_db"]),
        float(saved_full81["cv1_nmse_db"]),
        float(saved_full81["cv2_nmse_db"]),
        float(saved_full81["cv3_nmse_db"]),
        float(saved_full81["W_dB"]),
    )
    recomputed_values = (
        full81_metrics.full_train_nmse_db,
        full81_metrics.cv_nmse_db[0],
        full81_metrics.cv_nmse_db[1],
        full81_metrics.cv_nmse_db[2],
        full81_metrics.worst_nmse_db,
    )
    full81_max_abs_delta = float(
        max(
            abs(actual - saved)
            for actual, saved in zip(recomputed_values, saved_values, strict=True)
        )
    )
    if full81_max_abs_delta > 1e-9:
        raise RuntimeError(
            f"state {state_id} Full81 reproduction changed by {full81_max_abs_delta:.3e} dB"
        )

    m1_fit = fit_ols(frozen_bank[:, REFERENCE_MODELS[0].support], full_target, scale_columns=True)
    full81_fit = fit_ols(strict_bank, full_target, scale_columns=True)
    sse_m1 = float(np.sum(np.abs(full_target - m1_fit.prediction) ** 2))
    sse_full81 = float(np.sum(np.abs(full_target - full81_fit.prediction) ** 2))
    relative_excess = (sse_full81 - sse_m1) / max(sse_m1, np.finfo(np.float64).tiny)
    if relative_excess > 1e-10:
        raise RuntimeError(
            f"state {state_id} violates Full81 subset SSE gate: "
            f"relative excess={relative_excess:.3e}"
        )
    for row in rows:
        row["hard20_rank"] = int(hard20_rank)
        row["full81_train_nmse_db"] = saved_values[0]
        row["full81_W_db"] = saved_values[4]
        row["delta_train_vs_full81_db"] = float(row["train_nmse_db"]) - saved_values[0]
        row["delta_W_vs_full81_db"] = float(row["W_db"]) - saved_values[4]
    return {
        "state_id": int(state_id),
        "ranking_nmse_db": ranking_nmse,
        "ranking_abs_delta_db": ranking_delta,
        "full81_max_abs_delta_db": full81_max_abs_delta,
        "full81_subset_relative_sse_excess": relative_excess,
        "rough_delay": prepared.rough_delay,
        "fine_delay": prepared.fine_delay,
        "rows": rows,
    }


def formal_ridge_gate(state_id: int) -> dict[str, object]:
    """Check the Frozen10 raw-basis augmented Ridge path on one real state."""

    prepared = prepare_train_only_state(state_id)
    bank = build_frozen_bank(prepared.x_train)
    target = prepared.y_train_adjusted[DMAX:]
    actual = fit_ridge(bank, target, 1e-8)
    n_samples, k = bank.shape
    augmented = np.vstack([bank, np.sqrt(n_samples * 1e-8) * np.eye(k, dtype=np.complex128)])
    augmented_target = np.concatenate([target, np.zeros(k, dtype=np.complex128)])
    expected, *_ = np.linalg.lstsq(augmented, augmented_target, rcond=None)
    max_error = float(np.max(np.abs(actual.theta - expected)))
    if max_error > 1e-12:
        raise RuntimeError(f"Formal Ridge gate failed: {max_error:.3e}")
    return {"state_id": state_id, "theta_max_abs_error": max_error, "pass": True}


__all__ = [
    "FROZEN_BASES",
    "REFERENCE_MODELS",
    "FrozenBasis",
    "ReferenceModel",
    "build_frozen_bank",
    "evaluate_reference_state",
    "formal_ridge_gate",
    "frozen135_subset_gate",
    "validate_reference_definitions",
]
