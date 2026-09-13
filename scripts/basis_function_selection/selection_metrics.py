"""Frozen support scoring for the Hard-20 selection experiment."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .config import TARGET_NMSE_DB
from .model_solver import StateModelMetrics


@dataclass(frozen=True)
class SupportScore:
    support: tuple[int, ...]
    n40: int
    worst_w_db: float
    worst_deficit_db: float
    q90_deficit_db: float
    mean_deficit_db: float
    median_w_db: float
    q99_condition: float
    all_full_rank: bool
    state_metrics: tuple[StateModelMetrics, ...]

    @property
    def k(self) -> int:
        return len(self.support)

    def sort_key(self) -> tuple[object, ...]:
        """Return the immutable lexicographic objective (lower is better)."""

        return (
            -self.n40,
            self.worst_deficit_db,
            self.q90_deficit_db,
            self.mean_deficit_db,
            self.median_w_db,
            self.k,
            self.support,
        )


@dataclass(frozen=True)
class FrozenCenteredSupportScore:
    """Task-specific score with W-pass then Train-pass priority."""

    support: tuple[int, ...]
    n_w40: int
    n_train40: int
    worst_w_db: float
    worst_deficit_db: float
    q90_deficit_db: float
    mean_deficit_db: float
    median_w_db: float
    q99_condition: float
    all_full_rank: bool
    state_metrics: tuple[StateModelMetrics, ...]

    @property
    def k(self) -> int:
        return len(self.support)

    def sort_key(self) -> tuple[object, ...]:
        return (
            -self.n_w40,
            -self.n_train40,
            self.worst_deficit_db,
            self.q90_deficit_db,
            self.mean_deficit_db,
            self.median_w_db,
            self.k,
            self.support,
        )


def aggregate_support(
    support: Sequence[int],
    metrics: Sequence[StateModelMetrics],
) -> SupportScore:
    """Aggregate 20 state metrics into the frozen score definition."""

    ordered = tuple(sorted(metrics, key=lambda item: item.state_id))
    if not ordered:
        raise ValueError("metrics cannot be empty")
    w_values = np.asarray([item.worst_nmse_db for item in ordered], dtype=np.float64)
    deficits = np.maximum(0.0, w_values - TARGET_NMSE_DB)
    conditions = np.asarray([item.max_condition_number for item in ordered], dtype=np.float64)
    return SupportScore(
        support=tuple(sorted(int(index) for index in support)),
        n40=int(np.count_nonzero(w_values < TARGET_NMSE_DB)),
        worst_w_db=float(np.max(w_values)),
        worst_deficit_db=float(np.max(deficits)),
        q90_deficit_db=float(np.quantile(deficits, 0.90)),
        mean_deficit_db=float(np.mean(deficits)),
        median_w_db=float(np.median(w_values)),
        q99_condition=float(np.quantile(conditions, 0.99)),
        all_full_rank=bool(all(item.min_rank_ratio >= 1.0 for item in ordered)),
        state_metrics=ordered,
    )


def backward_deletion_allowed(current: SupportScore, removed: SupportScore) -> bool:
    """Apply the fixed backward-pruning tolerance."""

    from .config import BACKWARD_TOLERANCE_DB

    return bool(
        removed.n40 >= current.n40
        and removed.worst_deficit_db <= current.worst_deficit_db + BACKWARD_TOLERANCE_DB
        and removed.mean_deficit_db <= current.mean_deficit_db + BACKWARD_TOLERANCE_DB
    )


def aggregate_frozen_centered_support(
    support: Sequence[int],
    metrics: Sequence[StateModelMetrics],
) -> FrozenCenteredSupportScore:
    """Aggregate state metrics using the new frozen-centered score."""

    ordered = tuple(sorted(metrics, key=lambda item: item.state_id))
    if not ordered:
        raise ValueError("metrics cannot be empty")
    train = np.asarray([item.full_train_nmse_db for item in ordered], dtype=np.float64)
    w_values = np.asarray([item.worst_nmse_db for item in ordered], dtype=np.float64)
    deficits = np.maximum(0.0, w_values - TARGET_NMSE_DB)
    conditions = np.asarray([item.max_condition_number for item in ordered], dtype=np.float64)
    return FrozenCenteredSupportScore(
        support=tuple(sorted(int(index) for index in support)),
        n_w40=int(np.count_nonzero(w_values < TARGET_NMSE_DB)),
        n_train40=int(np.count_nonzero(train < TARGET_NMSE_DB)),
        worst_w_db=float(np.max(w_values)),
        worst_deficit_db=float(np.max(deficits)),
        q90_deficit_db=float(np.quantile(deficits, 0.90)),
        mean_deficit_db=float(np.mean(deficits)),
        median_w_db=float(np.median(w_values)),
        q99_condition=float(np.quantile(conditions, 0.99)),
        all_full_rank=bool(all(item.min_rank_ratio >= 1.0 for item in ordered)),
        state_metrics=ordered,
    )


def frozen_centered_deletion_allowed(
    current: FrozenCenteredSupportScore,
    removed: FrozenCenteredSupportScore,
) -> bool:
    """Apply the frozen 0.02 dB backward tolerance."""

    from .config import BACKWARD_TOLERANCE_DB

    return bool(
        removed.n_w40 >= current.n_w40
        and removed.worst_deficit_db <= current.worst_deficit_db + BACKWARD_TOLERANCE_DB
        and removed.mean_deficit_db <= current.mean_deficit_db + BACKWARD_TOLERANCE_DB
    )


def dominates_frozen_centered(
    left: FrozenCenteredSupportScore,
    right: FrozenCenteredSupportScore,
) -> bool:
    """Return whether left dominates right on the frozen Pareto axes."""

    left_values = (-left.n_w40, left.worst_deficit_db, left.mean_deficit_db, left.k)
    right_values = (-right.n_w40, right.worst_deficit_db, right.mean_deficit_db, right.k)
    return all(a <= b for a, b in zip(left_values, right_values, strict=True)) and any(
        a < b for a, b in zip(left_values, right_values, strict=True)
    )


def dominates(left: SupportScore, right: SupportScore) -> bool:
    """Return whether ``left`` Pareto-dominates ``right``."""

    left_values = (-left.n40, left.worst_deficit_db, left.mean_deficit_db, left.k)
    right_values = (-right.n40, right.worst_deficit_db, right.mean_deficit_db, right.k)
    return all(a <= b for a, b in zip(left_values, right_values, strict=True)) and any(
        a < b for a, b in zip(left_values, right_values, strict=True)
    )


__all__ = [
    "SupportScore",
    "FrozenCenteredSupportScore",
    "aggregate_frozen_centered_support",
    "aggregate_support",
    "backward_deletion_allowed",
    "dominates",
    "dominates_frozen_centered",
    "frozen_centered_deletion_allowed",
]
