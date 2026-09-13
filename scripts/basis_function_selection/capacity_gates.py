"""Train-only evaluation and aggregation for nested capacity gates."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .config import DMAX, OLS_CONDITION_HARD_LIMIT, TARGET_NMSE_DB, TRAIN_LENGTH
from .data_preparation import prepare_train_only_state
from .frozen_centered_dictionary import (
    FrozenCenteredBasis,
    build_frozen_centered_bank,
    gate_indices,
)
from .model_solver import evaluate_state_support, fit_ols
from .selection_metrics import (
    FrozenCenteredSupportScore,
    aggregate_frozen_centered_support,
)

_CENTERED_TERMS: tuple[FrozenCenteredBasis, ...] | None = None
_CENTERED_GATE_ID: str | None = None
_CENTERED_CACHE: dict[int, tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]] = {}


def _block_slices() -> tuple[slice, slice, slice]:
    blocks = np.array_split(np.arange(TRAIN_LENGTH), 3)
    return tuple(slice(int(block[0]), int(block[-1]) + 1) for block in blocks)  # type: ignore[return-value]


def evaluate_capacity_state(
    state_id: int,
    hard20_rank: int,
    gate_id: str,
    terms: Sequence[FrozenCenteredBasis],
    frozen_reference: Mapping[str, object],
) -> dict[str, object]:
    """Evaluate one full candidate gate on one state without Test access."""

    prepared = prepare_train_only_state(state_id)
    indices = gate_indices(terms, gate_id)
    bank = build_frozen_centered_bank(prepared.x_train, terms, indices)
    target = prepared.y_train_adjusted[DMAX:]
    block_banks = []
    block_targets = []
    for block in _block_slices():
        x_block = prepared.x_train[block]
        y_block = prepared.y_train_adjusted[block]
        block_banks.append(build_frozen_centered_bank(x_block, terms, indices))
        block_targets.append(y_block[DMAX:])
    metrics = evaluate_state_support(
        state_id,
        bank,
        target,
        block_banks,
        block_targets,
        tuple(range(len(indices))),
        ridge_lambda=0.0,
    )
    full_fit = fit_ols(bank, target, scale_columns=True)
    frozen_train = float(frozen_reference["train_nmse_db"])
    frozen_w = float(frozen_reference["W_db"])
    return {
        "state_id": int(state_id),
        "hard20_rank": int(hard20_rank),
        "gate_id": gate_id,
        "K": len(indices),
        "train_nmse_db": metrics.full_train_nmse_db,
        "cv1_nmse_db": metrics.cv_nmse_db[0],
        "cv2_nmse_db": metrics.cv_nmse_db[1],
        "cv3_nmse_db": metrics.cv_nmse_db[2],
        "W_db": metrics.worst_nmse_db,
        "pass_train40": bool(metrics.full_train_nmse_db < TARGET_NMSE_DB),
        "pass_W40": bool(metrics.worst_nmse_db < TARGET_NMSE_DB),
        "rank": int(round(metrics.min_rank_ratio * len(indices))),
        "condition_number": metrics.max_condition_number,
        "delta_train_vs_frozen10_db": metrics.full_train_nmse_db - frozen_train,
        "delta_W_vs_frozen10_db": metrics.worst_nmse_db - frozen_w,
        "train_sse": float(np.sum(np.abs(target - full_fit.prediction) ** 2)),
    }


def summarize_capacity_gate(
    rows: Sequence[Mapping[str, object]],
    previous_rows: Sequence[Mapping[str, object]] | None,
) -> dict[str, object]:
    """Aggregate one gate and classify capacity success/failure/inconclusive."""

    if len(rows) != 20:
        raise RuntimeError(f"Capacity gate requires 20 state rows, got {len(rows)}")
    gate_ids = {str(row["gate_id"]) for row in rows}
    k_values = {int(row["K"]) for row in rows}
    if len(gate_ids) != 1 or len(k_values) != 1:
        raise RuntimeError("Capacity rows mix gate IDs or K values")
    train = np.asarray([float(row["train_nmse_db"]) for row in rows])
    w_values = np.asarray([float(row["W_db"]) for row in rows])
    conditions = np.asarray([float(row["condition_number"]) for row in rows])
    all_full_rank = all(int(row["rank"]) == int(row["K"]) for row in rows)
    reliable = bool(
        all_full_rank
        and np.all(np.isfinite(conditions))
        and np.quantile(conditions, 0.99) <= OLS_CONDITION_HARD_LIMIT
    )
    n_train40 = int(np.count_nonzero(train < TARGET_NMSE_DB))
    n_w40 = int(np.count_nonzero(w_values < TARGET_NMSE_DB))
    capacity_success = n_train40 == 20
    capacity_failure = bool(reliable and not capacity_success)
    capacity_inconclusive = bool(not reliable)
    median_previous_train = np.nan
    median_previous_w = np.nan
    if previous_rows is not None:
        previous_by_state = {int(row["state_id"]): row for row in previous_rows}
        train_deltas = []
        w_deltas = []
        for row in rows:
            previous = previous_by_state[int(row["state_id"])]
            current_sse = float(row["train_sse"])
            previous_sse = float(previous["train_sse"])
            relative_excess = (current_sse - previous_sse) / max(
                previous_sse,
                np.finfo(np.float64).tiny,
            )
            if relative_excess > 1e-10:
                raise RuntimeError(
                    f"Nested SSE monotonicity failed at state {row['state_id']}: "
                    f"relative excess={relative_excess:.3e}"
                )
            train_deltas.append(float(row["train_nmse_db"]) - float(previous["train_nmse_db"]))
            w_deltas.append(float(row["W_db"]) - float(previous["W_db"]))
        median_previous_train = float(np.median(train_deltas))
        median_previous_w = float(np.median(w_deltas))
    return {
        "gate_id": next(iter(gate_ids)),
        "K": next(iter(k_values)),
        "all_full_rank": all_full_rank,
        "condition_q90": float(np.quantile(conditions, 0.90)),
        "condition_q99": float(np.quantile(conditions, 0.99)),
        "condition_max": float(np.max(conditions)),
        "numerically_reliable": reliable,
        "train_mean": float(np.mean(train)),
        "train_median": float(np.median(train)),
        "train_best": float(np.min(train)),
        "train_worst": float(np.max(train)),
        "W_mean": float(np.mean(w_values)),
        "W_median": float(np.median(w_values)),
        "W_best": float(np.min(w_values)),
        "W_worst": float(np.max(w_values)),
        "N_train40": n_train40,
        "N_W40": n_w40,
        "median_delta_train_vs_previous_gate": median_previous_train,
        "median_delta_W_vs_previous_gate": median_previous_w,
        "capacity_success": capacity_success,
        "capacity_failure": capacity_failure,
        "capacity_inconclusive": capacity_inconclusive,
    }


def _centered_worker_init(
    terms: tuple[FrozenCenteredBasis, ...],
    gate_id: str,
) -> None:
    global _CENTERED_TERMS, _CENTERED_GATE_ID, _CENTERED_CACHE
    _CENTERED_TERMS = terms
    _CENTERED_GATE_ID = gate_id
    _CENTERED_CACHE = {}


def _load_worker_state(
    state_id: int,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    if _CENTERED_TERMS is None or _CENTERED_GATE_ID is None:
        raise RuntimeError("Frozen-centered worker was not initialized")
    if state_id in _CENTERED_CACHE:
        return _CENTERED_CACHE[state_id]
    prepared = prepare_train_only_state(state_id)
    dictionary_indices = gate_indices(_CENTERED_TERMS, _CENTERED_GATE_ID)
    full_bank = build_frozen_centered_bank(
        prepared.x_train,
        _CENTERED_TERMS,
        dictionary_indices,
    )
    full_target = prepared.y_train_adjusted[DMAX:]
    block_banks = []
    block_targets = []
    for block in _block_slices():
        x_block = prepared.x_train[block]
        y_block = prepared.y_train_adjusted[block]
        block_banks.append(build_frozen_centered_bank(x_block, _CENTERED_TERMS, dictionary_indices))
        block_targets.append(y_block[DMAX:])
    if len(_CENTERED_CACHE) >= 2:
        _CENTERED_CACHE.pop(next(iter(_CENTERED_CACHE)))
    _CENTERED_CACHE[state_id] = (full_bank, full_target, block_banks, block_targets)
    return _CENTERED_CACHE[state_id]


def _centered_state_batch(
    state_id: int,
    supports: tuple[tuple[int, ...], ...],
    ridge_lambda: float,
) -> tuple[int, list[object]]:
    if _CENTERED_TERMS is None or _CENTERED_GATE_ID is None:
        raise RuntimeError("Frozen-centered worker was not initialized")
    dictionary_indices = gate_indices(_CENTERED_TERMS, _CENTERED_GATE_ID)
    local_by_global = {global_index: local for local, global_index in enumerate(dictionary_indices)}
    full_bank, full_target, block_banks, block_targets = _load_worker_state(state_id)
    results = []
    for support in supports:
        local_support = tuple(local_by_global[index] for index in support)
        results.append(
            evaluate_state_support(
                state_id,
                full_bank,
                full_target,
                block_banks,
                block_targets,
                local_support,
                ridge_lambda=ridge_lambda,
            )
        )
    return state_id, results


class FrozenCenteredEvaluator:
    """Persistent state-major evaluator for one capacity-success dictionary."""

    def __init__(
        self,
        terms: Sequence[FrozenCenteredBasis],
        gate_id: str,
        state_ids: Sequence[int],
        worker_count: int,
    ) -> None:
        self.terms = tuple(terms)
        self.gate_id = gate_id
        self.state_ids = tuple(int(value) for value in state_ids)
        self.dictionary_indices = gate_indices(self.terms, gate_id)
        self._executor = ProcessPoolExecutor(
            max_workers=int(worker_count),
            initializer=_centered_worker_init,
            initargs=(self.terms, gate_id),
        )
        self._cache: dict[tuple[tuple[int, ...], float], FrozenCenteredSupportScore] = {}

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)

    def __enter__(self) -> FrozenCenteredEvaluator:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def evaluate_supports(
        self,
        supports: Sequence[Sequence[int]],
        *,
        ridge_lambda: float = 0.0,
    ) -> list[FrozenCenteredSupportScore]:
        normalized = [tuple(sorted(int(index) for index in support)) for support in supports]
        dictionary_set = set(self.dictionary_indices)
        if any(
            not support
            or len(set(support)) != len(support)
            or not set(support).issubset(dictionary_set)
            for support in normalized
        ):
            raise ValueError("Support is empty, duplicated, or outside the selected gate")
        missing: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()
        for support in normalized:
            key = (support, float(ridge_lambda))
            if key not in self._cache and support not in seen:
                missing.append(support)
                seen.add(support)
        if missing:
            task_supports = tuple(missing)
            per_support: list[list[object]] = [[] for _ in task_supports]
            futures = {
                self._executor.submit(
                    _centered_state_batch,
                    state_id,
                    task_supports,
                    float(ridge_lambda),
                ): state_id
                for state_id in self.state_ids
            }
            for future in as_completed(futures):
                _, metrics = future.result()
                for support_index, state_metrics in enumerate(metrics):
                    per_support[support_index].append(state_metrics)
            for support, metrics in zip(task_supports, per_support, strict=True):
                self._cache[(support, float(ridge_lambda))] = aggregate_frozen_centered_support(
                    support, metrics
                )
        return [self._cache[(support, float(ridge_lambda))] for support in normalized]


__all__ = [
    "FrozenCenteredEvaluator",
    "evaluate_capacity_state",
    "summarize_capacity_gate",
]
