"""Persistent state-major sparse-support search for the Hard-20 experiment."""

from __future__ import annotations

import itertools
import time
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass

from .config import (
    K_MAX,
    OLS_CONDITION_HARD_LIMIT,
    PAIR_MAX_ROUNDS,
    PAIR_MEAN_IMPROVEMENT_DB,
    PAIR_TOP_L,
    PAIR_WORST_IMPROVEMENT_DB,
    PLATEAU_STEPS,
    PLATEAU_WORST_IMPROVEMENT_DB,
    SWAP_MAX_ACCEPTS,
    SWAP_TOP_L,
    SWAP_WORST_IMPROVEMENT_DB,
)
from .data_preparation import HardCacheSpec, load_selection_cache_arrays
from .model_solver import StateModelMetrics, evaluate_state_support
from .selection_metrics import SupportScore, aggregate_support, backward_deletion_allowed
from .volterra_dictionary import VolterraBasis

_WORKER_SPEC: HardCacheSpec | None = None
_WORKER_ARRAYS: dict[str, object] | None = None


def _worker_init(spec: HardCacheSpec) -> None:
    global _WORKER_SPEC, _WORKER_ARRAYS
    _WORKER_SPEC = spec
    _WORKER_ARRAYS = load_selection_cache_arrays(spec)


def _state_batch_worker(
    state_index: int,
    supports: tuple[tuple[int, ...], ...],
    ridge_lambda: float,
) -> tuple[int, list[StateModelMetrics]]:
    if _WORKER_SPEC is None or _WORKER_ARRAYS is None:
        raise RuntimeError("Selection worker was not initialized")
    state_id = int(_WORKER_SPEC.state_ids[state_index])
    arrays = _WORKER_ARRAYS
    full_bank = arrays["full_bank"][state_index]
    full_target = arrays["full_target"][state_index]
    block_banks = [arrays[f"block{index}_bank"][state_index] for index in range(3)]
    block_targets = [arrays[f"block{index}_target"][state_index] for index in range(3)]
    results = [
        evaluate_state_support(
            state_id,
            full_bank,
            full_target,
            block_banks,
            block_targets,
            support,
            ridge_lambda=ridge_lambda,
        )
        for support in supports
    ]
    return state_index, results


class PersistentStateEvaluator:
    """Persistent process pool evaluating all supports state-major."""

    def __init__(self, spec: HardCacheSpec, worker_count: int):
        self.spec = spec
        self.worker_count = max(1, int(worker_count))
        self._executor = ProcessPoolExecutor(
            max_workers=self.worker_count,
            initializer=_worker_init,
            initargs=(spec,),
        )
        self._cache: dict[tuple[tuple[int, ...], float], SupportScore] = {}

    def close(self) -> None:
        self._executor.shutdown(wait=True, cancel_futures=False)

    def __enter__(self) -> PersistentStateEvaluator:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def evaluate_supports(
        self,
        supports: Sequence[Sequence[int]],
        *,
        ridge_lambda: float = 0.0,
    ) -> list[SupportScore]:
        normalized = [tuple(sorted(int(index) for index in support)) for support in supports]
        if any(not support or len(set(support)) != len(support) for support in normalized):
            raise ValueError("Every support must be nonempty and unique")
        missing = []
        seen: set[tuple[int, ...]] = set()
        for support in normalized:
            key = (support, float(ridge_lambda))
            if key not in self._cache and support not in seen:
                missing.append(support)
                seen.add(support)
        if missing:
            task_supports = tuple(missing)
            per_support: list[list[StateModelMetrics]] = [[] for _ in task_supports]
            futures = {
                self._executor.submit(
                    _state_batch_worker, state_index, task_supports, float(ridge_lambda)
                ): state_index
                for state_index in range(len(self.spec.state_ids))
            }
            for future in as_completed(futures):
                _, state_results = future.result()
                for support_index, metrics in enumerate(state_results):
                    per_support[support_index].append(metrics)
            for support, metrics in zip(task_supports, per_support, strict=True):
                self._cache[(support, float(ridge_lambda))] = aggregate_support(support, metrics)
        return [self._cache[(support, float(ridge_lambda))] for support in normalized]


@dataclass(frozen=True)
class SearchResult:
    selected_ols: SupportScore
    accepted_scores: tuple[SupportScore, ...]
    path_rows: tuple[dict[str, object], ...]
    stop_reason: str


@dataclass(frozen=True)
class FrozenCenteredSearchResult:
    """Accepted path and selected OLS support for the nested Gate task."""

    selected_ols: object
    accepted_scores: tuple[object, ...]
    path_rows: tuple[dict[str, object], ...]
    stop_reason: str


def _valid(score: SupportScore) -> bool:
    return bool(score.all_full_rank and score.q99_condition <= OLS_CONDITION_HARD_LIMIT)


def _path_row(
    step: int,
    action: str,
    score: SupportScore,
    *,
    added: Sequence[int] = (),
    removed: Sequence[int] = (),
    elapsed_sec: float,
) -> dict[str, object]:
    return {
        "step": int(step),
        "action": action,
        "added_indices": ";".join(str(value) for value in added),
        "removed_indices": ";".join(str(value) for value in removed),
        "K": score.k,
        "N40": score.n40,
        "worst_W_dB": score.worst_w_db,
        "worst_deficit_dB": score.worst_deficit_db,
        "Q90_deficit_dB": score.q90_deficit_db,
        "mean_deficit_dB": score.mean_deficit_db,
        "median_W_dB": score.median_w_db,
        "Q99_condition": score.q99_condition,
        "support_indices": ";".join(str(value) for value in score.support),
        "elapsed_sec": float(elapsed_sec),
    }


def _best_valid(scores: Iterable[SupportScore]) -> SupportScore | None:
    valid = [score for score in scores if _valid(score)]
    return min(valid, key=lambda score: score.sort_key()) if valid else None


def _backward_cleanup(
    evaluator: PersistentStateEvaluator,
    current: SupportScore,
    mandatory_index: int,
    path_rows: list[dict[str, object]],
    accepted: list[SupportScore],
    started: float,
    step_ref: list[int],
) -> SupportScore:
    while True:
        removable = [index for index in current.support if index != mandatory_index]
        if not removable:
            return current
        supports = [
            tuple(index for index in current.support if index != removed) for removed in removable
        ]
        scores = evaluator.evaluate_supports(supports)
        allowed = [
            (removed, score)
            for removed, score in zip(removable, scores, strict=True)
            if _valid(score) and backward_deletion_allowed(current, score)
        ]
        if not allowed:
            return current
        removed, selected = min(allowed, key=lambda item: item[1].sort_key())
        current = selected
        accepted.append(current)
        step_ref[0] += 1
        path_rows.append(
            _path_row(
                step_ref[0],
                "backward",
                current,
                removed=(removed,),
                elapsed_sec=time.perf_counter() - started,
            )
        )


def _passed_state_ids(score: SupportScore) -> set[int]:
    return {item.state_id for item in score.state_metrics if item.worst_nmse_db < -40.0}


def _pair_acceptable(current: SupportScore, candidate: SupportScore) -> bool:
    if candidate.n40 > current.n40:
        return True
    if candidate.n40 != current.n40:
        return False
    if current.worst_deficit_db - candidate.worst_deficit_db >= PAIR_WORST_IMPROVEMENT_DB:
        return True
    no_damage = _passed_state_ids(current).issubset(_passed_state_ids(candidate))
    return bool(
        no_damage
        and current.mean_deficit_db - candidate.mean_deficit_db >= PAIR_MEAN_IMPROVEMENT_DB
    )


def run_sparse_search(
    evaluator: PersistentStateEvaluator,
    terms: Sequence[VolterraBasis],
) -> SearchResult:
    """Run frozen Forward/Backward/Pair/Swap search from x[n]."""

    mandatory = next(term.index for term in terms if term.mandatory)
    started = time.perf_counter()
    current = evaluator.evaluate_supports([(mandatory,)])[0]
    path_rows = [_path_row(0, "initial", current, elapsed_sec=0.0)]
    accepted = [current]
    step_ref = [0]
    plateau_count = 0
    pair_rounds = 0
    swap_accepts = 0
    last_single_scores: list[SupportScore] = []
    rejected_at_current: set[int] = set()
    stop_reason = "unknown"

    while current.k < K_MAX:
        if current.n40 == 20:
            current = _backward_cleanup(
                evaluator, current, mandatory, path_rows, accepted, started, step_ref
            )
            stop_reason = "OLS_20_of_20_compressed"
            break
        remaining = [
            term.index
            for term in terms
            if term.index not in current.support and term.index not in rejected_at_current
        ]
        if not remaining:
            stop_reason = "no_remaining_candidate"
            break
        forward_supports = [tuple(sorted((*current.support, candidate))) for candidate in remaining]
        forward_scores = evaluator.evaluate_supports(forward_supports)
        last_single_scores = forward_scores
        selected = _best_valid(forward_scores)
        if selected is None:
            stop_reason = "no_numerically_valid_forward_candidate"
            break
        added = next(index for index in selected.support if index not in current.support)
        previous = current
        current = selected
        step_ref[0] += 1
        path_rows.append(
            _path_row(
                step_ref[0],
                "forward",
                current,
                added=(added,),
                elapsed_sec=time.perf_counter() - started,
            )
        )
        accepted.append(current)
        current = _backward_cleanup(
            evaluator, current, mandatory, path_rows, accepted, started, step_ref
        )
        if current.support == previous.support:
            rejected_at_current.add(added)
        else:
            rejected_at_current.clear()
        n40_improved = current.n40 > previous.n40
        worst_improvement = previous.worst_deficit_db - current.worst_deficit_db
        plateau_count = (
            0
            if n40_improved or worst_improvement >= PLATEAU_WORST_IMPROVEMENT_DB
            else plateau_count + 1
        )
        print(
            f"[SEARCH] step={step_ref[0]} K={current.k} N40={current.n40}/20 "
            f"worst_deficit={current.worst_deficit_db:.4f} dB "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )
        if current.n40 == 20:
            continue
        if plateau_count < PLATEAU_STEPS:
            continue

        rescued = False
        if pair_rounds < PAIR_MAX_ROUNDS and current.k <= K_MAX - 2:
            pair_rounds += 1
            ranked_single = sorted(
                [score for score in last_single_scores if _valid(score)],
                key=lambda score: score.sort_key(),
            )[:PAIR_TOP_L]
            incoming = [
                next(index for index in score.support if index not in current.support)
                for score in ranked_single
            ]
            pair_supports = [
                tuple(sorted((*current.support, left, right)))
                for left, right in itertools.combinations(incoming, 2)
            ]
            pair_scores = evaluator.evaluate_supports(pair_supports)
            acceptable = [
                score for score in pair_scores if _valid(score) and _pair_acceptable(current, score)
            ]
            if acceptable:
                selected_pair = min(acceptable, key=lambda score: score.sort_key())
                added_pair = tuple(
                    index for index in selected_pair.support if index not in current.support
                )
                current = selected_pair
                step_ref[0] += 1
                path_rows.append(
                    _path_row(
                        step_ref[0],
                        "pair_rescue",
                        current,
                        added=added_pair,
                        elapsed_sec=time.perf_counter() - started,
                    )
                )
                accepted.append(current)
                current = _backward_cleanup(
                    evaluator, current, mandatory, path_rows, accepted, started, step_ref
                )
                rescued = True
                plateau_count = 0
                rejected_at_current.clear()
                print(f"[PAIR RESCUE] accepted K={current.k} N40={current.n40}/20", flush=True)
        if rescued:
            continue

        while swap_accepts < SWAP_MAX_ACCEPTS and current.n40 < 20:
            remaining = [term.index for term in terms if term.index not in current.support]
            single_scores = evaluator.evaluate_supports(
                [tuple(sorted((*current.support, candidate))) for candidate in remaining]
            )
            ranked_single = sorted(
                [score for score in single_scores if _valid(score)],
                key=lambda score: score.sort_key(),
            )[:SWAP_TOP_L]
            incoming = [
                next(index for index in score.support if index not in current.support)
                for score in ranked_single
            ]
            swap_specs: list[tuple[int, int, tuple[int, ...]]] = []
            for removed in current.support:
                if removed == mandatory:
                    continue
                for added_index in incoming:
                    support = tuple(
                        sorted(
                            index for index in (*current.support, added_index) if index != removed
                        )
                    )
                    swap_specs.append((removed, added_index, support))
            swap_scores = evaluator.evaluate_supports([item[2] for item in swap_specs])
            acceptable_swaps = []
            for spec, score in zip(swap_specs, swap_scores, strict=True):
                if not _valid(score):
                    continue
                if score.n40 > current.n40 or (
                    score.n40 == current.n40
                    and current.worst_deficit_db - score.worst_deficit_db
                    >= SWAP_WORST_IMPROVEMENT_DB
                ):
                    acceptable_swaps.append((spec, score))
            if not acceptable_swaps:
                break
            (removed, added_index, _), selected_swap = min(
                acceptable_swaps,
                key=lambda item: item[1].sort_key(),
            )
            current = selected_swap
            swap_accepts += 1
            step_ref[0] += 1
            path_rows.append(
                _path_row(
                    step_ref[0],
                    "swap",
                    current,
                    added=(added_index,),
                    removed=(removed,),
                    elapsed_sec=time.perf_counter() - started,
                )
            )
            accepted.append(current)
            current = _backward_cleanup(
                evaluator, current, mandatory, path_rows, accepted, started, step_ref
            )
            plateau_count = 0
            rejected_at_current.clear()
            print(f"[SWAP] accepted={swap_accepts} K={current.k} N40={current.n40}/20", flush=True)
        if plateau_count >= PLATEAU_STEPS:
            stop_reason = "plateau_after_pair_and_swap"
            break

    if stop_reason == "unknown":
        stop_reason = "K_max_reached"
    selected_ols = min(
        (score for score in accepted if _valid(score)), key=lambda score: score.sort_key()
    )
    return SearchResult(
        selected_ols=selected_ols,
        accepted_scores=tuple(accepted),
        path_rows=tuple(path_rows),
        stop_reason=stop_reason,
    )


def run_frozen_centered_search(
    evaluator: object,
    dictionary_indices: Sequence[int],
    frozen_support: Sequence[int],
    mandatory_index: int,
    *,
    full_space_score: object,
    k_max: int,
) -> FrozenCenteredSearchResult:
    """Run the frozen-centered Forward/Backward/Pair/Swap procedure."""

    from .selection_metrics import frozen_centered_deletion_allowed

    started = time.perf_counter()
    accepted: list[object] = []
    path: list[dict[str, object]] = []
    step = 0

    def score_key(score: object) -> tuple[object, ...]:
        return score.sort_key()

    def valid(score: object) -> bool:
        return bool(score.all_full_rank and score.q99_condition <= OLS_CONDITION_HARD_LIMIT)

    def pass_ids(score: object) -> set[int]:
        return {metric.state_id for metric in score.state_metrics if metric.worst_nmse_db < -40.0}

    def add_path(
        action: str,
        score: object,
        previous: object | None,
        added: Sequence[int] = (),
        removed: Sequence[int] = (),
    ) -> None:
        nonlocal step
        current_pass = pass_ids(score)
        previous_pass = pass_ids(previous) if previous is not None else set()
        path.append(
            {
                "step": step,
                "action": action,
                "added_indices": ";".join(str(value) for value in added),
                "removed_indices": ";".join(str(value) for value in removed),
                "K": score.k,
                "N_train40": score.n_train40,
                "N_W40": score.n_w40,
                "worst_deficit_db": score.worst_deficit_db,
                "Q90_deficit_db": score.q90_deficit_db,
                "mean_deficit_db": score.mean_deficit_db,
                "median_W_db": score.median_w_db,
                "N_rescue": len(current_pass - previous_pass),
                "N_damage": len(previous_pass - current_pass),
                "support_indices": ";".join(str(value) for value in score.support),
                "elapsed_sec": time.perf_counter() - started,
            }
        )

    def backward(current: object) -> object:
        nonlocal step
        while True:
            removable = [index for index in current.support if index != mandatory_index]
            if not removable:
                return current
            supports = [
                tuple(index for index in current.support if index != removed)
                for removed in removable
            ]
            scores = evaluator.evaluate_supports(supports)
            allowed = [
                (removed, score)
                for removed, score in zip(removable, scores, strict=True)
                if valid(score) and frozen_centered_deletion_allowed(current, score)
            ]
            if not allowed:
                return current
            removed, selected = min(allowed, key=lambda item: score_key(item[1]))
            previous = current
            current = selected
            accepted.append(current)
            step += 1
            add_path("backward", current, previous, removed=(removed,))

    current = evaluator.evaluate_supports([tuple(sorted(frozen_support))])[0]
    if not valid(current):
        raise RuntimeError("Frozen10 start support is numerically invalid")
    accepted.append(current)
    add_path("initial_frozen10", current, None)
    plateau = 0
    pair_rounds = 0
    swaps = 0
    last_single_scores: list[object] = []
    stop_reason = "unknown"

    while current.k < k_max and current.n_w40 < 20:
        remaining = [index for index in dictionary_indices if index not in current.support]
        if not remaining:
            stop_reason = "no_remaining_candidate"
            break
        candidates = [tuple(sorted((*current.support, index))) for index in remaining]
        scores = evaluator.evaluate_supports(candidates)
        last_single_scores = [score for score in scores if valid(score)]
        if not last_single_scores:
            stop_reason = "no_valid_forward_candidate"
            break
        selected = min(last_single_scores, key=score_key)
        added = next(index for index in selected.support if index not in current.support)
        previous = current
        current = selected
        accepted.append(current)
        step += 1
        add_path("forward", current, previous, added=(added,))
        current = backward(current)
        n_w_improved = current.n_w40 > previous.n_w40
        worst_improvement = previous.worst_deficit_db - current.worst_deficit_db
        plateau = 0 if n_w_improved or worst_improvement >= 0.05 else plateau + 1
        print(
            f"[SEARCH] step={step} K={current.k} Train={current.n_train40}/20 "
            f"W={current.n_w40}/20 worst_deficit={current.worst_deficit_db:.4f} dB",
            flush=True,
        )
        if current.n_w40 == 20:
            current = backward(current)
            stop_reason = "W20_compressed"
            break
        if plateau < PLATEAU_STEPS:
            continue

        rescued = False
        if pair_rounds < PAIR_MAX_ROUNDS and current.k <= k_max - 2:
            pair_rounds += 1
            ranked = sorted(last_single_scores, key=score_key)
            incoming = []
            for score in ranked:
                added_indices = [index for index in score.support if index not in current.support]
                if len(added_indices) == 1 and added_indices[0] not in incoming:
                    incoming.append(added_indices[0])
                if len(incoming) >= PAIR_TOP_L:
                    break
            pair_supports = [
                tuple(sorted((*current.support, left, right)))
                for left, right in itertools.combinations(incoming, 2)
            ]
            pair_scores = [
                score for score in evaluator.evaluate_supports(pair_supports) if valid(score)
            ]
            current_pass = pass_ids(current)
            acceptable = []
            for score in pair_scores:
                no_damage = current_pass.issubset(pass_ids(score))
                if (
                    score.n_w40 > current.n_w40
                    or (
                        score.n_w40 == current.n_w40
                        and current.worst_deficit_db - score.worst_deficit_db >= 0.10
                    )
                    or (
                        score.n_w40 == current.n_w40
                        and no_damage
                        and current.mean_deficit_db - score.mean_deficit_db >= 0.05
                    )
                ):
                    acceptable.append(score)
            if acceptable:
                selected = min(acceptable, key=score_key)
                added_pair = tuple(
                    index for index in selected.support if index not in current.support
                )
                previous = current
                current = selected
                accepted.append(current)
                step += 1
                add_path("pair_rescue", current, previous, added=added_pair)
                current = backward(current)
                plateau = 0
                rescued = True
                print(
                    f"[PAIR RESCUE] round={pair_rounds} K={current.k} W={current.n_w40}/20",
                    flush=True,
                )
        if rescued:
            continue

        while swaps < SWAP_MAX_ACCEPTS and current.n_w40 < 20:
            remaining = [index for index in dictionary_indices if index not in current.support]
            single_scores = evaluator.evaluate_supports(
                [tuple(sorted((*current.support, index))) for index in remaining]
            )
            ranked = sorted([score for score in single_scores if valid(score)], key=score_key)[
                :SWAP_TOP_L
            ]
            incoming = [
                next(index for index in score.support if index not in current.support)
                for score in ranked
            ]
            specs = []
            for removed in current.support:
                if removed == mandatory_index:
                    continue
                for added_index in incoming:
                    support = tuple(
                        sorted(
                            index for index in (*current.support, added_index) if index != removed
                        )
                    )
                    specs.append((removed, added_index, support))
            swap_scores = evaluator.evaluate_supports([spec[2] for spec in specs])
            acceptable = []
            for spec, score in zip(specs, swap_scores, strict=True):
                if not valid(score):
                    continue
                if score.n_w40 > current.n_w40 or (
                    score.n_w40 == current.n_w40
                    and current.worst_deficit_db - score.worst_deficit_db >= 0.05
                ):
                    acceptable.append((spec, score))
            if not acceptable:
                break
            (removed, added_index, _), selected = min(
                acceptable,
                key=lambda item: score_key(item[1]),
            )
            previous = current
            current = selected
            accepted.append(current)
            swaps += 1
            step += 1
            add_path("swap", current, previous, added=(added_index,), removed=(removed,))
            current = backward(current)
            plateau = 0
            print(f"[SWAP] accepted={swaps} K={current.k} W={current.n_w40}/20", flush=True)
        if plateau >= PLATEAU_STEPS:
            stop_reason = "plateau_after_rescue_and_swap"
            break

    if current.n_w40 == 20 and stop_reason == "unknown":
        current = backward(current)
        stop_reason = "W20_compressed"
    elif stop_reason == "unknown":
        stop_reason = "K_max_reached" if current.k >= k_max else "search_stopped"

    if full_space_score.n_w40 == 20 and current.n_w40 < 20:
        previous = current
        full_current = full_space_score
        accepted.append(full_current)
        step += 1
        add_path("backward_from_full_start", full_current, previous)
        while full_current.k > 1:
            removable = [index for index in full_current.support if index != mandatory_index]
            candidates = [
                tuple(index for index in full_current.support if index != removed)
                for removed in removable
            ]
            scores = evaluator.evaluate_supports(candidates)
            preserving = [
                (removed, score)
                for removed, score in zip(removable, scores, strict=True)
                if valid(score) and score.n_w40 == 20
            ]
            if not preserving:
                break
            removed, selected = min(preserving, key=lambda item: score_key(item[1]))
            previous = full_current
            full_current = selected
            accepted.append(full_current)
            step += 1
            add_path("backward_from_full", full_current, previous, removed=(removed,))
        current = min(
            [score for score in accepted if score.n_w40 == 20],
            key=score_key,
        )
        stop_reason = "backward_from_full_W20"

    selected_ols = min([score for score in accepted if valid(score)], key=score_key)
    return FrozenCenteredSearchResult(
        selected_ols=selected_ols,
        accepted_scores=tuple(accepted),
        path_rows=tuple(path),
        stop_reason=stop_reason,
    )


__all__ = [
    "FrozenCenteredSearchResult",
    "PersistentStateEvaluator",
    "SearchResult",
    "run_frozen_centered_search",
    "run_sparse_search",
]
