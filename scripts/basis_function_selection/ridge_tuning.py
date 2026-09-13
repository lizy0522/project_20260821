"""Pareto support selection and raw-basis unified Ridge tuning."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .config import PARETO_MAX_SUPPORTS, RIDGE_LAMBDA_GRID
from .search import PersistentStateEvaluator
from .selection_metrics import SupportScore, dominates


@dataclass(frozen=True)
class RidgeTuningResult:
    final_score: SupportScore
    final_lambda: float
    pareto_ols_scores: tuple[SupportScore, ...]
    evaluated: tuple[tuple[SupportScore, float], ...]


@dataclass(frozen=True)
class FrozenCenteredRidgeResult:
    """Ridge result for the frozen-centered task-specific score."""

    final_score: object
    final_lambda: float
    pareto_ols_scores: tuple[object, ...]
    evaluated: tuple[tuple[object, float], ...]


def select_pareto_supports(
    scores: Sequence[SupportScore],
    selected_ols: SupportScore,
) -> tuple[SupportScore, ...]:
    """Return at most six unique nondominated supports, preserving required models."""

    unique = {score.support: score for score in scores if score.all_full_rank}
    unique[selected_ols.support] = selected_ols
    values = list(unique.values())
    front = [
        score
        for score in values
        if not any(dominates(other, score) for other in values if other.support != score.support)
    ]
    required: list[SupportScore] = [selected_ols]
    successful = [score for score in values if score.n40 == 20]
    if successful:
        smallest_success = min(successful, key=lambda score: (score.k, score.sort_key()))
        if smallest_success.support != selected_ols.support:
            required.append(smallest_success)
    ordered = sorted(front, key=lambda score: score.sort_key())
    result: list[SupportScore] = []
    for score in [*required, *ordered]:
        if score.support not in {item.support for item in result}:
            result.append(score)
        if len(result) >= PARETO_MAX_SUPPORTS:
            break
    return tuple(result)


def tune_ridge(
    evaluator: PersistentStateEvaluator,
    candidate_scores: Sequence[SupportScore],
    selected_ols: SupportScore,
) -> RidgeTuningResult:
    """Tune one shared raw-basis lambda on no more than six supports."""

    pareto = select_pareto_supports(candidate_scores, selected_ols)
    evaluated: list[tuple[SupportScore, float]] = []
    for support_index, score in enumerate(pareto, start=1):
        for ridge_lambda in RIDGE_LAMBDA_GRID:
            tuned = evaluator.evaluate_supports([score.support], ridge_lambda=ridge_lambda)[0]
            evaluated.append((tuned, float(ridge_lambda)))
        print(f"[RIDGE] support {support_index}/{len(pareto)}", flush=True)
    final_score, final_lambda = min(
        evaluated,
        key=lambda item: (*item[0].sort_key(), item[1]),
    )
    return RidgeTuningResult(
        final_score=final_score,
        final_lambda=final_lambda,
        pareto_ols_scores=pareto,
        evaluated=tuple(evaluated),
    )


def tune_frozen_centered_ridge(
    evaluator: object,
    candidate_scores: Sequence[object],
    selected_ols: object,
) -> FrozenCenteredRidgeResult:
    """Tune raw-basis Ridge on at most six frozen-centered Pareto supports."""

    from .selection_metrics import dominates_frozen_centered

    unique = {score.support: score for score in candidate_scores if score.all_full_rank}
    unique[selected_ols.support] = selected_ols
    values = list(unique.values())
    front = [
        score
        for score in values
        if not any(
            dominates_frozen_centered(other, score)
            for other in values
            if other.support != score.support
        )
    ]
    required = [selected_ols]
    successful = [score for score in values if score.n_w40 == 20]
    if successful:
        smallest = min(successful, key=lambda score: (score.k, score.sort_key()))
        if smallest.support != selected_ols.support:
            required.append(smallest)
    pareto = []
    for score in [*required, *sorted(front, key=lambda item: item.sort_key())]:
        if score.support not in {item.support for item in pareto}:
            pareto.append(score)
        if len(pareto) >= PARETO_MAX_SUPPORTS:
            break
    evaluated = []
    for support_index, score in enumerate(pareto, start=1):
        for ridge_lambda in RIDGE_LAMBDA_GRID:
            tuned = evaluator.evaluate_supports(
                [score.support],
                ridge_lambda=ridge_lambda,
            )[0]
            evaluated.append((tuned, float(ridge_lambda)))
        print(f"[RIDGE] centered support {support_index}/{len(pareto)}", flush=True)
    final_score, final_lambda = min(
        evaluated,
        key=lambda item: (*item[0].sort_key(), item[1]),
    )
    return FrozenCenteredRidgeResult(
        final_score=final_score,
        final_lambda=final_lambda,
        pareto_ols_scores=tuple(pareto),
        evaluated=tuple(evaluated),
    )


__all__ = [
    "FrozenCenteredRidgeResult",
    "RidgeTuningResult",
    "select_pareto_supports",
    "tune_frozen_centered_ridge",
    "tune_ridge",
]
