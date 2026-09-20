"""Deterministic single-bandwidth search framework with injectable evaluation.

No real states, MAT files, Ridge grids, worker pool, or champion are configured
here.  A future route task supplies the evaluator, explicit grids and budget.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .shareability_parallel import CandidateEvaluation
from .shareability_sufficient_stats import canonical_support, support_id
from .shareability_types import CandidateModelSpec, ObservationBandwidthSpec, RidgeGridSpec

LEGACY_STAGES = ("forward", "backward", "swap", "ridge")
SCREENED_STAGES = (
    "fixed_seed",
    "screen_forward",
    "evaluate_forward",
    "select_beam",
    "backward",
    "screen_swap",
    "evaluate_swap",
    "dense_ridge",
    "complete",
)
STAGES = tuple(dict.fromkeys(LEGACY_STAGES + SCREENED_STAGES))


@dataclass(frozen=True)
class SearchConfig:
    stages: tuple[str, ...]
    beam_width: int
    candidate_budget: int
    random_seed: int

    def __post_init__(self) -> None:
        if not self.stages or any(stage not in STAGES for stage in self.stages):
            raise ValueError("explicit search stages are required")
        if self.beam_width < 1 or self.candidate_budget < 1:
            raise ValueError("beam width and budget must be positive")


@dataclass(frozen=True)
class SearchCandidate:
    model: CandidateModelSpec
    stage: str

    @property
    def candidate_id(self) -> str:
        return self.model.candidate_id


@dataclass
class SearchState:
    bandwidth: ObservationBandwidthSpec
    dictionary_hash: str
    config_hash: str
    random_seed: int
    stage_index: int = 0
    frontier: list[CandidateModelSpec] = field(default_factory=list)
    pending: list[SearchCandidate] = field(default_factory=list)
    scores: dict[str, int] = field(default_factory=dict)
    evaluated: dict[str, dict[str, object]] = field(default_factory=dict)
    ridge_values_evaluated: list[float] = field(default_factory=list)
    screening: dict[str, object] = field(default_factory=dict)
    execution_metadata: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "bandwidth": self.bandwidth.to_dict(),
            "dictionary_hash": self.dictionary_hash,
            "config_hash": self.config_hash,
            "random_seed": self.random_seed,
            "stage_index": self.stage_index,
            "current_k": [model.k_selected for model in self.frontier],
            "beam_candidates": [model.to_dict() for model in self.frontier],
            "pending": [
                {"model": candidate.model.to_dict(), "stage": candidate.stage}
                for candidate in self.pending
            ],
            "evaluated_candidate_ids": sorted(self.evaluated),
            "candidate_scores": self.scores,
            "evaluated": self.evaluated,
            "ridge_values_already_evaluated": self.ridge_values_evaluated,
            "screening": self.screening,
            "parallel_execution": self.execution_metadata,
        }

    def scientific_dict(self) -> dict[str, object]:
        """Comparable scientific state, independent of worker count and runtime."""

        payload = self.to_dict()
        payload.pop("parallel_execution")
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> SearchState:
        bandwidth = ObservationBandwidthSpec.from_dict(data["bandwidth"])  # type: ignore[arg-type]
        return cls(
            bandwidth=bandwidth,
            dictionary_hash=str(data["dictionary_hash"]),
            config_hash=str(data["config_hash"]),
            random_seed=int(data["random_seed"]),
            stage_index=int(data["stage_index"]),
            frontier=[CandidateModelSpec.from_dict(row) for row in data["beam_candidates"]],  # type: ignore[arg-type]
            pending=[
                SearchCandidate(CandidateModelSpec.from_dict(row["model"]), str(row["stage"]))
                for row in data["pending"]  # type: ignore[union-attr]
            ],
            scores={str(key): int(value) for key, value in data["candidate_scores"].items()},  # type: ignore[union-attr]
            evaluated={str(key): value for key, value in data["evaluated"].items()},  # type: ignore[union-attr]
            ridge_values_evaluated=[
                float(value) for value in data["ridge_values_already_evaluated"]
            ],  # type: ignore[arg-type]
            screening=dict(data.get("screening", {})),  # type: ignore[arg-type]
            execution_metadata=dict(data.get("parallel_execution", {})),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class SearchResult:
    state: SearchState
    complete: bool
    tied_primary_ids: tuple[str, ...]


def record_screening_progress(
    state: SearchState,
    parent_support: tuple[str, ...],
    *,
    candidate_pool_hash: str,
    screening_spec_hash: str,
    total_blocks: int,
    completed_blocks: Sequence[int],
    score_artifact_path: str | None = None,
    shortlist: Sequence[str] = (),
    exact_evaluation_completed: int = 0,
) -> None:
    """Store resumable proposal state separately from exact CandidateScores."""
    canonical = canonical_support(parent_support)
    if len(candidate_pool_hash) != 64 or len(screening_spec_hash) != 64:
        raise ValueError("screening candidate-pool/spec fingerprints must be SHA256 values")
    if total_blocks < 1 or exact_evaluation_completed < 0:
        raise ValueError("invalid screening progress counts")
    completed = tuple(sorted(set(int(index) for index in completed_blocks)))
    if any(index < 0 or index >= total_blocks for index in completed):
        raise ValueError("completed screening block is outside the current plan")
    if len(set(shortlist)) != len(tuple(shortlist)):
        raise ValueError("screening shortlist contains duplicate basis IDs")
    state.screening[support_id(canonical)] = {
        "parent_support": list(canonical),
        "candidate_pool_hash": candidate_pool_hash,
        "screening_spec_hash": screening_spec_hash,
        "total_blocks": total_blocks,
        "completed_blocks": list(completed),
        "score_artifact_path": score_artifact_path,
        "shortlist": list(shortlist),
        "exact_evaluation_completed": exact_evaluation_completed,
    }


def remaining_screening_blocks(
    state: SearchState,
    parent_support: tuple[str, ...],
    *,
    candidate_pool_hash: str,
    screening_spec_hash: str,
) -> tuple[int, ...]:
    """Reject incompatible cache state; return only blocks not safely completed."""
    canonical = canonical_support(parent_support)
    payload = state.screening.get(support_id(canonical))
    if not isinstance(payload, dict):
        raise KeyError("no screening progress for parent support")
    if (
        payload.get("parent_support") != list(canonical)
        or payload.get("candidate_pool_hash") != candidate_pool_hash
        or payload.get("screening_spec_hash") != screening_spec_hash
    ):
        raise ValueError("screening checkpoint is incompatible with current candidate pool/spec")
    total = int(payload["total_blocks"])
    completed = {int(value) for value in payload["completed_blocks"]}
    return tuple(index for index in range(total) if index not in completed)


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def dictionary_fingerprint(dictionary_ids: Sequence[str]) -> str:
    if len(set(dictionary_ids)) != len(dictionary_ids):
        raise ValueError("duplicate dictionary basis IDs")
    return _hash(sorted(dictionary_ids))


def make_search_state(
    seed: CandidateModelSpec,
    dictionary_ids: Sequence[str],
    ridge_grid: RidgeGridSpec,
    config: SearchConfig,
) -> SearchState:
    pool = set(dictionary_ids)
    if not set(seed.basis_ids) <= pool:
        raise ValueError("seed outside dictionary")
    config_hash = _hash({"config": config.__dict__, "ridge_values": ridge_grid.values})
    return SearchState(
        seed.bandwidth_spec,
        dictionary_fingerprint(dictionary_ids),
        config_hash,
        config.random_seed,
        frontier=[seed],
    )


def neighbors(
    model: CandidateModelSpec,
    stage: str,
    dictionary_ids: Sequence[str],
    ridge_grid: RidgeGridSpec,
) -> tuple[CandidateModelSpec, ...]:
    selected = set(model.basis_ids)
    pool = sorted(set(dictionary_ids))
    if not selected <= set(pool):
        raise ValueError("selected basis missing from dictionary")
    supports: list[tuple[str, ...]] = []
    if stage == "forward" and len(selected) < 20:
        supports = [tuple(sorted(selected | {item})) for item in pool if item not in selected]
    elif stage == "backward" and len(selected) > 1:
        supports = [tuple(sorted(selected - {item})) for item in sorted(selected)]
    elif stage == "swap":
        supports = [
            tuple(sorted((selected - {removed}) | {added}))
            for removed in sorted(selected)
            for added in pool
            if added not in selected
        ]
    elif stage == "ridge":
        return tuple(
            CandidateModelSpec(model.bandwidth_spec, model.basis_ids, value)
            for value in ridge_grid.values
            if value != model.ridge_lambda
        )
    elif stage in SCREENED_STAGES:
        raise ValueError("screened stages require the route task's screening orchestrator")
    elif stage not in STAGES:
        raise ValueError("unknown stage")
    return tuple(
        CandidateModelSpec(model.bandwidth_spec, support, model.ridge_lambda)
        for support in supports
    )


def run_shareability_search(
    state: SearchState,
    dictionary_ids: Sequence[str],
    ridge_grid: RidgeGridSpec,
    config: SearchConfig,
    evaluate: Callable[[CandidateModelSpec], int],
    *,
    max_this_call: int | None = None,
    evaluate_many: Callable[[Sequence[CandidateModelSpec]], Sequence[int | CandidateEvaluation]]
    | None = None,
    checkpoint_path_on_interrupt: Path | None = None,
) -> SearchResult:
    """Run or resume with a caller-supplied evaluator; primary ties are retained.

    ``evaluate_many`` may be a spawn-safe parallel evaluator in a future task.
    This engine itself never creates processes or inspects any real dataset.
    """

    expected = make_search_state(state.frontier[0], dictionary_ids, ridge_grid, config)
    if (state.dictionary_hash, state.config_hash, state.random_seed) != (
        expected.dictionary_hash,
        expected.config_hash,
        expected.random_seed,
    ):
        raise ValueError("checkpoint dictionary/config/seed fingerprint mismatch")
    if any(model.bandwidth_spec != state.bandwidth for model in state.frontier):
        raise ValueError("cannot inherit a champion across bandwidths")
    remaining = float("inf") if max_this_call is None else max_this_call
    if remaining < 0:
        raise ValueError("max_this_call must be nonnegative")
    seed = state.frontier[0]

    def evaluate_batch(batch: Sequence[CandidateModelSpec]) -> list[int]:
        if evaluate_many is None:
            return [evaluate(candidate) for candidate in batch]
        results = list(evaluate_many(batch))
        if len(results) != len(batch):
            raise ValueError("parallel evaluator returned a different number of scores")
        if all(isinstance(item, CandidateEvaluation) for item in results):
            by_id = {item.candidate_id: item for item in results}
            if len(by_id) != len(batch) or set(by_id) != {item.candidate_id for item in batch}:
                raise ValueError("parallel evaluator returned missing/duplicate candidate IDs")
            values = [by_id[item.candidate_id].n_shareable for item in batch]
        elif all(isinstance(item, int) for item in results):
            values = results
        else:
            raise TypeError("parallel evaluator must return integer or ID-tagged results")
        owner = getattr(evaluate_many, "__self__", None)
        if owner is not None and hasattr(owner, "runtime_metadata"):
            state.execution_metadata = dict(owner.runtime_metadata)
        return values

    if seed.candidate_id not in state.evaluated and remaining > 0:
        try:
            value = evaluate_batch([seed])[0]
        except KeyboardInterrupt:
            if checkpoint_path_on_interrupt is not None:
                from .shareability_persistence import save_checkpoint

                save_checkpoint(state, checkpoint_path_on_interrupt, replace=True)
            raise
        if not isinstance(value, int) or not 0 <= value <= 425:
            raise ValueError("evaluator must return integer N_shareable in 0..425")
        state.scores[seed.candidate_id] = value
        state.evaluated[seed.candidate_id] = {
            "model": seed.to_dict(),
            "stage": "seed",
            "N_shareable": value,
        }
        remaining -= 1
    if remaining == 0:
        best = max(state.scores.values(), default=None)
        tied = tuple(sorted(key for key, score in state.scores.items() if score == best))
        return SearchResult(state, False, tied)
    while state.stage_index < len(config.stages):
        stage = config.stages[state.stage_index]
        if not state.pending:
            generated = {
                candidate.candidate_id: SearchCandidate(candidate, stage)
                for model in state.frontier
                for candidate in neighbors(model, stage, dictionary_ids, ridge_grid)
                if candidate.candidate_id not in state.evaluated
            }
            state.pending = [generated[key] for key in sorted(generated)]
        # A completed stage can be empty (e.g., an already saturated K=20 frontier).
        while state.pending and remaining > 0 and len(state.evaluated) < config.candidate_budget:
            batch_size = (
                1
                if evaluate_many is None
                else min(
                    len(state.pending),
                    int(remaining) if remaining != float("inf") else len(state.pending),
                    config.candidate_budget - len(state.evaluated),
                )
            )
            batch = state.pending[:batch_size]
            try:
                values = evaluate_batch([item.model for item in batch])
            except KeyboardInterrupt:
                if checkpoint_path_on_interrupt is not None:
                    from .shareability_persistence import save_checkpoint

                    save_checkpoint(state, checkpoint_path_on_interrupt, replace=True)
                raise
            for item, value in zip(batch, values, strict=True):
                if not isinstance(value, int) or not 0 <= value <= 425:
                    raise ValueError("evaluator must return an integer N_shareable in 0..425")
                state.scores[item.candidate_id] = value
                state.evaluated[item.candidate_id] = {
                    "model": item.model.to_dict(),
                    "stage": stage,
                    "N_shareable": value,
                }
                if stage == "ridge" and item.model.ridge_lambda not in state.ridge_values_evaluated:
                    state.ridge_values_evaluated.append(item.model.ridge_lambda)
            del state.pending[: len(batch)]
            remaining -= len(batch)
        if state.pending:
            break
        # Select by N_shareable only; preserve all candidates tied at the beam boundary.
        rows = [row for row in state.evaluated.values() if row["stage"] == stage]
        if rows:
            rows.sort(
                key=lambda row: (
                    -int(row["N_shareable"]),
                    CandidateModelSpec.from_dict(row["model"]).candidate_id,
                )
            )
            boundary = int(rows[min(config.beam_width, len(rows)) - 1]["N_shareable"])
            state.frontier = [
                CandidateModelSpec.from_dict(row["model"])
                for row in rows
                if int(row["N_shareable"]) >= boundary
            ]
        state.stage_index += 1
        if len(state.evaluated) >= config.candidate_budget:
            break
    best = max(state.scores.values(), default=None)
    tied = tuple(sorted(key for key, score in state.scores.items() if score == best))
    return SearchResult(state, state.stage_index == len(config.stages) and not state.pending, tied)


__all__ = [
    "LEGACY_STAGES",
    "SCREENED_STAGES",
    "STAGES",
    "SearchConfig",
    "SearchCandidate",
    "SearchState",
    "SearchResult",
    "dictionary_fingerprint",
    "make_search_state",
    "neighbors",
    "record_screening_progress",
    "remaining_screening_blocks",
    "run_shareability_search",
]
