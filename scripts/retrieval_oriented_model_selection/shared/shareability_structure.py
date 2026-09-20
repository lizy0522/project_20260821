"""Fixed-lambda structure-stage enumeration and content-addressed score cache."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from behavior_modeling.shared.volterra_terms import VolterraTermSpec

from .shareability_persistence import CacheKey, MetadataCache
from .shareability_screening import ParentScreeningScores
from .shareability_sufficient_stats import canonical_support, support_id
from .shareability_types import (
    CandidateModelSpec,
    CandidateScreeningSpec,
    ObservationBandwidthSpec,
    StructureSearchSpec,
)

SCORE_FIELDS = ("N_shareable", "N_self", "N_top3", "N_top5", "N_top10")


def seed_candidate(
    spec: StructureSearchSpec, observation: ObservationBandwidthSpec
) -> CandidateModelSpec:
    return spec.candidate(observation, (spec.seed_basis_id,))


def unused_candidate_ids(
    support: tuple[str, ...], dictionary_ids: Iterable[str]
) -> tuple[str, ...]:
    """Complete canonical unused pool; screening may only reduce evaluation cost."""
    canonical = canonical_support(support)
    pool = tuple(sorted(set(dictionary_ids)))
    if not set(canonical) <= set(pool):
        raise ValueError("selected support basis missing from dictionary")
    return tuple(basis for basis in pool if basis not in canonical)


@dataclass(frozen=True, slots=True)
class ScreenedChildProposal:
    """One exact-evaluation proposal, retaining its non-scientific provenance."""

    model: CandidateModelSpec
    parent_support_id: str
    screening_score: float
    screening_rank: int
    proposal_source: str
    screening_stratum: str

    def trace_fields(self) -> dict[str, object]:
        return {
            "screening_method": "somp_residual_correlation",
            "screening_score": self.screening_score,
            "screening_rank": self.screening_rank,
            "proposal_source": self.proposal_source,
            "screening_stratum": self.screening_stratum,
            "parent_support_id": self.parent_support_id,
        }


def screened_forward_proposals(
    parents: Iterable[CandidateModelSpec],
    screening_scores: dict[tuple[str, ...], ParentScreeningScores],
    screening_spec: CandidateScreeningSpec,
    structure: StructureSearchSpec,
    observation: ObservationBandwidthSpec,
    dictionary_ids: Iterable[str],
    dictionary: dict[str, VolterraTermSpec],
) -> tuple[ScreenedChildProposal, ...]:
    """Build canonical, cross-parent-deduplicated exact proposals from shortlists."""
    dictionary_pool = tuple(sorted(set(dictionary_ids)))
    proposals: dict[str, ScreenedChildProposal] = {}
    for parent in parents:
        if parent.ridge_lambda != structure.ridge_lambda or parent.bandwidth_spec != observation:
            raise ValueError("screened structure parent violates frozen scientific contract")
        support = canonical_support(parent.basis_ids)
        if support not in screening_scores:
            raise ValueError("screening score missing for active parent")
        score = screening_scores[support]
        if score.parent_support != support or set(score.candidate_ids) != set(dictionary_pool):
            raise ValueError("screening score pool/support mismatch")
        for entry in score.shortlist(screening_spec, dictionary):
            child = structure.candidate(observation, canonical_support((*support, entry.basis_id)))
            proposal = ScreenedChildProposal(
                child,
                support_id(support),
                entry.score,
                entry.rank,
                entry.proposal_source,
                entry.stratum,
            )
            existing = proposals.get(child.candidate_id)
            if existing is None or (
                proposal.parent_support_id,
                proposal.screening_rank,
            ) < (existing.parent_support_id, existing.screening_rank):
                proposals[child.candidate_id] = proposal
    return tuple(proposals[key] for key in sorted(proposals))


def backward_candidates(
    model: CandidateModelSpec,
    structure: StructureSearchSpec,
) -> tuple[CandidateModelSpec, ...]:
    """All valid deletions are exact-evaluation candidates; no screening here.

    The frozen x[n] seed defines every support in this task, so deleting it is
    not a valid structure-search move even when K is greater than one.
    """
    if model.ridge_lambda != structure.ridge_lambda:
        raise ValueError("backward parent violates frozen structure Ridge")
    candidates = []
    for basis in model.basis_ids:
        if basis == structure.seed_basis_id:
            continue
        candidates.append(
            structure.candidate(
                model.bandwidth_spec,
                canonical_support(tuple(item for item in model.basis_ids if item != basis)),
            )
        )
    return tuple(candidates)


def reduced_swap_parents(
    model: CandidateModelSpec,
    structure: StructureSearchSpec,
) -> tuple[CandidateModelSpec, ...]:
    """Reduced parents whose residuals drive screened swap additions."""
    return backward_candidates(model, structure)


def forward_children(
    parents: Iterable[CandidateModelSpec],
    dictionary_ids: Iterable[str],
    spec: StructureSearchSpec,
    observation: ObservationBandwidthSpec,
) -> tuple[CandidateModelSpec, ...]:
    """Deterministic, duplicate-free complete forward expansions (no shortlist)."""
    available = tuple(sorted(set(dictionary_ids)))
    if spec.seed_basis_id not in available:
        raise ValueError("fixed seed missing from dictionary")
    children = {}
    for parent in parents:
        if parent.ridge_lambda != spec.ridge_lambda or parent.bandwidth_spec != observation:
            raise ValueError("historical/multi-lambda parent rejected")
        if parent.k_selected >= spec.max_support_size:
            continue
        for basis_id in available:
            if basis_id not in parent.basis_ids:
                child = spec.candidate(
                    observation, canonical_support((*parent.basis_ids, basis_id))
                )
                children[support_id(child.basis_ids)] = child
    return tuple(children[key] for key in sorted(children))


def evaluate_support_ridge_grid(
    support: tuple[str, ...],
    observation: ObservationBandwidthSpec,
    ridge_values: tuple[float, ...],
) -> tuple[CandidateModelSpec, ...]:
    """Separate dense-phase API; the science grid must be explicitly supplied."""
    if not ridge_values:
        raise ValueError("dense Ridge grid has not been scientifically frozen")
    return tuple(CandidateModelSpec(observation, support, ridge) for ridge in ridge_values)


class CandidateScoreCache:
    """Exact-evaluation-fingerprint-scoped, write-once support scores.

    Search-mode and screening-policy fingerprints deliberately do not enter
    this key: a support evaluated exactly is reusable by screened/exhaustive
    policy variants when its exact model/evaluator fingerprint matches.
    """

    def __init__(self, root: Path, exact_evaluation_fingerprint: str, bandwidth_label: str):
        if len(exact_evaluation_fingerprint) != 64:
            raise ValueError("full SHA256 exact-evaluation fingerprint required")
        self.store = MetadataCache(root)
        self.fingerprint = exact_evaluation_fingerprint
        self.bandwidth_label = bandwidth_label

    def key(self, support: tuple[str, ...], ridge_lambda: float) -> CacheKey:
        return CacheKey(
            "candidate_score",
            None,
            None,
            canonical_support(support),
            self.bandwidth_label,
            self.fingerprint,
            ridge_lambda,
        )

    def load(self, support: tuple[str, ...], ridge_lambda: float) -> dict[str, int]:
        values = self.store.load(self.key(support, ridge_lambda))
        if values.shape != (len(SCORE_FIELDS),) or not np.all(np.equal(values, values.astype(int))):
            raise ValueError("invalid cached score payload")
        return dict(zip(SCORE_FIELDS, values.astype(int).tolist(), strict=True))

    def save(self, support: tuple[str, ...], ridge_lambda: float, score: dict[str, object]) -> None:
        values = np.array([score[field] for field in SCORE_FIELDS], dtype=np.int64)
        self.store.save(self.key(support, ridge_lambda), values)


__all__ = [
    "CandidateScoreCache",
    "SCORE_FIELDS",
    "ScreenedChildProposal",
    "backward_candidates",
    "evaluate_support_ridge_grid",
    "forward_children",
    "reduced_swap_parents",
    "screened_forward_proposals",
    "seed_candidate",
    "unused_candidate_ids",
]
