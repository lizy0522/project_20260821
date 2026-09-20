"""Machine-readable per-candidate result schema, independent of search task."""

from __future__ import annotations

from typing import Any

from .shareability_objective import ShareabilityScore
from .shareability_types import CandidateModelSpec

RESULT_FIELDS = (
    "candidate_id", "bandwidth_label", "sample_rate_hz", "basis_ids",
    "K_selected", "ridge_lambda", "N_states", "N_shareable", "shareable_rate",
    "N_self", "self_rate", "N_top3_contains_true", "top3_contains_true_rate",
    "N_top5_contains_true", "top5_contains_true_rate", "N_top10_contains_true",
    "top10_contains_true_rate",
)


def candidate_result_row(
    candidate: CandidateModelSpec, score: ShareabilityScore
) -> dict[str, Any]:
    row = {
        "candidate_id": candidate.candidate_id,
        "bandwidth_label": candidate.bandwidth_spec.label,
        "sample_rate_hz": candidate.bandwidth_spec.sample_rate_hz,
        "basis_ids": list(candidate.basis_ids),
        "K_selected": candidate.k_selected,
        "ridge_lambda": candidate.ridge_lambda,
        "N_states": score.n_states,
        "N_shareable": score.n_shareable,
        "shareable_rate": score.shareable_rate,
        "N_self": score.n_self,
        "self_rate": score.self_rate,
        "N_top3_contains_true": score.n_top3_contains_true,
        "top3_contains_true_rate": score.top3_contains_true_rate,
        "N_top5_contains_true": score.n_top5_contains_true,
        "top5_contains_true_rate": score.top5_contains_true_rate,
        "N_top10_contains_true": score.n_top10_contains_true,
        "top10_contains_true_rate": score.top10_contains_true_rate,
    }
    if tuple(row) != RESULT_FIELDS:
        raise RuntimeError("candidate result schema changed")
    return row


__all__ = ["RESULT_FIELDS", "candidate_result_row"]
