"""Synthetic-safe orchestration: 5B basis, observation, model, fingerprint, retrieval.

Segment ownership (Aend versus C2 and final-ILC indexing) remains with the
existing data_management/signal_segmentation shared loaders. Callers supply
already canonical 5B x/y segments; no MAT loading occurs in this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from behavior_fingerprint_ranking_consistency.shared.distance_matrix import (
    compute_cnmse_distance_matrix,
)
from behavior_modeling.shared.basis_function_selection.model_solver import fit_ridge
from behavior_modeling.shared.volterra_terms import VolterraTermSpec, build_basis_columns
from low_bandwidth_behavior_analysis.shared import LowBandwidthObservationOperator

from .shareability_types import CandidateModelSpec


@dataclass(frozen=True)
class StateModel:
    state_id: int
    side: str
    theta: np.ndarray
    candidate_id: str


@dataclass(frozen=True)
class RetrievalRanking:
    distance_matrix: np.ndarray
    ranking_matrix: np.ndarray
    top1_indices: np.ndarray
    true_state_rank: np.ndarray

    def top_k_indices(self, k: int) -> np.ndarray:
        if not 1 <= k <= self.ranking_matrix.shape[1]:
            raise ValueError("k outside ranking width")
        return self.ranking_matrix[:, :k].copy()


def observed_design_and_target(
    x_5b: np.ndarray,
    y_5b: np.ndarray,
    candidate: CandidateModelSpec,
    dictionary: Mapping[str, VolterraTermSpec],
    *,
    dmax: int = 4,
    operator: LowBandwidthObservationOperator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct nonlinear columns at 5B before applying the same operator to E/y."""

    x = np.asarray(x_5b, dtype=np.complex128)
    y = np.asarray(y_5b, dtype=np.complex128)
    if x.ndim != 1 or x.shape != y.shape or x.size <= dmax:
        raise ValueError("canonical 5B x/y must be aligned one-dimensional segments")
    phi_5b = build_basis_columns(x, candidate.basis_ids, dictionary, dmax=dmax)
    target_5b = y[dmax:]
    op = operator or LowBandwidthObservationOperator(candidate.bandwidth_spec.sample_rate)
    if op.spec != candidate.bandwidth_spec.sample_rate:
        raise ValueError("observation operator does not match bandwidth")
    return op.apply_pair(phi_5b, target_5b)


def fit_state_model(
    state_id: int,
    side: str,
    x_5b: np.ndarray,
    y_5b: np.ndarray,
    candidate: CandidateModelSpec,
    dictionary: Mapping[str, VolterraTermSpec],
    *,
    dmax: int = 4,
) -> StateModel:
    """Fit one presegmented Aend LUT or C2 Query with its own coefficient vector."""

    if side not in {"Aend", "C2"} or state_id < 0:
        raise ValueError("side must be Aend/C2 and state_id must be nonnegative")
    phi, target = observed_design_and_target(x_5b, y_5b, candidate, dictionary, dmax=dmax)
    fit = fit_ridge(phi, target, candidate.ridge_lambda)
    return StateModel(state_id, side, fit.theta, candidate.candidate_id)


def build_fingerprints(
    models: Sequence[StateModel],
    common_b_probe_5b: np.ndarray,
    candidate: CandidateModelSpec,
    dictionary: Mapping[str, VolterraTermSpec],
    *,
    side: str,
    dmax: int = 4,
) -> np.ndarray:
    """Evaluate every model on exactly the same observed Common-B design."""

    if not models or side not in {"Aend", "C2"}:
        raise ValueError("nonempty models and Aend/C2 side required")
    phi_5b = build_basis_columns(common_b_probe_5b, candidate.basis_ids, dictionary, dmax=dmax)
    op = LowBandwidthObservationOperator(candidate.bandwidth_spec.sample_rate)
    phi = op.apply_matrix(phi_5b)
    if {model.state_id for model in models} != set(range(len(models))):
        raise ValueError("models must cover canonical state IDs in 0..N-1")
    by_state = {model.state_id: model for model in models}
    if any(
        model.side != side or model.candidate_id != candidate.candidate_id
        or model.theta.shape != (candidate.k_selected,)
        for model in models
    ):
        raise ValueError("model side/support/shape mismatch")
    fingerprints = np.vstack([phi @ by_state[state_id].theta for state_id in range(len(models))])
    if not np.all(np.isfinite(fingerprints)):
        raise ValueError("nonfinite fingerprint")
    return np.asarray(fingerprints, dtype=np.complex128)


def build_lut_fingerprints(
    models: Sequence[StateModel], common_b_probe_5b: np.ndarray,
    candidate: CandidateModelSpec, dictionary: Mapping[str, VolterraTermSpec],
    *, dmax: int = 4,
) -> np.ndarray:
    return build_fingerprints(models, common_b_probe_5b, candidate, dictionary,
                              side="Aend", dmax=dmax)


def build_query_fingerprints(
    models: Sequence[StateModel], common_b_probe_5b: np.ndarray,
    candidate: CandidateModelSpec, dictionary: Mapping[str, VolterraTermSpec],
    *, dmax: int = 4,
) -> np.ndarray:
    return build_fingerprints(models, common_b_probe_5b, candidate, dictionary,
                              side="C2", dmax=dmax)


def retrieve(query: np.ndarray, lut: np.ndarray) -> RetrievalRanking:
    """Canonical Query→Candidate CNMSE, followed by stable distance-only ranking."""

    q, lut_array = np.asarray(query), np.asarray(lut)
    if q.ndim != 2 or q.shape != lut_array.shape or q.shape[0] < 2:
        raise ValueError("query/lut must be equal (N,L) matrices with N>=2")
    distance = compute_cnmse_distance_matrix(
        np.asarray(q, dtype=np.complex128), np.asarray(lut_array, dtype=np.complex128)
    )
    ids = np.arange(distance.shape[0])
    ranking = np.vstack([np.lexsort((ids, row)) for row in distance])
    true_rank = np.array([np.flatnonzero(row == i)[0] + 1 for i, row in enumerate(ranking)])
    return RetrievalRanking(distance, ranking, ranking[:, 0].copy(), true_rank)


__all__ = [
    "StateModel", "RetrievalRanking", "observed_design_and_target", "fit_state_model",
    "build_fingerprints", "build_lut_fingerprints", "build_query_fingerprints", "retrieve",
]
