"""Synthetic parity and deterministic-shortlist checks for SOMP screening."""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from behavior_modeling.shared.volterra_terms import build_candidate_dictionary
from low_bandwidth_behavior_analysis.shared import SampleRateSpec

from .shareability_persistence import (
    load_checkpoint,
    load_screening_artifact,
    save_checkpoint,
    save_screening_artifact,
)
from .shareability_screening import (
    ParentScreeningScores,
    build_shortlist,
    descriptor_stratum,
    normalized_residual_correlation_from_statistics,
    screen_candidate_pool,
)
from .shareability_search import (
    SearchConfig,
    make_search_state,
    record_screening_progress,
    remaining_screening_blocks,
)
from .shareability_structure import backward_candidates, screened_forward_proposals
from .shareability_sufficient_stats import ParentFactor, ParentSupportStats
from .shareability_types import (
    CandidateModelSpec,
    CandidateScreeningSpec,
    ObservationBandwidthSpec,
    RidgeGridSpec,
    StructureSearchSpec,
)


def test_statistics_score_matches_explicit_residual() -> None:
    rng = np.random.default_rng(710)
    parent_design = rng.normal(size=(97, 3)) + 1j * rng.normal(size=(97, 3))
    columns = rng.normal(size=(97, 5)) + 1j * rng.normal(size=(97, 5))
    target = rng.normal(size=97) + 1j * rng.normal(size=97)
    stats = ParentSupportStats.from_design(("a", "b", "c"), parent_design, target)
    factor = ParentFactor.build(stats, 1e-8)
    cross = parent_design.conj().T @ columns
    energy = np.einsum("nb,nb->b", columns.conj(), columns).real
    rhs = columns.conj().T @ target
    actual, diagnostics = normalized_residual_correlation_from_statistics(
        factor, cross, energy, rhs, float(np.vdot(target, target).real)
    )
    residual = target - parent_design @ factor.coefficients
    expected = np.abs(columns.conj().T @ residual) ** 2 / (
        energy * np.vdot(residual, residual).real
    )
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-13)
    assert diagnostics == {
        "zero_residual_count": 0,
        "near_zero_basis_energy_count": 0,
        "nan_count": 0,
        "inf_count": 0,
    }


def test_weighting_and_deterministic_exploration() -> None:
    dictionary = build_candidate_dictionary(orders=(1, 3), max_delay=4)
    ids = tuple(sorted(dictionary))
    scores = np.linspace(1.0, 0.0, len(ids))
    record = ParentScreeningScores(
        ("V1_u0",),
        ids,
        scores,
        scores * 0.8,
        scores * 1.2,
        tuple("" if item == "V1_u0" else "placeholder" for item in ids),
        {},
    )
    # Supply valid canonical strata because shortlist validates dictionary alignment.
    record.strata = tuple(descriptor_stratum(dictionary[item]) for item in ids)
    spec = CandidateScreeningSpec(shortlist_size=8, exploit_size=4, explore_size=4)
    first = build_shortlist(record, spec, dictionary)
    second = build_shortlist(record, spec, dictionary)
    assert first == second
    assert len(first) == 8
    assert len({entry.basis_id for entry in first}) == 8
    assert {entry.proposal_source for entry in first} == {"exploit", "exploration"}
    assert "V1_u0" not in {entry.basis_id for entry in first}


def test_multi_parent_and_partitioned_screening_are_identical() -> None:
    rng = np.random.default_rng(711)
    dictionary = build_candidate_dictionary(orders=(1,), max_delay=4)
    ids = ("V1_u0", "V1_u1", "V1_u2", "V1_u3", "V1_u4")
    arrays = {
        "x_a": rng.normal(size=(2, 12288)) + 1j * rng.normal(size=(2, 12288)),
        "y_a": rng.normal(size=(2, 12288)) + 1j * rng.normal(size=(2, 12288)),
        "x_c": rng.normal(size=(2, 7373)) + 1j * rng.normal(size=(2, 7373)),
        "y_c": rng.normal(size=(2, 7373)) + 1j * rng.normal(size=(2, 7373)),
    }
    parents = (("V1_u0",), ("V1_u1",))
    spec = CandidateScreeningSpec(shortlist_size=4, exploit_size=2, explore_size=2)
    shared = screen_candidate_pool(
        parents, ids, dictionary=dictionary, ridge_lambda=1e-8, spec=spec, block_size=2, **arrays
    )
    for parent in parents:
        alone = screen_candidate_pool(
            (parent,),
            ids,
            dictionary=dictionary,
            ridge_lambda=1e-8,
            spec=spec,
            block_size=2,
            **arrays,
        )[parent]
        np.testing.assert_allclose(shared[parent].scores, alone.scores, equal_nan=True)

    first = screen_candidate_pool(
        (("V1_u0",),), ids[:3], dictionary=dictionary, ridge_lambda=1e-8, spec=spec, **arrays
    )[("V1_u0",)]
    second = screen_candidate_pool(
        (("V1_u0",),), ids[3:], dictionary=dictionary, ridge_lambda=1e-8, spec=spec, **arrays
    )[("V1_u0",)]
    combined = {
        basis: score for basis, score in zip(first.candidate_ids, first.scores, strict=True)
    }
    combined.update(
        {basis: score for basis, score in zip(second.candidate_ids, second.scores, strict=True)}
    )
    for basis, score in zip(
        shared[("V1_u0",)].candidate_ids, shared[("V1_u0",)].scores, strict=True
    ):
        assert np.isnan(score) and np.isnan(combined[basis]) or np.isclose(score, combined[basis])


def test_screening_artifact_and_checkpoint_state_are_resumable() -> None:
    with TemporaryDirectory() as directory:
        root = Path(directory)
        metadata = {
            "exact_evaluation_fingerprint": "a" * 64,
            "search_policy_fingerprint": "b" * 64,
            "parent_support_id": "c" * 64,
        }
        save_screening_artifact(
            root,
            "k02_parent_test_scores",
            metadata,
            {"basis_ids": np.array(["b1", "b2"]), "scores": np.array([0.1, 0.2])},
        )
        loaded = load_screening_artifact(root, "k02_parent_test_scores", metadata)
        np.testing.assert_allclose(loaded["scores"], [0.1, 0.2])

        observation = ObservationBandwidthSpec("5B", SampleRateSpec(50))
        seed = CandidateModelSpec(observation, ("V1_u0",), 1e-8)
        state = make_search_state(
            seed,
            ("V1_u0", "V1_u1"),
            RidgeGridSpec((1e-8,)),
            SearchConfig(("screen_forward", "evaluate_forward", "complete"), 3, 10, 0),
        )
        record_screening_progress(
            state,
            ("V1_u0",),
            candidate_pool_hash="d" * 64,
            screening_spec_hash="e" * 64,
            total_blocks=5,
            completed_blocks=(0, 1),
            score_artifact_path="screening/k02_parent_test_scores.npz",
            shortlist=("V1_u1",),
        )
        checkpoint = root / "screened_checkpoint.json"
        save_checkpoint(state, checkpoint)
        resumed = load_checkpoint(checkpoint)
        assert resumed.screening == state.screening
        assert remaining_screening_blocks(
            resumed,
            ("V1_u0",),
            candidate_pool_hash="d" * 64,
            screening_spec_hash="e" * 64,
        ) == (2, 3, 4)


def test_screened_proposals_and_backward_preserve_fixed_seed() -> None:
    dictionary = build_candidate_dictionary(orders=(1,), max_delay=4)
    observation = ObservationBandwidthSpec("5B", SampleRateSpec(50))
    structure = StructureSearchSpec("V1_u0")
    parent = CandidateModelSpec(observation, ("V1_u0",), 1e-8)
    result = ParentScreeningScores(
        ("V1_u0",),
        tuple(sorted(dictionary)),
        np.array([np.nan, 0.9, 0.8, 0.7, 0.6]),
        np.array([np.nan, 0.9, 0.8, 0.7, 0.6]),
        np.array([np.nan, 0.9, 0.8, 0.7, 0.6]),
        tuple(descriptor_stratum(dictionary[basis]) for basis in sorted(dictionary)),
        {},
    )
    spec = CandidateScreeningSpec(shortlist_size=2, exploit_size=1, explore_size=1)
    proposals = screened_forward_proposals(
        (parent,),
        {("V1_u0",): result},
        spec,
        structure,
        observation,
        tuple(sorted(dictionary)),
        dictionary,
    )
    assert len(proposals) == 2
    assert all("parent_support_id" in proposal.trace_fields() for proposal in proposals)
    k2 = CandidateModelSpec(observation, ("V1_u0", "V1_u1"), 1e-8)
    backward = backward_candidates(k2, structure)
    assert len(backward) == 1 and backward[0].basis_ids == ("V1_u0",)
