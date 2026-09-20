"""Explicit (never collected as a formal search) 425-state K2 parity check."""

import numpy as np
from behavior_fingerprint_ranking_consistency.shared.distance_matrix import (
    compute_cnmse_distance_matrix,
)
from behavior_modeling.shared.basis_function_selection.model_solver import fit_ridge
from behavior_modeling.shared.volterra_terms import build_basis_columns

from retrieval_oriented_model_selection.shared.shareability_optimized_evaluator import (
    evaluate_fixed_seed,
    evaluate_partition,
    rank_and_score,
)
from retrieval_oriented_model_selection.shared.shareability_screening import (
    normalized_residual_correlation_from_statistics,
)
from retrieval_oriented_model_selection.shared.shareability_sufficient_stats import (
    ParentFactor,
    ParentSupportStats,
    coefficient_distance,
)

from .run_search import DATASET_FILES, validated_science_fingerprint
from .task_config import DICTIONARY, REAL_B_SOURCE, SEED_TERM, STRUCTURE


def test_real_k2_425_states() -> None:
    check_real_support(2)


def test_real_fixed_seed_425_states() -> None:
    validated_science_fingerprint()
    x_a, y_a, x_c, y_c, common_b = (
        np.load(DATASET_FILES[name], mmap_mode="r")
        for name in ("x_aend", "y_aend", "x_c2", "y_c2", "common_b")
    )
    real_b = np.load(REAL_B_SOURCE, mmap_mode="r")
    optimized = evaluate_fixed_seed(
        SEED_TERM.basis_id,
        x_a=x_a,
        y_a=y_a,
        x_c=x_c,
        y_c=y_c,
        common_b=common_b,
        real_b=real_b,
        dictionary=DICTIONARY,
        include_details=True,
    )
    ids = (SEED_TERM.basis_id,)
    theta_a = np.empty((425, 1), complex)
    theta_c = np.empty_like(theta_a)
    for i in range(425):
        pa = build_basis_columns(x_a[i], ids, DICTIONARY, dmax=4)
        pc = build_basis_columns(x_c[i], ids, DICTIONARY, dmax=4)
        theta_a[i] = fit_ridge(pa, y_a[i, 4:], 1e-8).theta
        theta_c[i] = fit_ridge(pc, y_c[i, 4:], 1e-8).theta
    reference = rank_and_score(
        theta_a,
        theta_c,
        build_basis_columns(common_b, ids, DICTIONARY, dmax=4),
        real_b,
        include_details=True,
    )
    for field in (
        "N_shareable",
        "N_self",
        "N_top3",
        "N_top5",
        "N_top10",
        "Q1",
        "top10",
        "true_state_rank",
    ):
        assert optimized[field] == reference[field], field
    finite = np.isfinite(reference["distance"])
    np.testing.assert_allclose(
        optimized["distance"][finite], reference["distance"][finite], atol=1e-5
    )


def test_real_somp_statistics_sample() -> None:
    """Check direct residual and sufficient-statistics SOMP on real 5B segments."""
    validated_science_fingerprint()
    x_a, y_a, x_c, y_c = (
        np.load(DATASET_FILES[name], mmap_mode="r")
        for name in ("x_aend", "y_aend", "x_c2", "y_c2")
    )
    candidates = tuple(
        next(
            basis
            for basis in sorted(DICTIONARY)
            if basis != SEED_TERM.basis_id and DICTIONARY[basis].nonlinear_order == order
        )
        for order in (1, 3, 9, 11)
    )
    for state in (0, 42, 424):
        for x, y in ((x_a[state], y_a[state]), (x_c[state], y_c[state])):
            parent = build_basis_columns(x, (SEED_TERM.basis_id,), DICTIONARY, dmax=4)
            target = y[4:]
            factor = ParentFactor.build(
                ParentSupportStats.from_design((SEED_TERM.basis_id,), parent, target), 1e-8
            )
            columns = build_basis_columns(x, candidates, DICTIONARY, dmax=4)
            actual, diagnostics = normalized_residual_correlation_from_statistics(
                factor,
                parent.conj().T @ columns,
                np.einsum("nb,nb->b", columns.conj(), columns).real,
                columns.conj().T @ target,
                float(np.vdot(target, target).real),
            )
            residual = target - parent @ factor.coefficients
            expected = np.abs(columns.conj().T @ residual) ** 2 / (
                np.einsum("nb,nb->b", columns.conj(), columns).real
                * float(np.vdot(residual, residual).real)
            )
            np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-13)
            assert not any(diagnostics[key] for key in ("nan_count", "inf_count"))


def check_real_support(k: int) -> dict[str, int]:
    """Full 425-state comparison for one deterministic K; no search/selection."""
    if k not in (2, 5, 10, 20):
        raise ValueError("unexpected regression support size")
    validated_science_fingerprint()
    x_a, y_a, x_c, y_c, common_b = (
        np.load(DATASET_FILES[name], mmap_mode="r")
        for name in ("x_aend", "y_aend", "x_c2", "y_c2", "common_b")
    )
    real_b = np.load(REAL_B_SOURCE, mmap_mode="r")
    remaining = sorted(
        (basis for basis in DICTIONARY if basis != SEED_TERM.basis_id),
        key=lambda basis: (DICTIONARY[basis].nonlinear_order, basis),
    )
    parent = tuple(sorted((SEED_TERM.basis_id, *remaining[: k - 2])))
    added = remaining[k - 2]
    child = tuple(sorted((*parent, added)))
    optimized, diagnostics = evaluate_partition(
        (parent,),
        (added,),
        x_a=x_a,
        y_a=y_a,
        x_c=x_c,
        y_c=y_c,
        common_b=common_b,
        real_b=real_b,
        dictionary=DICTIONARY,
        ridge_lambda=STRUCTURE.ridge_lambda,
        include_details=True,
    )
    theta_a = np.empty((425, k), complex)
    theta_c = np.empty((425, k), complex)
    for i in range(425):
        pa = build_basis_columns(x_a[i], child, DICTIONARY, dmax=4)
        pc = build_basis_columns(x_c[i], child, DICTIONARY, dmax=4)
        theta_a[i] = fit_ridge(pa, y_a[i, 4:], STRUCTURE.ridge_lambda).theta
        theta_c[i] = fit_ridge(pc, y_c[i, 4:], STRUCTURE.ridge_lambda).theta
    probe = build_basis_columns(common_b, child, DICTIONARY, dmax=4)
    reference = rank_and_score(theta_a, theta_c, probe, real_b, include_details=True)
    for field in (
        "N_shareable",
        "N_self",
        "N_top3",
        "N_top5",
        "N_top10",
        "Q1",
        "top10",
        "true_state_rank",
    ):
        assert optimized[child][field] == reference[field], field
    fast_distance = optimized[child]["distance"]
    reference_distance = reference["distance"]
    np.testing.assert_array_equal(np.isneginf(fast_distance), np.isneginf(reference_distance))
    finite = np.isfinite(reference_distance)
    diagnostics["waveform_max_distance_error_db"] = float(
        np.max(np.abs(fast_distance[finite] - reference_distance[finite]))
    )
    np.testing.assert_allclose(fast_distance[finite], reference_distance[finite], atol=1e-5)
    if k == 2:
        waveform = compute_cnmse_distance_matrix((probe @ theta_c.T).T, (probe @ theta_a.T).T)
        coefficient = coefficient_distance(theta_c, theta_a, probe.conj().T @ probe)
        finite = np.isfinite(waveform) & np.isfinite(coefficient)
        max_error = float(np.max(np.abs(waveform[finite] - coefficient[finite])))
        ids = np.arange(425)
        q1_wave = np.array([np.lexsort((ids, row))[0] for row in waveform])
        q1_coeff = np.array([np.lexsort((ids, row))[0] for row in coefficient])
        diagnostics["coefficient_max_distance_error_db"] = max_error
        diagnostics["coefficient_q1_mismatch"] = int(np.count_nonzero(q1_wave != q1_coeff))
    assert diagnostics["generated_columns"] == 850
    return diagnostics
