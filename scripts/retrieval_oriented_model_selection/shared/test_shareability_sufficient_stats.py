"""Deterministic synthetic parity for the active-parent incremental equations."""

from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from behavior_fingerprint_ranking_consistency.shared.distance_matrix import (
    compute_cnmse_distance_matrix,
)
from behavior_modeling.shared.basis_function_selection.model_solver import fit_ridge
from behavior_modeling.shared.volterra_terms import build_basis_columns, build_candidate_dictionary

from retrieval_oriented_model_selection.shared.shareability_sufficient_stats import (
    ParentFactor,
    ParentSupportStats,
    canonical_support,
    coefficient_distance,
    support_id,
)


def test_incremental_fits_and_common_probe() -> None:
    for k in (1, 2, 5, 10, 20):
        _exercise_support_size(k)


def _exercise_support_size(k: int) -> None:
    rng = np.random.default_rng(100 + k)
    phi = rng.normal(size=(128, k + 2)) + 1j * rng.normal(size=(128, k + 2))
    target = rng.normal(size=128) + 1j * rng.normal(size=128)
    ids = tuple(f"b{i:02d}" for i in range(k))
    parent = ParentSupportStats.from_design(ids, phi[:, :k], target)
    fit = ParentFactor.build(parent, 1e-8)
    children, stats, fallback = fit.expand_block(
        ("extra1", "extra2"), phi[:, :k], phi[:, k:], target
    )
    assert not fallback
    for i in range(2):
        full = phi[:, [*range(k), k + i]]
        direct = fit_ridge(full, target, 1e-8).theta
        np.testing.assert_allclose(stats[i].gram, full.conj().T @ full, rtol=1e-12, atol=1e-10)
        np.testing.assert_allclose(stats[i].rhs, full.conj().T @ target, rtol=1e-12, atol=1e-10)
        np.testing.assert_allclose(children[i], direct, rtol=1e-6, atol=1e-6)
        reduced = stats[i].without("extra1" if i == 0 else "extra2")
        np.testing.assert_allclose(reduced.gram, parent.gram)

    e = rng.normal(size=(75, k + 1)) + 1j * rng.normal(size=(75, k + 1))
    gram_b = e[:, :k].conj().T @ e[:, :k]
    cross_b = e[:, :k].conj().T @ e[:, k]
    energy_b = float(np.vdot(e[:, k], e[:, k]).real)
    b_stats = ParentSupportStats(ids, gram_b, np.zeros(k, complex), e.shape[0])
    increment = b_stats.append("probe", cross_b, energy_b, 0j)
    np.testing.assert_allclose(increment.gram, e.conj().T @ e)

    wq = rng.normal(size=(7, k + 1)) + 1j * rng.normal(size=(7, k + 1))
    wl = rng.normal(size=(8, k + 1)) + 1j * rng.normal(size=(8, k + 1))
    fast = coefficient_distance(wq, wl, increment.gram)
    slow = compute_cnmse_distance_matrix(wq @ e.T, wl @ e.T)
    np.testing.assert_allclose(fast, slow, atol=2e-10)


def test_local_history_and_fixed_effective_lengths() -> None:
    dictionary = build_candidate_dictionary(orders=(1,), max_delay=4)
    x = np.arange(1, 13, dtype=float).astype(complex)
    for n in (12288, 4915, 7373):
        segment = np.resize(x, n)
        phi = build_basis_columns(segment, ("V1_u4",), dictionary, dmax=4)
        assert phi.shape == (n - 4, 1)
        assert phi[0, 0] == segment[0]
    a = np.full(12288, 10 + 2j)
    b = np.full(4915, 20 + 3j)
    c = np.full(7373, 30 + 4j)
    assert build_basis_columns(b, ("V1_u4",), dictionary, dmax=4)[0, 0] == b[0] != a[-1]
    assert build_basis_columns(c, ("V1_u4",), dictionary, dmax=4)[0, 0] == c[0] != b[-1]


def test_canonical_support_id() -> None:
    assert canonical_support(["z", "a"]) == ("a", "z")
    assert support_id(["z", "a"]) == support_id(["a", "z"])


def test_score_cache_science_isolation() -> None:
    from .shareability_structure import CandidateScoreCache

    with TemporaryDirectory() as directory:
        cache = CandidateScoreCache(Path(directory), "a" * 64, "5B")
        score = {name: 2 for name in ("N_shareable", "N_self", "N_top3", "N_top5", "N_top10")}
        cache.save(("x", "z"), 1e-8, score)
        assert cache.load(("z", "x"), 1e-8) == score
        different = CandidateScoreCache(Path(directory), "b" * 64, "5B")
        assert not (Path(directory) / f"{different.key(('x', 'z'), 1e-8).digest}.json").exists()
