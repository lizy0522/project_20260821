"""Small synthetic contract tests: no real MAT data or 425-state search."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
from behavior_modeling.shared.basis_function_selection.volterra_dictionary import (
    build_basis_bank as build_frozen_basis_bank,
)
from behavior_modeling.shared.basis_function_selection.volterra_dictionary import (
    build_dictionary as build_frozen_dictionary,
)
from behavior_modeling.shared.volterra_terms import (
    VolterraTermSpec,
    build_basis_columns,
    build_candidate_dictionary,
    canonical_terms,
)
from low_bandwidth_behavior_analysis.shared import LowBandwidthObservationOperator, SampleRateSpec

from retrieval_oriented_model_selection.shared.shareability_objective import (
    RealBShareabilityOracle,
    score_ranking,
)
from retrieval_oriented_model_selection.shared.shareability_parallel import SpawnEvaluator
from retrieval_oriented_model_selection.shared.shareability_persistence import (
    CacheKey,
    MetadataCache,
    load_checkpoint,
    save_checkpoint,
)
from retrieval_oriented_model_selection.shared.shareability_pipeline import (
    build_lut_fingerprints,
    build_query_fingerprints,
    fit_state_model,
    observed_design_and_target,
    retrieve,
)
from retrieval_oriented_model_selection.shared.shareability_schema import (
    RESULT_FIELDS,
    candidate_result_row,
)
from retrieval_oriented_model_selection.shared.shareability_search import (
    SearchConfig,
    make_search_state,
    neighbors,
    run_shareability_search,
)
from retrieval_oriented_model_selection.shared.shareability_types import (
    CandidateModelSpec,
    ObservationBandwidthSpec,
    RidgeGridSpec,
)


def _synthetic_score(candidate: CandidateModelSpec) -> int:
    return candidate.k_selected


class InfrastructureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.bandwidth = ObservationBandwidthSpec("synthetic_5b", SampleRateSpec(50))
        self.dictionary = build_candidate_dictionary(orders=(1, 3), max_delay=4)
        self.linear = self.dictionary["V1_u0"]

    def test_volterra_canonical_permutations_and_legacy_convention(self) -> None:
        a = VolterraTermSpec.canonical((1, 0), (0,))
        b = VolterraTermSpec.canonical((0, 1), (0,))
        self.assertEqual(a.basis_id, b.basis_id)
        self.assertEqual(len(canonical_terms((a, b))), 1)
        self.assertTrue(VolterraTermSpec.canonical((2, 2), (2,)).is_diagonal)
        self.assertTrue(a.is_cross_memory)
        self.assertEqual(a.conjugate_pattern, (False, False, True))
        self.assertTrue(all(term.nonlinear_order <= 11 and term.max_delay <= 4
                            for term in self.dictionary.values()))
        self.assertEqual(
            VolterraTermSpec.canonical((0, 0), (0,)).basis_id,
            "V3_u00_u00_c00",
        )

    def test_independent_memory_and_lazy_columns(self) -> None:
        terms = build_candidate_dictionary(
            orders=(3, 5, 7), max_delay=4,
            memory_by_order={3: 4, 5: 1, 7: 3},
        )
        self.assertEqual({p: max(x.max_delay for x in terms.values()
                                 if x.nonlinear_order == p) for p in (3, 5, 7)},
                         {3: 4, 5: 1, 7: 3})
        x = np.array([1 + 1j, 2 + 0j, 3 + 2j, 4 + 1j, 5 + 2j, 6 + 0j])
        phi = build_basis_columns(x, ("V3_u00_u00_c00",), self.dictionary, dmax=4)
        np.testing.assert_allclose(phi[:, 0], x[4:] * np.abs(x[4:]) ** 2)
        self.assertEqual(phi.shape, (2, 1))

    def test_frozen_volterra_numerical_regression_and_order11(self) -> None:
        frozen = build_frozen_dictionary()
        dictionary = build_candidate_dictionary(orders=(1, 3, 5), max_delay=2)
        rng = np.random.default_rng(17)
        x = rng.normal(size=48) + 1j * rng.normal(size=48)
        old = build_frozen_basis_bank(x, frozen)
        new = build_basis_columns(x, tuple(term.basis_id for term in frozen),
                                  dictionary, dmax=2)
        np.testing.assert_allclose(new, old, rtol=0, atol=0)
        order11 = build_candidate_dictionary(
            orders=(11,), max_delay=4, memory_by_order={11: 0}
        )
        self.assertEqual(len(order11), 1)
        self.assertEqual(next(iter(order11.values())).nonlinear_order, 11)

    def test_support_and_candidate_identity(self) -> None:
        ids = tuple(f"id_{i:02d}" for i in range(21))
        candidate = CandidateModelSpec(self.bandwidth, ids[:20], 1e-8)
        self.assertEqual(candidate.k_selected, 20)
        with self.assertRaises(ValueError):
            CandidateModelSpec(self.bandwidth, ids, 1e-8)
        same = CandidateModelSpec(self.bandwidth, tuple(reversed(ids[:20])), 1e-8)
        self.assertEqual(candidate.candidate_id, same.candidate_id)
        self.assertEqual(candidate, CandidateModelSpec.from_dict(candidate.to_dict()))
        another_band = ObservationBandwidthSpec("synthetic_1b", SampleRateSpec(10))
        self.assertNotEqual(
            candidate.candidate_id,
            CandidateModelSpec(another_band, candidate.basis_ids, 1e-8).candidate_id,
        )

    def test_5b_basis_precedes_observation(self) -> None:
        candidate = CandidateModelSpec(self.bandwidth, (self.linear.basis_id,), 0.0)
        x = np.arange(16, dtype=float).astype(complex) + 1j
        y = (2 * x).copy()

        class Spy:
            spec = SampleRateSpec(50)

            def apply_pair(
                self, e_5b: np.ndarray, y_5b: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray]:
                np.testing.assert_allclose(e_5b[:, 0], x[4:])
                np.testing.assert_allclose(y_5b, y[4:])
                return e_5b, y_5b

        phi, target = observed_design_and_target(
            x, y, candidate, self.dictionary, operator=Spy()
        )
        self.assertEqual(phi.shape, (12, 1))
        self.assertEqual(target.shape, (12,))
        low = CandidateModelSpec(ObservationBandwidthSpec("synthetic_0p5b", SampleRateSpec(5)),
                                 (self.linear.basis_id,), 0.0)
        observed, target_low = observed_design_and_target(x, y, low, self.dictionary)
        op = LowBandwidthObservationOperator(SampleRateSpec(5))
        np.testing.assert_allclose(observed, op.apply_matrix(phi))
        np.testing.assert_allclose(target_low, op.apply_vector(target))

    def test_fit_independent_models_and_common_probe(self) -> None:
        candidate = CandidateModelSpec(self.bandwidth, (self.linear.basis_id,), 0.0)
        x = np.linspace(1, 2, 40).astype(complex) + 0.1j
        aend = [fit_state_model(i, "Aend", x, (i + 1) * x, candidate, self.dictionary)
                for i in range(3)]
        c2 = [fit_state_model(i, "C2", x, (i + 1) * x, candidate, self.dictionary)
              for i in range(3)]
        self.assertNotEqual(aend[0].theta[0], aend[1].theta[0])
        lut = build_lut_fingerprints(aend, x, candidate, self.dictionary)
        query = build_query_fingerprints(c2, x, candidate, self.dictionary)
        self.assertEqual(lut.shape, (3, 36))
        np.testing.assert_allclose(lut, query)

    def test_retrieval_oracle_isolation_and_topk_true_state(self) -> None:
        lut = np.eye(6, dtype=np.complex128)
        query = np.roll(lut, 1, axis=0)
        ranking = retrieve(query, lut)
        self.assertEqual(ranking.distance_matrix.shape, (6, 6))
        self.assertEqual(ranking.ranking_matrix.shape, (6, 6))
        np.testing.assert_array_equal(ranking.top_k_indices(3), ranking.ranking_matrix[:, :3])
        initial = np.full((6, 6), -39.0)
        np.fill_diagonal(initial, -np.inf)
        score_a = score_ranking(ranking, RealBShareabilityOracle(initial))
        changed = initial.copy()
        changed[np.arange(6), ranking.top1_indices] = -41.0
        score_b = score_ranking(ranking, RealBShareabilityOracle(changed))
        self.assertEqual(score_a.n_shareable, 0)
        self.assertEqual(score_b.n_shareable, 6)
        self.assertEqual(score_a.n_self, score_b.n_self)
        self.assertEqual(score_a.n_top3_contains_true, score_b.n_top3_contains_true)
        self.assertEqual(score_b.n_top5_contains_true,
                         int(np.count_nonzero(ranking.true_state_rank <= 5)))
        self.assertEqual(score_b.n_top10_contains_true, 6)
        np.testing.assert_array_equal(ranking.top1_indices, retrieve(query, lut).top1_indices)
        row = candidate_result_row(
            CandidateModelSpec(self.bandwidth, (self.linear.basis_id,), 0.0), score_b
        )
        self.assertEqual(tuple(row), RESULT_FIELDS)

    def test_strict_threshold_and_oracle_roundtrip(self) -> None:
        values = np.array([[-np.inf, -40.0001, -40.0],
                           [-39.9999, -np.inf, -39.0],
                           [-41.0, -40.0, -np.inf]])
        oracle = RealBShareabilityOracle(values)
        np.testing.assert_array_equal(oracle.shareable_mask[0], (True, True, False))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "oracle.npz"
            oracle.save(path)
            np.testing.assert_array_equal(RealBShareabilityOracle.load(path).matrix, values)

    def test_neighbors_ridge_ties_checkpoint_and_cache(self) -> None:
        ids = ("a", "b", "c")
        seed = CandidateModelSpec(self.bandwidth, ("a",), 0.0)
        grid = RidgeGridSpec((0.0, 1e-8), (1e-6,))
        refined = RidgeGridSpec.with_log_refinement((0.0,), lower=1e-9,
                                                     upper=1e-6, count=4)
        self.assertEqual(len(refined.values), 5)
        self.assertEqual(len(neighbors(seed, "forward", ids, grid)), 2)
        self.assertEqual(len(neighbors(seed, "ridge", ids, grid)), 2)
        self.assertEqual(len(neighbors(CandidateModelSpec(self.bandwidth, ("a", "b"), 0.0),
                                       "swap", ids, grid)), 2)
        config = SearchConfig(("forward", "ridge", "backward", "swap"), 1, 30, 42)

        def evaluator(candidate: CandidateModelSpec) -> int:
            return len(candidate.basis_ids)  # Synthetic score; ties must be kept.

        one_shot = run_shareability_search(make_search_state(seed, ids, grid, config),
                                           ids, grid, config, evaluator)
        first_stage = run_shareability_search(
            make_search_state(seed, ids, grid, SearchConfig(("forward",), 1, 30, 42)),
            ids, grid, SearchConfig(("forward",), 1, 30, 42), evaluator,
        )
        self.assertEqual(len(first_stage.state.frontier), 2)
        parallel_adapter = run_shareability_search(
            make_search_state(seed, ids, grid, SearchConfig(("forward",), 1, 30, 42)),
            ids, grid, SearchConfig(("forward",), 1, 30, 42), evaluator,
            evaluate_many=lambda batch: [evaluator(item) for item in batch],
        )
        self.assertEqual(parallel_adapter.state.scores, first_stage.state.scores)
        partial = run_shareability_search(make_search_state(seed, ids, grid, config),
                                          ids, grid, config, evaluator, max_this_call=1)
        self.assertFalse(partial.complete)
        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "checkpoint.json"
            save_checkpoint(partial.state, checkpoint)
            resumed = run_shareability_search(load_checkpoint(checkpoint), ids, grid,
                                              config, evaluator)
            self.assertEqual(one_shot.state.to_dict(), resumed.state.to_dict())
            self.assertEqual(one_shot.tied_primary_ids, resumed.tied_primary_ids)
            cache = MetadataCache(Path(tmp) / "cache")
            key = CacheKey("basis_column", 2, "Aend", ("b", "a"), "5B", "seg-v1")
            cache.save(key, np.array([1.0, 2.0]))
            np.testing.assert_array_equal(cache.load(key), (1.0, 2.0))
            (Path(tmp) / "cache" / f"{key.digest}.npy").write_bytes(b"corrupted")
            with self.assertRaises(ValueError):
                cache.load(key)
            with self.assertRaises(ValueError):
                CacheKey("basis_column", 2, "Aend", ("a",), "5B", "seg-v1", 1e-8)
        with self.assertRaises(ValueError):
            run_shareability_search(
                make_search_state(seed, ids, grid, config), ("a", "b"), grid,
                config, evaluator,
            )

    def test_one_worker_serial_only_on_synthetic_candidate(self) -> None:
        candidate = CandidateModelSpec(self.bandwidth, (self.linear.basis_id,), 0.0)
        with SpawnEvaluator(_synthetic_score, workers=1) as pool:
            self.assertEqual(pool.map((candidate,)), [1])
            self.assertEqual(pool.runtime_metadata["effective_workers"], 1)


if __name__ == "__main__":
    unittest.main()
