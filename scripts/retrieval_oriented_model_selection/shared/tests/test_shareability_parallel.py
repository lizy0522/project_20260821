"""Synthetic-only tests for candidate-level serial and 10-worker execution."""

from __future__ import annotations

import json
import multiprocessing
import os
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from low_bandwidth_behavior_analysis.shared import SampleRateSpec

from retrieval_oriented_model_selection.shared.shareability_objective import (
    RealBShareabilityOracle,
    score_ranking,
)
from retrieval_oriented_model_selection.shared.shareability_parallel import (
    CandidateEvaluationError,
    SpawnEvaluator,
    WorkerReferences,
    get_worker_cache,
    get_worker_oracle,
    worker_active,
)
from retrieval_oriented_model_selection.shared.shareability_persistence import (
    CacheKey,
    load_checkpoint,
    save_checkpoint,
)
from retrieval_oriented_model_selection.shared.shareability_pipeline import retrieve
from retrieval_oriented_model_selection.shared.shareability_schema import candidate_result_row
from retrieval_oriented_model_selection.shared.shareability_search import (
    SearchConfig,
    make_search_state,
    run_shareability_search,
)
from retrieval_oriented_model_selection.shared.shareability_types import (
    CandidateModelSpec,
    ObservationBandwidthSpec,
    ParallelExecutionSpec,
    RidgeGridSpec,
)


def _candidate(index: int) -> CandidateModelSpec:
    bandwidth = ObservationBandwidthSpec("synthetic_5b", SampleRateSpec(50))
    return CandidateModelSpec(bandwidth, (f"basis_{index:02d}",), 0.0)


def _probe(candidate: CandidateModelSpec) -> dict[str, object]:
    time.sleep(0.005 * (20 - int(candidate.basis_ids[0][-2:])))
    return {
        "candidate_id": candidate.candidate_id,
        "N_shareable": candidate.k_selected,
        "environment": {
            name: os.environ.get(name)
            for name in (
                "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
            )
        },
        "worker_active": worker_active(),
    }


def _scientific_fixture(candidate: CandidateModelSpec) -> dict[str, object]:
    """Six synthetic states only; no real 425-state model/search."""

    offset = int(candidate.basis_ids[-1][-2:]) % 6
    lut = np.eye(6, dtype=np.complex128)
    query = np.roll(lut, offset, axis=0)
    ranking = retrieve(query, lut)
    matrix = np.full((6, 6), -39.0)
    np.fill_diagonal(matrix, -np.inf)
    for real_id in range(3):
        selected_id = ranking.top1_indices[real_id]
        if selected_id != real_id:
            matrix[real_id, selected_id] = -41.0
    score = score_ranking(ranking, RealBShareabilityOracle(matrix))
    row = candidate_result_row(candidate, score)
    # Tiny test-only evidence of distance/rank/Q1 equivalence; formal workers
    # return only small score records, never actual 425x425 arrays.
    row["test_distance_matrix"] = ranking.distance_matrix.tolist()
    row["test_ranking_matrix"] = ranking.ranking_matrix.tolist()
    row["test_Q1"] = ranking.top1_indices.tolist()
    return row


def _nested_attempt(candidate: CandidateModelSpec) -> int:
    try:
        with SpawnEvaluator(_probe, workers=2):
            pass
    except RuntimeError as error:
        if "nested" in str(error):
            return 1
    raise AssertionError("nested pool unexpectedly permitted")


def _fail_one(candidate: CandidateModelSpec) -> int:
    if "bad" in candidate.basis_ids:
        raise ValueError("synthetic candidate failure")
    return 1


def _cache_and_oracle(candidate: CandidateModelSpec) -> int:
    oracle = get_worker_oracle()
    if oracle is not get_worker_oracle():
        raise AssertionError("oracle was reloaded within one worker")
    cache = get_worker_cache()
    key = CacheKey("basis_column", 0, "Aend", ("phi0",), "5B", "synthetic-segment")

    def build() -> np.ndarray:
        marker = cache.root / "builder_executed_once"
        with marker.open("x", encoding="utf-8") as handle:
            handle.write("one build\n")
        return np.array([1.0, 2.0])

    values = cache.get_or_compute(key, build)
    if not np.array_equal(values, [1.0, 2.0]):
        raise AssertionError("cache corruption")
    return int(oracle.shareable_mask.shape[0])


def _simple_score(candidate: CandidateModelSpec) -> int:
    return candidate.k_selected


def _delayed_score(candidate: CandidateModelSpec) -> int:
    time.sleep(0.02 if "d" in candidate.basis_ids else 0.005)
    return candidate.k_selected


class ParallelPolicyTests(unittest.TestCase):
    def test_defaults_validation_and_identity_independence(self) -> None:
        spec = ParallelExecutionSpec()
        self.assertEqual(spec.to_dict(effective_workers=0), {
            "requested_workers": 10, "effective_workers": 0, "start_method": "spawn",
            "blas_threads_per_worker": 1, "cpu_affinity": None, "gpu_enabled": False,
            "nested_parallelism": False,
        })
        for kwargs in (
            {"max_workers": 0}, {"start_method": "fork"}, {"cpu_affinity": (0,)},
            {"gpu_enabled": True}, {"allow_nested_parallelism": True},
        ):
            with self.assertRaises(ValueError):
                ParallelExecutionSpec(**kwargs)
        candidate = _candidate(0)
        self.assertEqual(candidate.candidate_id, _candidate(0).candidate_id)
        before = {process.pid for process in multiprocessing.active_children()}
        with SpawnEvaluator(_simple_score, workers=1) as serial:
            self.assertEqual(serial.map((candidate,)), [1])
            self.assertEqual(serial.runtime_metadata["effective_workers"], 1)
        self.assertEqual(before, {process.pid for process in multiprocessing.active_children()})
        with SpawnEvaluator(_simple_score) as parallel:
            self.assertEqual(parallel.spec.max_workers, 10)
            self.assertEqual(candidate.candidate_id, _candidate(0).candidate_id)

    def test_ten_spawned_workers_environment_and_reordering(self) -> None:
        candidates = [_candidate(index) for index in range(20)]
        names = (
            "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
        )
        original = {name: os.environ.get(name) for name in names}
        with SpawnEvaluator(_probe) as executor:
            results = executor.map_scored(candidates)
            metadata = executor.runtime_metadata
        self.assertEqual(original, {name: os.environ.get(name) for name in names})
        self.assertEqual([item.candidate_id for item in results],
                         [item.candidate_id for item in candidates])
        self.assertEqual(len(results), 20)
        self.assertGreater(len({item.worker_pid for item in results}), 1)
        self.assertLessEqual(len({item.worker_pid for item in results}), 10)
        self.assertEqual(metadata["effective_workers"], len({item.worker_pid for item in results}))
        self.assertEqual(metadata["requested_workers"], 10)
        self.assertTrue(all(item.metrics["worker_active"] is True for item in results))
        self.assertTrue(all(set(item.metrics["environment"].values()) == {"1"}
                            for item in results))

    def test_synthetic_serial_parallel_scientific_equivalence(self) -> None:
        candidates = [_candidate(index) for index in range(12)]
        with SpawnEvaluator(_scientific_fixture, workers=1) as serial:
            expected = serial.map_scored(candidates)
        with SpawnEvaluator(_scientific_fixture) as parallel:
            actual = parallel.map_scored(candidates)
        self.assertEqual([row.candidate_id for row in actual],
                         [row.candidate_id for row in expected])
        for one, many in zip(expected, actual, strict=True):
            np.testing.assert_allclose(
                one.metrics["test_distance_matrix"], many.metrics["test_distance_matrix"],
                rtol=0, atol=1e-12,
            )
            self.assertEqual(one.metrics, many.metrics)
            self.assertEqual(one.n_shareable, many.n_shareable)

    def test_nested_pool_rejection_and_worker_exception(self) -> None:
        with SpawnEvaluator(_nested_attempt, workers=2) as executor:
            self.assertEqual(executor.map((_candidate(0), _candidate(1))), [1, 1])
        bad = CandidateModelSpec(_candidate(0).bandwidth_spec, ("bad",), 0.0)
        with self.assertRaises(CandidateEvaluationError) as caught:
            with SpawnEvaluator(_fail_one, workers=2) as executor:
                executor.map_scored((_candidate(0), bad, _candidate(1)))
        self.assertEqual(caught.exception.result.candidate_id, bad.candidate_id)
        self.assertEqual(caught.exception.result.exception_type, "ValueError")
        self.assertIn("synthetic candidate failure", caught.exception.result.traceback_summary)

    def test_checkpoint_resume_across_worker_counts(self) -> None:
        bandwidth = _candidate(0).bandwidth_spec
        seed = CandidateModelSpec(bandwidth, ("a",), 0.0)
        ids = ("a", "b", "c", "d")
        grid = RidgeGridSpec((0.0, 1e-8))
        config = SearchConfig(("forward", "ridge"), 1, 30, 9)
        one_shot = run_shareability_search(
            make_search_state(seed, ids, grid, config), ids, grid, config, _simple_score
        )
        with tempfile.TemporaryDirectory() as tmp:
            with SpawnEvaluator(_simple_score, workers=2) as first:
                partial = run_shareability_search(
                    make_search_state(seed, ids, grid, config), ids, grid, config,
                    _simple_score, evaluate_many=first.map_scored, max_this_call=2,
                )
                self.assertEqual(partial.state.execution_metadata["requested_workers"], 2)
                save_checkpoint(partial.state, Path(tmp) / "checkpoint.json")
            with SpawnEvaluator(_delayed_score, workers=10) as second:
                resumed = run_shareability_search(
                    load_checkpoint(Path(tmp) / "checkpoint.json"), ids, grid, config,
                    _simple_score, evaluate_many=second.map_scored,
                )
                self.assertEqual(resumed.state.execution_metadata["requested_workers"], 10)
        self.assertEqual(one_shot.state.scientific_dict(), resumed.state.scientific_dict())
        self.assertEqual(one_shot.tied_primary_ids, resumed.tied_primary_ids)

    def test_file_backed_cache_single_writer_and_oracle_worker_reuse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            oracle = np.full((3, 3), -39.0)
            np.fill_diagonal(oracle, -np.inf)
            RealBShareabilityOracle(oracle).save(root / "oracle.npz")
            refs = WorkerReferences(str(root / "cache"), str(root / "oracle.npz"))
            with SpawnEvaluator(_cache_and_oracle, workers=10, references=refs) as executor:
                self.assertEqual(executor.map([_candidate(i) for i in range(20)]), [3] * 20)
            self.assertEqual((root / "cache" / "builder_executed_once").read_text(),
                             "one build\n")

    def test_import_has_no_spawn_side_effect(self) -> None:
        scripts = Path(__file__).resolve().parents[3]
        environment = dict(os.environ, PYTHONPATH=str(scripts))
        check = subprocess.run(
            [sys.executable, "-c", "import multiprocessing as mp; "
             "before=len(mp.active_children()); "
             "import retrieval_oriented_model_selection.shared.shareability_parallel; "
             "assert len(mp.active_children())==before"],
            env=environment, capture_output=True, text=True, check=False,
        )
        self.assertEqual(check.returncode, 0, check.stderr)

    def test_keyboard_interrupt_preserves_safe_controller_checkpoint(self) -> None:
        bandwidth = _candidate(0).bandwidth_spec
        seed = CandidateModelSpec(bandwidth, ("a",), 0.0)
        config = SearchConfig(("forward",), 1, 10, 3)
        grid = RidgeGridSpec((0.0,))

        def interrupt(_: CandidateModelSpec) -> int:
            raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as tmp:
            checkpoint = Path(tmp) / "interrupt_checkpoint.json"
            with self.assertRaises(KeyboardInterrupt):
                run_shareability_search(
                    make_search_state(seed, ("a", "b"), grid, config),
                    ("a", "b"), grid, config, interrupt,
                    checkpoint_path_on_interrupt=checkpoint,
                )
            restored = load_checkpoint(checkpoint)
            self.assertEqual(restored.stage_index, 0)
            self.assertEqual(restored.scores, {})

    def test_pool_interrupt_cancels_and_terminates_children(self) -> None:
        before = {process.pid for process in multiprocessing.active_children()}
        with self.assertRaises(KeyboardInterrupt):
            with SpawnEvaluator(_probe, workers=2) as executor:
                with patch(
                    "retrieval_oriented_model_selection.shared.shareability_parallel.as_completed",
                    side_effect=KeyboardInterrupt,
                ):
                    executor.map_scored((_candidate(0), _candidate(1)))
        self.assertEqual(before, {process.pid for process in multiprocessing.active_children()})


if __name__ == "__main__":
    if "--benchmark" in sys.argv:
        candidates = [_candidate(index) for index in range(20)]
        measurements: dict[str, object] = {"label": "synthetic infrastructure only", "jobs": 20}
        for workers, key in ((1, "serial"), (10, "parallel")):
            started = time.perf_counter()
            with SpawnEvaluator(_probe, workers=workers) as executor:
                results = executor.map_scored(candidates)
                measurements[key] = {
                    "wall_seconds": round(time.perf_counter() - started, 6),
                    "effective_worker_pids": sorted({row.worker_pid for row in results}),
                    "jobs_completed": len(results),
                }
        print(json.dumps(measurements, ensure_ascii=False, indent=2))
    else:
        unittest.main()
