"""One-level candidate-granularity CPU execution; import starts no workers."""

from __future__ import annotations

import os
import pickle
import time
import traceback
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from multiprocessing import get_context
from pathlib import Path
from typing import Any

from .shareability_types import CandidateModelSpec, ParallelExecutionSpec

_THREAD_VARIABLES = (
    "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
)
_WORKER_ACTIVE = False
_CONTROLLER_ACTIVE = False
_WORKER_SCORER: Callable[[CandidateModelSpec], int | Mapping[str, Any]] | None = None
_WORKER_REFERENCES: WorkerReferences | None = None
_WORKER_ORACLE: Any = None
_WORKER_CONTEXT: dict[str, Any] | None = None


@dataclass(frozen=True, slots=True)
class WorkerReferences:
    """Small paths sent once per worker; never an in-memory 425-state dataset."""

    cache_root: str | None = None
    oracle_path: str | None = None
    context_path: str | None = None


@dataclass(frozen=True, slots=True)
class CandidateEvaluation:
    candidate_id: str
    status: str
    n_shareable: int | None
    worker_pid: int
    elapsed_seconds: float
    metrics: dict[str, Any] = field(default_factory=dict)
    exception_type: str | None = None
    exception_message: str | None = None
    traceback_summary: str | None = None


class CandidateEvaluationError(RuntimeError):
    """Fail fast with the exact candidate and worker failure details."""

    def __init__(self, result: CandidateEvaluation):
        self.result = result
        super().__init__(
            f"candidate_id={result.candidate_id} worker_pid={result.worker_pid} "
            f"{result.exception_type}: {result.exception_message}"
        )


def worker_active() -> bool:
    return _WORKER_ACTIVE or os.environ.get("SHAREABILITY_WORKER_ACTIVE") == "1"


def _initialize_worker(
    spec: ParallelExecutionSpec,
    scorer: Callable[[CandidateModelSpec], int | Mapping[str, Any]],
    references: WorkerReferences,
) -> None:
    global _WORKER_ACTIVE, _WORKER_SCORER, _WORKER_REFERENCES, _WORKER_ORACLE, _WORKER_CONTEXT
    if worker_active():
        raise RuntimeError("nested shareability worker initialization is forbidden")
    expected = str(spec.blas_threads_per_worker)
    if any(os.environ.get(name) != expected for name in _THREAD_VARIABLES):
        raise RuntimeError("BLAS environment must be set before spawning workers")
    _WORKER_ACTIVE = True
    os.environ["SHAREABILITY_WORKER_ACTIVE"] = "1"
    _WORKER_SCORER = scorer
    _WORKER_REFERENCES = references
    _WORKER_ORACLE = None
    _WORKER_CONTEXT = None


def get_worker_context() -> dict[str, Any]:
    """Load a small JSON context once per worker; large arrays remain memmaps."""

    global _WORKER_CONTEXT
    if _WORKER_REFERENCES is None or _WORKER_REFERENCES.context_path is None:
        raise RuntimeError("no worker context reference configured")
    if _WORKER_CONTEXT is None:
        import json

        with open(_WORKER_REFERENCES.context_path, encoding="utf-8") as handle:
            _WORKER_CONTEXT = json.load(handle)
    return _WORKER_CONTEXT


def get_worker_oracle() -> Any:
    """Load a fixed file-backed Real-B oracle once per process, then reuse it."""

    global _WORKER_ORACLE
    if _WORKER_REFERENCES is None or _WORKER_REFERENCES.oracle_path is None:
        raise RuntimeError("no read-only oracle reference configured")
    if _WORKER_ORACLE is None:
        from .shareability_objective import RealBShareabilityOracle

        _WORKER_ORACLE = RealBShareabilityOracle.load(Path(_WORKER_REFERENCES.oracle_path))
    return _WORKER_ORACLE


def get_worker_cache() -> Any:
    """Return the same file-backed cache namespace in serial and spawned mode."""

    if _WORKER_REFERENCES is None or _WORKER_REFERENCES.cache_root is None:
        raise RuntimeError("no cache reference configured")
    from .shareability_persistence import MetadataCache

    return MetadataCache(Path(_WORKER_REFERENCES.cache_root))


def _evaluate(
    scorer: Callable[[CandidateModelSpec], int | Mapping[str, Any]],
    candidate: CandidateModelSpec,
) -> CandidateEvaluation:
    started = time.monotonic()
    try:
        value = scorer(candidate)
        if isinstance(value, Mapping):
            if value.get("candidate_id", candidate.candidate_id) != candidate.candidate_id:
                raise ValueError("scorer returned a different candidate_id")
            score = value["N_shareable"]
            metrics = dict(value)
        else:
            score = value
            metrics = {}
        if not isinstance(score, int) or isinstance(score, bool) or not 0 <= score <= 425:
            raise ValueError("N_shareable must be an integer in 0..425")
        if len(pickle.dumps(metrics)) > 256 * 1024:
            raise ValueError("candidate result too large; persist arrays to file-backed cache")
        return CandidateEvaluation(
            candidate.candidate_id, "success", score, os.getpid(),
            time.monotonic() - started, metrics,
        )
    except Exception as error:
        return CandidateEvaluation(
            candidate.candidate_id, "failed", None, os.getpid(),
            time.monotonic() - started, {}, type(error).__name__, str(error),
            "".join(traceback.format_exception(error))[-4000:],
        )


def _worker_job(candidate: CandidateModelSpec) -> CandidateEvaluation:
    if not worker_active() or _WORKER_SCORER is None:
        raise RuntimeError("candidate job must run in an initialized shareability worker")
    return _evaluate(_WORKER_SCORER, candidate)


class SpawnEvaluator:
    """Explicit pool; serial debug fallback, deterministic results, fail-fast errors."""

    def __init__(
        self,
        scorer: Callable[[CandidateModelSpec], int | Mapping[str, Any]],
        workers: int | None = None,
        *,
        parallel_spec: ParallelExecutionSpec | None = None,
        references: WorkerReferences | None = None,
    ) -> None:
        if workers is not None and parallel_spec is not None:
            raise ValueError("use ParallelExecutionSpec or workers, not both")
        self.spec = parallel_spec or ParallelExecutionSpec(
            max_workers=10 if workers is None else workers
        )
        self.scorer = scorer
        self.references = references or WorkerReferences()
        self._pool: ProcessPoolExecutor | None = None
        self._saved_env: dict[str, str | None] = {}
        self._pids: set[int] = set()
        self._entered = False

    @property
    def runtime_metadata(self) -> dict[str, int | str | bool | None]:
        return self.spec.to_dict(effective_workers=len(self._pids))

    def __enter__(self) -> SpawnEvaluator:
        global _CONTROLLER_ACTIVE, _WORKER_REFERENCES, _WORKER_ORACLE, _WORKER_CONTEXT
        if worker_active() or _CONTROLLER_ACTIVE or self._entered:
            raise RuntimeError("nested shareability parallelism is forbidden")
        # Reject captured large data and unpicklable local/closure scorers before
        # creating a pool; each job then sends only CandidateModelSpec.
        if self.spec.max_workers > 1:
            serialized = pickle.dumps(self.scorer)
            if len(serialized) > 64 * 1024:
                raise ValueError("scorer captures too much data; use file-backed worker references")
        _CONTROLLER_ACTIVE = True
        self._entered = True
        self._pids.clear()
        _WORKER_REFERENCES = self.references
        _WORKER_ORACLE = None
        _WORKER_CONTEXT = None
        if self.spec.max_workers > 1:
            self._saved_env = {name: os.environ.get(name) for name in _THREAD_VARIABLES}
            for name in _THREAD_VARIABLES:
                os.environ[name] = str(self.spec.blas_threads_per_worker)
            try:
                self._pool = ProcessPoolExecutor(
                    max_workers=self.spec.max_workers,
                    mp_context=get_context(self.spec.start_method),
                    initializer=_initialize_worker,
                    initargs=(self.spec, self.scorer, self.references),
                )
            except BaseException:
                self.__exit__(None, None, None)
                raise
        return self

    def _stop_pool(self, *, force: bool) -> None:
        pool = self._pool
        self._pool = None
        if pool is None:
            return
        if force:
            # Python 3.11 has no public terminate_workers(); capture the
            # executor-owned processes before shutdown for interrupt cleanup.
            processes = list((pool._processes or {}).values())
            pool.shutdown(wait=False, cancel_futures=True)
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=2)
        else:
            pool.shutdown(wait=True, cancel_futures=False)

    def map_scored(self, candidates: Sequence[CandidateModelSpec]) -> list[CandidateEvaluation]:
        if not self._entered or worker_active():
            raise RuntimeError("enter SpawnEvaluator in the main process before evaluation")
        ids = [item.candidate_id for item in candidates]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate candidate_id in one evaluation batch")
        if not candidates:
            return []
        try:
            if self.spec.max_workers == 1:
                results = [_evaluate(self.scorer, item) for item in candidates]
            else:
                assert self._pool is not None
                futures: dict[Future[CandidateEvaluation], str] = {
                    self._pool.submit(_worker_job, item): item.candidate_id
                    for item in candidates
                }
                by_id: dict[str, CandidateEvaluation] = {}
                for future in as_completed(futures):
                    result = future.result()
                    if result.candidate_id != futures[future]:
                        raise RuntimeError("worker returned a mismatched candidate_id")
                    self._pids.add(result.worker_pid)
                    if result.status != "success":
                        raise CandidateEvaluationError(result)
                    by_id[result.candidate_id] = result
                results = [by_id[key] for key in ids]
            for result in results:
                self._pids.add(result.worker_pid)
                if result.status != "success":
                    raise CandidateEvaluationError(result)
            return results
        except BaseException:
            self._stop_pool(force=True)
            raise

    def map(self, candidates: Sequence[CandidateModelSpec]) -> list[int]:
        """Backward-compatible numeric adapter for the existing search API."""

        return [int(result.n_shareable) for result in self.map_scored(candidates)]

    def __exit__(self, exception_type: object, *_: object) -> None:
        global _CONTROLLER_ACTIVE, _WORKER_REFERENCES, _WORKER_ORACLE, _WORKER_CONTEXT
        self._stop_pool(force=exception_type is not None)
        for name, old in self._saved_env.items():
            if old is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = old
        self._saved_env.clear()
        _WORKER_REFERENCES = None
        _WORKER_ORACLE = None
        _WORKER_CONTEXT = None
        _CONTROLLER_ACTIVE = False
        self._entered = False


__all__ = [
    "WorkerReferences", "CandidateEvaluation", "CandidateEvaluationError",
    "SpawnEvaluator", "worker_active", "get_worker_oracle", "get_worker_cache",
    "get_worker_context",
]
