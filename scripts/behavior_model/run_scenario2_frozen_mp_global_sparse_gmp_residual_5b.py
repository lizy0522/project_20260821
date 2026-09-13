"""Global unified residual Sparse-GMP validation around the frozen 5B MP.

The Frozen MP structure and raw-basis Ridge value are fixed.  A deterministic
30-term cross-memory GMP dictionary is searched with a forward path of G0--G8
using only A/C blocked validation on the existing 340-state Development split.
The 85-state Validation B metrics are opened only after the support is frozen.
No LUT retrieval or Frozen-config replacement is performed here.
"""

# Imports intentionally follow the BLAS-thread environment setup below.
# ruff: noqa: E402,I001,E501

from __future__ import annotations

import hashlib
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from multiprocessing import freeze_support
from multiprocessing.shared_memory import SharedMemory
from pathlib import Path
from typing import Any

# Set process-local numerical threading before importing NumPy/SciPy.
for _thread_name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_name] = "1"
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import numpy as np
import pandas as pd

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from behavior_model import frozen_neighborhood_memory_ridge_scan as frozen_model  # noqa: E402
from behavior_model import sparse_gmp as gmp  # noqa: E402


TASK_NAME = "scenario_2_frozen_mp_global_sparse_gmp_residual_5B"
RESULT_ROOT = PROJECT_ROOT / "results" / "behavior_model" / TASK_NAME
MODEL_LOG = PROJECT_ROOT / "work_logs" / "behavior_model" / "execution_log.txt"
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "codex_handoff" / "codex_handoff.txt"

SPLIT_PATH = PROJECT_ROOT / "results" / "behavior_model" / "scenario_2_ridge_analysis" / "state_split.csv"
ALL_ILC_METRICS = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2_all_ilc"
    / "all_ilc_model_metrics.csv"
)
ALL_ILC_THETA = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2_all_ilc"
    / "all_ilc_theta.npz"
)

STATE_COUNT = 425
DEVELOPMENT_COUNT = 340
VALIDATION_COUNT = 85
ORDERS = (1, 2, 3, 5, 7, 9)
MEMORY = (3, 2, 2, 1, 1, 1)
FROZEN_K = 10
DMAX = 2
RIDGE_LAMBDA = 1e-8
GMP_ORDERS = (2, 3, 5, 7, 9)
GMP_DELAYS = (0, 1, 2)
MAX_SPARSE_TERMS = 8
CV_FOLDS = 3
CPU_TARGET = 0.90
FROZEN_REPRODUCTION_IDS = (0, 100, 187, 340, 424)
VALIDATION_DELTA_THRESHOLD_DB = -0.10
GAP_DELTA_MAX_DB = 0.10

PROTECTED_RESULT_DIRS = {
    "formal_retrieval": PROJECT_ROOT / "results" / "behavior_fingerprint_lut_retrieval" / "scenario_2_C2_to_Aend",
    "formal_all_ilc": PROJECT_ROOT / "results" / "behavior_fingerprint_ranking_consistency" / "scenario_2_all_ilc",
    "unified_capacity": PROJECT_ROOT / "results" / "behavior_model" / "scenario_2_unified_odd_order_mp_capacity_scan_5B",
    "odd_only_scan": PROJECT_ROOT / "results" / "behavior_model" / "scenario_2_failed14_odd_order_mp_order_dependent_memory_scan_5B",
    "p5_ridge_scan": PROJECT_ROOT / "results" / "behavior_model" / "scenario_2_failed14_P5_variable_memory_ridge_scan_5B",
    "p5_order2_scan": PROJECT_ROOT / "results" / "behavior_model" / "scenario_2_failed14_P5_with_order2_variable_memory_OLS_scan_5B",
    "neighborhood_scan": PROJECT_ROOT / "results" / "behavior_model" / "scenario_2_failed14_frozen_neighborhood_memory_ridge_scan_5B",
    "prior_sparse_gmp": PROJECT_ROOT / "results" / "behavior_model" / "scenario_2_failed14_frozen_mp_residual_sparse_gmp_5B",
    "g4_retrieval": PROJECT_ROOT / "results" / "behavior_fingerprint_lut_retrieval" / "scenario_2_C2_to_Aend_G4_sparse_gmp_5B",
}

PROTECTED_SCRIPT_FILES = (
    PROJECT_ROOT / "scripts" / "behavior_model" / "config.py",
    PROJECT_ROOT / "scripts" / "behavior_model" / "basis.py",
    PROJECT_ROOT / "scripts" / "behavior_model" / "sparse_gmp.py",
    PROJECT_ROOT / "scripts" / "behavior_model" / "frozen_neighborhood_memory_ridge_scan.py",
    PROJECT_ROOT / "scripts" / "behavior_model" / "scenario2_ridge_analysis.py",
    PROJECT_ROOT / "scripts" / "signal_segmentation" / "preprocessing.py",
    PROJECT_ROOT / "scripts" / "signal_segmentation" / "partition.py",
    PROJECT_ROOT / "scripts" / "signal_segmentation" / "behavior_pairs.py",
)


@dataclass(frozen=True, order=True)
class GlobalGMPTerm:
    """One deterministic cross-memory GMP candidate."""

    term_index: int
    term_id: str
    p: int
    m: int
    q: int

    @property
    def expression(self) -> str:
        return f"x[n-{self.m}]|x[n-{self.q}]|^{self.p - 1}"

    @property
    def term_tuple(self) -> tuple[int, int, int]:
        return self.p, self.m, self.q


GLOBAL_TERMS = tuple(
    GlobalGMPTerm(index, f"GMP_p{p:02d}_m{m:02d}_q{q:02d}", p, m, q)
    for index, (p, m, q) in enumerate(
        (  # p-major, then m-major, then q-major; aligned terms are excluded.
            (p, m, q)
            for p in GMP_ORDERS
            for m in GMP_DELAYS
            for q in GMP_DELAYS
            if m != q
        ),
        start=1,
    )
)
TERM_BY_INDEX = {term.term_index - 1: term for term in GLOBAL_TERMS}


def _validate_term_pool() -> dict[str, Any]:
    tuples = [term.term_tuple for term in GLOBAL_TERMS]
    required_old_g4 = {(2, 2, 0), (3, 2, 0), (2, 0, 1), (5, 2, 0)}
    observed = set(tuples)
    checks = {
        "candidate_count": len(GLOBAL_TERMS) == 30,
        "unique_terms": len(observed) == 30,
        "orders_allowed": all(term.p in GMP_ORDERS for term in GLOBAL_TERMS),
        "delays_allowed": all(term.m in GMP_DELAYS and term.q in GMP_DELAYS for term in GLOBAL_TERMS),
        "cross_terms_only": all(term.m != term.q for term in GLOBAL_TERMS),
        "common_dmax": max(max(term.m, term.q) for term in GLOBAL_TERMS) == DMAX,
        "old_g4_terms_present": required_old_g4.issubset(observed),
    }
    if not all(checks.values()):
        raise RuntimeError(f"GMP candidate-pool gate failed: {checks}")
    return {
        **checks,
        "terms": [
            {
                "term_index": term.term_index,
                "term_id": term.term_id,
                "p": term.p,
                "m": term.m,
                "q": term.q,
                "expression": term.expression,
            }
            for term in GLOBAL_TERMS
        ],
        "old_g4_tuples": [list(item) for item in sorted(required_old_g4)],
    }


def _tree_digest(path: Path) -> str | None:
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    files = [
        file
        for file in path.rglob("*")
        if file.is_file() and "__pycache__" not in file.parts and file.suffix.lower() != ".pyc"
    ]
    for file in sorted(files, key=lambda item: str(item).lower()):
        content = file.read_bytes()
        digest.update(file.relative_to(path).as_posix().encode("utf-8") + b"\0")
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def _file_digest(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _raw_manifest() -> dict[str, Any]:
    raw_root = PROJECT_ROOT / "data" / "raw"
    digest = hashlib.sha256()
    files = [file for file in raw_root.rglob("*") if file.is_file()]
    total = 0
    for file in sorted(files, key=lambda item: str(item).lower()):
        content = file.read_bytes()
        digest.update(file.relative_to(raw_root).as_posix().encode("utf-8") + b"\0")
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(hashlib.sha256(content).digest())
        total += len(content)
    return {"sha256": digest.hexdigest(), "file_count": len(files), "bytes": total}


def _snapshot() -> dict[str, Any]:
    return {
        "data/raw": _raw_manifest(),
        "result_dirs": {name: _tree_digest(path) for name, path in PROTECTED_RESULT_DIRS.items()},
        "script_files": {str(path.relative_to(PROJECT_ROOT)): _file_digest(path) for path in PROTECTED_SCRIPT_FILES},
    }


def _protection_check(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    raw_unchanged = before["data/raw"] == after["data/raw"]
    result_unchanged = {
        name: before["result_dirs"].get(name) == after["result_dirs"].get(name)
        for name in PROTECTED_RESULT_DIRS
    }
    scripts_unchanged = {
        name: before["script_files"].get(name) == after["script_files"].get(name)
        for name in before["script_files"]
    }
    return {
        "data_raw_unchanged": raw_unchanged,
        "protected_results_unchanged": bool(all(result_unchanged.values())),
        "protected_scripts_unchanged": bool(all(scripts_unchanged.values())),
        "all_protected_unchanged": bool(raw_unchanged and all(result_unchanged.values()) and all(scripts_unchanged.values())),
        "result_dirs": result_unchanged,
        "script_files": scripts_unchanged,
    }


def _load_split() -> tuple[tuple[int, ...], tuple[int, ...], str, dict[str, Any]]:
    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(f"existing frozen Development/Validation split is missing: {SPLIT_PATH}")
    frame = pd.read_csv(SPLIT_PATH)
    required = {"state_id", "split"}
    if not required.issubset(frame.columns):
        raise RuntimeError(f"state split is missing columns: {sorted(required - set(frame.columns))}")
    frame = frame.sort_values("state_id").reset_index(drop=True)
    ids = frame["state_id"].to_numpy(dtype=np.int64)
    if frame.shape[0] != STATE_COUNT or not np.array_equal(ids, np.arange(STATE_COUNT, dtype=np.int64)):
        raise RuntimeError("state split must contain state IDs 0...424 exactly once")
    split_values = frame["split"].astype(str).str.lower().to_numpy()
    development = tuple(int(value) for value in ids[split_values == "development"])
    validation = tuple(int(value) for value in ids[split_values == "validation"])
    if len(development) != DEVELOPMENT_COUNT or len(validation) != VALIDATION_COUNT or set(development) & set(validation):
        raise RuntimeError(f"invalid frozen split counts: development={len(development)}, validation={len(validation)}")
    if set(development) | set(validation) != set(range(STATE_COUNT)):
        raise RuntimeError("Development and Validation do not cover 0...424")
    payload = {
        "path": str(SPLIT_PATH),
        "sha256": hashlib.sha256(SPLIT_PATH.read_bytes()).hexdigest(),
        "development_count": len(development),
        "validation_count": len(validation),
        "disjoint": True,
        "complete": True,
    }
    return development, validation, payload["sha256"], payload


def _nmse_db(reference: np.ndarray, prediction: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.complex128).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.complex128).reshape(-1)
    if reference.shape != prediction.shape or not np.all(np.isfinite(reference)) or not np.all(np.isfinite(prediction)):
        raise ValueError("NMSE inputs must be finite complex vectors with equal shape")
    denominator = float(np.sum(np.abs(reference) ** 2))
    numerator = float(np.sum(np.abs(reference - prediction) ** 2))
    if denominator <= 0:
        raise ValueError("NMSE reference energy must be positive")
    if numerator == 0.0:
        return float("-inf")
    return float(10.0 * np.log10(numerator / denominator))


def _build_global_gmp_basis(x: np.ndarray, terms: tuple[GlobalGMPTerm, ...] = GLOBAL_TERMS) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    if x.ndim != 1 or x.size <= DMAX or not np.all(np.isfinite(x)):
        raise ValueError("GMP input must be a finite complex vector longer than dmax")
    columns = []
    for term in terms:
        carrier = x[DMAX - term.m : x.size - term.m]
        envelope = np.abs(x[DMAX - term.q : x.size - term.q]) ** (term.p - 1)
        columns.append(carrier * envelope)
    result = np.column_stack(columns).astype(np.complex128, copy=False)
    expected = (x.size - DMAX, len(terms))
    if result.shape != expected or not np.all(np.isfinite(result)):
        raise RuntimeError(f"global GMP basis shape/finite gate failed: {result.shape} != {expected}")
    return result


def _fit_augmented(phi: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Fit the fixed raw-basis Ridge through the formal augmented LS system."""

    phi = np.asarray(phi, dtype=np.complex128)
    y = np.asarray(y, dtype=np.complex128).reshape(-1)
    if phi.ndim != 2 or y.ndim != 1 or phi.shape[0] != y.size or phi.shape[0] <= phi.shape[1]:
        raise ValueError("invalid augmented Ridge dimensions")
    penalty = math.sqrt(phi.shape[0] * RIDGE_LAMBDA)
    augmented_phi = np.vstack(
        (phi, penalty * np.eye(phi.shape[1], dtype=np.complex128))
    )
    augmented_y = np.concatenate((y, np.zeros(phi.shape[1], dtype=np.complex128)))
    theta, _, rank, _ = np.linalg.lstsq(augmented_phi, augmented_y, rcond=None)
    if int(rank) != phi.shape[1] or not np.all(np.isfinite(theta)):
        raise RuntimeError(f"augmented Ridge rank/finite gate failed: rank={rank}/{phi.shape[1]}")
    return np.asarray(theta, dtype=np.complex128)


def _folds(n_samples: int) -> tuple[np.ndarray, ...]:
    return tuple(np.asarray(index, dtype=np.int64) for index in np.array_split(np.arange(n_samples, dtype=np.int64), CV_FOLDS))


def _cv_score(phi: np.ndarray, y: np.ndarray, folds: tuple[np.ndarray, ...]) -> tuple[float, tuple[float, ...]]:
    values: list[float] = []
    for validation_index in folds:
        training_mask = np.ones(y.size, dtype=bool)
        training_mask[validation_index] = False
        theta = _fit_augmented(phi[training_mask], y[training_mask])
        values.append(_nmse_db(y[validation_index], phi[validation_index] @ theta))
    return float(np.median(values)), tuple(values)


def _combined_columns(bank: np.ndarray, selected_indices: tuple[int, ...]) -> np.ndarray:
    columns = list(range(FROZEN_K)) + [FROZEN_K + int(index) for index in selected_indices]
    return bank[:, columns]


_DEV_SHMS: list[SharedMemory] = []
_DEV_XA: np.ndarray | None = None
_DEV_YA: np.ndarray | None = None
_DEV_XC: np.ndarray | None = None
_DEV_YC: np.ndarray | None = None
_DEV_INDEX: dict[int, int] = {}


def _prepare_dev_worker(state_id: int) -> dict[str, Any]:
    prepared = frozen_model.prepare_state_any(int(state_id))
    return {
        "state_id": int(state_id),
        "ilc_A_end": int(prepared.ilc_A_end),
        "a_x": np.asarray(prepared.aend_A_input, dtype=np.complex128),
        "a_y": np.asarray(prepared.aend_A_output, dtype=np.complex128),
        "c_x": np.asarray(prepared.c2_C_input, dtype=np.complex128),
        "c_y": np.asarray(prepared.c2_C_output, dtype=np.complex128),
    }


def _prepare_development_arrays(state_ids: tuple[int, ...], workers: int) -> dict[str, Any]:
    started = time.perf_counter()
    values: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_prepare_dev_worker, state_id): state_id for state_id in state_ids}
        checkpoints = {math.ceil(len(state_ids) * fraction) for fraction in (0.25, 0.5, 0.75, 1.0)}
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            values[int(result["state_id"])] = result
            if done in checkpoints:
                print(f"Development canonical preprocessing: {done}/{len(state_ids)}", flush=True)
    if tuple(sorted(values)) != tuple(sorted(state_ids)):
        raise RuntimeError("Development preprocessing state set is incomplete")
    a_x = np.stack([values[state_id]["a_x"] for state_id in state_ids]).astype(np.complex128, copy=False)
    a_y = np.stack([values[state_id]["a_y"] for state_id in state_ids]).astype(np.complex128, copy=False)
    c_x = np.stack([values[state_id]["c_x"] for state_id in state_ids]).astype(np.complex128, copy=False)
    c_y = np.stack([values[state_id]["c_y"] for state_id in state_ids]).astype(np.complex128, copy=False)
    expected_shapes = {
        "a_x": (len(state_ids), frozen_model.FORMAL_A_LENGTH),
        "a_y": (len(state_ids), frozen_model.FORMAL_A_LENGTH),
        "c_x": (len(state_ids), frozen_model.FORMAL_C_LENGTH),
        "c_y": (len(state_ids), frozen_model.FORMAL_C_LENGTH),
    }
    arrays = {"a_x": a_x, "a_y": a_y, "c_x": c_x, "c_y": c_y}
    for name, array in arrays.items():
        if array.shape != expected_shapes[name] or not np.iscomplexobj(array) or not np.all(np.isfinite(array)):
            raise RuntimeError(f"Development shared-array gate failed for {name}: {array.shape}")
    return {"state_ids": state_ids, "arrays": arrays, "elapsed_seconds": time.perf_counter() - started}


def _create_shared_arrays(arrays: dict[str, np.ndarray]) -> tuple[dict[str, tuple[str, tuple[int, ...]]], list[SharedMemory]]:
    descriptors: dict[str, tuple[str, tuple[int, ...]]] = {}
    parent_shms: list[SharedMemory] = []
    for name, array in arrays.items():
        shm = SharedMemory(create=True, size=array.nbytes)
        view = np.ndarray(array.shape, dtype=np.complex128, buffer=shm.buf)
        view[:] = array
        descriptors[name] = (shm.name, tuple(array.shape))
        parent_shms.append(shm)
    return descriptors, parent_shms


def _dev_worker_init(
    descriptors: dict[str, tuple[str, tuple[int, ...]]], state_ids: tuple[int, ...]
) -> None:
    global _DEV_SHMS, _DEV_XA, _DEV_YA, _DEV_XC, _DEV_YC, _DEV_INDEX
    _DEV_SHMS = []
    views: dict[str, np.ndarray] = {}
    for name, (shm_name, shape) in descriptors.items():
        shm = SharedMemory(name=shm_name)
        _DEV_SHMS.append(shm)
        views[name] = np.ndarray(shape, dtype=np.complex128, buffer=shm.buf)
    _DEV_XA, _DEV_YA = views["a_x"], views["a_y"]
    _DEV_XC, _DEV_YC = views["c_x"], views["c_y"]
    _DEV_INDEX = {int(state_id): index for index, state_id in enumerate(state_ids)}
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[name] = "1"


def _state_stage_worker(payload: tuple[int, tuple[int, ...], tuple[int, ...], int]) -> dict[str, Any]:
    stage, current_indices, trial_indices, state_id = payload
    if _DEV_XA is None or _DEV_YA is None or _DEV_XC is None or _DEV_YC is None:
        raise RuntimeError("Development shared memory worker is not initialized")
    position = _DEV_INDEX[int(state_id)]
    a_x = _DEV_XA[position]
    a_y = _DEV_YA[position][DMAX:]
    c_x = _DEV_XC[position]
    c_y = _DEV_YC[position][DMAX:]
    a_bank = np.column_stack((gmp.build_frozen_mp_basis(a_x), _build_global_gmp_basis(a_x)))
    c_bank = np.column_stack((gmp.build_frozen_mp_basis(c_x), _build_global_gmp_basis(c_x)))
    if a_bank.shape != (a_y.size, FROZEN_K + len(GLOBAL_TERMS)) or c_bank.shape != (c_y.size, FROZEN_K + len(GLOBAL_TERMS)):
        raise RuntimeError("Development basis-bank shape mismatch")
    a_folds = _folds(a_y.size)
    c_folds = _folds(c_y.size)
    a_current_phi = _combined_columns(a_bank, current_indices)
    c_current_phi = _combined_columns(c_bank, current_indices)
    a_current, _ = _cv_score(a_current_phi, a_y, a_folds)
    c_current, _ = _cv_score(c_current_phi, c_y, c_folds)
    a_trial_values: list[float] = []
    c_trial_values: list[float] = []
    for trial_index in trial_indices:
        trial_support = tuple(current_indices) + (int(trial_index),)
        a_trial, _ = _cv_score(_combined_columns(a_bank, trial_support), a_y, a_folds)
        c_trial, _ = _cv_score(_combined_columns(c_bank, trial_support), c_y, c_folds)
        a_trial_values.append(a_trial)
        c_trial_values.append(c_trial)
    return {
        "stage": int(stage),
        "state_id": int(state_id),
        "current_A": float(a_current),
        "current_C": float(c_current),
        "trial_indices": tuple(int(index) for index in trial_indices),
        "trial_A": tuple(a_trial_values),
        "trial_C": tuple(c_trial_values),
    }


def _aggregate(values: np.ndarray) -> dict[str, float]:
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise RuntimeError("candidate aggregate values must be finite and one-dimensional")
    return {
        "median": float(np.median(values)),
        "q75": float(np.quantile(values, 0.75)),
        "q90": float(np.quantile(values, 0.90)),
    }


def _summary_row(
    stage: int,
    support: tuple[int, ...],
    added_index: int | None,
    a: dict[str, float],
    c: dict[str, float],
    is_final: bool = False,
) -> dict[str, Any]:
    term = TERM_BY_INDEX[added_index] if added_index is not None else None
    return {
        "stage_G": int(stage),
        "K_total": FROZEN_K + int(stage),
        "added_term_id": "" if term is None else term.term_id,
        "added_p": "" if term is None else term.p,
        "added_m": "" if term is None else term.m,
        "added_q": "" if term is None else term.q,
        "added_expression": "" if term is None else term.expression,
        "selected_support_ids": ";".join(TERM_BY_INDEX[index].term_id for index in support),
        "dev_A_median": a["median"],
        "dev_C_median": c["median"],
        "balanced_median": max(a["median"], c["median"]),
        "dev_A_q75": a["q75"],
        "dev_C_q75": c["q75"],
        "balanced_q75": max(a["q75"], c["q75"]),
        "dev_A_q90": a["q90"],
        "dev_C_q90": c["q90"],
        "balanced_q90": max(a["q90"], c["q90"]),
        "is_final_selected_G": bool(is_final),
    }


def _run_forward_selection(
    state_ids: tuple[int, ...],
    arrays: dict[str, np.ndarray],
    workers: int,
) -> tuple[pd.DataFrame, tuple[int, ...], dict[int, tuple[int, ...]], dict[str, Any]]:
    descriptors, parent_shms = _create_shared_arrays(arrays)
    selected: list[int] = []
    summaries: list[dict[str, Any]] = []
    supports: dict[int, tuple[int, ...]] = {0: ()}
    started = time.perf_counter()
    try:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_dev_worker_init,
            initargs=(descriptors, state_ids),
        ) as executor:
            for stage in range(1, MAX_SPARSE_TERMS + 1):
                remaining = tuple(index for index in range(len(GLOBAL_TERMS)) if index not in selected)
                if not remaining:
                    raise RuntimeError("GMP candidate pool was exhausted before G8")
                current_indices = tuple(selected)
                futures = {
                    executor.submit(_state_stage_worker, (stage, current_indices, remaining, state_id)): state_id
                    for state_id in state_ids
                }
                current_a = np.empty(len(state_ids), dtype=np.float64)
                current_c = np.empty(len(state_ids), dtype=np.float64)
                trial_a = np.empty((len(state_ids), len(remaining)), dtype=np.float64)
                trial_c = np.empty_like(trial_a)
                state_position = {state_id: index for index, state_id in enumerate(state_ids)}
                for done, future in enumerate(as_completed(futures), start=1):
                    result = future.result()
                    position = state_position[int(result["state_id"])]
                    current_a[position] = result["current_A"]
                    current_c[position] = result["current_C"]
                    trial_a[position, :] = np.asarray(result["trial_A"], dtype=np.float64)
                    trial_c[position, :] = np.asarray(result["trial_C"], dtype=np.float64)
                    if done in {math.ceil(len(state_ids) * fraction) for fraction in (0.25, 0.5, 0.75, 1.0)}:
                        print(f"Forward G{stage}: state tasks {done}/{len(state_ids)}", flush=True)
                if stage == 1:
                    summaries.append(
                        _summary_row(0, (), None, _aggregate(current_a), _aggregate(current_c))
                    )
                candidate_records: list[tuple[tuple[float, float, float, str], int, dict[str, float], dict[str, float]]] = []
                for column, trial_index in enumerate(remaining):
                    a_stats = _aggregate(trial_a[:, column])
                    c_stats = _aggregate(trial_c[:, column])
                    key = (
                        max(a_stats["median"], c_stats["median"]),
                        max(a_stats["q75"], c_stats["q75"]),
                        max(a_stats["q90"], c_stats["q90"]),
                        TERM_BY_INDEX[trial_index].term_id,
                    )
                    candidate_records.append((key, trial_index, a_stats, c_stats))
                _, best_index, best_a, best_c = min(candidate_records, key=lambda item: item[0])
                selected.append(int(best_index))
                support = tuple(selected)
                supports[stage] = support
                summaries.append(_summary_row(stage, support, best_index, best_a, best_c))
                term = TERM_BY_INDEX[best_index]
                print(
                    f"Forward G{stage} selected {term.term_id} ({term.expression}); "
                    f"balanced median={max(best_a['median'], best_c['median']):.6f} dB",
                    flush=True,
                )
    finally:
        for shm in parent_shms:
            shm.close()
            shm.unlink()
    summary = pd.DataFrame(summaries).sort_values("stage_G").reset_index(drop=True)
    if summary.shape[0] != MAX_SPARSE_TERMS + 1:
        raise RuntimeError("G0...G8 forward summary is incomplete")
    best_score = float(summary["balanced_median"].min())
    eligible = summary.loc[summary["balanced_median"] <= best_score + 0.10, "stage_G"].to_numpy(dtype=int)
    selected_g = int(np.min(eligible))
    summary["is_final_selected_G"] = summary["stage_G"] == selected_g
    selected_support = supports[selected_g]
    return summary, selected_support, supports, {
        "elapsed_seconds": time.perf_counter() - started,
        "trial_evaluation_count": sum(30 - stage + 1 for stage in range(1, MAX_SPARSE_TERMS + 1)),
        "fit_estimate": 212 * len(state_ids) * 2 * CV_FOLDS,
        "best_balanced_median": best_score,
        "selected_G": selected_g,
    }


def _frozen_reproduction_gate() -> dict[str, Any]:
    if not ALL_ILC_METRICS.is_file() or not ALL_ILC_THETA.is_file():
        raise FileNotFoundError("formal all-ILC Frozen references are missing")
    formal = pd.read_csv(ALL_ILC_METRICS)
    with np.load(ALL_ILC_THETA, allow_pickle=False) as data:
        formal_a_theta = np.asarray(data["theta_Y_A_actual"], dtype=np.complex128)
        formal_c_theta = np.asarray(data["theta_Y_C_actual"], dtype=np.complex128)
    rows: list[dict[str, Any]] = []
    for state_id in FROZEN_REPRODUCTION_IDS:
        prepared = frozen_model.prepare_state_any(state_id)
        a_phi = gmp.build_frozen_mp_basis(prepared.aend_A_input)
        c_phi = gmp.build_frozen_mp_basis(prepared.c2_C_input)
        a_theta, _ = gmp.fit_ridge(a_phi, prepared.aend_A_output[DMAX:], RIDGE_LAMBDA)
        c_theta, _ = gmp.fit_ridge(c_phi, prepared.c2_C_output[DMAX:], RIDGE_LAMBDA)
        a_b_phi = gmp.build_frozen_mp_basis(prepared.aend_B_input)
        c_b_phi = gmp.build_frozen_mp_basis(prepared.c2_B_input)
        current = {
            "A_train": _nmse_db(prepared.aend_A_output[DMAX:], a_phi @ a_theta),
            "A_B": _nmse_db(prepared.aend_B_output[DMAX:], a_b_phi @ a_theta),
            "C_train": _nmse_db(prepared.c2_C_output[DMAX:], c_phi @ c_theta),
            "C_B": _nmse_db(prepared.c2_B_output[DMAX:], c_b_phi @ c_theta),
        }
        formal_a = formal.loc[
            (formal["state_id"] == state_id)
            & (formal["actual_ilc_n"] == prepared.ilc_A_end)
            & (formal["model_role"] == "Y-A")
            & np.isclose(formal["ridge_lambda"].astype(float), RIDGE_LAMBDA)
        ]
        formal_c = formal.loc[
            (formal["state_id"] == state_id)
            & (formal["actual_ilc_n"] == 2)
            & (formal["model_role"] == "Y-C")
            & np.isclose(formal["ridge_lambda"].astype(float), RIDGE_LAMBDA)
        ]
        if formal_a.shape[0] != 1 or formal_c.shape[0] != 1:
            raise RuntimeError(f"formal Frozen metric mapping missing for state {state_id}")
        expected = {
            "A_train": float(formal_a.iloc[0]["train_nmse_db"]),
            "A_B": float(formal_a.iloc[0]["B_generalization_nmse_db"]),
            "C_train": float(formal_c.iloc[0]["train_nmse_db"]),
            "C_B": float(formal_c.iloc[0]["B_generalization_nmse_db"]),
        }
        metric_error = max(abs(current[name] - expected[name]) for name in current)
        formal_a_theta_row = formal_a_theta[state_id, prepared.ilc_A_end - 1]
        formal_c_theta_row = formal_c_theta[state_id, 1]
        theta_error = max(
            float(np.max(np.abs(a_theta - formal_a_theta_row))),
            float(np.max(np.abs(c_theta - formal_c_theta_row))),
        )
        rows.append(
            {
                "state_id": state_id,
                "ilc_A_end": prepared.ilc_A_end,
                "metric_max_abs_error_dB": metric_error,
                "theta_max_abs_error": theta_error,
                "pass": bool(metric_error <= 1e-6 and theta_error <= 1e-12),
            }
        )
    maximum_metric_error = max(row["metric_max_abs_error_dB"] for row in rows)
    maximum_theta_error = max(row["theta_max_abs_error"] for row in rows)
    passed = bool(all(row["pass"] for row in rows))
    return {
        "pass": passed,
        "maximum_metric_abs_error_dB": maximum_metric_error,
        "maximum_theta_abs_error": maximum_theta_error,
        "rows": rows,
    }


def _fit_full_model(prepared: Any, selected_indices: tuple[int, ...]) -> dict[str, float]:
    def fit_side(x: np.ndarray, y: np.ndarray, b_x: np.ndarray, b_y: np.ndarray) -> tuple[float, float]:
        phi = np.column_stack((gmp.build_frozen_mp_basis(x), _build_global_gmp_basis(x)[..., list(selected_indices)])) if selected_indices else gmp.build_frozen_mp_basis(x)
        phi_b = np.column_stack((gmp.build_frozen_mp_basis(b_x), _build_global_gmp_basis(b_x)[..., list(selected_indices)])) if selected_indices else gmp.build_frozen_mp_basis(b_x)
        theta, _ = gmp.fit_ridge(phi, np.asarray(y, dtype=np.complex128)[DMAX:], RIDGE_LAMBDA)
        return (
            _nmse_db(np.asarray(y, dtype=np.complex128)[DMAX:], phi @ theta),
            _nmse_db(np.asarray(b_y, dtype=np.complex128)[DMAX:], phi_b @ theta),
        )

    a_train, a_b = fit_side(
        prepared.aend_A_input,
        prepared.aend_A_output,
        prepared.aend_B_input,
        prepared.aend_B_output,
    )
    c_train, c_b = fit_side(
        prepared.c2_C_input,
        prepared.c2_C_output,
        prepared.c2_B_input,
        prepared.c2_B_output,
    )
    return {
        "Aend_train_NMSE_dB": a_train,
        "Aend_B_NMSE_dB": a_b,
        "C2_train_NMSE_dB": c_train,
        "C2_B_NMSE_dB": c_b,
    }


def _evaluation_worker(payload: tuple[int, tuple[int, ...]]) -> dict[str, Any]:
    state_id, selected_indices = payload
    prepared = frozen_model.prepare_state_any(int(state_id))
    frozen_values = _fit_full_model(prepared, ())
    selected_values = _fit_full_model(prepared, selected_indices)
    result: dict[str, Any] = {"state_id": int(state_id)}
    for prefix, values in (("Frozen", frozen_values), ("Selected", selected_values)):
        result.update({f"{prefix}_{key}": value for key, value in values.items()})
    for key in ("Aend_train_NMSE_dB", "Aend_B_NMSE_dB", "C2_train_NMSE_dB", "C2_B_NMSE_dB"):
        result[f"delta_{key}"] = result[f"Selected_{key}"] - result[f"Frozen_{key}"]
    result["Frozen_gap_A_dB"] = result["Frozen_Aend_B_NMSE_dB"] - result["Frozen_Aend_train_NMSE_dB"]
    result["Selected_gap_A_dB"] = result["Selected_Aend_B_NMSE_dB"] - result["Selected_Aend_train_NMSE_dB"]
    result["delta_gap_A_dB"] = result["Selected_gap_A_dB"] - result["Frozen_gap_A_dB"]
    result["Frozen_gap_C_dB"] = result["Frozen_C2_B_NMSE_dB"] - result["Frozen_C2_train_NMSE_dB"]
    result["Selected_gap_C_dB"] = result["Selected_C2_B_NMSE_dB"] - result["Selected_C2_train_NMSE_dB"]
    result["delta_gap_C_dB"] = result["Selected_gap_C_dB"] - result["Frozen_gap_C_dB"]
    result["Frozen_worst4_dB"] = max(frozen_values.values())
    result["Selected_worst4_dB"] = max(selected_values.values())
    result["delta_worst4_dB"] = result["Selected_worst4_dB"] - result["Frozen_worst4_dB"]
    return result


def _run_evaluation(state_ids: tuple[int, ...], selected_indices: tuple[int, ...], workers: int, label: str) -> pd.DataFrame:
    values: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_evaluation_worker, (state_id, selected_indices)): state_id for state_id in state_ids}
        checkpoints = {math.ceil(len(state_ids) * fraction) for fraction in (0.25, 0.5, 0.75, 1.0)}
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            values[int(result["state_id"])] = result
            if done in checkpoints:
                print(f"{label}: {done}/{len(state_ids)}", flush=True)
    if tuple(sorted(values)) != tuple(sorted(state_ids)):
        raise RuntimeError(f"{label} state evaluation is incomplete")
    frame = pd.DataFrame([values[state_id] for state_id in state_ids]).sort_values("state_id").reset_index(drop=True)
    return frame


def _validation_report(frame: pd.DataFrame) -> dict[str, Any]:
    delta_columns = {
        "Aend_B": "delta_Aend_B_NMSE_dB",
        "C2_B": "delta_C2_B_NMSE_dB",
        "worst4": "delta_worst4_dB",
    }
    medians = {name: float(frame[column].median()) for name, column in delta_columns.items()}
    metric_medians = {
        column: float(frame[column].median())
        for column in (
            "Frozen_Aend_train_NMSE_dB",
            "Selected_Aend_train_NMSE_dB",
            "Frozen_Aend_B_NMSE_dB",
            "Selected_Aend_B_NMSE_dB",
            "Frozen_C2_train_NMSE_dB",
            "Selected_C2_train_NMSE_dB",
            "Frozen_C2_B_NMSE_dB",
            "Selected_C2_B_NMSE_dB",
            "Frozen_gap_A_dB",
            "Selected_gap_A_dB",
            "Frozen_gap_C_dB",
            "Selected_gap_C_dB",
            "Frozen_worst4_dB",
            "Selected_worst4_dB",
        )
    }
    all_delta_medians = {
        column.removeprefix("delta_"): float(frame[column].median())
        for column in (
            "delta_Aend_train_NMSE_dB",
            "delta_Aend_B_NMSE_dB",
            "delta_C2_train_NMSE_dB",
            "delta_C2_B_NMSE_dB",
            "delta_worst4_dB",
        )
    }
    gap_medians = {
        "A": float(frame["delta_gap_A_dB"].median()),
        "C": float(frame["delta_gap_C_dB"].median()),
    }
    improved = frame["delta_worst4_dB"] < 0.0
    degraded = frame["delta_worst4_dB"] > 0.0
    criteria = {
        "median_delta_Aend_B_at_most_minus_0p10dB": medians["Aend_B"] <= VALIDATION_DELTA_THRESHOLD_DB,
        "median_delta_C2_B_at_most_minus_0p10dB": medians["C2_B"] <= VALIDATION_DELTA_THRESHOLD_DB,
        "median_delta_worst4_at_most_minus_0p10dB": medians["worst4"] <= VALIDATION_DELTA_THRESHOLD_DB,
        "improved_worst4_count_greater_than_degraded": int(improved.sum()) > int(degraded.sum()),
        "median_delta_gap_A_at_most_plus_0p10dB": gap_medians["A"] <= GAP_DELTA_MAX_DB,
        "median_delta_gap_C_at_most_plus_0p10dB": gap_medians["C"] <= GAP_DELTA_MAX_DB,
    }
    return {
        "validation_pass": bool(all(criteria.values())),
        "criteria": criteria,
        "metric_medians_dB": metric_medians,
        "delta_metric_medians_dB": all_delta_medians,
        "delta_medians_dB": medians,
        "delta_gap_medians_dB": gap_medians,
        "worst4_improved_count": int(improved.sum()),
        "worst4_unchanged_count": int((frame["delta_worst4_dB"] == 0.0).sum()),
        "worst4_degraded_count": int(degraded.sum()),
    }


def _characterization_summary(frame: pd.DataFrame) -> dict[str, Any]:
    return {
        "state_count": int(frame.shape[0]),
        "delta_medians_dB": {
            key: float(frame[key].median())
            for key in ("delta_Aend_train_NMSE_dB", "delta_Aend_B_NMSE_dB", "delta_C2_train_NMSE_dB", "delta_C2_B_NMSE_dB", "delta_worst4_dB")
        },
        "worst4_improved_count": int((frame["delta_worst4_dB"] < 0).sum()),
        "worst4_degraded_count": int((frame["delta_worst4_dB"] > 0).sum()),
        "gap_delta_medians_dB": {
            "A": float(frame["delta_gap_A_dB"].median()),
            "C": float(frame["delta_gap_C_dB"].median()),
        },
    }


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", na_rep="NaN", float_format="%.17g")


def _summary_text(
    *,
    split_info: dict[str, Any],
    pool_info: dict[str, Any],
    model_gate: dict[str, Any],
    forward_summary: pd.DataFrame,
    selected_support: tuple[int, ...],
    forward_info: dict[str, Any],
    validation_info: dict[str, Any],
    validation_frame: pd.DataFrame,
    characterization_info: dict[str, Any] | None,
    protection: dict[str, Any],
    logical_cpus: int,
    workers: int,
) -> str:
    selected_g = int(forward_info["selected_G"])
    selected_terms = [TERM_BY_INDEX[index] for index in selected_support]
    lines = [
        TASK_NAME,
        "Global unified residual Sparse-GMP model-layer validation around Frozen MP.",
        "",
        "Scope",
        f"state_count=425; development_count={split_info['development_count']}; validation_count={split_info['validation_count']}",
        f"split_source={split_info['path']}",
        f"split_sha256={split_info['sha256']}",
        "failure14_used_for_support_selection=False",
        "LUT_retrieval=False; Real-B_for_support_selection=False; DPD_replay=False",
        "",
        "Frozen backbone",
        f"orders={list(ORDERS)}",
        f"memory={list(MEMORY)}",
        f"K_Frozen={FROZEN_K}; dmax={DMAX}; ridge_lambda={RIDGE_LAMBDA}",
        "raw-basis Ridge through augmented np.linalg.lstsq; no normalization, centering, lambda scan or memory scan",
        "",
        "GMP candidate pool",
        f"candidate_count={len(GLOBAL_TERMS)}; p={list(GMP_ORDERS)}; m_q={list(GMP_DELAYS)}; m_not_equal_q=True; common_dmax={DMAX}",
        f"old_G4_tuples_present={pool_info['old_g4_terms_present']}",
        "",
        "Frozen reproduction and G0 equivalence",
        f"pass={model_gate['pass']}",
        f"maximum_metric_abs_error_dB={model_gate['maximum_metric_abs_error_dB']:.17g}",
        f"maximum_theta_abs_error={model_gate['maximum_theta_abs_error']:.17g}",
        "",
        "Forward selection path",
        f"stages=G0...G8; candidate_trial_count={forward_info['trial_evaluation_count']}; estimated_CV_fits={forward_info['fit_estimate']}",
    ]
    for row in forward_summary.itertuples(index=False):
        lines.append(
            f"G{int(row.stage_G)}: K={int(row.K_total)}, added={row.added_term_id or 'None'}, "
            f"A_median={float(row.dev_A_median):.9f}, C_median={float(row.dev_C_median):.9f}, "
            f"balanced_median={float(row.balanced_median):.9f}, balanced_q75={float(row.balanced_q75):.9f}, "
            f"balanced_q90={float(row.balanced_q90):.9f}, final={bool(row.is_final_selected_G)}"
        )
    lines.extend(
        [
            f"selected_G={selected_g}; final_K={FROZEN_K + selected_g}",
            "selected_terms=" + ", ".join(f"{term.term_id}({term.expression})" for term in selected_terms) if selected_terms else "selected_terms=None; Frozen MP retained",
            "selection_rule=smallest G within best BalancedMedian + 0.10 dB",
            "",
            "Independent Validation",
            f"validation_pass={validation_info['validation_pass']}",
            f"validation_criteria={validation_info['criteria']}",
            f"validation_metric_medians_dB={validation_info['metric_medians_dB']}",
            f"validation_delta_metric_medians_dB={validation_info['delta_metric_medians_dB']}",
            f"validation_delta_medians_dB={validation_info['delta_medians_dB']}",
            f"validation_delta_gap_medians_dB={validation_info['delta_gap_medians_dB']}",
            f"validation_worst4_counts=improved:{validation_info['worst4_improved_count']}, unchanged:{validation_info['worst4_unchanged_count']}, degraded:{validation_info['worst4_degraded_count']}",
            "B_used_for_support_selection=False; Real_B_used_for_support_selection=False",
        ]
    )
    if characterization_info is not None:
        lines.extend(
            [
                "",
                "425-state final characterization (post-selection only)",
                f"state_count={characterization_info['state_count']}",
                f"delta_medians_dB={characterization_info['delta_medians_dB']}",
                f"worst4_improved_count={characterization_info['worst4_improved_count']}; worst4_degraded_count={characterization_info['worst4_degraded_count']}",
                f"gap_delta_medians_dB={characterization_info['gap_delta_medians_dB']}",
                "425-state characterization did not feed back into support selection",
            ]
        )
    lines.extend(
        [
            "",
            "Execution and protection",
            f"logical_cpus={logical_cpus}; workers={workers}; cpu_target={CPU_TARGET}; BLAS_threads_per_worker=1; nested_parallelism=False",
            "development_preprocessing=one-time A/C-only arrays in shared memory",
            f"protection={protection}",
            "",
            "Stopping boundary",
            "No LUT retrieval, no Real-B ranking, no Type III compression, no low-bandwidth processing, and no Frozen config replacement was performed.",
            "Frozen MP remains the formal model unless a later separately authorized task changes it.",
            "",
            "Output files",
            f"{RESULT_ROOT / 'forward_selection_summary.csv'}",
            f"{RESULT_ROOT / 'validation_per_state_metrics.csv'}",
        ]
    )
    if characterization_info is not None:
        lines.append(str(RESULT_ROOT / "all425_per_state_metrics.csv"))
    lines.append(str(RESULT_ROOT / "final_result_summary.txt"))
    return "\n".join(lines) + "\n"


def _append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"\n[{timestamp}] {TASK_NAME}\n{text}\n")


def main() -> None:
    freeze_support()
    pool_info = _validate_term_pool()
    development_ids, validation_ids, _, split_info = _load_split()
    if RESULT_ROOT.exists() and any(RESULT_ROOT.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty result directory: {RESULT_ROOT}")
    before = _snapshot()
    logical_cpus = int(os.cpu_count() or 1)
    workers = max(1, int(math.floor(CPU_TARGET * logical_cpus)))
    print(
        f"{TASK_NAME}: logical_cpus={logical_cpus}, workers={workers}, "
        f"development={len(development_ids)}, validation={len(validation_ids)}, GMP_candidates={len(GLOBAL_TERMS)}",
        flush=True,
    )
    model_gate = _frozen_reproduction_gate()
    if not model_gate["pass"]:
        raise RuntimeError(f"Frozen reproduction gate failed: {model_gate}")
    print("Frozen reproduction/G0 equivalence: PASS", flush=True)
    prepared = _prepare_development_arrays(development_ids, workers)
    print(
        f"Development A/C arrays prepared once; elapsed={prepared['elapsed_seconds']:.3f}s; "
        f"shapes={{{', '.join(f'{key}:{value.shape}' for key, value in prepared['arrays'].items())}}}",
        flush=True,
    )
    forward_summary, selected_support, _, forward_info = _run_forward_selection(
        development_ids,
        prepared["arrays"],
        workers,
    )
    print(
        f"Support frozen before Validation: G{forward_info['selected_G']}; "
        f"terms={[TERM_BY_INDEX[index].term_id for index in selected_support]}",
        flush=True,
    )
    validation_frame = _run_evaluation(validation_ids, selected_support, workers, "Validation Frozen vs Selected")
    validation_info = _validation_report(validation_frame)
    print(f"Independent Validation pass={validation_info['validation_pass']}", flush=True)
    characterization_frame: pd.DataFrame | None = None
    characterization_info: dict[str, Any] | None = None
    if validation_info["validation_pass"]:
        characterization_frame = _run_evaluation(tuple(range(STATE_COUNT)), selected_support, workers, "425-state characterization")
        split_lookup = pd.read_csv(SPLIT_PATH).set_index("state_id")["split"].astype(str).str.lower()
        characterization_frame.insert(1, "split_role", [split_lookup.loc[int(state_id)] for state_id in characterization_frame["state_id"]])
        characterization_info = _characterization_summary(characterization_frame)
    RESULT_ROOT.mkdir(parents=True, exist_ok=False)
    _write_frame(forward_summary, RESULT_ROOT / "forward_selection_summary.csv")
    _write_frame(validation_frame, RESULT_ROOT / "validation_per_state_metrics.csv")
    if characterization_frame is not None:
        _write_frame(characterization_frame, RESULT_ROOT / "all425_per_state_metrics.csv")
    after = _snapshot()
    protection = _protection_check(before, after)
    if not protection["all_protected_unchanged"]:
        raise RuntimeError(f"raw/protected data changed: {protection}")
    summary = _summary_text(
        split_info=split_info,
        pool_info=pool_info,
        model_gate=model_gate,
        forward_summary=forward_summary,
        selected_support=selected_support,
        forward_info=forward_info,
        validation_info=validation_info,
        validation_frame=validation_frame,
        characterization_info=characterization_info,
        protection=protection,
        logical_cpus=logical_cpus,
        workers=workers,
    )
    (RESULT_ROOT / "final_result_summary.txt").write_text(summary, encoding="utf-8")
    expected_files = 4 if characterization_frame is not None else 3
    actual_files = [path for path in RESULT_ROOT.iterdir() if path.is_file()]
    if len(actual_files) != expected_files:
        raise RuntimeError(f"unexpected result-file count: {len(actual_files)} != {expected_files}")
    _append_log(MODEL_LOG, summary)
    _append_log(
        HANDOFF_LOG,
        f"{TASK_NAME} completed: selected_G={forward_info['selected_G']}, "
        f"selected_terms={[TERM_BY_INDEX[index].term_id for index in selected_support]}, "
        f"validation_pass={validation_info['validation_pass']}, "
        f"characterization_425={characterization_frame is not None}, "
        f"raw/protected_unchanged={protection['all_protected_unchanged']}; "
        f"results={RESULT_ROOT}",
    )
    print(f"Result directory: {RESULT_ROOT}", flush=True)
    print(f"Validation pass: {validation_info['validation_pass']}; output files: {expected_files}", flush=True)


if __name__ == "__main__":
    main()
