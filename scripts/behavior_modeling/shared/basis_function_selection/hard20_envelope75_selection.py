# ruff: noqa: E501

"""Task-local implementation for Scenario 2 Envelope75 Hard-20 selection.

The module keeps the new forward-behavior experiment separate from the older
Hard-20, Gate-B48, Aend/C2, and strict-Volterra runners.  It reuses the
canonical ``data_management`` loader, full-record timing alignment, complex-gain
calibration, shared NMSE, and the validated OLS/Ridge solvers.

The selection workers receive only Train design banks and targets.  Test banks
are opened by a separate worker initializer after the model is frozen, making
the application-level Test-unlock rule explicit in the code.
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# These variables must be set before NumPy/SciPy is imported in spawned workers.
# ruff: noqa: E402
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np
import pandas as pd
from numpy.lib.format import open_memmap

MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from core.shared.metrics import nmse  # noqa: E402
from core.shared.project_paths import legacy_raw_manifest  # noqa: E402
from core.shared.signal import adjust_complex_gain, fine_align, rough_align  # noqa: E402
from data_management.shared import build_state_table, load_by_id  # noqa: E402

from behavior_modeling.shared.basis_function_selection.envelope_dictionary import (  # noqa: E402
    DMAX,
    EnvelopeBasis,
    build_envelope_bank,
    build_envelope_dictionary,
)
from behavior_modeling.shared.basis_function_selection.model_solver import (  # noqa: E402
    fit_ols,
    fit_ridge,
)

TASK_NAME = "scenario_2_hard20_envelope75_basis_selection_5B"
RESULT_ROOT = PROJECT_ROOT / "results" / "behavior_modeling" / "scenario_2" /TASK_NAME
WORK_LOG = PROJECT_ROOT / "work_logs" / "behavior_modeling" / "scenario_2" /TASK_NAME / "execution_log.txt"
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"

STATE_COUNT = 425
FULL_LENGTH = 24_576
TRAIN_START = 0
TRAIN_END = 16_384
TEST_START = 16_384
TEST_END = 24_576
TRAIN_LENGTH = TRAIN_END - TRAIN_START
TEST_LENGTH = TEST_END - TEST_START
HARD_STATE_COUNT = 20
TARGET_NMSE_DB = -40.0
FINE_ALIGN_SUBTIME = 256
WORKER_COUNT = 10
BLAS_THREADS_PER_WORKER = 1
GPU_USED = False
OLS_CONDITION_HARD_LIMIT = 1e10
K_SAFETY_CAP = 30
INNER_CV_FOLDS = 3

RIDGE_GRID = (
    0.0,
    1e-10,
    3e-10,
    1e-9,
    3e-9,
    1e-8,
    3e-8,
    1e-7,
    3e-7,
    1e-6,
)
RIDGE_BOUNDARY_LOW = 3e-11
RIDGE_BOUNDARY_HIGH = 3e-6
RIDGE_EXTRA_HIGH = 1e-5

BASELINE_SUPPORT_IDS = (
    "LIN_d0",
    "LIN_d1",
    "LIN_d2",
    "ENV_p02_m0_q0",
    "ENV_p02_m0_q1",
    "ENV_p02_m0_q2",
    "ENV_p02_m1_q0",
    "ENV_p03_m0_q0",
    "ENV_p03_m0_q1",
    "ENV_p03_m1_q0",
    "ENV_p03_m1_q1",
    "ENV_p05_m0_q0",
    "ENV_p05_m1_q0",
    "ENV_p07_m0_q0",
    "ENV_p09_m0_q0",
    "ENV_p09_m0_q1",
    "ENV_p09_m1_q1",
)

EXPECTED_RAW_MANIFEST = "910b06e4be6b60b1f4dc63409a3083eb328765347fbec3b89a97a41d6633a1e0"
EXPECTED_RAW_FILE_COUNT = 429
EXPECTED_RAW_MAT_COUNT = 427
EXPECTED_RAW_BYTES = 2_258_448_137


@dataclass(frozen=True)
class PreparedForwardState:
    """Full-record-aligned state with independently calibrated Train/Test targets."""

    state_id: int
    x_train: np.ndarray
    y_train_adjusted: np.ndarray
    x_test: np.ndarray
    y_test_adjusted: np.ndarray
    train_gain: complex
    test_gain: complex
    rough_delay: int
    fine_delay: float


@dataclass(frozen=True)
class TrainCacheSpec:
    """Paths and shapes for task-local Train-only memmaps plus final Test data."""

    root: str
    state_ids: tuple[int, ...]
    block_lengths: tuple[int, int, int]

    def path(self, name: str) -> Path:
        return Path(self.root) / f"{name}.npy"


def _raw_manifest() -> dict[str, object]:
    """Return the frozen raw-tree manifest used by prior project checks."""
    return legacy_raw_manifest()


def _as_vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1 or array.size != FULL_LENGTH or not np.iscomplexobj(array):
        raise ValueError(
            f"{name} must be a complex vector of length {FULL_LENGTH}, got {array.shape}"
        )
    array = np.asarray(array, dtype=np.complex128)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def _as_scalar(value: Any, name: str) -> float:
    array = np.asarray(value).reshape(-1)
    if array.size != 1 or not np.isfinite(array[0]):
        raise ValueError(f"{name} must be one finite scalar")
    return float(array[0])


def prepare_forward_state(state_id: int) -> PreparedForwardState:
    """Load, full-record-align, split, and independently calibrate one state."""

    normalized_id = int(state_id)
    data = load_by_id(normalized_id)
    xin = _as_vector(data["xin"], "xin")
    y_raw = _as_vector(data["yout_withoutdpd_ori"], "yout_withoutdpd_ori")
    sample_rate = _as_scalar(data["freqSample_Hz"], "freqSample_Hz")
    bandwidth_mhz = _as_scalar(data["bandWidth_MHz"], "bandWidth_MHz")
    if sample_rate != 100_000_000.0:
        raise ValueError(f"state {normalized_id} sample rate is not exactly 100 MHz")
    if bandwidth_mhz != 20.0:
        raise ValueError(f"state {normalized_id} bandwidth is not exactly 20 MHz")
    if sample_rate / (bandwidth_mhz * 1e6) != 5.0:
        raise ValueError(f"state {normalized_id} is not a 5B record")

    y_rough, rough_delay = rough_align(xin, y_raw)
    y_aligned, fine_delay = fine_align(xin, y_rough, subtime=FINE_ALIGN_SUBTIME)
    if y_aligned.shape != (FULL_LENGTH,) or not np.all(np.isfinite(y_aligned)):
        raise RuntimeError(f"state {normalized_id} full-record alignment produced invalid output")

    x_train = np.asarray(xin[TRAIN_START:TRAIN_END], dtype=np.complex128)
    y_train_aligned = np.asarray(y_aligned[TRAIN_START:TRAIN_END], dtype=np.complex128)
    x_test = np.asarray(xin[TEST_START:TEST_END], dtype=np.complex128)
    y_test_aligned = np.asarray(y_aligned[TEST_START:TEST_END], dtype=np.complex128)
    if x_train.size != TRAIN_LENGTH or x_test.size != TEST_LENGTH:
        raise RuntimeError(f"state {normalized_id} frozen Train/Test lengths are invalid")
    y_train_adjusted, train_gain = adjust_complex_gain(x_train, y_train_aligned)
    y_test_adjusted, test_gain = adjust_complex_gain(x_test, y_test_aligned)
    return PreparedForwardState(
        state_id=normalized_id,
        x_train=x_train,
        y_train_adjusted=np.asarray(y_train_adjusted, dtype=np.complex128),
        x_test=x_test,
        y_test_adjusted=np.asarray(y_test_adjusted, dtype=np.complex128),
        train_gain=complex(train_gain),
        test_gain=complex(test_gain),
        rough_delay=int(rough_delay),
        fine_delay=float(fine_delay),
    )


def prepare_ranking_row(state_id: int) -> dict[str, object]:
    """Prepare one state and compute only the Train no-DPD ranking metric."""

    prepared = prepare_forward_state(state_id)
    state = build_state_table()[int(state_id)]
    return {
        "state_id": int(state_id),
        "funMng": int(state["funMng"]),
        "funAng": int(state["funAng"]),
        "secMng": int(state["secMng"]),
        "secAng": int(state["secAng"]),
        "Vm": float(state["Vm"]),
        "Pin": float(state["Pin"]),
        "rough_delay": int(prepared.rough_delay),
        "fine_delay": float(prepared.fine_delay),
        "Train_gain_real": float(prepared.train_gain.real),
        "Train_gain_imag": float(prepared.train_gain.imag),
        "Test_gain_real": float(prepared.test_gain.real),
        "Test_gain_imag": float(prepared.test_gain.imag),
        "Train_noDPD_NMSE_dB": float(nmse(prepared.x_train, prepared.y_train_adjusted)),
    }


def _ranking_worker(state_id: int) -> dict[str, object]:
    return prepare_ranking_row(int(state_id))


def run_ranking() -> pd.DataFrame:
    """Rank all 425 states with exactly ten spawned worker processes."""

    rows: list[dict[str, object]] = []
    context = __import__("multiprocessing").get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=WORKER_COUNT,
        mp_context=context,
    ) as executor:
        futures = {
            executor.submit(_ranking_worker, state_id): state_id for state_id in range(STATE_COUNT)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if completed % 50 == 0 or completed == STATE_COUNT:
                print(f"[RANKING] {completed}/{STATE_COUNT}", flush=True)
    if len(rows) != STATE_COUNT:
        raise RuntimeError(f"ranking returned {len(rows)} rows, expected {STATE_COUNT}")
    frame = pd.DataFrame(rows).sort_values(
        ["Train_noDPD_NMSE_dB", "state_id"],
        ascending=[False, True],
        kind="mergesort",
    )
    frame.insert(0, "rank", np.arange(1, len(frame) + 1, dtype=np.int64))
    frame["selected_Hard20"] = frame["rank"] <= HARD_STATE_COUNT
    frame = frame.reset_index(drop=True)
    if int(frame["selected_Hard20"].sum()) != HARD_STATE_COUNT:
        raise RuntimeError("Hard-20 ranking did not select exactly 20 states")
    return frame


def _block_slices() -> tuple[slice, slice, slice]:
    blocks = np.array_split(np.arange(TRAIN_LENGTH), INNER_CV_FOLDS)
    slices: list[slice] = []
    for block in blocks:
        if block.size == 0 or not np.array_equal(block, np.arange(block[0], block[-1] + 1)):
            raise RuntimeError("Train CV block is not contiguous")
        slices.append(slice(int(block[0]), int(block[-1]) + 1))
    return slices[0], slices[1], slices[2]


def create_train_cache(root: Path, state_ids: Sequence[int]) -> TrainCacheSpec:
    """Create task-local memmaps; full design matrices stay outside results/."""

    normalized_ids = tuple(int(value) for value in state_ids)
    if len(normalized_ids) != HARD_STATE_COUNT or len(set(normalized_ids)) != HARD_STATE_COUNT:
        raise ValueError("Train cache requires exactly 20 unique Hard-20 state IDs")
    root.mkdir(parents=True, exist_ok=False)
    block_lengths = tuple(block.stop - block.start - DMAX for block in _block_slices())
    shapes = {
        "full_bank": (HARD_STATE_COUNT, TRAIN_LENGTH - DMAX, 75),
        "full_target": (HARD_STATE_COUNT, TRAIN_LENGTH - DMAX),
        "x_test": (HARD_STATE_COUNT, TEST_LENGTH),
        "y_test": (HARD_STATE_COUNT, TEST_LENGTH),
    }
    for block_index, block_length in enumerate(block_lengths):
        shapes[f"block{block_index}_bank"] = (HARD_STATE_COUNT, block_length, 75)
        shapes[f"block{block_index}_target"] = (HARD_STATE_COUNT, block_length)
    for name, shape in shapes.items():
        array = open_memmap(root / f"{name}.npy", mode="w+", dtype=np.complex128, shape=shape)
        array.flush()
        del array
    return TrainCacheSpec(
        root=str(root),
        state_ids=normalized_ids,
        block_lengths=block_lengths,
    )


def _populate_cache_state(
    spec: TrainCacheSpec, state_index: int, state_id: int
) -> dict[str, object]:
    """Populate one disjoint cache row; the worker returns metadata only."""

    prepared = prepare_forward_state(state_id)
    terms = build_envelope_dictionary()
    full_bank = build_envelope_bank(prepared.x_train, terms)
    full_target = prepared.y_train_adjusted[DMAX:]
    if full_bank.shape != (TRAIN_LENGTH - DMAX, len(terms)):
        raise RuntimeError(f"state {state_id} full bank shape is invalid: {full_bank.shape}")

    full_bank_file = open_memmap(spec.path("full_bank"), mode="r+")
    full_target_file = open_memmap(spec.path("full_target"), mode="r+")
    x_test_file = open_memmap(spec.path("x_test"), mode="r+")
    y_test_file = open_memmap(spec.path("y_test"), mode="r+")
    full_bank_file[state_index] = full_bank
    full_target_file[state_index] = full_target
    x_test_file[state_index] = prepared.x_test
    y_test_file[state_index] = prepared.y_test_adjusted
    block_slices = _block_slices()
    for block_index, block in enumerate(block_slices):
        block_bank = build_envelope_bank(prepared.x_train[block], terms)
        block_target = prepared.y_train_adjusted[block][DMAX:]
        block_bank_file = open_memmap(spec.path(f"block{block_index}_bank"), mode="r+")
        block_target_file = open_memmap(spec.path(f"block{block_index}_target"), mode="r+")
        block_bank_file[state_index] = block_bank
        block_target_file[state_index] = block_target
        block_bank_file.flush()
        block_target_file.flush()
        del block_bank_file, block_target_file
    for array in (full_bank_file, full_target_file, x_test_file, y_test_file):
        array.flush()
    return {
        "state_id": int(state_id),
        "state_index": int(state_index),
        "rough_delay": int(prepared.rough_delay),
        "fine_delay": float(prepared.fine_delay),
        "Train_gain_real": float(prepared.train_gain.real),
        "Train_gain_imag": float(prepared.train_gain.imag),
        "Test_gain_real": float(prepared.test_gain.real),
        "Test_gain_imag": float(prepared.test_gain.imag),
        "full_bank_finite": bool(np.all(np.isfinite(full_bank))),
    }


def populate_train_cache(spec: TrainCacheSpec) -> list[dict[str, object]]:
    """Populate the Hard-20 cache with the fixed ten-worker configuration."""

    context = __import__("multiprocessing").get_context("spawn")
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=WORKER_COUNT, mp_context=context) as executor:
        futures = {
            executor.submit(_populate_cache_state, spec, index, state_id): state_id
            for index, state_id in enumerate(spec.state_ids)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            print(f"[CACHE] {completed}/{len(futures)}", flush=True)
    if len(rows) != HARD_STATE_COUNT:
        raise RuntimeError("Hard-20 Train cache population is incomplete")
    return sorted(rows, key=lambda row: int(row["state_index"]))


_SELECTION_ARRAYS: dict[str, np.ndarray] | None = None
_FINAL_ARRAYS: dict[str, np.ndarray] | None = None
_FINAL_TERMS: tuple[EnvelopeBasis, ...] | None = None


def _open_selection_arrays(root: str) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {
        "full_bank": open_memmap(Path(root) / "full_bank.npy", mode="r"),
        "full_target": open_memmap(Path(root) / "full_target.npy", mode="r"),
    }
    for block_index in range(INNER_CV_FOLDS):
        arrays[f"block{block_index}_bank"] = open_memmap(
            Path(root) / f"block{block_index}_bank.npy", mode="r"
        )
        arrays[f"block{block_index}_target"] = open_memmap(
            Path(root) / f"block{block_index}_target.npy", mode="r"
        )
    if any("test" in name.lower() for name in arrays):
        raise RuntimeError("Train selection worker unexpectedly opened Test arrays")
    return arrays


def _selection_worker_init(root: str) -> None:
    global _SELECTION_ARRAYS
    _SELECTION_ARRAYS = _open_selection_arrays(root)


def _raw_condition(matrix: np.ndarray) -> float:
    singular = np.linalg.svd(np.asarray(matrix, dtype=np.complex128), compute_uv=False)
    if singular.size == 0 or singular[-1] <= 0.0:
        return float("inf")
    return float(singular[0] / singular[-1])


def _fit_matrix(
    matrix: np.ndarray,
    target: np.ndarray,
    ridge_lambda: float,
    *,
    exact_condition: bool,
) -> tuple[Any, float]:
    if ridge_lambda == 0.0:
        fitted = fit_ols(matrix, target, scale_columns=True)
        condition = _raw_condition(matrix) if exact_condition else fitted.condition_number
    else:
        fitted = fit_ridge(matrix, target, ridge_lambda)
        condition = _raw_condition(matrix) if exact_condition else fitted.condition_number
    return fitted, float(condition)


def _evaluate_train_support(
    state_id: int,
    full_bank: np.ndarray,
    full_target: np.ndarray,
    block_banks: Sequence[np.ndarray],
    block_targets: Sequence[np.ndarray],
    support: tuple[int, ...],
    ridge_lambda: float,
    exact_condition: bool,
) -> dict[str, object]:
    indices = np.asarray(support, dtype=np.int64)
    if indices.size == 0 or np.unique(indices).size != indices.size:
        raise ValueError("support must contain unique basis indices")
    full_phi = np.asarray(full_bank[:, indices], dtype=np.complex128)
    full_fit, full_condition = _fit_matrix(
        full_phi,
        full_target,
        ridge_lambda,
        exact_condition=exact_condition,
    )
    cv_values: list[float] = []
    conditions = [full_condition]
    rank_ratios = [full_fit.rank / indices.size]
    finite = bool(
        np.all(np.isfinite(full_fit.theta))
        and np.all(np.isfinite(full_fit.prediction))
        and np.isfinite(full_condition)
    )
    for validation_index in range(INNER_CV_FOLDS):
        train_phi = np.concatenate(
            [
                np.asarray(block_banks[index][:, indices], dtype=np.complex128)
                for index in range(INNER_CV_FOLDS)
                if index != validation_index
            ],
            axis=0,
        )
        train_target = np.concatenate(
            [
                np.asarray(block_targets[index], dtype=np.complex128)
                for index in range(INNER_CV_FOLDS)
                if index != validation_index
            ],
            axis=0,
        )
        validation_phi = np.asarray(block_banks[validation_index][:, indices], dtype=np.complex128)
        validation_target = np.asarray(block_targets[validation_index], dtype=np.complex128)
        fitted, condition = _fit_matrix(
            train_phi,
            train_target,
            ridge_lambda,
            exact_condition=exact_condition,
        )
        prediction = validation_phi @ fitted.theta
        value = float(nmse(validation_target, prediction))
        cv_values.append(value)
        conditions.append(condition)
        rank_ratios.append(fitted.rank / indices.size)
        finite = bool(
            finite
            and np.all(np.isfinite(fitted.theta))
            and np.all(np.isfinite(prediction))
            and np.isfinite(condition)
        )
    worst = float(max([full_fit.nmse_db, *cv_values]))
    return {
        "state_id": int(state_id),
        "train_nmse_db": float(full_fit.nmse_db),
        "cv1_nmse_db": cv_values[0],
        "cv2_nmse_db": cv_values[1],
        "cv3_nmse_db": cv_values[2],
        "W_db": worst,
        "full_rank": int(full_fit.rank),
        "min_rank_ratio": float(min(rank_ratios)),
        "condition_number": float(max(conditions)),
        "coefficient_norm": float(np.linalg.norm(full_fit.theta)),
        "finite": finite,
    }


def _selection_state_batch(
    state_index: int,
    state_id: int,
    supports: tuple[tuple[int, ...], ...],
    ridge_lambda: float,
    exact_condition: bool,
) -> list[dict[str, object]]:
    if _SELECTION_ARRAYS is None:
        raise RuntimeError("selection worker was not initialized")
    full_bank = _SELECTION_ARRAYS["full_bank"][state_index]
    full_target = _SELECTION_ARRAYS["full_target"][state_index]
    block_banks = [
        _SELECTION_ARRAYS[f"block{index}_bank"][state_index] for index in range(INNER_CV_FOLDS)
    ]
    block_targets = [
        _SELECTION_ARRAYS[f"block{index}_target"][state_index] for index in range(INNER_CV_FOLDS)
    ]
    rows = []
    for support_position, support in enumerate(supports):
        row = _evaluate_train_support(
            state_id,
            full_bank,
            full_target,
            block_banks,
            block_targets,
            support,
            ridge_lambda,
            exact_condition,
        )
        row["support_position"] = support_position
        rows.append(row)
    return rows


def _capacity_state_batch(
    state_index: int,
    state_id: int,
    supports: tuple[tuple[str, tuple[int, ...]], ...],
) -> list[dict[str, object]]:
    if _SELECTION_ARRAYS is None:
        raise RuntimeError("capacity worker was not initialized")
    full_bank = _SELECTION_ARRAYS["full_bank"][state_index]
    full_target = _SELECTION_ARRAYS["full_target"][state_index]
    rows = []
    for support_id, support in supports:
        indices = np.asarray(support, dtype=np.int64)
        phi = np.asarray(full_bank[:, indices], dtype=np.complex128)
        fitted = fit_ols(phi, full_target, scale_columns=True)
        rows.append(
            {
                "state_id": int(state_id),
                "support_id": support_id,
                "K": int(len(support)),
                "train_nmse_db": float(fitted.nmse_db),
                "rank": int(fitted.rank),
                "condition_number": float(fitted.condition_number),
                "raw_condition_number": _raw_condition(phi),
                "coefficient_norm": float(np.linalg.norm(fitted.theta)),
                "finite": bool(
                    np.all(np.isfinite(fitted.theta))
                    and np.all(np.isfinite(fitted.prediction))
                    and np.isfinite(fitted.condition_number)
                ),
                "train_pass40": bool(fitted.nmse_db < TARGET_NMSE_DB),
                "train_sse": float(np.sum(np.abs(full_target - fitted.prediction) ** 2)),
            }
        )
    return rows


def _aggregate_rows(
    rows: Sequence[dict[str, object]],
    support_id: str,
    support: tuple[int, ...],
    terms: Sequence[EnvelopeBasis],
    ridge_lambda: float,
) -> dict[str, object]:
    if len(rows) != HARD_STATE_COUNT:
        raise RuntimeError(f"support {support_id} returned {len(rows)} rows, expected 20")
    ordered = sorted(rows, key=lambda row: int(row["state_id"]))
    w_values = np.asarray([float(row["W_db"]) for row in ordered], dtype=np.float64)
    train_values = np.asarray([float(row["train_nmse_db"]) for row in ordered], dtype=np.float64)
    conditions = np.asarray([float(row["condition_number"]) for row in ordered], dtype=np.float64)
    deficits = np.maximum(0.0, w_values - TARGET_NMSE_DB)
    basis_ids = tuple(terms[index].basis_id for index in support)
    return {
        "support_id": support_id,
        "support_indices": ";".join(str(index) for index in support),
        "basis_ids": ";".join(basis_ids),
        "K": len(support),
        "ridge_lambda": float(ridge_lambda),
        "pass_count_W40": int(np.count_nonzero(w_values < TARGET_NMSE_DB)),
        "pass_count_Train40": int(np.count_nonzero(train_values < TARGET_NMSE_DB)),
        "W_median_dB": float(np.median(w_values)),
        "W_Q95_dB": float(np.quantile(w_values, 0.95)),
        "W_worst_dB": float(np.max(w_values)),
        "W_mean_dB": float(np.mean(w_values)),
        "W_Q95_deficit_dB": float(np.quantile(deficits, 0.95)),
        "W_mean_deficit_dB": float(np.mean(deficits)),
        "condition_q99": float(np.quantile(conditions, 0.99)),
        "condition_max": float(np.max(conditions)),
        "all_full_rank": bool(all(int(row["full_rank"]) == len(support) for row in ordered)),
        "all_finite": bool(all(bool(row["finite"]) for row in ordered)),
    }


class TrainSelectionEvaluator:
    """Persistent ten-worker, state-major evaluator with Train-only workers."""

    def __init__(self, spec: TrainCacheSpec, terms: Sequence[EnvelopeBasis]) -> None:
        self.spec = spec
        self.terms = tuple(terms)
        self.state_ids = tuple(spec.state_ids)
        self._cache: dict[
            tuple[tuple[int, ...], float, bool],
            tuple[dict[str, object], tuple[dict[str, object], ...]],
        ] = {}
        context = __import__("multiprocessing").get_context("spawn")
        self.executor = ProcessPoolExecutor(
            max_workers=WORKER_COUNT,
            mp_context=context,
            initializer=_selection_worker_init,
            initargs=(spec.root,),
        )

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=False)

    def __enter__(self) -> TrainSelectionEvaluator:
        return self

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def _evaluate_missing(
        self,
        missing: Sequence[tuple[int, ...]],
        ridge_lambda: float,
        exact_condition: bool,
        progress_label: str,
    ) -> None:
        missing = tuple(missing)
        by_position: list[list[dict[str, object]]] = [[] for _ in missing]
        futures = {
            self.executor.submit(
                _selection_state_batch,
                state_index,
                state_id,
                missing,
                float(ridge_lambda),
                bool(exact_condition),
            ): state_id
            for state_index, state_id in enumerate(self.state_ids)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            rows = future.result()
            for row in rows:
                by_position[int(row["support_position"])].append(row)
            if completed % 5 == 0 or completed == len(futures):
                print(f"[{progress_label}] {completed}/{len(futures)} states", flush=True)
        for support_position, support in enumerate(missing):
            rows = tuple(
                {key: value for key, value in row.items() if key != "support_position"}
                for row in sorted(
                    by_position[support_position], key=lambda item: int(item["state_id"])
                )
            )
            summary = _aggregate_rows(
                rows,
                support_id="internal",
                support=support,
                terms=self.terms,
                ridge_lambda=ridge_lambda,
            )
            self._cache[(support, float(ridge_lambda), bool(exact_condition))] = (summary, rows)

    def evaluate(
        self,
        specs: Sequence[tuple[str, tuple[int, ...]]],
        ridge_lambda: float,
        *,
        exact_condition: bool = False,
        progress_label: str = "selection",
    ) -> tuple[dict[str, dict[str, object]], dict[str, pd.DataFrame]]:
        normalized = [
            (str(label), tuple(sorted(int(index) for index in support))) for label, support in specs
        ]
        if not normalized:
            raise ValueError("support specs cannot be empty")
        unique_supports: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()
        for _, support in normalized:
            if not support or len(set(support)) != len(support):
                raise ValueError("support is empty or duplicated")
            if support not in seen:
                unique_supports.append(support)
                seen.add(support)
        missing = [
            support
            for support in unique_supports
            if (support, float(ridge_lambda), bool(exact_condition)) not in self._cache
        ]
        if missing:
            self._evaluate_missing(missing, ridge_lambda, exact_condition, progress_label)
        summaries: dict[str, dict[str, object]] = {}
        frames: dict[str, pd.DataFrame] = {}
        for label, support in normalized:
            summary, rows = self._cache[(support, float(ridge_lambda), bool(exact_condition))]
            summary_copy = dict(summary)
            summary_copy["support_id"] = label
            summaries[label] = summary_copy
            frame = pd.DataFrame(rows)
            frame.insert(0, "support_id", label)
            frame["ridge_lambda"] = float(ridge_lambda)
            frames[label] = frame
        return summaries, frames

    def evaluate_capacity(
        self,
        specs: Sequence[tuple[str, tuple[int, ...]]],
        *,
        progress_label: str = "capacity",
    ) -> pd.DataFrame:
        normalized = tuple((str(label), tuple(support)) for label, support in specs)
        futures = {
            self.executor.submit(
                _capacity_state_batch,
                state_index,
                state_id,
                normalized,
            ): state_id
            for state_index, state_id in enumerate(self.state_ids)
        }
        rows: list[dict[str, object]] = []
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.extend(future.result())
            if completed % 5 == 0 or completed == len(futures):
                print(f"[{progress_label}] {completed}/{len(futures)} states", flush=True)
        expected = len(self.state_ids) * len(normalized)
        if len(rows) != expected:
            raise RuntimeError(f"capacity returned {len(rows)} rows, expected {expected}")
        return pd.DataFrame(rows).sort_values(["support_id", "state_id"]).reset_index(drop=True)


def _final_worker_init(root: str, terms: tuple[EnvelopeBasis, ...]) -> None:
    """Open Test arrays only for the post-freeze final evaluator."""

    global _FINAL_ARRAYS, _FINAL_TERMS
    _FINAL_ARRAYS = {
        "full_bank": open_memmap(Path(root) / "full_bank.npy", mode="r"),
        "full_target": open_memmap(Path(root) / "full_target.npy", mode="r"),
        "x_test": open_memmap(Path(root) / "x_test.npy", mode="r"),
        "y_test": open_memmap(Path(root) / "y_test.npy", mode="r"),
    }
    _FINAL_TERMS = tuple(terms)


def _final_state_worker(
    state_index: int,
    state_id: int,
    support: tuple[int, ...],
    ridge_lambda: float,
) -> dict[str, object]:
    """Fit on full Train and score the previously locked Test segment."""

    if _FINAL_ARRAYS is None or _FINAL_TERMS is None:
        raise RuntimeError("final worker was not initialized")
    indices = np.asarray(support, dtype=np.int64)
    train_bank = _FINAL_ARRAYS["full_bank"][state_index]
    train_target = _FINAL_ARRAYS["full_target"][state_index]
    train_phi = np.asarray(train_bank[:, indices], dtype=np.complex128)
    fitted, condition = _fit_matrix(
        train_phi,
        train_target,
        ridge_lambda,
        exact_condition=True,
    )
    x_test = np.asarray(_FINAL_ARRAYS["x_test"][state_index], dtype=np.complex128)
    y_test = np.asarray(_FINAL_ARRAYS["y_test"][state_index], dtype=np.complex128)
    test_bank = build_envelope_bank(x_test, _FINAL_TERMS)
    test_prediction = test_bank[:, indices] @ fitted.theta
    test_target = y_test[DMAX:]
    return {
        "state_id": int(state_id),
        "train_nmse_db": float(fitted.nmse_db),
        "test_nmse_db": float(nmse(test_target, test_prediction)),
        "rank": int(fitted.rank),
        "condition_number": float(condition),
        "coefficient_norm": float(np.linalg.norm(fitted.theta)),
        "finite": bool(
            np.all(np.isfinite(fitted.theta))
            and np.all(np.isfinite(fitted.prediction))
            and np.all(np.isfinite(test_prediction))
            and np.isfinite(condition)
        ),
        "train_pass40": bool(fitted.nmse_db < TARGET_NMSE_DB),
        "test_pass40": bool(nmse(test_target, test_prediction) < TARGET_NMSE_DB),
    }


def run_final_evaluation(
    spec: TrainCacheSpec,
    terms: Sequence[EnvelopeBasis],
    support: Sequence[int],
    ridge_lambda: float,
) -> pd.DataFrame:
    """Run the first Test evaluation after support/lambda freeze."""

    normalized_support = tuple(sorted(int(index) for index in support))
    context = __import__("multiprocessing").get_context("spawn")
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(
        max_workers=WORKER_COUNT,
        mp_context=context,
        initializer=_final_worker_init,
        initargs=(spec.root, tuple(terms)),
    ) as executor:
        futures = {
            executor.submit(
                _final_state_worker,
                state_index,
                state_id,
                normalized_support,
                float(ridge_lambda),
            ): state_id
            for state_index, state_id in enumerate(spec.state_ids)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if completed % 5 == 0 or completed == len(futures):
                print(f"[FINAL TEST] {completed}/{len(futures)} states", flush=True)
    if len(rows) != HARD_STATE_COUNT:
        raise RuntimeError("final Hard-20 evaluation is incomplete")
    return pd.DataFrame(rows).sort_values("state_id").reset_index(drop=True)


def support_sort_key(summary: dict[str, object]) -> tuple[object, ...]:
    """Primary Hard-20 support objective; lower tuple values are better."""

    return (
        -int(summary["pass_count_W40"]),
        float(summary["W_worst_dB"]),
        float(summary["W_Q95_deficit_dB"]),
        float(summary["W_mean_deficit_dB"]),
        float(summary["W_median_dB"]),
        int(summary["K"]),
        str(summary["basis_ids"]),
    )


def support_improves(current: dict[str, object], candidate: dict[str, object]) -> bool:
    """Require a meaningful Train/CV improvement for forward/pair/swap steps."""

    current_pass = int(current["pass_count_W40"])
    candidate_pass = int(candidate["pass_count_W40"])
    if candidate_pass < current_pass:
        return False
    if candidate_pass > current_pass:
        return True
    worst_gain = float(current["W_worst_dB"]) - float(candidate["W_worst_dB"])
    q95_gain = float(current["W_Q95_dB"]) - float(candidate["W_Q95_dB"])
    mean_gain = float(current["W_mean_dB"]) - float(candidate["W_mean_dB"])
    return bool(worst_gain >= 0.05 or q95_gain >= 0.05 or mean_gain >= 0.02)


def deletion_allowed(current: dict[str, object], candidate: dict[str, object]) -> bool:
    """Apply the existing 0.02 dB backward tolerance to W metrics."""

    return bool(
        int(candidate["pass_count_W40"]) >= int(current["pass_count_W40"])
        and float(candidate["W_worst_dB"]) <= float(current["W_worst_dB"]) + 0.02
        and float(candidate["W_mean_dB"]) <= float(current["W_mean_dB"]) + 0.02
    )


def pareto_filter(summaries: Iterable[dict[str, object]]) -> list[dict[str, object]]:
    """Return deterministic nondominated summaries on pass/W/K axes."""

    unique: dict[str, dict[str, object]] = {str(item["support_id"]): item for item in summaries}
    values = list(unique.values())
    nondominated: list[dict[str, object]] = []
    for candidate in values:
        candidate_values = (
            -int(candidate["pass_count_W40"]),
            float(candidate["W_Q95_deficit_dB"]),
            float(candidate["W_worst_dB"]),
            int(candidate["K"]),
        )
        dominated = False
        for other in values:
            if other["support_id"] == candidate["support_id"]:
                continue
            other_values = (
                -int(other["pass_count_W40"]),
                float(other["W_Q95_deficit_dB"]),
                float(other["W_worst_dB"]),
                int(other["K"]),
            )
            if all(
                left <= right for left, right in zip(other_values, candidate_values, strict=True)
            ) and any(
                left < right for left, right in zip(other_values, candidate_values, strict=True)
            ):
                dominated = True
                break
        if not dominated:
            nondominated.append(candidate)
    return sorted(nondominated, key=support_sort_key)


def checkpoint(path: Path, **payload: object) -> None:
    """Write a compact JSON checkpoint for restart/audit inspection."""

    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )


def support_ids(terms: Sequence[EnvelopeBasis], support: Sequence[int]) -> tuple[str, ...]:
    return tuple(terms[int(index)].basis_id for index in support)


def baseline_support(terms: Sequence[EnvelopeBasis]) -> tuple[int, ...]:
    by_id = {term.basis_id: term.index for term in terms}
    missing = [basis_id for basis_id in BASELINE_SUPPORT_IDS if basis_id not in by_id]
    if missing:
        raise RuntimeError(f"Baseline17 basis IDs missing from Envelope75: {missing}")
    support = tuple(int(by_id[basis_id]) for basis_id in BASELINE_SUPPORT_IDS)
    if len(support) != 17 or len(set(support)) != 17:
        raise RuntimeError("Baseline17 is not a unique 17-term support")
    if tuple(terms[index].basis_id for index in support) != BASELINE_SUPPORT_IDS:
        raise RuntimeError("Baseline17 canonical order changed")
    if any(terms[index].order == 1 and not terms[index].mandatory for index in support):
        raise RuntimeError("Baseline17 lost a mandatory linear term")
    return support


def _write_task_definition(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                f"Task: {TASK_NAME}",
                "Mode: new independent research task inside behavior_modeling.basis_function_selection.",
                "Data: Scenario 2 425 states, xin -> yout_withoutdpd_ori.",
                "Preprocess: full-record rough/fine timing alignment, then Train/Test split.",
                "Gain: independent complex gain on Train and Test; Test score remains locked.",
                "Split: Train=[0,16384), Test=[16384,24576), dmax=2.",
                "Candidate: LIN_d0..d2 plus p=2..9, m/q=0..2 (Envelope75).",
                "Search: Full48/Full75 OLS capacity, Baseline17-centered Train/CV "
                "sparse search, Ridge.",
                "Workers: 10 spawned CPU processes, one BLAS thread per worker, "
                "no benchmark, no GPU.",
                "Stop boundary: do not run dmax=3, LUT retrieval, clustering, DPD, "
                "or low-bandwidth tasks.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


__all__ = [
    "BASELINE_SUPPORT_IDS",
    "BLAS_THREADS_PER_WORKER",
    "EXPECTED_RAW_BYTES",
    "EXPECTED_RAW_FILE_COUNT",
    "EXPECTED_RAW_MANIFEST",
    "EXPECTED_RAW_MAT_COUNT",
    "FULL_LENGTH",
    "HARD_STATE_COUNT",
    "INNER_CV_FOLDS",
    "K_SAFETY_CAP",
    "OLS_CONDITION_HARD_LIMIT",
    "PROJECT_ROOT",
    "RESULT_ROOT",
    "RIDGE_GRID",
    "STATE_COUNT",
    "TASK_NAME",
    "TARGET_NMSE_DB",
    "TEST_LENGTH",
    "TEST_START",
    "TRAIN_END",
    "TRAIN_LENGTH",
    "TRAIN_START",
    "TrainCacheSpec",
    "TrainSelectionEvaluator",
    "_aggregate_rows",
    "_raw_manifest",
    "_write_task_definition",
    "baseline_support",
    "build_envelope_dictionary",
    "checkpoint",
    "create_train_cache",
    "deletion_allowed",
    "pareto_filter",
    "populate_train_cache",
    "prepare_forward_state",
    "prepare_ranking_row",
    "run_ranking",
    "run_final_evaluation",
    "support_ids",
    "support_improves",
    "support_sort_key",
]
