"""Train-only dual ILC behavior preparation for shared Envelope75 selection.

The two behaviors are deliberately named ``ILC_END`` and ``ILC_COL2`` to
avoid confusing the literal second ILC column with the ABC segment named C.
This module does not import any LUT, fingerprint, clustering, or Ridge path.
During search it returns only full Train and segment-local Train-CV matrices.
Test arrays and Test gains are created only when ``include_test=True`` after
the caller has frozen the shared support.
"""

# ruff: noqa: E402,E501

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np

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

from core.shared.signal import adjust_complex_gain, fine_align, rough_align  # noqa: E402
from data_management.shared import load_by_id  # noqa: E402
from signal_segmentation.shared import get_ilc_pair  # noqa: E402

from behavior_modeling.shared.basis_function_selection.envelope_dictionary import (  # noqa: E402
    DMAX,
    EnvelopeBasis,
    build_envelope_bank,
)

ILC_END = "ILC_END"
ILC_COL2 = "ILC_COL2"
BEHAVIORS = (ILC_END, ILC_COL2)
FULL_LENGTH = 24_576
TRAIN_START = 0
TRAIN_END = 16_384
TEST_START = 16_384
TEST_END = 24_576
TRAIN_LENGTH = TRAIN_END - TRAIN_START
TEST_LENGTH = TEST_END - TEST_START
CV_BLOCK_RAW_LENGTHS = (5_461, 5_462, 5_461)
CV_BLOCK_VALID_LENGTHS = (5_459, 5_460, 5_459)
CV_FOLD_COUNT = 3
FINE_ALIGN_SUBTIME = 256


@dataclass(frozen=True)
class PreparedBehavior:
    """One behavior with Train matrices and optional post-freeze Test data."""

    state_id: int
    behavior: str
    ilc_column_index: int
    valid_ilc_count: int
    x_train: np.ndarray
    y_train_adjusted: np.ndarray
    full_bank: np.ndarray
    full_target: np.ndarray
    block_banks: tuple[np.ndarray, np.ndarray, np.ndarray]
    block_targets: tuple[np.ndarray, np.ndarray, np.ndarray]
    rough_delay: int
    fine_delay: float
    train_gain: complex
    x_test: np.ndarray | None
    y_test_adjusted: np.ndarray | None
    test_gain: complex | None


@dataclass(frozen=True)
class PreparedDualBehaviorState:
    """Both independent behaviors for one state."""

    state_id: int
    end: PreparedBehavior
    col2: PreparedBehavior

    def behavior(self, name: str) -> PreparedBehavior:
        if name == ILC_END:
            return self.end
        if name == ILC_COL2:
            return self.col2
        raise KeyError(f"unknown behavior: {name}")


def _as_history(value: Any, name: str) -> np.ndarray:
    history = np.asarray(value)
    if (
        history.ndim != 2
        or history.shape[0] != FULL_LENGTH
        or history.shape[1] < 1
        or not np.iscomplexobj(history)
        or not np.all(np.isfinite(history))
    ):
        raise ValueError(f"{name} must be finite complex shape (24576, n), got {history.shape}")
    return history


def _behavior_column(input_history: np.ndarray, behavior: str) -> int:
    if behavior == ILC_END:
        return int(input_history.shape[1] - 1)
    if behavior == ILC_COL2:
        if input_history.shape[1] < 2:
            raise ValueError("ILC_COL2 requires the literal MATLAB second column")
        return 1
    raise KeyError(f"unknown behavior: {behavior}")


def _block_slices() -> tuple[slice, slice, slice]:
    boundaries = (0, 5_461, 10_923, TRAIN_LENGTH)
    slices = tuple(
        slice(boundaries[index], boundaries[index + 1])
        for index in range(CV_FOLD_COUNT)
    )
    raw_lengths = tuple(item.stop - item.start for item in slices)
    if raw_lengths != CV_BLOCK_RAW_LENGTHS:
        raise RuntimeError(f"dual-behavior CV block lengths changed: {raw_lengths}")
    return slices  # type: ignore[return-value]


def _validate_full_pair(input_full: np.ndarray, output_full: np.ndarray, state_id: int) -> None:
    if (
        input_full.shape != (FULL_LENGTH,)
        or output_full.shape != (FULL_LENGTH,)
        or not np.iscomplexobj(input_full)
        or not np.iscomplexobj(output_full)
        or not np.all(np.isfinite(input_full))
        or not np.all(np.isfinite(output_full))
    ):
        raise RuntimeError(f"state {state_id} full ILC pair is invalid")


def _prepare_behavior(
    state_id: int,
    data: dict[str, Any],
    behavior: str,
    terms: tuple[EnvelopeBasis, ...],
    *,
    include_test: bool,
) -> PreparedBehavior:
    input_history = _as_history(data["xin_pd_ori_ilc"], "xin_pd_ori_ilc")
    output_history = _as_history(data["yout_withdpd_ori_ilc"], "yout_withdpd_ori_ilc")
    if input_history.shape != output_history.shape:
        raise RuntimeError(f"state {state_id} ILC input/output history shapes differ")
    column = _behavior_column(input_history, behavior)
    pair = get_ilc_pair(data, column)
    if pair.iteration_index != column:
        raise RuntimeError(f"state {state_id} {behavior} column contract failed")
    input_full = np.asarray(pair.input_full, dtype=np.complex128)
    output_raw = np.asarray(pair.output_raw_full, dtype=np.complex128)
    _validate_full_pair(input_full, output_raw, state_id)

    # Full-record timing alignment is completed before either split or gain.
    output_rough, rough_delay = rough_align(input_full, output_raw)
    output_aligned, fine_delay = fine_align(
        input_full,
        output_rough,
        subtime=FINE_ALIGN_SUBTIME,
    )
    if output_aligned.shape != (FULL_LENGTH,) or not np.all(np.isfinite(output_aligned)):
        raise RuntimeError(f"state {state_id} {behavior} full alignment is invalid")

    x_train = np.asarray(input_full[TRAIN_START:TRAIN_END], dtype=np.complex128)
    y_train_aligned = np.asarray(output_aligned[TRAIN_START:TRAIN_END], dtype=np.complex128)
    y_train_adjusted, train_gain = adjust_complex_gain(x_train, y_train_aligned)
    if x_train.shape != (TRAIN_LENGTH,) or y_train_adjusted.shape != (TRAIN_LENGTH,):
        raise RuntimeError(f"state {state_id} {behavior} Train shape is invalid")

    full_bank = np.asarray(build_envelope_bank(x_train, terms), dtype=np.complex128)
    full_target = np.asarray(y_train_adjusted[DMAX:], dtype=np.complex128)
    if full_bank.shape != (TRAIN_LENGTH - DMAX, len(terms)):
        raise RuntimeError(f"state {state_id} {behavior} Full75 Train shape is {full_bank.shape}")

    block_banks: list[np.ndarray] = []
    block_targets: list[np.ndarray] = []
    for block in _block_slices():
        block_bank = np.asarray(build_envelope_bank(x_train[block], terms), dtype=np.complex128)
        block_target = np.asarray(y_train_adjusted[block][DMAX:], dtype=np.complex128)
        block_banks.append(block_bank)
        block_targets.append(block_target)
    if tuple(bank.shape[0] for bank in block_banks) != CV_BLOCK_VALID_LENGTHS:
        raise RuntimeError(f"state {state_id} {behavior} local CV shapes are invalid")

    x_test = None
    y_test_adjusted = None
    test_gain = None
    if include_test:
        x_test = np.asarray(input_full[TEST_START:TEST_END], dtype=np.complex128)
        y_test_aligned = np.asarray(output_aligned[TEST_START:TEST_END], dtype=np.complex128)
        y_test_adjusted, test_gain = adjust_complex_gain(x_test, y_test_aligned)
        if x_test.shape != (TEST_LENGTH,) or y_test_adjusted.shape != (TEST_LENGTH,):
            raise RuntimeError(f"state {state_id} {behavior} Test shape is invalid")

    return PreparedBehavior(
        state_id=int(state_id),
        behavior=behavior,
        ilc_column_index=column,
        valid_ilc_count=int(input_history.shape[1]),
        x_train=x_train,
        y_train_adjusted=np.asarray(y_train_adjusted, dtype=np.complex128),
        full_bank=full_bank,
        full_target=full_target,
        block_banks=(block_banks[0], block_banks[1], block_banks[2]),
        block_targets=(block_targets[0], block_targets[1], block_targets[2]),
        rough_delay=int(rough_delay),
        fine_delay=float(fine_delay),
        train_gain=complex(train_gain),
        x_test=x_test,
        y_test_adjusted=y_test_adjusted,
        test_gain=None if test_gain is None else complex(test_gain),
    )


def prepare_dual_state(
    state_id: int,
    terms: tuple[EnvelopeBasis, ...],
    *,
    include_test: bool = False,
) -> PreparedDualBehaviorState:
    """Load one state once and prepare both ILC behaviors independently."""

    data = load_by_id(int(state_id))
    end = _prepare_behavior(
        int(state_id), data, ILC_END, terms, include_test=include_test
    )
    col2 = _prepare_behavior(
        int(state_id), data, ILC_COL2, terms, include_test=include_test
    )
    return PreparedDualBehaviorState(state_id=int(state_id), end=end, col2=col2)


def preflight_rows(prepared: PreparedDualBehaviorState) -> list[dict[str, object]]:
    """Return long-format Train-side metadata; Test fields are added post-freeze."""

    rows = []
    for behavior in BEHAVIORS:
        item = prepared.behavior(behavior)
        rows.append(
            {
                "State_ID": prepared.state_id,
                "behavior": behavior,
                "ILC_column_index": item.ilc_column_index,
                "valid_ilc_count": item.valid_ilc_count,
                "full_length": FULL_LENGTH,
                "full_delay_rough": item.rough_delay,
                "full_delay_fine": item.fine_delay,
                "Train_raw_length": TRAIN_LENGTH,
                "Train_valid_length": TRAIN_LENGTH - DMAX,
                "Test_raw_length": TEST_LENGTH,
                "Test_valid_length": TEST_LENGTH - DMAX,
                "Train_gain_real": float(item.train_gain.real),
                "Train_gain_imag": float(item.train_gain.imag),
                "Train_gain_abs": float(abs(item.train_gain)),
                "Train_gain_phase": float(np.angle(item.train_gain)),
                "Test_gain_real": None if item.test_gain is None else float(item.test_gain.real),
                "Test_gain_imag": None if item.test_gain is None else float(item.test_gain.imag),
                "Test_gain_abs": None if item.test_gain is None else float(abs(item.test_gain)),
                "Test_gain_phase": None if item.test_gain is None else float(np.angle(item.test_gain)),
                "finite": bool(
                    np.all(np.isfinite(item.x_train))
                    and np.all(np.isfinite(item.y_train_adjusted))
                    and np.all(np.isfinite(item.full_bank))
                    and np.all(np.isfinite(item.full_target))
                    and all(np.all(np.isfinite(value)) for value in item.block_banks)
                    and all(np.all(np.isfinite(value)) for value in item.block_targets)
                    and (
                        item.x_test is None
                        or np.all(np.isfinite(item.x_test))
                    )
                    and (
                        item.y_test_adjusted is None
                        or np.all(np.isfinite(item.y_test_adjusted))
                    )
                ),
            }
        )
    return rows


__all__ = [
    "BEHAVIORS",
    "CV_BLOCK_RAW_LENGTHS",
    "CV_BLOCK_VALID_LENGTHS",
    "CV_FOLD_COUNT",
    "DMAX",
    "FULL_LENGTH",
    "ILC_END",
    "ILC_COL2",
    "PreparedBehavior",
    "PreparedDualBehaviorState",
    "TEST_LENGTH",
    "TRAIN_LENGTH",
    "prepare_dual_state",
    "preflight_rows",
]
