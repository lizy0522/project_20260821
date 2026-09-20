"""Task-local ilc_end data provider and Hard-20 Train-cache population.

The provider is intentionally different from the no-DPD preparation helper:
it selects each state's last valid ``xin_pd_ori_ilc`` /
``yout_withdpd_ori_ilc`` column, aligns that full 24576-sample pair once, and
only then applies the frozen Train/Test split and independent gains.
"""

from __future__ import annotations

import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Must be set before NumPy/SciPy imports in spawned workers.
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

from core.shared.signal import adjust_complex_gain, fine_align, rough_align  # noqa: E402
from data_management.shared import load_by_id  # noqa: E402
from signal_segmentation.shared import get_ilc_pair  # noqa: E402

from behavior_modeling.shared.basis_function_selection.envelope_dictionary import (  # noqa: E402
    DMAX,
    build_envelope_bank,
    build_envelope_dictionary,
)
from behavior_modeling.shared.basis_function_selection.hard20_envelope75_selection import (  # noqa: E402
    FINE_ALIGN_SUBTIME,
    FULL_LENGTH,
    TEST_END,
    TEST_LENGTH,
    TEST_START,
    TRAIN_END,
    TRAIN_LENGTH,
    _block_slices,
)


@dataclass(frozen=True)
class PreparedIlcEndState:
    state_id: int
    valid_ilc_count: int
    ilc_end_index: int
    x_train: np.ndarray
    y_train_adjusted: np.ndarray
    x_test: np.ndarray
    y_test_adjusted: np.ndarray
    train_gain: complex
    test_gain: complex
    rough_delay: int
    fine_delay: float
    nodpd_nmse_db: float


def _vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1 or array.size != FULL_LENGTH or not np.iscomplexobj(array):
        raise ValueError(f"{name} must be complex vector length {FULL_LENGTH}, got {array.shape}")
    array = np.asarray(array, dtype=np.complex128)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def _prepare_from_pair(state_id: int, data: dict[str, Any]) -> PreparedIlcEndState:
    _vector(data["xin"], "xin")
    input_history = np.asarray(data["xin_pd_ori_ilc"])
    output_history = np.asarray(data["yout_withdpd_ori_ilc"])
    if (
        input_history.ndim != 2
        or output_history.ndim != 2
        or input_history.shape != output_history.shape
        or input_history.shape[0] != FULL_LENGTH
        or input_history.shape[1] < 1
    ):
        raise ValueError(f"state {state_id} ILC history shape is invalid")
    if not np.iscomplexobj(input_history) or not np.iscomplexobj(output_history):
        raise ValueError(f"state {state_id} ILC history must be complex")
    if not np.all(np.isfinite(input_history)) or not np.all(np.isfinite(output_history)):
        raise ValueError(f"state {state_id} ILC history contains NaN/Inf")
    valid_count = int(input_history.shape[1])
    ilc_end_index = valid_count - 1
    pair = get_ilc_pair(data, ilc_end_index)
    y_rough, rough_delay = rough_align(pair.input_full, pair.output_raw_full)
    y_aligned, fine_delay = fine_align(
        pair.input_full,
        y_rough,
        subtime=FINE_ALIGN_SUBTIME,
    )
    if y_aligned.shape != (FULL_LENGTH,) or not np.all(np.isfinite(y_aligned)):
        raise RuntimeError(f"state {state_id} ilc_end full alignment is invalid")
    x_train = np.asarray(pair.input_full[:TRAIN_END], dtype=np.complex128)
    y_train_aligned = np.asarray(y_aligned[:TRAIN_END], dtype=np.complex128)
    x_test = np.asarray(pair.input_full[TEST_START:TEST_END], dtype=np.complex128)
    y_test_aligned = np.asarray(y_aligned[TEST_START:TEST_END], dtype=np.complex128)
    if x_train.size != TRAIN_LENGTH or x_test.size != TEST_LENGTH:
        raise RuntimeError(f"state {state_id} Train/Test raw lengths are invalid")
    y_train_adjusted, train_gain = adjust_complex_gain(x_train, y_train_aligned)
    y_test_adjusted, test_gain = adjust_complex_gain(x_test, y_test_aligned)
    nmse_value = np.asarray(data["nmse_withoutdpd"]).reshape(-1)
    if nmse_value.size != 1 or not np.isfinite(nmse_value[0]):
        raise ValueError(f"state {state_id} nmse_withoutdpd is invalid")
    return PreparedIlcEndState(
        state_id=int(state_id),
        valid_ilc_count=valid_count,
        ilc_end_index=ilc_end_index,
        x_train=x_train,
        y_train_adjusted=np.asarray(y_train_adjusted, dtype=np.complex128),
        x_test=x_test,
        y_test_adjusted=np.asarray(y_test_adjusted, dtype=np.complex128),
        train_gain=complex(train_gain),
        test_gain=complex(test_gain),
        rough_delay=int(rough_delay),
        fine_delay=float(fine_delay),
        nodpd_nmse_db=float(nmse_value[0]),
    )


def prepare_ilcend_state(state_id: int) -> PreparedIlcEndState:
    return _prepare_from_pair(int(state_id), load_by_id(int(state_id)))


def _preflight_worker(state_id: int) -> dict[str, object]:
    prepared = prepare_ilcend_state(int(state_id))
    return {
        "State_ID": prepared.state_id,
        "valid_ilc_count": prepared.valid_ilc_count,
        "ilc_end_index": prepared.ilc_end_index,
        "full_length": FULL_LENGTH,
        "full_delay_rough": prepared.rough_delay,
        "full_delay_fine": prepared.fine_delay,
        "Train_raw_length": TRAIN_LENGTH,
        "Train_valid_length": TRAIN_LENGTH - DMAX,
        "Test_raw_length": TEST_LENGTH,
        "Test_valid_length": TEST_LENGTH - DMAX,
        "Train_gain_real": float(prepared.train_gain.real),
        "Train_gain_imag": float(prepared.train_gain.imag),
        "Train_gain_abs": float(abs(prepared.train_gain)),
        "Train_gain_phase": float(np.angle(prepared.train_gain)),
        "Test_gain_real": float(prepared.test_gain.real),
        "Test_gain_imag": float(prepared.test_gain.imag),
        "Test_gain_abs": float(abs(prepared.test_gain)),
        "Test_gain_phase": float(np.angle(prepared.test_gain)),
        "finite": bool(
            np.all(np.isfinite(prepared.x_train))
            and np.all(np.isfinite(prepared.y_train_adjusted))
            and np.all(np.isfinite(prepared.x_test))
            and np.all(np.isfinite(prepared.y_test_adjusted))
        ),
    }


def run_ilcend_preflight(state_ids: range | list[int] | tuple[int, ...]) -> pd.DataFrame:
    ids = tuple(int(value) for value in state_ids)
    context = __import__("multiprocessing").get_context("spawn")
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=10, mp_context=context) as executor:
        futures = {executor.submit(_preflight_worker, state_id): state_id for state_id in ids}
        for future in as_completed(futures):
            rows.append(future.result())
    frame = pd.DataFrame(rows).sort_values("State_ID").reset_index(drop=True)
    if frame["State_ID"].tolist() != list(ids):
        raise RuntimeError("ilc_end preflight does not cover requested canonical State IDs")
    if not bool(frame["finite"].all()):
        raise RuntimeError("ilc_end preflight finite gate failed")
    return frame


def _cache_worker(spec: Any, state_index: int, state_id: int) -> dict[str, object]:
    prepared = prepare_ilcend_state(int(state_id))
    terms = tuple(build_envelope_dictionary())
    full_bank = build_envelope_bank(prepared.x_train, terms)
    full_target = prepared.y_train_adjusted[DMAX:]
    full_bank_file = open_memmap(Path(spec.root) / "full_bank.npy", mode="r+")
    full_target_file = open_memmap(Path(spec.root) / "full_target.npy", mode="r+")
    x_test_file = open_memmap(Path(spec.root) / "x_test.npy", mode="r+")
    y_test_file = open_memmap(Path(spec.root) / "y_test.npy", mode="r+")
    full_bank_file[state_index] = full_bank
    full_target_file[state_index] = full_target
    x_test_file[state_index] = prepared.x_test
    y_test_file[state_index] = prepared.y_test_adjusted
    for block_index, block in enumerate(_block_slices()):
        block_bank = build_envelope_bank(prepared.x_train[block], terms)
        block_target = prepared.y_train_adjusted[block][DMAX:]
        bank_file = open_memmap(Path(spec.root) / f"block{block_index}_bank.npy", mode="r+")
        target_file = open_memmap(Path(spec.root) / f"block{block_index}_target.npy", mode="r+")
        bank_file[state_index] = block_bank
        target_file[state_index] = block_target
        bank_file.flush()
        target_file.flush()
        del bank_file, target_file
    for array in (full_bank_file, full_target_file, x_test_file, y_test_file):
        array.flush()
    return {
        "state_id": int(state_id),
        "state_index": int(state_index),
        "valid_ilc_count": prepared.valid_ilc_count,
        "ilc_end_index": prepared.ilc_end_index,
        "rough_delay": prepared.rough_delay,
        "fine_delay": prepared.fine_delay,
        "Train_gain_real": float(prepared.train_gain.real),
        "Train_gain_imag": float(prepared.train_gain.imag),
        "Test_gain_real": float(prepared.test_gain.real),
        "Test_gain_imag": float(prepared.test_gain.imag),
        "full_bank_finite": bool(np.all(np.isfinite(full_bank))),
    }


def populate_ilcend_cache(spec: Any) -> list[dict[str, object]]:
    context = __import__("multiprocessing").get_context("spawn")
    rows: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=10, mp_context=context) as executor:
        futures = {
            executor.submit(_cache_worker, spec, state_index, state_id): state_id
            for state_index, state_id in enumerate(spec.state_ids)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            print(f"[ILCEND CACHE] {completed}/{len(futures)}", flush=True)
    return sorted(rows, key=lambda row: int(row["state_index"]))


__all__ = [
    "PreparedIlcEndState",
    "populate_ilcend_cache",
    "prepare_ilcend_state",
    "run_ilcend_preflight",
]
