"""Canonical raw no-DPD preparation and Hard-20 task-local memmap creation."""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from core.shared.metrics import nmse
from core.shared.signal import adjust_complex_gain, fine_align, rough_align
from data_management.shared import build_state_table, load_by_id
from numpy.lib.format import open_memmap

from .config import (
    CANDIDATE_COUNT,
    DMAX,
    FINE_ALIGN_SUBTIME,
    FULL_LENGTH,
    INNER_CV_FOLDS,
    PREPROCESS_WORKERS,
    STATE_COUNT,
    TEST_END,
    TEST_LENGTH,
    TEST_START,
    TRAIN_END,
    TRAIN_LENGTH,
    TRAIN_START,
)
from .volterra_dictionary import build_basis_bank, build_dictionary


@dataclass(frozen=True)
class AlignedState:
    state_id: int
    x_train: np.ndarray
    y_train_adjusted: np.ndarray
    x_test: np.ndarray
    y_test_time_aligned: np.ndarray
    train_gain: complex
    rough_delay: int
    fine_delay: float


@dataclass(frozen=True)
class TrainOnlyState:
    """Full-record-aligned state exposing only the frozen Train segment."""

    state_id: int
    x_train: np.ndarray
    y_train_adjusted: np.ndarray
    train_gain: complex
    rough_delay: int
    fine_delay: float


@dataclass(frozen=True)
class HardCacheSpec:
    root: str
    state_ids: tuple[int, ...]
    block_lengths: tuple[int, int, int]

    def path(self, name: str) -> Path:
        return Path(self.root) / f"{name}.npy"


def _vector(data: dict[str, Any], key: str) -> np.ndarray:
    if key not in data:
        raise KeyError(f"State data does not contain {key!r}")
    value = np.asarray(data[key])
    if value.ndim == 2 and 1 in value.shape:
        value = value.reshape(-1)
    if value.ndim != 1 or value.size != FULL_LENGTH:
        raise ValueError(f"{key} must be a length-{FULL_LENGTH} vector, got {value.shape}")
    if not np.iscomplexobj(value) or not np.all(np.isfinite(value)):
        raise ValueError(f"{key} must contain finite complex samples")
    return np.asarray(value, dtype=np.complex128)


def _scalar(data: dict[str, Any], key: str) -> float:
    if key not in data:
        raise KeyError(f"State data does not contain {key!r}")
    value = np.asarray(data[key]).reshape(-1)
    if value.size != 1 or not np.isfinite(value[0]):
        raise ValueError(f"{key} must be one finite scalar")
    return float(value[0])


def prepare_aligned_state(state_id: int) -> AlignedState:
    """Load raw signals, align the full record, split, and adjust Train only."""

    state_id = int(state_id)
    data = load_by_id(state_id)
    xin = _vector(data, "xin")
    y_raw = _vector(data, "yout_withoutdpd_ori")
    sample_rate = _scalar(data, "freqSample_Hz")
    bandwidth_mhz = _scalar(data, "bandWidth_MHz")
    if not np.isclose(sample_rate, 100_000_000.0, rtol=0.0, atol=0.0):
        raise ValueError(f"state {state_id} sample rate is not 100 MHz")
    if not np.isclose(bandwidth_mhz, 20.0, rtol=0.0, atol=0.0):
        raise ValueError(f"state {state_id} bandwidth is not 20 MHz")
    if not np.isclose(sample_rate / (bandwidth_mhz * 1e6), 5.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"state {state_id} is not a 5B record")

    y_rough, rough_delay = rough_align(xin, y_raw)
    y_aligned, fine_delay = fine_align(xin, y_rough, subtime=FINE_ALIGN_SUBTIME)
    if y_aligned.shape != (FULL_LENGTH,) or not np.all(np.isfinite(y_aligned)):
        raise RuntimeError(f"state {state_id} full-record alignment produced invalid output")

    x_train = np.asarray(xin[TRAIN_START:TRAIN_END], dtype=np.complex128)
    y_train_aligned = np.asarray(y_aligned[TRAIN_START:TRAIN_END], dtype=np.complex128)
    x_test = np.asarray(xin[TEST_START:TEST_END], dtype=np.complex128)
    y_test_time_aligned = np.asarray(y_aligned[TEST_START:TEST_END], dtype=np.complex128)
    if x_train.size != TRAIN_LENGTH or x_test.size != TEST_LENGTH:
        raise RuntimeError("Frozen Train/Test split is inconsistent")
    y_train_adjusted, train_gain = adjust_complex_gain(x_train, y_train_aligned)
    return AlignedState(
        state_id=state_id,
        x_train=x_train,
        y_train_adjusted=np.asarray(y_train_adjusted, dtype=np.complex128),
        x_test=x_test,
        y_test_time_aligned=y_test_time_aligned,
        train_gain=train_gain,
        rough_delay=int(rough_delay),
        fine_delay=float(fine_delay),
    )


def prepare_train_only_state(state_id: int) -> TrainOnlyState:
    """Align the full raw record, then return only Train and its local gain.

    Full-record alignment necessarily uses the entire capture.  This API does
    not slice, return, adjust, model, or score the frozen Test segment.
    """

    state_id = int(state_id)
    data = load_by_id(state_id)
    xin = _vector(data, "xin")
    y_raw = _vector(data, "yout_withoutdpd_ori")
    sample_rate = _scalar(data, "freqSample_Hz")
    bandwidth_mhz = _scalar(data, "bandWidth_MHz")
    if not np.isclose(sample_rate, 100_000_000.0, rtol=0.0, atol=0.0):
        raise ValueError(f"state {state_id} sample rate is not 100 MHz")
    if not np.isclose(bandwidth_mhz, 20.0, rtol=0.0, atol=0.0):
        raise ValueError(f"state {state_id} bandwidth is not 20 MHz")
    if not np.isclose(sample_rate / (bandwidth_mhz * 1e6), 5.0, rtol=0.0, atol=1e-12):
        raise ValueError(f"state {state_id} is not a 5B record")
    y_rough, rough_delay = rough_align(xin, y_raw)
    y_aligned, fine_delay = fine_align(xin, y_rough, subtime=FINE_ALIGN_SUBTIME)
    if y_aligned.shape != (FULL_LENGTH,) or not np.all(np.isfinite(y_aligned)):
        raise RuntimeError(f"state {state_id} full-record alignment produced invalid output")
    x_train = np.asarray(xin[TRAIN_START:TRAIN_END], dtype=np.complex128)
    y_train_aligned = np.asarray(y_aligned[TRAIN_START:TRAIN_END], dtype=np.complex128)
    if x_train.size != TRAIN_LENGTH or y_train_aligned.size != TRAIN_LENGTH:
        raise RuntimeError("Frozen Train split is inconsistent")
    y_train_adjusted, train_gain = adjust_complex_gain(x_train, y_train_aligned)
    return TrainOnlyState(
        state_id=state_id,
        x_train=x_train,
        y_train_adjusted=np.asarray(y_train_adjusted, dtype=np.complex128),
        train_gain=train_gain,
        rough_delay=int(rough_delay),
        fine_delay=float(fine_delay),
    )


def prepare_ranking_state(state_id: int) -> dict[str, object]:
    """Return one deterministic no-DPD Train-NMSE ranking row."""

    aligned = prepare_aligned_state(state_id)
    state_table = build_state_table()
    row = dict(state_table[int(state_id)])
    if int(row["state_id"]) != int(state_id):
        raise RuntimeError("State-table mapping changed unexpectedly")
    return {
        "state_id": int(state_id),
        "funMng": int(row["funMng"]),
        "funAng": int(row["funAng"]),
        "secMng": int(row["secMng"]),
        "secAng": int(row["secAng"]),
        "Vm": float(row["Vm"]),
        "Pin": float(row["Pin"]),
        "train_noDPD_NMSE_dB": float(nmse(aligned.x_train, aligned.y_train_adjusted)),
        "rough_delay": aligned.rough_delay,
        "fine_delay": aligned.fine_delay,
        "train_gain_real": float(aligned.train_gain.real),
        "train_gain_imag": float(aligned.train_gain.imag),
    }


def preprocess_all_for_ranking(worker_count: int = PREPROCESS_WORKERS) -> list[dict[str, object]]:
    """Prepare and rank-input all 425 states without returning waveform arrays."""

    worker_count = max(1, min(int(worker_count), STATE_COUNT))
    rows: dict[int, dict[str, object]] = {}
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        futures = {
            executor.submit(prepare_ranking_state, state_id): state_id
            for state_id in range(STATE_COUNT)
        }
        for done, future in enumerate(as_completed(futures), start=1):
            state_id = futures[future]
            rows[state_id] = future.result()
            if done % 50 == 0 or done == STATE_COUNT:
                print(f"[PREPROCESS] {done}/{STATE_COUNT}", flush=True)
    if sorted(rows) != list(range(STATE_COUNT)):
        raise RuntimeError("Preprocessing did not return all 425 states")
    return [rows[state_id] for state_id in range(STATE_COUNT)]


def _block_slices() -> tuple[slice, slice, slice]:
    indices = np.array_split(np.arange(TRAIN_LENGTH), INNER_CV_FOLDS)
    slices: list[slice] = []
    for index in indices:
        if index.size == 0 or not np.array_equal(index, np.arange(index[0], index[-1] + 1)):
            raise RuntimeError("Blocked CV split is not contiguous")
        slices.append(slice(int(index[0]), int(index[-1]) + 1))
    return (slices[0], slices[1], slices[2])


def create_hard_cache(scratch_root: Path, hard_state_ids: Sequence[int]) -> HardCacheSpec:
    """Create empty task-local memmaps for the selected Hard-20 states."""

    scratch_root.mkdir(parents=True, exist_ok=False)
    state_ids = tuple(int(value) for value in hard_state_ids)
    if len(state_ids) != 20 or len(set(state_ids)) != 20:
        raise ValueError("Hard cache requires exactly 20 unique state IDs")
    block_lengths = tuple(block.stop - block.start - DMAX for block in _block_slices())
    shapes = {
        "x_train": (20, TRAIN_LENGTH),
        "y_train": (20, TRAIN_LENGTH),
        "x_test": (20, TEST_LENGTH),
        "y_test_aligned": (20, TEST_LENGTH),
        "full_bank": (20, TRAIN_LENGTH - DMAX, CANDIDATE_COUNT),
        "full_target": (20, TRAIN_LENGTH - DMAX),
    }
    for block_index, block_length in enumerate(block_lengths):
        shapes[f"block{block_index}_bank"] = (20, block_length, CANDIDATE_COUNT)
        shapes[f"block{block_index}_target"] = (20, block_length)
    for name, shape in shapes.items():
        array = open_memmap(
            scratch_root / f"{name}.npy", mode="w+", dtype=np.complex128, shape=shape
        )
        array.flush()
        del array
    return HardCacheSpec(root=str(scratch_root), state_ids=state_ids, block_lengths=block_lengths)


def _write_hard_cache_state(
    spec: HardCacheSpec, state_index: int, state_id: int
) -> dict[str, object]:
    aligned = prepare_aligned_state(state_id)
    terms = build_dictionary()
    x_train_file = open_memmap(spec.path("x_train"), mode="r+")
    y_train_file = open_memmap(spec.path("y_train"), mode="r+")
    x_test_file = open_memmap(spec.path("x_test"), mode="r+")
    y_test_file = open_memmap(spec.path("y_test_aligned"), mode="r+")
    full_bank_file = open_memmap(spec.path("full_bank"), mode="r+")
    full_target_file = open_memmap(spec.path("full_target"), mode="r+")
    x_train_file[state_index] = aligned.x_train
    y_train_file[state_index] = aligned.y_train_adjusted
    x_test_file[state_index] = aligned.x_test
    y_test_file[state_index] = aligned.y_test_time_aligned
    full_bank = build_basis_bank(aligned.x_train, terms)
    full_bank_file[state_index] = full_bank
    full_target_file[state_index] = aligned.y_train_adjusted[DMAX:]
    block_slices = _block_slices()
    for block_index, block in enumerate(block_slices):
        x_block = aligned.x_train[block]
        y_block = aligned.y_train_adjusted[block]
        block_bank_file = open_memmap(spec.path(f"block{block_index}_bank"), mode="r+")
        block_target_file = open_memmap(spec.path(f"block{block_index}_target"), mode="r+")
        block_bank_file[state_index] = build_basis_bank(x_block, terms)
        block_target_file[state_index] = y_block[DMAX:]
        block_bank_file.flush()
        block_target_file.flush()
        del block_bank_file, block_target_file
    for array in (
        x_train_file,
        y_train_file,
        x_test_file,
        y_test_file,
        full_bank_file,
        full_target_file,
    ):
        array.flush()
    return {
        "state_id": int(state_id),
        "state_index": int(state_index),
        "rough_delay": aligned.rough_delay,
        "fine_delay": aligned.fine_delay,
        "train_gain_real": float(aligned.train_gain.real),
        "train_gain_imag": float(aligned.train_gain.imag),
        "full_bank_finite": bool(np.all(np.isfinite(full_bank))),
    }


def populate_hard_cache(spec: HardCacheSpec, worker_count: int) -> list[dict[str, object]]:
    """Populate disjoint memmap rows in parallel without pickling waveform arrays."""

    rows: dict[int, dict[str, object]] = {}
    with ProcessPoolExecutor(max_workers=max(1, int(worker_count))) as executor:
        futures = {
            executor.submit(_write_hard_cache_state, spec, state_index, state_id): state_id
            for state_index, state_id in enumerate(spec.state_ids)
        }
        for done, future in enumerate(as_completed(futures), start=1):
            state_id = futures[future]
            rows[state_id] = future.result()
            print(f"[HARD20 CACHE] {done}/20", flush=True)
    if set(rows) != set(spec.state_ids):
        raise RuntimeError("Hard cache population is incomplete")
    return [rows[state_id] for state_id in spec.state_ids]


def load_hard_cache_arrays(spec: HardCacheSpec) -> dict[str, np.ndarray]:
    """Open all cache arrays read-only for a worker or final evaluator."""

    arrays: dict[str, np.ndarray] = {}
    for name in ("x_train", "y_train", "x_test", "y_test_aligned", "full_bank", "full_target"):
        arrays[name] = open_memmap(spec.path(name), mode="r")
    for block_index in range(3):
        arrays[f"block{block_index}_bank"] = open_memmap(
            spec.path(f"block{block_index}_bank"), mode="r"
        )
        arrays[f"block{block_index}_target"] = open_memmap(
            spec.path(f"block{block_index}_target"), mode="r"
        )
    return arrays


def load_selection_cache_arrays(spec: HardCacheSpec) -> dict[str, np.ndarray]:
    """Open Train/full/CV arrays only; intentionally exclude every Test array."""

    arrays: dict[str, np.ndarray] = {}
    for name in ("full_bank", "full_target"):
        arrays[name] = open_memmap(spec.path(name), mode="r")
    for block_index in range(3):
        arrays[f"block{block_index}_bank"] = open_memmap(
            spec.path(f"block{block_index}_bank"), mode="r"
        )
        arrays[f"block{block_index}_target"] = open_memmap(
            spec.path(f"block{block_index}_target"), mode="r"
        )
    if any("test" in name.lower() for name in arrays):
        raise RuntimeError("Selection cache unexpectedly exposes Test data")
    return arrays


def spec_as_dict(spec: HardCacheSpec) -> dict[str, object]:
    return asdict(spec)


__all__ = [
    "AlignedState",
    "HardCacheSpec",
    "TrainOnlyState",
    "create_hard_cache",
    "load_hard_cache_arrays",
    "load_selection_cache_arrays",
    "populate_hard_cache",
    "prepare_aligned_state",
    "prepare_ranking_state",
    "prepare_train_only_state",
    "preprocess_all_for_ranking",
    "spec_as_dict",
]
