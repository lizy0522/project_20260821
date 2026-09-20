# ruff: noqa: E402,E501,I001

"""One-read-per-state K10/K9 model phase for the MP10 ablation task.

The historical MP10 basis and Ridge implementation are imported from the
previously validated ``scenario_2_mp10_full_lut_retrieval_5b`` backend.  This
module only adds candidate support slicing and returns in-memory model-phase
records.  It does not implement CNMSE, Top-1 retrieval, or Real-B selection.
"""

from __future__ import annotations

import os
import sys
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

from core.shared.metrics import nmse  # noqa: E402
from behavior_modeling.shared import sparse_gmp as historical_mp10  # noqa: E402
from data_management.shared import build_state_table, load_by_id  # noqa: E402
from signal_segmentation.shared import (  # noqa: E402
    build_off_segments,
    build_partition_from_xin,
    get_ilc_pair,
    preprocess_full_pair,
)

STATE_COUNT = 425
FULL_LENGTH = 24_576
ABC_LENGTHS = (12_288, 4_915, 7_373)
MP_K = int(historical_mp10.FROZEN_MP_K)
MP_DMAX = int(historical_mp10.MP_MAX_DELAY)
FINGERPRINT_LENGTH = ABC_LENGTHS[1] - MP_DMAX
COMMON_B_RAW_LENGTH = ABC_LENGTHS[1]
RIDGE_LAMBDA = float(historical_mp10.RIDGE_LAMBDA)

_WORKER_CANDIDATES: tuple[tuple[int, tuple[int, ...]], ...] | None = None
_WORKER_STATE_TABLE: dict[int, dict[str, int | float]] | None = None


def _vector(value: Any, name: str, expected_size: int = FULL_LENGTH) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1 or array.size != expected_size or not np.iscomplexobj(array):
        raise ValueError(f"{name} must be a complex length-{expected_size} vector")
    array = np.asarray(array, dtype=np.complex128)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN/Inf")
    return array


def _scalar(value: Any, name: str) -> float:
    array = np.asarray(value).reshape(-1)
    if array.size != 1 or not np.isfinite(array[0]):
        raise ValueError(f"{name} must be one finite scalar")
    return float(array[0])


def _fit_one_candidate(
    *,
    candidate_id: int,
    support: tuple[int, ...],
    phi_a_full: np.ndarray,
    y_a: np.ndarray,
    phi_a_b_full: np.ndarray,
    y_a_b: np.ndarray,
    phi_c_full: np.ndarray,
    y_c: np.ndarray,
    phi_c_b_full: np.ndarray,
    y_c_b: np.ndarray,
) -> dict[str, Any]:
    phi_a = np.asarray(phi_a_full[:, support], dtype=np.complex128)
    phi_c = np.asarray(phi_c_full[:, support], dtype=np.complex128)
    phi_a_b = np.asarray(phi_a_b_full[:, support], dtype=np.complex128)
    phi_c_b = np.asarray(phi_c_b_full[:, support], dtype=np.complex128)
    theta_a, diagnostics_a = historical_mp10.fit_ridge(phi_a, y_a, RIDGE_LAMBDA)
    theta_c, diagnostics_c = historical_mp10.fit_ridge(phi_c, y_c, RIDGE_LAMBDA)
    prediction_a = phi_a @ theta_a
    prediction_c = phi_c @ theta_c
    prediction_a_b = phi_a_b @ theta_a
    prediction_c_b = phi_c_b @ theta_c
    finite = bool(
        np.all(np.isfinite(phi_a))
        and np.all(np.isfinite(phi_c))
        and np.all(np.isfinite(theta_a))
        and np.all(np.isfinite(theta_c))
        and np.all(np.isfinite(prediction_a))
        and np.all(np.isfinite(prediction_c))
        and np.all(np.isfinite(prediction_a_b))
        and np.all(np.isfinite(prediction_c_b))
    )
    return {
        "candidate_id": int(candidate_id),
        "theta_Aend": np.asarray(theta_a, dtype=np.complex128),
        "theta_C2": np.asarray(theta_c, dtype=np.complex128),
        "Y_Aend_train_NMSE_dB": float(nmse(y_a, prediction_a)),
        "Y_Aend_B_NMSE_dB": float(nmse(y_a_b, prediction_a_b)),
        "Y_C2_train_NMSE_dB": float(nmse(y_c, prediction_c)),
        "Y_C2_B_NMSE_dB": float(nmse(y_c_b, prediction_c_b)),
        "Y_Aend_rank": int(diagnostics_a.rank_phi),
        "Y_Aend_rank_augmented": int(diagnostics_a.rank_augmented),
        "Y_Aend_condition_number": float(diagnostics_a.condition_number_phi),
        "Y_Aend_condition_number_augmented": float(
            diagnostics_a.condition_number_augmented
        ),
        "Y_C2_rank": int(diagnostics_c.rank_phi),
        "Y_C2_rank_augmented": int(diagnostics_c.rank_augmented),
        "Y_C2_condition_number": float(diagnostics_c.condition_number_phi),
        "Y_C2_condition_number_augmented": float(
            diagnostics_c.condition_number_augmented
        ),
        "finite": finite,
    }


def worker_init(
    candidate_specs: tuple[tuple[int, tuple[int, ...]], ...],
) -> None:
    global _WORKER_CANDIDATES, _WORKER_STATE_TABLE
    _WORKER_CANDIDATES = tuple(
        (int(candidate_id), tuple(int(index) for index in support))
        for candidate_id, support in candidate_specs
    )
    if len(_WORKER_CANDIDATES) != 11:
        raise RuntimeError("the ablation model phase requires one K10 and ten K9 candidates")
    if _WORKER_CANDIDATES[0][1] != tuple(range(MP_K)):
        raise RuntimeError("candidate 0 is not the complete historical MP10 support")
    if any(len(support) not in (9, 10) for _, support in _WORKER_CANDIDATES):
        raise RuntimeError("candidate support sizes must be K10 or K9")
    _WORKER_STATE_TABLE = {int(row["state_id"]): row for row in build_state_table()}


def state_worker(state_id: int) -> dict[str, Any]:
    if _WORKER_CANDIDATES is None or _WORKER_STATE_TABLE is None:
        raise RuntimeError("MP10 K9 worker is not initialized")
    normalized_id = int(state_id)
    data = load_by_id(normalized_id)
    xin = _vector(data["xin"], "xin")
    _vector(data["yout_withoutdpd_ori"], "yout_withoutdpd_ori")
    if _scalar(data["freqSample_Hz"], "freqSample_Hz") != 100_000_000.0:
        raise RuntimeError(f"state {normalized_id} sample rate is not 100 MHz")
    if _scalar(data["bandWidth_MHz"], "bandWidth_MHz") != 20.0:
        raise RuntimeError(f"state {normalized_id} bandwidth is not 20 MHz")
    partition = build_partition_from_xin(xin)
    if (partition.n_a, partition.n_b, partition.n_c) != ABC_LENGTHS:
        raise RuntimeError(f"state {normalized_id} ABC ownership changed")
    input_history = np.asarray(data["xin_pd_ori_ilc"])
    output_history = np.asarray(data["yout_withdpd_ori_ilc"])
    if input_history.ndim != 2 or output_history.shape != input_history.shape:
        raise RuntimeError(f"state {normalized_id} ILC history shape is invalid")
    if input_history.shape[1] < 2:
        raise RuntimeError(f"state {normalized_id} has no valid C2 input")

    aend_index = int(input_history.shape[1] - 1)
    aend_pair = get_ilc_pair(data, aend_index)
    c2_pair = get_ilc_pair(data, 1)
    aend = preprocess_full_pair(
        aend_pair.input_full,
        aend_pair.output_raw_full,
        partition,
        pair_type=aend_pair.pair_type,
        iteration_index=aend_pair.iteration_index,
        input_peak_normalization_factor=aend_pair.input_peak_normalization_factor,
    )
    c2 = preprocess_full_pair(
        c2_pair.input_full,
        c2_pair.output_raw_full,
        partition,
        pair_type=c2_pair.pair_type,
        iteration_index=c2_pair.iteration_index,
        input_peak_normalization_factor=c2_pair.input_peak_normalization_factor,
    )
    off = build_off_segments(data, partition)

    phi_a_full = historical_mp10.build_frozen_mp_basis(aend["A"].input)
    phi_a_b_full = historical_mp10.build_frozen_mp_basis(aend["B"].input)
    phi_c_full = historical_mp10.build_frozen_mp_basis(c2["C"].input)
    phi_c_b_full = historical_mp10.build_frozen_mp_basis(c2["B"].input)
    y_a = np.asarray(aend["A"].output, dtype=np.complex128)[MP_DMAX:]
    y_a_b = np.asarray(aend["B"].output, dtype=np.complex128)[MP_DMAX:]
    y_c = np.asarray(c2["C"].output, dtype=np.complex128)[MP_DMAX:]
    y_c_b = np.asarray(c2["B"].output, dtype=np.complex128)[MP_DMAX:]
    expected_shapes = {
        "phi_a": (12_286, MP_K),
        "phi_a_b": (FINGERPRINT_LENGTH, MP_K),
        "phi_c": (7_371, MP_K),
        "phi_c_b": (FINGERPRINT_LENGTH, MP_K),
    }
    actual_shapes = {
        "phi_a": phi_a_full.shape,
        "phi_a_b": phi_a_b_full.shape,
        "phi_c": phi_c_full.shape,
        "phi_c_b": phi_c_b_full.shape,
    }
    if actual_shapes != expected_shapes:
        raise RuntimeError(f"state {normalized_id} full MP10 Phi shapes changed: {actual_shapes}")
    if not all(np.all(np.isfinite(value)) for value in (phi_a_full, phi_a_b_full, phi_c_full, phi_c_b_full)):
        raise RuntimeError(f"state {normalized_id} full MP10 Phi contains NaN/Inf")

    candidate_results = [
        _fit_one_candidate(
            candidate_id=candidate_id,
            support=support,
            phi_a_full=phi_a_full,
            y_a=y_a,
            phi_a_b_full=phi_a_b_full,
            y_a_b=y_a_b,
            phi_c_full=phi_c_full,
            y_c=y_c,
            phi_c_b_full=phi_c_b_full,
            y_c_b=y_c_b,
        )
        for candidate_id, support in _WORKER_CANDIDATES
    ]
    if any(not result["finite"] for result in candidate_results):
        raise RuntimeError(f"state {normalized_id} has a non-finite K10/K9 fit")
    real_b = np.asarray(off["B"].output[MP_DMAX:], dtype=np.complex128)
    if real_b.shape != (FINGERPRINT_LENGTH,) or not np.all(np.isfinite(real_b)):
        raise RuntimeError(f"state {normalized_id} Real-B shape/finite gate failed")
    state = _WORKER_STATE_TABLE[normalized_id]
    return {
        "State_n_R": normalized_id,
        "funMng": int(state["funMng"]),
        "funAng": int(state["funAng"]),
        "secMng": int(state["secMng"]),
        "secAng": int(state["secAng"]),
        "nmse_withoutdpd_dB": _scalar(data["nmse_withoutdpd"], "nmse_withoutdpd"),
        "ACPR_withoutdpd_mean_dBc": (
            _scalar(data["acpr_low_withoutdpd"], "acpr_low_withoutdpd")
            + _scalar(data["acpr_upper_withoutdpd"], "acpr_upper_withoutdpd")
        )
        / 2.0,
        "ilc_A_end": aend_index + 1,
        "candidates": candidate_results,
        "real_B_valid": real_b,
    }


def model_worker_entry(state_id: int) -> dict[str, Any]:
    return state_worker(int(state_id))


__all__ = [
    "ABC_LENGTHS",
    "COMMON_B_RAW_LENGTH",
    "FINGERPRINT_LENGTH",
    "FULL_LENGTH",
    "MP_DMAX",
    "MP_K",
    "PROJECT_ROOT",
    "RIDGE_LAMBDA",
    "STATE_COUNT",
    "historical_mp10",
    "model_worker_entry",
    "state_worker",
    "worker_init",
]
