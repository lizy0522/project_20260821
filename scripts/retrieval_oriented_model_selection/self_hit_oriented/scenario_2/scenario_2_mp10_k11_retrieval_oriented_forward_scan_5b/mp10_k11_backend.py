# ruff: noqa: E402,E501,I001

"""One-read-per-state K10/K11 model phase for the Envelope75 forward scan."""

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
from behavior_modeling.shared.basis_function_selection.envelope_dictionary import (  # noqa: E402
    DMAX,
    EnvelopeBasis,
    build_envelope_bank,
    build_envelope_dictionary,
)
from behavior_modeling.shared import sparse_gmp as historical_mp10  # noqa: E402
from data_management.shared import build_state_table, load_by_id  # noqa: E402
from signal_segmentation.shared import (  # noqa: E402
    build_partition_from_xin,
    get_ilc_pair,
    preprocess_full_pair,
)

STATE_COUNT = 425
FULL_LENGTH = 24_576
ABC_LENGTHS = (12_288, 4_915, 7_373)
MP_K = int(historical_mp10.FROZEN_MP_K)
MODEL_K_MAX = MP_K + 1
MP_DMAX = int(historical_mp10.MP_MAX_DELAY)
FINGERPRINT_LENGTH = ABC_LENGTHS[1] - MP_DMAX
RIDGE_LAMBDA = float(historical_mp10.RIDGE_LAMBDA)

_WORKER_CANDIDATES: tuple[tuple[int, int | None], ...] | None = None
_WORKER_TERMS: tuple[EnvelopeBasis, ...] | None = None
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


def _fit_one(
    *,
    candidate_id: int,
    added_envelope_index: int | None,
    mp_a: np.ndarray,
    env_a: np.ndarray,
    y_a: np.ndarray,
    mp_a_b: np.ndarray,
    env_a_b: np.ndarray,
    y_a_b: np.ndarray,
    mp_c: np.ndarray,
    env_c: np.ndarray,
    y_c: np.ndarray,
    mp_c_b: np.ndarray,
    env_c_b: np.ndarray,
    y_c_b: np.ndarray,
) -> dict[str, Any]:
    if added_envelope_index is None:
        phi_a = mp_a
        phi_a_b = mp_a_b
        phi_c = mp_c
        phi_c_b = mp_c_b
    else:
        phi_a = np.column_stack((mp_a, env_a[:, int(added_envelope_index)]))
        phi_a_b = np.column_stack((mp_a_b, env_a_b[:, int(added_envelope_index)]))
        phi_c = np.column_stack((mp_c, env_c[:, int(added_envelope_index)]))
        phi_c_b = np.column_stack((mp_c_b, env_c_b[:, int(added_envelope_index)]))
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
    padded_a = np.zeros(MODEL_K_MAX, dtype=np.complex128)
    padded_c = np.zeros(MODEL_K_MAX, dtype=np.complex128)
    padded_a[: theta_a.size] = theta_a
    padded_c[: theta_c.size] = theta_c
    return {
        "candidate_id": int(candidate_id),
        "K": int(theta_a.size),
        "theta_Aend_padded": padded_a,
        "theta_C2_padded": padded_c,
        "Y_Aend_train_NMSE_dB": float(nmse(y_a, prediction_a)),
        "Y_Aend_B_NMSE_dB": float(nmse(y_a_b, prediction_a_b)),
        "Y_C2_train_NMSE_dB": float(nmse(y_c, prediction_c)),
        "Y_C2_B_NMSE_dB": float(nmse(y_c_b, prediction_c_b)),
        "Y_Aend_rank": int(diagnostics_a.rank_phi),
        "Y_Aend_rank_augmented": int(diagnostics_a.rank_augmented),
        "Y_Aend_condition_number": float(diagnostics_a.condition_number_phi),
        "Y_Aend_condition_number_augmented": float(diagnostics_a.condition_number_augmented),
        "Y_C2_rank": int(diagnostics_c.rank_phi),
        "Y_C2_rank_augmented": int(diagnostics_c.rank_augmented),
        "Y_C2_condition_number": float(diagnostics_c.condition_number_phi),
        "Y_C2_condition_number_augmented": float(diagnostics_c.condition_number_augmented),
        "finite": finite,
    }


def worker_init(candidate_specs: tuple[tuple[int, int | None], ...]) -> None:
    global _WORKER_CANDIDATES, _WORKER_TERMS, _WORKER_STATE_TABLE
    _WORKER_CANDIDATES = tuple(
        (int(candidate_id), None if envelope_index is None else int(envelope_index))
        for candidate_id, envelope_index in candidate_specs
    )
    _WORKER_TERMS = tuple(build_envelope_dictionary())
    if len(_WORKER_TERMS) != 75:
        raise RuntimeError(f"Envelope75 dictionary count changed: {len(_WORKER_TERMS)}")
    if len(_WORKER_CANDIDATES) != 66:
        raise RuntimeError("K11 forward scan requires one baseline plus 65 candidates")
    if _WORKER_CANDIDATES[0][1] is not None:
        raise RuntimeError("candidate 0 must be the MP10 baseline")
    if any(index is not None and not 0 <= index < len(_WORKER_TERMS) for _, index in _WORKER_CANDIDATES):
        raise RuntimeError("candidate Envelope75 index is out of range")
    _WORKER_STATE_TABLE = {int(row["state_id"]): row for row in build_state_table()}


def state_worker(state_id: int) -> dict[str, Any]:
    if _WORKER_CANDIDATES is None or _WORKER_TERMS is None or _WORKER_STATE_TABLE is None:
        raise RuntimeError("MP10 K11 worker is not initialized")
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
    if input_history.ndim != 2 or output_history.shape != input_history.shape or input_history.shape[1] < 2:
        raise RuntimeError(f"state {normalized_id} ILC history is invalid")

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

    mp_a = historical_mp10.build_frozen_mp_basis(aend["A"].input)
    mp_a_b = historical_mp10.build_frozen_mp_basis(aend["B"].input)
    mp_c = historical_mp10.build_frozen_mp_basis(c2["C"].input)
    mp_c_b = historical_mp10.build_frozen_mp_basis(c2["B"].input)
    env_a = build_envelope_bank(aend["A"].input, _WORKER_TERMS)
    env_a_b = build_envelope_bank(aend["B"].input, _WORKER_TERMS)
    env_c = build_envelope_bank(c2["C"].input, _WORKER_TERMS)
    env_c_b = build_envelope_bank(c2["B"].input, _WORKER_TERMS)
    expected_mp_shapes = {
        "mp_a": (12_286, MP_K),
        "mp_a_b": (FINGERPRINT_LENGTH, MP_K),
        "mp_c": (7_371, MP_K),
        "mp_c_b": (FINGERPRINT_LENGTH, MP_K),
    }
    actual_mp_shapes = {
        "mp_a": mp_a.shape,
        "mp_a_b": mp_a_b.shape,
        "mp_c": mp_c.shape,
        "mp_c_b": mp_c_b.shape,
    }
    expected_env_shapes = {
        "env_a": (12_286, 75),
        "env_a_b": (FINGERPRINT_LENGTH, 75),
        "env_c": (7_371, 75),
        "env_c_b": (FINGERPRINT_LENGTH, 75),
    }
    actual_env_shapes = {
        "env_a": env_a.shape,
        "env_a_b": env_a_b.shape,
        "env_c": env_c.shape,
        "env_c_b": env_c_b.shape,
    }
    if actual_mp_shapes != expected_mp_shapes or actual_env_shapes != expected_env_shapes:
        raise RuntimeError(
            f"state {normalized_id} basis shapes changed: mp={actual_mp_shapes}, env={actual_env_shapes}"
        )
    y_a = np.asarray(aend["A"].output, dtype=np.complex128)[DMAX:]
    y_a_b = np.asarray(aend["B"].output, dtype=np.complex128)[DMAX:]
    y_c = np.asarray(c2["C"].output, dtype=np.complex128)[DMAX:]
    y_c_b = np.asarray(c2["B"].output, dtype=np.complex128)[DMAX:]
    candidate_results = [
        _fit_one(
            candidate_id=candidate_id,
            added_envelope_index=envelope_index,
            mp_a=mp_a,
            env_a=env_a,
            y_a=y_a,
            mp_a_b=mp_a_b,
            env_a_b=env_a_b,
            y_a_b=y_a_b,
            mp_c=mp_c,
            env_c=env_c,
            y_c=y_c,
            mp_c_b=mp_c_b,
            env_c_b=env_c_b,
            y_c_b=y_c_b,
        )
        for candidate_id, envelope_index in _WORKER_CANDIDATES
    ]
    if any(not result["finite"] for result in candidate_results):
        raise RuntimeError(f"state {normalized_id} has a non-finite K10/K11 fit")
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
    }


def model_worker_entry(state_id: int) -> dict[str, Any]:
    return state_worker(int(state_id))


__all__ = [
    "ABC_LENGTHS",
    "FINGERPRINT_LENGTH",
    "FULL_LENGTH",
    "MODEL_K_MAX",
    "MP_DMAX",
    "MP_K",
    "PROJECT_ROOT",
    "RIDGE_LAMBDA",
    "STATE_COUNT",
    "build_envelope_dictionary",
    "historical_mp10",
    "model_worker_entry",
    "worker_init",
]
