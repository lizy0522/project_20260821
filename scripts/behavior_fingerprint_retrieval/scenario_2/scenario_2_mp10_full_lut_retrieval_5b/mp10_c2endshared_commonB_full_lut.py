# ruff: noqa: E402,E501,I001

"""Historical Frozen MP10 backend adapted to the current E19 Full-LUT protocol.

The model implementation is deliberately delegated to ``behavior_modeling.sparse_gmp``:
``build_frozen_mp_basis`` is the historical ten-column MP builder and ``fit_ridge``
is its historical augmented complex least-squares Ridge solver.  This adapter owns
only the current E19 state semantics and output contract: final ILC column for
Aend, ILC2 for C2, canonical ABC preprocessing, and OFF Real-B materialization.
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

from behavior_fingerprint_retrieval.shared.envelope19_c2endshared_commonB_full_lut import (  # noqa: E402
    COMMON_B_RAW_LENGTH,
    EXPECTED_COMMON_B_SHA,
    FULL_LENGTH,
    VALID_B_LENGTH,
    compute_cnmse_matrix,
    raw_manifest_gate,
    verify_common_b_contract,
)

STATE_COUNT = 425
ABC_LENGTHS = (12_288, 4_915, 7_373)
FINGERPRINT_LENGTH = VALID_B_LENGTH
MP_ORDERS = tuple(historical_mp10.MP_ORDERS)
MP_MEMORY = dict(historical_mp10.MP_MEMORY)
MP_K = int(historical_mp10.FROZEN_MP_K)
MP_DMAX = int(historical_mp10.MP_MAX_DELAY)
RIDGE_LAMBDA = float(historical_mp10.RIDGE_LAMBDA)
MODEL_NAME = "Historical-Frozen-MP10"

_WORKER_COMMON_B: np.ndarray | None = None
_WORKER_COMMON_PHI: np.ndarray | None = None
_WORKER_STATE_TABLE: dict[int, dict[str, int | float]] | None = None


def historical_contract() -> dict[str, Any]:
    """Return the historical MP10 contract directly from its backend constants."""

    memory_vector = [int(MP_MEMORY[order]) for order in MP_ORDERS]
    if MP_ORDERS != (1, 2, 3, 5, 7, 9):
        raise RuntimeError(f"historical MP orders changed: {MP_ORDERS}")
    if memory_vector != [3, 2, 2, 1, 1, 1]:
        raise RuntimeError(f"historical MP memory changed: {memory_vector}")
    if MP_K != 10 or MP_DMAX != 2 or RIDGE_LAMBDA != 1e-8:
        raise RuntimeError(
            f"historical MP contract changed: K={MP_K}, dmax={MP_DMAX}, "
            f"lambda={RIDGE_LAMBDA}"
        )
    return {
        "model_name": MODEL_NAME,
        "orders": list(MP_ORDERS),
        "memory": memory_vector,
        "K": MP_K,
        "dmax": MP_DMAX,
        "ridge_lambda": RIDGE_LAMBDA,
        "basis_source": "behavior_modeling.sparse_gmp.build_frozen_mp_basis",
        "ridge_source": "behavior_modeling.sparse_gmp.fit_ridge",
        "ridge_solver": "augmented complex numpy.linalg.lstsq",
        "basis_normalization": "none; raw MP columns",
        "ilc_input_normalization": "get_ilc_pair: each ILC input column divided by its own peak",
        "separate_hnorm_function_found": False,
    }


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


def _fit_side(x_segment: np.ndarray, y_segment: np.ndarray) -> tuple[dict[str, Any], np.ndarray]:
    phi = historical_mp10.build_frozen_mp_basis(x_segment)
    target = np.asarray(y_segment, dtype=np.complex128)[MP_DMAX:]
    if phi.shape != (x_segment.size - MP_DMAX, MP_K) or target.shape != (phi.shape[0],):
        raise RuntimeError(f"MP10 fit shape mismatch: phi={phi.shape}, target={target.shape}")
    theta, diagnostics = historical_mp10.fit_ridge(phi, target, RIDGE_LAMBDA)
    prediction = phi @ theta
    metrics = {
        "train_NMSE_dB": float(nmse(target, prediction)),
        "rank": int(diagnostics.rank_phi),
        "rank_augmented": int(diagnostics.rank_augmented),
        "condition_number": float(diagnostics.condition_number_phi),
        "condition_number_augmented": float(diagnostics.condition_number_augmented),
        "coefficient_norm": float(np.linalg.norm(theta)),
        "n_train": int(phi.shape[0]),
        "finite": bool(
            np.all(np.isfinite(phi))
            and np.all(np.isfinite(theta))
            and np.all(np.isfinite(prediction))
        ),
    }
    return metrics, np.asarray(theta, dtype=np.complex128)


def _worker_init(common_b: np.ndarray) -> None:
    global _WORKER_COMMON_B, _WORKER_COMMON_PHI, _WORKER_STATE_TABLE
    historical_contract()
    _WORKER_COMMON_B = np.asarray(common_b, dtype=np.complex128)
    if _WORKER_COMMON_B.shape != (COMMON_B_RAW_LENGTH,) or not np.all(np.isfinite(_WORKER_COMMON_B)):
        raise RuntimeError("MP10 common-B input shape/finite contract failed")
    _WORKER_COMMON_PHI = historical_mp10.build_frozen_mp_basis(_WORKER_COMMON_B)
    if _WORKER_COMMON_PHI.shape != (FINGERPRINT_LENGTH, MP_K):
        raise RuntimeError(f"MP10 common-B Phi shape is {_WORKER_COMMON_PHI.shape}")
    _WORKER_STATE_TABLE = {int(row["state_id"]): row for row in build_state_table()}


def _state_worker(state_id: int) -> dict[str, Any]:
    if _WORKER_COMMON_PHI is None or _WORKER_STATE_TABLE is None:
        raise RuntimeError("MP10 worker is not initialized")
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
    histories = np.asarray(data["xin_pd_ori_ilc"])
    outputs = np.asarray(data["yout_withdpd_ori_ilc"])
    if histories.ndim != 2 or outputs.shape != histories.shape or histories.shape[1] < 2:
        raise RuntimeError(f"state {normalized_id} ILC history is invalid")
    aend_index = int(histories.shape[1] - 1)
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
    a_metrics, theta_a = _fit_side(aend["A"].input, aend["A"].output)
    c_metrics, theta_c = _fit_side(c2["C"].input, c2["C"].output)
    a_b_phi = historical_mp10.build_frozen_mp_basis(aend["B"].input)
    c_b_phi = historical_mp10.build_frozen_mp_basis(c2["B"].input)
    a_b_y = np.asarray(aend["B"].output, dtype=np.complex128)[MP_DMAX:]
    c_b_y = np.asarray(c2["B"].output, dtype=np.complex128)[MP_DMAX:]
    a_b_nmse = float(nmse(a_b_y, a_b_phi @ theta_a))
    c_b_nmse = float(nmse(c_b_y, c_b_phi @ theta_c))
    real_b = np.asarray(off["B"].output[MP_DMAX:], dtype=np.complex128)
    if real_b.shape != (VALID_B_LENGTH,) or not np.all(np.isfinite(real_b)):
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
        "Y_Aend_train_NMSE_dB": a_metrics["train_NMSE_dB"],
        "Y_Aend_B_NMSE_dB": a_b_nmse,
        "Y_Aend_rank": a_metrics["rank"],
        "Y_Aend_rank_augmented": a_metrics["rank_augmented"],
        "Y_Aend_condition_number": a_metrics["condition_number"],
        "Y_Aend_condition_number_augmented": a_metrics["condition_number_augmented"],
        "Y_Aend_coefficient_norm": a_metrics["coefficient_norm"],
        "Y_Aend_finite": a_metrics["finite"],
        "Y_C2_train_NMSE_dB": c_metrics["train_NMSE_dB"],
        "Y_C2_B_NMSE_dB": c_b_nmse,
        "Y_C2_rank": c_metrics["rank"],
        "Y_C2_rank_augmented": c_metrics["rank_augmented"],
        "Y_C2_condition_number": c_metrics["condition_number"],
        "Y_C2_condition_number_augmented": c_metrics["condition_number_augmented"],
        "Y_C2_coefficient_norm": c_metrics["coefficient_norm"],
        "Y_C2_finite": c_metrics["finite"],
        "theta_Aend": theta_a,
        "theta_C2": theta_c,
        "real_B_valid": real_b,
    }


def model_worker_entry(state_id: int) -> dict[str, Any]:
    return _state_worker(int(state_id))


def verify_backend_contract(common_b: np.ndarray) -> dict[str, Any]:
    """Run the shared common-B gate and direct historical backend checks."""

    contract = historical_contract()
    if common_b.shape != (COMMON_B_RAW_LENGTH,):
        raise RuntimeError(f"common-B raw shape is {common_b.shape}")
    _worker_init(common_b)
    if _WORKER_COMMON_PHI is None:
        raise RuntimeError("common-B Phi was not initialized")
    return {
        **contract,
        "common_B_raw_length": int(common_b.size),
        "common_B_valid_length": int(_WORKER_COMMON_PHI.shape[0]),
        "common_B_phi_shape": list(_WORKER_COMMON_PHI.shape),
        "common_B_sha256": EXPECTED_COMMON_B_SHA,
        "raw_manifest": raw_manifest_gate(),
    }


__all__ = [
    "ABC_LENGTHS",
    "COMMON_B_RAW_LENGTH",
    "FINGERPRINT_LENGTH",
    "MODEL_NAME",
    "MP_DMAX",
    "MP_K",
    "MP_MEMORY",
    "MP_ORDERS",
    "PROJECT_ROOT",
    "RIDGE_LAMBDA",
    "STATE_COUNT",
    "VALID_B_LENGTH",
    "compute_cnmse_matrix",
    "historical_contract",
    "model_worker_entry",
    "raw_manifest_gate",
    "verify_backend_contract",
    "verify_common_b_contract",
    "_state_worker",
    "_worker_init",
]
