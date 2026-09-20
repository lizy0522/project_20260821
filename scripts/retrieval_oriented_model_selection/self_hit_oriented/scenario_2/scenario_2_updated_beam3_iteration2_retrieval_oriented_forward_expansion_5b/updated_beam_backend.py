# ruff: noqa: E402,E501,I001

"""One-read-per-state backend for the updated Beam=3 forward expansion."""

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
from behavior_modeling.shared.basis_function_selection.envelope_dictionary import DMAX, EnvelopeBasis, build_envelope_bank, build_envelope_dictionary  # noqa: E402
from behavior_modeling.shared import sparse_gmp as historical_mp10  # noqa: E402
from data_management.shared import build_state_table, load_by_id  # noqa: E402
from signal_segmentation.shared import build_partition_from_xin, get_ilc_pair, preprocess_full_pair  # noqa: E402

STATE_COUNT = 425
FULL_LENGTH = 24_576
ABC_LENGTHS = (12_288, 4_915, 7_373)
MP_K = int(historical_mp10.FROZEN_MP_K)
MP_DMAX = int(historical_mp10.MP_MAX_DELAY)
RIDGE_LAMBDA = float(historical_mp10.RIDGE_LAMBDA)
FINGERPRINT_LENGTH = ABC_LENGTHS[1] - MP_DMAX
MP_COLUMNS = MP_K

_WORKER_CHILDREN: tuple[tuple[int, tuple[int, ...]], ...] | None = None
_WORKER_CONDITION: tuple[tuple[str, tuple[int, ...]], ...] | None = None
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


def _select_columns(mp: np.ndarray, env: np.ndarray, columns: tuple[int, ...]) -> np.ndarray:
    return np.column_stack([mp[:, index] if index < MP_COLUMNS else env[:, index - MP_COLUMNS] for index in columns]).astype(np.complex128)


def _condition_number(matrix: np.ndarray) -> tuple[int, float, float, float]:
    singular = np.linalg.svd(matrix, compute_uv=False, full_matrices=False)
    sigma_max = float(singular[0])
    sigma_min = float(singular[-1])
    condition = float(sigma_max / sigma_min) if sigma_min > 0 else float("inf")
    return int(np.linalg.matrix_rank(matrix)), sigma_max, sigma_min, condition


def _condition_block(phi: np.ndarray, theta: np.ndarray | None) -> dict[str, Any]:
    norms = np.linalg.norm(phi, axis=0)
    if np.any(norms <= 0) or not np.all(np.isfinite(norms)):
        raise RuntimeError("conditioning audit encountered zero/nonfinite column norm")
    rank_raw, sigma_max_raw, sigma_min_raw, cond_raw = _condition_number(phi)
    normalized = phi / norms
    rank_norm, sigma_max_norm, sigma_min_norm, cond_norm = _condition_number(normalized)
    augmented = np.vstack((phi, np.sqrt(RIDGE_LAMBDA) * np.eye(phi.shape[1], dtype=np.complex128)))
    rank_aug, sigma_max_aug, sigma_min_aug, cond_aug = _condition_number(augmented)
    return {"rank_raw": rank_raw, "sigma_max_raw": sigma_max_raw, "sigma_min_raw": sigma_min_raw, "condition_number_raw": cond_raw, "rank_column_normalized": rank_norm, "sigma_max_column_normalized": sigma_max_norm, "sigma_min_column_normalized": sigma_min_norm, "condition_number_column_normalized": cond_norm, "rank_ridge_augmented": rank_aug, "sigma_max_ridge_augmented": sigma_max_aug, "sigma_min_ridge_augmented": sigma_min_aug, "condition_number_ridge_augmented": cond_aug, "theta_l2_norm": float(np.linalg.norm(theta)) if theta is not None else np.nan}


def _fit_one(candidate_id: int, columns: tuple[int, ...], mp_a: np.ndarray, env_a: np.ndarray, y_a: np.ndarray, mp_a_b: np.ndarray, env_a_b: np.ndarray, y_a_b: np.ndarray, mp_c: np.ndarray, env_c: np.ndarray, y_c: np.ndarray, mp_c_b: np.ndarray, env_c_b: np.ndarray, y_c_b: np.ndarray) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    phi_a = _select_columns(mp_a, env_a, columns)
    phi_a_b = _select_columns(mp_a_b, env_a_b, columns)
    phi_c = _select_columns(mp_c, env_c, columns)
    phi_c_b = _select_columns(mp_c_b, env_c_b, columns)
    theta_a, diag_a = historical_mp10.fit_ridge(phi_a, y_a, RIDGE_LAMBDA)
    theta_c, diag_c = historical_mp10.fit_ridge(phi_c, y_c, RIDGE_LAMBDA)
    prediction_a = phi_a @ theta_a
    prediction_a_b = phi_a_b @ theta_a
    prediction_c = phi_c @ theta_c
    prediction_c_b = phi_c_b @ theta_c
    finite = bool(np.all(np.isfinite(phi_a)) and np.all(np.isfinite(phi_a_b)) and np.all(np.isfinite(phi_c)) and np.all(np.isfinite(phi_c_b)) and np.all(np.isfinite(theta_a)) and np.all(np.isfinite(theta_c)) and np.all(np.isfinite(prediction_a)) and np.all(np.isfinite(prediction_a_b)) and np.all(np.isfinite(prediction_c)) and np.all(np.isfinite(prediction_c_b)))
    return {"candidate_id": int(candidate_id), "K": len(columns), "theta_Aend": np.asarray(theta_a, dtype=np.complex128), "theta_C2": np.asarray(theta_c, dtype=np.complex128), "Y_Aend_train_NMSE_dB": float(nmse(y_a, prediction_a)), "Y_Aend_B_NMSE_dB": float(nmse(y_a_b, prediction_a_b)), "Y_C2_train_NMSE_dB": float(nmse(y_c, prediction_c)), "Y_C2_B_NMSE_dB": float(nmse(y_c_b, prediction_c_b)), "Aend_rank": int(diag_a.rank_phi), "Aend_rank_augmented": int(diag_a.rank_augmented), "C2_rank": int(diag_c.rank_phi), "C2_rank_augmented": int(diag_c.rank_augmented), "Aend_condition_number": float(diag_a.condition_number_phi), "C2_condition_number": float(diag_c.condition_number_phi), "Aend_condition_number_augmented": float(diag_a.condition_number_augmented), "C2_condition_number_augmented": float(diag_c.condition_number_augmented), "Aend_theta_norm": float(np.linalg.norm(theta_a)), "C2_theta_norm": float(np.linalg.norm(theta_c)), "finite": finite}, theta_a, theta_c


def worker_init(child_specs: tuple[tuple[int, tuple[int, ...]], ...], condition_specs: tuple[tuple[str, tuple[int, ...]], ...]) -> None:
    global _WORKER_CHILDREN, _WORKER_CONDITION, _WORKER_TERMS, _WORKER_STATE_TABLE
    _WORKER_CHILDREN = tuple((int(candidate_id), tuple(int(index) for index in columns)) for candidate_id, columns in child_specs)
    _WORKER_CONDITION = tuple((str(name), tuple(int(index) for index in columns)) for name, columns in condition_specs)
    _WORKER_TERMS = tuple(build_envelope_dictionary())
    if len(_WORKER_TERMS) != 75 or len(_WORKER_CHILDREN) != 186 or len(_WORKER_CONDITION) != 7:
        raise RuntimeError("updated Beam worker count changed")
    _WORKER_STATE_TABLE = {int(row["state_id"]): row for row in build_state_table()}


def condition_only_worker_init(condition_specs: tuple[tuple[str, tuple[int, ...]], ...]) -> None:
    """Initialize the same one-read state path for post-search supports."""
    global _WORKER_CHILDREN, _WORKER_CONDITION, _WORKER_TERMS, _WORKER_STATE_TABLE
    _WORKER_CHILDREN = tuple()
    _WORKER_CONDITION = tuple((str(name), tuple(int(index) for index in columns)) for name, columns in condition_specs)
    _WORKER_TERMS = tuple(build_envelope_dictionary())
    if len(_WORKER_TERMS) != 75 or not _WORKER_CONDITION:
        raise RuntimeError("post-search conditioning worker contract changed")
    _WORKER_STATE_TABLE = {int(row["state_id"]): row for row in build_state_table()}


def state_worker(state_id: int) -> dict[str, Any]:
    if _WORKER_CHILDREN is None or _WORKER_CONDITION is None or _WORKER_TERMS is None or _WORKER_STATE_TABLE is None:
        raise RuntimeError("updated Beam worker is not initialized")
    normalized_id = int(state_id)
    data = load_by_id(normalized_id)
    xin = _vector(data["xin"], "xin")
    _vector(data["yout_withoutdpd_ori"], "yout_withoutdpd_ori")
    if _scalar(data["freqSample_Hz"], "freqSample_Hz") != 100_000_000.0 or _scalar(data["bandWidth_MHz"], "bandWidth_MHz") != 20.0:
        raise RuntimeError(f"state {normalized_id} is not formal 5B")
    partition = build_partition_from_xin(xin)
    if (partition.n_a, partition.n_b, partition.n_c) != ABC_LENGTHS:
        raise RuntimeError(f"state {normalized_id} ABC ownership changed")
    history = np.asarray(data["xin_pd_ori_ilc"])
    outputs = np.asarray(data["yout_withdpd_ori_ilc"])
    if history.ndim != 2 or outputs.shape != history.shape or history.shape[1] < 2:
        raise RuntimeError(f"state {normalized_id} ILC history invalid")
    a_pair = get_ilc_pair(data, int(history.shape[1] - 1))
    c_pair = get_ilc_pair(data, 1)
    aend = preprocess_full_pair(a_pair.input_full, a_pair.output_raw_full, partition, pair_type=a_pair.pair_type, iteration_index=a_pair.iteration_index, input_peak_normalization_factor=a_pair.input_peak_normalization_factor)
    c2 = preprocess_full_pair(c_pair.input_full, c_pair.output_raw_full, partition, pair_type=c_pair.pair_type, iteration_index=c_pair.iteration_index, input_peak_normalization_factor=c_pair.input_peak_normalization_factor)
    inputs = (aend["A"].input, aend["B"].input, c2["C"].input, c2["B"].input)
    mp_a, mp_a_b, mp_c, mp_c_b = (historical_mp10.build_frozen_mp_basis(value) for value in inputs)
    env_a, env_a_b, env_c, env_c_b = (build_envelope_bank(value, _WORKER_TERMS) for value in inputs)
    y_a = np.asarray(aend["A"].output, dtype=np.complex128)[DMAX:]
    y_a_b = np.asarray(aend["B"].output, dtype=np.complex128)[DMAX:]
    y_c = np.asarray(c2["C"].output, dtype=np.complex128)[DMAX:]
    y_c_b = np.asarray(c2["B"].output, dtype=np.complex128)[DMAX:]
    fits: list[dict[str, Any]] = []
    for candidate_id, columns in _WORKER_CHILDREN:
        row, _, _ = _fit_one(candidate_id, columns, mp_a, env_a, y_a, mp_a_b, env_a_b, y_a_b, mp_c, env_c, y_c, mp_c_b, env_c_b, y_c_b)
        if not row["finite"]:
            raise RuntimeError(f"state {normalized_id} novel candidate {candidate_id} is nonfinite")
        fits.append(row)
    conditioning: list[dict[str, Any]] = []
    for support_name, columns in _WORKER_CONDITION:
        for behavior_name, mp, env, target in (("Aend", mp_a, env_a, y_a), ("C2", mp_c, env_c, y_c)):
            phi = _select_columns(mp, env, columns)
            theta, _ = historical_mp10.fit_ridge(phi, target, RIDGE_LAMBDA)
            conditioning.append({"support_name": support_name, "behavior": behavior_name, "State_R": normalized_id, **_condition_block(phi, theta)})
    state = _WORKER_STATE_TABLE[normalized_id]
    return {"State_n_R": normalized_id, "funMng": int(state["funMng"]), "funAng": int(state["funAng"]), "secMng": int(state["secMng"]), "secAng": int(state["secAng"]), "nmse_withoutdpd_dB": _scalar(data["nmse_withoutdpd"], "nmse_withoutdpd"), "ACPR_withoutdpd_mean_dBc": (_scalar(data["acpr_low_withoutdpd"], "acpr_low_withoutdpd") + _scalar(data["acpr_upper_withoutdpd"], "acpr_upper_withoutdpd")) / 2.0, "ilc_A_end": int(history.shape[1]), "fits": fits, "conditioning": conditioning}


def model_worker_entry(state_id: int) -> dict[str, Any]:
    return state_worker(int(state_id))


def condition_only_worker_entry(state_id: int) -> dict[str, Any]:
    return state_worker(int(state_id))


__all__ = ["ABC_LENGTHS", "FINGERPRINT_LENGTH", "FULL_LENGTH", "MP_DMAX", "MP_K", "PROJECT_ROOT", "RIDGE_LAMBDA", "STATE_COUNT", "build_envelope_dictionary", "condition_only_worker_entry", "condition_only_worker_init", "historical_mp10", "model_worker_entry", "worker_init"]
