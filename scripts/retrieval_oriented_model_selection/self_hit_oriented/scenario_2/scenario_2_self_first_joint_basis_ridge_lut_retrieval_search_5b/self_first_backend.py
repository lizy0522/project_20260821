# ruff: noqa: E402,E501,I001

"""One-read-per-state backend for the Self-First support/lambda search."""

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

from behavior_modeling.shared.basis_function_selection.envelope_dictionary import (  # noqa: E402
    DMAX,
    EnvelopeBasis,
    build_envelope_bank,
    build_envelope_dictionary,
)
from behavior_modeling.shared.basis_function_selection.model_solver import fit_ols, fit_ridge  # noqa: E402
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
VALID_LENGTHS = (12_286, 4_913, 7_371)

_WORKER_TERMS: tuple[EnvelopeBasis, ...] | None = None
_WORKER_CANDIDATES: tuple[tuple[int, tuple[int, ...], float], ...] | None = None
_WORKER_STATE_TABLE: dict[int, dict[str, int | float]] | None = None


def _vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1 or array.size != FULL_LENGTH or not np.iscomplexobj(array):
        raise ValueError(f"{name} must be a complex length-{FULL_LENGTH} vector")
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
    phi: np.ndarray,
    target: np.ndarray,
    phi_b: np.ndarray,
    target_b: np.ndarray,
    ridge_lambda: float,
) -> dict[str, Any]:
    fit = fit_ols(phi, target, scale_columns=True) if ridge_lambda == 0.0 else fit_ridge(phi, target, ridge_lambda)
    prediction_b = np.asarray(phi_b @ fit.theta, dtype=np.complex128)
    finite = bool(
        np.all(np.isfinite(phi))
        and np.all(np.isfinite(target))
        and np.all(np.isfinite(phi_b))
        and np.all(np.isfinite(target_b))
        and np.all(np.isfinite(fit.theta))
        and np.all(np.isfinite(fit.prediction))
        and np.all(np.isfinite(prediction_b))
        and np.isfinite(fit.condition_number)
    )
    from core.shared.metrics import nmse

    return {
        "theta": np.asarray(fit.theta, dtype=np.complex128),
        "train_nmse": float(fit.nmse_db),
        "b_nmse": float(nmse(target_b, prediction_b)),
        "rank": int(fit.rank),
        "condition_number": float(fit.condition_number),
        "coefficient_norm": float(np.linalg.norm(fit.theta)),
        "finite": finite,
    }


def worker_init(candidate_specs: tuple[tuple[int, tuple[int, ...], float], ...]) -> None:
    """Initialize immutable dictionary, candidate specs, and state table."""

    global _WORKER_TERMS, _WORKER_CANDIDATES, _WORKER_STATE_TABLE
    _WORKER_TERMS = tuple(build_envelope_dictionary())
    _WORKER_CANDIDATES = tuple(
        (int(candidate_id), tuple(int(index) for index in support), float(ridge_lambda))
        for candidate_id, support, ridge_lambda in candidate_specs
    )
    _WORKER_STATE_TABLE = {int(row["state_id"]): row for row in build_state_table()}
    if len(_WORKER_TERMS) != 75:
        raise RuntimeError(f"Envelope75 dictionary changed: {len(_WORKER_TERMS)}")
    if not _WORKER_CANDIDATES:
        raise RuntimeError("Self-First worker received no candidates")


def state_worker(state_id: int) -> dict[str, Any]:
    """Load one MAT state and fit every assigned support/lambda candidate."""

    if _WORKER_TERMS is None or _WORKER_CANDIDATES is None or _WORKER_STATE_TABLE is None:
        raise RuntimeError("Self-First worker is not initialized")
    normalized_id = int(state_id)
    data = load_by_id(normalized_id)
    xin = _vector(data["xin"], "xin")
    sample_rate = _scalar(data["freqSample_Hz"], "freqSample_Hz")
    bandwidth_mhz = _scalar(data["bandWidth_MHz"], "bandWidth_MHz")
    if sample_rate != 100_000_000.0 or bandwidth_mhz != 20.0:
        raise RuntimeError(f"state {normalized_id} is not a 5B record")

    partition = build_partition_from_xin(xin)
    if (partition.n_a, partition.n_b, partition.n_c) != ABC_LENGTHS:
        raise RuntimeError(f"state {normalized_id} ABC bounds changed")
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
    real_b = np.asarray(off["B"].output[DMAX:], dtype=np.complex128)
    if real_b.shape != (VALID_LENGTHS[1],) or not np.all(np.isfinite(real_b)):
        raise RuntimeError(f"state {normalized_id} Real-B shape/finite gate failed")

    phi_a_full = build_envelope_bank(aend["A"].input, _WORKER_TERMS)
    phi_a_b_full = build_envelope_bank(aend["B"].input, _WORKER_TERMS)
    phi_c_full = build_envelope_bank(c2["C"].input, _WORKER_TERMS)
    phi_c_b_full = build_envelope_bank(c2["B"].input, _WORKER_TERMS)
    y_a = np.asarray(aend["A"].output[DMAX:], dtype=np.complex128)
    y_a_b = np.asarray(aend["B"].output[DMAX:], dtype=np.complex128)
    y_c = np.asarray(c2["C"].output[DMAX:], dtype=np.complex128)
    y_c_b = np.asarray(c2["B"].output[DMAX:], dtype=np.complex128)
    expected = {
        "phi_a": (VALID_LENGTHS[0], 75),
        "phi_a_b": (VALID_LENGTHS[1], 75),
        "phi_c": (VALID_LENGTHS[2], 75),
        "phi_c_b": (VALID_LENGTHS[1], 75),
    }
    actual = {
        "phi_a": phi_a_full.shape,
        "phi_a_b": phi_a_b_full.shape,
        "phi_c": phi_c_full.shape,
        "phi_c_b": phi_c_b_full.shape,
    }
    if actual != expected:
        raise RuntimeError(f"state {normalized_id} full Envelope75 Phi shapes changed: {actual}")

    candidate_results: list[dict[str, Any]] = []
    for candidate_id, support, ridge_lambda in _WORKER_CANDIDATES:
        if not support or len(set(support)) != len(support) or min(support) < 0 or max(support) >= 75:
            raise RuntimeError(f"candidate {candidate_id} has invalid support")
        phi_a = phi_a_full[:, support]
        phi_a_b = phi_a_b_full[:, support]
        phi_c = phi_c_full[:, support]
        phi_c_b = phi_c_b_full[:, support]
        fit_a = _fit_one(phi_a, y_a, phi_a_b, y_a_b, ridge_lambda)
        fit_c = _fit_one(phi_c, y_c, phi_c_b, y_c_b, ridge_lambda)
        candidate_results.append(
            {
                "candidate_id": candidate_id,
                "ridge_lambda": ridge_lambda,
                "theta_Aend": fit_a["theta"],
                "theta_C2": fit_c["theta"],
                "Y_Aend_train_NMSE_dB": fit_a["train_nmse"],
                "Y_Aend_B_NMSE_dB": fit_a["b_nmse"],
                "Y_C2_train_NMSE_dB": fit_c["train_nmse"],
                "Y_C2_B_NMSE_dB": fit_c["b_nmse"],
                "Aend_rank": fit_a["rank"],
                "C2_rank": fit_c["rank"],
                "Aend_condition_number": fit_a["condition_number"],
                "C2_condition_number": fit_c["condition_number"],
                "Aend_coefficient_norm": fit_a["coefficient_norm"],
                "C2_coefficient_norm": fit_c["coefficient_norm"],
                "Aend_finite": bool(fit_a["finite"] and fit_a["rank"] == len(support)),
                "C2_finite": bool(fit_c["finite"] and fit_c["rank"] == len(support)),
            }
        )
    state = _WORKER_STATE_TABLE[normalized_id]
    return {
        "State_R": normalized_id,
        "funMng": int(state["funMng"]),
        "funAng": int(state["funAng"]),
        "secMng": int(state["secMng"]),
        "secAng": int(state["secAng"]),
        "Vm": float(state["Vm"]),
        "Pin": float(state["Pin"]),
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
    "DMAX",
    "FULL_LENGTH",
    "PROJECT_ROOT",
    "STATE_COUNT",
    "VALID_LENGTHS",
    "model_worker_entry",
    "state_worker",
    "worker_init",
]
