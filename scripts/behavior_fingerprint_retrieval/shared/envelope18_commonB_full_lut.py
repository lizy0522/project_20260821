"""Frozen Envelope18 + Aend/C2 + common-B + Full-425 LUT retrieval.

This task-local module deliberately does not consume historical Ridge,
clustering, Type-III, or compressed-LUT artifacts.  It loads raw Scenario 2
states through ``data_management``, uses the canonical signal-segmentation
pipeline for Aend/C2 and Real-B, and obtains the frozen Envelope18 support from
the preceding basis-function-selection task.
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path
from typing import Any

# Set numerical threading before NumPy/SciPy imports in spawned workers.
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

from behavior_modeling.shared.basis_function_selection.all425_envelope18_evaluation import (  # noqa: E402
    FINAL_SUPPORT_IDS,
    HARD20_IDS,
    support_hash,
    verify_frozen_support,
)
from behavior_modeling.shared.basis_function_selection.envelope_dictionary import (  # noqa: E402
    DMAX,
    EnvelopeBasis,
    build_envelope_bank,
)
from behavior_modeling.shared.basis_function_selection.model_solver import fit_ols  # noqa: E402
from core.shared.metrics import nmse  # noqa: E402
from core.shared.project_paths import legacy_raw_manifest  # noqa: E402
from data_management.shared import build_state_table, load_by_id  # noqa: E402
from signal_segmentation.shared import (  # noqa: E402
    build_off_segments,
    build_partition_from_xin,
    get_common_probe,
    get_ilc_pair,
    preprocess_full_pair,
)

TASK_NAME = "scenario_2_envelope18_commonB_full_lut_retrieval_5B"
RESULT_ROOT = PROJECT_ROOT / "results" / "behavior_fingerprint_retrieval" / "scenario_2" /TASK_NAME
WORK_LOG = (
    PROJECT_ROOT / "work_logs" / "behavior_fingerprint_retrieval" / "scenario_2" /TASK_NAME / "execution_log.txt"  # noqa: E501
)
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"

STATE_COUNT = 425
FULL_LENGTH = 24_576
COMMON_B_RAW_LENGTH = 4_915
VALID_B_LENGTH = COMMON_B_RAW_LENGTH - DMAX
TRAIN_THRESHOLD_DB = -40.0
REAL_B_THRESHOLD_DB = -40.0
WORKER_COUNT = 10
BLAS_THREADS_PER_WORKER = 1
COMMON_B_SOURCE_STATE = 0
TIE_TOLERANCE_DB = 1e-12

HARD20_ORDERED_IDS = (
    325,
    332,
    333,
    334,
    336,
    353,
    324,
    338,
    350,
    326,
    349,
    339,
    351,
    342,
    328,
    323,
    355,
    331,
    327,
    356,
)

EXPECTED_RAW_MANIFEST = "910b06e4be6b60b1f4dc63409a3083eb328765347fbec3b89a97a41d6633a1e0"
EXPECTED_RAW_FILE_COUNT = 429
EXPECTED_RAW_MAT_COUNT = 427
EXPECTED_RAW_BYTES = 2_258_448_137

_WORKER_TERMS: tuple[EnvelopeBasis, ...] | None = None
_WORKER_SUPPORT: tuple[int, ...] | None = None
_WORKER_COMMON_PHI: np.ndarray | None = None


def raw_manifest() -> dict[str, object]:
    """Return the project's path-and-size raw manifest."""
    return legacy_raw_manifest()


def raw_manifest_gate() -> dict[str, object]:
    value = raw_manifest()
    expected = {
        "sha256": EXPECTED_RAW_MANIFEST,
        "file_count": EXPECTED_RAW_FILE_COUNT,
        "mat_count": EXPECTED_RAW_MAT_COUNT,
        "bytes": EXPECTED_RAW_BYTES,
    }
    if value != expected:
        raise RuntimeError(f"data/raw manifest differs from frozen baseline: {value}")
    return value


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


def _scalar(value: Any, name: str) -> float:
    array = np.asarray(value).reshape(-1)
    if array.size != 1 or not np.isfinite(array[0]):
        raise ValueError(f"{name} must be one finite scalar")
    return float(array[0])


def verify_runtime_contract() -> tuple[tuple[EnvelopeBasis, ...], tuple[int, ...], str]:
    terms, support_indices, digest = verify_frozen_support()
    if len(FINAL_SUPPORT_IDS) != 18 or len(support_indices) != 18:
        raise RuntimeError("Frozen Envelope18 support must have K=18")
    if digest != support_hash(FINAL_SUPPORT_IDS):
        raise RuntimeError("Envelope18 support hash is not deterministic")
    return terms, support_indices, digest


def build_common_b_probe() -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Build the fixed common-B input/design matrix from canonical State 0 xin."""

    data = load_by_id(COMMON_B_SOURCE_STATE)
    xin = _vector(data["xin"], "xin")
    partition = build_partition_from_xin(xin)
    if (partition.n_a, partition.n_b, partition.n_c) != (12_288, 4_915, 7_373):
        raise RuntimeError("Canonical ABC ownership lengths changed")
    common_b = np.asarray(get_common_probe(data, partition), dtype=np.complex128)
    if common_b.shape != (COMMON_B_RAW_LENGTH,) or not np.all(np.isfinite(common_b)):
        raise RuntimeError(f"common-B input shape/finite gate failed: {common_b.shape}")
    terms, support_indices, _ = verify_runtime_contract()
    phi_b_full = build_envelope_bank(common_b, terms)
    phi_b = np.asarray(phi_b_full[:, support_indices], dtype=np.complex128)
    if phi_b.shape != (VALID_B_LENGTH, 18) or not np.all(np.isfinite(phi_b)):
        raise RuntimeError(f"common-B Envelope18 design shape/finite gate failed: {phi_b.shape}")
    probe_hash = hashlib.sha256(np.ascontiguousarray(common_b).tobytes()).hexdigest()
    metadata = {
        "source_state_id": COMMON_B_SOURCE_STATE,
        "source": "canonical xin B segment via signal_segmentation.get_common_probe",
        "raw_length": int(common_b.size),
        "valid_length": int(phi_b.shape[0]),
        "dtype": str(common_b.dtype),
        "finite": bool(np.all(np.isfinite(common_b))),
        "sha256": probe_hash,
        "rms": float(np.sqrt(np.mean(np.abs(common_b) ** 2))),
        "peak": float(np.max(np.abs(common_b))),
        "mean_power": float(np.mean(np.abs(common_b) ** 2)),
    }
    return common_b, phi_b, metadata


def _field_scalar(data: dict[str, Any], name: str) -> float:
    return _scalar(data[name], name)


def _load_raw_scalars(data: dict[str, Any]) -> tuple[float, float]:
    nmse_withoutdpd = _field_scalar(data, "nmse_withoutdpd")
    acpr_low = _field_scalar(data, "acpr_low_withoutdpd")
    acpr_upper = _field_scalar(data, "acpr_upper_withoutdpd")
    return nmse_withoutdpd, (acpr_low + acpr_upper) / 2.0


def _fit_envelope_side(
    x_segment: np.ndarray,
    y_segment: np.ndarray,
) -> tuple[dict[str, object], np.ndarray]:
    if _WORKER_TERMS is None or _WORKER_SUPPORT is None:
        raise RuntimeError("Envelope18 worker is not initialized")
    full_bank = build_envelope_bank(x_segment, _WORKER_TERMS)
    phi = np.asarray(full_bank[:, _WORKER_SUPPORT], dtype=np.complex128)
    target = np.asarray(y_segment[DMAX:], dtype=np.complex128)
    if phi.shape[1] != 18 or phi.shape[0] != target.size:
        raise RuntimeError(f"Envelope18 fit shape mismatch: phi={phi.shape}, target={target.shape}")
    fit = fit_ols(phi, target, scale_columns=True)
    prediction = fit.prediction
    metrics = {
        "train_NMSE_dB": float(fit.nmse_db),
        "rank": int(fit.rank),
        "condition_number": float(fit.condition_number),
        "coefficient_norm": float(np.linalg.norm(fit.theta)),
        "finite": bool(
            np.all(np.isfinite(phi))
            and np.all(np.isfinite(fit.theta))
            and np.all(np.isfinite(prediction))
            and np.isfinite(fit.condition_number)
        ),
    }
    return metrics, np.asarray(fit.theta, dtype=np.complex128)


def _worker_init(support_indices: tuple[int, ...], common_phi: np.ndarray) -> None:
    global _WORKER_TERMS, _WORKER_SUPPORT, _WORKER_COMMON_PHI
    _WORKER_TERMS = tuple(verify_runtime_contract()[0])
    _WORKER_SUPPORT = tuple(int(index) for index in support_indices)
    _WORKER_COMMON_PHI = np.asarray(common_phi, dtype=np.complex128)
    if _WORKER_COMMON_PHI.shape != (VALID_B_LENGTH, 18):
        raise RuntimeError("common-B Phi worker shape must be (4913,18)")


def _state_worker(state_id: int) -> dict[str, object]:
    if _WORKER_COMMON_PHI is None:
        raise RuntimeError("Envelope18 worker common-B Phi is not initialized")
    normalized_id = int(state_id)
    data = load_by_id(normalized_id)
    xin = _vector(data["xin"], "xin")
    _vector(data["yout_withoutdpd_ori"], "yout_withoutdpd_ori")
    if _field_scalar(data, "freqSample_Hz") != 100_000_000.0:
        raise RuntimeError(f"state {normalized_id} sample rate is not 100 MHz")
    if _field_scalar(data, "bandWidth_MHz") != 20.0:
        raise RuntimeError(f"state {normalized_id} bandwidth is not 20 MHz")
    partition = build_partition_from_xin(xin)
    aend_count = int(np.asarray(data["xin_pd_ori_ilc"]).shape[1])
    if aend_count < 2:
        raise RuntimeError(f"state {normalized_id} has no valid C2 column")
    aend_pair = get_ilc_pair(data, aend_count - 1)
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
    a_metrics, theta_a = _fit_envelope_side(aend["A"].input, aend["A"].output)
    c_metrics, theta_c = _fit_envelope_side(c2["C"].input, c2["C"].output)
    a_b_phi = build_envelope_bank(aend["B"].input, _WORKER_TERMS)
    c_b_phi = build_envelope_bank(c2["B"].input, _WORKER_TERMS)
    a_b_phi = np.asarray(a_b_phi[:, _WORKER_SUPPORT], dtype=np.complex128)
    c_b_phi = np.asarray(c_b_phi[:, _WORKER_SUPPORT], dtype=np.complex128)
    a_b_prediction = a_b_phi @ theta_a
    c_b_prediction = c_b_phi @ theta_c
    a_b_nmse = float(nmse(aend["B"].output[DMAX:], a_b_prediction))
    c_b_nmse = float(nmse(c2["B"].output[DMAX:], c_b_prediction))
    real_b = np.asarray(off["B"].output[DMAX:], dtype=np.complex128)
    if real_b.shape != (VALID_B_LENGTH,) or not np.all(np.isfinite(real_b)):
        raise RuntimeError(f"state {normalized_id} Real-B shape/finite gate failed")
    nmse_withoutdpd, acpr_mean = _load_raw_scalars(data)
    state = build_state_table()[normalized_id]
    return {
        "state_id": normalized_id,
        "funMng": int(state["funMng"]),
        "funAng": int(state["funAng"]),
        "secMng": int(state["secMng"]),
        "secAng": int(state["secAng"]),
        "Vm": float(state["Vm"]),
        "Pin": float(state["Pin"]),
        "ilc_A_end": aend_count,
        "C2_available": True,
        "nmse_withoutdpd_dB": nmse_withoutdpd,
        "ACPR_withoutdpd_mean_dBc": acpr_mean,
        "Y_Aend_train_NMSE_dB": a_metrics["train_NMSE_dB"],
        "Y_Aend_B_NMSE_dB": a_b_nmse,
        "Y_Aend_rank": a_metrics["rank"],
        "Y_Aend_condition_number": a_metrics["condition_number"],
        "Y_Aend_coefficient_norm": a_metrics["coefficient_norm"],
        "Y_Aend_finite": a_metrics["finite"],
        "Y_C2_train_NMSE_dB": c_metrics["train_NMSE_dB"],
        "Y_C2_B_NMSE_dB": c_b_nmse,
        "Y_C2_rank": c_metrics["rank"],
        "Y_C2_condition_number": c_metrics["condition_number"],
        "Y_C2_coefficient_norm": c_metrics["coefficient_norm"],
        "Y_C2_finite": c_metrics["finite"],
        "Aend_theta": theta_a,
        "C2_theta": theta_c,
        "real_B_valid": real_b,
    }


def model_worker_entry(state_id: int) -> dict[str, object]:
    return _state_worker(int(state_id))


def compute_cnmse_matrix(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    """Use the project's canonical vectorized CNMSE implementation."""

    from behavior_fingerprint_ranking_consistency.shared.distance_matrix import (
        compute_cnmse_distance_matrix,
    )

    return compute_cnmse_distance_matrix(query, candidates)


def top1_retrieval(distance: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if distance.shape != (STATE_COUNT, STATE_COUNT):
        raise RuntimeError(f"retrieval distance shape must be (425,425), got {distance.shape}")
    selected = np.empty(STATE_COUNT, dtype=np.int64)
    selected_distance = np.empty(STATE_COUNT, dtype=np.float64)
    true_rank = np.empty(STATE_COUNT, dtype=np.int64)
    tie_count = np.empty(STATE_COUNT, dtype=np.int64)
    state_ids = np.arange(STATE_COUNT, dtype=np.int64)
    for real_id in range(STATE_COUNT):
        row = distance[real_id]
        minimum = float(np.min(row))
        candidates = np.flatnonzero(row <= minimum + TIE_TOLERANCE_DB)
        query_id = int(np.min(candidates))
        selected[real_id] = query_id
        selected_distance[real_id] = float(row[query_id])
        tie_count[real_id] = int(candidates.size)
        order = np.lexsort((state_ids, row))
        true_rank[real_id] = int(np.flatnonzero(order == real_id)[0] + 1)
    return selected, selected_distance, true_rank, tie_count


def load_type_from_state(state: dict[str, int | float]) -> str:
    fun = int(state["funMng"]) != 0
    sec = int(state["secMng"]) != 0
    if not fun and not sec:
        return "matched"
    if fun and not sec:
        return "fundamental_only"
    if not fun and sec:
        return "second_harmonic_only"
    return "joint_mismatch"


def _finite_summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {
            "count": 0,
            "median_dB": float("nan"),
            "mean_dB": float("nan"),
            "Q90_dB": float("nan"),
            "Q95_dB": float("nan"),
            "worst_dB": float("nan"),
        }
    return {
        "count": int(finite.size),
        "median_dB": float(np.median(finite)),
        "mean_dB": float(np.mean(finite)),
        "Q90_dB": float(np.quantile(finite, 0.90)),
        "Q95_dB": float(np.quantile(finite, 0.95)),
        "worst_dB": float(np.max(finite)),
    }


__all__ = [
    "BLAS_THREADS_PER_WORKER",
    "COMMON_B_SOURCE_STATE",
    "FINAL_SUPPORT_IDS",
    "HARD20_IDS",
    "RESULT_ROOT",
    "STATE_COUNT",
    "TASK_NAME",
    "VALID_B_LENGTH",
    "build_common_b_probe",
    "compute_cnmse_matrix",
    "load_type_from_state",
    "model_worker_entry",
    "raw_manifest",
    "raw_manifest_gate",
    "support_hash",
    "top1_retrieval",
    "verify_runtime_contract",
    "_worker_init",
    "_finite_summary",
]
