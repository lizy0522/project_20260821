"""Frozen Envelope19-C2EndShared common-B Full425 retrieval protocol."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

# ruff: noqa: E402,E501

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
from behavior_modeling.shared.basis_function_selection.model_solver import fit_ols  # noqa: E402
from core.shared.metrics import nmse  # noqa: E402
from data_management.shared import build_state_table, load_by_id  # noqa: E402
from signal_segmentation.shared import (  # noqa: E402
    build_off_segments,
    build_partition_from_xin,
    get_ilc_pair,
    preprocess_full_pair,
)

from behavior_fingerprint_retrieval.shared.envelope23_ilcend_commonB_full_lut_comparison import (  # noqa: E402
    build_common_b_probe,
    raw_manifest_gate,
)

TASK_NAME = "scenario_2_envelope19_c2endshared_commonB_full_lut_retrieval_5B"
RESULT_ROOT = PROJECT_ROOT / "results" / "behavior_fingerprint_retrieval" / "scenario_2" /TASK_NAME
STATE_COUNT = 425
FULL_LENGTH = 24_576
COMMON_B_RAW_LENGTH = 4_915
VALID_B_LENGTH = COMMON_B_RAW_LENGTH - DMAX
WORKER_COUNT = 10
BLAS_THREADS_PER_WORKER = 1
THRESHOLD_DB = -40.0
COMMON_B_SOURCE_STATE = 0
EXPECTED_COMMON_B_SHA = "b3a794e22b6beb7171f696d41685714da122f948ca8284553abaaa9a68c4abac"
EXPECTED_SUPPORT_HASH = "dda04e2fc270c9f5c2eb72b0b030059211ba0295934e3e3a4b2834172b11ac90"
MODEL_NAME = "Envelope19-C2EndShared"

FROZEN_MODEL_DEFINITION = (
    PROJECT_ROOT
    / "results"
    / "behavior_modeling"
    / "scenario_2" /"scenario_2_hard20_c2_ilcend_shared_envelope75_basis_selection_5B"
    / "07_final_model_definition.csv"
)
E23_ROOT = PROJECT_ROOT / "results" / "behavior_fingerprint_retrieval" / "scenario_2" /"scenario_2_envelope23_ilcend_commonB_full_lut_retrieval_5B"
E23_METRICS = E23_ROOT / "03_envelope23_all425_retrieval_metrics.csv"
E18_METRICS = PROJECT_ROOT / "results" / "behavior_fingerprint_retrieval" / "scenario_2" /"scenario_2_envelope18_commonB_full_lut_retrieval_5B" / "03_all425_retrieval_metrics.csv"
E18_SUMMARY = PROJECT_ROOT / "results" / "behavior_fingerprint_retrieval" / "scenario_2" /"scenario_2_envelope18_commonB_full_lut_retrieval_5B" / "10_final_result_summary.txt"
E18_COMMON_META = PROJECT_ROOT / "results" / "behavior_fingerprint_retrieval" / "scenario_2" /"scenario_2_envelope18_commonB_full_lut_retrieval_5B" / "01_common_B_probe_metadata.txt"

_WORKER_TERMS: tuple[EnvelopeBasis, ...] | None = None
_WORKER_SUPPORT: tuple[int, ...] | None = None
_WORKER_STATE_TABLE: dict[int, dict[str, int | float]] | None = None


def support_hash(ids: tuple[str, ...]) -> str:
    return hashlib.sha256(
        json.dumps(list(ids), ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def load_frozen_support() -> tuple[tuple[EnvelopeBasis, ...], tuple[int, ...], tuple[str, ...], str, pd.DataFrame]:
    if not FROZEN_MODEL_DEFINITION.is_file():
        raise FileNotFoundError(f"Frozen Envelope19 model definition is missing: {FROZEN_MODEL_DEFINITION}")
    frame = pd.read_csv(FROZEN_MODEL_DEFINITION)
    if len(frame) != 19 or set(frame["K"].astype(int)) != {19}:
        raise RuntimeError("Frozen Envelope19 K contract failed")
    if set(frame["dmax"].astype(int)) != {DMAX} or set(frame["lambda"].astype(float)) != {0.0}:
        raise RuntimeError("Frozen Envelope19 dmax/lambda contract failed")
    if set(frame["solver"].astype(str)) != {"OLS"} or set(frame["model_name"].astype(str)) != {MODEL_NAME}:
        raise RuntimeError("Frozen Envelope19 model metadata contract failed")
    ids = tuple(frame["basis_id"].astype(str).tolist())
    digest = str(frame["support_hash"].iloc[0])
    if digest != EXPECTED_SUPPORT_HASH or support_hash(ids) != EXPECTED_SUPPORT_HASH:
        raise RuntimeError(f"Frozen Envelope19 support hash changed: {digest}")
    terms = tuple(build_envelope_dictionary())
    by_id = {term.basis_id: term for term in terms}
    missing = [basis_id for basis_id in ids if basis_id not in by_id]
    if missing:
        raise RuntimeError(f"Frozen Envelope19 basis IDs missing from Envelope75: {missing}")
    indices = tuple(by_id[basis_id].index for basis_id in ids)
    index_column = "term_index" if "term_index" in frame else "global_index"
    if tuple(frame[index_column].astype(int).tolist()) != indices:
        raise RuntimeError("Frozen Envelope19 canonical term indices changed")
    return terms, indices, ids, digest, frame


def verify_common_b_contract() -> tuple[np.ndarray, dict[str, object]]:
    common_b, _, metadata = build_common_b_probe()
    sha = str(metadata["sha256"])
    if sha != EXPECTED_COMMON_B_SHA:
        raise RuntimeError(f"Frozen common-B hash changed: {sha}")
    if common_b.shape != (COMMON_B_RAW_LENGTH,) or not np.all(np.isfinite(common_b)):
        raise RuntimeError("common-B shape/finite contract failed")
    return common_b, metadata


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


def _fit_side(x_segment: np.ndarray, y_segment: np.ndarray) -> tuple[dict[str, object], np.ndarray]:
    if _WORKER_TERMS is None or _WORKER_SUPPORT is None:
        raise RuntimeError("Envelope19 worker support is not initialized")
    full_bank = build_envelope_bank(x_segment, _WORKER_TERMS)
    phi = np.asarray(full_bank[:, _WORKER_SUPPORT], dtype=np.complex128)
    target = np.asarray(y_segment[DMAX:], dtype=np.complex128)
    if phi.shape != (x_segment.size - DMAX, 19) or target.shape != (x_segment.size - DMAX,):
        raise RuntimeError(f"Envelope19 fit shape mismatch: phi={phi.shape}, target={target.shape}")
    fit = fit_ols(phi, target, scale_columns=True)
    return {
        "train_NMSE_dB": float(fit.nmse_db),
        "rank": int(fit.rank),
        "condition_number": float(fit.condition_number),
        "coefficient_norm": float(np.linalg.norm(fit.theta)),
        "finite": bool(
            np.all(np.isfinite(phi))
            and np.all(np.isfinite(fit.theta))
            and np.all(np.isfinite(fit.prediction))
            and np.isfinite(fit.condition_number)
        ),
    }, np.asarray(fit.theta, dtype=np.complex128)


def _worker_init(support_indices: tuple[int, ...]) -> None:
    global _WORKER_TERMS, _WORKER_SUPPORT, _WORKER_STATE_TABLE
    _WORKER_TERMS = tuple(build_envelope_dictionary())
    _WORKER_SUPPORT = tuple(int(index) for index in support_indices)
    _WORKER_STATE_TABLE = {int(row["state_id"]): row for row in build_state_table()}
    if len(_WORKER_SUPPORT) != 19:
        raise RuntimeError("Envelope19 worker support must contain 19 terms")


def _state_worker(state_id: int) -> dict[str, object]:
    if _WORKER_TERMS is None or _WORKER_SUPPORT is None or _WORKER_STATE_TABLE is None:
        raise RuntimeError("Envelope19 worker is not initialized")
    normalized_id = int(state_id)
    data = load_by_id(normalized_id)
    xin = _vector(data["xin"], "xin")
    sample_rate = _scalar(data["freqSample_Hz"], "freqSample_Hz")
    bandwidth_mhz = _scalar(data["bandWidth_MHz"], "bandWidth_MHz")
    if sample_rate != 100_000_000.0 or bandwidth_mhz != 20.0:
        raise RuntimeError(f"state {normalized_id} is not a 5B record")
    partition = build_partition_from_xin(xin)
    if (partition.n_a, partition.n_b, partition.n_c) != (12_288, 4_915, 7_373):
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
    a_metrics, theta_a = _fit_side(aend["A"].input, aend["A"].output)
    c_metrics, theta_c = _fit_side(c2["C"].input, c2["C"].output)
    a_b_phi = np.asarray(build_envelope_bank(aend["B"].input, _WORKER_TERMS)[:, _WORKER_SUPPORT], dtype=np.complex128)
    c_b_phi = np.asarray(build_envelope_bank(c2["B"].input, _WORKER_TERMS)[:, _WORKER_SUPPORT], dtype=np.complex128)
    a_b_nmse = float(nmse(aend["B"].output[DMAX:], a_b_phi @ theta_a))
    c_b_nmse = float(nmse(c2["B"].output[DMAX:], c_b_phi @ theta_c))
    real_b = np.asarray(off["B"].output[DMAX:], dtype=np.complex128)
    if real_b.shape != (VALID_B_LENGTH,) or not np.all(np.isfinite(real_b)):
        raise RuntimeError(f"state {normalized_id} Real-B shape/finite gate failed")
    acpr_low = _scalar(data["acpr_low_withoutdpd"], "acpr_low_withoutdpd")
    acpr_upper = _scalar(data["acpr_upper_withoutdpd"], "acpr_upper_withoutdpd")
    state = _WORKER_STATE_TABLE[normalized_id]
    return {
        "State_n_R": normalized_id,
        "funMng": int(state["funMng"]),
        "funAng": int(state["funAng"]),
        "secMng": int(state["secMng"]),
        "secAng": int(state["secAng"]),
        "nmse_withoutdpd_dB": _scalar(data["nmse_withoutdpd"], "nmse_withoutdpd"),
        "ACPR_withoutdpd_mean_dBc": (acpr_low + acpr_upper) / 2.0,
        "ilc_A_end": aend_index + 1,
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
        "theta_Aend": theta_a,
        "theta_C2": theta_c,
        "real_B_valid": real_b,
    }


def model_worker_entry(state_id: int) -> dict[str, object]:
    return _state_worker(int(state_id))


def compute_cnmse_matrix(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    from behavior_fingerprint_ranking_consistency.shared.distance_matrix import (
        compute_cnmse_distance_matrix,
    )

    return compute_cnmse_distance_matrix(query, candidates)


def _finite_summary(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    return {
        "count": int(finite.size),
        "median_dB": float(np.median(finite)) if finite.size else float("nan"),
        "mean_dB": float(np.mean(finite)) if finite.size else float("nan"),
        "Q90_dB": float(np.quantile(finite, 0.90)) if finite.size else float("nan"),
        "Q95_dB": float(np.quantile(finite, 0.95)) if finite.size else float("nan"),
        "Q99_dB": float(np.quantile(finite, 0.99)) if finite.size else float("nan"),
        "worst_dB": float(np.max(finite)) if finite.size else float("nan"),
    }


__all__ = [
    "COMMON_B_RAW_LENGTH",
    "EXPECTED_COMMON_B_SHA",
    "EXPECTED_SUPPORT_HASH",
    "MODEL_NAME",
    "RESULT_ROOT",
    "STATE_COUNT",
    "TASK_NAME",
    "VALID_B_LENGTH",
    "compute_cnmse_matrix",
    "load_frozen_support",
    "model_worker_entry",
    "raw_manifest_gate",
    "support_hash",
    "verify_common_b_contract",
    "_finite_summary",
    "_worker_init",
]
