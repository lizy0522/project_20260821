"""Paired Envelope18 versus Frozen Envelope23-ilcEnd LUT retrieval."""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

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

from behavior_fingerprint_ranking_consistency.shared.distance_matrix import (  # noqa: E402
    compute_cnmse_distance_matrix,
)
from behavior_modeling.shared.basis_function_selection.envelope_dictionary import (  # noqa: E402
    DMAX,
    EnvelopeBasis,
    build_envelope_bank,
    build_envelope_dictionary,
)
from behavior_modeling.shared.basis_function_selection.hard20_envelope75_selection import (  # noqa: E402
    _raw_manifest,
)
from behavior_modeling.shared.basis_function_selection.model_solver import fit_ols  # noqa: E402
from core.shared.metrics import nmse  # noqa: E402
from data_management.shared import build_state_table, load_by_id  # noqa: E402
from signal_segmentation.shared import (  # noqa: E402
    build_off_segments,
    build_partition_from_xin,
    get_common_probe,
    get_ilc_pair,
    preprocess_full_pair,
)

TASK_NAME = "scenario_2_envelope23_ilcend_commonB_full_lut_retrieval_5B"
RESULT_ROOT = PROJECT_ROOT / "results" / "behavior_fingerprint_retrieval" / "scenario_2" /TASK_NAME
STATE_COUNT = 425
FULL_LENGTH = 24_576
COMMON_B_RAW_LENGTH = 4_915
VALID_B_LENGTH = COMMON_B_RAW_LENGTH - DMAX
WORKER_COUNT = 10
TIE_TOLERANCE_DB = 1e-12
THRESHOLD_DB = -40.0
COMMON_B_SOURCE_STATE = 0

E18_ROOT = PROJECT_ROOT / "results" / "behavior_fingerprint_retrieval" / "scenario_2" /"scenario_2_envelope18_commonB_full_lut_retrieval_5B"
E18_METRICS = E18_ROOT / "03_all425_retrieval_metrics.csv"
E18_SUMMARY = E18_ROOT / "10_final_result_summary.txt"
E18_COMMON_META = E18_ROOT / "01_common_B_probe_metadata.txt"
E18_DISTANCE = E18_ROOT / "05_fingerprint_cnmse_matrix.npy"
E18_MODEL_DEFINITION = (
    PROJECT_ROOT
    / "results"
    / "behavior_modeling"
    / "scenario_2" /"scenario_2_hard20_envelope75_basis_selection_5B"
    / "09_final_model_definition.csv"
)
E23_MODEL_ROOT = PROJECT_ROOT / "results" / "behavior_modeling" / "scenario_2" /"scenario_2_hard20_ilcend_envelope75_basis_selection_5B"
E23_MODEL_DEFINITION = E23_MODEL_ROOT / "08_final_model_definition.csv"

OLD_E18_FAILURE_IDS = {
    192,
    195,
    201,
    202,
    203,
    205,
    206,
    324,
    326,
    329,
    330,
    333,
    337,
    338,
    340,
    342,
    346,
    354,
    358,
    363,
    364,
    365,
}

_WORKER_TERMS: tuple[EnvelopeBasis, ...] | None = None
_WORKER_SUPPORT_E18: tuple[int, ...] | None = None
_WORKER_SUPPORT_E23: tuple[int, ...] | None = None
_WORKER_COMMON_PHI_E18: np.ndarray | None = None
_WORKER_COMMON_PHI_E23: np.ndarray | None = None
_WORKER_STATE_TABLE: dict[int, dict[str, int | float]] | None = None


def raw_manifest_gate() -> dict[str, object]:
    value = _raw_manifest()
    expected = {
        "sha256": "910b06e4be6b60b1f4dc63409a3083eb328765347fbec3b89a97a41d6633a1e0",
        "file_count": 429,
        "mat_count": 427,
        "bytes": 2_258_448_137,
    }
    if value != expected:
        raise RuntimeError(f"raw manifest differs from frozen baseline: {value}")
    return value


def _support_digest(ids: tuple[str, ...]) -> str:
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()


def _load_support(
    path: Path, expected_k: int
) -> tuple[tuple[EnvelopeBasis, ...], tuple[int, ...], str]:
    if not path.is_file():
        raise FileNotFoundError(f"Frozen model definition missing: {path}")
    frame = pd.read_csv(path)
    if len(frame) != expected_k or set(frame["K"].astype(int)) != {expected_k}:
        raise RuntimeError(f"Frozen support K mismatch in {path}")
    if set(np.asarray(frame["lambda"], dtype=float)) != {0.0}:
        raise RuntimeError(f"Frozen lambda mismatch in {path}")
    if set(frame["dmax"].astype(int)) != {DMAX}:
        raise RuntimeError(f"Frozen dmax mismatch in {path}")
    ids = tuple(frame["basis_id"].astype(str).tolist())
    if len(set(ids)) != expected_k:
        raise RuntimeError(f"Frozen support IDs are not unique in {path}")
    terms = tuple(build_envelope_dictionary())
    by_id = {term.basis_id: term for term in terms}
    missing = [basis_id for basis_id in ids if basis_id not in by_id]
    if missing:
        raise RuntimeError(f"Frozen support IDs missing from Envelope75: {missing}")
    indices = tuple(by_id[basis_id].index for basis_id in ids)
    index_column = "global_index" if "global_index" in frame else "term_index"
    if tuple(frame[index_column].astype(int).tolist()) != indices:
        raise RuntimeError(
            f"Frozen support column indices differ from canonical dictionary: {path}"
        )
    return terms, indices, _support_digest(ids)


def _parse_common_sha(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("sha256:"):
            return line.split(":", 1)[1].strip()
    raise RuntimeError(f"common-B metadata has no sha256: {path}")


def verify_old_baseline() -> tuple[pd.DataFrame, str]:
    if not E18_METRICS.is_file() or not E18_SUMMARY.is_file() or not E18_COMMON_META.is_file():
        raise FileNotFoundError("Frozen Envelope18 retrieval artifacts are incomplete")
    frame = pd.read_csv(E18_METRICS).sort_values("State_n_R").reset_index(drop=True)
    if len(frame) != STATE_COUNT or frame["State_n_R"].tolist() != list(range(STATE_COUNT)):
        raise RuntimeError("Frozen Envelope18 metrics are not canonical 425 rows")
    if int(frame["is_exact_state_hit"].sum()) != 170:
        raise RuntimeError("Frozen Envelope18 exact-hit baseline changed")
    if int(frame["retrieved_real_B_pass"].sum()) != 403:
        raise RuntimeError("Frozen Envelope18 Real-B pass baseline changed")
    old_failures = set(frame.loc[~frame["retrieved_real_B_pass"], "State_n_R"].astype(int))
    if old_failures != OLD_E18_FAILURE_IDS:
        raise RuntimeError(f"Frozen Envelope18 failure IDs changed: {sorted(old_failures)}")
    common_sha = _parse_common_sha(E18_COMMON_META)
    if common_sha != "b3a794e22b6beb7171f696d41685714da122f948ca8284553abaaa9a68c4abac":
        raise RuntimeError(f"Frozen Envelope18 common-B SHA changed: {common_sha}")
    return frame, common_sha


def build_common_b_probe() -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    data = load_by_id(COMMON_B_SOURCE_STATE)
    xin = np.asarray(data["xin"])
    if xin.ndim == 2 and 1 in xin.shape:
        xin = xin.reshape(-1)
    xin = np.asarray(xin, dtype=np.complex128)
    partition = build_partition_from_xin(xin)
    if (partition.n_a, partition.n_b, partition.n_c) != (12_288, 4_915, 7_373):
        raise RuntimeError("Canonical ABC bounds changed")
    common_b = np.asarray(get_common_probe(data, partition), dtype=np.complex128)
    if common_b.shape != (COMMON_B_RAW_LENGTH,) or not np.all(np.isfinite(common_b)):
        raise RuntimeError("common-B probe shape/finite gate failed")
    terms = tuple(build_envelope_dictionary())
    _, e18_indices, _ = _load_support(E18_MODEL_DEFINITION, 18)
    _, e23_indices, _ = _load_support(E23_MODEL_DEFINITION, 23)
    phi_full = build_envelope_bank(common_b, terms)
    phi_e18 = np.asarray(phi_full[:, e18_indices], dtype=np.complex128)
    phi_e23 = np.asarray(phi_full[:, e23_indices], dtype=np.complex128)
    sha = hashlib.sha256(np.ascontiguousarray(common_b).tobytes()).hexdigest()
    if sha != "b3a794e22b6beb7171f696d41685714da122f948ca8284553abaaa9a68c4abac":
        raise RuntimeError(f"common-B SHA does not match frozen Envelope18 baseline: {sha}")
    return common_b, phi_e18, {"sha256": sha, "phi_e23": phi_e23}


def compute_cnmse_matrix(query: np.ndarray, candidates: np.ndarray) -> np.ndarray:
    return compute_cnmse_distance_matrix(query, candidates)


def _fit_side(
    x: np.ndarray, y: np.ndarray, support: tuple[int, ...]
) -> tuple[dict[str, object], np.ndarray]:
    if _WORKER_TERMS is None:
        raise RuntimeError("paired retrieval worker terms are not initialized")
    full_bank = build_envelope_bank(x, _WORKER_TERMS)
    phi = np.asarray(full_bank[:, support], dtype=np.complex128)
    target = np.asarray(y[DMAX:], dtype=np.complex128)
    fit = fit_ols(phi, target, scale_columns=True)
    return {
        "train_NMSE_dB": float(fit.nmse_db),
        "rank": int(fit.rank),
        "condition_number": float(fit.condition_number),
        "finite": bool(
            np.all(np.isfinite(phi))
            and np.all(np.isfinite(fit.theta))
            and np.all(np.isfinite(fit.prediction))
            and np.isfinite(fit.condition_number)
        ),
    }, np.asarray(fit.theta, dtype=np.complex128)


def _worker_init(
    support_e18: tuple[int, ...],
    support_e23: tuple[int, ...],
    phi_e18: np.ndarray,
    phi_e23: np.ndarray,
) -> None:
    global _WORKER_TERMS, _WORKER_SUPPORT_E18, _WORKER_SUPPORT_E23
    global _WORKER_COMMON_PHI_E18, _WORKER_COMMON_PHI_E23, _WORKER_STATE_TABLE
    _WORKER_TERMS = tuple(build_envelope_dictionary())
    _WORKER_SUPPORT_E18 = tuple(support_e18)
    _WORKER_SUPPORT_E23 = tuple(support_e23)
    _WORKER_COMMON_PHI_E18 = np.asarray(phi_e18, dtype=np.complex128)
    _WORKER_COMMON_PHI_E23 = np.asarray(phi_e23, dtype=np.complex128)
    _WORKER_STATE_TABLE = {int(row["state_id"]): row for row in build_state_table()}


def evaluate_state(state_id: int) -> dict[str, object]:
    if (
        _WORKER_TERMS is None
        or _WORKER_SUPPORT_E18 is None
        or _WORKER_SUPPORT_E23 is None
        or _WORKER_COMMON_PHI_E18 is None
        or _WORKER_COMMON_PHI_E23 is None
        or _WORKER_STATE_TABLE is None
    ):
        raise RuntimeError("paired retrieval worker is not initialized")
    data = load_by_id(int(state_id))
    xin = np.asarray(data["xin"])
    if xin.ndim == 2 and 1 in xin.shape:
        xin = xin.reshape(-1)
    xin = np.asarray(xin, dtype=np.complex128)
    partition = build_partition_from_xin(xin)
    history = np.asarray(data["xin_pd_ori_ilc"])
    if history.ndim != 2 or history.shape[0] != FULL_LENGTH or history.shape[1] < 2:
        raise RuntimeError(f"State {state_id} ILC history is invalid")
    ilc_end = int(history.shape[1] - 1)
    a_pair = get_ilc_pair(data, ilc_end)
    c_pair = get_ilc_pair(data, 1)
    a = preprocess_full_pair(
        a_pair.input_full,
        a_pair.output_raw_full,
        partition,
        pair_type=a_pair.pair_type,
        iteration_index=a_pair.iteration_index,
        input_peak_normalization_factor=a_pair.input_peak_normalization_factor,
    )
    c = preprocess_full_pair(
        c_pair.input_full,
        c_pair.output_raw_full,
        partition,
        pair_type=c_pair.pair_type,
        iteration_index=c_pair.iteration_index,
        input_peak_normalization_factor=c_pair.input_peak_normalization_factor,
    )
    off = build_off_segments(data, partition)
    a18, theta_a18 = _fit_side(a["A"].input, a["A"].output, _WORKER_SUPPORT_E18)
    c18, theta_c18 = _fit_side(c["C"].input, c["C"].output, _WORKER_SUPPORT_E18)
    a23, theta_a23 = _fit_side(a["A"].input, a["A"].output, _WORKER_SUPPORT_E23)
    c23, theta_c23 = _fit_side(c["C"].input, c["C"].output, _WORKER_SUPPORT_E23)
    a_b18 = build_envelope_bank(a["B"].input, _WORKER_TERMS)[:, _WORKER_SUPPORT_E18] @ theta_a18
    c_b18 = build_envelope_bank(c["B"].input, _WORKER_TERMS)[:, _WORKER_SUPPORT_E18] @ theta_c18
    a_b23 = build_envelope_bank(a["B"].input, _WORKER_TERMS)[:, _WORKER_SUPPORT_E23] @ theta_a23
    c_b23 = build_envelope_bank(c["B"].input, _WORKER_TERMS)[:, _WORKER_SUPPORT_E23] @ theta_c23
    state = _WORKER_STATE_TABLE[int(state_id)]
    raw_nmse = float(np.asarray(data["nmse_withoutdpd"]).reshape(-1)[0])
    acpr = (
        float(np.asarray(data["acpr_low_withoutdpd"]).reshape(-1)[0])
        + float(np.asarray(data["acpr_upper_withoutdpd"]).reshape(-1)[0])
    ) / 2.0
    real_b = np.asarray(off["B"].output[DMAX:], dtype=np.complex128)
    return {
        "State_n_R": int(state_id),
        "funMng": int(state["funMng"]),
        "funAng": int(state["funAng"]),
        "secMng": int(state["secMng"]),
        "secAng": int(state["secAng"]),
        "nmse_withoutdpd_dB": raw_nmse,
        "ACPR_withoutdpd_mean_dBc": acpr,
        "ilc_end": ilc_end,
        "E18_Aend_train": a18,
        "E18_Aend_B": float(nmse(a["B"].output[DMAX:], a_b18)),
        "E18_C2_train": c18,
        "E18_C2_B": float(nmse(c["B"].output[DMAX:], c_b18)),
        "E23_Aend_train": a23,
        "E23_Aend_B": float(nmse(a["B"].output[DMAX:], a_b23)),
        "E23_C2_train": c23,
        "E23_C2_B": float(nmse(c["B"].output[DMAX:], c_b23)),
        "theta_Aend_E18": theta_a18,
        "theta_C2_E18": theta_c18,
        "theta_Aend_E23": theta_a23,
        "theta_C2_E23": theta_c23,
        "real_B": real_b,
    }


def worker_entry(state_id: int) -> dict[str, object]:
    return evaluate_state(int(state_id))


__all__ = [
    "COMMON_B_SOURCE_STATE",
    "E18_METRICS",
    "E18_ROOT",
    "E23_MODEL_DEFINITION",
    "RESULT_ROOT",
    "STATE_COUNT",
    "TASK_NAME",
    "VALID_B_LENGTH",
    "build_common_b_probe",
    "compute_cnmse_matrix",
    "raw_manifest_gate",
    "verify_old_baseline",
    "_worker_init",
    "worker_entry",
]
