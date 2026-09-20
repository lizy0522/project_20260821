"""Run fixed K18 Aend/C2 Full-425 retrieval under the shared 0.1B operator."""

# ruff: noqa: E402,E501

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import numpy as np
import pandas as pd

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
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
from behavior_modeling.shared.basis_function_selection.model_solver import fit_ridge  # noqa: E402
from core.shared.metrics import nmse  # noqa: E402
from data_management.shared import build_state_table, load_by_id  # noqa: E402
from low_bandwidth_behavior_analysis.shared.behavior_indexed_dpd_signal import (  # noqa: E402
    LowBandwidthObservationBank,
    LowBandwidthObservationOperator,
)
from signal_segmentation.shared import (  # noqa: E402
    build_off_segments,
    build_partition_from_xin,
    get_common_probe,
    get_ilc_pair,
    preprocess_full_pair,
)

from behavior_fingerprint_retrieval.scenario_2.scenario_2_k18_aend_c2_full_lut_retrieval_0p1b.plot_scenario_2_k18_aend_c2_full_lut_retrieval_0p1b import (  # noqa: E402
    write_figures,
)

TASK_NAME = "scenario_2_k18_aend_c2_full_lut_retrieval_0p1b"
RESULT_ROOT = PROJECT_ROOT / "results" / "behavior_fingerprint_retrieval" / "scenario_2" /TASK_NAME
WORK_LOG = PROJECT_ROOT / "work_logs" / "behavior_fingerprint_retrieval" / "scenario_2" /TASK_NAME / "execution_log.txt"
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"
FROZEN_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "scenario_2_self_first_joint_basis_ridge_lut_retrieval_search_5b"
FROZEN_SUMMARY_PATH = FROZEN_ROOT / "22_final_result_summary.json"
FROZEN_CHECKPOINT_PATH = FROZEN_ROOT / "23_checkpoint.json"
REFERENCE_5B_SUMMARY = PROJECT_ROOT / "results" / "behavior_fingerprint_retrieval" / "scenario_2" /"scenario_2_k18_aend_c2_full_lut_retrieval_5b" / "02_retrieval_summary.json"

STATE_COUNT = 425
FULL_LENGTH = 24_576
ABC_LENGTHS = (12_288, 4_915, 7_373)
VALID_5B_LENGTHS = (12_286, 4_913, 7_371)
COMMON_B_SOURCE_STATE = 0
COMMON_B_RAW_LENGTH = 4_915
COMMON_B_SHA256 = "b3a794e22b6beb7171f696d41685714da122f948ca8284553abaaa9a68c4abac"
RIDGE_LAMBDA = 3.16227766e-12
MODEL_CANDIDATE = "final_add_ENV_p04_m2_q1_lambda_3em12"
MODEL_K = 18
MODEL_P_MAX = 8
MODEL_SUPPORT_HASH = "263d4d30d09d651f9097fe86b50ad769f0b7e0cc7ac41c0d703f0a1dbf8b4f8c"
OBSERVATION_INDEX_K = 1
OBSERVATION_N = 0.1
FS_OBS_HZ = 2_000_000
REAL_B_THRESHOLD_DB = -40.0
TIE_TOLERANCE_DB = 1e-12
WORKER_COUNT = 10
BLAS_THREADS_PER_WORKER = 1
EXPECTED_RAW_MANIFEST = "910b06e4be6b60b1f4dc63409a3083eb328765347fbec3b89a97a41d6633a1e0"
EXPECTED_RAW_FILE_COUNT = 429
EXPECTED_RAW_MAT_COUNT = 427
EXPECTED_RAW_BYTES = 2_258_448_137

SUPPORT_IDS = (
    "LIN_d0", "LIN_d1", "LIN_d2", "ENV_p02_m0_q0", "ENV_p02_m1_q1", "ENV_p02_m2_q2",
    "ENV_p03_m0_q0", "ENV_p03_m0_q2", "ENV_p03_m1_q0", "ENV_p03_m1_q2",
    "ENV_p04_m0_q0", "ENV_p04_m0_q1", "ENV_p04_m2_q1", "ENV_p05_m0_q0",
    "ENV_p05_m2_q0", "ENV_p06_m0_q0", "ENV_p07_m0_q0", "ENV_p08_m0_q0",
)

_WORKER_TERMS: tuple[EnvelopeBasis, ...] | None = None
_WORKER_SUPPORT: tuple[int, ...] | None = None
_WORKER_COMMON_PHI: np.ndarray | None = None
_WORKER_STATE_TABLE: dict[int, dict[str, int | float]] | None = None
_WORKER_OPERATOR: LowBandwidthObservationOperator | None = None
_WORKER_OBS_LENGTHS: dict[str, int] | None = None


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot JSON serialize {type(value)!r}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _append_log(message: str) -> None:
    WORK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with WORK_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{_now()} | {message.rstrip()}\n")


def _append_handoff(message: str) -> None:
    HANDOFF_LOG.parent.mkdir(parents=True, exist_ok=True)
    with HANDOFF_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{_now()} | 新任务：{TASK_NAME}\n{message.rstrip()}\n")


def _vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2 and 1 in array.shape:
        array = array.reshape(-1)
    if array.ndim != 1 or array.size != FULL_LENGTH or not np.iscomplexobj(array):
        raise ValueError(f"{name} must be a complex vector of length {FULL_LENGTH}")
    array = np.asarray(array, dtype=np.complex128)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN or Inf")
    return array


def _scalar(value: Any, name: str) -> float:
    array = np.asarray(value).reshape(-1)
    if array.size != 1 or not np.isfinite(array[0]):
        raise ValueError(f"{name} must be one finite scalar")
    return float(array[0])


def raw_manifest() -> dict[str, object]:
    raw_root = PROJECT_ROOT / "data" / "raw"
    digest = hashlib.sha256()
    files = sorted((file for file in raw_root.rglob("*") if file.is_file()), key=lambda item: str(item).lower())
    total_bytes = 0
    mat_count = 0
    for file in files:
        digest.update(file.relative_to(raw_root).as_posix().encode("utf-8") + b"\0")
        size = int(file.stat().st_size)
        digest.update(size.to_bytes(8, "little"))
        total_bytes += size
        mat_count += int(file.suffix.lower() == ".mat")
    return {"sha256": digest.hexdigest(), "file_count": len(files), "mat_count": mat_count, "bytes": total_bytes}


def _raw_gate() -> dict[str, object]:
    value = raw_manifest()
    expected = {"sha256": EXPECTED_RAW_MANIFEST, "file_count": EXPECTED_RAW_FILE_COUNT, "mat_count": EXPECTED_RAW_MAT_COUNT, "bytes": EXPECTED_RAW_BYTES}
    if value != expected:
        raise RuntimeError(f"data/raw manifest differs from frozen baseline: {value}")
    return value


def _support_hash() -> str:
    payload = json.dumps(list(SUPPORT_IDS), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_frozen_model_contract() -> dict[str, Any]:
    summary = json.loads(FROZEN_SUMMARY_PATH.read_text(encoding="utf-8"))
    checkpoint = json.loads(FROZEN_CHECKPOINT_PATH.read_text(encoding="utf-8"))
    candidate = summary.get("best_candidate", {})
    basis_ids = tuple(json.loads(candidate["basis_ids"]))
    support_indices = tuple(json.loads(candidate["support_indices"]))
    by_id = {term.basis_id: term for term in build_envelope_dictionary()}
    expected_indices = tuple(by_id[item].index for item in SUPPORT_IDS)
    checks = {
        "status": summary.get("status") == "SUCCESS" and checkpoint.get("status") == "SUCCESS",
        "phase": checkpoint.get("phase") == "completed",
        "candidate": candidate.get("candidate_name") == MODEL_CANDIDATE,
        "K": int(candidate.get("K", -1)) == MODEL_K,
        "lambda": math.isclose(float(candidate.get("lambda")), RIDGE_LAMBDA, rel_tol=0.0, abs_tol=1e-24),
        "support_hash": candidate.get("support_hash") == MODEL_SUPPORT_HASH,
        "basis_ids": basis_ids == SUPPORT_IDS,
        "support_indices": support_indices == expected_indices,
        "hash_computed": _support_hash() == MODEL_SUPPORT_HASH,
    }
    if not all(checks.values()):
        raise RuntimeError(f"frozen K18 contract failed: {checks}")
    return {"candidate": MODEL_CANDIDATE, "K": MODEL_K, "lambda": RIDGE_LAMBDA, "dmax": DMAX, "p_max": MODEL_P_MAX, "support_hash": MODEL_SUPPORT_HASH, "support_ids": list(SUPPORT_IDS), "support_indices": list(expected_indices), "source_summary": str(FROZEN_SUMMARY_PATH), "source_checkpoint": str(FROZEN_CHECKPOINT_PATH), "checks": checks}


def _build_contract() -> tuple[tuple[EnvelopeBasis, ...], tuple[int, ...]]:
    terms = tuple(build_envelope_dictionary())
    by_id = {term.basis_id: term for term in terms}
    indices = tuple(int(by_id[item].index) for item in SUPPORT_IDS)
    if len(terms) != 75 or len(indices) != MODEL_K:
        raise RuntimeError("Envelope75/K18 contract failed")
    return terms, indices


def build_common_b(terms: tuple[EnvelopeBasis, ...], support: tuple[int, ...], operator: LowBandwidthObservationOperator) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    data = load_by_id(COMMON_B_SOURCE_STATE)
    xin = _vector(data["xin"], "xin")
    partition = build_partition_from_xin(xin)
    if (partition.n_a, partition.n_b, partition.n_c) != ABC_LENGTHS:
        raise RuntimeError("Common-B ABC lengths changed")
    common_b = np.asarray(get_common_probe(data, partition), dtype=np.complex128)
    if common_b.shape != (COMMON_B_RAW_LENGTH,):
        raise RuntimeError("Common-B raw length changed")
    digest = hashlib.sha256(np.ascontiguousarray(common_b).tobytes()).hexdigest()
    if digest != COMMON_B_SHA256:
        raise RuntimeError(f"Common-B SHA changed: {digest}")
    phi_5b = np.asarray(build_envelope_bank(common_b, terms)[:, support], dtype=np.complex128)
    phi_obs = np.asarray(operator.apply_matrix(phi_5b), dtype=np.complex128)
    metadata = {"source_state": COMMON_B_SOURCE_STATE, "raw_length": COMMON_B_RAW_LENGTH, "valid_5B_length": int(phi_5b.shape[0]), "observed_length": int(phi_obs.shape[0]), "dtype": str(common_b.dtype), "sha256": digest, "operator": operator.describe(), "construction": "5B Common-B K18 basis first, then observation operator columnwise"}
    return common_b, phi_obs, metadata


def _fit_side(segment: Any, operator: LowBandwidthObservationOperator, terms: tuple[EnvelopeBasis, ...], support: tuple[int, ...], expected_length: int) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    phi_5b = np.asarray(build_envelope_bank(segment.input, terms)[:, support], dtype=np.complex128)
    y_5b = np.asarray(segment.output[DMAX:], dtype=np.complex128)
    phi_obs, y_obs = operator.apply_pair(phi_5b, y_5b)
    if phi_obs.shape != (expected_length, MODEL_K) or y_obs.shape != (expected_length,):
        raise RuntimeError(f"0.1B fit shape mismatch: {phi_obs.shape}, {y_obs.shape}")
    fitted = fit_ridge(phi_obs, y_obs, RIDGE_LAMBDA)
    finite = bool(np.all(np.isfinite(phi_obs)) and np.all(np.isfinite(y_obs)) and np.all(np.isfinite(fitted.theta)) and np.all(np.isfinite(fitted.prediction)) and np.isfinite(fitted.condition_number))
    metrics = {"train_NMSE_dB": float(fitted.nmse_db), "rank": int(fitted.rank), "condition_number": float(fitted.condition_number), "coefficient_norm": float(np.linalg.norm(fitted.theta)), "finite": finite, "observed_train_length": int(phi_obs.shape[0])}
    return metrics, np.asarray(fitted.theta, dtype=np.complex128), phi_obs


def _worker_init(support: tuple[int, ...], common_phi: np.ndarray, obs_lengths: dict[str, int]) -> None:
    global _WORKER_TERMS, _WORKER_SUPPORT, _WORKER_COMMON_PHI, _WORKER_STATE_TABLE, _WORKER_OPERATOR, _WORKER_OBS_LENGTHS
    _WORKER_TERMS = tuple(build_envelope_dictionary())
    _WORKER_SUPPORT = tuple(int(item) for item in support)
    _WORKER_COMMON_PHI = np.asarray(common_phi, dtype=np.complex128)
    _WORKER_STATE_TABLE = {int(row["state_id"]): row for row in build_state_table()}
    _WORKER_OPERATOR = LowBandwidthObservationBank().get_by_index(OBSERVATION_INDEX_K)
    _WORKER_OBS_LENGTHS = {str(key): int(value) for key, value in obs_lengths.items()}
    if _WORKER_COMMON_PHI.shape != (_WORKER_OBS_LENGTHS["B"], MODEL_K):
        raise RuntimeError("Common-B observed basis shape changed")


def state_worker(state_id: int) -> dict[str, object]:
    if any(item is None for item in (_WORKER_TERMS, _WORKER_SUPPORT, _WORKER_COMMON_PHI, _WORKER_STATE_TABLE, _WORKER_OPERATOR, _WORKER_OBS_LENGTHS)):
        raise RuntimeError("0.1B worker is not initialized")
    data = load_by_id(int(state_id))
    xin = _vector(data["xin"], "xin")
    if _scalar(data["freqSample_Hz"], "freqSample_Hz") != 100_000_000.0 or _scalar(data["bandWidth_MHz"], "bandWidth_MHz") != 20.0:
        raise RuntimeError(f"state {state_id} is not 5B Scenario 2 data")
    partition = build_partition_from_xin(xin)
    histories = np.asarray(data["xin_pd_ori_ilc"])
    outputs = np.asarray(data["yout_withdpd_ori_ilc"])
    if histories.ndim != 2 or outputs.shape != histories.shape or histories.shape[1] < 2:
        raise RuntimeError(f"state {state_id} lacks Aend/C2")
    aend_index = histories.shape[1] - 1
    c2_index = 1
    aend_pair = get_ilc_pair(data, aend_index)
    c2_pair = get_ilc_pair(data, c2_index)
    canonical_a = preprocess_full_pair(aend_pair.input_full, aend_pair.output_raw_full, partition, pair_type="ILC", iteration_index=aend_pair.iteration_index, input_peak_normalization_factor=aend_pair.input_peak_normalization_factor)
    canonical_c = preprocess_full_pair(c2_pair.input_full, c2_pair.output_raw_full, partition, pair_type="ILC", iteration_index=c2_pair.iteration_index, input_peak_normalization_factor=c2_pair.input_peak_normalization_factor)
    off = build_off_segments(data, partition)
    a_metrics, theta_a, _ = _fit_side(canonical_a["A"], _WORKER_OPERATOR, _WORKER_TERMS, _WORKER_SUPPORT, _WORKER_OBS_LENGTHS["A"])
    c_metrics, theta_c, _ = _fit_side(canonical_c["C"], _WORKER_OPERATOR, _WORKER_TERMS, _WORKER_SUPPORT, _WORKER_OBS_LENGTHS["C"])
    phi_a_b = np.asarray(build_envelope_bank(canonical_a["B"].input, _WORKER_TERMS)[:, _WORKER_SUPPORT], dtype=np.complex128)
    phi_c_b = np.asarray(build_envelope_bank(canonical_c["B"].input, _WORKER_TERMS)[:, _WORKER_SUPPORT], dtype=np.complex128)
    y_a_b = np.asarray(canonical_a["B"].output[DMAX:], dtype=np.complex128)
    y_c_b = np.asarray(canonical_c["B"].output[DMAX:], dtype=np.complex128)
    phi_a_b_obs, y_a_b_obs = _WORKER_OPERATOR.apply_pair(phi_a_b, y_a_b)
    phi_c_b_obs, y_c_b_obs = _WORKER_OPERATOR.apply_pair(phi_c_b, y_c_b)
    if phi_a_b_obs.shape != (_WORKER_OBS_LENGTHS["B"], MODEL_K) or phi_c_b_obs.shape != (_WORKER_OBS_LENGTHS["B"], MODEL_K):
        raise RuntimeError(f"state {state_id} 0.1B B length mismatch")
    real_b = np.asarray(off["B"].output[DMAX:], dtype=np.complex128)
    state = _WORKER_STATE_TABLE[int(state_id)]
    low_acpr = _scalar(data["acpr_low_withoutdpd"], "acpr_low_withoutdpd")
    high_acpr = _scalar(data["acpr_upper_withoutdpd"], "acpr_upper_withoutdpd")
    return {"state_R": int(state_id), "funMng": int(state["funMng"]), "funAng": int(state["funAng"]), "secMng": int(state["secMng"]), "secAng": int(state["secAng"]), "Vm": float(state["Vm"]), "Pin": float(state["Pin"]), "nmse_withoutdpd_dB": _scalar(data["nmse_withoutdpd"], "nmse_withoutdpd"), "acpr_low_withoutdpd_dBc": low_acpr, "acpr_high_withoutdpd_dBc": high_acpr, "acpr_withoutdpd_mean_dBc": (low_acpr + high_acpr) / 2.0, "ilc_A_end": int(aend_index + 1), "ilc_C_2": 2, "Y_Aend_train_NMSE_dB": float(a_metrics["train_NMSE_dB"]), "Y_Aend_B_NMSE_dB": float(nmse(y_a_b_obs, phi_a_b_obs @ theta_a)), "Y_Aend_rank": int(a_metrics["rank"]), "Y_Aend_condition_number": float(a_metrics["condition_number"]), "Y_Aend_coefficient_norm": float(a_metrics["coefficient_norm"]), "Y_Aend_finite": bool(a_metrics["finite"]), "Y_C2_train_NMSE_dB": float(c_metrics["train_NMSE_dB"]), "Y_C2_B_NMSE_dB": float(nmse(y_c_b_obs, phi_c_b_obs @ theta_c)), "Y_C2_rank": int(c_metrics["rank"]), "Y_C2_condition_number": float(c_metrics["condition_number"]), "Y_C2_coefficient_norm": float(c_metrics["coefficient_norm"]), "Y_C2_finite": bool(c_metrics["finite"]), "theta_Aend": theta_a, "theta_C2": theta_c, "LUT_fingerprint": np.asarray(_WORKER_COMMON_PHI @ theta_a, dtype=np.complex128), "Query_fingerprint": np.asarray(_WORKER_COMMON_PHI @ theta_c, dtype=np.complex128), "real_B_valid": real_b}


def _top1(distance: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if distance.shape != (STATE_COUNT, STATE_COUNT) or np.isnan(distance).any() or np.isposinf(distance).any():
        raise RuntimeError("0.1B distance matrix invalid")
    ids = np.arange(STATE_COUNT, dtype=np.int64)
    selected = np.empty(STATE_COUNT, dtype=np.int64)
    selected_distance = np.empty(STATE_COUNT, dtype=np.float64)
    true_rank = np.empty(STATE_COUNT, dtype=np.int64)
    tie_count = np.empty(STATE_COUNT, dtype=np.int64)
    for state_id in range(STATE_COUNT):
        row = distance[state_id]
        minimum = float(np.min(row))
        tied = np.flatnonzero(row <= minimum + TIE_TOLERANCE_DB)
        selected[state_id] = int(np.min(tied))
        selected_distance[state_id] = float(row[selected[state_id]])
        tie_count[state_id] = int(tied.size)
        order = np.lexsort((ids, row))
        true_rank[state_id] = int(np.flatnonzero(order == state_id)[0] + 1)
    return selected, selected_distance, true_rank, tie_count


def _basis_definition(terms: tuple[EnvelopeBasis, ...], support: tuple[int, ...]) -> list[dict[str, object]]:
    return [{"basis_index": i, "global_dictionary_index": int(index), "basis_name": terms[index].basis_id, "p": int(terms[index].order), "m": int(terms[index].signal_delay), "q": None if terms[index].envelope_delay is None else int(terms[index].envelope_delay), "type": "LINEAR" if terms[index].order == 1 else "ENVELOPE", "formula": terms[index].formula} for i, index in enumerate(support)]


def _load_config(row: dict[str, object]) -> str:
    return f"funMng={int(row['funMng'])}, funAng={int(row['funAng'])}, secMng={int(row['secMng'])}, secAng={int(row['secAng'])}"


def _finite_stats(values: np.ndarray) -> dict[str, object]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {"count": 0, "median_dB": None, "Q05_dB": None, "Q95_dB": None, "worst_dB": None}
    return {"count": int(finite.size), "median_dB": float(np.median(finite)), "mean_dB": float(np.mean(finite)), "Q05_dB": float(np.quantile(finite, 0.05)), "Q95_dB": float(np.quantile(finite, 0.95)), "worst_dB": float(np.max(finite))}


def _write_checkpoint(phase: str, **payload: object) -> None:
    _write_json(RESULT_ROOT / "09_checkpoint.json", {"task_name": TASK_NAME, "phase": phase, "state_count": STATE_COUNT, "K": MODEL_K, "lambda": RIDGE_LAMBDA, "bandwidth": "0.1B", "Fs_obs_Hz": FS_OBS_HZ, "observation_index_k": OBSERVATION_INDEX_K, "support_hash": MODEL_SUPPORT_HASH, **payload})


def run(*, allow_existing: bool = False) -> dict[str, object]:
    if RESULT_ROOT.exists() and any(RESULT_ROOT.iterdir()) and not allow_existing:
        raise RuntimeError(f"result directory is not empty: {RESULT_ROOT}")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    WORK_LOG.parent.mkdir(parents=True, exist_ok=True)
    _append_log("START fixed K18 0.1B Aend/C2 Full-425 retrieval")
    raw_before = _raw_gate()
    frozen = load_frozen_model_contract()
    terms, support = _build_contract()
    bank = LowBandwidthObservationBank()
    operator = bank.get_by_index(OBSERVATION_INDEX_K)
    if operator.spec.normalized_bandwidth_n != OBSERVATION_N or operator.spec.fs_out_hz != FS_OBS_HZ or operator.spec.up != 1 or operator.spec.down != 50:
        raise RuntimeError("0.1B observation operator contract failed")
    obs_lengths = {name: operator.expected_output_length(length) for name, length in zip(("A", "B", "C"), VALID_5B_LENGTHS, strict=True)}
    common_b, common_phi, common_meta = build_common_b(terms, support, operator)
    model_definition = {"candidate": MODEL_CANDIDATE, "K": MODEL_K, "lambda": RIDGE_LAMBDA, "dmax": DMAX, "p_max": MODEL_P_MAX, "support_hash": MODEL_SUPPORT_HASH, "support": _basis_definition(terms, support), "basis_definition": "phi[p,m,q](n)=x[n-m] * abs(x[n-q])**(p-1)", "source_frozen_summary": str(FROZEN_SUMMARY_PATH)}
    _write_json(RESULT_ROOT / "00_preflight_contract.json", {"frozen_model": frozen, "operator": common_meta["operator"], "Common_B": common_meta, "raw_manifest_before": raw_before, "observed_segment_lengths": obs_lengths, "fingerprint_length": obs_lengths["B"]})
    _write_json(RESULT_ROOT / "03_model_definition.json", model_definition)
    (RESULT_ROOT / "00_task_definition.txt").write_text("\n".join([f"Task: {TASK_NAME}", "Fixed K18 0.1B Aend/C2 Full-425 LUT retrieval.", "Observation order: 5B K18 basis -> shared 0.1B observation operator.", f"operator: {json.dumps(common_meta['operator'], ensure_ascii=False, sort_keys=True)}", f"observed segment lengths: {obs_lengths}", f"fingerprint length: {obs_lengths['B']}", "Aend: last valid ILC column; C2: second ILC column.", "Final Real-B validation: original 5B yout_withoutdpd_ori B segment.", "No model search, lambda search, Self-First rerun, Nested CV, or 5B recomputation." ]) + "\n", encoding="utf-8")
    _write_checkpoint("preflight_complete", raw_manifest_before=raw_before, operator=common_meta["operator"], observed_segment_lengths=obs_lengths, fingerprint_length=obs_lengths["B"])
    context = get_context("spawn")
    model_results: list[dict[str, object]] = []
    start = time.time()
    with ProcessPoolExecutor(max_workers=WORKER_COUNT, mp_context=context, initializer=_worker_init, initargs=(support, common_phi, obs_lengths)) as executor:
        futures = {executor.submit(state_worker, state_id): state_id for state_id in range(STATE_COUNT)}
        for completed, future in enumerate(as_completed(futures), start=1):
            state_id = futures[future]
            try:
                model_results.append(future.result())
            except Exception as exc:
                for pending in futures:
                    pending.cancel()
                raise RuntimeError(f"0.1B state {state_id} failed") from exc
            if completed == 1 or completed % 25 == 0 or completed == STATE_COUNT:
                print(f"0.1B K18 model states {completed}/{STATE_COUNT}", flush=True)
    model_results.sort(key=lambda row: int(row["state_R"]))
    if [int(row["state_R"]) for row in model_results] != list(range(STATE_COUNT)):
        raise RuntimeError("0.1B model state order changed")
    lut = np.stack([row["LUT_fingerprint"] for row in model_results]).astype(np.complex128)
    query = np.stack([row["Query_fingerprint"] for row in model_results]).astype(np.complex128)
    real_b = np.stack([row["real_B_valid"] for row in model_results]).astype(np.complex128)
    theta_a = np.stack([row["theta_Aend"] for row in model_results]).astype(np.complex128)
    theta_c = np.stack([row["theta_C2"] for row in model_results]).astype(np.complex128)
    if lut.shape != (STATE_COUNT, obs_lengths["B"]) or query.shape != lut.shape or not np.all(np.isfinite(lut)) or not np.all(np.isfinite(query)):
        raise RuntimeError("0.1B fingerprint shape/finite contract failed")
    distance = compute_cnmse_distance_matrix(query, lut)
    real_b_distance = compute_cnmse_distance_matrix(real_b, real_b)
    selected, selected_distance, true_rank, tie_count = _top1(distance)
    ids = np.arange(STATE_COUNT, dtype=np.int64)
    retrieved_real_b = real_b_distance[ids, selected]
    self_hit = selected == ids
    shareable = retrieved_real_b < REAL_B_THRESHOLD_DB
    fallback = (~self_hit) & shareable
    fail = (~self_hit) & (~shareable)
    classes = np.full(STATE_COUNT, "FAIL", dtype=object)
    classes[self_hit] = "SELF"
    classes[fallback] = "FALLBACK"
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for index, row in enumerate(model_results):
        row_public = {key: value for key, value in row.items() if key not in {"theta_Aend", "theta_C2", "LUT_fingerprint", "Query_fingerprint", "real_B_valid"}}
        row_public.update({"state_R": index, "load_config": _load_config(row_public), "retrieved_state_Q": int(selected[index]), "retrieved_state_delta": int(selected[index] - index), "retrieved_state_abs_delta": int(abs(selected[index] - index)), "retrieval_fingerprint_CNMSE_dB": float(selected_distance[index]), "retrieved_real_B_CNMSE_dB": float(retrieved_real_b[index]), "self_hit": bool(self_hit[index]), "real_B_shareable_lt_minus40": bool(shareable[index]), "retrieval_class": str(classes[index]), "true_state_rank": int(true_rank[index]), "minimum_tie_count": int(tie_count[index])})
        rows.append(row_public)
        diagnostics.append({"State_R": index, "State_Q": int(selected[index]), "State_Q_minus_State_R": int(selected[index] - index), "abs_State_Q_minus_State_R": int(abs(selected[index] - index)), "retrieval_fingerprint_CNMSE_dB": float(selected_distance[index]), "retrieved_real_B_CNMSE_dB": float(retrieved_real_b[index]), "retrieval_class": str(classes[index])})
    frame = pd.DataFrame(rows).sort_values("state_R").reset_index(drop=True)
    main_columns = ["state_R", "load_config", "nmse_withoutdpd_dB", "acpr_withoutdpd_mean_dBc", "Y_Aend_train_NMSE_dB", "Y_Aend_B_NMSE_dB", "Y_C2_train_NMSE_dB", "Y_C2_B_NMSE_dB", "retrieved_state_Q", "retrieved_real_B_CNMSE_dB"]
    frame[main_columns].to_csv(RESULT_ROOT / "01_state_results.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(RESULT_ROOT / "11_state_retrieval_diagnostics.csv", index=False)
    pd.DataFrame(rows).to_csv(RESULT_ROOT / "03_retrieval_diagnostics.csv", index=False)
    np.save(RESULT_ROOT / "04_fingerprint_distance_matrix_0p1b.npy", distance)
    np.savez_compressed(RESULT_ROOT / "05_lut_fingerprints_0p1b.npz", state_ids=ids, fingerprints=lut, common_B_input=common_b, observation_index_k=np.array(OBSERVATION_INDEX_K, dtype=np.int64))
    np.savez_compressed(RESULT_ROOT / "06_query_fingerprints_0p1b.npz", state_ids=ids, fingerprints=query, common_B_input=common_b, observation_index_k=np.array(OBSERVATION_INDEX_K, dtype=np.int64))
    np.savez_compressed(RESULT_ROOT / "10_model_coefficients_0p1b.npz", state_ids=ids, theta_Aend=theta_a, theta_C2=theta_c)
    xlsx_source = frame[main_columns].copy()
    xlsx_source["retrieved_real_B_CNMSE_dB"] = xlsx_source["retrieved_real_B_CNMSE_dB"].map(lambda value: "-Inf" if np.isneginf(float(value)) else float(value))
    _write_json(RESULT_ROOT / "00_xlsx_source.json", {"headers": main_columns, "rows": xlsx_source.to_dict(orient="records")})
    figure_meta = write_figures(frame)
    finite_real = _finite_stats(retrieved_real_b)
    counts = {"N_self": int(self_hit.sum()), "N_fallback": int(fallback.sum()), "N_valid": int((self_hit | fallback).sum()), "N_fail": int(fail.sum())}
    delta = np.abs(selected - ids)
    baseline = {}
    if REFERENCE_5B_SUMMARY.is_file():
        ref = json.loads(REFERENCE_5B_SUMMARY.read_text(encoding="utf-8"))
        baseline = {"baseline_5B_N_self": ref.get("N_self"), "baseline_5B_N_valid": ref.get("N_valid"), "baseline_5B_N_fail": ref.get("N_fail")}
    raw_after = _raw_gate()
    summary = {"status": "NUMERICAL_COMPLETE", "task_name": TASK_NAME, "module": "behavior_fingerprint_retrieval", "bandwidth": "0.1B", "B_MHz": 20, "Fs_original_MHz": 100, "Fs_observation_MHz": 2, "resampling_ratio": "1/50", "operator_name": common_meta["operator"]["resampler"], "operator_metadata": common_meta["operator"], "state_count": STATE_COUNT, "allow_self": True, "model_candidate": MODEL_CANDIDATE, "K": MODEL_K, "lambda": RIDGE_LAMBDA, "dmax": DMAX, "p_max": MODEL_P_MAX, "support_hash": MODEL_SUPPORT_HASH, "Aend_definition": "last valid ILC column", "C2_definition": "literal second ILC column", "Common_B_sha256": COMMON_B_SHA256, "observed_segment_lengths": obs_lengths, "fingerprint_length": int(obs_lengths["B"]), "distance_matrix_shape": list(distance.shape), **counts, "self_rate": counts["N_self"] / STATE_COUNT, "valid_rate": counts["N_valid"] / STATE_COUNT, "fail_rate": counts["N_fail"] / STATE_COUNT, "fail_states": np.flatnonzero(fail).astype(int).tolist(), "mean_abs_state_delta": float(np.mean(delta)), "median_abs_state_delta": float(np.median(delta)), "max_abs_state_delta": int(np.max(delta)), "finite_real_B_CNMSE": finite_real, "shareability_threshold_dB": REAL_B_THRESHOLD_DB, "shareability_operator": "<", "raw_manifest_before": raw_before, "raw_manifest_after": raw_after, "raw_unchanged": raw_before == raw_after, "model_metric_medians": {column: float(frame[column].median()) for column in ("Y_Aend_train_NMSE_dB", "Y_Aend_B_NMSE_dB", "Y_C2_train_NMSE_dB", "Y_C2_B_NMSE_dB")}, "figure_metadata": figure_meta, "model_search_started": False, "nested_cv_started": False, "self_first_runner_started": False, "construction_order": "5B K18 basis -> shared 0.1B observation operator", "final_validation_domain": "5B Real-B yout_withoutdpd_ori B segment", "outputs": {"csv": str(RESULT_ROOT / "01_state_results.csv"), "summary": str(RESULT_ROOT / "02_retrieval_summary.json"), "model_definition": str(RESULT_ROOT / "03_model_definition.json"), "distance": str(RESULT_ROOT / "04_fingerprint_distance_matrix_0p1b.npy"), "xlsx": str(RESULT_ROOT / "scenario_2_k18_aend_c2_full_lut_retrieval_0p1b.xlsx"), "figure_01": str(RESULT_ROOT / "figure_01_full_metrics_vs_state_0p1b.png"), "figure_02": str(RESULT_ROOT / "figure_02_retrieved_real_B_CNMSE_vs_state_0p1b.png")}, **baseline}
    _write_json(RESULT_ROOT / "02_retrieval_summary.json", summary)
    summary_lines = [f"Task: {TASK_NAME}", "Status: NUMERICAL_COMPLETE; final SUCCESS awaits artifact-tool Excel and independent validator.", f"Observation: 0.1B = 2 MHz; original Fs=100 MHz; ratio=1/50; operator={common_meta['operator']['resampler']}", f"Observed lengths: {obs_lengths}; fingerprint length={obs_lengths['B']}", f"Model: {MODEL_CANDIDATE}; K={MODEL_K}; lambda={RIDGE_LAMBDA:.12g}; support_hash={MODEL_SUPPORT_HASH}", "LUT=Aend; Query=C2; allow_self=true; final Real-B validation uses 5B data.", "", *[f"{key}={value}" for key, value in counts.items()], f"self_rate={summary['self_rate']}", f"valid_rate={summary['valid_rate']}", f"fail_states={summary['fail_states']}", f"mean_abs(State_Q-State_R)={summary['mean_abs_state_delta']}", f"median_abs(State_Q-State_R)={summary['median_abs_state_delta']}", f"max_abs(State_Q-State_R)={summary['max_abs_state_delta']}", f"finite Real-B={json.dumps(finite_real, ensure_ascii=False, sort_keys=True)}", f"model_metric_medians={json.dumps(summary['model_metric_medians'], ensure_ascii=False, sort_keys=True)}", "Outputs: strict 10-column CSV source, artifact-tool Excel, two figures with all R->Q labels."]
    (RESULT_ROOT / "07_final_result_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    _write_checkpoint("numerical_outputs_complete", raw_unchanged=bool(summary["raw_unchanged"]), counts=counts, validation_status="pending_artifact_tool_and_validator", outputs=summary["outputs"])
    _append_log(f"NUMERICAL OUTPUTS COMPLETE operator={common_meta['operator']['resampler']} obs_lengths={obs_lengths} fingerprint_length={obs_lengths['B']} counts={counts} raw_unchanged={summary['raw_unchanged']} elapsed_s={time.time()-start:.1f}")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--allow-existing", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(allow_existing=args.allow_existing), ensure_ascii=False, indent=2, default=_json_default))


if __name__ == "__main__":
    main()
