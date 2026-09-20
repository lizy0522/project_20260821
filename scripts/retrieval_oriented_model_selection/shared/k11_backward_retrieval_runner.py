# ruff: noqa: E402,E501,I001

"""Run one-step retrieval-oriented backward elimination from the best K11."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from multiprocessing import get_context
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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

from core.shared.metrics import cnmse  # noqa: E402
from behavior_modeling.shared.basis_function_selection.envelope_dictionary import (  # noqa: E402
    EnvelopeBasis,
    build_envelope_bank,
    build_envelope_dictionary,
    dictionary_gate,
)
from behavior_fingerprint_retrieval.shared.envelope19_c2endshared_commonB_full_lut import (  # noqa: E402
    EXPECTED_COMMON_B_SHA,
    raw_manifest_gate,
    verify_common_b_contract,
)
from data_management.shared import load_by_id  # noqa: E402
from signal_segmentation.shared import build_off_segments, build_partition_from_xin  # noqa: E402
from retrieval_oriented_model_selection.shared import k11_backward_retrieval_support as backend
from retrieval_oriented_model_selection.shared import mp10_k9_retrieval_runner as k9_utils

TASK_NAME = "scenario_2_k11_backward_retrieval_oriented_basis_ablation_5b"
RESULT_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /TASK_NAME
WORK_LOG = PROJECT_ROOT / "work_logs" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /TASK_NAME / "execution_log.txt"
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"
K11_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /"scenario_2_mp10_k11_retrieval_oriented_forward_scan_5b"
K9_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /"scenario_2_mp10_k9_retrieval_oriented_basis_ablation_5b"
MP10_ROOT = PROJECT_ROOT / "results" / "behavior_fingerprint_retrieval" / "scenario_2_mp10_full_lut_retrieval_5b"
BEST_K11_NAME = "MP10_plus_ENV_p04_m2_q0"
BEST_K11_BASIS_ID = "ENV_p04_m2_q0"
STATE_COUNT = backend.STATE_COUNT
WORKER_COUNT = 10
BLAS_THREADS_PER_WORKER = 1
REAL_B_THRESHOLD_DB = -40.0
REAL_B_SANITY_TOLERANCE_DB = 1e-9


@dataclass(frozen=True)
class BackwardCandidate:
    candidate_id: int
    candidate_name: str
    K: int
    removed_basis_id: str | None
    removed_order_p: int | None
    removed_delay_m: int | None
    removed_envelope_delay_q: int | None
    removed_family: str | None
    removed_is_cross_delay: bool | None
    removed_index: int | None
    support: tuple[int, ...]
    retained_basis_ids: tuple[str, ...]
    support_hash: str


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _append_log(message: str) -> None:
    WORK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with WORK_LOG.open("a", encoding="utf-8") as handle:
        handle.write(message)


def _append_handoff(message: str) -> None:
    HANDOFF_LOG.parent.mkdir(parents=True, exist_ok=True)
    with HANDOFF_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{_now()} | 新任务：{TASK_NAME}\n{message.rstrip()}\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot JSON serialize {type(value)!r}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _checkpoint(**payload: Any) -> None:
    _write_json(
        RESULT_ROOT / "20_checkpoint.json",
        {
            "task_name": TASK_NAME,
            "mode": "new_task",
            "state_count": STATE_COUNT,
            "worker_count": WORKER_COUNT,
            "blas_threads_per_worker": BLAS_THREADS_PER_WORKER,
            "abc_bounds": [0, 12_288, 17_203, 24_576],
            "common_B_source_state": 0,
            "common_B_sha256": EXPECTED_COMMON_B_SHA,
            "real_B_threshold_dB": REAL_B_THRESHOLD_DB,
            "real_B_sanity_tolerance_dB": REAL_B_SANITY_TOLERANCE_DB,
            "allow_self": True,
            "k11_baseline": BEST_K11_NAME,
            "candidate_count": 12,
            "deletion_candidate_count": 11,
            "dmax_fixed": backend.MP_DMAX,
            "ridge_lambda_fixed": backend.RIDGE_LAMBDA,
            "second_deletion_performed": False,
            "k12_performed": False,
            "multi_basis_pruning_performed": False,
            "swap_performed": False,
            "lambda_scan_performed": False,
            "order_scan_performed": False,
            "memory_scan_performed": False,
            "top_k_selection_performed": False,
            "real_B_used_for_top1_selection": False,
            **payload,
        },
    )


def _support_hash(ids: tuple[str, ...]) -> str:
    return hashlib.sha256(json.dumps(list(ids), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _load_frozen_k11_support() -> tuple[tuple[str, ...], dict[str, EnvelopeBasis], int, pd.DataFrame]:
    support_path = K11_ROOT / "04_candidate_supports.csv"
    if not support_path.is_file():
        raise FileNotFoundError(f"missing formal K11 support: {support_path}")
    support_frame = pd.read_csv(support_path)
    row = support_frame.loc[support_frame["model_name"].eq(BEST_K11_NAME)]
    if row.shape[0] != 1:
        raise RuntimeError("formal best K11 support row is missing or duplicated")
    support_ids = tuple(json.loads(row.iloc[0]["support_basis_ids"]))
    if len(support_ids) != 11 or support_ids[-1] != BEST_K11_BASIS_ID:
        raise RuntimeError(f"formal best K11 support changed: {support_ids}")
    envelope_terms = tuple(build_envelope_dictionary())
    envelope_by_id = {term.basis_id: term for term in envelope_terms}
    missing = [basis_id for basis_id in support_ids if basis_id not in envelope_by_id]
    if missing:
        raise RuntimeError(f"best K11 basis IDs missing from Envelope75: {missing}")
    added_index = int(envelope_by_id[BEST_K11_BASIS_ID].index)
    if len(envelope_terms) != 75:
        raise RuntimeError("Envelope75 dictionary count changed")
    return support_ids, envelope_by_id, added_index, support_frame


def _build_candidates(
    support_ids: tuple[str, ...],
    envelope_by_id: dict[str, EnvelopeBasis],
) -> tuple[pd.DataFrame, tuple[BackwardCandidate, ...]]:
    baseline = BackwardCandidate(
        candidate_id=0,
        candidate_name="K11_full",
        K=11,
        removed_basis_id=None,
        removed_order_p=None,
        removed_delay_m=None,
        removed_envelope_delay_q=None,
        removed_family=None,
        removed_is_cross_delay=None,
        removed_index=None,
        support=tuple(range(11)),
        retained_basis_ids=support_ids,
        support_hash=_support_hash(support_ids),
    )
    candidates = [baseline]
    support_rows = [
        {
            "candidate_id": 0,
            "candidate_name": baseline.candidate_name,
            "K": 11,
            "removed_basis_id": "NONE",
            "removed_basis_p": np.nan,
            "removed_basis_m": np.nan,
            "removed_basis_q": np.nan,
            "removed_is_cross_delay": False,
            "retained_basis_ids": json.dumps(list(support_ids), ensure_ascii=False),
            "support_indices": json.dumps(list(baseline.support)),
            "support_hash": baseline.support_hash,
            "dmax": backend.MP_DMAX,
            "ridge_lambda": backend.RIDGE_LAMBDA,
        }
    ]
    for index, basis_id in enumerate(support_ids):
        term = envelope_by_id[basis_id]
        support = tuple(item for item in range(11) if item != index)
        retained = tuple(support_ids[item] for item in support)
        is_cross = bool(term.order > 1 and term.signal_delay != term.envelope_delay)
        candidate = BackwardCandidate(
            candidate_id=index + 1,
            candidate_name=f"K11_minus_{basis_id}",
            K=10,
            removed_basis_id=basis_id,
            removed_order_p=int(term.order),
            removed_delay_m=None if term.signal_delay is None else int(term.signal_delay),
            removed_envelope_delay_q=None if term.envelope_delay is None else int(term.envelope_delay),
            removed_family=term.family,
            removed_is_cross_delay=is_cross,
            removed_index=index,
            support=support,
            retained_basis_ids=retained,
            support_hash=_support_hash(retained),
        )
        candidates.append(candidate)
        support_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "candidate_name": candidate.candidate_name,
                "K": candidate.K,
                "removed_basis_id": basis_id,
                "removed_basis_p": int(term.order),
                "removed_basis_m": term.signal_delay,
                "removed_basis_q": term.envelope_delay,
                "removed_is_cross_delay": is_cross,
                "retained_basis_ids": json.dumps(list(retained), ensure_ascii=False),
                "support_indices": json.dumps(list(support)),
                "support_hash": candidate.support_hash,
                "dmax": backend.MP_DMAX,
                "ridge_lambda": backend.RIDGE_LAMBDA,
            }
        )
    support_frame = pd.DataFrame(support_rows)
    support_frame.to_csv(RESULT_ROOT / "03_deletion_candidate_supports.csv", index=False)
    baseline_frame = pd.DataFrame(
        [
            {
                "baseline_column_index": index,
                "basis_id": basis_id,
                "order_p": int(envelope_by_id[basis_id].order),
                "delay_m": envelope_by_id[basis_id].signal_delay,
                "envelope_delay_q": envelope_by_id[basis_id].envelope_delay,
                "basis_family": envelope_by_id[basis_id].family,
            }
            for index, basis_id in enumerate(support_ids)
        ]
    )
    baseline_frame.to_csv(RESULT_ROOT / "02_k11_baseline_support.csv", index=False)
    return support_frame, tuple(candidates)


def _load_references() -> dict[str, Any]:
    previous_mp10 = k9_utils._load_previous_baseline()
    k11_model = pd.read_csv(K11_ROOT / "06_candidate_model_quality_long.csv")
    k11_query = pd.read_csv(K11_ROOT / "09_candidate_query_metrics_long.csv")
    with np.load(K11_ROOT / "07_candidate_coefficients.npz", allow_pickle=False) as data:
        k11_candidate_ids = np.asarray(data["model_ids"], dtype=np.int64)
        k11_theta_a = np.asarray(data["theta_Aend_padded"])
        k11_theta_c = np.asarray(data["theta_C2_padded"])
    k11_distance = np.load(K11_ROOT / "distance_matrices" / f"{BEST_K11_NAME}.npy")
    k11_lut = np.load(K11_ROOT / "fingerprints" / "diagnostic_best_k11_lut.npy")
    k11_query_fp = np.load(K11_ROOT / "fingerprints" / "diagnostic_best_k11_query.npy")
    real_b_distance = np.load(K9_ROOT / "04_realB_ground_truth_cnmse_matrix.npy")
    shareability = np.load(K9_ROOT / "05_realB_shareability_mask.npy")
    folds_k9 = pd.read_csv(K9_ROOT / "12_query_state_cv_folds.csv").sort_values("state_id").reset_index(drop=True)
    folds_k11 = pd.read_csv(K11_ROOT / "13_query_state_cv_folds.csv").sort_values("state_id").reset_index(drop=True)
    if not np.array_equal(folds_k9[["state_id", "fold"]].to_numpy(), folds_k11[["state_id", "fold"]].to_numpy()):
        raise RuntimeError("K11 and K9 Query-State folds are not identical")
    if real_b_distance.shape != (STATE_COUNT, STATE_COUNT) or shareability.shape != real_b_distance.shape:
        raise RuntimeError("Real-B ground truth/mask shape changed")
    if not np.all(np.isneginf(np.diag(real_b_distance))) or not shareability.diagonal().all():
        raise RuntimeError("Real-B diagonal/shareability contract changed")
    if k11_theta_a.shape != (66, STATE_COUNT, 11) or k11_theta_c.shape != k11_theta_a.shape:
        raise RuntimeError("formal K11 coefficient shape changed")
    return {
        "mp10": previous_mp10,
        "k11_model": k11_model,
        "k11_query": k11_query,
        "k11_candidate_ids": k11_candidate_ids,
        "k11_theta_a": k11_theta_a,
        "k11_theta_c": k11_theta_c,
        "k11_distance": k11_distance,
        "k11_lut": k11_lut,
        "k11_query_fp": k11_query_fp,
        "real_b_distance": real_b_distance,
        "shareability": shareability,
        "folds": folds_k11,
    }


def _random_real_b_check(real_b_distance: np.ndarray) -> float:
    rng = np.random.default_rng(20260916)
    cache: dict[int, np.ndarray] = {}
    maximum = 0.0
    for real_id, query_id in rng.integers(0, STATE_COUNT, size=(20, 2)):
        for state_id in (int(real_id), int(query_id)):
            if state_id not in cache:
                data = load_by_id(state_id)
                partition = build_partition_from_xin(np.asarray(data["xin"]))
                cache[state_id] = np.asarray(build_off_segments(data, partition)["B"].output[2:], dtype=np.complex128)
        direct = cnmse(cache[int(real_id)], cache[int(query_id)])
        stored = float(real_b_distance[int(real_id), int(query_id)])
        if np.isneginf(direct) and np.isneginf(stored):
            continue
        maximum = max(maximum, abs(direct - stored))
    if maximum > REAL_B_SANITY_TOLERANCE_DB:
        raise RuntimeError(f"Real-B matrix random-pair check failed: {maximum:.3e} dB")
    return maximum


def _run_model_phase(candidates: tuple[BackwardCandidate, ...], added_index: int, resume: bool) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    model_path = RESULT_ROOT / "05_candidate_model_quality_long.csv"
    coefficient_path = RESULT_ROOT / "06_candidate_coefficients.npz"
    if resume and model_path.is_file() and coefficient_path.is_file():
        try:
            frame = pd.read_csv(model_path)
            with np.load(coefficient_path, allow_pickle=False) as data:
                theta_a = np.asarray(data["theta_Aend_padded"])
                theta_c = np.asarray(data["theta_C2_padded"])
            if theta_a.shape == (len(candidates), STATE_COUNT, backend.K11) and theta_c.shape == theta_a.shape:
                return frame, theta_a, theta_c
        except (OSError, ValueError, KeyError):
            pass
    specs = tuple((candidate.candidate_id, candidate.removed_index) for candidate in candidates)
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=WORKER_COUNT,
        mp_context=get_context("spawn"),
        initializer=backend.worker_init,
        initargs=(specs, added_index),
    ) as executor:
        futures = {
            executor.submit(backend.model_worker_entry, state_id): state_id
            for state_id in range(STATE_COUNT)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed % 50 == 0 or completed == STATE_COUNT:
                _checkpoint(
                    phase="candidate_model_progress",
                    completed_model_states=completed,
                    completed_model_state_ids=sorted(int(item["State_n_R"]) for item in results),
                )
                print(f"[K11 BACKWARD MODEL] {completed}/{STATE_COUNT}", flush=True)
    results.sort(key=lambda item: int(item["State_n_R"]))
    if [int(item["State_n_R"]) for item in results] != list(range(STATE_COUNT)):
        raise RuntimeError("backward model state order is not 0...424")
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    theta_a = np.zeros((len(candidates), STATE_COUNT, backend.K11), dtype=np.complex128)
    theta_c = np.zeros_like(theta_a)
    rows: list[dict[str, Any]] = []
    for item in results:
        state_id = int(item["State_n_R"])
        for fit in item["candidates"]:
            candidate = candidate_by_id[int(fit["candidate_id"])]
            theta_a[candidate.candidate_id, state_id] = fit["theta_Aend_padded"]
            theta_c[candidate.candidate_id, state_id] = fit["theta_C2_padded"]
            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "candidate_name": candidate.candidate_name,
                    "K": candidate.K,
                    "removed_basis_id": candidate.removed_basis_id or "NONE",
                    "removed_basis_p": candidate.removed_order_p,
                    "removed_basis_m": candidate.removed_delay_m,
                    "removed_basis_q": candidate.removed_envelope_delay_q,
                    "removed_family": candidate.removed_family or "BASELINE",
                    "removed_is_cross_delay": bool(candidate.removed_is_cross_delay) if candidate.removed_is_cross_delay is not None else False,
                    "State_ID": state_id,
                    "State_R": state_id,
                    "ilc_A_end": int(item["ilc_A_end"]),
                    "funMng": int(item["funMng"]),
                    "funAng": int(item["funAng"]),
                    "secMng": int(item["secMng"]),
                    "secAng": int(item["secAng"]),
                    "nmse_withoutdpd_dB": float(item["nmse_withoutdpd_dB"]),
                    "ACPR_withoutdpd_mean_dBc": float(item["ACPR_withoutdpd_mean_dBc"]),
                    "Y_Aend_train_NMSE_dB": float(fit["Y_Aend_train_NMSE_dB"]),
                    "Y_Aend_B_NMSE_dB": float(fit["Y_Aend_B_NMSE_dB"]),
                    "Y_C2_train_NMSE_dB": float(fit["Y_C2_train_NMSE_dB"]),
                    "Y_C2_B_NMSE_dB": float(fit["Y_C2_B_NMSE_dB"]),
                    "Aend_rank": int(fit["Aend_rank"]),
                    "Aend_rank_augmented": int(fit["Aend_rank_augmented"]),
                    "C2_rank": int(fit["C2_rank"]),
                    "C2_rank_augmented": int(fit["C2_rank_augmented"]),
                    "Aend_finite": bool(fit["finite"]),
                    "C2_finite": bool(fit["finite"]),
                }
            )
    frame = pd.DataFrame(rows).sort_values(["candidate_id", "State_R"]).reset_index(drop=True)
    if frame.shape[0] != len(candidates) * STATE_COUNT:
        raise RuntimeError(f"backward model quality long shape failed: {frame.shape}")
    frame.to_csv(model_path, index=False)
    mask = np.zeros((len(candidates), backend.K11), dtype=bool)
    for candidate in candidates:
        mask[candidate.candidate_id, list(candidate.support)] = True
    support_indices = np.full((len(candidates), backend.K11), -1, dtype=np.int64)
    for candidate in candidates:
        support_indices[candidate.candidate_id, : candidate.K] = candidate.support
    np.savez_compressed(
        coefficient_path,
        candidate_ids=np.asarray([candidate.candidate_id for candidate in candidates], dtype=np.int64),
        candidate_names=np.asarray([candidate.candidate_name for candidate in candidates]),
        removed_basis_ids=np.asarray([candidate.removed_basis_id or "NONE" for candidate in candidates]),
        theta_Aend_padded=theta_a,
        theta_C2_padded=theta_c,
        theta_support_mask=mask,
        support_indices=support_indices,
        K=np.asarray([candidate.K for candidate in candidates], dtype=np.int64),
        dmax=np.asarray(backend.MP_DMAX),
        ridge_lambda=np.asarray(backend.RIDGE_LAMBDA),
    )
    return frame, theta_a, theta_c


def _max_abs_with_neginf(left: np.ndarray, right: np.ndarray) -> tuple[float, int]:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        return float("inf"), max(left.size, right.size)
    if np.iscomplexobj(left) or np.iscomplexobj(right):
        left_finite = np.isfinite(left)
        right_finite = np.isfinite(right)
        mismatch = int(np.count_nonzero(left_finite != right_finite))
        finite = left_finite & right_finite
        if not np.any(finite):
            return 0.0, mismatch
        return float(np.max(np.abs(left[finite] - right[finite]))), mismatch
    left_inf = np.isneginf(left)
    right_inf = np.isneginf(right)
    mismatch = int(np.count_nonzero(left_inf != right_inf))
    finite = np.isfinite(left) & np.isfinite(right)
    if not np.any(finite):
        return 0.0, mismatch
    return float(np.max(np.abs(left[finite] - right[finite]))), mismatch


def _baseline_regression(
    current_model: pd.DataFrame,
    current_frame: pd.DataFrame,
    theta_a: np.ndarray,
    theta_c: np.ndarray,
    current_distance: np.ndarray,
    current_lut: np.ndarray,
    current_query_fp: np.ndarray,
    previous: dict[str, Any],
) -> dict[str, Any]:
    reference_model = previous["k11_model"].loc[previous["k11_model"]["model_id"].eq(21)].sort_values("State_R").reset_index(drop=True)
    current_model = current_model.loc[current_model["candidate_id"].eq(0)].sort_values("State_R").reset_index(drop=True)
    metric_columns = ["Y_Aend_train_NMSE_dB", "Y_Aend_B_NMSE_dB", "Y_C2_train_NMSE_dB", "Y_C2_B_NMSE_dB"]
    metric_errors = {column: float(np.max(np.abs(current_model[column].to_numpy(dtype=float) - reference_model[column].to_numpy(dtype=float)))) for column in metric_columns}
    reference_theta_index = int(np.flatnonzero(previous["k11_candidate_ids"] == 21)[0])
    theta_a_error = float(np.max(np.abs(theta_a[0] - previous["k11_theta_a"][reference_theta_index])))
    theta_c_error = float(np.max(np.abs(theta_c[0] - previous["k11_theta_c"][reference_theta_index])))
    lut_error = float(np.max(np.abs(current_lut - previous["k11_lut"])))
    query_error = float(np.max(np.abs(current_query_fp - previous["k11_query_fp"])))
    distance_error = float(np.max(np.abs(current_distance - previous["k11_distance"])))
    reference_query = previous["k11_query"].loc[previous["k11_query"]["candidate_id"].eq(21)].sort_values("State_R").reset_index(drop=True)
    selected_equal = bool(np.array_equal(current_frame["State_Q"].to_numpy(dtype=int), reference_query["State_Q"].to_numpy(dtype=int)))
    real_b_error, real_b_mismatch = _max_abs_with_neginf(
        current_frame["retrieved_realB_CNMSE_dB"].to_numpy(dtype=float),
        reference_query["retrieved_realB_CNMSE_dB"].to_numpy(dtype=float),
    )
    result = {
        "pass": False,
        "model_metric_max_abs_error_dB": metric_errors,
        "theta_Aend_max_abs_error": theta_a_error,
        "theta_C2_max_abs_error": theta_c_error,
        "LUT_fingerprint_max_abs_error": lut_error,
        "Query_fingerprint_max_abs_error": query_error,
        "distance_max_abs_error_dB": distance_error,
        "retrieved_Real_B_max_abs_error_dB": real_b_error,
        "retrieved_Real_B_nonfinite_pattern_mismatch": real_b_mismatch,
        "state_q_equal": selected_equal,
        "exact_count": int(current_frame["exact_hit"].sum()),
        "real_B_pass_count": int(current_frame["realB_pass"].sum()),
        "failure_count": int((~current_frame["realB_pass"]).sum()),
        "nonself_count": int((current_frame["State_Q"].to_numpy(dtype=int) != np.arange(STATE_COUNT)).sum()),
        "nonself_pass_count": int(((current_frame["State_Q"].to_numpy(dtype=int) != np.arange(STATE_COUNT)) & current_frame["realB_pass"].to_numpy(dtype=bool)).sum()),
    }
    result["pass"] = bool(
        max(metric_errors.values()) < 1e-10
        and theta_a_error < 1e-12
        and theta_c_error < 1e-12
        and lut_error < 1e-12
        and query_error < 1e-12
        and distance_error < 1e-12
        and real_b_error < 1e-9
        and real_b_mismatch == 0
        and selected_equal
        and result["exact_count"] == 218
        and result["real_B_pass_count"] == 416
        and result["failure_count"] == 9
        and result["nonself_count"] == 207
        and result["nonself_pass_count"] == 198
    )
    if not result["pass"]:
        raise RuntimeError(f"HARD FAIL: current K11 baseline regression failed: {result}")
    return result


def _deletion_to_mp10_regression(candidate: BackwardCandidate, frame: pd.DataFrame, model: pd.DataFrame, theta_a: np.ndarray, theta_c: np.ndarray, distance: np.ndarray, lut: np.ndarray, query_fp: np.ndarray, previous: dict[str, Any]) -> dict[str, Any]:
    current_model = model.loc[model["candidate_id"].eq(candidate.candidate_id)].sort_values("State_R").reset_index(drop=True)
    previous_model = previous["mp10"]["model"].sort_values("State_n_R").reset_index(drop=True)
    current_frame = frame.sort_values("State_R").reset_index(drop=True)
    previous_retrieval = previous["mp10"]["retrieval"].sort_values("State_n_R").reset_index(drop=True)
    metric_columns = ["Y_Aend_train_NMSE_dB", "Y_Aend_B_NMSE_dB", "Y_C2_train_NMSE_dB", "Y_C2_B_NMSE_dB"]
    metric_errors = {column: float(np.max(np.abs(current_model[column].to_numpy(dtype=float) - previous_model[column].to_numpy(dtype=float)))) for column in metric_columns}
    theta_a_error = float(np.max(np.abs(theta_a[candidate.candidate_id, :, :10] - previous["mp10"]["theta_aend"])))
    theta_c_error = float(np.max(np.abs(theta_c[candidate.candidate_id, :, :10] - previous["mp10"]["theta_c2"])))
    lut_error = float(np.max(np.abs(lut - previous["mp10"]["lut"])))
    query_error = float(np.max(np.abs(query_fp - previous["mp10"]["query"])))
    distance_error = float(np.max(np.abs(distance - previous["mp10"]["distance"])))
    state_q_equal = bool(np.array_equal(current_frame["State_Q"].to_numpy(dtype=int), previous_retrieval["State_n_Q"].to_numpy(dtype=int)))
    real_b_error, real_b_mismatch = _max_abs_with_neginf(
        current_frame["retrieved_realB_CNMSE_dB"].to_numpy(dtype=float),
        previous_retrieval["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float),
    )
    result = {
        "candidate": candidate.candidate_name,
        "removed_basis_id": candidate.removed_basis_id,
        "state_q_equal": state_q_equal,
        "metric_max_abs_error_dB": metric_errors,
        "theta_Aend_max_abs_error": theta_a_error,
        "theta_C2_max_abs_error": theta_c_error,
        "LUT_fingerprint_max_abs_error": lut_error,
        "Query_fingerprint_max_abs_error": query_error,
        "distance_max_abs_error_dB": distance_error,
        "retrieved_Real_B_max_abs_error_dB": real_b_error,
        "retrieved_Real_B_nonfinite_pattern_mismatch": real_b_mismatch,
    }
    result["pass"] = bool(
        max(metric_errors.values()) < 1e-10
        and theta_a_error < 1e-12
        and theta_c_error < 1e-12
        and lut_error < 1e-12
        and query_error < 1e-12
        and distance_error < 1e-12
        and real_b_error < 1e-9
        and real_b_mismatch == 0
        and state_q_equal
    )
    if not result["pass"]:
        raise RuntimeError(f"HARD FAIL: deleting added K11 basis did not reproduce MP10: {result}")
    return result


def _candidate_retrieval(candidate: BackwardCandidate, theta_a: np.ndarray, theta_c: np.ndarray, common_phi: np.ndarray, real_b_distance: np.ndarray, shareability: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    frame, summary, distance, lut, query_fp = k9_utils._candidate_retrieval(candidate, theta_a, theta_c, common_phi, real_b_distance, shareability)
    frame["candidate_id"] = candidate.candidate_id
    frame["candidate_name"] = candidate.candidate_name
    frame["removed_basis_id"] = candidate.removed_basis_id or "NONE"
    frame["removed_basis_p"] = candidate.removed_order_p
    frame["removed_basis_m"] = candidate.removed_delay_m
    frame["removed_basis_q"] = candidate.removed_envelope_delay_q
    frame["removed_is_cross_delay"] = bool(candidate.removed_is_cross_delay) if candidate.removed_is_cross_delay is not None else False
    frame["retrieved_real_B_CNMSE_dB"] = frame["retrieved_realB_CNMSE_dB"]
    for key, value in {
        "candidate_id": candidate.candidate_id,
        "candidate_name": candidate.candidate_name,
        "removed_basis_id": candidate.removed_basis_id or "NONE",
        "removed_basis_p": candidate.removed_order_p,
        "removed_basis_m": candidate.removed_delay_m,
        "removed_basis_q": candidate.removed_envelope_delay_q,
        "removed_is_cross_delay": bool(candidate.removed_is_cross_delay) if candidate.removed_is_cross_delay is not None else False,
    }.items():
        summary[key] = value
    summary["top1_pass_count"] = summary["top1_realB_pass_count"]
    summary["top1_pass_rate"] = summary["top1_realB_pass_rate"]
    return frame, summary, distance, lut, query_fp


def _local_cv_results(candidates: tuple[BackwardCandidate, ...], frames: dict[int, pd.DataFrame], folds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for fold in range(5):
        validation_ids = folds.loc[folds["fold"].eq(fold), "state_id"].to_numpy(dtype=int)
        train_ids = folds.loc[~folds["fold"].eq(fold), "state_id"].to_numpy(dtype=int)
        train_summaries = {
            candidate.candidate_id: k9_utils._summarize_query_frame(
                frames[candidate.candidate_id].loc[frames[candidate.candidate_id]["State_R"].isin(train_ids)], candidate
            )
            for candidate in candidates
        }
        ranking = sorted(train_summaries.values(), key=k9_utils._selection_key)
        winner_id = int(ranking[0]["candidate_id"])
        rank_by_id = {int(item["candidate_id"]): index for index, item in enumerate(ranking, start=1)}
        for candidate in candidates:
            train_summary = train_summaries[candidate.candidate_id]
            validation_frame = frames[candidate.candidate_id].loc[frames[candidate.candidate_id]["State_R"].isin(validation_ids)]
            validation_summary = k9_utils._summarize_query_frame(validation_frame, candidate)
            rows.append(
                {
                    "fold": fold,
                    "candidate_id": candidate.candidate_id,
                    "candidate_name": candidate.candidate_name,
                    "removed_basis_id": candidate.removed_basis_id or "NONE",
                    "removed_basis_p": candidate.removed_order_p,
                    "removed_basis_m": candidate.removed_delay_m,
                    "removed_basis_q": candidate.removed_envelope_delay_q,
                    "removed_is_cross_delay": bool(candidate.removed_is_cross_delay) if candidate.removed_is_cross_delay is not None else False,
                    "K": candidate.K,
                    "train_query_count": len(train_ids),
                    "validation_query_count": len(validation_ids),
                    "train_is_winner": candidate.candidate_id == winner_id,
                    "train_selection_rank": rank_by_id[candidate.candidate_id],
                    "train_top1_pass": train_summary["top1_realB_pass_count"],
                    "train_nonself_pass_rate": train_summary["nonself_pass_rate"],
                    "train_Q05_margin": train_summary["share_margin_Q05"],
                    "train_MRR": train_summary["MRR"],
                    "train_Top3_oracle": train_summary["Top3_oracle"],
                    "validation_exact": validation_summary["exact_hit_count"],
                    "validation_top1_pass": validation_summary["top1_realB_pass_count"],
                    "validation_top1_pass_rate": validation_summary["top1_realB_pass_rate"],
                    "validation_nonself_count": validation_summary["nonself_count"],
                    "validation_nonself_pass": validation_summary["nonself_pass_count"],
                    "validation_nonself_pass_rate": validation_summary["nonself_pass_rate"],
                    "validation_Q05_margin": validation_summary["share_margin_Q05"],
                    "validation_MRR": validation_summary["MRR"],
                    "validation_Top1_oracle": validation_summary["Top1_oracle"],
                    "validation_Top2_oracle": validation_summary["Top2_oracle"],
                    "validation_Top3_oracle": validation_summary["Top3_oracle"],
                    "validation_Top5_oracle": validation_summary["Top5_oracle"],
                    "validation_Top10_oracle": validation_summary["Top10_oracle"],
                }
            )
    cv = pd.DataFrame(rows).sort_values(["fold", "candidate_id"]).reset_index(drop=True)
    stability_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        winners = cv.loc[cv["candidate_id"].eq(candidate.candidate_id) & cv["train_is_winner"]]
        stability_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "candidate_name": candidate.candidate_name,
                "removed_basis_id": candidate.removed_basis_id or "NONE",
                "K": candidate.K,
                "train_winner_count": int(winners.shape[0]),
                "train_winner_frequency": float(winners.shape[0] / 5),
                "winner_folds": json.dumps(winners["fold"].astype(int).tolist()),
                "winner_validation_top1_pass_mean": float(winners["validation_top1_pass"].mean()) if not winners.empty else float("nan"),
                "winner_validation_top1_pass_min": float(winners["validation_top1_pass"].min()) if not winners.empty else float("nan"),
                "winner_validation_nonself_rate_mean": float(winners["validation_nonself_pass_rate"].mean()) if not winners.empty else float("nan"),
                "winner_validation_Q05_margin_mean": float(winners["validation_Q05_margin"].mean()) if not winners.empty else float("nan"),
                "winner_validation_MRR_mean": float(winners["validation_MRR"].mean()) if not winners.empty else float("nan"),
                "winner_validation_Top3_mean": float(winners["validation_Top3_oracle"].mean()) if not winners.empty else float("nan"),
            }
        )
    stability = pd.DataFrame(stability_rows).sort_values("candidate_id").reset_index(drop=True)
    stability["model_name"] = stability["candidate_name"]
    return cv, stability


def _build_deletion_outputs(candidates: tuple[BackwardCandidate, ...], summary: pd.DataFrame, model_frame: pd.DataFrame, frames: dict[int, pd.DataFrame], cv: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline = summary.loc[summary["candidate_id"].eq(0)].iloc[0]
    base_model = model_frame.loc[model_frame["candidate_id"].eq(0)]
    base_medians = {column: float(base_model[column].median()) for column in ("Y_Aend_train_NMSE_dB", "Y_Aend_B_NMSE_dB", "Y_C2_train_NMSE_dB", "Y_C2_B_NMSE_dB")}
    baseline_cv = cv.loc[cv["candidate_id"].eq(0)].sort_values("fold")
    delta_rows: list[dict[str, Any]] = []
    all_recovered: list[dict[str, Any]] = []
    recommendations: list[dict[str, Any]] = []
    for candidate in candidates[1:]:
        current = summary.loc[summary["candidate_id"].eq(candidate.candidate_id)].iloc[0]
        current_model = model_frame.loc[model_frame["candidate_id"].eq(candidate.candidate_id)]
        current_medians = {column: float(current_model[column].median()) for column in base_medians}
        current_frame = frames[candidate.candidate_id].set_index("State_R")
        base_frame = frames[0].set_index("State_R")
        base_pass = base_frame["realB_pass"].to_numpy(dtype=bool)
        current_pass = current_frame["realB_pass"].to_numpy(dtype=bool)
        recovered = int((~base_pass & current_pass).sum())
        regressed = int((base_pass & ~current_pass).sum())
        cv_current = cv.loc[cv["candidate_id"].eq(candidate.candidate_id)].sort_values("fold")
        validation_delta = cv_current["validation_top1_pass"].to_numpy(dtype=float) - baseline_cv["validation_top1_pass"].to_numpy(dtype=float)
        cv_negative = int((validation_delta < 0).sum())
        cv_stable = bool(cv_negative <= 1 and float(validation_delta.mean()) >= 0)
        cv_strict_nonworse = bool(np.all(validation_delta >= 0))
        secondary_nonworse = bool(
            current["nonself_pass_rate"] >= baseline["nonself_pass_rate"]
            and current["share_margin_Q05"] >= baseline["share_margin_Q05"]
            and current["MRR"] >= baseline["MRR"]
            and current["Top3_oracle"] >= baseline["Top3_oracle"]
        )
        delta_top1 = int(current["top1_realB_pass_count"] - baseline["top1_realB_pass_count"])
        if delta_top1 > 0:
            effect_class = "beneficial_to_delete"
        elif delta_top1 == 0 and recovered == 0 and regressed == 0 and secondary_nonworse and cv_strict_nonworse:
            effect_class = "potentially_redundant"
        elif delta_top1 == 0 and (recovered > 0 or regressed > 0):
            effect_class = "retrieval-equivalent-but-reordered"
        else:
            effect_class = "necessary"
        recommend = bool((delta_top1 > 0 and cv_stable) or (effect_class == "potentially_redundant"))
        reason = (
            "Top1 improves and CV has no systematic decline"
            if delta_top1 > 0 and cv_stable
            else "Top1 unchanged with zero recovered/regressed, non-worse secondary metrics and non-worse validation"
            if effect_class == "potentially_redundant"
            else "Top1 unchanged but state assignments exchanged"
            if effect_class == "retrieval-equivalent-but-reordered"
            else "deletion lowers Top1 or fails the safety conditions"
        )
        delta_rows.append(
            {
                "removed_basis_id": candidate.removed_basis_id,
                "removed_basis_p": candidate.removed_order_p,
                "removed_basis_m": candidate.removed_delay_m,
                "removed_basis_q": candidate.removed_envelope_delay_q,
                "removed_is_cross_delay": candidate.removed_is_cross_delay,
                "baseline_top1": int(baseline["top1_realB_pass_count"]),
                "candidate_top1": int(current["top1_realB_pass_count"]),
                "delta_top1": delta_top1,
                "recovered_count": recovered,
                "regressed_count": regressed,
                "net_change": recovered - regressed,
                "baseline_nonself_rate": float(baseline["nonself_pass_rate"]),
                "candidate_nonself_rate": float(current["nonself_pass_rate"]),
                "delta_nonself_rate": float(current["nonself_pass_rate"] - baseline["nonself_pass_rate"]),
                "baseline_Q05_margin": float(baseline["share_margin_Q05"]),
                "candidate_Q05_margin": float(current["share_margin_Q05"]),
                "delta_Q05_margin": float(current["share_margin_Q05"] - baseline["share_margin_Q05"]),
                "baseline_MRR": float(baseline["MRR"]),
                "candidate_MRR": float(current["MRR"]),
                "delta_MRR": float(current["MRR"] - baseline["MRR"]),
                "baseline_Top3": int(baseline["Top3_oracle"]),
                "candidate_Top3": int(current["Top3_oracle"]),
                "delta_Top3": int(current["Top3_oracle"] - baseline["Top3_oracle"]),
                "delta_Aend_train_median": current_medians["Y_Aend_train_NMSE_dB"] - base_medians["Y_Aend_train_NMSE_dB"],
                "delta_Aend_B_median": current_medians["Y_Aend_B_NMSE_dB"] - base_medians["Y_Aend_B_NMSE_dB"],
                "delta_C2_train_median": current_medians["Y_C2_train_NMSE_dB"] - base_medians["Y_C2_train_NMSE_dB"],
                "delta_C2_B_median": current_medians["Y_C2_B_NMSE_dB"] - base_medians["Y_C2_B_NMSE_dB"],
                "deletion_effect_class": effect_class,
                "cv_stable": cv_stable,
            }
        )
        recommendations.append(
            {
                "basis_id": candidate.removed_basis_id,
                "all425_top1_after_delete": int(current["top1_realB_pass_count"]),
                "delta_top1": delta_top1,
                "recovered": recovered,
                "regressed": regressed,
                "secondary_metrics_nonworse": secondary_nonworse,
                "cv_stable": cv_stable,
                "cv_strict_nonworse": cv_strict_nonworse,
                "cv_validation_delta_mean": float(validation_delta.mean()),
                "cv_validation_delta_min": float(validation_delta.min()),
                "recommend_delete": recommend,
                "deletion_effect_class": effect_class,
                "recommendation_reason": reason,
            }
        )
        for state_id in range(STATE_COUNT):
            base_pass_state = bool(base_frame.loc[state_id, "realB_pass"])
            current_pass_state = bool(current_frame.loc[state_id, "realB_pass"])
            transition = "recovered" if (not base_pass_state and current_pass_state) else "regressed" if (base_pass_state and not current_pass_state) else "unchanged_pass" if base_pass_state else "unchanged_fail"
            all_recovered.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "candidate_name": candidate.candidate_name,
                    "removed_basis_id": candidate.removed_basis_id,
                    "State_R": state_id,
                    "baseline_State_Q": int(base_frame.loc[state_id, "State_Q"]),
                    "candidate_State_Q": int(current_frame.loc[state_id, "State_Q"]),
                    "baseline_real_B_CNMSE_dB": float(base_frame.loc[state_id, "retrieved_realB_CNMSE_dB"]),
                    "candidate_real_B_CNMSE_dB": float(current_frame.loc[state_id, "retrieved_realB_CNMSE_dB"]),
                    "baseline_first_shareable_rank": int(base_frame.loc[state_id, "first_shareable_rank"]),
                    "candidate_first_shareable_rank": int(current_frame.loc[state_id, "first_shareable_rank"]),
                    "baseline_shareability_margin_dB": float(base_frame.loc[state_id, "shareability_margin_dB"]),
                    "candidate_shareability_margin_dB": float(current_frame.loc[state_id, "shareability_margin_dB"]),
                    "baseline_pass": base_pass_state,
                    "candidate_pass": current_pass_state,
                    "transition": transition,
                }
            )
    return pd.DataFrame(delta_rows), pd.DataFrame(all_recovered), pd.DataFrame(recommendations)


def _failure9_detail(candidates: tuple[BackwardCandidate, ...], frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    baseline = frames[0]
    failure_ids = baseline.loc[~baseline["realB_pass"], "State_R"].astype(int).tolist()
    if len(failure_ids) != 9:
        raise RuntimeError(f"K11 baseline failure count changed: {len(failure_ids)}")
    rows: list[dict[str, Any]] = []
    base = baseline.set_index("State_R")
    for candidate in candidates[1:]:
        current = frames[candidate.candidate_id].set_index("State_R")
        for state_id in failure_ids:
            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "candidate_name": candidate.candidate_name,
                    "removed_basis_id": candidate.removed_basis_id,
                    "State_R": state_id,
                    "baseline_State_Q": int(base.loc[state_id, "State_Q"]),
                    "candidate_State_Q": int(current.loc[state_id, "State_Q"]),
                    "baseline_real_B_CNMSE_dB": float(base.loc[state_id, "retrieved_realB_CNMSE_dB"]),
                    "candidate_real_B_CNMSE_dB": float(current.loc[state_id, "retrieved_realB_CNMSE_dB"]),
                    "candidate_pass_after_delete": bool(current.loc[state_id, "realB_pass"]),
                    "baseline_first_shareable_rank": int(base.loc[state_id, "first_shareable_rank"]),
                    "candidate_first_shareable_rank": int(current.loc[state_id, "first_shareable_rank"]),
                    "baseline_shareability_margin_dB": float(base.loc[state_id, "shareability_margin_dB"]),
                    "candidate_shareability_margin_dB": float(current.loc[state_id, "shareability_margin_dB"]),
                }
            )
    return pd.DataFrame(rows)


def _write_figures(summary: pd.DataFrame, delta: pd.DataFrame) -> None:
    ordered = summary.sort_values("candidate_id")
    labels = ordered["candidate_name"].tolist()
    values = ordered["top1_pass_count"].to_numpy(dtype=float)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(15, 8), dpi=300)
    bars = ax.bar(x, values, color=plt.get_cmap("tab20")(np.linspace(0, 1, len(labels))))
    ax.axhline(416, color="black", linestyle="--", linewidth=1.0, label="K11 baseline = 416")
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.3, f"{int(value)}", ha="center", va="bottom", fontsize=7)
    ax.set_xticks(x, labels, rotation=65, ha="right", fontsize=7)
    ax.set_ylabel("Top-1 Real-B pass count")
    ax.set_xlabel("K11 support / one-basis deletion")
    ax.set_ylim(0, 425)
    ax.set_title("K11 backward deletion Top-1 Real-B comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "16_candidate_top1_comparison.png")
    plt.close(fig)

    ordered_delta = delta.sort_values("removed_basis_id")
    dvalues = ordered_delta["delta_top1"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    colors = ["tab:green" if value > 0 else "tab:red" if value < 0 else "tab:gray" for value in dvalues]
    bars = ax.bar(np.arange(len(ordered_delta)), dvalues, color=colors)
    ax.axhline(0, color="black", linewidth=0.9)
    for bar, value in zip(bars, dvalues, strict=True):
        y = value + (0.2 if value >= 0 else -0.35)
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{value:+.0f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
    ax.set_xticks(np.arange(len(ordered_delta)), ordered_delta["removed_basis_id"], rotation=65, ha="right", fontsize=8)
    ax.set_ylabel("Δ Top-1 Real-B pass count vs K11")
    ax.set_xlabel("Removed basis")
    ax.set_ylim(float(dvalues.min()) - 4, max(4, float(dvalues.max()) + 4))
    ax.set_title("Backward retrieval contribution of each K11 basis")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "17_backward_basis_contribution.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 8), dpi=300)
    ax.scatter(delta["delta_C2_train_median"], delta["delta_top1"], c=delta["removed_is_cross_delay"].astype(int), cmap="tab10", s=45, alpha=0.85)
    highlights = delta["delta_top1"].abs().sort_values(ascending=False).head(8).index
    for row in delta.loc[highlights].itertuples(index=False):
        ax.annotate(str(row.removed_basis_id), (row.delta_C2_train_median, row.delta_top1), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Deletion Δ C2 Train median NMSE (dB)")
    ax.set_ylabel("Deletion Δ Top-1 Real-B pass count")
    ax.set_title("K11 deletion: modeling contribution versus retrieval contribution")
    ax.grid(alpha=0.25)
    ax.text(0.99, 0.02, "Labels show the largest |ΔTop1| deletions; full basis IDs are in 09_backward_deletion_delta.csv.", transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "18_modeling_vs_retrieval_deletion_effect.png")
    plt.close(fig)


def _run(*, resume: bool = False) -> dict[str, Any]:
    if RESULT_ROOT.exists() and any(RESULT_ROOT.iterdir()) and not resume:
        raise RuntimeError(f"result directory is not empty; use --resume: {RESULT_ROOT}")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "distance_matrices").mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "fingerprints").mkdir(parents=True, exist_ok=True)
    raw_before = raw_manifest_gate()
    support_ids, envelope_by_id, added_index, formal_support_frame = _load_frozen_k11_support()
    if formal_support_frame.loc[formal_support_frame["model_name"].eq(BEST_K11_NAME), "K"].iloc[0] != 11:
        raise RuntimeError("formal best K11 K changed")
    envelope_gate = dictionary_gate()
    previous = _load_references()
    common_b, common_meta = verify_common_b_contract()
    common_b = np.asarray(common_b, dtype=np.complex128)
    if common_meta["sha256"] != EXPECTED_COMMON_B_SHA:
        raise RuntimeError("common-B hash changed")
    real_b_random_error = _random_real_b_check(previous["real_b_distance"])
    support_frame, candidates = _build_candidates(support_ids, envelope_by_id)
    previous["folds"].to_csv(RESULT_ROOT / "12_query_state_cv_folds.csv", index=False)
    _write_json(
        RESULT_ROOT / "01_reuse_audit.json",
        {
            "task_name": TASK_NAME,
            "formal_best_k11_source": str(K11_ROOT / "04_candidate_supports.csv"),
            "formal_best_k11_name": BEST_K11_NAME,
            "formal_best_k11_support_ids": list(support_ids),
            "envelope75_dictionary_gate": envelope_gate,
            "remaining_backward_candidate_count": 11,
            "historical_mp10_backend": "behavior_modeling.sparse_gmp.build_frozen_mp_basis + fit_ridge",
            "dmax": backend.MP_DMAX,
            "ridge_lambda": backend.RIDGE_LAMBDA,
            "separate_hnorm_found": False,
            "retrieval_helpers_reused": ["K9 metric definitions", "CNMSE", "Top1 allow-self", "Real-B matrix", "shareability mask", "Query-State folds"],
            "real_B_ground_truth_reused_from": str(K9_ROOT / "04_realB_ground_truth_cnmse_matrix.npy"),
            "real_B_random_pair_max_abs_error_dB": real_b_random_error,
        },
    )
    audit_lines = [
        f"Task: {TASK_NAME}",
        f"Frozen parent model: {BEST_K11_NAME}; support={list(support_ids)}; K=11.",
        "Parent support was read from the formal K11 candidate support output, not reconstructed from chat text.",
        f"Envelope75 dictionary gate: {json.dumps(envelope_gate, ensure_ascii=False, sort_keys=True)}",
        "Historical MP10 basis/Ridge, canonical ABC, common-B, Real-B matrix/mask, CNMSE, Top-1, margins, MRR, Top-k and Query-State CV definitions are reused.",
        "Each State worker builds the full K11 Phi once and evaluates the K11 baseline plus 11 one-column deletions sequentially.",
        "All deletion candidates retain dmax=2 and Ridge lambda=1e-8.",
        "No Hnorm stage exists in the current checkout and no Hnorm was added.",
        "Deletion of ENV_p04_m2_q0 is required to reproduce the previous MP10 Full-LUT artifact.",
        "No second deletion, K12, multi-basis pruning, swap, Top-k selection, ensemble, clustering, DPD replay or low-bandwidth task was run.",
    ]
    (RESULT_ROOT / "01_reuse_audit.txt").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    (RESULT_ROOT / "00_task_definition.txt").write_text(
        "\n".join(
            [
                f"Task: {TASK_NAME}",
                "Mode: retrieval-oriented backward elimination from the formal best K11.",
                f"Parent: {BEST_K11_NAME}; support={list(support_ids)}; K=11.",
                "Candidates: K11_full plus 11 K10 leave-one-basis-out candidates.",
                "All candidates retain dmax=2, Ridge lambda=1e-8, frozen ABC, common-B, Real-B and allow-self Top-1.",
                "Primary result: whether any single deletion improves or safely preserves 416/425.",
                "No second deletion or K12 is performed in this task.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _checkpoint(phase="reuse_audit_and_support_passed", raw_manifest_before=raw_before, candidate_count=12, real_B_random_pair_max_abs_error_dB=real_b_random_error)
    model_frame, theta_a, theta_c = _run_model_phase(candidates, added_index, resume)
    mp_common = backend.historical_mp10.build_frozen_mp_basis(common_b)
    env_common = build_envelope_bank(common_b, tuple(envelope_by_id.values()))
    # The dictionary values are insertion-ordered from the canonical builder; re-read to guarantee index alignment.
    envelope_terms = tuple(build_envelope_dictionary())
    env_common = build_envelope_bank(common_b, envelope_terms)
    common_phi = np.column_stack((mp_common, env_common[:, added_index]))
    if common_phi.shape != (backend.FINGERPRINT_LENGTH, backend.K11):
        raise RuntimeError(f"K11 common-B Phi shape failed: {common_phi.shape}")
    candidate_frames: dict[int, pd.DataFrame] = {}
    candidate_summaries: list[dict[str, Any]] = []
    distance_dir = RESULT_ROOT / "distance_matrices"
    for candidate in candidates:
        frame, summary, distance, lut, query_fp = _candidate_retrieval(candidate, theta_a, theta_c, common_phi, previous["real_b_distance"], previous["shareability"])
        candidate_frames[candidate.candidate_id] = frame
        candidate_summaries.append(summary)
        np.save(distance_dir / f"{candidate.candidate_name}.npy", distance)
        if candidate.candidate_id == 0:
            np.save(RESULT_ROOT / "fingerprints" / "K11_full_lut.npy", lut)
            np.save(RESULT_ROOT / "fingerprints" / "K11_full_query.npy", query_fp)
        if candidate.removed_basis_id == BEST_K11_BASIS_ID:
            deletion_mp10_regression = _deletion_to_mp10_regression(candidate, frame, model_frame, theta_a, theta_c, distance, lut, query_fp, previous)
    baseline_summary_frame = pd.DataFrame(candidate_summaries)
    baseline_summary_frame["model_id"] = baseline_summary_frame["candidate_id"]
    baseline_summary_frame["model_name"] = baseline_summary_frame["candidate_name"]
    baseline_frame = candidate_frames[0]
    baseline = _baseline_regression(
        model_frame,
        baseline_frame,
        theta_a,
        theta_c,
        np.load(distance_dir / "K11_full.npy"),
        np.load(RESULT_ROOT / "fingerprints" / "K11_full_lut.npy"),
        np.load(RESULT_ROOT / "fingerprints" / "K11_full_query.npy"),
        previous,
    )
    _write_json(
        RESULT_ROOT / "04_baseline_regression_check.json",
        {
            "k11_baseline_regression": baseline,
            "deletion_of_added_basis_to_mp10_regression": deletion_mp10_regression,
            "pass": bool(baseline["pass"] and deletion_mp10_regression["pass"]),
        },
    )
    _checkpoint(phase="k11_baseline_and_mp10_deletion_regression_passed", baseline_regression=baseline, deletion_to_mp10_regression=deletion_mp10_regression)
    query_frame = pd.concat(candidate_frames.values(), ignore_index=True).sort_values(["candidate_id", "State_R"]).reset_index(drop=True)
    query_frame.to_csv(RESULT_ROOT / "08_candidate_query_metrics_long.csv", index=False)
    summary = k9_utils._assign_diagnostic_ranks(candidate_summaries)
    summary["model_id"] = summary["candidate_id"].astype(int)
    summary["model_name"] = summary["candidate_name"]
    summary["removed_basis_id"] = summary["removed_basis_id"].fillna("NONE")
    summary["top1_pass_count"] = summary["top1_realB_pass_count"]
    summary["top1_pass_rate"] = summary["top1_realB_pass_rate"]
    summary["delta_top1"] = summary["top1_pass_count"] - int(summary.loc[summary["candidate_id"].eq(0), "top1_pass_count"].iloc[0])
    summary["all425_rank"] = summary["diagnostic_rank"]
    summary["recovered_count"] = 0
    summary["regressed_count"] = 0
    summary["deletion_effect_class"] = "baseline"
    cv_frame, stability = _local_cv_results(candidates, candidate_frames, previous["folds"])
    delta, recovered_frame, recommendations = _build_deletion_outputs(candidates, summary, model_frame, candidate_frames, cv_frame)
    for row in delta.itertuples(index=False):
        selector = summary["candidate_id"].eq(next(candidate.candidate_id for candidate in candidates if candidate.removed_basis_id == row.removed_basis_id))
        summary.loc[selector, "recovered_count"] = int(row.recovered_count)
        summary.loc[selector, "regressed_count"] = int(row.regressed_count)
        summary.loc[selector, "deletion_effect_class"] = row.deletion_effect_class
    summary.to_csv(RESULT_ROOT / "07_candidate_retrieval_summary.csv", index=False)
    delta.to_csv(RESULT_ROOT / "09_backward_deletion_delta.csv", index=False)
    recovered_frame.to_csv(RESULT_ROOT / "10_recovered_regressed_states.csv", index=False)
    failure9 = _failure9_detail(candidates, candidate_frames)
    failure9.to_csv(RESULT_ROOT / "11_failure9_detailed_analysis.csv", index=False)
    cv_frame.to_csv(RESULT_ROOT / "13_query_state_cv_results.csv", index=False)
    stability.to_csv(RESULT_ROOT / "14_query_state_cv_selection_stability.csv", index=False)
    recommendations.to_csv(RESULT_ROOT / "15_deletion_recommendation.csv", index=False)
    winner = summary.sort_values("all425_rank").iloc[0]
    winner_id = int(winner["candidate_id"])
    winner_candidate = candidates[winner_id]
    winner_frame = candidate_frames[winner_id]
    winner_frame.to_csv(RESULT_ROOT / "20_diagnostic_best_detail.csv", index=False)
    winner_support = winner_candidate.support
    winner_phi = common_phi[:, list(winner_support)]
    np.save(RESULT_ROOT / "fingerprints" / "diagnostic_best_deletion_lut.npy", (winner_phi @ theta_a[winner_id][:, list(winner_support)].T).T.astype(np.complex128))
    np.save(RESULT_ROOT / "fingerprints" / "diagnostic_best_deletion_query.npy", (winner_phi @ theta_c[winner_id][:, list(winner_support)].T).T.astype(np.complex128))
    _write_figures(summary, delta)
    raw_after = raw_manifest_gate()
    if raw_before != raw_after:
        raise RuntimeError("data/raw manifest changed")
    recommendation_row = recommendations.loc[recommendations["recommend_delete"]]
    summary_lines = [
        f"Task: {TASK_NAME}",
        f"Parent K11: {BEST_K11_NAME}; support={list(support_ids)}; K=11.",
        f"K11 baseline regression PASS={baseline['pass']}; Top1={baseline['real_B_pass_count']}/425; Exact={baseline['exact_count']}/425; Non-self={baseline['nonself_pass_count']}/{baseline['nonself_count']}; Q05={float(summary.loc[summary['candidate_id'].eq(0), 'share_margin_Q05'].iloc[0]):.9g}; MRR={float(summary.loc[summary['candidate_id'].eq(0), 'MRR'].iloc[0]):.9g}; Top3={int(summary.loc[summary['candidate_id'].eq(0), 'Top3_oracle'].iloc[0])}/425.",
        f"Deletion ENV_p04_m2_q0 -> MP10 regression PASS={deletion_mp10_regression['pass']}.",
        "",
        "All 11 deletion candidates:",
    ]
    for row in summary.sort_values("all425_rank").itertuples(index=False):
        summary_lines.append(
            f"rank={int(row.all425_rank)} {row.model_name}: removed={row.removed_basis_id}, K={int(row.K)}, Top1={int(row.top1_pass_count)}/425, delta={int(row.delta_top1)}, recovered={int(row.recovered_count)}, regressed={int(row.regressed_count)}, nonself={float(row.nonself_pass_rate):.9g}, Q05={float(row.share_margin_Q05):.9g}, MRR={float(row.MRR):.9g}, Top3={int(row.Top3_oracle)}, class={row.deletion_effect_class}"
        )
    summary_lines.extend(
        [
            "",
            f"All425 diagnostic ranking winner: {winner['model_name']}.",
            "This is a diagnostic result. No basis was physically removed or frozen as a new final model.",
            f"Deletion recommendations marked True: {recommendation_row['basis_id'].tolist()}.",
            "",
            "Five-fold Query-State CV:",
        ]
    )
    for row in stability.loc[stability["train_winner_count"].gt(0)].sort_values("train_winner_count", ascending=False).itertuples(index=False):
        summary_lines.append(
            f"{row.model_name}: train winner={int(row.train_winner_count)}/5, folds={row.winner_folds}, validation Top1 mean/min={float(row.winner_validation_top1_pass_mean):.6g}/{float(row.winner_validation_top1_pass_min):.6g}"
        )
    summary_lines.extend(
        [
            "",
            f"raw before={json.dumps(raw_before, ensure_ascii=False, sort_keys=True)}",
            f"raw after={json.dumps(raw_after, ensure_ascii=False, sort_keys=True)}",
            f"raw unchanged={raw_before == raw_after}",
            "No second deletion, K12, multi-basis pruning, swap, lambda/order/memory scan, Top-k selection, ensemble, clustering, DPD replay, or low-bandwidth task was run.",
            "CV is a Query-State stability analysis, not an independent final test.",
        ]
    )
    (RESULT_ROOT / "19_final_result_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    transition_counts = recovered_frame["transition"].value_counts().to_dict()
    result = {
        "task": TASK_NAME,
        "status": "SUCCESS",
        "baseline_regression_pass": bool(baseline["pass"]),
        "deletion_to_mp10_regression_pass": bool(deletion_mp10_regression["pass"]),
        "diagnostic_winner": str(winner["model_name"]),
        "diagnostic_winner_class": str(winner["deletion_effect_class"]),
        "diagnostic_winner_top1": int(winner["top1_pass_count"]),
        "max_deletion_top1": int(summary.loc[summary["candidate_id"].ne(0), "top1_pass_count"].max()),
        "recommend_delete_basis_ids": recommendation_row["basis_id"].tolist(),
        "transition_counts_across_deletions": {key: int(transition_counts.get(key, 0)) for key in ("recovered", "regressed", "unchanged_pass", "unchanged_fail")},
        "cv_winner_counts": stability.loc[stability["train_winner_count"].gt(0), ["model_name", "train_winner_count"]].to_dict("records"),
        "raw_data_modified": raw_before != raw_after,
        "result_root": str(RESULT_ROOT),
    }
    _write_json(RESULT_ROOT / "20_checkpoint.json", {"phase": "completed", **result})
    _append_log(
        f"\n[{_now()}] Complete {TASK_NAME}\nResult={json.dumps(result, ensure_ascii=False, sort_keys=True)}\n"
        "No second deletion or K12 was run.\n"
    )
    _append_handoff(
        f"完成当前最佳 K11 的 11 个单项 backward deletion candidates；结果目录：{RESULT_ROOT}\n"
        f"摘要：{json.dumps(result, ensure_ascii=False, sort_keys=True)}\n"
        "完成 K11/MP10 regression、Full425 retrieval、failure9、deletion recommendation、5-fold CV 和三张诊断图；未冻结新模型。"
    )
    return result


def main() -> None:
    result = _run(resume="--resume" in sys.argv[1:])
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    __import__("multiprocessing").freeze_support()
    main()
