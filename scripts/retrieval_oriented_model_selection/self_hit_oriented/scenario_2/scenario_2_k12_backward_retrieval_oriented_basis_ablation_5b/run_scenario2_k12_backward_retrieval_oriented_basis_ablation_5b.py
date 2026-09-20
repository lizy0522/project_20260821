# ruff: noqa: E402,E501,I001

"""Run one-step retrieval-oriented backward elimination from the best K12."""

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
from retrieval_oriented_model_selection.self_hit_oriented.scenario_2.scenario_2_k12_backward_retrieval_oriented_basis_ablation_5b import (  # noqa: E402
    k12_backward_backend as backend,
)
from retrieval_oriented_model_selection.shared import k11_backward_retrieval_runner as prior_backward

TASK_NAME = "scenario_2_k12_backward_retrieval_oriented_basis_ablation_5b"
RESULT_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /TASK_NAME
WORK_LOG = PROJECT_ROOT / "work_logs" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /TASK_NAME / "execution_log.txt"
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"
K12_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /"scenario_2_k12_retrieval_oriented_forward_scan_5b"
K11_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /"scenario_2_mp10_k11_retrieval_oriented_forward_scan_5b"
K9_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /"scenario_2_mp10_k9_retrieval_oriented_basis_ablation_5b"
BEST_K12_NAME = "K11_plus_ENV_p04_m0_q1"
BEST_K12_BASIS_ID = "ENV_p04_m0_q1"
K11_BASIS_ID = "ENV_p04_m2_q0"
K11_REFERENCE_NAME = "MP10_plus_ENV_p04_m2_q0"
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
        RESULT_ROOT / "23_checkpoint.json",
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
            "k12_baseline": BEST_K12_NAME,
            "candidate_count": 13,
            "deletion_candidate_count": 12,
            "dmax_fixed": backend.MP_DMAX,
            "ridge_lambda_fixed": backend.RIDGE_LAMBDA,
            "second_deletion_performed": False,
            "k13_performed": False,
            "multi_basis_pruning_performed": False,
            "swap_performed": False,
            "lambda_scan_performed": False,
            "order_scan_performed": False,
            "memory_scan_performed": False,
            "dmax_scan_performed": False,
            "top_k_selection_performed": False,
            "real_B_used_for_top1_selection": False,
            **payload,
        },
    )


def _support_hash(ids: tuple[str, ...]) -> str:
    return hashlib.sha256(json.dumps(list(ids), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _load_frozen_k12() -> tuple[tuple[str, ...], dict[str, EnvelopeBasis], int, int, pd.DataFrame]:
    path = K12_ROOT / "06_candidate_supports.csv"
    if not path.is_file():
        raise FileNotFoundError(f"formal K12 support is missing: {path}")
    frame = pd.read_csv(path)
    row = frame.loc[frame["model_name"].eq(BEST_K12_NAME)]
    if row.shape[0] != 1 or int(row.iloc[0]["K"]) != 12:
        raise RuntimeError("formal best K12 support is missing or invalid")
    support_ids = tuple(json.loads(row.iloc[0]["support_basis_ids"]))
    if len(support_ids) != 12 or support_ids[-2:] != (K11_BASIS_ID, BEST_K12_BASIS_ID):
        raise RuntimeError(f"frozen K12 support changed: {support_ids}")
    terms = tuple(build_envelope_dictionary())
    by_id = {term.basis_id: term for term in terms}
    if len(terms) != 75 or any(basis_id not in by_id for basis_id in support_ids):
        raise RuntimeError("Envelope75/K12 support relationship changed")
    return support_ids, by_id, int(by_id[K11_BASIS_ID].index), int(by_id[BEST_K12_BASIS_ID].index), frame


def _build_candidates(support_ids: tuple[str, ...], by_id: dict[str, EnvelopeBasis]) -> tuple[pd.DataFrame, tuple[BackwardCandidate, ...]]:
    candidates = [
        BackwardCandidate(0, "K12_full", 12, None, None, None, None, None, None, None, tuple(range(12)), support_ids, _support_hash(support_ids))
    ]
    rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = [
        {
            "candidate_id": 0,
            "candidate_name": "K12_full",
            "K": 12,
            "removed_basis_id": "NONE",
            "removed_basis_p": np.nan,
            "removed_basis_m": np.nan,
            "removed_basis_q": np.nan,
            "removed_basis_type": "BASELINE",
            "removed_is_cross_delay": False,
            "retained_basis_ids": json.dumps(list(support_ids), ensure_ascii=False),
            "support_indices": json.dumps(list(range(12))),
            "support_hash": _support_hash(support_ids),
            "retained_support_hash": _support_hash(support_ids),
            "dmax": backend.MP_DMAX,
            "ridge_lambda": backend.RIDGE_LAMBDA,
        }
    ]
    for index, basis_id in enumerate(support_ids):
        term = by_id[basis_id]
        support = tuple(item for item in range(12) if item != index)
        retained = tuple(support_ids[item] for item in support)
        is_cross = bool(term.order > 1 and term.signal_delay != term.envelope_delay)
        candidate = BackwardCandidate(index + 1, f"K12_minus_{basis_id}", 11, basis_id, int(term.order), term.signal_delay, term.envelope_delay, term.family, is_cross, index, support, retained, _support_hash(retained))
        candidates.append(candidate)
        rows.append(
            {
                "candidate_index": index + 1,
                "candidate_id": index + 1,
                "candidate_name": candidate.candidate_name,
                "removed_basis_id": basis_id,
                "removed_basis_type": term.family,
                "removed_order_p": int(term.order),
                "removed_delay_m": term.signal_delay,
                "removed_envelope_delay_q": term.envelope_delay,
                "removed_is_cross_delay": is_cross,
            }
        )
        support_rows.append(
            {
                "candidate_id": index + 1,
                "candidate_name": candidate.candidate_name,
                "K": 11,
                "removed_basis_id": basis_id,
                "removed_basis_p": int(term.order),
                "removed_basis_m": term.signal_delay,
                "removed_basis_q": term.envelope_delay,
                "removed_basis_type": term.family,
                "removed_is_cross_delay": is_cross,
                "retained_basis_ids": json.dumps(list(retained), ensure_ascii=False),
                "support_indices": json.dumps(list(support)),
                "support_hash": candidate.support_hash,
                "retained_support_hash": candidate.support_hash,
                "dmax": backend.MP_DMAX,
                "ridge_lambda": backend.RIDGE_LAMBDA,
            }
        )
    pd.DataFrame(support_rows).to_csv(RESULT_ROOT / "03_deletion_candidate_supports.csv", index=False)
    pd.DataFrame(
        [
            {
                "baseline_column_index": index,
                "basis_id": basis_id,
                "order_p": int(by_id[basis_id].order),
                "delay_m": by_id[basis_id].signal_delay,
                "envelope_delay_q": by_id[basis_id].envelope_delay,
                "basis_family": by_id[basis_id].family,
            }
            for index, basis_id in enumerate(support_ids)
        ]
    ).to_csv(RESULT_ROOT / "02_k12_baseline_support.csv", index=False)
    return pd.DataFrame(support_rows), tuple(candidates)


def _load_references() -> dict[str, Any]:
    previous_mp10 = prior_backward._load_references()["mp10"]
    k12_model = pd.read_csv(K12_ROOT / "08_candidate_model_quality_long.csv")
    k12_query = pd.read_csv(K12_ROOT / "11_candidate_query_metrics_long.csv")
    with np.load(K12_ROOT / "09_candidate_coefficients.npz", allow_pickle=False) as data:
        k12_ids = np.asarray(data["model_ids"], dtype=np.int64)
        k12_theta_a = np.asarray(data["theta_Aend_padded"])
        k12_theta_c = np.asarray(data["theta_C2_padded"])
    k11_model = pd.read_csv(K11_ROOT / "06_candidate_model_quality_long.csv")
    k11_query = pd.read_csv(K11_ROOT / "09_candidate_query_metrics_long.csv")
    k11_reference_rows = k11_model.loc[k11_model["model_name"].eq(K11_REFERENCE_NAME), "model_id"].drop_duplicates()
    if k11_reference_rows.shape[0] != 1:
        raise RuntimeError(f"formal K11 reference is missing or ambiguous: {K11_REFERENCE_NAME}")
    k11_reference_id = int(k11_reference_rows.iloc[0])
    with np.load(K11_ROOT / "07_candidate_coefficients.npz", allow_pickle=False) as data:
        k11_ids = np.asarray(data["model_ids"], dtype=np.int64)
        k11_theta_a = np.asarray(data["theta_Aend_padded"])
        k11_theta_c = np.asarray(data["theta_C2_padded"])
    real_b_distance = np.load(K9_ROOT / "04_realB_ground_truth_cnmse_matrix.npy")
    shareability = np.load(K9_ROOT / "05_realB_shareability_mask.npy")
    folds = pd.read_csv(K12_ROOT / "16_query_state_cv_folds.csv").sort_values("state_id").reset_index(drop=True)
    k11_folds = pd.read_csv(K11_ROOT / "13_query_state_cv_folds.csv").sort_values("state_id").reset_index(drop=True)
    k9_folds = pd.read_csv(K9_ROOT / "12_query_state_cv_folds.csv").sort_values("state_id").reset_index(drop=True)
    if not np.array_equal(folds[["state_id", "fold"]].to_numpy(), k11_folds[["state_id", "fold"]].to_numpy()) or not np.array_equal(folds[["state_id", "fold"]].to_numpy(), k9_folds[["state_id", "fold"]].to_numpy()):
        raise RuntimeError("K12/K11/K9 Query-State folds differ")
    baseline_query = k12_query.loc[k12_query["candidate_id"].eq(16)].sort_values("State_R").reset_index(drop=True)
    expected_failures = [189, 195, 323, 335, 340, 346]
    observed_failures = baseline_query.loc[~baseline_query["realB_pass"], "State_R"].astype(int).tolist()
    if baseline_query.shape[0] != STATE_COUNT or observed_failures != expected_failures:
        raise RuntimeError(f"K12 baseline failure contract changed: {observed_failures}")
    if int(baseline_query["realB_pass"].sum()) != 419 or int(baseline_query["exact_hit"].sum()) != 235:
        raise RuntimeError("K12 baseline counts changed")
    if real_b_distance.shape != (STATE_COUNT, STATE_COUNT) or shareability.shape != real_b_distance.shape or not np.all(np.isneginf(np.diag(real_b_distance))) or not shareability.diagonal().all():
        raise RuntimeError("Real-B ground truth contract changed")
    return {
        "mp10": previous_mp10,
        "k12_model": k12_model,
        "k12_query": k12_query,
        "k12_ids": k12_ids,
        "k12_theta_a": k12_theta_a,
        "k12_theta_c": k12_theta_c,
        "k12_distance": np.load(K12_ROOT / "distance_matrices" / f"{BEST_K12_NAME}.npy"),
        "k12_lut": np.load(K12_ROOT / "fingerprints" / "diagnostic_best_k12_lut.npy"),
        "k12_query_fp": np.load(K12_ROOT / "fingerprints" / "diagnostic_best_k12_query.npy"),
        "k12_baseline_query": baseline_query,
        "k11_model": k11_model,
        "k11_query": k11_query,
        "k11_ids": k11_ids,
        "k11_theta_a": k11_theta_a,
        "k11_theta_c": k11_theta_c,
        "k11_reference_id": k11_reference_id,
        "k11_distance": np.load(K11_ROOT / "distance_matrices" / f"{K11_REFERENCE_NAME}.npy"),
        "k11_lut": np.load(K11_ROOT / "fingerprints" / "diagnostic_best_k11_lut.npy"),
        "k11_query_fp": np.load(K11_ROOT / "fingerprints" / "diagnostic_best_k11_query.npy"),
        "real_b_distance": real_b_distance,
        "shareability": shareability,
        "folds": folds,
    }


def _random_real_b_check(matrix: np.ndarray) -> float:
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
        stored = float(matrix[int(real_id), int(query_id)])
        if np.isneginf(direct) and np.isneginf(stored):
            continue
        maximum = max(maximum, abs(direct - stored))
    if maximum > REAL_B_SANITY_TOLERANCE_DB:
        raise RuntimeError(f"Real-B random-pair check failed: {maximum:.3e} dB")
    return maximum


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
        return (float(np.max(np.abs(left[finite] - right[finite]))) if np.any(finite) else 0.0, mismatch)
    left_inf = np.isneginf(left)
    right_inf = np.isneginf(right)
    mismatch = int(np.count_nonzero(left_inf != right_inf))
    finite = np.isfinite(left) & np.isfinite(right)
    return (float(np.max(np.abs(left[finite] - right[finite]))) if np.any(finite) else 0.0, mismatch)


def _failure_rank_profile(
    distance: np.ndarray,
    shareability: np.ndarray,
    real_b_distance: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive the frozen parent's rank profile without writing into another task."""
    rows: list[dict[str, Any]] = []
    hard_rows: list[dict[str, Any]] = []
    state_ids = np.arange(STATE_COUNT, dtype=np.int64)
    for state_id in range(STATE_COUNT):
        order = np.lexsort((state_ids, distance[state_id]))
        top1 = int(order[0])
        top2 = int(order[1])
        top3 = int(order[2])
        share_positions = np.flatnonzero(shareability[state_id, order])
        if share_positions.size == 0:
            raise RuntimeError(f"state {state_id} has no shareable candidate in parent profile")
        first_position = int(share_positions[0])
        first_state = int(order[first_position])
        top1_shareable = bool(shareability[state_id, top1])
        rows.append(
            {
                "State_R": state_id,
                "Top1_State_Q": top1,
                "Top1_shareable": top1_shareable,
                "Top2_State_Q": top2,
                "Top2_shareable": bool(shareability[state_id, top2]),
                "Top3_State_Q": top3,
                "Top3_shareable": bool(shareability[state_id, top3]),
                "first_shareable_rank": first_position + 1,
                "first_shareable_State_Q": first_state,
                "Top1_distance": float(distance[state_id, top1]),
                "Top2_distance": float(distance[state_id, top2]),
                "first_shareable_distance": float(distance[state_id, first_state]),
                "ranking_gap": float(distance[state_id, top2] - distance[state_id, top1]),
                "Top1_realB_CNMSE_dB": float(real_b_distance[state_id, top1]),
                "Top2_realB_CNMSE_dB": float(real_b_distance[state_id, top2]),
            }
        )
        if not top1_shareable and bool(shareability[state_id, top2]):
            hard_rows.append(
                {
                    "State_R": state_id,
                    "Top1_wrong_State_Q": top1,
                    "Top2_shareable_State_Q": top2,
                    "Top1_wrong_fingerprint_distance": float(distance[state_id, top1]),
                    "Top2_shareable_fingerprint_distance": float(distance[state_id, top2]),
                    "hard_pair_margin_before": float(distance[state_id, top1] - distance[state_id, top2]),
                    "Top1_wrong_realB_CNMSE_dB": float(real_b_distance[state_id, top1]),
                    "Top2_shareable_realB_CNMSE_dB": float(real_b_distance[state_id, top2]),
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(hard_rows)


def _run_model_phase(candidates: tuple[BackwardCandidate, ...], k11_index: int, k12_index: int, resume: bool) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    model_path = RESULT_ROOT / "06_candidate_model_quality_long.csv"
    coefficient_path = RESULT_ROOT / "07_candidate_coefficients.npz"
    if resume and model_path.is_file() and coefficient_path.is_file():
        try:
            frame = pd.read_csv(model_path)
            with np.load(coefficient_path, allow_pickle=False) as data:
                theta_a = np.asarray(data["theta_Aend_padded"])
                theta_c = np.asarray(data["theta_C2_padded"])
            if theta_a.shape == (len(candidates), STATE_COUNT, backend.K12) and theta_c.shape == theta_a.shape:
                return frame, theta_a, theta_c
        except (OSError, ValueError, KeyError):
            pass
    specs = tuple((candidate.candidate_id, candidate.removed_index) for candidate in candidates)
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=WORKER_COUNT, mp_context=get_context("spawn"), initializer=backend.worker_init, initargs=(specs, k11_index, k12_index)) as executor:
        futures = {executor.submit(backend.model_worker_entry, state_id): state_id for state_id in range(STATE_COUNT)}
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed % 50 == 0 or completed == STATE_COUNT:
                _checkpoint(phase="candidate_model_progress", completed_model_states=completed, completed_model_state_ids=sorted(int(item["State_n_R"]) for item in results))
                print(f"[K12 BACKWARD MODEL] {completed}/{STATE_COUNT}", flush=True)
    results.sort(key=lambda item: int(item["State_n_R"]))
    if [int(item["State_n_R"]) for item in results] != list(range(STATE_COUNT)):
        raise RuntimeError("K12 backward model state order is not 0...424")
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    theta_a = np.zeros((len(candidates), STATE_COUNT, backend.K12), dtype=np.complex128)
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
        raise RuntimeError(f"K12 backward model quality shape failed: {frame.shape}")
    frame.to_csv(model_path, index=False)
    support_indices = np.full((len(candidates), backend.K12), -1, dtype=np.int64)
    support_mask = np.zeros((len(candidates), backend.K12), dtype=bool)
    for candidate in candidates:
        support_indices[candidate.candidate_id, : candidate.K] = candidate.support
        support_mask[candidate.candidate_id, list(candidate.support)] = True
    np.savez_compressed(coefficient_path, candidate_ids=np.asarray([candidate.candidate_id for candidate in candidates], dtype=np.int64), candidate_names=np.asarray([candidate.candidate_name for candidate in candidates]), removed_basis_ids=np.asarray([candidate.removed_basis_id or "NONE" for candidate in candidates]), theta_Aend_padded=theta_a, theta_C2_padded=theta_c, theta_support_mask=support_mask, support_indices=support_indices, K=np.asarray([candidate.K for candidate in candidates], dtype=np.int64), dmax=np.asarray(backend.MP_DMAX), ridge_lambda=np.asarray(backend.RIDGE_LAMBDA))
    return frame, theta_a, theta_c


def _candidate_retrieval(candidate: BackwardCandidate, theta_a: np.ndarray, theta_c: np.ndarray, common_phi: np.ndarray, real_b_distance: np.ndarray, shareability: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    frame, summary, distance, lut, query = prior_backward._candidate_retrieval(candidate, theta_a, theta_c, common_phi, real_b_distance, shareability)
    frame["model_id"] = candidate.candidate_id
    frame["model_name"] = candidate.candidate_name
    frame["removed_basis_id"] = candidate.removed_basis_id or "NONE"
    frame["removed_basis_p"] = candidate.removed_order_p
    frame["removed_basis_m"] = candidate.removed_delay_m
    frame["removed_basis_q"] = candidate.removed_envelope_delay_q
    frame["removed_is_cross_delay"] = bool(candidate.removed_is_cross_delay) if candidate.removed_is_cross_delay is not None else False
    frame["retrieved_real_B_CNMSE_dB"] = frame["retrieved_realB_CNMSE_dB"]
    summary["model_id"] = candidate.candidate_id
    summary["model_name"] = candidate.candidate_name
    summary["removed_basis_id"] = candidate.removed_basis_id or "NONE"
    summary["removed_basis_p"] = candidate.removed_order_p
    summary["removed_basis_m"] = candidate.removed_delay_m
    summary["removed_basis_q"] = candidate.removed_envelope_delay_q
    summary["removed_is_cross_delay"] = bool(candidate.removed_is_cross_delay) if candidate.removed_is_cross_delay is not None else False
    summary["top1_pass_count"] = summary["top1_realB_pass_count"]
    summary["top1_pass_rate"] = summary["top1_realB_pass_rate"]
    values = frame["top12_margin_dB"].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    summary["top12_margin_Q05"] = float(np.quantile(finite, 0.05)) if finite.size else float("nan")
    summary["top12_margin_Q10"] = float(np.quantile(finite, 0.10)) if finite.size else float("nan")
    return frame, summary, distance, lut, query


def _baseline_regression(current_model: pd.DataFrame, current_frame: pd.DataFrame, theta_a: np.ndarray, theta_c: np.ndarray, distance: np.ndarray, lut: np.ndarray, query_fp: np.ndarray, references: dict[str, Any]) -> dict[str, Any]:
    reference_model = references["k12_model"].loc[references["k12_model"]["model_id"].eq(16)].sort_values("State_R").reset_index(drop=True)
    current_model = current_model.loc[current_model["candidate_id"].eq(0)].sort_values("State_R").reset_index(drop=True)
    metric_columns = ["Y_Aend_train_NMSE_dB", "Y_Aend_B_NMSE_dB", "Y_C2_train_NMSE_dB", "Y_C2_B_NMSE_dB"]
    metric_errors = {column: float(np.max(np.abs(current_model[column].to_numpy(dtype=float) - reference_model[column].to_numpy(dtype=float)))) for column in metric_columns}
    ref_index = int(np.flatnonzero(references["k12_ids"] == 16)[0])
    theta_a_error = float(np.max(np.abs(theta_a[0] - references["k12_theta_a"][ref_index])))
    theta_c_error = float(np.max(np.abs(theta_c[0] - references["k12_theta_c"][ref_index])))
    real_b_error, real_b_mismatch = _max_abs_with_neginf(current_frame["retrieved_realB_CNMSE_dB"].to_numpy(dtype=float), references["k12_baseline_query"]["retrieved_realB_CNMSE_dB"].to_numpy(dtype=float))
    distance_error, distance_mismatch = _max_abs_with_neginf(distance, references["k12_distance"])
    reference_query = references["k12_baseline_query"]
    result = {
        "pass": False,
        "model_metric_max_abs_error_dB": metric_errors,
        "theta_Aend_max_abs_error": theta_a_error,
        "theta_C2_max_abs_error": theta_c_error,
        "LUT_fingerprint_max_abs_error": float(np.max(np.abs(lut - references["k12_lut"]))),
        "Query_fingerprint_max_abs_error": float(np.max(np.abs(query_fp - references["k12_query_fp"]))),
        "distance_max_abs_error_dB": distance_error,
        "distance_nonfinite_pattern_mismatch": distance_mismatch,
        "retrieved_Real_B_max_abs_error_dB": real_b_error,
        "retrieved_Real_B_nonfinite_pattern_mismatch": real_b_mismatch,
        "fingerprint_top1_max_abs_error_dB": float(np.max(np.abs(current_frame["fingerprint_top1_CNMSE_dB"].to_numpy(dtype=float) - reference_query["fingerprint_top1_CNMSE_dB"].to_numpy(dtype=float)))),
        "fingerprint_top2_max_abs_error_dB": float(np.max(np.abs(current_frame["fingerprint_top2_CNMSE_dB"].to_numpy(dtype=float) - reference_query["fingerprint_top2_CNMSE_dB"].to_numpy(dtype=float)))),
        "shareability_margin_max_abs_error_dB": float(np.max(np.abs(current_frame["shareability_margin_dB"].to_numpy(dtype=float) - reference_query["shareability_margin_dB"].to_numpy(dtype=float)))),
        "first_shareable_rank_equal": bool(np.array_equal(current_frame["first_shareable_rank"].to_numpy(dtype=int), reference_query["first_shareable_rank"].to_numpy(dtype=int))),
        "state_q_equal": bool(np.array_equal(current_frame["State_Q"].to_numpy(dtype=int), reference_query["State_Q"].to_numpy(dtype=int))),
        "exact_count": int(current_frame["exact_hit"].sum()),
        "real_B_pass_count": int(current_frame["realB_pass"].sum()),
        "failure_count": int((~current_frame["realB_pass"]).sum()),
        "nonself_count": int((current_frame["State_Q"].to_numpy(dtype=int) != np.arange(STATE_COUNT)).sum()),
        "nonself_pass_count": int(((current_frame["State_Q"].to_numpy(dtype=int) != np.arange(STATE_COUNT)) & current_frame["realB_pass"].to_numpy(dtype=bool)).sum()),
    }
    result["pass"] = bool(max(metric_errors.values()) < 1e-10 and theta_a_error < 1e-12 and theta_c_error < 1e-12 and result["LUT_fingerprint_max_abs_error"] < 1e-12 and result["Query_fingerprint_max_abs_error"] < 1e-12 and result["distance_max_abs_error_dB"] < 1e-12 and distance_mismatch == 0 and real_b_error < 1e-9 and real_b_mismatch == 0 and result["fingerprint_top1_max_abs_error_dB"] < 1e-12 and result["fingerprint_top2_max_abs_error_dB"] < 1e-12 and result["shareability_margin_max_abs_error_dB"] < 1e-12 and result["first_shareable_rank_equal"] and result["state_q_equal"] and result["exact_count"] == 235 and result["real_B_pass_count"] == 419 and result["failure_count"] == 6 and result["nonself_count"] == 190 and result["nonself_pass_count"] == 184)
    if not result["pass"]:
        raise RuntimeError(f"HARD FAIL: K12 baseline regression failed: {result}")
    return result


def _deletion_to_k11_regression(candidate: BackwardCandidate, frame: pd.DataFrame, model: pd.DataFrame, theta_a: np.ndarray, theta_c: np.ndarray, distance: np.ndarray, lut: np.ndarray, query_fp: np.ndarray, references: dict[str, Any]) -> dict[str, Any]:
    current_model = model.loc[model["candidate_id"].eq(candidate.candidate_id)].sort_values("State_R").reset_index(drop=True)
    reference_model = references["k11_model"].loc[references["k11_model"]["model_id"].eq(references["k11_reference_id"])].sort_values("State_R").reset_index(drop=True)
    reference_query = references["k11_query"].loc[references["k11_query"]["candidate_id"].eq(references["k11_reference_id"])].sort_values("State_R").reset_index(drop=True)
    current_frame = frame.sort_values("State_R").reset_index(drop=True)
    metric_columns = ["Y_Aend_train_NMSE_dB", "Y_Aend_B_NMSE_dB", "Y_C2_train_NMSE_dB", "Y_C2_B_NMSE_dB"]
    metric_errors = {column: float(np.max(np.abs(current_model[column].to_numpy(dtype=float) - reference_model[column].to_numpy(dtype=float)))) for column in metric_columns}
    ref_index = int(np.flatnonzero(references["k11_ids"] == references["k11_reference_id"])[0])
    theta_a_error = float(np.max(np.abs(theta_a[candidate.candidate_id, :, :11] - references["k11_theta_a"][ref_index])))
    theta_c_error = float(np.max(np.abs(theta_c[candidate.candidate_id, :, :11] - references["k11_theta_c"][ref_index])))
    real_b_error, real_b_mismatch = _max_abs_with_neginf(current_frame["retrieved_realB_CNMSE_dB"].to_numpy(dtype=float), reference_query["retrieved_realB_CNMSE_dB"].to_numpy(dtype=float))
    distance_error, distance_mismatch = _max_abs_with_neginf(distance, references["k11_distance"])
    result = {
        "candidate": candidate.candidate_name,
        "removed_basis_id": candidate.removed_basis_id,
        "state_q_equal": bool(np.array_equal(current_frame["State_Q"].to_numpy(dtype=int), reference_query["State_Q"].to_numpy(dtype=int))),
        "metric_max_abs_error_dB": metric_errors,
        "theta_Aend_max_abs_error": theta_a_error,
        "theta_C2_max_abs_error": theta_c_error,
        "LUT_fingerprint_max_abs_error": float(np.max(np.abs(lut - references["k11_lut"]))),
        "Query_fingerprint_max_abs_error": float(np.max(np.abs(query_fp - references["k11_query_fp"]))),
        "distance_max_abs_error_dB": distance_error,
        "distance_nonfinite_pattern_mismatch": distance_mismatch,
        "retrieved_Real_B_max_abs_error_dB": real_b_error,
        "retrieved_Real_B_nonfinite_pattern_mismatch": real_b_mismatch,
    }
    result["pass"] = bool(max(metric_errors.values()) < 1e-10 and theta_a_error < 1e-12 and theta_c_error < 1e-12 and result["LUT_fingerprint_max_abs_error"] < 1e-12 and result["Query_fingerprint_max_abs_error"] < 1e-12 and result["distance_max_abs_error_dB"] < 1e-12 and distance_mismatch == 0 and real_b_error < 1e-9 and real_b_mismatch == 0 and result["state_q_equal"])
    if not result["pass"]:
        raise RuntimeError(f"HARD FAIL: deleting ENV_p04_m0_q1 did not reproduce K11: {result}")
    return result


def _local_cv_results(candidates: tuple[BackwardCandidate, ...], frames: dict[int, pd.DataFrame], folds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for fold in range(5):
        val_ids = folds.loc[folds["fold"].eq(fold), "state_id"].to_numpy(dtype=int)
        train_ids = folds.loc[~folds["fold"].eq(fold), "state_id"].to_numpy(dtype=int)
        train_summaries = {candidate.candidate_id: prior_backward.k9_utils._summarize_query_frame(frames[candidate.candidate_id].loc[frames[candidate.candidate_id]["State_R"].isin(train_ids)], candidate) for candidate in candidates}
        ranking = sorted(train_summaries.values(), key=prior_backward.k9_utils._selection_key)
        winner_id = int(ranking[0]["candidate_id"])
        rank_by_id = {int(item["candidate_id"]): index for index, item in enumerate(ranking, start=1)}
        for candidate in candidates:
            train_summary = train_summaries[candidate.candidate_id]
            val_frame = frames[candidate.candidate_id].loc[frames[candidate.candidate_id]["State_R"].isin(val_ids)]
            val_summary = prior_backward.k9_utils._summarize_query_frame(val_frame, candidate)
            rows.append({"fold": fold, "candidate_id": candidate.candidate_id, "candidate_name": candidate.candidate_name, "removed_basis_id": candidate.removed_basis_id or "NONE", "K": candidate.K, "train_query_count": len(train_ids), "validation_query_count": len(val_ids), "train_is_winner": candidate.candidate_id == winner_id, "train_selection_rank": rank_by_id[candidate.candidate_id], "train_top1_pass": train_summary["top1_realB_pass_count"], "train_nonself_pass_rate": train_summary["nonself_pass_rate"], "train_Q05_margin": train_summary["share_margin_Q05"], "train_MRR": train_summary["MRR"], "train_Top3_oracle": train_summary["Top3_oracle"], "validation_exact": val_summary["exact_hit_count"], "validation_top1_pass": val_summary["top1_realB_pass_count"], "validation_top1_pass_rate": val_summary["top1_realB_pass_rate"], "validation_nonself_count": val_summary["nonself_count"], "validation_nonself_pass": val_summary["nonself_pass_count"], "validation_nonself_pass_rate": val_summary["nonself_pass_rate"], "validation_Q05_margin": val_summary["share_margin_Q05"], "validation_MRR": val_summary["MRR"], "validation_Top1_oracle": val_summary["Top1_oracle"], "validation_Top2_oracle": val_summary["Top2_oracle"], "validation_Top3_oracle": val_summary["Top3_oracle"], "validation_Top5_oracle": val_summary["Top5_oracle"], "validation_Top10_oracle": val_summary["Top10_oracle"]})
    cv = pd.DataFrame(rows).sort_values(["fold", "candidate_id"]).reset_index(drop=True)
    stability_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        winners = cv.loc[cv["candidate_id"].eq(candidate.candidate_id) & cv["train_is_winner"]]
        stability_rows.append({"candidate_id": candidate.candidate_id, "candidate_name": candidate.candidate_name, "removed_basis_id": candidate.removed_basis_id or "NONE", "K": candidate.K, "train_winner_count": int(winners.shape[0]), "train_winner_frequency": float(winners.shape[0] / 5), "winner_folds": json.dumps(winners["fold"].astype(int).tolist()), "winner_validation_top1_pass_mean": float(winners["validation_top1_pass"].mean()) if not winners.empty else float("nan"), "winner_validation_top1_pass_min": float(winners["validation_top1_pass"].min()) if not winners.empty else float("nan"), "winner_validation_nonself_rate_mean": float(winners["validation_nonself_pass_rate"].mean()) if not winners.empty else float("nan"), "winner_validation_Q05_margin_mean": float(winners["validation_Q05_margin"].mean()) if not winners.empty else float("nan"), "winner_validation_MRR_mean": float(winners["validation_MRR"].mean()) if not winners.empty else float("nan"), "winner_validation_Top3_mean": float(winners["validation_Top3_oracle"].mean()) if not winners.empty else float("nan")})
    stability = pd.DataFrame(stability_rows).sort_values("candidate_id").reset_index(drop=True)
    stability["model_name"] = stability["candidate_name"]
    return cv, stability


def _build_deltas(candidates: tuple[BackwardCandidate, ...], summary: pd.DataFrame, model_frame: pd.DataFrame, frames: dict[int, pd.DataFrame], cv: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline = summary.loc[summary["candidate_id"].eq(0)].iloc[0]
    base_model = model_frame.loc[model_frame["candidate_id"].eq(0)]
    metric_columns = ["Y_Aend_train_NMSE_dB", "Y_Aend_B_NMSE_dB", "Y_C2_train_NMSE_dB", "Y_C2_B_NMSE_dB"]
    base_medians = {column: float(base_model[column].median()) for column in metric_columns}
    base_frame = frames[0].set_index("State_R")
    baseline_cv = cv.loc[cv["candidate_id"].eq(0)].sort_values("fold")
    delta_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    recommendations: list[dict[str, Any]] = []
    for candidate in candidates[1:]:
        current = summary.loc[summary["candidate_id"].eq(candidate.candidate_id)].iloc[0]
        current_model = model_frame.loc[model_frame["candidate_id"].eq(candidate.candidate_id)]
        current_medians = {column: float(current_model[column].median()) for column in metric_columns}
        current_frame = frames[candidate.candidate_id].set_index("State_R")
        base_pass = base_frame["realB_pass"].to_numpy(dtype=bool)
        current_pass = current_frame["realB_pass"].to_numpy(dtype=bool)
        recovered = int((~base_pass & current_pass).sum())
        regressed = int((base_pass & ~current_pass).sum())
        validation = cv.loc[cv["candidate_id"].eq(candidate.candidate_id)].sort_values("fold")
        validation_delta = validation["validation_top1_pass"].to_numpy(dtype=float) - baseline_cv["validation_top1_pass"].to_numpy(dtype=float)
        cv_negative = int((validation_delta < 0).sum())
        cv_stable = bool(cv_negative <= 1 and float(validation_delta.mean()) >= 0)
        cv_strict = bool(np.all(validation_delta >= 0))
        secondary_nonworse = bool(current["nonself_pass_rate"] >= baseline["nonself_pass_rate"] and current["share_margin_Q05"] >= baseline["share_margin_Q05"] and current["MRR"] >= baseline["MRR"] and current["Top3_oracle"] >= baseline["Top3_oracle"])
        delta_top1 = int(current["top1_realB_pass_count"] - baseline["top1_realB_pass_count"])
        if delta_top1 > 0:
            effect_class = "beneficial_to_delete"
        elif delta_top1 == 0 and recovered == 0 and regressed == 0 and secondary_nonworse and cv_strict:
            effect_class = "strict_redundant"
        elif delta_top1 == 0 and (recovered > 0 or regressed > 0):
            effect_class = "retrieval-equivalent-but-reordered"
        else:
            effect_class = "necessary"
        recommend = bool((delta_top1 > 0 and cv_stable) or effect_class == "strict_redundant")
        reason = "Top1 improves and CV has no systematic decline" if recommend and delta_top1 > 0 else "strict no-change deletion conditions are met" if recommend else "deletion lowers Top1 or changes/weakens the retrieval geometry"
        delta_rows.append({"removed_basis_id": candidate.removed_basis_id, "removed_basis_p": candidate.removed_order_p, "removed_basis_m": candidate.removed_delay_m, "removed_basis_q": candidate.removed_envelope_delay_q, "removed_is_cross_delay": candidate.removed_is_cross_delay, "baseline_top1": int(baseline["top1_realB_pass_count"]), "candidate_top1": int(current["top1_realB_pass_count"]), "delta_top1": delta_top1, "recovered_count": recovered, "regressed_count": regressed, "net_change": recovered - regressed, "baseline_exact": int(baseline["exact_hit_count"]), "candidate_exact": int(current["exact_hit_count"]), "delta_exact": int(current["exact_hit_count"] - baseline["exact_hit_count"]), "baseline_nonself_rate": float(baseline["nonself_pass_rate"]), "candidate_nonself_rate": float(current["nonself_pass_rate"]), "delta_nonself_rate": float(current["nonself_pass_rate"] - baseline["nonself_pass_rate"]), "baseline_Q05": float(baseline["share_margin_Q05"]), "candidate_Q05": float(current["share_margin_Q05"]), "delta_Q05": float(current["share_margin_Q05"] - baseline["share_margin_Q05"]), "baseline_MRR": float(baseline["MRR"]), "candidate_MRR": float(current["MRR"]), "delta_MRR": float(current["MRR"] - baseline["MRR"]), "baseline_Top2": int(baseline["Top2_oracle"]), "candidate_Top2": int(current["Top2_oracle"]), "baseline_Top3": int(baseline["Top3_oracle"]), "candidate_Top3": int(current["Top3_oracle"]), "delta_Aend_train_median": current_medians["Y_Aend_train_NMSE_dB"] - base_medians["Y_Aend_train_NMSE_dB"], "delta_Aend_B_median": current_medians["Y_Aend_B_NMSE_dB"] - base_medians["Y_Aend_B_NMSE_dB"], "delta_C2_train_median": current_medians["Y_C2_train_NMSE_dB"] - base_medians["Y_C2_train_NMSE_dB"], "delta_C2_B_median": current_medians["Y_C2_B_NMSE_dB"] - base_medians["Y_C2_B_NMSE_dB"], "deletion_effect_class": effect_class, "cv_stable": cv_stable})
        recommendations.append({"basis_id": candidate.removed_basis_id, "all425_top1_after_delete": int(current["top1_realB_pass_count"]), "delta_top1": delta_top1, "recovered": recovered, "regressed": regressed, "secondary_metrics_nonworse": secondary_nonworse, "cv_stable": cv_stable, "cv_strict_nonworse": cv_strict, "cv_validation_delta_mean": float(validation_delta.mean()), "cv_validation_delta_min": float(validation_delta.min()), "recommend_delete": recommend, "deletion_effect_class": effect_class, "recommendation_reason": reason})
        for state_id in range(STATE_COUNT):
            base_state_pass = bool(base_frame.loc[state_id, "realB_pass"])
            current_state_pass = bool(current_frame.loc[state_id, "realB_pass"])
            transition = "recovered" if not base_state_pass and current_state_pass else "regressed" if base_state_pass and not current_state_pass else "unchanged_pass" if base_state_pass else "unchanged_fail"
            transition_rows.append({"candidate_id": candidate.candidate_id, "candidate_name": candidate.candidate_name, "removed_basis_id": candidate.removed_basis_id, "State_R": state_id, "baseline_State_Q": int(base_frame.loc[state_id, "State_Q"]), "candidate_State_Q": int(current_frame.loc[state_id, "State_Q"]), "baseline_realB_CNMSE_dB": float(base_frame.loc[state_id, "retrieved_realB_CNMSE_dB"]), "candidate_realB_CNMSE_dB": float(current_frame.loc[state_id, "retrieved_realB_CNMSE_dB"]), "baseline_pass": base_state_pass, "candidate_pass": current_state_pass, "transition": transition})
    return pd.DataFrame(delta_rows), pd.DataFrame(transition_rows), pd.DataFrame(recommendations)


def _failure6_detail(candidates: tuple[BackwardCandidate, ...], frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    baseline = frames[0].set_index("State_R")
    failure_ids = baseline.loc[~baseline["realB_pass"]].index.astype(int).tolist()
    if failure_ids != [189, 195, 323, 335, 340, 346]:
        raise RuntimeError(f"K12 failure set changed: {failure_ids}")
    rows: list[dict[str, Any]] = []
    for candidate in candidates[1:]:
        current = frames[candidate.candidate_id].set_index("State_R")
        for state_id in failure_ids:
            baseline_rank = int(baseline.loc[state_id, "first_shareable_rank"])
            candidate_rank = int(current.loc[state_id, "first_shareable_rank"])
            rows.append({"candidate_id": candidate.candidate_id, "candidate_name": candidate.candidate_name, "removed_basis_id": candidate.removed_basis_id, "State_R": state_id, "baseline_State_Q": int(baseline.loc[state_id, "State_Q"]), "candidate_State_Q": int(current.loc[state_id, "State_Q"]), "baseline_first_shareable_rank": baseline_rank, "candidate_first_shareable_rank": candidate_rank, "rank_delta": candidate_rank - baseline_rank, "baseline_shareability_margin_dB": float(baseline.loc[state_id, "shareability_margin_dB"]), "candidate_shareability_margin_dB": float(current.loc[state_id, "shareability_margin_dB"]), "baseline_realB_CNMSE_dB": float(baseline.loc[state_id, "retrieved_realB_CNMSE_dB"]), "candidate_realB_CNMSE_dB": float(current.loc[state_id, "retrieved_realB_CNMSE_dB"]), "baseline_pass": bool(baseline.loc[state_id, "realB_pass"]), "candidate_pass": bool(current.loc[state_id, "realB_pass"]), "recovered": bool(current.loc[state_id, "realB_pass"]), "rank_improved": candidate_rank < baseline_rank, "rank_degraded": candidate_rank > baseline_rank})
    return pd.DataFrame(rows)


def _crossdelay_interaction(summary: pd.DataFrame, frames: dict[int, pd.DataFrame], candidates: tuple[BackwardCandidate, ...]) -> pd.DataFrame:
    wanted = {"K12_full": 0, "K12_minus_ENV_p04_m2_q0": next(candidate.candidate_id for candidate in candidates if candidate.removed_basis_id == K11_BASIS_ID), "K12_minus_ENV_p04_m0_q1": next(candidate.candidate_id for candidate in candidates if candidate.removed_basis_id == BEST_K12_BASIS_ID)}
    rows: list[dict[str, Any]] = []
    for name, candidate_id in wanted.items():
        current = summary.loc[summary["candidate_id"].eq(candidate_id)].iloc[0]
        has_m2 = name in {"K12_full", "K12_minus_ENV_p04_m0_q1"}
        has_m0q1 = name in {"K12_full", "K12_minus_ENV_p04_m2_q0"}
        rows.append({"model": name, "has_p04_m2_q0": has_m2, "has_p04_m0_q1": has_m0q1, "Top1": int(current["top1_pass_count"]), "Exact": int(current["exact_hit_count"]), "Nonself_pass": int(current["nonself_pass_count"]), "Q05_margin": float(current["share_margin_Q05"]), "MRR": float(current["MRR"]), "Top2": int(current["Top2_oracle"]), "Top3": int(current["Top3_oracle"]), "failure_state_ids": frames[candidate_id].loc[~frames[candidate_id]["realB_pass"], "State_R"].astype(int).tolist()})
    return pd.DataFrame(rows)


def _write_figures(summary: pd.DataFrame, delta: pd.DataFrame, failure6: pd.DataFrame, baseline_profile: pd.DataFrame) -> None:
    ordered = summary.sort_values("candidate_id")
    labels = ordered["candidate_name"].tolist()
    values = ordered["top1_pass_count"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
    bars = ax.bar(np.arange(len(labels)), values, color=plt.get_cmap("tab20")(np.linspace(0, 1, len(labels))))
    ax.axhline(419, color="black", linestyle="--", linewidth=1.0, label="K12 baseline = 419")
    for bar, value in zip(bars, values, strict=True):
        if value > 419:
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.3, str(int(value)), ha="center", va="bottom", fontsize=7)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=70, ha="right", fontsize=7)
    ax.set_ylabel("Top-1 Real-B pass count")
    ax.set_xlabel("K12 support / one-basis deletion")
    ax.set_ylim(0, 425)
    ax.set_title("K12 backward deletion Top-1 Real-B comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "18_candidate_top1_comparison.png")
    plt.close(fig)

    d = delta.sort_values("removed_basis_id")
    vals = d["delta_top1"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(15, 7), dpi=300)
    colors = ["tab:green" if value > 0 else "tab:red" if value < 0 else "tab:gray" for value in vals]
    bars = ax.bar(np.arange(len(d)), vals, color=colors)
    ax.axhline(0, color="black", linewidth=0.9)
    for bar, value in zip(bars, vals, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + (0.2 if value >= 0 else -0.3), f"{value:+.0f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
    ax.set_xticks(np.arange(len(d)), d["removed_basis_id"], rotation=70, ha="right", fontsize=8)
    ax.set_ylabel("Δ Top-1 Real-B pass count vs K12")
    ax.set_xlabel("Removed basis")
    ax.set_ylim(float(vals.min()) - 5, max(5, float(vals.max()) + 5))
    ax.set_title("Backward retrieval contribution of each K12 basis")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "19_backward_basis_contribution.png")
    plt.close(fig)

    fail_ids = baseline_profile.loc[~baseline_profile["Top1_shareable"], "State_R"].astype(int).tolist()
    model_ids = [0] + sorted(failure6["candidate_id"].unique().tolist())
    xlabels = ["K12_full"] + [str(name).replace("K12_minus_", "-") for name in summary.loc[summary["candidate_id"].ne(0)].sort_values("candidate_id")["candidate_name"]]
    fig, ax = plt.subplots(figsize=(18, 8), dpi=300)
    for state_id in fail_ids:
        y = [int(baseline_profile.loc[baseline_profile["State_R"].eq(state_id), "first_shareable_rank"].iloc[0])]
        for candidate_id in model_ids[1:]:
            y.append(int(failure6.loc[(failure6["candidate_id"].eq(candidate_id)) & failure6["State_R"].eq(state_id), "candidate_first_shareable_rank"].iloc[0]))
        ax.plot(np.arange(len(model_ids)), y, marker="o", linewidth=1.0, label=f"State {state_id}")
    ax.set_xticks(np.arange(len(model_ids)), xlabels, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("First-shareable rank")
    ax.set_xlabel("K12 baseline / one-basis deletion")
    ax.set_title("K12 deletion effect on the six baseline failure states")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "20_failure6_rank_change.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 8), dpi=300)
    ax.scatter(delta["delta_C2_train_median"], delta["delta_top1"], c=delta["removed_is_cross_delay"].astype(int), cmap="tab10", s=45, alpha=0.85)
    for row in delta.loc[delta["delta_top1"].abs().sort_values(ascending=False).head(8).index].itertuples(index=False):
        ax.annotate(str(row.removed_basis_id), (row.delta_C2_train_median, row.delta_top1), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Deletion Δ C2 Train median NMSE (dB)")
    ax.set_ylabel("Deletion Δ Top-1 Real-B pass count")
    ax.set_title("K12 deletion: modeling contribution versus retrieval contribution")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "21_modeling_vs_retrieval_deletion_effect.png")
    plt.close(fig)


def _run(*, resume: bool = False) -> dict[str, Any]:
    if RESULT_ROOT.exists() and any(RESULT_ROOT.iterdir()) and not resume:
        raise RuntimeError(f"result directory is not empty; use --resume: {RESULT_ROOT}")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "distance_matrices").mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "fingerprints").mkdir(parents=True, exist_ok=True)
    raw_before = raw_manifest_gate()
    support_ids, by_id, k11_index, k12_index, formal_support = _load_frozen_k12()
    envelope_gate = dictionary_gate()
    references = _load_references()
    common_b, common_meta = verify_common_b_contract()
    common_b = np.asarray(common_b, dtype=np.complex128)
    if common_meta["sha256"] != EXPECTED_COMMON_B_SHA:
        raise RuntimeError("common-B hash changed")
    real_b_random_error = _random_real_b_check(references["real_b_distance"])
    baseline_profile, _ = _failure_rank_profile(references["k12_distance"], references["shareability"], references["real_b_distance"])
    baseline_profile.to_csv(RESULT_ROOT / "04_k12_failure_rank_profile.csv", index=False)
    _, candidates = _build_candidates(support_ids, by_id)
    if len(candidates) != 13:
        raise RuntimeError("K12 backward candidate count is not 13")
    rank_distribution = {"rank_1": int((baseline_profile["first_shareable_rank"] == 1).sum()), "rank_2": int((baseline_profile["first_shareable_rank"] == 2).sum()), "rank_3_to_5": int(baseline_profile["first_shareable_rank"].between(3, 5).sum()), "rank_6_to_10": int(baseline_profile["first_shareable_rank"].between(6, 10).sum()), "rank_gt_10": int((baseline_profile["first_shareable_rank"] > 10).sum())}
    references["folds"].to_csv(RESULT_ROOT / "13_query_state_cv_folds.csv", index=False)
    (RESULT_ROOT / "00_task_definition.txt").write_text("\n".join([f"Task: {TASK_NAME}", f"Parent K12: {BEST_K12_NAME}; support={list(support_ids)}; K=12.", "Models: K12_full plus 12 K11 leave-one-basis-out candidates.", "All candidates use dmax=2, Ridge lambda=1e-8, frozen ABC/common-B/Real-B/allow-self protocol.", "Primary question: whether any single deletion improves or safely preserves 419/425.", "No second deletion, K13, swap, beam search, multi-basis change, lambda/order/memory/dmax scan, Top-k reranking, ensemble, clustering, DPD replay or low-bandwidth task."]) + "\n", encoding="utf-8")
    _write_json(RESULT_ROOT / "01_reuse_audit.json", {"task_name": TASK_NAME, "formal_k12_support_source": str(K12_ROOT / "06_candidate_supports.csv"), "support_ids": list(support_ids), "envelope75_gate": envelope_gate, "historical_mp10_backend": "behavior_modeling.sparse_gmp", "dmax": backend.MP_DMAX, "ridge_lambda": backend.RIDGE_LAMBDA, "separate_hnorm_found": False, "real_B_source": str(K9_ROOT / "04_realB_ground_truth_cnmse_matrix.npy"), "cv_source": str(K12_ROOT / "16_query_state_cv_folds.csv"), "k12_failure_set": baseline_profile.loc[~baseline_profile["Top1_shareable"], "State_R"].astype(int).tolist(), "failure_rank_distribution": rank_distribution, "real_B_random_pair_max_abs_error_dB": real_b_random_error})
    (RESULT_ROOT / "01_reuse_audit.txt").write_text("\n".join([f"Task: {TASK_NAME}", f"Frozen K12 parent: {BEST_K12_NAME}", f"Support read from formal metadata: {list(support_ids)}", "Historical MP10, Envelope75, ABC, common-B, Real-B matrix/mask, CNMSE, Top1, margins, MRR, Top-k and Query-State CV definitions are reused.", "Each State builds full K12 once and evaluates K12 baseline plus 12 one-column deletions sequentially.", "All deletion candidates keep dmax=2 and lambda=1e-8. No Hnorm was added.", f"K12 failure set={baseline_profile.loc[~baseline_profile['Top1_shareable'], 'State_R'].astype(int).tolist()}; rank distribution={rank_distribution}; hard-pair profile is inherited from K12 distance ordering.", "Deletion ENV_p04_m0_q1 must reproduce K11; deletion ENV_p04_m2_q0 is a separate substitution test.", "No second deletion or K13 was run."]) + "\n", encoding="utf-8")
    _checkpoint(phase="reuse_audit_support_failure_profile_passed", raw_manifest_before=raw_before, candidate_count=13, failure_rank_distribution=rank_distribution, real_B_random_pair_max_abs_error_dB=real_b_random_error)
    model_frame, theta_a, theta_c = _run_model_phase(candidates, k11_index, k12_index, resume)
    mp_common = backend.historical_mp10.build_frozen_mp_basis(common_b)
    env_common = build_envelope_bank(common_b, tuple(by_id.values()))
    envelope_terms = tuple(build_envelope_dictionary())
    env_common = build_envelope_bank(common_b, envelope_terms)
    full_common = np.column_stack((mp_common, env_common[:, k11_index], env_common[:, k12_index]))
    if full_common.shape != (backend.FINGERPRINT_LENGTH, backend.K12):
        raise RuntimeError(f"K12 common-B Phi shape failed: {full_common.shape}")
    frames: dict[int, pd.DataFrame] = {}
    summaries: list[dict[str, Any]] = []
    distances: dict[int, np.ndarray] = {}
    baseline_frame, baseline_summary, baseline_distance, baseline_lut, baseline_query = _candidate_retrieval(candidates[0], theta_a, theta_c, full_common, references["real_b_distance"], references["shareability"])
    frames[0] = baseline_frame
    summaries.append(baseline_summary)
    distances[0] = baseline_distance
    np.save(RESULT_ROOT / "distance_matrices" / "K12_full.npy", baseline_distance)
    np.save(RESULT_ROOT / "fingerprints" / "K12_full_lut.npy", baseline_lut)
    np.save(RESULT_ROOT / "fingerprints" / "K12_full_query.npy", baseline_query)
    baseline = _baseline_regression(model_frame, baseline_frame, theta_a, theta_c, baseline_distance, baseline_lut, baseline_query, references)
    _write_json(
        RESULT_ROOT / "05_baseline_regression_check.json",
        {
            "pass": bool(baseline["pass"]),
            "K12_baseline": baseline,
            "reference_candidate": BEST_K12_NAME,
            "deletion_to_k11_reference_pending": True,
        },
    )
    deletion_to_k11 = None
    _checkpoint(phase="k12_baseline_regression_passed", baseline_regression=baseline)
    for candidate in candidates[1:]:
        frame, summary, distance, lut, query = _candidate_retrieval(candidate, theta_a, theta_c, full_common, references["real_b_distance"], references["shareability"])
        frames[candidate.candidate_id] = frame
        summaries.append(summary)
        distances[candidate.candidate_id] = distance
        np.save(RESULT_ROOT / "distance_matrices" / f"{candidate.candidate_name}.npy", distance)
        if candidate.removed_basis_id == BEST_K12_BASIS_ID:
            deletion_to_k11 = _deletion_to_k11_regression(candidate, frame, model_frame, theta_a, theta_c, distance, lut, query, references)
    if deletion_to_k11 is None or not deletion_to_k11["pass"]:
        raise RuntimeError("critical deletion-to-K11 regression was not completed")
    _write_json(
        RESULT_ROOT / "05_baseline_regression_check.json",
        {
            "pass": bool(baseline["pass"] and deletion_to_k11["pass"]),
            "K12_baseline": baseline,
            "reference_candidate": BEST_K12_NAME,
            "deletion_to_k11": deletion_to_k11,
        },
    )
    summary = pd.DataFrame(summaries)
    summary["model_id"] = summary["candidate_id"].astype(int)
    summary["model_name"] = summary["candidate_name"]
    summary["removed_basis_id"] = summary["removed_basis_id"].fillna("NONE")
    summary["top1_pass_count"] = summary["top1_realB_pass_count"]
    summary["top1_pass_rate"] = summary["top1_realB_pass_rate"]
    summary["delta_top1_vs_k12"] = summary["top1_pass_count"] - int(summary.loc[summary["candidate_id"].eq(0), "top1_pass_count"].iloc[0])
    summary["all425_rank"] = prior_backward.k9_utils._assign_diagnostic_ranks(summaries)["diagnostic_rank"]
    summary["recovered_count"] = 0
    summary["regressed_count"] = 0
    summary["deletion_effect_class"] = "baseline"
    query_frame = pd.concat(frames.values(), ignore_index=True).sort_values(["candidate_id", "State_R"]).reset_index(drop=True)
    query_frame.to_csv(RESULT_ROOT / "09_candidate_query_metrics_long.csv", index=False)
    cv_frame, stability = _local_cv_results(candidates, frames, references["folds"])
    delta, recovered_frame, recommendations = _build_deltas(candidates, summary, model_frame, frames, cv_frame)
    for row in delta.itertuples(index=False):
        candidate_id = next(candidate.candidate_id for candidate in candidates if candidate.removed_basis_id == row.removed_basis_id)
        selector = summary["candidate_id"].eq(candidate_id)
        summary.loc[selector, "recovered_count"] = int(row.recovered_count)
        summary.loc[selector, "regressed_count"] = int(row.regressed_count)
        summary.loc[selector, "deletion_effect_class"] = row.deletion_effect_class
    summary["hard_pair_corrected_count"] = 0
    summary.to_csv(RESULT_ROOT / "08_candidate_retrieval_summary.csv", index=False)
    delta.to_csv(RESULT_ROOT / "10_backward_deletion_delta.csv", index=False)
    recovered_frame.to_csv(RESULT_ROOT / "12_recovered_regressed_states.csv", index=False)
    failure6 = _failure6_detail(candidates, frames)
    failure6.to_csv(RESULT_ROOT / "11_failure6_detailed_analysis.csv", index=False)
    cv_frame.to_csv(RESULT_ROOT / "14_query_state_cv_results.csv", index=False)
    stability.to_csv(RESULT_ROOT / "15_query_state_cv_selection_stability.csv", index=False)
    recommendations.to_csv(RESULT_ROOT / "16_deletion_recommendation.csv", index=False)
    interaction = _crossdelay_interaction(summary, frames, candidates)
    interaction.to_csv(RESULT_ROOT / "17_crossdelay_interaction_summary.csv", index=False)
    winner = summary.sort_values("all425_rank").iloc[0]
    winner_id = int(winner["candidate_id"])
    winner_candidate = candidates[winner_id]
    winner_phi = full_common[:, list(winner_candidate.support)]
    np.save(RESULT_ROOT / "fingerprints" / "diagnostic_best_deletion_lut.npy", (winner_phi @ theta_a[winner_id][:, list(winner_candidate.support)].T).T.astype(np.complex128))
    np.save(RESULT_ROOT / "fingerprints" / "diagnostic_best_deletion_query.npy", (winner_phi @ theta_c[winner_id][:, list(winner_candidate.support)].T).T.astype(np.complex128))
    _write_figures(summary, delta, failure6, baseline_profile)
    raw_after = raw_manifest_gate()
    if raw_before != raw_after:
        raise RuntimeError("data/raw manifest changed")
    recommendations_true = recommendations.loc[recommendations["recommend_delete"]]
    summary_lines = [
        f"Task: {TASK_NAME}",
        f"Parent K12: {BEST_K12_NAME}; support={list(support_ids)}; K=12.",
        f"K12 baseline regression PASS={baseline['pass']}; Top1=419/425; Failure=6; Exact=235/425; Non-self=184/190; Q05={float(summary.loc[summary['candidate_id'].eq(0), 'share_margin_Q05'].iloc[0]):.9g}; MRR={float(summary.loc[summary['candidate_id'].eq(0), 'MRR'].iloc[0]):.9g}; Top2=423; Top3=424; Top5=424; Top10=425.",
        f"Current K12 failure states: {baseline_profile.loc[~baseline_profile['Top1_shareable'], 'State_R'].astype(int).tolist()}.",
        f"Failure rank distribution: {rank_distribution}; deletion ENV_p04_m0_q1 -> K11 regression PASS={deletion_to_k11['pass']}.",
        "",
        "All 12 deletion candidates:",
    ]
    for row in summary.sort_values("all425_rank").itertuples(index=False):
        summary_lines.append(f"rank={int(row.all425_rank)} {row.model_name}: removed={row.removed_basis_id}, Top1={int(row.top1_pass_count)}/425, delta={int(row.delta_top1_vs_k12)}, recovered={int(row.recovered_count)}, regressed={int(row.regressed_count)}, Exact={int(row.exact_hit_count)}, nonself={float(row.nonself_pass_rate):.9g}, Q05={float(row.share_margin_Q05):.9g}, MRR={float(row.MRR):.9g}, Top2={int(row.Top2_oracle)}, Top3={int(row.Top3_oracle)}, class={row.deletion_effect_class}")
    summary_lines.extend(["", f"All425 diagnostic winner: {winner['model_name']}.", f"recommend_delete count={int(recommendations['recommend_delete'].sum())}; basis_ids={recommendations_true['basis_id'].tolist()}.", "No basis was physically removed or frozen as a new final model.", "", "Cross-delay interaction:"])
    for row in interaction.itertuples(index=False):
        summary_lines.append(f"{row.model}: has_p04_m2_q0={row.has_p04_m2_q0}, has_p04_m0_q1={row.has_p04_m0_q1}, Top1={int(row.Top1)}, Exact={int(row.Exact)}, Nonself={int(row.Nonself_pass)}, Q05={float(row.Q05_margin):.9g}, MRR={float(row.MRR):.9g}, failures={row.failure_state_ids}")
    summary_lines.append("")
    summary_lines.append("Five-fold Query-State CV:")
    for row in stability.loc[stability["train_winner_count"].gt(0)].sort_values("train_winner_count", ascending=False).itertuples(index=False):
        summary_lines.append(f"{row.model_name}: train winner={int(row.train_winner_count)}/5, folds={row.winner_folds}, validation Top1 mean/min={float(row.winner_validation_top1_pass_mean):.6g}/{float(row.winner_validation_top1_pass_min):.6g}")
    summary_lines.extend(["", f"raw before={json.dumps(raw_before, ensure_ascii=False, sort_keys=True)}", f"raw after={json.dumps(raw_after, ensure_ascii=False, sort_keys=True)}", f"raw unchanged={raw_before == raw_after}", "No second deletion, K13, swap, beam search, multi-basis pruning/addition, lambda/order/memory/dmax scan, Top-k reranking, ensemble/fusion, clustering, Type III, low-bandwidth or DPD replay was run.", "CV is Query-State stability analysis, not an independent final test."])
    (RESULT_ROOT / "22_final_result_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    transition_counts = recovered_frame["transition"].value_counts().to_dict()
    result = {"task": TASK_NAME, "status": "SUCCESS", "baseline_regression_pass": bool(baseline["pass"]), "deletion_to_k11_regression_pass": bool(deletion_to_k11["pass"]), "diagnostic_winner": str(winner["model_name"]), "diagnostic_winner_top1": int(winner["top1_pass_count"]), "max_deletion_top1": int(summary.loc[summary["candidate_id"].ne(0), "top1_pass_count"].max()), "recommend_delete_basis_ids": recommendations_true["basis_id"].tolist(), "transition_counts_across_deletions": {key: int(transition_counts.get(key, 0)) for key in ("recovered", "regressed", "unchanged_pass", "unchanged_fail")}, "cv_winner_counts": stability.loc[stability["train_winner_count"].gt(0), ["model_name", "train_winner_count"]].to_dict("records"), "raw_data_modified": raw_before != raw_after, "result_root": str(RESULT_ROOT)}
    _write_json(RESULT_ROOT / "23_checkpoint.json", {"phase": "completed", **result, "independent_validation_pending": True})
    _append_log(f"\n[{_now()}] Complete {TASK_NAME}\nResult={json.dumps(result, ensure_ascii=False, sort_keys=True)}\nNo second deletion or K13 was run.\n")
    _append_handoff(f"完成当前最佳 K12 的 12 个单项 backward deletion；结果目录：{RESULT_ROOT}\n摘要：{json.dumps(result, ensure_ascii=False, sort_keys=True)}\n完成 K12/K11 regression、failure6、cross-delay interaction、Full425 retrieval、5-fold CV 和 deletion recommendation；未冻结新模型。")
    return result


def main() -> None:
    result = _run(resume="--resume" in sys.argv[1:])
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    __import__("multiprocessing").freeze_support()
    main()
