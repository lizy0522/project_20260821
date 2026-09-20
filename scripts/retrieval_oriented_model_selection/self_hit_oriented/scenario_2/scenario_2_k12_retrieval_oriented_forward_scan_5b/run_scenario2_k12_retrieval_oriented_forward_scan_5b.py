# ruff: noqa: E402,E501,I001

"""Run one-step retrieval-oriented K12 forward additions from the best K11."""

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
from retrieval_oriented_model_selection.self_hit_oriented.scenario_2.scenario_2_k12_retrieval_oriented_forward_scan_5b import (  # noqa: E402
    k12_backend as backend,
)
from retrieval_oriented_model_selection.shared import mp10_k9_retrieval_runner as k9_utils

TASK_NAME = "scenario_2_k12_retrieval_oriented_forward_scan_5b"
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
class ForwardCandidate:
    candidate_id: int
    candidate_name: str
    K: int
    added_basis_id: str | None
    added_order_p: int | None
    added_delay_m: int | None
    added_envelope_delay_q: int | None
    added_family: str | None
    is_cross_delay: bool | None
    envelope_index: int | None
    support: tuple[int, ...]
    retained_basis_ids: tuple[str, ...]
    support_hash: str

    @property
    def removed_basis_id(self) -> str | None:
        # Reuse K9 ranking helpers without changing their semantics.
        return self.added_basis_id


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
        RESULT_ROOT / "27_checkpoint.json",
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
            "candidate_count": 65,
            "remaining_envelope75_count": 64,
            "dmax_fixed": backend.MP_DMAX,
            "ridge_lambda_fixed": backend.RIDGE_LAMBDA,
            "k13_or_higher_performed": False,
            "second_forward_addition_performed": False,
            "backward_elimination_performed": False,
            "multi_basis_addition_performed": False,
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


def _load_frozen_k11() -> tuple[tuple[str, ...], dict[str, EnvelopeBasis], int, pd.DataFrame]:
    path = K11_ROOT / "04_candidate_supports.csv"
    if not path.is_file():
        raise FileNotFoundError(f"formal K11 support is missing: {path}")
    frame = pd.read_csv(path)
    row = frame.loc[frame["model_name"].eq(BEST_K11_NAME)]
    if row.shape[0] != 1 or int(row.iloc[0]["K"]) != 11:
        raise RuntimeError("formal best K11 support is missing or invalid")
    support_ids = tuple(json.loads(row.iloc[0]["support_basis_ids"]))
    if len(support_ids) != 11 or support_ids[-1] != BEST_K11_BASIS_ID:
        raise RuntimeError(f"frozen K11 support changed: {support_ids}")
    terms = tuple(build_envelope_dictionary())
    by_id = {term.basis_id: term for term in terms}
    if len(terms) != 75 or any(basis_id not in by_id for basis_id in support_ids):
        raise RuntimeError("Envelope75/K11 support relationship changed")
    added_index = int(by_id[BEST_K11_BASIS_ID].index)
    return support_ids, by_id, added_index, frame


def _build_candidates(support_ids: tuple[str, ...], by_id: dict[str, EnvelopeBasis]) -> tuple[pd.DataFrame, tuple[ForwardCandidate, ...]]:
    candidates = [
        ForwardCandidate(
            candidate_id=0,
            candidate_name="K11_full",
            K=11,
            added_basis_id=None,
            added_order_p=None,
            added_delay_m=None,
            added_envelope_delay_q=None,
            added_family=None,
            is_cross_delay=None,
            envelope_index=None,
            support=tuple(range(11)),
            retained_basis_ids=support_ids,
            support_hash=_support_hash(support_ids),
        )
    ]
    remaining = tuple(term for term in by_id.values() if term.basis_id not in set(support_ids))
    if len(remaining) != 64:
        raise RuntimeError(f"remaining Envelope75 candidate count is {len(remaining)}, expected 64")
    rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = [
        {
            "model_id": 0,
            "model_name": "K11_full",
            "K": 11,
            "added_basis_id": "NONE",
            "added_order_p": np.nan,
            "added_delay_m": np.nan,
            "added_envelope_delay_q": np.nan,
            "is_cross_delay": False,
            "support_basis_ids": json.dumps(list(support_ids), ensure_ascii=False),
            "support_indices": json.dumps(list(range(11))),
            "support_hash": _support_hash(support_ids),
            "dmax": backend.MP_DMAX,
            "ridge_lambda": backend.RIDGE_LAMBDA,
        }
    ]
    for candidate_id, term in enumerate(remaining, start=1):
        is_cross = bool(term.order > 1 and term.signal_delay != term.envelope_delay)
        candidate_ids = support_ids + (term.basis_id,)
        candidate = ForwardCandidate(
            candidate_id=candidate_id,
            candidate_name=f"K11_plus_{term.basis_id}",
            K=12,
            added_basis_id=term.basis_id,
            added_order_p=int(term.order),
            added_delay_m=None if term.signal_delay is None else int(term.signal_delay),
            added_envelope_delay_q=None if term.envelope_delay is None else int(term.envelope_delay),
            added_family=term.family,
            is_cross_delay=is_cross,
            envelope_index=int(term.index),
            support=tuple(range(12)),
            retained_basis_ids=candidate_ids,
            support_hash=_support_hash(candidate_ids),
        )
        candidates.append(candidate)
        rows.append(
            {
                "candidate_index": candidate_id,
                "basis_id": term.basis_id,
                "basis_type": term.family,
                "order_p": int(term.order),
                "delay_m": term.signal_delay,
                "envelope_delay_q": term.envelope_delay,
                "is_cross_delay": is_cross,
                "was_in_mp10": False,
                "was_k11_added_basis": False,
                "envelope_dictionary_index": int(term.index),
                "formula": term.formula,
            }
        )
        support_rows.append(
            {
                "model_id": candidate_id,
                "model_name": candidate.candidate_name,
                "K": 12,
                "added_basis_id": term.basis_id,
                "added_order_p": int(term.order),
                "added_delay_m": term.signal_delay,
                "added_envelope_delay_q": term.envelope_delay,
                "is_cross_delay": is_cross,
                "support_basis_ids": json.dumps(list(candidate_ids), ensure_ascii=False),
                "support_indices": json.dumps(list(range(12))),
                "support_hash": candidate.support_hash,
                "dmax": backend.MP_DMAX,
                "ridge_lambda": backend.RIDGE_LAMBDA,
            }
        )
    support_frame = pd.DataFrame(support_rows)
    support_frame.to_csv(RESULT_ROOT / "06_candidate_supports.csv", index=False)
    pd.DataFrame(rows).to_csv(RESULT_ROOT / "03_remaining_envelope75_candidates.csv", index=False)
    baseline_frame = pd.DataFrame(
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
    )
    baseline_frame.to_csv(RESULT_ROOT / "02_k11_baseline_support.csv", index=False)
    return support_frame, tuple(candidates)


def _load_references() -> dict[str, Any]:
    previous_mp10 = k9_utils._load_previous_baseline()
    required = [
        K11_ROOT / "06_candidate_model_quality_long.csv",
        K11_ROOT / "07_candidate_coefficients.npz",
        K11_ROOT / "09_candidate_query_metrics_long.csv",
        K11_ROOT / "distance_matrices" / f"{BEST_K11_NAME}.npy",
        K11_ROOT / "fingerprints" / "diagnostic_best_k11_lut.npy",
        K11_ROOT / "fingerprints" / "diagnostic_best_k11_query.npy",
        K9_ROOT / "04_realB_ground_truth_cnmse_matrix.npy",
        K9_ROOT / "05_realB_shareability_mask.npy",
        K11_ROOT / "13_query_state_cv_folds.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"K11/K9 reference artifacts missing: {missing}")
    k11_model = pd.read_csv(K11_ROOT / "06_candidate_model_quality_long.csv")
    k11_query = pd.read_csv(K11_ROOT / "09_candidate_query_metrics_long.csv")
    with np.load(K11_ROOT / "07_candidate_coefficients.npz", allow_pickle=False) as data:
        candidate_ids = np.asarray(data["model_ids"], dtype=np.int64)
        theta_a = np.asarray(data["theta_Aend_padded"])
        theta_c = np.asarray(data["theta_C2_padded"])
    real_b_distance = np.load(K9_ROOT / "04_realB_ground_truth_cnmse_matrix.npy")
    shareability = np.load(K9_ROOT / "05_realB_shareability_mask.npy")
    folds_k11 = pd.read_csv(K11_ROOT / "13_query_state_cv_folds.csv").sort_values("state_id").reset_index(drop=True)
    folds_k9 = pd.read_csv(K9_ROOT / "12_query_state_cv_folds.csv").sort_values("state_id").reset_index(drop=True)
    baseline_query = k11_query.loc[k11_query["candidate_id"].eq(21)].sort_values("State_R").reset_index(drop=True)
    if baseline_query.shape[0] != STATE_COUNT:
        raise RuntimeError("K11 reference baseline query table is not 425 rows")
    if int(baseline_query["realB_pass"].sum()) != 416 or int(baseline_query["exact_hit"].sum()) != 218:
        raise RuntimeError("K11 reference baseline counts changed")
    expected_failures = [189, 195, 199, 323, 327, 330, 335, 340, 346]
    observed_failures = baseline_query.loc[~baseline_query["realB_pass"], "State_R"].astype(int).tolist()
    if observed_failures != expected_failures:
        raise RuntimeError(f"K11 failure set changed: {observed_failures}")
    if not np.array_equal(folds_k11[["state_id", "fold"]].to_numpy(), folds_k9[["state_id", "fold"]].to_numpy()):
        raise RuntimeError("K11 and K9 Query-State folds differ")
    if real_b_distance.shape != (STATE_COUNT, STATE_COUNT) or shareability.shape != real_b_distance.shape:
        raise RuntimeError("Real-B ground truth/mask shape changed")
    if not np.all(np.isneginf(np.diag(real_b_distance))) or not shareability.diagonal().all():
        raise RuntimeError("Real-B diagonal/shareability contract changed")
    if theta_a.shape != (66, STATE_COUNT, 11) or theta_c.shape != theta_a.shape:
        raise RuntimeError("K11 reference coefficient shape changed")
    return {
        "mp10": previous_mp10,
        "k11_model": k11_model,
        "k11_query": k11_query,
        "k11_candidate_ids": candidate_ids,
        "k11_theta_a": theta_a,
        "k11_theta_c": theta_c,
        "k11_distance": np.load(K11_ROOT / "distance_matrices" / f"{BEST_K11_NAME}.npy"),
        "k11_lut": np.load(K11_ROOT / "fingerprints" / "diagnostic_best_k11_lut.npy"),
        "k11_query_fp": np.load(K11_ROOT / "fingerprints" / "diagnostic_best_k11_query.npy"),
        "baseline_query": baseline_query,
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
                cache[state_id] = np.asarray(
                    build_off_segments(data, partition)["B"].output[2:],
                    dtype=np.complex128,
                )
        direct = cnmse(cache[int(real_id)], cache[int(query_id)])
        stored = float(real_b_distance[int(real_id), int(query_id)])
        if np.isneginf(direct) and np.isneginf(stored):
            continue
        maximum = max(maximum, abs(direct - stored))
    if maximum > REAL_B_SANITY_TOLERANCE_DB:
        raise RuntimeError(f"Real-B ground truth random-pair check failed: {maximum:.3e} dB")
    return maximum


def _failure_rank_profile(distance: np.ndarray, shareability: np.ndarray, real_b_distance: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    hard_rows: list[dict[str, Any]] = []
    state_ids = np.arange(STATE_COUNT, dtype=np.int64)
    for state_id in range(STATE_COUNT):
        order = np.lexsort((state_ids, distance[state_id]))
        top1 = int(order[0])
        top2 = int(order[1])
        top3 = int(order[2])
        share_positions = np.flatnonzero(shareability[state_id, order])
        first_position = int(share_positions[0])
        first_state = int(order[first_position])
        top1_shareable = bool(shareability[state_id, top1])
        top2_shareable = bool(shareability[state_id, top2])
        row = {
            "State_R": state_id,
            "Top1_State_Q": top1,
            "Top1_shareable": top1_shareable,
            "Top2_State_Q": top2,
            "Top2_shareable": top2_shareable,
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
        rows.append(row)
        if not top1_shareable and top2_shareable:
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
    profile = pd.DataFrame(rows)
    hard = pd.DataFrame(hard_rows)
    profile.to_csv(RESULT_ROOT / "04_k11_failure_rank_profile.csv", index=False)
    hard.to_csv(RESULT_ROOT / "05_k11_hard_pairs.csv", index=False)
    return profile, hard


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


def _run_model_phase(candidates: tuple[ForwardCandidate, ...], added_index: int, resume: bool) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    model_path = RESULT_ROOT / "08_candidate_model_quality_long.csv"
    coefficient_path = RESULT_ROOT / "09_candidate_coefficients.npz"
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
    specs = tuple((candidate.candidate_id, candidate.envelope_index) for candidate in candidates)
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=WORKER_COUNT,
        mp_context=get_context("spawn"),
        initializer=backend.worker_init,
        initargs=(specs, added_index),
    ) as executor:
        futures = {executor.submit(backend.model_worker_entry, state_id): state_id for state_id in range(STATE_COUNT)}
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed % 50 == 0 or completed == STATE_COUNT:
                _checkpoint(phase="candidate_model_progress", completed_model_states=completed, completed_model_state_ids=sorted(int(item["State_n_R"]) for item in results))
                print(f"[K12 MODEL] {completed}/{STATE_COUNT}", flush=True)
    results.sort(key=lambda item: int(item["State_n_R"]))
    if [int(item["State_n_R"]) for item in results] != list(range(STATE_COUNT)):
        raise RuntimeError("K12 model phase state order is not 0...424")
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
                    "model_id": candidate.candidate_id,
                    "model_name": candidate.candidate_name,
                    "K": candidate.K,
                    "added_basis_id": candidate.added_basis_id or "NONE",
                    "added_order_p": candidate.added_order_p,
                    "added_delay_m": candidate.added_delay_m,
                    "added_envelope_delay_q": candidate.added_envelope_delay_q,
                    "added_family": candidate.added_family or "BASELINE",
                    "is_cross_delay": bool(candidate.is_cross_delay) if candidate.is_cross_delay is not None else False,
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
    frame = pd.DataFrame(rows).sort_values(["model_id", "State_R"]).reset_index(drop=True)
    if frame.shape[0] != len(candidates) * STATE_COUNT:
        raise RuntimeError(f"K12 model quality table shape failed: {frame.shape}")
    frame.to_csv(model_path, index=False)
    support_indices = np.full((len(candidates), backend.K12), -1, dtype=np.int64)
    support_mask = np.zeros((len(candidates), backend.K12), dtype=bool)
    for candidate in candidates:
        support_indices[candidate.candidate_id, : candidate.K] = candidate.support
        support_mask[candidate.candidate_id, list(candidate.support)] = True
    np.savez_compressed(
        coefficient_path,
        model_ids=np.asarray([candidate.candidate_id for candidate in candidates], dtype=np.int64),
        model_names=np.asarray([candidate.candidate_name for candidate in candidates]),
        added_basis_ids=np.asarray([candidate.added_basis_id or "NONE" for candidate in candidates]),
        theta_Aend_padded=theta_a,
        theta_C2_padded=theta_c,
        theta_support_mask=support_mask,
        support_indices=support_indices,
        K=np.asarray([candidate.K for candidate in candidates], dtype=np.int64),
        dmax=np.asarray(backend.MP_DMAX),
        ridge_lambda=np.asarray(backend.RIDGE_LAMBDA),
    )
    return frame, theta_a, theta_c


def _candidate_retrieval(candidate: ForwardCandidate, theta_a: np.ndarray, theta_c: np.ndarray, common_phi: np.ndarray, real_b_distance: np.ndarray, shareability: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    frame, summary, distance, lut, query = k9_utils._candidate_retrieval(candidate, theta_a, theta_c, common_phi, real_b_distance, shareability)
    frame["model_id"] = candidate.candidate_id
    frame["model_name"] = candidate.candidate_name
    frame["added_basis_id"] = candidate.added_basis_id or "NONE"
    frame["added_order_p"] = candidate.added_order_p
    frame["added_delay_m"] = candidate.added_delay_m
    frame["added_envelope_delay_q"] = candidate.added_envelope_delay_q
    frame["added_family"] = candidate.added_family or "BASELINE"
    frame["is_cross_delay"] = bool(candidate.is_cross_delay) if candidate.is_cross_delay is not None else False
    frame["retrieved_real_B_CNMSE_dB"] = frame["retrieved_realB_CNMSE_dB"]
    summary["model_id"] = candidate.candidate_id
    summary["model_name"] = candidate.candidate_name
    summary["added_basis_id"] = candidate.added_basis_id or "NONE"
    summary["added_order_p"] = candidate.added_order_p
    summary["added_delay_m"] = candidate.added_delay_m
    summary["added_envelope_delay_q"] = candidate.added_envelope_delay_q
    summary["added_family"] = candidate.added_family or "BASELINE"
    summary["is_cross_delay"] = bool(candidate.is_cross_delay) if candidate.is_cross_delay is not None else False
    summary["top1_pass_count"] = summary["top1_realB_pass_count"]
    summary["top1_pass_rate"] = summary["top1_realB_pass_rate"]
    for quantile, key in ((0.05, "top12_margin_Q05"), (0.10, "top12_margin_Q10")):
        values = frame["top12_margin_dB"].to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        summary[key] = float(np.quantile(finite, quantile)) if finite.size else float("nan")
    return frame, summary, distance, lut, query


def _baseline_regression(current_model: pd.DataFrame, current_frame: pd.DataFrame, theta_a: np.ndarray, theta_c: np.ndarray, current_distance: np.ndarray, current_lut: np.ndarray, current_query_fp: np.ndarray, references: dict[str, Any]) -> dict[str, Any]:
    ref_model = references["k11_model"].loc[references["k11_model"]["model_id"].eq(21)].sort_values("State_R").reset_index(drop=True)
    cur_model = current_model.loc[current_model["model_id"].eq(0)].sort_values("State_R").reset_index(drop=True)
    metric_columns = ["Y_Aend_train_NMSE_dB", "Y_Aend_B_NMSE_dB", "Y_C2_train_NMSE_dB", "Y_C2_B_NMSE_dB"]
    metric_errors = {column: float(np.max(np.abs(cur_model[column].to_numpy(dtype=float) - ref_model[column].to_numpy(dtype=float)))) for column in metric_columns}
    ref_index = int(np.flatnonzero(references["k11_candidate_ids"] == 21)[0])
    theta_a_error = float(np.max(np.abs(theta_a[0, :, :11] - references["k11_theta_a"][ref_index])))
    theta_c_error = float(np.max(np.abs(theta_c[0, :, :11] - references["k11_theta_c"][ref_index])))
    lut_error = float(np.max(np.abs(current_lut - references["k11_lut"])))
    query_error = float(np.max(np.abs(current_query_fp - references["k11_query_fp"])))
    distance_error = float(np.max(np.abs(current_distance - references["k11_distance"])))
    ref_query = references["baseline_query"]
    state_q_equal = bool(np.array_equal(current_frame["State_Q"].to_numpy(dtype=int), ref_query["State_Q"].to_numpy(dtype=int)))
    real_b_error, real_b_mismatch = _max_abs_with_neginf(current_frame["retrieved_realB_CNMSE_dB"].to_numpy(dtype=float), ref_query["retrieved_realB_CNMSE_dB"].to_numpy(dtype=float))
    fp_top1_error = float(np.max(np.abs(current_frame["fingerprint_top1_CNMSE_dB"].to_numpy(dtype=float) - ref_query["fingerprint_top1_CNMSE_dB"].to_numpy(dtype=float))))
    margin_error = float(np.max(np.abs(current_frame["shareability_margin_dB"].to_numpy(dtype=float) - ref_query["shareability_margin_dB"].to_numpy(dtype=float))))
    first_rank_equal = bool(np.array_equal(current_frame["first_shareable_rank"].to_numpy(dtype=int), ref_query["first_shareable_rank"].to_numpy(dtype=int)))
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
        "fingerprint_top1_max_abs_error_dB": fp_top1_error,
        "shareability_margin_max_abs_error_dB": margin_error,
        "first_shareable_rank_equal": first_rank_equal,
        "state_q_equal": state_q_equal,
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
        and fp_top1_error < 1e-12
        and margin_error < 1e-12
        and first_rank_equal
        and state_q_equal
        and result["exact_count"] == 218
        and result["real_B_pass_count"] == 416
        and result["failure_count"] == 9
        and result["nonself_count"] == 207
        and result["nonself_pass_count"] == 198
    )
    if not result["pass"]:
        raise RuntimeError(f"HARD FAIL: K11 baseline regression failed: {result}")
    return result


def _local_cv_results(candidates: tuple[ForwardCandidate, ...], frames: dict[int, pd.DataFrame], folds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for fold in range(5):
        validation_ids = folds.loc[folds["fold"].eq(fold), "state_id"].to_numpy(dtype=int)
        train_ids = folds.loc[~folds["fold"].eq(fold), "state_id"].to_numpy(dtype=int)
        train_summaries = {candidate.candidate_id: k9_utils._summarize_query_frame(frames[candidate.candidate_id].loc[frames[candidate.candidate_id]["State_R"].isin(train_ids)], candidate) for candidate in candidates}
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
                    "model_id": candidate.candidate_id,
                    "model_name": candidate.candidate_name,
                    "added_basis_id": candidate.added_basis_id or "NONE",
                    "added_order_p": candidate.added_order_p,
                    "added_delay_m": candidate.added_delay_m,
                    "added_envelope_delay_q": candidate.added_envelope_delay_q,
                    "is_cross_delay": bool(candidate.is_cross_delay) if candidate.is_cross_delay is not None else False,
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
    cv = pd.DataFrame(rows).sort_values(["fold", "model_id"]).reset_index(drop=True)
    stability_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        winners = cv.loc[cv["model_id"].eq(candidate.candidate_id) & cv["train_is_winner"]]
        stability_rows.append(
            {
                "model_id": candidate.candidate_id,
                "model_name": candidate.candidate_name,
                "added_basis_id": candidate.added_basis_id or "NONE",
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
    return cv, pd.DataFrame(stability_rows).sort_values("model_id").reset_index(drop=True)


def _build_deltas(candidates: tuple[ForwardCandidate, ...], summary: pd.DataFrame, model_frame: pd.DataFrame, frames: dict[int, pd.DataFrame], hard_pair_corrections: dict[int, int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline = summary.loc[summary["candidate_id"].eq(0)].iloc[0]
    base_model = model_frame.loc[model_frame["model_id"].eq(0)]
    metric_columns = ["Y_Aend_train_NMSE_dB", "Y_Aend_B_NMSE_dB", "Y_C2_train_NMSE_dB", "Y_C2_B_NMSE_dB"]
    base_medians = {column: float(base_model[column].median()) for column in metric_columns}
    base_frame = frames[0].set_index("State_R")
    delta_rows: list[dict[str, Any]] = []
    transition_rows: list[dict[str, Any]] = []
    for candidate in candidates[1:]:
        current = summary.loc[summary["candidate_id"].eq(candidate.candidate_id)].iloc[0]
        current_model = model_frame.loc[model_frame["model_id"].eq(candidate.candidate_id)]
        current_medians = {column: float(current_model[column].median()) for column in metric_columns}
        current_frame = frames[candidate.candidate_id].set_index("State_R")
        base_pass = base_frame["realB_pass"].to_numpy(dtype=bool)
        current_pass = current_frame["realB_pass"].to_numpy(dtype=bool)
        recovered = int((~base_pass & current_pass).sum())
        regressed = int((base_pass & ~current_pass).sum())
        for state_id in range(STATE_COUNT):
            base_state_pass = bool(base_frame.loc[state_id, "realB_pass"])
            current_state_pass = bool(current_frame.loc[state_id, "realB_pass"])
            transition = "recovered" if not base_state_pass and current_state_pass else "regressed" if base_state_pass and not current_state_pass else "unchanged_pass" if base_state_pass else "unchanged_fail"
            transition_rows.append(
                {
                    "model_id": candidate.candidate_id,
                    "model_name": candidate.candidate_name,
                    "added_basis_id": candidate.added_basis_id,
                    "State_R": state_id,
                    "baseline_State_Q": int(base_frame.loc[state_id, "State_Q"]),
                    "candidate_State_Q": int(current_frame.loc[state_id, "State_Q"]),
                    "baseline_real_B_CNMSE_dB": float(base_frame.loc[state_id, "retrieved_realB_CNMSE_dB"]),
                    "candidate_real_B_CNMSE_dB": float(current_frame.loc[state_id, "retrieved_realB_CNMSE_dB"]),
                    "baseline_pass": base_state_pass,
                    "candidate_pass": current_state_pass,
                    "transition": transition,
                }
            )
        delta_rows.append(
            {
                "added_basis_id": candidate.added_basis_id,
                "p": candidate.added_order_p,
                "m": candidate.added_delay_m,
                "q": candidate.added_envelope_delay_q,
                "is_cross_delay": candidate.is_cross_delay,
                "baseline_top1": int(baseline["top1_realB_pass_count"]),
                "candidate_top1": int(current["top1_realB_pass_count"]),
                "delta_top1": int(current["top1_realB_pass_count"] - baseline["top1_realB_pass_count"]),
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
                "baseline_Top2": int(baseline["Top2_oracle"]),
                "candidate_Top2": int(current["Top2_oracle"]),
                "delta_Top2": int(current["Top2_oracle"] - baseline["Top2_oracle"]),
                "baseline_Top3": int(baseline["Top3_oracle"]),
                "candidate_Top3": int(current["Top3_oracle"]),
                "delta_Top3": int(current["Top3_oracle"] - baseline["Top3_oracle"]),
                "hard_pair_corrected_count": hard_pair_corrections[candidate.candidate_id],
                "delta_Aend_train_median": current_medians["Y_Aend_train_NMSE_dB"] - base_medians["Y_Aend_train_NMSE_dB"],
                "delta_Aend_B_median": current_medians["Y_Aend_B_NMSE_dB"] - base_medians["Y_Aend_B_NMSE_dB"],
                "delta_C2_train_median": current_medians["Y_C2_train_NMSE_dB"] - base_medians["Y_C2_train_NMSE_dB"],
                "delta_C2_B_median": current_medians["Y_C2_B_NMSE_dB"] - base_medians["Y_C2_B_NMSE_dB"],
            }
        )
    return pd.DataFrame(delta_rows), pd.DataFrame(transition_rows), pd.DataFrame()


def _failure9_detail(candidates: tuple[ForwardCandidate, ...], frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    baseline = frames[0].set_index("State_R")
    failure_ids = baseline.loc[~baseline["realB_pass"], :].index.astype(int).tolist()
    if len(failure_ids) != 9:
        raise RuntimeError("K11 failure set no longer contains 9 states")
    rows: list[dict[str, Any]] = []
    for candidate in candidates[1:]:
        current = frames[candidate.candidate_id].set_index("State_R")
        for state_id in failure_ids:
            rows.append(
                {
                    "model_id": candidate.candidate_id,
                    "model_name": candidate.candidate_name,
                    "added_basis_id": candidate.added_basis_id,
                    "State_R": state_id,
                    "K11_State_Q": int(baseline.loc[state_id, "State_Q"]),
                    "K12_State_Q": int(current.loc[state_id, "State_Q"]),
                    "K11_realB_CNMSE_dB": float(baseline.loc[state_id, "retrieved_realB_CNMSE_dB"]),
                    "K12_realB_CNMSE_dB": float(current.loc[state_id, "retrieved_realB_CNMSE_dB"]),
                    "K11_pass": bool(baseline.loc[state_id, "realB_pass"]),
                    "K12_pass": bool(current.loc[state_id, "realB_pass"]),
                    "recovered": bool(current.loc[state_id, "realB_pass"]),
                    "K11_first_shareable_rank": int(baseline.loc[state_id, "first_shareable_rank"]),
                    "K12_first_shareable_rank": int(current.loc[state_id, "first_shareable_rank"]),
                    "K11_shareability_margin_dB": float(baseline.loc[state_id, "shareability_margin_dB"]),
                    "K12_shareability_margin_dB": float(current.loc[state_id, "shareability_margin_dB"]),
                    "rank_improved": bool(current.loc[state_id, "first_shareable_rank"] < baseline.loc[state_id, "first_shareable_rank"]),
                }
            )
    return pd.DataFrame(rows)


def _hard_pair_correction(candidates: tuple[ForwardCandidate, ...], hard_pairs: pd.DataFrame, distance_by_id: dict[int, np.ndarray]) -> tuple[pd.DataFrame, dict[int, int]]:
    rows: list[dict[str, Any]] = []
    counts: dict[int, int] = {candidate.candidate_id: 0 for candidate in candidates}
    for candidate in candidates[1:]:
        distance = distance_by_id[candidate.candidate_id]
        for pair in hard_pairs.itertuples(index=False):
            state_id = int(pair.State_R)
            wrong = int(pair.Top1_wrong_State_Q)
            shareable = int(pair.Top2_shareable_State_Q)
            before_wrong = float(distance_by_id[0][state_id, wrong])
            before_shareable = float(distance_by_id[0][state_id, shareable])
            after_wrong = float(distance[state_id, wrong])
            after_shareable = float(distance[state_id, shareable])
            corrected = bool(after_shareable < after_wrong)
            counts[candidate.candidate_id] += int(corrected)
            rows.append(
                {
                    "model_id": candidate.candidate_id,
                    "model_name": candidate.candidate_name,
                    "added_basis_id": candidate.added_basis_id,
                    "State_R": state_id,
                    "wrong_Top1_State_Q": wrong,
                    "shareable_Top2_State_Q": shareable,
                    "hard_pair_margin_before": before_wrong - before_shareable,
                    "hard_pair_margin_after": after_wrong - after_shareable,
                    "wrong_distance_after": after_wrong,
                    "shareable_distance_after": after_shareable,
                    "hard_pair_corrected": corrected,
                }
            )
    return pd.DataFrame(rows), counts


def _write_figures(summary: pd.DataFrame, delta: pd.DataFrame, failure9: pd.DataFrame) -> None:
    ordered = summary.sort_values("candidate_id")
    labels = ordered["candidate_name"].tolist()
    values = ordered["top1_pass_count"].to_numpy(dtype=float)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(32, 9), dpi=300)
    bars = ax.bar(x, values, color=plt.get_cmap("tab20")(np.linspace(0, 1, len(labels))))
    ax.axhline(416, color="black", linestyle="--", linewidth=1.0, label="K11 baseline = 416")
    for bar, value in zip(bars, values, strict=True):
        if value > 416:
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.5, str(int(value)), ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x, labels, rotation=90, fontsize=5)
    ax.set_ylabel("Top-1 Real-B pass count")
    ax.set_xlabel("K11 baseline and one added Envelope75 basis")
    ax.set_ylim(0, 425)
    ax.set_title("K12 single-basis forward Top-1 Real-B comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "21_candidate_top1_comparison.png")
    plt.close(fig)

    d = delta.sort_values("added_basis_id")
    dvalues = d["delta_top1"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(32, 8), dpi=300)
    colors = ["tab:green" if value > 0 else "tab:red" if value < 0 else "tab:gray" for value in dvalues]
    bars = ax.bar(np.arange(len(d)), dvalues, color=colors)
    ax.axhline(0, color="black", linewidth=0.9)
    for bar, value in zip(bars, dvalues, strict=True):
        if value != 0:
            ax.text(bar.get_x() + bar.get_width() / 2, value + (0.25 if value > 0 else -0.4), f"{value:+.0f}", ha="center", va="bottom" if value > 0 else "top", fontsize=6)
    ax.set_xticks(np.arange(len(d)), d["added_basis_id"], rotation=90, fontsize=5)
    ax.set_ylabel("Δ Top-1 Real-B pass count vs K11")
    ax.set_xlabel("Added basis")
    ax.set_ylim(float(dvalues.min()) - 6, max(6, float(dvalues.max()) + 6))
    ax.set_title("Forward retrieval contribution of each added basis")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "22_forward_basis_contribution.png")
    plt.close(fig)

    grouped = failure9.groupby(["model_id", "model_name", "added_basis_id"], as_index=False).agg(recovered=("recovered", "sum"))
    regressed = delta[["added_basis_id", "regressed_count"]].rename(columns={"regressed_count": "regressed"})
    grouped = grouped.merge(regressed, on="added_basis_id", how="left")
    grouped = grouped.sort_values("recovered", ascending=False)
    fig, ax = plt.subplots(figsize=(30, 8), dpi=300)
    xx = np.arange(grouped.shape[0])
    width = 0.38
    ax.bar(xx - width / 2, grouped["recovered"], width, label="K11 failure-9 recovered", color="tab:green")
    ax.bar(xx + width / 2, grouped["regressed"], width, label="K11 success regressed", color="tab:red")
    ax.set_xticks(xx, grouped["added_basis_id"], rotation=90, fontsize=5)
    ax.set_ylabel("State count")
    ax.set_xlabel("Added basis")
    ax.set_title("K12 impact on K11 failures and successes")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "23_failure9_recovery_comparison.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    ax.scatter(delta["delta_C2_train_median"], delta["delta_top1"], c=delta["is_cross_delay"].astype(int), cmap="tab10", s=40, alpha=0.85)
    highlights = delta["delta_top1"].abs().sort_values(ascending=False).head(10).index
    for row in delta.loc[highlights].itertuples(index=False):
        ax.annotate(str(row.added_basis_id), (row.delta_C2_train_median, row.delta_top1), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Δ C2 Train median NMSE (dB)")
    ax.set_ylabel("Δ Top-1 Real-B pass count")
    ax.set_title("K12 modeling contribution versus retrieval contribution")
    ax.grid(alpha=0.25)
    ax.text(0.99, 0.02, "Labels show the largest |ΔTop1| candidates; full IDs are in 12_forward_addition_delta.csv.", transform=ax.transAxes, ha="right", va="bottom", fontsize=7)
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "24_modeling_vs_retrieval_contribution.png")
    plt.close(fig)


def _run(*, resume: bool = False) -> dict[str, Any]:
    if RESULT_ROOT.exists() and any(RESULT_ROOT.iterdir()) and not resume:
        raise RuntimeError(f"result directory is not empty; use --resume: {RESULT_ROOT}")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "distance_matrices").mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "fingerprints").mkdir(parents=True, exist_ok=True)
    raw_before = raw_manifest_gate()
    support_ids, envelope_by_id, added_index, formal_support_frame = _load_frozen_k11()
    envelope_terms = tuple(build_envelope_dictionary())
    envelope_gate = dictionary_gate()
    references = _load_references()
    common_b, common_meta = verify_common_b_contract()
    common_b = np.asarray(common_b, dtype=np.complex128)
    if common_meta["sha256"] != EXPECTED_COMMON_B_SHA:
        raise RuntimeError("common-B hash changed")
    real_b_random_error = _random_real_b_check(references["real_b_distance"])
    profile, hard_pairs = _failure_rank_profile(references["k11_distance"], references["shareability"], references["real_b_distance"])
    rank_distribution = {
        "rank_1": int((profile["first_shareable_rank"] == 1).sum()),
        "rank_2": int((profile["first_shareable_rank"] == 2).sum()),
        "rank_3_to_5": int(profile["first_shareable_rank"].between(3, 5).sum()),
        "rank_6_to_10": int(profile["first_shareable_rank"].between(6, 10).sum()),
        "rank_gt_10": int((profile["first_shareable_rank"] > 10).sum()),
    }
    _, candidates = _build_candidates(support_ids, envelope_by_id)
    if len(candidates) != 65:
        raise RuntimeError("K12 candidate count is not 65")
    references["folds"].to_csv(RESULT_ROOT / "16_query_state_cv_folds.csv", index=False)
    history = {
        "orders": list(backend.historical_mp10.MP_ORDERS),
        "memory": [backend.historical_mp10.MP_MEMORY[order] for order in backend.historical_mp10.MP_ORDERS],
        "K11": 11,
        "K12": 12,
        "dmax": backend.MP_DMAX,
        "ridge_lambda": backend.RIDGE_LAMBDA,
    }
    baseline_reference_gate = {
        "K11_expected_top1": 416,
        "K11_expected_failure": 9,
        "K11_expected_exact": 218,
        "K11_failure_states": profile.loc[~profile["Top1_shareable"], "State_R"].astype(int).tolist(),
        "rank_distribution": rank_distribution,
        "hard_pair_count": int(hard_pairs.shape[0]),
        "common_B_sha256": EXPECTED_COMMON_B_SHA,
        "real_B_ground_truth_reused": True,
    }
    _write_json(RESULT_ROOT / "01_reuse_audit.json", {
        "task_name": TASK_NAME,
        "frozen_k11_support_source": str(K11_ROOT / "04_candidate_supports.csv"),
        "frozen_k11_support_ids": list(support_ids),
        "remaining_envelope75_count": 64,
        "envelope75_dictionary_gate": envelope_gate,
        "historical_mp10_backend": "behavior_modeling.sparse_gmp.build_frozen_mp_basis + fit_ridge",
        "historical_contract": history,
        "separate_hnorm_found": False,
        "real_B_ground_truth_source": str(K9_ROOT / "04_realB_ground_truth_cnmse_matrix.npy"),
        "cv_fold_source": str(K11_ROOT / "13_query_state_cv_folds.csv"),
        "retrieval_helpers_reused": ["CNMSE", "Top1 allow-self", "shareability margin", "MRR", "Top-k oracle", "Query-State CV"],
        "baseline_reference_gate": baseline_reference_gate,
    })
    (RESULT_ROOT / "01_reuse_audit.txt").write_text(
        "\n".join([
            f"Task: {TASK_NAME}",
            f"Frozen K11 parent: {BEST_K11_NAME}",
            f"Support: {list(support_ids)}",
            "Support was read from the formal K11 candidate support metadata.",
            "K12 candidates are the canonical Envelope75 set difference: Envelope75 minus the frozen K11 support.",
            f"Remaining count: {len(candidates) - 1}.",
            f"Historical MP10 contract: {json.dumps(history, ensure_ascii=False, sort_keys=True)}",
            "No separate Hnorm exists in the current checkout and none was added.",
            "All K12 use dmax=2 and historical augmented complex Ridge lambda=1e-8.",
            "Real-B matrix/mask and Query-State folds are reused from the preceding K9/K11 work.",
            f"K11 failure rank distribution: {rank_distribution}; hard-pair count={len(hard_pairs)}.",
            "No K13, K12 backward, multi-basis addition, swap, beam search, lambda/order/memory/dmax scan, Top-k selection, ensemble, clustering, DPD replay or low-bandwidth task was run.",
        ]) + "\n", encoding="utf-8"
    )
    (RESULT_ROOT / "00_task_definition.txt").write_text(
        "\n".join([
            f"Task: {TASK_NAME}",
            "Mode: Frozen best K11 plus one remaining Envelope75 basis forward scan.",
            f"Parent support: {list(support_ids)}; K11={BEST_K11_NAME}.",
            "Models: K11_full plus 64 K12 candidates.",
            "All candidates use dmax=2, lambda=1e-8, frozen ABC/common-B/Real-B/allow-self protocol.",
            "Primary ranking: Top-1 Real-B pass, non-self pass rate, Q05 margin, MRR, Top-3 oracle, then K.",
            "CV is Query-State stability analysis, not an independent final test.",
            "No K13 or additional optimization is performed.",
        ]) + "\n", encoding="utf-8"
    )
    _checkpoint(phase="reuse_audit_k11_baseline_reference_and_candidate_pool_passed", raw_manifest_before=raw_before, remaining_candidate_count=64, k11_failure_rank_distribution=rank_distribution, hard_pair_count=int(hard_pairs.shape[0]), real_B_random_pair_max_abs_error_dB=real_b_random_error)

    model_frame, theta_a, theta_c = _run_model_phase(candidates, added_index, resume)
    mp_common = backend.historical_mp10.build_frozen_mp_basis(common_b)
    env_common = build_envelope_bank(common_b, envelope_terms)
    k11_common = np.column_stack((mp_common, env_common[:, added_index]))
    if k11_common.shape != (backend.FINGERPRINT_LENGTH, backend.K11):
        raise RuntimeError(f"K11 common-B Phi shape failed: {k11_common.shape}")
    candidate_frames: dict[int, pd.DataFrame] = {}
    candidate_summaries: list[dict[str, Any]] = []
    distance_by_id: dict[int, np.ndarray] = {}
    baseline_frame, baseline_summary, baseline_distance, baseline_lut, baseline_query = _candidate_retrieval(candidates[0], theta_a, theta_c, k11_common, references["real_b_distance"], references["shareability"])
    candidate_frames[0] = baseline_frame
    candidate_summaries.append(baseline_summary)
    distance_by_id[0] = baseline_distance
    np.save(RESULT_ROOT / "distance_matrices" / "K11_full.npy", baseline_distance)
    np.save(RESULT_ROOT / "fingerprints" / "K11_full_lut.npy", baseline_lut)
    np.save(RESULT_ROOT / "fingerprints" / "K11_full_query.npy", baseline_query)
    baseline = _baseline_regression(model_frame, baseline_frame, theta_a, theta_c, baseline_distance, baseline_lut, baseline_query, references)
    _write_json(RESULT_ROOT / "07_baseline_regression_check.json", {"pass": baseline["pass"], "K11_baseline": baseline, "reference_candidate": BEST_K11_NAME})
    _checkpoint(phase="k11_baseline_regression_passed", baseline_regression=baseline)
    for candidate in candidates[1:]:
        full_phi = np.column_stack((k11_common, env_common[:, candidate.envelope_index]))
        frame, summary, distance, lut, query = _candidate_retrieval(candidate, theta_a, theta_c, full_phi, references["real_b_distance"], references["shareability"])
        candidate_frames[candidate.candidate_id] = frame
        candidate_summaries.append(summary)
        distance_by_id[candidate.candidate_id] = distance
        np.save(RESULT_ROOT / "distance_matrices" / f"{candidate.candidate_name}.npy", distance)
    query_frame = pd.concat(candidate_frames.values(), ignore_index=True).sort_values(["candidate_id", "State_R"]).reset_index(drop=True)
    query_frame.to_csv(RESULT_ROOT / "11_candidate_query_metrics_long.csv", index=False)
    summary = k9_utils._assign_diagnostic_ranks(candidate_summaries)
    summary["model_id"] = summary["candidate_id"].astype(int)
    summary["model_name"] = summary["candidate_name"]
    summary["added_basis_id"] = summary["removed_basis_id"].fillna("NONE")
    summary["added_order_p"] = summary["added_order_p"].where(summary["added_order_p"].notna(), np.nan) if "added_order_p" in summary else np.nan
    summary["top1_pass_count"] = summary["top1_realB_pass_count"]
    summary["top1_pass_rate"] = summary["top1_realB_pass_rate"]
    summary["delta_top1_vs_k11"] = summary["top1_pass_count"] - int(summary.loc[summary["candidate_id"].eq(0), "top1_pass_count"].iloc[0])
    summary["all425_rank"] = summary["diagnostic_rank"]
    hard_pair_corrections: dict[int, int] = {0: 0}
    hard_correction_frame, hard_pair_corrections = _hard_pair_correction(candidates, hard_pairs, distance_by_id)
    hard_correction_frame.to_csv(RESULT_ROOT / "15_hard_pair_correction_long.csv", index=False)
    delta, recovered_frame, _ = _build_deltas(candidates, summary, model_frame, candidate_frames, hard_pair_corrections)
    for row in delta.itertuples(index=False):
        mask = summary["candidate_id"].eq(next(candidate.candidate_id for candidate in candidates if candidate.added_basis_id == row.added_basis_id))
        summary.loc[mask, "recovered_count"] = int(row.recovered_count)
        summary.loc[mask, "regressed_count"] = int(row.regressed_count)
        summary.loc[mask, "hard_pair_corrected_count"] = int(row.hard_pair_corrected_count)
    summary["recovered_count"] = summary["recovered_count"].fillna(0).astype(int)
    summary["regressed_count"] = summary["regressed_count"].fillna(0).astype(int)
    summary["hard_pair_corrected_count"] = summary["hard_pair_corrected_count"].fillna(0).astype(int)
    summary.to_csv(RESULT_ROOT / "10_candidate_retrieval_summary.csv", index=False)
    delta.to_csv(RESULT_ROOT / "12_forward_addition_delta.csv", index=False)
    recovered_frame.to_csv(RESULT_ROOT / "13_recovered_regressed_states.csv", index=False)
    failure9 = _failure9_detail(candidates, candidate_frames)
    failure9.to_csv(RESULT_ROOT / "14_failure9_detailed_analysis.csv", index=False)
    cv_frame, stability = _local_cv_results(candidates, candidate_frames, references["folds"])
    cv_frame.to_csv(RESULT_ROOT / "17_query_state_cv_results.csv", index=False)
    stability.to_csv(RESULT_ROOT / "18_query_state_cv_selection_stability.csv", index=False)
    basis_type = summary.loc[summary["candidate_id"].ne(0)].groupby(["added_family", "is_cross_delay"], dropna=False).agg(candidate_count=("model_id", "count"), top1_mean=("top1_pass_count", "mean"), top1_max=("top1_pass_count", "max"), delta_top1_mean=("delta_top1_vs_k11", "mean"), positive_delta_count=("delta_top1_vs_k11", lambda values: int((values > 0).sum())), margin_mean=("share_margin_Q05", "mean"), MRR_mean=("MRR", "mean")).reset_index()
    basis_type.to_csv(RESULT_ROOT / "19_basis_type_summary.csv", index=False)
    basis_structure = summary.loc[summary["candidate_id"].ne(0), ["added_basis_id", "added_order_p", "added_delay_m", "added_envelope_delay_q", "is_cross_delay", "top1_pass_count", "delta_top1_vs_k11", "recovered_count", "regressed_count", "share_margin_Q05", "MRR", "hard_pair_corrected_count"]].copy()
    basis_structure.to_csv(RESULT_ROOT / "20_basis_structure_summary.csv", index=False)
    winner = summary.sort_values("all425_rank").iloc[0]
    winner_id = int(winner["candidate_id"])
    winner_candidate = candidates[winner_id]
    winner_frame = candidate_frames[winner_id]
    winner_frame.to_csv(RESULT_ROOT / "25_diagnostic_best_k12_detail.csv", index=False)
    if winner_candidate.envelope_index is None:
        winner_phi = k11_common
    else:
        winner_phi = np.column_stack((k11_common, env_common[:, winner_candidate.envelope_index]))
    np.save(RESULT_ROOT / "fingerprints" / "diagnostic_best_k12_lut.npy", (winner_phi @ theta_a[winner_id, :, : winner_candidate.K].T).T.astype(np.complex128))
    np.save(RESULT_ROOT / "fingerprints" / "diagnostic_best_k12_query.npy", (winner_phi @ theta_c[winner_id, :, : winner_candidate.K].T).T.astype(np.complex128))
    _write_figures(summary, delta, failure9)
    raw_after = raw_manifest_gate()
    if raw_before != raw_after:
        raise RuntimeError("data/raw manifest changed")
    top1_improvement = int(summary.loc[summary["candidate_id"].ne(0), "top1_pass_count"].max()) > 416
    summary_lines = [
        f"Task: {TASK_NAME}",
        f"Frozen K11 support: {list(support_ids)}; parent={BEST_K11_NAME}.",
        f"K11 baseline regression PASS={baseline['pass']}; Top1=416/425; Failure=9; Exact=218/425; Non-self=198/207; Q05=0.323862; MRR=0.987983; Top2=423; Top3=423; Top5=424; Top10=425.",
        f"Current K11 failure states: {profile.loc[~profile['Top1_shareable'], 'State_R'].astype(int).tolist()}.",
        f"Failure rank distribution: {rank_distribution}; hard-pair count={len(hard_pairs)}; Real-B random-pair max error={real_b_random_error:.3e} dB.",
        "",
        "All 64 K12 candidates ranked by the frozen dictionary order:",
    ]
    for row in summary.sort_values("all425_rank").head(10).itertuples(index=False):
        summary_lines.append(
            f"rank={int(row.all425_rank)} {row.model_name}: added={row.added_basis_id}, p/m/q={row.added_order_p}/{row.added_delay_m}/{row.added_envelope_delay_q}, cross={row.is_cross_delay}, Top1={int(row.top1_pass_count)}/425, delta={int(row.delta_top1_vs_k11)}, recovered={int(row.recovered_count)}, regressed={int(row.regressed_count)}, nonself={float(row.nonself_pass_rate):.9g}, Q05={float(row.share_margin_Q05):.9g}, MRR={float(row.MRR):.9g}, Top2={int(row.Top2_oracle)}, Top3={int(row.Top3_oracle)}, hard_corrected={int(row.hard_pair_corrected_count)}"
        )
    summary_lines.extend([
        "",
        f"All425 diagnostic best: {winner['model_name']} / {winner['added_basis_id']}; Top1={int(winner['top1_pass_count'])}/425; delta={int(winner['delta_top1_vs_k11'])}.",
        "Single-basis K12 forward improvement found." if top1_improvement else "No Top-1 improvement from any single additional basis.",
        "This is an All425 diagnostic best K12, not a final frozen model.",
        "",
        "Five-fold Query-State CV:",
    ])
    for row in stability.loc[stability["train_winner_count"].gt(0)].sort_values("train_winner_count", ascending=False).itertuples(index=False):
        summary_lines.append(
            f"{row.model_name}: train winner={int(row.train_winner_count)}/5, folds={row.winner_folds}, validation Top1 mean/min={float(row.winner_validation_top1_pass_mean):.6g}/{float(row.winner_validation_top1_pass_min):.6g}"
        )
    summary_lines.extend([
        "",
        f"raw before={json.dumps(raw_before, ensure_ascii=False, sort_keys=True)}",
        f"raw after={json.dumps(raw_after, ensure_ascii=False, sort_keys=True)}",
        f"raw unchanged={raw_before == raw_after}",
        "No K13, second forward addition, K12 backward elimination, multi-basis addition, swap, beam search, lambda/order/memory/dmax scan, Top-k reranking, ensemble/fusion, clustering, Type III, low-bandwidth or DPD replay was run.",
        "CV is a Query-State stability analysis, not an independent final test.",
    ])
    (RESULT_ROOT / "26_final_result_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    transition_counts = recovered_frame["transition"].value_counts().to_dict()
    result = {
        "task": TASK_NAME,
        "status": "SUCCESS",
        "baseline_regression_pass": bool(baseline["pass"]),
        "diagnostic_best_model": str(winner["model_name"]),
        "diagnostic_best_basis": str(winner["added_basis_id"]),
        "diagnostic_best_top1": int(winner["top1_pass_count"]),
        "baseline_top1": 416,
        "max_k12_top1": int(summary.loc[summary["candidate_id"].ne(0), "top1_pass_count"].max()),
        "positive_k12_delta_count": int((summary.loc[summary["candidate_id"].ne(0), "delta_top1_vs_k11"] > 0).sum()),
        "k11_failure_rank_distribution": rank_distribution,
        "hard_pair_count": int(hard_pairs.shape[0]),
        "transition_counts_across_k12": {key: int(transition_counts.get(key, 0)) for key in ("recovered", "regressed", "unchanged_pass", "unchanged_fail")},
        "cv_winner_counts": stability.loc[stability["train_winner_count"].gt(0), ["model_name", "train_winner_count"]].to_dict("records"),
        "raw_data_modified": raw_before != raw_after,
        "result_root": str(RESULT_ROOT),
    }
    _write_json(RESULT_ROOT / "27_checkpoint.json", {"phase": "completed", **result})
    _append_log(
        f"\n[{_now()}] Complete {TASK_NAME}\nResult={json.dumps(result, ensure_ascii=False, sort_keys=True)}\n"
        "No K13 or additional optimization was run.\n"
    )
    _append_handoff(
        f"完成当前最佳 K11 的 64 个单项 K12 forward candidates；结果目录：{RESULT_ROOT}\n"
        f"摘要：{json.dumps(result, ensure_ascii=False, sort_keys=True)}\n"
        "完成 failure rank/hard-pair、Full425 retrieval、K12 delta、5-fold Query-State CV、类型与结构分析；未冻结新模型。"
    )
    return result


def main() -> None:
    result = _run(resume="--resume" in sys.argv[1:])
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    __import__("multiprocessing").freeze_support()
    main()
