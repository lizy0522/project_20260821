# ruff: noqa: E402,E501,I001

"""Frozen MP10 plus one canonical Envelope75 basis forward scan."""

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
from behavior_modeling.shared.config import BASIS_TERMS, MP_CONFIG  # noqa: E402
from behavior_fingerprint_retrieval.shared.envelope19_c2endshared_commonB_full_lut import (  # noqa: E402
    EXPECTED_COMMON_B_SHA,
    compute_cnmse_matrix,
    raw_manifest_gate,
    verify_common_b_contract,
)
from data_management.shared import load_by_id  # noqa: E402
from signal_segmentation.shared import (  # noqa: E402
    build_partition_from_xin,
    get_ilc_pair,
    preprocess_full_pair,
)
from retrieval_oriented_model_selection.self_hit_oriented.scenario_2.scenario_2_mp10_k11_retrieval_oriented_forward_scan_5b import (  # noqa: E402
    mp10_k11_backend as backend,
)
from retrieval_oriented_model_selection.shared import mp10_k9_retrieval_runner as k9_utils

TASK_NAME = "scenario_2_mp10_k11_retrieval_oriented_forward_scan_5b"
RESULT_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /TASK_NAME
WORK_LOG = PROJECT_ROOT / "work_logs" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /TASK_NAME / "execution_log.txt"
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"
PREVIOUS_MP10_ROOT = PROJECT_ROOT / "results" / "behavior_fingerprint_retrieval" / "scenario_2_mp10_full_lut_retrieval_5b"
PREVIOUS_K9_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /"scenario_2_mp10_k9_retrieval_oriented_basis_ablation_5b"
STATE_COUNT = backend.STATE_COUNT
WORKER_COUNT = 10
BLAS_THREADS_PER_WORKER = 1
REAL_B_THRESHOLD_DB = -40.0
REAL_B_SANITY_TOLERANCE_DB = 1e-9
MP10_K9_FOLD_FILE = PREVIOUS_K9_ROOT / "12_query_state_cv_folds.csv"
PREVIOUS_REAL_B_MATRIX = PREVIOUS_K9_ROOT / "04_realB_ground_truth_cnmse_matrix.npy"
PREVIOUS_SHAREABILITY = PREVIOUS_K9_ROOT / "05_realB_shareability_mask.npy"


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
        # K9 helper functions use this field name; the forward scan reports it as added_basis_id.
        return self.added_basis_id

    @property
    def removed_baseline_column(self) -> int | None:
        return None


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
        RESULT_ROOT / "22_checkpoint.json",
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
            "candidate_count": 66,
            "remaining_envelope75_count": 65,
            "baseline_candidate": "MP10_full",
            "dmax_fixed": backend.MP_DMAX,
            "ridge_lambda_fixed": backend.RIDGE_LAMBDA,
            "envelope75_multi_addition_performed": False,
            "k12_or_higher_performed": False,
            "lambda_scan_performed": False,
            "memory_scan_performed": False,
            "order_scan_performed": False,
            "top_k_selection_performed": False,
            "real_B_used_for_top1_selection": False,
            **payload,
        },
    )


def _basis_id(term: EnvelopeBasis) -> str:
    return str(term.basis_id)


def _support_hash(ids: tuple[str, ...]) -> str:
    return hashlib.sha256(json.dumps(list(ids), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _mp10_basis_ids() -> tuple[str, ...]:
    terms = tuple((int(order), int(delay)) for order, delay in BASIS_TERMS)
    if tuple(MP_CONFIG["orders"]) != tuple(backend.historical_mp10.MP_ORDERS):
        raise RuntimeError("MP10 order contract changed")
    expected_terms = tuple(
        (int(order), int(delay))
        for order in backend.historical_mp10.MP_ORDERS
        for delay in range(backend.historical_mp10.MP_MEMORY[order])
    )
    if terms != expected_terms:
        raise RuntimeError("MP10 BASIS_TERMS no longer matches historical builder")
    result: list[str] = []
    for order, delay in terms:
        if order == 1:
            result.append(f"LIN_d{delay}")
        else:
            result.append(f"ENV_p{order:02d}_m{delay}_q{delay}")
    return tuple(result)


def _build_candidates(envelope_terms: tuple[EnvelopeBasis, ...], mp10_ids: tuple[str, ...]) -> tuple[pd.DataFrame, tuple[ForwardCandidate, ...], dict[str, int]]:
    envelope_by_id = {term.basis_id: term for term in envelope_terms}
    missing = [basis_id for basis_id in mp10_ids if basis_id not in envelope_by_id]
    if missing:
        raise RuntimeError(f"MP10 basis IDs missing from Envelope75: {missing}")
    overlap_indices = {basis_id: envelope_by_id[basis_id].index for basis_id in mp10_ids}
    remaining = tuple(term for term in envelope_terms if term.basis_id not in set(mp10_ids))
    if len(envelope_terms) != 75 or len(remaining) != 65:
        raise RuntimeError(f"Envelope75 remaining count failed: total={len(envelope_terms)}, remaining={len(remaining)}")
    combined_baseline = tuple(mp10_ids)
    candidates = [
        ForwardCandidate(
            candidate_id=0,
            candidate_name="MP10_full",
            K=10,
            added_basis_id=None,
            added_order_p=None,
            added_delay_m=None,
            added_envelope_delay_q=None,
            added_family=None,
            is_cross_delay=None,
            envelope_index=None,
            support=tuple(range(10)),
            retained_basis_ids=combined_baseline,
            support_hash=_support_hash(combined_baseline),
        )
    ]
    remaining_rows: list[dict[str, Any]] = []
    support_rows: list[dict[str, Any]] = [
        {
            "model_id": 0,
            "model_name": "MP10_full",
            "K": 10,
            "added_basis_id": "NONE",
            "added_order_p": np.nan,
            "added_delay_m": np.nan,
            "added_envelope_delay_q": np.nan,
            "is_cross_delay": False,
            "support_basis_ids": json.dumps(list(combined_baseline), ensure_ascii=False),
            "support_hash": _support_hash(combined_baseline),
            "dmax": backend.MP_DMAX,
            "ridge_lambda": backend.RIDGE_LAMBDA,
        }
    ]
    for candidate_index, term in enumerate(remaining, start=1):
        is_cross = bool(term.order > 1 and term.signal_delay != term.envelope_delay)
        candidate_ids = tuple(mp10_ids) + (term.basis_id,)
        candidate = ForwardCandidate(
            candidate_id=candidate_index,
            candidate_name=f"MP10_plus_{term.basis_id}",
            K=11,
            added_basis_id=term.basis_id,
            added_order_p=int(term.order),
            added_delay_m=None if term.signal_delay is None else int(term.signal_delay),
            added_envelope_delay_q=None if term.envelope_delay is None else int(term.envelope_delay),
            added_family=term.family,
            is_cross_delay=is_cross,
            envelope_index=int(term.index),
            support=tuple(range(11)),
            retained_basis_ids=candidate_ids,
            support_hash=_support_hash(candidate_ids),
        )
        candidates.append(candidate)
        remaining_rows.append(
            {
                "candidate_index": candidate_index,
                "basis_id": term.basis_id,
                "basis_type": term.family,
                "order_p": int(term.order),
                "delay_m": term.signal_delay,
                "envelope_delay_q": term.envelope_delay,
                "is_cross_delay": is_cross,
                "envelope_dictionary_index": int(term.index),
                "formula": term.formula,
            }
        )
        support_rows.append(
            {
                "model_id": candidate_index,
                "model_name": candidate.candidate_name,
                "K": 11,
                "added_basis_id": term.basis_id,
                "added_order_p": int(term.order),
                "added_delay_m": term.signal_delay,
                "added_envelope_delay_q": term.envelope_delay,
                "is_cross_delay": is_cross,
                "support_basis_ids": json.dumps(list(candidate_ids), ensure_ascii=False),
                "support_hash": candidate.support_hash,
                "dmax": backend.MP_DMAX,
                "ridge_lambda": backend.RIDGE_LAMBDA,
            }
        )
    pd.DataFrame(remaining_rows).to_csv(RESULT_ROOT / "03_remaining_envelope75_candidates.csv", index=False)
    pd.DataFrame(support_rows).to_csv(RESULT_ROOT / "04_candidate_supports.csv", index=False)
    return pd.DataFrame(remaining_rows), tuple(candidates), overlap_indices


def _run_overlap_regression(
    state_ids: tuple[int, ...],
    common_b: np.ndarray,
    envelope_terms: tuple[EnvelopeBasis, ...],
    mp10_ids: tuple[str, ...],
    overlap_indices: dict[str, int],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for state_id in state_ids:
        data = load_by_id(state_id)
        partition = build_partition_from_xin(np.asarray(data["xin"]))
        count = np.asarray(data["xin_pd_ori_ilc"]).shape[1]
        a_pair = get_ilc_pair(data, int(count - 1))
        c_pair = get_ilc_pair(data, 1)
        aend = preprocess_full_pair(
            a_pair.input_full,
            a_pair.output_raw_full,
            partition,
            pair_type=a_pair.pair_type,
            iteration_index=a_pair.iteration_index,
            input_peak_normalization_factor=a_pair.input_peak_normalization_factor,
        )
        c2 = preprocess_full_pair(
            c_pair.input_full,
            c_pair.output_raw_full,
            partition,
            pair_type=c_pair.pair_type,
            iteration_index=c_pair.iteration_index,
            input_peak_normalization_factor=c_pair.input_peak_normalization_factor,
        )
        contexts = (
            ("Aend_A", aend["A"].input),
            ("Aend_B", aend["B"].input),
            ("C2_C", c2["C"].input),
            ("C2_B", c2["B"].input),
        )
        for context, signal in contexts:
            mp_bank = backend.historical_mp10.build_frozen_mp_basis(signal)
            env_bank = build_envelope_bank(signal, envelope_terms)
            for column, basis_id in enumerate(mp10_ids):
                envelope_index = overlap_indices[basis_id]
                delta = float(np.max(np.abs(mp_bank[:, column] - env_bank[:, envelope_index])))
                relative = delta / max(float(np.linalg.norm(mp_bank[:, column])), np.finfo(float).eps)
                rows.append(
                    {
                        "state_id": state_id,
                        "context": context,
                        "baseline_column": column,
                        "basis_id": basis_id,
                        "envelope_index": envelope_index,
                        "max_abs_delta": delta,
                        "relative_error": relative,
                        "finite": bool(np.all(np.isfinite(mp_bank[:, column])) and np.all(np.isfinite(env_bank[:, envelope_index]))),
                    }
                )
    mp_common = backend.historical_mp10.build_frozen_mp_basis(common_b)
    env_common = build_envelope_bank(common_b, envelope_terms)
    for column, basis_id in enumerate(mp10_ids):
        envelope_index = overlap_indices[basis_id]
        delta = float(np.max(np.abs(mp_common[:, column] - env_common[:, envelope_index])))
        relative = delta / max(float(np.linalg.norm(mp_common[:, column])), np.finfo(float).eps)
        rows.append(
            {
                "state_id": 0,
                "context": "common_B",
                "baseline_column": column,
                "basis_id": basis_id,
                "envelope_index": envelope_index,
                "max_abs_delta": delta,
                "relative_error": relative,
                "finite": True,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULT_ROOT / "02_mp10_envelope75_overlap_regression.csv", index=False)
    if float(frame["max_abs_delta"].max()) > 1e-12 or not frame["finite"].all():
        raise RuntimeError(
            f"MP10/Envelope75 overlap regression failed: max={frame['max_abs_delta'].max():.3e}"
        )
    return frame


def _write_audit(contract: dict[str, Any], envelope_gate: dict[str, Any], baseline_gate: dict[str, Any], common_meta: dict[str, Any]) -> None:
    lines = [
        f"Task: {TASK_NAME}",
        "Purpose: add one canonical remaining Envelope75 basis to the frozen MP10 and evaluate retrieval-oriented Full425 performance.",
        "",
        "Reused historical MP10:",
        "- behavior_modeling.sparse_gmp.build_frozen_mp_basis",
        "- behavior_modeling.basis.build_mp_basis",
        "- behavior_modeling.sparse_gmp.fit_ridge",
        "- orders=[1,2,3,5,7,9]; memory=[3,2,2,1,1,1]; K=10; dmax=2; lambda=1e-8.",
        "- get_ilc_pair peak normalization; no separate Hnorm stage exists in the current checkout.",
        "",
        "Reused canonical Envelope75:",
        "- scripts/basis_function_selection/envelope_dictionary.py",
        f"- dictionary gate: {json.dumps(envelope_gate, ensure_ascii=False, sort_keys=True)}",
        "- MP10-to-Envelope75 overlap was mapped by basis_id and checked numerically on four representative states, Aend A/B, C2 C/B and common-B.",
        "- K11 Phi is historical MP10 Phi10 followed by one canonical Envelope75 column; MP10 columns are not reordered or regenerated through a new formula.",
        "",
        "Reused retrieval components:",
        "- previous MP10 Full-LUT baseline and previous K9 CV fold assignment",
        "- current common-B verification, CNMSE matrix, Top-1 allow-self, Real-B ground truth and shareability mask",
        "- previous K9 retrieval metric definitions: margin, first-shareable rank, MRR and Top-k oracle",
        "",
        f"Baseline artifact regression: {json.dumps(baseline_gate, ensure_ascii=False, sort_keys=True)}",
        f"Common-B metadata: {json.dumps(common_meta, ensure_ascii=False, sort_keys=True, default=lambda value: str(value) if not isinstance(value, (str, int, float, bool)) else value)}",
        "No K12, multi-basis addition, swap, lambda/order/memory scan, Top-k selection, ensemble, clustering, DPD replay or low-bandwidth task was run.",
    ]
    (RESULT_ROOT / "01_reuse_audit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _load_previous_inputs() -> tuple[dict[str, Any], np.ndarray, np.ndarray, pd.DataFrame]:
    previous = k9_utils._load_previous_baseline()
    real_b_distance = np.load(PREVIOUS_REAL_B_MATRIX)
    shareability = np.load(PREVIOUS_SHAREABILITY)
    folds = pd.read_csv(MP10_K9_FOLD_FILE).sort_values("state_id").reset_index(drop=True)
    if real_b_distance.shape != (STATE_COUNT, STATE_COUNT) or shareability.shape != real_b_distance.shape:
        raise RuntimeError("previous Real-B ground truth/mask shape changed")
    if not np.all(np.isneginf(np.diag(real_b_distance))) or not shareability.diagonal().all():
        raise RuntimeError("previous Real-B diagonal/shareability contract changed")
    if folds.shape[0] != STATE_COUNT or folds["fold"].value_counts().sort_index().to_dict() != {0: 85, 1: 85, 2: 85, 3: 85, 4: 85}:
        raise RuntimeError("previous Query-State folds changed")
    if not np.array_equal(folds["state_id"].to_numpy(dtype=int), np.arange(STATE_COUNT)):
        raise RuntimeError("previous Query-State fold state order changed")
    return previous, real_b_distance, shareability, folds


def _random_real_b_check(real_b_distance: np.ndarray) -> float:
    rng = np.random.default_rng(20260916)
    pairs = rng.integers(0, STATE_COUNT, size=(20, 2))
    cache: dict[int, np.ndarray] = {}
    maximum = 0.0
    for real_id, query_id in pairs:
        for state_id in (int(real_id), int(query_id)):
            if state_id not in cache:
                data = load_by_id(state_id)
                partition = build_partition_from_xin(np.asarray(data["xin"]))
                from signal_segmentation.shared import build_off_segments

                cache[state_id] = np.asarray(build_off_segments(data, partition)["B"].output[2:], dtype=np.complex128)
        direct = cnmse(cache[int(real_id)], cache[int(query_id)])
        stored = float(real_b_distance[int(real_id), int(query_id)])
        if np.isneginf(direct) and np.isneginf(stored):
            continue
        maximum = max(maximum, abs(direct - stored))
    if maximum > REAL_B_SANITY_TOLERANCE_DB:
        raise RuntimeError(f"previous Real-B matrix random-pair check failed: {maximum:.3e} dB")
    return maximum


def _run_model_phase(candidates: tuple[ForwardCandidate, ...]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    candidate_specs = tuple((candidate.candidate_id, candidate.envelope_index) for candidate in candidates)
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=WORKER_COUNT,
        mp_context=get_context("spawn"),
        initializer=backend.worker_init,
        initargs=(candidate_specs,),
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
                print(f"[MP10 K11 MODEL] {completed}/{STATE_COUNT}", flush=True)
    results.sort(key=lambda item: int(item["State_n_R"]))
    if [int(item["State_n_R"]) for item in results] != list(range(STATE_COUNT)):
        raise RuntimeError("K11 model phase state order is not 0...424")
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    theta_a = np.zeros((len(candidates), STATE_COUNT, backend.MODEL_K_MAX), dtype=np.complex128)
    theta_c = np.zeros_like(theta_a)
    rows: list[dict[str, Any]] = []
    for item in results:
        state_id = int(item["State_n_R"])
        for fit in item["candidates"]:
            candidate_id = int(fit["candidate_id"])
            candidate = candidate_by_id[candidate_id]
            theta_a[candidate_id, state_id] = fit["theta_Aend_padded"]
            theta_c[candidate_id, state_id] = fit["theta_C2_padded"]
            rows.append(
                {
                    "model_id": candidate_id,
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
                    "Aend_rank": int(fit["Y_Aend_rank"]),
                    "Aend_rank_augmented": int(fit["Y_Aend_rank_augmented"]),
                    "C2_rank": int(fit["Y_C2_rank"]),
                    "C2_rank_augmented": int(fit["Y_C2_rank_augmented"]),
                    "Aend_finite": bool(fit["finite"]),
                    "C2_finite": bool(fit["finite"]),
                }
            )
    frame = pd.DataFrame(rows).sort_values(["model_id", "State_R"]).reset_index(drop=True)
    if frame.shape[0] != len(candidates) * STATE_COUNT:
        raise RuntimeError(f"K11 model quality long table shape failed: {frame.shape}")
    return frame, theta_a, theta_c


def _candidate_retrieval(
    candidate: ForwardCandidate,
    theta_a: np.ndarray,
    theta_c: np.ndarray,
    mp_common: np.ndarray,
    envelope_common: np.ndarray,
    real_b_distance: np.ndarray,
    shareability: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    if candidate.envelope_index is None:
        common_phi = mp_common
    else:
        common_phi = np.column_stack((mp_common, envelope_common[:, candidate.envelope_index]))
    common_phi = np.asarray(common_phi, dtype=np.complex128)
    K = candidate.K
    lut = (common_phi @ theta_a[candidate.candidate_id, :, :K].T).T.astype(np.complex128)
    query = (common_phi @ theta_c[candidate.candidate_id, :, :K].T).T.astype(np.complex128)
    if lut.shape != (STATE_COUNT, backend.FINGERPRINT_LENGTH) or query.shape != lut.shape:
        raise RuntimeError(f"{candidate.candidate_name} fingerprint shape failed")
    distance = compute_cnmse_matrix(query, lut)
    if distance.shape != (STATE_COUNT, STATE_COUNT) or np.isnan(distance).any() or np.isposinf(distance).any():
        raise RuntimeError(f"{candidate.candidate_name} distance matrix failed")
    frame, summary, _, _, _ = k9_utils._candidate_retrieval(
        candidate,
        theta_a,
        theta_c,
        common_phi,
        real_b_distance,
        shareability,
    )
    frame["added_basis_id"] = candidate.added_basis_id or "NONE"
    frame["added_order_p"] = candidate.added_order_p
    frame["added_delay_m"] = candidate.added_delay_m
    frame["added_envelope_delay_q"] = candidate.added_envelope_delay_q
    frame["added_family"] = candidate.added_family or "BASELINE"
    frame["is_cross_delay"] = bool(candidate.is_cross_delay) if candidate.is_cross_delay is not None else False
    summary["added_basis_id"] = candidate.added_basis_id or "NONE"
    summary["added_order_p"] = candidate.added_order_p
    summary["added_delay_m"] = candidate.added_delay_m
    summary["added_envelope_delay_q"] = candidate.added_envelope_delay_q
    summary["added_family"] = candidate.added_family or "BASELINE"
    summary["is_cross_delay"] = bool(candidate.is_cross_delay) if candidate.is_cross_delay is not None else False
    return frame, summary, distance, lut, query


def _max_abs(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left)
    right = np.asarray(right)
    finite = np.isfinite(left) & np.isfinite(right)
    if not np.any(finite):
        return 0.0
    return float(np.max(np.abs(left[finite] - right[finite])))


def _baseline_regression(
    model_frame: pd.DataFrame,
    theta_a: np.ndarray,
    theta_c: np.ndarray,
    baseline_lut: np.ndarray,
    baseline_query: np.ndarray,
    baseline_distance: np.ndarray,
    baseline_frame: pd.DataFrame,
    previous: dict[str, Any],
) -> dict[str, Any]:
    current = model_frame.loc[model_frame["model_id"].eq(0)].sort_values("State_R").reset_index(drop=True)
    previous_model = previous["model"]
    metric_columns = ["Y_Aend_train_NMSE_dB", "Y_Aend_B_NMSE_dB", "Y_C2_train_NMSE_dB", "Y_C2_B_NMSE_dB"]
    metric_errors = {
        column: float(np.max(np.abs(current[column].to_numpy(dtype=float) - previous_model[column].to_numpy(dtype=float))))
        for column in metric_columns
    }
    theta_a_error = float(np.max(np.abs(theta_a[0, :, :10] - previous["theta_aend"])))
    theta_c_error = float(np.max(np.abs(theta_c[0, :, :10] - previous["theta_c2"])))
    lut_error = _max_abs(baseline_lut, previous["lut"])
    query_error = _max_abs(baseline_query, previous["query"])
    distance_error = _max_abs(baseline_distance, previous["distance"])
    previous_selected = previous["retrieval"]["State_n_Q"].to_numpy(dtype=int)
    current_selected = baseline_frame["State_Q"].to_numpy(dtype=int)
    retrieved_current = baseline_frame["retrieved_realB_CNMSE_dB"].to_numpy(dtype=float)
    retrieved_previous = previous["retrieval"]["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float)
    retrieved_error = _max_abs(retrieved_current, retrieved_previous)
    result = {
        "previous_artifact_gate": previous["artifact_gate"]["pass"],
        "model_metric_max_abs_error_dB": metric_errors,
        "theta_Aend_max_abs_error": theta_a_error,
        "theta_C2_max_abs_error": theta_c_error,
        "LUT_fingerprint_max_abs_error": lut_error,
        "Query_fingerprint_max_abs_error": query_error,
        "distance_max_abs_error_dB": distance_error,
        "retrieved_Real_B_max_abs_error_dB": retrieved_error,
        "state_q_equal": bool(np.array_equal(current_selected, previous_selected)),
        "exact_count": int(baseline_frame["exact_hit"].sum()),
        "real_B_pass_count": int(baseline_frame["realB_pass"].sum()),
        "failure_count": int((~baseline_frame["realB_pass"]).sum()),
        "nonself_count": int((current_selected != np.arange(STATE_COUNT)).sum()),
        "nonself_pass_count": int(((current_selected != np.arange(STATE_COUNT)) & baseline_frame["realB_pass"].to_numpy(dtype=bool)).sum()),
    }
    result["pass"] = bool(
        result["previous_artifact_gate"]
        and max(metric_errors.values()) < 1e-10
        and theta_a_error < 1e-12
        and theta_c_error < 1e-12
        and lut_error < 1e-12
        and query_error < 1e-12
        and distance_error < 1e-12
        and retrieved_error < 1e-9
        and result["state_q_equal"]
        and result["exact_count"] == 218
        and result["real_B_pass_count"] == 411
        and result["failure_count"] == 14
        and result["nonself_count"] == 207
        and result["nonself_pass_count"] == 193
    )
    if not result["pass"]:
        raise RuntimeError(f"HARD FAIL: MP10 baseline regression failed: {result}")
    return result


def _local_cv_results(candidates: tuple[ForwardCandidate, ...], frames: dict[int, pd.DataFrame], folds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for fold in range(5):
        validation_ids = folds.loc[folds["fold"].eq(fold), "state_id"].to_numpy(dtype=int)
        train_ids = folds.loc[~folds["fold"].eq(fold), "state_id"].to_numpy(dtype=int)
        train_summaries = {
            c.candidate_id: k9_utils._summarize_query_frame(
                frames[c.candidate_id].loc[frames[c.candidate_id]["State_R"].isin(train_ids)], c
            )
            for c in candidates
        }
        ranking = sorted(train_summaries.values(), key=k9_utils._selection_key)
        winner_id = int(ranking[0]["candidate_id"])
        rank_by_id = {int(item["candidate_id"]): index for index, item in enumerate(ranking, start=1)}
        for c in candidates:
            train_summary = train_summaries[c.candidate_id]
            val_frame = frames[c.candidate_id].loc[frames[c.candidate_id]["State_R"].isin(validation_ids)]
            val_summary = k9_utils._summarize_query_frame(val_frame, c)
            rows.append(
                {
                    "fold": fold,
                    "model_id": c.candidate_id,
                    "model_name": c.candidate_name,
                    "added_basis_id": c.added_basis_id or "NONE",
                    "added_order_p": c.added_order_p,
                    "added_delay_m": c.added_delay_m,
                    "added_envelope_delay_q": c.added_envelope_delay_q,
                    "is_cross_delay": bool(c.is_cross_delay) if c.is_cross_delay is not None else False,
                    "K": c.K,
                    "train_query_count": len(train_ids),
                    "validation_query_count": len(validation_ids),
                    "train_is_winner": c.candidate_id == winner_id,
                    "train_selection_rank": rank_by_id[c.candidate_id],
                    "train_top1_pass": train_summary["top1_realB_pass_count"],
                    "train_nonself_pass_rate": train_summary["nonself_pass_rate"],
                    "train_Q05_margin": train_summary["share_margin_Q05"],
                    "train_MRR": train_summary["MRR"],
                    "train_Top3_oracle": train_summary["Top3_oracle"],
                    "validation_exact": val_summary["exact_hit_count"],
                    "validation_top1_pass": val_summary["top1_realB_pass_count"],
                    "validation_top1_pass_rate": val_summary["top1_realB_pass_rate"],
                    "validation_nonself_count": val_summary["nonself_count"],
                    "validation_nonself_pass": val_summary["nonself_pass_count"],
                    "validation_nonself_pass_rate": val_summary["nonself_pass_rate"],
                    "validation_Q05_margin": val_summary["share_margin_Q05"],
                    "validation_MRR": val_summary["MRR"],
                    "validation_Top1_oracle": val_summary["Top1_oracle"],
                    "validation_Top2_oracle": val_summary["Top2_oracle"],
                    "validation_Top3_oracle": val_summary["Top3_oracle"],
                    "validation_Top5_oracle": val_summary["Top5_oracle"],
                    "validation_Top10_oracle": val_summary["Top10_oracle"],
                }
            )
    cv_frame = pd.DataFrame(rows).sort_values(["fold", "model_id"]).reset_index(drop=True)
    stability: list[dict[str, Any]] = []
    for c in candidates:
        wins = cv_frame.loc[cv_frame["model_id"].eq(c.candidate_id) & cv_frame["train_is_winner"]]
        stability.append(
            {
                "model_id": c.candidate_id,
                "model_name": c.candidate_name,
                "added_basis_id": c.added_basis_id or "NONE",
                "added_order_p": c.added_order_p,
                "added_delay_m": c.added_delay_m,
                "added_envelope_delay_q": c.added_envelope_delay_q,
                "is_cross_delay": bool(c.is_cross_delay) if c.is_cross_delay is not None else False,
                "K": c.K,
                "train_winner_count": int(wins.shape[0]),
                "train_winner_frequency": float(wins.shape[0] / 5),
                "winner_folds": json.dumps(wins["fold"].astype(int).tolist()),
                "winner_validation_top1_pass_mean": float(wins["validation_top1_pass"].mean()) if not wins.empty else float("nan"),
                "winner_validation_top1_pass_min": float(wins["validation_top1_pass"].min()) if not wins.empty else float("nan"),
                "winner_validation_nonself_rate_mean": float(wins["validation_nonself_pass_rate"].mean()) if not wins.empty else float("nan"),
                "winner_validation_Q05_margin_mean": float(wins["validation_Q05_margin"].mean()) if not wins.empty else float("nan"),
                "winner_validation_MRR_mean": float(wins["validation_MRR"].mean()) if not wins.empty else float("nan"),
                "winner_validation_Top3_mean": float(wins["validation_Top3_oracle"].mean()) if not wins.empty else float("nan"),
            }
        )
    return cv_frame, pd.DataFrame(stability).sort_values("model_id").reset_index(drop=True)


def _write_figures(summary: pd.DataFrame, delta: pd.DataFrame, envelope_candidates: pd.DataFrame) -> None:
    ordered = summary.sort_values("model_id")
    nonbaseline = ordered.loc[ordered["model_id"].ne(0)].copy()
    labels = nonbaseline["added_basis_id"].tolist()
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(32, 9), dpi=300)
    bars = ax.bar(x, nonbaseline["top1_realB_pass_count"], color=plt.get_cmap("tab20")(np.linspace(0, 1, len(labels))))
    ax.axhline(411, color="black", linestyle="--", linewidth=1.0, label="MP10 baseline = 411")
    for bar, value in zip(bars, nonbaseline["top1_realB_pass_count"], strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, float(value) + 0.5, str(int(value)), ha="center", va="bottom", fontsize=6)
    ax.set_xticks(x, labels, rotation=90, fontsize=6)
    ax.set_ylabel("Top-1 Real-B pass count")
    ax.set_xlabel("Added Envelope75 basis")
    ax.set_ylim(0, 425)
    ax.set_title("Frozen MP10 + one Envelope75 basis: Top-1 Real-B comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "17_candidate_top1_comparison.png")
    plt.close(fig)

    delta_ordered = delta.sort_values("added_basis_id")
    fig, ax = plt.subplots(figsize=(32, 8), dpi=300)
    values = delta_ordered["delta_top1"].to_numpy(dtype=float)
    colors = ["tab:green" if value > 0 else "tab:red" if value < 0 else "tab:gray" for value in values]
    bars = ax.bar(np.arange(len(delta_ordered)), values, color=colors)
    ax.axhline(0, color="black", linewidth=0.9)
    for bar, value in zip(bars, values, strict=True):
        y = float(value) + (0.3 if value >= 0 else -0.6)
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{value:+.0f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=6)
    ax.set_xticks(np.arange(len(delta_ordered)), delta_ordered["added_basis_id"], rotation=90, fontsize=6)
    ax.set_ylabel("Δ Top-1 Real-B pass count vs MP10")
    ax.set_xlabel("Added Envelope75 basis")
    ax.set_ylim(float(values.min()) - 8.0, max(8.0, float(values.max()) + 8.0))
    ax.set_title("Forward retrieval contribution of each added basis")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "18_forward_basis_contribution.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    ax.scatter(
        delta["delta_C2_train_median"],
        delta["delta_top1"],
        c=delta["is_cross_delay"].astype(int),
        cmap="tab10",
        s=35,
        alpha=0.85,
    )
    highlight_indices = delta["delta_top1"].abs().sort_values(ascending=False).head(12).index
    for row in delta.loc[highlight_indices].itertuples(index=False):
        ax.annotate(str(row.added_basis_id), (row.delta_C2_train_median, row.delta_top1), fontsize=5, xytext=(2, 2), textcoords="offset points")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Δ C2 Train median NMSE (dB)")
    ax.set_ylabel("Δ Top-1 Real-B pass count")
    ax.set_title("Modeling contribution versus retrieval contribution")
    ax.grid(alpha=0.25)
    ax.text(
        0.99,
        0.02,
        "Labels show the 12 largest |ΔTop1| candidates; full basis IDs are in 10_forward_addition_delta.csv.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
    )
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "19_retrieval_vs_modeling_contribution.png")
    plt.close(fig)


def _write_summary(history: dict[str, Any], baseline_gate: dict[str, Any], summary: pd.DataFrame, delta: pd.DataFrame, stability: pd.DataFrame, raw_before: dict[str, Any], raw_after: dict[str, Any], winner: pd.Series) -> None:
    lines = [
        f"Task: {TASK_NAME}",
        "Task type: Frozen MP10 plus one remaining Envelope75 basis retrieval-oriented forward scan.",
        "",
        "Baseline regression",
        f"- PASS={baseline_gate['pass']}; Exact={baseline_gate['exact_count']}/425; Top1 Real-B pass={baseline_gate['real_B_pass_count']}/425; failure={baseline_gate['failure_count']}.",
        f"- Non-self pass={baseline_gate['nonself_pass_count']}/{baseline_gate['nonself_count']}; State_Q equal={baseline_gate['state_q_equal']}.",
        f"- Max model metric error={max(baseline_gate['model_metric_max_abs_error_dB'].values()):.3e} dB; max theta errors={baseline_gate['theta_Aend_max_abs_error']:.3e}/{baseline_gate['theta_C2_max_abs_error']:.3e}.",
        "",
        "Frozen contract",
        f"- MP10 orders={history['orders']}; memory={history['memory']}; K10=10; K11=11; dmax={history['dmax']}; lambda={history['ridge_lambda']}.",
        "- Envelope75 dictionary=75; remaining single-basis candidates=65; total models=66.",
        f"- ABC valid lengths=12286/4913/7371; common-B SHA={EXPECTED_COMMON_B_SHA}; Real-B threshold={REAL_B_THRESHOLD_DB} dB.",
        "- All K11 keep dmax=2; MP10 columns stay in their historical order; the added column is canonical Envelope75.",
        "",
        "All425 forward ranking",
    ]
    for row in summary.sort_values("all425_rank").head(10).itertuples(index=False):
        lines.append(
            f"- rank={int(row.all425_rank)} {row.model_name}: added={row.added_basis_id}, p/m/q={row.added_order_p}/{row.added_delay_m}/{row.added_envelope_delay_q}, cross={row.is_cross_delay}, Top1={int(row.top1_pass_count)}/425, ΔTop1={int(row.delta_top1)}, recovered={int(row.recovered_count)}, regressed={int(row.regressed_count)}, nonself={float(row.nonself_pass_rate):.9g}, Q05={float(row.share_margin_Q05):.9g}, MRR={float(row.MRR):.9g}, Top3={int(row.Top3_oracle)}."
        )
    lines.append("- ...")
    for row in summary.sort_values("all425_rank").tail(10).itertuples(index=False):
        lines.append(
            f"- tail rank={int(row.all425_rank)} {row.model_name}: added={row.added_basis_id}, Top1={int(row.top1_pass_count)}/425, ΔTop1={int(row.delta_top1)}, recovered={int(row.recovered_count)}, regressed={int(row.regressed_count)}."
        )
    lines.extend(
        [
            "",
            "Diagnostic best K11",
            f"- {winner['model_name']} with added basis {winner['added_basis_id']}.",
            f"- Top1={int(winner['top1_pass_count'])}/425; ΔTop1={int(winner['delta_top1'])}; rank={int(winner['all425_rank'])}.",
            "- This is a diagnostic best K11, not a final frozen model.",
            "",
            "Five-fold Query-State CV stability",
            "- The previous K9 fold assignment was reused exactly: 5 folds × 85 Query States; LUT candidates remain all 425 states.",
        ]
    )
    for row in stability.loc[stability["train_winner_count"].gt(0)].sort_values("train_winner_count", ascending=False).itertuples(index=False):
        lines.append(
            f"- {row.model_name}: train winner={int(row.train_winner_count)}/5, folds={row.winner_folds}, validation Top1 mean/min={float(row.winner_validation_top1_pass_mean):.6g}/{float(row.winner_validation_top1_pass_min):.6g}."
        )
    lines.extend(
        [
            "",
            "Data protection and stop boundary",
            f"- raw manifest before: {json.dumps(raw_before, ensure_ascii=False, sort_keys=True)}",
            f"- raw manifest after: {json.dumps(raw_after, ensure_ascii=False, sort_keys=True)}",
            f"- raw data unchanged: {raw_before == raw_after}",
            "- No K12, multi-basis addition, swap, lambda/order/memory scan, Top-k selection, ensemble, clustering, DPD replay, or low-bandwidth task was run.",
            "- CV is a Query-State stability analysis, not an independent final test.",
        ]
    )
    (RESULT_ROOT / "21_final_result_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run(*, resume: bool = False) -> dict[str, Any]:
    if RESULT_ROOT.exists() and any(RESULT_ROOT.iterdir()) and not resume:
        raise RuntimeError(f"result directory is not empty; use --resume: {RESULT_ROOT}")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "distance_matrices").mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "fingerprints").mkdir(parents=True, exist_ok=True)
    raw_before = raw_manifest_gate()
    envelope_terms = tuple(build_envelope_dictionary())
    envelope_gate = dictionary_gate()
    mp10_ids = _mp10_basis_ids()
    common_b, common_meta = verify_common_b_contract()
    common_b = np.asarray(common_b, dtype=np.complex128)
    if common_meta["sha256"] != EXPECTED_COMMON_B_SHA:
        raise RuntimeError("common-B hash changed")
    previous, real_b_distance, shareability, folds = _load_previous_inputs()
    _random_real_b_check(real_b_distance)
    overlap = _run_overlap_regression((0, 187, 325, 424), common_b, envelope_terms, mp10_ids, {
        term.basis_id: term.index for term in envelope_terms
    })
    _, candidates, overlap_indices = _build_candidates(envelope_terms, mp10_ids)
    if len(candidates) != 66 or len(overlap_indices) != 10:
        raise RuntimeError("candidate pool contract failed")
    history = {
        "orders": list(backend.historical_mp10.MP_ORDERS),
        "memory": [backend.historical_mp10.MP_MEMORY[order] for order in backend.historical_mp10.MP_ORDERS],
        "K": 10,
        "dmax": backend.MP_DMAX,
        "ridge_lambda": backend.RIDGE_LAMBDA,
    }
    _write_json(
        RESULT_ROOT / "05_baseline_regression_check.json",
        {
            "previous_artifact_gate": previous["artifact_gate"],
            "real_B_ground_truth_reused_from": str(PREVIOUS_REAL_B_MATRIX),
            "shareability_reused_from": str(PREVIOUS_SHAREABILITY),
            "overlap_regression_max_abs_delta": float(overlap["max_abs_delta"].max()),
            "envelope_gate": envelope_gate,
            "pass_before_candidate_models": bool(previous["artifact_gate"]["pass"] and float(overlap["max_abs_delta"].max()) <= 1e-12 and len(candidates) == 66),
        },
    )
    if not previous["artifact_gate"]["pass"] or float(overlap["max_abs_delta"].max()) > 1e-12:
        raise RuntimeError("HARD FAIL: baseline or MP10/Envelope75 overlap preflight failed")
    _write_json(
        RESULT_ROOT / "01_reuse_audit.json",
        {
            "task_name": TASK_NAME,
            "historical_mp10_contract": history,
            "envelope75_dictionary_gate": envelope_gate,
            "mp10_basis_ids": list(mp10_ids),
            "mp10_envelope75_overlap_max_abs_delta": float(overlap["max_abs_delta"].max()),
            "remaining_candidate_count": len(candidates) - 1,
            "previous_k9_fold_file": str(MP10_K9_FOLD_FILE),
            "real_B_ground_truth_reused": True,
            "separate_hnorm_found": False,
            "no_new_core_retrieval_logic": True,
        },
    )
    (RESULT_ROOT / "00_task_definition.txt").write_text(
        "\n".join(
            [
                f"Task: {TASK_NAME}",
                "Mode: Frozen MP10 + one remaining canonical Envelope75 basis forward scan.",
                "Models: MP10_full plus 65 K11 candidates; no K12 or multi-basis addition.",
                f"MP10: orders={history['orders']}; memory={history['memory']}; K=10; dmax=2; lambda=1e-8.",
                "All K11: historical MP10 columns plus exactly one canonical Envelope75 column; dmax remains 2.",
                "Aend=final ILC; C2=ILC2; common-B=State 0 B; Top-1 allow-self; Real-B labels only after State_Q.",
                "Primary ranking: Top1 Real-B pass, non-self pass rate, Q05 margin, MRR, Top3 oracle, smaller K.",
                "No Hnorm, no lambda/order/memory scan, no swap, no ensemble, no clustering, no DPD, no low-bandwidth.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    audit_lines = [
        f"Task: {TASK_NAME}",
        "Historical MP10 backend: behavior_modeling.sparse_gmp build_frozen_mp_basis + fit_ridge.",
        f"Historical contract: {json.dumps(history, ensure_ascii=False, sort_keys=True)}",
        "Envelope75 source: scripts/basis_function_selection/envelope_dictionary.py.",
        f"Envelope75 gate: {json.dumps(envelope_gate, ensure_ascii=False, sort_keys=True)}",
        f"MP10 basis IDs: {list(mp10_ids)}",
        f"Overlap regression max_abs_delta: {float(overlap['max_abs_delta'].max()):.3e}.",
        "Overlap contexts: State 0/187/325/424 Aend A/B, C2 C/B and common-B.",
        "Real-B ground truth and shareability mask reused from previous K9 task after shape/diagonal/random-pair validation.",
        "Previous Query-State CV folds reused exactly from scenario_2_mp10_k9_retrieval_oriented_basis_ablation_5b.",
        "No separate Hnorm was found or added.",
        "No existing CNMSE, Top-1, Real-B or retrieval core was copied into this task.",
    ]
    (RESULT_ROOT / "01_reuse_audit.txt").write_text("\n".join(audit_lines) + "\n", encoding="utf-8")
    folds.to_csv(RESULT_ROOT / "13_query_state_cv_folds.csv", index=False)
    _checkpoint(phase="overlap_and_candidate_pool_passed", raw_manifest_before=raw_before, candidate_count=66, remaining_count=65, overlap_max_abs_delta=float(overlap["max_abs_delta"].max()))

    model_path = RESULT_ROOT / "06_candidate_model_quality_long.csv"
    coefficient_path = RESULT_ROOT / "07_candidate_coefficients.npz"
    if resume and model_path.is_file() and coefficient_path.is_file():
        model_frame = pd.read_csv(model_path)
        with np.load(coefficient_path, allow_pickle=False) as coefficient_data:
            theta_a = np.asarray(coefficient_data["theta_Aend_padded"])
            theta_c = np.asarray(coefficient_data["theta_C2_padded"])
        if theta_a.shape != (len(candidates), STATE_COUNT, backend.MODEL_K_MAX) or theta_c.shape != theta_a.shape:
            raise RuntimeError("resumed K11 coefficient cache shape failed")
    else:
        model_frame, theta_a, theta_c = _run_model_phase(candidates)
        model_frame.to_csv(model_path, index=False)
        theta_mask = np.zeros((len(candidates), backend.MODEL_K_MAX), dtype=bool)
        for candidate in candidates:
            theta_mask[candidate.candidate_id, : candidate.K] = True
        np.savez_compressed(
            coefficient_path,
            model_ids=np.asarray([candidate.candidate_id for candidate in candidates], dtype=np.int64),
            model_names=np.asarray([candidate.candidate_name for candidate in candidates]),
            added_basis_ids=np.asarray([candidate.added_basis_id or "NONE" for candidate in candidates]),
            theta_Aend_padded=theta_a,
            theta_C2_padded=theta_c,
            theta_support_mask=theta_mask,
            K=np.asarray([candidate.K for candidate in candidates], dtype=np.int64),
            dmax=np.asarray(backend.MP_DMAX),
            ridge_lambda=np.asarray(backend.RIDGE_LAMBDA),
        )
    mp_common = backend.historical_mp10.build_frozen_mp_basis(common_b)
    env_common = build_envelope_bank(common_b, envelope_terms)
    candidate_frames: dict[int, pd.DataFrame] = {}
    candidate_summaries: list[dict[str, Any]] = []
    distance_dir = RESULT_ROOT / "distance_matrices"
    fingerprint_dir = RESULT_ROOT / "fingerprints"
    for candidate in candidates:
        frame, summary, distance, lut, query = _candidate_retrieval(
            candidate,
            theta_a,
            theta_c,
            mp_common,
            env_common,
            real_b_distance,
            shareability,
        )
        candidate_frames[candidate.candidate_id] = frame
        candidate_summaries.append(summary)
        np.save(distance_dir / f"{candidate.candidate_name}.npy", distance)
        if candidate.candidate_id == 0:
            np.save(fingerprint_dir / "MP10_lut.npy", lut)
            np.save(fingerprint_dir / "MP10_query.npy", query)
    baseline_frame = candidate_frames[0]
    baseline = _baseline_regression(
        model_frame,
        theta_a,
        theta_c,
        (mp_common @ theta_a[0, :, :10].T).T,
        (mp_common @ theta_c[0, :, :10].T).T,
        np.load(distance_dir / "MP10_full.npy"),
        baseline_frame,
        previous,
    )
    _checkpoint(phase="mp10_baseline_regression_passed", baseline_regression=baseline, real_B_reused=True)
    query_metrics = pd.concat(candidate_frames.values(), ignore_index=True).sort_values(["candidate_id", "State_R"]).reset_index(drop=True)
    query_metrics.to_csv(RESULT_ROOT / "09_candidate_query_metrics_long.csv", index=False)
    summary = k9_utils._assign_diagnostic_ranks(candidate_summaries)
    summary["model_id"] = summary["candidate_id"].astype(int)
    summary["model_name"] = summary["candidate_name"]
    summary["added_basis_id"] = summary["removed_basis_id"].fillna("NONE")
    summary["top1_pass_count"] = summary["top1_realB_pass_count"]
    summary["top1_pass_rate"] = summary["top1_realB_pass_rate"]
    summary["all425_rank"] = summary["diagnostic_rank"]
    summary = summary.drop(columns=["removed_basis_id"]).sort_values("model_id").reset_index(drop=True)
    baseline_summary = summary.loc[summary["model_id"].eq(0)].iloc[0]
    for candidate in candidates[1:]:
        current = summary.loc[summary["model_id"].eq(candidate.candidate_id)].iloc[0]
        frame = candidate_frames[candidate.candidate_id]
        base_frame = candidate_frames[0]
        summary.loc[summary["model_id"].eq(candidate.candidate_id), "recovered_count"] = int((~base_frame["realB_pass"].to_numpy(dtype=bool) & frame["realB_pass"].to_numpy(dtype=bool)).sum())
        summary.loc[summary["model_id"].eq(candidate.candidate_id), "regressed_count"] = int((base_frame["realB_pass"].to_numpy(dtype=bool) & ~frame["realB_pass"].to_numpy(dtype=bool)).sum())
        summary.loc[summary["model_id"].eq(candidate.candidate_id), "delta_top1"] = int(current["top1_realB_pass_count"] - baseline_summary["top1_realB_pass_count"])
    for column in ("recovered_count", "regressed_count", "delta_top1"):
        summary[column] = summary[column].fillna(0).astype(int)
    summary.to_csv(RESULT_ROOT / "08_candidate_retrieval_summary.csv", index=False)
    delta = k9_utils._build_deltas(candidates, summary.drop(columns=["model_id"]), model_frame.rename(columns={"model_id": "candidate_id"}))
    delta = delta.rename(columns={"removed_basis_id": "added_basis_id", "candidate_id": "model_id", "candidate_name": "model_name"})
    delta["delta_top1"] = delta["delta_top1_pass"]
    delta = delta.merge(summary[["model_id", "added_order_p", "added_delay_m", "added_envelope_delay_q", "is_cross_delay"]], on="model_id", how="left")
    recovered_rows: list[dict[str, Any]] = []
    base = candidate_frames[0].set_index("State_R")
    for candidate in candidates[1:]:
        frame = candidate_frames[candidate.candidate_id].set_index("State_R")
        for state_id in range(STATE_COUNT):
            base_pass = bool(base.loc[state_id, "realB_pass"])
            current_pass = bool(frame.loc[state_id, "realB_pass"])
            transition = "recovered" if (not base_pass and current_pass) else "regressed" if (base_pass and not current_pass) else "unchanged_pass" if base_pass else "unchanged_fail"
            recovered_rows.append(
                {
                    "model_id": candidate.candidate_id,
                    "model_name": candidate.candidate_name,
                    "added_basis_id": candidate.added_basis_id,
                    "added_order_p": candidate.added_order_p,
                    "added_delay_m": candidate.added_delay_m,
                    "added_envelope_delay_q": candidate.added_envelope_delay_q,
                    "is_cross_delay": candidate.is_cross_delay,
                    "State_R": state_id,
                    "baseline_State_Q": int(base.loc[state_id, "State_Q"]),
                    "candidate_State_Q": int(frame.loc[state_id, "State_Q"]),
                    "baseline_pass": base_pass,
                    "candidate_pass": current_pass,
                    "transition": transition,
                }
            )
    recovered_frame = pd.DataFrame(recovered_rows)
    transition_counts_by_model = (
        recovered_frame.groupby(["model_id", "transition"]).size().unstack(fill_value=0)
        if not recovered_frame.empty
        else pd.DataFrame()
    )
    delta["recovered_count"] = delta["model_id"].map(
        transition_counts_by_model.get("recovered", pd.Series(dtype=int))
    ).fillna(0).astype(int)
    delta["regressed_count"] = delta["model_id"].map(
        transition_counts_by_model.get("regressed", pd.Series(dtype=int))
    ).fillna(0).astype(int)
    delta["net_recovery"] = delta["recovered_count"] - delta["regressed_count"]
    delta.to_csv(RESULT_ROOT / "10_forward_addition_delta.csv", index=False)
    recovered_frame.to_csv(RESULT_ROOT / "11_recovered_regressed_states.csv", index=False)
    cv_frame, stability = _local_cv_results(candidates, candidate_frames, folds)
    cv_frame.to_csv(RESULT_ROOT / "14_query_state_cv_results.csv", index=False)
    stability.to_csv(RESULT_ROOT / "15_query_state_cv_selection_stability.csv", index=False)
    winner = summary.sort_values("all425_rank").iloc[0]
    winner_id = int(winner["model_id"])
    winner_frame = candidate_frames[winner_id].copy()
    winner_frame.to_csv(RESULT_ROOT / "20_diagnostic_best_detail.csv", index=False)
    winner_candidate = candidates[winner_id]
    if winner_candidate.envelope_index is None:
        winner_lut = (mp_common @ theta_a[winner_id, :, :10].T).T
        winner_query = (mp_common @ theta_c[winner_id, :, :10].T).T
    else:
        winner_phi = np.column_stack((mp_common, env_common[:, winner_candidate.envelope_index]))
        winner_lut = (winner_phi @ theta_a[winner_id, :, :11].T).T
        winner_query = (winner_phi @ theta_c[winner_id, :, :11].T).T
    np.save(fingerprint_dir / "diagnostic_best_k11_lut.npy", winner_lut.astype(np.complex128))
    np.save(fingerprint_dir / "diagnostic_best_k11_query.npy", winner_query.astype(np.complex128))
    type_summary = summary.loc[summary["model_id"].ne(0)].groupby(["added_family", "is_cross_delay"], dropna=False).agg(
        candidate_count=("model_id", "count"),
        top1_pass_mean=("top1_pass_count", "mean"),
        top1_pass_max=("top1_pass_count", "max"),
        delta_top1_mean=("delta_top1", "mean"),
        positive_delta_count=("delta_top1", lambda values: int((values > 0).sum())),
    ).reset_index()
    type_summary.to_csv(RESULT_ROOT / "16_basis_type_summary.csv", index=False)
    _write_figures(summary, delta, pd.DataFrame([{"basis_id": c.added_basis_id, "is_cross_delay": c.is_cross_delay} for c in candidates[1:]]))
    raw_after = raw_manifest_gate()
    summary_lines = [
        f"Task: {TASK_NAME}",
        "Frozen MP10 + one remaining Envelope75 basis retrieval-oriented forward scan.",
        f"Baseline regression PASS={baseline['pass']}; MP10 Top1={baseline['real_B_pass_count']}/425; Exact={baseline['exact_count']}/425; Non-self={baseline['nonself_pass_count']}/{baseline['nonself_count']}; Top3=423/425.",
        f"Envelope75 dictionary count={len(envelope_terms)}; MP10 overlap=10; remaining candidates=65; total models=66.",
        f"MP10/Envelope75 overlap max_abs_delta={float(overlap['max_abs_delta'].max()):.3e}.",
        "All K11 use dmax=2 and lambda=1e-8; Real-B ground truth/mask and Query-State folds are reused from the preceding K9 task.",
        "",
        "Top diagnostic candidates:",
    ]
    for row in summary.sort_values("all425_rank").head(10).itertuples(index=False):
        summary_lines.append(
            f"rank={int(row.all425_rank)} {row.model_name}: added={row.added_basis_id}, p/m/q={row.added_order_p}/{row.added_delay_m}/{row.added_envelope_delay_q}, cross={row.is_cross_delay}, Top1={int(row.top1_pass_count)}/425, delta={int(row.delta_top1)}, recovered={int(row.recovered_count)}, regressed={int(row.regressed_count)}, nonself={float(row.nonself_pass_rate):.9g}, Q05={float(row.share_margin_Q05):.9g}, MRR={float(row.MRR):.9g}, Top3={int(row.Top3_oracle)}"
        )
    summary_lines.extend(
        [
            "",
            f"Diagnostic best: {winner['model_name']} / {winner['added_basis_id']}; Top1={int(winner['top1_pass_count'])}/425; delta={int(winner['delta_top1'])}.",
            "This is a diagnostic best K11, not a final frozen model.",
            "",
            "CV winner frequency:",
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
            "No K12, multi-basis addition, swap, lambda/order/memory scan, Top-k selection, ensemble, clustering, DPD replay, or low-bandwidth task was run.",
            "CV is a Query-State stability analysis, not an independent final test.",
        ]
    )
    (RESULT_ROOT / "21_final_result_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    if raw_before != raw_after:
        raise RuntimeError("data/raw manifest changed")
    transition_counts = recovered_frame["transition"].value_counts().to_dict()
    result = {
        "task": TASK_NAME,
        "status": "SUCCESS",
        "baseline_regression_pass": bool(baseline["pass"]),
        "diagnostic_best_model": str(winner["model_name"]),
        "diagnostic_best_basis": str(winner["added_basis_id"]),
        "diagnostic_best_top1_pass": int(winner["top1_pass_count"]),
        "baseline_top1_pass": 411,
        "max_k11_top1_pass": int(summary.loc[summary["K"].eq(11), "top1_pass_count"].max()),
        "k11_positive_delta_count": int((summary.loc[summary["K"].eq(11), "delta_top1"] > 0).sum()),
        "transition_counts_across_k11": {key: int(transition_counts.get(key, 0)) for key in ("recovered", "regressed", "unchanged_pass", "unchanged_fail")},
        "cv_winner_counts": stability.loc[stability["train_winner_count"].gt(0), ["model_name", "train_winner_count"]].to_dict("records"),
        "raw_data_modified": raw_before != raw_after,
        "result_root": str(RESULT_ROOT),
    }
    _write_json(RESULT_ROOT / "22_checkpoint.json", {"phase": "completed", **result})
    _append_log(
        f"\n[{_now()}] Complete {TASK_NAME}\nResult={json.dumps(result, ensure_ascii=False, sort_keys=True)}\n"
        "No K12 or additional optimization was run.\n"
    )
    _append_handoff(
        f"完成 Frozen MP10 + 65 个单项 Envelope75 K11 forward candidates；结果目录：{RESULT_ROOT}\n"
        f"摘要：{json.dumps(result, ensure_ascii=False, sort_keys=True)}\n"
        "完成 MP10/Envelope75 overlap regression、Full425 retrieval、recovered/regressed、basis type summary 和 5-fold Query-State CV；未冻结新模型。"
    )
    return result


def main() -> None:
    result = _run(resume="--resume" in sys.argv[1:])
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    __import__("multiprocessing").freeze_support()
    main()
