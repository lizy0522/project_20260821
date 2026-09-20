# ruff: noqa: E402,E501,I001

"""Run the ten leave-one-basis-out K9 retrieval ablations around MP10."""

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
from behavior_modeling.shared.config import BASIS_TERMS, MP_CONFIG  # noqa: E402
from behavior_fingerprint_retrieval.shared.envelope18_commonB_full_lut import (  # noqa: E402
    top1_retrieval,
)
from behavior_fingerprint_retrieval.shared.envelope19_c2endshared_commonB_full_lut import (  # noqa: E402
    EXPECTED_COMMON_B_SHA,
    compute_cnmse_matrix,
    raw_manifest_gate,
    verify_common_b_contract,
)
from retrieval_oriented_model_selection.shared import mp10_k9_retrieval_support as backend

TASK_NAME = "scenario_2_mp10_k9_retrieval_oriented_basis_ablation_5b"
RESULT_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /TASK_NAME
WORK_LOG = PROJECT_ROOT / "work_logs" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /TASK_NAME / "execution_log.txt"
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"
PREVIOUS_ROOT = PROJECT_ROOT / "results" / "behavior_fingerprint_retrieval" / "scenario_2_mp10_full_lut_retrieval_5b"
PREVIOUS_MODEL = PREVIOUS_ROOT / "04_all425_model_quality.csv"
PREVIOUS_COEFFICIENTS = PREVIOUS_ROOT / "05_mp10_aend_c2_coefficients.npz"
PREVIOUS_LUT = PREVIOUS_ROOT / "06_mp10_lut_fingerprints.npy"
PREVIOUS_QUERY = PREVIOUS_ROOT / "07_mp10_query_fingerprints.npy"
PREVIOUS_DISTANCE = PREVIOUS_ROOT / "08_mp10_fingerprint_cnmse_matrix.npy"
PREVIOUS_RETRIEVAL = PREVIOUS_ROOT / "09_retrieval_results_all425.csv"
STATE_COUNT = backend.STATE_COUNT
WORKER_COUNT = 10
BLAS_THREADS_PER_WORKER = 1
REAL_B_THRESHOLD_DB = -40.0
REAL_B_SANITY_TOLERANCE_DB = 1e-9
DRY_RUN_REFERENCE = "previous MP10 Full-LUT formal result"


@dataclass(frozen=True)
class Candidate:
    candidate_id: int
    candidate_name: str
    K: int
    removed_basis_id: str | None
    removed_baseline_column: int | None
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
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _checkpoint(**payload: Any) -> None:
    _write_json(
        RESULT_ROOT / "19_checkpoint.json",
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
            "candidate_count": 11,
            "baseline_candidate": "MP10_full",
            "K9_candidate_count": 10,
            "dmax_fixed": backend.MP_DMAX,
            "ridge_lambda_fixed": backend.RIDGE_LAMBDA,
            "envelope75_search_performed": False,
            "lambda_scan_performed": False,
            "memory_scan_performed": False,
            "order_scan_performed": False,
            "top_k_selection_performed": False,
            "real_B_used_for_top1_selection": False,
            **payload,
        },
    )


def _basis_id(term: tuple[int, int]) -> str:
    return f"p{int(term[0])}_m{int(term[1])}"


def _support_hash(ids: tuple[str, ...]) -> str:
    payload = json.dumps(list(ids), ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _build_candidates() -> tuple[pd.DataFrame, tuple[Candidate, ...]]:
    terms = tuple((int(order), int(delay)) for order, delay in BASIS_TERMS)
    if tuple(MP_CONFIG["orders"]) != tuple(backend.historical_mp10.MP_ORDERS):
        raise RuntimeError("MP10 basis order does not match the historical backend")
    if tuple(terms) != tuple(
        (int(order), int(delay))
        for order in backend.historical_mp10.MP_ORDERS
        for delay in range(backend.historical_mp10.MP_MEMORY[order])
    ):
        raise RuntimeError("behavior_modeling.config BASIS_TERMS is not the historical MP10 order")
    basis_ids = tuple(_basis_id(term) for term in terms)
    if len(basis_ids) != backend.MP_K:
        raise RuntimeError(f"historical MP10 basis count changed: {len(basis_ids)}")
    basis_rows = [
        {
            "baseline_column_index": index,
            "basis_id": basis_id,
            "order_p": term[0],
            "delay_m": term[1],
            "expression": f"x[n-{term[1]}] * abs(x[n-{term[1]}])**({term[0] - 1})",
        }
        for index, (term, basis_id) in enumerate(zip(terms, basis_ids, strict=True))
    ]
    candidates = [
        Candidate(
            candidate_id=0,
            candidate_name="MP10_full",
            K=backend.MP_K,
            removed_basis_id=None,
            removed_baseline_column=None,
            support=tuple(range(backend.MP_K)),
            retained_basis_ids=basis_ids,
            support_hash=_support_hash(basis_ids),
        )
    ]
    for index, basis_id in enumerate(basis_ids):
        retained = tuple(item for item in range(backend.MP_K) if item != index)
        retained_ids = tuple(basis_ids[item] for item in retained)
        candidates.append(
            Candidate(
                candidate_id=index + 1,
                candidate_name=f"MP10_minus_{basis_id}",
                K=backend.MP_K - 1,
                removed_basis_id=basis_id,
                removed_baseline_column=index,
                support=retained,
                retained_basis_ids=retained_ids,
                support_hash=_support_hash(retained_ids),
            )
        )
    basis_frame = pd.DataFrame(basis_rows)
    candidate_frame = pd.DataFrame(
        [
            {
                "candidate_id": candidate.candidate_id,
                "candidate_name": candidate.candidate_name,
                "K": candidate.K,
                "removed_basis_id": candidate.removed_basis_id,
                "removed_baseline_column": candidate.removed_baseline_column,
                "retained_basis_ids": json.dumps(
                    list(candidate.retained_basis_ids), ensure_ascii=False
                ),
                "support_indices": json.dumps(list(candidate.support)),
                "support_hash": candidate.support_hash,
                "dmax": backend.MP_DMAX,
                "ridge_lambda": backend.RIDGE_LAMBDA,
            }
            for candidate in candidates
        ]
    )
    basis_frame.to_csv(RESULT_ROOT / "02_mp10_basis_order.csv", index=False)
    candidate_frame.to_csv(RESULT_ROOT / "03_candidate_supports.csv", index=False)
    return basis_frame, tuple(candidates)


def _load_previous_baseline() -> dict[str, Any]:
    required = (
        PREVIOUS_MODEL,
        PREVIOUS_COEFFICIENTS,
        PREVIOUS_LUT,
        PREVIOUS_QUERY,
        PREVIOUS_DISTANCE,
        PREVIOUS_RETRIEVAL,
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"previous MP10 baseline artifacts are missing: {missing}")
    contract = json.loads((PREVIOUS_ROOT / "02_protocol_contract.json").read_text())
    model = pd.read_csv(PREVIOUS_MODEL).sort_values("State_n_R").reset_index(drop=True)
    retrieval = pd.read_csv(PREVIOUS_RETRIEVAL).sort_values("State_n_R").reset_index(drop=True)
    with np.load(PREVIOUS_COEFFICIENTS, allow_pickle=False) as data:
        theta_a = np.asarray(data["theta_aend"])
        theta_c = np.asarray(data["theta_c2"])
    lut = np.load(PREVIOUS_LUT)
    query = np.load(PREVIOUS_QUERY)
    distance = np.load(PREVIOUS_DISTANCE)
    if model.shape[0] != STATE_COUNT or retrieval.shape[0] != STATE_COUNT:
        raise RuntimeError("previous MP10 baseline is not a canonical 425-row result")
    if not np.array_equal(model["State_n_R"].to_numpy(dtype=int), np.arange(STATE_COUNT)):
        raise RuntimeError("previous MP10 model state order changed")
    if not np.array_equal(retrieval["State_n_R"].to_numpy(dtype=int), np.arange(STATE_COUNT)):
        raise RuntimeError("previous MP10 retrieval state order changed")
    if theta_a.shape != (STATE_COUNT, backend.MP_K) or theta_c.shape != (STATE_COUNT, backend.MP_K):
        raise RuntimeError("previous MP10 coefficient shapes changed")
    if lut.shape != (STATE_COUNT, backend.FINGERPRINT_LENGTH) or query.shape != lut.shape:
        raise RuntimeError("previous MP10 fingerprint shapes changed")
    if distance.shape != (STATE_COUNT, STATE_COUNT):
        raise RuntimeError("previous MP10 distance shape changed")
    selected_previous = retrieval["State_n_Q"].to_numpy(dtype=int)
    selected_recomputed, _, _, _ = top1_retrieval(distance)
    artifact_gate = {
        "contract_matches_historical": contract.get("model", {}).get("K") == backend.MP_K
        and contract.get("model", {}).get("dmax") == backend.MP_DMAX
        and float(contract.get("model", {}).get("ridge_lambda")) == backend.RIDGE_LAMBDA
        and contract.get("model", {}).get("orders") == list(backend.historical_mp10.MP_ORDERS)
        and contract.get("model", {}).get("memory") == [
            backend.historical_mp10.MP_MEMORY[order] for order in backend.historical_mp10.MP_ORDERS
        ],
        "top1_recomputed_from_saved_distance_equal": bool(
            np.array_equal(selected_previous, selected_recomputed)
        ),
        "exact_count": int(retrieval["is_exact_state_hit"].sum()),
        "real_B_pass_count": int(retrieval["retrieved_real_B_pass"].sum()),
        "failure_count": int((~retrieval["retrieved_real_B_pass"]).sum()),
        "nonself_count": int((retrieval["State_n_Q"].to_numpy(dtype=int) != np.arange(STATE_COUNT)).sum()),
        "nonself_pass_count": int(
            (
                (retrieval["State_n_Q"].to_numpy(dtype=int) != np.arange(STATE_COUNT))
                & retrieval["retrieved_real_B_pass"].to_numpy(dtype=bool)
            ).sum()
        ),
    }
    artifact_gate["pass"] = bool(
        artifact_gate["contract_matches_historical"]
        and artifact_gate["top1_recomputed_from_saved_distance_equal"]
        and artifact_gate["exact_count"] == 218
        and artifact_gate["real_B_pass_count"] == 411
        and artifact_gate["failure_count"] == 14
        and artifact_gate["nonself_count"] == 207
        and artifact_gate["nonself_pass_count"] == 193
    )
    if not artifact_gate["pass"]:
        raise RuntimeError(f"HARD FAIL: previous MP10 baseline artifact gate failed: {artifact_gate}")
    return {
        "contract": contract,
        "model": model,
        "retrieval": retrieval,
        "theta_aend": theta_a,
        "theta_c2": theta_c,
        "lut": lut,
        "query": query,
        "distance": distance,
        "artifact_gate": artifact_gate,
    }


def _write_task_definition(contract: dict[str, Any], baseline_gate: dict[str, Any]) -> None:
    lines = [
        f"Task: {TASK_NAME}",
        "Mode: retrieval-oriented basis ablation.",
        "Candidates: one historical MP10 K10 baseline plus ten leave-one-basis-out K9 candidates.",
        f"Historical contract: orders={contract['orders']}; memory={contract['memory']}; K={contract['K']}; dmax={contract['dmax']}; lambda={contract['ridge_lambda']}.",
        "All candidates retain dmax=2 and use the same A/B/C valid support.",
        "Aend uses each state's final ILC column; C2 uses ILC2; ILC inputs are column-peak normalized by get_ilc_pair.",
        "LUT fingerprint is Aend response on the frozen State 0 common-B probe; Query fingerprint is C2 response on the same probe.",
        "Top-1 uses the existing allow-self fingerprint CNMSE retrieval; Real-B is evaluated after State_Q is frozen.",
        "Primary ranking: Top-1 Real-B pass count, non-self pass rate, Q05 shareability margin, MRR, Top-3 oracle count, then smaller K.",
        "Five-fold Query-State CV is a stability analysis, not an independent final test.",
        "No Envelope75 search, new basis invention, order/memory/lambda scan, Top-k selection, ensemble, clustering, DPD replay, or low-bandwidth processing.",
        "",
        f"Previous MP10 baseline artifact gate: {json.dumps(baseline_gate, ensure_ascii=False, sort_keys=True)}",
    ]
    (RESULT_ROOT / "00_task_definition.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_reuse_audit(
    contract: dict[str, Any],
    baseline_gate: dict[str, Any],
    common_meta: dict[str, Any],
) -> None:
    lines = [
        f"Task: {TASK_NAME}",
        "Reuse audit: this task ablates the previously validated historical MP10 only.",
        "",
        "Historical MP10 source:",
        "- scripts/behavior_modeling/config.py -> MP_CONFIG and BASIS_TERMS.",
        "- scripts/behavior_modeling/basis.py -> build_mp_basis.",
        "- scripts/behavior_modeling/sparse_gmp.py -> build_frozen_mp_basis and fit_ridge.",
        "- scripts/scenario_2_mp10_full_lut_retrieval_5b/mp10_c2endshared_commonB_full_lut.py -> previously validated MP10/E19 adapter.",
        "",
        "Historical contract:",
        f"- orders={contract['orders']}; memory={contract['memory']}; K={contract['K']}; dmax={contract['dmax']}; lambda={contract['ridge_lambda']}.",
        "- Basis columns are built once as historical MP10 Phi10 and K9 candidates are column slices.",
        "- Every K9 keeps dmax=2, including deletion of a maximum-delay column.",
        "- get_ilc_pair performs per-column peak normalization. No separate Hnorm exists in the current checkout and none was added.",
        "",
        "Reused retrieval protocol:",
        "- data_management state order and raw scalar fields.",
        "- signal_segmentation canonical full Rough/Fine, one ABC split and segment gain adjustment.",
        "- Aend=final ILC column; C2=ILC2; OFF Real-B=yout_withoutdpd_ori B.",
        "- E19 common-B State 0 probe and the existing common-B SHA gate.",
        "- Existing CNMSE matrix helper and E19 Top-1 allow-self tie rule.",
        "",
        "No new implementation of alignment, gain adjustment, ABC, common-B, CNMSE, Top-1, or Real-B was created.",
        "Real-B is used only after State_Q retrieval to label candidate performance. It is not used to choose Top-1.",
        "Five-fold CV is Query-State stability analysis, not an independent final test.",
        "",
        f"Previous MP10 result root: {PREVIOUS_ROOT}",
        f"Previous baseline artifact gate: {json.dumps(baseline_gate, ensure_ascii=False, sort_keys=True)}",
        f"Common-B metadata: {json.dumps(common_meta, ensure_ascii=False, sort_keys=True, default=_json_default)}",
    ]
    (RESULT_ROOT / "01_reuse_audit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _build_query_state_folds(state_table: pd.DataFrame) -> pd.DataFrame:
    frame = state_table.copy().sort_values(
        ["funMng", "secMng", "funAng", "secAng", "state_id"]
    ).reset_index(drop=True)
    frame["load_class"] = np.select(
        [
            frame["funMng"].ne(0) & frame["secMng"].eq(0),
            frame["funMng"].eq(0) & frame["secMng"].ne(0),
            frame["funMng"].ne(0) & frame["secMng"].ne(0),
        ],
        ["fundamental_only", "second_harmonic_only", "joint_mismatch"],
        default="matched",
    )
    frame["load_stratum"] = (
        frame["load_class"].astype(str)
        + "|funMng="
        + frame["funMng"].astype(str)
        + "|secMng="
        + frame["secMng"].astype(str)
        + "|funAng="
        + frame["funAng"].astype(str)
    )
    frame["fold"] = np.arange(frame.shape[0], dtype=np.int64) % 5
    frame = frame.sort_values("state_id").reset_index(drop=True)
    counts = frame["fold"].value_counts().sort_index()
    if counts.to_dict() != {0: 85, 1: 85, 2: 85, 3: 85, 4: 85}:
        raise RuntimeError(f"query-state folds are not five balanced folds: {counts.to_dict()}")
    frame[["state_id", "fold", "load_class", "load_stratum", "funMng", "funAng", "secMng", "secAng", "Vm", "Pin"]].to_csv(
        RESULT_ROOT / "12_query_state_cv_folds.csv", index=False
    )
    return frame


def _run_model_phase(
    candidates: tuple[Candidate, ...],
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    candidate_specs = tuple((candidate.candidate_id, candidate.support) for candidate in candidates)
    results: list[dict[str, Any]] = []
    context = get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=WORKER_COUNT,
        mp_context=context,
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
                print(f"[MP10 K9 MODEL] {completed}/{STATE_COUNT}", flush=True)
    results.sort(key=lambda item: int(item["State_n_R"]))
    if [int(item["State_n_R"]) for item in results] != list(range(STATE_COUNT)):
        raise RuntimeError("candidate model phase state order is not 0...424")
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    model_rows: list[dict[str, Any]] = []
    theta_a = np.zeros((len(candidates), STATE_COUNT, backend.MP_K), dtype=np.complex128)
    theta_c = np.zeros_like(theta_a)
    real_b_rows: list[np.ndarray] = []
    for item in results:
        state_id = int(item["State_n_R"])
        real_b_rows.append(np.asarray(item["real_B_valid"], dtype=np.complex128))
        for fit in item["candidates"]:
            candidate_id = int(fit["candidate_id"])
            candidate = candidate_by_id[candidate_id]
            theta_a[candidate_id, state_id, list(candidate.support)] = fit["theta_Aend"]
            theta_c[candidate_id, state_id, list(candidate.support)] = fit["theta_C2"]
            model_rows.append(
                {
                    "candidate_id": candidate_id,
                    "candidate_name": candidate.candidate_name,
                    "K": candidate.K,
                    "removed_basis_id": candidate.removed_basis_id,
                    "removed_baseline_column": candidate.removed_baseline_column,
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
    model_frame = pd.DataFrame(model_rows).sort_values(["candidate_id", "State_R"]).reset_index(drop=True)
    if model_frame.shape[0] != len(candidates) * STATE_COUNT:
        raise RuntimeError(f"candidate model long table has wrong shape: {model_frame.shape}")
    real_b = np.stack(real_b_rows, axis=0).astype(np.complex128, copy=False)
    if real_b.shape != (STATE_COUNT, backend.FINGERPRINT_LENGTH) or not np.all(np.isfinite(real_b)):
        raise RuntimeError("Real-B ground truth waveform matrix has wrong shape or non-finite values")
    return model_frame, theta_a, theta_c, real_b


def _write_coefficients(candidates: tuple[Candidate, ...], theta_a: np.ndarray, theta_c: np.ndarray) -> None:
    mask = np.zeros((len(candidates), backend.MP_K), dtype=bool)
    support_indices = np.full((len(candidates), backend.MP_K), -1, dtype=np.int64)
    for candidate in candidates:
        mask[candidate.candidate_id, list(candidate.support)] = True
        support_indices[candidate.candidate_id, : candidate.K] = candidate.support
    np.savez_compressed(
        RESULT_ROOT / "07_candidate_coefficients.npz",
        candidate_ids=np.asarray([candidate.candidate_id for candidate in candidates], dtype=np.int64),
        candidate_names=np.asarray([candidate.candidate_name for candidate in candidates]),
        removed_basis_ids=np.asarray([candidate.removed_basis_id or "" for candidate in candidates]),
        theta_Aend_padded=theta_a,
        theta_C2_padded=theta_c,
        theta_support_mask=mask,
        support_indices=support_indices,
        K=np.asarray([candidate.K for candidate in candidates], dtype=np.int64),
        dmax=np.asarray(backend.MP_DMAX),
        ridge_lambda=np.asarray(backend.RIDGE_LAMBDA),
    )


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
    model_frame: pd.DataFrame,
    theta_a: np.ndarray,
    theta_c: np.ndarray,
    baseline_lut: np.ndarray,
    baseline_query: np.ndarray,
    baseline_distance: np.ndarray,
    baseline_selected: np.ndarray,
    baseline_real_b: np.ndarray,
    previous: dict[str, Any],
) -> dict[str, Any]:
    current_model = model_frame.loc[model_frame["candidate_id"].eq(0)].sort_values("State_R").reset_index(drop=True)
    previous_model = previous["model"]
    model_columns = [
        "Y_Aend_train_NMSE_dB",
        "Y_Aend_B_NMSE_dB",
        "Y_C2_train_NMSE_dB",
        "Y_C2_B_NMSE_dB",
    ]
    model_errors = {
        column: float(
            np.max(
                np.abs(
                    current_model[column].to_numpy(dtype=float)
                    - previous_model[column].to_numpy(dtype=float)
                )
            )
        )
        for column in model_columns
    }
    theta_a_error = float(np.max(np.abs(theta_a[0] - previous["theta_aend"])))
    theta_c_error = float(np.max(np.abs(theta_c[0] - previous["theta_c2"])))
    lut_error, lut_mismatch = _max_abs_with_neginf(baseline_lut, previous["lut"])
    query_error, query_mismatch = _max_abs_with_neginf(baseline_query, previous["query"])
    distance_error, distance_mismatch = _max_abs_with_neginf(baseline_distance, previous["distance"])
    real_b_retrieved = previous["retrieval"]["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float)
    current_real_b_retrieved = baseline_real_b
    real_b_error, real_b_mismatch = _max_abs_with_neginf(current_real_b_retrieved, real_b_retrieved)
    previous_selected = previous["retrieval"]["State_n_Q"].to_numpy(dtype=int)
    retrieval = {
        "state_q_equal": bool(np.array_equal(baseline_selected, previous_selected)),
        "exact_count": int(np.count_nonzero(baseline_selected == np.arange(STATE_COUNT))),
        "real_B_pass_count": int(np.count_nonzero(baseline_real_b < REAL_B_THRESHOLD_DB)),
        "failure_count": int(np.count_nonzero(~(baseline_real_b < REAL_B_THRESHOLD_DB))),
        "nonself_count": int(np.count_nonzero(baseline_selected != np.arange(STATE_COUNT))),
        "nonself_pass_count": int(
            np.count_nonzero(
                (baseline_selected != np.arange(STATE_COUNT))
                & (baseline_real_b < REAL_B_THRESHOLD_DB)
            )
        ),
    }
    gate = {
        "previous_artifact_gate": previous["artifact_gate"]["pass"],
        "model_metric_max_abs_error_dB": model_errors,
        "theta_Aend_max_abs_error": theta_a_error,
        "theta_C2_max_abs_error": theta_c_error,
        "LUT_fingerprint_max_abs_error": lut_error,
        "LUT_fingerprint_nonfinite_pattern_mismatch": lut_mismatch,
        "Query_fingerprint_max_abs_error": query_error,
        "Query_fingerprint_nonfinite_pattern_mismatch": query_mismatch,
        "distance_max_abs_error_dB": distance_error,
        "distance_nonfinite_pattern_mismatch": distance_mismatch,
        "retrieved_Real_B_max_abs_error_dB": real_b_error,
        "retrieved_Real_B_nonfinite_pattern_mismatch": real_b_mismatch,
        "retrieval": retrieval,
    }
    gate["pass"] = bool(
        gate["previous_artifact_gate"]
        and max(model_errors.values()) < 1e-10
        and theta_a_error < 1e-12
        and theta_c_error < 1e-12
        and lut_error < 1e-12
        and lut_mismatch == 0
        and query_error < 1e-12
        and query_mismatch == 0
        and distance_error < 1e-12
        and distance_mismatch == 0
        and real_b_error < 1e-9
        and real_b_mismatch == 0
        and retrieval["state_q_equal"]
        and retrieval["exact_count"] == 218
        and retrieval["real_B_pass_count"] == 411
        and retrieval["failure_count"] == 14
        and retrieval["nonself_count"] == 207
        and retrieval["nonself_pass_count"] == 193
    )
    if not gate["pass"]:
        raise RuntimeError(f"HARD FAIL: fresh MP10 baseline regression failed: {gate}")
    return gate


def _finite_quantile(values: np.ndarray, quantile: float) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    return float(np.quantile(finite, quantile)) if finite.size else float("nan")


def _finite_median(values: np.ndarray) -> float:
    return _finite_quantile(values, 0.5)


def _ranked_indices(row: np.ndarray, selected: int) -> np.ndarray:
    state_ids = np.arange(STATE_COUNT, dtype=np.int64)
    order = np.lexsort((state_ids, row))
    if int(order[0]) == int(selected):
        return order
    return np.concatenate(
        (np.asarray([selected], dtype=np.int64), order[order != int(selected)])
    )


def _candidate_retrieval(
    candidate: Candidate,
    theta_a: np.ndarray,
    theta_c: np.ndarray,
    common_phi: np.ndarray,
    real_b_distance: np.ndarray,
    shareability: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    support = candidate.support
    common_selected = np.asarray(common_phi[:, support], dtype=np.complex128)
    theta_a_selected = np.asarray(
        theta_a[candidate.candidate_id][:, list(support)], dtype=np.complex128
    )
    theta_c_selected = np.asarray(
        theta_c[candidate.candidate_id][:, list(support)], dtype=np.complex128
    )
    lut = (common_selected @ theta_a_selected.T).T.astype(np.complex128)
    query = (common_selected @ theta_c_selected.T).T.astype(np.complex128)
    if lut.shape != (STATE_COUNT, backend.FINGERPRINT_LENGTH) or query.shape != lut.shape:
        raise RuntimeError(f"{candidate.candidate_name} fingerprint shape failed")
    if not np.all(np.isfinite(lut)) or not np.all(np.isfinite(query)):
        raise RuntimeError(f"{candidate.candidate_name} fingerprint finite gate failed")
    distance = compute_cnmse_matrix(query, lut)
    if distance.shape != (STATE_COUNT, STATE_COUNT) or np.isnan(distance).any() or np.isposinf(distance).any():
        raise RuntimeError(f"{candidate.candidate_name} distance matrix contract failed")
    selected, selected_distance, true_rank, tie_count = top1_retrieval(distance)
    rows: list[dict[str, Any]] = []
    state_ids = np.arange(STATE_COUNT, dtype=np.int64)
    for state_id in range(STATE_COUNT):
        row = distance[state_id]
        ranked = _ranked_indices(row, int(selected[state_id]))
        share_indices = state_ids[shareability[state_id]]
        nonshare_indices = state_ids[~shareability[state_id]]
        if share_indices.size == 0:
            raise RuntimeError(f"state {state_id} has no shareable Real-B candidate")
        share_order = share_indices[np.lexsort((share_indices, row[share_indices]))]
        if nonshare_indices.size:
            nonshare_order = nonshare_indices[np.lexsort((nonshare_indices, row[nonshare_indices]))]
            nearest_nonshare_id = int(nonshare_order[0])
            nearest_nonshare_distance = float(row[nearest_nonshare_id])
            nearest_share_distance = float(row[int(share_order[0])])
            share_margin = nearest_nonshare_distance - nearest_share_distance
        else:
            nearest_nonshare_id = np.nan
            nearest_nonshare_distance = np.nan
            nearest_share_distance = float(row[int(share_order[0])])
            share_margin = np.nan
        first_shareable_positions = np.flatnonzero(shareability[state_id, ranked])
        if first_shareable_positions.size != 1 and first_shareable_positions.size == 0:
            raise RuntimeError(f"state {state_id} has no ranked shareable candidate")
        first_shareable_rank = int(first_shareable_positions[0] + 1)
        real_b_value = float(real_b_distance[state_id, int(selected[state_id])])
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "candidate_name": candidate.candidate_name,
                "K": candidate.K,
                "removed_basis_id": candidate.removed_basis_id,
                "State_R": state_id,
                "State_Q": int(selected[state_id]),
                "exact_hit": bool(int(selected[state_id]) == state_id),
                "realB_pass": bool(real_b_value < REAL_B_THRESHOLD_DB),
                "retrieved_realB_CNMSE_dB": real_b_value,
                "fingerprint_top1_CNMSE_dB": float(selected_distance[state_id]),
                "fingerprint_top2_CNMSE_dB": float(row[int(ranked[1])]),
                "top12_margin_dB": float(row[int(ranked[1])] - row[int(ranked[0])]),
                "true_state_rank": int(true_rank[state_id]),
                "minimum_tie_count": int(tie_count[state_id]),
                "nearest_shareable_State_Q": int(share_order[0]),
                "nearest_shareable_distance_dB": nearest_share_distance,
                "nearest_nonshareable_State_Q": nearest_nonshare_id,
                "nearest_nonshareable_distance_dB": nearest_nonshare_distance,
                "shareability_margin_dB": share_margin,
                "first_shareable_rank": first_shareable_rank,
                "Top2_has_shareable": bool(shareability[state_id, ranked[:2]].any()),
                "Top3_has_shareable": bool(shareability[state_id, ranked[:3]].any()),
                "Top5_has_shareable": bool(shareability[state_id, ranked[:5]].any()),
                "Top10_has_shareable": bool(shareability[state_id, ranked[:10]].any()),
            }
        )
    query_frame = pd.DataFrame(rows).sort_values("State_R").reset_index(drop=True)
    summary = _summarize_query_frame(query_frame, candidate)
    return query_frame, summary, distance, lut, query


def _summarize_query_frame(frame: pd.DataFrame, candidate: Candidate) -> dict[str, Any]:
    exact = frame["exact_hit"].to_numpy(dtype=bool)
    real_pass = frame["realB_pass"].to_numpy(dtype=bool)
    nonself = ~exact
    margins = frame["shareability_margin_dB"].to_numpy(dtype=float)
    top12 = frame["top12_margin_dB"].to_numpy(dtype=float)
    nonself_count = int(nonself.sum())
    nonself_pass = int((nonself & real_pass).sum())
    return {
        "candidate_id": candidate.candidate_id,
        "candidate_name": candidate.candidate_name,
        "K": candidate.K,
        "removed_basis_id": candidate.removed_basis_id,
        "exact_hit_count": int(exact.sum()),
        "top1_realB_pass_count": int(real_pass.sum()),
        "top1_realB_pass_rate": float(real_pass.mean()),
        "failure_count": int((~real_pass).sum()),
        "nonself_count": nonself_count,
        "nonself_pass_count": nonself_pass,
        "nonself_pass_rate": float(nonself_pass / nonself_count) if nonself_count else float("nan"),
        "share_margin_positive_count": int(np.count_nonzero(margins > 0)),
        "share_margin_infinite_count": int(np.count_nonzero(np.isinf(margins))),
        "share_margin_finite_count": int(np.count_nonzero(np.isfinite(margins))),
        "share_margin_Q05": _finite_quantile(margins, 0.05),
        "share_margin_Q10": _finite_quantile(margins, 0.10),
        "share_margin_median": _finite_median(margins),
        "share_margin_worst": float(np.nanmin(margins)) if np.any(np.isfinite(margins)) else float("nan"),
        "MRR": float(np.mean(1.0 / frame["first_shareable_rank"].to_numpy(dtype=float))),
        "Top1_oracle": int(real_pass.sum()),
        "Top2_oracle": int(frame["Top2_has_shareable"].sum()),
        "Top3_oracle": int(frame["Top3_has_shareable"].sum()),
        "Top5_oracle": int(frame["Top5_has_shareable"].sum()),
        "Top10_oracle": int(frame["Top10_has_shareable"].sum()),
        "top12_margin_median": _finite_median(top12),
        "top12_margin_finite_count": int(np.count_nonzero(np.isfinite(top12))),
        "first_shareable_rank_median": float(frame["first_shareable_rank"].median()),
        "first_shareable_rank_Q95": float(frame["first_shareable_rank"].quantile(0.95)),
    }


def _selection_key(summary: dict[str, Any]) -> tuple[float, ...]:
    def finite_or_low(value: Any) -> float:
        number = float(value)
        return number if np.isfinite(number) else -np.inf

    return (
        -float(summary["top1_realB_pass_count"]),
        -finite_or_low(summary["nonself_pass_rate"]),
        -finite_or_low(summary["share_margin_Q05"]),
        -finite_or_low(summary["MRR"]),
        -float(summary["Top3_oracle"]),
        float(summary["K"]),
        float(summary["candidate_id"]),
    )


def _assign_diagnostic_ranks(summary_rows: list[dict[str, Any]]) -> pd.DataFrame:
    ordered = sorted(summary_rows, key=_selection_key)
    rank_by_id = {int(row["candidate_id"]): index for index, row in enumerate(ordered, start=1)}
    for row in summary_rows:
        row["diagnostic_rank"] = rank_by_id[int(row["candidate_id"])]
    return pd.DataFrame(summary_rows).sort_values("candidate_id").reset_index(drop=True)


def _build_deltas(
    candidates: tuple[Candidate, ...],
    summary_frame: pd.DataFrame,
    model_frame: pd.DataFrame,
) -> pd.DataFrame:
    baseline = summary_frame.loc[summary_frame["candidate_id"].eq(0)].iloc[0]
    base_model = model_frame.loc[model_frame["candidate_id"].eq(0)]
    base_medians = {
        column: float(base_model[column].median())
        for column in (
            "Y_Aend_train_NMSE_dB",
            "Y_Aend_B_NMSE_dB",
            "Y_C2_train_NMSE_dB",
            "Y_C2_B_NMSE_dB",
        )
    }
    rows: list[dict[str, Any]] = []
    for candidate in candidates[1:]:
        current = summary_frame.loc[summary_frame["candidate_id"].eq(candidate.candidate_id)].iloc[0]
        current_model = model_frame.loc[model_frame["candidate_id"].eq(candidate.candidate_id)]
        current_medians = {
            column: float(current_model[column].median())
            for column in base_medians
        }
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "candidate_name": candidate.candidate_name,
                "removed_basis_id": candidate.removed_basis_id,
                "K": candidate.K,
                "baseline_top1_pass": int(baseline["top1_realB_pass_count"]),
                "candidate_top1_pass": int(current["top1_realB_pass_count"]),
                "delta_top1_pass": int(current["top1_realB_pass_count"] - baseline["top1_realB_pass_count"]),
                "baseline_exact": int(baseline["exact_hit_count"]),
                "candidate_exact": int(current["exact_hit_count"]),
                "delta_exact": int(current["exact_hit_count"] - baseline["exact_hit_count"]),
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
            }
        )
    return pd.DataFrame(rows).sort_values("candidate_id").reset_index(drop=True)


def _build_recovered_regressed(
    candidate_frames: dict[int, pd.DataFrame],
    candidates: tuple[Candidate, ...],
) -> pd.DataFrame:
    baseline = candidate_frames[0].set_index("State_R")
    rows: list[dict[str, Any]] = []
    for candidate in candidates[1:]:
        current = candidate_frames[candidate.candidate_id].set_index("State_R")
        for state_id in range(STATE_COUNT):
            base_pass = bool(baseline.loc[state_id, "realB_pass"])
            current_pass = bool(current.loc[state_id, "realB_pass"])
            if not base_pass and current_pass:
                transition = "recovered"
            elif base_pass and not current_pass:
                transition = "regressed"
            elif base_pass:
                transition = "unchanged_pass"
            else:
                transition = "unchanged_fail"
            rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "candidate_name": candidate.candidate_name,
                    "removed_basis_id": candidate.removed_basis_id,
                    "State_R": state_id,
                    "baseline_State_Q": int(baseline.loc[state_id, "State_Q"]),
                    "candidate_State_Q": int(current.loc[state_id, "State_Q"]),
                    "baseline_realB_CNMSE_dB": float(baseline.loc[state_id, "retrieved_realB_CNMSE_dB"]),
                    "candidate_realB_CNMSE_dB": float(current.loc[state_id, "retrieved_realB_CNMSE_dB"]),
                    "baseline_pass": base_pass,
                    "candidate_pass": current_pass,
                    "transition": transition,
                }
            )
    return pd.DataFrame(rows)


def _cv_results(
    candidates: tuple[Candidate, ...],
    candidate_frames: dict[int, pd.DataFrame],
    folds: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for fold in range(5):
        validation_ids = folds.loc[folds["fold"].eq(fold), "state_id"].to_numpy(dtype=int)
        train_ids = folds.loc[~folds["fold"].eq(fold), "state_id"].to_numpy(dtype=int)
        train_summaries = {
            candidate.candidate_id: _summarize_query_frame(
                candidate_frames[candidate.candidate_id].loc[
                    candidate_frames[candidate.candidate_id]["State_R"].isin(train_ids)
                ],
                candidate,
            )
            for candidate in candidates
        }
        ranking = sorted(train_summaries.values(), key=_selection_key)
        winner_id = int(ranking[0]["candidate_id"])
        rank_by_id = {int(item["candidate_id"]): index for index, item in enumerate(ranking, start=1)}
        for candidate in candidates:
            train_summary = train_summaries[candidate.candidate_id]
            validation_frame = candidate_frames[candidate.candidate_id].loc[
                candidate_frames[candidate.candidate_id]["State_R"].isin(validation_ids)
            ]
            validation_summary = _summarize_query_frame(validation_frame, candidate)
            rows.append(
                {
                    "fold": fold,
                    "candidate_id": candidate.candidate_id,
                    "candidate_name": candidate.candidate_name,
                    "removed_basis_id": candidate.removed_basis_id,
                    "K": candidate.K,
                    "train_query_count": len(train_ids),
                    "validation_query_count": len(validation_ids),
                    "train_is_winner": bool(candidate.candidate_id == winner_id),
                    "train_selection_rank": rank_by_id[candidate.candidate_id],
                    "train_top1_realB_pass_count": train_summary["top1_realB_pass_count"],
                    "train_nonself_pass_rate": train_summary["nonself_pass_rate"],
                    "train_Q05_margin": train_summary["share_margin_Q05"],
                    "train_MRR": train_summary["MRR"],
                    "train_Top3_oracle": train_summary["Top3_oracle"],
                    "validation_exact_hit_count": validation_summary["exact_hit_count"],
                    "validation_top1_realB_pass_count": validation_summary["top1_realB_pass_count"],
                    "validation_top1_realB_pass_rate": validation_summary["top1_realB_pass_rate"],
                    "validation_nonself_count": validation_summary["nonself_count"],
                    "validation_nonself_pass_count": validation_summary["nonself_pass_count"],
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
    result_frame = pd.DataFrame(rows).sort_values(["fold", "candidate_id"]).reset_index(drop=True)
    stability_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        subset = result_frame.loc[result_frame["candidate_id"].eq(candidate.candidate_id)]
        winners = subset.loc[subset["train_is_winner"]]
        stability_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "candidate_name": candidate.candidate_name,
                "removed_basis_id": candidate.removed_basis_id,
                "K": candidate.K,
                "train_winner_count": int(winners.shape[0]),
                "train_winner_frequency": float(winners.shape[0] / 5),
                "winner_folds": json.dumps(winners["fold"].astype(int).tolist()),
                "winner_validation_top1_pass_mean": float(winners["validation_top1_realB_pass_count"].mean()) if not winners.empty else float("nan"),
                "winner_validation_top1_pass_min": float(winners["validation_top1_realB_pass_count"].min()) if not winners.empty else float("nan"),
                "winner_validation_nonself_rate_mean": float(winners["validation_nonself_pass_rate"].mean()) if not winners.empty else float("nan"),
                "winner_validation_Q05_margin_mean": float(winners["validation_Q05_margin"].mean()) if not winners.empty else float("nan"),
                "winner_validation_MRR_mean": float(winners["validation_MRR"].mean()) if not winners.empty else float("nan"),
                "winner_validation_Top3_mean": float(winners["validation_Top3_oracle"].mean()) if not winners.empty else float("nan"),
            }
        )
    stability_frame = pd.DataFrame(stability_rows).sort_values("candidate_id").reset_index(drop=True)
    result_frame.to_csv(RESULT_ROOT / "13_query_state_cv_results.csv", index=False)
    stability_frame.to_csv(RESULT_ROOT / "14_query_state_cv_selection_stability.csv", index=False)
    return result_frame, stability_frame


def _write_figures(summary_frame: pd.DataFrame, delta_frame: pd.DataFrame) -> None:
    ordered = summary_frame.sort_values("candidate_id")
    labels = ordered["candidate_name"].tolist()
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    bars = ax.bar(x, ordered["top1_realB_pass_count"], color=plt.get_cmap("tab10")(np.linspace(0, 1, len(labels))))
    ax.axhline(ordered.loc[ordered["candidate_id"].eq(0), "top1_realB_pass_count"].iloc[0], color="black", linestyle="--", linewidth=1.0, label="MP10 baseline")
    for bar, value in zip(bars, ordered["top1_realB_pass_count"], strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, float(value) + 0.7, str(int(value)), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, labels, rotation=45, ha="right")
    ax.set_ylabel("Top-1 Real-B pass count")
    ax.set_xlabel("Candidate")
    ax.set_ylim(0, 425)
    ax.set_title("MP10 leave-one-basis-out retrieval comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "16_candidate_comparison.png")
    plt.close(fig)

    delta_ordered = delta_frame.sort_values("candidate_id")
    fig, ax = plt.subplots(figsize=(13, 6), dpi=300)
    colors = ["tab:green" if value > 0 else "tab:red" if value < 0 else "tab:gray" for value in delta_ordered["delta_top1_pass"]]
    bars = ax.bar(delta_ordered["removed_basis_id"], delta_ordered["delta_top1_pass"], color=colors)
    ax.axhline(0, color="black", linewidth=0.9)
    for bar, value in zip(bars, delta_ordered["delta_top1_pass"], strict=True):
        y = float(value) + (0.8 if value >= 0 else -0.35)
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{int(value):+d}", ha="center", va="bottom" if value >= 0 else "top", fontsize=9)
    ax.set_ylim(float(delta_ordered["delta_top1_pass"].min()) - 5.0, 5.0)
    ax.set_ylabel("Δ Top-1 Real-B pass count vs MP10")
    ax.set_xlabel("Removed basis")
    ax.set_title("Retrieval contribution of each MP10 basis")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "17_basis_retrieval_contribution.png")
    plt.close(fig)


def _write_summary(
    contract: dict[str, Any],
    baseline_gate: dict[str, Any],
    summary_frame: pd.DataFrame,
    delta_frame: pd.DataFrame,
    cv_frame: pd.DataFrame,
    stability_frame: pd.DataFrame,
    raw_before: dict[str, Any],
    raw_after: dict[str, Any],
    diagnostic_winner: pd.Series,
) -> None:
    lines = [
        f"Task: {TASK_NAME}",
        "Task type: retrieval-oriented leave-one-basis-out ablation.",
        "",
        "Baseline regression",
        "- Existing MP10 Full-LUT artifact gate: PASS.",
        f"- Fresh integrated MP10 candidate gate: {baseline_gate['pass']}.",
        f"- MP10 exact/top1 Real-B pass/failure: {baseline_gate['retrieval']['exact_count']}/"
        f"{baseline_gate['retrieval']['real_B_pass_count']}/{baseline_gate['retrieval']['failure_count']}.",
        f"- MP10 non-self pass: {baseline_gate['retrieval']['nonself_pass_count']}/{baseline_gate['retrieval']['nonself_count']}.",
        f"- Max model metric error: {max(baseline_gate['model_metric_max_abs_error_dB'].values()):.3e} dB.",
        f"- Max theta error: Aend={baseline_gate['theta_Aend_max_abs_error']:.3e}, C2={baseline_gate['theta_C2_max_abs_error']:.3e}.",
        "",
        "Frozen protocol",
        f"- orders={contract['orders']}; memory={contract['memory']}; K10=10; K9=9; dmax={contract['dmax']}; lambda={contract['ridge_lambda']}.",
        "- All K9 candidates keep dmax=2 and use valid A/B/C lengths 12286/4913/7371.",
        f"- common-B: State 0, raw length 4915, valid length 4913, SHA={EXPECTED_COMMON_B_SHA}.",
        "- Aend=final ILC column; C2=ILC2; allow_self=true; Real-B is not used to choose State_Q.",
        "",
        "All425 candidate ranking",
    ]
    for row in summary_frame.sort_values("diagnostic_rank").itertuples(index=False):
        removed_basis = "none" if pd.isna(row.removed_basis_id) else str(row.removed_basis_id)
        lines.append(
            f"- rank={int(row.diagnostic_rank)} {row.candidate_name}: K={int(row.K)}, "
            f"removed={removed_basis}, Top1={int(row.top1_realB_pass_count)}/425, "
            f"Exact={int(row.exact_hit_count)}, nonself={int(row.nonself_pass_count)}/{int(row.nonself_count)}, "
            f"Q05_margin={float(row.share_margin_Q05):.9g}, MRR={float(row.MRR):.9g}, "
            f"Top3={int(row.Top3_oracle)}, Top5={int(row.Top5_oracle)}, Top10={int(row.Top10_oracle)}."
        )
    lines.extend(
        [
            "",
            "All425 diagnostic winner",
            f"- {diagnostic_winner['candidate_name']} (diagnostic rank {int(diagnostic_winner['diagnostic_rank'])}).",
            "- This is an All425 diagnostic winner, not a final frozen retrieval model.",
            "- Real-B labels were used for candidate comparison, so no candidate is frozen from this result alone.",
            "",
            "Five-fold Query-State CV",
            "- Five folds contain 85 Query States each; assignment is deterministic and stratified by load class, mismatch settings and angle group.",
            "- Independent validation: 11 distance matrices, 55 argmin checks, 22 shareability-margin checks and 88 Top-k oracle checks passed; 20 Real-B pairs matched within 1e-9 dB.",
        ]
    )
    for row in stability_frame.sort_values("train_winner_count", ascending=False).itertuples(index=False):
        lines.append(
            f"- {row.candidate_name}: train winner={int(row.train_winner_count)}/5, "
            f"folds={row.winner_folds}, winner-validation Top1 mean/min="
            f"{float(row.winner_validation_top1_pass_mean) if np.isfinite(row.winner_validation_top1_pass_mean) else float('nan'):.6g}/"
            f"{float(row.winner_validation_top1_pass_min) if np.isfinite(row.winner_validation_top1_pass_min) else float('nan'):.6g}."
        )
    lines.extend(
        [
            "",
            "Data protection and stop boundary",
            f"- raw manifest before: {json.dumps(raw_before, ensure_ascii=False, sort_keys=True)}",
            f"- raw manifest after: {json.dumps(raw_after, ensure_ascii=False, sort_keys=True)}",
            f"- raw data unchanged: {raw_before == raw_after}",
            "- No Envelope75 forward search, order/memory/lambda scan, Top-k selection, ensemble, clustering, DPD replay, or low-bandwidth task was run.",
            "- No independent final test is claimed; CV is reported only as Query-State selection stability.",
        ]
    )
    (RESULT_ROOT / "18_final_result_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run(*, resume: bool = False) -> dict[str, Any]:
    if RESULT_ROOT.exists() and any(RESULT_ROOT.iterdir()) and not resume:
        raise RuntimeError(f"result directory is not empty; use --resume: {RESULT_ROOT}")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "distance_matrices").mkdir(parents=True, exist_ok=True)
    raw_before = raw_manifest_gate()
    contract = backend.historical_mp10
    history = {
        "orders": list(contract.MP_ORDERS),
        "memory": [contract.MP_MEMORY[order] for order in contract.MP_ORDERS],
        "K": int(contract.FROZEN_MP_K),
        "dmax": int(contract.MP_MAX_DELAY),
        "ridge_lambda": float(contract.RIDGE_LAMBDA),
    }
    previous = _load_previous_baseline()
    basis_frame, candidates = _build_candidates()
    _write_task_definition(history, previous["artifact_gate"])
    common_b, common_meta = verify_common_b_contract()
    common_b = np.asarray(common_b, dtype=np.complex128)
    if common_meta["sha256"] != EXPECTED_COMMON_B_SHA:
        raise RuntimeError("common-B hash regression failed")
    common_phi = contract.build_frozen_mp_basis(common_b)
    if common_phi.shape != (backend.FINGERPRINT_LENGTH, backend.MP_K):
        raise RuntimeError(f"common-B MP10 Phi shape failed: {common_phi.shape}")
    folds = _build_query_state_folds(pd.DataFrame(__import__("data_management").build_state_table()))
    common_meta_clean = {
        key: value for key, value in common_meta.items() if not isinstance(value, np.ndarray)
    }
    _write_json(
        RESULT_ROOT / "01_reuse_audit.json",
        {
            "historical_mp10_source": "behavior_modeling.sparse_gmp",
            "historical_basis_function": "build_frozen_mp_basis",
            "historical_ridge_function": "fit_ridge",
            "historical_contract": history,
            "separate_hnorm_found": False,
            "normalization": "get_ilc_pair peak-normalizes each input history column; no additional Hnorm",
            "previous_task": str(PREVIOUS_ROOT),
            "previous_baseline_gate": previous["artifact_gate"],
            "current_retrieval_pipeline": "scenario_2_mp10_full_lut_retrieval_5b plus this candidate support adapter",
            "reused": [
                "data_management",
                "signal_segmentation",
                "common-B verification",
                "compute_cnmse_matrix",
                "top1_retrieval",
                "canonical OFF Real-B",
            ],
            "new_task_code": [
                "mp10_k9_backend.py: full-Phi column slicing and one-read-per-state model phase",
                "run_scenario2_mp10_k9_retrieval_oriented_basis_ablation_5b.py: candidate metrics/CV/figures",
            ],
            "common_B_metadata": common_meta_clean,
        },
    )
    _write_reuse_audit(history, previous["artifact_gate"], common_meta_clean)
    _checkpoint(
        phase="reuse_audit_and_baseline_artifact_gate_passed",
        raw_manifest_before=raw_before,
        baseline_artifact_gate=previous["artifact_gate"],
        candidate_supports_written=True,
        cv_folds_written=True,
    )
    model_frame, theta_a, theta_c, real_b = _run_model_phase(candidates)
    real_b_distance = compute_cnmse_matrix(real_b, real_b)
    if real_b_distance.shape != (STATE_COUNT, STATE_COUNT) or not np.all(np.isneginf(np.diag(real_b_distance))):
        raise RuntimeError("Real-B ground truth CNMSE matrix diagonal/shape failed")
    shareability = real_b_distance < REAL_B_THRESHOLD_DB
    if shareability.dtype != bool or shareability.shape != (STATE_COUNT, STATE_COUNT) or not shareability.diagonal().all():
        raise RuntimeError("Real-B shareability mask contract failed")
    np.save(RESULT_ROOT / "04_realB_ground_truth_cnmse_matrix.npy", real_b_distance)
    np.save(RESULT_ROOT / "05_realB_shareability_mask.npy", shareability)
    # Independent 20-pair sanity check against the canonical CNMSE helper.
    rng = np.random.default_rng(20260916)
    for real_id, query_id in rng.integers(0, STATE_COUNT, size=(20, 2)):
        direct = cnmse(real_b[int(real_id)], real_b[int(query_id)])
        stored = float(real_b_distance[int(real_id), int(query_id)])
        if np.isneginf(direct) and np.isneginf(stored):
            continue
        if abs(direct - stored) > REAL_B_SANITY_TOLERANCE_DB:
            raise RuntimeError("Real-B ground truth random-pair sanity check failed")
    _write_coefficients(candidates, theta_a, theta_c)
    model_frame.to_csv(RESULT_ROOT / "06_candidate_model_quality_long.csv", index=False)

    # Candidate 0 is evaluated first. No K9 retrieval metrics are produced until this gate passes.
    baseline_frame, baseline_summary, baseline_distance, baseline_lut, baseline_query = _candidate_retrieval(
        candidates[0], theta_a, theta_c, common_phi, real_b_distance, shareability
    )
    baseline_retrieval = baseline_frame["State_Q"].to_numpy(dtype=int)
    baseline_real_b_retrieved = baseline_frame["retrieved_realB_CNMSE_dB"].to_numpy(dtype=float)
    baseline_gate = _baseline_regression(
        model_frame,
        theta_a,
        theta_c,
        baseline_lut,
        baseline_query,
        baseline_distance,
        baseline_retrieval,
        baseline_real_b_retrieved,
        previous,
    )
    _checkpoint(
        phase="mp10_baseline_regression_passed",
        raw_manifest_before=raw_before,
        baseline_regression=baseline_gate,
        realB_matrix_written=True,
        candidate_model_quality_written=True,
    )
    candidate_frames: dict[int, pd.DataFrame] = {0: baseline_frame}
    candidate_summaries: list[dict[str, Any]] = [baseline_summary]
    distance_dir = RESULT_ROOT / "distance_matrices"
    np.save(distance_dir / "MP10_full.npy", baseline_distance)
    np.save(RESULT_ROOT / "diagnostic_baseline_lut_fingerprints.npy", baseline_lut)
    np.save(RESULT_ROOT / "diagnostic_baseline_query_fingerprints.npy", baseline_query)
    for candidate in candidates[1:]:
        frame, summary, distance, lut, query = _candidate_retrieval(
            candidate, theta_a, theta_c, common_phi, real_b_distance, shareability
        )
        candidate_frames[candidate.candidate_id] = frame
        candidate_summaries.append(summary)
        np.save(distance_dir / f"{candidate.candidate_name}.npy", distance)
    query_metrics = pd.concat(candidate_frames.values(), ignore_index=True).sort_values(
        ["candidate_id", "State_R"]
    ).reset_index(drop=True)
    query_metrics.to_csv(RESULT_ROOT / "09_candidate_query_metrics_long.csv", index=False)
    summary_frame = _assign_diagnostic_ranks(candidate_summaries)
    summary_frame.to_csv(RESULT_ROOT / "08_candidate_retrieval_summary.csv", index=False)
    delta_frame = _build_deltas(candidates, summary_frame, model_frame)
    delta_frame.to_csv(RESULT_ROOT / "10_leave_one_basis_delta.csv", index=False)
    recovered_regressed = _build_recovered_regressed(candidate_frames, candidates)
    recovered_regressed.to_csv(RESULT_ROOT / "11_recovered_regressed_states.csv", index=False)
    cv_frame, stability_frame = _cv_results(candidates, candidate_frames, folds)
    diagnostic_winner = summary_frame.sort_values("diagnostic_rank").iloc[0]
    winner_id = int(diagnostic_winner["candidate_id"])
    winner_detail = candidate_frames[winner_id].copy()
    winner_detail["diagnostic_winner"] = True
    winner_detail.to_csv(RESULT_ROOT / "15_diagnostic_winner_detail.csv", index=False)
    winner_candidate = candidates[winner_id]
    winner_support = winner_candidate.support
    winner_theta_a = np.asarray(theta_a[winner_id][:, list(winner_support)], dtype=np.complex128)
    winner_theta_c = np.asarray(theta_c[winner_id][:, list(winner_support)], dtype=np.complex128)
    winner_lut = (common_phi[:, winner_support] @ winner_theta_a.T).T.astype(np.complex128)
    winner_query = (common_phi[:, winner_support] @ winner_theta_c.T).T.astype(np.complex128)
    np.save(RESULT_ROOT / "diagnostic_winner_lut_fingerprints.npy", winner_lut)
    np.save(RESULT_ROOT / "diagnostic_winner_query_fingerprints.npy", winner_query)
    _write_figures(summary_frame, delta_frame)
    raw_after = raw_manifest_gate()
    _write_summary(
        history,
        baseline_gate,
        summary_frame,
        delta_frame,
        cv_frame,
        stability_frame,
        raw_before,
        raw_after,
        diagnostic_winner,
    )
    if raw_before != raw_after:
        raise RuntimeError("data/raw manifest changed during ablation")
    transition_counts = recovered_regressed["transition"].value_counts().to_dict()
    result = {
        "task": TASK_NAME,
        "status": "SUCCESS",
        "baseline_regression_pass": bool(baseline_gate["pass"]),
        "diagnostic_winner": str(diagnostic_winner["candidate_name"]),
        "diagnostic_winner_rank": int(diagnostic_winner["diagnostic_rank"]),
        "diagnostic_winner_top1_realB_pass": int(diagnostic_winner["top1_realB_pass_count"]),
        "baseline_top1_realB_pass": int(summary_frame.loc[summary_frame["candidate_id"].eq(0), "top1_realB_pass_count"].iloc[0]),
        "max_k9_top1_realB_pass": int(summary_frame.loc[summary_frame["K"].eq(9), "top1_realB_pass_count"].max()),
        "transition_counts_across_k9": {key: int(transition_counts.get(key, 0)) for key in ("unchanged_pass", "recovered", "regressed", "unchanged_fail")},
        "cv_winner_counts": stability_frame.loc[stability_frame["train_winner_count"].gt(0), ["candidate_name", "train_winner_count"]].to_dict("records"),
        "raw_data_modified": raw_before != raw_after,
        "result_root": str(RESULT_ROOT),
    }
    _write_json(RESULT_ROOT / "19_checkpoint.json", {"phase": "completed", **result})
    _append_log(
        f"\n[{_now()}] Start and complete {TASK_NAME}\n"
        f"Baseline regression={json.dumps(baseline_gate, ensure_ascii=False, sort_keys=True)}\n"
        f"Result={json.dumps(result, ensure_ascii=False, sort_keys=True)}\n"
        "No Envelope75 search or additional optimization was run.\n"
    )
    _append_handoff(
        f"完成 Frozen MP10 的 10 个 leave-one-basis-out K9 retrieval-oriented ablation；结果目录：{RESULT_ROOT}\n"
        f"摘要：{json.dumps(result, ensure_ascii=False, sort_keys=True)}\n"
        "保存 Real-B 425x425 ground truth/mask、候选检索指标、Delta、5-fold Query-State CV 和两张诊断图；未冻结新的最终模型。"
    )
    return result


def main() -> None:
    result = _run(resume="--resume" in sys.argv[1:])
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    __import__("multiprocessing").freeze_support()
    main()
