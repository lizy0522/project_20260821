# ruff: noqa: E402,E501,I001

"""Run the Self-First joint Envelope75 support/Ridge LUT search.

The runner is intentionally resumable.  It first re-ranks available historical
``(support, lambda)`` artifacts, then calibrates a shared Ridge lambda and
performs Beam/Forward/Backward/Swap search under the Self-First objective.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections.abc import Mapping, Sequence
from collections import defaultdict
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
from behavior_fingerprint_ranking_consistency.shared.distance_matrix import (  # noqa: E402
    compute_cnmse_distance_matrix,
)
from data_management.shared import build_state_table  # noqa: E402
from retrieval_oriented_model_selection.self_hit_oriented.scenario_2.scenario_2_self_first_joint_basis_ridge_lut_retrieval_search_5b.historical_registry import build_registry  # noqa: E402
from retrieval_oriented_model_selection.self_hit_oriented.scenario_2.scenario_2_self_first_joint_basis_ridge_lut_retrieval_search_5b.self_first_backend import (  # noqa: E402
    DMAX,
    model_worker_entry,
    worker_init,
)
from retrieval_oriented_model_selection.self_hit_oriented.scenario_2.scenario_2_self_first_joint_basis_ridge_lut_retrieval_search_5b.self_first_metrics import (  # noqa: E402
    REAL_B_THRESHOLD_DB,
    canonical_support_hash,
    evaluate_distance,
    lexicographic_key,
    rank_frame,
    sha256_json,
    transition_matrix,
)

TASK_NAME = "scenario_2_self_first_joint_basis_ridge_lut_retrieval_search_5b"
RESULT_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /TASK_NAME
WORK_LOG = PROJECT_ROOT / "work_logs" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /TASK_NAME / "execution_log.txt"
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"

STATE_COUNT = 425
BEAM_WIDTH = 5
MAX_FORWARD_BACKWARD_ROUNDS = 6
MAX_K = 18
WORKER_COUNT = 10
BLAS_THREADS_PER_WORKER = 1
OBJECTIVE_VERSION = "SELF_FIRST_V1"
COMMON_B_SOURCE_STATE = 0
COMMON_B_RAW_LENGTH = 4_915
COMMON_B_VALID_LENGTH = 4_913
TIE_TOLERANCE_DB = 1e-12

COARSE_LAMBDA_GRID = (
    0.0,
    1e-14,
    1e-12,
    1e-10,
    1e-8,
    1e-6,
    1e-4,
    1e-2,
    1.0,
    1e2,
)
LAMBDA_BOUNDARY_MIN = 1e-16
LAMBDA_BOUNDARY_MAX = 1e6

K9_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /"scenario_2_mp10_k9_retrieval_oriented_basis_ablation_5b"
REAL_B_DISTANCE_PATH = K9_ROOT / "04_realB_ground_truth_cnmse_matrix.npy"
SHAREABILITY_PATH = K9_ROOT / "05_realB_shareability_mask.npy"


@dataclass(frozen=True)
class Candidate:
    """One support/lambda retrieval model."""

    candidate_id: int
    candidate_name: str
    support_hash: str
    basis_ids: tuple[str, ...]
    support_indices: tuple[int, ...]
    lambda_value: float
    operation: str = "HISTORICAL"
    parent_support_hash: str = ""
    parent_candidate_name: str = ""
    source_task: str = ""

    @property
    def K(self) -> int:
        return len(self.basis_ids)

    def to_record(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "candidate_name": self.candidate_name,
            "support_hash": self.support_hash,
            "basis_ids": json.dumps(list(self.basis_ids), ensure_ascii=False),
            "support_indices": json.dumps(list(self.support_indices)),
            "K": self.K,
            "lambda": self.lambda_value,
            "operation": self.operation,
            "parent_support_hash": self.parent_support_hash,
            "parent_candidate_name": self.parent_candidate_name,
            "source_task": self.source_task,
        }


@dataclass
class CandidateEvaluation:
    """In-memory result needed for Beam transitions."""

    candidate: Candidate
    summary: dict[str, Any]
    state_frame: pd.DataFrame
    classes: np.ndarray
    distance_path: Path | None = None


@dataclass
class RuntimeContext:
    """Frozen inputs and hashes shared by all phases."""

    terms: tuple[EnvelopeBasis, ...]
    common_b: np.ndarray
    common_phi: np.ndarray
    real_b_distance: np.ndarray
    shareability: np.ndarray
    state_table: pd.DataFrame
    contract: dict[str, Any]


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


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


def _append_log(message: str) -> None:
    WORK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with WORK_LOG.open("a", encoding="utf-8") as handle:
        handle.write(message.rstrip() + "\n")


def _append_handoff(message: str) -> None:
    HANDOFF_LOG.parent.mkdir(parents=True, exist_ok=True)
    with HANDOFF_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{_now()} | 新任务：{TASK_NAME}\n{message.rstrip()}\n")


def _checkpoint(context: RuntimeContext, *, phase: str, **payload: Any) -> None:
    base = {
        "task_name": TASK_NAME,
        "objective_version": OBJECTIVE_VERSION,
        "phase": phase,
        "state_count": STATE_COUNT,
        "worker_count": WORKER_COUNT,
        "blas_threads_per_worker": BLAS_THREADS_PER_WORKER,
        "abc_bounds": list(context.contract["abc_bounds"]),
        "dmax": DMAX,
        "Aend_definition": "last_valid_ilc_column",
        "C2_definition": "literal_ilc_column_2",
        "allow_self": True,
        "real_B_threshold_dB": REAL_B_THRESHOLD_DB,
        "common_B_source_state": COMMON_B_SOURCE_STATE,
        "common_B_sha256": context.contract["common_B_sha256"],
        "raw_manifest": context.contract["raw_manifest"],
        "real_B_matrix_sha256": context.contract["real_B_matrix_sha256"],
        "envelope75_dictionary_sha256": context.contract["envelope75_dictionary_sha256"],
        "cv_fold_sha256": context.contract["cv_fold_sha256"],
        "lambda_grid_sha256": context.contract["lambda_grid_sha256"],
    }
    _write_json(RESULT_ROOT / "23_checkpoint.json", {**base, **payload})


def _load_checkpoint() -> dict[str, Any] | None:
    path = RESULT_ROOT / "23_checkpoint.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _verify_checkpoint(context: RuntimeContext) -> dict[str, Any] | None:
    checkpoint = _load_checkpoint()
    if checkpoint is None:
        return None
    expected = {
        "objective_version": OBJECTIVE_VERSION,
        "state_count": STATE_COUNT,
        "abc_bounds": list(context.contract["abc_bounds"]),
        "dmax": DMAX,
        "Aend_definition": "last_valid_ilc_column",
        "C2_definition": "literal_ilc_column_2",
        "allow_self": True,
        "real_B_threshold_dB": REAL_B_THRESHOLD_DB,
        "common_B_sha256": context.contract["common_B_sha256"],
        "raw_manifest": context.contract["raw_manifest"],
        "real_B_matrix_sha256": context.contract["real_B_matrix_sha256"],
        "envelope75_dictionary_sha256": context.contract["envelope75_dictionary_sha256"],
        "cv_fold_sha256": context.contract["cv_fold_sha256"],
        "lambda_grid_sha256": context.contract["lambda_grid_sha256"],
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise RuntimeError(f"resume contract HARD FAIL: {key} changed")
    return checkpoint


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_real_b() -> tuple[np.ndarray, np.ndarray, str]:
    if not REAL_B_DISTANCE_PATH.is_file() or not SHAREABILITY_PATH.is_file():
        raise FileNotFoundError("historical Real-B matrix or shareability mask is missing")
    real_b_distance = np.asarray(np.load(REAL_B_DISTANCE_PATH, allow_pickle=False), dtype=np.float64)
    shareability = np.asarray(np.load(SHAREABILITY_PATH, allow_pickle=False), dtype=bool)
    if real_b_distance.shape != (STATE_COUNT, STATE_COUNT):
        raise RuntimeError(f"Real-B distance shape changed: {real_b_distance.shape}")
    if shareability.shape != real_b_distance.shape:
        raise RuntimeError("Real-B shareability mask shape changed")
    if not np.all(np.isneginf(np.diag(real_b_distance))):
        raise RuntimeError("Real-B distance diagonal is not -Inf")
    if not shareability.diagonal().all():
        raise RuntimeError("Real-B shareability diagonal must be True")
    return real_b_distance, shareability, _file_sha256(REAL_B_DISTANCE_PATH)


def _load_cv_folds() -> pd.DataFrame:
    path = K9_ROOT / "12_query_state_cv_folds.csv"
    if not path.is_file():
        raise FileNotFoundError(f"fixed Query-State fold file is missing: {path}")
    frame = pd.read_csv(path).sort_values("state_id").reset_index(drop=True)
    required = {"state_id", "fold"}
    if not required.issubset(frame.columns) or frame.shape[0] != STATE_COUNT:
        raise RuntimeError("historical Query-State fold contract is invalid")
    if frame["fold"].value_counts().sort_index().to_dict() != {0: 85, 1: 85, 2: 85, 3: 85, 4: 85}:
        raise RuntimeError("Query-State folds are not five balanced 85-state folds")
    return frame[["state_id", "fold"]]


def _build_context() -> RuntimeContext:
    terms = tuple(build_envelope_dictionary())
    if len(terms) != 75:
        raise RuntimeError(f"Envelope75 dictionary count changed: {len(terms)}")
    dictionary_gate_result = dictionary_gate()
    if not dictionary_gate_result.get("pass"):
        raise RuntimeError("Envelope75 dictionary gate failed")
    common_b, common_meta = verify_common_b_contract()
    if common_b.shape != (COMMON_B_RAW_LENGTH,) or not np.all(np.isfinite(common_b)):
        raise RuntimeError("common-B raw contract failed")
    if str(common_meta.get("sha256")) != EXPECTED_COMMON_B_SHA:
        raise RuntimeError("common-B SHA256 contract failed")
    common_phi = build_envelope_bank(common_b, terms)
    if common_phi.shape != (COMMON_B_VALID_LENGTH, 75):
        raise RuntimeError(f"common-B Envelope75 bank shape changed: {common_phi.shape}")
    real_b_distance, shareability, real_b_hash = _load_real_b()
    state_table = pd.DataFrame(build_state_table()).sort_values("state_id").reset_index(drop=True)
    if state_table.shape[0] != STATE_COUNT or not np.array_equal(state_table["state_id"], np.arange(STATE_COUNT)):
        raise RuntimeError("state table contract is not canonical 0...424")
    cv_folds = _load_cv_folds()
    raw_manifest = raw_manifest_gate()
    contract = {
        "raw_manifest": raw_manifest,
        "common_B_sha256": str(common_meta["sha256"]),
        "real_B_matrix_sha256": real_b_hash,
        "envelope75_dictionary_sha256": canonical_support_hash(tuple(term.basis_id for term in terms)),
        "cv_fold_sha256": sha256_json(cv_folds.to_dict("records")),
        "lambda_grid_sha256": sha256_json(list(COARSE_LAMBDA_GRID)),
        "abc_bounds": (0, 12_288, 17_203, 24_576),
        "common_B_raw_length": COMMON_B_RAW_LENGTH,
        "common_B_valid_length": COMMON_B_VALID_LENGTH,
        "real_B_matrix_path": str(REAL_B_DISTANCE_PATH),
    }
    return RuntimeContext(
        terms=terms,
        common_b=np.asarray(common_b, dtype=np.complex128),
        common_phi=np.asarray(common_phi, dtype=np.complex128),
        real_b_distance=real_b_distance,
        shareability=shareability,
        state_table=state_table,
        contract=contract,
    )


def _write_task_definition(context: RuntimeContext) -> None:
    lines = [
        f"Task: {TASK_NAME}",
        "Mode: long Self-First joint support/Ridge LUT retrieval search.",
        f"OBJECTIVE_VERSION={OBJECTIVE_VERSION}",
        "Primary objective: maximize Top-1 SELF retrieval; shareable Real-B fallback is secondary.",
        "Model candidate is the joint pair (support, shared lambda), not support alone.",
        "Retrieval distance D selects State_Q; Real-B G labels a non-self State_Q as SHAREABLE_FALLBACK.",
        "Three classes: SELF, SHAREABLE_FALLBACK, FAIL.",
        "Ranking: N_self, N_valid, self_Q05, self_MRR, fallback_Q05, fallback_MRR, Self_Top3, smaller K, numerical tie-break.",
        "SELF->FALLBACK is a regression even when N_valid is unchanged; every edge stores the full 3x3 transition.",
        f"States=0...{STATE_COUNT - 1}; A/B/C={context.contract['abc_bounds']}; dmax={DMAX}; allow_self=True; Real-B threshold={REAL_B_THRESHOLD_DB} dB.",
        "Aend=last valid ILC column; C2=literal ILC column 2; common-B=State 0; Envelope75 is the only candidate dictionary.",
        f"Shared lambda coarse grid={list(COARSE_LAMBDA_GRID)}; active set max=7; boundary range={LAMBDA_BOUNDARY_MIN}...{LAMBDA_BOUNDARY_MAX}.",
        f"Parallelism: {WORKER_COUNT} spawn workers, {BLAS_THREADS_PER_WORKER} BLAS/OpenBLAS thread per worker, no nested multiprocessing.",
        "Query-State CV is 5 folds of 340 train Query / 85 validation Query and is not called an independent test.",
        "Raw data and common-B hashes are frozen; any resume contract change is HARD FAIL.",
        "No DPD replay, clustering, Type-III, or low-bandwidth nB processing is part of this task.",
        "",
        f"Raw manifest: {json.dumps(context.contract['raw_manifest'], ensure_ascii=False, sort_keys=True)}",
        f"Common-B metadata: {json.dumps(context.contract, ensure_ascii=False, sort_keys=True, default=_json_default)}",
    ]
    (RESULT_ROOT / "00_task_definition.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_fallback_availability(context: RuntimeContext) -> None:
    share = context.shareability.copy()
    np.fill_diagonal(share, False)
    rows: list[dict[str, Any]] = []
    for state_id in range(STATE_COUNT):
        ids = np.flatnonzero(share[state_id]).astype(int).tolist()
        rows.append(
            {
                "State_R": state_id,
                "fallback_available": bool(ids),
                "fallback_candidate_count": len(ids),
                "fallback_candidate_ids": json.dumps(ids),
                "self_only": not bool(ids),
            }
        )
    pd.DataFrame(rows).to_csv(RESULT_ROOT / "04_fallback_availability.csv", index=False)


def _write_reuse_audit(context: RuntimeContext, registry: pd.DataFrame, audit: dict[str, Any]) -> None:
    available = registry.loc[registry["distance_matrix_available"].astype(bool)]
    unavailable = registry.loc[~registry["distance_matrix_available"].astype(bool)]
    lines = [
        f"Task: {TASK_NAME}",
        f"OBJECTIVE_VERSION={OBJECTIVE_VERSION}",
        "Historical supports are candidates only; all historical ranking is invalidated and recomputed under Self-First.",
        f"Registry rows={registry.shape[0]}; unique supports={registry['support_hash'].nunique()}; distance-matrix candidates={available.shape[0]}; metadata-only candidates={unavailable.shape[0]}.",
        "Available 425x425 matrices are reused without re-Ridge. Candidates without complete artifacts remain explicitly unavailable.",
        f"Result roots inspected: {json.dumps(audit.get('result_roots_seen', []), ensure_ascii=False)}",
        f"Summary rows inspected: {audit.get('summary_rows_seen', 0)}",
        f"Raw manifest: {json.dumps(context.contract['raw_manifest'], ensure_ascii=False, sort_keys=True)}",
        f"Real-B matrix: {REAL_B_DISTANCE_PATH}; SHA256={context.contract['real_B_matrix_sha256']}",
        f"Common-B SHA256={context.contract['common_B_sha256']}; Envelope75 SHA256={context.contract['envelope75_dictionary_sha256']}",
        "",
        "Metadata-only/unavailable notes:",
        *[f"- {item}" for item in audit.get("unavailable_notes", [])],
        "",
        "The current script-only or incomplete task roots are not treated as measured historical candidates until result artifacts exist.",
    ]
    (RESULT_ROOT / "01_reuse_audit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _candidate_from_registry(row: Mapping[str, Any], candidate_id: int, *, operation: str = "HISTORICAL") -> Candidate:
    ids = tuple(json.loads(str(row["basis_ids"])))
    indices = tuple(int(next(term.index for term in build_envelope_dictionary() if term.basis_id == basis_id)) for basis_id in ids)
    return Candidate(
        candidate_id=candidate_id,
        candidate_name=str(row.get("source_model_name", row.get("support_id", f"candidate_{candidate_id}"))).split(";")[0],
        support_hash=str(row["support_hash"]),
        basis_ids=ids,
        support_indices=indices,
        lambda_value=float(row["lambda"]),
        operation=operation,
        source_task=str(row.get("source_task", "")),
    )


def _build_historical_ranking(context: RuntimeContext, registry: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[int, CandidateEvaluation]]:
    summaries: list[dict[str, Any]] = []
    classes_by_key: dict[str, np.ndarray] = {}
    evaluations: dict[int, CandidateEvaluation] = {}
    detail_frames: list[pd.DataFrame] = []
    for _, row in registry.iterrows():
        path_text = str(row.get("distance_matrix_path", ""))
        if not path_text or path_text.lower() in {"nan", "none"}:
            continue
        path = Path(path_text)
        if not path.is_file():
            continue
        candidate = _candidate_from_registry(row, int(row["candidate_id"]))
        try:
            distance = np.asarray(np.load(path, allow_pickle=False, mmap_mode="r"), dtype=np.float64)
            summary, state_frame_dict, classes = evaluate_distance(
                distance,
                context.real_b_distance,
                context.shareability,
                candidate={
                    "candidate_id": candidate.candidate_id,
                    "candidate_name": candidate.candidate_name,
                    "support_hash": candidate.support_hash,
                    "K": candidate.K,
                    "lambda": candidate.lambda_value,
                },
            )
        except (OSError, ValueError, RuntimeError) as exc:
            _append_log(f"[{_now()}] Historical candidate skipped: {path}: {type(exc).__name__}: {exc}")
            continue
        for key, value in row.to_dict().items():
            if key not in summary:
                summary[key] = value
        summary["candidate_key"] = f"{candidate.support_hash}|{candidate.lambda_value:.17g}"
        summaries.append(summary)
        classes_by_key[summary["candidate_key"]] = classes
        detail = pd.DataFrame(state_frame_dict)
        detail.insert(0, "candidate_key", summary["candidate_key"])
        detail.insert(1, "candidate_name", candidate.candidate_name)
        detail.insert(2, "support_hash", candidate.support_hash)
        detail.insert(3, "K", candidate.K)
        detail.insert(4, "lambda", candidate.lambda_value)
        detail_frames.append(detail)
        evaluations[candidate.candidate_id] = CandidateEvaluation(candidate, summary, detail, classes, path)
    if not summaries:
        raise RuntimeError("no historical 425x425 distance matrix is available for Self-First ranking")
    ranking = rank_frame(pd.DataFrame(summaries))
    ranking.to_csv(RESULT_ROOT / "03_historical_self_first_ranking.csv", index=False)
    if detail_frames:
        pd.concat(detail_frames, ignore_index=True).to_csv(RESULT_ROOT / "historical_self_first_query_metrics.csv", index=False)
    top = ranking.head(20)
    lines = [
        "Historical Self-First Top20:",
        "rank,candidate_name,K,lambda,N_self,N_fallback,N_valid,N_fail,self_Q05,self_MRR,fallback_Q05,fallback_MRR,Self_Top3",
    ]
    for item in top.to_dict("records"):
        lines.append(
            ",".join(
                str(item.get(key, ""))
                for key in (
                    "rank",
                    "candidate_name",
                    "K",
                    "lambda",
                    "N_self",
                    "N_fallback",
                    "N_valid",
                    "N_fail",
                    "self_Q05",
                    "self_MRR",
                    "fallback_Q05",
                    "fallback_MRR",
                    "Self_Top3",
                )
            )
        )
    (RESULT_ROOT / "historical_top20.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("[SELF-FIRST HISTORICAL TOP20]", flush=True)
    for line in lines[2:]:
        print(line, flush=True)
    return ranking, classes_by_key, evaluations


def _support_indices(ids: Sequence[str], terms: Sequence[EnvelopeBasis]) -> tuple[int, ...]:
    by_id = {term.basis_id: term.index for term in terms}
    missing = [basis_id for basis_id in ids if basis_id not in by_id]
    if missing:
        raise RuntimeError(f"support contains basis IDs outside Envelope75: {missing}")
    return tuple(by_id[basis_id] for basis_id in ids)


def _candidate_from_support(
    candidate_id: int,
    support_ids: Sequence[str],
    lambda_value: float,
    *,
    name: str,
    operation: str,
    context: RuntimeContext,
    parent: Candidate | None = None,
    source_task: str = "",
) -> Candidate:
    ids = tuple(support_ids)
    return Candidate(
        candidate_id=candidate_id,
        candidate_name=name,
        support_hash=canonical_support_hash(ids),
        basis_ids=ids,
        support_indices=_support_indices(ids, context.terms),
        lambda_value=float(lambda_value),
        operation=operation,
        parent_support_hash=parent.support_hash if parent else "",
        parent_candidate_name=parent.candidate_name if parent else "",
        source_task=source_task,
    )


def _phase_manifest(candidates: Sequence[Candidate]) -> str:
    return sha256_json(
        [
            {
                "candidate_id": c.candidate_id,
                "support_hash": c.support_hash,
                "basis_ids": list(c.basis_ids),
                "lambda": c.lambda_value,
            }
            for c in candidates
        ]
    )


def _load_state_cache(path: Path, manifest: str, candidate_count: int, max_k: int) -> dict[str, np.ndarray] | None:
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            if str(np.asarray(data["manifest"]).reshape(-1)[0]) != manifest:
                return None
            if np.asarray(data["candidate_ids"]).shape != (candidate_count,):
                return None
            if np.asarray(data["theta_Aend"]).shape != (candidate_count, max_k):
                return None
            return {key: np.asarray(data[key]) for key in data.files}
    except (OSError, ValueError, KeyError):
        return None


def _save_state_cache(path: Path, manifest: str, candidates: Sequence[Candidate], item: dict[str, Any], max_k: int) -> None:
    candidate_count = len(candidates)
    theta_a = np.zeros((candidate_count, max_k), dtype=np.complex128)
    theta_c = np.zeros_like(theta_a)
    fields = {
        "Y_Aend_train_NMSE_dB": np.zeros(candidate_count, dtype=float),
        "Y_Aend_B_NMSE_dB": np.zeros(candidate_count, dtype=float),
        "Y_C2_train_NMSE_dB": np.zeros(candidate_count, dtype=float),
        "Y_C2_B_NMSE_dB": np.zeros(candidate_count, dtype=float),
        "Aend_rank": np.zeros(candidate_count, dtype=np.int64),
        "C2_rank": np.zeros(candidate_count, dtype=np.int64),
        "Aend_condition_number": np.zeros(candidate_count, dtype=float),
        "C2_condition_number": np.zeros(candidate_count, dtype=float),
        "Aend_coefficient_norm": np.zeros(candidate_count, dtype=float),
        "C2_coefficient_norm": np.zeros(candidate_count, dtype=float),
        "Aend_finite": np.zeros(candidate_count, dtype=bool),
        "C2_finite": np.zeros(candidate_count, dtype=bool),
    }
    by_id = {int(value["candidate_id"]): value for value in item["candidates"]}
    for index, candidate in enumerate(candidates):
        fit = by_id[candidate.candidate_id]
        theta_a[index, : candidate.K] = np.asarray(fit["theta_Aend"], dtype=np.complex128)
        theta_c[index, : candidate.K] = np.asarray(fit["theta_C2"], dtype=np.complex128)
        for key in fields:
            fields[key][index] = fit[key]
    payload = {
        "manifest": np.asarray(manifest),
        "candidate_ids": np.asarray([c.candidate_id for c in candidates], dtype=np.int64),
        "theta_Aend": theta_a,
        "theta_C2": theta_c,
        **fields,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _run_model_phase(context: RuntimeContext, candidates: Sequence[Candidate], phase_name: str, resume: bool) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    if not candidates:
        raise RuntimeError(f"{phase_name} received no candidates")
    if [candidate.candidate_id for candidate in candidates] != list(range(len(candidates))):
        raise RuntimeError(f"{phase_name} candidate IDs must be contiguous")
    manifest = _phase_manifest(candidates)
    max_k = max(candidate.K for candidate in candidates)
    cache_dir = RESULT_ROOT / "_cache" / phase_name
    loaded: dict[int, dict[str, np.ndarray]] = {}
    missing: list[int] = []
    for state_id in range(STATE_COUNT):
        cached = _load_state_cache(cache_dir / f"state_{state_id:03d}.npz", manifest, len(candidates), max_k) if resume else None
        if cached is None:
            missing.append(state_id)
        else:
            loaded[state_id] = cached
    if missing:
        specs = tuple((candidate.candidate_id, candidate.support_indices, candidate.lambda_value) for candidate in candidates)
        context_mp = get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=WORKER_COUNT,
            mp_context=context_mp,
            initializer=worker_init,
            initargs=(specs,),
        ) as executor:
            futures = {executor.submit(model_worker_entry, state_id): state_id for state_id in missing}
            completed = STATE_COUNT - len(missing)
            for future in as_completed(futures):
                state_id = futures[future]
                item = future.result()
                _save_state_cache(cache_dir / f"state_{state_id:03d}.npz", manifest, candidates, item, max_k)
                loaded[state_id] = _load_state_cache(cache_dir / f"state_{state_id:03d}.npz", manifest, len(candidates), max_k)
                completed += 1
                if completed % 25 == 0 or completed == STATE_COUNT:
                    _checkpoint(context, phase=f"{phase_name}_model_progress", phase_manifest=manifest, candidate_count=len(candidates), max_k=max_k, completed_model_states=completed)
                    print(f"[{phase_name} MODEL] {completed}/{STATE_COUNT}", flush=True)
    if len(loaded) != STATE_COUNT:
        raise RuntimeError(f"{phase_name} model cache incomplete: {len(loaded)}/{STATE_COUNT}")

    theta_a = np.zeros((len(candidates), STATE_COUNT, max_k), dtype=np.complex128)
    theta_c = np.zeros_like(theta_a)
    rows: list[dict[str, Any]] = []
    for state_id in range(STATE_COUNT):
        cache = loaded[state_id]
        theta_a[:, state_id, :] = cache["theta_Aend"]
        theta_c[:, state_id, :] = cache["theta_C2"]
        for candidate_index, candidate in enumerate(candidates):
            row = {
                "candidate_id": candidate.candidate_id,
                "candidate_name": candidate.candidate_name,
                "support_hash": candidate.support_hash,
                "K": candidate.K,
                "lambda": candidate.lambda_value,
                "operation": candidate.operation,
                "parent_support_hash": candidate.parent_support_hash,
                "parent_candidate_name": candidate.parent_candidate_name,
                "source_task": candidate.source_task,
                "State_R": state_id,
                "State_ID": state_id,
                "funMng": int(context.state_table.iloc[state_id]["funMng"]),
                "funAng": int(context.state_table.iloc[state_id]["funAng"]),
                "secMng": int(context.state_table.iloc[state_id]["secMng"]),
                "secAng": int(context.state_table.iloc[state_id]["secAng"]),
                "Vm": float(context.state_table.iloc[state_id]["Vm"]),
                "Pin": float(context.state_table.iloc[state_id]["Pin"]),
                "Y_Aend_train_NMSE_dB": float(cache["Y_Aend_train_NMSE_dB"][candidate_index]),
                "Y_Aend_B_NMSE_dB": float(cache["Y_Aend_B_NMSE_dB"][candidate_index]),
                "Y_C2_train_NMSE_dB": float(cache["Y_C2_train_NMSE_dB"][candidate_index]),
                "Y_C2_B_NMSE_dB": float(cache["Y_C2_B_NMSE_dB"][candidate_index]),
                "Aend_rank": int(cache["Aend_rank"][candidate_index]),
                "C2_rank": int(cache["C2_rank"][candidate_index]),
                "Aend_condition_number": float(cache["Aend_condition_number"][candidate_index]),
                "C2_condition_number": float(cache["C2_condition_number"][candidate_index]),
                "Aend_coefficient_norm": float(cache["Aend_coefficient_norm"][candidate_index]),
                "C2_coefficient_norm": float(cache["C2_coefficient_norm"][candidate_index]),
                "Aend_finite": bool(cache["Aend_finite"][candidate_index]),
                "C2_finite": bool(cache["C2_finite"][candidate_index]),
            }
            rows.append(row)
    model_frame = pd.DataFrame(rows).sort_values(["candidate_id", "State_R"]).reset_index(drop=True)
    expected_rows = len(candidates) * STATE_COUNT
    if model_frame.shape[0] != expected_rows:
        raise RuntimeError(f"{phase_name} model rows changed: {model_frame.shape[0]} != {expected_rows}")
    return model_frame, theta_a, theta_c


def _fit_summary(model_frame: pd.DataFrame, candidate: Candidate) -> dict[str, Any]:
    frame = model_frame.loc[model_frame["candidate_id"].eq(candidate.candidate_id)]
    return {
        "modeling_finite_count": int(np.count_nonzero(frame["Aend_finite"] & frame["C2_finite"])),
        "Aend_rank_min": int(frame["Aend_rank"].min()),
        "C2_rank_min": int(frame["C2_rank"].min()),
        "Aend_condition_Q95": float(frame["Aend_condition_number"].quantile(0.95)),
        "C2_condition_Q95": float(frame["C2_condition_number"].quantile(0.95)),
        "Aend_condition_max": float(frame["Aend_condition_number"].max()),
        "C2_condition_max": float(frame["C2_condition_number"].max()),
        "Aend_train_median_dB": float(frame["Y_Aend_train_NMSE_dB"].median()),
        "Aend_B_median_dB": float(frame["Y_Aend_B_NMSE_dB"].median()),
        "C2_train_median_dB": float(frame["Y_C2_train_NMSE_dB"].median()),
        "C2_B_median_dB": float(frame["Y_C2_B_NMSE_dB"].median()),
    }


def _evaluate_model_phase(
    context: RuntimeContext,
    candidates: Sequence[Candidate],
    model_frame: pd.DataFrame,
    theta_a: np.ndarray,
    theta_c: np.ndarray,
    phase_name: str,
    resume: bool,
) -> tuple[pd.DataFrame, dict[int, CandidateEvaluation]]:
    distance_dir = RESULT_ROOT / phase_name / "distance_matrices"
    distance_dir.mkdir(parents=True, exist_ok=True)
    evaluations: dict[int, CandidateEvaluation] = {}
    summaries: list[dict[str, Any]] = []
    query_frames: list[pd.DataFrame] = []
    for candidate in candidates:
        distance_path = distance_dir / f"candidate_{candidate.candidate_id:04d}.npy"
        if resume and distance_path.is_file():
            distance = np.asarray(np.load(distance_path, allow_pickle=False, mmap_mode="r"), dtype=np.float64)
        else:
            common = context.common_phi[:, candidate.support_indices]
            lut = np.asarray(common @ theta_a[candidate.candidate_id, :, : candidate.K].T, dtype=np.complex128).T
            query = np.asarray(common @ theta_c[candidate.candidate_id, :, : candidate.K].T, dtype=np.complex128).T
            distance = compute_cnmse_distance_matrix(query, lut)
            np.save(distance_path, distance)
        summary, state_frame_dict, classes = evaluate_distance(
            distance,
            context.real_b_distance,
            context.shareability,
            candidate={
                "candidate_id": candidate.candidate_id,
                "candidate_name": candidate.candidate_name,
                "support_hash": candidate.support_hash,
                "K": candidate.K,
                "lambda": candidate.lambda_value,
            },
        )
        summary.update(candidate.to_record())
        summary.update(_fit_summary(model_frame, candidate))
        summary["candidate_key"] = f"{candidate.support_hash}|{candidate.lambda_value:.17g}"
        summary["distance_matrix_path"] = str(distance_path)
        detail = pd.DataFrame(state_frame_dict)
        for key, value in candidate.to_record().items():
            if key == "basis_ids":
                continue
            detail.insert(0 if key == "candidate_id" else detail.shape[1], key, value)
        detail["candidate_key"] = summary["candidate_key"]
        detail["support_basis_ids"] = json.dumps(list(candidate.basis_ids), ensure_ascii=False)
        detail["operation"] = candidate.operation
        detail["parent_support_hash"] = candidate.parent_support_hash
        detail["parent_candidate_name"] = candidate.parent_candidate_name
        detail["source_task"] = candidate.source_task
        query_frames.append(detail)
        evaluations[candidate.candidate_id] = CandidateEvaluation(candidate, summary, detail, classes, distance_path)
        summaries.append(summary)
        if len(summaries) % 10 == 0 or len(summaries) == len(candidates):
            _checkpoint(context, phase=f"{phase_name}_retrieval_progress", phase_manifest=_phase_manifest(candidates), completed_candidates=len(summaries), candidate_count=len(candidates))
            print(f"[{phase_name} RETRIEVAL] {len(summaries)}/{len(candidates)}", flush=True)
    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(RESULT_ROOT / phase_name / "support_lambda_results.csv", index=False)
    if query_frames:
        pd.concat(query_frames, ignore_index=True).to_csv(RESULT_ROOT / phase_name / "query_metrics_long.csv", index=False)
    return summary_frame, evaluations


def _representative_supports(ranking: pd.DataFrame) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    chosen_hashes: set[str] = set()
    for row in ranking.to_dict("records"):
        if row["support_hash"] in chosen_hashes:
            continue
        chosen.append(row)
        chosen_hashes.add(row["support_hash"])
        if len(chosen) == 5:
            break
    top_pool = ranking.head(min(100, ranking.shape[0])).to_dict("records")
    for row in top_pool:
        if len(chosen) >= 8 or row["support_hash"] in chosen_hashes:
            continue
        ids = set(json.loads(str(row["basis_ids"])))
        diverse = False
        for previous in chosen:
            previous_ids = set(json.loads(str(previous["basis_ids"])))
            if len(ids.symmetric_difference(previous_ids)) >= 2:
                diverse = True
                break
        if diverse:
            chosen.append(row)
            chosen_hashes.add(row["support_hash"])
    mp_rows = ranking.loc[ranking["source_model_name"].astype(str).str.contains("MP10", case=False, na=False)]
    if not mp_rows.empty and mp_rows.iloc[0]["support_hash"] not in chosen_hashes:
        chosen.append(mp_rows.iloc[0].to_dict())
    return chosen[:9]


def _lambda_slug(value: float) -> str:
    if value == 0.0:
        return "0"
    text = f"{value:.0e}".replace("+", "p").replace("-", "m")
    return text


def _calibration_candidates(
    supports: Sequence[dict[str, Any]],
    lambdas_by_support: Mapping[str, Sequence[float]],
    context: RuntimeContext,
    operation: str,
    prefix: str,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for support in supports:
        ids = tuple(json.loads(str(support["basis_ids"])))
        support_hash = canonical_support_hash(ids)
        for lambda_value in sorted(set(float(value) for value in lambdas_by_support[support_hash])):
            candidates.append(
                _candidate_from_support(
                    len(candidates),
                    ids,
                    lambda_value,
                    name=f"{prefix}_{support_hash[:10]}_lambda_{_lambda_slug(lambda_value)}",
                    operation=operation,
                    context=context,
                    source_task=str(support.get("source_task", "")),
                )
            )
    return candidates


def _select_best_per_support(summary_frame: pd.DataFrame) -> pd.DataFrame:
    selected: list[dict[str, Any]] = []
    for support_hash, group in summary_frame.groupby("support_hash", sort=False):
        ranked = rank_frame(group)
        selected.append(ranked.iloc[0].to_dict())
    return rank_frame(pd.DataFrame(selected))


def _build_active_lambda_set(coarse: pd.DataFrame, best: pd.DataFrame) -> list[float]:
    values: set[float] = {0.0, 1e-8}
    tested = sorted(set(float(value) for value in coarse["lambda"].dropna()))
    for value in best["lambda"].dropna().astype(float):
        values.add(float(value))
        if value in tested:
            index = tested.index(value)
            if index > 0:
                values.add(tested[index - 1])
            if index + 1 < len(tested):
                values.add(tested[index + 1])
    ordered = sorted(value for value in values if LAMBDA_BOUNDARY_MIN <= value <= LAMBDA_BOUNDARY_MAX or value == 0.0)
    if len(ordered) <= 7:
        return ordered
    # Keep 0 and 1e-8, then retain values that appeared most often as winners.
    winner_counts = best["lambda"].astype(float).value_counts().to_dict()
    ordered.sort(key=lambda value: (-int(winner_counts.get(value, 0)), 0 if value in {0.0, 1e-8} else 1, value))
    return sorted(ordered[:7])


def _calibrate_ridge(context: RuntimeContext, ranking: pd.DataFrame, resume: bool) -> tuple[pd.DataFrame, list[float]]:
    supports = _representative_supports(ranking)
    support_hashes = [canonical_support_hash(tuple(json.loads(str(row["basis_ids"])))) for row in supports]
    coarse_lambdas = {support_hash: list(COARSE_LAMBDA_GRID) for support_hash in support_hashes}
    coarse_candidates = _calibration_candidates(supports, coarse_lambdas, context, "RIDGE_COARSE", "calibration_coarse")
    coarse_model, coarse_theta_a, coarse_theta_c = _run_model_phase(context, coarse_candidates, "calibration_coarse", resume)
    coarse_summary, _ = _evaluate_model_phase(context, coarse_candidates, coarse_model, coarse_theta_a, coarse_theta_c, "calibration_coarse", resume)
    refine_values: dict[str, list[float]] = defaultdict(list)
    coarse_best = _select_best_per_support(coarse_summary)
    coarse_values = set(COARSE_LAMBDA_GRID)
    for row in coarse_best.to_dict("records"):
        value = float(row["lambda"])
        if value == 0.0:
            candidates = (0.0, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12)
        else:
            candidates = tuple(value * factor for factor in (0.1, 0.31622777, 1.0, 3.16227766, 10.0))
        refine_values[str(row["support_hash"])] = [float(item) for item in candidates if item not in coarse_values and LAMBDA_BOUNDARY_MIN <= item <= LAMBDA_BOUNDARY_MAX]
    refine_supports = [row for row in supports if canonical_support_hash(tuple(json.loads(str(row["basis_ids"])))) in refine_values]
    refine_candidates = _calibration_candidates(refine_supports, refine_values, context, "RIDGE_LOCAL_REFINEMENT", "calibration_refine")
    frames = [coarse_summary]
    if refine_candidates:
        refine_model, refine_theta_a, refine_theta_c = _run_model_phase(context, refine_candidates, "calibration_refine", resume)
        refine_summary, _ = _evaluate_model_phase(context, refine_candidates, refine_model, refine_theta_a, refine_theta_c, "calibration_refine", resume)
        frames.append(refine_summary)
    grid = pd.concat(frames, ignore_index=True)
    grid = rank_frame(grid)
    grid.to_csv(RESULT_ROOT / "05_ridge_calibration_grid.csv", index=False)
    best = _select_best_per_support(grid)
    best.to_csv(RESULT_ROOT / "06_ridge_calibration_best.csv", index=False)
    active = _build_active_lambda_set(grid, best)
    _write_json(
        RESULT_ROOT / "07_lambda_active_set.json",
        {
            "objective_version": OBJECTIVE_VERSION,
            "active_lambda_set": active,
            "max_size": 7,
            "forced_values": [0.0, 1e-8],
            "boundary_range": [LAMBDA_BOUNDARY_MIN, LAMBDA_BOUNDARY_MAX],
            "support_count": len(supports),
        },
    )
    return best, active


def _diversity_distance(left: Candidate, right: Candidate) -> int:
    return len(set(left.basis_ids).symmetric_difference(right.basis_ids))


def _select_beam(evaluations: Sequence[CandidateEvaluation]) -> list[CandidateEvaluation]:
    if not evaluations:
        raise RuntimeError("cannot select an empty Beam")
    ordered = sorted(evaluations, key=lambda item: lexicographic_key(item.summary))
    selected: list[CandidateEvaluation] = []
    for item in ordered:
        if not selected:
            selected.append(item)
            continue
        if any(
            _diversity_distance(item.candidate, previous.candidate) >= 2
            or set(json.loads(item.summary.get("self_miss_state_ids", "[]")))
            != set(json.loads(previous.summary.get("self_miss_state_ids", "[]")))
            for previous in selected
        ):
            selected.append(item)
        if len(selected) == BEAM_WIDTH:
            break
    if len(selected) < BEAM_WIDTH:
        for item in ordered:
            if item not in selected:
                selected.append(item)
            if len(selected) == BEAM_WIDTH:
                break
    return selected


def _initial_beam(context: RuntimeContext, calibration_best: pd.DataFrame, resume: bool) -> list[CandidateEvaluation]:
    candidates: list[Candidate] = []
    for index, row in calibration_best.head(BEAM_WIDTH).iterrows():
        ids = tuple(json.loads(str(row["basis_ids"])))
        candidates.append(
            _candidate_from_support(
                len(candidates),
                ids,
                float(row["lambda"]),
                name=f"initial_beam_{len(candidates)+1}_{row['support_hash'][:10]}",
                operation="INITIAL",
                context=context,
                source_task=str(row.get("source_task", "")),
            )
        )
    model, theta_a, theta_c = _run_model_phase(context, candidates, "initial_beam", resume)
    summary, evaluations = _evaluate_model_phase(context, candidates, model, theta_a, theta_c, "initial_beam", resume)
    ranked = rank_frame(summary)
    ranked.to_csv(RESULT_ROOT / "08_initial_beam.csv", index=False)
    selected = _select_beam(list(evaluations.values()))
    return selected


def _write_transition(edge_path: Path, parent: CandidateEvaluation, child: CandidateEvaluation, edge: dict[str, Any]) -> dict[str, Any]:
    transition = transition_matrix(parent.classes, child.classes)
    counts = transition["counts"]
    edge = {
        **edge,
        "transition_SELF_SELF": counts[0][0],
        "transition_SELF_FALLBACK": counts[0][1],
        "transition_SELF_FAIL": counts[0][2],
        "transition_FALLBACK_SELF": counts[1][0],
        "transition_FALLBACK_FALLBACK": counts[1][1],
        "transition_FALLBACK_FAIL": counts[1][2],
        "transition_FAIL_SELF": counts[2][0],
        "transition_FAIL_FALLBACK": counts[2][1],
        "transition_FAIL_FAIL": counts[2][2],
        "self_recovered": counts[1][0] + counts[2][0],
        "self_regressed": counts[0][1] + counts[0][2],
        "valid_recovered": counts[2][0] + counts[2][1],
        "valid_regressed": counts[0][2] + counts[1][2],
        "transition_state_ids": json.dumps(transition["state_ids"], ensure_ascii=False),
    }
    return edge


def _generate_add_candidates(context: RuntimeContext, beam: Sequence[CandidateEvaluation], active_lambdas: Sequence[float], round_index: int) -> tuple[list[Candidate], list[dict[str, Any]], dict[str, Candidate]]:
    by_hash: dict[str, Candidate] = {}
    edges: list[dict[str, Any]] = []
    for parent_index, parent_eval in enumerate(beam):
        parent = parent_eval.candidate
        if parent.K >= MAX_K:
            continue
        existing = set(parent.basis_ids)
        for term in context.terms:
            if term.basis_id in existing:
                continue
            ids = tuple(sorted((*parent.basis_ids, term.basis_id), key=lambda item: next(t.index for t in context.terms if t.basis_id == item)))
            support_hash = canonical_support_hash(ids)
            child = by_hash.get(support_hash)
            if child is None:
                child = _candidate_from_support(
                    len(by_hash),
                    ids,
                    float(active_lambdas[0]),
                    name=f"round_{round_index:02d}_add_{len(by_hash):04d}",
                    operation="ADD",
                    context=context,
                    parent=parent,
                )
                by_hash[support_hash] = child
            edges.append(
                {
                    "parent_beam_index": parent_index,
                    "parent_candidate_name": parent.candidate_name,
                    "parent_support_hash": parent.support_hash,
                    "added_basis_id": term.basis_id,
                    "child_support_hash": support_hash,
                }
            )
    candidates: list[Candidate] = []
    for support_index, child in enumerate(by_hash.values()):
        for lambda_value in active_lambdas:
            candidates.append(
                _candidate_from_support(
                    len(candidates),
                    child.basis_ids,
                    float(lambda_value),
                    name=f"round_{round_index:02d}_add_{support_index:04d}_lambda_{_lambda_slug(float(lambda_value))}",
                    operation="ADD",
                    context=context,
                    parent=child,
                )
            )
    return candidates, edges, by_hash


def _generate_delete_candidates(context: RuntimeContext, beam: Sequence[CandidateEvaluation], active_lambdas: Sequence[float], round_index: int) -> tuple[list[Candidate], list[dict[str, Any]], dict[str, Candidate]]:
    by_hash: dict[str, Candidate] = {}
    edges: list[dict[str, Any]] = []
    for parent_index, parent_eval in enumerate(beam):
        parent = parent_eval.candidate
        for removed in parent.basis_ids:
            ids = tuple(item for item in parent.basis_ids if item != removed)
            if not ids:
                continue
            support_hash = canonical_support_hash(ids)
            child = by_hash.get(support_hash)
            if child is None:
                child = _candidate_from_support(
                    len(by_hash),
                    ids,
                    float(active_lambdas[0]),
                    name=f"round_{round_index:02d}_delete_{len(by_hash):04d}",
                    operation="DELETE",
                    context=context,
                    parent=parent,
                )
                by_hash[support_hash] = child
            edges.append(
                {
                    "parent_beam_index": parent_index,
                    "parent_candidate_name": parent.candidate_name,
                    "parent_support_hash": parent.support_hash,
                    "removed_basis_id": removed,
                    "child_support_hash": support_hash,
                }
            )
    candidates: list[Candidate] = []
    for support_index, child in enumerate(by_hash.values()):
        for lambda_value in active_lambdas:
            candidates.append(
                _candidate_from_support(
                    len(candidates),
                    child.basis_ids,
                    float(lambda_value),
                    name=f"round_{round_index:02d}_delete_{support_index:04d}_lambda_{_lambda_slug(float(lambda_value))}",
                    operation="DELETE",
                    context=context,
                    parent=child,
                )
            )
    return candidates, edges, by_hash


def _best_support_evaluations(summary: pd.DataFrame, evaluations: dict[int, CandidateEvaluation]) -> dict[str, CandidateEvaluation]:
    result: dict[str, CandidateEvaluation] = {}
    for support_hash, group in summary.groupby("support_hash", sort=False):
        row = rank_frame(group).iloc[0]
        result[str(support_hash)] = evaluations[int(row["candidate_id"])]
    return result


def _run_add_delete_round(context: RuntimeContext, beam: list[CandidateEvaluation], active_lambdas: Sequence[float], round_index: int, resume: bool) -> tuple[list[CandidateEvaluation], bool, list[dict[str, Any]]]:
    round_root = RESULT_ROOT / "search_rounds" / f"round_{round_index:02d}"
    round_root.mkdir(parents=True, exist_ok=True)
    forward_candidates, forward_edges, _ = _generate_add_candidates(context, beam, active_lambdas, round_index)
    forward_best: dict[str, CandidateEvaluation] = {}
    if forward_candidates:
        model, theta_a, theta_c = _run_model_phase(context, forward_candidates, f"round_{round_index:02d}_forward", resume)
        summary, evaluations = _evaluate_model_phase(context, forward_candidates, model, theta_a, theta_c, f"round_{round_index:02d}_forward", resume)
        summary.to_csv(round_root / "support_lambda_results_forward.csv", index=False)
        forward_best = _best_support_evaluations(summary, evaluations)
        edges_with_transition = []
        for edge in forward_edges:
            child = forward_best[edge["child_support_hash"]]
            parent = next(item for item in beam if item.candidate.candidate_name == edge["parent_candidate_name"])
            edges_with_transition.append(_write_transition(round_root / "forward_edges.csv", parent, child, {**edge, "child_candidate_name": child.candidate.candidate_name, "child_N_self": child.summary["N_self"], "child_N_valid": child.summary["N_valid"]}))
        pd.DataFrame(edges_with_transition).to_csv(round_root / "raw_forward_edges.csv", index=False)
        pd.DataFrame([item.summary for item in forward_best.values()]).pipe(rank_frame).to_csv(round_root / "forward_summary.csv", index=False)
    after_forward = _select_beam(beam + list(forward_best.values()))
    pd.DataFrame([item.summary for item in after_forward]).to_csv(round_root / "beam_after_forward.csv", index=False)

    backward_candidates, backward_edges, _ = _generate_delete_candidates(context, after_forward, active_lambdas, round_index)
    backward_best: dict[str, CandidateEvaluation] = {}
    if backward_candidates:
        model, theta_a, theta_c = _run_model_phase(context, backward_candidates, f"round_{round_index:02d}_backward", resume)
        summary, evaluations = _evaluate_model_phase(context, backward_candidates, model, theta_a, theta_c, f"round_{round_index:02d}_backward", resume)
        summary.to_csv(round_root / "support_lambda_results_backward.csv", index=False)
        backward_best = _best_support_evaluations(summary, evaluations)
        edges_with_transition = []
        for edge in backward_edges:
            child = backward_best[edge["child_support_hash"]]
            parent = next(item for item in after_forward if item.candidate.candidate_name == edge["parent_candidate_name"])
            edges_with_transition.append(_write_transition(round_root / "backward_edges.csv", parent, child, {**edge, "child_candidate_name": child.candidate.candidate_name, "child_N_self": child.summary["N_self"], "child_N_valid": child.summary["N_valid"]}))
        pd.DataFrame(edges_with_transition).to_csv(round_root / "raw_backward_edges.csv", index=False)
        pd.DataFrame([item.summary for item in backward_best.values()]).pipe(rank_frame).to_csv(round_root / "backward_summary.csv", index=False)
    updated = _select_beam(after_forward + list(backward_best.values()))
    pd.DataFrame([item.summary for item in updated]).to_csv(round_root / "beam_after_backward.csv", index=False)
    all_edges = [
        {"operation": "ADD", **edge}
        for edge in forward_edges
    ] + [
        {"operation": "DELETE", **edge}
        for edge in backward_edges
    ]
    previous_best = min(beam, key=lambda item: lexicographic_key(item.summary))
    current_best = min(updated, key=lambda item: lexicographic_key(item.summary))
    primary_improved = (
        int(current_best.summary["N_self"]) > int(previous_best.summary["N_self"])
        or (
            int(current_best.summary["N_self"]) == int(previous_best.summary["N_self"])
            and int(current_best.summary["N_valid"]) > int(previous_best.summary["N_valid"])
        )
    )
    _checkpoint(context, phase=f"round_{round_index:02d}_completed", round_index=round_index, primary_improved=primary_improved, beam=[item.candidate.to_record() for item in updated])
    return updated, primary_improved, all_edges


def _run_restricted_swap(context: RuntimeContext, beam: list[CandidateEvaluation], active_lambdas: Sequence[float], round_index: int, resume: bool) -> tuple[list[CandidateEvaluation], bool]:
    swap_root = RESULT_ROOT / "swap_rounds" / f"swap_{round_index:02d}"
    swap_root.mkdir(parents=True, exist_ok=True)
    candidates: list[Candidate] = []
    edges: list[dict[str, Any]] = []
    seen: set[str] = set()
    all_terms = {term.basis_id for term in context.terms}
    for parent_index, parent_eval in enumerate(beam):
        parent = parent_eval.candidate
        weakest = list(parent.basis_ids)[:5]
        unselected = [term.basis_id for term in context.terms if term.basis_id not in set(parent.basis_ids)][:10]
        for removed in weakest:
            for added in unselected:
                ids = tuple(sorted((set(parent.basis_ids) - {removed}) | {added}, key=lambda item: next(t.index for t in context.terms if t.basis_id == item)))
                if not ids or not set(ids).issubset(all_terms):
                    continue
                support_hash = canonical_support_hash(ids)
                if support_hash not in seen:
                    seen.add(support_hash)
                    base = _candidate_from_support(len(seen) - 1, ids, float(active_lambdas[0]), name=f"swap_{round_index:02d}_{len(seen)-1:04d}", operation="SWAP", context=context, parent=parent)
                    for lambda_value in active_lambdas:
                        candidates.append(_candidate_from_support(len(candidates), ids, float(lambda_value), name=f"swap_{round_index:02d}_{len(seen)-1:04d}_lambda_{_lambda_slug(float(lambda_value))}", operation="SWAP", context=context, parent=base))
                edges.append({"parent_candidate_name": parent.candidate_name, "parent_support_hash": parent.support_hash, "removed_basis_id": removed, "added_basis_id": added, "child_support_hash": support_hash})
    if not candidates:
        return beam, False
    model, theta_a, theta_c = _run_model_phase(context, candidates, f"swap_{round_index:02d}", resume)
    summary, evaluations = _evaluate_model_phase(context, candidates, model, theta_a, theta_c, f"swap_{round_index:02d}", resume)
    summary.to_csv(swap_root / "support_lambda_results.csv", index=False)
    best = _best_support_evaluations(summary, evaluations)
    pd.DataFrame([item.summary for item in best.values()]).pipe(rank_frame).to_csv(swap_root / "swap_summary.csv", index=False)
    updated = _select_beam(beam + list(best.values()))
    previous_best = min(beam, key=lambda item: lexicographic_key(item.summary))
    current_best = min(updated, key=lambda item: lexicographic_key(item.summary))
    improved = (
        int(current_best.summary["N_self"]) > int(previous_best.summary["N_self"])
        or (
            int(current_best.summary["N_self"]) == int(previous_best.summary["N_self"])
            and int(current_best.summary["N_valid"]) > int(previous_best.summary["N_valid"])
        )
    )
    pd.DataFrame(edges).to_csv(swap_root / "raw_swap_edges.csv", index=False)
    return updated, improved


def _save_figures(summary: pd.DataFrame) -> None:
    if summary.empty:
        return
    figures = RESULT_ROOT / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    ranked = rank_frame(summary).head(50)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(np.arange(ranked.shape[0]), ranked["N_self"].to_numpy(), marker=".", label="N_self")
    ax.plot(np.arange(ranked.shape[0]), ranked["N_valid"].to_numpy(), marker=".", label="N_valid")
    ax.set_xlabel("Self-First candidate rank")
    ax.set_ylabel("count / 425")
    ax.set_title("Self-First candidate ranking")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(figures / "self_first_ranking_progress.png", dpi=150)
    plt.close(fig)


def _run_search(
    context: RuntimeContext,
    initial_beam: list[CandidateEvaluation],
    active_lambdas: Sequence[float],
    resume: bool,
) -> tuple[list[CandidateEvaluation], list[dict[str, Any]], str, int]:
    beam = initial_beam
    all_edge_rows: list[dict[str, Any]] = []
    plateau = 0
    swap_plateau = 0
    stop_reason = "MAX_SEARCH_ROUNDS_REACHED"
    rounds_completed = 0
    for round_index in range(1, MAX_FORWARD_BACKWARD_ROUNDS + 1):
        rounds_completed = round_index
        beam, improved, edges = _run_add_delete_round(context, beam, active_lambdas, round_index, resume)
        all_edge_rows.extend(edges)
        if improved:
            plateau = 0
        else:
            plateau += 1
        if plateau >= 2:
            beam, swap_improved = _run_restricted_swap(context, beam, active_lambdas, round_index, resume)
            if swap_improved:
                plateau = 0
                swap_plateau = 0
            else:
                swap_plateau += 1
                if swap_plateau >= 2:
                    stop_reason = "ADD_DELETE_SWAP_PLATEAU"
                    break
    return beam, all_edge_rows, stop_reason, rounds_completed


def _subset_summary(evaluation: CandidateEvaluation, indices: np.ndarray) -> dict[str, Any]:
    """Aggregate the same Self-First metrics on a Query-State subset."""

    frame = evaluation.state_frame.iloc[np.asarray(indices, dtype=np.int64)]
    classes = frame["retrieval_class"].astype(str).to_numpy(dtype=object)
    ranks = frame["true_state_rank"].to_numpy(dtype=float)
    fallback_ranks = frame["fallback_rank"].to_numpy(dtype=float)
    self_margin = frame["self_margin_dB"].to_numpy(dtype=float)
    fallback_margin = frame["fallback_margin_dB"].to_numpy(dtype=float)
    n_self = int(np.count_nonzero(classes == "SELF"))
    n_fallback = int(np.count_nonzero(classes == "SHAREABLE_FALLBACK"))
    n_fail = int(np.count_nonzero(classes == "FAIL"))
    if n_self + n_fallback + n_fail != frame.shape[0]:
        raise RuntimeError("subset Self-First class identity failed")

    def q05(values: np.ndarray) -> float:
        values = values[np.isfinite(values)]
        return float(np.quantile(values, 0.05)) if values.size else float("nan")

    fallback_valid = fallback_ranks > 0
    return {
        "candidate_id": evaluation.candidate.candidate_id,
        "candidate_name": evaluation.candidate.candidate_name,
        "support_hash": evaluation.candidate.support_hash,
        "K": evaluation.candidate.K,
        "lambda": evaluation.candidate.lambda_value,
        "N_self": n_self,
        "N_fallback": n_fallback,
        "N_valid": n_self + n_fallback,
        "N_fail": n_fail,
        "self_Q05": q05(self_margin),
        "self_MRR": float(np.mean(1.0 / ranks)),
        "fallback_Q05": q05(fallback_margin),
        "fallback_MRR": float(np.mean(np.divide(1.0, fallback_ranks, out=np.zeros_like(fallback_ranks), where=fallback_valid))),
        "Self_Top3": int(np.count_nonzero(ranks <= 3)),
        "query_count": int(frame.shape[0]),
    }


def _load_search_candidate_pool() -> pd.DataFrame:
    """Read all completed support/lambda summaries produced by search rounds."""

    paths: list[Path] = []
    initial = RESULT_ROOT / "08_initial_beam.csv"
    if initial.is_file():
        paths.append(initial)
    paths.extend(sorted((RESULT_ROOT / "search_rounds").glob("round_*/support_lambda_results_*.csv")))
    paths.extend(sorted((RESULT_ROOT / "swap_rounds").glob("swap_*/support_lambda_results.csv")))
    frames: list[pd.DataFrame] = []
    for path in paths:
        try:
            frame = pd.read_csv(path)
        except (OSError, pd.errors.ParserError):
            continue
        if {"support_hash", "basis_ids", "lambda", "N_self", "N_valid"}.issubset(frame.columns):
            frames.append(frame)
    if not frames:
        raise RuntimeError("no completed search candidate summaries are available")
    return pd.concat(frames, ignore_index=True)


def _candidate_from_summary_row(
    context: RuntimeContext,
    row: Mapping[str, Any],
    candidate_id: int,
    *,
    name_prefix: str,
    operation: str,
) -> Candidate:
    ids = tuple(json.loads(str(row["basis_ids"])))
    return _candidate_from_support(
        candidate_id,
        ids,
        float(row["lambda"]),
        name=f"{name_prefix}_{candidate_id:04d}",
        operation=operation,
        context=context,
        source_task=str(row.get("source_task", "")),
    )


def _dense_lambda_values(best_lambda: float) -> list[float]:
    values: set[float] = {0.0, 1e-8}
    if best_lambda > 0.0:
        for exponent_step in np.arange(-1.5, 1.5001, 0.25):
            value = best_lambda * (10.0**float(exponent_step))
            if LAMBDA_BOUNDARY_MIN <= value <= LAMBDA_BOUNDARY_MAX:
                values.add(float(value))
    else:
        values.update((1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9))
    return sorted(values)


def _unique_support_rows(frame: pd.DataFrame, limit: int) -> list[dict[str, Any]]:
    chosen: list[dict[str, Any]] = []
    seen: set[str] = set()
    ranked = rank_frame(frame)
    for row in ranked.to_dict("records"):
        support_hash = str(row["support_hash"])
        if support_hash in seen:
            continue
        seen.add(support_hash)
        chosen.append(row)
        if len(chosen) >= limit:
            break
    return chosen


def _write_candidate_stage(
    stage_root: Path,
    summary: pd.DataFrame,
    evaluations: Mapping[int, CandidateEvaluation],
) -> None:
    stage_root.mkdir(parents=True, exist_ok=True)
    summary.to_csv(stage_root / "support_lambda_results.csv", index=False)
    query_frames = [evaluation.state_frame for evaluation in evaluations.values()]
    if query_frames:
        pd.concat(query_frames, ignore_index=True).to_csv(stage_root / "query_metrics_long.csv", index=False)


def _final_dense_ridge(
    context: RuntimeContext,
    search_pool: pd.DataFrame,
    active_lambdas: Sequence[float],
    resume: bool,
) -> tuple[pd.DataFrame, dict[int, CandidateEvaluation]]:
    top_supports = _unique_support_rows(search_pool, 10)
    candidates: list[Candidate] = []
    for support in top_supports:
        ids = tuple(json.loads(str(support["basis_ids"])))
        for value in _dense_lambda_values(float(support["lambda"])):
            candidates.append(
                _candidate_from_support(
                    len(candidates),
                    ids,
                    value,
                    name=f"dense_{support['support_hash'][:10]}_lambda_{_lambda_slug(value)}",
                    operation="FINAL_DENSE_RIDGE",
                    context=context,
                    source_task=str(support.get("source_task", "")),
                )
            )
    model, theta_a, theta_c = _run_model_phase(context, candidates, "final_dense_ridge", resume)
    summary, evaluations = _evaluate_model_phase(context, candidates, model, theta_a, theta_c, "final_dense_ridge", resume)
    ranked = rank_frame(summary)
    ranked.to_csv(RESULT_ROOT / "12_final_dense_ridge_scan.csv", index=False)
    _write_candidate_stage(RESULT_ROOT / "final_dense_ridge", ranked, evaluations)
    return ranked, evaluations


def _best_evaluation_per_support(summary: pd.DataFrame, evaluations: Mapping[int, CandidateEvaluation]) -> dict[str, CandidateEvaluation]:
    result: dict[str, CandidateEvaluation] = {}
    for support_hash, group in summary.groupby("support_hash", sort=False):
        row = rank_frame(group).iloc[0]
        result[str(support_hash)] = evaluations[int(row["candidate_id"])]
    return result


def _run_final_leave_one_out(
    context: RuntimeContext,
    parent: CandidateEvaluation,
    active_lambdas: Sequence[float],
    resume: bool,
) -> tuple[pd.DataFrame, dict[int, CandidateEvaluation]]:
    candidates: list[Candidate] = []
    removed_by_hash: dict[str, str] = {}
    for removed in parent.candidate.basis_ids:
        ids = tuple(item for item in parent.candidate.basis_ids if item != removed)
        support_hash = canonical_support_hash(ids)
        removed_by_hash[support_hash] = removed
        for value in active_lambdas:
            candidates.append(
                _candidate_from_support(
                    len(candidates),
                    ids,
                    float(value),
                    name=f"final_delete_{removed}_lambda_{_lambda_slug(float(value))}",
                    operation="FINAL_DELETE",
                    context=context,
                    parent=parent.candidate,
                )
            )
    model, theta_a, theta_c = _run_model_phase(context, candidates, "final_leave_one_out", resume)
    summary, evaluations = _evaluate_model_phase(context, candidates, model, theta_a, theta_c, "final_leave_one_out", resume)
    best = _best_evaluation_per_support(summary, evaluations)
    rows: list[dict[str, Any]] = []
    edge_rows: list[dict[str, Any]] = []
    for support_hash, child in best.items():
        removed = removed_by_hash[support_hash]
        trans = transition_matrix(parent.classes, child.classes)
        counts = trans["counts"]
        rows.append(
            {
                "basis_id": removed,
                "parent_support_hash": parent.candidate.support_hash,
                "child_support_hash": child.candidate.support_hash,
                "parent_K": parent.candidate.K,
                "child_K": child.candidate.K,
                "parent_lambda": parent.candidate.lambda_value,
                "best_lambda_after_deletion": child.candidate.lambda_value,
                "parent_N_self": parent.summary["N_self"],
                "child_N_self": child.summary["N_self"],
                "C_self_minus": int(parent.summary["N_self"] - child.summary["N_self"]),
                "parent_N_valid": parent.summary["N_valid"],
                "child_N_valid": child.summary["N_valid"],
                "C_valid_minus": int(parent.summary["N_valid"] - child.summary["N_valid"]),
                "parent_self_Q05": parent.summary["self_Q05"],
                "child_self_Q05": child.summary["self_Q05"],
                "self_Q05_change": child.summary["self_Q05"] - parent.summary["self_Q05"],
                "SELF_to_SELF": counts[0][0],
                "SELF_to_FALLBACK": counts[0][1],
                "SELF_to_FAIL": counts[0][2],
                "FALLBACK_to_SELF": counts[1][0],
                "FAIL_to_SELF": counts[2][0],
                "transition_state_ids": json.dumps(trans["state_ids"], ensure_ascii=False),
            }
        )
        edge_rows.append({"removed_basis_id": removed, **trans})
    importance = pd.DataFrame(rows).sort_values("C_self_minus", ascending=True).reset_index(drop=True)
    importance.to_csv(RESULT_ROOT / "13_final_basis_importance.csv", index=False)
    pd.DataFrame(edge_rows).to_json(RESULT_ROOT / "14_final_leave_one_out_edges.json", orient="records", force_ascii=False, indent=2)
    ranked = rank_frame(summary)
    _write_candidate_stage(RESULT_ROOT / "final_leave_one_out", ranked, evaluations)
    return importance, evaluations


def _run_final_single_add(
    context: RuntimeContext,
    parent: CandidateEvaluation,
    active_lambdas: Sequence[float],
    resume: bool,
) -> tuple[pd.DataFrame, dict[int, CandidateEvaluation]]:
    candidates: list[Candidate] = []
    added_by_hash: dict[str, str] = {}
    existing = set(parent.candidate.basis_ids)
    for term in context.terms:
        if term.basis_id in existing:
            continue
        ids = tuple(sorted((*parent.candidate.basis_ids, term.basis_id), key=lambda item: next(t.index for t in context.terms if t.basis_id == item)))
        support_hash = canonical_support_hash(ids)
        added_by_hash[support_hash] = term.basis_id
        for value in active_lambdas:
            candidates.append(
                _candidate_from_support(
                    len(candidates),
                    ids,
                    float(value),
                    name=f"final_add_{term.basis_id}_lambda_{_lambda_slug(float(value))}",
                    operation="FINAL_ADD",
                    context=context,
                    parent=parent.candidate,
                )
            )
    model, theta_a, theta_c = _run_model_phase(context, candidates, "final_single_add", resume)
    summary, evaluations = _evaluate_model_phase(context, candidates, model, theta_a, theta_c, "final_single_add", resume)
    best = _best_evaluation_per_support(summary, evaluations)
    rows: list[dict[str, Any]] = []
    for support_hash, child in best.items():
        rows.append(
            {
                "added_basis_id": added_by_hash[support_hash],
                "parent_support_hash": parent.candidate.support_hash,
                "child_support_hash": child.candidate.support_hash,
                "child_K": child.candidate.K,
                "best_lambda": child.candidate.lambda_value,
                "parent_N_self": parent.summary["N_self"],
                "child_N_self": child.summary["N_self"],
                "delta_N_self": int(child.summary["N_self"] - parent.summary["N_self"]),
                "parent_N_valid": parent.summary["N_valid"],
                "child_N_valid": child.summary["N_valid"],
                "delta_N_valid": int(child.summary["N_valid"] - parent.summary["N_valid"]),
                "self_Q05": child.summary["self_Q05"],
                "self_MRR": child.summary["self_MRR"],
            }
        )
    result = pd.DataFrame(rows)
    result = rank_frame(result) if not result.empty else result
    result.to_csv(RESULT_ROOT / "15_final_single_add_validation.csv", index=False)
    ranked = rank_frame(summary)
    _write_candidate_stage(RESULT_ROOT / "final_single_add", ranked, evaluations)
    return result, evaluations


def _run_final_restricted_swap(
    context: RuntimeContext,
    parent: CandidateEvaluation,
    importance: pd.DataFrame,
    add_summary: pd.DataFrame,
    add_evaluations: Mapping[int, CandidateEvaluation],
    active_lambdas: Sequence[float],
    resume: bool,
) -> tuple[pd.DataFrame, dict[int, CandidateEvaluation]]:
    if add_summary.empty:
        return pd.DataFrame(), {}
    weak = importance["basis_id"].astype(str).head(5).tolist()
    promising = add_summary.sort_values(
        ["delta_N_self", "delta_N_valid", "self_Q05", "self_MRR"],
        ascending=[False, False, False, False],
    )["added_basis_id"].astype(str).head(10).tolist()
    candidates: list[Candidate] = []
    edge_rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for removed in weak:
        for added in promising:
            ids = tuple(sorted((set(parent.candidate.basis_ids) - {removed}) | {added}, key=lambda item: next(t.index for t in context.terms if t.basis_id == item)))
            support_hash = canonical_support_hash(ids)
            if support_hash in seen:
                continue
            seen.add(support_hash)
            edge_rows.append({"removed_basis_id": removed, "added_basis_id": added, "child_support_hash": support_hash})
            for value in active_lambdas:
                candidates.append(
                    _candidate_from_support(
                        len(candidates),
                        ids,
                        float(value),
                        name=f"final_swap_{removed}_{added}_lambda_{_lambda_slug(float(value))}",
                        operation="FINAL_SWAP",
                        context=context,
                        parent=parent.candidate,
                    )
                )
    if not candidates:
        return pd.DataFrame(), {}
    model, theta_a, theta_c = _run_model_phase(context, candidates, "final_restricted_swap", resume)
    summary, evaluations = _evaluate_model_phase(context, candidates, model, theta_a, theta_c, "final_restricted_swap", resume)
    best = _best_evaluation_per_support(summary, evaluations)
    edge_by_hash = {row["child_support_hash"]: row for row in edge_rows}
    rows: list[dict[str, Any]] = []
    for support_hash, child in best.items():
        edge = edge_by_hash[support_hash]
        rows.append(
            {
                **edge,
                "child_K": child.candidate.K,
                "best_lambda": child.candidate.lambda_value,
                "parent_N_self": parent.summary["N_self"],
                "child_N_self": child.summary["N_self"],
                "delta_N_self": int(child.summary["N_self"] - parent.summary["N_self"]),
                "parent_N_valid": parent.summary["N_valid"],
                "child_N_valid": child.summary["N_valid"],
                "delta_N_valid": int(child.summary["N_valid"] - parent.summary["N_valid"]),
                "self_Q05": child.summary["self_Q05"],
                "self_MRR": child.summary["self_MRR"],
            }
        )
    result = rank_frame(pd.DataFrame(rows))
    result.to_csv(RESULT_ROOT / "16_final_swap_validation.csv", index=False)
    pd.DataFrame(edge_rows).to_csv(RESULT_ROOT / "16_final_swap_edges.csv", index=False)
    ranked = rank_frame(summary)
    _write_candidate_stage(RESULT_ROOT / "final_restricted_swap", ranked, evaluations)
    return result, evaluations


def _build_final_candidate_pool(
    dense_evaluations: Mapping[int, CandidateEvaluation],
    add_evaluations: Mapping[int, CandidateEvaluation],
    swap_evaluations: Mapping[int, CandidateEvaluation],
) -> list[CandidateEvaluation]:
    values: list[CandidateEvaluation] = []
    for evaluations in (dense_evaluations, add_evaluations, swap_evaluations):
        if not evaluations:
            continue
        frame = pd.DataFrame([evaluation.summary for evaluation in evaluations.values()])
        for row in _unique_support_rows(frame, 10):
            evaluation = evaluations.get(int(row["candidate_id"]))
            if evaluation is not None:
                values.append(evaluation)
    unique: dict[tuple[str, float], CandidateEvaluation] = {}
    for evaluation in values:
        key = (evaluation.candidate.support_hash, evaluation.candidate.lambda_value)
        current = unique.get(key)
        if current is None or lexicographic_key(evaluation.summary) < lexicographic_key(current.summary):
            unique[key] = evaluation
    return sorted(unique.values(), key=lambda item: lexicographic_key(item.summary))


def _run_query_state_cv(context: RuntimeContext, evaluations: Sequence[CandidateEvaluation]) -> tuple[pd.DataFrame, pd.DataFrame]:
    folds = _load_cv_folds()
    all_rows: list[dict[str, Any]] = []
    for fold in range(5):
        train = folds.loc[folds["fold"].ne(fold), "state_id"].to_numpy(dtype=np.int64)
        validation = folds.loc[folds["fold"].eq(fold), "state_id"].to_numpy(dtype=np.int64)
        fold_rows: list[dict[str, Any]] = []
        for evaluation in evaluations:
            train_row = _subset_summary(evaluation, train)
            validation_row = _subset_summary(evaluation, validation)
            row = {
                "fold": fold,
                "candidate_id": evaluation.candidate.candidate_id,
                "candidate_name": evaluation.candidate.candidate_name,
                "support_hash": evaluation.candidate.support_hash,
                "K": evaluation.candidate.K,
                "lambda": evaluation.candidate.lambda_value,
                "train_N_self": train_row["N_self"],
                "train_N_fallback": train_row["N_fallback"],
                "train_N_valid": train_row["N_valid"],
                "train_N_fail": train_row["N_fail"],
                "train_self_Q05": train_row["self_Q05"],
                "train_self_MRR": train_row["self_MRR"],
                "validation_N_self": validation_row["N_self"],
                "validation_N_fallback": validation_row["N_fallback"],
                "validation_N_valid": validation_row["N_valid"],
                "validation_N_fail": validation_row["N_fail"],
                "validation_self_Q05": validation_row["self_Q05"],
                "validation_self_MRR": validation_row["self_MRR"],
            }
            fold_rows.append(row)
        winner = min(
            fold_rows,
            key=lambda row: lexicographic_key(
                {
                    "N_self": row["train_N_self"],
                    "N_valid": row["train_N_valid"],
                    "self_Q05": row["train_self_Q05"],
                    "self_MRR": row["train_self_MRR"],
                    "fallback_Q05": float("nan"),
                    "fallback_MRR": float("nan"),
                    "Self_Top3": 0,
                    "K": row["K"],
                    "support_hash": row["support_hash"],
                    "lambda": row["lambda"],
                }
            ),
        )
        for row in fold_rows:
            row["train_is_fold_winner"] = bool(row["candidate_id"] == winner["candidate_id"])
        all_rows.extend(fold_rows)
    cv_frame = pd.DataFrame(all_rows)
    stability = (
        cv_frame.groupby(["candidate_id", "candidate_name", "support_hash", "K", "lambda"], as_index=False)
        .agg(
            train_winner_count=("train_is_fold_winner", "sum"),
            validation_N_self_mean=("validation_N_self", "mean"),
            validation_N_self_min=("validation_N_self", "min"),
            validation_N_valid_mean=("validation_N_valid", "mean"),
            validation_N_valid_min=("validation_N_valid", "min"),
            validation_self_Q05_mean=("validation_self_Q05", "mean"),
            validation_self_MRR_mean=("validation_self_MRR", "mean"),
        )
    )
    cv_frame.to_csv(RESULT_ROOT / "20_query_state_cv_results.csv", index=False)
    stability.to_csv(RESULT_ROOT / "21_query_state_cv_stability.csv", index=False)
    return cv_frame, stability


def _write_numerical_conditioning(evaluations: Sequence[CandidateEvaluation]) -> None:
    rows = []
    for evaluation in evaluations:
        summary = evaluation.summary
        rows.append(
            {
                "candidate_id": evaluation.candidate.candidate_id,
                "candidate_name": evaluation.candidate.candidate_name,
                "support_hash": evaluation.candidate.support_hash,
                "K": evaluation.candidate.K,
                "lambda": evaluation.candidate.lambda_value,
                "Aend_rank_min": summary.get("Aend_rank_min"),
                "C2_rank_min": summary.get("C2_rank_min"),
                "Aend_condition_Q95": summary.get("Aend_condition_Q95"),
                "C2_condition_Q95": summary.get("C2_condition_Q95"),
                "Aend_condition_max": summary.get("Aend_condition_max"),
                "C2_condition_max": summary.get("C2_condition_max"),
                "Aend_coefficient_norm_median": summary.get("Aend_train_median_dB"),
                "C2_coefficient_norm_median": summary.get("C2_train_median_dB"),
            }
        )
    pd.DataFrame(rows).to_csv(RESULT_ROOT / "19_numerical_conditioning.csv", index=False)


def _write_context_dependence(
    importance: pd.DataFrame,
    add_summary: pd.DataFrame,
    parent: CandidateEvaluation,
) -> None:
    rows: list[dict[str, Any]] = []
    for row in importance.to_dict("records"):
        rows.append(
            {
                "basis_id": row["basis_id"],
                "context_support_hash": parent.candidate.support_hash,
                "operation": "DELETE",
                "C_self_conditional": row["C_self_minus"],
                "C_valid_conditional": row["C_valid_minus"],
                "delta_self_Q05": row["self_Q05_change"],
            }
        )
    for row in add_summary.to_dict("records"):
        rows.append(
            {
                "basis_id": row["added_basis_id"],
                "context_support_hash": parent.candidate.support_hash,
                "operation": "ADD",
                "C_self_conditional": row["delta_N_self"],
                "C_valid_conditional": row["delta_N_valid"],
                "delta_self_Q05": row["self_Q05"] - float(parent.summary["self_Q05"]),
            }
        )
    pd.DataFrame(rows).to_csv(RESULT_ROOT / "18_context_dependence_summary.csv", index=False)


def _run_validation(
    context: RuntimeContext,
    evaluations: Sequence[CandidateEvaluation],
    *,
    rng_seed: int = 20260916,
) -> dict[str, Any]:
    if not evaluations:
        raise RuntimeError("final validation received no evaluations")
    rng = np.random.default_rng(rng_seed)
    top1_checks = 0
    top1_mismatches = 0
    self_checks = 0
    self_mismatches = 0
    fallback_checks = 0
    fallback_mismatches = 0
    eligible_fallback: list[tuple[CandidateEvaluation, int]] = []
    for evaluation in evaluations:
        for state_id, available in enumerate(evaluation.state_frame["fallback_available"].to_numpy(dtype=bool)):
            if available:
                eligible_fallback.append((evaluation, state_id))
    for _ in range(500):
        evaluation = evaluations[int(rng.integers(0, len(evaluations)))]
        state_id = int(rng.integers(0, STATE_COUNT))
        distance = np.asarray(np.load(evaluation.distance_path, allow_pickle=False, mmap_mode="r"), dtype=np.float64)
        row = distance[state_id]
        minimum = float(np.min(row))
        tied = np.flatnonzero(row <= minimum + TIE_TOLERANCE_DB)
        selected = int(np.min(tied))
        stored = int(evaluation.state_frame.iloc[state_id]["State_Q"])
        top1_checks += 1
        top1_mismatches += int(selected != stored)
    for _ in range(300):
        evaluation = evaluations[int(rng.integers(0, len(evaluations)))]
        state_id = int(rng.integers(0, STATE_COUNT))
        distance = np.asarray(np.load(evaluation.distance_path, allow_pickle=False, mmap_mode="r"), dtype=np.float64)
        row = distance[state_id].copy()
        self_distance = float(row[state_id])
        row[state_id] = np.inf
        expected = float(np.min(row))
        stored = float(evaluation.state_frame.iloc[state_id]["self_margin_dB"])
        actual = float(expected - self_distance) if np.isfinite(self_distance) else float("inf")
        self_checks += 1
        self_mismatches += int(not (np.isinf(actual) and np.isinf(stored)) and not np.isclose(actual, stored, atol=1e-9, rtol=0))
    for index in rng.choice(len(eligible_fallback), size=min(300, len(eligible_fallback)), replace=False):
        evaluation, state_id = eligible_fallback[int(index)]
        distance = np.asarray(np.load(evaluation.distance_path, allow_pickle=False, mmap_mode="r"), dtype=np.float64)
        row = distance[state_id]
        share = context.shareability[state_id].copy()
        share[state_id] = False
        order = np.lexsort((np.arange(STATE_COUNT), np.where(np.arange(STATE_COUNT) == state_id, np.inf, row)))
        shareable = [int(item) for item in order if share[item]]
        stored = float(evaluation.state_frame.iloc[state_id]["fallback_rank"])
        actual = float(np.flatnonzero(order == shareable[0])[0]) if shareable else 0.0
        fallback_checks += 1
        fallback_mismatches += int((stored > 0 and int(actual) + 1 != int(stored)) or (stored == 0 and not shareable))
    class_identity = []
    for evaluation in evaluations:
        counts = evaluation.state_frame["retrieval_class"].value_counts().to_dict()
        class_identity.append(
            {
                "candidate_name": evaluation.candidate.candidate_name,
                "N_self": int(counts.get("SELF", 0)),
                "N_fallback": int(counts.get("SHAREABLE_FALLBACK", 0)),
                "N_fail": int(counts.get("FAIL", 0)),
                "identity_pass": int(
                    sum(counts.get(name, 0) for name in ("SELF", "SHAREABLE_FALLBACK", "FAIL"))
                )
                == STATE_COUNT,
            }
        )
    raw_after = raw_manifest_gate()
    result = {
        "objective_version": OBJECTIVE_VERSION,
        "top1_checks": top1_checks,
        "top1_mismatches": top1_mismatches,
        "self_margin_checks": self_checks,
        "self_margin_mismatches": self_mismatches,
        "fallback_checks": fallback_checks,
        "fallback_mismatches": fallback_mismatches,
        "class_identity": class_identity,
        "raw_manifest_after": raw_after,
        "raw_data_modified": raw_after != context.contract["raw_manifest"],
    }
    result["pass"] = bool(
        top1_mismatches == 0
        and self_mismatches == 0
        and fallback_mismatches == 0
        and all(item["identity_pass"] for item in class_identity)
        and not result["raw_data_modified"]
    )
    _write_json(RESULT_ROOT / "24_validation_checks.json", result)
    return result


def _finish_search(
    context: RuntimeContext,
    initial_beam: list[CandidateEvaluation],
    active_lambdas: Sequence[float],
    registry_hash: str,
    stop_reason: str,
    rounds_completed: int,
    resume: bool,
) -> dict[str, Any]:
    search_pool = _load_search_candidate_pool()
    dense_summary, dense_evaluations = _final_dense_ridge(context, search_pool, active_lambdas, resume)
    dense_best = min(dense_evaluations.values(), key=lambda item: lexicographic_key(item.summary))
    importance, loo_evaluations = _run_final_leave_one_out(context, dense_best, active_lambdas, resume)
    add_summary, add_evaluations = _run_final_single_add(context, dense_best, active_lambdas, resume)
    swap_summary, swap_evaluations = _run_final_restricted_swap(
        context,
        dense_best,
        importance,
        add_summary,
        add_evaluations,
        active_lambdas,
        resume,
    )
    final_candidates = _build_final_candidate_pool(
        dense_evaluations,
        add_evaluations,
        swap_evaluations,
    )
    if not final_candidates:
        final_candidates = [dense_best]
    final_top10 = final_candidates[:10]
    final_best = final_candidates[0]
    _write_context_dependence(importance, add_summary, final_best)
    _write_numerical_conditioning(final_top10)
    cv_frame, cv_stability = _run_query_state_cv(context, final_top10)
    validation = _run_validation(context, final_top10)
    primary_add_improvements = int(np.count_nonzero(add_summary.get("delta_N_self", pd.Series(dtype=float)) > 0)) if not add_summary.empty else 0
    primary_swap_improvements = int(np.count_nonzero(swap_summary.get("delta_N_self", pd.Series(dtype=float)) > 0)) if not swap_summary.empty else 0
    result = {
        "task": TASK_NAME,
        "status": "SUCCESS" if validation["pass"] else "VALIDATION_FAILED",
        "objective_version": OBJECTIVE_VERSION,
        "historical_registry_hash": registry_hash,
        "stop_reason": stop_reason,
        "rounds_completed": rounds_completed,
        "best_candidate": final_best.candidate.to_record(),
        "best_N_self": int(final_best.summary["N_self"]),
        "best_N_fallback": int(final_best.summary["N_fallback"]),
        "best_N_valid": int(final_best.summary["N_valid"]),
        "best_N_fail": int(final_best.summary["N_fail"]),
        "best_self_Q05": final_best.summary["self_Q05"],
        "best_self_MRR": final_best.summary["self_MRR"],
        "best_fallback_Q05": final_best.summary["fallback_Q05"],
        "best_fallback_MRR": final_best.summary["fallback_MRR"],
        "best_Self_Top3": final_best.summary["Self_Top3"],
        "final_top10_count": len(final_top10),
        "primary_single_add_improvement_count": primary_add_improvements,
        "primary_swap_improvement_count": primary_swap_improvements,
        "nested_cv_status": "deferred",
        "validation": validation,
        "cv_winner_counts": cv_stability.sort_values("train_winner_count", ascending=False).head(10).to_dict("records"),
        "raw_data_modified": False,
        "result_root": str(RESULT_ROOT),
    }
    _write_json(RESULT_ROOT / "22_final_result_summary.json", result)
    summary_lines = [
        f"Task: {TASK_NAME}",
        f"OBJECTIVE_VERSION={OBJECTIVE_VERSION}",
        f"Stop reason: {stop_reason}",
        f"Rounds completed: {rounds_completed}",
        f"Final candidate: {final_best.candidate.candidate_name}",
        f"Support: {json.dumps(list(final_best.candidate.basis_ids), ensure_ascii=False)}",
        f"lambda={final_best.candidate.lambda_value}",
        f"N_self={final_best.summary['N_self']}",
        f"N_fallback={final_best.summary['N_fallback']}",
        f"N_valid={final_best.summary['N_valid']}",
        f"N_fail={final_best.summary['N_fail']}",
        f"self_Q05={final_best.summary['self_Q05']}",
        f"self_MRR={final_best.summary['self_MRR']}",
        f"fallback_Q05={final_best.summary['fallback_Q05']}",
        f"fallback_MRR={final_best.summary['fallback_MRR']}",
        f"Self_Top3={final_best.summary['Self_Top3']}",
        "",
        "Final Top10:",
    ]
    for rank, evaluation in enumerate(final_top10, start=1):
        summary_lines.append(
            f"rank={rank} {evaluation.candidate.candidate_name}: K={evaluation.candidate.K}, lambda={evaluation.candidate.lambda_value}, N_self={evaluation.summary['N_self']}, N_fallback={evaluation.summary['N_fallback']}, N_valid={evaluation.summary['N_valid']}, N_fail={evaluation.summary['N_fail']}"
        )
    summary_lines.extend(
        [
            "",
            f"Nested CV: {result['nested_cv_status']}",
            f"Validation: {json.dumps(validation, ensure_ascii=False, default=_json_default)}",
            "Search result is a searched-neighborhood result; it is not a global-optimum claim.",
        ]
    )
    (RESULT_ROOT / "22_final_result_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    _checkpoint(context, phase="completed", **result)
    _append_log(f"[{_now()}] Completed {TASK_NAME}: {json.dumps(result, ensure_ascii=False, default=_json_default)}")
    _append_handoff(
        f"完成 Self-First joint support+Ridge 搜索；OBJECTIVE_VERSION={OBJECTIVE_VERSION}；stop_reason={stop_reason}；best N_self={result['best_N_self']}, N_valid={result['best_N_valid']}；结果目录：{RESULT_ROOT}"
    )
    print(json.dumps(result, ensure_ascii=False, default=_json_default), flush=True)
    return result


def _resume_completed_structure_search(
    context: RuntimeContext,
    checkpoint: Mapping[str, Any],
    resume: bool,
) -> dict[str, Any]:
    """Continue from a legacy completed-search checkpoint without rerunning search rounds."""

    registry_path = RESULT_ROOT / "02_historical_support_registry.csv"
    if not registry_path.is_file():
        raise RuntimeError("completed-search resume requires the historical support registry")
    registry = pd.read_csv(registry_path)
    required_columns = {"support_hash", "basis_ids", "lambda", "source_task"}
    if not required_columns.issubset(registry.columns):
        missing = sorted(required_columns.difference(registry.columns))
        raise RuntimeError(f"completed-search resume registry is missing columns: {missing}")
    registry_hash = str(
        checkpoint.get("historical_registry_hash")
        or sha256_json(registry[["support_hash", "basis_ids", "lambda", "source_task"]].to_dict("records"))
    )
    active_lambdas = [float(value) for value in checkpoint.get("active_lambda_set", [])]
    if not active_lambdas:
        raise RuntimeError("completed-search resume requires active_lambda_set")
    round_numbers = []
    search_round_root = RESULT_ROOT / "search_rounds"
    for path in search_round_root.glob("round_*"):
        try:
            round_numbers.append(int(path.name.split("_")[-1]))
        except ValueError:
            continue
    rounds_completed = int(checkpoint.get("rounds_completed") or max(round_numbers, default=0))
    if rounds_completed <= 0:
        raise RuntimeError("completed-search resume could not determine completed search rounds")
    stop_reason = str(checkpoint.get("stop_reason") or "MAX_SEARCH_ROUNDS_REACHED")
    _append_log(
        f"[{_now()}] Resuming from legacy completed structure-search checkpoint; "
        f"skipping completed Add/Delete rounds 1-{rounds_completed} and entering Final stages."
    )
    return _finish_search(
        context,
        [],
        active_lambdas,
        registry_hash,
        stop_reason,
        rounds_completed,
        resume,
    )


def _structure_search_artifacts_complete() -> bool:
    """Check the durable artifacts that prove all six Add/Delete rounds are complete."""

    required = [RESULT_ROOT / "08_initial_beam.csv"]
    for round_index in range(1, MAX_FORWARD_BACKWARD_ROUNDS + 1):
        round_root = RESULT_ROOT / "search_rounds" / f"round_{round_index:02d}"
        required.extend(
            (
                round_root / "support_lambda_results_forward.csv",
                round_root / "support_lambda_results_backward.csv",
                round_root / "beam_after_forward.csv",
                round_root / "beam_after_backward.csv",
            )
        )
    return all(path.is_file() for path in required)


def _run(context: RuntimeContext, resume: bool) -> dict[str, Any]:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    checkpoint = None
    if resume:
        checkpoint = _verify_checkpoint(context)
        completed_status = checkpoint and checkpoint.get("phase") == "completed" and checkpoint.get("status") == "SEARCH_COMPLETE"
        final_progress = checkpoint and str(checkpoint.get("phase", "")).startswith("final_")
        completed_artifacts = checkpoint and _structure_search_artifacts_complete()
        if completed_status or final_progress or completed_artifacts:
            return _resume_completed_structure_search(context, checkpoint, resume)
    _write_task_definition(context)
    registry, audit = build_registry(PROJECT_ROOT)
    registry_hash = sha256_json(registry[["support_hash", "basis_ids", "lambda", "source_task"]].to_dict("records"))
    registry["historical_registry_hash"] = registry_hash
    registry.to_csv(RESULT_ROOT / "02_historical_support_registry.csv", index=False)
    _write_reuse_audit(context, registry, audit)
    _write_fallback_availability(context)
    _checkpoint(context, phase="phase0_preflight_and_registry", historical_registry_hash=registry_hash, registry_candidate_count=int(registry.shape[0]))
    ranking, _, _ = _build_historical_ranking(context, registry)
    _checkpoint(context, phase="phase1_historical_self_first_reranking", historical_matrix_candidate_count=int(ranking.shape[0]), historical_registry_hash=registry_hash)
    calibration_best, active_lambdas = _calibrate_ridge(context, ranking, resume)
    _checkpoint(context, phase="phase2_ridge_calibration", active_lambda_set=active_lambdas, calibration_support_count=int(calibration_best.shape[0]), historical_registry_hash=registry_hash)
    initial_beam = _initial_beam(context, calibration_best, resume)
    _checkpoint(context, phase="phase4_initial_beam", active_lambda_set=active_lambdas, initial_beam=[item.candidate.to_record() for item in initial_beam], historical_registry_hash=registry_hash)
    final_beam, edge_rows, stop_reason, rounds_completed = _run_search(
        context,
        initial_beam,
        active_lambdas,
        resume,
    )
    return _finish_search(
        context,
        final_beam,
        active_lambdas,
        registry_hash,
        stop_reason,
        rounds_completed,
        resume,
    )


def _run_registry_only(context: RuntimeContext) -> pd.DataFrame:
    """Execute the inexpensive Phase 0/1 registry and reranking checkpoint."""

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_task_definition(context)
    registry, audit = build_registry(PROJECT_ROOT)
    registry_hash = sha256_json(
        registry[["support_hash", "basis_ids", "lambda", "source_task"]].to_dict("records")
    )
    registry["historical_registry_hash"] = registry_hash
    registry.to_csv(RESULT_ROOT / "02_historical_support_registry.csv", index=False)
    _write_reuse_audit(context, registry, audit)
    _write_fallback_availability(context)
    _checkpoint(
        context,
        phase="phase0_preflight_and_registry",
        historical_registry_hash=registry_hash,
        registry_candidate_count=int(registry.shape[0]),
    )
    ranking, _, _ = _build_historical_ranking(context, registry)
    _checkpoint(
        context,
        phase="phase1_historical_self_first_reranking",
        historical_matrix_candidate_count=int(ranking.shape[0]),
        historical_registry_hash=registry_hash,
    )
    return ranking


def main() -> None:
    resume = "--resume" in sys.argv[1:]
    context = _build_context()
    if "--phase" in sys.argv[1:]:
        phase_index = sys.argv.index("--phase")
        if phase_index + 1 >= len(sys.argv):
            raise SystemExit("--phase requires a value")
        phase = sys.argv[phase_index + 1]
        if phase == "registry":
            _run_registry_only(context)
            return
        raise SystemExit(f"unsupported phase: {phase}")
    _run(context, resume=resume)


if __name__ == "__main__":
    __import__("multiprocessing").freeze_support()
    main()
