# ruff: noqa: E402,E501,I001

"""Run unified, deduplicated floating backward elimination from three K13 frontiers."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, replace
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
from retrieval_oriented_model_selection.self_hit_oriented.scenario_2.scenario_2_beam3_k13_floating_backward_retrieval_ablation_5b import (  # noqa: E402
    floating_backward_backend as backend,
)
from retrieval_oriented_model_selection.shared import mp10_k9_retrieval_runner as k9_utils
from signal_segmentation.shared import build_off_segments, build_partition_from_xin  # noqa: E402

TASK_NAME = "scenario_2_beam3_k13_floating_backward_retrieval_ablation_5b"
RESULT_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /TASK_NAME
WORK_LOG = PROJECT_ROOT / "work_logs" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /TASK_NAME / "execution_log.txt"
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"
BEAM_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /"scenario_2_smallbeam3_k13_retrieval_oriented_forward_expansion_5b"
K12_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /"scenario_2_k12_retrieval_oriented_forward_scan_5b"
K11_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /"scenario_2_mp10_k11_retrieval_oriented_forward_scan_5b"
K9_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /"scenario_2_mp10_k9_retrieval_oriented_basis_ablation_5b"
CORE_MODEL_NAME = "MP10_plus_ENV_p04_m2_q0"
FRONTIER_SPECS = (
    (0, "frontier_A_global_best", 74, "K13_plus_ENV_p03_m0_q1__ENV_p03_m1_q0", ("ENV_p03_m0_q1", "ENV_p03_m1_q0")),
    (1, "frontier_B_safe_best", 125, "K13_plus_ENV_p03_m0_q1__ENV_p09_m2_q0", ("ENV_p03_m0_q1", "ENV_p09_m2_q0")),
    (2, "frontier_C_seedA_branch_best", 12, "K13_plus_ENV_p03_m1_q0__ENV_p04_m0_q1", ("ENV_p03_m1_q0", "ENV_p04_m0_q1")),
)
FRONTIER_A_ID = 0
FRONTIER_B_ID = 1
FRONTIER_C_ID = 2
CORE_K = backend.CORE_K
K13 = CORE_K + 2
K12 = backend.K12
STATE_COUNT = backend.STATE_COUNT
WORKER_COUNT = 10
BLAS_THREADS_PER_WORKER = 1
REAL_B_THRESHOLD_DB = -40.0
REAL_B_SANITY_TOLERANCE_DB = 1e-9
REPRESENTATIVE_STATES = (0, 42, 187, 325, 424)
STATE330 = 330


@dataclass(frozen=True)
class FrontierCandidate:
    candidate_id: int
    candidate_name: str
    frontier_id: str
    formal_model_id: int
    support_basis_ids: tuple[str, ...]
    support_hash: str
    extension_basis_ids: tuple[str, ...]
    extension_indices: tuple[int, ...]
    K: int = K13

    @property
    def support(self) -> tuple[int, ...]:
        return tuple(range(self.K))

    @property
    def removed_basis_id(self) -> str | None:
        return None


@dataclass(frozen=True)
class K12Candidate:
    candidate_id: int
    candidate_name: str
    parent_frontier_ids: tuple[str, ...]
    removed_basis_by_parent: tuple[str, ...]
    support_basis_ids: tuple[str, ...]
    support_hash: str
    removed_core_basis_id: str | None
    removed_core_index: int | None
    extension_basis_ids: tuple[str, ...]
    extension_indices: tuple[int, ...]
    is_historical_reuse: bool
    historical_source_task: str | None
    historical_source_model_name: str | None
    historical_source_model_id: int | None
    K: int = K12

    @property
    def support(self) -> tuple[int, ...]:
        return tuple(range(self.K))

    @property
    def removed_basis_id(self) -> str | None:
        return self.removed_core_basis_id


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
        RESULT_ROOT / "31_checkpoint.json",
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
            "frontier_count": 3,
            "raw_deletion_edges": 39,
            "unique_k12_children": 37,
            "historical_reuse_count": 4,
            "novel_k12_fit_count": 33,
            "evaluation_pool_count": 40,
            "core_K": CORE_K,
            "parent_K": K13,
            "child_K": K12,
            "dmax_fixed": backend.MP_DMAX,
            "ridge_lambda_fixed": backend.RIDGE_LAMBDA,
            "second_backward_performed": False,
            "k14_performed": False,
            "second_beam_add_performed": False,
            "swap_performed": False,
            "multi_basis_deletion_performed": False,
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
        raise RuntimeError(f"Real-B random pair check failed: {maximum:.3e} dB")
    return maximum


def _load_core(terms: tuple[EnvelopeBasis, ...]) -> tuple[tuple[str, ...], dict[str, EnvelopeBasis], int, pd.DataFrame]:
    frame = pd.read_csv(K11_ROOT / "04_candidate_supports.csv")
    row = frame.loc[frame["model_name"].eq(CORE_MODEL_NAME)]
    if row.shape[0] != 1 or int(row.iloc[0]["K"]) != CORE_K:
        raise RuntimeError("formal K11 core support is missing")
    core_ids = tuple(json.loads(row.iloc[0]["support_basis_ids"]))
    by_id = {term.basis_id: term for term in terms}
    if len(core_ids) != CORE_K or core_ids[-1] != backend.CORE_BASIS_ID or any(basis_id not in by_id for basis_id in core_ids):
        raise RuntimeError(f"formal K11 core changed: {core_ids}")
    if str(row.iloc[0]["support_hash"]) != _support_hash(core_ids):
        raise RuntimeError("formal K11 core support hash changed")
    return core_ids, by_id, int(by_id[backend.CORE_BASIS_ID].index), frame


def _load_references(core_ids: tuple[str, ...]) -> dict[str, Any]:
    beam_summary = pd.read_csv(BEAM_ROOT / "12_retrieval_summary.csv")
    beam_model = pd.read_csv(BEAM_ROOT / "09_model_quality_long.csv")
    beam_query = pd.read_csv(BEAM_ROOT / "13_query_metrics_long.csv")
    with np.load(BEAM_ROOT / "09_all_candidate_coefficients.npz", allow_pickle=False) as data:
        beam_ids = np.asarray(data["model_ids"], dtype=np.int64)
        beam_theta_a = np.asarray(data["theta_Aend_padded"])
        beam_theta_c = np.asarray(data["theta_C2_padded"])
    k12_support = pd.read_csv(K12_ROOT / "06_candidate_supports.csv")
    k12_summary = pd.read_csv(K12_ROOT / "10_candidate_retrieval_summary.csv")
    k12_model = pd.read_csv(K12_ROOT / "08_candidate_model_quality_long.csv")
    k12_query = pd.read_csv(K12_ROOT / "11_candidate_query_metrics_long.csv")
    with np.load(K12_ROOT / "09_candidate_coefficients.npz", allow_pickle=False) as data:
        k12_ids = np.asarray(data["model_ids"], dtype=np.int64)
        k12_theta_a = np.asarray(data["theta_Aend_padded"])
        k12_theta_c = np.asarray(data["theta_C2_padded"])
    k11_summary = pd.read_csv(K11_ROOT / "08_candidate_retrieval_summary.csv")
    k11_model = pd.read_csv(K11_ROOT / "06_candidate_model_quality_long.csv")
    k11_query = pd.read_csv(K11_ROOT / "09_candidate_query_metrics_long.csv")
    real_b = np.load(K9_ROOT / "04_realB_ground_truth_cnmse_matrix.npy")
    shareability = np.load(K9_ROOT / "05_realB_shareability_mask.npy")
    formal_frontiers: dict[str, dict[str, Any]] = {}
    for frontier_id, frontier_name, formal_id, formal_name, extensions in FRONTIER_SPECS:
        row = beam_summary.loc[beam_summary["model_id"].eq(formal_id)]
        if row.shape[0] != 1 or str(row.iloc[0]["model_name"]) != formal_name or int(row.iloc[0]["top1_pass_count"]) != 421:
            raise RuntimeError(f"formal frontier changed: {frontier_name}")
        support_row = pd.read_csv(BEAM_ROOT / "07_unique_k13_supports.csv").loc[pd.read_csv(BEAM_ROOT / "07_unique_k13_supports.csv")["child_model_id"].eq(formal_id)]
        expected_support = core_ids + tuple(sorted(extensions, key=lambda basis_id: next(term.index for term in build_envelope_dictionary() if term.basis_id == basis_id)))
        if support_row.shape[0] != 1 or tuple(json.loads(support_row.iloc[0]["basis_ids"])) != expected_support:
            raise RuntimeError(f"formal frontier support changed: {frontier_name}")
        formal_frontiers[frontier_name] = {"formal_id": formal_id, "formal_name": formal_name, "summary": row.iloc[0].to_dict(), "support_ids": expected_support}
    historical_children: dict[str, dict[str, Any]] = {}
    for basis_id in ("ENV_p03_m0_q1", "ENV_p03_m1_q0", "ENV_p09_m2_q0", "ENV_p04_m0_q1"):
        support_ids = core_ids + (basis_id,)
        support_row = k12_support.loc[k12_support["support_hash"].eq(_support_hash(support_ids))]
        if support_row.shape[0] != 1 or int(support_row.iloc[0]["K"]) != K12:
            raise RuntimeError(f"historical K12 child is missing: {basis_id}")
        source_name = str(support_row.iloc[0]["model_name"])
        source_id = int(support_row.iloc[0]["model_id"])
        historical_children[_support_hash(support_ids)] = {"basis_id": basis_id, "source_name": source_name, "source_id": source_id, "support_ids": support_ids}
    k11_core_summary = k11_summary.loc[k11_summary["model_name"].eq(CORE_MODEL_NAME)]
    core_query = k11_query.loc[k11_query["candidate_id"].eq(21)].sort_values("State_R").reset_index(drop=True)
    if k11_core_summary.shape[0] != 1 or core_query.shape[0] != STATE_COUNT:
        raise RuntimeError("formal K11 core retrieval reference is missing")
    k9_folds = pd.read_csv(K9_ROOT / "12_query_state_cv_folds.csv").sort_values("state_id").reset_index(drop=True)
    k11_folds = pd.read_csv(K11_ROOT / "13_query_state_cv_folds.csv").sort_values("state_id").reset_index(drop=True)
    k12_folds = pd.read_csv(K12_ROOT / "16_query_state_cv_folds.csv").sort_values("state_id").reset_index(drop=True)
    if not np.array_equal(k9_folds[["state_id", "fold"]].to_numpy(), k11_folds[["state_id", "fold"]].to_numpy()) or not np.array_equal(k9_folds[["state_id", "fold"]].to_numpy(), k12_folds[["state_id", "fold"]].to_numpy()):
        raise RuntimeError("historical Query-State folds differ")
    historical_seed_a_query = k12_query.loc[k12_query["candidate_id"].eq(16)].sort_values("State_R").reset_index(drop=True)
    if historical_seed_a_query.shape[0] != STATE_COUNT:
        raise RuntimeError("historical Seed A query reference is not 425 rows")
    return {"beam_summary": beam_summary, "beam_model": beam_model, "beam_query": beam_query, "beam_ids": beam_ids, "beam_theta_a": beam_theta_a, "beam_theta_c": beam_theta_c, "k12_support": k12_support, "k12_summary": k12_summary, "k12_model": k12_model, "k12_query": k12_query, "k12_ids": k12_ids, "k12_theta_a": k12_theta_a, "k12_theta_c": k12_theta_c, "k11_summary": k11_summary, "k11_model": k11_model, "k11_query": k11_query, "real_b": real_b, "shareability": shareability, "folds": k12_folds, "formal_frontiers": formal_frontiers, "historical_children": historical_children, "historical_seedA_query": historical_seed_a_query, "core_summary": k11_core_summary.iloc[0].to_dict(), "core_model": k11_model.loc[k11_model["model_id"].eq(21)], "core_query": core_query}


def _build_candidates(core_ids: tuple[str, ...], by_id: dict[str, EnvelopeBasis], references: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, tuple[FrontierCandidate, ...], tuple[K12Candidate, ...], pd.DataFrame]:
    frontiers: list[FrontierCandidate] = []
    frontier_rows: list[dict[str, Any]] = []
    for frontier_id, frontier_name, formal_id, formal_name, extensions in FRONTIER_SPECS:
        extension_ids = tuple(sorted(extensions, key=lambda basis_id: by_id[basis_id].index))
        support_ids = core_ids + extension_ids
        frontiers.append(FrontierCandidate(frontier_id, frontier_name, frontier_name, formal_id, support_ids, _support_hash(support_ids), extension_ids, tuple(int(by_id[basis_id].index) for basis_id in extension_ids)))
        frontier_rows.append({"frontier_id": frontier_name, "frontier_model_id": formal_id, "formal_model_name": formal_name, "extension_basis_1": extension_ids[0], "extension_basis_2": extension_ids[1], "support_basis_ids": json.dumps(list(support_ids), ensure_ascii=False), "support_hash": _support_hash(support_ids), "K": K13, "dmax": backend.MP_DMAX, "ridge_lambda": backend.RIDGE_LAMBDA})
    children_by_hash: dict[str, K12Candidate] = {}
    parent_map: dict[str, list[str]] = {}
    removed_map: dict[str, list[str]] = {}
    edge_rows: list[dict[str, Any]] = []
    edge_id = 0
    for frontier in frontiers:
        for local_index, removed_basis_id in enumerate(frontier.support_basis_ids):
            child_support = tuple(basis_id for index, basis_id in enumerate(frontier.support_basis_ids) if index != local_index)
            child_hash = _support_hash(child_support)
            removed_term = by_id[removed_basis_id]
            removed_core = removed_basis_id in core_ids
            if child_hash not in children_by_hash:
                child_id = 3 + len(children_by_hash)
                extension_ids = tuple(basis_id for basis_id in child_support if basis_id not in core_ids)
                extension_indices = tuple(int(by_id[basis_id].index) for basis_id in extension_ids)
                historical = references["historical_children"].get(child_hash)
                children_by_hash[child_hash] = K12Candidate(child_id, f"child_K12_{child_id:02d}", (frontier.frontier_id,), (f"{frontier.frontier_id}:-{removed_basis_id}",), child_support, child_hash, removed_basis_id if removed_core else None, local_index if removed_core else None, extension_ids, extension_indices, historical is not None, "scenario_2_k12_retrieval_oriented_forward_scan_5b" if historical else None, historical["source_name"] if historical else None, historical["source_id"] if historical else None)
                parent_map[child_hash] = [frontier.frontier_id]
                removed_map[child_hash] = [f"{frontier.frontier_id}:-{removed_basis_id}"]
                is_duplicate = False
            else:
                parent_map[child_hash].append(frontier.frontier_id)
                removed_map[child_hash].append(f"{frontier.frontier_id}:-{removed_basis_id}")
                is_duplicate = True
                child_id = children_by_hash[child_hash].candidate_id
            edge_rows.append({"edge_id": edge_id, "parent_frontier_id": frontier.frontier_id, "parent_model_id": frontier.formal_model_id, "parent_support_hash": frontier.support_hash, "removed_basis_id": removed_basis_id, "removed_p": int(removed_term.order), "removed_m": removed_term.signal_delay, "removed_q": removed_term.envelope_delay, "removed_is_cross_delay": bool(removed_term.family == "CROSS_ENVELOPE"), "child_support_hash": child_hash, "child_model_id": child_id, "child_known_from_previous_task": child_hash in references["historical_children"], "is_duplicate_child": is_duplicate, "child_generation_count": 0})
            edge_id += 1
    children: list[K12Candidate] = []
    for child in children_by_hash.values():
        parents = tuple(parent_map[child.support_hash])
        removals = tuple(removed_map[child.support_hash])
        children.append(replace(child, parent_frontier_ids=parents, removed_basis_by_parent=removals))
    finalized_by_hash = {child.support_hash: child for child in children}
    for edge in edge_rows:
        edge["child_generation_count"] = len(finalized_by_hash[edge["child_support_hash"]].parent_frontier_ids)
    edge_frame = pd.DataFrame(edge_rows).sort_values("edge_id").reset_index(drop=True)
    unique_rows = [{"child_model_id": child.candidate_id, "model_name": child.candidate_name, "support_hash": child.support_hash, "K": child.K, "basis_ids": json.dumps(list(child.support_basis_ids), ensure_ascii=False), "parent_frontier_ids": json.dumps(list(child.parent_frontier_ids), ensure_ascii=False), "generation_count": len(child.parent_frontier_ids), "removed_basis_by_parent": json.dumps(list(child.removed_basis_by_parent), ensure_ascii=False), "is_historical_reuse": child.is_historical_reuse, "historical_source_task": child.historical_source_task or "", "historical_source_model_name": child.historical_source_model_name or "", "historical_source_model_id": child.historical_source_model_id if child.historical_source_model_id is not None else np.nan, "removed_core_basis_id": child.removed_core_basis_id or "", "removed_core_index": child.removed_core_index if child.removed_core_index is not None else np.nan, "extension_basis_1": child.extension_basis_ids[0] if child.extension_basis_ids else "NONE", "extension_basis_2": child.extension_basis_ids[1] if len(child.extension_basis_ids) > 1 else "NONE", "support_indices": json.dumps(list(range(K12)))} for child in children]
    unique_frame = pd.DataFrame(unique_rows).sort_values("child_model_id").reset_index(drop=True)
    duplicate_frame = unique_frame.loc[unique_frame["generation_count"].gt(1), ["child_model_id", "extension_basis_1", "extension_basis_2", "parent_frontier_ids", "removed_basis_by_parent", "generation_count"]].copy()
    expected_pairs = {("ENV_p03_m0_q1", "NONE"), ("ENV_p03_m1_q0", "NONE")}
    observed_pairs = {tuple(row) for row in duplicate_frame[["extension_basis_1", "extension_basis_2"]].itertuples(index=False, name=None)}
    if edge_frame.shape[0] != 39 or unique_frame.shape[0] != 37 or duplicate_frame.shape[0] != 2 or observed_pairs != expected_pairs:
        raise RuntimeError(f"floating deletion graph changed: raw={edge_frame.shape[0]}, unique={unique_frame.shape[0]}, duplicate={observed_pairs}")
    dedup_frame = duplicate_frame.reset_index(drop=True)
    return pd.DataFrame(frontier_rows), edge_frame, tuple(frontiers), tuple(children), dedup_frame


def _candidate_metadata(candidate: FrontierCandidate | K12Candidate) -> dict[str, Any]:
    if isinstance(candidate, FrontierCandidate):
        parent_ids = (candidate.frontier_id,)
        removed_by_parent: tuple[str, ...] = ()
        model_type = "K13_parent"
        historical = False
        source_task = "scenario_2_smallbeam3_k13_retrieval_oriented_forward_expansion_5b"
        source_name = candidate.candidate_name
        source_id = candidate.formal_model_id
        extension_1 = candidate.extension_basis_ids[0]
        extension_2 = candidate.extension_basis_ids[1]
    else:
        parent_ids = candidate.parent_frontier_ids
        removed_by_parent = candidate.removed_basis_by_parent
        model_type = "K12_child"
        historical = candidate.is_historical_reuse
        source_task = candidate.historical_source_task
        source_name = candidate.historical_source_model_name
        source_id = candidate.historical_source_model_id
        extension_1 = candidate.extension_basis_ids[0] if candidate.extension_basis_ids else "NONE"
        extension_2 = candidate.extension_basis_ids[1] if len(candidate.extension_basis_ids) > 1 else "NONE"
    return {
        "model_id": candidate.candidate_id,
        "model_name": candidate.candidate_name,
        "model_type": model_type,
        "K": candidate.K,
        "support_hash": candidate.support_hash,
        "support_basis_ids": json.dumps(list(candidate.support_basis_ids), ensure_ascii=False),
        "parent_frontier_ids": json.dumps(list(parent_ids), ensure_ascii=False),
        "removed_basis_by_parent": json.dumps(list(removed_by_parent), ensure_ascii=False),
        "extension_basis_1": extension_1,
        "extension_basis_2": extension_2,
        "is_historical_reuse": historical,
        "historical_source_task": source_task or "",
        "historical_source_model_name": source_name or "",
        "historical_source_model_id": source_id if source_id is not None else np.nan,
    }


def _decorate_model_rows(frame: pd.DataFrame, candidate: FrontierCandidate | K12Candidate) -> pd.DataFrame:
    result = frame.copy()
    metadata = _candidate_metadata(candidate)
    for key, value in metadata.items():
        result[key] = value
    result["State_ID"] = result["State_ID"].astype(int)
    result["State_R"] = result["State_R"].astype(int)
    return result


def _decorate_query_frame(frame: pd.DataFrame, candidate: FrontierCandidate | K12Candidate) -> pd.DataFrame:
    result = frame.copy()
    metadata = _candidate_metadata(candidate)
    result["candidate_id"] = candidate.candidate_id
    result["candidate_name"] = candidate.candidate_name
    for key, value in metadata.items():
        result[key] = value
    if "retrieved_real_B_CNMSE_dB" not in result:
        result["retrieved_real_B_CNMSE_dB"] = result["retrieved_realB_CNMSE_dB"]
    return result.sort_values("State_R").reset_index(drop=True)


def _summary_from_frame(frame: pd.DataFrame, candidate: FrontierCandidate | K12Candidate) -> dict[str, Any]:
    summary = k9_utils._summarize_query_frame(frame, candidate)
    summary.update(_candidate_metadata(candidate))
    summary["top1_pass_count"] = summary["top1_realB_pass_count"]
    summary["top1_pass_rate"] = summary["top1_realB_pass_rate"]
    values = frame["top12_margin_dB"].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    summary["top12_margin_Q05"] = float(np.quantile(finite, 0.05)) if finite.size else float("nan")
    summary["top12_margin_Q10"] = float(np.quantile(finite, 0.10)) if finite.size else float("nan")
    return summary


def _candidate_phi_for_common(common_core: np.ndarray, common_env: np.ndarray, candidate: FrontierCandidate | K12Candidate) -> np.ndarray:
    return np.column_stack((common_core, *(common_env[:, index] for index in candidate.extension_indices))).astype(np.complex128)


def _load_parent_payload(candidate: FrontierCandidate, references: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    old_id = candidate.formal_model_id
    model = _decorate_model_rows(references["beam_model"].loc[references["beam_model"]["model_id"].eq(old_id)], candidate)
    query = _decorate_query_frame(references["beam_query"].loc[references["beam_query"]["model_id"].eq(old_id)], candidate)
    distance = np.load(BEAM_ROOT / "distance_matrices" / f"k13_{old_id:03d}.npy")
    index = int(np.flatnonzero(references["beam_ids"] == old_id)[0])
    theta_a = np.asarray(references["beam_theta_a"][index, :, :K13], dtype=np.complex128)
    theta_c = np.asarray(references["beam_theta_c"][index, :, :K13], dtype=np.complex128)
    return model, query, distance, theta_a, theta_c


def _load_historical_child_payload(candidate: K12Candidate, references: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    source_id = int(candidate.historical_source_model_id)
    model = _decorate_model_rows(references["k12_model"].loc[references["k12_model"]["model_id"].eq(source_id)], candidate)
    query = _decorate_query_frame(references["k12_query"].loc[references["k12_query"]["candidate_id"].eq(source_id)], candidate)
    source_name = str(candidate.historical_source_model_name)
    distance = np.load(K12_ROOT / "distance_matrices" / f"{source_name}.npy")
    index = int(np.flatnonzero(references["k12_ids"] == source_id)[0])
    theta_a = np.asarray(references["k12_theta_a"][index, :, :K12], dtype=np.complex128)
    theta_c = np.asarray(references["k12_theta_c"][index, :, :K12], dtype=np.complex128)
    return model, query, distance, theta_a, theta_c


def _seed_or_historical_regression(candidate: FrontierCandidate | K12Candidate, current_model: pd.DataFrame, current_query: pd.DataFrame, current_distance: np.ndarray, current_theta_a: np.ndarray, current_theta_c: np.ndarray, current_lut: np.ndarray, current_query_fp: np.ndarray, references: dict[str, Any], reference_kind: str) -> dict[str, Any]:
    if reference_kind == "parent":
        old_id = candidate.formal_model_id  # type: ignore[union-attr]
        reference_model = references["beam_model"].loc[references["beam_model"]["model_id"].eq(old_id)].sort_values("State_R").reset_index(drop=True)
        reference_query = references["beam_query"].loc[references["beam_query"]["model_id"].eq(old_id)].sort_values("State_R").reset_index(drop=True)
        index = int(np.flatnonzero(references["beam_ids"] == old_id)[0])
        reference_theta_a = references["beam_theta_a"][index, :, :K13]
        reference_theta_c = references["beam_theta_c"][index, :, :K13]
        reference_distance = np.load(BEAM_ROOT / "distance_matrices" / f"k13_{old_id:03d}.npy")
        fingerprint_stem = "global_best_k13" if old_id == 74 else "per_branch_best_C" if old_id == 125 else "per_branch_best_A"
        reference_lut = np.load(BEAM_ROOT / "fingerprints" / f"{fingerprint_stem}_lut.npy")
        reference_query_fp = np.load(BEAM_ROOT / "fingerprints" / f"{fingerprint_stem}_query.npy")
    else:
        old_id = int(candidate.historical_source_model_id)  # type: ignore[union-attr]
        reference_model = references["k12_model"].loc[references["k12_model"]["model_id"].eq(old_id)].sort_values("State_R").reset_index(drop=True)
        reference_query = references["k12_query"].loc[references["k12_query"]["candidate_id"].eq(old_id)].sort_values("State_R").reset_index(drop=True)
        index = int(np.flatnonzero(references["k12_ids"] == old_id)[0])
        reference_theta_a = references["k12_theta_a"][index, :, :K12]
        reference_theta_c = references["k12_theta_c"][index, :, :K12]
        source_name = str(candidate.historical_source_model_name)  # type: ignore[union-attr]
        reference_distance = np.load(K12_ROOT / "distance_matrices" / f"{source_name}.npy")
        current_phi = _candidate_phi_for_common(references["common_core"], references["common_env"], candidate)
        reference_lut = (current_phi @ reference_theta_a.T).T.astype(np.complex128)
        reference_query_fp = (current_phi @ reference_theta_c.T).T.astype(np.complex128)
    metric_columns = ["Y_Aend_train_NMSE_dB", "Y_Aend_B_NMSE_dB", "Y_C2_train_NMSE_dB", "Y_C2_B_NMSE_dB"]
    metric_errors = {column: float(np.max(np.abs(current_model[column].to_numpy(dtype=float) - reference_model[column].to_numpy(dtype=float)))) for column in metric_columns}
    theta_a_error = float(np.max(np.abs(current_theta_a - reference_theta_a)))
    theta_c_error = float(np.max(np.abs(current_theta_c - reference_theta_c)))
    lut_error = float(np.max(np.abs(current_lut - reference_lut)))
    query_fp_error = float(np.max(np.abs(current_query_fp - reference_query_fp)))
    distance_error, distance_mismatch = _max_abs_with_neginf(current_distance, reference_distance)
    retrieved_error, retrieved_mismatch = _max_abs_with_neginf(current_query["retrieved_realB_CNMSE_dB"].to_numpy(dtype=float), reference_query["retrieved_realB_CNMSE_dB"].to_numpy(dtype=float))
    state_q_equal = bool(np.array_equal(current_query["State_Q"].to_numpy(dtype=int), reference_query["State_Q"].to_numpy(dtype=int)))
    pass_equal = bool(np.array_equal(current_query["realB_pass"].to_numpy(dtype=bool), reference_query["realB_pass"].to_numpy(dtype=bool)))
    rank_equal = bool(np.array_equal(current_query["first_shareable_rank"].to_numpy(dtype=int), reference_query["first_shareable_rank"].to_numpy(dtype=int)))
    result = {"candidate": candidate.candidate_name, "reference_kind": reference_kind, "reference_model_id": old_id, "pass": bool(max(metric_errors.values()) < 1e-10 and theta_a_error < 1e-12 and theta_c_error < 1e-12 and lut_error < 1e-12 and query_fp_error < 1e-12 and distance_error < 1e-12 and distance_mismatch == 0 and retrieved_error < 1e-9 and retrieved_mismatch == 0 and state_q_equal and pass_equal and rank_equal), "model_metric_max_abs_error_dB": metric_errors, "theta_Aend_max_abs_error": theta_a_error, "theta_C2_max_abs_error": theta_c_error, "LUT_fingerprint_max_abs_error": lut_error, "Query_fingerprint_max_abs_error": query_fp_error, "distance_max_abs_error_dB": distance_error, "distance_nonfinite_pattern_mismatch": distance_mismatch, "retrieved_Real_B_max_abs_error_dB": retrieved_error, "retrieved_Real_B_nonfinite_pattern_mismatch": retrieved_mismatch, "State_Q_equal": state_q_equal, "realB_pass_equal": pass_equal, "first_shareable_rank_equal": rank_equal}
    if not result["pass"]:
        raise RuntimeError(f"HARD FAIL: {candidate.candidate_name} {reference_kind} regression failed: {result}")
    return result


def _run_novel_model_phase(novel_children: tuple[K12Candidate, ...], core_index: int, resume: bool) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    model_path = RESULT_ROOT / "08_novel_model_quality_long.csv"
    coefficient_path = RESULT_ROOT / "09_coefficients_novel_k12.npz"
    if resume and model_path.is_file() and coefficient_path.is_file():
        try:
            frame = pd.read_csv(model_path)
            with np.load(coefficient_path, allow_pickle=False) as data:
                theta_a = np.asarray(data["theta_Aend_padded"])
                theta_c = np.asarray(data["theta_C2_padded"])
            if frame.shape[0] == len(novel_children) * STATE_COUNT and theta_a.shape == (len(novel_children), STATE_COUNT, K12) and theta_c.shape == theta_a.shape:
                return frame, theta_a, theta_c
        except (OSError, ValueError, KeyError):
            pass
    specs = tuple((candidate.candidate_id, int(candidate.removed_core_index), candidate.extension_indices) for candidate in novel_children)
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=WORKER_COUNT, mp_context=get_context("spawn"), initializer=backend.worker_init, initargs=(specs, core_index)) as executor:
        futures = {executor.submit(backend.model_worker_entry, state_id): state_id for state_id in range(STATE_COUNT)}
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed % 50 == 0 or completed == STATE_COUNT:
                _checkpoint(phase="novel_k12_model_progress", completed_model_states=completed, completed_model_state_ids=sorted(int(item["State_n_R"]) for item in results))
                print(f"[FLOATING NOVEL K12 MODEL] {completed}/{STATE_COUNT}", flush=True)
    results.sort(key=lambda item: int(item["State_n_R"]))
    if [int(item["State_n_R"]) for item in results] != list(range(STATE_COUNT)):
        raise RuntimeError("novel K12 model phase state order changed")
    candidate_by_id = {candidate.candidate_id: candidate for candidate in novel_children}
    theta_a = np.zeros((len(novel_children), STATE_COUNT, K12), dtype=np.complex128)
    theta_c = np.zeros_like(theta_a)
    local_index = {candidate.candidate_id: index for index, candidate in enumerate(novel_children)}
    rows: list[dict[str, Any]] = []
    for item in results:
        state_id = int(item["State_n_R"])
        for fit in item["candidates"]:
            candidate = candidate_by_id[int(fit["candidate_id"])]
            position = local_index[candidate.candidate_id]
            theta_a[position, state_id] = fit["theta_Aend_padded"]
            theta_c[position, state_id] = fit["theta_C2_padded"]
            rows.append({"model_id": candidate.candidate_id, "model_name": candidate.candidate_name, "K": K12, "support_hash": candidate.support_hash, "parent_frontier_ids": json.dumps(list(candidate.parent_frontier_ids), ensure_ascii=False), "removed_basis_by_parent": json.dumps(list(candidate.removed_basis_by_parent), ensure_ascii=False), "removed_core_basis_id": candidate.removed_core_basis_id, "removed_core_index": candidate.removed_core_index, "extension_basis_1": candidate.extension_basis_ids[0], "extension_basis_2": candidate.extension_basis_ids[1], "State_ID": state_id, "State_R": state_id, "ilc_A_end": int(item["ilc_A_end"]), "funMng": int(item["funMng"]), "funAng": int(item["funAng"]), "secMng": int(item["secMng"]), "secAng": int(item["secAng"]), "nmse_withoutdpd_dB": float(item["nmse_withoutdpd_dB"]), "ACPR_withoutdpd_mean_dBc": float(item["ACPR_withoutdpd_mean_dBc"]), "Y_Aend_train_NMSE_dB": float(fit["Y_Aend_train_NMSE_dB"]), "Y_Aend_B_NMSE_dB": float(fit["Y_Aend_B_NMSE_dB"]), "Y_C2_train_NMSE_dB": float(fit["Y_C2_train_NMSE_dB"]), "Y_C2_B_NMSE_dB": float(fit["Y_C2_B_NMSE_dB"]), "Aend_rank": int(fit["Aend_rank"]), "Aend_rank_augmented": int(fit["Aend_rank_augmented"]), "C2_rank": int(fit["C2_rank"]), "C2_rank_augmented": int(fit["C2_rank_augmented"]), "Aend_condition_number": float(fit["Aend_condition_number"]), "C2_condition_number": float(fit["C2_condition_number"]), "Aend_condition_number_augmented": float(fit["Aend_condition_number_augmented"]), "C2_condition_number_augmented": float(fit["C2_condition_number_augmented"]), "Aend_finite": bool(fit["finite"]), "C2_finite": bool(fit["finite"]), "is_historical_reuse": False, "historical_source_task": "", "historical_source_model_name": "", "historical_source_model_id": np.nan})
    frame = pd.DataFrame(rows).sort_values(["model_id", "State_R"]).reset_index(drop=True)
    if frame.shape[0] != len(novel_children) * STATE_COUNT:
        raise RuntimeError(f"novel K12 model quality row count failed: {frame.shape}")
    frame.to_csv(model_path, index=False)
    np.savez_compressed(coefficient_path, model_ids=np.asarray([candidate.candidate_id for candidate in novel_children], dtype=np.int64), model_names=np.asarray([candidate.candidate_name for candidate in novel_children]), theta_Aend_padded=theta_a, theta_C2_padded=theta_c, support_hashes=np.asarray([candidate.support_hash for candidate in novel_children]), removed_core_indices=np.asarray([candidate.removed_core_index for candidate in novel_children], dtype=np.int64), extension_indices=np.asarray([candidate.extension_indices for candidate in novel_children], dtype=np.int64))
    return frame, theta_a, theta_c


def _novel_retrieval(candidate: K12Candidate, theta_a_bank: np.ndarray, theta_c_bank: np.ndarray, common_phi: np.ndarray, real_b: np.ndarray, shareability: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    # The candidate ID remains the unified evaluation-pool ID; its coefficients
    # are placed at that ID so the canonical K9 retrieval helper is reused.
    frame, summary, distance, lut, query = k9_utils._candidate_retrieval(candidate, theta_a_bank, theta_c_bank, common_phi, real_b, shareability)
    frame = _decorate_query_frame(frame, candidate)
    summary = _summary_from_frame(frame, candidate)
    return frame, summary, distance, lut, query


def _compare_frames(current: pd.DataFrame, reference: pd.DataFrame) -> dict[str, Any]:
    current = current.sort_values("State_R").reset_index(drop=True)
    reference = reference.sort_values("State_R").reset_index(drop=True)
    current_pass = current["realB_pass"].to_numpy(dtype=bool)
    reference_pass = reference["realB_pass"].to_numpy(dtype=bool)
    recovered = np.flatnonzero(~reference_pass & current_pass).astype(int).tolist()
    regressed = np.flatnonzero(reference_pass & ~current_pass).astype(int).tolist()
    current_ranks = current["first_shareable_rank"].to_numpy(dtype=float)
    reference_ranks = reference["first_shareable_rank"].to_numpy(dtype=float)
    return {
        "Top1": int(current_pass.sum()),
        "Exact": int(current["exact_hit"].sum()),
        "nonself_count": int((current["State_Q"].to_numpy(dtype=int) != np.arange(len(current))).sum()),
        "nonself_pass": int(((current["State_Q"].to_numpy(dtype=int) != np.arange(len(current))) & current_pass).sum()),
        "Q05": float(current["shareability_margin_dB"].quantile(0.05)),
        "MRR": float(np.mean(1.0 / current_ranks)),
        "Top2": int(current["Top2_has_shareable"].sum()),
        "Top3": int(current["Top3_has_shareable"].sum()),
        "Top5": int(current["Top5_has_shareable"].sum()),
        "Top10": int(current["Top10_has_shareable"].sum()),
        "recovered_count": len(recovered),
        "recovered_state_ids": recovered,
        "regressed_count": len(regressed),
        "regressed_state_ids": regressed,
        "unchanged_pass": int((reference_pass & current_pass).sum()),
        "unchanged_fail": int((~reference_pass & ~current_pass).sum()),
        "delta_top1": int(current_pass.sum() - reference_pass.sum()),
        "delta_Q05": float(current["shareability_margin_dB"].quantile(0.05) - reference["shareability_margin_dB"].quantile(0.05)),
        "delta_MRR": float(np.mean(1.0 / current_ranks) - np.mean(1.0 / reference_ranks)),
    }


def _summary_from_loaded_frame(frame: pd.DataFrame, candidate: FrontierCandidate | K12Candidate) -> dict[str, Any]:
    return _summary_from_frame(_decorate_query_frame(frame, candidate), candidate)


def _fragile_summary(candidate: FrontierCandidate | K12Candidate, frame: pd.DataFrame) -> dict[str, Any]:
    selected_real_b = frame["retrieved_realB_CNMSE_dB"].to_numpy(dtype=float)
    passed = selected_real_b < REAL_B_THRESHOLD_DB
    headroom = -40.0 - selected_real_b
    fragile = passed & (headroom <= 1.0)
    finite_headroom = headroom[passed & np.isfinite(headroom)]
    return {"model_id": candidate.candidate_id, "model_name": candidate.candidate_name, "model_type": "K13_parent" if isinstance(candidate, FrontierCandidate) else "K12_child", "success_count": int(passed.sum()), "fragile_success_count": int(fragile.sum()), "fragile_fraction": float(fragile.sum() / passed.sum()) if passed.any() else float("nan"), "fragile_state_ids": json.dumps(np.flatnonzero(fragile).astype(int).tolist()), "minimum_headroom_dB": float(np.min(finite_headroom)) if finite_headroom.size else float("nan"), "Q05_headroom_dB": float(np.quantile(finite_headroom, 0.05)) if finite_headroom.size else float("nan"), "median_headroom_dB": float(np.median(finite_headroom)) if finite_headroom.size else float("nan")}


def _build_fragile_edge_fields(parent: pd.DataFrame, child: pd.DataFrame) -> dict[str, Any]:
    parent_real = parent["retrieved_realB_CNMSE_dB"].to_numpy(dtype=float)
    child_real = child["retrieved_realB_CNMSE_dB"].to_numpy(dtype=float)
    parent_pass = parent["realB_pass"].to_numpy(dtype=bool)
    child_pass = child["realB_pass"].to_numpy(dtype=bool)
    parent_headroom = -40.0 - parent_real
    parent_fragile = parent_pass & (parent_headroom <= 1.0)
    child_fragile = child_pass & ((-40.0 - child_real) <= 1.0)
    preserved = parent_fragile & child_fragile
    regressed = parent_fragile & ~child_pass
    return {"parent_fragile_count": int(parent_fragile.sum()), "fragile_preserved": int(preserved.sum()), "fragile_regressed": int(regressed.sum()), "fragile_regressed_state_ids": json.dumps(np.flatnonzero(regressed).astype(int).tolist())}


def _local_cv(candidates: tuple[FrontierCandidate | K12Candidate, ...], frames: dict[int, pd.DataFrame], folds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    frontier_names = [candidate.frontier_id for candidate in candidates[:3]]
    rows: list[dict[str, Any]] = []
    branch_rows: list[dict[str, Any]] = []
    for fold in range(5):
        validation_ids = folds.loc[folds["fold"].eq(fold), "state_id"].to_numpy(dtype=int)
        train_ids = folds.loc[~folds["fold"].eq(fold), "state_id"].to_numpy(dtype=int)
        train_summaries = {candidate.candidate_id: k9_utils._summarize_query_frame(frames[candidate.candidate_id].loc[frames[candidate.candidate_id]["State_R"].isin(train_ids)], candidate) for candidate in candidates}
        global_ranking = sorted(train_summaries.values(), key=k9_utils._selection_key)
        global_rank = {int(item["candidate_id"]): rank for rank, item in enumerate(global_ranking, start=1)}
        global_winner = int(global_ranking[0]["candidate_id"])
        branch_rankings: dict[str, dict[int, int]] = {}
        branch_winners: dict[str, int] = {}
        for frontier_name in frontier_names:
            branch_ids = [candidate.candidate_id for candidate in candidates if frontier_name == getattr(candidate, "frontier_id", None) or (isinstance(candidate, K12Candidate) and frontier_name in candidate.parent_frontier_ids)]
            ranking = sorted((train_summaries[candidate_id] for candidate_id in branch_ids), key=k9_utils._selection_key)
            branch_rankings[frontier_name] = {int(item["candidate_id"]): rank for rank, item in enumerate(ranking, start=1)}
            branch_winners[frontier_name] = int(ranking[0]["candidate_id"])
            branch_rows.append({"fold": fold, "frontier_id": frontier_name, "winner_model_id": branch_winners[frontier_name], "winner_model_name": next(candidate.candidate_name for candidate in candidates if candidate.candidate_id == branch_winners[frontier_name]), "train_query_count": len(train_ids), "validation_query_count": len(validation_ids)})
        for candidate in candidates:
            train_summary = train_summaries[candidate.candidate_id]
            validation_frame = frames[candidate.candidate_id].loc[frames[candidate.candidate_id]["State_R"].isin(validation_ids)]
            validation_summary = k9_utils._summarize_query_frame(validation_frame, candidate)
            branch_ranks = {frontier_name: branch_rankings[frontier_name][candidate.candidate_id] for frontier_name in frontier_names if candidate.candidate_id in branch_rankings[frontier_name]}
            rows.append({"fold": fold, "model_id": candidate.candidate_id, "model_name": candidate.candidate_name, "model_type": "K13_parent" if isinstance(candidate, FrontierCandidate) else "K12_child", "K": candidate.K, "train_query_count": len(train_ids), "validation_query_count": len(validation_ids), "train_is_global_winner": candidate.candidate_id == global_winner, "global_train_rank": global_rank[candidate.candidate_id], "frontier_train_rank": json.dumps(branch_ranks, ensure_ascii=False, sort_keys=True), "train_top1": int(train_summary["top1_realB_pass_count"]), "train_nonself_rate": float(train_summary["nonself_pass_rate"]), "train_Q05": float(train_summary["share_margin_Q05"]), "train_MRR": float(train_summary["MRR"]), "train_Top3": int(train_summary["Top3_oracle"]), "validation_exact": int(validation_summary["exact_hit_count"]), "validation_top1": int(validation_summary["top1_realB_pass_count"]), "validation_nonself_rate": float(validation_summary["nonself_pass_rate"]), "validation_Q05": float(validation_summary["share_margin_Q05"]), "validation_MRR": float(validation_summary["MRR"]), "validation_Top2": int(validation_summary["Top2_oracle"]), "validation_Top3": int(validation_summary["Top3_oracle"]), "validation_Top5": int(validation_summary["Top5_oracle"]), "validation_Top10": int(validation_summary["Top10_oracle"])})
    cv = pd.DataFrame(rows).sort_values(["fold", "model_id"]).reset_index(drop=True)
    stability_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        current = cv.loc[cv["model_id"].eq(candidate.candidate_id)]
        global_wins = current.loc[current["train_is_global_winner"]]
        branch_counts: dict[str, int] = {frontier_name: 0 for frontier_name in frontier_names}
        for value in current["frontier_train_rank"]:
            parsed = json.loads(value)
            for frontier_name in frontier_names:
                if parsed.get(frontier_name) == 1:
                    branch_counts[frontier_name] += 1
        stability_rows.append({"model_id": candidate.candidate_id, "model_name": candidate.candidate_name, "model_type": "K13_parent" if isinstance(candidate, FrontierCandidate) else "K12_child", "global_train_winner_count": int(global_wins.shape[0]), "global_train_winner_frequency": float(global_wins.shape[0] / 5), "global_winner_folds": json.dumps(global_wins["fold"].astype(int).tolist()), "frontier_A_winner_count": branch_counts[frontier_names[0]], "frontier_B_winner_count": branch_counts[frontier_names[1]], "frontier_C_winner_count": branch_counts[frontier_names[2]], "validation_top1_mean": float(current["validation_top1"].mean()), "validation_top1_min": int(current["validation_top1"].min()), "validation_top1_max": int(current["validation_top1"].max()), "global_winner_validation_top1_mean": float(global_wins["validation_top1"].mean()) if not global_wins.empty else float("nan"), "global_winner_validation_top1_min": int(global_wins["validation_top1"].min()) if not global_wins.empty else np.nan})
    return cv, pd.DataFrame(stability_rows).sort_values("model_id").reset_index(drop=True), pd.DataFrame(branch_rows).sort_values(["frontier_id", "fold"]).reset_index(drop=True)


def _build_edge_delta(edges: pd.DataFrame, candidates: tuple[FrontierCandidate | K12Candidate, ...], frames: dict[int, pd.DataFrame], summaries: pd.DataFrame, cv: pd.DataFrame, fragile: pd.DataFrame) -> pd.DataFrame:
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    frontier_by_name = {candidate.frontier_id: candidate for candidate in candidates[:3]}
    rows: list[dict[str, Any]] = []
    for edge in edges.itertuples(index=False):
        parent = frontier_by_name[str(edge.parent_frontier_id)]
        child = candidate_by_id[int(edge.child_model_id)]
        comparison = _compare_frames(frames[child.candidate_id], frames[parent.candidate_id])
        parent_summary = summaries.loc[summaries["model_id"].eq(parent.candidate_id)].iloc[0]
        child_summary = summaries.loc[summaries["model_id"].eq(child.candidate_id)].iloc[0]
        parent_cv = cv.loc[cv["model_id"].eq(parent.candidate_id)].sort_values("fold")
        child_cv = cv.loc[cv["model_id"].eq(child.candidate_id)].sort_values("fold")
        validation_delta = child_cv["validation_top1"].to_numpy(dtype=float) - parent_cv["validation_top1"].to_numpy(dtype=float)
        cv_stable = bool(int(np.count_nonzero(validation_delta < 0)) <= 1 and float(validation_delta.mean()) >= 0)
        fragile_fields = _build_fragile_edge_fields(frames[parent.candidate_id], frames[child.candidate_id])
        secondary_nonworse = bool(child_summary["nonself_pass_rate"] >= parent_summary["nonself_pass_rate"] and child_summary["share_margin_Q05"] >= parent_summary["share_margin_Q05"] and child_summary["MRR"] >= parent_summary["MRR"])
        strict_safe = bool(comparison["Top1"] == 421 and comparison["recovered_count"] == 0 and comparison["regressed_count"] == 0 and secondary_nonworse and cv_stable)
        equivalent_reordered = bool(comparison["Top1"] == 421 and comparison["recovered_count"] > 0 and comparison["regressed_count"] > 0)
        beneficial = bool(comparison["Top1"] > 421)
        necessary = bool(comparison["Top1"] < 421)
        if beneficial:
            deletion_class = "beneficial_to_delete"
        elif strict_safe:
            deletion_class = "strict_safe_compression"
        elif equivalent_reordered:
            deletion_class = "performance_preserving_but_reordered"
        elif necessary:
            deletion_class = "necessary_in_parent_context"
        else:
            deletion_class = "performance_preserving_but_not_safe"
        rows.append({"edge_id": int(edge.edge_id), "parent_frontier_id": parent.frontier_id, "parent_model_id": parent.candidate_id, "child_model_id": child.candidate_id, "child_model_name": child.candidate_name, "child_global_rank": int(child_summary["global_rank"]), "removed_basis_id": edge.removed_basis_id, "removed_p": edge.removed_p, "removed_m": edge.removed_m, "removed_q": edge.removed_q, "parent_top1": int(parent_summary["top1_pass_count"]), "child_top1": int(child_summary["top1_pass_count"]), "delta_top1": comparison["delta_top1"], "recovered_count": comparison["recovered_count"], "regressed_count": comparison["regressed_count"], "recovered_state_ids": json.dumps(comparison["recovered_state_ids"]), "regressed_state_ids": json.dumps(comparison["regressed_state_ids"]), "unchanged_pass": comparison["unchanged_pass"], "unchanged_fail": comparison["unchanged_fail"], "parent_exact": int(parent_summary["exact_hit_count"]), "child_exact": int(child_summary["exact_hit_count"]), "parent_nonself_rate": float(parent_summary["nonself_pass_rate"]), "child_nonself_rate": float(child_summary["nonself_pass_rate"]), "parent_Q05": float(parent_summary["share_margin_Q05"]), "child_Q05": float(child_summary["share_margin_Q05"]), "parent_MRR": float(parent_summary["MRR"]), "child_MRR": float(child_summary["MRR"]), "parent_Top2": int(parent_summary["Top2_oracle"]), "child_Top2": int(child_summary["Top2_oracle"]), "parent_Top3": int(parent_summary["Top3_oracle"]), "child_Top3": int(child_summary["Top3_oracle"]), "parent_Top5": int(parent_summary["Top5_oracle"]), "child_Top5": int(child_summary["Top5_oracle"]), "parent_Top10": int(parent_summary["Top10_oracle"]), "child_Top10": int(child_summary["Top10_oracle"]), "parent_fragile_count": int(fragile_fields["parent_fragile_count"]), "fragile_preserved": int(fragile_fields["fragile_preserved"]), "fragile_regressed": int(fragile_fields["fragile_regressed"]), "fragile_regressed_state_ids": fragile_fields["fragile_regressed_state_ids"], "cv_validation_delta_mean": float(validation_delta.mean()), "cv_validation_delta_min": float(validation_delta.min()), "cv_stable": cv_stable, "beneficial_to_delete": beneficial, "strict_safe_compression": strict_safe, "performance_preserving_but_reordered": equivalent_reordered, "necessary_in_parent_context": necessary, "deletion_class": deletion_class})
    return pd.DataFrame(rows).sort_values("edge_id").reset_index(drop=True)


def _build_parent_failure_outputs(candidates: tuple[FrontierCandidate | K12Candidate, ...], frames: dict[int, pd.DataFrame], edges: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    parents = candidates[:3]
    parent_frames = {candidate.frontier_id: frames[candidate.candidate_id].set_index("State_R") for candidate in parents}
    all_states = range(STATE_COUNT)
    overlap_rows: list[dict[str, Any]] = []
    for state_id in all_states:
        failed = {candidate.frontier_id: not bool(parent_frames[candidate.frontier_id].loc[state_id, "realB_pass"]) for candidate in parents}
        ranks = [int(parent_frames[candidate.frontier_id].loc[state_id, "first_shareable_rank"]) for candidate in parents]
        failed_count = int(sum(failed.values()))
        hard_type = "deeper_mismatch" if failed_count and max(ranks) >= 3 else "boundary_conflict" if failed_count else "not_parent_failure"
        overlap_rows.append({"State_R": state_id, "failed_in_frontier_A": failed[parents[0].frontier_id], "failed_in_frontier_B": failed[parents[1].frontier_id], "failed_in_frontier_C": failed[parents[2].frontier_id], "failure_count_across_3": failed_count, "persistent_all3": bool(failed_count == 3), "frontier_A_first_shareable_rank": ranks[0], "frontier_B_first_shareable_rank": ranks[1], "frontier_C_first_shareable_rank": ranks[2], "hard_state_type": hard_type})
    detailed_rows: list[dict[str, Any]] = []
    for edge in edges.itertuples(index=False):
        parent = next(candidate for candidate in parents if candidate.frontier_id == edge.parent_frontier_id)
        parent_frame = frames[parent.candidate_id].set_index("State_R")
        child_frame = frames[int(edge.child_model_id)].set_index("State_R")
        failure_ids = parent_frame.loc[~parent_frame["realB_pass"], :].index.astype(int).tolist()
        for state_id in failure_ids:
            parent_row = parent_frame.loc[state_id]
            child_row = child_frame.loc[state_id]
            detailed_rows.append({"edge_id": int(edge.edge_id), "parent_frontier_id": parent.frontier_id, "parent_model_id": parent.candidate_id, "child_model_id": int(edge.child_model_id), "removed_basis_id": edge.removed_basis_id, "State_R": state_id, "parent_State_Q": int(parent_row["State_Q"]), "child_State_Q": int(child_row["State_Q"]), "parent_first_shareable_rank": int(parent_row["first_shareable_rank"]), "child_first_shareable_rank": int(child_row["first_shareable_rank"]), "parent_margin": float(parent_row["shareability_margin_dB"]), "child_margin": float(child_row["shareability_margin_dB"]), "parent_realB": float(parent_row["retrieved_realB_CNMSE_dB"]), "child_realB": float(child_row["retrieved_realB_CNMSE_dB"]), "recovered": bool(not parent_row["realB_pass"] and child_row["realB_pass"]), "regressed": bool(parent_row["realB_pass"] and not child_row["realB_pass"]), "rank_improved": bool(child_row["first_shareable_rank"] < parent_row["first_shareable_rank"]), "rank_degraded": bool(child_row["first_shareable_rank"] > parent_row["first_shareable_rank"])})
    return pd.DataFrame(overlap_rows), pd.DataFrame(detailed_rows)


def _state330_trace(candidates: tuple[FrontierCandidate | K12Candidate, ...], frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        row = frames[candidate.candidate_id].loc[frames[candidate.candidate_id]["State_R"].eq(STATE330)].iloc[0]
        real_b = float(row["retrieved_realB_CNMSE_dB"])
        rows.append({"model_id": candidate.candidate_id, "model_name": candidate.candidate_name, "model_type": "K13_parent" if isinstance(candidate, FrontierCandidate) else "K12_child", "K": candidate.K, "State_R": STATE330, "State_Q": int(row["State_Q"]), "pass": bool(row["realB_pass"]), "retrieved_realB_CNMSE_dB": real_b, "headroom_dB": float(-40.0 - real_b), "first_shareable_rank": int(row["first_shareable_rank"]), "shareability_margin_dB": float(row["shareability_margin_dB"])})
    return pd.DataFrame(rows).sort_values("model_id").reset_index(drop=True)


def _seed_a_child_comparison(children: tuple[K12Candidate, ...], frames: dict[int, pd.DataFrame], references: dict[str, Any]) -> pd.DataFrame:
    seed_a = references["historical_seedA_query"].sort_values("State_R").reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    for child in children:
        comparison = _compare_frames(frames[child.candidate_id], seed_a)
        rows.append({"child_model_id": child.candidate_id, "child_model_name": child.candidate_name, "support_hash": child.support_hash, "Top1": comparison["Top1"], "delta_vs_SeedA": comparison["delta_top1"], "recovered_count": comparison["recovered_count"], "recovered_state_ids": json.dumps(comparison["recovered_state_ids"]), "regressed_count": comparison["regressed_count"], "regressed_state_ids": json.dumps(comparison["regressed_state_ids"]), "unchanged_pass": comparison["unchanged_pass"], "unchanged_fail": comparison["unchanged_fail"], "Exact": comparison["Exact"], "nonself_rate": comparison["nonself_pass"] / comparison["nonself_count"] if comparison["nonself_count"] else float("nan"), "Q05": comparison["Q05"], "MRR": comparison["MRR"], "Top2": comparison["Top2"], "Top3": comparison["Top3"], "Top5": comparison["Top5"], "Top10": comparison["Top10"]})
    return pd.DataFrame(rows).sort_values("child_model_id").reset_index(drop=True)


def _modeling_vs_retrieval(edges: pd.DataFrame, candidates: tuple[FrontierCandidate | K12Candidate, ...], model_frame: pd.DataFrame, summaries: pd.DataFrame) -> pd.DataFrame:
    medians = model_frame.groupby("model_id")[["Y_Aend_train_NMSE_dB", "Y_Aend_B_NMSE_dB", "Y_C2_train_NMSE_dB", "Y_C2_B_NMSE_dB"]].median()
    summary_by_id = summaries.set_index("model_id")
    parent_id_by_frontier = {candidate.frontier_id: candidate.candidate_id for candidate in candidates[:3]}
    rows: list[dict[str, Any]] = []
    for edge in edges.itertuples(index=False):
        child_id = int(edge.child_model_id)
        parent_id = parent_id_by_frontier[str(edge.parent_frontier_id)]
        rows.append({"edge_id": int(edge.edge_id), "parent_frontier_id": edge.parent_frontier_id, "child_model_id": child_id, "removed_basis_id": edge.removed_basis_id, "delta_Aend_train_median": float(medians.loc[child_id, "Y_Aend_train_NMSE_dB"] - medians.loc[parent_id, "Y_Aend_train_NMSE_dB"]), "delta_Aend_B_median": float(medians.loc[child_id, "Y_Aend_B_NMSE_dB"] - medians.loc[parent_id, "Y_Aend_B_NMSE_dB"]), "delta_C2_train_median": float(medians.loc[child_id, "Y_C2_train_NMSE_dB"] - medians.loc[parent_id, "Y_C2_train_NMSE_dB"]), "delta_C2_B_median": float(medians.loc[child_id, "Y_C2_B_NMSE_dB"] - medians.loc[parent_id, "Y_C2_B_NMSE_dB"]), "delta_Top1": int(summary_by_id.loc[child_id, "top1_pass_count"] - summary_by_id.loc[parent_id, "top1_pass_count"]), "delta_Q05": float(summary_by_id.loc[child_id, "share_margin_Q05"] - summary_by_id.loc[parent_id, "share_margin_Q05"]), "delta_MRR": float(summary_by_id.loc[child_id, "MRR"] - summary_by_id.loc[parent_id, "MRR"])})
    return pd.DataFrame(rows).sort_values("edge_id").reset_index(drop=True)


def _annotate_summary(candidates: tuple[FrontierCandidate | K12Candidate, ...], summaries: list[dict[str, Any]], frames: dict[int, pd.DataFrame], fragile: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(summaries).sort_values("model_id").reset_index(drop=True)
    summary["Top1"] = summary["top1_pass_count"]
    summary["Top1_rate"] = summary["top1_pass_rate"]
    summary["delta_vs_421"] = summary["top1_pass_count"] - 421
    summary["strict_breakthrough"] = summary["top1_pass_count"].gt(421)
    fragile_by_id = fragile.set_index("model_id")
    summary["fragile_success_count"] = summary["model_id"].map(fragile_by_id["fragile_success_count"])
    summary["fragile_fraction"] = summary["model_id"].map(fragile_by_id["fragile_fraction"])
    diagnostic_ranks = k9_utils._assign_diagnostic_ranks(summaries)["diagnostic_rank"].to_numpy(dtype=int)
    summary["global_rank"] = diagnostic_ranks
    summary["global_diagnostic_rank"] = summary["global_rank"]
    summary["model_type"] = summary["model_type"].astype(str)
    summary["is_K12_child"] = summary["model_type"].eq("K12_child")
    summary["distance_file"] = summary.apply(lambda row: "frontier_A.npy" if int(row.model_id) == 0 else "frontier_B.npy" if int(row.model_id) == 1 else "frontier_C.npy" if int(row.model_id) == 2 else f"child_{int(row.model_id):02d}.npy", axis=1)
    summary["parent_frontier_ids"] = summary["parent_frontier_ids"].fillna("[]")
    return summary


def _write_figures(summary: pd.DataFrame, edge_delta: pd.DataFrame, branch_summary: pd.DataFrame, overlap: pd.DataFrame, fragile: pd.DataFrame, model_frame: pd.DataFrame, cv_stability: pd.DataFrame, modeling: pd.DataFrame) -> None:
    ordered = summary.sort_values("global_rank").reset_index(drop=True)
    colors = ["tab:blue" if row.model_type == "K13_parent" else "tab:purple" for row in ordered.itertuples(index=False)]
    fig, ax = plt.subplots(figsize=(15, 8), dpi=300)
    ax.scatter(np.arange(ordered.shape[0]), ordered["Top1"], c=colors, s=30, alpha=0.8)
    ax.axhline(421, color="black", linestyle="--", linewidth=1.0, label="K13 parent benchmark = 421")
    top = ordered.head(3)
    for index, row in top.iterrows():
        ax.annotate(f"{int(row.model_id)}:{int(row.Top1)}", (index, row.Top1), rotation=90, ha="center", fontsize=7, xytext=(0, 4), textcoords="offset points")
    ax.set_xlabel("Global diagnostic rank")
    ax.set_ylabel("Top-1 Real-B pass count")
    ax.set_ylim(min(0, float(ordered["Top1"].min()) - 5), max(425, float(ordered["Top1"].max()) + 2))
    ax.set_title("Floating backward: global Top-1 Real-B comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "22_global_top1_comparison.png")
    plt.close(fig)

    edge_plot = edge_delta.sort_values(["parent_frontier_id", "delta_top1", "edge_id"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(18, 8), dpi=300)
    x = np.arange(edge_plot.shape[0])
    branch_colors = {"frontier_A_global_best": "tab:blue", "frontier_B_safe_best": "tab:orange", "frontier_C_seedA_branch_best": "tab:green"}
    ax.bar(x, edge_plot["delta_top1"], color=[branch_colors[value] for value in edge_plot["parent_frontier_id"]])
    ax.axhline(0, color="black", linewidth=0.9)
    ax.set_xticks(x, edge_plot["removed_basis_id"], rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("Top1(child) - 421")
    ax.set_xlabel("Deletion edge grouped by frontier")
    ax.set_title("Floating backward deletion contribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "23_backward_deletion_contribution.png")
    plt.close(fig)

    branch_labels = []
    branch_values = []
    for row in branch_summary.itertuples(index=False):
        branch_labels.extend([f"{row.frontier_id}\nK13", f"{row.frontier_id}\nK12 best"])
        branch_values.extend([int(row.parent_top1), int(row.best_child_top1)])
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    ax.bar(np.arange(len(branch_values)), branch_values, color=["#9ecae1", "#2171b5"] * 3)
    ax.axhline(421, color="black", linestyle="--", linewidth=1.0)
    ax.set_xticks(np.arange(len(branch_labels)), branch_labels, rotation=25, ha="right")
    ax.set_ylabel("Top-1 Real-B pass count")
    ax.set_title("K13 frontier versus best K12 deletion child")
    ax.set_ylim(0, 425)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "24_frontier_compression_comparison.png")
    plt.close(fig)

    persistent = overlap.loc[overlap["persistent_all3"]].copy()
    fig, ax = plt.subplots(figsize=(12, 7), dpi=300)
    if persistent.empty:
        ax.text(0.5, 0.5, "No persistent failure across all three frontiers", ha="center", va="center")
    else:
        for column, label, color in (("frontier_A_first_shareable_rank", "Frontier A", "tab:blue"), ("frontier_B_first_shareable_rank", "Frontier B", "tab:orange"), ("frontier_C_first_shareable_rank", "Frontier C", "tab:green")):
            ax.plot(persistent["State_R"], persistent[column], marker="o", label=label, color=color)
        ax.set_xticks(persistent["State_R"].astype(int), persistent["State_R"].astype(int))
        ax.legend(frameon=False)
    ax.set_xlabel("Persistent failure State_R")
    ax.set_ylabel("First-shareable rank")
    ax.set_title("Persistent hard-state rank across the three frontiers")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "25_persistent_hard_state_rank_change.png")
    plt.close(fig)

    fragile_edge = edge_delta.sort_values(["parent_frontier_id", "edge_id"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(18, 8), dpi=300)
    x = np.arange(fragile_edge.shape[0])
    ax.bar(x - 0.2, fragile_edge["fragile_preserved"], width=0.4, label="fragile preserved", color="tab:green")
    ax.bar(x + 0.2, fragile_edge["fragile_regressed"], width=0.4, label="fragile regressed", color="tab:red")
    ax.set_xticks(x, fragile_edge["removed_basis_id"], rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("State count")
    ax.set_title("Fragile-success preservation under floating deletion")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "26_fragile_success_preservation.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    ax.scatter(modeling["delta_C2_train_median"], modeling["delta_Top1"], c=modeling["delta_Top1"], cmap="coolwarm", s=45, alpha=0.85)
    for row in modeling.loc[modeling["delta_Top1"].abs().nlargest(10).index].itertuples(index=False):
        ax.annotate(str(row.removed_basis_id), (row.delta_C2_train_median, row.delta_Top1), fontsize=6, xytext=(3, 3), textcoords="offset points")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Deletion Δ C2 Train median NMSE (dB)")
    ax.set_ylabel("Deletion Δ Top1 vs parent")
    ax.set_title("Floating deletion: modeling versus retrieval contribution")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "27_modeling_vs_retrieval_deletion.png")
    plt.close(fig)

    frequency = cv_stability.sort_values(["global_train_winner_count", "validation_top1_mean"], ascending=[False, False]).head(20)
    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
    bars = ax.bar(np.arange(frequency.shape[0]), frequency["global_train_winner_count"], color="tab:blue")
    ax.set_xticks(np.arange(frequency.shape[0]), frequency["model_id"].astype(str), rotation=70, ha="right")
    for bar, value in zip(bars, frequency["global_train_winner_count"], strict=True):
        if value:
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.03, str(int(value)), ha="center", fontsize=7)
    ax.set_xlabel("Model ID (top 20 by global train-winner frequency)")
    ax.set_ylabel("Global train-winner count across 5 folds")
    ax.set_title("Floating backward Query-State CV selection frequency")
    ax.set_ylim(0, max(1, int(frequency["global_train_winner_count"].max()) + 1))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "28_cv_selection_frequency.png")
    plt.close(fig)


def _branch_summary(candidates: tuple[FrontierCandidate, ...], children: tuple[K12Candidate, ...], summary: pd.DataFrame, edge_delta: pd.DataFrame, cv_stability: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for frontier in candidates:
        edge_subset = edge_delta.loc[edge_delta["parent_frontier_id"].eq(frontier.frontier_id)]
        child_summary = summary.loc[summary["model_id"].isin(edge_subset["child_model_id"].astype(int))].sort_values("global_rank")
        best = child_summary.iloc[0]
        best_edge = edge_subset.loc[edge_subset["child_model_id"].eq(int(best["model_id"]))].sort_values("edge_id").iloc[0]
        best_cv = cv_stability.loc[cv_stability["model_id"].eq(int(best["model_id"]))].iloc[0]
        rows.append({"frontier_id": frontier.frontier_id, "parent_model_id": frontier.candidate_id, "parent_formal_model_id": frontier.formal_model_id, "parent_support_hash": frontier.support_hash, "parent_top1": int(summary.loc[summary["model_id"].eq(frontier.candidate_id), "Top1"].iloc[0]), "best_child_model_id": int(best["model_id"]), "best_child_model_name": best["model_name"], "best_child_removed_basis": best_edge["removed_basis_id"], "best_child_support_hash": best["support_hash"], "best_child_top1": int(best["Top1"]), "best_child_global_rank": int(best["global_rank"]), "delta_top1": int(best_edge["delta_top1"]), "recovered_count": int(best_edge["recovered_count"]), "recovered_state_ids": best_edge["recovered_state_ids"], "regressed_count": int(best_edge["regressed_count"]), "regressed_state_ids": best_edge["regressed_state_ids"], "K": int(best["K"]), "Q05": float(best["share_margin_Q05"]), "MRR": float(best["MRR"]), "Top2": int(best["Top2_oracle"]), "Top3": int(best["Top3_oracle"]), "validation_top1_mean": float(best_cv["validation_top1_mean"]), "validation_top1_min": int(best_cv["validation_top1_min"]), "validation_top1_max": int(best_cv["validation_top1_max"])})
    return pd.DataFrame(rows)


def _save_fingerprint(path_stem: str, lut: np.ndarray, query: np.ndarray) -> None:
    np.save(RESULT_ROOT / "fingerprints" / f"{path_stem}_lut.npy", np.asarray(lut, dtype=np.complex128))
    np.save(RESULT_ROOT / "fingerprints" / f"{path_stem}_query.npy", np.asarray(query, dtype=np.complex128))


def _run(*, resume: bool = False) -> dict[str, Any]:
    if RESULT_ROOT.exists() and any(RESULT_ROOT.iterdir()) and not resume:
        raise RuntimeError(f"result directory is not empty; use --resume: {RESULT_ROOT}")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "distance_matrices").mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "fingerprints").mkdir(parents=True, exist_ok=True)
    raw_before = raw_manifest_gate()
    terms = tuple(build_envelope_dictionary())
    envelope_gate = dictionary_gate()
    core_ids, by_id, core_index, formal_core_frame = _load_core(terms)
    references = _load_references(core_ids)
    common_b, common_meta = verify_common_b_contract()
    common_b = np.asarray(common_b, dtype=np.complex128)
    if str(common_meta["sha256"]) != EXPECTED_COMMON_B_SHA:
        raise RuntimeError("common-B hash changed")
    common_mp = backend.historical_mp10.build_frozen_mp_basis(common_b)
    common_env = build_envelope_bank(common_b, terms)
    common_core = np.column_stack((common_mp, common_env[:, core_index])).astype(np.complex128)
    references["common_core"] = common_core
    references["common_env"] = common_env
    real_b_random_error = _random_real_b_check(references["real_b"])
    frontier_frame, edges, frontiers, children, duplicate_frame = _build_candidates(core_ids, by_id, references)
    if len(frontiers) != 3 or len(children) != 37 or edges.shape[0] != 39:
        raise RuntimeError("floating support graph cardinality changed")
    frontier_frame.to_csv(RESULT_ROOT / "02_frontier_supports.csv", index=False)
    edges.to_csv(RESULT_ROOT / "04_raw_deletion_edges.csv", index=False)
    unique_rows = []
    for child in children:
        unique_rows.append({"child_model_id": child.candidate_id, "model_name": child.candidate_name, "support_hash": child.support_hash, "K": child.K, "basis_ids": json.dumps(list(child.support_basis_ids), ensure_ascii=False), "parent_frontier_ids": json.dumps(list(child.parent_frontier_ids), ensure_ascii=False), "generation_count": len(child.parent_frontier_ids), "removed_basis_by_parent": json.dumps(list(child.removed_basis_by_parent), ensure_ascii=False), "is_historical_reuse": child.is_historical_reuse, "historical_source_task": child.historical_source_task or "", "historical_source_model_name": child.historical_source_model_name or "", "historical_source_model_id": child.historical_source_model_id if child.historical_source_model_id is not None else np.nan, "removed_core_basis_id": child.removed_core_basis_id or "", "removed_core_index": child.removed_core_index if child.removed_core_index is not None else np.nan, "extension_basis_1": child.extension_basis_ids[0] if child.extension_basis_ids else "NONE", "extension_basis_2": child.extension_basis_ids[1] if len(child.extension_basis_ids) > 1 else "NONE", "support_indices": json.dumps(list(range(K12)))})
    unique_frame = pd.DataFrame(unique_rows).sort_values("child_model_id").reset_index(drop=True)
    unique_frame.to_csv(RESULT_ROOT / "05_unique_k12_children.csv", index=False)
    dedup_payload = {"raw_deletion_edges": int(edges.shape[0]), "unique_k12_children": int(unique_frame.shape[0]), "duplicate_occurrences": int(edges["is_duplicate_child"].sum()), "multi_parent_children": int((unique_frame["generation_count"] > 1).sum()), "historical_reuse_count": int(unique_frame["is_historical_reuse"].sum()), "novel_k12_fit_count": int((~unique_frame["is_historical_reuse"]).sum()), "multi_parent_supports": duplicate_frame.to_dict("records"), "pass": bool(edges.shape[0] == 39 and unique_frame.shape[0] == 37 and int(edges["is_duplicate_child"].sum()) == 2 and int((unique_frame["generation_count"] > 1).sum()) == 2)}
    _write_json(RESULT_ROOT / "06_support_dedup_audit.json", dedup_payload)
    (RESULT_ROOT / "00_task_definition.txt").write_text("\n".join([f"Task: {TASK_NAME}", "Mode: unified multi-frontier floating backward retrieval ablation.", "Frontiers: Model 74, Model 125, Model 12; each K13 frontier deletes one basis at a time.", "Raw deletion edges=39; canonical support dedup produces 37 unique K12 children; 4 historical children are reused and 33 novel children are fitted.", "Evaluation pool=3 K13 parents + 37 K12 children=40 unique supports.", f"Frozen protocol: dmax={backend.MP_DMAX}, Ridge lambda={backend.RIDGE_LAMBDA}, ABC valid={backend.ABC_LENGTHS}, common-B SHA={EXPECTED_COMMON_B_SHA}, Real-B threshold={REAL_B_THRESHOLD_DB} dB, allow-self=True.", "No second backward, K14, second beam add, swap, multi-basis deletion/addition, parameter scan, reranking, ensemble, clustering, low-bandwidth, DPD replay or nested CV final evaluation."]) + "\n", encoding="utf-8")
    _write_json(RESULT_ROOT / "01_reuse_audit.json", {"task_name": TASK_NAME, "formal_core_source": str(K11_ROOT / "04_candidate_supports.csv"), "core_basis_ids": list(core_ids), "core_support_hash": _support_hash(core_ids), "frontier_source": str(BEAM_ROOT / "12_retrieval_summary.csv"), "historical_k12_source": str(K12_ROOT / "10_candidate_retrieval_summary.csv"), "envelope75_gate": envelope_gate, "real_B_source": str(K9_ROOT / "04_realB_ground_truth_cnmse_matrix.npy"), "cv_source": str(K12_ROOT / "16_query_state_cv_folds.csv"), "common_B_sha256": EXPECTED_COMMON_B_SHA, "real_B_random_pair_max_abs_error_dB": real_b_random_error, "raw_deletion_edges": 39, "unique_k12_children": 37, "historical_reuse": 4, "novel_fit": 33, "no_new_hnorm": True})
    (RESULT_ROOT / "01_reuse_audit.txt").write_text("\n".join([f"Task: {TASK_NAME}", f"Frozen core={list(core_ids)}; support_hash={_support_hash(core_ids)}.", "Frontier A/B/C are pre-frozen as Model 74/125/12 from the formal Beam=3 result.", "All parent and four historical K12 children are read/verified/reused; only 33 novel K12 supports enter Ridge.", "Support dedup uses canonical support IDs and SHA256 before fitting.", f"common-B SHA={EXPECTED_COMMON_B_SHA}; Real-B random max error={real_b_random_error:.3e} dB.", "No second backward, K14, second beam, swap, parameter scan, reranking, ensemble, clustering, low-bandwidth or DPD replay was run."]) + "\n", encoding="utf-8")
    _checkpoint(phase="reuse_audit_graph_passed", raw_manifest_before=raw_before, raw_deletion_edges=39, unique_k12_children=37, historical_reuse_count=4, novel_k12_fit_count=33, evaluation_pool_count=40, multi_parent_child_count=2, real_B_random_pair_max_abs_error_dB=real_b_random_error)

    all_candidates: tuple[FrontierCandidate | K12Candidate, ...] = tuple(frontiers) + tuple(children)
    frames: dict[int, pd.DataFrame] = {}
    model_rows: dict[int, pd.DataFrame] = {}
    distances: dict[int, np.ndarray] = {}
    theta_a_by_id: dict[int, np.ndarray] = {}
    theta_c_by_id: dict[int, np.ndarray] = {}
    fingerprints: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    parent_regressions: list[dict[str, Any]] = []
    for frontier in frontiers:
        model, query, distance, theta_a, theta_c = _load_parent_payload(frontier, references)
        phi = _candidate_phi_for_common(common_core, common_env, frontier)
        lut = (phi @ theta_a.T).T.astype(np.complex128)
        query_fp = (phi @ theta_c.T).T.astype(np.complex128)
        frames[frontier.candidate_id] = query
        model_rows[frontier.candidate_id] = model
        distances[frontier.candidate_id] = distance
        theta_a_by_id[frontier.candidate_id] = theta_a
        theta_c_by_id[frontier.candidate_id] = theta_c
        fingerprints[frontier.candidate_id] = (lut, query_fp)
        np.save(RESULT_ROOT / "distance_matrices" / f"frontier_{'A' if frontier.candidate_id == 0 else 'B' if frontier.candidate_id == 1 else 'C'}.npy", distance)
        _save_fingerprint(f"frontier_{'A' if frontier.candidate_id == 0 else 'B' if frontier.candidate_id == 1 else 'C'}", lut, query_fp)
        parent_regressions.append(_seed_or_historical_regression(frontier, model, query, distance, theta_a, theta_c, lut, query_fp, references, "parent"))
    if not all(item["pass"] for item in parent_regressions):
        raise RuntimeError("HARD FAIL: frontier regression failed")
    _write_json(RESULT_ROOT / "03_frontier_regression.json", {"pass": True, "frontier_regressions": parent_regressions})
    _checkpoint(phase="frontier_regression_passed", frontier_regressions=parent_regressions)

    historical_regressions: list[dict[str, Any]] = []
    for child in children:
        if not child.is_historical_reuse:
            continue
        model, query, distance, theta_a, theta_c = _load_historical_child_payload(child, references)
        phi = _candidate_phi_for_common(common_core, common_env, child)
        lut = (phi @ theta_a.T).T.astype(np.complex128)
        query_fp = (phi @ theta_c.T).T.astype(np.complex128)
        frames[child.candidate_id] = query
        model_rows[child.candidate_id] = model
        distances[child.candidate_id] = distance
        theta_a_by_id[child.candidate_id] = theta_a
        theta_c_by_id[child.candidate_id] = theta_c
        fingerprints[child.candidate_id] = (lut, query_fp)
        np.save(RESULT_ROOT / "distance_matrices" / f"child_{child.candidate_id:02d}.npy", distance)
        historical_regressions.append(_seed_or_historical_regression(child, model, query, distance, theta_a, theta_c, lut, query_fp, references, "historical_child"))
    if len(historical_regressions) != 4 or not all(item["pass"] for item in historical_regressions):
        raise RuntimeError("HARD FAIL: historical K12 child reuse regression failed")
    _write_json(RESULT_ROOT / "07_historical_child_reuse_audit.json", {"pass": True, "historical_child_count": len(historical_regressions), "historical_child_regressions": historical_regressions})
    _checkpoint(phase="historical_child_reuse_passed", historical_child_count=4)

    novel_children = tuple(child for child in children if not child.is_historical_reuse)
    novel_model_frame, novel_theta_a, novel_theta_c = _run_novel_model_phase(novel_children, core_index, resume)
    for child in novel_children:
        model_rows[child.candidate_id] = _decorate_model_rows(novel_model_frame.loc[novel_model_frame["model_id"].eq(child.candidate_id)], child)
    local_novel_index = {child.candidate_id: index for index, child in enumerate(novel_children)}
    theta_bank_a = np.zeros((40, STATE_COUNT, K12), dtype=np.complex128)
    theta_bank_c = np.zeros_like(theta_bank_a)
    for child in novel_children:
        theta_bank_a[child.candidate_id] = novel_theta_a[local_novel_index[child.candidate_id]]
        theta_bank_c[child.candidate_id] = novel_theta_c[local_novel_index[child.candidate_id]]
    for child in novel_children:
        phi = _candidate_phi_for_common(common_core, common_env, child)
        query, summary, distance, lut, query_fp = _novel_retrieval(child, theta_bank_a, theta_bank_c, phi, references["real_b"], references["shareability"])
        frames[child.candidate_id] = query
        distances[child.candidate_id] = distance
        theta_a_by_id[child.candidate_id] = novel_theta_a[local_novel_index[child.candidate_id]]
        theta_c_by_id[child.candidate_id] = novel_theta_c[local_novel_index[child.candidate_id]]
        fingerprints[child.candidate_id] = (lut, query_fp)
        np.save(RESULT_ROOT / "distance_matrices" / f"child_{child.candidate_id:02d}.npy", distance)
    _checkpoint(phase="novel_k12_fit_and_retrieval_completed", novel_child_count=len(novel_children))

    unified_model = pd.concat(model_rows.values(), ignore_index=True).sort_values(["model_id", "State_R"]).reset_index(drop=True)
    if unified_model.shape[0] != 40 * STATE_COUNT:
        raise RuntimeError(f"unified model quality shape failed: {unified_model.shape}")
    unified_model.to_csv(RESULT_ROOT / "08_model_quality_long.csv", index=False)
    all_summaries = [_summary_from_frame(frames[candidate.candidate_id], candidate) for candidate in all_candidates]
    fragile = pd.DataFrame([_fragile_summary(candidate, frames[candidate.candidate_id]) for candidate in all_candidates]).sort_values("model_id").reset_index(drop=True)
    fragile.to_csv(RESULT_ROOT / "16_fragile_success_summary.csv", index=False)
    historical_seed_a = references["historical_seedA_query"].sort_values("State_R").reset_index(drop=True).copy()
    historical_seed_a["headroom_dB"] = -40.0 - historical_seed_a["retrieved_realB_CNMSE_dB"].to_numpy(dtype=float)
    historical_seed_a["fragile_success"] = historical_seed_a["realB_pass"].to_numpy(dtype=bool) & (historical_seed_a["headroom_dB"].to_numpy(dtype=float) <= 1.0)
    historical_seed_a.loc[~historical_seed_a["realB_pass"], "headroom_dB"] = np.nan
    historical_seed_a.loc[~historical_seed_a["fragile_success"], "fragile_success"] = False
    historical_seed_a[["State_R", "State_Q", "retrieved_realB_CNMSE_dB", "headroom_dB", "first_shareable_rank", "shareability_margin_dB", "fragile_success"]].to_csv(RESULT_ROOT / "16_seedA_historical_fragile_success.csv", index=False)
    summary = _annotate_summary(all_candidates, all_summaries, frames, fragile)
    query_frame = pd.concat(frames.values(), ignore_index=True).sort_values(["model_id", "State_R"]).reset_index(drop=True)
    if query_frame.shape[0] != 40 * STATE_COUNT:
        raise RuntimeError(f"unified query metric shape failed: {query_frame.shape}")
    query_frame.to_csv(RESULT_ROOT / "10_query_metrics_long.csv", index=False)
    cv, cv_stability, branch_cv = _local_cv(all_candidates, frames, references["folds"])
    cv.to_csv(RESULT_ROOT / "18_query_state_cv_results.csv", index=False)
    cv_stability.to_csv(RESULT_ROOT / "19_query_state_cv_selection_stability.csv", index=False)
    branch_cv.to_csv(RESULT_ROOT / "20_frontier_branch_cv_summary.csv", index=False)
    edge_delta = _build_edge_delta(edges, all_candidates, frames, summary, cv, fragile)
    edge_delta.to_csv(RESULT_ROOT / "12_deletion_edge_delta.csv", index=False)
    overlap, parent_failure_detail = _build_parent_failure_outputs(all_candidates, frames, edges)
    overlap.to_csv(RESULT_ROOT / "13_parent_failure_overlap.csv", index=False)
    parent_failure_detail.to_csv(RESULT_ROOT / "14_parent_failure_detailed_analysis.csv", index=False)
    state330 = _state330_trace(all_candidates, frames)
    state330.to_csv(RESULT_ROOT / "15_state330_fragility_trace.csv", index=False)
    seed_a_child = _seed_a_child_comparison(children, frames, references)
    seed_a_child.to_csv(RESULT_ROOT / "17_recovered_regressed_vs_seedA.csv", index=False)
    modeling = _modeling_vs_retrieval(edges, all_candidates, unified_model, summary)
    modeling.to_csv(RESULT_ROOT / "21_modeling_vs_retrieval_deletion.csv", index=False)
    branch_summary = _branch_summary(frontiers, children, summary, edge_delta, cv_stability)
    branch_summary.to_csv(RESULT_ROOT / "23_branch_summary.csv", index=False)
    # Add edge-context classification counts to the support-level summary.
    for field in ("beneficial_to_delete", "strict_safe_compression", "performance_preserving_but_reordered", "necessary_in_parent_context"):
        counts = edge_delta.groupby("child_model_id")[field].sum()
        summary[f"{field}_edge_count"] = summary["model_id"].map(counts).fillna(0).astype(int)
    summary.to_csv(RESULT_ROOT / "11_retrieval_summary.csv", index=False)
    global_best = summary.sort_values("global_rank").iloc[0]
    k12_summary = summary.loc[summary["model_type"].eq("K12_child")].sort_values("global_rank")
    best_k12 = k12_summary.iloc[0]
    # Save selected diagnostic fingerprints only.
    selected_ids = {"best_k12_child": int(best_k12["model_id"]), "global_best": int(global_best["model_id"])}
    for name, candidate_id in selected_ids.items():
        lut, query_fp = fingerprints[candidate_id]
        _save_fingerprint(name, lut, query_fp)
    for row in branch_summary.itertuples(index=False):
        lut, query_fp = fingerprints[int(row.best_child_model_id)]
        short = "A" if row.frontier_id == frontiers[0].frontier_id else "B" if row.frontier_id == frontiers[1].frontier_id else "C"
        _save_fingerprint(f"best_child_frontier_{short}", lut, query_fp)
    frames[int(global_best["model_id"])].to_csv(RESULT_ROOT / "29_best_child_detail.csv", index=False)
    _write_figures(summary, edge_delta, branch_summary, overlap, fragile, unified_model, cv_stability, modeling)
    raw_after = raw_manifest_gate()
    if raw_before != raw_after:
        raise RuntimeError("data/raw manifest changed")
    beneficial_count = int(edge_delta["beneficial_to_delete"].sum())
    strict_safe_count = int(edge_delta["strict_safe_compression"].sum())
    reordered_count = int(edge_delta["performance_preserving_but_reordered"].sum())
    necessary_count = int(edge_delta["necessary_in_parent_context"].sum())
    k12_421_count = int(k12_summary["Top1"].eq(421).sum())
    top20 = k12_summary.head(20)
    persistent = overlap.loc[overlap["persistent_all3"], "State_R"].astype(int).tolist()
    global_winners = cv.loc[cv["train_is_global_winner"]].sort_values("fold")
    summary_lines = [
        f"Task: {TASK_NAME}",
        "3 K13 frontier unified floating backward elimination completed.",
        "Frontier parents=3; raw deletion edges=39; unique K12 children=37; historical reuse=4; novel K12 fits=33; evaluation pool=40.",
        f"Frontier regression PASS={all(item['pass'] for item in parent_regressions)}; historical child reuse regression PASS={all(item['pass'] for item in historical_regressions)}.",
        f"Beneficial deletion edges (child Top1>421)={beneficial_count}; strict safe compression edges={strict_safe_count}; reordered edges={reordered_count}; necessary-in-parent-context edges={necessary_count}.",
        f"Unique K12 children with Top1=421: {k12_421_count}; best global support={global_best['model_name']} Top1={int(global_best['Top1'])}/425; best K12 child={best_k12['model_name']} Top1={int(best_k12['Top1'])}/425.",
        f"Persistent failure intersection across three K13 parents: {persistent}.",
        "",
        "Three frontier parents:",
    ]
    for candidate in frontiers:
        row = summary.loc[summary["model_id"].eq(candidate.candidate_id)].iloc[0]
        failure_ids = frames[candidate.candidate_id].loc[~frames[candidate.candidate_id]["realB_pass"], "State_R"].astype(int).tolist()
        summary_lines.append(f"{candidate.frontier_id}: support={list(candidate.support_basis_ids)}, Top1={int(row['Top1'])}, Exact={int(row['exact_hit_count'])}, nonself={int(row['nonself_pass_count'])}/{int(row['nonself_count'])}, Q05={float(row['share_margin_Q05']):.9g}, MRR={float(row['MRR']):.9g}, Top2={int(row['Top2_oracle'])}, Top3={int(row['Top3_oracle'])}, failures={failure_ids}")
    summary_lines.extend(["", "All unique K12 children by global rank (first 20):"])
    for row in top20.itertuples(index=False):
        summary_lines.append(f"rank={int(row.global_rank)} id={int(row.model_id)} {row.model_name}: parent={row.parent_frontier_ids}, Top1={int(row.Top1)}, delta421={int(row.delta_vs_421)}, Exact={int(row.exact_hit_count)}, nonself={float(row.nonself_pass_rate):.9g}, Q05={float(row.share_margin_Q05):.9g}, MRR={float(row.MRR):.9g}, Top2={int(row.Top2_oracle)}, Top3={int(row.Top3_oracle)}, fragile={int(row.fragile_success_count)}")
    summary_lines.extend(["", "Best deletion child per frontier:"])
    for row in branch_summary.itertuples(index=False):
        summary_lines.append(f"{row.frontier_id}: removed={row.best_child_removed_basis}, child={row.best_child_model_name}, 421->{int(row.best_child_top1)}, recovered={int(row.recovered_count)} {row.recovered_state_ids}, regressed={int(row.regressed_count)} {row.regressed_state_ids}, Q05={float(row.Q05):.9g}, MRR={float(row.MRR):.9g}, validation Top1 mean/min/max={float(row.validation_top1_mean):.6g}/{int(row.validation_top1_min)}/{int(row.validation_top1_max)}")
    summary_lines.extend(["", "Persistent hard states:"])
    for row in overlap.loc[overlap["persistent_all3"]].itertuples(index=False):
        summary_lines.append(f"State {int(row.State_R)}: ranks A/B/C={int(row.frontier_A_first_shareable_rank)}/{int(row.frontier_B_first_shareable_rank)}/{int(row.frontier_C_first_shareable_rank)}, type={row.hard_state_type}")
    summary_lines.extend(["", "State330 trace:"])
    for row in state330.itertuples(index=False):
        pass_value = bool(state330.loc[state330["model_id"].eq(int(row.model_id)), "pass"].iloc[0])
        summary_lines.append(f"{row.model_name}: State_Q={int(row.State_Q)}, pass={pass_value}, Real-B={float(row.retrieved_realB_CNMSE_dB):.9g}, headroom={float(row.headroom_dB):.9g}, rank={int(row.first_shareable_rank)}")
    summary_lines.extend(["", "Fragile-success counts:"])
    for row in fragile.itertuples(index=False):
        summary_lines.append(f"{row.model_name}: success={int(row.success_count)}, fragile={int(row.fragile_success_count)}, min_headroom={float(row.minimum_headroom_dB):.9g}, Q05_headroom={float(row.Q05_headroom_dB):.9g}")
    summary_lines.extend(["", "Global CV winners:"])
    for row in global_winners.itertuples(index=False):
        summary_lines.append(f"fold={int(row.fold)} winner={row.model_name}, validation Top1={int(row.validation_top1)}, nonself={float(row.validation_nonself_rate):.9g}, Q05={float(row.validation_Q05):.9g}, MRR={float(row.validation_MRR):.9g}")
    summary_lines.extend(["", f"raw before={json.dumps(raw_before, ensure_ascii=False, sort_keys=True)}", f"raw after={json.dumps(raw_after, ensure_ascii=False, sort_keys=True)}", f"raw unchanged={raw_before == raw_after}", "No second backward, K14, second beam add, swap, multi-basis deletion/addition, parameter scan, reranking, ensemble, clustering, low-bandwidth or DPD replay was run.", "CV is Query-State support-selection stability analysis, not an independent final test."])
    (RESULT_ROOT / "30_final_result_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    result = {"task": TASK_NAME, "status": "SUCCESS", "frontier_regression_pass": bool(all(item["pass"] for item in parent_regressions)), "historical_reuse_pass": bool(all(item["pass"] for item in historical_regressions)), "raw_deletion_edges": int(edges.shape[0]), "unique_k12_children": int(unique_frame.shape[0]), "historical_reuse_count": int(unique_frame["is_historical_reuse"].sum()), "novel_k12_fit_count": int((~unique_frame["is_historical_reuse"]).sum()), "evaluation_pool": int(len(all_candidates)), "beneficial_deletion_count": beneficial_count, "strict_safe_compression_count": strict_safe_count, "performance_preserving_reordered_count": reordered_count, "necessary_in_parent_context_count": necessary_count, "k12_top1_421_count": k12_421_count, "global_best_model": str(global_best["model_name"]), "global_best_top1": int(global_best["Top1"]), "best_k12_child": str(best_k12["model_name"]), "best_k12_top1": int(best_k12["Top1"]), "persistent_failure_intersection": persistent, "cv_global_winners": global_winners[["fold", "model_id", "model_name"]].to_dict("records"), "raw_data_modified": raw_before != raw_after, "result_root": str(RESULT_ROOT)}
    _write_json(RESULT_ROOT / "31_checkpoint.json", {"phase": "completed", **result, "independent_validation_pending": True})
    _append_log(f"\n[{_now()}] Complete {TASK_NAME}\nResult={json.dumps(result, ensure_ascii=False, sort_keys=True)}\nNo second backward, K14, second beam or swap was run.\n")
    _append_handoff(f"完成三条 K13 frontier 的统一 Floating Backward Elimination；39 raw edges 去重为 37 unique K12 children，4 个历史复用、33 个 novel fit，evaluation pool=40。结果：global best K12={best_k12['model_name']} Top1={int(best_k12['Top1'])}/425，beneficial={beneficial_count}，strict safe compression={strict_safe_count}，reordered={reordered_count}；未启动 K14、第二次 backward、第二层 beam 或 swap。结果目录：{RESULT_ROOT}")
    return result


def main() -> None:
    result = _run(resume="--resume" in sys.argv[1:])
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    __import__("multiprocessing").freeze_support()
    main()
