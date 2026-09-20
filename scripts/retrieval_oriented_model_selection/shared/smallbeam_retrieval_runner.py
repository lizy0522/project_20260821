# ruff: noqa: E402,E501,I001

"""Run the first Beam=3 one-layer retrieval-oriented K13 expansion."""

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
from retrieval_oriented_model_selection.shared import mp10_k9_retrieval_runner as k9_utils
from retrieval_oriented_model_selection.shared import smallbeam_retrieval_support as backend
from signal_segmentation.shared import build_off_segments, build_partition_from_xin  # noqa: E402

TASK_NAME = "scenario_2_smallbeam3_k13_retrieval_oriented_forward_expansion_5b"
RESULT_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /TASK_NAME
WORK_LOG = PROJECT_ROOT / "work_logs" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /TASK_NAME / "execution_log.txt"
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"
K11_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /"scenario_2_mp10_k11_retrieval_oriented_forward_scan_5b"
K12_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /"scenario_2_k12_retrieval_oriented_forward_scan_5b"
K9_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /"scenario_2_mp10_k9_retrieval_oriented_basis_ablation_5b"
CORE_MODEL_NAME = "MP10_plus_ENV_p04_m2_q0"
SEED_SPECS = (
    (0, "seed_A_global_champion", "K11_plus_ENV_p04_m0_q1", "ENV_p04_m0_q1"),
    (1, "seed_B_margin_branch", "K11_plus_ENV_p03_m0_q1", "ENV_p03_m0_q1"),
    (2, "seed_C_structural_branch", "K11_plus_ENV_p09_m2_q0", "ENV_p09_m2_q0"),
)
SEED_A_ID = 0
SEED_B_ID = 1
SEED_C_ID = 2
CORE_K = backend.CORE_K
SEED_K = backend.SEED_K
CHILD_K = backend.CHILD_K
STATE_COUNT = backend.STATE_COUNT
WORKER_COUNT = 10
BLAS_THREADS_PER_WORKER = 1
REAL_B_THRESHOLD_DB = -40.0
REAL_B_SANITY_TOLERANCE_DB = 1e-9
REPRESENTATIVE_STATES = (0, 42, 187, 325, 424)
CHAMPION_FAILURES = [189, 195, 323, 335, 340, 346]


@dataclass(frozen=True)
class BeamCandidate:
    candidate_id: int
    candidate_name: str
    model_type: str
    branch_seed_ids: tuple[str, ...]
    primary_parent_seed_id: str | None
    K: int
    extension_basis_ids: tuple[str, ...]
    extension_indices: tuple[int, ...]
    support_basis_ids: tuple[str, ...]
    support_hash: str

    @property
    def candidate_name_for_retrieval(self) -> str:
        return self.candidate_name

    @property
    def support(self) -> tuple[int, ...]:
        return tuple(range(self.K))

    @property
    def removed_basis_id(self) -> str | None:
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
        RESULT_ROOT / "32_checkpoint.json",
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
            "beam_width": 3,
            "raw_expansion_edge_count": 189,
            "unique_k13_child_count": 186,
            "evaluation_pool_count": 189,
            "core_K": CORE_K,
            "seed_K": SEED_K,
            "child_K": CHILD_K,
            "dmax_fixed": backend.MP_DMAX,
            "ridge_lambda_fixed": backend.RIDGE_LAMBDA,
            "second_beam_layer_performed": False,
            "k14_performed": False,
            "backward_elimination_performed": False,
            "swap_performed": False,
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


def _load_frozen_core(terms: tuple[EnvelopeBasis, ...]) -> tuple[tuple[str, ...], dict[str, EnvelopeBasis], int, pd.DataFrame]:
    path = K11_ROOT / "04_candidate_supports.csv"
    frame = pd.read_csv(path)
    row = frame.loc[frame["model_name"].eq(CORE_MODEL_NAME)]
    if row.shape[0] != 1 or int(row.iloc[0]["K"]) != CORE_K:
        raise RuntimeError("formal K11 core support is missing or invalid")
    core_ids = tuple(json.loads(row.iloc[0]["support_basis_ids"]))
    by_id = {term.basis_id: term for term in terms}
    if len(core_ids) != CORE_K or core_ids[-1] != backend.CORE_BASIS_ID or any(basis_id not in by_id for basis_id in core_ids):
        raise RuntimeError(f"K11 core support changed: {core_ids}")
    if str(row.iloc[0]["support_hash"]) != _support_hash(core_ids):
        raise RuntimeError("K11 core support hash does not match canonical IDs")
    return core_ids, by_id, int(by_id[backend.CORE_BASIS_ID].index), frame


def _load_references(core_ids: tuple[str, ...]) -> dict[str, Any]:
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
    with np.load(K11_ROOT / "07_candidate_coefficients.npz", allow_pickle=False) as data:
        k11_ids = np.asarray(data["model_ids"], dtype=np.int64)
        k11_theta_a = np.asarray(data["theta_Aend_padded"])
        k11_theta_c = np.asarray(data["theta_C2_padded"])
    k9_folds = pd.read_csv(K9_ROOT / "12_query_state_cv_folds.csv").sort_values("state_id").reset_index(drop=True)
    k11_folds = pd.read_csv(K11_ROOT / "13_query_state_cv_folds.csv").sort_values("state_id").reset_index(drop=True)
    k12_folds = pd.read_csv(K12_ROOT / "16_query_state_cv_folds.csv").sort_values("state_id").reset_index(drop=True)
    if not np.array_equal(k9_folds[["state_id", "fold"]].to_numpy(), k11_folds[["state_id", "fold"]].to_numpy()) or not np.array_equal(k9_folds[["state_id", "fold"]].to_numpy(), k12_folds[["state_id", "fold"]].to_numpy()):
        raise RuntimeError("historical K9/K11/K12 CV folds differ")
    real_b_distance = np.load(K9_ROOT / "04_realB_ground_truth_cnmse_matrix.npy")
    shareability = np.load(K9_ROOT / "05_realB_shareability_mask.npy")
    if real_b_distance.shape != (STATE_COUNT, STATE_COUNT) or shareability.shape != real_b_distance.shape or not np.all(np.isneginf(np.diag(real_b_distance))) or not shareability.diagonal().all():
        raise RuntimeError("Real-B ground truth contract changed")
    formal_seed_ids: dict[str, int] = {}
    for _, seed_name, formal_name, _ in SEED_SPECS:
        rows = k12_summary.loc[k12_summary["model_name"].eq(formal_name), "model_id"].drop_duplicates()
        if rows.shape[0] != 1:
            raise RuntimeError(f"formal seed result is missing or ambiguous: {formal_name}")
        formal_seed_ids[seed_name] = int(rows.iloc[0])
        support_row = k12_support.loc[k12_support["model_name"].eq(formal_name)]
        expected = tuple(core_ids) + (next(spec[3] for spec in SEED_SPECS if spec[1] == seed_name),)
        if support_row.shape[0] != 1 or tuple(json.loads(support_row.iloc[0]["support_basis_ids"])) != expected:
            raise RuntimeError(f"formal {seed_name} support changed")
    core_rows = k11_summary.loc[k11_summary["model_name"].eq(CORE_MODEL_NAME)]
    if core_rows.shape[0] != 1:
        raise RuntimeError("formal K11 core summary is missing")
    core_query = k11_query.loc[k11_query["candidate_id"].eq(21)].sort_values("State_R").reset_index(drop=True)
    if core_query.shape[0] != STATE_COUNT:
        raise RuntimeError("formal K11 core query table is not 425 rows")
    seed_distances = {
        seed_name: np.load(K12_ROOT / "distance_matrices" / f"{formal_name}.npy")
        for _, seed_name, formal_name, _ in SEED_SPECS
    }
    return {
        "k12_support": k12_support,
        "k12_summary": k12_summary,
        "k12_model": k12_model,
        "k12_query": k12_query,
        "k12_ids": k12_ids,
        "k12_theta_a": k12_theta_a,
        "k12_theta_c": k12_theta_c,
        "k11_summary": k11_summary,
        "k11_model": k11_model,
        "k11_query": k11_query,
        "k11_ids": k11_ids,
        "k11_theta_a": k11_theta_a,
        "k11_theta_c": k11_theta_c,
        "formal_seed_ids": formal_seed_ids,
        "seed_distances": seed_distances,
        "real_b_distance": real_b_distance,
        "shareability": shareability,
        "folds": k12_folds,
        "core_summary": core_rows.iloc[0].to_dict(),
        "core_model": k11_model.loc[k11_model["model_id"].eq(21)],
        "core_query": core_query,
        "single_add_summary": k12_summary.loc[k12_summary["candidate_id"].ne(0)].copy(),
    }


def _build_candidates(core_ids: tuple[str, ...], by_id: dict[str, EnvelopeBasis]) -> tuple[pd.DataFrame, pd.DataFrame, tuple[BeamCandidate, ...], pd.DataFrame]:
    seed_candidates: list[BeamCandidate] = []
    seed_rows: list[dict[str, Any]] = []
    for seed_id, seed_name, _, extra_basis_id in SEED_SPECS:
        term = by_id[extra_basis_id]
        support_ids = core_ids + (extra_basis_id,)
        candidate = BeamCandidate(seed_id, seed_name, "seed", (seed_name,), seed_name, SEED_K, (extra_basis_id,), (int(term.index),), support_ids, _support_hash(support_ids))
        seed_candidates.append(candidate)
        seed_rows.append({"seed_id": seed_id, "seed_name": seed_name, "formal_model_name": next(spec[2] for spec in SEED_SPECS if spec[1] == seed_name), "extra_basis_id": extra_basis_id, "extra_p": int(term.order), "extra_m": term.signal_delay, "extra_q": term.envelope_delay, "extra_is_cross_delay": bool(term.family == "CROSS_ENVELOPE"), "K": SEED_K, "support_basis_ids": json.dumps(list(support_ids), ensure_ascii=False), "support_indices": json.dumps(list(range(SEED_K))), "support_hash": candidate.support_hash, "dmax": backend.MP_DMAX, "ridge_lambda": backend.RIDGE_LAMBDA})
    remaining = tuple(term for term in by_id.values() if term.basis_id not in set(core_ids))
    if len(remaining) != 64:
        raise RuntimeError(f"remaining Envelope75 terms changed: {len(remaining)}")
    children: dict[str, BeamCandidate] = {}
    parent_names: dict[str, list[str]] = {}
    edge_rows: list[dict[str, Any]] = []
    edge_id = 0
    for seed in seed_candidates:
        seed_name = seed.branch_seed_ids[0]
        for term in remaining:
            if term.basis_id in seed.extension_basis_ids:
                continue
            extension_ids = tuple(sorted((seed.extension_basis_ids[0], term.basis_id), key=lambda basis_id: by_id[basis_id].index))
            extension_indices = tuple(int(by_id[basis_id].index) for basis_id in extension_ids)
            support_ids = core_ids + extension_ids
            support_hash = _support_hash(support_ids)
            if support_hash not in children:
                child_id = 3 + len(children)
                child = BeamCandidate(child_id, f"K13_plus_{extension_ids[0]}__{extension_ids[1]}", "k13", (seed_name,), seed_name, CHILD_K, extension_ids, extension_indices, support_ids, support_hash)
                children[support_hash] = child
                parent_names[support_hash] = [seed_name]
                is_duplicate = False
            else:
                child = children[support_hash]
                parent_names[support_hash].append(seed_name)
                is_duplicate = True
            edge_rows.append({"edge_id": edge_id, "parent_seed_id": seed_name, "parent_support_hash": seed.support_hash, "added_basis_id": term.basis_id, "added_p": int(term.order), "added_m": term.signal_delay, "added_q": term.envelope_delay, "added_is_cross_delay": bool(term.family == "CROSS_ENVELOPE"), "child_support_hash": support_hash, "child_model_id": child.candidate_id, "is_duplicate_child": is_duplicate, "child_generation_count": 0})
            edge_id += 1
    if edge_id != 189 or len(children) != 186:
        raise RuntimeError(f"Beam expansion graph changed: raw={edge_id}, unique={len(children)}")
    finalized_children: list[BeamCandidate] = []
    for child in children.values():
        parents = tuple(parent_names[child.support_hash])
        finalized_children.append(replace(child, branch_seed_ids=parents, primary_parent_seed_id=parents[0]))
    candidates = tuple(seed_candidates + finalized_children)
    candidate_by_hash = {candidate.support_hash: candidate for candidate in finalized_children}
    for row in edge_rows:
        row["child_generation_count"] = len(candidate_by_hash[row["child_support_hash"]].branch_seed_ids)
    unique_rows = [{"child_model_id": child.candidate_id, "model_name": child.candidate_name, "support_hash": child.support_hash, "K": child.K, "extension_basis_1": child.extension_basis_ids[0], "extension_basis_2": child.extension_basis_ids[1], "parent_seed_ids": json.dumps(list(child.branch_seed_ids), ensure_ascii=False), "primary_parent_seed_id": child.primary_parent_seed_id, "generation_count": len(child.branch_seed_ids), "basis_ids": json.dumps(list(child.support_basis_ids), ensure_ascii=False), "support_indices": json.dumps(list(range(child.K))), "dmax": backend.MP_DMAX, "ridge_lambda": backend.RIDGE_LAMBDA} for child in finalized_children]
    seed_frame = pd.DataFrame(seed_rows)
    unique_frame = pd.DataFrame(unique_rows)
    dedup_pairs = unique_frame.loc[unique_frame["generation_count"].gt(1), ["child_model_id", "extension_basis_1", "extension_basis_2", "parent_seed_ids", "generation_count"]]
    expected_pairs = {tuple(sorted(pair, key=lambda basis_id: by_id[basis_id].index)) for pair in (("ENV_p04_m0_q1", "ENV_p03_m0_q1"), ("ENV_p04_m0_q1", "ENV_p09_m2_q0"), ("ENV_p03_m0_q1", "ENV_p09_m2_q0"))}
    observed_pairs = {tuple(row) for row in dedup_pairs[["extension_basis_1", "extension_basis_2"]].itertuples(index=False, name=None)}
    if observed_pairs != expected_pairs or dedup_pairs.shape[0] != 3:
        raise RuntimeError(f"multi-parent support set changed: {observed_pairs}")
    dedup_audit = pd.DataFrame([{"child_model_id": int(row.child_model_id), "extension_basis_1": row.extension_basis_1, "extension_basis_2": row.extension_basis_2, "parent_seed_ids": row.parent_seed_ids, "generation_count": int(row.generation_count)} for row in dedup_pairs.itertuples(index=False)])
    return seed_frame, unique_frame, candidates, dedup_audit


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


def _candidate_phi(common_core: np.ndarray, common_env: np.ndarray, candidate: BeamCandidate) -> np.ndarray:
    return np.column_stack((common_core, *(common_env[:, index] for index in candidate.extension_indices))).astype(np.complex128)


def _candidate_retrieval(
    candidate: BeamCandidate,
    theta_a: np.ndarray,
    theta_c: np.ndarray,
    common_phi: np.ndarray,
    real_b_distance: np.ndarray,
    shareability: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    frame, summary, distance, lut, query = k9_utils._candidate_retrieval(candidate, theta_a, theta_c, common_phi, real_b_distance, shareability)
    metadata = {
        "model_id": candidate.candidate_id,
        "model_name": candidate.candidate_name,
        "model_type": candidate.model_type,
        "branch_seed_ids": json.dumps(list(candidate.branch_seed_ids), ensure_ascii=False),
        "primary_parent_seed_id": candidate.primary_parent_seed_id,
        "support_hash": candidate.support_hash,
        "extension_basis_1": candidate.extension_basis_ids[0],
        "extension_basis_2": candidate.extension_basis_ids[1] if len(candidate.extension_basis_ids) == 2 else "NONE",
        "K": candidate.K,
    }
    for key, value in metadata.items():
        frame[key] = value
        summary[key] = value
    frame["retrieved_real_B_CNMSE_dB"] = frame["retrieved_realB_CNMSE_dB"]
    summary["top1_pass_count"] = summary["top1_realB_pass_count"]
    summary["top1_pass_rate"] = summary["top1_realB_pass_rate"]
    values = frame["top12_margin_dB"].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    summary["top12_margin_Q05"] = float(np.quantile(finite, 0.05)) if finite.size else float("nan")
    summary["top12_margin_Q10"] = float(np.quantile(finite, 0.10)) if finite.size else float("nan")
    return frame, summary, distance, lut, query


def _seed_regression(
    candidate: BeamCandidate,
    current_model: pd.DataFrame,
    current_frame: pd.DataFrame,
    theta_a: np.ndarray,
    theta_c: np.ndarray,
    current_distance: np.ndarray,
    current_lut: np.ndarray,
    current_query: np.ndarray,
    references: dict[str, Any],
    common_phi: np.ndarray,
) -> dict[str, Any]:
    formal_id = int(references["formal_seed_ids"][candidate.candidate_name])
    ref_model = references["k12_model"].loc[references["k12_model"]["model_id"].eq(formal_id)].sort_values("State_R").reset_index(drop=True)
    cur_model = current_model.loc[current_model["model_id"].eq(candidate.candidate_id)].sort_values("State_R").reset_index(drop=True)
    ref_frame = references["k12_query"].loc[references["k12_query"]["candidate_id"].eq(formal_id)].sort_values("State_R").reset_index(drop=True)
    ref_index = int(np.flatnonzero(references["k12_ids"] == formal_id)[0])
    metric_columns = ["Y_Aend_train_NMSE_dB", "Y_Aend_B_NMSE_dB", "Y_C2_train_NMSE_dB", "Y_C2_B_NMSE_dB"]
    metric_errors = {column: float(np.max(np.abs(cur_model[column].to_numpy(dtype=float) - ref_model[column].to_numpy(dtype=float)))) for column in metric_columns}
    theta_a_error = float(np.max(np.abs(theta_a[candidate.candidate_id, :, :SEED_K] - references["k12_theta_a"][ref_index])))
    theta_c_error = float(np.max(np.abs(theta_c[candidate.candidate_id, :, :SEED_K] - references["k12_theta_c"][ref_index])))
    ref_lut = (common_phi @ references["k12_theta_a"][ref_index].T).T.astype(np.complex128)
    ref_query_fp = (common_phi @ references["k12_theta_c"][ref_index].T).T.astype(np.complex128)
    lut_error = float(np.max(np.abs(current_lut - ref_lut)))
    query_error = float(np.max(np.abs(current_query - ref_query_fp)))
    distance_error, distance_mismatch = _max_abs_with_neginf(current_distance, references["seed_distances"][candidate.candidate_name])
    retrieved_error, retrieved_mismatch = _max_abs_with_neginf(current_frame["retrieved_realB_CNMSE_dB"].to_numpy(dtype=float), ref_frame["retrieved_realB_CNMSE_dB"].to_numpy(dtype=float))
    fp_top1_error = float(np.max(np.abs(current_frame["fingerprint_top1_CNMSE_dB"].to_numpy(dtype=float) - ref_frame["fingerprint_top1_CNMSE_dB"].to_numpy(dtype=float))))
    fp_top2_error = float(np.max(np.abs(current_frame["fingerprint_top2_CNMSE_dB"].to_numpy(dtype=float) - ref_frame["fingerprint_top2_CNMSE_dB"].to_numpy(dtype=float))))
    margin_error = float(np.max(np.abs(current_frame["shareability_margin_dB"].to_numpy(dtype=float) - ref_frame["shareability_margin_dB"].to_numpy(dtype=float))))
    state_q_equal = bool(np.array_equal(current_frame["State_Q"].to_numpy(dtype=int), ref_frame["State_Q"].to_numpy(dtype=int)))
    pass_flags_equal = bool(np.array_equal(current_frame["realB_pass"].to_numpy(dtype=bool), ref_frame["realB_pass"].to_numpy(dtype=bool)))
    first_rank_equal = bool(np.array_equal(current_frame["first_shareable_rank"].to_numpy(dtype=int), ref_frame["first_shareable_rank"].to_numpy(dtype=int)))
    result = {
        "seed_name": candidate.candidate_name,
        "formal_model_name": next(spec[2] for spec in SEED_SPECS if spec[1] == candidate.candidate_name),
        "formal_model_id": formal_id,
        "pass": False,
        "model_metric_max_abs_error_dB": metric_errors,
        "theta_Aend_max_abs_error": theta_a_error,
        "theta_C2_max_abs_error": theta_c_error,
        "LUT_fingerprint_max_abs_error": lut_error,
        "Query_fingerprint_max_abs_error": query_error,
        "distance_max_abs_error_dB": distance_error,
        "distance_nonfinite_pattern_mismatch": distance_mismatch,
        "retrieved_Real_B_max_abs_error_dB": retrieved_error,
        "retrieved_Real_B_nonfinite_pattern_mismatch": retrieved_mismatch,
        "fingerprint_top1_max_abs_error_dB": fp_top1_error,
        "fingerprint_top2_max_abs_error_dB": fp_top2_error,
        "shareability_margin_max_abs_error_dB": margin_error,
        "State_Q_equal": state_q_equal,
        "realB_pass_equal": pass_flags_equal,
        "first_shareable_rank_equal": first_rank_equal,
        "Top1": int(current_frame["realB_pass"].sum()),
        "Exact": int(current_frame["exact_hit"].sum()),
        "Nonself": int((current_frame["State_Q"].to_numpy(dtype=int) != np.arange(STATE_COUNT)).sum()),
        "Nonself_pass": int(((current_frame["State_Q"].to_numpy(dtype=int) != np.arange(STATE_COUNT)) & current_frame["realB_pass"].to_numpy(dtype=bool)).sum()),
        "Q05": float(current_frame["shareability_margin_dB"].quantile(0.05)),
        "MRR": float(np.mean(1.0 / current_frame["first_shareable_rank"].to_numpy(dtype=float))),
        "Top2": int(current_frame["Top2_has_shareable"].sum()),
        "Top3": int(current_frame["Top3_has_shareable"].sum()),
        "Top5": int(current_frame["Top5_has_shareable"].sum()),
        "Top10": int(current_frame["Top10_has_shareable"].sum()),
        "failure_state_ids": current_frame.loc[~current_frame["realB_pass"], "State_R"].astype(int).tolist(),
    }
    result["pass"] = bool(
        max(metric_errors.values()) < 1e-10
        and theta_a_error < 1e-12
        and theta_c_error < 1e-12
        and lut_error < 1e-12
        and query_error < 1e-12
        and distance_error < 1e-12
        and distance_mismatch == 0
        and retrieved_error < 1e-9
        and retrieved_mismatch == 0
        and fp_top1_error < 1e-12
        and fp_top2_error < 1e-12
        and margin_error < 1e-12
        and state_q_equal
        and pass_flags_equal
        and first_rank_equal
    )
    if not result["pass"]:
        raise RuntimeError(f"HARD FAIL: {candidate.candidate_name} seed regression failed: {result}")
    return result


def _run_model_phase(candidates: tuple[BeamCandidate, ...], core_index: int, resume: bool) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    model_path = RESULT_ROOT / "09_model_quality_long.csv"
    all_coefficient_path = RESULT_ROOT / "09_all_candidate_coefficients.npz"
    if resume and model_path.is_file() and all_coefficient_path.is_file():
        try:
            frame = pd.read_csv(model_path)
            with np.load(all_coefficient_path, allow_pickle=False) as data:
                theta_a = np.asarray(data["theta_Aend_padded"])
                theta_c = np.asarray(data["theta_C2_padded"])
            if frame.shape[0] == len(candidates) * STATE_COUNT and theta_a.shape == (len(candidates), STATE_COUNT, CHILD_K) and theta_c.shape == theta_a.shape:
                return frame, theta_a, theta_c
        except (OSError, ValueError, KeyError):
            pass
    specs = tuple((candidate.candidate_id, candidate.extension_indices) for candidate in candidates)
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=WORKER_COUNT, mp_context=get_context("spawn"), initializer=backend.worker_init, initargs=(specs, core_index)) as executor:
        futures = {executor.submit(backend.model_worker_entry, state_id): state_id for state_id in range(STATE_COUNT)}
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed % 25 == 0 or completed == STATE_COUNT:
                _checkpoint(phase="k13_model_progress", completed_model_states=completed, completed_model_state_ids=sorted(int(item["State_n_R"]) for item in results))
                print(f"[SMALLBEAM K13 MODEL] {completed}/{STATE_COUNT}", flush=True)
    results.sort(key=lambda item: int(item["State_n_R"]))
    if [int(item["State_n_R"]) for item in results] != list(range(STATE_COUNT)):
        raise RuntimeError("Beam model phase state order is not 0...424")
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    theta_a = np.zeros((len(candidates), STATE_COUNT, CHILD_K), dtype=np.complex128)
    theta_c = np.zeros_like(theta_a)
    rows: list[dict[str, Any]] = []
    for item in results:
        state_id = int(item["State_n_R"])
        for fit in item["candidates"]:
            candidate = candidate_by_id[int(fit["candidate_id"])]
            theta_a[candidate.candidate_id, state_id] = fit["theta_Aend_padded"]
            theta_c[candidate.candidate_id, state_id] = fit["theta_C2_padded"]
            rows.append({
                "model_id": candidate.candidate_id,
                "model_name": candidate.candidate_name,
                "model_type": candidate.model_type,
                "branch_seed_ids": json.dumps(list(candidate.branch_seed_ids), ensure_ascii=False),
                "primary_parent_seed_id": candidate.primary_parent_seed_id,
                "K": candidate.K,
                "support_hash": candidate.support_hash,
                "extension_basis_1": candidate.extension_basis_ids[0],
                "extension_basis_2": candidate.extension_basis_ids[1] if len(candidate.extension_basis_ids) == 2 else "NONE",
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
                "Aend_condition_number": float(fit["Aend_condition_number"]),
                "C2_condition_number": float(fit["C2_condition_number"]),
                "Aend_condition_number_augmented": float(fit["Aend_condition_number_augmented"]),
                "C2_condition_number_augmented": float(fit["C2_condition_number_augmented"]),
                "Aend_finite": bool(fit["finite"]),
                "C2_finite": bool(fit["finite"]),
            })
    frame = pd.DataFrame(rows).sort_values(["model_id", "State_R"]).reset_index(drop=True)
    if frame.shape[0] != len(candidates) * STATE_COUNT:
        raise RuntimeError(f"Beam model quality row count failed: {frame.shape}")
    frame.to_csv(model_path, index=False)
    extension_indices = np.full((len(candidates), 2), -1, dtype=np.int64)
    for candidate in candidates:
        extension_indices[candidate.candidate_id, : len(candidate.extension_indices)] = np.asarray(candidate.extension_indices, dtype=np.int64)
    np.savez_compressed(all_coefficient_path, model_ids=np.asarray([candidate.candidate_id for candidate in candidates], dtype=np.int64), model_names=np.asarray([candidate.candidate_name for candidate in candidates]), theta_Aend_padded=theta_a, theta_C2_padded=theta_c, K=np.asarray([candidate.K for candidate in candidates], dtype=np.int64), extension_indices=extension_indices, support_hashes=np.asarray([candidate.support_hash for candidate in candidates]))
    return frame, theta_a, theta_c


def _write_coefficient_artifacts(candidates: tuple[BeamCandidate, ...], theta_a: np.ndarray, theta_c: np.ndarray) -> None:
    seeds = candidates[:3]
    children = candidates[3:]
    np.savez_compressed(RESULT_ROOT / "10_coefficients_seeds.npz", model_ids=np.asarray([candidate.candidate_id for candidate in seeds], dtype=np.int64), model_names=np.asarray([candidate.candidate_name for candidate in seeds]), theta_Aend_padded=theta_a[:3, :, :SEED_K], theta_C2_padded=theta_c[:3, :, :SEED_K], K=np.asarray([candidate.K for candidate in seeds], dtype=np.int64), support_hashes=np.asarray([candidate.support_hash for candidate in seeds]), extension_indices=np.asarray([candidate.extension_indices[0] for candidate in seeds], dtype=np.int64))
    np.savez_compressed(RESULT_ROOT / "11_coefficients_k13.npz", model_ids=np.asarray([candidate.candidate_id for candidate in children], dtype=np.int64), model_names=np.asarray([candidate.candidate_name for candidate in children]), theta_Aend_padded=theta_a[3:, :, :CHILD_K], theta_C2_padded=theta_c[3:, :, :CHILD_K], K=np.asarray([candidate.K for candidate in children], dtype=np.int64), support_hashes=np.asarray([candidate.support_hash for candidate in children]), extension_indices=np.asarray([candidate.extension_indices for candidate in children], dtype=np.int64))


def _write_seed_support_outputs(core_ids: tuple[str, ...], by_id: dict[str, EnvelopeBasis], seed_frame: pd.DataFrame) -> None:
    pd.DataFrame([{"core_column_index": index, "basis_id": basis_id, "basis_family": by_id[basis_id].family, "order_p": int(by_id[basis_id].order), "delay_m": by_id[basis_id].signal_delay, "envelope_delay_q": by_id[basis_id].envelope_delay, "formula": by_id[basis_id].formula} for index, basis_id in enumerate(core_ids)]).to_csv(RESULT_ROOT / "02_common_k11_core_support.csv", index=False)
    seed_frame.to_csv(RESULT_ROOT / "03_beam_seed_supports.csv", index=False)


def _save_fingerprint(path_stem: str, lut: np.ndarray, query: np.ndarray) -> None:
    np.save(RESULT_ROOT / "fingerprints" / f"{path_stem}_lut.npy", np.asarray(lut, dtype=np.complex128))
    np.save(RESULT_ROOT / "fingerprints" / f"{path_stem}_query.npy", np.asarray(query, dtype=np.complex128))


def _compare_frames(current: pd.DataFrame, reference: pd.DataFrame) -> dict[str, Any]:
    current = current.sort_values("State_R").reset_index(drop=True)
    reference = reference.sort_values("State_R").reset_index(drop=True)
    current_pass = current["realB_pass"].to_numpy(dtype=bool)
    reference_pass = reference["realB_pass"].to_numpy(dtype=bool)
    recovered_ids = np.flatnonzero(~reference_pass & current_pass).astype(int).tolist()
    regressed_ids = np.flatnonzero(reference_pass & ~current_pass).astype(int).tolist()
    unchanged_pass = int((reference_pass & current_pass).sum())
    unchanged_fail = int((~reference_pass & ~current_pass).sum())
    return {
        "top1": int(current_pass.sum()),
        "exact": int(current["exact_hit"].sum()),
        "nonself_count": int((current["State_Q"].to_numpy(dtype=int) != np.arange(STATE_COUNT)).sum()),
        "nonself_pass": int(((current["State_Q"].to_numpy(dtype=int) != np.arange(STATE_COUNT)) & current_pass).sum()),
        "Q05": float(current["shareability_margin_dB"].quantile(0.05)),
        "MRR": float(np.mean(1.0 / current["first_shareable_rank"].to_numpy(dtype=float))),
        "Top2": int(current["Top2_has_shareable"].sum()),
        "Top3": int(current["Top3_has_shareable"].sum()),
        "Top5": int(current["Top5_has_shareable"].sum()),
        "Top10": int(current["Top10_has_shareable"].sum()),
        "recovered_count": len(recovered_ids),
        "recovered_state_ids": recovered_ids,
        "regressed_count": len(regressed_ids),
        "regressed_state_ids": regressed_ids,
        "unchanged_pass": unchanged_pass,
        "unchanged_fail": unchanged_fail,
        "delta_top1": int(current_pass.sum() - reference_pass.sum()),
        "delta_Q05": float(current["shareability_margin_dB"].quantile(0.05) - reference["shareability_margin_dB"].quantile(0.05)),
        "delta_MRR": float(np.mean(1.0 / current["first_shareable_rank"].to_numpy(dtype=float)) - np.mean(1.0 / reference["first_shareable_rank"].to_numpy(dtype=float))),
    }


def _comparison_row(candidate: BeamCandidate, comparison: dict[str, Any]) -> dict[str, Any]:
    return {
        "model_id": candidate.candidate_id,
        "model_name": candidate.candidate_name,
        "model_type": candidate.model_type,
        "branch_seed_ids": json.dumps(list(candidate.branch_seed_ids), ensure_ascii=False),
        "support_hash": candidate.support_hash,
        **comparison,
    }


def _build_global_comparisons(candidates: tuple[BeamCandidate, ...], frames: dict[int, pd.DataFrame], summary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    champion = frames[SEED_A_ID]
    rows: list[dict[str, Any]] = []
    for candidate in candidates[3:]:
        comparison = _compare_frames(frames[candidate.candidate_id], champion)
        rows.append(_comparison_row(candidate, comparison))
    return pd.DataFrame(rows), pd.DataFrame()


def _build_edge_deltas(edges: pd.DataFrame, candidates: tuple[BeamCandidate, ...], frames: dict[int, pd.DataFrame], summary: pd.DataFrame) -> pd.DataFrame:
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    seed_by_name = {candidate.candidate_name: candidate for candidate in candidates[:3]}
    rows: list[dict[str, Any]] = []
    for edge in edges.itertuples(index=False):
        child = candidate_by_id[int(edge.child_model_id)]
        parent = seed_by_name[str(edge.parent_seed_id)]
        comparison = _compare_frames(frames[child.candidate_id], frames[parent.candidate_id])
        parent_summary = summary.loc[summary["model_id"].eq(parent.candidate_id)].iloc[0]
        child_summary = summary.loc[summary["model_id"].eq(child.candidate_id)].iloc[0]
        rows.append({
            "edge_id": int(edge.edge_id),
            "parent_seed_id": parent.candidate_name,
            "child_model_id": child.candidate_id,
            "child_model_name": child.candidate_name,
            "child_support_hash": child.support_hash,
            "added_basis_id": edge.added_basis_id,
            "parent_top1": int(parent_summary["top1_pass_count"]),
            "child_top1": int(child_summary["top1_pass_count"]),
            "delta_top1_vs_parent": comparison["delta_top1"],
            "recovered_vs_parent": comparison["recovered_count"],
            "regressed_vs_parent": comparison["regressed_count"],
            "recovered_state_ids_vs_parent": json.dumps(comparison["recovered_state_ids"]),
            "regressed_state_ids_vs_parent": json.dumps(comparison["regressed_state_ids"]),
            "parent_Q05": float(parent_summary["share_margin_Q05"]),
            "child_Q05": float(child_summary["share_margin_Q05"]),
            "delta_Q05_vs_parent": float(child_summary["share_margin_Q05"] - parent_summary["share_margin_Q05"]),
            "parent_MRR": float(parent_summary["MRR"]),
            "child_MRR": float(child_summary["MRR"]),
            "delta_MRR_vs_parent": float(child_summary["MRR"] - parent_summary["MRR"]),
            "strict_improvement_vs_parent": bool(comparison["delta_top1"] > 0),
        })
    return pd.DataFrame(rows).sort_values("edge_id").reset_index(drop=True)


def _build_champion_failure6(candidates: tuple[BeamCandidate, ...], frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    champion = frames[SEED_A_ID].set_index("State_R")
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        current = frames[candidate.candidate_id].set_index("State_R")
        for state_id in CHAMPION_FAILURES:
            rows.append({
                "model_id": candidate.candidate_id,
                "model_name": candidate.candidate_name,
                "model_type": candidate.model_type,
                "branch_seed_ids": json.dumps(list(candidate.branch_seed_ids), ensure_ascii=False),
                "State_R": state_id,
                "SeedA_State_Q": int(champion.loc[state_id, "State_Q"]),
                "model_State_Q": int(current.loc[state_id, "State_Q"]),
                "SeedA_pass": bool(champion.loc[state_id, "realB_pass"]),
                "model_pass": bool(current.loc[state_id, "realB_pass"]),
                "SeedA_first_shareable_rank": int(champion.loc[state_id, "first_shareable_rank"]),
                "model_first_shareable_rank": int(current.loc[state_id, "first_shareable_rank"]),
                "SeedA_margin": float(champion.loc[state_id, "shareability_margin_dB"]),
                "model_margin": float(current.loc[state_id, "shareability_margin_dB"]),
                "fingerprint_top1_distance_change_dB": float(current.loc[state_id, "fingerprint_top1_CNMSE_dB"] - champion.loc[state_id, "fingerprint_top1_CNMSE_dB"]),
                "model_realB_CNMSE_dB": float(current.loc[state_id, "retrieved_realB_CNMSE_dB"]),
                "recovered_vs_SeedA": bool(not champion.loc[state_id, "realB_pass"] and current.loc[state_id, "realB_pass"]),
                "regressed_vs_SeedA": bool(champion.loc[state_id, "realB_pass"] and not current.loc[state_id, "realB_pass"]),
                "rank_improved": bool(current.loc[state_id, "first_shareable_rank"] < champion.loc[state_id, "first_shareable_rank"]),
                "rank_degraded": bool(current.loc[state_id, "first_shareable_rank"] > champion.loc[state_id, "first_shareable_rank"]),
            })
    return pd.DataFrame(rows).sort_values(["model_id", "State_R"]).reset_index(drop=True)


def _build_pairwise_interaction(
    children: tuple[BeamCandidate, ...],
    frames: dict[int, pd.DataFrame],
    summary: pd.DataFrame,
    references: dict[str, Any],
) -> pd.DataFrame:
    single_summary = references["single_add_summary"].set_index("added_basis_id")
    core_frame = references["core_query"]
    n0 = int(references["core_summary"]["top1_pass_count"])
    rows: list[dict[str, Any]] = []
    for child in children:
        basis_a, basis_b = child.extension_basis_ids
        if basis_a not in single_summary.index or basis_b not in single_summary.index:
            raise RuntimeError(f"single-add reference missing for {child.candidate_name}")
        row_a = single_summary.loc[basis_a]
        row_b = single_summary.loc[basis_b]
        current = summary.loc[summary["model_id"].eq(child.candidate_id)].iloc[0]
        comparison_core = _compare_frames(frames[child.candidate_id], core_frame)
        n_a = int(row_a["top1_pass_count"])
        n_b = int(row_b["top1_pass_count"])
        n_ab = int(current["top1_pass_count"])
        rows.append({
            "child_model_id": child.candidate_id,
            "model_name": child.candidate_name,
            "support_hash": child.support_hash,
            "basis_a": basis_a,
            "basis_b": basis_b,
            "N_core": n0,
            "N_a": n_a,
            "N_b": n_b,
            "N_ab": n_ab,
            "gain_a_vs_core": n_a - n0,
            "gain_b_vs_core": n_b - n0,
            "gain_combined_vs_core": n_ab - n0,
            "gain_over_best_single": n_ab - max(n_a, n_b),
            "interaction_excess": n_ab - n_a - n_b + n0,
            "recovered_vs_core": comparison_core["recovered_count"],
            "regressed_vs_core": comparison_core["regressed_count"],
            "Q05_ab": float(current["share_margin_Q05"]),
            "MRR_ab": float(current["MRR"]),
            "Top1_ab": n_ab,
            "Top2_ab": int(current["Top2_oracle"]),
            "Top3_ab": int(current["Top3_oracle"]),
            "strict_breakthrough": bool(n_ab > 419),
            "branch_seed_ids": json.dumps(list(child.branch_seed_ids), ensure_ascii=False),
        })
    return pd.DataFrame(rows).sort_values("child_model_id").reset_index(drop=True)


def _classify_pair(term_a: EnvelopeBasis, term_b: EnvelopeBasis) -> str:
    cross_a = term_a.family == "CROSS_ENVELOPE"
    cross_b = term_b.family == "CROSS_ENVELOPE"
    return "cross+cross" if cross_a and cross_b else "cross+aligned" if cross_a or cross_b else "aligned+aligned"


def _build_structure_summaries(
    children: tuple[BeamCandidate, ...],
    by_id: dict[str, EnvelopeBasis],
    summary: pd.DataFrame,
    interaction: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    interaction_by_id = interaction.set_index("child_model_id")
    child_summary = summary.set_index("model_id")
    pair_rows: list[dict[str, Any]] = []
    for child in children:
        term_a = by_id[child.extension_basis_ids[0]]
        term_b = by_id[child.extension_basis_ids[1]]
        pair_type = _classify_pair(term_a, term_b)
        pair_rows.append({
            "child_model_id": child.candidate_id,
            "basis_a": child.extension_basis_ids[0],
            "basis_b": child.extension_basis_ids[1],
            "pair_type": pair_type,
            "Top1": int(child_summary.loc[child.candidate_id, "top1_pass_count"]),
            "strict_breakthrough": bool(child_summary.loc[child.candidate_id, "strict_breakthrough"]),
            "safe_breakthrough": bool(child_summary.loc[child.candidate_id, "safe_breakthrough"]),
            "gain_over_best_single": float(interaction_by_id.loc[child.candidate_id, "gain_over_best_single"]),
            "interaction_excess": float(interaction_by_id.loc[child.candidate_id, "interaction_excess"]),
        })
    pair_frame = pd.DataFrame(pair_rows)
    pair_type_frame = pair_frame.groupby("pair_type", as_index=False).agg(candidate_count=("child_model_id", "count"), Top1_mean=("Top1", "mean"), Top1_max=("Top1", "max"), strict_breakthrough_count=("strict_breakthrough", "sum"), safe_breakthrough_count=("safe_breakthrough", "sum"), mean_gain_over_best_single=("gain_over_best_single", "mean"), max_gain_over_best_single=("gain_over_best_single", "max"), mean_interaction_excess=("interaction_excess", "mean"), max_interaction_excess=("interaction_excess", "max"))
    core_basis_ids = set(children[0].support_basis_ids[:CORE_K])
    basis_ids = tuple(term.basis_id for term in by_id.values() if term.basis_id not in core_basis_ids)
    basis_rows: list[dict[str, Any]] = []
    for basis_id in basis_ids:
        term = by_id[basis_id]
        containing = [child for child in children if basis_id in child.extension_basis_ids]
        subset = child_summary.loc[[child.candidate_id for child in containing]]
        subset_interaction = interaction_by_id.loc[[child.candidate_id for child in containing]]
        basis_rows.append({
            "basis_id": basis_id,
            "basis_family": term.family,
            "p": int(term.order),
            "m": term.signal_delay,
            "q": term.envelope_delay,
            "is_cross_delay": bool(term.family == "CROSS_ENVELOPE"),
            "child_count": len(containing),
            "child_Top1_mean": float(subset["top1_pass_count"].mean()),
            "child_Top1_max": int(subset["top1_pass_count"].max()),
            "strict_breakthrough_count": int(subset["strict_breakthrough"].sum()),
            "safe_breakthrough_count": int(subset["safe_breakthrough"].sum()),
            "mean_gain_over_best_single": float(subset_interaction["gain_over_best_single"].mean()),
            "max_gain_over_best_single": float(subset_interaction["gain_over_best_single"].max()),
        })
    return pair_type_frame, pd.DataFrame(basis_rows).sort_values(["p", "m", "q"]).reset_index(drop=True)


def _local_cv_results(candidates: tuple[BeamCandidate, ...], frames: dict[int, pd.DataFrame], folds: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    seed_names = [candidate.candidate_name for candidate in candidates[:3]]
    for fold in range(5):
        validation_ids = folds.loc[folds["fold"].eq(fold), "state_id"].to_numpy(dtype=int)
        train_ids = folds.loc[~folds["fold"].eq(fold), "state_id"].to_numpy(dtype=int)
        train_summaries = {candidate.candidate_id: k9_utils._summarize_query_frame(frames[candidate.candidate_id].loc[frames[candidate.candidate_id]["State_R"].isin(train_ids)], candidate) for candidate in candidates}
        global_ranking = sorted(train_summaries.values(), key=k9_utils._selection_key)
        global_winner_id = int(global_ranking[0]["candidate_id"])
        global_rank = {int(item["candidate_id"]): rank for rank, item in enumerate(global_ranking, start=1)}
        branch_rankings: dict[str, dict[int, int]] = {}
        branch_winners: dict[str, int] = {}
        for branch_name in seed_names:
            branch_ids = [candidate.candidate_id for candidate in candidates if branch_name in candidate.branch_seed_ids]
            ranking = sorted((train_summaries[candidate_id] for candidate_id in branch_ids), key=k9_utils._selection_key)
            branch_rankings[branch_name] = {int(item["candidate_id"]): rank for rank, item in enumerate(ranking, start=1)}
            branch_winners[branch_name] = int(ranking[0]["candidate_id"])
        for candidate in candidates:
            train_summary = train_summaries[candidate.candidate_id]
            validation_frame = frames[candidate.candidate_id].loc[frames[candidate.candidate_id]["State_R"].isin(validation_ids)]
            validation_summary = k9_utils._summarize_query_frame(validation_frame, candidate)
            branch_ranks = {branch: branch_rankings[branch][candidate.candidate_id] for branch in seed_names if candidate.candidate_id in branch_rankings[branch]}
            branch_winner_flags = {branch: candidate.candidate_id == winner for branch, winner in branch_winners.items()}
            rows.append({
                "fold": fold,
                "model_id": candidate.candidate_id,
                "model_name": candidate.candidate_name,
                "model_type": candidate.model_type,
                "branch_seed_ids": json.dumps(list(candidate.branch_seed_ids), ensure_ascii=False),
                "K": candidate.K,
                "train_query_count": len(train_ids),
                "validation_query_count": len(validation_ids),
                "train_is_global_winner": candidate.candidate_id == global_winner_id,
                "global_train_rank": global_rank[candidate.candidate_id],
                "branch_train_rank": json.dumps(branch_ranks, ensure_ascii=False, sort_keys=True),
                "branch_train_winner": json.dumps(branch_winner_flags, ensure_ascii=False, sort_keys=True),
                "train_top1": int(train_summary["top1_realB_pass_count"]),
                "train_nonself_pass_rate": float(train_summary["nonself_pass_rate"]),
                "train_Q05": float(train_summary["share_margin_Q05"]),
                "train_MRR": float(train_summary["MRR"]),
                "train_Top3": int(train_summary["Top3_oracle"]),
                "validation_exact": int(validation_summary["exact_hit_count"]),
                "validation_top1": int(validation_summary["top1_realB_pass_count"]),
                "validation_nonself": int(validation_summary["nonself_pass_count"]),
                "validation_nonself_rate": float(validation_summary["nonself_pass_rate"]),
                "validation_Q05": float(validation_summary["share_margin_Q05"]),
                "validation_MRR": float(validation_summary["MRR"]),
                "validation_Top3": int(validation_summary["Top3_oracle"]),
                "validation_Top2": int(validation_summary["Top2_oracle"]),
                "validation_Top5": int(validation_summary["Top5_oracle"]),
                "validation_Top10": int(validation_summary["Top10_oracle"]),
            })
    cv = pd.DataFrame(rows).sort_values(["fold", "model_id"]).reset_index(drop=True)
    stability_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        current = cv.loc[cv["model_id"].eq(candidate.candidate_id)]
        global_wins = current.loc[current["train_is_global_winner"]]
        branch_counts = {branch_name: int(current["branch_train_winner"].map(lambda value, branch=branch_name: bool(json.loads(value).get(branch, False))).sum()) for branch_name in seed_names}
        stability_rows.append({
            "model_id": candidate.candidate_id,
            "model_name": candidate.candidate_name,
            "model_type": candidate.model_type,
            "branch_seed_ids": json.dumps(list(candidate.branch_seed_ids), ensure_ascii=False),
            "support_hash": candidate.support_hash,
            "global_train_winner_count": int(global_wins.shape[0]),
            "global_train_winner_frequency": float(global_wins.shape[0] / 5),
            "global_winner_folds": json.dumps(global_wins["fold"].astype(int).tolist()),
            "seedA_branch_winner_count": branch_counts[seed_names[0]],
            "seedB_branch_winner_count": branch_counts[seed_names[1]],
            "seedC_branch_winner_count": branch_counts[seed_names[2]],
            "validation_top1_mean": float(current["validation_top1"].mean()),
            "validation_top1_min": int(current["validation_top1"].min()),
            "validation_top1_max": int(current["validation_top1"].max()),
            "global_winner_validation_top1_mean": float(global_wins["validation_top1"].mean()) if not global_wins.empty else float("nan"),
            "global_winner_validation_top1_min": int(global_wins["validation_top1"].min()) if not global_wins.empty else np.nan,
            "global_winner_validation_top1_max": int(global_wins["validation_top1"].max()) if not global_wins.empty else np.nan,
            "validation_nonself_rate_mean": float(current["validation_nonself_rate"].mean()),
            "validation_Q05_mean": float(current["validation_Q05"].mean()),
            "validation_MRR_mean": float(current["validation_MRR"].mean()),
        })
    return cv, pd.DataFrame(stability_rows).sort_values("model_id").reset_index(drop=True)


def _write_figures(
    candidates: tuple[BeamCandidate, ...],
    summary: pd.DataFrame,
    interaction: pd.DataFrame,
    failure6: pd.DataFrame,
    model_frame: pd.DataFrame,
    cv_stability: pd.DataFrame,
    branch_summary: pd.DataFrame,
) -> None:
    ordered = summary.sort_values("global_rank").reset_index(drop=True)
    color_map = {"seed_A_global_champion": "tab:blue", "seed_B_margin_branch": "tab:orange", "seed_C_structural_branch": "tab:green"}
    colors = [color_map.get(branch.split(",")[0].strip("[]\"'"), "tab:purple") if row.model_type == "seed" else "tab:purple" for row, branch in zip(ordered.itertuples(index=False), ordered["branch_seed_ids"], strict=True)]
    fig, ax = plt.subplots(figsize=(18, 8), dpi=300)
    x = np.arange(ordered.shape[0])
    ax.scatter(x, ordered["top1_pass_count"], c=colors, s=12, alpha=0.75, label="189 evaluation supports")
    ax.axhline(419, color="black", linestyle="--", linewidth=1.0, label="Current champion = 419")
    top_rows = ordered.head(1)
    for index, row in top_rows.iterrows():
        ax.annotate(f"{row.model_id}:{row.top1_pass_count}", (index, row.top1_pass_count), fontsize=6, rotation=90, ha="center", xytext=(0, 4), textcoords="offset points")
    ax.set_xlabel("Global diagnostic rank")
    ax.set_ylabel("Top-1 Real-B pass count")
    ax.set_ylim(min(0, float(ordered["top1_pass_count"].min()) - 5), 425)
    ax.set_title("Beam=3 one-layer expansion: global Top-1 Real-B comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "24_global_top1_comparison.png")
    plt.close(fig)

    branch_rows = []
    for row in branch_summary.itertuples(index=False):
        branch_rows.append((row.seed_name, "parent", row.parent_top1))
        branch_rows.append((row.seed_name, "best_child", row.best_child_top1))
    fig, ax = plt.subplots(figsize=(13, 7), dpi=300)
    labels = [f"{branch}\n{kind}" for branch, kind, _ in branch_rows]
    values = [value for _, _, value in branch_rows]
    bars = ax.bar(np.arange(len(values)), values, color=["#6baed6" if kind == "parent" else "#2171b5" for _, kind, _ in branch_rows])
    ax.axhline(419, color="black", linestyle="--", linewidth=1.0, label="Current champion = 419")
    for bar, value in zip(bars, values, strict=True):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.4, f"{value:.0f}", ha="center", fontsize=8)
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_ylabel("Top-1 Real-B pass count")
    ax.set_title("Seed branches and their best one-layer K13 child")
    ax.set_ylim(0, 425)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "25_branch_best_comparison.png")
    plt.close(fig)

    interaction_ordered = interaction.sort_values("gain_over_best_single", ascending=False).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(18, 8), dpi=300)
    values = interaction_ordered["gain_over_best_single"].to_numpy(dtype=float)
    bar_colors = ["tab:green" if value > 0 else "tab:red" if value < 0 else "tab:gray" for value in values]
    ax.bar(np.arange(values.size), values, color=bar_colors)
    ax.axhline(0, color="black", linewidth=0.9)
    for index, row in interaction_ordered.head(1).iterrows():
        ax.annotate(f"{row.basis_a}+{row.basis_b}", (index, row.gain_over_best_single), fontsize=6, rotation=90, ha="center", xytext=(0, 4 if row.gain_over_best_single >= 0 else -8), textcoords="offset points")
    ax.set_xlabel("K13 child ordered by gain over best single add")
    ax.set_ylabel("N_ab - max(N_a, N_b)")
    ax.set_title("Pairwise interaction gain over the best single addition")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "26_pairwise_interaction_gain.png")
    plt.close(fig)

    heat = failure6.pivot(index="model_id", columns="State_R", values="model_first_shareable_rank").reindex(ordered["model_id"].tolist())
    fig, ax = plt.subplots(figsize=(13, 12), dpi=300)
    image = ax.imshow(heat.to_numpy(dtype=float), aspect="auto", interpolation="nearest", cmap="viridis", vmin=1, vmax=max(10, float(np.nanmax(heat.to_numpy()))))
    ax.set_xticks(np.arange(len(CHAMPION_FAILURES)), [str(state_id) for state_id in CHAMPION_FAILURES])
    tick_positions = np.linspace(0, heat.shape[0] - 1, min(20, heat.shape[0]), dtype=int)
    ax.set_yticks(tick_positions, [str(int(heat.index[position])) for position in tick_positions])
    ax.set_xlabel("Seed A failure State_R")
    ax.set_ylabel("Evaluation support, ordered by global rank")
    ax.set_title("Seed A failure-6 first-shareable rank across the Beam evaluation pool")
    fig.colorbar(image, ax=ax, label="first-shareable rank")
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "27_champion_failure6_recovery.png")
    plt.close(fig)

    core_model = model_frame.loc[model_frame["model_id"].eq(SEED_A_ID)]
    core_c2_median = float(core_model["Y_C2_train_NMSE_dB"].median())
    child_model = model_frame.loc[model_frame["model_type"].eq("k13")].groupby(["model_id", "model_name"], as_index=False).agg(c2_train_median=("Y_C2_train_NMSE_dB", "median"))
    child_model["delta_C2_train_median_vs_seedA"] = child_model["c2_train_median"] - core_c2_median
    child_model = child_model.merge(summary[["model_id", "top1_pass_count"]], on="model_id", how="left")
    child_model["delta_top1_vs_seedA"] = child_model["top1_pass_count"] - int(summary.loc[summary["model_id"].eq(SEED_A_ID), "top1_pass_count"].iloc[0])
    fig, ax = plt.subplots(figsize=(12, 8), dpi=300)
    ax.scatter(child_model["delta_C2_train_median_vs_seedA"], child_model["delta_top1_vs_seedA"], s=20, alpha=0.7, color="tab:purple")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    for row in child_model.loc[child_model["delta_top1_vs_seedA"].abs().nlargest(10).index].itertuples(index=False):
        ax.annotate(str(row.model_id), (row.delta_C2_train_median_vs_seedA, row.delta_top1_vs_seedA), fontsize=6, xytext=(2, 2), textcoords="offset points")
    ax.set_xlabel("Δ C2 Train median NMSE vs Seed A (dB)")
    ax.set_ylabel("Δ Top-1 Real-B pass vs Seed A")
    ax.set_title("K13 modeling contribution versus retrieval contribution")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "28_modeling_vs_retrieval_contribution.png")
    plt.close(fig)

    frequency = cv_stability.sort_values(["global_train_winner_count", "validation_top1_mean"], ascending=[False, False]).head(20).copy()
    fig, ax = plt.subplots(figsize=(16, 8), dpi=300)
    bars = ax.bar(np.arange(frequency.shape[0]), frequency["global_train_winner_count"], color="tab:blue")
    ax.set_xticks(np.arange(frequency.shape[0]), frequency["model_id"].astype(str), rotation=70, ha="right")
    for bar, value in zip(bars, frequency["global_train_winner_count"], strict=True):
        if value:
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.03, str(int(value)), ha="center", fontsize=7)
    ax.set_xlabel("Model ID (top 20 by global train-winner frequency)")
    ax.set_ylabel("Global train-winner count across 5 folds")
    ax.set_title("Query-State CV global selection frequency")
    ax.set_ylim(0, max(1, int(frequency["global_train_winner_count"].max()) + 1))
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "29_cv_selection_frequency.png")
    plt.close(fig)


def _annotate_summary(candidates: tuple[BeamCandidate, ...], summaries: list[dict[str, Any]], frames: dict[int, pd.DataFrame], references: dict[str, Any]) -> pd.DataFrame:
    summary_rows = pd.DataFrame(summaries)
    summary_rows = summary_rows.sort_values("model_id").reset_index(drop=True)
    summary_rows["strict_breakthrough"] = (summary_rows["model_type"].eq("k13")) & summary_rows["top1_pass_count"].gt(419)
    champion = frames[SEED_A_ID]
    global_comparisons = {candidate.candidate_id: _compare_frames(frames[candidate.candidate_id], champion) for candidate in candidates}
    summary_rows["delta_top1_vs_global_champion"] = summary_rows["model_id"].map(lambda model_id: global_comparisons[int(model_id)]["delta_top1"])
    summary_rows["recovered_vs_global_champion"] = summary_rows["model_id"].map(lambda model_id: global_comparisons[int(model_id)]["recovered_count"])
    summary_rows["regressed_vs_global_champion"] = summary_rows["model_id"].map(lambda model_id: global_comparisons[int(model_id)]["regressed_count"])
    summary_rows["safe_breakthrough"] = summary_rows["strict_breakthrough"] & summary_rows["regressed_vs_global_champion"].eq(0)
    summary_rows["global_rank"] = k9_utils._assign_diagnostic_ranks(summaries)["diagnostic_rank"].to_numpy(dtype=int)
    branch_names = [candidate.candidate_name for candidate in candidates[:3]]
    for branch_name in branch_names:
        branch_ids = [candidate.candidate_id for candidate in candidates if branch_name in candidate.branch_seed_ids]
        branch_summary_rows = [summaries[model_id] for model_id in branch_ids]
        ordered = sorted(branch_summary_rows, key=k9_utils._selection_key)
        rank_by_id = {int(row["candidate_id"]): rank for rank, row in enumerate(ordered, start=1)}
        summary_rows[f"{branch_name}_branch_rank"] = summary_rows["model_id"].map(lambda model_id: rank_by_id.get(int(model_id), np.nan))
    # Keep the requested compact branch aliases as well as the explicit seed names.
    summary_rows["seedA_branch_rank"] = summary_rows["seed_A_global_champion_branch_rank"]
    summary_rows["seedB_branch_rank"] = summary_rows["seed_B_margin_branch_branch_rank"]
    summary_rows["seedC_branch_rank"] = summary_rows["seed_C_structural_branch_branch_rank"]
    summary_rows["distance_file"] = summary_rows.apply(lambda row: "seed_A.npy" if int(row.model_id) == 0 else "seed_B.npy" if int(row.model_id) == 1 else "seed_C.npy" if int(row.model_id) == 2 else f"k13_{int(row.model_id):03d}.npy", axis=1)
    summary_rows["global_top1_rank"] = summary_rows["global_rank"]
    return summary_rows


def _build_branch_summary(candidates: tuple[BeamCandidate, ...], summary: pd.DataFrame, edges: pd.DataFrame, cv_stability: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for seed in candidates[:3]:
        branch_children = [candidate for candidate in candidates[3:] if seed.candidate_name in candidate.branch_seed_ids]
        branch_summary = summary.loc[summary["model_id"].isin([candidate.candidate_id for candidate in branch_children])].sort_values("global_rank")
        best = branch_summary.iloc[0]
        branch_edges = edges.loc[edges["parent_seed_id"].eq(seed.candidate_name)]
        best_edge = branch_edges.loc[branch_edges["child_model_id"].eq(int(best["model_id"]))].iloc[0]
        best_stability = cv_stability.loc[cv_stability["model_id"].eq(int(best["model_id"]))].iloc[0]
        rows.append({
            "seed_name": seed.candidate_name,
            "seed_model_id": seed.candidate_id,
            "seed_extra_basis": seed.extension_basis_ids[0],
            "parent_top1": int(summary.loc[summary["model_id"].eq(seed.candidate_id), "top1_pass_count"].iloc[0]),
            "best_child_model_id": int(best["model_id"]),
            "best_child_model_name": best["model_name"],
            "best_child_extension_basis_1": best["extension_basis_1"],
            "best_child_extension_basis_2": best["extension_basis_2"],
            "best_child_top1": int(best["top1_pass_count"]),
            "best_child_global_rank": int(best["global_rank"]),
            "best_child_delta_vs_parent": int(best_edge["delta_top1_vs_parent"]),
            "best_child_recovered_vs_parent": int(best_edge["recovered_vs_parent"]),
            "best_child_regressed_vs_parent": int(best_edge["regressed_vs_parent"]),
            "best_child_delta_vs_global_champion": int(best["delta_top1_vs_global_champion"]),
            "best_child_recovered_vs_global_champion": int(best["recovered_vs_global_champion"]),
            "best_child_regressed_vs_global_champion": int(best["regressed_vs_global_champion"]),
            "best_child_Q05": float(best["share_margin_Q05"]),
            "best_child_MRR": float(best["MRR"]),
            "best_child_Top2": int(best["Top2_oracle"]),
            "best_child_Top3": int(best["Top3_oracle"]),
            "best_child_validation_top1_mean": float(best_stability["validation_top1_mean"]),
            "best_child_validation_top1_min": int(best_stability["validation_top1_min"]),
            "best_child_validation_top1_max": int(best_stability["validation_top1_max"]),
        })
    return pd.DataFrame(rows)


def _safe_model_stem(candidate: BeamCandidate) -> str:
    return "seed_A" if candidate.candidate_id == 0 else "seed_B" if candidate.candidate_id == 1 else "seed_C" if candidate.candidate_id == 2 else f"k13_{candidate.candidate_id:03d}"


def _run(*, resume: bool = False) -> dict[str, Any]:
    if RESULT_ROOT.exists() and any(RESULT_ROOT.iterdir()) and not resume:
        raise RuntimeError(f"result directory is not empty; use --resume: {RESULT_ROOT}")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "distance_matrices").mkdir(parents=True, exist_ok=True)
    (RESULT_ROOT / "fingerprints").mkdir(parents=True, exist_ok=True)
    raw_before = raw_manifest_gate()
    terms = tuple(build_envelope_dictionary())
    envelope_gate = dictionary_gate()
    core_ids, by_id, core_index, formal_core_frame = _load_frozen_core(terms)
    references = _load_references(core_ids)
    common_b, common_meta = verify_common_b_contract()
    common_b = np.asarray(common_b, dtype=np.complex128)
    if str(common_meta["sha256"]) != EXPECTED_COMMON_B_SHA:
        raise RuntimeError("common-B hash changed")
    real_b_random_error = _random_real_b_check(references["real_b_distance"])
    seed_frame, unique_frame, candidates, dedup_audit = _build_candidates(core_ids, by_id)
    if len(candidates) != 189 or sum(candidate.model_type == "k13" for candidate in candidates) != 186:
        raise RuntimeError("Beam evaluation pool count changed")
    _write_seed_support_outputs(core_ids, by_id, seed_frame)
    unique_frame.to_csv(RESULT_ROOT / "07_unique_k13_supports.csv", index=False)
    edges = pd.read_csv(RESULT_ROOT / "06_raw_expansion_edges.csv") if (RESULT_ROOT / "06_raw_expansion_edges.csv").is_file() else pd.DataFrame()
    # The raw edge table is created here so it remains the exact pre-fit graph.
    # Reconstruct it from the deterministic candidate/parent relation if a prior
    # resume stopped before the first write.
    if edges.empty or edges.shape[0] != 189:
        rows: list[dict[str, Any]] = []
        edge_id = 0
        candidate_by_hash = {candidate.support_hash: candidate for candidate in candidates[3:]}
        for seed in candidates[:3]:
            for term in terms:
                if term.basis_id in core_ids or term.basis_id == seed.extension_basis_ids[0]:
                    continue
                extension_ids = tuple(sorted((seed.extension_basis_ids[0], term.basis_id), key=lambda basis_id: by_id[basis_id].index))
                child_hash = _support_hash(core_ids + extension_ids)
                child = candidate_by_hash[child_hash]
                parent_names = child.branch_seed_ids
                occurrence_index = parent_names.index(seed.candidate_name)
                rows.append({"edge_id": edge_id, "parent_seed_id": seed.candidate_name, "parent_support_hash": seed.support_hash, "added_basis_id": term.basis_id, "added_p": int(term.order), "added_m": term.signal_delay, "added_q": term.envelope_delay, "added_is_cross_delay": bool(term.family == "CROSS_ENVELOPE"), "child_support_hash": child_hash, "child_model_id": child.candidate_id, "is_duplicate_child": occurrence_index > 0, "child_generation_count": len(parent_names)})
                edge_id += 1
        edges = pd.DataFrame(rows)
    edges.to_csv(RESULT_ROOT / "06_raw_expansion_edges.csv", index=False)
    dedup_audit.to_csv(RESULT_ROOT / "08_multi_parent_children.csv", index=False)
    dedup_payload = {
        "raw_expansion_edges": int(edges.shape[0]),
        "unique_k13_children": int(unique_frame.shape[0]),
        "evaluation_pool": int(len(candidates)),
        "duplicate_occurrence_count": int(edges["is_duplicate_child"].sum()),
        "multi_parent_child_count": int((unique_frame["generation_count"] > 1).sum()),
        "multi_parent_children": dedup_audit.to_dict("records"),
        "expected_raw_edges": 189,
        "expected_unique_children": 186,
        "pass": bool(edges.shape[0] == 189 and unique_frame.shape[0] == 186 and dedup_audit.shape[0] == 3),
    }
    _write_json(RESULT_ROOT / "08_support_dedup_audit.json", dedup_payload)
    (RESULT_ROOT / "00_task_definition.txt").write_text("\n".join([
        f"Task: {TASK_NAME}",
        "Mode: Beam=3 one-layer retrieval-oriented forward expansion.",
        "Seeds: seed_A_global_champion, seed_B_margin_branch, seed_C_structural_branch.",
        "Each seed expands by one unselected Envelope75 basis; duplicate K13 supports are removed before fitting.",
        "Raw expansion edges=189; unique K13 children=186; evaluation pool=3 seeds + 186 children=189.",
        f"Frozen protocol: dmax={backend.MP_DMAX}, Ridge lambda={backend.RIDGE_LAMBDA}, ABC valid={backend.ABC_LENGTHS}, common-B SHA={EXPECTED_COMMON_B_SHA}, Real-B threshold={REAL_B_THRESHOLD_DB} dB, allow-self=True.",
        "Primary ranking: Top-1 Real-B pass, non-self pass rate, Q05 shareability margin, MRR, Top-3 oracle, then K/model ID.",
        "No second beam layer, K14, backward elimination, swap, multi-basis simultaneous addition, parameter scan, Top-k reranking, ensemble, clustering, Type III, low-bandwidth, DPD replay or nested-CV final evaluation.",
    ]) + "\n", encoding="utf-8")
    _write_json(RESULT_ROOT / "01_reuse_audit.json", {
        "task_name": TASK_NAME,
        "formal_k11_core_source": str(K11_ROOT / "04_candidate_supports.csv"),
        "core_basis_ids": list(core_ids),
        "core_support_hash": _support_hash(core_ids),
        "formal_k12_seed_source": str(K12_ROOT / "06_candidate_supports.csv"),
        "seed_formal_model_ids": references["formal_seed_ids"],
        "envelope75_gate": envelope_gate,
        "historical_mp10_backend": "behavior_modeling.sparse_gmp",
        "separate_hnorm_found": False,
        "real_B_source": str(K9_ROOT / "04_realB_ground_truth_cnmse_matrix.npy"),
        "cv_source": str(K12_ROOT / "16_query_state_cv_folds.csv"),
        "common_B_sha256": EXPECTED_COMMON_B_SHA,
        "real_B_random_pair_max_abs_error_dB": real_b_random_error,
        "raw_expansion_edges": 189,
        "unique_k13_children": 186,
        "evaluation_pool": 189,
        "prohibited_second_layer": True,
    })
    (RESULT_ROOT / "01_reuse_audit.txt").write_text("\n".join([
        f"Task: {TASK_NAME}",
        f"Frozen common K11 core read from formal metadata: {list(core_ids)}; hash={_support_hash(core_ids)}.",
        "The three seeds were read from the formal K12 support/summary/coefficients and are not selected from this run.",
        "Seed A=ENV_p04_m0_q1; Seed B=ENV_p03_m0_q1; Seed C=ENV_p09_m2_q0.",
        "K13 support construction uses core + two extension basis IDs sorted by canonical Envelope75 order.",
        "Duplicate supports are removed before Ridge fitting using the canonical support SHA256.",
        "Historical augmented complex Ridge, dmax=2, lambda=1e-8, Aend last ILC column and C2 ILC2 are reused.",
        f"Real-B random-pair max error={real_b_random_error:.3e} dB; common-B SHA={EXPECTED_COMMON_B_SHA}.",
        "No second beam layer, K14, backward, swap, parameter scan, reranking, ensemble, clustering, low-bandwidth or DPD replay was run.",
    ]) + "\n", encoding="utf-8")
    _checkpoint(phase="reuse_audit_graph_seed_supports_passed", raw_manifest_before=raw_before, raw_expansion_edges=189, unique_k13_children=186, evaluation_pool=189, real_B_random_pair_max_abs_error_dB=real_b_random_error, multi_parent_child_count=3)

    model_frame, theta_a, theta_c = _run_model_phase(candidates, core_index, resume)
    _write_coefficient_artifacts(candidates, theta_a, theta_c)
    common_mp = backend.historical_mp10.build_frozen_mp_basis(common_b)
    common_env = build_envelope_bank(common_b, terms)
    common_core = np.column_stack((common_mp, common_env[:, core_index])).astype(np.complex128)
    if common_core.shape != (backend.FINGERPRINT_LENGTH, CORE_K):
        raise RuntimeError(f"common core Phi shape changed: {common_core.shape}")
    frames: dict[int, pd.DataFrame] = {}
    summaries: list[dict[str, Any]] = []
    distances: dict[int, np.ndarray] = {}
    fingerprints: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    seed_regressions: list[dict[str, Any]] = []
    # Seed retrieval and regression gate are completed before any K13 child is evaluated.
    for candidate in candidates[:3]:
        phi = _candidate_phi(common_core, common_env, candidate)
        frame, summary, distance, lut, query = _candidate_retrieval(candidate, theta_a, theta_c, phi, references["real_b_distance"], references["shareability"])
        frames[candidate.candidate_id] = frame
        summaries.append(summary)
        distances[candidate.candidate_id] = distance
        fingerprints[candidate.candidate_id] = (lut, query)
        np.save(RESULT_ROOT / "distance_matrices" / f"seed_{'A' if candidate.candidate_id == 0 else 'B' if candidate.candidate_id == 1 else 'C'}.npy", distance)
        _save_fingerprint(f"seed_{'A' if candidate.candidate_id == 0 else 'B' if candidate.candidate_id == 1 else 'C'}", lut, query)
        seed_regressions.append(_seed_regression(candidate, model_frame, frame, theta_a, theta_c, distance, lut, query, references, phi))
    expected_seed_values = {
        "seed_A_global_champion": {"Top1": 419, "Q05": 0.3796019374662876, "MRR": 0.9917086834733894, "Top2": 423, "Top3": 424},
        "seed_B_margin_branch": {"Top1": 418, "Q05": 0.413804, "MRR": 0.990924, "Top2": 424, "Top3": 424},
        "seed_C_structural_branch": {"Top1": 418, "Q05": 0.279299, "MRR": 0.990336, "Top2": 423, "Top3": 423},
    }
    for item in seed_regressions:
        expected = expected_seed_values[item["seed_name"]]
        for key, value in expected.items():
            actual = item[key] if key in item else item["Q05"] if key == "Q05" else item[key]
            if abs(float(actual) - float(value)) > 1e-5:
                raise RuntimeError(f"formal seed metric changed: {item['seed_name']} {key}={actual}, expected={value}")
    _write_json(RESULT_ROOT / "04_seed_baseline_regression.json", {"pass": all(item["pass"] for item in seed_regressions), "seed_regressions": seed_regressions, "known_seed_metric_gate": expected_seed_values})
    seed_profiles: list[pd.DataFrame] = []
    for candidate in candidates[:3]:
        profile = frames[candidate.candidate_id][["State_R", "State_Q", "exact_hit", "realB_pass", "retrieved_realB_CNMSE_dB", "fingerprint_top1_CNMSE_dB", "fingerprint_top2_CNMSE_dB", "shareability_margin_dB", "first_shareable_rank", "Top2_has_shareable", "Top3_has_shareable", "Top5_has_shareable", "Top10_has_shareable"]].copy()
        profile.insert(0, "seed_id", candidate.candidate_id)
        profile.insert(1, "seed_name", candidate.candidate_name)
        seed_profiles.append(profile)
    pd.concat(seed_profiles, ignore_index=True).to_csv(RESULT_ROOT / "05_seed_failure_profiles.csv", index=False)
    _checkpoint(phase="seed_regressions_passed_before_k13_retrieval", seed_regressions=seed_regressions)

    for candidate in candidates[3:]:
        phi = _candidate_phi(common_core, common_env, candidate)
        frame, summary, distance, lut, query = _candidate_retrieval(candidate, theta_a, theta_c, phi, references["real_b_distance"], references["shareability"])
        frames[candidate.candidate_id] = frame
        summaries.append(summary)
        distances[candidate.candidate_id] = distance
        fingerprints[candidate.candidate_id] = (lut, query)
        np.save(RESULT_ROOT / "distance_matrices" / f"k13_{candidate.candidate_id:03d}.npy", distance)
        if candidate.candidate_id % 20 == 0 or candidate.candidate_id == candidates[-1].candidate_id:
            _checkpoint(phase="k13_retrieval_progress", completed_k13_children=candidate.candidate_id - 2, total_k13_children=186)
            print(f"[SMALLBEAM K13 RETRIEVAL] {candidate.candidate_id - 2}/186", flush=True)
    summary = _annotate_summary(candidates, summaries, frames, references)
    summary["model_type"] = summary["model_type"].astype(str)
    summary["strict_breakthrough"] = summary["strict_breakthrough"].astype(bool)
    summary["safe_breakthrough"] = summary["safe_breakthrough"].astype(bool)
    query_frame = pd.concat(frames.values(), ignore_index=True).sort_values(["model_id", "State_R"]).reset_index(drop=True)
    query_frame.to_csv(RESULT_ROOT / "13_query_metrics_long.csv", index=False)
    summary.to_csv(RESULT_ROOT / "12_retrieval_summary.csv", index=False)
    edge_delta = _build_edge_deltas(edges, candidates, frames, summary)
    edge_delta.to_csv(RESULT_ROOT / "14_expansion_edge_delta.csv", index=False)
    global_comparison, _ = _build_global_comparisons(candidates, frames, summary)
    global_comparison.to_csv(RESULT_ROOT / "17_recovered_regressed_vs_global_champion.csv", index=False)
    parent_comparison = edge_delta.copy()
    parent_comparison.to_csv(RESULT_ROOT / "18_recovered_regressed_vs_parent.csv", index=False)
    failure6 = _build_champion_failure6(candidates, frames)
    failure6.to_csv(RESULT_ROOT / "15_global_champion_failure6_analysis.csv", index=False)
    interaction = _build_pairwise_interaction(candidates[3:], frames, summary, references)
    interaction.to_csv(RESULT_ROOT / "16_pairwise_interaction_summary.csv", index=False)
    cv_frame, cv_stability = _local_cv_results(candidates, frames, references["folds"])
    cv_frame.to_csv(RESULT_ROOT / "19_query_state_cv_results.csv", index=False)
    cv_stability.to_csv(RESULT_ROOT / "20_query_state_cv_selection_stability.csv", index=False)
    pair_type_summary, basis_structure = _build_structure_summaries(candidates[3:], by_id, summary, interaction)
    pair_type_summary.to_csv(RESULT_ROOT / "21_extension_pair_type_summary.csv", index=False)
    basis_structure.to_csv(RESULT_ROOT / "22_basis_structure_summary.csv", index=False)
    branch_summary = _build_branch_summary(candidates, summary, edge_delta, cv_stability)
    branch_summary.to_csv(RESULT_ROOT / "23_branch_summary.csv", index=False)
    global_best = summary.sort_values("global_rank").iloc[0]
    k13_summary = summary.loc[summary["model_type"].eq("k13")].sort_values("global_rank").reset_index(drop=True)
    global_best_k13 = k13_summary.iloc[0]
    best_branch_ids = {row.seed_name: int(row.best_child_model_id) for row in branch_summary.itertuples(index=False)}
    for candidate_id, (lut, query) in fingerprints.items():
        if candidate_id == int(global_best_k13["model_id"]):
            _save_fingerprint("global_best_k13", lut, query)
        for branch_name, branch_id in best_branch_ids.items():
            if candidate_id == branch_id:
                short = "A" if branch_name == "seed_A_global_champion" else "B" if branch_name == "seed_B_margin_branch" else "C"
                _save_fingerprint(f"per_branch_best_{short}", lut, query)
    # Seed fingerprints already exist; the detail file is the globally best support from all 189.
    frames[int(global_best["model_id"])].to_csv(RESULT_ROOT / "30_global_best_detail.csv", index=False)
    _write_figures(candidates, summary, interaction, failure6, model_frame, cv_stability, branch_summary)
    raw_after = raw_manifest_gate()
    if raw_before != raw_after:
        raise RuntimeError("data/raw manifest changed")
    strict_breakthrough_count = int(k13_summary["strict_breakthrough"].sum())
    safe_breakthrough_count = int(k13_summary["safe_breakthrough"].sum())
    top_interactions = interaction.sort_values(["gain_over_best_single", "interaction_excess"], ascending=False).head(10)
    summary_lines = [
        f"Task: {TASK_NAME}",
        "Beam=3 one-layer expansion completed.",
        "Seeds=3; expansions per seed=63; raw expansion edges=189; unique K13 children=186; evaluation pool=189.",
        f"Seed regression PASS={all(item['pass'] for item in seed_regressions)}; seed A failure set={frames[0].loc[~frames[0]['realB_pass'], 'State_R'].astype(int).tolist()}.",
        f"Global benchmark Seed A: Top1=419/425; Exact={int(frames[0]['exact_hit'].sum())}; Q05={float(summary.loc[summary['model_id'].eq(0), 'share_margin_Q05'].iloc[0]):.9g}; MRR={float(summary.loc[summary['model_id'].eq(0), 'MRR'].iloc[0]):.9g}; Top2={int(summary.loc[summary['model_id'].eq(0), 'Top2_oracle'].iloc[0])}; Top3={int(summary.loc[summary['model_id'].eq(0), 'Top3_oracle'].iloc[0])}.",
        f"Global best support={global_best['model_name']}; Top1={int(global_best['top1_pass_count'])}/425; global rank={int(global_best['global_rank'])}.",
        f"Best K13={global_best_k13['model_name']}; Top1={int(global_best_k13['top1_pass_count'])}/425; strict_breakthrough_count={strict_breakthrough_count}; safe_breakthrough_count={safe_breakthrough_count}.",
        "",
        "Global top 15 K13 children:",
    ]
    for row in k13_summary.head(15).itertuples(index=False):
        summary_lines.append(f"rank={int(row.global_rank)} {row.model_name}: branch={row.branch_seed_ids}, pair={row.extension_basis_1}+{row.extension_basis_2}, Top1={int(row.top1_pass_count)}/425, delta419={int(row.top1_pass_count)-419}, rec={int(row.recovered_vs_global_champion)}, reg={int(row.regressed_vs_global_champion)}, Exact={int(row.exact_hit_count)}, nonself={float(row.nonself_pass_rate):.9g}, Q05={float(row.share_margin_Q05):.9g}, MRR={float(row.MRR):.9g}, Top2={int(row.Top2_oracle)}, Top3={int(row.Top3_oracle)}")
    summary_lines.extend(["", "Best child per seed branch:"])
    for row in branch_summary.itertuples(index=False):
        summary_lines.append(f"{row.seed_name}: child={row.best_child_model_name}, pair={row.best_child_extension_basis_1}+{row.best_child_extension_basis_2}, parent Top1={int(row.parent_top1)}, child Top1={int(row.best_child_top1)}, delta_parent={int(row.best_child_delta_vs_parent)}, recovered_parent={int(row.best_child_recovered_vs_parent)}, regressed_parent={int(row.best_child_regressed_vs_parent)}, validation Top1 mean/min/max={float(row.best_child_validation_top1_mean):.6g}/{int(row.best_child_validation_top1_min)}/{int(row.best_child_validation_top1_max)}")
    summary_lines.extend(["", "Strongest pairwise interactions by gain over best single:"])
    for row in top_interactions.itertuples(index=False):
        summary_lines.append(f"{row.basis_a}+{row.basis_b}: N_a={int(row.N_a)}, N_b={int(row.N_b)}, N_ab={int(row.N_ab)}, gain_over_best={int(row.gain_over_best_single)}, interaction_excess={int(row.interaction_excess)}")
    summary_lines.extend(["", "Cross+cross / cross+aligned summary:"])
    for row in pair_type_summary.itertuples(index=False):
        summary_lines.append(f"{row.pair_type}: count={int(row.candidate_count)}, Top1 mean/max={float(row.Top1_mean):.6g}/{int(row.Top1_max)}, strict={int(row.strict_breakthrough_count)}, safe={int(row.safe_breakthrough_count)}, gain_over_best mean/max={float(row.mean_gain_over_best_single):.6g}/{float(row.max_gain_over_best_single):.6g}")
    summary_lines.extend(["", "Seed A failure-6 global-best-K13 comparison:"])
    for state_id in CHAMPION_FAILURES:
        base = frames[0].set_index("State_R").loc[state_id]
        child = frames[int(global_best_k13["model_id"])].set_index("State_R").loc[state_id]
        summary_lines.append(f"State {state_id}: SeedA rank={int(base['first_shareable_rank'])}, Q={int(base['State_Q'])}; K13 rank={int(child['first_shareable_rank'])}, Q={int(child['State_Q'])}; recovered={bool(not base['realB_pass'] and child['realB_pass'])}; K13 Top1 Real-B={float(child['retrieved_realB_CNMSE_dB']):.9g} dB")
    summary_lines.extend(["", "Five-fold Query-State CV global winners:"])
    for row in cv_frame.loc[cv_frame["train_is_global_winner"]].sort_values("fold").itertuples(index=False):
        summary_lines.append(f"fold={int(row.fold)} winner={row.model_name}, validation Top1={int(row.validation_top1)}/85, validation nonself={float(row.validation_nonself_rate):.9g}, Q05={float(row.validation_Q05):.9g}, MRR={float(row.validation_MRR):.9g}")
    summary_lines.append(f"CV global winner frequency top: {cv_stability.loc[cv_stability['global_train_winner_count'].gt(0), ['model_name', 'global_train_winner_count']].to_dict('records')}")
    summary_lines.extend(["", f"Raw before={json.dumps(raw_before, ensure_ascii=False, sort_keys=True)}", f"Raw after={json.dumps(raw_after, ensure_ascii=False, sort_keys=True)}", f"Raw unchanged={raw_before == raw_after}", "No second beam layer, K14, backward, swap, multi-basis simultaneous addition, parameter scan, reranking, ensemble, clustering, Type III, low-bandwidth or DPD replay was run.", "CV is Query-State support-selection stability analysis, not an independent final test."])
    (RESULT_ROOT / "31_final_result_summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")
    transition_counts = global_comparison[["recovered_count", "regressed_count", "unchanged_pass", "unchanged_fail"]].sum().to_dict()
    result = {
        "task": TASK_NAME,
        "status": "SUCCESS",
        "seed_regression_pass": bool(all(item["pass"] for item in seed_regressions)),
        "raw_expansion_edges": 189,
        "unique_k13_children": 186,
        "evaluation_pool": 189,
        "multi_parent_child_count": int(dedup_audit.shape[0]),
        "global_best_model": str(global_best["model_name"]),
        "global_best_model_type": str(global_best["model_type"]),
        "global_best_top1": int(global_best["top1_pass_count"]),
        "global_best_k13_model": str(global_best_k13["model_name"]),
        "global_best_k13_top1": int(global_best_k13["top1_pass_count"]),
        "strict_breakthrough_count": strict_breakthrough_count,
        "safe_breakthrough_count": safe_breakthrough_count,
        "branch_best_models": best_branch_ids,
        "cv_global_winner_counts": cv_stability.loc[cv_stability["global_train_winner_count"].gt(0), ["model_name", "global_train_winner_count"]].to_dict("records"),
        "global_child_transition_totals": {key: int(value) for key, value in transition_counts.items()},
        "raw_data_modified": raw_before != raw_after,
        "result_root": str(RESULT_ROOT),
    }
    _write_json(RESULT_ROOT / "32_checkpoint.json", {"phase": "completed", **result, "independent_validation_pending": True})
    _append_log(f"\n[{_now()}] Complete {TASK_NAME}\nResult={json.dumps(result, ensure_ascii=False, sort_keys=True)}\nNo second beam layer or K14 was run.\n")
    _append_handoff(f"完成 Beam=3 one-layer retrieval-oriented K13 expansion；结果目录：{RESULT_ROOT}\n摘要：{json.dumps(result, ensure_ascii=False, sort_keys=True)}\n完成 3 seed regression、189 raw edges/186 unique children、Full425 retrieval、pairwise interaction、failure6、branch comparison 和 5-fold Query-State stability；未启动第二层 beam。")
    return result


def main() -> None:
    result = _run(resume="--resume" in sys.argv[1:])
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    __import__("multiprocessing").freeze_support()
    main()
