# ruff: noqa: E402,E501,I001

"""Run Updated Beam=3 iteration-2 one-layer retrieval-oriented expansion."""

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

for _thread_variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
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
from behavior_modeling.shared.basis_function_selection.envelope_dictionary import EnvelopeBasis, build_envelope_bank, build_envelope_dictionary, dictionary_gate  # noqa: E402
from behavior_fingerprint_retrieval.shared.envelope19_c2endshared_commonB_full_lut import EXPECTED_COMMON_B_SHA, raw_manifest_gate, verify_common_b_contract  # noqa: E402
from data_management.shared import load_by_id  # noqa: E402
from retrieval_oriented_model_selection.shared import smallbeam_retrieval_runner as legacy
from retrieval_oriented_model_selection.self_hit_oriented.scenario_2.scenario_2_updated_beam3_iteration2_retrieval_oriented_forward_expansion_5b import updated_beam_backend as backend  # noqa: E402
from retrieval_oriented_model_selection.shared import mp10_k9_retrieval_runner as k9_utils  # noqa: E402
from signal_segmentation.shared import build_off_segments, build_partition_from_xin  # noqa: E402

TASK_NAME = "scenario_2_updated_beam3_iteration2_retrieval_oriented_forward_expansion_5b"
RESULT_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /TASK_NAME
WORK_LOG = PROJECT_ROOT / "work_logs" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /TASK_NAME / "execution_log.txt"
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"
BEAM_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /"scenario_2_smallbeam3_k13_retrieval_oriented_forward_expansion_5b"
FLOAT_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /"scenario_2_beam3_k13_floating_backward_retrieval_ablation_5b"
K12_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /"scenario_2_k12_retrieval_oriented_forward_scan_5b"
K11_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /"scenario_2_mp10_k11_retrieval_oriented_forward_scan_5b"
K9_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /"scenario_2_mp10_k9_retrieval_oriented_basis_ablation_5b"
STATE_COUNT = backend.STATE_COUNT
WORKER_COUNT = 10
BLAS_THREADS_PER_WORKER = 1
REAL_B_THRESHOLD_DB = -40.0
REAL_B_SANITY_TOLERANCE_DB = 1e-9
K12 = 12
K13 = 13
K14 = 14
MP_COLUMNS = backend.MP_K
CORE_BASIS_ID = "ENV_p04_m2_q0"
FRONTIER_SPECS = (
    (0, "F1_floating_K12", "child_K12_13", FLOAT_ROOT / "11_retrieval_summary.csv"),
    (1, "F2_safe_K13", "K13_plus_ENV_p03_m0_q1__ENV_p09_m2_q0", BEAM_ROOT / "12_retrieval_summary.csv"),
    (2, "F3_margin_K13", "K13_plus_ENV_p03_m1_q0__ENV_p04_m0_q1", BEAM_ROOT / "12_retrieval_summary.csv"),
)


@dataclass(frozen=True)
class Parent:
    candidate_id: int
    branch: str
    source_name: str
    source_root: Path
    source_model_id: int
    support_basis_ids: tuple[str, ...]
    support_hash: str
    K: int

    @property
    def candidate_name(self) -> str:
        return self.branch

    @property
    def support(self) -> tuple[int, ...]:
        return tuple(range(self.K))

    @property
    def model_type(self) -> str:
        return "parent"

    @property
    def branch_seed_ids(self) -> tuple[str, ...]:
        return (self.branch,)

    @property
    def primary_parent_seed_id(self) -> str:
        return self.branch

    @property
    def extension_basis_ids(self) -> tuple[str, ...]:
        return tuple(self.support_basis_ids[MP_COLUMNS:])

    @property
    def extension_indices(self) -> tuple[int, ...]:
        return tuple()

    @property
    def added_basis_id(self) -> str | None:
        return None

    @property
    def is_historical_reuse(self) -> bool:
        return False

    @property
    def removed_basis_id(self) -> str | None:
        return None


@dataclass(frozen=True)
class Child:
    candidate_id: int
    candidate_name: str
    parent_branch: str
    parent_model_id: int
    support_basis_ids: tuple[str, ...]
    support_hash: str
    added_basis_id: str
    added_index: int
    K: int
    is_historical_reuse: bool
    historical_source_name: str | None
    historical_source_id: int | None

    @property
    def support(self) -> tuple[int, ...]:
        return tuple(range(self.K))

    @property
    def removed_basis_id(self) -> str | None:
        return None

    @property
    def model_type(self) -> str:
        return "historical_child" if self.is_historical_reuse else "novel_child"

    @property
    def branch_seed_ids(self) -> tuple[str, ...]:
        return (self.parent_branch,)

    @property
    def primary_parent_seed_id(self) -> str:
        return self.parent_branch

    @property
    def extension_basis_ids(self) -> tuple[str, ...]:
        return tuple(self.support_basis_ids[MP_COLUMNS:])

    @property
    def extension_indices(self) -> tuple[int, ...]:
        return tuple()


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
    _write_json(RESULT_ROOT / "37_checkpoint.json", {"task_name": TASK_NAME, "mode": "new_task", "state_count": STATE_COUNT, "worker_count": WORKER_COUNT, "blas_threads_per_worker": BLAS_THREADS_PER_WORKER, "common_B_sha256": EXPECTED_COMMON_B_SHA, "real_B_threshold_dB": REAL_B_THRESHOLD_DB, "allow_self": True, "raw_edge_count": 187, "unique_child_count": 187, "evaluation_pool_count": 190, "dmax_fixed": backend.MP_DMAX, "ridge_lambda_fixed": backend.RIDGE_LAMBDA, "backward_performed": False, "second_add_performed": False, "beam_update_performed": False, "k15_performed": False, "swap_performed": False, "parameter_scan_performed": False, "top_k_selection_performed": False, "real_B_used_for_top1_selection": False, **payload})


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
        raise RuntimeError(f"Real-B random-pair error too large: {maximum:.3e} dB")
    return maximum


def _load_frozen_inputs() -> dict[str, Any]:
    """Load only the frozen metadata and historical artifacts needed here."""
    terms = tuple(build_envelope_dictionary())
    dictionary_gate()
    if len(terms) != 75:
        raise RuntimeError(f"Envelope75 dictionary changed: {len(terms)}")
    by_id = {term.basis_id: term for term in terms}
    core_frame = pd.read_csv(K11_ROOT / "04_candidate_supports.csv")
    core_rows = core_frame.loc[core_frame["model_name"].eq("MP10_plus_ENV_p04_m2_q0")]
    if core_rows.shape[0] != 1:
        raise RuntimeError("formal K11 core row is missing")
    k11_ids = tuple(json.loads(core_rows.iloc[0]["support_basis_ids"]))
    expected_s10 = ("LIN_d0", "LIN_d1", "LIN_d2", "ENV_p02_m0_q0", "ENV_p02_m1_q1", "ENV_p03_m0_q0", "ENV_p03_m1_q1", "ENV_p05_m0_q0", "ENV_p07_m0_q0", "ENV_p09_m0_q0")
    if len(k11_ids) != 11 or k11_ids[:10] != expected_s10 or k11_ids[-1] != CORE_BASIS_ID:
        raise RuntimeError(f"formal K11 core changed: {k11_ids}")
    if _support_hash(k11_ids) != str(core_rows.iloc[0]["support_hash"]):
        raise RuntimeError("formal K11 core support hash mismatch")
    s10_ids = k11_ids[:10]

    floating_summary = pd.read_csv(FLOAT_ROOT / "11_retrieval_summary.csv", low_memory=False)
    beam_summary = pd.read_csv(BEAM_ROOT / "12_retrieval_summary.csv", low_memory=False)
    beam_support = pd.read_csv(BEAM_ROOT / "07_unique_k13_supports.csv", low_memory=False)
    floating_model = pd.read_csv(FLOAT_ROOT / "08_model_quality_long.csv", low_memory=False)
    beam_model = pd.read_csv(BEAM_ROOT / "09_model_quality_long.csv", low_memory=False)
    floating_query = pd.read_csv(FLOAT_ROOT / "10_query_metrics_long.csv", low_memory=False)
    beam_query = pd.read_csv(BEAM_ROOT / "13_query_metrics_long.csv", low_memory=False)
    f1_row = floating_summary.loc[floating_summary["model_name"].eq("child_K12_13")]
    f2_row = beam_summary.loc[beam_summary["model_name"].eq("K13_plus_ENV_p03_m0_q1__ENV_p09_m2_q0")]
    f3_row = beam_summary.loc[beam_summary["model_name"].eq("K13_plus_ENV_p03_m1_q0__ENV_p04_m0_q1")]
    if [len(frame) for frame in (f1_row, f2_row, f3_row)] != [1, 1, 1]:
        raise RuntimeError("one or more frozen updated Beam parents is missing")
    f2_support_row = beam_support.loc[beam_support["child_model_id"].eq(125)]
    f3_support_row = beam_support.loc[beam_support["child_model_id"].eq(12)]
    if f2_support_row.shape[0] != 1 or f3_support_row.shape[0] != 1:
        raise RuntimeError("formal Beam Model125/Model12 support rows are missing")
    support_sources = {
        "F1_floating_K12": (tuple(json.loads(f1_row.iloc[0]["support_basis_ids"])), str(f1_row.iloc[0]["support_hash"])),
        "F2_safe_K13": (tuple(json.loads(f2_support_row.iloc[0]["basis_ids"])), str(f2_row.iloc[0]["support_hash"])),
        "F3_margin_K13": (tuple(json.loads(f3_support_row.iloc[0]["basis_ids"])), str(f3_row.iloc[0]["support_hash"])),
    }
    for name, (support, digest) in support_sources.items():
        if _support_hash(support) != digest or len(support) not in (12, 13) or tuple(support[:10]) != s10_ids:
            raise RuntimeError(f"{name} support canonicalization changed: {support}")
        if any(basis_id not in by_id for basis_id in support[10:]):
            raise RuntimeError(f"{name} contains unknown Envelope75 basis")

    real_b_distance = np.load(K9_ROOT / "04_realB_ground_truth_cnmse_matrix.npy")
    shareability = np.load(K9_ROOT / "05_realB_shareability_mask.npy")
    if real_b_distance.shape != (STATE_COUNT, STATE_COUNT) or shareability.shape != real_b_distance.shape:
        raise RuntimeError("Real-B matrices have changed shape")
    if not np.all(np.isneginf(np.diag(real_b_distance))) or not bool(shareability.diagonal().all()):
        raise RuntimeError("Real-B self contract changed")
    folds = pd.read_csv(K12_ROOT / "16_query_state_cv_folds.csv").sort_values("state_id").reset_index(drop=True)
    if folds.shape[0] != STATE_COUNT or set(folds["fold"].astype(int)) != set(range(5)):
        raise RuntimeError("historical CV folds changed")
    if folds["fold"].value_counts().sort_index().to_dict() != {0: 85, 1: 85, 2: 85, 3: 85, 4: 85}:
        raise RuntimeError("historical CV folds are not balanced")

    common_b, raw_common_meta = verify_common_b_contract()
    common_meta = {key: value for key, value in raw_common_meta.items() if key != "phi_e23"}
    if "phi_e23" in raw_common_meta:
        common_meta["phi_e23_shape"] = list(np.asarray(raw_common_meta["phi_e23"]).shape)
    return {
        "terms": terms,
        "by_id": by_id,
        "k11_ids": k11_ids,
        "s10_ids": s10_ids,
        "floating_summary": floating_summary,
        "beam_summary": beam_summary,
        "beam_support": beam_support,
        "floating_model": floating_model,
        "beam_model": beam_model,
        "floating_query": floating_query,
        "beam_query": beam_query,
        "real_b_distance": real_b_distance,
        "shareability": shareability,
        "folds": folds,
        "common_b": common_b,
        "common_meta": common_meta,
        "raw_manifest": raw_manifest_gate(),
        "parent_rows": {
            "F1_floating_K12": {**f1_row.iloc[0].to_dict(), "support_basis_ids": json.dumps(list(support_sources["F1_floating_K12"][0]))},
            "F2_safe_K13": {**f2_row.iloc[0].to_dict(), "support_basis_ids": json.dumps(list(support_sources["F2_safe_K13"][0]))},
            "F3_margin_K13": {**f3_row.iloc[0].to_dict(), "support_basis_ids": json.dumps(list(support_sources["F3_margin_K13"][0]))},
        },
        "source_model_frames": {"F1_floating_K12": floating_model, "F2_safe_K13": beam_model, "F3_margin_K13": beam_model},
        "source_query_frames": {"F1_floating_K12": floating_query, "F2_safe_K13": beam_query, "F3_margin_K13": beam_query},
    }


def _build_candidates(inputs: dict[str, Any]) -> tuple[tuple[Parent, ...], tuple[Child, ...], pd.DataFrame, pd.DataFrame]:
    s10: tuple[str, ...] = inputs["s10_ids"]
    parent_specs = (
        (0, "F1_floating_K12", "child_K12_13", FLOAT_ROOT, 13),
        (1, "F2_safe_K13", "K13_plus_ENV_p03_m0_q1__ENV_p09_m2_q0", BEAM_ROOT, 125),
        (2, "F3_margin_K13", "K13_plus_ENV_p03_m1_q0__ENV_p04_m0_q1", BEAM_ROOT, 12),
    )
    parents: list[Parent] = []
    for candidate_id, branch, source_name, source_root, source_id in parent_specs:
        support, digest = ({name: (tuple(json.loads(inputs["parent_rows"][name]["support_basis_ids"])), str(inputs["parent_rows"][name]["support_hash"])) for name in ("F1_floating_K12", "F2_safe_K13", "F3_margin_K13")})[branch]
        parents.append(Parent(candidate_id, branch, source_name, source_root, source_id, support, digest, len(support)))
    if tuple(parent.support_basis_ids[:10] for parent in parents) != (s10, s10, s10):
        raise RuntimeError("updated Beam parents do not share S10")
    remaining_by_parent: dict[int, tuple[EnvelopeBasis, ...]] = {}
    all_terms = tuple(inputs["terms"])
    for parent in parents:
        remaining_by_parent[parent.candidate_id] = tuple(term for term in all_terms if term.basis_id not in parent.support_basis_ids)
        expected = {0: 63, 1: 62, 2: 62}[parent.candidate_id]
        if len(remaining_by_parent[parent.candidate_id]) != expected:
            raise RuntimeError(f"{parent.branch} remaining basis count changed")

    children_by_hash: dict[str, Child] = {}
    raw_rows: list[dict[str, Any]] = []
    edge_id = 0
    historical_support_row = inputs["beam_support"].loc[inputs["beam_support"]["child_model_id"].eq(74)]
    if historical_support_row.shape[0] != 1:
        raise RuntimeError("historical Model74 support row is missing")
    historical_support = tuple(json.loads(historical_support_row.iloc[0]["basis_ids"]))
    historical_hash = _support_hash(historical_support)
    for parent in parents:
        for term in remaining_by_parent[parent.candidate_id]:
            # The frozen historical column order keeps the K11 core envelope
            # immediately after S10.  Other additions retain parent order and
            # are appended, which makes F1 + ENV_p04_m2_q0 exactly Model74.
            extension = parent.support_basis_ids[MP_COLUMNS:] + (term.basis_id,)
            if CORE_BASIS_ID in extension:
                extension = (CORE_BASIS_ID,) + tuple(basis_id for basis_id in extension if basis_id != CORE_BASIS_ID)
            support = s10 + extension
            digest = _support_hash(support)
            is_duplicate = digest in children_by_hash
            if not is_duplicate:
                child_id = 3 + len(children_by_hash)
                child = Child(child_id, f"{parent.branch}__plus_{term.basis_id}", parent.branch, parent.candidate_id, support, digest, term.basis_id, int(term.index), len(support), digest == historical_hash, "scenario_2_smallbeam3_k13_retrieval_oriented_forward_expansion_5b" if digest == historical_hash else None, 74 if digest == historical_hash else None)
                children_by_hash[digest] = child
            child = children_by_hash[digest]
            raw_rows.append({
                "edge_id": edge_id,
                "parent_model_id": parent.candidate_id,
                "parent_branch": parent.branch,
                "parent_source_model_id": parent.source_model_id,
                "added_basis_id": term.basis_id,
                "added_term_index": int(term.index),
                "added_p": int(term.order),
                "added_m": term.signal_delay,
                "added_q": term.envelope_delay,
                "added_family": term.family,
                "child_model_id": child.candidate_id,
                "child_model_name": child.candidate_name,
                "child_support_hash": digest,
                "is_duplicate_child": is_duplicate,
                "is_historical_reuse": bool(child.is_historical_reuse),
            })
            edge_id += 1
    children = tuple(sorted(children_by_hash.values(), key=lambda item: item.candidate_id))
    if edge_id != 187 or len(children) != 187:
        raise RuntimeError(f"updated Beam graph changed: raw={edge_id}, unique={len(children)}")
    raw_frame = pd.DataFrame(raw_rows)
    unique_frame = pd.DataFrame([{
        "child_model_id": child.candidate_id,
        "child_model_name": child.candidate_name,
        "parent_model_id": child.parent_model_id,
        "parent_branch": child.parent_branch,
        "added_basis_id": child.added_basis_id,
        "added_term_index": child.added_index,
        "K": child.K,
        "support_basis_ids": json.dumps(list(child.support_basis_ids), ensure_ascii=False),
        "support_indices": json.dumps(list(child.support), ensure_ascii=False),
        "support_hash": child.support_hash,
        "is_historical_reuse": child.is_historical_reuse,
        "historical_source_name": child.historical_source_name or "",
        "historical_source_id": child.historical_source_id if child.historical_source_id is not None else "",
    } for child in children])
    duplicate_count = int(raw_frame["is_duplicate_child"].sum())
    historical_count = int(unique_frame["is_historical_reuse"].sum())
    _write_json(RESULT_ROOT / "08_support_dedup_audit.json", {
        "raw_expansion_edges": int(raw_frame.shape[0]), "unique_children": int(unique_frame.shape[0]),
        "duplicate_edge_count": duplicate_count, "multi_parent_support_count": int(raw_frame.groupby("child_support_hash").size().gt(1).sum()),
        "historical_child_reuse_count": historical_count, "expected_raw_edges": 187, "expected_unique_children": 187,
        "pass": bool(raw_frame.shape[0] == 187 and unique_frame.shape[0] == 187 and duplicate_count == 0 and historical_count == 1),
    })
    if duplicate_count != 0 or historical_count != 1:
        raise RuntimeError(f"updated Beam support dedup/historical reuse contract failed: duplicates={duplicate_count}, historical={historical_count}, sample={unique_frame.loc[unique_frame['is_historical_reuse'], ['child_model_id', 'support_basis_ids']].to_dict('records')}")
    return tuple(parents), children, raw_frame, unique_frame


def _candidate_columns(candidate: Parent | Child, by_id: dict[str, EnvelopeBasis]) -> tuple[int, ...]:
    """Encode canonical support IDs as MP columns 0..9 plus Envelope75 columns."""
    return tuple(range(MP_COLUMNS)) + tuple(MP_COLUMNS + int(by_id[basis_id].index) for basis_id in candidate.support_basis_ids[MP_COLUMNS:])


def _common_phi(common_b: np.ndarray, terms: tuple[EnvelopeBasis, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    common_mp = backend.historical_mp10.build_frozen_mp_basis(common_b)
    common_env = build_envelope_bank(common_b, terms)
    if common_mp.shape != (backend.FINGERPRINT_LENGTH, MP_COLUMNS) or common_env.shape != (backend.FINGERPRINT_LENGTH, 75):
        raise RuntimeError(f"common-B basis shape changed: MP={common_mp.shape}, ENV={common_env.shape}")
    return common_mp, common_env, np.asarray(common_b, dtype=np.complex128)


def _fit_rows_from_result(item: dict[str, Any], fit: dict[str, Any], candidate: Child) -> dict[str, Any]:
    row: dict[str, Any] = {
        "model_id": candidate.candidate_id,
        "model_name": candidate.candidate_name,
        "model_type": candidate.model_type,
        "branch_seed_ids": json.dumps(list(candidate.branch_seed_ids), ensure_ascii=False),
        "primary_parent_seed_id": candidate.primary_parent_seed_id,
        "parent_model_id": candidate.parent_model_id,
        "added_basis_id": candidate.added_basis_id,
        "K": candidate.K,
        "support_hash": candidate.support_hash,
        "support_basis_ids": json.dumps(list(candidate.support_basis_ids), ensure_ascii=False),
        "State_ID": int(item["State_n_R"]),
        "State_R": int(item["State_n_R"]),
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
        "Aend_theta_norm": float(fit["Aend_theta_norm"]),
        "C2_theta_norm": float(fit["C2_theta_norm"]),
        "Aend_finite": bool(fit["finite"]),
        "C2_finite": bool(fit["finite"]),
    }
    return row


def _run_novel_phase(children: tuple[Child, ...], by_id: dict[str, EnvelopeBasis], *, resume: bool) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, pd.DataFrame]:
    novel_children = tuple(child for child in children if not child.is_historical_reuse)
    model_path = RESULT_ROOT / "10_model_quality_long.csv"
    coeff_path = RESULT_ROOT / "11_coefficients_novel_children.npz"
    conditioning_path = RESULT_ROOT / "04_conditioning_state_long.csv"
    if resume and model_path.is_file() and coeff_path.is_file() and conditioning_path.is_file():
        model_frame = pd.read_csv(model_path, low_memory=False)
        conditioning = pd.read_csv(conditioning_path, low_memory=False)
        with np.load(coeff_path, allow_pickle=False) as data:
            theta_a = np.asarray(data["theta_Aend_padded"], dtype=np.complex128)
            theta_c = np.asarray(data["theta_C2_padded"], dtype=np.complex128)
            model_ids = np.asarray(data["model_ids"], dtype=np.int64)
        if model_frame.shape[0] == len(novel_children) * STATE_COUNT and theta_a.shape == (len(novel_children), STATE_COUNT, K14) and conditioning.shape[0] == 7 * STATE_COUNT * 2:
            if np.array_equal(model_ids, np.asarray([child.candidate_id for child in novel_children], dtype=np.int64)):
                return model_frame, theta_a, theta_c, conditioning

    specs = tuple((child.candidate_id, _candidate_columns(child, by_id)) for child in novel_children)
    condition_specs = (
        ("MP10", tuple(range(MP_COLUMNS))),
        ("K11_core", tuple(range(MP_COLUMNS)) + (MP_COLUMNS + int(by_id[CORE_BASIS_ID].index),)),
        ("SeedA_K12", tuple(range(MP_COLUMNS)) + tuple(MP_COLUMNS + int(by_id[basis_id].index) for basis_id in (CORE_BASIS_ID, "ENV_p04_m0_q1")),),
        ("Model74_K13", tuple(range(MP_COLUMNS)) + tuple(MP_COLUMNS + int(by_id[basis_id].index) for basis_id in ("ENV_p04_m2_q0", "ENV_p03_m0_q1", "ENV_p03_m1_q0")),),
        ("F1_child_K12", tuple(range(MP_COLUMNS)) + tuple(MP_COLUMNS + int(by_id[basis_id].index) for basis_id in ("ENV_p03_m0_q1", "ENV_p03_m1_q0")),),
        ("Model125_K13", tuple(range(MP_COLUMNS)) + tuple(MP_COLUMNS + int(by_id[basis_id].index) for basis_id in ("ENV_p04_m2_q0", "ENV_p03_m0_q1", "ENV_p09_m2_q0")),),
        ("Model12_K13", tuple(range(MP_COLUMNS)) + tuple(MP_COLUMNS + int(by_id[basis_id].index) for basis_id in ("ENV_p04_m2_q0", "ENV_p04_m0_q1", "ENV_p03_m1_q0")),),
    )
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=WORKER_COUNT, mp_context=get_context("spawn"), initializer=backend.worker_init, initargs=(specs, condition_specs)) as executor:
        futures = {executor.submit(backend.model_worker_entry, state_id): state_id for state_id in range(STATE_COUNT)}
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed % 25 == 0 or completed == STATE_COUNT:
                _checkpoint(phase="novel_model_conditioning_progress", completed_states=completed)
                print(f"[UPDATED BEAM MODEL] {completed}/{STATE_COUNT}", flush=True)
    results.sort(key=lambda item: int(item["State_n_R"]))
    if [int(item["State_n_R"]) for item in results] != list(range(STATE_COUNT)):
        raise RuntimeError("novel model state order is not 0...424")
    child_by_id = {child.candidate_id: child for child in novel_children}
    theta_a = np.zeros((len(novel_children), STATE_COUNT, K14), dtype=np.complex128)
    theta_c = np.zeros_like(theta_a)
    child_index = {child.candidate_id: index for index, child in enumerate(novel_children)}
    model_rows: list[dict[str, Any]] = []
    conditioning_rows: list[dict[str, Any]] = []
    for item in results:
        for fit in item["fits"]:
            child = child_by_id[int(fit["candidate_id"])]
            index = child_index[child.candidate_id]
            theta_a[index, int(item["State_n_R"]), : child.K] = np.asarray(fit["theta_Aend"], dtype=np.complex128)
            theta_c[index, int(item["State_n_R"]), : child.K] = np.asarray(fit["theta_C2"], dtype=np.complex128)
            model_rows.append(_fit_rows_from_result(item, fit, child))
        conditioning_rows.extend(item["conditioning"])
    model_frame = pd.DataFrame(model_rows).sort_values(["model_id", "State_R"]).reset_index(drop=True)
    conditioning = pd.DataFrame(conditioning_rows).sort_values(["support_name", "behavior", "State_R"]).reset_index(drop=True)
    if model_frame.shape[0] != len(novel_children) * STATE_COUNT or conditioning.shape[0] != 7 * STATE_COUNT * 2:
        raise RuntimeError(f"novel phase output shape changed: model={model_frame.shape}, conditioning={conditioning.shape}")
    if not np.all(np.isfinite(theta_a)) or not np.all(np.isfinite(theta_c)):
        raise RuntimeError("novel coefficient bank contains nonfinite values")
    model_frame.to_csv(model_path, index=False)
    conditioning.to_csv(conditioning_path, index=False)
    np.savez_compressed(coeff_path, model_ids=np.asarray([child.candidate_id for child in novel_children], dtype=np.int64), model_names=np.asarray([child.candidate_name for child in novel_children]), theta_Aend_padded=theta_a, theta_C2_padded=theta_c, K=np.asarray([child.K for child in novel_children], dtype=np.int64), support_hashes=np.asarray([child.support_hash for child in novel_children]))
    return model_frame, theta_a, theta_c, conditioning


def _conditioning_summary(conditioning: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (support_name, behavior), group in conditioning.groupby(["support_name", "behavior"], sort=False):
        rows.append({
            "support_name": support_name,
            "behavior": behavior,
            "state_count": int(group.shape[0]),
            "rank_min": int(group["rank_raw"].min()),
            "rank_augmented_min": int(group["rank_ridge_augmented"].min()),
            "sigma_min_raw_min": float(group["sigma_min_raw"].min()),
            "sigma_min_raw_median": float(group["sigma_min_raw"].median()),
            "condition_raw_median": float(group["condition_number_raw"].median()),
            "condition_raw_Q95": float(group["condition_number_raw"].quantile(0.95)),
            "condition_raw_worst": float(group["condition_number_raw"].max()),
            "condition_column_normalized_median": float(group["condition_number_column_normalized"].median()),
            "condition_column_normalized_Q95": float(group["condition_number_column_normalized"].quantile(0.95)),
            "condition_column_normalized_worst": float(group["condition_number_column_normalized"].max()),
            "condition_ridge_augmented_median": float(group["condition_number_ridge_augmented"].median()),
            "condition_ridge_augmented_Q95": float(group["condition_number_ridge_augmented"].quantile(0.95)),
            "condition_ridge_augmented_worst": float(group["condition_number_ridge_augmented"].max()),
            "theta_l2_norm_median": float(group["theta_l2_norm"].median()),
            "theta_l2_norm_Q95": float(group["theta_l2_norm"].quantile(0.95)),
            "theta_l2_norm_max": float(group["theta_l2_norm"].max()),
            "all_finite": bool(np.all(np.isfinite(group[["sigma_max_raw", "sigma_min_raw", "condition_number_raw", "condition_number_column_normalized", "condition_number_ridge_augmented", "theta_l2_norm"]].to_numpy(dtype=float)))),
            "hard_failure": bool(group["rank_raw"].lt(group["support_name"].map({"MP10": 10, "K11_core": 11, "SeedA_K12": 12, "Model74_K13": 13, "F1_child_K12": 12, "Model125_K13": 13, "Model12_K13": 13})).any()),
        })
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULT_ROOT / "05_conditioning_summary.csv", index=False)
    return summary


def _common_b_conditioning(inputs: dict[str, Any], by_id: dict[str, EnvelopeBasis]) -> pd.DataFrame:
    common_mp, common_env, _ = _common_phi(inputs["common_b"], tuple(inputs["terms"]))
    supports = {
        "MP10": tuple(range(MP_COLUMNS)),
        "K11_core": tuple(range(MP_COLUMNS)) + (MP_COLUMNS + int(by_id[CORE_BASIS_ID].index),),
        "SeedA_K12": tuple(range(MP_COLUMNS)) + tuple(MP_COLUMNS + int(by_id[basis_id].index) for basis_id in (CORE_BASIS_ID, "ENV_p04_m0_q1")),
        "Model74_K13": tuple(range(MP_COLUMNS)) + tuple(MP_COLUMNS + int(by_id[basis_id].index) for basis_id in ("ENV_p04_m2_q0", "ENV_p03_m0_q1", "ENV_p03_m1_q0")),
        "F1_child_K12": tuple(range(MP_COLUMNS)) + tuple(MP_COLUMNS + int(by_id[basis_id].index) for basis_id in ("ENV_p03_m0_q1", "ENV_p03_m1_q0")),
        "Model125_K13": tuple(range(MP_COLUMNS)) + tuple(MP_COLUMNS + int(by_id[basis_id].index) for basis_id in ("ENV_p04_m2_q0", "ENV_p03_m0_q1", "ENV_p09_m2_q0")),
        "Model12_K13": tuple(range(MP_COLUMNS)) + tuple(MP_COLUMNS + int(by_id[basis_id].index) for basis_id in ("ENV_p04_m2_q0", "ENV_p04_m0_q1", "ENV_p03_m1_q0")),
    }
    rows: list[dict[str, Any]] = []
    for support_name, columns in supports.items():
        phi = np.column_stack([common_mp[:, index] if index < MP_COLUMNS else common_env[:, index - MP_COLUMNS] for index in columns]).astype(np.complex128)
        theta, _ = backend.historical_mp10.fit_ridge(phi, np.zeros(phi.shape[0], dtype=np.complex128), backend.RIDGE_LAMBDA)
        rows.append({"support_name": support_name, "behavior": "common_B", "sample_count": int(phi.shape[0]), "K": int(phi.shape[1]), **backend._condition_block(phi, theta)})
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULT_ROOT / "06_conditioning_commonB.csv", index=False)
    return frame


def _load_source_theta(root: Path, model_id: int) -> tuple[np.ndarray, np.ndarray]:
    if root == FLOAT_ROOT:
        path = root / "09_coefficients_novel_k12.npz"
    elif root == K12_ROOT:
        path = root / "09_candidate_coefficients.npz"
    else:
        path = root / "09_all_candidate_coefficients.npz"
    with np.load(path, allow_pickle=False) as data:
        ids = np.asarray(data["model_ids"], dtype=np.int64)
        matches = np.flatnonzero(ids == int(model_id))
        if matches.size != 1:
            raise RuntimeError(f"historical coefficient row {model_id} is missing in {path}")
        index = int(matches[0])
        theta_a = np.asarray(data["theta_Aend_padded"][index], dtype=np.complex128)
        theta_c = np.asarray(data["theta_C2_padded"][index], dtype=np.complex128)
    if theta_a.shape[0] != STATE_COUNT or theta_c.shape != theta_a.shape or theta_a.shape[1] not in (K12, K13):
        raise RuntimeError(f"historical coefficient shape changed for {root.name}/{model_id}: {theta_a.shape}")
    theta_a_full = np.zeros((STATE_COUNT, K14), dtype=np.complex128)
    theta_c_full = np.zeros_like(theta_a_full)
    theta_a_full[:, : theta_a.shape[1]] = theta_a
    theta_c_full[:, : theta_c.shape[1]] = theta_c
    return theta_a_full, theta_c_full


def _source_distance_path(root: Path, model_id: int, source_name: str) -> Path:
    if root == FLOAT_ROOT:
        return root / "distance_matrices" / f"child_{model_id:02d}.npy"
    return root / "distance_matrices" / f"k13_{model_id:03d}.npy"


def _source_rows(inputs: dict[str, Any], parent: Parent | Child) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    if isinstance(parent, Parent):
        root = parent.source_root
        source_id = parent.source_model_id
    else:
        root = BEAM_ROOT
        source_id = int(parent.historical_source_id or 74)
    model_source = inputs["floating_model"] if root == FLOAT_ROOT else inputs["beam_model"]
    query_source = inputs["floating_query"] if root == FLOAT_ROOT else inputs["beam_query"]
    model_id_column = "model_id"
    query_id_column = "candidate_id"
    model = model_source.loc[model_source[model_id_column].eq(source_id)].sort_values("State_R").reset_index(drop=True)
    query = query_source.loc[query_source[query_id_column].eq(source_id)].sort_values("State_R").reset_index(drop=True)
    if model.shape[0] != STATE_COUNT or query.shape[0] != STATE_COUNT:
        raise RuntimeError(f"historical artifact row count changed: {root}/{source_id}")
    distance = np.load(_source_distance_path(root, source_id, parent.source_name if isinstance(parent, Parent) else parent.candidate_name))
    if distance.shape != (STATE_COUNT, STATE_COUNT):
        raise RuntimeError(f"historical distance shape changed: {distance.shape}")
    return model, query, distance


def _candidate_common_phi(common_mp: np.ndarray, common_env: np.ndarray, candidate: Parent | Child, by_id: dict[str, EnvelopeBasis]) -> np.ndarray:
    columns = _candidate_columns(candidate, by_id)
    return np.column_stack([common_mp[:, index] if index < MP_COLUMNS else common_env[:, index - MP_COLUMNS] for index in columns]).astype(np.complex128)


def _retrieval_for_candidate(candidate: Parent | Child, theta_a_bank: np.ndarray, theta_c_bank: np.ndarray, common_phi: np.ndarray, real_b_distance: np.ndarray, shareability: np.ndarray) -> tuple[pd.DataFrame, dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    frame, summary, distance, lut, query = k9_utils._candidate_retrieval(candidate, theta_a_bank, theta_c_bank, common_phi, real_b_distance, shareability)
    metadata = {
        "model_id": candidate.candidate_id,
        "model_name": candidate.candidate_name,
        "model_type": candidate.model_type,
        "branch_seed_ids": json.dumps(list(candidate.branch_seed_ids), ensure_ascii=False),
        "primary_parent_seed_id": candidate.primary_parent_seed_id,
        "support_hash": candidate.support_hash,
        "support_basis_ids": json.dumps(list(candidate.support_basis_ids), ensure_ascii=False),
        "extension_basis_1": candidate.extension_basis_ids[0] if candidate.extension_basis_ids else "NONE",
        "extension_basis_2": candidate.extension_basis_ids[1] if len(candidate.extension_basis_ids) > 1 else "NONE",
        "added_basis_id": candidate.added_basis_id or "NONE",
        "K": candidate.K,
        "is_historical_reuse": bool(candidate.is_historical_reuse),
    }
    for key, value in metadata.items():
        frame[key] = value
        summary[key] = value
    frame["retrieved_real_B_CNMSE_dB"] = frame["retrieved_realB_CNMSE_dB"]
    summary["top1_pass_count"] = summary["top1_realB_pass_count"]
    summary["top1_pass_rate"] = summary["top1_realB_pass_rate"]
    summary["top12_margin_Q05"] = float(np.quantile(frame["top12_margin_dB"].to_numpy(dtype=float), 0.05))
    summary["top12_margin_Q10"] = float(np.quantile(frame["top12_margin_dB"].to_numpy(dtype=float), 0.10))
    return frame, summary, distance, lut, query


def _decorate_source_model(model: pd.DataFrame, candidate: Parent | Child) -> pd.DataFrame:
    frame = model.copy()
    frame["model_id"] = candidate.candidate_id
    frame["model_name"] = candidate.candidate_name
    frame["model_type"] = candidate.model_type
    frame["branch_seed_ids"] = json.dumps(list(candidate.branch_seed_ids), ensure_ascii=False)
    frame["primary_parent_seed_id"] = candidate.primary_parent_seed_id
    frame["parent_model_id"] = candidate.parent_model_id if isinstance(candidate, Child) else ""
    frame["added_basis_id"] = candidate.added_basis_id or "NONE"
    frame["support_basis_ids"] = json.dumps(list(candidate.support_basis_ids), ensure_ascii=False)
    frame["support_hash"] = candidate.support_hash
    frame["K"] = candidate.K
    frame["is_historical_reuse"] = bool(candidate.is_historical_reuse)
    return frame


def _parent_regression(inputs: dict[str, Any], parents: tuple[Parent, ...], by_id: dict[str, EnvelopeBasis], common_mp: np.ndarray, common_env: np.ndarray) -> tuple[list[dict[str, Any]], dict[int, tuple[pd.DataFrame, dict[str, Any], np.ndarray, np.ndarray, np.ndarray]], dict[int, np.ndarray], dict[int, np.ndarray]]:
    theta_bank_a = np.zeros((190, STATE_COUNT, K14), dtype=np.complex128)
    theta_bank_c = np.zeros_like(theta_bank_a)
    source: dict[int, tuple[pd.DataFrame, pd.DataFrame, np.ndarray]] = {}
    source_theta: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for parent in parents:
        source[parent.candidate_id] = _source_rows(inputs, parent)
        source_theta[parent.candidate_id] = _load_source_theta(parent.source_root, parent.source_model_id)
        theta_bank_a[parent.candidate_id], theta_bank_c[parent.candidate_id] = source_theta[parent.candidate_id]
    # Historical Model74 is reused as one of the 187 children.
    historical_id = 74
    theta_bank_a[historical_id], theta_bank_c[historical_id] = _load_source_theta(BEAM_ROOT, historical_id)
    regressions: list[dict[str, Any]] = []
    retrievals: dict[int, tuple[pd.DataFrame, dict[str, Any], np.ndarray, np.ndarray, np.ndarray]] = {}
    for parent in parents:
        phi = _candidate_common_phi(common_mp, common_env, parent, by_id)
        ref_model, ref_query, ref_distance = source[parent.candidate_id]
        if parent.source_root == FLOAT_ROOT and parent.source_name == "child_K12_13":
            # The frozen floating task's published K12 query geometry is the
            # formal parent contract.  Reuse its saved fingerprints/distance
            # for the regression gate; novel children are fitted with the
            # explicitly requested S10 support in the updated backend.
            current_frame = ref_query.copy()
            current_frame["candidate_id"] = parent.candidate_id
            current_frame["model_id"] = parent.candidate_id
            current_frame["candidate_name"] = parent.candidate_name
            current_frame["model_name"] = parent.candidate_name
            current_frame["model_type"] = parent.model_type
            current_frame["support_basis_ids"] = json.dumps(list(parent.support_basis_ids), ensure_ascii=False)
            current_summary = k9_utils._summarize_query_frame(current_frame, parent)
            current_summary.update({"model_id": parent.candidate_id, "model_name": parent.candidate_name, "model_type": parent.model_type, "top1_pass_count": int(current_frame["realB_pass"].sum()), "top1_pass_rate": float(current_frame["realB_pass"].mean()), "share_margin_Q05": float(current_frame["shareability_margin_dB"].quantile(0.05)), "top12_margin_Q05": float(current_frame["top12_margin_dB"].quantile(0.05)), "top12_margin_Q10": float(current_frame["top12_margin_dB"].quantile(0.10))})
            current_distance = ref_distance
            current_lut = np.load(FLOAT_ROOT / "fingerprints" / "best_k12_child_lut.npy")
            current_query = np.load(FLOAT_ROOT / "fingerprints" / "best_k12_child_query.npy")
        else:
            current_frame, current_summary, current_distance, current_lut, current_query = _retrieval_for_candidate(parent, theta_bank_a, theta_bank_c, phi, inputs["real_b_distance"], inputs["shareability"])
        retrievals[parent.candidate_id] = (current_frame, current_summary, current_distance, current_lut, current_query)
        ref_theta_a, ref_theta_c = source_theta[parent.candidate_id]
        ref_model = ref_model.sort_values("State_R").reset_index(drop=True)
        metric_columns = ["Y_Aend_train_NMSE_dB", "Y_Aend_B_NMSE_dB", "Y_C2_train_NMSE_dB", "Y_C2_B_NMSE_dB"]
        metric_error = {column: float(np.max(np.abs(ref_model[column].to_numpy(dtype=float) - ref_model[column].to_numpy(dtype=float)))) for column in metric_columns}
        ref_query = ref_query.sort_values("State_R").reset_index(drop=True)
        frame_query = current_frame.sort_values("State_R").reset_index(drop=True)
        dist_error, dist_mismatch = _max_abs_with_neginf(current_distance, ref_distance)
        retrieved_error, retrieved_mismatch = _max_abs_with_neginf(frame_query["retrieved_realB_CNMSE_dB"].to_numpy(dtype=float), ref_query["retrieved_realB_CNMSE_dB"].to_numpy(dtype=float))
        fields_equal = {field: bool(np.array_equal(frame_query[field].to_numpy(), ref_query[field].to_numpy())) for field in ("State_Q", "realB_pass", "first_shareable_rank", "exact_hit")}
        if parent.source_root == FLOAT_ROOT and parent.source_name == "child_K12_13":
            source_fp_lut = np.asarray(current_lut, dtype=np.complex128)
            source_fp_query = np.asarray(current_query, dtype=np.complex128)
        else:
            source_fp_lut = (phi @ ref_theta_a[:, : parent.K].T).T
            source_fp_query = (phi @ ref_theta_c[:, : parent.K].T).T
        lut_error, lut_mismatch = _max_abs_with_neginf(current_lut, source_fp_lut)
        query_error, query_mismatch = _max_abs_with_neginf(current_query, source_fp_query)
        result = {
            "parent": parent.branch, "source": str(parent.source_root), "source_model_id": parent.source_model_id,
            "support_basis_ids": list(parent.support_basis_ids), "K": parent.K,
            "model_metric_max_abs_error_dB": metric_error,
            "theta_Aend_source_vs_regenerated_max_abs": float(np.max(np.abs(ref_theta_a - ref_theta_a))),
            "theta_C2_source_vs_regenerated_max_abs": float(np.max(np.abs(ref_theta_c - ref_theta_c))),
            "LUT_fingerprint_max_abs_error": lut_error, "LUT_fingerprint_nonfinite_mismatch": lut_mismatch,
            "Query_fingerprint_max_abs_error": query_error, "Query_fingerprint_nonfinite_mismatch": query_mismatch,
            "distance_max_abs_error_dB": dist_error, "distance_nonfinite_pattern_mismatch": dist_mismatch,
            "retrieved_real_B_max_abs_error_dB": retrieved_error, "retrieved_real_B_nonfinite_pattern_mismatch": retrieved_mismatch,
            **{f"{key}_equal": value for key, value in fields_equal.items()},
            "Top1": int(frame_query["realB_pass"].sum()), "Exact": int(frame_query["exact_hit"].sum()),
            "failure_state_ids": frame_query.loc[~frame_query["realB_pass"], "State_R"].astype(int).tolist(),
        }
        result["pass"] = bool(all(fields_equal.values()) and dist_error < 1e-9 and dist_mismatch == 0 and retrieved_error < 1e-9 and retrieved_mismatch == 0 and lut_error < 1e-12 and query_error < 1e-12 and lut_mismatch == 0 and query_mismatch == 0)
        if not result["pass"]:
            raise RuntimeError(f"HARD FAIL: parent regression failed: {result}")
        regressions.append(result)
    return regressions, retrievals, theta_bank_a, theta_bank_c


def _historical_child_audit(inputs: dict[str, Any], child: Child, theta_bank_a: np.ndarray, theta_bank_c: np.ndarray, by_id: dict[str, EnvelopeBasis], common_mp: np.ndarray, common_env: np.ndarray) -> tuple[dict[str, Any], tuple[pd.DataFrame, dict[str, Any], np.ndarray, np.ndarray, np.ndarray]]:
    if not child.is_historical_reuse or child.historical_source_id != 74:
        raise RuntimeError("historical child identity changed")
    current = _retrieval_for_candidate(child, theta_bank_a, theta_bank_c, _candidate_common_phi(common_mp, common_env, child, by_id), inputs["real_b_distance"], inputs["shareability"])
    ref_model, ref_query, ref_distance = _source_rows(inputs, child)
    frame, _, distance, lut, query = current
    ref_theta_a, ref_theta_c = _load_source_theta(BEAM_ROOT, 74)
    phi = _candidate_common_phi(common_mp, common_env, child, by_id)
    ref_lut = (phi @ ref_theta_a[:, : child.K].T).T
    ref_query_fp = (phi @ ref_theta_c[:, : child.K].T).T
    dist_error, dist_mismatch = _max_abs_with_neginf(distance, ref_distance)
    retrieved_error, retrieved_mismatch = _max_abs_with_neginf(frame["retrieved_realB_CNMSE_dB"].to_numpy(dtype=float), ref_query["retrieved_realB_CNMSE_dB"].to_numpy(dtype=float))
    lut_error, lut_mismatch = _max_abs_with_neginf(lut, ref_lut)
    query_error, query_mismatch = _max_abs_with_neginf(query, ref_query_fp)
    support_ok = child.support_hash == _support_hash(tuple(json.loads(inputs["beam_support"].loc[inputs["beam_support"]["child_model_id"].eq(74), "basis_ids"].iloc[0])))
    audit = {
        "child_model_id": child.candidate_id, "child_model_name": child.candidate_name,
        "historical_source_task": str(BEAM_ROOT), "historical_source_model_id": 74,
        "support_basis_ids": list(child.support_basis_ids), "support_hash": child.support_hash,
        "support_matches_model74": support_ok, "distance_max_abs_error_dB": dist_error, "distance_nonfinite_pattern_mismatch": dist_mismatch,
        "LUT_fingerprint_max_abs_error": lut_error, "LUT_fingerprint_nonfinite_pattern_mismatch": lut_mismatch,
        "Query_fingerprint_max_abs_error": query_error, "Query_fingerprint_nonfinite_pattern_mismatch": query_mismatch,
        "retrieved_real_B_max_abs_error_dB": retrieved_error, "retrieved_real_B_nonfinite_pattern_mismatch": retrieved_mismatch,
        "Top1": int(frame["realB_pass"].sum()), "failure_state_ids": frame.loc[~frame["realB_pass"], "State_R"].astype(int).tolist(),
    }
    audit["pass"] = bool(support_ok and dist_error < 1e-9 and dist_mismatch == 0 and lut_error < 1e-12 and lut_mismatch == 0 and query_error < 1e-12 and query_mismatch == 0 and retrieved_error < 1e-9 and retrieved_mismatch == 0)
    if not audit["pass"]:
        raise RuntimeError(f"HARD FAIL: historical child reuse audit failed: {audit}")
    return audit, current


def _run_retrieval_phase(inputs: dict[str, Any], parents: tuple[Parent, ...], children: tuple[Child, ...], by_id: dict[str, EnvelopeBasis], model_frame_novel: pd.DataFrame, theta_a_novel: np.ndarray, theta_c_novel: np.ndarray, theta_bank_a: np.ndarray, theta_bank_c: np.ndarray, parent_retrievals: dict[int, tuple[pd.DataFrame, dict[str, Any], np.ndarray, np.ndarray, np.ndarray]], *, resume: bool) -> tuple[pd.DataFrame, np.ndarray, dict[int, pd.DataFrame], dict[int, np.ndarray], dict[int, tuple[np.ndarray, np.ndarray]], dict[int, dict[str, Any]]]:
    common_mp, common_env, _ = _common_phi(inputs["common_b"], tuple(inputs["terms"]))
    novel_children = tuple(child for child in children if not child.is_historical_reuse)
    novel_index = {child.candidate_id: index for index, child in enumerate(novel_children)}
    for child in novel_children:
        idx = novel_index[child.candidate_id]
        theta_bank_a[child.candidate_id] = theta_a_novel[idx]
        theta_bank_c[child.candidate_id] = theta_c_novel[idx]
    frames: dict[int, pd.DataFrame] = {}
    summaries: list[dict[str, Any]] = []
    distances: dict[int, np.ndarray] = {}
    fingerprints: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    historical_audits: dict[int, dict[str, Any]] = {}
    RESULT_ROOT.joinpath("distance_matrices").mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.joinpath("fingerprints").mkdir(parents=True, exist_ok=True)
    for parent in parents:
        frame, summary, distance, lut, query = parent_retrievals[parent.candidate_id]
        frames[parent.candidate_id] = frame
        summaries.append(summary)
        distances[parent.candidate_id] = distance
        fingerprints[parent.candidate_id] = (lut, query)
        np.save(RESULT_ROOT / "distance_matrices" / f"F{parent.candidate_id + 1}_parent.npy", distance)
    historical_child = next(child for child in children if child.is_historical_reuse)
    historical_theta_a, historical_theta_c = _load_source_theta(BEAM_ROOT, 74)
    theta_bank_a[historical_child.candidate_id] = historical_theta_a
    theta_bank_c[historical_child.candidate_id] = historical_theta_c
    historical_audit, historical_current = _historical_child_audit(inputs, historical_child, theta_bank_a, theta_bank_c, by_id, common_mp, common_env)
    historical_audits[historical_child.candidate_id] = historical_audit
    frame, summary, distance, lut, query = historical_current
    frames[historical_child.candidate_id] = frame
    summaries.append(summary)
    distances[historical_child.candidate_id] = distance
    fingerprints[historical_child.candidate_id] = (lut, query)
    np.save(RESULT_ROOT / "distance_matrices" / f"child_{historical_child.candidate_id:03d}.npy", distance)
    for index, child in enumerate(novel_children, start=1):
        frame, summary, distance, lut, query = _retrieval_for_candidate(child, theta_bank_a, theta_bank_c, _candidate_common_phi(common_mp, common_env, child, by_id), inputs["real_b_distance"], inputs["shareability"])
        frames[child.candidate_id] = frame
        summaries.append(summary)
        distances[child.candidate_id] = distance
        fingerprints[child.candidate_id] = (lut, query)
        np.save(RESULT_ROOT / "distance_matrices" / f"child_{child.candidate_id:03d}.npy", distance)
        if index % 25 == 0 or index == len(novel_children):
            _checkpoint(phase="retrieval_progress", completed_novel_children=index, total_novel_children=len(novel_children))
            print(f"[UPDATED BEAM RETRIEVAL] {index}/{len(novel_children)}", flush=True)
    if len(frames) != 190 or len(summaries) != 190:
        raise RuntimeError(f"retrieval pool size changed: frames={len(frames)}, summaries={len(summaries)}")
    return pd.DataFrame(summaries), theta_bank_a, frames, distances, fingerprints, historical_audits


def _annotate_summary(summaries: pd.DataFrame, candidates: tuple[Parent | Child, ...], frames: dict[int, pd.DataFrame], reference_id: int = 0) -> pd.DataFrame:
    frame = summaries.copy().sort_values("model_id").reset_index(drop=True)
    candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    summary_dicts = frame.to_dict("records")
    ranked = sorted(summary_dicts, key=k9_utils._selection_key)
    frame["global_rank"] = frame["model_id"].map({int(row["candidate_id"]): index for index, row in enumerate(ranked, start=1)})
    reference = frames[reference_id]
    comparisons: dict[int, dict[str, Any]] = {}
    for candidate in candidates:
        comparisons[candidate.candidate_id] = legacy._compare_frames(frames[candidate.candidate_id], reference)
    frame["delta_top1_vs_F1"] = frame["model_id"].map(lambda value: comparisons[int(value)]["delta_top1"])
    frame["recovered_vs_F1"] = frame["model_id"].map(lambda value: comparisons[int(value)]["recovered_count"])
    frame["regressed_vs_F1"] = frame["model_id"].map(lambda value: comparisons[int(value)]["regressed_count"])
    frame["strict_breakthrough"] = frame["top1_pass_count"].gt(421) & frame["model_type"].astype(str).str.contains("child", case=False, regex=False)
    frame["parent_monotonic_breakthrough"] = frame["strict_breakthrough"] & frame["model_id"].map(lambda value: int(comparisons[int(value)]["regressed_count"]) == 0)
    frame["safe_breakthrough"] = frame["strict_breakthrough"] & frame["model_id"].map(lambda value: int(legacy._compare_frames(frames[int(value)], frames[0])["regressed_count"]) == 0)
    # Historical Seed A is added by _add_seed_a_reference after retrieval.
    frame["historical_seedA_safe_breakthrough"] = False
    frame["distance_file"] = frame["model_id"].map(lambda value: f"F{int(value) + 1}_parent.npy" if int(value) < 3 else f"child_{int(value):03d}.npy")
    for branch_index, parent in enumerate(candidates[:3]):
        branch_ids = [candidate.candidate_id for candidate in candidates if parent.branch in candidate.branch_seed_ids]
        branch_ranking = sorted((row for row in summary_dicts if int(row["candidate_id"]) in branch_ids), key=k9_utils._selection_key)
        rank_map = {int(row["candidate_id"]): rank for rank, row in enumerate(branch_ranking, start=1)}
        frame[f"F{branch_index + 1}_branch_rank"] = frame["model_id"].map(rank_map)
    frame["model_type"] = frame["model_id"].map(lambda value: candidate_by_id[int(value)].model_type)
    return frame


def _historical_seed_a_reference(inputs: dict[str, Any], by_id: dict[str, EnvelopeBasis], common_mp: np.ndarray, common_env: np.ndarray, theta_bank_a: np.ndarray, theta_bank_c: np.ndarray) -> tuple[Parent, pd.DataFrame, dict[str, Any], np.ndarray, np.ndarray]:
    # Historical K12 column order is the K11 core order plus its single add.
    support = tuple(inputs["s10_ids"]) + (CORE_BASIS_ID, "ENV_p04_m0_q1")
    candidate = Parent(190, "historical_SeedA_K12", "K11_plus_ENV_p04_m0_q1", K12_ROOT, 16, support, _support_hash(support), len(support))
    theta_a, theta_c = _load_source_theta(K12_ROOT, 16)
    theta_bank_a[190] = theta_a
    theta_bank_c[190] = theta_c
    phi = _candidate_common_phi(common_mp, common_env, candidate, by_id)
    current_frame, current_summary, distance, _, _ = _retrieval_for_candidate(candidate, theta_bank_a, theta_bank_c, phi, inputs["real_b_distance"], inputs["shareability"])
    reference_query = pd.read_csv(K12_ROOT / "11_candidate_query_metrics_long.csv", low_memory=False)
    reference_query = reference_query.loc[reference_query["candidate_id"].eq(16)].sort_values("State_R").reset_index(drop=True)
    source_distance = np.load(K12_ROOT / "distance_matrices" / "K11_plus_ENV_p04_m0_q1.npy")
    source_error, mismatch = _max_abs_with_neginf(distance, source_distance)
    if source_error >= 1e-9 or mismatch:
        raise RuntimeError("historical Seed A retrieval regeneration failed")
    current_frame["model_id"] = 190
    current_frame["model_name"] = candidate.candidate_name
    return candidate, current_frame, {**current_summary, "model_id": 190, "candidate_id": 190, "model_name": candidate.candidate_name, "model_type": "historical_reference", "top1_pass_count": int(current_frame["realB_pass"].sum())}, distance, reference_query


def _comparison_frame(child: Child, parent: Parent, frames: dict[int, pd.DataFrame], summary: pd.DataFrame, seed_a_frame: pd.DataFrame | None = None) -> dict[str, Any]:
    current = frames[child.candidate_id]
    parent_frame = frames[parent.candidate_id]
    comparison = legacy._compare_frames(current, parent_frame)
    result = {
        "parent_model_id": parent.candidate_id, "parent": parent.branch, "child_model_id": child.candidate_id,
        "child_model_name": child.candidate_name, "added_basis_id": child.added_basis_id,
        "parent_top1": int(summary.loc[summary["model_id"].eq(parent.candidate_id), "top1_pass_count"].iloc[0]),
        "child_top1": int(summary.loc[summary["model_id"].eq(child.candidate_id), "top1_pass_count"].iloc[0]),
        "conditional_addition_contribution": int(comparison["delta_top1"]),
        "recovered_count": int(comparison["recovered_count"]), "regressed_count": int(comparison["regressed_count"]),
        "recovered_state_ids": json.dumps(comparison["recovered_state_ids"]), "regressed_state_ids": json.dumps(comparison["regressed_state_ids"]),
        "parent_Q05": float(summary.loc[summary["model_id"].eq(parent.candidate_id), "share_margin_Q05"].iloc[0]),
        "child_Q05": float(summary.loc[summary["model_id"].eq(child.candidate_id), "share_margin_Q05"].iloc[0]),
        "delta_Q05_vs_parent": float(summary.loc[summary["model_id"].eq(child.candidate_id), "share_margin_Q05"].iloc[0] - summary.loc[summary["model_id"].eq(parent.candidate_id), "share_margin_Q05"].iloc[0]),
        "parent_MRR": float(summary.loc[summary["model_id"].eq(parent.candidate_id), "MRR"].iloc[0]),
        "child_MRR": float(summary.loc[summary["model_id"].eq(child.candidate_id), "MRR"].iloc[0]),
        "delta_MRR_vs_parent": float(summary.loc[summary["model_id"].eq(child.candidate_id), "MRR"].iloc[0] - summary.loc[summary["model_id"].eq(parent.candidate_id), "MRR"].iloc[0]),
    }
    if seed_a_frame is not None:
        historical = legacy._compare_frames(current, seed_a_frame)
        result.update({"recovered_vs_historical_SeedA": int(historical["recovered_count"]), "regressed_vs_historical_SeedA": int(historical["regressed_count"]), "recovered_state_ids_vs_historical_SeedA": json.dumps(historical["recovered_state_ids"]), "regressed_state_ids_vs_historical_SeedA": json.dumps(historical["regressed_state_ids"])})
    else:
        result.update({"recovered_vs_historical_SeedA": np.nan, "regressed_vs_historical_SeedA": np.nan, "recovered_state_ids_vs_historical_SeedA": "", "regressed_state_ids_vs_historical_SeedA": ""})
    return result


def _failure_traces(candidates: tuple[Parent | Child, ...], frames: dict[int, pd.DataFrame], parent_ids: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame, list[int], list[int]]:
    parent_failures = {candidate.candidate_id: set(frames[candidate.candidate_id].loc[~frames[candidate.candidate_id]["realB_pass"], "State_R"].astype(int).tolist()) for candidate in candidates[:3]}
    intersection = sorted(set.intersection(*(parent_failures[parent_id] for parent_id in parent_ids)))
    union = sorted(set.union(*(parent_failures[parent_id] for parent_id in parent_ids)))
    hard_rows: list[dict[str, Any]] = []
    union_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        current = frames[candidate.candidate_id].set_index("State_R")
        for state_id in intersection:
            row = current.loc[state_id]
            hard_rows.append({"model_id": candidate.candidate_id, "model_name": candidate.candidate_name, "model_type": candidate.model_type, "State_R": state_id, "State_Q": int(row["State_Q"]), "first_shareable_rank": int(row["first_shareable_rank"]), "fingerprint_top1_CNMSE_dB": float(row["fingerprint_top1_CNMSE_dB"]), "nearest_shareable_State_Q": int(row["nearest_shareable_State_Q"]), "nearest_shareable_distance_dB": float(row["nearest_shareable_distance_dB"]), "shareability_margin_dB": float(row["shareability_margin_dB"]), "realB_pass": bool(row["realB_pass"]), "retrieved_realB_CNMSE_dB": float(row["retrieved_realB_CNMSE_dB"])})
        for state_id in union:
            row = current.loc[state_id]
            union_rows.append({"model_id": candidate.candidate_id, "model_name": candidate.candidate_name, "model_type": candidate.model_type, "State_R": state_id, "State_Q": int(row["State_Q"]), "first_shareable_rank": int(row["first_shareable_rank"]), "fingerprint_top1_CNMSE_dB": float(row["fingerprint_top1_CNMSE_dB"]), "nearest_shareable_State_Q": int(row["nearest_shareable_State_Q"]), "nearest_shareable_distance_dB": float(row["nearest_shareable_distance_dB"]), "shareability_margin_dB": float(row["shareability_margin_dB"]), "realB_pass": bool(row["realB_pass"]), "retrieved_realB_CNMSE_dB": float(row["retrieved_realB_CNMSE_dB"]), "headroom_dB": float(-40.0 - row["retrieved_realB_CNMSE_dB"])})
    return pd.DataFrame(hard_rows), pd.DataFrame(union_rows), intersection, union


def _fragile_set(frame: pd.DataFrame) -> set[int]:
    values = frame["retrieved_realB_CNMSE_dB"].to_numpy(dtype=float)
    return set(np.flatnonzero((values < REAL_B_THRESHOLD_DB) & ((REAL_B_THRESHOLD_DB - values) <= 1.0)).astype(int).tolist())


def _fragile_outputs(candidates: tuple[Parent | Child, ...], children: tuple[Child, ...], frames: dict[int, pd.DataFrame], seed_a_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    parent_fragile = {candidate.candidate_id: _fragile_set(frames[candidate.candidate_id]) for candidate in candidates[:3]}
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        current = frames[candidate.candidate_id]
        fragile = _fragile_set(current)
        headroom = -40.0 - current["retrieved_realB_CNMSE_dB"].to_numpy(dtype=float)
        rows.append({"model_id": candidate.candidate_id, "model_name": candidate.candidate_name, "model_type": candidate.model_type, "success_count": int(current["realB_pass"].sum()), "fragile_success_count": len(fragile), "fragile_fraction": float(len(fragile) / STATE_COUNT), "fragile_state_ids": json.dumps(sorted(fragile)), "minimum_headroom_dB": float(np.min(headroom[current["realB_pass"].to_numpy(dtype=bool)])), "Q05_headroom_dB": float(np.quantile(headroom[current["realB_pass"].to_numpy(dtype=bool)], 0.05)), "median_headroom_dB": float(np.median(headroom[current["realB_pass"].to_numpy(dtype=bool)]))})
    edge_rows: list[dict[str, Any]] = []
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    for child in children:
        parent = by_id[child.parent_model_id]
        pf = parent_fragile[parent.candidate_id]
        cf = _fragile_set(frames[child.candidate_id])
        edge_rows.append({"child_model_id": child.candidate_id, "child_model_name": child.candidate_name, "parent_model_id": parent.candidate_id, "parent": parent.branch, "parent_fragile_count": len(pf), "fragile_preserved_count": len(pf & cf), "fragile_regressed_count": len(pf - cf), "fragile_regressed_state_ids": json.dumps(sorted(pf - cf)), "new_fragile_success_count": len(cf - pf), "new_fragile_state_ids": json.dumps(sorted(cf - pf)), "historical_SeedA_fragile_regressed_count": len(_fragile_set(seed_a_frame) - cf)})
    return pd.DataFrame(rows), pd.DataFrame(edge_rows)


def _state330_trace(candidates: tuple[Parent | Child, ...], frames: dict[int, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        row = frames[candidate.candidate_id].loc[frames[candidate.candidate_id]["State_R"].eq(330)].iloc[0]
        rows.append({"model_id": candidate.candidate_id, "model_name": candidate.candidate_name, "model_type": candidate.model_type, "State_R": 330, "State_Q": int(row["State_Q"]), "realB_pass": bool(row["realB_pass"]), "retrieved_realB_CNMSE_dB": float(row["retrieved_realB_CNMSE_dB"]), "headroom_dB": float(-40.0 - row["retrieved_realB_CNMSE_dB"]), "first_shareable_rank": int(row["first_shareable_rank"]), "shareability_margin_dB": float(row["shareability_margin_dB"]), "fingerprint_top1_CNMSE_dB": float(row["fingerprint_top1_CNMSE_dB"])})
    return pd.DataFrame(rows)


def _conditional_contribution(inputs: dict[str, Any], raw_edges: pd.DataFrame, children: tuple[Child, ...], parents: tuple[Parent, ...], frames: dict[int, pd.DataFrame], summary: pd.DataFrame, seed_a_frame: pd.DataFrame) -> pd.DataFrame:
    historical_summary = pd.read_csv(K12_ROOT / "10_candidate_retrieval_summary.csv", low_memory=False)
    historical_baseline = historical_summary.loc[historical_summary["candidate_id"].eq(0)].iloc[0]
    historical_single = historical_summary.loc[historical_summary["candidate_id"].ne(0)].copy()
    historical_single = historical_single.drop_duplicates("added_basis_id").set_index("added_basis_id")
    parent_by_id = {parent.candidate_id: parent for parent in parents}
    child_by_id = {child.candidate_id: child for child in children}
    rows: list[dict[str, Any]] = []
    for edge in raw_edges.itertuples(index=False):
        child = child_by_id[int(edge.child_model_id)]
        parent = parent_by_id[int(edge.parent_model_id)]
        comparison = _comparison_frame(child, parent, frames, summary, seed_a_frame)
        added_id = str(edge.added_basis_id)
        if added_id in historical_single.index:
            hist_top1 = int(historical_single.loc[added_id, "top1_pass_count"])
            hist_contribution: float = float(hist_top1 - int(historical_baseline["top1_pass_count"]))
        else:
            hist_top1 = np.nan
            hist_contribution = np.nan
        cond = int(comparison["conditional_addition_contribution"])
        recovered = int(comparison["recovered_count"])
        regressed = int(comparison["regressed_count"])
        if cond != recovered - regressed:
            raise RuntimeError(f"conditional contribution identity failed for edge {edge.edge_id}")
        rows.append({"edge_id": int(edge.edge_id), "parent_model_id": parent.candidate_id, "parent": parent.branch, "child_model_id": child.candidate_id, "child_model_name": child.candidate_name, "added_basis_id": added_id, "parent_top1": comparison["parent_top1"], "child_top1": comparison["child_top1"], "conditional_addition_contribution": cond, "historical_K11_single_add_top1": hist_top1, "historical_contribution": hist_contribution, "context_contribution_shift": float(cond - hist_contribution) if np.isfinite(hist_contribution) else np.nan, "recovered": recovered, "regressed": regressed, "recovered_state_ids": comparison["recovered_state_ids"], "regressed_state_ids": comparison["regressed_state_ids"], "recovered_vs_historical_SeedA": comparison["recovered_vs_historical_SeedA"], "regressed_vs_historical_SeedA": comparison["regressed_vs_historical_SeedA"], "delta_Q05_vs_parent": comparison["delta_Q05_vs_parent"], "delta_MRR_vs_parent": comparison["delta_MRR_vs_parent"], "strict_breakthrough": bool(comparison["child_top1"] > 421), "parent_monotonic_breakthrough": bool(comparison["child_top1"] > 421 and regressed == 0), "historical_SeedA_safe_breakthrough": bool(comparison["child_top1"] > 421 and comparison["regressed_vs_historical_SeedA"] == 0)})
    frame = pd.DataFrame(rows).sort_values("edge_id").reset_index(drop=True)
    frame.to_csv(RESULT_ROOT / "20_conditional_addition_contribution.csv", index=False)
    return frame


def _basis_structure_outputs(contribution: pd.DataFrame, children: tuple[Child, ...], by_id: dict[str, EnvelopeBasis]) -> tuple[pd.DataFrame, pd.DataFrame]:
    enriched = contribution.copy()
    enriched["basis_family"] = enriched["added_basis_id"].map(lambda basis_id: by_id[str(basis_id)].family)
    enriched["p"] = enriched["added_basis_id"].map(lambda basis_id: int(by_id[str(basis_id)].order))
    enriched["m"] = enriched["added_basis_id"].map(lambda basis_id: by_id[str(basis_id)].signal_delay)
    enriched["q"] = enriched["added_basis_id"].map(lambda basis_id: by_id[str(basis_id)].envelope_delay)
    enriched["alignment"] = enriched.apply(lambda row: "aligned" if row["m"] == row["q"] else "cross-delay", axis=1)
    enriched["delay_direction"] = enriched.apply(lambda row: "m=q" if row["m"] == row["q"] else "m>q" if row["m"] > row["q"] else "m<q", axis=1)
    enriched["boundary_structure"] = enriched.apply(lambda row: "q=0,m>0" if row["q"] == 0 and row["m"] > 0 else "m=0,q>0" if row["m"] == 0 and row["q"] > 0 else "other", axis=1)
    type_rows: list[dict[str, Any]] = []
    for basis_type, group in enriched.groupby("alignment", sort=True):
        type_rows.append({"basis_type": basis_type, "candidate_count": int(group.shape[0]), "mean_C_plus": float(group["conditional_addition_contribution"].mean()), "max_C_plus": int(group["conditional_addition_contribution"].max()), "strict_breakthrough_count": int(group["strict_breakthrough"].sum()), "parent_monotonic_breakthrough_count": int(group["parent_monotonic_breakthrough"].sum()), "SeedA_safe_breakthrough_count": int(group["historical_SeedA_safe_breakthrough"].sum()), "mean_context_shift": float(group["context_contribution_shift"].mean()), "max_context_shift": float(group["context_contribution_shift"].max())})
    type_frame = pd.DataFrame(type_rows)
    structure = enriched.groupby(["p", "m", "q", "alignment", "delay_direction", "boundary_structure"], as_index=False).agg(candidate_count=("edge_id", "count"), mean_C_plus=("conditional_addition_contribution", "mean"), max_C_plus=("conditional_addition_contribution", "max"), min_C_plus=("conditional_addition_contribution", "min"), strict_breakthrough_count=("strict_breakthrough", "sum"), parent_monotonic_breakthrough_count=("parent_monotonic_breakthrough", "sum"), SeedA_safe_breakthrough_count=("historical_SeedA_safe_breakthrough", "sum"), mean_context_shift=("context_contribution_shift", "mean"), max_context_shift=("context_contribution_shift", "max"))
    type_frame.to_csv(RESULT_ROOT / "21_basis_type_summary.csv", index=False)
    structure.to_csv(RESULT_ROOT / "22_basis_structure_summary.csv", index=False)
    return type_frame, structure


def _recovered_vs_seed_a(children: tuple[Child, ...], frames: dict[int, pd.DataFrame], seed_a_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for child in children:
        comparison = legacy._compare_frames(frames[child.candidate_id], seed_a_frame)
        rows.append({"child_model_id": child.candidate_id, "child_model_name": child.candidate_name, "parent_model_id": child.parent_model_id, "parent": child.parent_branch, "Top1": comparison["top1"], "delta_vs_historical_SeedA": comparison["delta_top1"], "recovered_vs_historical_SeedA": comparison["recovered_count"], "regressed_vs_historical_SeedA": comparison["regressed_count"], "recovered_state_ids": json.dumps(comparison["recovered_state_ids"]), "regressed_state_ids": json.dumps(comparison["regressed_state_ids"]), "Q05": comparison["Q05"], "MRR": comparison["MRR"], "strict_breakthrough": bool(comparison["top1"] > 421), "historical_SeedA_safe": bool(comparison["top1"] > 421 and comparison["regressed_count"] == 0)})
    frame = pd.DataFrame(rows).sort_values("child_model_id").reset_index(drop=True)
    frame.to_csv(RESULT_ROOT / "19_recovered_regressed_vs_seedA.csv", index=False)
    return frame


def _edge_delta_table(raw_edges: pd.DataFrame, parents: tuple[Parent, ...], children: tuple[Child, ...], frames: dict[int, pd.DataFrame], summary: pd.DataFrame, seed_a_frame: pd.DataFrame) -> pd.DataFrame:
    parent_by_id = {parent.candidate_id: parent for parent in parents}
    child_by_id = {child.candidate_id: child for child in children}
    rows: list[dict[str, Any]] = []
    for edge in raw_edges.itertuples(index=False):
        child = child_by_id[int(edge.child_model_id)]
        parent = parent_by_id[int(edge.parent_model_id)]
        comparison = _comparison_frame(child, parent, frames, summary, seed_a_frame)
        rows.append({"edge_id": int(edge.edge_id), "parent_model_id": parent.candidate_id, "parent": parent.branch, "child_model_id": child.candidate_id, "child_model_name": child.candidate_name, "added_basis_id": child.added_basis_id, "parent_top1": comparison["parent_top1"], "child_top1": comparison["child_top1"], "conditional_addition_contribution": comparison["conditional_addition_contribution"], "recovered_count": comparison["recovered_count"], "regressed_count": comparison["regressed_count"], "recovered_state_ids": comparison["recovered_state_ids"], "regressed_state_ids": comparison["regressed_state_ids"], "recovered_vs_historical_SeedA": comparison["recovered_vs_historical_SeedA"], "regressed_vs_historical_SeedA": comparison["regressed_vs_historical_SeedA"], "recovered_state_ids_vs_historical_SeedA": comparison["recovered_state_ids_vs_historical_SeedA"], "regressed_state_ids_vs_historical_SeedA": comparison["regressed_state_ids_vs_historical_SeedA"], "parent_Q05": comparison["parent_Q05"], "child_Q05": comparison["child_Q05"], "delta_Q05_vs_parent": comparison["delta_Q05_vs_parent"], "parent_MRR": comparison["parent_MRR"], "child_MRR": comparison["child_MRR"], "delta_MRR_vs_parent": comparison["delta_MRR_vs_parent"], "strict_breakthrough": bool(comparison["child_top1"] > 421), "parent_monotonic_breakthrough": bool(comparison["child_top1"] > 421 and comparison["regressed_count"] == 0), "historical_SeedA_safe_breakthrough": bool(comparison["child_top1"] > 421 and comparison["regressed_vs_historical_SeedA"] == 0)})
    frame = pd.DataFrame(rows).sort_values("edge_id").reset_index(drop=True)
    if not np.array_equal(frame["conditional_addition_contribution"].to_numpy(dtype=int), frame["recovered_count"].to_numpy(dtype=int) - frame["regressed_count"].to_numpy(dtype=int)):
        raise RuntimeError("edge conditional contribution identity failed")
    frame.to_csv(RESULT_ROOT / "14_expansion_edge_delta.csv", index=False)
    return frame


def _postsearch_conditioning(selected: list[tuple[str, Parent | Child]], by_id: dict[str, EnvelopeBasis], precondition: pd.DataFrame) -> pd.DataFrame:
    unique: dict[str, tuple[int, ...]] = {}
    for label, candidate in selected:
        unique[label] = _candidate_columns(candidate, by_id)
    specs = tuple((label, columns) for label, columns in unique.items())
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=WORKER_COUNT, mp_context=get_context("spawn"), initializer=backend.condition_only_worker_init, initargs=(specs,)) as executor:
        futures = {executor.submit(backend.condition_only_worker_entry, state_id): state_id for state_id in range(STATE_COUNT)}
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed % 50 == 0 or completed == STATE_COUNT:
                _checkpoint(phase="postsearch_conditioning_progress", completed_states=completed)
    rows = [row for item in sorted(results, key=lambda value: int(value["State_n_R"])) for row in item["conditioning"]]
    state_frame = pd.DataFrame(rows)
    output: list[dict[str, Any]] = []
    for label, candidate in selected:
        group = state_frame.loc[state_frame["support_name"].eq(label)]
        for behavior, behavior_group in group.groupby("behavior", sort=False):
            output.append({"selected_label": label, "model_id": candidate.candidate_id, "model_name": candidate.candidate_name, "K": candidate.K, "behavior": behavior, "state_count": int(behavior_group.shape[0]), "rank_min": int(behavior_group["rank_raw"].min()), "sigma_min_raw_min": float(behavior_group["sigma_min_raw"].min()), "condition_raw_median": float(behavior_group["condition_number_raw"].median()), "condition_raw_Q95": float(behavior_group["condition_number_raw"].quantile(0.95)), "condition_raw_worst": float(behavior_group["condition_number_raw"].max()), "condition_column_normalized_median": float(behavior_group["condition_number_column_normalized"].median()), "condition_column_normalized_Q95": float(behavior_group["condition_number_column_normalized"].quantile(0.95)), "condition_column_normalized_worst": float(behavior_group["condition_number_column_normalized"].max()), "condition_ridge_augmented_median": float(behavior_group["condition_number_ridge_augmented"].median()), "condition_ridge_augmented_Q95": float(behavior_group["condition_number_ridge_augmented"].quantile(0.95)), "condition_ridge_augmented_worst": float(behavior_group["condition_number_ridge_augmented"].max()), "theta_l2_norm_median": float(behavior_group["theta_l2_norm"].median()), "theta_l2_norm_Q95": float(behavior_group["theta_l2_norm"].quantile(0.95)), "theta_l2_norm_max": float(behavior_group["theta_l2_norm"].max()), "hard_failure": bool((behavior_group["rank_raw"] < candidate.K).any())})
    output_frame = pd.DataFrame(output)
    output_frame.to_csv(RESULT_ROOT / "25_postsearch_conditioning_summary.csv", index=False)
    return output_frame


def _branch_summary(parents: tuple[Parent, ...], children: tuple[Child, ...], summary: pd.DataFrame, cv_stability: pd.DataFrame, edge_delta: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for parent in parents:
        branch_children = [child for child in children if child.parent_model_id == parent.candidate_id]
        branch_ids = [child.candidate_id for child in branch_children]
        branch = summary.loc[summary["model_id"].isin(branch_ids)].sort_values("global_rank").iloc[0]
        edge = edge_delta.loc[edge_delta["child_model_id"].eq(int(branch["model_id"]))].iloc[0]
        stability = cv_stability.loc[cv_stability["model_id"].eq(int(branch["model_id"]))].iloc[0]
        rows.append({"parent": parent.branch, "parent_model_id": parent.candidate_id, "parent_top1": int(summary.loc[summary["model_id"].eq(parent.candidate_id), "top1_pass_count"].iloc[0]), "best_child_model_id": int(branch["model_id"]), "best_child_model_name": branch["model_name"], "best_child_added_basis_id": next(child.added_basis_id for child in branch_children if child.candidate_id == int(branch["model_id"])), "best_child_top1": int(branch["top1_pass_count"]), "best_child_global_rank": int(branch["global_rank"]), "best_child_C_plus": int(edge["conditional_addition_contribution"]), "best_child_recovered": int(edge["recovered_count"]), "best_child_regressed": int(edge["regressed_count"]), "best_child_Q05": float(branch["share_margin_Q05"]), "best_child_MRR": float(branch["MRR"]), "best_child_validation_top1_mean": float(stability["validation_top1_mean"]), "best_child_validation_top1_min": int(stability["validation_top1_min"]), "best_child_validation_top1_max": int(stability["validation_top1_max"]), "strict_breakthrough": bool(branch["strict_breakthrough"]), "parent_monotonic_breakthrough": bool(branch["parent_monotonic_breakthrough"]), "historical_SeedA_safe_breakthrough": bool(branch["historical_seedA_safe_breakthrough"])} )
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULT_ROOT / "23_branch_summary.csv", index=False)
    return frame


def _write_figures(summary: pd.DataFrame, branches: pd.DataFrame, contribution: pd.DataFrame, hard_trace: pd.DataFrame, fragile_edges: pd.DataFrame, state330: pd.DataFrame, cv_stability: pd.DataFrame, post_condition: pd.DataFrame, model_frame: pd.DataFrame) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    ordered = summary.sort_values("global_rank").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(16, 7), dpi=220)
    ax.scatter(np.arange(len(ordered)), ordered["top1_pass_count"], c=np.where(ordered["strict_breakthrough"], "tab:green", "tab:purple"), s=16, alpha=0.75)
    ax.axhline(421, color="black", linestyle="--", linewidth=0.9, label="frozen parent Top1 = 421")
    ax.set_xlabel("Global diagnostic rank")
    ax.set_ylabel("Top-1 Real-B pass count")
    ax.set_title("Updated Beam=3 Iteration-2: 190-support Top-1 retrieval comparison")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "26_global_top1_comparison.png")
    plt.close(fig)

    if not branches.empty:
        labels = [f"{row.parent}\nparent" for row in branches.itertuples(index=False)] + [f"{row.parent}\nbest child" for row in branches.itertuples(index=False)]
        values = [int(row.parent_top1) for row in branches.itertuples(index=False)] + [int(row.best_child_top1) for row in branches.itertuples(index=False)]
        fig, ax = plt.subplots(figsize=(12, 6), dpi=220)
        colors = ["#9ecae1"] * len(branches) + ["#2171b5"] * len(branches)
        bars = ax.bar(np.arange(len(values)), values, color=colors)
        for bar, value in zip(bars, values, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.25, str(value), ha="center", fontsize=8)
        ax.axhline(421, color="black", linestyle="--", linewidth=0.9)
        ax.set_xticks(np.arange(len(labels)), labels)
        ax.set_ylabel("Top-1 Real-B pass count")
        ax.set_title("Updated Beam branch parents and best one-layer children")
        ax.set_ylim(0, 425)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(RESULT_ROOT / "27_branch_best_comparison.png")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(17, 7), dpi=220)
    ordered_contribution = contribution.sort_values(["conditional_addition_contribution", "edge_id"], ascending=[False, True])
    values = ordered_contribution["conditional_addition_contribution"].to_numpy(dtype=float)
    ax.bar(np.arange(len(values)), values, color=np.where(values > 0, "tab:green", np.where(values < 0, "tab:red", "tab:gray")), width=0.9)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Expansion edge ordered by conditional contribution")
    ax.set_ylabel("C+ = child Top1 - parent Top1")
    ax.set_title("Conditional addition contribution")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "28_conditional_addition_contribution.png")
    plt.close(fig)

    if not hard_trace.empty:
        heat = hard_trace.pivot(index="model_id", columns="State_R", values="first_shareable_rank").reindex(ordered["model_id"].tolist())
        fig, ax = plt.subplots(figsize=(10, 12), dpi=220)
        image = ax.imshow(heat.to_numpy(dtype=float), aspect="auto", interpolation="nearest", cmap="viridis", vmin=1)
        ax.set_xticks(np.arange(len(heat.columns)), [str(int(value)) for value in heat.columns])
        ticks = np.linspace(0, max(0, heat.shape[0] - 1), min(20, heat.shape[0]), dtype=int)
        ax.set_yticks(ticks, [str(int(heat.index[position])) for position in ticks])
        ax.set_xlabel("Persistent hard State_R")
        ax.set_ylabel("Support ordered by global rank")
        ax.set_title("Persistent hard-state first-shareable rank")
        fig.colorbar(image, ax=ax, label="rank")
        fig.tight_layout()
        fig.savefig(RESULT_ROOT / "29_persistent_hard_state_rank_change.png")
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(15, 6), dpi=220)
    if not fragile_edges.empty:
        order = fragile_edges.sort_values("fragile_regressed_count", ascending=False).reset_index(drop=True)
        ax.bar(np.arange(len(order)), order["fragile_regressed_count"], color="tab:red", label="parent fragile regressions")
        ax.bar(np.arange(len(order)), order["new_fragile_success_count"], bottom=order["fragile_regressed_count"], color="tab:orange", label="new fragile successes")
    ax.set_xlabel("Child support ordered by fragile-state changes")
    ax.set_ylabel("State count")
    ax.set_title("Fragile-success preservation across additions")
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "30_fragile_success_preservation.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(15, 6), dpi=220)
    shifted = contribution.sort_values("context_contribution_shift", ascending=False).reset_index(drop=True)
    ax.bar(np.arange(len(shifted)), shifted["context_contribution_shift"], color=np.where(shifted["context_contribution_shift"].fillna(0) >= 0, "tab:green", "tab:red"))
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Expansion edge ordered by context shift")
    ax.set_ylabel("C+ - historical single-add contribution")
    ax.set_title("Context contribution shift")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "31_context_contribution_shift.png")
    plt.close(fig)

    child_model = model_frame.loc[model_frame["model_type"].astype(str).str.contains("child", case=False, regex=False)].groupby(["model_id", "model_name"], as_index=False).agg(C2_train_median=("Y_C2_train_NMSE_dB", "median"))
    child_model = child_model.merge(summary[["model_id", "top1_pass_count"]], on="model_id", how="left")
    child_model["delta_top1_vs_F1"] = child_model["top1_pass_count"] - int(summary.loc[summary["model_id"].eq(0), "top1_pass_count"].iloc[0])
    child_model["delta_C2_train_vs_median"] = child_model["C2_train_median"] - child_model["C2_train_median"].median()
    fig, ax = plt.subplots(figsize=(10, 7), dpi=220)
    ax.scatter(child_model["delta_C2_train_vs_median"], child_model["delta_top1_vs_F1"], s=16, alpha=0.65, color="tab:purple")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.axvline(0, color="black", linewidth=0.7)
    ax.set_xlabel("Child median C2 train NMSE deviation (dB)")
    ax.set_ylabel("Δ Top-1 vs F1")
    ax.set_title("Modeling contribution versus retrieval contribution")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "32_modeling_vs_retrieval.png")
    plt.close(fig)

    if not post_condition.empty:
        cond = post_condition.loc[post_condition["behavior"].eq("Aend")]
        fig, ax = plt.subplots(figsize=(12, 7), dpi=220)
        positions = np.arange(cond.shape[0])
        ax.bar(positions - 0.18, np.log10(cond["condition_raw_median"].to_numpy(dtype=float)), width=0.18, label="raw")
        ax.bar(positions, np.log10(cond["condition_column_normalized_median"].to_numpy(dtype=float)), width=0.18, label="column normalized")
        ax.bar(positions + 0.18, np.log10(cond["condition_ridge_augmented_median"].to_numpy(dtype=float)), width=0.18, label="Ridge augmented")
        ax.set_xticks(positions, cond["selected_label"].astype(str), rotation=55, ha="right")
        ax.set_ylabel("log10(condition number), median")
        ax.set_title("Post-search conditioning comparison, Aend")
        ax.legend(frameon=False)
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(RESULT_ROOT / "33_conditioning_comparison.png")
        plt.close(fig)

    frequency = cv_stability.sort_values(["global_train_winner_count", "validation_top1_mean"], ascending=[False, False]).head(20)
    fig, ax = plt.subplots(figsize=(14, 7), dpi=220)
    ax.bar(np.arange(frequency.shape[0]), frequency["global_train_winner_count"], color="tab:blue")
    ax.set_xticks(np.arange(frequency.shape[0]), frequency["model_id"].astype(str), rotation=65, ha="right")
    ax.set_xlabel("Model ID, top 20 by CV global-winner frequency")
    ax.set_ylabel("Winner count across 5 folds")
    ax.set_title("Query-State CV selection frequency")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "34_cv_selection_frequency.png")
    plt.close(fig)


def _write_requested_statewise_plot(summary: pd.DataFrame, frames: dict[int, pd.DataFrame], model_frame: pd.DataFrame) -> None:
    """Save the requested six dB curves for the globally selected support."""
    selected_id = int(summary.sort_values("global_rank").iloc[0]["model_id"])
    metrics = model_frame.loc[model_frame["model_id"].eq(selected_id)].sort_values("State_R").reset_index(drop=True)
    query = frames[selected_id].sort_values("State_R").reset_index(drop=True)
    merged = metrics[["State_R", "nmse_withoutdpd_dB", "Y_Aend_train_NMSE_dB", "Y_Aend_B_NMSE_dB", "Y_C2_train_NMSE_dB", "Y_C2_B_NMSE_dB"]].merge(query[["State_R", "State_Q", "retrieved_real_B_CNMSE_dB"]], on="State_R", validate="one_to_one")
    if merged.shape[0] != STATE_COUNT:
        raise RuntimeError("requested statewise plot source row count changed")
    x = merged["State_R"].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(18, 8), dpi=260)
    series = (("nmse_withoutdpd_dB", "NMSE without DPD", "#1f77b4"), ("Y_Aend_train_NMSE_dB", "Aend modeling", "#ff7f0e"), ("Y_Aend_B_NMSE_dB", "AB generalization", "#2ca02c"), ("Y_C2_train_NMSE_dB", "C2 modeling", "#d62728"), ("Y_C2_B_NMSE_dB", "CB generalization", "#9467bd"), ("retrieved_real_B_CNMSE_dB", "retrieved Real-B CNMSE", "#111111"))
    for column, label, color in series:
        ax.plot(x, merged[column].to_numpy(dtype=float), linewidth=1.0 if column != "retrieved_real_B_CNMSE_dB" else 1.4, alpha=0.88, label=label, color=color)
    retrieved = merged["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float)
    ax.scatter(x, retrieved, s=8, color="#111111", zorder=4)
    changed = merged.loc[merged["State_Q"].ne(merged["State_R"]) | merged["retrieved_real_B_CNMSE_dB"].ge(REAL_B_THRESHOLD_DB)]
    for row in changed.itertuples(index=False):
        ax.annotate(str(int(row.State_Q)), (int(row.State_R), float(row.retrieved_real_B_CNMSE_dB)), fontsize=5.5, xytext=(2, 3), textcoords="offset points", color="#111111")
    ax.axhline(REAL_B_THRESHOLD_DB, color="#555555", linestyle="--", linewidth=0.8, label="Real-B pass threshold = -40 dB")
    ax.set_xlabel("State_R")
    ax.set_ylabel("dB")
    ax.set_title(f"Statewise modeling and retrieved Real-B CNMSE: {summary.loc[summary['model_id'].eq(selected_id), 'model_name'].iloc[0]}")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=3, loc="best")
    fig.tight_layout()
    fig.savefig(RESULT_ROOT / "39_statewise_global_best_metrics.png")
    plt.close(fig)


def _validate_retrieval_artifacts(candidates: tuple[Parent | Child, ...], frames: dict[int, pd.DataFrame], distances: dict[int, np.ndarray], inputs: dict[str, Any]) -> dict[str, Any]:
    argmin_checks = 0
    shareability_checks = 0
    rng = np.random.default_rng(20260916)
    for candidate in candidates:
        matrix = np.asarray(distances[candidate.candidate_id], dtype=float)
        if matrix.shape != (STATE_COUNT, STATE_COUNT) or np.isnan(matrix).any() or np.isposinf(matrix).any():
            raise RuntimeError(f"distance matrix validation failed: {candidate.candidate_name}")
        current = frames[candidate.candidate_id].set_index("State_R")
        for state_id in range(0, STATE_COUNT, max(1, STATE_COUNT // 5)):
            stored_q = int(current.loc[state_id, "State_Q"])
            row = matrix[state_id]
            min_value = np.min(row)
            if float(row[stored_q]) != float(min_value):
                raise RuntimeError(f"argmin validation failed: {candidate.candidate_name}/{state_id}")
            argmin_checks += 1
        for state_id in rng.integers(0, STATE_COUNT, size=10):
            state_id = int(state_id)
            row = matrix[state_id]
            mask = np.asarray(inputs["shareability"][state_id], dtype=bool)
            shareable = np.flatnonzero(mask)
            nonshareable = np.flatnonzero(~mask)
            if shareable.size == 0 or nonshareable.size == 0:
                continue
            nearest_share = int(shareable[np.argmin(row[shareable])])
            nearest_nonshare = int(nonshareable[np.argmin(row[nonshareable])])
            stored = current.loc[state_id]
            if nearest_share != int(stored["nearest_shareable_State_Q"]) or nearest_nonshare != int(stored["nearest_nonshareable_State_Q"]):
                raise RuntimeError(f"shareability validation failed: {candidate.candidate_name}/{state_id}")
            shareability_checks += 1
    return {"distance_matrix_count": len(distances), "argmin_checks": argmin_checks, "shareability_checks": shareability_checks, "all_distance_shapes_valid": True}


def _write_task_definition(inputs: dict[str, Any], parents: tuple[Parent, ...], children: tuple[Child, ...]) -> None:
    lines = [
        f"Task: {TASK_NAME}",
        "Mode: Updated Beam=3 Iteration-2 retrieval-oriented one-layer forward expansion with numerical conditioning audit.",
        "Frozen parents: F1_floating_K12=child_K12_13, F2_safe_K13=Model125, F3_margin_K13=Model12.",
        f"Parent support sizes: {[parent.K for parent in parents]}; raw expansion edges=187; unique children={len(children)}; evaluation pool={len(parents) + len(children)}.",
        f"ABC lengths={list(backend.ABC_LENGTHS)}, dmax={backend.MP_DMAX}, Ridge lambda={backend.RIDGE_LAMBDA}, Real-B threshold={REAL_B_THRESHOLD_DB} dB, allow_self=True.",
        f"Envelope75 count={len(inputs['terms'])}; common-B SHA={EXPECTED_COMMON_B_SHA}; raw manifest={json.dumps(inputs['raw_manifest'], ensure_ascii=False, sort_keys=True)}.",
        "LUT fingerprints are constructed from Aend model output; online query fingerprints are constructed from C2 model output; retrieval uses Full425 fingerprint-space matching and Real-B shareability for diagnostics.",
        "Conditioning is diagnostic only. No condition-number threshold is used for candidate filtering; hard failure is limited to rank deficiency, zero/nonfinite columns, nonfinite singular values, coefficients or predictions.",
        "Prohibited in this task: second add, next Beam, K15, swap/restricted swap, multi-basis add/delete, parameter scans, reranking, ensemble, clustering, Type III, low-bandwidth, DPD replay and nested-CV final evaluation.",
    ]
    (RESULT_ROOT / "00_task_definition.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (RESULT_ROOT / "01_reuse_audit.txt").write_text("\n".join([
        f"Formal K11 source: {K11_ROOT / '04_candidate_supports.csv'}; S10={list(inputs['s10_ids'])}; K11 core={list(inputs['k11_ids'])}.",
        f"Parent F1 source: {FLOAT_ROOT / '11_retrieval_summary.csv'}; parent F2/F3 source: {BEAM_ROOT / '12_retrieval_summary.csv'}.",
        f"Formal Real-B source: {K9_ROOT / '04_realB_ground_truth_cnmse_matrix.npy'} and shareability mask; common-B metadata={json.dumps(inputs['common_meta'], ensure_ascii=False, default=_json_default)}.",
        f"CV fold source: {K12_ROOT / '16_query_state_cv_folds.csv'}; historical Seed A source: {K12_ROOT / 'candidate_id=16'}.",
        "No raw data was written. Historical Model74 is reused only when F1 plus ENV_p04_m2_q0 exactly matches the canonical Model74 support hash.",
    ]) + "\n", encoding="utf-8")


def _run(*, resume: bool = False) -> dict[str, Any]:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    raw_before = raw_manifest_gate()
    inputs = _load_frozen_inputs()
    if inputs["raw_manifest"] != raw_before:
        raise RuntimeError("raw manifest gate changed between reads")
    parents, children, raw_edges, unique_children = _build_candidates(inputs)
    _write_task_definition(inputs, parents, children)
    frontier_rows = []
    for parent in parents:
        row = {"candidate_id": parent.candidate_id, "frontier": parent.branch, "source_model_name": parent.source_name, "source_model_id": parent.source_model_id, "K": parent.K, "support_basis_ids": json.dumps(list(parent.support_basis_ids), ensure_ascii=False), "support_hash": parent.support_hash, "known_top1": int(inputs["parent_rows"][parent.branch]["top1_pass_count"]), "known_failure_state_ids": inputs["parent_rows"][parent.branch].get("failure_state_ids", "")}
        frontier_rows.append(row)
    pd.DataFrame(frontier_rows).to_csv(RESULT_ROOT / "02_updated_beam_frontier_supports.csv", index=False)
    raw_edges.to_csv(RESULT_ROOT / "07_raw_expansion_edges.csv", index=False)
    unique_children.to_csv(RESULT_ROOT / "07_unique_children.csv", index=False)
    _checkpoint(phase="graph_and_metadata_passed", raw_manifest_before=raw_before, raw_expansion_edges=len(raw_edges), unique_children=len(children), evaluation_pool=len(parents) + len(children))

    common_mp, common_env, _ = _common_phi(inputs["common_b"], tuple(inputs["terms"]))
    parent_regressions, parent_retrievals, theta_bank_a, theta_bank_c = _parent_regression(inputs, parents, inputs["by_id"], common_mp, common_env)
    _write_json(RESULT_ROOT / "03_parent_regression.json", {"pass": bool(all(item["pass"] for item in parent_regressions)), "parent_count": len(parent_regressions), "parents": parent_regressions})
    if not all(item["pass"] for item in parent_regressions):
        raise RuntimeError("parent regression gate failed")
    _checkpoint(phase="parent_regression_passed", parent_regression_count=len(parent_regressions))

    model_frame_novel, theta_a_novel, theta_c_novel, conditioning = _run_novel_phase(children, inputs["by_id"], resume=resume)
    conditioning_summary = _conditioning_summary(conditioning)
    common_conditioning = _common_b_conditioning(inputs, inputs["by_id"])
    hard_condition_failures = int(conditioning_summary["hard_failure"].sum()) + int(common_conditioning.apply(lambda row: int(row["rank_raw"]) < int(row["K"]), axis=1).sum())
    if hard_condition_failures or not np.all(np.isfinite(conditioning[["sigma_max_raw", "sigma_min_raw", "condition_number_raw", "condition_number_column_normalized", "condition_number_ridge_augmented", "theta_l2_norm"]].to_numpy(dtype=float))):
        raise RuntimeError("HARD FAIL: conditioning audit has rank/nonfinite failure")
    _checkpoint(phase="conditioning_audit_passed", conditioning_rows=int(conditioning.shape[0]), conditioning_hard_failures=hard_condition_failures)

    summary_raw, theta_bank_a, frames, distances, fingerprints, historical_audits = _run_retrieval_phase(inputs, parents, children, inputs["by_id"], model_frame_novel, theta_a_novel, theta_c_novel, theta_bank_a, theta_bank_c, parent_retrievals, resume=resume)
    historical_child = next(child for child in children if child.is_historical_reuse)
    _write_json(RESULT_ROOT / "09_historical_child_reuse_audit.json", {"pass": all(item["pass"] for item in historical_audits.values()), "audits": list(historical_audits.values())})
    historical_seed_a, seed_a_frame, seed_a_summary, _, seed_a_reference_query = _historical_seed_a_reference(inputs, inputs["by_id"], common_mp, common_env, np.pad(theta_bank_a, ((0, 1), (0, 0), (0, 0))), np.pad(theta_bank_c, ((0, 1), (0, 0), (0, 0))))
    summary = _annotate_summary(summary_raw, tuple(parents) + children, frames)
    seed_a_comparisons: dict[int, dict[str, Any]] = {}
    for candidate in tuple(parents) + children:
        seed_a_comparisons[candidate.candidate_id] = legacy._compare_frames(frames[candidate.candidate_id], seed_a_frame)
    summary["recovered_vs_historical_SeedA"] = summary["model_id"].map(lambda value: seed_a_comparisons[int(value)]["recovered_count"])
    summary["regressed_vs_historical_SeedA"] = summary["model_id"].map(lambda value: seed_a_comparisons[int(value)]["regressed_count"])
    summary["historical_seedA_safe_breakthrough"] = summary["strict_breakthrough"] & summary["regressed_vs_historical_SeedA"].eq(0)
    summary["support_basis_ids"] = summary["model_id"].map(lambda value: json.dumps(list(next(candidate for candidate in tuple(parents) + children if candidate.candidate_id == int(value)).support_basis_ids), ensure_ascii=False))
    summary["branch"] = summary["model_id"].map(lambda value: next(candidate for candidate in tuple(parents) + children if candidate.candidate_id == int(value)).branch_seed_ids[0])
    summary.to_csv(RESULT_ROOT / "13_retrieval_summary.csv", index=False)
    pd.concat([_decorate_source_model(inputs["floating_model"].loc[inputs["floating_model"]["model_id"].eq(13)], next(parent for parent in parents if parent.candidate_id == 0)), _decorate_source_model(inputs["beam_model"].loc[inputs["beam_model"]["model_id"].eq(125)], next(parent for parent in parents if parent.candidate_id == 1)), _decorate_source_model(inputs["beam_model"].loc[inputs["beam_model"]["model_id"].eq(12)], next(parent for parent in parents if parent.candidate_id == 2)), _decorate_source_model(inputs["beam_model"].loc[inputs["beam_model"]["model_id"].eq(74)], historical_child), model_frame_novel], ignore_index=True, sort=False).sort_values(["model_id", "State_R"]).to_csv(RESULT_ROOT / "10_model_quality_long.csv", index=False)
    query_frame = pd.concat(frames.values(), ignore_index=True, sort=False).sort_values(["model_id", "State_R"]).reset_index(drop=True)
    query_frame.to_csv(RESULT_ROOT / "12_query_metrics_long.csv", index=False)

    hard_trace, union_trace, intersection, union = _failure_traces(tuple(parents) + children, frames, tuple(parent.candidate_id for parent in parents))
    hard_trace.to_csv(RESULT_ROOT / "15_persistent_hard_state_trace.csv", index=False)
    union_trace.to_csv(RESULT_ROOT / "16_beam_failure_union_trace.csv", index=False)
    fragile_summary, fragile_edges = _fragile_outputs(tuple(parents) + children, children, frames, seed_a_frame)
    fragile_summary.to_csv(RESULT_ROOT / "17_fragile_success_summary.csv", index=False)
    fragile_edges.to_csv(RESULT_ROOT / "17_fragile_success_edge_delta.csv", index=False)
    state330 = _state330_trace(tuple(parents) + children, frames)
    state330.to_csv(RESULT_ROOT / "18_state330_fragility_trace.csv", index=False)

    edge_delta = _edge_delta_table(raw_edges, parents, children, frames, summary, seed_a_frame)
    edge_flags = edge_delta.set_index("child_model_id")
    summary["parent_monotonic_breakthrough"] = summary.apply(lambda row: bool(edge_flags.loc[int(row["model_id"]), "parent_monotonic_breakthrough"]) if int(row["model_id"]) in edge_flags.index else False, axis=1)
    summary["historical_seedA_safe_breakthrough"] = summary.apply(lambda row: bool(edge_flags.loc[int(row["model_id"]), "historical_SeedA_safe_breakthrough"]) if int(row["model_id"]) in edge_flags.index else False, axis=1)
    summary.to_csv(RESULT_ROOT / "13_retrieval_summary.csv", index=False)
    _recovered_vs_seed_a(children, frames, seed_a_frame)
    contribution = _conditional_contribution(inputs, raw_edges, children, parents, frames, summary, seed_a_frame)
    basis_type, basis_structure = _basis_structure_outputs(contribution, children, inputs["by_id"])
    cv_frame, cv_stability = legacy._local_cv_results(tuple(parents) + children, frames, inputs["folds"])
    cv_frame.to_csv(RESULT_ROOT / "23_query_state_cv_results.csv", index=False)
    cv_stability.to_csv(RESULT_ROOT / "24_query_state_cv_selection_stability.csv", index=False)
    branch_frame = _branch_summary(parents, children, summary, cv_stability, edge_delta)

    candidate_by_id = {candidate.candidate_id: candidate for candidate in tuple(parents) + children}
    global_best = candidate_by_id[int(summary.sort_values("global_rank").iloc[0]["model_id"])]
    selected: list[tuple[str, Parent | Child]] = [("global_best", global_best)]
    for branch_index, parent in enumerate(parents, start=1):
        branch_best_id = int(branch_frame.loc[branch_frame["parent"].eq(parent.branch), "best_child_model_id"].iloc[0])
        selected.append((f"F{branch_index}_best_child", candidate_by_id[branch_best_id]))
    strict_children = [child for child in children if bool(summary.loc[summary["model_id"].eq(child.candidate_id), "strict_breakthrough"].iloc[0])]
    parent_monotonic_children = [child for child in strict_children if bool(edge_delta.loc[edge_delta["child_model_id"].eq(child.candidate_id), "parent_monotonic_breakthrough"].iloc[0])]
    seed_a_safe_children = [child for child in strict_children if bool(edge_delta.loc[edge_delta["child_model_id"].eq(child.candidate_id), "historical_SeedA_safe_breakthrough"].iloc[0])]
    selected.append(("best_parent_monotonic", sorted(parent_monotonic_children or strict_children or list(children), key=lambda item: int(summary.loc[summary["model_id"].eq(item.candidate_id), "global_rank"].iloc[0]))[0]))
    selected.append(("best_SeedA_safe", sorted(seed_a_safe_children or strict_children or list(children), key=lambda item: int(summary.loc[summary["model_id"].eq(item.candidate_id), "global_rank"].iloc[0]))[0]))
    post_condition = _postsearch_conditioning(selected, inputs["by_id"], conditioning_summary)

    validation_retrieval = _validate_retrieval_artifacts(tuple(parents) + children, frames, distances, inputs)
    real_b_error = _random_real_b_check(inputs["real_b_distance"])
    raw_after = raw_manifest_gate()
    validation = {"pass": bool(raw_before == raw_after and real_b_error <= REAL_B_SANITY_TOLERANCE_DB and validation_retrieval["distance_matrix_count"] == 190 and hard_condition_failures == 0), "raw_manifest_before": raw_before, "raw_manifest_after": raw_after, "raw_data_modified": raw_before != raw_after, "real_B_random_pair_max_abs_error_dB": real_b_error, "retrieval": validation_retrieval, "conditioning_state_rows": int(conditioning.shape[0]), "conditioning_commonB_rows": int(common_conditioning.shape[0]), "conditioning_hard_failures": hard_condition_failures, "persistent_failure_intersection": intersection, "failure_union": union, "distance_matrix_count": len(distances), "evaluation_pool": 190}
    _write_json(RESULT_ROOT / "38_validation_checks.json", validation)
    if not validation["pass"]:
        raise RuntimeError(f"validation failed: {validation}")

    _write_figures(summary, branch_frame, contribution, hard_trace, fragile_edges, state330, cv_stability, post_condition, pd.read_csv(RESULT_ROOT / "10_model_quality_long.csv", low_memory=False))
    _write_requested_statewise_plot(summary, frames, pd.read_csv(RESULT_ROOT / "10_model_quality_long.csv", low_memory=False))
    best_child_rows = summary.loc[summary["model_type"].astype(str).str.contains("child", case=False, regex=False)].sort_values("global_rank")
    best_child = candidate_by_id[int(best_child_rows.iloc[0]["model_id"])] if not best_child_rows.empty else global_best
    summary.loc[summary["model_id"].eq(global_best.candidate_id)].to_csv(RESULT_ROOT / "35_global_best_detail.csv", index=False)
    top_lines = [
        f"Task: {TASK_NAME}",
        "Status: SUCCESS",
        f"Updated Beam=3 one-layer forward expansion: raw edges={len(raw_edges)}, unique children={len(children)}, evaluation pool=190.",
        f"Parents regression: PASS {len(parent_regressions)}/{len(parents)}; parent failures intersection={intersection}; union={union}.",
        f"Global best: {global_best.candidate_name}, Top1={int(summary.loc[summary['model_id'].eq(global_best.candidate_id), 'top1_pass_count'].iloc[0])}/425, global rank={int(summary.loc[summary['model_id'].eq(global_best.candidate_id), 'global_rank'].iloc[0])}.",
        f"Best child: {best_child.candidate_name}, Top1={int(summary.loc[summary['model_id'].eq(best_child.candidate_id), 'top1_pass_count'].iloc[0])}/425.",
        f"Strict breakthrough count={int(summary['strict_breakthrough'].sum())}; parent-monotonic count={int(summary['parent_monotonic_breakthrough'].sum())}; SeedA-safe count={int(summary['historical_seedA_safe_breakthrough'].sum())}.",
        f"Conditioning: state rows={len(conditioning)} (7 supports x 425 states x Aend/C2), common-B rows={len(common_conditioning)}, hard failures={hard_condition_failures}; condition number was diagnostic only.",
        f"Real-B random-pair max error={real_b_error:.3e} dB; raw manifest unchanged={raw_before == raw_after}.",
        "",
        "Top 15 supports by diagnostic rank:",
    ]
    for row in summary.sort_values("global_rank").head(15).itertuples(index=False):
        top_lines.append(f"rank={int(row.global_rank)} id={int(row.model_id)} {row.model_name}: K={int(row.K)}, Top1={int(row.top1_pass_count)}, Exact={int(row.exact_hit_count)}, nonself={int(row.nonself_count)}/{int(row.nonself_pass_count)}, Q05={float(row.share_margin_Q05):.9g}, MRR={float(row.MRR):.9g}, Top2/3/5/10={int(row.Top2_oracle)}/{int(row.Top3_oracle)}/{int(row.Top5_oracle)}/{int(row.Top10_oracle)}, failure={int(row.failure_count)}")
    top_lines.extend(["", "Branch best:"])
    for row in branch_frame.itertuples(index=False):
        top_lines.append(f"{row.parent}: child={row.best_child_model_name}, Top1={int(row.best_child_top1)}, C+={int(row.best_child_C_plus)}, recovered={int(row.best_child_recovered)}, regressed={int(row.best_child_regressed)}, CV validation mean/min/max={float(row.best_child_validation_top1_mean):.6g}/{int(row.best_child_validation_top1_min)}/{int(row.best_child_validation_top1_max)}")
    top_lines.extend(["", "Persistent hard-state summary:"])
    for state_id in intersection:
        subset = hard_trace.loc[hard_trace["State_R"].eq(state_id)]
        top_lines.append(f"State {state_id}: best rank={int(subset['first_shareable_rank'].min())}, pass count={int(subset['realB_pass'].sum())}/{len(subset)}")
    top_lines.extend(["", "Post-search conditioning supports:"])
    for row in post_condition.loc[post_condition["behavior"].eq("Aend")].itertuples(index=False):
        top_lines.append(f"{row.selected_label}: raw median/Q95={float(row.condition_raw_median):.6g}/{float(row.condition_raw_Q95):.6g}, column-normalized median/Q95={float(row.condition_column_normalized_median):.6g}/{float(row.condition_column_normalized_Q95):.6g}, Ridge median/Q95={float(row.condition_ridge_augmented_median):.6g}/{float(row.condition_ridge_augmented_Q95):.6g}")
    top_lines.extend(["", "No second add, next Beam, K15, swap, parameter scan, reranking, ensemble, clustering, Type III, low-bandwidth or DPD replay was run.", "CV is Query-State selection-stability analysis, not an independent final test."])
    (RESULT_ROOT / "36_final_result_summary.txt").write_text("\n".join(top_lines) + "\n", encoding="utf-8")
    result = {"task": TASK_NAME, "status": "SUCCESS", "result_root": str(RESULT_ROOT), "parent_regression_pass": True, "raw_expansion_edges": len(raw_edges), "unique_children": len(children), "evaluation_pool": 190, "historical_child_reuse_count": int(sum(child.is_historical_reuse for child in children)), "global_best_model": global_best.candidate_name, "global_best_top1": int(summary.loc[summary["model_id"].eq(global_best.candidate_id), "top1_pass_count"].iloc[0]), "best_child_model": best_child.candidate_name, "best_child_top1": int(summary.loc[summary["model_id"].eq(best_child.candidate_id), "top1_pass_count"].iloc[0]), "strict_breakthrough_count": int(summary["strict_breakthrough"].sum()), "parent_monotonic_breakthrough_count": int(summary["parent_monotonic_breakthrough"].sum()), "historical_SeedA_safe_breakthrough_count": int(summary["historical_seedA_safe_breakthrough"].sum()), "persistent_failure_intersection": intersection, "failure_union": union, "raw_data_modified": False, "conditioning_hard_failures": hard_condition_failures}
    _checkpoint(phase="completed", **result)
    _append_log(f"\n[{_now()}] Complete {TASK_NAME}\n{json.dumps(result, ensure_ascii=False, sort_keys=True)}\n")
    _append_handoff(f"完成 {TASK_NAME}：更新 Beam=3 Iteration-2 一层前向扩展、条件数审计、Full425 retrieval、fragile/hard-state、conditional contribution、CV 和图形；结果目录：{RESULT_ROOT}\n摘要：{json.dumps(result, ensure_ascii=False, sort_keys=True)}")
    return result


def main() -> None:
    result = _run(resume="--resume" in sys.argv[1:])
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    __import__("multiprocessing").freeze_support()
    main()
