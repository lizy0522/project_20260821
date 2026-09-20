# ruff: noqa: E402,E501,I001

"""Independent validation for the three-frontier floating backward task."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
from behavior_modeling.shared.basis_function_selection.envelope_dictionary import build_envelope_bank, build_envelope_dictionary  # noqa: E402
from behavior_fingerprint_retrieval.shared.envelope18_commonB_full_lut import top1_retrieval  # noqa: E402
from behavior_fingerprint_retrieval.shared.envelope19_c2endshared_commonB_full_lut import EXPECTED_COMMON_B_SHA, raw_manifest_gate, verify_common_b_contract  # noqa: E402
from data_management.shared import load_by_id  # noqa: E402
from retrieval_oriented_model_selection.self_hit_oriented.scenario_2.scenario_2_beam3_k13_floating_backward_retrieval_ablation_5b import run_scenario2_beam3_k13_floating_backward_retrieval_ablation_5b as runner  # noqa: E402
from signal_segmentation.shared import build_off_segments, build_partition_from_xin, get_ilc_pair, preprocess_full_pair  # noqa: E402

TASK_NAME = "scenario_2_beam3_k13_floating_backward_retrieval_ablation_5b"
RESULT_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /TASK_NAME
K9_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /"scenario_2_mp10_k9_retrieval_oriented_basis_ablation_5b"
K12_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /"scenario_2_k12_retrieval_oriented_forward_scan_5b"
BEAM_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /"scenario_2_smallbeam3_k13_retrieval_oriented_forward_expansion_5b"
STATE_COUNT = runner.STATE_COUNT
K12 = runner.K12
REPRESENTATIVE_STATES = (0, 42, 187, 325, 424)
THRESHOLD_DB = -40.0


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


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
    _assert(maximum <= 1e-9, f"Real-B random-pair error too large: {maximum:.3e} dB")
    return maximum


def _state_blocks(state_id: int, terms: tuple[Any, ...], core_index: int) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
    data = load_by_id(state_id)
    partition = build_partition_from_xin(np.asarray(data["xin"]))
    _assert((partition.n_a, partition.n_b, partition.n_c) == runner.backend.ABC_LENGTHS, f"State {state_id} ABC contract failed")
    history = np.asarray(data["xin_pd_ori_ilc"])
    a_pair = get_ilc_pair(data, int(history.shape[1] - 1))
    c_pair = get_ilc_pair(data, 1)
    aend = preprocess_full_pair(a_pair.input_full, a_pair.output_raw_full, partition, pair_type=a_pair.pair_type, iteration_index=a_pair.iteration_index, input_peak_normalization_factor=a_pair.input_peak_normalization_factor)
    c2 = preprocess_full_pair(c_pair.input_full, c_pair.output_raw_full, partition, pair_type=c_pair.pair_type, iteration_index=c_pair.iteration_index, input_peak_normalization_factor=c_pair.input_peak_normalization_factor)
    inputs = (aend["A"].input, aend["B"].input, c2["C"].input, c2["B"].input)
    cores: list[np.ndarray] = []
    envs: list[np.ndarray] = []
    for value in inputs:
        mp = runner.backend.historical_mp10.build_frozen_mp_basis(value)
        env = build_envelope_bank(value, terms)
        cores.append(np.column_stack((mp, env[:, core_index])).astype(np.complex128))
        envs.append(env)
    expected = ((12_286, runner.CORE_K), (4_913, runner.CORE_K), (7_371, runner.CORE_K), (4_913, runner.CORE_K))
    for core, shape in zip(cores, expected, strict=True):
        _assert(core.shape == shape and np.all(np.isfinite(core)), f"State {state_id} core Phi shape failed")
    return tuple(cores), tuple(envs)


def _check_phi_slicing(children: tuple[Any, ...], terms: tuple[Any, ...], core_index: int, common_b: np.ndarray) -> dict[str, Any]:
    novel = [child for child in children if not child.is_historical_reuse]
    chosen_ids = sorted(np.random.default_rng(20260916).choice([child.candidate_id for child in novel], size=20, replace=False).astype(int).tolist())
    by_id = {child.candidate_id: child for child in novel}
    common_mp = runner.backend.historical_mp10.build_frozen_mp_basis(common_b)
    common_env = build_envelope_bank(common_b, terms)
    common_core = np.column_stack((common_mp, common_env[:, core_index])).astype(np.complex128)
    checks = 0
    for state_id in REPRESENTATIVE_STATES:
        cores, envs = _state_blocks(state_id, terms, core_index)
        for child_id in chosen_ids:
            child = by_id[child_id]
            removed = int(child.removed_core_index)
            for core, env in zip(cores, envs, strict=True):
                parent_full = np.column_stack((core, env[:, child.extension_indices[0]], env[:, child.extension_indices[1]]))
                child_rebuilt = np.column_stack((core[:, tuple(index for index in range(runner.CORE_K) if index != removed)], env[:, child.extension_indices[0]], env[:, child.extension_indices[1]]))
                _assert(np.array_equal(np.delete(parent_full, removed, axis=1), child_rebuilt), f"{child.candidate_name} State {state_id} canonical Phi12 deletion mismatch")
                checks += 1
            parent_common = np.column_stack((common_core, common_env[:, child.extension_indices[0]], common_env[:, child.extension_indices[1]]))
            child_common = np.column_stack((common_core[:, tuple(index for index in range(runner.CORE_K) if index != removed)], common_env[:, child.extension_indices[0]], common_env[:, child.extension_indices[1]]))
            _assert(np.array_equal(np.delete(parent_common, removed, axis=1), child_common), f"{child.candidate_name} common-B deletion mismatch")
            checks += 1
    return {"random_novel_supports": 20, "representative_states": len(REPRESENTATIVE_STATES), "blocks_per_state": 5, "phi_slicing_checks": checks, "pass": True}


def _check_retrieval(summary: pd.DataFrame, query: pd.DataFrame, real_b: np.ndarray, shareability: np.ndarray) -> dict[str, Any]:
    argmin_checks = 0
    shareability_checks = 0
    for row in summary.sort_values("model_id").itertuples(index=False):
        distance = np.load(RESULT_ROOT / "distance_matrices" / str(row.distance_file))
        _assert(distance.shape == (STATE_COUNT, STATE_COUNT) and not np.isnan(distance).any() and not np.isposinf(distance).any(), f"{row.model_name} distance matrix failed")
        selected, _, _, _ = top1_retrieval(distance)
        frame = query.loc[query["model_id"].eq(int(row.model_id))].sort_values("State_R").reset_index(drop=True)
        _assert(frame.shape[0] == STATE_COUNT and np.array_equal(frame["State_Q"].to_numpy(dtype=int), selected), f"{row.model_name} State_Q mismatch")
        expected_pass = np.asarray([float(real_b[state_id, int(selected[state_id])]) < THRESHOLD_DB for state_id in range(STATE_COUNT)])
        _assert(np.array_equal(frame["realB_pass"].to_numpy(dtype=bool), expected_pass), f"{row.model_name} Real-B pass mismatch")
        for state_id in REPRESENTATIVE_STATES:
            order = np.lexsort((np.arange(STATE_COUNT), distance[state_id]))
            _assert(int(order[0]) == int(selected[state_id]), f"{row.model_name} argmin mismatch at State {state_id}")
            argmin_checks += 1
    rng = np.random.default_rng(20260916)
    model_ids = summary["model_id"].astype(int).to_numpy()
    for _ in range(200):
        model_id = int(rng.choice(model_ids))
        state_id = int(rng.integers(0, STATE_COUNT))
        row = summary.loc[summary["model_id"].eq(model_id)].iloc[0]
        distance = np.load(RESULT_ROOT / "distance_matrices" / str(row.distance_file))
        frame = query.loc[query["model_id"].eq(model_id)].sort_values("State_R").reset_index(drop=True)
        order = np.lexsort((np.arange(STATE_COUNT), distance[state_id]))
        share_idx = np.flatnonzero(shareability[state_id])
        nonshare_idx = np.flatnonzero(~shareability[state_id])
        _assert(share_idx.size > 0 and nonshare_idx.size > 0, "shareability partition is empty")
        share_order = share_idx[np.lexsort((share_idx, distance[state_id, share_idx]))]
        nonshare_order = nonshare_idx[np.lexsort((nonshare_idx, distance[state_id, nonshare_idx]))]
        expected_margin = float(distance[state_id, nonshare_order[0]] - distance[state_id, share_order[0]])
        stored = frame.iloc[state_id]
        _assert(abs(expected_margin - float(stored["shareability_margin_dB"])) <= 1e-12, f"{row.model_name} margin mismatch")
        first_rank = int(np.flatnonzero(shareability[state_id, order])[0] + 1)
        _assert(first_rank == int(stored["first_shareable_rank"]), f"{row.model_name} first rank mismatch")
        for k, column in ((2, "Top2_has_shareable"), (3, "Top3_has_shareable"), (5, "Top5_has_shareable"), (10, "Top10_has_shareable")):
            _assert(bool(shareability[state_id, order[:k]].any()) == bool(stored[column]), f"{row.model_name} {column} mismatch")
        shareability_checks += 1
    return {"distance_matrix_count": int(summary.shape[0]), "argmin_checks": argmin_checks, "shareability_checks": shareability_checks, "pass": True}


def _check_fragile(summary: pd.DataFrame, query: pd.DataFrame) -> dict[str, Any]:
    fragile_summary = pd.read_csv(RESULT_ROOT / "16_fragile_success_summary.csv")
    historical = pd.read_csv(RESULT_ROOT / "16_seedA_historical_fragile_success.csv")
    _assert(fragile_summary.shape[0] == 40 and historical.shape[0] == STATE_COUNT, "fragile output shape failed")
    _assert(int(historical["fragile_success"].sum()) == 9, "historical Seed A fragile count changed")
    pass_rows = query.loc[query["realB_pass"]].copy()
    sampled = pass_rows.sample(n=50, random_state=20260916)
    fragile_sets = {int(row.model_id): set(json.loads(row.fragile_state_ids)) for row in fragile_summary.itertuples(index=False)}
    for row in sampled.itertuples(index=False):
        headroom = -40.0 - float(row.retrieved_realB_CNMSE_dB)
        expected = bool(0 < headroom <= 1.0)
        _assert(expected == (int(row.State_R) in fragile_sets[int(row.model_id)]), f"fragile flag mismatch at model={row.model_id}, State={row.State_R}")
    return {"support_rows": int(fragile_summary.shape[0]), "historical_seedA_fragile_count": int(historical["fragile_success"].sum()), "random_fragile_checks": 50, "pass": True}


def main() -> None:
    raw = raw_manifest_gate()
    expected_raw = {"sha256": "910b06e4be6b60b1f4dc63409a3083eb328765347fbec3b89a97a41d6633a1e0", "file_count": 429, "mat_count": 427, "bytes": 2_258_448_137}
    _assert(raw == expected_raw, "raw manifest changed")
    common_b, common_meta = verify_common_b_contract()
    _assert(common_b.shape == (4_915,) and str(common_meta["sha256"]) == EXPECTED_COMMON_B_SHA, "common-B contract failed")
    terms = tuple(build_envelope_dictionary())
    core_ids, by_id, core_index, _ = runner._load_core(terms)
    references = runner._load_references(core_ids)
    _, _, frontiers, children, _ = runner._build_candidates(core_ids, by_id, references)
    _assert(len(frontiers) == 3 and len(children) == 37, "candidate graph reconstruction failed")
    graph = json.loads((RESULT_ROOT / "06_support_dedup_audit.json").read_text(encoding="utf-8"))
    _assert(graph["pass"] is True and graph["raw_deletion_edges"] == 39 and graph["unique_k12_children"] == 37 and graph["duplicate_occurrences"] == 2 and graph["multi_parent_children"] == 2 and graph["historical_reuse_count"] == 4 and graph["novel_k12_fit_count"] == 33, "support dedup audit failed")
    frontier_regression = json.loads((RESULT_ROOT / "03_frontier_regression.json").read_text(encoding="utf-8"))
    reuse_regression = json.loads((RESULT_ROOT / "07_historical_child_reuse_audit.json").read_text(encoding="utf-8"))
    _assert(frontier_regression["pass"] is True and len(frontier_regression["frontier_regressions"]) == 3, "frontier regression failed")
    _assert(reuse_regression["pass"] is True and reuse_regression["historical_child_count"] == 4, "historical child reuse failed")
    summary = pd.read_csv(RESULT_ROOT / "11_retrieval_summary.csv")
    query = pd.read_csv(RESULT_ROOT / "10_query_metrics_long.csv", low_memory=False)
    model = pd.read_csv(RESULT_ROOT / "08_model_quality_long.csv", low_memory=False)
    edges = pd.read_csv(RESULT_ROOT / "04_raw_deletion_edges.csv")
    unique = pd.read_csv(RESULT_ROOT / "05_unique_k12_children.csv")
    edge_delta = pd.read_csv(RESULT_ROOT / "12_deletion_edge_delta.csv")
    _assert(summary.shape[0] == 40 and model.shape[0] == 40 * STATE_COUNT and query.shape[0] == 40 * STATE_COUNT, "unified table row counts failed")
    _assert(edges.shape[0] == 39 and unique.shape[0] == 37 and edge_delta.shape[0] == 39, "edge/unique table shapes failed")
    with np.load(RESULT_ROOT / "09_coefficients_novel_k12.npz", allow_pickle=False) as data:
        theta_a = np.asarray(data["theta_Aend_padded"])
        theta_c = np.asarray(data["theta_C2_padded"])
    _assert(theta_a.shape == (33, STATE_COUNT, K12) and theta_c.shape == theta_a.shape, "novel coefficient shape failed")
    _assert(model["Aend_finite"].all() and model["C2_finite"].all(), "model finite gate failed")
    _assert(model.loc[model["model_type"].eq("K13_parent"), "Aend_rank"].eq(13).all() and model.loc[model["model_type"].eq("K12_child"), "Aend_rank"].eq(12).all(), "model rank gate failed")
    real_b = np.load(K9_ROOT / "04_realB_ground_truth_cnmse_matrix.npy")
    shareability = np.load(K9_ROOT / "05_realB_shareability_mask.npy")
    retrieval = _check_retrieval(summary, query, real_b, shareability)
    random_error = _random_real_b_check(real_b)
    phi = _check_phi_slicing(children, terms, core_index, common_b)
    fragile = _check_fragile(summary, query)
    _assert(int(summary.loc[summary["model_type"].eq("K12_child"), "Top1"].eq(421).sum()) == 1, "K12 Top1=421 count changed")
    _assert(int(edge_delta["beneficial_to_delete"].sum()) == 0 and int(edge_delta["strict_safe_compression"].sum()) == 0 and int(edge_delta["performance_preserving_but_reordered"].sum()) == 1 and int(edge_delta["necessary_in_parent_context"].sum()) == 38, "edge classification counts changed")
    best_child = edge_delta.loc[edge_delta["child_model_id"].eq(13) & edge_delta["parent_frontier_id"].eq("frontier_A_global_best")].iloc[0]
    _assert(int(best_child["child_top1"]) == 421 and int(best_child["recovered_count"]) == 1 and int(best_child["regressed_count"]) == 1 and bool(best_child["performance_preserving_but_reordered"]), "best K12 reordered child contract failed")
    overlap = pd.read_csv(RESULT_ROOT / "13_parent_failure_overlap.csv")
    _assert(overlap.loc[overlap["persistent_all3"], "State_R"].astype(int).tolist() == [189, 323, 340], "persistent failure intersection changed")
    _assert(overlap.loc[overlap["failure_count_across_3"].gt(0), "State_R"].astype(int).tolist() == [189, 190, 323, 330, 340, 346], "failure union changed")
    cv = pd.read_csv(RESULT_ROOT / "18_query_state_cv_results.csv")
    stability = pd.read_csv(RESULT_ROOT / "19_query_state_cv_selection_stability.csv")
    branch_cv = pd.read_csv(RESULT_ROOT / "20_frontier_branch_cv_summary.csv")
    _assert(cv.shape[0] == 200 and cv.groupby("fold").size().to_dict() == {0: 40, 1: 40, 2: 40, 3: 40, 4: 40}, "CV result shape failed")
    _assert(cv.groupby("fold")["train_is_global_winner"].sum().eq(1).all() and stability.shape[0] == 40 and branch_cv.shape[0] == 15, "CV winner/stability failed")
    checks = {"task_name": TASK_NAME, "raw_manifest": raw, "common_B": {"raw_length": int(common_b.size), "sha256": str(common_meta["sha256"]), "pass": True}, "frontier_regression": {"count": 3, "pass": True}, "support_graph": graph, "historical_reuse": reuse_regression, "novel_coefficient_shape": list(theta_a.shape), "model_quality_rows": int(model.shape[0]), "query_metric_rows": int(query.shape[0]), "retrieval": retrieval, "real_B_random_pair_count": 20, "real_B_random_pair_max_abs_error_dB": random_error, "phi_slicing": phi, "fragile_success": fragile, "edge_classification": {"beneficial": 0, "strict_safe_compression": 0, "performance_preserving_but_reordered": 1, "necessary_in_parent_context": 38, "pass": True}, "failure_overlap": {"persistent_intersection": [189, 323, 340], "union": [189, 190, 323, 330, 340, 346], "pass": True}, "cv": {"rows": int(cv.shape[0]), "stability_rows": int(stability.shape[0]), "branch_rows": int(branch_cv.shape[0]), "pass": True}, "pass": True}
    (RESULT_ROOT / "32_validation_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
