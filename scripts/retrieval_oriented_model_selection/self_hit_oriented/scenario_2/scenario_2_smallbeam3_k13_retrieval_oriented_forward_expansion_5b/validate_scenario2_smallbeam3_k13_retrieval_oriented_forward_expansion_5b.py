# ruff: noqa: E402,E501,I001

"""Independent validation for the Beam=3 one-layer K13 expansion."""

from __future__ import annotations

import json
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
from behavior_fingerprint_retrieval.shared.envelope18_commonB_full_lut import (  # noqa: E402
    top1_retrieval,
)
from behavior_fingerprint_retrieval.shared.envelope19_c2endshared_commonB_full_lut import (  # noqa: E402
    EXPECTED_COMMON_B_SHA,
    raw_manifest_gate,
    verify_common_b_contract,
)
from behavior_modeling.shared.basis_function_selection.envelope_dictionary import (  # noqa: E402
    build_envelope_bank,
    build_envelope_dictionary,
)
from core.shared.metrics import cnmse  # noqa: E402
from data_management.shared import load_by_id  # noqa: E402
from retrieval_oriented_model_selection.self_hit_oriented.scenario_2.scenario_2_smallbeam3_k13_retrieval_oriented_forward_expansion_5b import (  # noqa: E402
    run_scenario2_smallbeam3_k13_retrieval_oriented_forward_expansion_5b as runner,
)
from signal_segmentation.shared import build_off_segments, build_partition_from_xin, get_ilc_pair, preprocess_full_pair  # noqa: E402

TASK_NAME = "scenario_2_smallbeam3_k13_retrieval_oriented_forward_expansion_5b"
RESULT_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /TASK_NAME
K9_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /"scenario_2_mp10_k9_retrieval_oriented_basis_ablation_5b"
K11_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /"scenario_2_mp10_k11_retrieval_oriented_forward_scan_5b"
K12_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /"scenario_2_k12_retrieval_oriented_forward_scan_5b"
STATE_COUNT = runner.STATE_COUNT
THRESHOLD_DB = -40.0
REPRESENTATIVE_STATES = (0, 42, 187, 325, 424)


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


def _build_state_blocks(state_id: int, terms: tuple[Any, ...], core_index: int) -> tuple[np.ndarray, ...]:
    data = load_by_id(state_id)
    partition = build_partition_from_xin(np.asarray(data["xin"]))
    _assert((partition.n_a, partition.n_b, partition.n_c) == runner.backend.ABC_LENGTHS, f"State {state_id} ABC contract changed")
    history = np.asarray(data["xin_pd_ori_ilc"])
    a_pair = get_ilc_pair(data, int(history.shape[1] - 1))
    c_pair = get_ilc_pair(data, 1)
    aend = preprocess_full_pair(a_pair.input_full, a_pair.output_raw_full, partition, pair_type=a_pair.pair_type, iteration_index=a_pair.iteration_index, input_peak_normalization_factor=a_pair.input_peak_normalization_factor)
    c2 = preprocess_full_pair(c_pair.input_full, c_pair.output_raw_full, partition, pair_type=c_pair.pair_type, iteration_index=c_pair.iteration_index, input_peak_normalization_factor=c_pair.input_peak_normalization_factor)
    inputs = (aend["A"].input, aend["B"].input, c2["C"].input, c2["B"].input)
    blocks: list[np.ndarray] = []
    for value in inputs:
        mp = runner.backend.historical_mp10.build_frozen_mp_basis(value)
        env = build_envelope_bank(value, terms)
        blocks.append(np.column_stack((mp, env[:, core_index])).astype(np.complex128))
    expected_shapes = ((12_286, runner.CORE_K), (4_913, runner.CORE_K), (7_371, runner.CORE_K), (4_913, runner.CORE_K))
    for block, expected in zip(blocks, expected_shapes, strict=True):
        _assert(block.shape == expected and np.all(np.isfinite(block)), f"State {state_id} core Phi shape/finite contract failed")
    return tuple(blocks)


def _check_support_graph(terms: tuple[Any, ...], core_ids: tuple[str, ...], core_index: int) -> dict[str, Any]:
    seed_frame = pd.read_csv(RESULT_ROOT / "03_beam_seed_supports.csv")
    edges = pd.read_csv(RESULT_ROOT / "06_raw_expansion_edges.csv")
    unique = pd.read_csv(RESULT_ROOT / "07_unique_k13_supports.csv")
    audit = json.loads((RESULT_ROOT / "08_support_dedup_audit.json").read_text(encoding="utf-8"))
    _assert(seed_frame.shape[0] == 3 and edges.shape[0] == 189 and unique.shape[0] == 186, "Beam support graph row counts failed")
    _assert(int(edges["is_duplicate_child"].sum()) == 3 and int((unique["generation_count"] > 1).sum()) == 3, "Beam duplicate count failed")
    _assert(audit["raw_expansion_edges"] == 189 and audit["unique_k13_children"] == 186 and audit["evaluation_pool"] == 189 and audit["pass"] is True, "dedup audit failed")
    expected_seed_extras = [spec[3] for spec in runner.SEED_SPECS]
    _assert(seed_frame["extra_basis_id"].tolist() == expected_seed_extras, "seed ordering changed")
    _assert(seed_frame["K"].eq(runner.SEED_K).all(), "seed K failed")
    expected_multi = {
        tuple(sorted(("ENV_p04_m0_q1", "ENV_p03_m0_q1"), key=lambda basis_id: next(term.index for term in terms if term.basis_id == basis_id))),
        tuple(sorted(("ENV_p04_m0_q1", "ENV_p09_m2_q0"), key=lambda basis_id: next(term.index for term in terms if term.basis_id == basis_id))),
        tuple(sorted(("ENV_p03_m0_q1", "ENV_p09_m2_q0"), key=lambda basis_id: next(term.index for term in terms if term.basis_id == basis_id))),
    }
    observed_multi = {tuple(row) for row in unique.loc[unique["generation_count"].gt(1), ["extension_basis_1", "extension_basis_2"]].itertuples(index=False, name=None)}
    _assert(observed_multi == expected_multi, f"multi-parent child pair set failed: {observed_multi}")
    _assert(str(seed_frame.loc[0, "support_hash"]) == runner._support_hash(tuple(json.loads(seed_frame.loc[0, "support_basis_ids"]))), "seed support hash failed")
    return {"raw_expansion_edges": int(edges.shape[0]), "unique_k13_children": int(unique.shape[0]), "evaluation_pool": 189, "duplicate_occurrence_count": int(edges["is_duplicate_child"].sum()), "multi_parent_child_count": int((unique["generation_count"] > 1).sum()), "pass": True}


def _check_phi_slicing(candidates: tuple[Any, ...], terms: tuple[Any, ...], core_index: int, common_b: np.ndarray) -> dict[str, Any]:
    child_ids = [candidate.candidate_id for candidate in candidates[3:]]
    selected_ids = sorted(np.random.default_rng(20260916).choice(child_ids, size=20, replace=False).astype(int).tolist())
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    common_mp = runner.backend.historical_mp10.build_frozen_mp_basis(common_b)
    common_env = build_envelope_bank(common_b, terms)
    common_core = np.column_stack((common_mp, common_env[:, core_index])).astype(np.complex128)
    checks = 0
    for state_id in REPRESENTATIVE_STATES:
        blocks = _build_state_blocks(state_id, terms, core_index)
        state_data = load_by_id(state_id)
        partition = build_partition_from_xin(np.asarray(state_data["xin"]))
        history = np.asarray(state_data["xin_pd_ori_ilc"])
        a_pair = get_ilc_pair(state_data, int(history.shape[1] - 1))
        c_pair = get_ilc_pair(state_data, 1)
        aend = preprocess_full_pair(a_pair.input_full, a_pair.output_raw_full, partition, pair_type=a_pair.pair_type, iteration_index=a_pair.iteration_index, input_peak_normalization_factor=a_pair.input_peak_normalization_factor)
        c2 = preprocess_full_pair(c_pair.input_full, c_pair.output_raw_full, partition, pair_type=c_pair.pair_type, iteration_index=c_pair.iteration_index, input_peak_normalization_factor=c_pair.input_peak_normalization_factor)
        state_inputs = (aend["A"].input, aend["B"].input, c2["C"].input, c2["B"].input)
        state_envs = tuple(build_envelope_bank(value, terms) for value in state_inputs)
        for candidate_id in selected_ids:
            candidate = by_id[candidate_id]
            for block, env in zip(blocks, state_envs, strict=True):
                canonical = np.column_stack((block, *(env[:, index] for index in candidate.extension_indices)))
                reassembled = np.column_stack([canonical[:, index] for index in range(candidate.K)])
                _assert(np.array_equal(canonical, reassembled), f"{candidate.candidate_name} state {state_id} Phi slicing failed")
                checks += 1
            common_candidate = np.column_stack((common_core, *(common_env[:, index] for index in candidate.extension_indices)))
            _assert(common_candidate.shape == (runner.backend.FINGERPRINT_LENGTH, runner.CHILD_K) and np.all(np.isfinite(common_candidate)), f"{candidate.candidate_name} common-B Phi failed")
            _assert(np.array_equal(common_candidate, np.column_stack([common_candidate[:, index] for index in range(candidate.K)])), f"{candidate.candidate_name} common-B slicing failed")
            checks += 1
    return {"random_k13_support_count": 20, "representative_state_count": len(REPRESENTATIVE_STATES), "phi_block_types": 5, "candidate_phi_slicing_checks": checks, "pass": True}


def _check_outputs(summary: pd.DataFrame, query: pd.DataFrame, model: pd.DataFrame, real_b: np.ndarray, shareability: np.ndarray) -> dict[str, Any]:
    _assert(summary.shape[0] == 189 and summary["model_id"].astype(int).tolist() == list(range(189)), "summary candidate coverage failed")
    _assert(model.shape[0] == 189 * STATE_COUNT and query.shape[0] == 189 * STATE_COUNT, "model/query row count failed")
    _assert(model["Aend_finite"].all() and model["C2_finite"].all(), "model finite flags failed")
    _assert(model.loc[model["model_type"].eq("seed"), "Aend_rank"].eq(runner.SEED_K).all() and model.loc[model["model_type"].eq("seed"), "C2_rank"].eq(runner.SEED_K).all(), "seed rank failed")
    _assert(model.loc[model["model_type"].eq("k13"), "Aend_rank"].eq(runner.CHILD_K).all() and model.loc[model["model_type"].eq("k13"), "C2_rank"].eq(runner.CHILD_K).all(), "K13 rank failed")
    _assert(real_b.shape == (STATE_COUNT, STATE_COUNT) and shareability.shape == real_b.shape and shareability.dtype == bool, "Real-B matrix/mask shape failed")
    _assert(np.all(np.isneginf(np.diag(real_b))) and bool(shareability.diagonal().all()), "Real-B diagonal contract failed")
    argmin_checks = 0
    for row in summary.itertuples(index=False):
        path = RESULT_ROOT / "distance_matrices" / str(row.distance_file)
        distance = np.load(path)
        _assert(distance.shape == (STATE_COUNT, STATE_COUNT) and not np.isnan(distance).any() and not np.isposinf(distance).any(), f"{row.model_name} distance contract failed")
        selected, _, _, _ = top1_retrieval(distance)
        frame = query.loc[query["model_id"].eq(int(row.model_id))].sort_values("State_R").reset_index(drop=True)
        _assert(frame.shape[0] == STATE_COUNT and np.array_equal(frame["State_Q"].to_numpy(dtype=int), selected), f"{row.model_name} State_Q mismatch")
        expected_pass = np.asarray([float(real_b[state_id, int(selected[state_id])]) < THRESHOLD_DB for state_id in range(STATE_COUNT)])
        _assert(np.array_equal(frame["realB_pass"].to_numpy(dtype=bool), expected_pass), f"{row.model_name} Real-B pass mismatch")
        _assert(int(frame["realB_pass"].sum()) == int(row.top1_pass_count), f"{row.model_name} Top1 count mismatch")
        for state_id in REPRESENTATIVE_STATES:
            order = np.lexsort((np.arange(STATE_COUNT), distance[state_id]))
            _assert(int(order[0]) == int(selected[state_id]), f"{row.model_name} argmin mismatch at {state_id}")
            argmin_checks += 1
    _assert(int(summary.loc[summary["model_type"].eq("k13"), "strict_breakthrough"].sum()) == 25, "strict breakthrough count changed")
    _assert(int(summary.loc[summary["model_type"].eq("k13"), "safe_breakthrough"].sum()) == 5, "safe breakthrough count changed")
    _assert(int(summary.sort_values("global_rank").iloc[0]["top1_pass_count"]) == 421, "global best Top1 changed")
    global_ids = summary["global_rank"].astype(int).tolist()
    _assert(sorted(global_ids) == list(range(1, 190)), "global ranking is not a permutation")
    return {"distance_matrix_count": 189, "argmin_checks": argmin_checks, "model_quality_rows": int(model.shape[0]), "query_metric_rows": int(query.shape[0]), "strict_breakthrough_count": 25, "safe_breakthrough_count": 5, "global_best_top1": 421, "pass": True}


def _check_failure_interaction_cv() -> dict[str, Any]:
    failure6 = pd.read_csv(RESULT_ROOT / "15_global_champion_failure6_analysis.csv")
    interaction = pd.read_csv(RESULT_ROOT / "16_pairwise_interaction_summary.csv")
    global_compare = pd.read_csv(RESULT_ROOT / "17_recovered_regressed_vs_global_champion.csv")
    parent_compare = pd.read_csv(RESULT_ROOT / "18_recovered_regressed_vs_parent.csv")
    cv = pd.read_csv(RESULT_ROOT / "19_query_state_cv_results.csv")
    stability = pd.read_csv(RESULT_ROOT / "20_query_state_cv_selection_stability.csv")
    pair_type = pd.read_csv(RESULT_ROOT / "21_extension_pair_type_summary.csv")
    branch = pd.read_csv(RESULT_ROOT / "23_branch_summary.csv")
    _assert(failure6.shape[0] == 189 * 6 and set(failure6["State_R"].astype(int)) == {189, 195, 323, 335, 340, 346}, "champion failure6 coverage failed")
    _assert(interaction.shape[0] == 186 and global_compare.shape[0] == 186 and parent_compare.shape[0] == 189, "interaction/comparison row counts failed")
    _assert(cv.shape[0] == 189 * 5 and stability.shape[0] == 189 and branch.shape[0] == 3, "CV/branch row counts failed")
    _assert(cv.groupby("fold").size().to_dict() == {0: 189, 1: 189, 2: 189, 3: 189, 4: 189}, "CV fold rows failed")
    _assert(cv.groupby("fold")["train_is_global_winner"].sum().eq(1).all(), "global CV winner-per-fold failed")
    _assert(pair_type["candidate_count"].sum() == 186 and set(pair_type["pair_type"]) == {"cross+cross", "cross+aligned"}, "extension pair type summary failed")
    _assert(stability["global_train_winner_count"].sum() == 5, "CV global winner count failed")
    winners = cv.loc[cv["train_is_global_winner"], "model_id"].tolist()
    _assert(winners == [74, 125, 74, 74, 74], f"CV global winners changed: {winners}")
    return {"failure6_rows": int(failure6.shape[0]), "interaction_rows": int(interaction.shape[0]), "global_comparison_rows": int(global_compare.shape[0]), "parent_edge_rows": int(parent_compare.shape[0]), "cv_rows": int(cv.shape[0]), "cv_stability_rows": int(stability.shape[0]), "cv_global_winners": winners, "branch_rows": int(branch.shape[0]), "pair_type_rows": int(pair_type.shape[0]), "pass": True}


def main() -> None:
    raw = raw_manifest_gate()
    expected_raw = {"sha256": "910b06e4be6b60b1f4dc63409a3083eb328765347fbec3b89a97a41d6633a1e0", "file_count": 429, "mat_count": 427, "bytes": 2_258_448_137}
    _assert(raw == expected_raw, "raw manifest changed")
    common_b, common_meta = verify_common_b_contract()
    _assert(common_b.shape == (4_915,) and str(common_meta["sha256"]) == EXPECTED_COMMON_B_SHA, "common-B contract failed")
    terms = tuple(build_envelope_dictionary())
    core_ids, by_id, core_index, _ = runner._load_frozen_core(terms)
    _, _, candidates, _ = runner._build_candidates(core_ids, by_id)
    support_graph = _check_support_graph(terms, core_ids, core_index)
    phi_slicing = _check_phi_slicing(candidates, terms, core_index, common_b)
    with np.load(RESULT_ROOT / "10_coefficients_seeds.npz", allow_pickle=False) as data:
        seed_theta_a = np.asarray(data["theta_Aend_padded"])
        seed_theta_c = np.asarray(data["theta_C2_padded"])
    with np.load(RESULT_ROOT / "11_coefficients_k13.npz", allow_pickle=False) as data:
        child_theta_a = np.asarray(data["theta_Aend_padded"])
        child_theta_c = np.asarray(data["theta_C2_padded"])
    _assert(seed_theta_a.shape == (3, STATE_COUNT, runner.SEED_K) and seed_theta_c.shape == seed_theta_a.shape, "seed coefficient artifact shape failed")
    _assert(child_theta_a.shape == (186, STATE_COUNT, runner.CHILD_K) and child_theta_c.shape == child_theta_a.shape, "K13 coefficient artifact shape failed")
    summary = pd.read_csv(RESULT_ROOT / "12_retrieval_summary.csv")
    query = pd.read_csv(RESULT_ROOT / "13_query_metrics_long.csv")
    model = pd.read_csv(RESULT_ROOT / "09_model_quality_long.csv")
    real_b = np.load(K9_ROOT / "04_realB_ground_truth_cnmse_matrix.npy")
    shareability = np.load(K9_ROOT / "05_realB_shareability_mask.npy")
    seed_regression = json.loads((RESULT_ROOT / "04_seed_baseline_regression.json").read_text(encoding="utf-8"))
    _assert(seed_regression["pass"] is True and all(item["pass"] for item in seed_regression["seed_regressions"]), "seed regression failed")
    outputs = _check_outputs(summary, query, model, real_b, shareability)
    random_error = _random_real_b_check(real_b)
    failure_cv = _check_failure_interaction_cv()
    checks = {
        "task_name": TASK_NAME,
        "raw_manifest": raw,
        "common_B": {"raw_length": int(common_b.size), "sha256": str(common_meta["sha256"]), "pass": True},
        "seed_regression": {"pass": True, "seed_count": 3},
        "support_graph": support_graph,
        "phi_slicing": phi_slicing,
        "coefficient_artifacts": {"seed_theta_Aend_shape": list(seed_theta_a.shape), "seed_theta_C2_shape": list(seed_theta_c.shape), "k13_theta_Aend_shape": list(child_theta_a.shape), "k13_theta_C2_shape": list(child_theta_c.shape), "pass": True},
        "outputs": outputs,
        "real_B_random_pair_count": 20,
        "real_B_random_pair_max_abs_error_dB": random_error,
        "failure_interaction_cv": failure_cv,
        "pass": True,
    }
    (RESULT_ROOT / "33_validation_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
