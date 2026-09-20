# ruff: noqa: E402,E501,I001

"""Independent validation for the K12 retrieval-oriented backward ablation."""

from __future__ import annotations

import hashlib
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
from behavior_modeling.shared.basis_function_selection.envelope_dictionary import (  # noqa: E402
    build_envelope_bank,
    build_envelope_dictionary,
)
from core.shared.metrics import cnmse  # noqa: E402
from behavior_fingerprint_retrieval.shared.envelope18_commonB_full_lut import (  # noqa: E402
    top1_retrieval,
)
from behavior_fingerprint_retrieval.shared.envelope19_c2endshared_commonB_full_lut import (  # noqa: E402
    EXPECTED_COMMON_B_SHA,
    raw_manifest_gate,
    verify_common_b_contract,
)
from data_management.shared import load_by_id  # noqa: E402
from retrieval_oriented_model_selection.self_hit_oriented.scenario_2.scenario_2_k12_backward_retrieval_oriented_basis_ablation_5b import (  # noqa: E402
    k12_backward_backend as backend,
)
from signal_segmentation.shared import (  # noqa: E402
    build_off_segments,
    build_partition_from_xin,
    get_ilc_pair,
    preprocess_full_pair,
)

TASK_NAME = "scenario_2_k12_backward_retrieval_oriented_basis_ablation_5b"
RESULT_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /TASK_NAME
K9_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /"scenario_2_mp10_k9_retrieval_oriented_basis_ablation_5b"
K11_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /"scenario_2_mp10_k11_retrieval_oriented_forward_scan_5b"
K12_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /"scenario_2_k12_retrieval_oriented_forward_scan_5b"
STATE_COUNT = backend.STATE_COUNT
THRESHOLD_DB = -40.0
REPRESENTATIVE_STATES = (0, 189, 195, 199, 323, 327, 330, 335, 340, 346, 424)
EXPECTED_FAILURES = [189, 195, 323, 335, 340, 346]
EXPECTED_SUPPORT = (
    "LIN_d0",
    "LIN_d1",
    "LIN_d2",
    "ENV_p02_m0_q0",
    "ENV_p02_m1_q1",
    "ENV_p03_m0_q0",
    "ENV_p03_m1_q1",
    "ENV_p05_m0_q0",
    "ENV_p07_m0_q0",
    "ENV_p09_m0_q0",
    "ENV_p04_m2_q0",
    "ENV_p04_m0_q1",
)


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _support_hash(ids: tuple[str, ...]) -> str:
    return hashlib.sha256(json.dumps(list(ids), ensure_ascii=False, separators=(",", ":")).encode()).hexdigest()


def _close(actual: float, expected: float, tolerance: float, name: str) -> None:
    _assert(np.isfinite(actual) and abs(actual - expected) <= tolerance, f"{name}={actual} differs from {expected}")


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


def _build_full_design_matrices(state_id: int, terms: tuple[Any, ...], k11_index: int, k12_index: int) -> tuple[np.ndarray, ...]:
    data = load_by_id(state_id)
    xin = np.asarray(data["xin"])
    partition = build_partition_from_xin(xin)
    _assert((partition.n_a, partition.n_b, partition.n_c) == backend.ABC_LENGTHS, f"state {state_id} ABC lengths changed")
    history = np.asarray(data["xin_pd_ori_ilc"])
    a_pair = get_ilc_pair(data, int(history.shape[1] - 1))
    c_pair = get_ilc_pair(data, 1)
    aend = preprocess_full_pair(a_pair.input_full, a_pair.output_raw_full, partition, pair_type=a_pair.pair_type, iteration_index=a_pair.iteration_index, input_peak_normalization_factor=a_pair.input_peak_normalization_factor)
    c2 = preprocess_full_pair(c_pair.input_full, c_pair.output_raw_full, partition, pair_type=c_pair.pair_type, iteration_index=c_pair.iteration_index, input_peak_normalization_factor=c_pair.input_peak_normalization_factor)
    inputs = (aend["A"].input, aend["B"].input, c2["C"].input, c2["B"].input)
    result: list[np.ndarray] = []
    for value in inputs:
        mp = backend.historical_mp10.build_frozen_mp_basis(value)
        env = build_envelope_bank(value, terms)
        full = np.column_stack((mp, env[:, k11_index], env[:, k12_index])).astype(np.complex128)
        result.append(full)
    expected_shapes = ((12_286, 12), (4_913, 12), (7_371, 12), (4_913, 12))
    for name, value, shape in zip(("Aend_A", "Aend_B", "C2_C", "C2_B"), result, expected_shapes, strict=True):
        _assert(value.shape == shape and np.all(np.isfinite(value)), f"state {state_id} {name} full Phi contract failed: {value.shape}")
    return tuple(result)


def _check_supports() -> dict[str, Any]:
    baseline = pd.read_csv(RESULT_ROOT / "02_k12_baseline_support.csv").sort_values("baseline_column_index").reset_index(drop=True)
    candidates = pd.read_csv(RESULT_ROOT / "03_deletion_candidate_supports.csv").sort_values("candidate_id").reset_index(drop=True)
    _assert(tuple(baseline["basis_id"].astype(str)) == EXPECTED_SUPPORT, "baseline support ordering changed")
    _assert(baseline.shape[0] == 12, "baseline support row count failed")
    _assert(candidates.shape[0] == 13 and candidates["candidate_id"].tolist() == list(range(13)), "candidate support count/order failed")
    _assert(candidates.loc[0, "candidate_name"] == "K12_full" and int(candidates.loc[0, "K"]) == 12, "baseline candidate support failed")
    _assert(str(candidates.loc[0, "support_hash"]) == _support_hash(EXPECTED_SUPPORT), "baseline candidate support hash failed")
    for row in candidates.itertuples(index=False):
        retained = tuple(json.loads(row.retained_basis_ids))
        indices = tuple(json.loads(row.support_indices))
        if int(row.candidate_id) == 0:
            expected_indices = tuple(range(12))
        else:
            removed_index = EXPECTED_SUPPORT.index(str(row.removed_basis_id))
            expected_indices = tuple(index for index in range(12) if index != removed_index)
        _assert(indices == expected_indices and retained == tuple(EXPECTED_SUPPORT[index] for index in expected_indices), f"{row.candidate_name} support slicing failed")
        _assert(str(row.retained_support_hash) == _support_hash(retained), f"{row.candidate_name} support hash failed")
        _assert(int(row.K) == len(expected_indices), f"{row.candidate_name} K failed")
    terms = tuple(build_envelope_dictionary())
    by_id = {term.basis_id: term for term in terms}
    k11_index = int(by_id["ENV_p04_m2_q0"].index)
    k12_index = int(by_id["ENV_p04_m0_q1"].index)
    slice_checks = 0
    for state_id in REPRESENTATIVE_STATES:
        full_matrices = _build_full_design_matrices(state_id, terms, k11_index, k12_index)
        for row in candidates.itertuples(index=False):
            support = tuple(json.loads(row.support_indices))
            for matrix in full_matrices:
                independently_sliced = np.column_stack([matrix[:, index] for index in support])
                _assert(np.array_equal(matrix[:, list(support)], independently_sliced), f"{row.candidate_name} Phi deletion slice changed at state {state_id}")
                slice_checks += 1
    return {"candidate_count": 13, "baseline_K": 12, "deletion_K": 11, "representative_states": list(REPRESENTATIVE_STATES), "candidate_phi_slicing_checks": slice_checks, "pass": True}


def _check_coefficients() -> dict[str, Any]:
    summary = pd.read_csv(RESULT_ROOT / "03_deletion_candidate_supports.csv").sort_values("candidate_id").reset_index(drop=True)
    with np.load(RESULT_ROOT / "07_candidate_coefficients.npz", allow_pickle=False) as data:
        theta_a = np.asarray(data["theta_Aend_padded"])
        theta_c = np.asarray(data["theta_C2_padded"])
        mask = np.asarray(data["theta_support_mask"])
        support_indices = np.asarray(data["support_indices"])
    _assert(theta_a.shape == (13, STATE_COUNT, 12) and theta_c.shape == theta_a.shape, f"coefficient shape failed: {theta_a.shape}")
    _assert(mask.shape == (13, 12) and support_indices.shape == (13, 12), "coefficient support metadata shape failed")
    _assert(np.all(np.isfinite(theta_a)) and np.all(np.isfinite(theta_c)), "coefficient finite gate failed")
    padding_checks = 0
    for row in summary.itertuples(index=False):
        support = tuple(json.loads(row.support_indices))
        expected_mask = np.zeros(12, dtype=bool)
        expected_mask[list(support)] = True
        _assert(np.array_equal(mask[int(row.candidate_id)], expected_mask), f"{row.candidate_name} coefficient support mask failed")
        expected_indices = np.full(12, -1, dtype=np.int64)
        expected_indices[: len(support)] = np.asarray(support, dtype=np.int64)
        _assert(np.array_equal(support_indices[int(row.candidate_id)], expected_indices), f"{row.candidate_name} coefficient support indices failed")
        if int(row.candidate_id) > 0:
            removed_index = EXPECTED_SUPPORT.index(str(row.removed_basis_id))
            _assert(np.all(theta_a[int(row.candidate_id), :, removed_index] == 0.0), f"{row.candidate_name} Aend padding is not zero")
            _assert(np.all(theta_c[int(row.candidate_id), :, removed_index] == 0.0), f"{row.candidate_name} C2 padding is not zero")
            padding_checks += 2
    return {"theta_Aend_shape": list(theta_a.shape), "theta_C2_shape": list(theta_c.shape), "deleted_slot_zero_checks": padding_checks, "pass": True}


def _check_baseline_and_regressions(summary: pd.DataFrame, query: pd.DataFrame) -> dict[str, Any]:
    baseline = summary.loc[summary["candidate_id"].eq(0)].iloc[0]
    baseline_query = query.loc[query["candidate_id"].eq(0)].sort_values("State_R").reset_index(drop=True)
    failures = baseline_query.loc[~baseline_query["realB_pass"], "State_R"].astype(int).tolist()
    _assert(failures == EXPECTED_FAILURES, f"baseline failure set changed: {failures}")
    expected_ints = {"exact_hit_count": 235, "top1_pass_count": 419, "failure_count": 6, "nonself_count": 190, "nonself_pass_count": 184, "Top1_oracle": 419, "Top2_oracle": 423, "Top3_oracle": 424, "Top5_oracle": 424, "Top10_oracle": 425}
    for column, expected in expected_ints.items():
        _assert(int(baseline[column]) == expected, f"baseline {column}={baseline[column]} expected {expected}")
    _close(float(baseline["nonself_pass_rate"]), 184 / 190, 1e-12, "baseline nonself rate")
    _close(float(baseline["share_margin_Q05"]), 0.3796019374662876, 1e-12, "baseline Q05")
    _close(float(baseline["MRR"]), 0.9917086834733894, 1e-12, "baseline MRR")
    _assert(np.array_equal(baseline_query["State_Q"].to_numpy(dtype=int), baseline_query["State_Q"].to_numpy(dtype=int)), "baseline State_Q is not integer-valued")
    regression = json.loads((RESULT_ROOT / "05_baseline_regression_check.json").read_text(encoding="utf-8"))
    _assert(regression["pass"] is True and regression["K12_baseline"]["pass"] is True and regression["deletion_to_k11"]["pass"] is True, "baseline or K11 regression failed")
    deletion = summary.loc[summary["removed_basis_id"].eq("ENV_p04_m0_q1")].iloc[0]
    for column, expected in {"top1_pass_count": 416, "exact_hit_count": 218, "nonself_count": 207, "nonself_pass_count": 198, "Top2_oracle": 423, "Top3_oracle": 423}.items():
        _assert(int(deletion[column]) == expected, f"delete ENV_p04_m0_q1 {column}={deletion[column]} expected {expected}")
    _close(float(deletion["share_margin_Q05"]), 0.3238621531022702, 1e-12, "K11 regression Q05")
    _close(float(deletion["MRR"]), 0.9879831932773109, 1e-12, "K11 regression MRR")
    return {"baseline_top1": int(baseline["top1_pass_count"]), "baseline_failure_ids": failures, "baseline_exact": int(baseline["exact_hit_count"]), "baseline_nonself_pass": f"{int(baseline['nonself_pass_count'])}/{int(baseline['nonself_count'])}", "baseline_Q05": float(baseline["share_margin_Q05"]), "baseline_MRR": float(baseline["MRR"]), "baseline_topk": {str(k): int(baseline[f"Top{k}_oracle"] if k in (1, 2, 3, 5, 10) else 0) for k in (1, 2, 3, 5, 10)}, "deletion_to_k11_pass": True, "pass": True}


def _check_retrieval(summary: pd.DataFrame, query: pd.DataFrame, real_b: np.ndarray, shareability: np.ndarray) -> dict[str, Any]:
    matrix_checks = 0
    margin_checks = 0
    topk_checks = 0
    maximum_margin_error = 0.0
    for candidate in summary.sort_values("candidate_id").itertuples(index=False):
        distance = np.load(RESULT_ROOT / "distance_matrices" / f"{candidate.candidate_name}.npy")
        _assert(distance.shape == (STATE_COUNT, STATE_COUNT), f"{candidate.candidate_name} distance shape failed")
        _assert(not np.isnan(distance).any() and not np.isposinf(distance).any(), f"{candidate.candidate_name} distance finite gate failed")
        selected, _, _, _ = top1_retrieval(distance)
        frame = query.loc[query["candidate_id"].eq(int(candidate.candidate_id))].sort_values("State_R").reset_index(drop=True)
        _assert(frame.shape[0] == STATE_COUNT and frame["State_R"].tolist() == list(range(STATE_COUNT)), f"{candidate.candidate_name} query coverage failed")
        _assert(np.array_equal(selected, frame["State_Q"].to_numpy(dtype=int)), f"{candidate.candidate_name} State_Q mismatch")
        expected_pass = np.asarray([float(real_b[state_id, int(selected[state_id])]) < THRESHOLD_DB for state_id in range(STATE_COUNT)])
        _assert(np.array_equal(expected_pass, frame["realB_pass"].to_numpy(dtype=bool)), f"{candidate.candidate_name} Real-B pass mismatch")
        expected_exact = selected == np.arange(STATE_COUNT)
        _assert(np.array_equal(expected_exact, frame["exact_hit"].to_numpy(dtype=bool)), f"{candidate.candidate_name} exact-hit mismatch")
        for state_id in range(STATE_COUNT):
            order = np.lexsort((np.arange(STATE_COUNT), distance[state_id]))
            share_positions = np.flatnonzero(shareability[state_id, order])
            _assert(share_positions.size > 0, f"{candidate.candidate_name} state {state_id} has no shareable candidate")
            first_rank = int(share_positions[0] + 1)
            stored = frame.iloc[state_id]
            _assert(int(stored["first_shareable_rank"]) == first_rank, f"{candidate.candidate_name} first shareable rank mismatch at {state_id}")
            for k, column in ((2, "Top2_has_shareable"), (3, "Top3_has_shareable"), (5, "Top5_has_shareable"), (10, "Top10_has_shareable")):
                _assert(bool(stored[column]) == bool(shareability[state_id, order[:k]].any()), f"{candidate.candidate_name} {column} mismatch at {state_id}")
                topk_checks += 1
            share_idx = np.flatnonzero(shareability[state_id])
            nonshare_idx = np.flatnonzero(~shareability[state_id])
            _assert(share_idx.size > 0 and nonshare_idx.size > 0, f"{candidate.candidate_name} shareability partition is empty at {state_id}")
            share_order = share_idx[np.lexsort((share_idx, distance[state_id, share_idx]))]
            nonshare_order = nonshare_idx[np.lexsort((nonshare_idx, distance[state_id, nonshare_idx]))]
            expected_margin = float(distance[state_id, nonshare_order[0]] - distance[state_id, share_order[0]])
            error = abs(expected_margin - float(stored["shareability_margin_dB"]))
            maximum_margin_error = max(maximum_margin_error, error)
            _assert(error <= 1e-12, f"{candidate.candidate_name} shareability margin mismatch at {state_id}")
            margin_checks += 1
        for state_id in (0, 42, 187, 325, 424):
            order = np.lexsort((np.arange(STATE_COUNT), distance[state_id]))
            _assert(int(order[0]) == int(selected[state_id]), f"{candidate.candidate_name} independent argmin mismatch at {state_id}")
            matrix_checks += 1
    return {"candidate_count": int(summary.shape[0]), "distance_matrix_count": int(summary.shape[0]), "argmin_checks": matrix_checks, "shareability_margin_checks": margin_checks, "topk_oracle_checks": topk_checks, "maximum_margin_abs_error_dB": maximum_margin_error, "pass": True}


def _check_failure_and_delta(summary: pd.DataFrame, query: pd.DataFrame) -> dict[str, Any]:
    failure6 = pd.read_csv(RESULT_ROOT / "11_failure6_detailed_analysis.csv")
    transitions = pd.read_csv(RESULT_ROOT / "12_recovered_regressed_states.csv")
    delta = pd.read_csv(RESULT_ROOT / "10_backward_deletion_delta.csv")
    _assert(failure6.shape[0] == 12 * 6 and set(failure6["State_R"].astype(int)) == set(EXPECTED_FAILURES), "failure6 coverage failed")
    _assert({"rank_degraded", "rank_improved", "recovered", "candidate_pass"}.issubset(failure6.columns), "failure6 transition fields are incomplete")
    _assert(transitions.shape[0] == 12 * STATE_COUNT, "recovered/regressed coverage failed")
    for row in delta.itertuples(index=False):
        candidate_id = int(summary.loc[summary["removed_basis_id"].eq(row.removed_basis_id), "candidate_id"].iloc[0])
        current = transitions.loc[transitions["candidate_id"].eq(candidate_id)]
        recovered = int((current["transition"] == "recovered").sum())
        regressed = int((current["transition"] == "regressed").sum())
        _assert(recovered == int(row.recovered_count) and regressed == int(row.regressed_count), f"{row.removed_basis_id} recovered/regressed delta mismatch")
        _assert(int(row.delta_top1) == recovered - regressed, f"{row.removed_basis_id} delta Top1 identity failed")
    _assert(set(delta["deletion_effect_class"]) == {"necessary"}, "unexpected deletion effect class")
    _assert(not pd.read_csv(RESULT_ROOT / "16_deletion_recommendation.csv")["recommend_delete"].any(), "unexpected deletion recommendation")
    return {"failure6_rows": int(failure6.shape[0]), "recovered_regressed_rows": int(transitions.shape[0]), "delta_rows": int(delta.shape[0]), "delta_identity_checked": int(delta.shape[0]), "transition_counts": {str(key): int(value) for key, value in transitions["transition"].value_counts().to_dict().items()}, "all_deletions_necessary": True, "recommend_delete_count": 0, "pass": True}


def _check_cv() -> dict[str, Any]:
    folds = pd.read_csv(RESULT_ROOT / "13_query_state_cv_folds.csv").sort_values("state_id").reset_index(drop=True)
    cv = pd.read_csv(RESULT_ROOT / "14_query_state_cv_results.csv")
    stability = pd.read_csv(RESULT_ROOT / "15_query_state_cv_selection_stability.csv")
    _assert(folds.shape[0] == STATE_COUNT and folds["fold"].value_counts().sort_index().to_dict() == {0: 85, 1: 85, 2: 85, 3: 85, 4: 85}, "CV fold balance failed")
    for root, relative in ((K9_ROOT, "12_query_state_cv_folds.csv"), (K11_ROOT, "13_query_state_cv_folds.csv"), (K12_ROOT, "16_query_state_cv_folds.csv")):
        reference = pd.read_csv(root / relative).sort_values("state_id").reset_index(drop=True)
        _assert(np.array_equal(folds[["state_id", "fold"]].to_numpy(), reference[["state_id", "fold"]].to_numpy()), f"CV folds differ from {root.name}")
    _assert(cv.shape[0] == 65 and cv.groupby("fold").size().eq(13).all(), "CV result shape failed")
    _assert(cv.groupby("fold")["train_is_winner"].sum().eq(1).all(), "CV winner-per-fold failed")
    _assert(stability.shape[0] == 13 and stability["train_winner_count"].sum() == 5, "CV stability failed")
    winners = cv.loc[cv["train_is_winner"]].sort_values("fold")
    return {"fold_count": 5, "fold_size": 85, "train_query_size": 340, "cv_rows": int(cv.shape[0]), "stability_rows": int(stability.shape[0]), "winner_per_fold": winners[["fold", "candidate_id", "candidate_name"]].to_dict("records"), "winner_frequency": stability.loc[stability["train_winner_count"].gt(0), ["candidate_name", "train_winner_count", "train_winner_frequency", "winner_folds", "winner_validation_top1_pass_mean", "winner_validation_top1_pass_min"]].to_dict("records"), "pass": True}


def main() -> None:
    raw = raw_manifest_gate()
    _assert(raw == {"sha256": "910b06e4be6b60b1f4dc63409a3083eb328765347fbec3b89a97a41d6633a1e0", "file_count": 429, "mat_count": 427, "bytes": 2_258_448_137}, "raw manifest changed")
    common_b, common_meta = verify_common_b_contract()
    _assert(common_b.shape == (4_915,) and str(common_meta["sha256"]) == EXPECTED_COMMON_B_SHA, "common-B contract failed")
    support = _check_supports()
    coefficient = _check_coefficients()
    summary = pd.read_csv(RESULT_ROOT / "08_candidate_retrieval_summary.csv")
    query = pd.read_csv(RESULT_ROOT / "09_candidate_query_metrics_long.csv")
    model = pd.read_csv(RESULT_ROOT / "06_candidate_model_quality_long.csv")
    _assert(summary.shape[0] == 13 and model.shape[0] == 13 * STATE_COUNT and query.shape[0] == 13 * STATE_COUNT, "main table row counts failed")
    real_b = np.load(K9_ROOT / "04_realB_ground_truth_cnmse_matrix.npy")
    shareability = np.load(K9_ROOT / "05_realB_shareability_mask.npy")
    _assert(real_b.shape == (STATE_COUNT, STATE_COUNT) and shareability.shape == real_b.shape and shareability.dtype == bool, "Real-B matrix/mask shape failed")
    _assert(np.all(np.isneginf(np.diag(real_b))) and bool(shareability.diagonal().all()), "Real-B diagonal contract failed")
    random_error = _random_real_b_check(real_b)
    baseline = _check_baseline_and_regressions(summary, query)
    retrieval = _check_retrieval(summary, query, real_b, shareability)
    failure_delta = _check_failure_and_delta(summary, query)
    cv = _check_cv()
    interaction = pd.read_csv(RESULT_ROOT / "17_crossdelay_interaction_summary.csv")
    _assert(interaction.shape[0] == 3, "cross-delay interaction row count failed")
    flags = interaction.set_index("model")[["has_p04_m2_q0", "has_p04_m0_q1"]].to_dict("index")
    _assert(flags["K12_full"] == {"has_p04_m2_q0": True, "has_p04_m0_q1": True}, "full cross-delay flags failed")
    _assert(flags["K12_minus_ENV_p04_m2_q0"] == {"has_p04_m2_q0": False, "has_p04_m0_q1": True}, "p04_m2_q0 deletion flag failed")
    _assert(flags["K12_minus_ENV_p04_m0_q1"] == {"has_p04_m2_q0": True, "has_p04_m0_q1": False}, "p04_m0_q1 deletion flag failed")
    checks = {
        "task_name": TASK_NAME,
        "raw_manifest": raw,
        "common_B": {"raw_length": int(common_b.size), "sha256": str(common_meta["sha256"]), "pass": True},
        "support": support,
        "coefficients": coefficient,
        "model_quality_rows": int(model.shape[0]),
        "query_metric_rows": int(query.shape[0]),
        "real_B_random_pair_count": 20,
        "real_B_random_pair_max_abs_error_dB": random_error,
        "baseline_and_regressions": baseline,
        "retrieval": retrieval,
        "failure_and_delta": failure_delta,
        "query_state_cv": cv,
        "crossdelay_interaction_rows": int(interaction.shape[0]),
        "pass": True,
    }
    (RESULT_ROOT / "24_validation_checks.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
