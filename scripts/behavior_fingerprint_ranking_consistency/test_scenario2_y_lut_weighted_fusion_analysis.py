"""非等权全局 Y-A LUT Development/Validation 分离研究的契约测试。"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
TASK_SCRIPTS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from behavior_fingerprint_ranking_consistency.distance_matrix import (  # noqa: E402
    compute_cnmse_distance_matrix,
)
from behavior_fingerprint_ranking_consistency.ranking import spearman_statistic  # noqa: E402
from behavior_fingerprint_ranking_consistency.scenario2_y_lut_weighted_fusion_analysis import (  # noqa: E402
    FORMAL_LUT_TYPES,
    METRIC_TOLERANCE,
    STATE_COUNT,
    _distance_from_scan_terms,
    _precompute_scan_terms,
    build_query_split,
    build_weighted_fingerprint,
    compare_weight_records,
    evaluate_validation_success,
    load_a123_inputs,
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2_y_lut_weighted_fusion"
)
A123_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2_y_lut_fusion_A123"
)
REFERENCE_ROOT = (
    PROJECT_ROOT / "results" / "behavior_fingerprint_ranking_consistency" / "scenario_2"
)


@lru_cache(maxsize=1)
def _validation() -> dict[str, object]:
    with (RESULT_ROOT / "validation.json").open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def _selected_weight() -> dict[str, object]:
    with (RESULT_ROOT / "selected_weight.json").open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def _validation_long() -> pd.DataFrame:
    return pd.read_csv(RESULT_ROOT / "validation_spearman_by_state.csv")


@lru_cache(maxsize=1)
def _validation_summary() -> pd.DataFrame:
    return pd.read_csv(RESULT_ROOT / "validation_summary.csv")


def test_split_is_340_85_disjoint_and_balanced() -> None:
    split = pd.read_csv(RESULT_ROOT / "split_definition.csv")
    assert split.shape[0] == STATE_COUNT
    assert split["split"].value_counts().to_dict() == {"Development": 340, "Validation": 85}
    dev = set(split.loc[split["split"] == "Development", "state_id"])
    val = set(split.loc[split["split"] == "Validation", "state_id"])
    assert dev.isdisjoint(val)
    assert dev | val == set(range(STATE_COUNT))
    for angle in (135, 180, 225):
        assert ((split["funAng"] == angle) & (split["split"] == "Development")).any()
        assert ((split["funAng"] == angle) & (split["split"] == "Validation")).any()
    validation = _validation()
    assert validation["development_count"] == 340
    assert validation["validation_count"] == 85
    assert validation["split_seed"] == 20260827
    assert validation["split_overlap_count"] == 0
    assert validation["split_union_count"] == 425


def test_candidate_library_remains_full_425() -> None:
    validation = _validation()
    assert validation["candidate_count"] == 425
    assert validation["query_split_only"] is True
    with np.load(RESULT_ROOT / "lut_fingerprints_selected.npz", allow_pickle=False) as data:
        assert data["Y_A2"].shape == (425, 4913)
        assert data["Y_A_weighted"].shape == (425, 4913)


def test_simplex_weight_grid_has_anchors_and_valid_weights() -> None:
    grid = pd.read_csv(RESULT_ROOT / "weight_grid.csv")
    selected = _selected_weight()
    assert grid["grid_phase"].value_counts()["coarse"] == 232
    assert grid.shape[0] == 232 + selected["fine_candidate_count"]
    weights = grid[["w1_A1", "w2_A2", "w3_A3"]].to_numpy(dtype=float)
    assert np.all(weights >= -1e-12)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, rtol=0, atol=1e-12)
    anchors = grid[grid["is_anchor"]]
    assert set(anchors["anchor_name"]) == {"A1", "A2", "A3", "A12", "A13", "A23", "A123"}
    assert selected["coarse_candidate_count"] == 232
    assert selected["coarse_grid_step"] == 0.05
    assert selected["fine_grid_step"] == 0.01


def test_development_scan_is_development_only_and_ranked() -> None:
    scan = pd.read_csv(RESULT_ROOT / "development_weight_scan.csv")
    ranking = pd.read_csv(RESULT_ROOT / "development_weight_ranking.csv")
    selected = _selected_weight()
    assert scan.shape[0] == 232 + selected["fine_candidate_count"]
    assert scan["development_state_count"].eq(340).all()
    assert set(scan["grid_phase"]) == {"coarse", "fine"}
    assert ranking.shape[0] == scan.shape[0]
    assert ranking["development_rank"].to_numpy().tolist() == list(range(1, ranking.shape[0] + 1))
    assert ranking.iloc[0]["weight_id"] == selected["selected_weight_id"]
    assert ranking.iloc[0]["w1"] == selected["w1"]
    assert ranking.iloc[0]["w2"] == selected["w2"]
    assert ranking.iloc[0]["w3"] == selected["w3"]
    validation = _validation()
    assert validation["validation_used_for_weight_selection"] is False
    assert selected["selection_split"] == "Development"
    assert selected["weights_frozen_marker"] == "WEIGHTS_FROZEN"


def test_selected_weight_development_acceleration_matches_direct_cnmse() -> None:
    inputs = load_a123_inputs(A123_ROOT, REFERENCE_ROOT)
    _, development_ids, _ = build_query_split(inputs.state_frame)
    terms = _precompute_scan_terms(
        inputs.base_fingerprints,
        inputs.query_fingerprints,
        inputs.reference_ranking,
        development_ids,
    )
    selected = _selected_weight()
    weights = np.asarray([selected["w1"], selected["w2"], selected["w3"]], dtype=float)
    candidate = build_weighted_fingerprint(inputs.base_fingerprints, weights)
    for stage in range(1, 6):
        fast_distance = _distance_from_scan_terms(
            terms.query_energy[stage],
            terms.query_base_inner[stage],
            terms.candidate_gram,
            weights,
        )
        direct_distance = compute_cnmse_distance_matrix(
            inputs.query_fingerprints[stage][development_ids], candidate
        )
        fast_rank = rankdata(fast_distance, axis=1, method="average")
        direct_rank = rankdata(direct_distance, axis=1, method="average")
        assert np.array_equal(fast_rank, direct_rank)
        fast_spearman = np.asarray(
            [
                spearman_statistic(terms.reference_dev[index], fast_rank[index])
                for index in range(development_ids.size)
            ]
        )
        direct_spearman = np.asarray(
            [
                spearman_statistic(
                    inputs.reference_ranking[development_ids[index]], direct_rank[index]
                )
                for index in range(development_ids.size)
            ]
        )
        np.testing.assert_array_equal(fast_spearman, direct_spearman)


def test_selected_weight_and_weighted_fingerprint_contract() -> None:
    selected = _selected_weight()
    validation = _validation()
    weights = np.asarray([selected["w1"], selected["w2"], selected["w3"]], dtype=float)
    assert np.all(weights >= 0)
    np.testing.assert_allclose(weights.sum(), 1.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(weights, [0.0, 1.0, 0.0], rtol=0, atol=1e-12)
    assert validation["selected_weight_is_A2"] is True
    assert validation["selected_weight_is_A2_dominant"] is True
    with np.load(A123_ROOT / "lut_fingerprints.npz", allow_pickle=False) as source:
        with np.load(
            RESULT_ROOT / "lut_fingerprints_selected.npz", allow_pickle=False
        ) as selected_data:
            np.testing.assert_array_equal(selected_data["Y_A2"], source["Y_A2_fingerprints"])
            expected = build_weighted_fingerprint(
                np.stack(
                    [
                        source["Y_A1_fingerprints"],
                        source["Y_A2_fingerprints"],
                        source["Y_A3_fingerprints"],
                    ],
                    axis=0,
                ),
                weights,
            )
            np.testing.assert_allclose(selected_data["Y_A_weighted"], expected, rtol=0, atol=1e-12)
            assert selected_data["Y_A_weighted"].dtype == np.complex128
            assert np.all(np.isfinite(selected_data["Y_A_weighted"]))


def test_validation_matrices_are_ten_85_by_425_and_finite() -> None:
    validation = _validation()
    assert validation["validation_distance_matrix_count"] == 10
    assert validation["validation_ranking_matrix_count"] == 10
    assert validation["validation_matrix_shape"] == [85, 425]
    with np.load(RESULT_ROOT / "validation_distance_matrices.npz", allow_pickle=False) as data:
        names = [name for name in data.files if name != "state_ids"]
        assert len(names) == 10
        assert data["state_ids"].shape == (85,)
        for name in names:
            assert data[name].shape == (85, 425)
            assert not np.isnan(data[name]).any()
            assert not np.isposinf(data[name]).any()
    with np.load(RESULT_ROOT / "validation_ranking_matrices.npz", allow_pickle=False) as data:
        names = [name for name in data.files if name != "state_ids"]
        assert len(names) == 10
        for name in names:
            assert data[name].shape == (85, 425)
            assert np.all(np.isfinite(data[name]))


def test_validation_spearman_count_summary_and_scope() -> None:
    validation = _validation()
    long_table = _validation_long()
    summary = _validation_summary()
    assert validation["validation_spearman_count"] == 850
    assert validation["expected_validation_spearman_count"] == 850
    assert validation["validation_all_spearman_finite"] is True
    assert long_table.shape[0] == 850
    assert set(long_table["lut_type"]) == set(FORMAL_LUT_TYPES)
    assert long_table["spearman"].between(-1.0, 1.0).all()
    assert summary.shape[0] == 2
    assert set(summary["lut_type"]) == set(FORMAL_LUT_TYPES)
    assert summary["state_count"].eq(85).all()


def test_previous_a123_baseline_regression() -> None:
    validation = _validation()
    assert validation["baseline_regression_pass"] is True
    regression = validation["baseline_regression"]
    assert regression["fingerprint_max_abs_error"] <= 1e-12
    assert regression["distance_max_abs_error"] <= regression["distance_diagnostic_tolerance"]
    assert regression["ranking_max_abs_error"] <= 1e-12
    assert regression["spearman_max_abs_error"] <= 1e-12
    assert regression["robustness_max_abs_error"] <= 1e-12
    assert regression["distance_nonfinite_pattern_mismatch_count"] == 0
    assert regression["ranking_nonfinite_pattern_mismatch_count"] == 0


def test_validation_success_rule_is_strict_and_currently_fails() -> None:
    validation = _validation()
    detail = validation["validation_success_detail"]
    assert validation["validation_success"] is False
    assert detail["conditions"]["worst_C_strictly_better"] is False
    assert detail["conditions"]["statewise_worst_not_degraded"] is True
    assert detail["conditions"]["q05_not_degraded_beyond_limit"] is True
    assert detail["worst_C_gain"] == 0
    assert detail["statewise_worst_C_median_gain"] == 0
    assert detail["statewise_worst_q05_gain"] == 0
    artificial = pd.DataFrame(
        [
            {
                "lut_type": "Y-A2",
                "C1_median": 0.9,
                "C2_median": 0.9,
                "C3_median": 0.9,
                "C4_median": 0.9,
                "C5_median": 0.9,
                "worst_C_median": 0.9,
                "statewise_worst_C_median": 0.9,
                "mean_C_median": 0.9,
                "std_C_median": 0.01,
                "statewise_worst_q05": 0.9,
            },
            {
                "lut_type": "Y-A-Weighted",
                "C1_median": 0.91,
                "C2_median": 0.91,
                "C3_median": 0.91,
                "C4_median": 0.91,
                "C5_median": 0.91,
                "worst_C_median": 0.91,
                "statewise_worst_C_median": 0.91,
                "mean_C_median": 0.91,
                "std_C_median": 0.01,
                "statewise_worst_q05": 0.9,
            },
        ]
    )
    assert evaluate_validation_success(artificial)["success"] is True


def test_validation_paired_table_and_condition_regions() -> None:
    paired = pd.read_csv(RESULT_ROOT / "validation_paired_vs_A2.csv")
    assert paired.shape == (425, 5)
    assert set(paired.columns) == {"state_id", "C_stage", "rho_A2", "rho_weighted", "delta"}
    np.testing.assert_allclose(
        paired["delta"], paired["rho_weighted"] - paired["rho_A2"], atol=1e-15
    )
    assert paired["state_id"].nunique() == 85
    regions = pd.read_csv(RESULT_ROOT / "condition_region_analysis.csv")
    assert set(regions["grouping"]) == {"funAng", "funMng", "secMng", "secAng"}
    angles = set(regions.loc[regions["grouping"] == "funAng", "condition"])
    assert {135, 180, 225}.issubset(angles)
    assert regions[["median_rho_A2", "median_rho_weighted", "median_delta"]].notna().all().all()


def test_tail_and_sensitivity_outputs() -> None:
    tail = pd.read_csv(RESULT_ROOT / "tail_analysis.csv")
    assert tail.shape[0] == 4
    assert set(tail["lut_type"]) == set(FORMAL_LUT_TYPES)
    assert set(tail["tail_fraction"]) == {0.05, 0.10}
    sensitivity = pd.read_csv(RESULT_ROOT / "saturation_sensitivity_summary.csv")
    assert set(sensitivity["analysis"]) == {"state374_sensitivity", "C1_C3_sensitivity"}
    assert set(
        sensitivity.loc[sensitivity["analysis"] == "state374_sensitivity", "state_count"]
    ) == {425, 424}
    assert set(sensitivity.loc[sensitivity["analysis"] == "C1_C3_sensitivity", "C_stage_set"]) == {
        "C1,C2,C3"
    }
    assert _validation()["saturation_sensitivity_winner_unchanged"] is True


def test_figures_have_required_structure_and_actual_y_ranges() -> None:
    validation = _validation()
    long_table = _validation_long()
    summary = _validation_summary()
    figure1 = validation["figure1"]
    figure2 = validation["figure2"]
    figure3 = validation["figure3"]
    assert figure1["plot_type"] == "simplex scatter landscape"
    assert figure1["anchor_count"] == 7
    assert figure1["weight_point_count"] == 232 + _selected_weight()["fine_candidate_count"]
    assert figure1["selected_weight"] == [
        _selected_weight()["w1"],
        _selected_weight()["w2"],
        _selected_weight()["w3"],
    ]
    assert figure2["subplot_shape"] == [2, 2]
    assert figure2["top_boxplot_count_per_subplot"] == 5
    assert figure2["bottom_curve_count_per_subplot"] == 5
    assert figure2["points_per_curve"] == 85
    assert figure2["all_axes_same_ylim"] is True
    np.testing.assert_allclose(figure2["y_min"], long_table["spearman"].min(), atol=1e-15)
    np.testing.assert_allclose(figure2["y_max"], long_table["spearman"].max(), atol=1e-15)
    assert figure3["curve_count"] == 2
    assert figure3["points_per_curve"] == 5
    np.testing.assert_allclose(
        figure3["y_min"],
        summary.filter(
            items=["C1_median", "C2_median", "C3_median", "C4_median", "C5_median"], axis="columns"
        )
        .to_numpy()
        .min(),
        atol=1e-15,
    )
    np.testing.assert_allclose(
        figure3["y_max"],
        summary.filter(
            items=["C1_median", "C2_median", "C3_median", "C4_median", "C5_median"], axis="columns"
        )
        .to_numpy()
        .max(),
        atol=1e-15,
    )
    assert figure2["ylim_padding"] is False
    assert figure3["ylim_padding"] is False
    assert "marker and color" in figure2["marker_contract"]
    assert "marker and color" in figure3["marker_contract"]


def test_figures_are_readable_and_output_paths_exist() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name in (
        "figure1_development_weight_landscape.png",
        "figure2_validation_A2_vs_weighted.png",
        "figure3_validation_median_vs_C_stage.png",
    ):
        path = RESULT_ROOT / name
        assert path.is_file()
        assert path.stat().st_size > 0
        image = plt.imread(path)
        assert image.size > 0
        assert image.ndim in (2, 3)
    plt.close("all")


def test_winner_comparator_and_tolerance_contract() -> None:
    better_right = {
        "weight_id": "right",
        "w1": 0.0,
        "w2": 0.9,
        "w3": 0.1,
        "worst_C_median": 0.91,
        "statewise_worst_C_median": 0.9,
        "mean_C_median": 0.9,
        "std_C_median": 0.1,
    }
    worse_left = {**better_right, "weight_id": "left", "worst_C_median": 0.90}
    assert compare_weight_records(worse_left, better_right) > 0
    tied_a = {**better_right, "weight_id": "a", "worst_C_median": 0.9}
    tied_b = {
        **better_right,
        "weight_id": "b",
        "worst_C_median": 0.9,
        "w1": 0.1,
        "w2": 0.9,
        "w3": 0.0,
    }
    assert compare_weight_records(tied_a, tied_b) < 0
    assert METRIC_TOLERANCE == 1e-6


def test_no_raw_waveform_names_in_weighted_analysis() -> None:
    for name in (
        "scenario2_y_lut_weighted_fusion_analysis.py",
        "run_scenario2_y_lut_weighted_fusion_analysis.py",
    ):
        source = (TASK_SCRIPTS_ROOT / name).read_text(encoding="utf-8")
        assert "xin_pd_ori_ilc" not in source
        assert "yout_withdpd_ori_ilc" not in source


def test_frozen_scope_flags_and_validation_unlock_marker() -> None:
    validation = _validation()
    selected = _selected_weight()
    assert validation["baseline"] == "Y-A2"
    assert validation["models_retrained"] is False
    assert validation["lambda_rescanned"] is False
    assert validation["mp_changed"] is False
    assert validation["canonical_changed"] is False
    assert validation["common_B_changed"] is False
    assert validation["D_RR_changed"] is False
    assert validation["X_A_used_as_LUT"] is False
    assert validation["Real_B_used_as_LUT"] is False
    assert validation["Real_B_used_as_reference"] is True
    assert selected["validation_unlocked_marker"] == "VALIDATION_UNLOCKED_AFTER_WEIGHT_FREEZE"
    assert validation["validation_unlocked_marker"] == "VALIDATION_UNLOCKED_AFTER_WEIGHT_FREEZE"


def main() -> None:
    tests = [
        test_split_is_340_85_disjoint_and_balanced,
        test_candidate_library_remains_full_425,
        test_simplex_weight_grid_has_anchors_and_valid_weights,
        test_development_scan_is_development_only_and_ranked,
        test_selected_weight_development_acceleration_matches_direct_cnmse,
        test_selected_weight_and_weighted_fingerprint_contract,
        test_validation_matrices_are_ten_85_by_425_and_finite,
        test_validation_spearman_count_summary_and_scope,
        test_previous_a123_baseline_regression,
        test_validation_success_rule_is_strict_and_currently_fails,
        test_validation_paired_table_and_condition_regions,
        test_tail_and_sensitivity_outputs,
        test_figures_have_required_structure_and_actual_y_ranges,
        test_figures_are_readable_and_output_paths_exist,
        test_winner_comparator_and_tolerance_contract,
        test_no_raw_waveform_names_in_weighted_analysis,
        test_frozen_scope_flags_and_validation_unlock_marker,
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: PASS")


if __name__ == "__main__":
    main()
