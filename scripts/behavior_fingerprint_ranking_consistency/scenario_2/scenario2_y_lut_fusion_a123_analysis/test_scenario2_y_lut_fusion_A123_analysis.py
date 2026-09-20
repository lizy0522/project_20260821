"""A1/A2/A3 及七类等权 Y-A LUT 融合研究的结果与保护契约测试。"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
TASK_SCRIPTS_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from behavior_fingerprint_ranking_consistency.shared.scenario2_y_lut_fusion_A123_analysis import (  # noqa: E402
    LUT_DEFINITIONS,
    LUT_TYPES,
    NEW_LUT_TYPES,
    choose_a123_robust_lut,
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2" /"scenario_2_y_lut_fusion_A123"
)
PREVIOUS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2" /"scenario_2_y_lut_fusion"
)


@lru_cache(maxsize=1)
def _validation() -> dict[str, object]:
    with (RESULT_ROOT / "validation.json").open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def _summary() -> pd.DataFrame:
    return pd.read_csv(RESULT_ROOT / "spearman_summary.csv")


@lru_cache(maxsize=1)
def _long_table() -> pd.DataFrame:
    return pd.read_csv(RESULT_ROOT / "spearman_by_state_long.csv")


def _fingerprint_key(lut_type: str) -> str:
    return f"{lut_type.replace('-', '_')}_fingerprints"


def test_availability_and_unique_missing_a3_state() -> None:
    validation = _validation()
    assert validation["A1_available"] == 425
    assert validation["A2_available"] == 425
    assert validation["A3_available"] == 424
    assert validation["missing_A3_state_id"] == 374
    assert validation["missing_A3_state"] == {
        "state_id": 374,
        "funMng": 30,
        "funAng": 225,
        "secMng": 0,
        "secAng": 0,
        "Vm": 2.3,
        "Pin": -21.0,
    }
    assert validation["A3_effective_stage_distribution"] == {"2": 1, "3": 424}


def test_lut_definition_and_scope_contract() -> None:
    assert LUT_TYPES == (
        "Y-A1",
        "Y-A2",
        "Y-A3",
        "Y-A12-Mean",
        "Y-A13-Mean",
        "Y-A23-Mean",
        "Y-A123-Mean",
    )
    assert LUT_DEFINITIONS["Y-A12-Mean"] == ((1, 0.5), (2, 0.5))
    assert LUT_DEFINITIONS["Y-A13-Mean"] == ((1, 0.5), (3, 0.5))
    assert LUT_DEFINITIONS["Y-A23-Mean"] == ((2, 0.5), (3, 0.5))
    assert LUT_DEFINITIONS["Y-A123-Mean"] == (
        (1, 1.0 / 3.0),
        (2, 1.0 / 3.0),
        (3, 1.0 / 3.0),
    )
    validation = _validation()
    assert validation["continuous_weight_search"] is False
    assert validation["fusion_layer"] == "common-B fingerprint layer"
    assert validation["X_A_used_as_LUT"] is False
    assert validation["Real_B_used_as_LUT"] is False
    assert validation["Real_B_used_as_reference"] is True
    assert validation["fusion_deduplication"] is False


def test_no_raw_waveform_averaging_in_new_analysis_code() -> None:
    validation = _validation()
    assert validation["raw_waveform_averaging"] is False
    source = (TASK_SCRIPTS_ROOT / "scenario2_y_lut_fusion_A123_analysis.py").read_text(
        encoding="utf-8"
    )
    assert "xin_pd_ori_ilc" not in source
    assert "yout_withdpd_ori_ilc" not in source


def test_all_seven_fingerprints_are_complex_finite_and_correct_shape() -> None:
    validation = _validation()
    assert validation["fingerprint_shapes"] == {lut_type: [425, 4913] for lut_type in LUT_TYPES}
    assert validation["all_lut_fingerprints_complex128_finite"] is True
    with np.load(RESULT_ROOT / "lut_fingerprints.npz", allow_pickle=False) as data:
        for lut_type in LUT_TYPES:
            value = data[_fingerprint_key(lut_type)]
            assert value.shape == (425, 4913)
            assert value.dtype == np.complex128
            assert np.all(np.isfinite(value))


def test_a3_saturation_and_missing_state_fusion_rules() -> None:
    with np.load(RESULT_ROOT / "lut_fingerprints.npz", allow_pickle=False) as data:
        a1 = data["Y_A1_fingerprints"]
        a2 = data["Y_A2_fingerprints"]
        a3 = data["Y_A3_fingerprints"]
        a12 = data["Y_A12_Mean_fingerprints"]
        a13 = data["Y_A13_Mean_fingerprints"]
        a23 = data["Y_A23_Mean_fingerprints"]
        a123 = data["Y_A123_Mean_fingerprints"]
        assert int(data["A3_effective_stage_per_state"][374]) == 2
        np.testing.assert_array_equal(a3[374], a2[374])
        np.testing.assert_array_equal(a23[374], a2[374])
        np.testing.assert_allclose(a13[374], a12[374], rtol=0, atol=1e-12)
        np.testing.assert_allclose(a123[374], (a1[374] + 2.0 * a2[374]) / 3.0, rtol=0, atol=1e-12)
        np.testing.assert_allclose(a12, 0.5 * (a1 + a2), rtol=0, atol=1e-12)
        np.testing.assert_allclose(a13, 0.5 * (a1 + a3), rtol=0, atol=1e-12)
        np.testing.assert_allclose(a23, 0.5 * (a2 + a3), rtol=0, atol=1e-12)
        np.testing.assert_allclose(a123, (a1 + a2 + a3) / 3.0, rtol=0, atol=1e-12)


def test_all_fingerprint_theta_equivalence_errors() -> None:
    validation = _validation()
    errors = validation["fusion_fingerprint_theta_max_abs_errors"]
    assert set(errors) == set(LUT_TYPES)
    assert max(errors.values()) <= 1e-10
    assert validation["fusion_fingerprint_theta_equivalence_pass"] is True


def test_effective_query_shapes_and_saturation() -> None:
    validation = _validation()
    assert validation["requested_C_stages"] == [1, 2, 3, 4, 5]
    assert validation["C_statewise_saturation"] is True
    assert validation["C_effective_rule"] == "effective_C_n = min(requested_C_n, Ns)"
    with np.load(RESULT_ROOT / "query_fingerprints.npz", allow_pickle=False) as data:
        assert data["effective_C_n"].shape == (425, 5)
        assert np.all(data["effective_C_n"][:, 0] == 1)
        assert np.all(data["effective_C_n"][:, 4] <= 5)
        for stage in range(1, 6):
            assert data[f"Q_C{stage}"].shape == (425, 4913)
            assert data[f"Q_C{stage}"].dtype == np.complex128
            assert np.all(np.isfinite(data[f"Q_C{stage}"]))


def test_thirty_five_distance_and_ranking_matrices() -> None:
    validation = _validation()
    assert validation["distance_matrix_count"] == 35
    assert validation["ranking_matrix_count"] == 35
    with np.load(RESULT_ROOT / "distance_matrices.npz", allow_pickle=False) as data:
        names = [name for name in data.files if name != "state_ids"]
        assert len(names) == 35
        for name in names:
            assert data[name].shape == (425, 425)
            assert not np.isnan(data[name]).any()
            assert not np.isposinf(data[name]).any()
    with np.load(RESULT_ROOT / "ranking_matrices.npz", allow_pickle=False) as data:
        names = [name for name in data.files if name != "state_ids"]
        assert len(names) == 35
        for name in names:
            assert data[name].shape == (425, 425)
            assert np.all(np.isfinite(data[name]))


def test_candidate_count_self_match_and_spearman_total() -> None:
    validation = _validation()
    assert validation["candidate_count"] == 425
    assert validation["candidate_count_per_query"] == 425
    assert validation["self_match"] is True
    assert validation["self_match_included"] is True
    assert validation["spearman_count"] == 14875
    assert validation["spearman_total_count"] == 14875
    assert validation["all_spearman_finite"] is True
    long_table = _long_table()
    assert long_table.shape == (14875, 11)
    assert set(long_table["lut_fingerprint_type"].unique()) == set(LUT_TYPES)
    assert long_table["spearman"].between(-1.0, 1.0).all()
    summary = _summary()
    assert summary.shape[0] == 35
    assert summary["count"].eq(425).all()


def test_previous_a1_a2_a12_regression() -> None:
    validation = _validation()
    assert validation["baseline_regression_pass"] is True
    regression = validation["previous_fusion_baseline_regression"]
    assert regression["fingerprint_max_abs_error"] <= 1e-12
    assert regression["query_fingerprint_max_abs_error"] <= 1e-12
    assert regression["ranking_max_abs_error"] <= 1e-12
    assert regression["spearman_max_abs_error"] <= 1e-12
    assert regression["robustness_max_abs_error"] <= 1e-12
    assert regression["distance_nonfinite_pattern_mismatch_count"] == 0
    assert regression["ranking_nonfinite_pattern_mismatch_count"] == 0


def test_robustness_summary_and_selected_lut() -> None:
    validation = _validation()
    robustness = pd.read_csv(RESULT_ROOT / "lut_robustness_summary.csv")
    assert robustness.shape[0] == 7
    assert set(robustness["lut_type"]) == set(LUT_TYPES)
    for field in (
        "C1_median",
        "C2_median",
        "C3_median",
        "C4_median",
        "C5_median",
        "worst_C_median",
        "statewise_worst_C_median",
        "mean_C_median",
        "std_C_median",
        "range_C_median",
    ):
        assert np.all(np.isfinite(robustness[field]))
    assert validation["selected_robust_lut_type"] in LUT_TYPES


def test_paired_vs_a2_wide_table_and_summary() -> None:
    paired = pd.read_csv(RESULT_ROOT / "paired_vs_A2_by_state.csv")
    expected_columns = {
        "state_id",
        "C_stage",
        "rho_A2",
        "rho_A3",
        "delta_A3_vs_A2",
        "rho_A12",
        "delta_A12_vs_A2",
        "rho_A13",
        "delta_A13_vs_A2",
        "rho_A23",
        "delta_A23_vs_A2",
        "rho_A123",
        "delta_A123_vs_A2",
    }
    assert paired.shape == (2125, len(expected_columns))
    assert set(paired.columns) == expected_columns
    summary = pd.read_csv(RESULT_ROOT / "paired_vs_A2_summary.csv")
    assert summary.shape[0] == 25
    assert set(summary["lut_type"]) == set(NEW_LUT_TYPES)
    assert summary["count"].eq(425).all()
    assert summary[["better_count", "equal_count", "worse_count"]].sum(axis=1).eq(425).all()


def test_saturation_sensitivity_has_full_and_excluded_scopes() -> None:
    validation = _validation()
    sensitivity = pd.read_csv(RESULT_ROOT / "saturation_sensitivity_summary.csv")
    assert sensitivity.shape[0] == 14
    assert set(sensitivity["sample_scope"]) == {"all_425_states", "exclude_Ns2_state"}
    assert set(sensitivity["state_count"]) == {425, 424}
    assert set(sensitivity["lut_type"]) == set(LUT_TYPES)
    assert validation["saturation_sensitivity"]["winner_unchanged"] is True


def test_figure1_contract_and_global_y_range() -> None:
    validation = _validation()["figure1"]
    long_table = _long_table()
    assert validation["subplot_shape"] == [2, 7]
    assert validation["boxplots_per_top_subplot"] == 5
    assert validation["curves_per_bottom_subplot"] == 5
    assert validation["points_per_curve"] == 425
    assert validation["all_axes_same_ylim"] is True
    assert validation["ylim_padding"] is False
    np.testing.assert_allclose(validation["y_min"], long_table["spearman"].min(), atol=1e-15)
    np.testing.assert_allclose(validation["y_max"], long_table["spearman"].max(), atol=1e-15)
    assert validation["all_spearman_value_count"] == 14875
    assert "marker and color" in validation["marker_contract"]


def test_figure2_contract_and_median_source() -> None:
    validation = _validation()["figure2"]
    summary = _summary()
    assert validation["subplot_shape"] == [1, 1]
    assert validation["curve_count"] == 7
    assert validation["points_per_curve"] == 5
    assert validation["ylim_padding"] is False
    np.testing.assert_allclose(validation["y_min"], summary["median"].min(), atol=1e-15)
    np.testing.assert_allclose(validation["y_max"], summary["median"].max(), atol=1e-15)
    assert "marker and color" in validation["marker_contract"]


def test_figures_exist_and_are_readable() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name in (
        "figure1_A123_lut_fingerprint_robustness.png",
        "figure2_median_spearman_vs_C_stage_A123.png",
    ):
        path = RESULT_ROOT / name
        assert path.is_file()
        assert path.stat().st_size > 0
        image = plt.imread(path)
        assert image.size > 0
        assert image.ndim in (2, 3)
    plt.close("all")


def test_winner_rule_order_is_frozen() -> None:
    summary = pd.DataFrame(
        [
            {
                "lut_type": lut_type,
                "worst_C_median": 0.90,
                "statewise_worst_C_median": 0.90,
                "mean_C_median": 0.90,
                "std_C_median": 0.01,
            }
            for lut_type in LUT_TYPES
        ]
    )
    summary.loc[summary["lut_type"] == "Y-A123-Mean", "std_C_median"] = 0.02
    selected, _ = choose_a123_robust_lut(summary)
    assert selected == "Y-A1"
    assert _validation()["winner_rule"] == [
        "worst_C_median",
        "statewise_worst_C_median",
        "mean_C_median",
        "std_C_median",
    ]


def test_frozen_scope_and_output_separation() -> None:
    validation = _validation()
    assert validation["ridge_lambda"] == 1e-8
    assert validation["ridge_rescanned"] is False
    assert validation["mp_changed"] is False
    assert validation["canonical_changed"] is False
    assert validation["common_B_changed"] is False
    assert validation["D_RR_changed"] is False
    assert validation["figure_count"] == 2
    assert Path(validation["result_root"]) == RESULT_ROOT
    assert PREVIOUS_ROOT.is_dir()
    assert (PREVIOUS_ROOT / "lut_fingerprints.npz").is_file()


def main() -> None:
    tests = [
        test_availability_and_unique_missing_a3_state,
        test_lut_definition_and_scope_contract,
        test_no_raw_waveform_averaging_in_new_analysis_code,
        test_all_seven_fingerprints_are_complex_finite_and_correct_shape,
        test_a3_saturation_and_missing_state_fusion_rules,
        test_all_fingerprint_theta_equivalence_errors,
        test_effective_query_shapes_and_saturation,
        test_thirty_five_distance_and_ranking_matrices,
        test_candidate_count_self_match_and_spearman_total,
        test_previous_a1_a2_a12_regression,
        test_robustness_summary_and_selected_lut,
        test_paired_vs_a2_wide_table_and_summary,
        test_saturation_sensitivity_has_full_and_excluded_scopes,
        test_figure1_contract_and_global_y_range,
        test_figure2_contract_and_median_source,
        test_figures_exist_and_are_readable,
        test_winner_rule_order_is_frozen,
        test_frozen_scope_and_output_separation,
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: PASS")


if __name__ == "__main__":
    main()
