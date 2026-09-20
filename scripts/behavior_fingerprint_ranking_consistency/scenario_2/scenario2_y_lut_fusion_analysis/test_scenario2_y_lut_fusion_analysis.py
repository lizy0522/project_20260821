"""Scenario 2 Y-A LUT 融合结果、数学定义、矩阵、鲁棒性和图形契约测试。"""

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

from behavior_fingerprint_ranking_consistency.scenario_2.scenario2_y_lut_fusion_analysis.plot_y_lut_fusion_analysis import (  # noqa: E402,E501
    plot_figure2_median_spearman_vs_c_stage,
)
from behavior_fingerprint_ranking_consistency.shared.scenario2_y_lut_fusion_analysis import (  # noqa: E402
    LUT_TYPES,
    build_fusion_paired_tables,
    build_robustness_tables,
    choose_robust_lut,
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2" /"scenario2_y_lut_fusion_analysis"
)
SOURCE_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2" /"scenario2_all_ilc_analysis"
    / "scenario_2"
)


@lru_cache(maxsize=1)
def _validation() -> dict[str, object]:
    with (RESULT_ROOT / "validation.json").open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def _long_table() -> pd.DataFrame:
    return pd.read_csv(RESULT_ROOT / "spearman_by_state_long.csv")


@lru_cache(maxsize=1)
def _summary() -> pd.DataFrame:
    return pd.read_csv(RESULT_ROOT / "spearman_summary.csv")


def test_a1_a2_availability_and_fingerprint_shapes() -> None:
    validation = _validation()
    assert validation["A1_available_state_count"] == 425
    assert validation["A2_available_state_count"] == 425
    assert validation["Y_A1_fingerprint_shape"] == [425, 4913]
    assert validation["Y_A2_fingerprint_shape"] == [425, 4913]
    assert validation["Y_A12_mean_fingerprint_shape"] == [425, 4913]
    assert validation["all_lut_fingerprints_complex128_finite"] is True
    with np.load(RESULT_ROOT / "lut_fingerprints.npz", allow_pickle=False) as data:
        for name in ("Y_A1_fingerprints", "Y_A2_fingerprints", "Y_A12_mean_fingerprints"):
            assert data[name].shape == (425, 4913)
            assert data[name].dtype == np.complex128
            assert np.all(np.isfinite(data[name]))


def test_fusion_is_direct_complex_arithmetic_mean() -> None:
    a1 = np.asarray([1.0 + 1.0j, 2.0 - 4.0j])
    a2 = np.asarray([3.0 + 3.0j, 4.0 + 2.0j])
    expected = np.asarray([2.0 + 2.0j, 3.0 - 1.0j])
    np.testing.assert_array_equal(0.5 * (a1 + a2), expected)
    assert not np.array_equal(0.5 * (np.abs(a1) + np.abs(a2)), expected)
    with np.load(RESULT_ROOT / "lut_fingerprints.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(
            data["Y_A12_mean_fingerprints"],
            0.5 * (data["Y_A1_fingerprints"] + data["Y_A2_fingerprints"]),
        )


def test_fingerprint_mean_equals_theta_mean_and_frozen_probe() -> None:
    validation = _validation()
    assert validation["fingerprint_mean_theta_mean_max_abs_error"] <= 1e-10
    assert validation["common_B_probe_changed"] is False
    assert validation["common_B_input_shape"] == [4915]
    assert validation["phi_B_shape"] == [4913, 10]
    with np.load(RESULT_ROOT / "lut_fingerprints.npz", allow_pickle=False) as data:
        assert data["common_B_input"].shape == (4915,)
        assert data["common_B_input"].dtype == np.complex128


def test_raw_waveform_is_not_an_input_to_fusion() -> None:
    validation = _validation()
    assert validation["raw_waveform_averaging"] is False
    source = (TASK_SCRIPTS_ROOT / "scenario2_y_lut_fusion_analysis.py").read_text(encoding="utf-8")
    assert "xin_pd_ori_ilc" not in source
    assert "yout_withdpd_ori_ilc" not in source
    assert "Y_A1_fingerprints" in (
        TASK_SCRIPTS_ROOT / "run_scenario2_y_lut_fusion_analysis.py"
    ).read_text(encoding="utf-8")


def test_lut_scope_and_real_reference_contract() -> None:
    validation = _validation()
    assert validation["lut_types"] == list(LUT_TYPES)
    assert validation["X_A_used_as_LUT"] is False
    assert validation["Real_B_used_as_LUT"] is False
    assert validation["Real_B_used_as_reference"] is True
    assert validation["reference_ranking"] == "R_RR = frozen Real-B to Real-B"


def test_query_saturation_and_saved_query_shapes() -> None:
    validation = _validation()
    assert validation["requested_C_stages"] == [1, 2, 3, 4, 5]
    assert validation["statewise_C_saturation"] is True
    assert validation["effective_C_rule"] == "min(requested_C_n, state_max_n)"
    with np.load(RESULT_ROOT / "query_fingerprints.npz", allow_pickle=False) as data:
        assert data["state_ids"].shape == (425,)
        assert data["effective_C_n"].shape == (425, 5)
        assert np.all(data["effective_C_n"][:, 0] == 1)
        assert np.all(data["effective_C_n"][:, 1] <= 2)
        assert np.all(data["effective_C_n"][:, 4] <= 5)
        for stage in range(1, 6):
            assert data[f"Q_C{stage}"].shape == (425, 4913)
            assert data[f"Q_C{stage}"].dtype == np.complex128
            assert np.all(np.isfinite(data[f"Q_C{stage}"]))


def test_distance_and_ranking_matrix_counts_and_shapes() -> None:
    validation = _validation()
    assert validation["distance_matrix_count"] == 15
    assert validation["ranking_matrix_count"] == 15
    with np.load(RESULT_ROOT / "distance_matrices.npz", allow_pickle=False) as data:
        names = [name for name in data.files if name != "state_ids"]
        assert len(names) == 15
        for name in names:
            assert data[name].shape == (425, 425)
            assert not np.isnan(data[name]).any()
            assert not np.isposinf(data[name]).any()
    with np.load(RESULT_ROOT / "ranking_matrices.npz", allow_pickle=False) as data:
        names = [name for name in data.files if name != "state_ids"]
        assert len(names) == 15
        for name in names:
            assert data[name].shape == (425, 425)
            assert np.all(np.isfinite(data[name]))


def test_candidate_count_and_self_match_contract() -> None:
    validation = _validation()
    assert validation["candidate_count_per_query"] == 425
    assert validation["self_match_included"] is True
    with np.load(RESULT_ROOT / "distance_matrices.npz", allow_pickle=False) as data:
        # The diagonal is present in every matrix; no leave-one-out candidate removal.
        for name in data.files:
            if name != "state_ids":
                diagonal = np.diag(data[name])
                assert diagonal.shape == (425,)


def test_spearman_counts_range_and_summary() -> None:
    validation = _validation()
    assert validation["spearman_total_count"] == 6375
    assert validation["all_spearman_finite"] is True
    long_table = _long_table()
    assert long_table.shape == (6375, 11)
    assert set(long_table["lut_fingerprint_type"].unique()) == set(LUT_TYPES)
    assert long_table["requested_C_n"].nunique() == 5
    assert long_table["spearman"].between(-1.0, 1.0).all()
    summary = _summary()
    assert summary.shape[0] == 15
    assert summary["count"].eq(425).all()


def test_a1_a2_baseline_regression() -> None:
    validation = _validation()
    assert validation["baseline_regression_pass"] is True
    regression = validation["baseline_regression"]
    assert regression["spearman_max_abs_error"] <= 1e-10
    assert regression["distance_max_abs_error"] <= regression["distance_diagnostic_tolerance"]
    assert regression["ranking_max_abs_error"] <= 1e-10
    assert regression["distance_nonfinite_pattern_mismatch_count"] == 0
    assert regression["ranking_nonfinite_pattern_mismatch_count"] == 0


def test_worst_c_metric_with_artificial_example() -> None:
    state_frame = pd.DataFrame({"state_id": np.arange(3)})
    values = {
        (1, "Y-A1"): np.asarray([0.99, 0.99, 0.99]),
        (2, "Y-A1"): np.asarray([0.98, 0.98, 0.98]),
        (3, "Y-A1"): np.asarray([0.97, 0.97, 0.97]),
        (4, "Y-A1"): np.asarray([0.96, 0.96, 0.96]),
        (5, "Y-A1"): np.asarray([0.95, 0.95, 0.95]),
    }
    for lut_type in ("Y-A2", "Y-A12-Mean"):
        values.update({(stage, lut_type): values[(stage, "Y-A1")] for stage in range(1, 6)})
    summary, _ = build_robustness_tables(state_frame, values)
    row = summary[summary["lut_type"] == "Y-A1"].iloc[0]
    assert row["worst_C_median"] == 0.95
    assert row["mean_C_median"] == 0.97


def test_statewise_worst_c_and_stage() -> None:
    state_frame = pd.DataFrame({"state_id": np.arange(3)})
    values = {}
    for stage, row in enumerate(
        (
            [0.99, 0.97, 0.96],
            [0.98, 0.98, 0.95],
            [0.97, 0.96, 0.94],
            [0.95, 0.95, 0.93],
            [0.94, 0.94, 0.92],
        ),
        start=1,
    ):
        for lut_type in LUT_TYPES:
            values[(stage, lut_type)] = np.asarray(row, dtype=float)
    _, statewise = build_robustness_tables(state_frame, values)
    np.testing.assert_array_equal(statewise["worst_C_Y_A1"].to_numpy(), [0.94, 0.94, 0.92])
    np.testing.assert_array_equal(statewise["worst_C_stage_Y_A1"].to_numpy(), [5, 5, 5])


def test_paired_delta_definition_and_counts() -> None:
    state_frame = pd.DataFrame({"state_id": np.arange(3)})
    values = {}
    for stage in range(1, 6):
        values[(stage, "Y-A1")] = np.asarray([0.80, 0.90, 0.95])
        values[(stage, "Y-A2")] = np.asarray([0.70, 0.92, 0.95])
        values[(stage, "Y-A12-Mean")] = np.asarray([0.85, 0.91, 0.96])
    comparison, summary = build_fusion_paired_tables(state_frame, values)
    row = comparison[(comparison["C_stage"] == 1) & (comparison["state_id"] == 0)].iloc[0]
    np.testing.assert_allclose(row["delta_A12_vs_A1"], 0.05)
    np.testing.assert_allclose(row["delta_A12_vs_A2"], 0.15)
    c1 = summary[(summary["scope"] == "C_stage") & (summary["C_stage"] == 1)].iloc[0]
    assert c1["A12_gt_A1_state_count"] == 3
    assert c1["A12_gt_A2_state_count"] == 2
    np.testing.assert_allclose(c1["median_delta_vs_A1"], 0.01)


def test_robust_lut_selection_follows_predeclared_rule() -> None:
    summary = pd.DataFrame(
        [
            {
                "lut_type": "Y-A1",
                "worst_C_median": 0.90,
                "statewise_worst_median": 0.91,
                "mean_C_median": 0.93,
                "std_C_median": 0.01,
            },
            {
                "lut_type": "Y-A2",
                "worst_C_median": 0.92,
                "statewise_worst_median": 0.90,
                "mean_C_median": 0.92,
                "std_C_median": 0.02,
            },
            {
                "lut_type": "Y-A12-Mean",
                "worst_C_median": 0.92,
                "statewise_worst_median": 0.93,
                "mean_C_median": 0.94,
                "std_C_median": 0.01,
            },
        ]
    )
    selected, ranked = choose_robust_lut(summary)
    assert selected == "Y-A12-Mean"
    assert ranked.iloc[0]["lut_type"] == "Y-A12-Mean"


def test_figure1_contract_and_exact_global_y_range() -> None:
    validation = _validation()["figure1"]
    long_table = _long_table()
    y_min = float(long_table["spearman"].min())
    y_max = float(long_table["spearman"].max())
    assert validation["subplot_shape"] == [2, 3]
    assert validation["boxplots_per_top_subplot"] == 5
    assert validation["curves_per_bottom_subplot"] == 5
    assert validation["points_per_curve"] == 425
    assert validation["all_axes_same_ylim"] is True
    assert validation["ylim_padding"] is False
    np.testing.assert_allclose(validation["y_min"], y_min, rtol=0, atol=1e-15)
    np.testing.assert_allclose(validation["y_max"], y_max, rtol=0, atol=1e-15)
    assert "marker and color" in validation["marker_contract"]


def test_figure2_contract_uses_summary_medians_and_exact_range() -> None:
    validation = _validation()["figure2"]
    summary = _summary()
    assert validation["subplot_shape"] == [1, 1]
    assert validation["curve_count"] == 3
    assert validation["points_per_curve"] == 5
    assert validation["ylim_padding"] is False
    np.testing.assert_allclose(
        validation["y_min"], float(summary["median"].min()), rtol=0, atol=1e-15
    )
    np.testing.assert_allclose(
        validation["y_max"], float(summary["median"].max()), rtol=0, atol=1e-15
    )
    assert "marker and color" in validation["marker_contract"]


def test_figures_exist_and_are_readable() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name in (
        "figure1_y_lut_fingerprint_robustness.png",
        "figure2_median_spearman_vs_online_ilc_stage.png",
    ):
        path = RESULT_ROOT / name
        assert path.is_file()
        assert path.stat().st_size > 0
        image = plt.imread(path)
        assert image.size > 0
        assert image.ndim in (2, 3)
    plt.close("all")


def test_figure2_plotter_rejects_missing_summary_fields() -> None:
    output = RESULT_ROOT / "_test_unused_figure2.png"
    try:
        try:
            plot_figure2_median_spearman_vs_c_stage(pd.DataFrame({"median": [1.0]}), output)
        except ValueError as exc:
            assert "缺少字段" in str(exc)
        else:
            raise AssertionError("missing summary fields should fail")
    finally:
        if output.exists():
            output.unlink()


def test_frozen_scope_flags_and_output_contract() -> None:
    validation = _validation()
    assert validation["ridge_lambda"] == 1e-8
    assert validation["ridge_rescanned"] is False
    assert validation["models_retrained"] is False
    assert validation["mp_changed"] is False
    assert validation["canonical_changed"] is False
    assert validation["D_RR_recomputed"] is False
    assert validation["figure_count"] == 2
    assert Path(validation["result_root"]) == RESULT_ROOT


def main() -> None:
    tests = [
        test_a1_a2_availability_and_fingerprint_shapes,
        test_fusion_is_direct_complex_arithmetic_mean,
        test_fingerprint_mean_equals_theta_mean_and_frozen_probe,
        test_raw_waveform_is_not_an_input_to_fusion,
        test_lut_scope_and_real_reference_contract,
        test_query_saturation_and_saved_query_shapes,
        test_distance_and_ranking_matrix_counts_and_shapes,
        test_candidate_count_and_self_match_contract,
        test_spearman_counts_range_and_summary,
        test_a1_a2_baseline_regression,
        test_worst_c_metric_with_artificial_example,
        test_statewise_worst_c_and_stage,
        test_paired_delta_definition_and_counts,
        test_robust_lut_selection_follows_predeclared_rule,
        test_figure1_contract_and_exact_global_y_range,
        test_figure2_contract_uses_summary_medians_and_exact_range,
        test_figures_exist_and_are_readable,
        test_figure2_plotter_rejects_missing_summary_fields,
        test_frozen_scope_flags_and_output_contract,
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: PASS")


if __name__ == "__main__":
    main()
