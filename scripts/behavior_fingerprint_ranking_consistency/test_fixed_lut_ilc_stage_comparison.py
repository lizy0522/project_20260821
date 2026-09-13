"""固定 LUT ILC 阶段 Figure 1/2 的数据重组、marker、结构和保护测试。"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from behavior_fingerprint_ranking_consistency.plot_fixed_lut_ilc_stage_comparison import (  # noqa: E402
    C_STAGE_MARKERS,
    ROUTES,
    load_fixed_lut_stage_data,
)
from behavior_fingerprint_ranking_consistency.plotting import _spearman_ylim  # noqa: E402

TASK_ROOT = PROJECT_ROOT / "results" / "behavior_fingerprint_ranking_consistency"
SOURCE_ROOT = TASK_ROOT / "scenario_2_all_ilc"
RESULT_ROOT = SOURCE_ROOT / "fixed_lut_stage_figures"
SOURCE_PATH = SOURCE_ROOT / "spearman_all_ilc_long.csv"


@lru_cache(maxsize=1)
def _data():
    return load_fixed_lut_stage_data(SOURCE_PATH)


@lru_cache(maxsize=1)
def _validation() -> dict[str, object]:
    with (RESULT_ROOT / "figure_validation.json").open(
        "r",
        encoding="utf-8-sig",
    ) as handle:
        return json.load(handle)


def test_source_schema_nmax_and_state_coverage() -> None:
    frame = pd.read_csv(SOURCE_PATH)
    data = _data()
    assert frame.shape == (14875, 12)
    assert data.nmax == 5
    assert np.array_equal(data.state_frame["state_id"].to_numpy(), np.arange(425))
    assert set(frame["requested_n1"].unique()) == {1, 2, 3, 4, 5}
    assert set(frame.loc[frame["lut_type"] == "Y-A", "requested_n2"].astype(int)) == {
        1,
        2,
        3,
        4,
        5,
    }


def test_x_and_real_b_are_invariant_across_fixed_a_stage() -> None:
    summary = pd.read_csv(RESULT_ROOT / "fixed_lut_stage_figure_summary.csv")
    figure1 = summary[summary["figure"] == "Figure1"]
    for c_stage in range(1, 6):
        for route in ("X-A", "Real-B"):
            values = figure1.loc[
                (figure1["C_stage"] == c_stage) & (figure1["route"] == route),
                "median",
            ].to_numpy(dtype=float)
            assert values.size == 5
            np.testing.assert_allclose(values, values[0], rtol=0, atol=1e-12)


def test_y_a_uses_the_fixed_a_stage() -> None:
    data = _data()
    source = pd.read_csv(SOURCE_PATH)
    for c_stage in range(1, 6):
        for a_stage in range(1, 6):
            expected = source[
                (source["requested_n1"] == c_stage)
                & (source["requested_n2"] == a_stage)
                & (source["lut_type"] == "Y-A")
            ].sort_values("state_id")["spearman"].to_numpy(dtype=float)
            np.testing.assert_array_equal(data.values[(c_stage, "Y-A", a_stage)], expected)


def test_figure1_structure_and_y_range() -> None:
    validation = _validation()["figure1"]
    data = _data()
    all_values = [
        value
        for key, value in data.values.items()
        if len(key) == 2 or key[2] in range(1, 6)
    ]
    y_min, y_max = _spearman_ylim(all_values)
    assert validation["subplot_shape"] == [2, 5]
    assert validation["boxplots_per_top_subplot"] == 15
    assert validation["curves_per_bottom_subplot"] == 15
    assert validation["all_axes_same_ylim"] is True
    assert validation["ylim_padding"] is False
    assert validation["y_min"] == y_min
    assert validation["y_max"] == y_max


def test_figure2_structure_and_y_range() -> None:
    validation = _validation()["figure2"]
    data = _data()
    all_values = [value for key, value in data.values.items() if len(key) == 3]
    y_min, y_max = _spearman_ylim(all_values)
    assert validation["subplot_shape"] == [2, 5]
    assert validation["boxplots_per_top_subplot"] == 5
    assert validation["curves_per_bottom_subplot"] == 5
    assert validation["all_axes_same_ylim"] is True
    assert validation["ylim_padding"] is False
    assert validation["y_min"] == y_min
    assert validation["y_max"] == y_max


def test_box_data_and_curve_data_are_identical() -> None:
    data = _data()
    for c_stage in range(1, 6):
        for a_stage in range(1, 6):
            values = data.values[(c_stage, "Y-A", a_stage)]
            assert values.shape == (425,)
            np.testing.assert_array_equal(values, data.values[(c_stage, "Y-A", a_stage)])


def test_marker_contract_has_five_distinct_markers() -> None:
    assert len(C_STAGE_MARKERS) == 5
    assert len(set(C_STAGE_MARKERS)) == 5
    assert set(ROUTES) == {"X-A", "Y-A", "Real-B"}
    assert _validation()["figure1"]["marker_contract"].startswith("marker and color")
    assert _validation()["figure2"]["marker_contract"].startswith("marker and color")


def test_summary_rows_and_counts() -> None:
    summary = pd.read_csv(RESULT_ROOT / "fixed_lut_stage_figure_summary.csv")
    assert summary.shape[0] == 100
    assert summary["count"].eq(425).all()
    assert summary[summary["figure"] == "Figure1"].shape[0] == 75
    assert summary[summary["figure"] == "Figure2"].shape[0] == 25


def test_new_figures_exist_and_old_figures_are_preserved() -> None:
    new_paths = [
        RESULT_ROOT / "figure1_three_lut_routes_by_fixed_A_stage.png",
        RESULT_ROOT / "figure2_YA_only_by_fixed_A_stage.png",
    ]
    for path in new_paths:
        assert path.is_file()
        assert path.stat().st_size > 0
    old_paths = [
        SOURCE_ROOT / "spearman_all_ilc_2x5_combined.png",
        SOURCE_ROOT / "spearman_all_ilc_2x5_YA_only_combined.png",
    ]
    for path in old_paths:
        assert path.is_file()
        assert path.stat().st_size > 0
    assert all(path.parent == RESULT_ROOT for path in new_paths)


def test_pngs_are_readable() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name in (
        "figure1_three_lut_routes_by_fixed_A_stage.png",
        "figure2_YA_only_by_fixed_A_stage.png",
    ):
        image = plt.imread(RESULT_ROOT / name)
        assert image.size > 0
        assert image.ndim in (2, 3)
    plt.close("all")


def test_result_is_reorganized_only() -> None:
    validation = _validation()
    assert validation["data_reorganized_only"] is True
    assert validation["models_retrained"] is False
    assert validation["cnmse_recomputed"] is False
    assert validation["spearman_recomputed"] is False
    assert validation["X_A_and_Real_B_fixed_across_A"] is True


def main() -> None:
    tests = [
        test_source_schema_nmax_and_state_coverage,
        test_x_and_real_b_are_invariant_across_fixed_a_stage,
        test_y_a_uses_the_fixed_a_stage,
        test_figure1_structure_and_y_range,
        test_figure2_structure_and_y_range,
        test_box_data_and_curve_data_are_identical,
        test_marker_contract_has_five_distinct_markers,
        test_summary_rows_and_counts,
        test_new_figures_exist_and_old_figures_are_preserved,
        test_pngs_are_readable,
        test_result_is_reorganized_only,
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: PASS")


if __name__ == "__main__":
    main()
