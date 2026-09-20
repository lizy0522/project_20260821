"""
功能说明：验证ACPR dB平均公式、425状态结构、动态iteration、NaN语义、数据来源和PNG输出。
输入：小型合成DataFrame以及data_manager只读加载的scenario_2真实425状态。
输出：8项以上测试PASS信息；测试不写正式results目录。
用途：确保每次ILC单图单系列，且缺失迭代不补值、不重新计算频谱。
"""

from __future__ import annotations

import sys
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from data_management.shared import load_by_id  # noqa: E402

from pa_performance_evaluation.shared.ilc_acpr_observation import (  # noqa: E402
    _average_acpr_db,
    _iteration_columns,
    collect_scenario2_ilc_acpr,
    extract_state_ilc_acpr,
    plot_acpr_by_iteration,
    plot_all_iterations_together,
)


@lru_cache(maxsize=1)
def _dataframe() -> pd.DataFrame:
    return collect_scenario2_ilc_acpr()


def test_acpr_average_formula() -> None:
    assert _average_acpr_db(-48.0, -52.0) == -50.0


def test_state0_structure_matches_iteration_history() -> None:
    data = load_by_id(0)
    record = extract_state_ilc_acpr(0)
    nth_inter = int(np.asarray(data["nth_inter"]).reshape(-1)[0])
    assert np.asarray(data["acpr_withdpd_ilc_tmp"]).shape == (nth_inter, 2)
    assert np.asarray(data["yout_withdpd_ori_ilc"]).shape[1] == nth_inter
    assert record["iteration_acpr_averages"].shape == (nth_inter,)


def test_all_state_ids_are_complete_and_unique() -> None:
    dataframe = _dataframe()
    assert dataframe.shape[0] == 425
    np.testing.assert_array_equal(dataframe["state_id"].to_numpy(), np.arange(425))
    assert dataframe["state_id"].is_unique


def test_max_iteration_is_discovered_from_columns() -> None:
    synthetic = pd.DataFrame(
        {
            "state_id": [0, 1],
            "iter_1_acpr_avg_db": [-40.0, -41.0],
            "iter_2_acpr_avg_db": [-45.0, np.nan],
            "iter_3_acpr_avg_db": [-50.0, np.nan],
        }
    )
    assert [iteration for iteration, _ in _iteration_columns(synthetic)] == [1, 2, 3]
    assert len(_iteration_columns(_dataframe())) == 5


def test_current_valid_state_counts() -> None:
    dataframe = _dataframe()
    counts = [
        int(np.count_nonzero(np.isfinite(dataframe[f"iter_{iteration}_acpr_avg_db"])))
        for iteration in range(1, 6)
    ]
    assert counts == [425, 425, 424, 8, 1]


def test_missing_iterations_remain_nan() -> None:
    dataframe = _dataframe()
    for row in dataframe.itertuples(index=False):
        for iteration in range(1, 6):
            value = getattr(row, f"iter_{iteration}_acpr_avg_db")
            if iteration > row.nth_inter:
                assert np.isnan(value)
            else:
                assert np.isfinite(value)


def test_dataframe_values_equal_saved_acpr_average() -> None:
    dataframe = _dataframe().set_index("state_id")
    for state_id in (0, 100, 424):
        data = load_by_id(state_id)
        history = np.asarray(data["acpr_withdpd_ilc_tmp"], dtype=float)
        for iteration in range(1, history.shape[0] + 1):
            expected = (history[iteration - 1, 0] + history[iteration - 1, 1]) / 2.0
            actual = dataframe.loc[state_id, f"iter_{iteration}_acpr_avg_db"]
            np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)


def test_plot_count_and_image_readability() -> None:
    with tempfile.TemporaryDirectory(prefix="pa_ilc_acpr_test_") as temp_dir:
        paths = plot_acpr_by_iteration(_dataframe(), Path(temp_dir))
        assert len(paths) == 5
        assert [path.name for path in paths] == [
            f"ilc_iteration_{iteration:02d}_acpr_average.png" for iteration in range(1, 6)
        ]
        for path in paths:
            assert path.is_file() and path.stat().st_size > 0
            with Image.open(path) as image:
                image.verify()


def test_combined_plot_contains_all_iterations_and_is_readable() -> None:
    dataframe = _dataframe()
    assert len(_iteration_columns(dataframe)) == 5
    with tempfile.TemporaryDirectory(prefix="pa_ilc_acpr_combined_test_") as temp_dir:
        path = plot_all_iterations_together(dataframe, Path(temp_dir))
        assert path.name == "ilc_iterations_01_to_05_acpr_average.png"
        assert path.is_file() and path.stat().st_size > 0
        with Image.open(path) as image:
            image.verify()


def main() -> None:
    tests = [
        ("ACPR average formula test", test_acpr_average_formula),
        ("State0 structure test", test_state0_structure_matches_iteration_history),
        ("425 state completeness test", test_all_state_ids_are_complete_and_unique),
        ("dynamic max iteration test", test_max_iteration_is_discovered_from_columns),
        ("valid state count test", test_current_valid_state_counts),
        ("missing iteration NaN test", test_missing_iterations_remain_nan),
        ("saved ACPR source test", test_dataframe_values_equal_saved_acpr_average),
        ("PNG count and readability test", test_plot_count_and_image_readability),
        (
            "combined five-iteration PNG test",
            test_combined_plot_contains_all_iterations_and_is_readable,
        ),
    ]
    for label, test in tests:
        test()
        print(f"{label}: PASS")


if __name__ == "__main__":
    main()
