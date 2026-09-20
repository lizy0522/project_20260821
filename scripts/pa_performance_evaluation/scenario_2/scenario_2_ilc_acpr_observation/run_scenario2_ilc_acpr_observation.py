"""
功能说明：执行scenario_2全部425状态的ILC迭代平均ACPR收集、CSV保存和逐iteration绘图。
输入：data_manager只读加载的acpr_withdpd_ilc_tmp、nth_inter及状态参数。
输出：results/pa_performance_evaluation/scenario_2/scenario_2中的1个CSV和动态N张PNG。
用途：每次ILC单独一张图且只有一个数据系列，不输出其它PA性能指标。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from pa_performance_evaluation.shared import (  # noqa: E402
    collect_scenario2_ilc_acpr,
    plot_acpr_by_iteration,
    plot_all_iterations_together,
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "results"
    / "pa_performance_evaluation"
    / "scenario_2" /"scenario_2_ilc_acpr_observation"
)
CSV_PATH = OUTPUT_DIR / "ilc_acpr_average_by_state.csv"


def main() -> None:
    dataframe = collect_scenario2_ilc_acpr()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dataframe.to_csv(CSV_PATH, index=False, encoding="utf-8-sig", na_rep="NaN")
    figure_paths = plot_acpr_by_iteration(dataframe, OUTPUT_DIR)
    combined_figure_path = plot_all_iterations_together(dataframe, OUTPUT_DIR)
    iteration_columns = [column for column in dataframe.columns if column.startswith("iter_")]

    print("Scenario 2 ILC ACPR observation completed.")
    print()
    print(f"States: {dataframe.shape[0]}")
    print(f"Max ILC iteration: {len(iteration_columns)}")
    print()
    print("Valid states:")
    for iteration, column in enumerate(iteration_columns, start=1):
        valid_count = int(np.count_nonzero(np.isfinite(dataframe[column].to_numpy())))
        print(f"Iteration {iteration}: {valid_count}")
    print()
    print(f"CSV: {CSV_PATH}")
    print(f"Individual figures: {len(figure_paths)}")
    print(f"Combined figure: {combined_figure_path}")
    print(f"Total figures: {len(figure_paths) + 1}")


if __name__ == "__main__":
    main()
