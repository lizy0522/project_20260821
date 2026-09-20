"""
功能说明：批量收集scenario_2各状态、各次ILC迭代保存的上下邻道ACPR算术平均并分图绘制。
输入：data_manager中的state_id 0...424及MAT保存的acpr_withdpd_ilc_tmp、nth_inter。
输出：425行DataFrame，以及每个动态ILC iteration对应的一张单数据系列PNG。
用途：只观察ILC迭代ACPR，不重新计算频谱、不引入其它PA指标或通用分析框架。
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from data_management.shared import get_state_info, load_by_id

STATE_COUNT = 425
ITERATION_COLUMN_PATTERN = re.compile(r"^iter_(?P<iteration>\d+)_acpr_avg_db$")


def _average_acpr_db(lower_db: float, upper_db: float) -> float:
    """按冻结定义直接计算两个dBc数值的算术平均。"""
    return (float(lower_db) + float(upper_db)) / 2.0


def _scalar_integer(value: Any, variable_name: str) -> int:
    array = np.asarray(value)
    if array.size != 1:
        raise ValueError(f"{variable_name}必须是标量")
    scalar = float(array.reshape(-1)[0])
    integer = int(scalar)
    if not np.isfinite(scalar) or scalar != integer or integer <= 0:
        raise ValueError(f"{variable_name}必须是有限正整数，实际为{scalar}")
    return integer


def extract_state_ilc_acpr(state_id: int) -> dict[str, Any]:
    """提取一个状态的ILC ACPR历史、结构元数据和逐次上下邻道dB算术平均。"""
    data = load_by_id(state_id)
    for variable_name in ("nth_inter", "acpr_withdpd_ilc_tmp", "yout_withdpd_ori_ilc"):
        if variable_name not in data:
            raise KeyError(f"state_id={state_id}缺少变量{variable_name!r}")

    nth_inter = _scalar_integer(data["nth_inter"], "nth_inter")
    acpr_history = np.asarray(data["acpr_withdpd_ilc_tmp"], dtype=np.float64)
    output_history = np.asarray(data["yout_withdpd_ori_ilc"])
    if acpr_history.shape != (nth_inter, 2):
        raise ValueError(
            f"state_id={state_id} ACPR历史shape应为({nth_inter},2)，实际为{acpr_history.shape}"
        )
    if output_history.ndim != 2 or output_history.shape[1] != nth_inter:
        raise ValueError(
            f"state_id={state_id} yout_withdpd_ori_ilc列数必须等于nth_inter={nth_inter}"
        )
    if not np.all(np.isfinite(acpr_history)):
        raise ValueError(f"state_id={state_id} ACPR历史包含NaN或Inf")

    averages = np.array(
        [_average_acpr_db(lower, upper) for lower, upper in acpr_history],
        dtype=np.float64,
    )
    return {
        "state_id": state_id,
        **get_state_info(state_id),
        "nth_inter": nth_inter,
        "acpr_history": acpr_history,
        "iteration_acpr_averages": averages,
    }


def collect_scenario2_ilc_acpr() -> pd.DataFrame:
    """读取425个冻结state_id，动态识别最大ILC次数并生成含缺失NaN的宽表。"""
    records = [extract_state_ilc_acpr(state_id) for state_id in range(STATE_COUNT)]
    max_iteration = max(int(record["nth_inter"]) for record in records)
    rows: list[dict[str, int | float]] = []
    for record in records:
        row: dict[str, int | float] = {
            "state_id": int(record["state_id"]),
            "funMng": int(record["funMng"]),
            "funAng": int(record["funAng"]),
            "secMng": int(record["secMng"]),
            "secAng": int(record["secAng"]),
            "Vm": float(record["Vm"]),
            "Pin": float(record["Pin"]),
            "nth_inter": int(record["nth_inter"]),
        }
        averages = np.asarray(record["iteration_acpr_averages"], dtype=np.float64)
        for iteration in range(1, max_iteration + 1):
            row[f"iter_{iteration}_acpr_avg_db"] = (
                float(averages[iteration - 1]) if iteration <= averages.size else float("nan")
            )
        rows.append(row)

    dataframe = pd.DataFrame(rows)
    expected_ids = np.arange(STATE_COUNT)
    actual_ids = dataframe["state_id"].to_numpy(dtype=int)
    if dataframe.shape[0] != STATE_COUNT or not np.array_equal(actual_ids, expected_ids):
        raise RuntimeError("scenario_2 DataFrame必须严格包含state_id 0...424且无重复缺失")
    return dataframe


def _iteration_columns(dataframe: pd.DataFrame) -> list[tuple[int, str]]:
    columns: list[tuple[int, str]] = []
    for column in dataframe.columns:
        match = ITERATION_COLUMN_PATTERN.fullmatch(str(column))
        if match is not None:
            columns.append((int(match.group("iteration")), str(column)))
    if not columns:
        raise ValueError("DataFrame中没有iteration ACPR列")
    return sorted(columns)


def _global_y_limits(
    dataframe: pd.DataFrame,
    iteration_columns: list[tuple[int, str]],
) -> tuple[float, float]:
    finite_values = np.concatenate(
        [
            dataframe[column].to_numpy(dtype=float)[
                np.isfinite(dataframe[column].to_numpy(dtype=float))
            ]
            for _, column in iteration_columns
        ]
    )
    if finite_values.size == 0:
        raise ValueError("没有可绘制的有限ACPR值")
    global_min = float(np.min(finite_values))
    global_max = float(np.max(finite_values))
    value_range = global_max - global_min
    margin = 0.05 * value_range if value_range > 0 else 1.0
    return global_min - margin, global_max + margin


def plot_acpr_by_iteration(
    dataframe: pd.DataFrame,
    output_dir: Path,
) -> list[Path]:
    """每个动态iteration只画一组平均ACPR数据，并统一全部图的纵轴范围。"""
    if "state_id" not in dataframe:
        raise KeyError("DataFrame缺少state_id列")
    iteration_columns = _iteration_columns(dataframe)
    y_min, y_max = _global_y_limits(dataframe, iteration_columns)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths: list[Path] = []
    state_ids = dataframe["state_id"].to_numpy(dtype=int)
    for iteration, column in iteration_columns:
        values = dataframe[column].to_numpy(dtype=float)
        valid_mask = np.isfinite(values)
        x_values = state_ids[valid_mask]
        y_values = values[valid_mask]
        if y_values.size == 0:
            raise ValueError(f"iteration={iteration}没有有效状态")

        figure, axis = plt.subplots(figsize=(10.0, 4.8))
        if y_values.size == 1:
            axis.plot(
                x_values,
                y_values,
                color="#377EB8",
                marker="o",
                markersize=5,
                linestyle="none",
            )
        else:
            marker_size = 2 if y_values.size > 20 else 5
            axis.plot(
                x_values,
                y_values,
                color="#377EB8",
                linewidth=0.9,
                marker="o",
                markersize=marker_size,
            )
        axis.set_xlim(0, STATE_COUNT - 1)
        axis.set_ylim(y_min, y_max)
        axis.set_xlabel("State ID")
        axis.set_ylabel("Average ACPR (dBc)")
        axis.set_title(f"Scenario 2 - ILC Iteration {iteration}")
        axis.grid(True, alpha=0.3)
        figure.tight_layout()
        output_path = output_dir / f"ilc_iteration_{iteration:02d}_acpr_average.png"
        figure.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(figure)
        output_paths.append(output_path)
    return output_paths


def plot_all_iterations_together(
    dataframe: pd.DataFrame,
    output_dir: Path,
) -> Path:
    """在一张图中绘制全部动态ILC iteration，每个iteration保持一个独立数据系列。"""
    if "state_id" not in dataframe:
        raise KeyError("DataFrame缺少state_id列")
    iteration_columns = _iteration_columns(dataframe)
    y_min, y_max = _global_y_limits(dataframe, iteration_columns)
    state_ids = dataframe["state_id"].to_numpy(dtype=int)
    colors = plt.get_cmap("tab10")

    figure, axis = plt.subplots(figsize=(10.0, 4.8))
    for color_index, (iteration, column) in enumerate(iteration_columns):
        values = dataframe[column].to_numpy(dtype=float)
        valid_mask = np.isfinite(values)
        x_values = state_ids[valid_mask]
        y_values = values[valid_mask]
        if y_values.size == 0:
            raise ValueError(f"iteration={iteration}没有有效状态")
        if y_values.size == 1:
            axis.plot(
                x_values,
                y_values,
                color=colors(color_index),
                marker="o",
                markersize=6,
                linestyle="none",
                label=f"Iteration {iteration}",
            )
        else:
            marker_size = 2 if y_values.size > 20 else 5
            axis.plot(
                x_values,
                y_values,
                color=colors(color_index),
                linewidth=0.9,
                marker="o",
                markersize=marker_size,
                label=f"Iteration {iteration}",
            )
    axis.set_xlim(0, STATE_COUNT - 1)
    axis.set_ylim(y_min, y_max)
    axis.set_xlabel("State ID")
    axis.set_ylabel("Average ACPR (dBc)")
    axis.set_title("Scenario 2 - ILC Iterations 1 to 5")
    axis.grid(True, alpha=0.3)
    axis.legend()
    figure.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "ilc_iterations_01_to_05_acpr_average.png"
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return output_path


__all__ = [
    "collect_scenario2_ilc_acpr",
    "extract_state_ilc_acpr",
    "plot_all_iterations_together",
    "plot_acpr_by_iteration",
]
