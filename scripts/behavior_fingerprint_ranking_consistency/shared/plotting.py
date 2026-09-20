"""只生成本任务规定的两张 Spearman 科研图。"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from .scenario2_analysis import LUT_LABELS, SPEARMAN_COLUMNS


def _pyplot():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _spearman_ylim(values: list[np.ndarray]) -> tuple[float, float]:
    """返回所有已绘制Spearman值的实际最小/最大范围，不增加外边距。"""

    combined = np.concatenate(values)
    lower = float(np.min(combined))
    upper = float(np.max(combined))
    if not lower < upper:
        raise ValueError("Spearman值的最小值必须小于最大值")
    return lower, upper


def plot_spearman_boxplot(table: pd.DataFrame, output_path: Path) -> None:
    """绘制三种 LUT 的 Spearman 箱线图，纵轴严格取实际最小/最大值。"""

    plt = _pyplot()
    values = [table[column].to_numpy(dtype=float) for column in SPEARMAN_COLUMNS]
    if any(array.size != 425 or not np.all(np.isfinite(array)) for array in values):
        raise ValueError("箱线图输入必须是三组425个finite Spearman")
    y_min, y_max = _spearman_ylim(values)
    fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
    ax.boxplot(
        values,
        tick_labels=LUT_LABELS,
        patch_artist=True,
        showmeans=False,
        boxprops={"facecolor": "#cfe3f2", "edgecolor": "#2c5d7c"},
        medianprops={"color": "#b2182b", "linewidth": 1.5},
        whiskerprops={"color": "#2c5d7c"},
        capprops={"color": "#2c5d7c"},
        flierprops={"marker": ".", "markersize": 3, "alpha": 0.45},
    )
    ax.set_title("Scenario 2 - Ranking Consistency")
    ax.set_ylabel("Spearman Rank Correlation")
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", alpha=0.25)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_spearman_by_state(table: pd.DataFrame, output_path: Path) -> None:
    """绘制按state_id=0...424排列的三条曲线，纵轴取实际最小/最大值。"""

    plt = _pyplot()
    if table.shape[0] != 425 or not np.array_equal(
        table["state_id"].to_numpy(dtype=int),
        np.arange(425, dtype=int),
    ):
        raise ValueError("逐状态图必须包含按0...424顺序排列的425个状态")
    values_by_lut = [table[column].to_numpy(dtype=float) for column in SPEARMAN_COLUMNS]
    if any(not np.all(np.isfinite(values)) for values in values_by_lut):
        raise ValueError("三条Spearman曲线必须全部finite")
    y_min, y_max = _spearman_ylim(values_by_lut)
    fig, ax = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    colors = ("#2166ac", "#b2182b", "#4d9221")
    for column, label, color in zip(SPEARMAN_COLUMNS, LUT_LABELS, colors, strict=True):
        values = table[column].to_numpy(dtype=float)
        ax.plot(table["state_id"], values, linewidth=1.1, label=label, color=color)
    ax.set_xlabel("State ID")
    ax.set_ylabel("Spearman Rank Correlation")
    ax.set_xlim(0, 424)
    ax.set_ylim(y_min, y_max)
    ax.set_title("Scenario 2 - Spearman by State")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _all_ilc_series(nmax: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    if not isinstance(nmax, int) or nmax < 1:
        raise ValueError("nmax必须是正整数")
    columns = (
        ("spearman_X_A",)
        + tuple(f"spearman_Y_A_n2_{n:02d}" for n in range(1, nmax + 1))
        + ("spearman_Real_B",)
    )
    labels = (
        ("X-A",)
        + tuple(f"Y-A-{n}" for n in range(1, nmax + 1))
        + ("Real-B",)
    )
    return columns, labels


def _validate_all_ilc_table(
    table: pd.DataFrame,
    nmax: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    columns, labels = _all_ilc_series(nmax)
    if table.shape[0] != 425 or not np.array_equal(
        table["state_id"].to_numpy(dtype=int),
        np.arange(425, dtype=int),
    ):
        raise ValueError("全ILC图必须包含按0...424顺序排列的425个状态")
    missing = [column for column in columns if column not in table.columns]
    if missing:
        raise ValueError(f"全ILC逐状态表缺少字段：{missing}")
    return columns, labels


def plot_all_ilc_spearman_boxplot(
    table: pd.DataFrame,
    requested_n1: int,
    nmax: int,
    output_path: Path,
) -> None:
    """绘制一个requested n1下所有LUT的Spearman箱线图。"""

    plt = _pyplot()
    columns, labels = _validate_all_ilc_table(table, nmax)
    values = [table[column].to_numpy(dtype=float) for column in columns]
    if any(not np.all(np.isfinite(value)) for value in values):
        raise ValueError("全ILC箱线图输入必须全部finite")
    y_min, y_max = _spearman_ylim(values)
    fig, ax = plt.subplots(figsize=(8.4, 5.0), constrained_layout=True)
    ax.boxplot(
        values,
        tick_labels=labels,
        patch_artist=True,
        showmeans=False,
        boxprops={"facecolor": "#cfe3f2", "edgecolor": "#2c5d7c"},
        medianprops={"color": "#b2182b", "linewidth": 1.4},
        whiskerprops={"color": "#2c5d7c"},
        capprops={"color": "#2c5d7c"},
        flierprops={"marker": ".", "markersize": 2.8, "alpha": 0.4},
    )
    ax.set_title(f"Scenario 2 Ranking Consistency - Query n1 = {requested_n1}")
    ax.set_ylabel("Spearman Rank Correlation")
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", alpha=0.25)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_all_ilc_spearman_by_state(
    table: pd.DataFrame,
    requested_n1: int,
    nmax: int,
    output_path: Path,
) -> None:
    """绘制一个requested n1下所有LUT按state_id排列的Spearman曲线。"""

    plt = _pyplot()
    columns, labels = _validate_all_ilc_table(table, nmax)
    values = [table[column].to_numpy(dtype=float) for column in columns]
    if any(not np.all(np.isfinite(value)) for value in values):
        raise ValueError("全ILC曲线输入必须全部finite")
    y_min, y_max = _spearman_ylim(values)
    fig, ax = plt.subplots(figsize=(10.0, 5.0), constrained_layout=True)
    colors = ("#2166ac", "#b2182b", "#4d9221", "#762a83", "#e08214", "#1b9e77", "#666666")
    for index, (column, label) in enumerate(zip(columns, labels, strict=True)):
        ax.plot(
            table["state_id"],
            table[column],
            linewidth=1.0,
            label=label,
            color=colors[index % len(colors)],
        )
    ax.set_xlabel("State ID")
    ax.set_ylabel("Spearman Rank Correlation")
    ax.set_xlim(0, 424)
    ax.set_ylim(y_min, y_max)
    ax.set_title(f"Y-C Query, Requested ILC Iteration n1 = {requested_n1}")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_all_ilc_spearman_2x5(
    tables: Mapping[int, pd.DataFrame],
    output_path: Path,
) -> tuple[float, float]:
    """绘制n1=1...5的2×5组合图，并让10个子图共享实际极值范围。"""

    requested_n1_values = tuple(range(1, 6))
    if tuple(sorted(tables)) != requested_n1_values:
        raise ValueError("2×5组合图必须提供requested n1=1...5的五张逐状态表")
    columns, labels = _all_ilc_series(5)
    series_by_n1: dict[int, list[np.ndarray]] = {}
    all_values: list[np.ndarray] = []
    for requested_n1 in requested_n1_values:
        _validate_all_ilc_table(tables[requested_n1], 5)
        series = [tables[requested_n1][column].to_numpy(dtype=float) for column in columns]
        if any(not np.all(np.isfinite(values)) for values in series):
            raise ValueError(f"n1={requested_n1}的Spearman数据必须全部finite")
        series_by_n1[requested_n1] = series
        all_values.extend(series)
    y_min, y_max = _spearman_ylim(all_values)

    plt = _pyplot()
    fig, axes = plt.subplots(2, 5, figsize=(24.0, 8.2), sharey=True)
    colors = ("#2166ac", "#b2182b", "#4d9221", "#762a83", "#e08214", "#1b9e77", "#666666")
    for column_index, requested_n1 in enumerate(requested_n1_values):
        top_axis = axes[0, column_index]
        bottom_axis = axes[1, column_index]
        series = series_by_n1[requested_n1]
        top_axis.boxplot(
            series,
            tick_labels=labels,
            patch_artist=True,
            showmeans=False,
            boxprops={"facecolor": "#cfe3f2", "edgecolor": "#2c5d7c"},
            medianprops={"color": "#b2182b", "linewidth": 1.2},
            whiskerprops={"color": "#2c5d7c"},
            capprops={"color": "#2c5d7c"},
            flierprops={"marker": ".", "markersize": 1.8, "alpha": 0.35},
        )
        top_axis.set_title(f"n1 = {requested_n1}", fontsize=10)
        top_axis.tick_params(axis="x", labelrotation=45, labelsize=7)
        top_axis.grid(axis="y", alpha=0.22)

        for values, label, color in zip(series, labels, colors, strict=True):
            bottom_axis.plot(
                np.arange(425),
                values,
                linewidth=0.75,
                label=label,
                color=color,
            )
        bottom_axis.set_title(f"n1 = {requested_n1}", fontsize=10)
        bottom_axis.set_xlim(0, 424)
        bottom_axis.set_xlabel("State ID")
        bottom_axis.grid(True, alpha=0.22)
        for axis in (top_axis, bottom_axis):
            axis.set_ylim(y_min, y_max)
        if column_index == 0:
            top_axis.set_ylabel("Spearman Rank Correlation")
            bottom_axis.set_ylabel("Spearman Rank Correlation")
        else:
            top_axis.tick_params(axis="y", labelleft=False)
            bottom_axis.tick_params(axis="y", labelleft=False)

    handles, legend_labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=7,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle("Scenario 2 - All ILC Ranking Consistency", fontsize=15)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.94))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return y_min, y_max


def plot_all_ilc_spearman_2x5_y_a_only(
    tables: Mapping[int, pd.DataFrame],
    output_path: Path,
) -> tuple[float, float]:
    """绘制仅含Y-A-1...Y-A-5的2×5组合图，共用Y-A实际极值范围。"""

    requested_n1_values = tuple(range(1, 6))
    if tuple(sorted(tables)) != requested_n1_values:
        raise ValueError("Y-A-only组合图必须提供requested n1=1...5的五张逐状态表")
    columns = tuple(f"spearman_Y_A_n2_{n:02d}" for n in range(1, 6))
    labels = tuple(f"Y-A-{n}" for n in range(1, 6))
    series_by_n1: dict[int, list[np.ndarray]] = {}
    all_values: list[np.ndarray] = []
    for requested_n1 in requested_n1_values:
        _validate_all_ilc_table(tables[requested_n1], 5)
        series = [tables[requested_n1][column].to_numpy(dtype=float) for column in columns]
        if any(not np.all(np.isfinite(values)) for values in series):
            raise ValueError(f"n1={requested_n1}的Y-A Spearman数据必须全部finite")
        series_by_n1[requested_n1] = series
        all_values.extend(series)
    y_min, y_max = _spearman_ylim(all_values)

    plt = _pyplot()
    fig, axes = plt.subplots(2, 5, figsize=(23.0, 8.2), sharey=True)
    colors = ("#b2182b", "#4d9221", "#762a83", "#e08214", "#1b9e77")
    for column_index, requested_n1 in enumerate(requested_n1_values):
        top_axis = axes[0, column_index]
        bottom_axis = axes[1, column_index]
        series = series_by_n1[requested_n1]
        top_axis.boxplot(
            series,
            tick_labels=labels,
            patch_artist=True,
            showmeans=False,
            boxprops={"facecolor": "#f4c7c3", "edgecolor": "#7f1d1d"},
            medianprops={"color": "#2166ac", "linewidth": 1.2},
            whiskerprops={"color": "#7f1d1d"},
            capprops={"color": "#7f1d1d"},
            flierprops={"marker": ".", "markersize": 1.8, "alpha": 0.35},
        )
        top_axis.set_title(f"n1 = {requested_n1}", fontsize=10)
        top_axis.tick_params(axis="x", labelrotation=45, labelsize=7)
        top_axis.grid(axis="y", alpha=0.22)

        for values, label, color in zip(series, labels, colors, strict=True):
            bottom_axis.plot(
                np.arange(425),
                values,
                linewidth=0.8,
                label=label,
                color=color,
            )
        bottom_axis.set_title(f"n1 = {requested_n1}", fontsize=10)
        bottom_axis.set_xlim(0, 424)
        bottom_axis.set_xlabel("State ID")
        bottom_axis.grid(True, alpha=0.22)
        for axis in (top_axis, bottom_axis):
            axis.set_ylim(y_min, y_max)
        if column_index == 0:
            top_axis.set_ylabel("Spearman Rank Correlation")
            bottom_axis.set_ylabel("Spearman Rank Correlation")
        else:
            top_axis.tick_params(axis="y", labelleft=False)
            bottom_axis.tick_params(axis="y", labelleft=False)

    handles, legend_labels = axes[1, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=5,
        frameon=False,
        fontsize=9,
    )
    fig.suptitle("Scenario 2 - Y-A-only All ILC Ranking Consistency", fontsize=15)
    fig.tight_layout(rect=(0.0, 0.08, 1.0, 0.94))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return y_min, y_max


__all__ = [
    "plot_all_ilc_spearman_boxplot",
    "plot_all_ilc_spearman_by_state",
    "plot_all_ilc_spearman_2x5",
    "plot_all_ilc_spearman_2x5_y_a_only",
    "plot_spearman_boxplot",
    "plot_spearman_by_state",
]
