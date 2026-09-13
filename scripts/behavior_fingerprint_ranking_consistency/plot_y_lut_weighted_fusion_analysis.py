"""非等权全局 Y-A LUT 研究的 Python/matplotlib 正式图。

图形契约：Figure 1 展示 Development simplex 权重空间中的 Worst-C Median；
Figure 2 在权重冻结后并列展示 Validation 的 Y-A2 与 Weighted 分布/状态轨迹；
Figure 3 展示两者的 C1...C5 median 曲线。全部图形只用 Python/matplotlib，
实际数据范围不加 padding，曲线同时使用颜色与不同 marker。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .plotting import _pyplot, _spearman_ylim
from .scenario2_y_lut_weighted_fusion_analysis import (
    FORMAL_LUT_TYPES,
    NMAX,
    VALIDATION_COUNT,
)

C_STAGE_COLORS = ("#2166ac", "#b2182b", "#4d9221", "#762a83", "#e08214")
C_STAGE_MARKERS = ("o", "s", "^", "D", "P")
SCHEME_COLORS = {"Y-A2": "#b2182b", "Y-A-Weighted": "#1b9e77"}
SCHEME_MARKERS = {"Y-A2": "s", "Y-A-Weighted": "D"}


def _configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.75,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.frameon": False,
        }
    )


def _set_common_ylim(axes: np.ndarray, y_min: float, y_max: float) -> bool:
    for axis in axes.flat:
        axis.set_ylim(y_min, y_max)
    return all(
        tuple(float(value) for value in axis.get_ylim()) == (y_min, y_max) for axis in axes.flat
    )


def plot_figure1_development_landscape(
    development_scan: pd.DataFrame,
    selected_weights: np.ndarray,
    output_path: Path,
) -> dict[str, Any]:
    """绘制 Development simplex 权重景观并标出 anchor/最终权重。"""

    required = {"w1", "w2", "w3", "worst_C_median", "grid_phase", "anchor_name"}
    if not required.issubset(development_scan.columns):
        raise ValueError(
            f"Development扫描缺少字段：{sorted(required - set(development_scan.columns))}"
        )
    data = development_scan.copy()
    values = data["worst_C_median"].to_numpy(dtype=float)
    if data.shape[0] < 232 or not np.all(np.isfinite(values)):
        raise ValueError("Development扫描必须包含至少232个finite权重结果")
    y_min, y_max = float(values.min()), float(values.max())
    _configure_matplotlib()
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.3, 5.7))
    scatter = ax.scatter(
        data["w1"],
        data["w3"],
        c=values,
        cmap="viridis",
        s=28,
        edgecolors="white",
        linewidths=0.35,
        alpha=0.92,
        zorder=2,
    )
    colorbar = fig.colorbar(scatter, ax=ax, pad=0.03)
    colorbar.set_label("Development Worst-C Median")
    anchors = data[data["anchor_name"].astype(str).str.len() > 0]
    for row in anchors.itertuples(index=False):
        ax.scatter(
            row.w1,
            row.w3,
            marker="o",
            facecolors="none",
            edgecolors="black",
            linewidths=0.9,
            s=62,
            zorder=4,
        )
        ax.annotate(
            str(row.anchor_name),
            (row.w1, row.w3),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6.5,
            color="black",
        )
    selected = np.asarray(selected_weights, dtype=float)
    if selected.shape != (3,):
        raise ValueError("selected_weights必须有三个分量")
    ax.scatter(
        selected[0],
        selected[2],
        marker="*",
        s=145,
        color="#d73027",
        edgecolors="black",
        linewidths=0.7,
        zorder=5,
        label="selected w*",
    )
    ax.plot([0, 1, 0, 0], [0, 0, 1, 0], color="black", linewidth=0.85, zorder=1)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("w1 (A1 contribution)")
    ax.set_ylabel("w3 (A3 contribution)")
    ax.set_title("Development landscape for global Y-A fusion", fontsize=9.0, fontweight="bold")
    ax.grid(alpha=0.16, linewidth=0.5)
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "path": str(output_path),
        "plot_type": "simplex scatter landscape",
        "weight_point_count": int(data.shape[0]),
        "color_metric": "Development Worst-C Median",
        "y_min": y_min,
        "y_max": y_max,
        "ylim_padding": False,
        "selected_weight": selected.tolist(),
        "anchor_count": int(anchors.shape[0]),
    }


def _validate_validation_values(
    spearman_values: dict[tuple[int, str], np.ndarray],
    validation_ids: np.ndarray,
) -> None:
    expected = {(stage, lut) for stage in range(1, NMAX + 1) for lut in FORMAL_LUT_TYPES}
    if set(spearman_values) != expected:
        raise ValueError("Validation Spearman键必须覆盖C1...C5与A2/Weighted")
    if validation_ids.shape != (VALIDATION_COUNT,):
        raise ValueError("Validation state id数量必须为85")
    for key, value in spearman_values.items():
        array = np.asarray(value, dtype=float)
        if array.shape != (VALIDATION_COUNT,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{key}必须是85个finite Spearman")


def plot_figure2_validation_A2_vs_weighted(
    spearman_values: dict[tuple[int, str], np.ndarray],
    validation_ids: np.ndarray,
    output_path: Path,
) -> dict[str, Any]:
    """绘制 Validation A2/Weighted 的 2×2 分布和逐状态曲线。"""

    validation_ids = np.asarray(validation_ids, dtype=np.int64)
    _validate_validation_values(spearman_values, validation_ids)
    all_values = [
        spearman_values[(stage, lut_type)]
        for lut_type in FORMAL_LUT_TYPES
        for stage in range(1, NMAX + 1)
    ]
    y_min, y_max = _spearman_ylim(all_values)
    _configure_matplotlib()
    plt = _pyplot()
    fig, axes = plt.subplots(2, 2, figsize=(8.3, 6.4), sharey=True)
    for column, lut_type in enumerate(FORMAL_LUT_TYPES):
        top = axes[0, column]
        bottom = axes[1, column]
        for stage in range(1, NMAX + 1):
            values = spearman_values[(stage, lut_type)]
            color = C_STAGE_COLORS[stage - 1]
            top.boxplot(
                [values],
                positions=[stage],
                widths=0.58,
                patch_artist=True,
                boxprops={"facecolor": color, "edgecolor": color, "alpha": 0.34, "linewidth": 0.8},
                medianprops={"color": "black", "linewidth": 1.0},
                whiskerprops={"color": color, "linewidth": 0.75},
                capprops={"color": color, "linewidth": 0.75},
                flierprops={"marker": ".", "markersize": 1.3, "alpha": 0.25},
            )
            bottom.plot(
                validation_ids,
                values,
                color=color,
                marker=C_STAGE_MARKERS[stage - 1],
                markevery=5,
                markersize=3.2,
                markerfacecolor="white",
                markeredgewidth=0.65,
                linewidth=0.9,
                alpha=0.9,
                label=f"C{stage}",
            )
        top.set_title(lut_type, fontsize=8.2, fontweight="bold")
        top.set_xlim(0.5, NMAX + 0.5)
        top.set_xticks(range(1, NMAX + 1), [f"C{stage}" for stage in range(1, NMAX + 1)])
        top.set_xlabel("Online C stage")
        top.grid(axis="y", alpha=0.22, linewidth=0.55)
        bottom.set_xlim(int(validation_ids.min()), int(validation_ids.max()))
        bottom.set_xlabel("Validation state ID")
        bottom.grid(axis="y", alpha=0.22, linewidth=0.55)
        if column == 0:
            top.set_ylabel("Spearman rank correlation")
            bottom.set_ylabel("Spearman rank correlation")
    handles = [
        Line2D(
            [0],
            [0],
            color=C_STAGE_COLORS[index],
            marker=C_STAGE_MARKERS[index],
            markerfacecolor="white",
            markeredgewidth=0.65,
            linewidth=0.9,
            markersize=4.0,
            label=f"C{index + 1}",
        )
        for index in range(NMAX)
    ]
    fig.legend(
        handles=handles,
        labels=[f"C{stage}" for stage in range(1, NMAX + 1)],
        loc="upper center",
        ncol=NMAX,
        bbox_to_anchor=(0.5, 0.995),
    )
    fig.suptitle(
        "Independent validation: Y-A2 versus frozen weighted fusion", y=1.025, fontsize=9.0
    )
    common_ylim = _set_common_ylim(axes, y_min, y_max)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), w_pad=1.0, h_pad=1.0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "path": str(output_path),
        "subplot_shape": [2, 2],
        "top_boxplot_count_per_subplot": 5,
        "bottom_curve_count_per_subplot": 5,
        "points_per_curve": VALIDATION_COUNT,
        "all_axes_same_ylim": common_ylim,
        "y_min": y_min,
        "y_max": y_max,
        "ylim_padding": False,
        "marker_contract": "marker and color encode C stage: C1=o, C2=s, C3=^, C4=D, C5=P",
    }


def plot_figure3_validation_median_vs_c(
    validation_summary: pd.DataFrame,
    output_path: Path,
) -> dict[str, Any]:
    """绘制 Validation A2/Weighted 的五阶段 median 曲线。"""

    required = {"lut_type", "C1_median", "C2_median", "C3_median", "C4_median", "C5_median"}
    if not required.issubset(validation_summary.columns):
        raise ValueError(
            f"validation_summary缺少字段：{sorted(required - set(validation_summary.columns))}"
        )
    values: dict[str, np.ndarray] = {}
    for lut_type in FORMAL_LUT_TYPES:
        row = validation_summary[validation_summary["lut_type"] == lut_type]
        if row.shape[0] != 1:
            raise ValueError(f"validation_summary缺少{lut_type}唯一行")
        values[lut_type] = (
            row[[f"C{stage}_median" for stage in range(1, NMAX + 1)]].to_numpy(dtype=float).ravel()
        )
    y_min, y_max = _spearman_ylim(list(values.values()))
    _configure_matplotlib()
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    x = np.arange(1, NMAX + 1)
    for lut_type in FORMAL_LUT_TYPES:
        ax.plot(
            x,
            values[lut_type],
            color=SCHEME_COLORS[lut_type],
            marker=SCHEME_MARKERS[lut_type],
            markersize=5.5,
            markerfacecolor="white",
            markeredgewidth=0.8,
            linewidth=1.45,
            label=lut_type,
        )
    ax.set_xlim(1, NMAX)
    ax.set_xticks(x, [f"C{stage}" for stage in x])
    ax.set_xlabel("Online C stage")
    ax.set_ylabel("Validation median Spearman")
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", alpha=0.24, linewidth=0.6)
    ax.legend(frameon=False, loc="lower left")
    ax.set_title("Validation median ranking consistency", fontsize=9.0, fontweight="bold")
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "path": str(output_path),
        "subplot_shape": [1, 1],
        "curve_count": 2,
        "points_per_curve": NMAX,
        "all_axes_same_ylim": True,
        "y_min": y_min,
        "y_max": y_max,
        "ylim_padding": False,
        "marker_contract": "marker and color encode scheme: Y-A2=s, Y-A-Weighted=D",
    }


__all__ = [
    "C_STAGE_COLORS",
    "C_STAGE_MARKERS",
    "SCHEME_COLORS",
    "SCHEME_MARKERS",
    "plot_figure1_development_landscape",
    "plot_figure2_validation_A2_vs_weighted",
    "plot_figure3_validation_median_vs_c",
]
