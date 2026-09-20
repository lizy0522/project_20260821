"""Scenario 2 Y-A LUT 融合研究的两张 Nature-style Python/matplotlib 图。

Figure 1 是定量网格：三列固定 LUT、第一行逐 C 箱线图、第二行逐状态曲线，
用于同时检查每个固定 LUT 在 C1...C5 下的分布和状态级变化。Figure 2 是鲁棒性
总结曲线：三种 LUT 的五个 C-stage median，用于支持“高且平”的比较结论。
所有曲线同时用颜色和 marker 区分；每张图的纵轴严格使用该图实际数据的 min/max，
不添加 padding。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from behavior_fingerprint_ranking_consistency.shared.plotting import _pyplot, _spearman_ylim
from behavior_fingerprint_ranking_consistency.shared.scenario2_y_lut_fusion_analysis import (  # noqa: E501
    LUT_TYPES,
    NMAX,
    STATE_COUNT,
)

C_STAGE_COLORS = ("#2166ac", "#b2182b", "#4d9221", "#762a83", "#e08214")
C_STAGE_MARKERS = ("o", "s", "^", "D", "P")
LUT_COLORS = {"Y-A1": "#2166ac", "Y-A2": "#b2182b", "Y-A12-Mean": "#4d9221"}
LUT_MARKERS = {"Y-A1": "o", "Y-A2": "s", "Y-A12-Mean": "^"}
MARK_EVERY = 28


def _configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "legend.frameon": False,
        }
    )


def _validate_combination_values(
    values: dict[tuple[int, str], np.ndarray],
) -> None:
    expected_keys = {
        (c_stage, lut_type) for c_stage in range(1, NMAX + 1) for lut_type in LUT_TYPES
    }
    if set(values) != expected_keys:
        raise ValueError("Spearman组合键必须覆盖C1...C5与三种LUT")
    for key, array in values.items():
        data = np.asarray(array, dtype=float)
        if data.shape != (STATE_COUNT,) or not np.all(np.isfinite(data)):
            raise ValueError(f"{key}必须是425个finite Spearman")
        if np.any(data < -1.0) or np.any(data > 1.0):
            raise ValueError(f"{key}包含超出[-1,1]的Spearman")


def _set_common_ylim(axes: np.ndarray, y_min: float, y_max: float) -> bool:
    for axis in axes.flat:
        axis.set_ylim(y_min, y_max)
    return all(
        tuple(float(value) for value in axis.get_ylim()) == (y_min, y_max) for axis in axes.flat
    )


def plot_figure1_y_lut_fingerprint_robustness(
    spearman_values: dict[tuple[int, str], np.ndarray],
    output_path: Path,
) -> dict[str, Any]:
    """绘制 Figure 1：2×3，第一行 5 boxes，第二行 5 条逐状态曲线。"""

    _validate_combination_values(spearman_values)
    all_values = [
        spearman_values[(c_stage, lut_type)]
        for lut_type in LUT_TYPES
        for c_stage in range(1, NMAX + 1)
    ]
    y_min, y_max = _spearman_ylim(all_values)
    _configure_matplotlib()
    plt = _pyplot()
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.8), sharey=True)
    for column, lut_type in enumerate(LUT_TYPES):
        top = axes[0, column]
        bottom = axes[1, column]
        for c_stage in range(1, NMAX + 1):
            values = spearman_values[(c_stage, lut_type)]
            result = top.boxplot(
                [values],
                positions=[c_stage],
                widths=0.58,
                patch_artist=True,
                showmeans=False,
                boxprops={
                    "facecolor": C_STAGE_COLORS[c_stage - 1],
                    "edgecolor": C_STAGE_COLORS[c_stage - 1],
                    "alpha": 0.35,
                    "linewidth": 0.8,
                },
                medianprops={"color": "black", "linewidth": 1.0},
                whiskerprops={"color": C_STAGE_COLORS[c_stage - 1], "linewidth": 0.75},
                capprops={"color": C_STAGE_COLORS[c_stage - 1], "linewidth": 0.75},
                flierprops={"marker": ".", "markersize": 1.3, "alpha": 0.28},
            )
            for box in result["boxes"]:
                box.set_label(f"C{c_stage}")
            x = np.arange(STATE_COUNT)
            bottom.plot(
                x,
                values,
                color=C_STAGE_COLORS[c_stage - 1],
                marker=C_STAGE_MARKERS[c_stage - 1],
                markevery=MARK_EVERY,
                markersize=3.0,
                markerfacecolor="white",
                markeredgewidth=0.65,
                linewidth=0.9,
                alpha=0.9,
                label=f"C{c_stage}",
            )
        top.set_title(lut_type, fontsize=8.0, fontweight="bold")
        top.set_xlim(0.5, NMAX + 0.5)
        top.set_xticks(range(1, NMAX + 1), [f"C{stage}" for stage in range(1, NMAX + 1)])
        top.set_xlabel("Online C stage")
        top.grid(axis="y", alpha=0.22, linewidth=0.55)
        bottom.set_xlim(0, STATE_COUNT - 1)
        bottom.set_xlabel("State ID")
        bottom.grid(axis="y", alpha=0.22, linewidth=0.55)
        if column == 0:
            top.set_ylabel("Spearman rank correlation")
            bottom.set_ylabel("Spearman rank correlation")
    handles = [
        plt.Line2D(
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
        bbox_to_anchor=(0.5, 0.99),
    )
    fig.suptitle(
        "Robustness of fixed Y-A LUT fingerprints across online C stages", y=1.015, fontsize=9.0
    )
    common_ylim = _set_common_ylim(axes, y_min, y_max)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "path": str(output_path),
        "subplot_shape": [2, 3],
        "boxplots_per_top_subplot": 5,
        "curves_per_bottom_subplot": 5,
        "points_per_curve": STATE_COUNT,
        "all_axes_same_ylim": common_ylim,
        "y_min": y_min,
        "y_max": y_max,
        "ylim_padding": False,
        "marker_contract": "marker and color encode online C stage: C1=o, C2=s, C3=^, C4=D, C5=P",
        "lut_columns": list(LUT_TYPES),
    }


def plot_figure2_median_spearman_vs_c_stage(
    spearman_summary: pd.DataFrame,
    output_path: Path,
) -> dict[str, Any]:
    """绘制 Figure 2：三种固定 LUT 的五点 median Spearman 曲线。"""

    required = {"requested_C_n", "lut_fingerprint_type", "median"}
    if not required.issubset(spearman_summary.columns):
        raise ValueError(
            f"spearman_summary缺少字段：{sorted(required - set(spearman_summary.columns))}"
        )
    medians: dict[str, np.ndarray] = {}
    for lut_type in LUT_TYPES:
        group = spearman_summary[spearman_summary["lut_fingerprint_type"] == lut_type].sort_values(
            "requested_C_n"
        )
        if group.shape[0] != NMAX or not np.array_equal(
            group["requested_C_n"].to_numpy(dtype=int), np.arange(1, NMAX + 1)
        ):
            raise ValueError(f"{lut_type} summary必须包含C1...C5各一行")
        values = group["median"].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{lut_type} median包含非有限值")
        medians[lut_type] = values
    y_min, y_max = _spearman_ylim(list(medians.values()))
    _configure_matplotlib()
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(6.3, 4.1))
    x = np.arange(1, NMAX + 1)
    for lut_type in LUT_TYPES:
        ax.plot(
            x,
            medians[lut_type],
            color=LUT_COLORS[lut_type],
            marker=LUT_MARKERS[lut_type],
            markersize=5.0,
            markerfacecolor="white",
            markeredgewidth=0.8,
            linewidth=1.35,
            label=lut_type,
        )
    ax.set_xlim(1, NMAX)
    ax.set_xticks(x, [f"C{stage}" for stage in x])
    ax.set_xlabel("Online C stage")
    ax.set_ylabel("Median Spearman rank correlation")
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", alpha=0.24, linewidth=0.6)
    ax.legend(frameon=False, ncol=3, loc="lower left")
    ax.set_title(
        "Fixed Y-A LUT robustness across online DPD mismatch", fontsize=8.7, fontweight="bold"
    )
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "path": str(output_path),
        "subplot_shape": [1, 1],
        "curve_count": 3,
        "points_per_curve": 5,
        "all_axes_same_ylim": True,
        "y_min": y_min,
        "y_max": y_max,
        "ylim_padding": False,
        "marker_contract": "marker and color encode LUT type: Y-A1=o, Y-A2=s, Y-A12-Mean=^",
        "x_stages": list(range(1, NMAX + 1)),
    }


__all__ = [
    "C_STAGE_COLORS",
    "C_STAGE_MARKERS",
    "LUT_COLORS",
    "LUT_MARKERS",
    "MARK_EVERY",
    "plot_figure1_y_lut_fingerprint_robustness",
    "plot_figure2_median_spearman_vs_c_stage",
]
