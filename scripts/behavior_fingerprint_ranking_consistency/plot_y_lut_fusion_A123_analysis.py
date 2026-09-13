"""A1/A2/A3/等权融合研究的 Python/matplotlib 正式图。

Figure 1 是 2×7 quantitative grid：七列固定 LUT，第一行展示 C1...C5 的
425 状态 Spearman 分布，第二行展示同一批数据的逐状态轨迹。Figure 2 汇总
七种 LUT 在五个在线 C 阶段的 median Spearman。两张图均使用实际数据范围，
不加 padding；曲线同时使用颜色与不同 marker，避免只依赖颜色编码。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .plotting import _pyplot, _spearman_ylim
from .scenario2_y_lut_fusion_A123_analysis import LUT_TYPES, NMAX, STATE_COUNT

C_STAGE_COLORS = ("#2166ac", "#b2182b", "#4d9221", "#762a83", "#e08214")
C_STAGE_MARKERS = ("o", "s", "^", "D", "P")
LUT_COLORS = {
    "Y-A1": "#2166ac",
    "Y-A2": "#b2182b",
    "Y-A3": "#4d9221",
    "Y-A12-Mean": "#762a83",
    "Y-A13-Mean": "#e08214",
    "Y-A23-Mean": "#1b9e77",
    "Y-A123-Mean": "#8c6bb1",
}
LUT_MARKERS = {
    "Y-A1": "o",
    "Y-A2": "s",
    "Y-A3": "^",
    "Y-A12-Mean": "D",
    "Y-A13-Mean": "P",
    "Y-A23-Mean": "X",
    "Y-A123-Mean": "v",
}
MARK_EVERY = 28


def _configure_matplotlib() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 6.8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.7,
            "xtick.major.width": 0.65,
            "ytick.major.width": 0.65,
            "legend.frameon": False,
        }
    )


def _validate_values(values: dict[tuple[int, str], np.ndarray]) -> None:
    expected = {(c_stage, lut_type) for c_stage in range(1, NMAX + 1) for lut_type in LUT_TYPES}
    if set(values) != expected:
        raise ValueError("Spearman数据必须覆盖C1...C5与7种LUT")
    for key, data in values.items():
        array = np.asarray(data, dtype=float)
        if array.shape != (STATE_COUNT,) or not np.all(np.isfinite(array)):
            raise ValueError(f"{key}必须是425个finite值")
        if np.any(array < -1.0) or np.any(array > 1.0):
            raise ValueError(f"{key}包含超出[-1,1]的值")


def _set_common_ylim(axes: np.ndarray, y_min: float, y_max: float) -> bool:
    for axis in axes.flat:
        axis.set_ylim(y_min, y_max)
    return all(
        tuple(float(value) for value in axis.get_ylim()) == (y_min, y_max) for axis in axes.flat
    )


def plot_figure1_A123_robustness(
    spearman_values: dict[tuple[int, str], np.ndarray],
    output_path: Path,
) -> dict[str, Any]:
    """绘制 2×7 Figure 1，每个上排子图5个box、下排5条曲线。"""

    _validate_values(spearman_values)
    all_values = [
        spearman_values[(c_stage, lut_type)]
        for lut_type in LUT_TYPES
        for c_stage in range(1, NMAX + 1)
    ]
    y_min, y_max = _spearman_ylim(all_values)
    _configure_matplotlib()
    plt = _pyplot()
    fig, axes = plt.subplots(2, len(LUT_TYPES), figsize=(22.0, 7.0), sharey=True)
    x_states = np.arange(STATE_COUNT)
    for column, lut_type in enumerate(LUT_TYPES):
        top = axes[0, column]
        bottom = axes[1, column]
        for c_stage in range(1, NMAX + 1):
            values = spearman_values[(c_stage, lut_type)]
            color = C_STAGE_COLORS[c_stage - 1]
            top.boxplot(
                [values],
                positions=[c_stage],
                widths=0.58,
                patch_artist=True,
                showmeans=False,
                boxprops={
                    "facecolor": color,
                    "edgecolor": color,
                    "alpha": 0.34,
                    "linewidth": 0.75,
                },
                medianprops={"color": "black", "linewidth": 0.95},
                whiskerprops={"color": color, "linewidth": 0.7},
                capprops={"color": color, "linewidth": 0.7},
                flierprops={"marker": ".", "markersize": 1.2, "alpha": 0.24},
            )
            bottom.plot(
                x_states,
                values,
                color=color,
                marker=C_STAGE_MARKERS[c_stage - 1],
                markevery=MARK_EVERY,
                markersize=2.8,
                markerfacecolor="white",
                markeredgewidth=0.6,
                linewidth=0.78,
                alpha=0.9,
            )
        top.set_title(lut_type, fontsize=7.5, fontweight="bold")
        top.set_xlim(0.5, NMAX + 0.5)
        top.set_xticks(range(1, NMAX + 1), [f"C{stage}" for stage in range(1, NMAX + 1)])
        top.set_xlabel("Online C stage")
        top.grid(axis="y", alpha=0.22, linewidth=0.5)
        bottom.set_xlim(0, STATE_COUNT - 1)
        bottom.set_xlabel("State ID")
        bottom.grid(axis="y", alpha=0.22, linewidth=0.5)
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
            markeredgewidth=0.6,
            linewidth=0.8,
            markersize=3.8,
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
        "Robustness of fixed Y-A LUT fingerprints across online C stages",
        y=1.025,
        fontsize=9.0,
    )
    common_ylim = _set_common_ylim(axes, y_min, y_max)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94), w_pad=0.8, h_pad=1.0)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "path": str(output_path),
        "subplot_shape": [2, len(LUT_TYPES)],
        "boxplots_per_top_subplot": NMAX,
        "curves_per_bottom_subplot": NMAX,
        "points_per_curve": STATE_COUNT,
        "all_axes_same_ylim": common_ylim,
        "y_min": y_min,
        "y_max": y_max,
        "ylim_padding": False,
        "marker_contract": "marker and color encode online C stage: C1=o, C2=s, C3=^, C4=D, C5=P",
        "lut_columns": list(LUT_TYPES),
        "all_spearman_value_count": len(all_values) * STATE_COUNT,
    }


def plot_figure2_median_vs_c_stage(
    spearman_summary: pd.DataFrame,
    output_path: Path,
) -> dict[str, Any]:
    """绘制 7 条 LUT median Spearman 趋势曲线。"""

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
            raise ValueError(f"{lut_type}必须包含C1...C5五行")
        values = group["median"].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{lut_type} median包含非有限值")
        medians[lut_type] = values
    y_min, y_max = _spearman_ylim(list(medians.values()))
    _configure_matplotlib()
    plt = _pyplot()
    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    x = np.arange(1, NMAX + 1)
    for lut_type in LUT_TYPES:
        ax.plot(
            x,
            medians[lut_type],
            color=LUT_COLORS[lut_type],
            marker=LUT_MARKERS[lut_type],
            markersize=5.2,
            markerfacecolor="white",
            markeredgewidth=0.75,
            linewidth=1.35,
            label=lut_type,
        )
    ax.set_xlim(1, NMAX)
    ax.set_xticks(x, [f"C{stage}" for stage in x])
    ax.set_xlabel("Online C stage")
    ax.set_ylabel("Median Spearman rank correlation")
    ax.set_ylim(y_min, y_max)
    ax.grid(axis="y", alpha=0.24, linewidth=0.55)
    ax.legend(frameon=False, ncol=4, loc="lower left", columnspacing=1.0)
    ax.set_title(
        "Fixed Y-A LUT robustness across online DPD mismatch",
        fontsize=9.0,
        fontweight="bold",
    )
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {
        "path": str(output_path),
        "subplot_shape": [1, 1],
        "curve_count": len(LUT_TYPES),
        "points_per_curve": NMAX,
        "all_axes_same_ylim": True,
        "y_min": y_min,
        "y_max": y_max,
        "ylim_padding": False,
        "marker_contract": "marker and color encode LUT type: "
        "Y-A1=o, Y-A2=s, Y-A3=^, Y-A12-Mean=D, Y-A13-Mean=P, "
        "Y-A23-Mean=X, Y-A123-Mean=v",
        "x_stages": list(range(1, NMAX + 1)),
    }


__all__ = [
    "C_STAGE_COLORS",
    "C_STAGE_MARKERS",
    "LUT_COLORS",
    "LUT_MARKERS",
    "MARK_EVERY",
    "plot_figure1_A123_robustness",
    "plot_figure2_median_vs_c_stage",
]
