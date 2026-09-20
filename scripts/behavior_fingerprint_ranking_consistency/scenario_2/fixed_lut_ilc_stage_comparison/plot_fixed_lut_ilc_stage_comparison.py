"""
基于已完成的 scenario_2_all_ilc Spearman 长表生成两幅固定 LUT 阶段比较图。

本模块只重组结果并绘图，不重新训练模型、不重新计算 CNMSE、排名或 Spearman。
Figure 1 比较 X-A、固定 Y-A 和 Real-B 三条路线；Figure 2 只比较固定 Y-A
指纹空间。两幅图均为 2×5，列为固定离线 A 阶段，子图内横坐标为在线 C 阶段。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from behavior_fingerprint_ranking_consistency.shared.plotting import _pyplot, _spearman_ylim

STATE_COUNT = 425
NMAX = 5
STATE_COLUMNS = ("state_id", "funMng", "funAng", "secMng", "secAng", "Vm", "Pin")
REQUIRED_COLUMNS = (*STATE_COLUMNS, "requested_n1", "requested_n2", "lut_type", "spearman")
ROUTES = ("X-A", "Y-A", "Real-B")
YA_ONLY_ROUTES = ("Y-A",)
C_STAGE_LABELS = tuple(f"C{stage}" for stage in range(1, NMAX + 1))
C_STAGE_COLORS = ("#2166ac", "#b2182b", "#4d9221", "#762a83", "#e08214")
C_STAGE_MARKERS = ("o", "s", "^", "D", "P")
ROUTE_COLORS = {"X-A": "#2166ac", "Y-A": "#b2182b", "Real-B": "#4d9221"}
ROUTE_LINESTYLES = {"X-A": "-", "Y-A": "--", "Real-B": ":"}
MARK_EVERY = 28


@dataclass(frozen=True)
class FixedLUTStageData:
    """按 ``(C stage, route, A stage)`` 索引的425点Spearman结果。"""

    source_path: Path
    nmax: int
    state_frame: pd.DataFrame
    values: dict[tuple[Any, ...], np.ndarray]


@dataclass(frozen=True)
class FixedLUTStageFigureResult:
    """两幅组合图、summary和验证元数据。"""

    data: FixedLUTStageData
    figure1_summary: pd.DataFrame
    figure2_summary: pd.DataFrame
    figure1_validation: dict[str, Any]
    figure2_validation: dict[str, Any]


def _validate_state_vector(group: pd.DataFrame, label: str) -> np.ndarray:
    """按state_id排序并确认一组数据恰好覆盖425个状态。"""

    if group.empty:
        raise ValueError(f"{label}没有数据")
    if group["state_id"].isna().any():
        raise ValueError(f"{label}包含空state_id")
    group = group.copy()
    group["state_id"] = group["state_id"].astype(int)
    if group["state_id"].duplicated().any():
        collapsed: list[dict[str, Any]] = []
        for state_id, state_group in group.groupby("state_id", sort=True):
            values = state_group["spearman"].to_numpy(dtype=float)
            if not np.allclose(values, values[0], rtol=0, atol=1e-12):
                raise ValueError(f"{label}的state_id={state_id}重复值不一致")
            collapsed.append({"state_id": int(state_id), "spearman": float(values[0])})
        group = pd.DataFrame(collapsed)
    group = group.sort_values("state_id")
    expected_ids = np.arange(STATE_COUNT, dtype=np.int64)
    if not np.array_equal(group["state_id"].to_numpy(dtype=np.int64), expected_ids):
        raise ValueError(f"{label}必须覆盖按0...424排列的425个状态")
    values = group["spearman"].to_numpy(dtype=float)
    if not np.all(np.isfinite(values)) or np.any(values < -1) or np.any(values > 1):
        raise ValueError(f"{label}包含非法Spearman")
    return values


def _validate_metadata(frame: pd.DataFrame) -> pd.DataFrame:
    state_frame = frame.loc[:, list(STATE_COLUMNS)].drop_duplicates("state_id")
    state_frame["state_id"] = state_frame["state_id"].astype(int)
    state_frame = state_frame.sort_values("state_id").reset_index(drop=True)
    if not np.array_equal(
        state_frame["state_id"].to_numpy(dtype=np.int64),
        np.arange(STATE_COUNT, dtype=np.int64),
    ):
        raise ValueError("state metadata必须覆盖state_id=0...424")
    return state_frame


def load_fixed_lut_stage_data(source_path: Path) -> FixedLUTStageData:
    """读取并索引已有 Spearman 长表，不触碰raw或模型。"""

    source_path = Path(source_path)
    if not source_path.is_file():
        raise FileNotFoundError(f"缺少Spearman长表：{source_path}")
    frame = pd.read_csv(source_path)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Spearman长表缺少字段：{missing}")
    frame = frame.copy()
    frame["state_id"] = frame["state_id"].astype(int)
    frame["requested_n1"] = frame["requested_n1"].astype(int)
    frame["requested_n2"] = pd.to_numeric(frame["requested_n2"], errors="coerce")
    frame["spearman"] = frame["spearman"].astype(float)
    if set(frame["lut_type"].dropna().unique()) != set(ROUTES):
        raise ValueError("lut_type必须包含X-A、Y-A、Real-B")
    if frame["state_id"].min() != 0 or frame["state_id"].max() != STATE_COUNT - 1:
        raise ValueError("state_id范围必须是0...424")
    if not np.all(np.isfinite(frame["spearman"].to_numpy(dtype=float))):
        raise ValueError("Spearman长表包含NaN或Inf")

    n1_values = tuple(sorted(frame["requested_n1"].unique()))
    y_n2_values = tuple(
        sorted(frame.loc[frame["lut_type"] == "Y-A", "requested_n2"].dropna().astype(int).unique())
    )
    if n1_values != tuple(range(1, NMAX + 1)) or y_n2_values != tuple(range(1, NMAX + 1)):
        raise ValueError(f"requested n1/n2必须为1...5，实际为{n1_values}/{y_n2_values}")
    state_frame = _validate_metadata(frame)
    values: dict[tuple[Any, ...], np.ndarray] = {}
    for c_stage in range(1, NMAX + 1):
        for route in ("X-A", "Real-B"):
            group = frame[
                (frame["requested_n1"] == c_stage)
                & (frame["lut_type"] == route)
            ]
            values[(c_stage, route)] = _validate_state_vector(
                group,
                f"C{c_stage}->{route}",
            )
        for a_stage in range(1, NMAX + 1):
            group = frame[
                (frame["requested_n1"] == c_stage)
                & (frame["lut_type"] == "Y-A")
                & (frame["requested_n2"] == a_stage)
            ]
            values[(c_stage, "Y-A", a_stage)] = _validate_state_vector(
                group,
                f"C{c_stage}->Y-A{a_stage}",
            )

    for c_stage in range(1, NMAX + 1):
        for route in ("X-A", "Real-B"):
            reference = values[(c_stage, route)]
            if reference.shape != (STATE_COUNT,):
                raise RuntimeError(f"{route} C{c_stage}数据长度错误")
    return FixedLUTStageData(
        source_path=source_path,
        nmax=NMAX,
        state_frame=state_frame,
        values=values,
    )


def _summary_rows(
    data: FixedLUTStageData,
    *,
    figure: str,
    routes: tuple[str, ...],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for a_stage in range(1, NMAX + 1):
        for c_stage in range(1, NMAX + 1):
            for route in routes:
                key = (c_stage, route) if route != "Y-A" else (c_stage, route, a_stage)
                values = data.values[key]
                rows.append({
                    "figure": figure,
                    "fixed_A_stage": a_stage,
                    "C_stage": c_stage,
                    "route": route,
                    "count": int(values.size),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values, ddof=1)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                })
    return pd.DataFrame(rows)


def _axes_ylim(axes: np.ndarray, y_min: float, y_max: float) -> bool:
    limits: list[tuple[float, float]] = []
    for axis in axes.flat:
        axis.set_ylim(y_min, y_max)
        limits.append(tuple(float(value) for value in axis.get_ylim()))
    return all(np.array_equal(limit, (y_min, y_max)) for limit in limits)


def _route_boxplot(axis: Any, values: list[np.ndarray], positions: np.ndarray, route: str) -> None:
    result = axis.boxplot(
        values,
        positions=positions,
        widths=0.16,
        patch_artist=True,
        showmeans=False,
        boxprops={
            "facecolor": ROUTE_COLORS[route],
            "edgecolor": ROUTE_COLORS[route],
            "alpha": 0.28,
        },
        medianprops={"color": "black", "linewidth": 1.1},
        whiskerprops={"color": ROUTE_COLORS[route], "linewidth": 0.9},
        capprops={"color": ROUTE_COLORS[route], "linewidth": 0.9},
        flierprops={"marker": ".", "markersize": 1.8, "alpha": 0.35},
    )
    for box in result["boxes"]:
        box.set_label(route)


def plot_figure1_three_routes(
    data: FixedLUTStageData,
    output_path: Path,
) -> dict[str, Any]:
    """Figure 1：固定A阶段下，三条路线的15 box/15 curve比较。"""

    all_values = [
        value
        for key, value in data.values.items()
        if len(key) == 2 or key[2] in range(1, NMAX + 1)
    ]
    y_min, y_max = _spearman_ylim(all_values)
    plt = _pyplot()
    fig, axes = plt.subplots(2, 5, figsize=(24.0, 9.2), sharey=True)
    centers = np.arange(1, NMAX + 1, dtype=float)
    offsets = {"X-A": -0.22, "Y-A": 0.0, "Real-B": 0.22}
    for column_index, a_stage in enumerate(range(1, NMAX + 1)):
        top = axes[0, column_index]
        bottom = axes[1, column_index]
        for route in ROUTES:
            route_values = [
                data.values[(c_stage, route)]
                if route != "Y-A"
                else data.values[(c_stage, route, a_stage)]
                for c_stage in range(1, NMAX + 1)
            ]
            _route_boxplot(top, route_values, centers + offsets[route], route)
        top.set_xticks(centers, C_STAGE_LABELS)
        top.set_xlim(0.5, 5.5)
        top.set_title(f"Fixed LUT Y-A Stage: A{a_stage}", fontsize=10)
        top.grid(axis="y", alpha=0.22)
        for c_stage, (color, marker) in enumerate(
            zip(C_STAGE_COLORS, C_STAGE_MARKERS, strict=True),
            start=1,
        ):
            for route in ROUTES:
                values = (
                    data.values[(c_stage, route)]
                    if route != "Y-A"
                    else data.values[(c_stage, route, a_stage)]
                )
                bottom.plot(
                    np.arange(STATE_COUNT),
                    values,
                    color=color,
                    linestyle=ROUTE_LINESTYLES[route],
                    linewidth=0.8,
                    alpha=0.82,
                    marker=marker,
                    markersize=3.0,
                    markevery=MARK_EVERY,
                )
        bottom.set_xlim(0, STATE_COUNT - 1)
        bottom.set_xticks((0, 50, 100, 150, 200, 250, 300, 350, 400, 424))
        bottom.set_xlabel("State ID")
        bottom.set_title(f"Online Query C1...C5; fixed A{a_stage}", fontsize=10)
        bottom.grid(True, alpha=0.22)
        if column_index == 0:
            top.set_ylabel("Spearman Rank Correlation")
            bottom.set_ylabel("Spearman Rank Correlation")
        else:
            top.tick_params(axis="y", labelleft=False)
            bottom.tick_params(axis="y", labelleft=False)
    for axis in axes.flat:
        axis.set_ylim(y_min, y_max)
    c_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            marker=marker,
            linestyle="-",
            linewidth=1.0,
            markersize=4,
            label=f"C{index}",
        )
        for index, (color, marker) in enumerate(
            zip(C_STAGE_COLORS, C_STAGE_MARKERS, strict=True),
            start=1,
        )
    ]
    route_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle=ROUTE_LINESTYLES[route],
            linewidth=1.2,
            label=f"{route} LUT",
        )
        for route in ROUTES
    ]
    fig.legend(
        c_handles,
        [handle.get_label() for handle in c_handles],
        loc="lower center",
        bbox_to_anchor=(0.38, 0.005),
        ncol=5,
        frameon=False,
        title="Online Query stage (color + marker)",
        fontsize=8,
        title_fontsize=8,
    )
    fig.legend(
        route_handles,
        [handle.get_label() for handle in route_handles],
        loc="lower center",
        bbox_to_anchor=(0.75, 0.005),
        ncol=3,
        frameon=False,
        title="LUT route (line style)",
        fontsize=8,
        title_fontsize=8,
    )
    fig.suptitle("Ranking Consistency of Three LUT Fingerprint Types", fontsize=15)
    fig.tight_layout(rect=(0.0, 0.12, 1.0, 0.94))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    all_axes_same = _axes_ylim(axes, y_min, y_max)
    plt.close(fig)
    return {
        "figure": "Figure1",
        "subplot_shape": [2, 5],
        "boxplots_per_top_subplot": 15,
        "curves_per_bottom_subplot": 15,
        "y_min": y_min,
        "y_max": y_max,
        "all_axes_same_ylim": all_axes_same,
        "ylim_padding": False,
        "marker_contract": "marker and color encode C stage; line style encodes LUT route",
        "png_path": str(output_path),
    }


def plot_figure2_ya_only(
    data: FixedLUTStageData,
    output_path: Path,
) -> dict[str, Any]:
    """Figure 2：固定A阶段下，仅Y-A路线的5 box/5 curve比较。"""

    all_values = [value for key, value in data.values.items() if len(key) == 3]
    y_min, y_max = _spearman_ylim(all_values)
    plt = _pyplot()
    fig, axes = plt.subplots(2, 5, figsize=(23.0, 9.2), sharey=True)
    for column_index, a_stage in enumerate(range(1, NMAX + 1)):
        top = axes[0, column_index]
        bottom = axes[1, column_index]
        values = [data.values[(c_stage, "Y-A", a_stage)] for c_stage in range(1, NMAX + 1)]
        top.boxplot(
            values,
            positions=np.arange(1, NMAX + 1, dtype=float),
            widths=0.45,
            patch_artist=True,
            showmeans=False,
            boxprops={"facecolor": "#f4c7c3", "edgecolor": "#7f1d1d", "alpha": 0.5},
            medianprops={"color": "black", "linewidth": 1.1},
            whiskerprops={"color": "#7f1d1d", "linewidth": 0.9},
            capprops={"color": "#7f1d1d", "linewidth": 0.9},
            flierprops={"marker": ".", "markersize": 1.8, "alpha": 0.35},
        )
        top.set_xticks(np.arange(1, NMAX + 1), C_STAGE_LABELS)
        top.set_xlim(0.5, 5.5)
        top.set_title(f"Fixed LUT Y-A Stage: A{a_stage}", fontsize=10)
        top.grid(axis="y", alpha=0.22)
        for c_stage, (color, marker) in enumerate(
            zip(C_STAGE_COLORS, C_STAGE_MARKERS, strict=True),
            start=1,
        ):
            bottom.plot(
                np.arange(STATE_COUNT),
                data.values[(c_stage, "Y-A", a_stage)],
                color=color,
                linestyle="-",
                linewidth=1.0,
                alpha=0.86,
                marker=marker,
                markersize=3.2,
                markevery=MARK_EVERY,
                label=f"C{c_stage}",
            )
        bottom.set_xlim(0, STATE_COUNT - 1)
        bottom.set_xticks((0, 50, 100, 150, 200, 250, 300, 350, 400, 424))
        bottom.set_xlabel("State ID")
        bottom.set_title(f"Online Query C1...C5; fixed A{a_stage}", fontsize=10)
        bottom.grid(True, alpha=0.22)
        if column_index == 0:
            top.set_ylabel("Spearman Rank Correlation")
            bottom.set_ylabel("Spearman Rank Correlation")
        else:
            top.tick_params(axis="y", labelleft=False)
            bottom.tick_params(axis="y", labelleft=False)
    for axis in axes.flat:
        axis.set_ylim(y_min, y_max)
    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            marker=marker,
            linestyle="-",
            linewidth=1.0,
            markersize=4,
            label=f"C{index}",
        )
        for index, (color, marker) in enumerate(
            zip(C_STAGE_COLORS, C_STAGE_MARKERS, strict=True),
            start=1,
        )
    ]
    fig.legend(
        handles,
        [handle.get_label() for handle in handles],
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=5,
        frameon=False,
        title="Online Query stage (color + marker)",
        fontsize=8,
        title_fontsize=8,
    )
    fig.suptitle(
        "Ranking Consistency of Fixed Y-A LUT Fingerprint Space",
        fontsize=15,
    )
    fig.tight_layout(rect=(0.0, 0.10, 1.0, 0.94))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    all_axes_same = _axes_ylim(axes, y_min, y_max)
    plt.close(fig)
    return {
        "figure": "Figure2",
        "subplot_shape": [2, 5],
        "boxplots_per_top_subplot": 5,
        "curves_per_bottom_subplot": 5,
        "y_min": y_min,
        "y_max": y_max,
        "all_axes_same_ylim": all_axes_same,
        "ylim_padding": False,
        "marker_contract": "marker and color encode C stage; all curves use solid line",
        "png_path": str(output_path),
    }


def build_fixed_lut_stage_figures(
    source_path: Path,
    output_root: Path,
) -> FixedLUTStageFigureResult:
    """构建数据索引、绘制两幅图并返回验证信息。"""

    data = load_fixed_lut_stage_data(source_path)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    figure1_summary = _summary_rows(data, figure="Figure1", routes=ROUTES)
    figure2_summary = _summary_rows(data, figure="Figure2", routes=YA_ONLY_ROUTES)
    figure1_validation = plot_figure1_three_routes(
        data,
        output_root / "figure1_three_lut_routes_by_fixed_A_stage.png",
    )
    figure2_validation = plot_figure2_ya_only(
        data,
        output_root / "figure2_YA_only_by_fixed_A_stage.png",
    )
    return FixedLUTStageFigureResult(
        data=data,
        figure1_summary=figure1_summary,
        figure2_summary=figure2_summary,
        figure1_validation=figure1_validation,
        figure2_validation=figure2_validation,
    )


__all__ = [
    "C_STAGE_COLORS",
    "C_STAGE_MARKERS",
    "FixedLUTStageData",
    "FixedLUTStageFigureResult",
    "build_fixed_lut_stage_figures",
    "load_fixed_lut_stage_data",
    "plot_figure1_three_routes",
    "plot_figure2_ya_only",
]
