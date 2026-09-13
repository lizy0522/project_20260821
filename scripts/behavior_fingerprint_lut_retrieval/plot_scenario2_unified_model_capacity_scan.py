"""Python/matplotlib figures for the unified model-capacity study.

The figure contract is intentionally explicit: the figures test whether a
Development-selected common Y-Aend/Y-C2 model improves B-segment
generalisation and whether that improvement transfers to C2 -> Aend retrieval.
All panels are generated with Python/matplotlib only; finite data ranges are
used as axis limits and the formal -40 dB line is included where relevant.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from .scenario2_unified_model_capacity_scan import DPD_SHAREABLE_THRESHOLD_DB


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.frameon": False,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _save_png(fig: plt.Figure, output_path: Path) -> dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    width, height = fig.get_size_inches()
    plt.close(fig)
    return {
        "path": str(output_path),
        "width_in": float(width),
        "height_in": float(height),
        "dpi": 300,
    }


def _limits(values: np.ndarray, references: tuple[float, ...] = ()) -> tuple[float, float]:
    values = np.asarray(values, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("figure至少需要一个finite值")
    combined = np.concatenate([finite, np.asarray(references, dtype=float)])
    combined = combined[np.isfinite(combined)]
    low = float(np.min(combined))
    high = float(np.max(combined))
    if low == high:
        delta = max(abs(low) * 1e-6, np.finfo(float).eps * 10.0)
        return low - delta, high + delta
    return low, high


def plot_candidate_scan(
    scan: pd.DataFrame, selected: dict[str, Any], output_path: Path
) -> dict[str, Any]:
    """Figure 1: candidate complexity versus worst-side Development generalisation."""

    _configure_matplotlib()
    required = {
        "candidate_id",
        "n_complex_coefficients",
        "worst_side_B_median_dB",
        "order_profile",
        "lambda",
        "is_baseline",
    }
    missing = sorted(required - set(scan.columns))
    if missing:
        raise ValueError(f"candidate scan缺少字段：{missing}")
    figure, axis = plt.subplots(figsize=(8.2, 5.3), constrained_layout=True)
    profiles = ["P7", "P9", "P11", "P13"]
    markers = {"P7": "o", "P9": "s", "P11": "^", "P13": "D"}
    lambda_values = sorted(scan["lambda"].astype(float).unique())
    color_map = plt.get_cmap("viridis")
    color_by_lambda = {
        value: color_map(index / max(len(lambda_values) - 1, 1))
        for index, value in enumerate(lambda_values)
    }
    for profile in profiles:
        group = scan.loc[scan["order_profile"] == profile]
        axis.scatter(
            group["n_complex_coefficients"],
            group["worst_side_B_median_dB"],
            c=[color_by_lambda[float(value)] for value in group["lambda"]],
            marker=markers[profile],
            s=32,
            alpha=0.75,
            linewidths=0.25,
            edgecolors="white",
            label=profile,
        )
    for _, row in scan.loc[scan["is_baseline"].astype(bool)].iterrows():
        axis.scatter(
            [row["n_complex_coefficients"]],
            [row["worst_side_B_median_dB"]],
            marker="*",
            s=145,
            facecolor="white",
            edgecolor="black",
            linewidth=0.9,
            zorder=5,
        )
        axis.annotate(
            "Baseline",
            (row["n_complex_coefficients"], row["worst_side_B_median_dB"]),
            xytext=(5, 6),
            textcoords="offset points",
            fontsize=7,
        )
    selected_row = scan.loc[scan["candidate_id"] == int(selected["candidate_id"])]
    if selected_row.shape[0] == 1:
        row = selected_row.iloc[0]
        axis.scatter(
            [row["n_complex_coefficients"]],
            [row["worst_side_B_median_dB"]],
            marker="P",
            s=110,
            facecolor="#d73027",
            edgecolor="black",
            linewidth=0.7,
            zorder=6,
        )
        axis.annotate(
            "Selected",
            (row["n_complex_coefficients"], row["worst_side_B_median_dB"]),
            xytext=(5, -12),
            textcoords="offset points",
            fontsize=7,
            color="#d73027",
        )
    axis.set_xlabel("Number of complex coefficients")
    axis.set_ylabel("Worst-side B median NMSE (dB)")
    axis.set_title("Development model-capacity scan")
    axis.set_ylim(*_limits(scan["worst_side_B_median_dB"].to_numpy(dtype=float)))
    axis.set_xlim(
        float(scan["n_complex_coefficients"].min()), float(scan["n_complex_coefficients"].max())
    )
    profile_handles = [
        Line2D([0], [0], marker=markers[p], color="black", linestyle="None", label=p)
        for p in profiles
    ]
    axis.legend(
        handles=profile_handles, title="Order profile", loc="best", fontsize=7, title_fontsize=7
    )
    sm = mpl.cm.ScalarMappable(
        norm=mpl.colors.Normalize(vmin=0, vmax=max(len(lambda_values) - 1, 1)),
        cmap=color_map,
    )
    sm.set_array([])
    colorbar = figure.colorbar(sm, ax=axis, pad=0.02)
    colorbar.set_label("Ridge lambda (grid order)")
    if len(lambda_values) <= 10:
        colorbar.set_ticks(np.linspace(0, max(len(lambda_values) - 1, 1), len(lambda_values)))
        colorbar.set_ticklabels(
            ["OLS" if value == 0 else f"{value:.0e}" for value in lambda_values]
        )
    details = _save_png(figure, output_path)
    details.update(
        {
            "candidate_count": int(scan.shape[0]),
            "order_profile_count": len(profiles),
            "selected_candidate_id": int(selected["candidate_id"]),
            "y_min": float(scan["worst_side_B_median_dB"].min()),
            "y_max": float(scan["worst_side_B_median_dB"].max()),
        }
    )
    return details


def plot_pa_quartile_gains(quartile_summary: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    """Figure 2: model B-generalisation gain by PA nonlinearity quartile."""

    _configure_matplotlib()
    required = {"pa_nonlinearity_quartile", "gain_Y_Aend_B_median_dB", "gain_Y_C2_B_median_dB"}
    missing = sorted(required - set(quartile_summary.columns))
    if missing:
        raise ValueError(f"quartile summary缺少字段：{missing}")
    ordered = (
        quartile_summary.set_index("pa_nonlinearity_quartile")
        .loc[["Q1", "Q2", "Q3", "Q4"]]
        .reset_index()
    )
    x = np.arange(4, dtype=float)
    figure, axis = plt.subplots(figsize=(7.1, 4.6), constrained_layout=True)
    axis.plot(
        x,
        ordered["gain_Y_Aend_B_median_dB"],
        color="#2166ac",
        marker="o",
        lw=1.2,
        ms=5,
        label="Y-Aend -> B",
    )
    axis.plot(
        x,
        ordered["gain_Y_C2_B_median_dB"],
        color="#b2182b",
        marker="s",
        lw=1.2,
        ms=5,
        label="Y-C2 -> B",
    )
    axis.axhline(0.0, color="#555555", lw=0.8, ls="--")
    axis.set_xticks(x, ordered["pa_nonlinearity_quartile"])
    axis.set_xlabel("PA nonlinearity quartile (Q1 best, Q4 worst)")
    axis.set_ylabel("Selected minus baseline gain (dB)")
    axis.set_title("B-segment generalisation gain across PA nonlinearity")
    axis.set_ylim(
        *_limits(
            ordered[["gain_Y_Aend_B_median_dB", "gain_Y_C2_B_median_dB"]].to_numpy(),
            references=(0.0,),
        )
    )
    axis.legend(loc="best", fontsize=7)
    details = _save_png(figure, output_path)
    details.update(
        {
            "quartile_count": int(ordered.shape[0]),
            "y_min": float(axis.get_ylim()[0]),
            "y_max": float(axis.get_ylim()[1]),
        }
    )
    return details


def plot_state_model_metrics(statewise: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    """Figure 3: four statewise model metrics, baseline versus selected."""

    _configure_matplotlib()
    columns = [
        ("baseline_Y_Aend_B_NMSE_dB", "Baseline Y-Aend -> B", "o", "-"),
        ("selected_Y_Aend_B_NMSE_dB", "Selected Y-Aend -> B", "o", "--"),
        ("baseline_Y_C2_B_NMSE_dB", "Baseline Y-C2 -> B", "s", "-"),
        ("selected_Y_C2_B_NMSE_dB", "Selected Y-C2 -> B", "s", "--"),
    ]
    missing = [column for column, *_ in columns if column not in statewise.columns]
    if missing:
        raise ValueError(f"statewise comparison缺少字段：{missing}")
    x = statewise["state_id"].to_numpy(dtype=float)
    values = np.concatenate([statewise[column].to_numpy(dtype=float) for column, *_ in columns])
    figure, axis = plt.subplots(figsize=(11.5, 4.9), constrained_layout=True)
    colors = ["#2166ac", "#2166ac", "#b2182b", "#b2182b"]
    for (column, label, marker, linestyle), color in zip(columns, colors, strict=True):
        axis.plot(
            x,
            statewise[column],
            color=color,
            marker=marker,
            linestyle=linestyle,
            lw=0.8,
            ms=2.5,
            markevery=15,
            alpha=0.85,
            label=label,
        )
    axis.set_xlabel("State ID")
    axis.set_ylabel("NMSE (dB)")
    axis.set_title("Baseline versus selected unified-model B generalisation")
    axis.set_xlim(float(x.min()), float(x.max()))
    axis.set_ylim(*_limits(values))
    axis.legend(loc="best", fontsize=7, ncol=2)
    details = _save_png(figure, output_path)
    details.update(
        {
            "curve_count": len(columns),
            "state_count": int(statewise.shape[0]),
            "y_min": float(np.min(values)),
            "y_max": float(np.max(values)),
        }
    )
    return details


def plot_retrieval_comparison(statewise: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    """Figure 4: finite Real-B retrieval curves and exact-hit strip."""

    _configure_matplotlib()
    required = {
        "state_id",
        "baseline_retrieved_real_B_CNMSE_dB",
        "selected_retrieved_real_B_CNMSE_dB",
        "baseline_exact_hit",
        "selected_exact_hit",
    }
    missing = sorted(required - set(statewise.columns))
    if missing:
        raise ValueError(f"statewise comparison缺少字段：{missing}")
    x = statewise["state_id"].to_numpy(dtype=float)
    baseline = statewise["baseline_retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float)
    selected = statewise["selected_retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float)
    finite = np.isfinite(np.concatenate([baseline, selected]))
    finite_values = np.concatenate(
        [baseline[np.isfinite(baseline)], selected[np.isfinite(selected)]]
    )
    y_min, y_max = _limits(finite_values, references=(DPD_SHAREABLE_THRESHOLD_DB,))
    figure, axis = plt.subplots(figsize=(11.5, 4.9), constrained_layout=True)
    axis.plot(
        x[np.isfinite(baseline)],
        baseline[np.isfinite(baseline)],
        color="#2166ac",
        lw=0.8,
        marker="o",
        ms=2.2,
        markevery=15,
        label="Baseline finite retrieval",
    )
    axis.plot(
        x[np.isfinite(selected)],
        selected[np.isfinite(selected)],
        color="#b2182b",
        lw=0.8,
        marker="s",
        ms=2.2,
        markevery=15,
        label="Selected finite retrieval",
    )
    axis.axhline(
        DPD_SHAREABLE_THRESHOLD_DB,
        color="#444444",
        lw=0.9,
        ls="--",
        label="Formal threshold: -40 dB",
    )
    finite_min = float(np.min(finite_values))
    baseline_exact = statewise["baseline_exact_hit"].to_numpy(dtype=bool)
    selected_exact = statewise["selected_exact_hit"].to_numpy(dtype=bool)
    if baseline_exact.any():
        axis.scatter(
            x[baseline_exact],
            np.full(int(baseline_exact.sum()), finite_min),
            color="#2166ac",
            marker="|",
            s=56,
            linewidths=1.0,
            label="Baseline exact (-Inf)",
        )
    if selected_exact.any():
        axis.scatter(
            x[selected_exact],
            np.full(int(selected_exact.sum()), finite_min),
            color="#b2182b",
            marker="_",
            s=48,
            linewidths=1.0,
            label="Selected exact (-Inf)",
        )
    axis.set_xlim(float(x.min()), float(x.max()))
    axis.set_ylim(y_min, y_max)
    axis.set_xlabel("State ID")
    axis.set_ylabel("Retrieved Real-B CNMSE (dB)")
    axis.set_title("Baseline versus selected C2 -> Aend retrieval")
    axis.legend(loc="best", fontsize=7, ncol=2)
    details = _save_png(figure, output_path)
    details.update(
        {
            "y_min": y_min,
            "y_max": y_max,
            "baseline_exact_count": int(baseline_exact.sum()),
            "selected_exact_count": int(selected_exact.sum()),
            "finite_value_count": int(finite.sum()),
        }
    )
    return details


def plot_transition_classes(statewise: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    """Figure 5: statewise shareability transition classes."""

    _configure_matplotlib()
    classes = [
        "failure_to_success",
        "failure_to_failure",
        "success_to_success",
        "success_to_failure",
    ]
    colors = {
        "failure_to_success": "#1b9e77",
        "failure_to_failure": "#d95f02",
        "success_to_success": "#2166ac",
        "success_to_failure": "#b2182b",
    }
    markers = {
        "failure_to_success": "o",
        "failure_to_failure": "x",
        "success_to_success": ".",
        "success_to_failure": "s",
    }
    if "transition_class" not in statewise.columns:
        raise ValueError("statewise comparison缺少transition_class")
    figure, axis = plt.subplots(figsize=(11.5, 3.9), constrained_layout=True)
    y_map = {value: index for index, value in enumerate(classes)}
    for value in classes:
        group = statewise.loc[statewise["transition_class"] == value]
        if group.empty:
            continue
        axis.scatter(
            group["state_id"],
            np.full(group.shape[0], y_map[value]),
            color=colors[value],
            marker=markers[value],
            s=20 if value != "success_to_success" else 12,
            linewidths=0.75,
            alpha=0.85,
            label=f"{value} (n={group.shape[0]})",
        )
    axis.set_xlim(float(statewise["state_id"].min()), float(statewise["state_id"].max()))
    axis.set_ylim(0.0, 3.0)
    axis.set_yticks(range(4), classes)
    axis.set_xlabel("State ID")
    axis.set_ylabel("Baseline -> selected")
    axis.set_title("DPD-shareability transition classes")
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=7)
    details = _save_png(figure, output_path)
    details.update(
        {
            "class_count": len(classes),
            "state_count": int(statewise.shape[0]),
            "transition_counts": {
                value: int(np.count_nonzero(statewise["transition_class"] == value))
                for value in classes
            },
        }
    )
    return details


def _plot_gain_correlation(
    statewise: pd.DataFrame,
    x_column: str,
    x_label: str,
    output_path: Path,
) -> dict[str, Any]:
    from scipy.stats import pearsonr, spearmanr

    mask = np.isfinite(statewise[x_column].to_numpy(dtype=float)) & np.isfinite(
        statewise["retrieval_gain_dB"].to_numpy(dtype=float)
    )
    x = statewise.loc[mask, x_column].to_numpy(dtype=float)
    y = statewise.loc[mask, "retrieval_gain_dB"].to_numpy(dtype=float)
    if x.size < 3:
        raise ValueError("相关性图需要至少3个finite/finite状态")
    correlation_defined = np.ptp(x) > 0.0 and np.ptp(y) > 0.0
    if correlation_defined:
        pearson = float(pearsonr(x, y).statistic)
        spearman = float(spearmanr(x, y).statistic)
    else:
        pearson = np.nan
        spearman = np.nan
    figure, axis = plt.subplots(figsize=(5.7, 4.7), constrained_layout=True)
    axis.scatter(x, y, s=22, color="#4d4d4d", alpha=0.72, edgecolors="white", linewidths=0.25)
    if np.ptp(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        axis.plot(
            np.array([x.min(), x.max()]),
            slope * np.array([x.min(), x.max()]) + intercept,
            color="#2166ac",
            lw=1.0,
            label="Least-squares trend",
        )
    axis.axhline(0.0, color="#777777", lw=0.7, ls="--")
    axis.axvline(0.0, color="#777777", lw=0.7, ls="--")
    axis.set_xlim(*_limits(x, references=(0.0,)))
    axis.set_ylim(*_limits(y, references=(0.0,)))
    axis.set_xlabel(x_label)
    axis.set_ylabel("Retrieval gain (dB)")
    axis.set_title("Model generalisation gain versus retrieval gain")
    correlation_text = (
        f"Pearson r={pearson:.3f}\nSpearman rho={spearman:.3f}"
        if correlation_defined
        else "Pearson r=undefined\nSpearman rho=undefined\n(constant gain)"
    )
    axis.text(
        0.03,
        0.97,
        f"n={x.size}\n{correlation_text}",
        transform=axis.transAxes,
        va="top",
        fontsize=7,
    )
    if np.ptp(x) > 0:
        axis.legend(loc="best", fontsize=7)
    details = _save_png(figure, output_path)
    details.update(
        {
            "finite_finite_count": int(x.size),
            "pearson_r": pearson,
            "spearman_rho": spearman,
            "correlation_defined": correlation_defined,
            "x_min": float(x.min()),
            "x_max": float(x.max()),
            "y_min": float(y.min()),
            "y_max": float(y.max()),
        }
    )
    return details


def plot_gain_correlations(statewise: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    """Figure 6a-c: independent correlation plots for all three gain definitions."""

    output_dir = Path(output_dir)
    return {
        "figure6a_C2_B": _plot_gain_correlation(
            statewise,
            "gain_Y_C2_B_dB",
            "Gain Y-C2 -> B (dB)",
            output_dir / "figure6a_C2_B_gain_vs_retrieval_gain.png",
        ),
        "figure6b_Aend_B": _plot_gain_correlation(
            statewise,
            "gain_Y_Aend_B_dB",
            "Gain Y-Aend -> B (dB)",
            output_dir / "figure6b_Aend_B_gain_vs_retrieval_gain.png",
        ),
        "figure6c_worst_side": _plot_gain_correlation(
            statewise,
            "gain_worst_side_B_dB",
            "Gain worst-side B (dB)",
            output_dir / "figure6c_worst_side_B_gain_vs_retrieval_gain.png",
        ),
    }


__all__ = [
    "plot_candidate_scan",
    "plot_gain_correlations",
    "plot_pa_quartile_gains",
    "plot_retrieval_comparison",
    "plot_state_model_metrics",
    "plot_transition_classes",
]
