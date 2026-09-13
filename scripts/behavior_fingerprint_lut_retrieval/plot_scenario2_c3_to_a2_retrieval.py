"""Publication-oriented matplotlib figures for the C3-to-A2 retrieval result."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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


def _save_png(fig: plt.Figure, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _actual_limits(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("图形需要至少一个有限值")
    return float(np.min(finite)), float(np.max(finite))


def plot_state_retrieval_mapping(
    retrieval_results: pd.DataFrame, output_path: Path
) -> dict[str, Any]:
    """Plot State_R -> State_Q, with exact and non-exact hits distinguished."""

    _configure_matplotlib()
    x = retrieval_results["State_n_R"].to_numpy(dtype=float)
    y = retrieval_results["State_n_Q"].to_numpy(dtype=float)
    exact = retrieval_results["exact_hit"].to_numpy(dtype=bool)
    y_min, y_max = _actual_limits(y)
    fig, ax = plt.subplots(figsize=(5.5, 4.0), constrained_layout=True)
    ax.scatter(
        x[~exact],
        y[~exact],
        s=12,
        c="#2166ac",
        marker="o",
        alpha=0.75,
        linewidths=0.25,
        edgecolors="white",
        label="Non-exact retrieval",
    )
    ax.scatter(
        x[exact],
        y[exact],
        s=15,
        c="#b2182b",
        marker="s",
        alpha=0.9,
        linewidths=0.25,
        edgecolors="white",
        label="Exact hit",
    )
    ax.plot([x.min(), x.max()], [x.min(), x.max()], color="#555555", lw=0.9, ls="--")
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"Real state $State_{n_R}$")
    ax.set_ylabel(r"Retrieved LUT state $State_{n_Q}$")
    ax.set_title("Y-C3 query to fixed Y-A2 LUT")
    ax.legend(loc="best", fontsize=7)
    ax.text(
        0.02,
        0.98,
        "Diagonal indicates exact state-index hit;\nindex offset is diagnostic only.",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=7,
        color="#333333",
    )
    _save_png(fig, output_path)
    return {
        "path": str(Path(output_path)),
        "y_min": y_min,
        "y_max": y_max,
        "exact_hit_count": int(np.count_nonzero(exact)),
        "non_exact_count": int(np.count_nonzero(~exact)),
        "reference_line": "y=x",
        "index_offset_is_not_behavioral_distance": True,
    }


def plot_retrieved_real_b_cnmse_by_state(
    retrieval_results: pd.DataFrame,
    real_b_diagnostics: pd.DataFrame,
    output_path: Path,
) -> dict[str, Any]:
    """Plot finite non-exact Real-B CNMSE and mark exact hits in a bottom strip."""

    _configure_matplotlib()
    merged = retrieval_results.loc[:, ["State_n_R", "exact_hit"]].merge(
        real_b_diagnostics.loc[:, ["State_n_R", "retrieved_real_B_CNMSE_dB"]],
        on="State_n_R",
        validate="one_to_one",
    )
    x = merged["State_n_R"].to_numpy(dtype=float)
    exact = merged["exact_hit"].to_numpy(dtype=bool)
    values = merged["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float)
    finite_nonexact = (~exact) & np.isfinite(values)
    y_min, y_max = _actual_limits(values[finite_nonexact])
    fig, ax = plt.subplots(figsize=(7.0, 3.6), constrained_layout=True)
    ax.plot(
        x[finite_nonexact],
        values[finite_nonexact],
        color="#2166ac",
        marker="o",
        ms=2.8,
        lw=0.65,
        alpha=0.85,
        label="Non-exact Real-B CNMSE",
    )
    exact_y = np.full(np.count_nonzero(exact), y_min, dtype=float)
    ax.scatter(
        x[exact],
        exact_y,
        color="#b2182b",
        marker="|",
        s=48,
        linewidths=1.0,
        label="Exact hit (stored as $-\\infty$)",
        zorder=3,
    )
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"Real state $State_{n_R}$")
    ax.set_ylabel(r"Retrieved Real-B CNMSE (dB)")
    ax.set_title("Behavioral distance of retrieved states")
    ax.legend(loc="best", fontsize=7)
    ax.text(
        0.99,
        0.03,
        "Exact hits are retained as -inf in the data;\nmarkers show their state IDs only.",
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=7,
        color="#333333",
    )
    _save_png(fig, output_path)
    return {
        "path": str(Path(output_path)),
        "y_min": y_min,
        "y_max": y_max,
        "finite_nonexact_count": int(np.count_nonzero(finite_nonexact)),
        "exact_hit_count": int(np.count_nonzero(exact)),
        "negative_infinity_preserved_in_data": True,
    }


def plot_retrieved_real_b_cnmse_distribution(
    real_b_diagnostics: pd.DataFrame, retrieval_results: pd.DataFrame, output_path: Path
) -> dict[str, Any]:
    """Plot the finite Non-exact Real-B CNMSE distribution and its boxplot."""

    _configure_matplotlib()
    exact = retrieval_results["exact_hit"].to_numpy(dtype=bool)
    values = real_b_diagnostics["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float)
    finite_nonexact = values[(~exact) & np.isfinite(values)]
    y_min, y_max = _actual_limits(finite_nonexact)
    fig, (hist_ax, box_ax) = plt.subplots(
        2,
        1,
        figsize=(5.8, 4.7),
        gridspec_kw={"height_ratios": [4, 1]},
        sharex=True,
        constrained_layout=True,
    )
    hist_ax.hist(
        finite_nonexact,
        bins=min(30, max(8, int(np.sqrt(finite_nonexact.size)))),
        color="#2166ac",
        alpha=0.82,
        edgecolor="white",
        linewidth=0.45,
    )
    hist_ax.axvline(
        float(np.median(finite_nonexact)),
        color="#b2182b",
        lw=1.0,
        ls="--",
        label=f"Median = {np.median(finite_nonexact):.2f} dB",
    )
    hist_ax.set_ylabel("Count")
    hist_ax.set_title("Non-exact retrieved Real-B CNMSE distribution")
    hist_ax.legend(loc="best", fontsize=7)
    box_ax.boxplot(
        finite_nonexact,
        vert=False,
        patch_artist=True,
        boxprops={"facecolor": "#92c5de", "edgecolor": "#2166ac"},
        medianprops={"color": "#b2182b", "linewidth": 1.0},
        whiskerprops={"color": "#2166ac"},
        capprops={"color": "#2166ac"},
        flierprops={"marker": ".", "markersize": 2.0, "markerfacecolor": "#2166ac"},
    )
    box_ax.set_xlabel("Retrieved Real-B CNMSE (dB)")
    box_ax.set_yticks([])
    box_ax.set_xlim(y_min, y_max)
    _save_png(fig, output_path)
    return {
        "path": str(Path(output_path)),
        "x_min": y_min,
        "x_max": y_max,
        "finite_nonexact_count": int(finite_nonexact.size),
        "histogram_and_boxplot": True,
    }


__all__ = [
    "plot_retrieved_real_b_cnmse_by_state",
    "plot_retrieved_real_b_cnmse_distribution",
    "plot_state_retrieval_mapping",
]
