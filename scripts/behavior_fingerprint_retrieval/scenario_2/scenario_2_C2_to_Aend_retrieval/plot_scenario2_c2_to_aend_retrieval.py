"""Python/matplotlib figures for the formal C2-to-A_end retrieval study."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from behavior_fingerprint_retrieval.shared.scenario2_c2_to_aend_retrieval import (
    DPD_SHAREABLE_THRESHOLD_DB,
)


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


def _save_png(fig: plt.Figure, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=600, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _limits_with_reference(
    values: np.ndarray, reference: float | None = None
) -> tuple[float, float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("图形至少需要一个有限值")
    y_min = float(np.min(finite))
    y_max = float(np.max(finite))
    if reference is not None:
        y_min = min(y_min, float(reference))
        y_max = max(y_max, float(reference))
    return y_min, y_max


def plot_state_retrieval_mapping(
    retrieval_results: pd.DataFrame, real_b_diagnostics: pd.DataFrame, output_path: Path
) -> dict[str, Any]:
    """Plot State_R -> State_Q with exact/shareable/failure categories."""

    _configure_matplotlib()
    merged = retrieval_results.loc[
        :, ["State_n_R", "State_n_Q", "exact_hit"]
    ].merge(
        real_b_diagnostics.loc[:, ["State_n_R", "dpd_shareable"]],
        on="State_n_R",
        validate="one_to_one",
    )
    x = merged["State_n_R"].to_numpy(dtype=float)
    y = merged["State_n_Q"].to_numpy(dtype=float)
    exact = merged["exact_hit"].to_numpy(dtype=bool)
    shareable = merged["dpd_shareable"].to_numpy(dtype=bool)
    nonexact_shareable = (~exact) & shareable
    failure = ~shareable
    y_min, y_max = _limits_with_reference(y)
    fig, ax = plt.subplots(figsize=(5.7, 4.1), constrained_layout=True)
    if nonexact_shareable.any():
        ax.scatter(
            x[nonexact_shareable],
            y[nonexact_shareable],
            s=14,
            c="#1b9e77",
            marker="o",
            alpha=0.8,
            linewidths=0.25,
            edgecolors="white",
            label="Non-exact, DPD-shareable",
        )
    if failure.any():
        ax.scatter(
            x[failure],
            y[failure],
            s=22,
            c="#d95f02",
            marker="x",
            alpha=0.85,
            linewidths=0.75,
            label="Non-shareable failure",
        )
    if exact.any():
        ax.scatter(
            x[exact],
            y[exact],
            s=16,
            c="#2166ac",
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
    ax.set_title("Y-C2 query to state-specific Y-Aend LUT")
    ax.legend(loc="best", fontsize=6.8)
    ax.text(
        0.02,
        0.98,
        "Diagonal: exact state-index hit; index offset is diagnostic only.",
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
        "exact_hit_count": int(exact.sum()),
        "nonexact_shareable_count": int(nonexact_shareable.sum()),
        "failure_count": int(failure.sum()),
        "reference_line": "y=x",
    }


def plot_real_b_cnmse_by_state(
    retrieval_results: pd.DataFrame,
    real_b_diagnostics: pd.DataFrame,
    output_path: Path,
) -> dict[str, Any]:
    """Plot retrieved Real-B CNMSE, strict -40 dB line and exact-hit strip."""

    _configure_matplotlib()
    merged = retrieval_results.loc[:, ["State_n_R", "exact_hit"]].merge(
        real_b_diagnostics.loc[
            :, ["State_n_R", "retrieved_real_B_CNMSE_dB", "dpd_shareable"]
        ],
        on="State_n_R",
        validate="one_to_one",
    )
    x = merged["State_n_R"].to_numpy(dtype=float)
    values = merged["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float)
    exact = merged["exact_hit"].to_numpy(dtype=bool)
    shareable = merged["dpd_shareable"].to_numpy(dtype=bool)
    finite_nonexact = (~exact) & np.isfinite(values)
    y_min, y_max = _limits_with_reference(values[finite_nonexact], DPD_SHAREABLE_THRESHOLD_DB)
    fig, ax = plt.subplots(figsize=(7.0, 3.8), constrained_layout=True)
    if finite_nonexact.any():
        ax.plot(
            x[finite_nonexact],
            values[finite_nonexact],
            color="#4d4d4d",
            marker="o",
            ms=2.8,
            lw=0.55,
            alpha=0.65,
            label="Non-exact retrieval",
            zorder=1,
        )
        ax.scatter(
            x[finite_nonexact & shareable],
            values[finite_nonexact & shareable],
            s=17,
            color="#1b9e77",
            marker="o",
            linewidths=0.25,
            edgecolors="white",
            label="DPD-shareable (< -40 dB)",
            zorder=3,
        )
        ax.scatter(
            x[finite_nonexact & ~shareable],
            values[finite_nonexact & ~shareable],
            s=24,
            color="#d95f02",
            marker="x",
            linewidths=0.8,
            label="Failure (≥ -40 dB)",
            zorder=4,
        )
    ax.axhline(
        DPD_SHAREABLE_THRESHOLD_DB,
        color="#b2182b",
        lw=0.95,
        ls="--",
        label="Formal threshold: -40 dB",
        zorder=2,
    )
    exact_y = np.full(int(exact.sum()), y_min, dtype=float)
    ax.scatter(
        x[exact],
        exact_y,
        color="#2166ac",
        marker="|",
        s=48,
        linewidths=1.0,
        label="Exact hit (stored as $-\\infty$)",
        zorder=5,
    )
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel(r"Real state $State_{n_R}$")
    ax.set_ylabel(r"Retrieved Real-B CNMSE (dB)")
    ax.set_title("C2-to-Aend retrieved behavioral distance")
    ax.legend(loc="best", fontsize=6.5, ncol=2)
    ax.text(
        0.99,
        0.03,
        "Exact hits remain -inf in the data; bottom markers show their IDs.",
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
        "threshold_dB": DPD_SHAREABLE_THRESHOLD_DB,
        "finite_nonexact_count": int(finite_nonexact.sum()),
        "exact_hit_count": int(exact.sum()),
        "negative_infinity_preserved_in_data": True,
    }


def plot_real_b_cnmse_distribution(
    retrieval_results: pd.DataFrame,
    real_b_diagnostics: pd.DataFrame,
    output_path: Path,
) -> dict[str, Any]:
    """Plot the finite Non-exact Real-B CNMSE distribution."""

    _configure_matplotlib()
    exact = retrieval_results["exact_hit"].to_numpy(dtype=bool)
    values = real_b_diagnostics["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float)
    finite_nonexact = _finite_nonexact(values, exact)
    x_min, x_max = _limits_with_reference(finite_nonexact, DPD_SHAREABLE_THRESHOLD_DB)
    fig, (hist_ax, box_ax) = plt.subplots(
        2,
        1,
        figsize=(5.8, 4.8),
        gridspec_kw={"height_ratios": [4, 1]},
        sharex=True,
        constrained_layout=True,
    )
    bins = min(30, max(8, int(np.sqrt(finite_nonexact.size))))
    hist_ax.hist(
        finite_nonexact,
        bins=bins,
        color="#4d9221",
        alpha=0.82,
        edgecolor="white",
        linewidth=0.45,
    )
    hist_ax.axvline(
        float(np.median(finite_nonexact)),
        color="#2166ac",
        lw=1.0,
        ls="--",
        label=f"Median = {np.median(finite_nonexact):.2f} dB",
    )
    hist_ax.axvline(
        DPD_SHAREABLE_THRESHOLD_DB,
        color="#b2182b",
        lw=0.95,
        ls="-.",
        label="Formal threshold = -40 dB",
    )
    hist_ax.set_ylabel("Count")
    hist_ax.set_title("Non-exact retrieved Real-B CNMSE distribution")
    hist_ax.legend(loc="best", fontsize=7)
    box_ax.boxplot(
        finite_nonexact,
        vert=False,
        patch_artist=True,
        boxprops={"facecolor": "#b8e186", "edgecolor": "#4d9221"},
        medianprops={"color": "#2166ac", "linewidth": 1.0},
        whiskerprops={"color": "#4d9221"},
        capprops={"color": "#4d9221"},
        flierprops={"marker": ".", "markersize": 2.0, "markerfacecolor": "#4d9221"},
    )
    box_ax.axvline(DPD_SHAREABLE_THRESHOLD_DB, color="#b2182b", lw=0.95, ls="-.")
    box_ax.set_xlabel("Retrieved Real-B CNMSE (dB)")
    box_ax.set_yticks([])
    box_ax.set_xlim(x_min, x_max)
    _save_png(fig, output_path)
    return {
        "path": str(Path(output_path)),
        "x_min": x_min,
        "x_max": x_max,
        "threshold_dB": DPD_SHAREABLE_THRESHOLD_DB,
        "finite_nonexact_count": int(finite_nonexact.size),
    }


def _finite_nonexact(values: np.ndarray, exact: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    exact = np.asarray(exact, dtype=bool)
    return values[(~exact) & np.isfinite(values)]


def plot_dpd_shareable_success_by_state(
    retrieval_results: pd.DataFrame,
    real_b_diagnostics: pd.DataFrame,
    output_path: Path,
) -> dict[str, Any]:
    """Plot the three formal categories: failure, shareable, exact."""

    _configure_matplotlib()
    merged = retrieval_results.loc[:, ["State_n_R", "exact_hit"]].merge(
        real_b_diagnostics.loc[:, ["State_n_R", "dpd_shareable"]],
        on="State_n_R",
        validate="one_to_one",
    )
    x = merged["State_n_R"].to_numpy(dtype=float)
    exact = merged["exact_hit"].to_numpy(dtype=bool)
    shareable = merged["dpd_shareable"].to_numpy(dtype=bool)
    category = np.where(exact, 2, np.where(shareable, 1, 0))
    labels = {
        0: "Failure (CNMSE ≥ -40 dB)",
        1: "Non-exact, DPD-shareable",
        2: "Exact hit",
    }
    colors = {0: "#d95f02", 1: "#1b9e77", 2: "#2166ac"}
    markers = {0: "x", 1: "o", 2: "s"}
    fig, ax = plt.subplots(figsize=(7.0, 2.7), constrained_layout=True)
    for value in (0, 1, 2):
        mask = category == value
        if mask.any():
            scatter_kwargs = {
                "color": colors[value],
                "marker": markers[value],
                "s": 18 if value != 0 else 24,
                "linewidths": 0.75 if value == 0 else 0.25,
                "alpha": 0.85,
                "label": labels[value],
            }
            if value != 0:
                scatter_kwargs["edgecolors"] = "white"
            ax.scatter(x[mask], category[mask], **scatter_kwargs)
    ax.set_xlim(float(np.min(x)), float(np.max(x)))
    ax.set_ylim(0, 2)
    ax.set_yticks([0, 1, 2], ["Failure", "Non-exact\nshareable", "Exact"])
    ax.set_xlabel(r"Real state $State_{n_R}$")
    ax.set_ylabel("Retrieval class")
    ax.set_title("Formal DPD-shareable retrieval class by state")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.26), ncol=3, fontsize=6.5)
    _save_png(fig, output_path)
    return {
        "path": str(Path(output_path)),
        "category_y_values": {str(key): int((category == key).sum()) for key in (0, 1, 2)},
        "threshold_dB": DPD_SHAREABLE_THRESHOLD_DB,
    }


__all__ = [
    "plot_dpd_shareable_success_by_state",
    "plot_real_b_cnmse_by_state",
    "plot_real_b_cnmse_distribution",
    "plot_state_retrieval_mapping",
]
