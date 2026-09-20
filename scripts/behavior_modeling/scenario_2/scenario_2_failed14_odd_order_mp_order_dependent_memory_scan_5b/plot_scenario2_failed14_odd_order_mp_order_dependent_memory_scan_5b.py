"""Publication-oriented Python/matplotlib figures for the failed-14 scan.

Figure contract: the panels test whether compact odd-order, order-dependent
memory profiles improve the joint Aend/C2 modeling-to-B generalization tradeoff
for the 14 previously failed states.  The archetype is a quantitative grid with
candidate-level summaries, Pareto evidence, and state-by-candidate heatmaps.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 8
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["legend.frameon"] = False

COLORS = {
    "Aend_train": "#0F4D92",
    "Aend_B": "#D07A00",
    "C2_train": "#2E7D32",
    "C2_B": "#B64342",
}
MARKERS = {1: "o", 3: "s", 5: "^", 7: "D", 9: "P"}


def _save(fig: plt.Figure, stem: Path) -> dict[str, str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        "png": stem.with_suffix(".png"),
        "svg": stem.with_suffix(".svg"),
        "pdf": stem.with_suffix(".pdf"),
    }
    fig.savefig(outputs["png"], dpi=600, bbox_inches="tight")
    fig.savefig(outputs["svg"], bbox_inches="tight")
    fig.savefig(outputs["pdf"], bbox_inches="tight")
    plt.close(fig)
    return {key: str(value) for key, value in outputs.items()}


def _style_ax(ax: plt.Axes) -> None:
    ax.tick_params(direction="out", length=3, width=0.7)
    ax.grid(False)


def _figure1(summary: pd.DataFrame, output: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = summary["candidate_id"].to_numpy(dtype=float)
    for column, label in (
        ("Aend_train_median", "Aend train median"),
        ("Aend_B_median", "Aend → B median"),
        ("C2_train_median", "C2 train median"),
        ("C2_B_median", "C2 → B median"),
    ):
        ax.plot(
            x,
            summary[column],
            color=COLORS[column.removesuffix("_median")],
            marker="o",
            markersize=2.4,
            markevery=8,
            linewidth=0.9,
            label=label,
        )
    ax.axhline(-40.0, color="#666666", linestyle="--", linewidth=0.8, label="−40 dB")
    ax.set_xlabel("Candidate ID")
    ax.set_ylabel("NMSE (dB), candidate median")
    ax.set_title("125 compact odd-order MP structures")
    ax.legend(
        ncol=5,
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        borderaxespad=0.0,
    )
    _style_ax(ax)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.87])
    return _save(fig, output / "figure1_candidate_median_curves")


def _figure2(summary: pd.DataFrame, output: Path) -> dict[str, str]:
    fig, axes = plt.subplots(1, 4, figsize=(8.0, 2.7), sharey=True)
    metrics = (
        ("Aend_train_median", "Aend train"),
        ("Aend_B_median", "Aend → B"),
        ("C2_train_median", "C2 train"),
        ("C2_B_median", "C2 → B"),
    )
    p_values = sorted(summary["P"].unique())
    for ax, (column, title) in zip(axes, metrics, strict=True):
        values = [summary.loc[summary["P"] == p, column].to_numpy() for p in p_values]
        artists = ax.boxplot(
            values,
            positions=np.arange(len(p_values)),
            widths=0.65,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#272727", "linewidth": 0.9},
            whiskerprops={"color": "#555555", "linewidth": 0.7},
            capprops={"color": "#555555", "linewidth": 0.7},
            boxprops={"edgecolor": "#555555", "linewidth": 0.7},
        )
        for patch, p in zip(artists["boxes"], p_values, strict=True):
            patch.set_facecolor(plt.get_cmap("Blues")(0.35 + 0.1 * p_values.index(p)))
            patch.set_alpha(0.85)
        ax.axhline(-40.0, color="#888888", linestyle="--", linewidth=0.7)
        ax.set_xticks(np.arange(len(p_values)), [f"P={p}" for p in p_values])
        ax.set_title(title)
        _style_ax(ax)
    axes[0].set_ylabel("NMSE (dB), candidate median")
    fig.suptitle("Candidate-level distributions grouped by maximum odd order", y=1.03)
    fig.tight_layout()
    return _save(fig, output / "figure2_P_grouped_boxplots")


def _scatter_by_p(ax: plt.Axes, summary: pd.DataFrame, y_column: str, ylabel: str) -> None:
    for p in sorted(summary["P"].unique()):
        group = summary.loc[summary["P"] == p]
        ax.scatter(
            group["coefficient_count"],
            group[y_column],
            s=24,
            alpha=0.82,
            color=plt.get_cmap("viridis")((p - 1) / 8),
            marker=MARKERS[int(p)],
            label=f"P={p}",
            edgecolors="white",
            linewidths=0.3,
        )
    ax.set_xlabel("Coefficient count K")
    ax.set_ylabel(ylabel)
    ax.legend(ncol=3, fontsize=7)
    _style_ax(ax)


def _figure3_4(summary: pd.DataFrame, output: Path) -> dict[str, dict[str, str]]:
    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    _scatter_by_p(ax, summary, "Aend_B_median", "Aend → B median NMSE (dB)")
    ax.set_title("Complexity versus Aend generalization")
    first = _save(fig, output / "figure3_complexity_vs_Aend_generalization")

    fig, ax = plt.subplots(figsize=(5.0, 3.6))
    _scatter_by_p(ax, summary, "C2_B_median", "C2 → B median NMSE (dB)")
    ax.set_title("Complexity versus C2 generalization")
    second = _save(fig, output / "figure4_complexity_vs_C2_generalization")
    return {"figure3": first, "figure4": second}


def _figure5(summary: pd.DataFrame, output: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    non_pareto = summary.loc[~summary["pareto_optimal"].astype(bool)]
    pareto = summary.loc[summary["pareto_optimal"].astype(bool)]
    ax.scatter(
        non_pareto["worst_train_median"],
        non_pareto["worst_B_median"],
        color="#B8B8B8",
        s=22,
        label="Other candidates",
    )
    ax.scatter(
        pareto["worst_train_median"],
        pareto["worst_B_median"],
        color="#0F4D92",
        s=38,
        marker="D",
        label="Pareto frontier",
    )
    for row in pareto.itertuples():
        ax.annotate(
            f"{int(row.candidate_id)}",
            (row.worst_train_median, row.worst_B_median),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=6,
        )
    ax.set_xlabel("Worst train median NMSE (dB)")
    ax.set_ylabel("Worst B median NMSE (dB)")
    ax.set_title("Modeling–generalization Pareto candidates")
    ax.legend(fontsize=7)
    _style_ax(ax)
    fig.tight_layout()
    return _save(fig, output / "figure5_modeling_generalization_pareto")


def _figure6_7(summary: pd.DataFrame, output: Path) -> dict[str, dict[str, str]]:
    x = summary["candidate_id"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(
        x,
        summary["Aend_gap_median"],
        color=COLORS["Aend_B"],
        marker="o",
        markersize=2.2,
        markevery=8,
        linewidth=0.9,
    )
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Candidate ID")
    ax.set_ylabel("Aend B − train gap (dB)")
    ax.set_title("Aend generalization gap")
    _style_ax(ax)
    first = _save(fig, output / "figure6_Aend_generalization_gap")

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.plot(
        x,
        summary["C2_gap_median"],
        color=COLORS["C2_B"],
        marker="D",
        markersize=2.2,
        markevery=8,
        linewidth=0.9,
    )
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Candidate ID")
    ax.set_ylabel("C2 B − train gap (dB)")
    ax.set_title("C2 generalization gap")
    _style_ax(ax)
    second = _save(fig, output / "figure7_C2_generalization_gap")
    return {"figure6": first, "figure7": second}


def _heatmap(
    state_metrics: pd.DataFrame,
    metric: str,
    title: str,
    output: Path,
    filename: str,
) -> dict[str, str]:
    piv = state_metrics.pivot(index="state_id", columns="candidate_id", values=metric)
    piv = piv.sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(9.0, 3.8))
    image = ax.imshow(piv.to_numpy(dtype=float), aspect="auto", cmap="magma")
    x_ticks = np.arange(0, piv.shape[1], 10)
    ax.set_xticks(x_ticks, [str(int(piv.columns[i])) for i in x_ticks])
    ax.set_yticks(np.arange(piv.shape[0]), [str(int(value)) for value in piv.index])
    ax.set_xlabel("Candidate ID")
    ax.set_ylabel("Failed state ID")
    ax.set_title(title)
    cbar = fig.colorbar(image, ax=ax, shrink=0.85)
    cbar.set_label("NMSE (dB)")
    _style_ax(ax)
    fig.tight_layout()
    return _save(fig, output / filename)


def _figure10(selected: pd.DataFrame, output: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = selected["state_id"].to_numpy(dtype=float)
    for column, label, color, marker in (
        ("Aend_train_NMSE_dB", "Aend train", COLORS["Aend_train"], "o"),
        ("Aend_B_NMSE_dB", "Aend → B", COLORS["Aend_B"], "s"),
        ("C2_train_NMSE_dB", "C2 train", COLORS["C2_train"], "^"),
        ("C2_B_NMSE_dB", "C2 → B", COLORS["C2_B"], "D"),
    ):
        ax.plot(
            x,
            selected[column],
            color=color,
            marker=marker,
            linewidth=1.0,
            markersize=3.2,
            label=label,
        )
    ax.axhline(-40.0, color="#666666", linestyle="--", linewidth=0.8, label="−40 dB")
    ax.set_xlabel("Failed state ID")
    ax.set_ylabel("NMSE (dB)")
    ax.set_title("Selected balanced candidate across the 14 failed states")
    ax.legend(ncol=3, fontsize=7)
    _style_ax(ax)
    fig.tight_layout()
    return _save(fig, output / "figure10_selected_candidate_statewise")


def generate_figures(
    summary: pd.DataFrame,
    state_metrics: pd.DataFrame,
    selected: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    """Generate all ten requested figure groups using Python/matplotlib only."""

    details: dict[str, Any] = {"figure1": _figure1(summary, output_dir)}
    details["figure2"] = _figure2(summary, output_dir)
    details.update(_figure3_4(summary, output_dir))
    details["figure5"] = _figure5(summary, output_dir)
    details.update(_figure6_7(summary, output_dir))
    details["figure8"] = _heatmap(
        state_metrics,
        "C2_B_NMSE_dB",
        "C2 → B NMSE across failed states and candidates",
        output_dir,
        "figure8_C2_B_heatmap",
    )
    details["figure9"] = _heatmap(
        state_metrics,
        "Aend_B_NMSE_dB",
        "Aend → B NMSE across failed states and candidates",
        output_dir,
        "figure9_Aend_B_heatmap",
    )
    details["figure10"] = _figure10(selected, output_dir)
    return details


__all__ = ["generate_figures"]
