"""Python/matplotlib figures for the P5 order-2 OLS ablation.

Figure contract: test whether adding the second-order basis term to the
odd-only P5 family changes segment fitting, common-B generalization and
conditioning for the 14 failed states.  The figure archetype is a quantitative
grid with candidate summaries, an old/new ablation panel, and statewise curves.
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


def _style(ax: plt.Axes) -> None:
    ax.tick_params(direction="out", length=3, width=0.7)
    ax.grid(False)


def _figure1(summary: pd.DataFrame, output: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = summary["candidate_id"].to_numpy(dtype=float)
    for column, label, color, marker in (
        ("Aend_train_median", "Aend train", COLORS["Aend_train"], "o"),
        ("Aend_B_median", "Aend → B", COLORS["Aend_B"], "s"),
        ("C2_train_median", "C2 train", COLORS["C2_train"], "^"),
        ("C2_B_median", "C2 → B", COLORS["C2_B"], "D"),
    ):
        ax.plot(
            x,
            summary[column],
            color=color,
            marker=marker,
            markersize=2.6,
            markevery=3,
            linewidth=0.9,
            label=label,
        )
    ax.axhline(-40.0, color="#666666", linestyle="--", linewidth=0.8, label="−40 dB")
    ax.set_xlabel("Candidate ID")
    ax.set_ylabel("NMSE (dB), candidate median")
    ax.set_title("P5 + order-2 OLS: 35 memory profiles")
    ax.legend(
        ncol=3,
        fontsize=7,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        borderaxespad=0.0,
    )
    _style(ax)
    fig.tight_layout(rect=[0.0, 0.10, 1.0, 1.0])
    return _save(fig, output / "figure1_candidate_median_curves")


def _profile_bar(
    summary: pd.DataFrame, output: Path, metric: str, title: str, filename: str
) -> dict[str, str]:
    group = summary.sort_values(["candidate_id"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    colors = [
        plt.get_cmap("viridis")(float(p) / max(1, group["max_delay"].max()))
        for p in group["max_delay"]
    ]
    ax.bar(
        np.arange(group.shape[0]), group[metric], color=colors, edgecolor="white", linewidth=0.25
    )
    ax.axhline(-40.0, color="#666666", linestyle="--", linewidth=0.8)
    ax.set_xticks(
        np.arange(group.shape[0]), group["memory_profile_string"], rotation=65, ha="right"
    )
    ax.set_xlabel("Memory profile")
    ax.set_ylabel("NMSE (dB), candidate median")
    ax.set_title(title)
    _style(ax)
    fig.tight_layout()
    return _save(fig, output / filename)


def _figure4(summary: pd.DataFrame, output: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(5.2, 4.0))
    scatter = ax.scatter(
        summary["worst_train_median"],
        summary["worst_B_median"],
        c=summary["coefficient_count"],
        cmap="viridis",
        s=28,
        edgecolors="white",
        linewidths=0.3,
    )
    fig.colorbar(scatter, ax=ax, label="K")
    ax.set_xlabel("Worst train median NMSE (dB)")
    ax.set_ylabel("Worst B median NMSE (dB)")
    ax.set_title("Train–generalization tradeoff")
    _style(ax)
    fig.tight_layout()
    return _save(fig, output / "figure4_train_generalization_tradeoff")


def _figure5(summary: pd.DataFrame, output: Path) -> dict[str, str]:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), sharex=True)
    x = summary["candidate_id"].to_numpy(dtype=float)
    axes[0].plot(
        x,
        summary["Aend_gap_median"],
        color=COLORS["Aend_B"],
        marker="o",
        markersize=2.4,
        markevery=3,
        linewidth=0.9,
    )
    axes[1].plot(
        x,
        summary["C2_gap_median"],
        color=COLORS["C2_B"],
        marker="D",
        markersize=2.4,
        markevery=3,
        linewidth=0.9,
    )
    axes[0].set_title("Aend gap")
    axes[1].set_title("C2 gap")
    for ax in axes:
        ax.axhline(0.0, color="#666666", linestyle="--", linewidth=0.8)
        ax.set_xlabel("Candidate ID")
        _style(ax)
    axes[0].set_ylabel("B − train NMSE (dB)")
    fig.suptitle("Generalization gap")
    fig.tight_layout()
    return _save(fig, output / "figure5_generalization_gaps")


def _figure6(comparison: pd.DataFrame, output: Path) -> dict[str, str]:
    metrics = [
        ("Aend_train_NMSE_dB", "Aend train"),
        ("Aend_B_NMSE_dB", "Aend → B"),
        ("C2_train_NMSE_dB", "C2 train"),
        ("C2_B_NMSE_dB", "C2 → B"),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    labels = comparison["model_label"].tolist()
    x = np.arange(len(metrics))
    width = 0.25
    for index, row in comparison.iterrows():
        ax.bar(
            x + (index - 1) * width,
            [row[m[0]] for m in metrics],
            width,
            label=labels[index],
            color=("#767676", "#0F4D92", "#D07A00")[index],
        )
    ax.axhline(-40.0, color="#666666", linestyle="--", linewidth=0.8)
    ax.set_xticks(x, [title for _, title in metrics])
    ax.set_ylabel("Median NMSE (dB)")
    ax.set_title("Order-2 ablation and historical frozen reference")
    ax.legend(
        fontsize=7,
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        borderaxespad=0.0,
    )
    _style(ax)
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    return _save(fig, output / "figure6_order2_ablation_comparison")


def _figure7(old_state: pd.DataFrame, new_state: pd.DataFrame, output: Path) -> dict[str, str]:
    merged = new_state.merge(old_state, on="state_id", suffixes=("_new", "_old"))
    metrics = [
        ("Aend_train_NMSE_dB", "Aend train"),
        ("Aend_B_NMSE_dB", "Aend → B"),
        ("C2_train_NMSE_dB", "C2 train"),
        ("C2_B_NMSE_dB", "C2 → B"),
    ]
    fig, ax = plt.subplots(figsize=(7.0, 3.7))
    x = np.arange(merged.shape[0])
    for column, label, color, marker in (
        (metrics[0][0], metrics[0][1], COLORS["Aend_train"], "o"),
        (metrics[1][0], metrics[1][1], COLORS["Aend_B"], "s"),
        (metrics[2][0], metrics[2][1], COLORS["C2_train"], "^"),
        (metrics[3][0], metrics[3][1], COLORS["C2_B"], "D"),
    ):
        delta = merged[f"{column}_new"] - merged[f"{column}_old"]
        ax.plot(x, delta, color=color, marker=marker, markersize=3.0, linewidth=0.9, label=label)
    ax.axhline(0.0, color="#666666", linestyle="--", linewidth=0.8)
    ax.set_xticks(x, merged["state_id"].astype(str))
    ax.set_xlabel("Failed state ID")
    ax.set_ylabel("New − odd-only OLS (dB)")
    ax.set_title("Adding p=2: statewise change")
    ax.legend(ncol=2, fontsize=7)
    _style(ax)
    fig.tight_layout()
    return _save(fig, output / "figure7_statewise_order2_change")


def _figure8(selected: pd.DataFrame, output: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
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
    ax.set_title("Selected P5 + order-2 OLS candidate")
    ax.legend(ncol=3, fontsize=7)
    _style(ax)
    fig.tight_layout()
    return _save(fig, output / "figure8_selected_candidate_statewise")


def _figure9(summary: pd.DataFrame, output: Path) -> dict[str, str]:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    colors = summary["coefficient_count"].to_numpy(dtype=float)
    axes[0].scatter(
        summary["coefficient_count"],
        summary["condition_number_Aend_median"],
        c=colors,
        cmap="viridis",
        s=24,
    )
    axes[1].scatter(
        summary["coefficient_count"],
        summary["Aend_theta_norm_median"],
        c=colors,
        cmap="viridis",
        s=24,
    )
    axes[0].set_ylabel("Aend condition number (median)")
    axes[1].set_ylabel("Aend theta L2 norm (median)")
    for ax in axes:
        ax.set_xlabel("K")
        _style(ax)
    fig.suptitle("Conditioning and coefficient norm")
    fig.tight_layout()
    return _save(fig, output / "figure9_condition_number_theta_norm")


def generate_figures(
    summary: pd.DataFrame,
    state_metrics: pd.DataFrame,
    selected: pd.DataFrame,
    comparison: pd.DataFrame,
    old_state: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    details: dict[str, Any] = {"figure1": _figure1(summary, output_dir)}
    details["figure2"] = _profile_bar(
        summary,
        output_dir,
        "Aend_train_median",
        "Memory profile versus Aend training",
        "figure2_memory_profile_train",
    )
    details["figure3"] = _profile_bar(
        summary,
        output_dir,
        "Aend_B_median",
        "Memory profile versus Aend B generalization",
        "figure3_memory_profile_B_generalization",
    )
    details["figure4"] = _figure4(summary, output_dir)
    details["figure5"] = _figure5(summary, output_dir)
    details["figure6"] = _figure6(comparison, output_dir)
    details["figure7"] = _figure7(old_state, selected, output_dir)
    details["figure8"] = _figure8(selected, output_dir)
    details["figure9"] = _figure9(summary, output_dir)
    return details


__all__ = ["generate_figures"]
