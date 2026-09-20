"""Python/matplotlib figures for the frozen-neighborhood Ridge experiment.

The figures are deliberately generated from the saved tables rather than
recomputing any model quantities.  Every figure is exported as editable-text
SVG, PDF and a high-resolution PNG so the result can be inspected and placed
in a manuscript without changing the numerical scan.
"""

# ruff: noqa: E501

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
    "frozen": "#555555",
    "selected": "#5E3C99",
    "failure14": "#C44E52",
    "success411": "#4C72B0",
}
CORE = ("Aend_train_NMSE_dB", "Aend_B_NMSE_dB", "C2_train_NMSE_dB", "C2_B_NMSE_dB")
METRIC_COLORS = {
    "Aend_train_NMSE_dB": COLORS["Aend_train"],
    "Aend_B_NMSE_dB": COLORS["Aend_B"],
    "C2_train_NMSE_dB": COLORS["C2_train"],
    "C2_B_NMSE_dB": COLORS["C2_B"],
}
LABELS = {
    "Aend_train_NMSE_dB": "Aend train",
    "Aend_B_NMSE_dB": "Aend → B",
    "C2_train_NMSE_dB": "C2 train",
    "C2_B_NMSE_dB": "C2 → B",
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


def _lambda_x(value: float) -> float:
    return -13.5 if float(value) == 0.0 else float(np.log10(value))


def _lambda_label(value: float) -> str:
    return "OLS" if float(value) == 0.0 else f"{float(value):.0e}"


def _initial(summary: pd.DataFrame, initial_candidate_count: int) -> pd.DataFrame:
    return summary.loc[summary["candidate_id"].astype(int) < int(initial_candidate_count)].copy()


def _heatmap(
    summary: pd.DataFrame,
    value_column: str,
    title: str,
    output_dir: Path,
    stem: str,
    initial_candidate_count: int,
) -> dict[str, str]:
    frame = _initial(summary, initial_candidate_count)
    structures = list(frame["structure_id"].drop_duplicates())
    lambdas = sorted(frame["lambda"].drop_duplicates().astype(float))
    matrix = (
        frame.pivot(index="structure_id", columns="lambda", values=value_column)
        .reindex(index=structures, columns=lambdas)
        .to_numpy(dtype=float)
    )
    fig, ax = plt.subplots(figsize=(7.3, 3.8))
    image = ax.imshow(matrix, aspect="auto", cmap="magma")
    ax.set_xticks(np.arange(len(lambdas)), [_lambda_label(v) for v in lambdas], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(structures)), structures)
    ax.set_xlabel("Ridge lambda")
    ax.set_ylabel("Memory structure")
    ax.set_title(title)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            if np.isfinite(matrix[row, col]):
                ax.text(col, row, f"{matrix[row, col]:.1f}", ha="center", va="center", fontsize=5, color="white")
    cbar = fig.colorbar(image, ax=ax, pad=0.02)
    cbar.set_label("NMSE (dB), 14-state median")
    _style(ax)
    fig.tight_layout()
    return _save(fig, output_dir / stem)


def _figure1(summary: pd.DataFrame, output_dir: Path, initial_candidate_count: int) -> dict[str, str]:
    return _heatmap(
        summary,
        "worst4_median",
        "Local memory structure × Ridge: worst-four median",
        output_dir,
        "figure1_worst4_median_heatmap",
        initial_candidate_count,
    )


def _figure2(summary: pd.DataFrame, output_dir: Path, initial_candidate_count: int) -> dict[str, str]:
    return _heatmap(
        summary,
        "Bworst_median",
        "Local memory structure × Ridge: B-balanced median",
        output_dir,
        "figure2_Bworst_median_heatmap",
        initial_candidate_count,
    )


def _figure3(summary: pd.DataFrame, output_dir: Path, initial_candidate_count: int) -> dict[str, str]:
    frame = _initial(summary, initial_candidate_count)
    structures = list(frame["structure_id"].drop_duplicates())
    fig, axes = plt.subplots(4, 2, figsize=(8.0, 9.0), sharex=True, sharey=True)
    axes = axes.ravel()
    specs = tuple((f"{column.removesuffix('_NMSE_dB')}_median", LABELS[column], METRIC_COLORS[column]) for column in CORE)
    for ax, structure in zip(axes, structures, strict=False):
        group = frame.loc[frame["structure_id"] == structure].sort_values("lambda").copy()
        group["lambda_x"] = group["lambda"].map(_lambda_x)
        for column, label, color in specs:
            ax.plot(group["lambda_x"], group[column], marker="o", markersize=2.4, linewidth=0.85, color=color, label=label)
        ax.axhline(-40.0, color="#777777", linestyle="--", linewidth=0.7)
        ax.set_title(structure)
        _style(ax)
    for ax in axes[len(structures) :]:
        ax.set_visible(False)
    for ax in axes[-2:]:
        if ax.get_visible():
            ax.set_xticks([-13.5, -10, -8, -6, -4, -3], ["OLS", "1e−10", "1e−8", "1e−6", "1e−4", "1e−3"], rotation=35, ha="right")
            ax.set_xlabel("Ridge lambda")
    axes[0].set_ylabel("NMSE (dB), median")
    axes[2].set_ylabel("NMSE (dB), median")
    axes[-2].legend(ncol=2, fontsize=6, loc="best")
    fig.suptitle("Four metrics versus Ridge strength for each local structure", y=0.995)
    fig.tight_layout()
    return _save(fig, output_dir / "figure3_metrics_vs_lambda")


def _figure4(summary: pd.DataFrame, output_dir: Path, initial_candidate_count: int) -> dict[str, str]:
    frame = _initial(summary, initial_candidate_count).copy()
    frame["lambda_x"] = frame["lambda"].map(_lambda_x)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5), sharex=True)
    for ax, column, title in (
        (axes[0], "Aend_theta_norm_median", "Aend coefficient norm"),
        (axes[1], "C2_theta_norm_median", "C2 coefficient norm"),
    ):
        for structure, group in frame.groupby("structure_id", sort=False):
            ax.plot(group["lambda_x"], group[column], marker="o", markersize=2.2, linewidth=0.8, label=structure)
        ax.set_title(title)
        ax.set_xlabel("Ridge lambda")
        _style(ax)
    axes[0].set_ylabel("Median coefficient L2 norm")
    axes[1].legend(ncol=2, fontsize=6, loc="best")
    axes[0].set_xticks([-13.5, -10, -8, -6, -4, -3], ["OLS", "1e−10", "1e−8", "1e−6", "1e−4", "1e−3"], rotation=35, ha="right")
    fig.suptitle("Ridge coefficient shrinkage", y=1.02)
    fig.tight_layout()
    return _save(fig, output_dir / "figure4_theta_norm_vs_lambda")


def _figure5(summary: pd.DataFrame, output_dir: Path, initial_candidate_count: int) -> dict[str, str]:
    frame = _initial(summary, initial_candidate_count).copy()
    frame["lambda_x"] = frame["lambda"].map(_lambda_x)
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.5), sharex=True)
    for ax, column, title in (
        (axes[0], "Aend_gap_median", "Aend generalization gap"),
        (axes[1], "C2_gap_median", "C2 generalization gap"),
    ):
        for structure, group in frame.groupby("structure_id", sort=False):
            ax.plot(group["lambda_x"], group[column], marker="o", markersize=2.2, linewidth=0.8, label=structure)
        ax.axhline(0.0, color="#666666", linestyle="--", linewidth=0.7)
        ax.set_title(title)
        ax.set_xlabel("Ridge lambda")
        _style(ax)
    axes[0].set_ylabel("B NMSE − train NMSE (dB), median")
    axes[1].legend(ncol=2, fontsize=6, loc="best")
    axes[0].set_xticks([-13.5, -10, -8, -6, -4, -3], ["OLS", "1e−10", "1e−8", "1e−6", "1e−4", "1e−3"], rotation=35, ha="right")
    fig.suptitle("Generalization gap versus Ridge strength", y=1.02)
    fig.tight_layout()
    return _save(fig, output_dir / "figure5_generalization_gap_vs_lambda")


def _figure6(selected_summary: pd.DataFrame, frozen_summary: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    metrics = list(CORE)
    labels = [LABELS[value] for value in metrics]
    median_columns = [f"{value.removesuffix('_NMSE_dB')}_median" for value in metrics]
    selected_values = [float(selected_summary[column].iloc[0]) for column in median_columns]
    frozen_values = [float(frozen_summary[column].iloc[0]) for column in median_columns]
    x = np.arange(len(metrics))
    width = 0.36
    fig, ax = plt.subplots(figsize=(7.3, 3.9))
    ax.bar(x - width / 2, frozen_values, width, color=COLORS["frozen"], label="Frozen S0 + 1e−8")
    ax.bar(x + width / 2, selected_values, width, color=COLORS["selected"], label="Selected candidate")
    ax.axhline(-40.0, color="#777777", linestyle="--", linewidth=0.8)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("NMSE (dB), failure14 median")
    ax.set_title("Selected candidate versus frozen baseline")
    ax.legend(fontsize=7)
    _style(ax)
    fig.tight_layout()
    return _save(fig, output_dir / "figure6_selected_vs_frozen_failure14")


def _figure7(failure_comparison: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    frame = failure_comparison.sort_values("state_id")
    fig, ax = plt.subplots(figsize=(8.4, 3.9))
    ax.bar(frame["state_id"].astype(str), frame["worst4_delta_selected_minus_frozen"], color=COLORS["selected"], width=0.75)
    ax.axhline(0.0, color="#555555", linewidth=0.8)
    ax.set_xlabel("State ID")
    ax.set_ylabel("Selected − frozen worst-four (dB)")
    ax.set_title("Failure14 statewise change (negative is an improvement)")
    ax.tick_params(axis="x", rotation=45)
    _style(ax)
    fig.tight_layout()
    return _save(fig, output_dir / "figure7_failure14_delta_worst4")


def _figure8(summary: pd.DataFrame, output_dir: Path, initial_candidate_count: int) -> dict[str, str]:
    frame = _initial(summary, initial_candidate_count)
    fig, ax = plt.subplots(figsize=(5.4, 4.3))
    scatter = ax.scatter(frame["worst_train_median"], frame["worst_B_median"], c=frame["coefficient_count"], cmap="viridis", s=28, edgecolors="white", linewidths=0.3)
    frozen = frame.loc[frame["is_frozen_baseline"].astype(bool)]
    ax.scatter(frozen["worst_train_median"], frozen["worst_B_median"], color=COLORS["frozen"], marker="x", s=42, label="Frozen baseline")
    ax.set_xlabel("Worst train median NMSE (dB)")
    ax.set_ylabel("Worst B median NMSE (dB)")
    ax.set_title("Train–B Pareto view of the 63 initial candidates")
    fig.colorbar(scatter, ax=ax, label="Coefficient count K")
    ax.legend(fontsize=7)
    _style(ax)
    fig.tight_layout()
    return _save(fig, output_dir / "figure8_train_B_pareto")


def _figure9_10(all425: pd.DataFrame, output_dir: Path, column_prefix: str, title: str, stem: str) -> dict[str, str]:
    selected = f"{column_prefix}_B_NMSE_dB"
    frozen = f"frozen_{column_prefix.removeprefix('selected_')}_B_NMSE_dB"
    frame = all425.sort_values("state_id")
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    ax.plot(frame["state_id"], frame[frozen], color=COLORS["frozen"], linewidth=0.75, label="Frozen")
    ax.plot(frame["state_id"], frame[selected], color=COLORS["selected"], linewidth=0.75, label="Selected")
    ax.axhline(-40.0, color="#777777", linestyle="--", linewidth=0.7)
    ax.set_xlabel("State ID")
    ax.set_ylabel("NMSE (dB)")
    ax.set_title(title)
    ax.legend(fontsize=7)
    _style(ax)
    fig.tight_layout()
    return _save(fig, output_dir / stem)


def _figure11(all425: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    frame = all425.sort_values("state_id")
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    colors = np.where(frame["split"].eq("failure14"), COLORS["failure14"], COLORS["success411"])
    ax.bar(frame["state_id"], frame["worst4_delta_selected_minus_frozen"], color=colors, width=0.9)
    ax.axhline(0.0, color="#555555", linewidth=0.8)
    ax.set_xlabel("State ID")
    ax.set_ylabel("Selected − frozen worst-four (dB)")
    ax.set_title("All-425 statewise change")
    ax.plot([], [], color=COLORS["failure14"], linewidth=5, label="Failure14")
    ax.plot([], [], color=COLORS["success411"], linewidth=5, label="Success411")
    ax.legend(fontsize=7, ncol=2)
    _style(ax)
    fig.tight_layout()
    return _save(fig, output_dir / "figure11_all425_delta_worst4")


def _figure12(all425: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    groups = ["failure14", "success411", "all425"]
    values = [all425.loc[all425["split"] == group, "worst4_delta_selected_minus_frozen"].to_numpy(dtype=float) for group in groups[:2]]
    values.append(all425["worst4_delta_selected_minus_frozen"].to_numpy(dtype=float))
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    box = ax.boxplot(values, tick_labels=["Failure14", "Success411", "All425"], patch_artist=True, widths=0.6, showmeans=True)
    for patch, color in zip(box["boxes"], (COLORS["failure14"], COLORS["success411"], COLORS["selected"]), strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    ax.axhline(0.0, color="#555555", linewidth=0.8)
    ax.set_ylabel("Selected − frozen worst-four (dB)")
    ax.set_title("Improvement distribution after frozen-neighborhood selection")
    _style(ax)
    fig.tight_layout()
    return _save(fig, output_dir / "figure12_improvement_distribution")


def generate_figures(
    summary: pd.DataFrame,
    selected_summary: pd.DataFrame,
    frozen_summary: pd.DataFrame,
    failure_comparison: pd.DataFrame,
    all425: pd.DataFrame,
    output_dir: Path,
    *,
    initial_candidate_count: int = 63,
) -> dict[str, Any]:
    """Generate the twelve requested figures from persisted result tables."""

    details: dict[str, Any] = {}
    details["figure1"] = _figure1(summary, output_dir, initial_candidate_count)
    details["figure2"] = _figure2(summary, output_dir, initial_candidate_count)
    details["figure3"] = _figure3(summary, output_dir, initial_candidate_count)
    details["figure4"] = _figure4(summary, output_dir, initial_candidate_count)
    details["figure5"] = _figure5(summary, output_dir, initial_candidate_count)
    details["figure6"] = _figure6(selected_summary, frozen_summary, output_dir)
    details["figure7"] = _figure7(failure_comparison, output_dir)
    details["figure8"] = _figure8(summary, output_dir, initial_candidate_count)
    details["figure9"] = _figure9_10(all425, output_dir, "selected_Aend", "All-425 Aend → B: frozen versus selected", "figure9_all425_Aend_B")
    details["figure10"] = _figure9_10(all425, output_dir, "selected_C2", "All-425 C2 → B: frozen versus selected", "figure10_all425_C2_B")
    details["figure11"] = _figure11(all425, output_dir)
    details["figure12"] = _figure12(all425, output_dir)
    return details


__all__ = ["generate_figures"]
