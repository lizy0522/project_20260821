"""Publication-oriented Python figures for the Sparse-GMP experiment."""

# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)

COLORS = {"A": "#0F4D92", "C": "#2E7D32", "B": "#D07A00", "balanced": "#7B3294", "frozen": "#555555", "selected": "#5E3C99", "failure": "#C44E52", "success": "#4C72B0"}
METRICS = ("Aend_train", "Aend_B", "C2_train", "C2_B")
LABELS = {"Aend_train": "Aend train", "Aend_B": "Aend → B", "C2_train": "C2 train", "C2_B": "C2 → B"}


def _save(fig: plt.Figure, stem: Path) -> dict[str, str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = {"png": stem.with_suffix(".png"), "svg": stem.with_suffix(".svg"), "pdf": stem.with_suffix(".pdf")}
    fig.savefig(outputs["png"], dpi=600, bbox_inches="tight")
    fig.savefig(outputs["svg"], bbox_inches="tight")
    fig.savefig(outputs["pdf"], bbox_inches="tight")
    plt.close(fig)
    return {key: str(value) for key, value in outputs.items()}


def _style(ax: plt.Axes) -> None:
    ax.tick_params(direction="out", length=3, width=0.7)
    ax.grid(False)


def _figure1(history: pd.DataFrame, output: Path) -> dict[str, str]:
    frame = history.sort_values("step")
    fig, ax = plt.subplots(figsize=(8.0, 3.4))
    x = frame["step"].to_numpy(float)
    ax.plot(x, np.zeros_like(x), color="#BBBBBB", linewidth=0.8)
    for _, row in frame.iterrows():
        label = f"{row['selected_term_id']}\n(p={int(row['p'])}, m={int(row['m'])}, q={int(row['q'])})"
        ax.scatter([row["step"]], [0], s=42, color=COLORS["selected"], zorder=3)
        ax.text(row["step"], 0.03 if int(row["step"]) % 2 else -0.06, label, ha="center", va="bottom" if int(row["step"]) % 2 else "top", fontsize=7)
    ax.set_xlim(0.5, 8.5)
    ax.set_ylim(-0.13, 0.13)
    ax.set_yticks([])
    ax.set_xticks(np.arange(1, 9), [f"G{i}" for i in range(1, 9)])
    ax.set_xlabel("Forward-selected model step")
    ax.set_title("Forward-selected Sparse-GMP term sequence")
    _style(ax)
    fig.tight_layout()
    return _save(fig, output / "figure1_forward_selected_term_sequence")


def _figure2(sequence: pd.DataFrame, selected_count: int, output: Path) -> dict[str, str]:
    frame = sequence.sort_values("gmp_term_count")
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(frame["gmp_term_count"], frame["Aend_CV_median"], marker="o", color=COLORS["A"], label="Aend CV median")
    ax.plot(frame["gmp_term_count"], frame["C2_CV_median"], marker="s", color=COLORS["C"], label="C2 CV median")
    ax.plot(frame["gmp_term_count"], frame["balanced_CV_median"], marker="D", color=COLORS["balanced"], label="Balanced CV median")
    ax.axvline(selected_count, color=COLORS["selected"], linestyle="--", linewidth=0.8, label=f"selected k={selected_count}")
    ax.set_xlabel("Number of GMP terms")
    ax.set_ylabel("Blocked-CV NMSE (dB), median")
    ax.set_title("A/C internal blocked validation along G0–G8")
    ax.legend(fontsize=7)
    _style(ax)
    fig.tight_layout()
    return _save(fig, output / "figure2_internal_cv_vs_term_count")


def _figure3(b_curve: pd.DataFrame, selected_count: int, output: Path) -> dict[str, str]:
    grouped = b_curve.groupby("gmp_term_count", sort=True).agg({"Aend_train_NMSE_dB": "median", "Aend_B_NMSE_dB": "median", "C2_train_NMSE_dB": "median", "C2_B_NMSE_dB": "median", "Aend_CV_median": "median", "C2_CV_median": "median"}).reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(7.6, 3.6), sharey=True)
    for ax, prefix, title in ((axes[0], "Aend", "Aend"), (axes[1], "C2", "C2")):
        ax.plot(grouped["gmp_term_count"], grouped[f"{prefix}_train_NMSE_dB"], marker="o", color=COLORS["A" if prefix == "Aend" else "C"], label="Train")
        ax.plot(grouped["gmp_term_count"], grouped[f"{prefix}_CV_median"], marker="^", linestyle="--", color=COLORS["balanced"], label="A/C CV" if prefix == "Aend" else None)
        ax.plot(grouped["gmp_term_count"], grouped[f"{prefix}_B_NMSE_dB"], marker="s", color=COLORS["B"], label="B")
        ax.axvline(selected_count, color=COLORS["selected"], linestyle="--", linewidth=0.8)
        ax.axhline(-40, color="#777777", linestyle=":", linewidth=0.7)
        ax.set_title(title)
        ax.set_xlabel("GMP term count")
        _style(ax)
    axes[0].set_ylabel("NMSE (dB), 14-state median")
    axes[1].legend(fontsize=7)
    fig.suptitle("Train, internal CV and B behavior across the nested path", y=1.02)
    fig.tight_layout()
    return _save(fig, output / "figure3_train_cv_B_nested_path")


def _figure4(comparison: pd.DataFrame, output: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(8.0, 3.9))
    x = np.arange(comparison.shape[0])
    width = 0.19
    for index, (name, color) in enumerate(zip(METRICS, (COLORS["A"], COLORS["B"], COLORS["C"], "#B64342"), strict=True)):
        ax.bar(x + (index - 1.5) * width, comparison[f"selected_{name}"], width, color=color, label=LABELS[name])
    ax.axhline(-40, color="#777777", linestyle="--", linewidth=0.8)
    ax.set_xticks(x, comparison["state_id"].astype(str), rotation=45, ha="right")
    ax.set_ylabel("NMSE (dB)")
    ax.set_title("Frozen versus selected Sparse-GMP on failure14")
    ax.legend(ncol=2, fontsize=7)
    _style(ax)
    fig.tight_layout()
    return _save(fig, output / "figure4_selected_vs_frozen_failure14")


def _figure5(comparison: pd.DataFrame, output: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(8.0, 3.8))
    x = np.arange(comparison.shape[0])
    width = 0.19
    for index, (name, color) in enumerate(zip(METRICS, (COLORS["A"], COLORS["B"], COLORS["C"], "#B64342"), strict=True)):
        ax.bar(x + (index - 1.5) * width, comparison[f"delta_{name}_selected_minus_frozen_dB"], width, color=color, label=LABELS[name])
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xticks(x, comparison["state_id"].astype(str), rotation=45, ha="right")
    ax.set_ylabel("Selected − frozen (dB)")
    ax.set_title("Four-metric change on failure14 (negative is improvement)")
    ax.legend(ncol=2, fontsize=7)
    _style(ax)
    fig.tight_layout()
    return _save(fig, output / "figure5_delta_metrics_failure14")


def _figure6(trial_metrics: pd.DataFrame, output: Path) -> dict[str, str]:
    frame = trial_metrics.loc[trial_metrics["step"] == 1].copy()
    frame["group"] = frame["state_id"].astype(str) + "/" + frame["side"]
    matrix = frame.pivot(index="group", columns="trial_term_id", values="residual_correlation").reindex(columns=[f"GMP{i:03d}" for i in range(1, 19)])
    fig, ax = plt.subplots(figsize=(10.2, 6.0))
    image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="viridis", vmin=0, vmax=max(0.05, float(np.nanmax(matrix.to_numpy(dtype=float)))) )
    ax.set_xticks(np.arange(matrix.shape[1]), matrix.columns, rotation=65, ha="right")
    ax.set_yticks(np.arange(matrix.shape[0]), matrix.index, fontsize=6)
    ax.set_xlabel("GMP candidate term")
    ax.set_ylabel("Discovery group (State/side)")
    ax.set_title("Frozen-model residual correlation for GMP candidates")
    fig.colorbar(image, ax=ax, label="Normalized |φᴴe|")
    _style(ax)
    fig.tight_layout()
    return _save(fig, output / "figure6_frozen_residual_correlation_heatmap")


def _figure7(b_curve: pd.DataFrame, output: Path) -> dict[str, str]:
    grouped = b_curve.groupby("gmp_term_count", sort=True).agg({"Aend_condition_number": "median", "C2_condition_number": "median"}).reset_index()
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.plot(grouped["gmp_term_count"], grouped["Aend_condition_number"], marker="o", color=COLORS["A"], label="Aend Φ")
    ax.plot(grouped["gmp_term_count"], grouped["C2_condition_number"], marker="s", color=COLORS["C"], label="C2 Φ")
    ax.set_yscale("log")
    ax.set_xlabel("GMP term count")
    ax.set_ylabel("Median condition number (log scale)")
    ax.set_title("Conditioning along the nested Sparse-GMP path")
    ax.legend(fontsize=7)
    _style(ax)
    fig.tight_layout()
    return _save(fig, output / "figure7_condition_number_vs_term_count")


def _figure8(b_curve: pd.DataFrame, selected_count: int, output: Path) -> dict[str, str]:
    frame = b_curve.loc[b_curve["gmp_term_count"] == selected_count].sort_values("state_id")
    fig, ax = plt.subplots(figsize=(8.0, 3.7))
    x = np.arange(frame.shape[0])
    ax.plot(x, frame["Aend_GMP_response_energy_ratio"], marker="o", color=COLORS["A"], label="Aend")
    ax.plot(x, frame["C2_GMP_response_energy_ratio"], marker="s", color=COLORS["C"], label="C2")
    ax.set_xticks(x, frame["state_id"].astype(str), rotation=45, ha="right")
    ax.set_xlabel("State ID")
    ax.set_ylabel("GMP response energy / total response")
    ax.set_title(f"Selected G{selected_count} GMP block contribution")
    ax.legend(fontsize=7)
    _style(ax)
    fig.tight_layout()
    return _save(fig, output / "figure8_gmp_block_energy_contribution")


def _figure9_10(all425: pd.DataFrame, output: Path, side: str, stem: str) -> dict[str, str]:
    frame = all425.sort_values("state_id")
    fig, ax = plt.subplots(figsize=(8.8, 3.9))
    ax.plot(frame["state_id"], frame[f"frozen_{side}_B_NMSE_dB"], color=COLORS["frozen"], linewidth=0.75, label="Frozen")
    ax.plot(frame["state_id"], frame[f"selected_{side}_B_NMSE_dB"], color=COLORS["selected"], linewidth=0.75, label="Sparse-GMP")
    ax.axhline(-40, color="#777777", linestyle="--", linewidth=0.7)
    ax.set_xlabel("State ID")
    ax.set_ylabel("NMSE (dB)")
    ax.set_title(f"All-425 {side} → B: frozen versus selected Sparse-GMP")
    ax.legend(fontsize=7)
    _style(ax)
    fig.tight_layout()
    return _save(fig, output / stem)


def _figure11(all425: pd.DataFrame, output: Path) -> dict[str, str]:
    frame = all425.sort_values("state_id")
    colors = np.where(frame["split"].eq("failure14"), COLORS["failure"], COLORS["success"])
    fig, ax = plt.subplots(figsize=(8.8, 3.9))
    ax.bar(frame["state_id"], frame["worst4_delta_selected_minus_frozen"], color=colors, width=0.9)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_xlabel("State ID")
    ax.set_ylabel("Selected − frozen worst-four (dB)")
    ax.set_title("All-425 statewise worst-four change")
    ax.plot([], [], color=COLORS["failure"], linewidth=5, label="Failure14")
    ax.plot([], [], color=COLORS["success"], linewidth=5, label="Success411")
    ax.legend(fontsize=7, ncol=2)
    _style(ax)
    fig.tight_layout()
    return _save(fig, output / "figure11_all425_delta_worst4")


def _figure12(all425: pd.DataFrame, output: Path) -> dict[str, str]:
    values = [all425.loc[all425["split"] == name, "worst4_delta_selected_minus_frozen"].to_numpy(float) for name in ("failure14", "success411")]
    values.append(all425["worst4_delta_selected_minus_frozen"].to_numpy(float))
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    box = ax.boxplot(values, tick_labels=["Failure14", "Success411", "All425"], patch_artist=True, widths=0.6, showmeans=True)
    for patch, color in zip(box["boxes"], (COLORS["failure"], COLORS["success"], COLORS["selected"]), strict=True):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    ax.axhline(0, color="#555555", linewidth=0.8)
    ax.set_ylabel("Selected − frozen worst-four (dB)")
    ax.set_title("Improvement distribution")
    _style(ax)
    fig.tight_layout()
    return _save(fig, output / "figure12_improvement_distribution")


def generate_figures(sequence: pd.DataFrame, history: pd.DataFrame, trial_metrics: pd.DataFrame, b_curve: pd.DataFrame, comparison: pd.DataFrame, all425: pd.DataFrame, residuals: pd.DataFrame, numerical: pd.DataFrame, validation_summary: pd.DataFrame, output_dir: Path, *, selected_count: int) -> dict[str, Any]:
    del residuals, numerical, validation_summary
    details: dict[str, Any] = {}
    details["figure1"] = _figure1(history, output_dir)
    details["figure2"] = _figure2(sequence, selected_count, output_dir)
    details["figure3"] = _figure3(b_curve, selected_count, output_dir)
    details["figure4"] = _figure4(comparison, output_dir)
    details["figure5"] = _figure5(comparison, output_dir)
    details["figure6"] = _figure6(trial_metrics, output_dir)
    details["figure7"] = _figure7(b_curve, output_dir)
    details["figure8"] = _figure8(b_curve, selected_count, output_dir)
    details["figure9"] = _figure9_10(all425, output_dir, "Aend", "figure9_all425_Aend_B")
    details["figure10"] = _figure9_10(all425, output_dir, "C2", "figure10_all425_C2_B")
    details["figure11"] = _figure11(all425, output_dir)
    details["figure12"] = _figure12(all425, output_dir)
    return details


__all__ = ["generate_figures"]
