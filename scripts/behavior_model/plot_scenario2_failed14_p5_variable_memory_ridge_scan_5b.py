"""Python/matplotlib figures for the finite P5 variable-memory Ridge scan.

Figure contract: establish whether Ridge changes the training/generalization
tradeoff for the 20 P5 memory profiles on the 14 failed states.  The figure
archetype is a quantitative grid with lambda curves, profile-by-lambda heatmaps,
and one statewise validation panel.  All rendering is Python/matplotlib only.
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
REPRESENTATIVE_PROFILES = (
    "P5_M[1,1,1]",
    "P5_M[2,2,1]",
    "P5_M[3,2,1]",
    "P5_M[4,3,1]",
    "P5_M[4,4,1]",
    "P5_M[4,4,4]",
)


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


def _figure1(profile_lambda: pd.DataFrame, output: Path) -> dict[str, str]:
    fig, axes = plt.subplots(2, 3, figsize=(8.2, 5.2), sharey=True)
    axes = axes.ravel()
    line_specs = (
        ("Aend_train_median", "Aend train", COLORS["Aend_train"]),
        ("Aend_B_median", "Aend → B", COLORS["Aend_B"]),
        ("C2_train_median", "C2 train", COLORS["C2_train"]),
        ("C2_B_median", "C2 → B", COLORS["C2_B"]),
    )
    for ax, profile in zip(axes, REPRESENTATIVE_PROFILES, strict=True):
        group = profile_lambda.loc[profile_lambda["memory_profile_string"] == profile].copy()
        group["lambda_x"] = group["lambda"].map(_lambda_x)
        for column, label, color in line_specs:
            ax.plot(
                group["lambda_x"],
                group[column],
                marker="o",
                markersize=2.6,
                linewidth=0.9,
                color=color,
                label=label,
            )
        ax.axhline(-40.0, color="#777777", linestyle="--", linewidth=0.7)
        ax.set_title(profile)
        ax.set_xticks(
            [-13.5, -12, -10, -8, -6, -4],
            ["OLS", "1e−12", "1e−10", "1e−8", "1e−6", "1e−4"],
            rotation=35,
            ha="right",
        )
        _style(ax)
    axes[0].set_ylabel("NMSE (dB), median")
    axes[3].set_ylabel("NMSE (dB), median")
    axes[-1].legend(ncol=2, fontsize=6, loc="upper left")
    fig.suptitle("Representative memory profiles across Ridge strength", y=1.01)
    fig.tight_layout()
    return _save(fig, output / "figure1_representative_profiles_vs_lambda")


def _figure2(profile_lambda: pd.DataFrame, output: Path) -> dict[str, str]:
    base = profile_lambda.loc[profile_lambda["lambda"] == 0.0].set_index("memory_profile_string")
    positive = profile_lambda.loc[profile_lambda["lambda"] > 0].copy()
    positive["lambda_x"] = positive["lambda"].map(_lambda_x)
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2), sharex=True)
    for profile, group in positive.groupby("memory_profile_string", sort=False):
        base_row = base.loc[profile]
        delta_a = group["Aend_B_median"] - float(base_row["Aend_B_median"])
        delta_c = group["C2_B_median"] - float(base_row["C2_B_median"])
        axes[0].plot(group["lambda_x"], delta_a, color="#9BB7D4", linewidth=0.8, alpha=0.7)
        axes[1].plot(group["lambda_x"], delta_c, color="#D5A39F", linewidth=0.8, alpha=0.7)
    axes[0].axhline(0.0, color="#555555", linestyle="--", linewidth=0.8)
    axes[1].axhline(0.0, color="#555555", linestyle="--", linewidth=0.8)
    axes[0].set_title("Aend → B change")
    axes[1].set_title("C2 → B change")
    for ax in axes:
        ax.set_xlabel("log10(lambda)")
        _style(ax)
    axes[0].set_ylabel("Ridge NMSE − OLS NMSE (dB)")
    fig.suptitle("Ridge generalization change across all 20 profiles", y=1.02)
    fig.tight_layout()
    return _save(fig, output / "figure2_ridge_generalization_change")


def _figure3(summary: pd.DataFrame, output: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(5.1, 4.0))
    scatter = ax.scatter(
        summary["worst_train_median"],
        summary["worst_B_median"],
        c=np.log10(summary["lambda"].replace(0.0, np.nan)),
        cmap="viridis",
        s=24,
        alpha=0.85,
        edgecolors="white",
        linewidths=0.3,
    )
    ax.scatter(
        summary.loc[summary["lambda"] == 0.0, "worst_train_median"],
        summary.loc[summary["lambda"] == 0.0, "worst_B_median"],
        color="#4D4D4D",
        marker="x",
        s=28,
        label="OLS (lambda=0)",
    )
    fig.colorbar(scatter, ax=ax, label="log10(lambda), Ridge")
    ax.set_xlabel("Worst train median NMSE (dB)")
    ax.set_ylabel("Worst B median NMSE (dB)")
    ax.set_title("Train–generalization tradeoff")
    ax.legend(fontsize=7)
    _style(ax)
    fig.tight_layout()
    return _save(fig, output / "figure3_train_generalization_tradeoff")


def _figure4(summary: pd.DataFrame, output: Path) -> dict[str, str]:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), constrained_layout=True)
    for ax, norm_col, metric_col, title, ylabel in (
        (
            axes[0],
            "Aend_theta_norm_median",
            "Aend_B_median",
            "Aend coefficients",
            "Aend → B median NMSE (dB)",
        ),
        (
            axes[1],
            "C2_theta_norm_median",
            "C2_B_median",
            "C2 coefficients",
            "C2 → B median NMSE (dB)",
        ),
    ):
        scatter = ax.scatter(
            summary[norm_col],
            summary[metric_col],
            c=np.log10(summary["lambda"].replace(0.0, np.nan)),
            cmap="viridis",
            s=24,
            alpha=0.85,
            edgecolors="white",
            linewidths=0.3,
        )
        ax.set_xlabel("Median coefficient L2 norm")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        _style(ax)
    fig.colorbar(scatter, ax=axes, shrink=0.82, pad=0.04, label="log10(lambda), Ridge")
    fig.suptitle("Coefficient shrinkage versus B generalization", y=1.02)
    return _save(fig, output / "figure4_theta_norm_vs_generalization")


def _figure5(profile_lambda: pd.DataFrame, output: Path) -> dict[str, str]:
    profiles = list(profile_lambda["memory_profile_string"].drop_duplicates())
    lambdas = list(profile_lambda["lambda"].drop_duplicates())
    labels = ["OLS" if value == 0 else f"{value:.0e}" for value in lambdas]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 5.2), sharey=True)
    for ax, column, title in (
        (axes[0], "Aend_B_median", "Aend → B median"),
        (axes[1], "C2_B_median", "C2 → B median"),
        (axes[2], "worst4_median", "Worst-four median"),
    ):
        matrix = profile_lambda.pivot(
            index="memory_profile_string", columns="lambda", values=column
        ).reindex(index=profiles, columns=lambdas)
        image = ax.imshow(matrix.to_numpy(dtype=float), aspect="auto", cmap="magma")
        ax.set_xticks(np.arange(len(lambdas)), labels, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(profiles)), profiles)
        ax.set_title(title)
        ax.set_xlabel("lambda")
        cbar = fig.colorbar(image, ax=ax, shrink=0.75)
        cbar.set_label("NMSE (dB)")
        _style(ax)
    axes[0].set_ylabel("Memory profile")
    fig.suptitle("Memory profile × Ridge strength", y=1.01)
    fig.tight_layout()
    return _save(fig, output / "figure5_memory_profile_lambda_heatmaps")


def _figure6(selected: pd.DataFrame, output: Path) -> dict[str, str]:
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
    ax.set_title("Selected P5 variable-memory Ridge candidate")
    ax.legend(ncol=3, fontsize=7)
    _style(ax)
    fig.tight_layout()
    return _save(fig, output / "figure6_selected_candidate_statewise")


def _figure7(ols_vs_best: pd.DataFrame, output: Path) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(8.3, 3.8))
    x = np.arange(ols_vs_best.shape[0])
    width = 0.38
    ax.bar(
        x - width / 2,
        ols_vs_best["Aend_B_improvement_dB"],
        width,
        color=COLORS["Aend_B"],
        label="Aend → B improvement",
    )
    ax.bar(
        x + width / 2,
        ols_vs_best["C2_B_improvement_dB"],
        width,
        color=COLORS["C2_B"],
        label="C2 → B improvement",
    )
    ax.axhline(0.0, color="#555555", linewidth=0.8)
    ax.set_xticks(x, ols_vs_best["memory_profile_string"], rotation=65, ha="right")
    ax.set_ylabel("Best Ridge − OLS B median (dB)")
    ax.set_title("OLS versus best lambda within each memory profile")
    ax.legend(fontsize=7)
    _style(ax)
    fig.tight_layout()
    return _save(fig, output / "figure7_ols_vs_best_ridge")


def generate_figures(
    summary: pd.DataFrame,
    state_metrics: pd.DataFrame,
    selected: pd.DataFrame,
    profile_lambda: pd.DataFrame,
    ols_vs_best: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Any]:
    details: dict[str, Any] = {"figure1": _figure1(profile_lambda, output_dir)}
    details["figure2"] = _figure2(profile_lambda, output_dir)
    details["figure3"] = _figure3(summary, output_dir)
    details["figure4"] = _figure4(summary, output_dir)
    details["figure5"] = _figure5(profile_lambda, output_dir)
    details["figure6"] = _figure6(selected, output_dir)
    details["figure7"] = _figure7(ols_vs_best, output_dir)
    return details


__all__ = ["generate_figures"]
