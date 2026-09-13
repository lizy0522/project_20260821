"""PNG-only figures for the independent multi-branch OLS search."""

# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"], "font.size": 8, "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.8, "legend.frameon": False})


def _save(fig: plt.Figure, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _style(ax: plt.Axes) -> None:
    ax.tick_params(direction="out", length=3, width=0.7)
    ax.grid(False)


def _figure1(trace: pd.DataFrame, output: Path) -> str:
    frame = trace.copy()
    if frame.empty:
        frame = pd.DataFrame({"parameter_block": ["none"], "BalancedMedian": [np.nan], "candidate_id": [0]})
    frame = frame.sort_values("candidate_id").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9.2, 3.9))
    ax.plot(np.arange(frame.shape[0]), frame["BalancedMedian"], color="#5E3C99", linewidth=0.8, marker="o", markersize=2.2)
    ax.set_xlabel("Evaluated candidate index")
    ax.set_ylabel("Balanced CV median (dB)")
    ax.set_title("Coordinate-beam search progress")
    _style(ax)
    fig.tight_layout()
    return _save(fig, output / "01_search_progress.png")


def _reference_medians(refs: pd.DataFrame, metric: str) -> pd.DataFrame:
    return refs.groupby("model")[metric].median().reindex(["Frozen", "G4", "SourceTop", "Selected"])


def _figure2(refs: pd.DataFrame, output: Path) -> str:
    metrics = [("Aend", "train_NMSE_dB", "Aend train"), ("Aend", "B_NMSE_dB", "Aend → B"), ("C2", "train_NMSE_dB", "C2 train"), ("C2", "B_NMSE_dB", "C2 → B")]
    models = ["Frozen", "G4", "SourceTop", "Selected"]
    values = np.array([[float(refs.loc[(refs.model == model) & (refs.side == side), metric].median()) for side, metric, _ in metrics] for model in models])
    x = np.arange(len(metrics))
    width = 0.19
    fig, ax = plt.subplots(figsize=(8.0, 4.0))
    colors = ["#555555", "#D07A00", "#4C72B0", "#5E3C99"]
    for index, (model, color) in enumerate(zip(models, colors, strict=True)):
        ax.bar(x + (index - 1.5) * width, values[index], width, color=color, label=model)
    ax.axhline(-40, color="#777777", linestyle="--", linewidth=0.8)
    ax.set_xticks(x, [item[2] for item in metrics], rotation=20, ha="right")
    ax.set_ylabel("NMSE (dB), 14-state median")
    ax.set_title("Selected multibranch model versus references")
    ax.legend(fontsize=7)
    _style(ax)
    fig.tight_layout()
    return _save(fig, output / "02_selected_vs_references_four_metrics.png")


def _figure3_4(refs: pd.DataFrame, output: Path, side: str, stem: str) -> str:
    frame = refs.loc[refs["side"] == side].copy()
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    for model, color in (("Frozen", "#555555"), ("G4", "#D07A00"), ("SourceTop", "#4C72B0"), ("Selected", "#5E3C99")):
        group = frame.loc[frame["model"] == model]
        ax.scatter(group["train_NMSE_dB"], group["B_NMSE_dB"], s=15, alpha=0.75, color=color, label=model)
    ax.plot([-60, -20], [-60, -20], color="#AAAAAA", linestyle="--", linewidth=0.7)
    ax.set_xlabel(f"{side} train NMSE (dB)")
    ax.set_ylabel(f"{side} → B NMSE (dB)")
    ax.set_title(f"{side} train versus B generalization")
    ax.legend(fontsize=7)
    _style(ax)
    fig.tight_layout()
    return _save(fig, output / stem)


def _figure5(refs: pd.DataFrame, output: Path) -> str:
    frame = refs.copy()
    frame["gap"] = frame["B_NMSE_dB"] - frame["train_NMSE_dB"]
    fig, ax = plt.subplots(figsize=(8.0, 3.7))
    for model, color in (("Frozen", "#555555"), ("G4", "#D07A00"), ("SourceTop", "#4C72B0"), ("Selected", "#5E3C99")):
        group = frame.loc[frame["model"] == model].sort_values(["side", "state_id"])
        for side, marker in (("Aend", "o"), ("C2", "s")):
            values = group.loc[group["side"] == side, "gap"].to_numpy(float)
            if values.size:
                ax.plot(np.arange(values.size), values, linewidth=0.7, marker=marker, markersize=2, color=color, alpha=0.7, label=f"{model} {side}")
    ax.axhline(0, color="#777777", linestyle="--", linewidth=0.7)
    ax.set_xlabel("State rank within side")
    ax.set_ylabel("Generalization gap (dB)")
    ax.set_title("Generalization gap across references")
    ax.legend(ncol=2, fontsize=6)
    _style(ax)
    fig.tight_layout()
    return _save(fig, output / "05_generalization_gap.png")


def _figure6(trace: pd.DataFrame, output: Path) -> str:
    frame = trace.drop_duplicates("candidate_id")
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    scatter = ax.scatter(frame["basis_count"], frame["BalancedMedian"], c=frame["max_effective_delay"], cmap="viridis", s=22, edgecolors="white", linewidths=0.25)
    ax.set_xlabel("Total basis count K")
    ax.set_ylabel("Balanced CV median (dB)")
    ax.set_title("Basis complexity versus blocked-CV score")
    fig.colorbar(scatter, ax=ax, label="Effective max delay")
    _style(ax)
    fig.tight_layout()
    return _save(fig, output / "06_basis_count_vs_cv.png")


def _figure7(refs: pd.DataFrame, output: Path) -> str:
    frame = refs.copy()
    frame["BalancedCV"] = frame.groupby(["model"])["train_NMSE_dB"].transform("median")
    med = frame.groupby("model").agg(condition_number=("condition_number", "median"), BalancedCV=("BalancedCV", "first")).reset_index()
    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    for _, row in med.iterrows():
        ax.scatter(row["condition_number"], row["BalancedCV"], s=45, label=row["model"])
        ax.annotate(row["model"], (row["condition_number"], row["BalancedCV"]), xytext=(4, 3), textcoords="offset points", fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Median condition number")
    ax.set_ylabel("Median train NMSE (dB)")
    ax.set_title("Conditioning versus fit score")
    _style(ax)
    fig.tight_layout()
    return _save(fig, output / "07_condition_number_vs_cv.png")


def _figure8(counts: pd.DataFrame, output: Path) -> str:
    frame = counts.groupby("family", sort=False)["basis_count"].first().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8.4, 4.0))
    ax.bar(np.arange(frame.size), frame.to_numpy(float), color="#4C72B0")
    ax.set_xticks(np.arange(frame.size), frame.index, rotation=65, ha="right")
    ax.set_ylabel("Basis count")
    ax.set_title("Selected basis family counts")
    _style(ax)
    fig.tight_layout()
    return _save(fig, output / "08_selected_basis_family_counts.png")


def _figure9(norms: pd.DataFrame, output: Path) -> str:
    frame = norms.groupby(["side", "family"], sort=False)["coefficient_norm"].median().reset_index()
    families = list(frame["family"].drop_duplicates())
    x = np.arange(len(families))
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    for side, marker, color in (("Aend", "o", "#0F4D92"), ("C2", "s", "#2E7D32")):
        values = frame.loc[frame["side"] == side].set_index("family")["coefficient_norm"].reindex(families)
        ax.plot(x, values, marker=marker, color=color, linewidth=0.9, label=side)
    ax.set_xticks(x, families, rotation=65, ha="right")
    ax.set_ylabel("Median coefficient L2 norm")
    ax.set_title("Selected coefficient norms by family")
    ax.legend(fontsize=7)
    _style(ax)
    fig.tight_layout()
    return _save(fig, output / "09_coefficient_family_norms.png")


def _figure10(trace: pd.DataFrame, output: Path) -> str:
    frame = trace.groupby("parameter_block", sort=False)["BalancedMedian"].min().reset_index()
    fig, ax = plt.subplots(figsize=(9.2, 4.0))
    ax.plot(np.arange(frame.shape[0]), frame["BalancedMedian"], marker="o", color="#7B3294", linewidth=0.9)
    ax.set_xticks(np.arange(frame.shape[0]), frame["parameter_block"], rotation=75, ha="right", fontsize=6)
    ax.set_ylabel("Best Balanced CV median (dB)")
    ax.set_title("Parameter-block sensitivity overview")
    _style(ax)
    fig.tight_layout()
    return _save(fig, output / "10_parameter_sensitivity_overview.png")


def _family_sensitivity(trace: pd.DataFrame, output: Path, family: str) -> str:
    mask = trace["parameter_block"].astype(str).str.contains(family, case=False, regex=False)
    frame = trace.loc[mask].sort_values("candidate_id")
    fig, ax = plt.subplots(figsize=(7.2, 3.7))
    if frame.empty:
        ax.text(0.5, 0.5, "No evaluated block", ha="center", va="center")
    else:
        ax.scatter(np.arange(frame.shape[0]), frame["BalancedMedian"], s=18, color="#4C72B0")
    ax.set_xlabel("Candidate index")
    ax.set_ylabel("Balanced CV median (dB)")
    ax.set_title(f"Sensitivity: {family}")
    _style(ax)
    fig.tight_layout()
    return _save(fig, output / f"sensitivity_{family}.png")


def generate_figures(trace: pd.DataFrame, history: pd.DataFrame, refs: pd.DataFrame, gaps: pd.DataFrame, conditioning: pd.DataFrame, counts: pd.DataFrame, norms: pd.DataFrame, uniqueness: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    del history, gaps, uniqueness
    details: dict[str, Any] = {}
    details["01_search_progress"] = _figure1(trace, output_dir)
    details["02_selected_vs_references_four_metrics"] = _figure2(refs, output_dir)
    details["03_Aend_train_vs_B"] = _figure3_4(refs, output_dir, "Aend", "03_Aend_train_vs_B.png")
    details["04_C2_train_vs_B"] = _figure3_4(refs, output_dir, "C2", "04_C2_train_vs_B.png")
    details["05_generalization_gap"] = _figure5(refs, output_dir)
    details["06_basis_count_vs_cv"] = _figure6(trace, output_dir)
    details["07_condition_number_vs_cv"] = _figure7(refs, output_dir)
    details["08_selected_basis_family_counts"] = _figure8(counts, output_dir)
    details["09_coefficient_family_norms"] = _figure9(norms, output_dir)
    details["10_parameter_sensitivity_overview"] = _figure10(trace, output_dir)
    for family in ("MP", "EMem", "Lag", "Lead", "V3", "V5", "Dynamic", "Static", "M_pre", "Env"):
        details[f"sensitivity_{family}"] = _family_sensitivity(trace, output_dir, family)
    return details


__all__ = ["generate_figures"]
