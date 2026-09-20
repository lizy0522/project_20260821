"""Python/matplotlib figures for the formal G4 C2→Aend retrieval."""

# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"], "svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 8, "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.8, "legend.frameon": False})

COLORS = {"raw": "#555555", "Atrain": "#0F4D92", "AB": "#D07A00", "Ctrain": "#2E7D32", "CB": "#B64342", "real": "#7B3294", "frozen": "#4D4D4D", "g4": "#5E3C99", "shareable": "#2E8B57", "failure": "#C44E52"}


def _save(fig: plt.Figure, stem: Path) -> dict[str, str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    # Keep PDF exports in a fresh file so an externally opened historical PDF
    # cannot block the final artifact on Windows.  PNG/SVG retain the stable
    # figure stem; the validation manifest records the fresh PDF path.
    outputs = {"png": stem.with_suffix(".png"), "svg": stem.with_suffix(".svg"), "pdf": stem.with_name(stem.name + "_final").with_suffix(".pdf")}
    fig.savefig(outputs["png"], dpi=600, bbox_inches="tight")
    fig.savefig(outputs["svg"], bbox_inches="tight")
    fig.savefig(outputs["pdf"], bbox_inches="tight")
    plt.close(fig)
    return {key: str(value) for key, value in outputs.items()}


def _style(ax: plt.Axes) -> None:
    ax.tick_params(direction="out", length=3, width=0.7)
    ax.grid(False)


def _plot_six(summary: pd.DataFrame, output: Path, *, nonexact_only: bool) -> dict[str, str]:
    frame = summary.sort_values("state_id").copy()
    x = frame["state_id"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    specs = (("nmse_withoutdpd_dB", "PA without DPD NMSE", COLORS["raw"], ""), ("Y_Aend_train_NMSE_dB", "Y-Aend train", COLORS["Atrain"], ""), ("Y_Aend_B_NMSE_dB", "Y-Aend → B", COLORS["AB"], ""), ("Y_C2_train_NMSE_dB", "Y-C2 train", COLORS["Ctrain"], ""), ("Y_C2_B_NMSE_dB", "Y-C2 → B", COLORS["CB"], ""), ("retrieved_real_B_CNMSE_dB", "Retrieved Real-B CNMSE", COLORS["real"], "o"))
    for column, label, color, marker in specs:
        values = frame[column].to_numpy(float)
        if column == "retrieved_real_B_CNMSE_dB":
            finite = np.isfinite(values)
            plot_values = np.where(finite, values, -55.0)
            ax.plot(x, plot_values, color=color, linewidth=0.7, marker=marker, markersize=2.2, label=label, zorder=3)
            if not nonexact_only:
                for index, record in enumerate(frame.itertuples(index=False)):
                    ax.annotate(str(int(record.retrieved_state_id)), (x[index], plot_values[index]), textcoords="offset points", xytext=(0, 3 if index % 2 == 0 else -5), ha="center", va="bottom" if index % 2 == 0 else "top", fontsize=3.0, color=color, alpha=0.85)
            else:
                mask = frame["retrieved_state_id"].to_numpy(int) != frame["state_id"].to_numpy(int)
                for index in np.flatnonzero(mask):
                    ax.annotate(str(int(frame.iloc[index]["retrieved_state_id"])), (x[index], plot_values[index]), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=3.2, color=color)
        else:
            ax.plot(x, values, color=color, linewidth=0.75, label=label)
    ax.axhline(-40, color="#777777", linestyle="--", linewidth=0.8, label="−40 dB")
    ax.set_xlabel("State ID (R)")
    ax.set_ylabel("dB")
    ax.set_title("G4 C2→Aend retrieval and Real-B verification" + (" (all Top-1 labels)" if not nonexact_only else " (non-Exact labels)"))
    ax.legend(ncol=3, fontsize=7, loc="best")
    _style(ax)
    fig.tight_layout()
    return _save(fig, output / ("figure1_six_metrics_with_retrieved_ids" if not nonexact_only else "figure1b_six_metrics_nonexact_labels"))


def _plot_map(summary: pd.DataFrame, output: Path) -> dict[str, str]:
    frame = summary.sort_values("state_id")
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    exact = frame["exact_hit"].astype(bool).to_numpy()
    ax.scatter(frame.loc[~exact, "state_id"], frame.loc[~exact, "retrieved_state_id"], s=12, color=COLORS["g4"], alpha=0.75, label="Non-Exact")
    ax.scatter(frame.loc[exact, "state_id"], frame.loc[exact, "retrieved_state_id"], s=16, color=COLORS["shareable"], alpha=0.9, label="Exact")
    ax.plot([0, 424], [0, 424], color="#777777", linestyle="--", linewidth=0.8, label="Q=R")
    ax.set_xlim(-3, 427)
    ax.set_ylim(-3, 427)
    ax.set_xlabel("State ID R")
    ax.set_ylabel("Retrieved State ID Q")
    ax.set_title("G4 retrieval map")
    ax.legend(fontsize=7)
    _style(ax)
    fig.tight_layout()
    return _save(fig, output / "figure2_retrieval_map")


def _plot_real_b(summary: pd.DataFrame, output: Path) -> dict[str, str]:
    frame = summary.sort_values("state_id")
    fig, ax = plt.subplots(figsize=(8.5, 3.9))
    shareable = frame["shareable_under_minus40dB"].astype(bool)
    values = frame["retrieved_real_B_CNMSE_dB"].to_numpy(float)
    plot_values = np.where(np.isfinite(values), values, -55.0)
    ax.scatter(frame.loc[shareable, "state_id"], plot_values[shareable.to_numpy()], color=COLORS["shareable"], s=13, label="Shareable (<−40 dB)")
    ax.scatter(frame.loc[~shareable, "state_id"], plot_values[(~shareable).to_numpy()], color=COLORS["failure"], s=13, label="Failure")
    ax.axhline(-40, color="#777777", linestyle="--", linewidth=0.8)
    ax.set_xlabel("State ID R")
    ax.set_ylabel("Retrieved Real-B CNMSE (dB)")
    ax.set_title("G4 retrieved Real-B CNMSE")
    ax.legend(fontsize=7)
    _style(ax)
    fig.tight_layout()
    return _save(fig, output / "figure3_retrieved_real_B_CNMSE")


def _plot_comparison(summary: pd.DataFrame, output: Path) -> dict[str, str]:
    groups = ["Exact count", "Shareable count", "Failure count"]
    frozen_values = [int(summary["Frozen_exact"].sum()), int(summary["Frozen_shareable"].sum()), int((~summary["Frozen_shareable"].astype(bool)).sum())]
    g4_values = [int(summary["exact_hit"].sum()), int(summary["shareable_under_minus40dB"].sum()), int((~summary["shareable_under_minus40dB"].astype(bool)).sum())]
    x = np.arange(len(groups))
    fig, ax = plt.subplots(figsize=(6.7, 3.9))
    ax.bar(x - 0.19, frozen_values, 0.38, color=COLORS["frozen"], label="Frozen")
    ax.bar(x + 0.19, g4_values, 0.38, color=COLORS["g4"], label="G4")
    ax.set_xticks(x, groups)
    ax.set_ylabel("State count")
    ax.set_title("Frozen versus G4 retrieval counts")
    ax.legend(fontsize=7)
    _style(ax)
    fig.tight_layout()
    return _save(fig, output / "figure4_frozen_vs_G4")


def generate_figures(summary: pd.DataFrame, output_dir: Path) -> dict[str, Any]:
    details: dict[str, Any] = {}
    details["figure1"] = _plot_six(summary, output_dir, nonexact_only=False)
    details["figure1b"] = _plot_six(summary, output_dir, nonexact_only=True)
    details["figure2"] = _plot_map(summary, output_dir)
    details["figure3"] = _plot_real_b(summary, output_dir)
    details["figure4"] = _plot_comparison(summary, output_dir)
    return details


__all__ = ["generate_figures"]
