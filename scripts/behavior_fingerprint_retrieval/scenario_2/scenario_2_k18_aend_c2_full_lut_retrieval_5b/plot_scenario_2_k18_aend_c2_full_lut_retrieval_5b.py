"""Plots for the fixed K18 Aend/C2 Full-425 retrieval task."""

# ruff: noqa: E402,E501

from __future__ import annotations

import os
import sys
from pathlib import Path

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

TASK_NAME = "scenario_2_k18_aend_c2_full_lut_retrieval_5b"
RESULT_ROOT = PROJECT_ROOT / "results" / "behavior_fingerprint_retrieval" / "scenario_2" /TASK_NAME
STATE_COUNT = 425
THRESHOLD_DB = -40.0


def choose_plot_floor(values: np.ndarray) -> float:
    """Choose a display-only floor below every finite Real-B value."""

    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -60.0
    return float(5.0 * np.floor((float(np.min(finite)) - 5.0) / 5.0))


def add_query_labels(
    axis: plt.Axes,
    state_r: np.ndarray,
    values: np.ndarray,
    state_q: np.ndarray,
    *,
    color: str,
) -> None:
    """Annotate every Real-B marker, including exact self hits."""

    for index, (query, value, retrieved) in enumerate(
        zip(state_r, values, state_q, strict=True)
    ):
        axis.annotate(
            f"Q{int(retrieved)}",
            (int(query), float(value)),
            xytext=(0, 2.0 + float(index % 4) * 2.2),
            textcoords="offset points",
            fontsize=3.8,
            rotation=90,
            ha="center",
            va="bottom",
            color=color,
        )


def _prepare(frame: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    ordered = frame.sort_values("state_R").reset_index(drop=True)
    state_r = ordered["state_R"].to_numpy(dtype=int)
    if not np.array_equal(state_r, np.arange(STATE_COUNT, dtype=int)):
        raise ValueError("state_R must be the canonical 0...424 order")
    raw = ordered["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float)
    floor = choose_plot_floor(raw)
    ordered["retrieved_real_B_CNMSE_plot_dB"] = np.where(np.isfinite(raw), raw, floor)
    return ordered, floor


def write_figures(
    frame: pd.DataFrame,
    figure_01_path: Path | None = None,
    figure_02_path: Path | None = None,
) -> dict[str, object]:
    """Write the required six-metric and retrieval-only figures."""

    ordered, plot_floor = _prepare(frame)
    figure_01_path = figure_01_path or RESULT_ROOT / "figure_01_full_metrics_vs_state.png"
    figure_02_path = figure_02_path or RESULT_ROOT / "figure_02_retrieved_real_B_CNMSE_vs_state.png"
    x = ordered["state_R"].to_numpy(dtype=int)
    q = ordered["retrieved_state_Q"].to_numpy(dtype=int)
    real_values = ordered["retrieved_real_B_CNMSE_plot_dB"].to_numpy(dtype=float)

    metric_specs = (
        ("nmse_withoutdpd_dB", "NMSE without DPD", "#7f7f7f", "-"),
        ("Y_Aend_train_NMSE_dB", "Y-Aend train", "#1f77b4", "-"),
        ("Y_Aend_B_NMSE_dB", "Y-Aend -> B", "#2ca02c", "-"),
        ("Y_C2_train_NMSE_dB", "Y-C2 train", "#9467bd", "-"),
        ("Y_C2_B_NMSE_dB", "Y-C2 -> B", "#8c564b", "-"),
        (
            "retrieved_real_B_CNMSE_plot_dB",
            "Retrieved Real-B CNMSE",
            "#d62728",
            "o-",
        ),
    )
    fig, axis = plt.subplots(figsize=(38, 15), dpi=260)
    for column, label, color, style in metric_specs:
        values = ordered[column].to_numpy(dtype=float)
        axis.plot(
            x,
            values,
            style,
            color=color,
            linewidth=0.85,
            markersize=2.2 if column.startswith("retrieved") else 0.0,
            label=label,
        )
        if column == "retrieved_real_B_CNMSE_plot_dB":
            add_query_labels(axis, x, values, q, color=color)
    axis.axhline(
        THRESHOLD_DB,
        color="#111111",
        linestyle="--",
        linewidth=1.1,
        label="-40 dB shareability threshold",
    )
    axis.set_xlim(0, STATE_COUNT - 1)
    axis.set_xlabel("State index R")
    axis.set_ylabel("Metric (dB)")
    axis.set_title("Fixed K18 Aend LUT / C2 Query Full-425 Retrieval")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, ncol=3, loc="best")
    axis.text(
        0.01,
        0.01,
        f"Self-hit Real-B CNMSE = -Inf; plotted at {plot_floor:g} dB only for visualization. "
        "Non-self shareable criterion: Real-B CNMSE < -40 dB.",
        transform=axis.transAxes,
        fontsize=8,
        va="bottom",
    )
    fig.tight_layout()
    fig.savefig(figure_01_path)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(38, 14), dpi=260)
    axis.plot(
        x,
        real_values,
        "o-",
        color="#d62728",
        linewidth=0.9,
        markersize=2.4,
        label="retrieved_real_B_CNMSE_dB",
    )
    add_query_labels(axis, x, real_values, q, color="#d62728")
    axis.axhline(
        THRESHOLD_DB,
        color="#111111",
        linestyle="--",
        linewidth=1.2,
        label="-40 dB shareability threshold",
    )
    axis.set_xlim(0, STATE_COUNT - 1)
    axis.set_xlabel("State_R")
    axis.set_ylabel("CNMSE (dB)")
    axis.set_title("Retrieved Real-B CNMSE vs State_R: Fixed K18 Aend/C2 LUT")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, loc="best")
    axis.text(
        0.01,
        0.01,
        f"Self-hit Real-B CNMSE = -Inf; plot floor = {plot_floor:g} dB for display only.",
        transform=axis.transAxes,
        fontsize=8,
        va="bottom",
    )
    fig.tight_layout()
    fig.savefig(figure_02_path)
    plt.close(fig)

    return {
        "plot_floor_dB": plot_floor,
        "figure_01": str(figure_01_path),
        "figure_02": str(figure_02_path),
    }


def main() -> None:
    frame = pd.read_csv(RESULT_ROOT / "01_state_results.csv")
    metadata = write_figures(frame)
    print(metadata)


if __name__ == "__main__":
    main()
