"""Plots for the fixed K18 0.5B Aend/C2 retrieval task."""

# ruff: noqa: E402,E501

from __future__ import annotations

import json
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

TASK_NAME = "scenario_2_k18_aend_c2_full_lut_retrieval_0p5b"
RESULT_ROOT = PROJECT_ROOT / "results" / "behavior_fingerprint_retrieval" / "scenario_2" /TASK_NAME
STATE_COUNT = 425


def choose_plot_floor(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return -60.0
    return float(5.0 * np.floor((float(np.min(finite)) - 5.0) / 5.0))


def _label(axis: plt.Axes, state_r: int, value: float, state_q: int, index: int) -> None:
    axis.annotate(
        f"R{state_r}{RIGHTWARDS}Q{state_q}",
        (state_r, value),
        xytext=(0, 2.0 + float(index % 4) * 2.2),
        textcoords="offset points",
        fontsize=3.8,
        rotation=90,
        ha="center",
        va="bottom",
        color="#c62828",
    )


RIGHTWARDS = "\N{RIGHTWARDS ARROW}"


def write_figures(frame: pd.DataFrame) -> dict[str, object]:
    ordered = frame.sort_values("state_R").reset_index(drop=True)
    state_r = ordered["state_R"].to_numpy(dtype=int)
    if not np.array_equal(state_r, np.arange(STATE_COUNT, dtype=int)):
        raise ValueError("state_R must be canonical 0...424")
    state_q = ordered["retrieved_state_Q"].to_numpy(dtype=int)
    raw_real_b = ordered["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float)
    floor = choose_plot_floor(raw_real_b)
    plot_real_b = np.where(np.isfinite(raw_real_b), raw_real_b, floor)
    ordered["retrieved_real_B_CNMSE_plot_dB"] = plot_real_b

    specs = (
        ("nmse_withoutdpd_dB", "NMSE without DPD", "#7f7f7f"),
        ("Y_Aend_train_NMSE_dB", "Y-Aend train", "#1f77b4"),
        ("Y_Aend_B_NMSE_dB", "Y-Aend -> B", "#2ca02c"),
        ("Y_C2_train_NMSE_dB", "Y-C2 train", "#9467bd"),
        ("Y_C2_B_NMSE_dB", "Y-C2 -> B", "#8c564b"),
        ("retrieved_real_B_CNMSE_plot_dB", "Retrieved Real-B CNMSE", "#d62728"),
    )
    figure_01 = RESULT_ROOT / "figure_01_full_metrics_vs_state_0p5b.png"
    fig, axis = plt.subplots(figsize=(48, 15), dpi=260)
    for column, label, color in specs:
        values = ordered[column].to_numpy(dtype=float)
        axis.plot(
            state_r,
            values,
            "o-" if column == "retrieved_real_B_CNMSE_plot_dB" else "-",
            color=color,
            linewidth=0.85,
            markersize=2.2 if column == "retrieved_real_B_CNMSE_plot_dB" else 0.0,
            label=label,
        )
        if column == "retrieved_real_B_CNMSE_plot_dB":
            for index, (r, value, q) in enumerate(zip(state_r, values, state_q, strict=True)):
                _label(axis, int(r), float(value), int(q), index)
    axis.set_xlim(0, STATE_COUNT - 1)
    axis.set_xlabel("State index")
    axis.set_ylabel("Metric (dB)")
    axis.set_title("0.5B Aend-C2 LUT Retrieval: Model Metrics and Real-B CNMSE")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, ncol=3, loc="best")
    axis.text(
        0.01,
        0.01,
        f"Self-hit Real-B CNMSE = -Inf; displayed at {floor:g} dB for visualization only.",
        transform=axis.transAxes,
        fontsize=8,
        va="bottom",
    )
    fig.tight_layout()
    fig.savefig(figure_01)
    plt.close(fig)

    figure_02 = RESULT_ROOT / "figure_02_retrieved_real_B_CNMSE_vs_state_0p5b.png"
    fig, axis = plt.subplots(figsize=(48, 14), dpi=260)
    axis.plot(
        state_r,
        plot_real_b,
        "o-",
        color="#d62728",
        linewidth=0.9,
        markersize=2.4,
        label="retrieved_real_B_CNMSE_dB",
    )
    for index, (r, value, q) in enumerate(zip(state_r, plot_real_b, state_q, strict=True)):
        _label(axis, int(r), float(value), int(q), index)
    axis.set_xlim(0, STATE_COUNT - 1)
    axis.set_xlabel("State_R")
    axis.set_ylabel("CNMSE (dB)")
    axis.set_title("0.5B Aend-C2 LUT Retrieval: Retrieved Real-B CNMSE")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, loc="best")
    axis.text(
        0.01,
        0.01,
        f"Self-hit Real-B CNMSE = -Inf; displayed at {floor:g} dB for visualization only.",
        transform=axis.transAxes,
        fontsize=8,
        va="bottom",
    )
    fig.tight_layout()
    fig.savefig(figure_02)
    plt.close(fig)

    metadata = {
        "plot_floor_dB": floor,
        "figure_01": str(figure_01),
        "figure_02": str(figure_02),
        "figure_01_data_series_count": 6,
        "figure_02_data_series_count": 1,
        "marker_label_count_figure_01": STATE_COUNT,
        "marker_label_count_figure_02": STATE_COUNT,
        "marker_label_format": "R<State_R>→Q<State_Q>",
    }
    (RESULT_ROOT / "12_figure_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def main() -> None:
    write_figures(pd.read_csv(RESULT_ROOT / "01_state_results.csv"))


if __name__ == "__main__":
    main()
