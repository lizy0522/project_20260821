"""Plots for Scenario 1 load-drift candidate selection."""

# ruff: noqa: E402,E501

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir())
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

TASK_NAME = "scenario_1_load_drift_state_selection_from_scenario_2"
RESULT_ROOT = PROJECT_ROOT / "results" / "pa_performance_evaluation" / "cross_scenario" /TASK_NAME


def _phase_frame(candidate: pd.DataFrame, value_column: str) -> pd.DataFrame:
    selected = candidate.loc[candidate["funMng"].isin([10, 20])].copy()
    selected["gamma"] = selected["funMng"] / 100.0
    return selected.pivot(index="funAng_deg", columns="gamma", values=value_column).sort_index()


def write_plots(candidate_path: Path | None = None, pairwise_path: Path | None = None) -> dict[str, object]:
    candidate = pd.read_csv(candidate_path or RESULT_ROOT / "01_candidate_summary.csv")
    pairwise = np.load(pairwise_path or RESULT_ROOT / "05_pairwise_B_CNMSE.npy", allow_pickle=False)
    phase = _phase_frame(candidate, "ACPR_avg_stale_dBc")
    colors = {0.1: "#1f77b4", 0.2: "#d62728"}

    figure_01, axis = plt.subplots(figsize=(11, 6), dpi=300)
    for gamma in (0.1, 0.2):
        axis.plot(phase.index, phase[gamma], marker="o", linewidth=1.3, label=f"|Gamma|={gamma:.1f}", color=colors[gamma])
    axis.set_xlabel("Fundamental load phase (deg)")
    axis.set_ylabel("Stale-DPD average ACPR (dBc)")
    axis.set_title("Scenario 1 Load Drift: Stale-DPD ACPR by Phase")
    axis.set_xticks(phase.index)
    axis.grid(True, alpha=0.3)
    axis.legend(frameon=False)
    figure_01.tight_layout()
    path_01 = RESULT_ROOT / "figure_01_stale_dpd_vs_load_phase.png"
    figure_01.savefig(path_01, bbox_inches="tight")
    plt.close(figure_01)

    phase_nmse = _phase_frame(candidate, "NMSE_stale_dB")
    figure_02, axis = plt.subplots(figsize=(11, 6), dpi=300)
    for gamma in (0.1, 0.2):
        axis.plot(phase_nmse.index, phase_nmse[gamma], marker="o", linewidth=1.3, label=f"|Gamma|={gamma:.1f}", color=colors[gamma])
    axis.set_xlabel("Fundamental load phase (deg)")
    axis.set_ylabel("Stale-DPD NMSE (dB)")
    axis.set_title("Scenario 1 Load Drift: Stale-DPD NMSE by Phase")
    axis.set_xticks(phase_nmse.index)
    axis.grid(True, alpha=0.3)
    axis.legend(frameon=False)
    figure_02.tight_layout()
    path_02 = RESULT_ROOT / "figure_02_stale_nmse_vs_load_phase.png"
    figure_02.savefig(path_02, bbox_inches="tight")
    plt.close(figure_02)

    phase_cnmse = _phase_frame(candidate, "CNMSE_B_to_nominal_dB")
    figure_03, axis = plt.subplots(figsize=(11, 6), dpi=300)
    for gamma in (0.1, 0.2):
        axis.plot(phase_cnmse.index, phase_cnmse[gamma], marker="o", linewidth=1.3, label=f"|Gamma|={gamma:.1f}", color=colors[gamma])
    axis.set_xlabel("Fundamental load phase (deg)")
    axis.set_ylabel("B-segment CNMSE to nominal (dB)")
    axis.set_title("Scenario 1 Load Drift: Behavior Shift by Phase")
    axis.set_xticks(phase_cnmse.index)
    axis.grid(True, alpha=0.3)
    axis.legend(frameon=False)
    figure_03.tight_layout()
    path_03 = RESULT_ROOT / "figure_03_cnmse_to_nominal_vs_phase.png"
    figure_03.savefig(path_03, bbox_inches="tight")
    plt.close(figure_03)

    labels = candidate["load_label"].tolist()
    mask = np.isfinite(pairwise)
    display = np.where(mask, pairwise, np.nan)
    figure_04, axis = plt.subplots(figsize=(13, 11), dpi=260)
    image = axis.imshow(display, cmap="viridis", aspect="auto")
    figure_04.colorbar(image, ax=axis, label="B-segment CNMSE (dB)")
    axis.set_xticks(np.arange(len(labels)), labels, rotation=90, fontsize=6)
    axis.set_yticks(np.arange(len(labels)), labels, fontsize=6)
    axis.set_xlabel("Candidate state")
    axis.set_ylabel("Candidate state")
    axis.set_title("Candidate Pairwise Real-B CNMSE")
    figure_04.tight_layout()
    path_04 = RESULT_ROOT / "figure_04_candidate_pairwise_B_CNMSE_heatmap.png"
    figure_04.savefig(path_04, bbox_inches="tight")
    plt.close(figure_04)

    figure_05, axis = plt.subplots(figsize=(10, 7), dpi=300)
    for gamma, color in ((0.1, "#1f77b4"), (0.2, "#d62728")):
        selected = candidate.loc[candidate["funMng"] == int(round(gamma * 100))]
        axis.scatter(selected["CNMSE_B_to_nominal_dB"], selected["Delta_ACPR_stale_vs_nominal_dB"], label=f"|Gamma|={gamma:.1f}", color=color, s=35)
        for row in selected.itertuples(index=False):
            axis.annotate(f"S{row.State}", (row.CNMSE_B_to_nominal_dB, row.Delta_ACPR_stale_vs_nominal_dB), fontsize=6, xytext=(2, 2), textcoords="offset points")
    axis.set_xlabel("B-segment CNMSE to nominal (dB)")
    axis.set_ylabel("Stale-DPD ACPR degradation vs nominal (dB)")
    axis.set_title("Scenario 1 Load Drift: Stale Mismatch vs Behavior Shift")
    axis.grid(True, alpha=0.3)
    axis.legend(frameon=False)
    figure_05.tight_layout()
    path_05 = RESULT_ROOT / "figure_05_stale_mismatch_vs_behavior_shift.png"
    figure_05.savefig(path_05, bbox_inches="tight")
    plt.close(figure_05)

    metadata = {"figure_01": str(path_01), "figure_02": str(path_02), "figure_03": str(path_03), "figure_04": str(path_04), "figure_05": str(path_05), "phase_series_count": 2, "pairwise_matrix_shape": list(pairwise.shape)}
    (RESULT_ROOT / "12_figure_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    write_plots()


if __name__ == "__main__":
    main()
