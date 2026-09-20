"""
功能说明：执行State0 16点统一lambda Ridge诊断扫描并保存长表、聚合表、theta、验证JSON和5张图。
输入：一次生成的State0 canonical OFF/ILC A/B/C和既有OLS xy_equivalence基线。
输出：results/behavior_modeling/scenario_2/state0_ridge_scan下的轻量诊断结果。
用途：只识别候选lambda区域，不修改OLS默认配置、不冻结最终lambda、不扩展全状态。
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from behavior_modeling.scenario_2.state0_ridge_scan.ridge_scan import (  # noqa: E402
    RidgeScanResult,
    run_state0_ridge_scan,
    validate_ols_reproduction,
)

RESULT_ROOT = PROJECT_ROOT / "results" / "behavior_modeling" / "scenario_2" /"state0_ridge_scan"
OLS_ROOT = PROJECT_ROOT / "results" / "behavior_modeling" / "scenario_2" /"state0_xy_equivalence"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plot_scan(
    result: RidgeScanResult,
    *,
    fields: tuple[str, ...],
    labels: tuple[str, ...],
    ylabel: str,
    title: str,
    filename: str,
) -> None:
    positive_rows = [row for row in result.summary_rows if row["ridge_lambda"] > 0]
    x_values = np.array([row["ridge_lambda"] for row in positive_rows])
    figure, axis = plt.subplots(figsize=(8.8, 5.0))
    colors = ("#4D4D4D", "#377EB8", "#E68613")
    for field, label, color in zip(
        fields,
        labels,
        colors[: len(fields)],
        strict=True,
    ):
        y_values = [row[field] for row in positive_rows]
        baseline = result.summary_rows[0][field]
        axis.plot(x_values, y_values, marker="o", linewidth=1.5, label=label, color=color)
        axis.axhline(
            baseline,
            color=color,
            linestyle="--",
            alpha=0.45,
            linewidth=1.0,
        )
    axis.set_xscale("log")
    axis.set_xlabel("Ridge lambda (OLS baselines shown as dashed lines)")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(RESULT_ROOT / filename, dpi=180, bbox_inches="tight")
    plt.close(figure)


def _save_plots(result: RidgeScanResult) -> None:
    _plot_scan(
        result,
        fields=("X_train_NMSE_dB", "Y_A_train_median_dB", "Y_C_train_median_dB"),
        labels=("X", "Y-A median", "Y-C median"),
        ylabel="NMSE (dB)",
        title="State0 Ridge Scan: Train NMSE",
        filename="ridge_train_nmse_vs_lambda.png",
    )
    _plot_scan(
        result,
        fields=(
            "X_B_generalization_NMSE_dB",
            "Y_A_B_generalization_median_dB",
            "Y_C_B_generalization_median_dB",
        ),
        labels=("X", "Y-A median", "Y-C median"),
        ylabel="NMSE (dB)",
        title="State0 Ridge Scan: B-Segment Generalization",
        filename="ridge_B_generalization_vs_lambda.png",
    )
    _plot_scan(
        result,
        fields=("Y_A_vs_X_median_dB", "Y_C_vs_X_median_dB"),
        labels=("Y-A median", "Y-C median"),
        ylabel="CNMSE (dB)",
        title="State0 Ridge Scan: Common-B Y vs X",
        filename="ridge_Y_vs_X_cnmse_vs_lambda.png",
    )
    _plot_scan(
        result,
        fields=("Y_A_vs_realB_median_dB", "Y_C_vs_realB_median_dB"),
        labels=("Y-A median", "Y-C median"),
        ylabel="CNMSE (dB)",
        title="State0 Ridge Scan: Common-B Y vs Real-B",
        filename="ridge_Y_vs_realB_cnmse_vs_lambda.png",
    )
    _plot_scan(
        result,
        fields=("X_theta_l2_norm", "Y_A_theta_norm_median", "Y_C_theta_norm_median"),
        labels=("X", "Y-A median", "Y-C median"),
        ylabel="Theta L2 norm",
        title="State0 Ridge Scan: Coefficient Norm",
        filename="ridge_theta_norm_vs_lambda.png",
    )


def _save_result(result: RidgeScanResult, baseline_validation: dict[str, Any]) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_csv(RESULT_ROOT / "state0_ridge_scan_models.csv", result.model_rows)
    _write_csv(RESULT_ROOT / "state0_ridge_scan_summary.csv", result.summary_rows)
    np.savez_compressed(
        RESULT_ROOT / "theta_ridge_scan.npz",
        lambdas=result.lambdas,
        model_ids=np.asarray(result.model_ids),
        theta=result.theta,
    )
    validation = {
        **result.diagnostic,
        **baseline_validation,
        "signal_segmentation_unchanged": None,
        "data_raw_unchanged": None,
        "note": "Hash and raw-data checks are completed after execution and recorded in the log.",
    }
    with (RESULT_ROOT / "ridge_scan_validation.json").open("w", encoding="utf-8") as handle:
        json.dump(validation, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    _save_plots(result)


def main() -> None:
    result = run_state0_ridge_scan()
    baseline_validation = validate_ols_reproduction(result, OLS_ROOT, atol=1e-10)
    _save_result(result, baseline_validation)
    print(
        "lambda,X_B,Y-A_B_median,Y-C_B_median,Y-A_vs_X,Y-C_vs_X,"
        "Y-A_vs_Real,Y-C_vs_Real,Y-C_theta_norm"
    )
    for row in result.summary_rows:
        print(
            f"{row['ridge_lambda']:.12g},"
            f"{row['X_B_generalization_NMSE_dB']:.9f},"
            f"{row['Y_A_B_generalization_median_dB']:.9f},"
            f"{row['Y_C_B_generalization_median_dB']:.9f},"
            f"{row['Y_A_vs_X_median_dB']:.9f},"
            f"{row['Y_C_vs_X_median_dB']:.9f},"
            f"{row['Y_A_vs_realB_median_dB']:.9f},"
            f"{row['Y_C_vs_realB_median_dB']:.9f},"
            f"{row['Y_C_theta_norm_median']:.9g}"
        )
    print(json.dumps(result.diagnostic, ensure_ascii=False, indent=2))
    print(f"Results: {RESULT_ROOT}")


if __name__ == "__main__":
    main()
