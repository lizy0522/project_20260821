"""
执行 Scenario 2 全425状态统一 Ridge lambda 开发、独立验证和条件式最终提取。

运行顺序固定为：全状态 lambda=0 回归门槛、确定性 split、Development 粗/细扫描、
写出冻结参数、一次性 Validation；只有 Validation PASS 才生成 final Ridge 模型。
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from behavior_model.scenario2_ridge_analysis import (  # noqa: E402
    COARSE_LAMBDAS,
    MODEL_IDS,
    augment_final_model_metrics,
    build_scan_summary,
    build_selection_payload,
    build_validation_result,
    compare_lambda_zero_to_baseline,
    load_or_create_state_split,
    make_fine_lambda_grid,
    run_full_lambda_zero_regression,
    run_lambda_scan,
    run_ols_vs_ridge_comparison,
    select_lambda,
    threshold_counts,
)

RESULT_ROOT = PROJECT_ROOT / "results" / "behavior_model" / "scenario_2_ridge_analysis"
BASELINE_ROOT = PROJECT_ROOT / "results" / "behavior_model" / "scenario_2_xy_analysis"
EXECUTION_LOG = PROJECT_ROOT / "work_logs" / "behavior_model" / "execution_log.txt"
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "codex_handoff" / "codex_handoff.txt"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    raise TypeError(f"不能序列化类型：{type(value)!r}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", na_rep="NaN")


def _plot_development_scan(summary: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    positive = summary[summary["ridge_lambda"] > 0].copy()
    if positive.empty:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    curves = (
        ("median_YC_B_gen_gain_dB", "Y-C B-gen"),
        ("median_YC_vs_X_gain_dB", "Y-C vs X"),
        ("median_YC_vs_real_gain_dB", "Y-C vs Real-B"),
    )
    for column, label in curves:
        ax.plot(
            positive["ridge_lambda"],
            positive[column],
            marker="o",
            linewidth=1.4,
            markersize=3.5,
            label=label,
        )
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xscale("log")
    ax.set_xlabel("Unified Ridge lambda")
    ax.set_ylabel("Paired median gain (dB)")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(frameon=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_validation_bgen(state_summary: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x = state_summary["Y_C_1_B_generalization_nmse_db_ols_db"].to_numpy(dtype=float)
    y = state_summary["Y_C_1_B_generalization_nmse_db_ridge_db"].to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return
    fig, ax = plt.subplots(figsize=(6.4, 5.4), constrained_layout=True)
    ax.scatter(x[finite], y[finite], s=18, alpha=0.75, edgecolors="none")
    lower = float(np.min(np.concatenate([x[finite], y[finite]])))
    upper = float(np.max(np.concatenate([x[finite], y[finite]])))
    ax.plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Y-C-1 OLS B-gen NMSE (dB)")
    ax.set_ylabel("Y-C-1 Ridge B-gen NMSE (dB)")
    ax.set_title("Validation: Y-C-1 C→B generalization")
    ax.grid(True, alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_validation_common_b(state_summary: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pairs = (
        (
            "Y_C_1_commonB_vs_X_cnmse_db_ols_db",
            "Y_C_1_commonB_vs_X_cnmse_db_ridge_db",
            "Y-C-1 vs X",
        ),
        (
            "Y_C_1_commonB_vs_realB_cnmse_db_ols_db",
            "Y_C_1_commonB_vs_realB_cnmse_db_ridge_db",
            "Y-C-1 vs Real-B",
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.5), constrained_layout=True)
    for ax, (x_name, y_name, title) in zip(axes, pairs, strict=True):
        x = state_summary[x_name].to_numpy(dtype=float)
        y = state_summary[y_name].to_numpy(dtype=float)
        finite = np.isfinite(x) & np.isfinite(y)
        if not np.any(finite):
            ax.set_title(title + " (no finite data)")
            continue
        ax.scatter(x[finite], y[finite], s=18, alpha=0.75, edgecolors="none")
        lower = float(np.min(np.concatenate([x[finite], y[finite]])))
        upper = float(np.max(np.concatenate([x[finite], y[finite]])))
        ax.plot([lower, upper], [lower, upper], color="black", linestyle="--", linewidth=0.9)
        ax.set_xlabel("OLS (dB)")
        ax.set_ylabel("Ridge (dB)")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _append_execution_log(text: str) -> None:
    EXECUTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    with EXECUTION_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n[{timestamp}] Scenario 2 unified Ridge analysis\n")
        handle.write(text.rstrip() + "\n")


def _append_handoff(text: str) -> None:
    HANDOFF_LOG.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    with HANDOFF_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{timestamp} | behavior_model持续维护：Scenario 2统一Ridge λ开发与验证\n")
        handle.write(text.rstrip() + "\n")


def _print_scan_table(name: str, summary: pd.DataFrame) -> None:
    columns = [
        "ridge_lambda",
        "median_YC_B_gen_gain_dB",
        "median_YC_vs_X_gain_dB",
        "median_YC_vs_real_gain_dB",
        "median_X_B_gen_degradation_dB",
        "median_YA_B_gen_degradation_dB",
        "median_YC_train_degradation_dB",
        "feasible",
    ]
    available = [column for column in columns if column in summary.columns]
    print(f"{name} summary:")
    print(summary[available].to_string(index=False))


def _state_split_lookup(split_frame: pd.DataFrame) -> dict[int, str]:
    return {
        int(row.state_id): str(row.split)
        for row in split_frame.itertuples(index=False)
    }


def _write_validation_outputs(
    comparison: Any,
    validation_result: dict[str, Any],
    validation_root: Path,
) -> None:
    ols = comparison.ols_metrics.copy()
    ridge = comparison.ridge_metrics.copy()
    ols["evaluation_role"] = "OLS"
    ridge["evaluation_role"] = "Ridge"
    validation_metrics = pd.concat([ols, ridge], ignore_index=True)
    _write_frame(validation_metrics, validation_root / "validation_state_metrics.csv")
    _write_frame(comparison.state_summary, validation_root / "validation_state_summary.csv")
    _write_frame(comparison.aggregate_summary, validation_root / "validation_aggregate_summary.csv")
    _write_json(validation_root / "validation_result.json", validation_result)


def _write_final_outputs(
    comparison: Any,
    selected_lambda: float,
    final_root: Path,
) -> dict[str, Any]:
    final_root.mkdir(parents=True, exist_ok=True)
    final_metrics = augment_final_model_metrics(comparison)
    _write_frame(final_metrics, final_root / "scenario2_ridge_model_metrics.csv")
    _write_frame(comparison.state_summary, final_root / "scenario2_ridge_state_summary.csv")

    aggregate = comparison.aggregate_summary.copy()
    y_c_state = comparison.state_summary[comparison.state_summary["state_id"].notna()]
    threshold_info = {
        "Y_C_vs_X": {
            "OLS": threshold_counts(y_c_state, source="ols", metric_column="Y_C_vs_X"),
            "Ridge": threshold_counts(y_c_state, source="ridge", metric_column="Y_C_vs_X"),
        },
        "Y_C_vs_realB": {
            "OLS": threshold_counts(y_c_state, source="ols", metric_column="Y_C_vs_realB"),
            "Ridge": threshold_counts(y_c_state, source="ridge", metric_column="Y_C_vs_realB"),
        },
    }
    aggregate_rows = aggregate.to_dict("records")
    for row in aggregate_rows:
        if row["model_id"] == "Y-C-1":
            row.update({
                "Y_C_vs_X_threshold_counts": json.dumps(
                    threshold_info["Y_C_vs_X"], ensure_ascii=False
                ),
                "Y_C_vs_realB_threshold_counts": json.dumps(
                    threshold_info["Y_C_vs_realB"], ensure_ascii=False
                ),
            })
    aggregate = pd.DataFrame(aggregate_rows)
    _write_frame(aggregate, final_root / "scenario2_ridge_aggregate_summary.csv")
    np.savez_compressed(
        final_root / "scenario2_ridge_theta.npz",
        state_ids=np.asarray(comparison.state_ids, dtype=np.int64),
        model_ids=np.asarray(MODEL_IDS),
        theta=comparison.theta_ridge,
    )
    final_validation = {
        "selected_lambda": float(selected_lambda),
        "state_count": int(len(comparison.state_ids)),
        "model_count": int(final_metrics.shape[0]),
        "theta_shape": list(comparison.theta_ridge.shape),
        "theta_dtype": str(comparison.theta_ridge.dtype),
        "all_theta_finite": bool(np.all(np.isfinite(comparison.theta_ridge))),
        "model_ids": list(MODEL_IDS),
        "used_ilc_iteration_1_only": bool(
            all(final_metrics["ilc_iteration_used"].fillna(1).astype(int) <= 1)
        ),
        "used_stale": False,
        "used_ridge": True,
        "threshold_counts": threshold_info,
    }
    _write_json(final_root / "scenario2_ridge_validation.json", final_validation)
    return final_validation


def main() -> None:
    print("Scenario 2 unified Ridge analysis started.")
    print(f"Project root: {PROJECT_ROOT}")
    print("Step 1/8: full-state lambda=0 OLS regression gate")
    ols_metrics, ols_theta = run_full_lambda_zero_regression(progress_interval=50)
    baseline_check = compare_lambda_zero_to_baseline(
        ols_metrics,
        ols_theta,
        BASELINE_ROOT,
        atol=1e-10,
    )
    print(
        "lambda=0 regression: "
        f"{'PASS' if baseline_check['lambda_zero_regression_passed'] else 'FAIL'} "
        f"(theta_error={baseline_check['theta_max_abs_error']}, "
        f"metric_error={baseline_check['metric_max_abs_error']})"
    )
    if not baseline_check["lambda_zero_regression_passed"]:
        raise RuntimeError("lambda=0未完整复现425状态OLS，停止Ridge扫描")

    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    development_root = RESULT_ROOT / "development"
    validation_root = RESULT_ROOT / "validation"
    split_frame = load_or_create_state_split(RESULT_ROOT / "state_split.csv")
    split_lookup = _state_split_lookup(split_frame)
    development_ids = tuple(
        int(value)
        for value in split_frame.loc[split_frame["split"] == "development", "state_id"]
    )
    validation_ids = tuple(
        int(value)
        for value in split_frame.loc[split_frame["split"] == "validation", "state_id"]
    )
    print(
        f"Step 2/8: fixed split ready (development={len(development_ids)}, "
        f"validation={len(validation_ids)}, state0_split={split_lookup[0]})"
    )

    print("Step 3/8: Development coarse lambda scan")
    coarse_raw = run_lambda_scan(
        development_ids,
        COARSE_LAMBDAS,
        development_state_ids=development_ids,
        validation_state_ids=validation_ids,
        scan_stage="coarse",
        progress_interval=25,
    )
    coarse_metrics, coarse_summary = build_scan_summary(coarse_raw)
    _write_frame(coarse_metrics, development_root / "coarse_scan_state_metrics.csv")
    _write_frame(coarse_summary, development_root / "coarse_scan_summary.csv")
    _print_scan_table("Coarse", coarse_summary)
    coarse_selection = select_lambda(coarse_summary, stage="coarse")
    coarse_winner = coarse_selection.get("selected_lambda")
    if coarse_selection["status"] != "SELECTED":
        selection_payload = build_selection_payload(
            baseline_check=baseline_check,
            split_frame=split_frame,
            coarse_lambdas=COARSE_LAMBDAS,
            coarse_selection=coarse_selection,
            fine_lambdas=None,
            fine_selection=None,
        )
        selection_payload["decision"] = "OLS_RETAINED_NO_FEASIBLE_NONZERO_LAMBDA"
        _write_json(development_root / "lambda_selection.json", selection_payload)
        _plot_development_scan(
            coarse_summary,
            development_root / "coarse_scan_gain.png",
            "Development coarse unified Ridge scan",
        )
        message = (
            "Ridge统一lambda开发结束：粗扫描没有满足全部保护/改善约束的非零候选；"
            "未执行细扫描和Validation，OLS继续作为正式模型。\n"
            f"lambda_zero_regression={baseline_check}\n"
            f"split={len(development_ids)} development, {len(validation_ids)} validation\n"
            f"coarse_summary={development_root / 'coarse_scan_summary.csv'}\n"
            "final Ridge model not generated."
        )
        _append_execution_log(message)
        _append_handoff("- 统一Ridge粗扫描无可行非零lambda；OLS保持正式模型，未生成final Ridge。")
        print("NO_FEASIBLE_NONZERO_LAMBDA: OLS retained.")
        return

    print(f"Step 4/8: Development fine lambda scan around coarse winner={coarse_winner}")
    fine_lambdas = make_fine_lambda_grid(float(coarse_winner))
    fine_raw = run_lambda_scan(
        development_ids,
        fine_lambdas,
        development_state_ids=development_ids,
        validation_state_ids=validation_ids,
        scan_stage="fine",
        progress_interval=25,
    )
    fine_metrics, fine_summary = build_scan_summary(fine_raw)
    _write_frame(fine_metrics, development_root / "fine_scan_state_metrics.csv")
    _write_frame(fine_summary, development_root / "fine_scan_summary.csv")
    _print_scan_table("Fine", fine_summary)
    fine_selection = select_lambda(
        fine_summary,
        stage="fine",
        coarse_winner=float(coarse_winner),
    )
    selection_payload = build_selection_payload(
        baseline_check=baseline_check,
        split_frame=split_frame,
        coarse_lambdas=COARSE_LAMBDAS,
        coarse_selection=coarse_selection,
        fine_lambdas=fine_lambdas,
        fine_selection=fine_selection,
    )
    _plot_development_scan(
        coarse_summary,
        development_root / "coarse_scan_gain.png",
        "Development coarse unified Ridge scan",
    )
    _plot_development_scan(
        fine_summary,
        development_root / "fine_scan_gain.png",
        "Development fine unified Ridge scan",
    )
    if fine_selection["status"] != "SELECTED":
        selection_payload["decision"] = "OLS_RETAINED_NO_FEASIBLE_NONZERO_LAMBDA"
        _write_json(development_root / "lambda_selection.json", selection_payload)
        message = (
            "Ridge统一lambda开发结束：细扫描没有满足全部保护/改善约束的非零候选；"
            "未执行Validation，OLS继续作为正式模型。\n"
            f"coarse_winner={coarse_winner}\n"
            f"coarse_summary={development_root / 'coarse_scan_summary.csv'}\n"
            f"fine_summary={development_root / 'fine_scan_summary.csv'}\n"
            "final Ridge model not generated."
        )
        _append_execution_log(message)
        _append_handoff("- 统一Ridge细扫描无可行非零lambda；OLS保持正式模型，未生成final Ridge。")
        print("NO_FEASIBLE_NONZERO_LAMBDA after fine scan: OLS retained.")
        return

    selected_lambda = float(fine_selection["selected_lambda"])
    selection_payload["decision"] = "FROZEN_BEFORE_VALIDATION"
    _write_json(development_root / "lambda_selection.json", selection_payload)
    print(f"Step 5/8: lambda frozen before Validation: {selected_lambda}")

    print("Step 6/8: independent Validation (OLS versus frozen Ridge only)")
    validation_comparison = run_ols_vs_ridge_comparison(
        validation_ids,
        selected_lambda,
        split_lookup=split_lookup,
        progress_interval=25,
    )
    validation_result = build_validation_result(
        validation_comparison.state_summary,
        selected_lambda=selected_lambda,
        validation_state_count=len(validation_ids),
    )
    validation_result["lambda_selection_path"] = str(
        development_root / "lambda_selection.json"
    )
    _write_validation_outputs(validation_comparison, validation_result, validation_root)
    _plot_validation_bgen(
        validation_comparison.state_summary,
        validation_root / "validation_YC_Bgen_OLS_vs_Ridge.png",
    )
    _plot_validation_common_b(
        validation_comparison.state_summary,
        validation_root / "validation_YC_commonB_OLS_vs_Ridge.png",
    )
    print(
        f"Validation: {'PASS' if validation_result['pass'] else 'FAIL'}; "
        f"YC_B={validation_result['observed_medians_dB']['Y_C_B_gen']:.6f} dB, "
        f"YC_vsX={validation_result['observed_medians_dB']['Y_C_vs_X']:.6f} dB, "
        f"YC_vsReal={validation_result['observed_medians_dB']['Y_C_vs_real']:.6f} dB"
    )

    if not validation_result["pass"]:
        message = (
            "统一Ridge独立Validation FAIL；selected lambda已冻结但不接受为正式模型，"
            "OLS继续作为正式模型，未生成final目录。\n"
            f"selected_lambda={selected_lambda}\n"
            f"validation_result={validation_result}\n"
            f"selection={development_root / 'lambda_selection.json'}\n"
            f"validation={validation_root / 'validation_result.json'}"
        )
        _append_execution_log(message)
        _append_handoff(
            "- 统一Ridge Validation FAIL；selected lambda未冻结为正式模型，"
            "OLS保持正式模型。"
        )
        print("Validation FAIL: OLS retained; final Ridge model not generated.")
        return

    print("Step 7/8: Validation PASS; extract final 425-state Ridge models")
    all_state_ids = tuple(int(value) for value in split_frame["state_id"])
    final_comparison = run_ols_vs_ridge_comparison(
        all_state_ids,
        selected_lambda,
        split_lookup=split_lookup,
        progress_interval=50,
    )
    final_root = RESULT_ROOT / "final"
    final_validation = _write_final_outputs(
        final_comparison,
        selected_lambda,
        final_root,
    )
    print("Step 8/8: final outputs and logs written")
    print(f"Final results: {final_root}")
    print(
        f"Final states={final_validation['state_count']}, "
        f"models={final_validation['model_count']}, "
        f"theta_shape={final_validation['theta_shape']}"
    )
    message = (
        "统一Ridge独立Validation PASS；selected lambda已冻结并提取425状态1275模型正式Ridge结果。\n"
        f"selected_lambda={selected_lambda}\n"
        f"validation_result={validation_result}\n"
        f"final_validation={final_validation}\n"
        f"development={development_root}\n"
        f"validation={validation_root}\n"
        f"final={final_root}\n"
        "Ridge只改变系数求解；正式Y仍只使用ILC1；data/raw与旧结果应保持不变。"
    )
    _append_execution_log(message)
    _append_handoff(
        f"- 统一Ridge Validation PASS，冻结lambda={selected_lambda}并生成"
        "425状态/1275模型final结果。"
    )


if __name__ == "__main__":
    main()
