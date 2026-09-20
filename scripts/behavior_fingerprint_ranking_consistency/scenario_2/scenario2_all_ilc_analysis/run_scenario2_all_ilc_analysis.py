"""
正式运行 Scenario 2 全 ILC requested n1×n2 排序保持性扫描。

旧 scenario_2 的 X-A/Real-B/reference 结果只读复用；本入口只写 scenario_2_all_ilc。
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from behavior_fingerprint_ranking_consistency.shared.plotting import (  # noqa: E402
    plot_all_ilc_spearman_2x5,
    plot_all_ilc_spearman_2x5_y_a_only,
    plot_all_ilc_spearman_boxplot,
    plot_all_ilc_spearman_by_state,
)
from behavior_fingerprint_ranking_consistency.shared.scenario2_all_ilc_analysis import (  # noqa: E402
    RIDGE_LAMBDA,
    run_scenario2_all_ilc_analysis,
)

TASK_NAME = "scenario2_all_ilc_analysis"
SOURCE_ROOT = (
    PROJECT_ROOT / "results" / "behavior_fingerprint_ranking_consistency" / "scenario_2" /TASK_NAME / "scenario_2"  # noqa: E501
)
RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2" /TASK_NAME
    / "scenario_2_all_ilc"
)
LOG_ROOT = PROJECT_ROOT / "work_logs" / "behavior_fingerprint_ranking_consistency" / "scenario_2" /TASK_NAME  # noqa: E501
EXECUTION_LOG = LOG_ROOT / "execution_log.txt"
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"
FROZEN_ROOT = PROJECT_ROOT / "results" / "behavior_modeling" / "scenario_2" /"scenario_2_ridge_analysis" / "final"  # noqa: E501
FROZEN_VALIDATION = FROZEN_ROOT / "scenario2_ridge_validation.json"


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"不能序列化类型：{type(value)!r}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")


def _write_frame(frame: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", na_rep="NaN")


def _append_log(text: str) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    with EXECUTION_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n[{timestamp}] Scenario 2 all ILC ranking consistency scan\n")
        handle.write(text.rstrip() + "\n")


def _append_handoff(text: str) -> None:
    HANDOFF_LOG.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    with HANDOFF_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{timestamp} | {TASK_NAME}持续维护：Scenario 2全ILC排序扫描\n")
        handle.write(text.rstrip() + "\n")


def _write_dynamic_npz(path: Path, state_ids: np.ndarray, arrays: dict[str, np.ndarray]) -> None:
    payload = {"state_ids": state_ids}
    payload.update(arrays)
    np.savez_compressed(path, **payload)


def _validate_frozen_ridge_lambda() -> None:
    if not FROZEN_VALIDATION.is_file():
        raise FileNotFoundError(f"缺少冻结Ridge validation：{FROZEN_VALIDATION}")
    with FROZEN_VALIDATION.open("r", encoding="utf-8-sig") as handle:
        validation = json.load(handle)
    if float(validation.get("selected_lambda", np.nan)) != RIDGE_LAMBDA:
        raise ValueError("冻结Ridge lambda不是1e-8")
    if validation.get("used_ridge") is not True:
        raise ValueError("冻结Ridge validation未标记used_ridge=true")


def main() -> None:
    print("Scenario 2 all ILC ranking consistency scan started.")
    print(f"Project root: {PROJECT_ROOT}")
    print("Step 1/6: frozen scenario_2 inputs and actual ILC model extraction")
    _validate_frozen_ridge_lambda()
    result = run_scenario2_all_ilc_analysis(
        SOURCE_ROOT,
        progress_interval=25,
    )
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_frame(result.availability, RESULT_ROOT / "ilc_availability.csv")
    _write_frame(result.availability_summary, RESULT_ROOT / "ilc_availability_summary.csv")
    _write_frame(result.effective_iteration_map, RESULT_ROOT / "effective_iteration_map.csv")
    _write_frame(
        result.effective_iteration_summary,
        RESULT_ROOT / "effective_iteration_summary.csv",
    )
    _write_dynamic_npz(
        RESULT_ROOT / "all_ilc_theta.npz",
        result.frozen.state_ids,
        {
            "theta_Y_A_actual": result.actual_models.theta_Y_A_actual,
            "theta_Y_C_actual": result.actual_models.theta_Y_C_actual,
            "available_mask": result.actual_models.available_mask,
            "ilc_column_counts": result.actual_models.ilc_column_counts,
        },
    )
    _write_frame(result.actual_models.model_metrics, RESULT_ROOT / "all_ilc_model_metrics.csv")

    print("Step 2/6: write all distance matrices")
    _write_dynamic_npz(
        RESULT_ROOT / "distance_matrices_all_ilc.npz",
        result.frozen.state_ids,
        result.distance_matrices,
    )
    print("Step 3/6: write all ranking matrices")
    _write_dynamic_npz(
        RESULT_ROOT / "ranking_matrices_all_ilc.npz",
        result.frozen.state_ids,
        result.ranking_matrices,
    )

    print("Step 4/6: write Spearman long/summary tables and per-n1 tables")
    _write_frame(result.spearman_long, RESULT_ROOT / "spearman_all_ilc_long.csv")
    _write_frame(result.spearman_summary, RESULT_ROOT / "spearman_summary_all_ilc.csv")
    figure_paths: list[Path] = []
    for requested_n1, table in result.spearman_by_n1.items():
        wide = table.copy()
        wide.insert(7, "requested_n1", requested_n1)
        _write_frame(
            wide,
            RESULT_ROOT / f"spearman_n1_{requested_n1:02d}_by_state.csv",
        )
        boxplot_path = RESULT_ROOT / f"n1_{requested_n1:02d}_spearman_boxplot.png"
        curve_path = RESULT_ROOT / f"n1_{requested_n1:02d}_spearman_by_state.png"
        plot_all_ilc_spearman_boxplot(
            table,
            requested_n1,
            result.nmax,
            boxplot_path,
        )
        plot_all_ilc_spearman_by_state(
            table,
            requested_n1,
            result.nmax,
            curve_path,
        )
        figure_paths.extend((boxplot_path, curve_path))

    combined_path = RESULT_ROOT / "spearman_all_ilc_2x5_combined.png"
    combined_y_range = None
    if result.nmax == 5:
        combined_y_range = plot_all_ilc_spearman_2x5(
            result.spearman_by_n1,
            combined_path,
        )
        figure_paths.append(combined_path)

    y_a_only_path = RESULT_ROOT / "spearman_all_ilc_2x5_YA_only_combined.png"
    y_a_only_range = None
    if result.nmax == 5:
        y_a_only_range = plot_all_ilc_spearman_2x5_y_a_only(
            result.spearman_by_n1,
            y_a_only_path,
        )
        figure_paths.append(y_a_only_path)

    print("Step 5/6: write validation and log")
    png_paths = sorted(RESULT_ROOT.glob("n1_*_spearman_*.png"))
    result.validation["formal_png_count"] = len(png_paths)
    result.validation["expected_formal_png_count"] = 2 * result.nmax
    result.validation["formal_png_paths"] = [str(path) for path in png_paths]
    result.validation["formal_png_count_valid"] = len(png_paths) == 2 * result.nmax
    result.validation["combined_2x5_path"] = str(combined_path) if combined_y_range else None
    result.validation["combined_2x5_y_min"] = (
        None if combined_y_range is None else combined_y_range[0]
    )
    result.validation["combined_2x5_y_max"] = (
        None if combined_y_range is None else combined_y_range[1]
    )
    result.validation["combined_2x5_YA_only_path"] = str(y_a_only_path) if y_a_only_range else None
    result.validation["combined_2x5_YA_only_y_min"] = (
        None if y_a_only_range is None else y_a_only_range[0]
    )
    result.validation["combined_2x5_YA_only_y_max"] = (
        None if y_a_only_range is None else y_a_only_range[1]
    )
    result.validation["combined_2x5_YA_only_series_count"] = 5 if y_a_only_range else None
    if not result.validation["formal_png_count_valid"]:
        raise RuntimeError(f"正式PNG数量错误：{len(png_paths)} != {2 * result.nmax}")
    _write_json(RESULT_ROOT / "validation.json", result.validation)

    availability_distribution = {
        str(int(row.actual_ilc_column_count)): int(row.state_count)
        for row in result.availability_summary.itertuples(index=False)
    }
    log_lines = [
        "研究定义：requested n1为在线Y-C Query列，requested n2为离线Y-A LUT列；",
        "effective_n=min(requested_n,state_actual_max_n)，超过状态实际列数时保持最后可用ILC行为。",
        f"Nmax={result.nmax}；ILC availability={availability_distribution}。",
        f"输入/输出列严格对应；实际Y-A模型={result.validation['Y_A_actual_model_count']}，"
        f"实际Y-C模型={result.validation['Y_C_actual_model_count']}；"
        f"available theta finite={result.validation['all_available_theta_finite']}，"
        f"rank deficient={result.validation['rank_deficient_available_models']}。",
        f"ILC1回归Y-A误差={result.validation['ILC1_Y_A_max_abs_error']}，"
        f"Y-C误差={result.validation['ILC1_Y_C_max_abs_error']}；"
        "旧scenario_2指纹回归通过。",
        f"requested扫描距离矩阵={result.validation['distance_matrix_count']}，"
        f"排名矩阵={result.validation['ranking_matrix_count']}；"
        f"Spearman总数={result.validation['spearman_total_count']}。",
        f"PNG数量={len(png_paths)}，期望={2 * result.nmax}；每张图y轴使用该图实际最小/最大值。",
        f"结果目录：{RESULT_ROOT}",
        f"Ridge lambda固定={RIDGE_LAMBDA}，未重新调参；MP/canonical/X-A/Real-B/reference均冻结。",
    ]
    _append_log("\n".join(log_lines))
    _append_handoff(
        f"完成Scenario 2全ILC requested n1×n2排序扫描，Nmax={result.nmax}，"
        f"Spearman总数={result.validation['spearman_total_count']}，PNG={len(png_paths)}；"
        f"结果位于{RESULT_ROOT}。"
    )
    print("Step 6/6: all ILC scan completed.")
    print(f"Results: {RESULT_ROOT}")


if __name__ == "__main__":
    main()
