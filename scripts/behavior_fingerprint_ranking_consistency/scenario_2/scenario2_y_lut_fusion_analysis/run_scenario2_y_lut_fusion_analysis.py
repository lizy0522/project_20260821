"""运行 Scenario 2 多阶段 Y-A LUT 指纹融合研究。"""

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

from behavior_fingerprint_ranking_consistency.scenario_2.scenario2_y_lut_fusion_analysis.plot_y_lut_fusion_analysis import (  # noqa: E402,E501
    plot_figure1_y_lut_fingerprint_robustness,
    plot_figure2_median_spearman_vs_c_stage,
)
from behavior_fingerprint_ranking_consistency.shared.scenario2_y_lut_fusion_analysis import (  # noqa: E402
    LUT_TYPES,
    run_y_lut_fusion_analysis,
)

TASK_NAME = "scenario2_y_lut_fusion_analysis"
REFERENCE_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2" /"scenario2_ranking_consistency"
    / "scenario_2"
)
ALL_ILC_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2" /"scenario2_all_ilc_analysis"
    / "scenario_2"
)
RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2" /TASK_NAME
    / "scenario_2_y_lut_fusion"
)
LOG_ROOT = PROJECT_ROOT / "work_logs" / "behavior_fingerprint_ranking_consistency" / "scenario_2" /TASK_NAME  # noqa: E501
EXECUTION_LOG = LOG_ROOT / "execution_log.txt"
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"


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
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        na_rep="NaN",
        float_format="%.17g",
    )


def _write_npz(path: Path, state_ids: np.ndarray, arrays: dict[str, np.ndarray]) -> None:
    payload = {"state_ids": np.asarray(state_ids, dtype=np.int64)}
    payload.update(arrays)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _append_log(text: str) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    with EXECUTION_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n[{timestamp}] Scenario 2 - Robust Y-A LUT Fingerprint Fusion\n")
        handle.write(text.rstrip() + "\n")


def _append_handoff(text: str) -> None:
    HANDOFF_LOG.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    with HANDOFF_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{timestamp} | {TASK_NAME}持续维护：Scenario 2 Y-A LUT指纹融合\n")
        handle.write(text.rstrip() + "\n")


def main() -> None:
    print("Scenario 2 robust Y-A LUT fingerprint fusion started.")
    print(f"Project root: {PROJECT_ROOT}")
    print("Step 1/5: load frozen Scenario 2 reference and all-ILC theta")
    result = run_y_lut_fusion_analysis(REFERENCE_ROOT, ALL_ILC_ROOT)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    print("Step 2/5: write fixed LUT and Query fingerprints")
    _write_npz(
        RESULT_ROOT / "lut_fingerprints.npz",
        result.state_ids,
        {
            "common_B_input": result.common_B_input,
            "Y_A1_fingerprints": result.lut_fingerprints["Y-A1"],
            "Y_A2_fingerprints": result.lut_fingerprints["Y-A2"],
            "Y_A12_mean_fingerprints": result.lut_fingerprints["Y-A12-Mean"],
            "fusion_definition": np.asarray(result.validation["fusion_definition"]),
        },
    )
    _write_npz(
        RESULT_ROOT / "query_fingerprints.npz",
        result.state_ids,
        {
            "effective_C_n": result.effective_C,
            **{f"Q_C{stage}": result.query_fingerprints[stage] for stage in range(1, 6)},
        },
    )

    print("Step 3/5: write 15 distance/ranking matrices and statistics")
    _write_npz(RESULT_ROOT / "distance_matrices.npz", result.state_ids, result.distance_matrices)
    _write_npz(RESULT_ROOT / "ranking_matrices.npz", result.state_ids, result.ranking_matrices)
    _write_frame(result.spearman_long, RESULT_ROOT / "spearman_by_state_long.csv")
    _write_frame(result.spearman_summary, RESULT_ROOT / "spearman_summary.csv")
    _write_frame(result.lut_robustness_summary, RESULT_ROOT / "lut_robustness_summary.csv")
    _write_frame(result.statewise_worst_over_C, RESULT_ROOT / "statewise_worst_over_C.csv")
    _write_frame(result.fusion_paired_comparison, RESULT_ROOT / "fusion_paired_comparison.csv")
    _write_frame(result.fusion_paired_summary, RESULT_ROOT / "fusion_paired_summary.csv")

    print("Step 4/5: generate Figure 1 and Figure 2")
    figure1_validation = plot_figure1_y_lut_fingerprint_robustness(
        result.spearman_by_combination,
        RESULT_ROOT / "figure1_y_lut_fingerprint_robustness.png",
    )
    figure2_validation = plot_figure2_median_spearman_vs_c_stage(
        result.spearman_summary,
        RESULT_ROOT / "figure2_median_spearman_vs_online_ilc_stage.png",
    )
    result.validation["figure1"] = figure1_validation
    result.validation["figure2"] = figure2_validation
    result.validation["figure_paths"] = [
        str(RESULT_ROOT / "figure1_y_lut_fingerprint_robustness.png"),
        str(RESULT_ROOT / "figure2_median_spearman_vs_online_ilc_stage.png"),
    ]
    result.validation["result_root"] = str(RESULT_ROOT)
    result.validation["output_files"] = {
        "lut_fingerprints": str(RESULT_ROOT / "lut_fingerprints.npz"),
        "query_fingerprints": str(RESULT_ROOT / "query_fingerprints.npz"),
        "distance_matrices": str(RESULT_ROOT / "distance_matrices.npz"),
        "ranking_matrices": str(RESULT_ROOT / "ranking_matrices.npz"),
        "spearman_by_state_long": str(RESULT_ROOT / "spearman_by_state_long.csv"),
        "spearman_summary": str(RESULT_ROOT / "spearman_summary.csv"),
        "lut_robustness_summary": str(RESULT_ROOT / "lut_robustness_summary.csv"),
        "statewise_worst_over_C": str(RESULT_ROOT / "statewise_worst_over_C.csv"),
        "fusion_paired_comparison": str(RESULT_ROOT / "fusion_paired_comparison.csv"),
        "fusion_paired_summary": str(RESULT_ROOT / "fusion_paired_summary.csv"),
        "validation": str(RESULT_ROOT / "validation.json"),
    }

    print("Step 5/5: write validation and task logs")
    _write_json(RESULT_ROOT / "validation.json", result.validation)
    medians = result.lut_robustness_summary.set_index("lut_type")
    paired_by_stage = result.fusion_paired_summary[
        result.fusion_paired_summary["scope"] == "C_stage"
    ]
    log_lines = [
        "研究问题：将 common-B 下独立生成的 Y-A1/Y-A2 行为指纹逐复数采样点平均，"
        "形成固定 Y-A12-Mean LUT，并评价其面对在线 C1...C5 的排序鲁棒性。",
        "只使用 Y-A1、Y-A2、Y-A12-Mean 作为LUT；X-A未作为LUT；Real-B仅作为冻结R_RR reference。",
        "未读取raw waveform做平均；未重新canonical、训练模型、扫描lambda、修改MP或重建D_RR。",
        f"A1/A2 availability={result.validation['A1_available_state_count']}/"
        f"{result.validation['A2_available_state_count']}；"
        f"fingerprints={result.validation['Y_A1_fingerprint_shape']}、"
        f"{result.validation['Y_A2_fingerprint_shape']}、"
        f"{result.validation['Y_A12_mean_fingerprint_shape']}；"
        "fingerprint-vs-theta mean max error="
        f"{result.validation['fingerprint_mean_theta_mean_max_abs_error']:.3e}。",
        f"15个距离矩阵和15个排名矩阵均为425x425；Spearman={result.validation['spearman_total_count']}，"
        f"全部finite；A1/A2 baseline回归最大Spearman误差="
        f"{result.baseline_regression['spearman_max_abs_error']:.3e}。",
        "跨C中位数："
        + "; ".join(
            f"{lut}="
            + ",".join(
                f"C{stage}:{medians.loc[lut, f'C{stage}_median']:.9f}" for stage in range(1, 6)
            )
            for lut in LUT_TYPES
        ),
        "Worst-C/mean-C/std-C/statewise-worst-median："
        + "; ".join(
            f"{lut}={medians.loc[lut, 'worst_C_median']:.9f}/"
            f"{medians.loc[lut, 'mean_C_median']:.9f}/"
            f"{medians.loc[lut, 'std_C_median']:.9f}/"
            f"{medians.loc[lut, 'statewise_worst_median']:.9f}"
            for lut in LUT_TYPES
        ),
        f"预声明规则选择的鲁棒LUT：{result.validation['selected_robust_lut_type']}。",
        "A12 paired median delta："
        + "; ".join(
            f"C{int(row.C_stage)} vs A1={row.median_delta_vs_A1:.9f}, "
            f"vs A2={row.median_delta_vs_A2:.9f}"
            for row in paired_by_stage.itertuples(index=False)
        ),
        "Figure1实际y范围="
        f"[{figure1_validation['y_min']:.15g}, {figure1_validation['y_max']:.15g}]；"
        "Figure2实际y范围="
        f"[{figure2_validation['y_min']:.15g}, {figure2_validation['y_max']:.15g}]；"
        "两图均无padding，Figure1六个轴共享范围。",
        f"结果目录：{RESULT_ROOT}。",
    ]
    _append_log("\n".join(log_lines))
    _append_handoff(
        f"完成Scenario 2固定Y-A LUT融合分析：Y-A1/Y-A2/Y-A12-Mean，"
        f"15个Query→LUT组合、6375个Spearman；按预声明Worst-C规则选择"
        f"{result.validation['selected_robust_lut_type']}；生成两张正式图，结果位于{RESULT_ROOT}。"
    )
    print("Scenario 2 robust Y-A LUT fingerprint fusion completed.")
    print(f"Selected robust LUT: {result.validation['selected_robust_lut_type']}")
    print(f"Results: {RESULT_ROOT}")


if __name__ == "__main__":
    main()
