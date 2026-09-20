"""运行 Scenario 2 A1/A2/A3 及等权多阶段 Y-A LUT 指纹融合分析。"""

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

from behavior_fingerprint_ranking_consistency.scenario_2.scenario2_y_lut_fusion_a123_analysis.plot_y_lut_fusion_A123_analysis import (  # noqa: E402,E501
    plot_figure1_A123_robustness,
    plot_figure2_median_vs_c_stage,
)
from behavior_fingerprint_ranking_consistency.shared.scenario2_y_lut_fusion_A123_analysis import (  # noqa: E402
    LUT_DEFINITIONS,
    LUT_TYPES,
    run_a123_y_lut_fusion_analysis,
)

TASK_NAME = "scenario2_y_lut_fusion_a123_analysis"
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
PREVIOUS_FUSION_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2" /TASK_NAME
    / "scenario_2_y_lut_fusion"
)
RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2" /TASK_NAME
    / "scenario_2_y_lut_fusion_A123"
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
        handle.write(f"\n\n[{timestamp}] Scenario 2 - A123 Y-A LUT Fingerprint Fusion\n")
        handle.write(text.rstrip() + "\n")


def _append_handoff(text: str) -> None:
    HANDOFF_LOG.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    with HANDOFF_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{timestamp} | {TASK_NAME}持续维护：Scenario 2 A123 Y-A LUT融合\n")
        handle.write(text.rstrip() + "\n")


def main() -> None:
    print("Scenario 2 A123 Y-A LUT fingerprint fusion started.")
    print(f"Project root: {PROJECT_ROOT}")
    print("Step 1/5: load frozen reference, all-ILC theta and previous fusion baseline")
    result = run_a123_y_lut_fusion_analysis(
        REFERENCE_ROOT,
        ALL_ILC_ROOT,
        PREVIOUS_FUSION_ROOT,
    )
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)

    print("Step 2/5: write seven LUT fingerprints and five Query fingerprints")
    definitions_json = json.dumps(
        {
            lut_type: [[int(stage), float(weight)] for stage, weight in definition]
            for lut_type, definition in LUT_DEFINITIONS.items()
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    _write_npz(
        RESULT_ROOT / "lut_fingerprints.npz",
        result.state_ids,
        {
            "common_B_input": result.common_B_input,
            **{
                f"{lut_type.replace('-', '_')}_fingerprints": result.lut_fingerprints[lut_type]
                for lut_type in LUT_TYPES
            },
            "A3_effective_stage_per_state": result.A3_effective_stage_per_state,
            "fusion_definitions_json": np.asarray(definitions_json),
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

    print("Step 3/5: write 35 distance/ranking matrices and all CSV tables")
    _write_npz(RESULT_ROOT / "distance_matrices.npz", result.state_ids, result.distance_matrices)
    _write_npz(RESULT_ROOT / "ranking_matrices.npz", result.state_ids, result.ranking_matrices)
    _write_frame(result.spearman_long, RESULT_ROOT / "spearman_by_state_long.csv")
    _write_frame(result.spearman_summary, RESULT_ROOT / "spearman_summary.csv")
    _write_frame(result.lut_robustness_summary, RESULT_ROOT / "lut_robustness_summary.csv")
    _write_frame(result.statewise_worst_over_C, RESULT_ROOT / "statewise_worst_over_C.csv")
    _write_frame(result.paired_vs_A2_by_state, RESULT_ROOT / "paired_vs_A2_by_state.csv")
    _write_frame(result.paired_vs_A2_summary, RESULT_ROOT / "paired_vs_A2_summary.csv")
    _write_frame(
        result.saturation_sensitivity_summary,
        RESULT_ROOT / "saturation_sensitivity_summary.csv",
    )

    print("Step 4/5: generate Figure 1 and Figure 2")
    figure1_validation = plot_figure1_A123_robustness(
        result.spearman_by_combination,
        RESULT_ROOT / "figure1_A123_lut_fingerprint_robustness.png",
    )
    figure2_validation = plot_figure2_median_vs_c_stage(
        result.spearman_summary,
        RESULT_ROOT / "figure2_median_spearman_vs_C_stage_A123.png",
    )
    result.validation["figure1"] = figure1_validation
    result.validation["figure2"] = figure2_validation
    result.validation["figure_paths"] = [
        str(RESULT_ROOT / "figure1_A123_lut_fingerprint_robustness.png"),
        str(RESULT_ROOT / "figure2_median_spearman_vs_C_stage_A123.png"),
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
        "paired_vs_A2_by_state": str(RESULT_ROOT / "paired_vs_A2_by_state.csv"),
        "paired_vs_A2_summary": str(RESULT_ROOT / "paired_vs_A2_summary.csv"),
        "saturation_sensitivity_summary": str(RESULT_ROOT / "saturation_sensitivity_summary.csv"),
        "validation": str(RESULT_ROOT / "validation.json"),
    }

    print("Step 5/5: write validation and task logs")
    _write_json(RESULT_ROOT / "validation.json", result.validation)
    robustness = result.lut_robustness_summary.set_index("lut_type")
    paired = result.paired_vs_A2_summary
    lines = [
        "研究问题：固定Y-A LUT仅使用A1/A2/A3及其等权fingerprint-level融合，"
        "评价面对在线C1...C5时的排序鲁棒性。",
        "LUT候选为Y-A1/Y-A2/Y-A3/Y-A12-Mean/Y-A13-Mean/Y-A23-Mean/Y-A123-Mean；"
        "X-A未作为LUT，Real-B仅用于冻结R_RR reference。",
        "融合发生在相同common-B probe生成的最终复数指纹层；未平均raw waveform，"
        "未连续扫描权重，未重新训练/canonical/调整MP/lambda，未重建D_RR。",
        f"A1/A2/A3 availability={result.validation['A1_available']}/"
        f"{result.validation['A2_available']}/{result.validation['A3_available']}；"
        f"唯一缺失A3状态={result.validation['missing_A3_state_id']}"
        f" ({result.validation['missing_A3_state']})；"
        f"A3 effective stage distribution={result.validation['A3_effective_stage_distribution']}。",
        f"七类fingerprint均为(425,4913) complex128 finite；"
        f"fusion/theta最大误差={max(result.fusion_equivalence_errors.values()):.3e}。",
        f"35个距离矩阵/35个排名矩阵均为425x425；Spearman={result.validation['spearman_count']}，"
        f"全部finite；上一轮A1/A2/A12回归={result.validation['baseline_regression_pass']}。",
        "鲁棒性排序："
        + "; ".join(
            f"{lut}=W{robustness.loc[lut, 'worst_C_median']:.9f},"
            f"SW{robustness.loc[lut, 'statewise_worst_C_median']:.9f},"
            f"M{robustness.loc[lut, 'mean_C_median']:.9f},"
            f"S{robustness.loc[lut, 'std_C_median']:.9f}"
            for lut in LUT_TYPES
        ),
        f"按预声明规则winner={result.validation['selected_robust_lut_type']}；"
        f"saturation sensitivity full/exclude winner="
        f"{result.sensitivity_validation['full_425_winner']}/"
        f"{result.sensitivity_validation['exclude_Ns2_winner']}，"
        f"winner unchanged={result.sensitivity_validation['winner_unchanged']}。",
        "相对Y-A2的paired统计已保存；"
        + "; ".join(
            f"C{int(row.C_stage)}-{row.lut_type}:"
            f"median_delta={row.median_delta:.9f},better={int(row.better_count)}"
            for row in paired.itertuples(index=False)
        ),
        f"Figure1 y=[{figure1_validation['y_min']:.15g},{figure1_validation['y_max']:.15g}]；"
        f"Figure2 y=[{figure2_validation['y_min']:.15g},{figure2_validation['y_max']:.15g}]；"
        "均使用实际min/max且无padding。",
        f"结果目录：{RESULT_ROOT}。",
    ]
    _append_log("\n".join(lines))
    _append_handoff(
        f"完成Scenario 2 A123 Y-A LUT融合研究：7类固定LUT、35组Query→LUT、"
        f"{result.validation['spearman_count']}个Spearman；winner={result.validation['selected_robust_lut_type']}，"
        f"saturation sensitivity unchanged={result.sensitivity_validation['winner_unchanged']}；"
        f"生成Figure1/2，结果位于{RESULT_ROOT}。"
    )
    print("Scenario 2 A123 Y-A LUT fingerprint fusion completed.")
    print(f"Selected robust LUT: {result.validation['selected_robust_lut_type']}")
    print(f"Results: {RESULT_ROOT}")


if __name__ == "__main__":
    main()
