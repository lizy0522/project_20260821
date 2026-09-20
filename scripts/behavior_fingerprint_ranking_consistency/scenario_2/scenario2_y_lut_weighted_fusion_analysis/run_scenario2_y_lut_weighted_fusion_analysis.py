"""运行 Scenario 2 Development/Validation 非等权全局 Y-A LUT 融合研究。"""

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

from behavior_fingerprint_ranking_consistency.scenario_2.scenario2_y_lut_weighted_fusion_analysis.plot_y_lut_weighted_fusion_analysis import (  # noqa: E402,E501
    plot_figure1_development_landscape,
    plot_figure2_validation_A2_vs_weighted,
    plot_figure3_validation_median_vs_c,
)
from behavior_fingerprint_ranking_consistency.shared.scenario2_y_lut_weighted_fusion_analysis import (  # noqa: E402,E501
    FORMAL_LUT_TYPES,
    run_weighted_fusion_analysis,
)

TASK_NAME = "scenario2_y_lut_weighted_fusion_analysis"
A123_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2" /TASK_NAME
    / "scenario_2_y_lut_fusion_A123"
)
REFERENCE_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2" /"scenario2_ranking_consistency"
    / "scenario_2"
)
RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2" /TASK_NAME
    / "scenario_2_y_lut_weighted_fusion"
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
        handle.write(f"\n\n[{timestamp}] Scenario 2 - Non-equal Global Weighted Y-A LUT Fusion\n")
        handle.write(text.rstrip() + "\n")


def _append_handoff(text: str) -> None:
    HANDOFF_LOG.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    with HANDOFF_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{timestamp} | {TASK_NAME}持续维护：Scenario 2非等权全局Y-A融合\n")
        handle.write(text.rstrip() + "\n")


def main() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    print("Scenario 2 non-equal global weighted Y-A LUT fusion started.")
    print(f"Project root: {PROJECT_ROOT}")

    def freeze_selected_weight(payload: dict[str, Any]) -> None:
        _write_json(RESULT_ROOT / "selected_weight.json", payload)
        print("WEIGHTS_FROZEN: selected_weight.json written from Development only.")

    print("Step 1/6: split Query states and scan coarse/fine simplex weights on Development")
    result = run_weighted_fusion_analysis(
        A123_ROOT,
        REFERENCE_ROOT,
        on_weights_frozen=freeze_selected_weight,
    )
    print("VALIDATION_UNLOCKED_AFTER_WEIGHT_FREEZE")

    print("Step 2/6: write split, grid, Development scans and frozen LUT")
    _write_frame(result.split_definition, RESULT_ROOT / "split_definition.csv")
    _write_frame(result.weight_grid, RESULT_ROOT / "weight_grid.csv")
    _write_frame(result.development_scan, RESULT_ROOT / "development_weight_scan.csv")
    _write_frame(result.development_ranking, RESULT_ROOT / "development_weight_ranking.csv")
    _write_npz(
        RESULT_ROOT / "lut_fingerprints_selected.npz",
        result.inputs.state_ids,
        {
            "Y_A2": result.inputs.base_fingerprints[1],
            "Y_A_weighted": result.selected_weight_fingerprint,
            "weights": result.selected_weights,
        },
    )
    _write_npz(
        RESULT_ROOT / "query_fingerprints.npz",
        result.inputs.state_ids,
        {
            "effective_C_n": result.inputs.effective_C_n,
            **{f"Q_C{stage}": result.inputs.query_fingerprints[stage] for stage in range(1, 6)},
        },
    )

    print("Step 3/6: write frozen-weight Validation matrices and statistics")
    _write_npz(
        RESULT_ROOT / "validation_distance_matrices.npz",
        result.validation_ids,
        result.validation_distance_matrices,
    )
    _write_npz(
        RESULT_ROOT / "validation_ranking_matrices.npz",
        result.validation_ids,
        result.validation_ranking_matrices,
    )
    _write_frame(result.validation_long, RESULT_ROOT / "validation_spearman_by_state.csv")
    _write_frame(result.validation_summary, RESULT_ROOT / "validation_summary.csv")
    _write_frame(result.validation_paired, RESULT_ROOT / "validation_paired_vs_A2.csv")
    _write_frame(result.tail_analysis, RESULT_ROOT / "tail_analysis.csv")
    _write_frame(result.condition_region_analysis, RESULT_ROOT / "condition_region_analysis.csv")
    _write_frame(result.saturation_sensitivity, RESULT_ROOT / "saturation_sensitivity_summary.csv")

    print("Step 4/6: generate Development and Validation figures")
    figure1_validation = plot_figure1_development_landscape(
        result.development_scan,
        result.selected_weights,
        RESULT_ROOT / "figure1_development_weight_landscape.png",
    )
    figure2_validation = plot_figure2_validation_A2_vs_weighted(
        result.validation_spearman,
        result.validation_ids,
        RESULT_ROOT / "figure2_validation_A2_vs_weighted.png",
    )
    figure3_validation = plot_figure3_validation_median_vs_c(
        result.validation_summary,
        RESULT_ROOT / "figure3_validation_median_vs_C_stage.png",
    )
    result.validation["figure1"] = figure1_validation
    result.validation["figure2"] = figure2_validation
    result.validation["figure3"] = figure3_validation
    result.validation["figure_paths"] = [
        str(RESULT_ROOT / "figure1_development_weight_landscape.png"),
        str(RESULT_ROOT / "figure2_validation_A2_vs_weighted.png"),
        str(RESULT_ROOT / "figure3_validation_median_vs_C_stage.png"),
    ]
    result.validation["result_root"] = str(RESULT_ROOT)
    result.validation["output_files"] = {
        "split_definition": str(RESULT_ROOT / "split_definition.csv"),
        "weight_grid": str(RESULT_ROOT / "weight_grid.csv"),
        "development_weight_scan": str(RESULT_ROOT / "development_weight_scan.csv"),
        "development_weight_ranking": str(RESULT_ROOT / "development_weight_ranking.csv"),
        "selected_weight": str(RESULT_ROOT / "selected_weight.json"),
        "lut_fingerprints_selected": str(RESULT_ROOT / "lut_fingerprints_selected.npz"),
        "query_fingerprints": str(RESULT_ROOT / "query_fingerprints.npz"),
        "validation_distance_matrices": str(RESULT_ROOT / "validation_distance_matrices.npz"),
        "validation_ranking_matrices": str(RESULT_ROOT / "validation_ranking_matrices.npz"),
        "validation_spearman_by_state": str(RESULT_ROOT / "validation_spearman_by_state.csv"),
        "validation_summary": str(RESULT_ROOT / "validation_summary.csv"),
        "validation_paired_vs_A2": str(RESULT_ROOT / "validation_paired_vs_A2.csv"),
        "tail_analysis": str(RESULT_ROOT / "tail_analysis.csv"),
        "condition_region_analysis": str(RESULT_ROOT / "condition_region_analysis.csv"),
        "saturation_sensitivity_summary": str(RESULT_ROOT / "saturation_sensitivity_summary.csv"),
        "validation": str(RESULT_ROOT / "validation.json"),
    }

    print("Step 5/6: write validation and task logs")
    _write_json(RESULT_ROOT / "validation.json", result.validation)
    selected = result.selected_weights
    dev = result.selected_weight["development_selection_metrics"]
    val_summary = result.validation_summary.set_index("lut_type")
    paired = result.validation_paired
    paired_by_c = paired.groupby("C_stage", sort=True)["delta"].agg(["median", "mean"])
    paired_counts = paired.assign(better=paired["delta"] > 0).groupby("C_stage")["better"].sum()
    log_lines = [
        "研究问题：在冻结A1/A2/A3 common-B fingerprint的前提下，"
        "只用Development选择一组全局非负且和为1的权重，"
        "冻结后在独立Validation比较Y-A2与Weighted Fusion。",
        "未平均/加权raw waveform；未重新canonical、建模、训练Ridge、扫描lambda、"
        "修改MP/ABC/common-B或重建Real-B reference。",
        f"split seed={result.validation['split_seed']}；"
        f"Development/Validation={result.validation['development_count']}/"
        f"{result.validation['validation_count']}；candidate始终425，overlap={result.validation['split_overlap_count']}，"
        f"union={result.validation['split_union_count']}。",
        f"coarse candidates={result.validation['coarse_candidate_count']}；"
        f"fine candidates={result.validation['fine_candidate_count']}；"
        f"coarse winner={result.selected_weight['coarse_winner']['weight_id']}；"
        f"final weights=({selected[0]:.12g},{selected[1]:.12g},{selected[2]:.12g})，"
        f"phase={result.selected_weight['selected_phase']}。",
        "WEIGHTS_FROZEN；VALIDATION_UNLOCKED_AFTER_WEIGHT_FREEZE；Validation未参与权重选择。",
        f"Development selected metrics：Worst-C={dev['worst_C_median']:.9f}，"
        f"StatewiseWorst={dev['statewise_worst_C_median']:.9f}，Mean-C={dev['mean_C_median']:.9f}，"
        f"Std-C={dev['std_C_median']:.9f}，Q05={dev['statewise_worst_q05']:.9f}。",
        f"Validation success={result.validation_success['success']}；"
        f"Worst-C gain={result.validation_success['worst_C_gain']:.9f}；"
        f"StatewiseWorst gain={result.validation_success['statewise_worst_C_median_gain']:.9f}；"
        f"Q05 gain={result.validation_success['statewise_worst_q05_gain']:.9f}。",
        "Validation C-stage median："
        + "; ".join(
            f"{lut}="
            + ",".join(
                f"C{stage}:{val_summary.loc[lut].get(f'C{stage}_median', np.nan):.9f}"
                for stage in range(1, 6)
            )
            for lut in FORMAL_LUT_TYPES
        ),
        "Validation paired median delta / better count："
        + "; ".join(
            f"C{stage}:{paired_by_c.loc[stage, 'median']:.9f}/{int(paired_counts.loc[stage])}"
            for stage in range(1, 6)
        ),
        f"Figure1 y=[{figure1_validation['y_min']:.15g},{figure1_validation['y_max']:.15g}]；"
        f"Figure2 y=[{figure2_validation['y_min']:.15g},{figure2_validation['y_max']:.15g}]；"
        f"Figure3 y=[{figure3_validation['y_min']:.15g},{figure3_validation['y_max']:.15g}]。",
        f"结果目录：{RESULT_ROOT}。",
    ]
    _append_log("\n".join(log_lines))
    _append_handoff(
        f"完成Scenario 2非等权全局Y-A融合：Development {result.validation['development_count']}、"
        f"Validation {result.validation['validation_count']}，coarse/fine候选"
        f"{result.validation['coarse_candidate_count']}/{result.validation['fine_candidate_count']}，"
        f"最终权重=({selected[0]:.6g},{selected[1]:.6g},{selected[2]:.6g})，"
        f"Validation success={result.validation_success['success']}，结果位于{RESULT_ROOT}。"
    )
    print("Step 6/6: weighted fusion research completed.")
    print(f"Selected weights: {selected.tolist()}")
    print(f"Validation success: {result.validation_success['success']}")
    print(f"Results: {RESULT_ROOT}")


if __name__ == "__main__":
    main()
