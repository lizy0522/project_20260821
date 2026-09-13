"""
功能说明：执行Scenario 2全425状态正式X-A、Y-A-1、Y-C-1 OLS分析并保存五类结果文件。
输入：data_manager和signal_segmentation提供的State0/全状态canonical OFF与ILC第1列。
输出：1275行长表、425行宽表、3行聚合表、theta(425,3,10) NPZ和validation JSON。
用途：建立论文正式全状态三模型基线；不使用ILC2以后、STALE、Ridge或科研图。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from behavior_model.scenario2_xy_analysis import (  # noqa: E402
    MODEL_IDS,
    analyze_state_xy,
    collect_scenario2_xy_analysis,
    validate_state0_regression,
)

RESULT_ROOT = PROJECT_ROOT / "results" / "behavior_model" / "scenario_2_xy_analysis"
BASELINE_ROOT = PROJECT_ROOT / "results" / "behavior_model" / "state_000" / "xy_equivalence"


def main() -> None:
    state0_result = analyze_state_xy(0)
    state0_validation = validate_state0_regression(
        state0_result,
        BASELINE_ROOT,
        atol=1e-10,
    )
    print(
        "State0 regression: PASS "
        f"(metric_error={state0_validation['state0_metric_max_abs_error']}, "
        f"theta_error={state0_validation['state0_theta_max_abs_error']})"
    )

    result = collect_scenario2_xy_analysis(state0_result=state0_result)
    result.validation.update(state0_validation)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    result.model_metrics.to_csv(
        RESULT_ROOT / "scenario2_xy_model_metrics.csv",
        index=False,
        encoding="utf-8-sig",
        na_rep="NaN",
    )
    result.state_summary.to_csv(
        RESULT_ROOT / "scenario2_xy_state_summary.csv",
        index=False,
        encoding="utf-8-sig",
        na_rep="NaN",
    )
    result.aggregate_summary.to_csv(
        RESULT_ROOT / "scenario2_xy_aggregate_summary.csv",
        index=False,
        encoding="utf-8-sig",
        na_rep="NaN",
    )
    np.savez_compressed(
        RESULT_ROOT / "scenario2_xy_theta.npz",
        state_ids=np.arange(425, dtype=np.int64),
        model_ids=np.asarray(MODEL_IDS),
        theta=result.theta,
    )
    with (RESULT_ROOT / "scenario2_xy_validation.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(result.validation, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("Scenario 2 XY analysis completed.")
    print("States: 425")
    print("Models: 1275")
    print("X-A: 425")
    print("Y-A-1: 425")
    print("Y-C-1: 425")
    print("Train NMSE: 1275")
    print("B-gen NMSE: 1275")
    print("vs-X CNMSE: 850")
    print("vs-Real-B CNMSE: 1275")
    print(f"Rank deficient models: {result.validation['rank_deficient_count']}")
    print(f"Results: {RESULT_ROOT}")

    if result.validation["rank_deficient_count"] > 0:
        raise RuntimeError("存在rank<10模型；结果已保留，但validation失败")
    if not result.validation["all_theta_finite"]:
        raise RuntimeError("存在非有限theta；结果已保留，但validation失败")


if __name__ == "__main__":
    main()
