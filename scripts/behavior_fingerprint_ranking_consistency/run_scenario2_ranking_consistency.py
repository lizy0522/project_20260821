"""
正式执行 Scenario 2 行为指纹排序一致性分析。

只读取已经冻结的 Ridge final theta，逐状态生成四类指纹，计算四个 CNMSE 距离矩阵、
四个排名矩阵、三类逐状态 Spearman 和两张固定纵轴图。
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from behavior_fingerprint_ranking_consistency.fingerprint_builder import (  # noqa: E402
    build_scenario2_fingerprints,
    load_frozen_ridge_models,
)
from behavior_fingerprint_ranking_consistency.plotting import (  # noqa: E402
    plot_spearman_boxplot,
    plot_spearman_by_state,
)
from behavior_fingerprint_ranking_consistency.scenario2_analysis import (  # noqa: E402
    build_scenario2_ranking_analysis,
)

TASK_NAME = "behavior_fingerprint_ranking_consistency"
RESULT_ROOT = PROJECT_ROOT / "results" / TASK_NAME / "scenario_2"
LOG_ROOT = PROJECT_ROOT / "work_logs" / TASK_NAME
EXECUTION_LOG = LOG_ROOT / "execution_log.txt"
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "codex_handoff" / "codex_handoff.txt"
FROZEN_ROOT = PROJECT_ROOT / "results" / "behavior_model" / "scenario_2_ridge_analysis" / "final"
THETA_PATH = FROZEN_ROOT / "scenario2_ridge_theta.npz"
FROZEN_VALIDATION_PATH = FROZEN_ROOT / "scenario2_ridge_validation.json"


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


def _append_log(text: str) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    with EXECUTION_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n[{timestamp}] {TASK_NAME}\n")
        handle.write(text.rstrip() + "\n")


def _append_handoff(text: str) -> None:
    HANDOFF_LOG.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    with HANDOFF_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{timestamp} | {TASK_NAME}\n")
        handle.write(text.rstrip() + "\n")


def main() -> None:
    print("Scenario 2 behavior fingerprint ranking consistency started.")
    print(f"Project root: {PROJECT_ROOT}")
    print("Step 1/7: load and validate frozen Ridge models")
    frozen_models = load_frozen_ridge_models(
        THETA_PATH,
        validation_path=FROZEN_VALIDATION_PATH,
        expected_lambda=1e-8,
    )
    print(
        f"Frozen models: states={len(frozen_models.state_ids)}, "
        f"models={frozen_models.model_ids}, lambda={frozen_models.ridge_lambda}"
    )

    print("Step 2/7: build four fingerprints and verify common B probe")
    fingerprints = build_scenario2_fingerprints(frozen_models, progress_interval=50)
    print(
        f"Fingerprints ready: common_B={fingerprints.common_B_input.shape}, "
        f"max_probe_difference={fingerprints.common_B_max_abs_difference}"
    )
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        RESULT_ROOT / "fingerprints.npz",
        state_ids=fingerprints.state_ids,
        common_B_input=fingerprints.common_B_input,
        X_A_fingerprints=fingerprints.X_A_fingerprints,
        Y_A_fingerprints=fingerprints.Y_A_fingerprints,
        Y_C_query_fingerprints=fingerprints.Y_C_query_fingerprints,
        real_B_fingerprints=fingerprints.real_B_fingerprints,
    )

    print("Step 3/7: compute D_RR, D_CX, D_CY and D_CR")
    analysis = build_scenario2_ranking_analysis(frozen_models, fingerprints)
    np.savez_compressed(
        RESULT_ROOT / "distance_matrices.npz",
        state_ids=analysis.frozen_models.state_ids,
        D_RR=analysis.distance_matrices["D_RR"],
        D_CX=analysis.distance_matrices["D_CX"],
        D_CY=analysis.distance_matrices["D_CY"],
        D_CR=analysis.distance_matrices["D_CR"],
    )

    print("Step 4/7: convert distances to average ranks")
    np.savez_compressed(
        RESULT_ROOT / "ranking_matrices.npz",
        state_ids=analysis.frozen_models.state_ids,
        R_RR=analysis.ranking_matrices["R_RR"],
        R_CX=analysis.ranking_matrices["R_CX"],
        R_CY=analysis.ranking_matrices["R_CY"],
        R_CR=analysis.ranking_matrices["R_CR"],
    )

    print("Step 5/7: write Spearman tables and validation")
    analysis.spearman_by_state.to_csv(
        RESULT_ROOT / "spearman_by_state.csv",
        index=False,
        encoding="utf-8-sig",
    )
    analysis.spearman_summary.to_csv(
        RESULT_ROOT / "spearman_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    analysis.validation["timestamp_utc"] = datetime.now(UTC).isoformat()
    _write_json(RESULT_ROOT / "validation.json", analysis.validation)

    print("Step 6/7: generate exactly two Spearman figures")
    plot_spearman_boxplot(
        analysis.spearman_by_state,
        RESULT_ROOT / "spearman_boxplot_three_luts.png",
    )
    plot_spearman_by_state(
        analysis.spearman_by_state,
        RESULT_ROOT / "spearman_by_state_three_luts.png",
    )

    print("Step 7/7: append task log and handoff")
    summary_lines = [
        "只读取冻结Ridge final theta；未重新训练OLS/Ridge/MP，lambda=1e-8。",
        "正式Query=Y-C-1 Ridge公共B响应；LUT=X-A Ridge公共B响应、Y-A-1 Ridge公共B响应、Real-B。",
        f"state_count=425，candidate_count_per_query=425，common_B_consistent={fingerprints.common_B_consistent}，"
        f"max_probe_difference={fingerprints.common_B_max_abs_difference}。",
        f"D/R矩阵均为425x425；Spearman counts="
        f"{analysis.validation['spearman_X_count']}/"
        f"{analysis.validation['spearman_Y_count']}/"
        f"{analysis.validation['spearman_real_count']}，全部finite。",
        f"结果目录：{RESULT_ROOT}",
        "两张正式图：spearman_boxplot_three_luts.png、spearman_by_state_three_luts.png。",
        "Real-B→Real-B作为唯一reference ranking；self-match保留；CNMSE越负越相似，排名升序。",
    ]
    _append_log("\n".join(summary_lines))
    _append_handoff(
        f"完成425状态行为指纹排序一致性分析；冻结lambda=1e-8，生成四个距离矩阵、四个排名矩阵、"
        f"三类Spearman表和两张图，结果位于{RESULT_ROOT}。"
    )
    print("Analysis completed.")
    print(f"Results: {RESULT_ROOT}")


if __name__ == "__main__":
    main()
