"""运行固定 LUT ILC 阶段比较的纯结果重组与绘图任务。"""

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

from behavior_fingerprint_ranking_consistency.plot_fixed_lut_ilc_stage_comparison import (  # noqa: E402
    build_fixed_lut_stage_figures,
)

TASK_ROOT = PROJECT_ROOT / "results" / "behavior_fingerprint_ranking_consistency"
SOURCE_ROOT = TASK_ROOT / "scenario_2_all_ilc"
SOURCE_PATH = SOURCE_ROOT / "spearman_all_ilc_long.csv"
RESULT_ROOT = SOURCE_ROOT / "fixed_lut_stage_figures"
EXECUTION_LOG = (
    PROJECT_ROOT
    / "work_logs"
    / "behavior_fingerprint_ranking_consistency"
    / "execution_log.txt"
)
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "codex_handoff" / "codex_handoff.txt"


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
    EXECUTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    with EXECUTION_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n[{timestamp}] fixed LUT ILC stage figure comparison\n")
        handle.write(text.rstrip() + "\n")


def _append_handoff(text: str) -> None:
    HANDOFF_LOG.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    with HANDOFF_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{timestamp} | behavior_fingerprint_ranking_consistency持续维护\n")
        handle.write(text.rstrip() + "\n")


def _check_reference_invariance(summary: pd.DataFrame) -> bool:
    reference = summary[summary["route"].isin(("X-A", "Real-B"))]
    grouped = reference.groupby(["C_stage", "route"])["median"]
    # The same source vector is reused for every fixed A stage; summary medians must match.
    for (c_stage, route), group in grouped:
        if route not in ("X-A", "Real-B") or c_stage not in range(1, 6):
            raise RuntimeError("固定参考路线索引错误")
        if group.size != 5:
            raise RuntimeError("固定参考路线未覆盖五个A阶段")
        if not np.allclose(group.to_numpy(dtype=float), group.iloc[0], rtol=0, atol=1e-12):
            return False
    return True


def main() -> None:
    print("Fixed LUT ILC stage comparison started.")
    print(f"Source: {SOURCE_PATH}")
    print("Only reorganize existing Spearman results; no model/CNMSE/Spearman recomputation.")
    result = build_fixed_lut_stage_figures(SOURCE_PATH, RESULT_ROOT)
    combined_summary = pd.concat(
        [result.figure1_summary, result.figure2_summary],
        ignore_index=True,
    )
    combined_summary.to_csv(
        RESULT_ROOT / "fixed_lut_stage_figure_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    figure_validation = {
        "source_path": str(SOURCE_PATH),
        "source_shape": list(pd.read_csv(SOURCE_PATH).shape),
        "global_ilc_max": result.data.nmax,
        "state_count": 425,
        "requested_ilc_C_stages": list(range(1, result.data.nmax + 1)),
        "requested_ilc_A_stages": list(range(1, result.data.nmax + 1)),
        "statewise_saturation_reused": True,
        "data_reorganized_only": True,
        "models_retrained": False,
        "cnmse_recomputed": False,
        "spearman_recomputed": False,
        "X_A_and_Real_B_fixed_across_A": _check_reference_invariance(
            result.figure1_summary
        ),
        "figure1": result.figure1_validation,
        "figure2": result.figure2_validation,
        "figure1_summary_row_count": int(result.figure1_summary.shape[0]),
        "figure2_summary_row_count": int(result.figure2_summary.shape[0]),
        "figure1_box_data_points_per_top_subplot": 15 * 425,
        "figure1_curve_points_per_bottom_subplot": 15 * 425,
        "figure2_box_data_points_per_top_subplot": 5 * 425,
        "figure2_curve_points_per_bottom_subplot": 5 * 425,
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    _write_json(RESULT_ROOT / "figure_validation.json", figure_validation)
    log_lines = [
        "本次仅读取spearman_all_ilc_long.csv及必要的n1宽表语义，未重新建模、计算CNMSE或Spearman。",
        "requested_n1解释为在线C阶段，requested_n2解释为固定离线Y-A阶段。",
        "Figure1固定A阶段下比较X-A/Y-A/Real-B；每个top subplot=15 boxes、"
        "bottom subplot=15 curves。",
        "Figure2固定A阶段下只比较Y-A；每个top subplot=5 boxes、bottom subplot=5 curves。",
        "Figure1与Figure2各自10个axes使用本图全部实际数据的共同min/max，未使用padding。",
        f"Figure1 y=[{result.figure1_validation['y_min']}, {result.figure1_validation['y_max']}]; "
        f"Figure2 y=[{result.figure2_validation['y_min']}, {result.figure2_validation['y_max']}]。",
        f"新图与summary位于：{RESULT_ROOT}；旧scenario_2_all_ilc结果未覆盖。",
    ]
    _append_log("\n".join(log_lines))
    _append_handoff(
        f"新增固定LUT阶段Figure1/2（各2×5，颜色+marker区分C阶段，线型区分Figure1路线），"
        f"共同y轴实际范围分别为[{result.figure1_validation['y_min']},"
        f"{result.figure1_validation['y_max']}]和[{result.figure2_validation['y_min']},"
        f"{result.figure2_validation['y_max']}]；旧图和旧结果保留，产物位于{RESULT_ROOT}。"
    )
    print("Figure 1 and Figure 2 completed.")
    print(
        f"Figure1 y-range: {result.figure1_validation['y_min']} .. "
        f"{result.figure1_validation['y_max']}"
    )
    print(
        f"Figure2 y-range: {result.figure2_validation['y_min']} .. "
        f"{result.figure2_validation['y_max']}"
    )
    print(f"Results: {RESULT_ROOT}")


if __name__ == "__main__":
    main()
