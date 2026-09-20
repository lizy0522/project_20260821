# ruff: noqa: E501

"""Prepare the current per-state best-metric table for Excel export.

The four behavior columns intentionally take independent minima from the
persisted Aend/C2 candidate logs.  This is a source-data preparation step;
the workbook itself is authored by the bundled artifact-tool JavaScript
builder.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from data_management.shared import build_state_table  # noqa: E402
from retrieval_oriented_model_selection.shared.scenario2_retrieval_oriented_model_scan import (  # noqa: E402
    load_raw_scalar_metrics,
)

STATE_COUNT = 425
MODEL_ROOT = (
    PROJECT_ROOT / "results" / "behavior_modeling" / "scenario_2" /"scenario_2_statewise_adaptive_Aend_C2_5B"
)
OUTPUT_JSON = MODEL_ROOT / "statewise_adaptive_best_metrics_excel_source.json"
AEND_LOG = MODEL_ROOT / "candidate_search_Aend.csv.gz"
C2_LOG = MODEL_ROOT / "candidate_search_C2.csv.gz"
CHECKPOINT = MODEL_ROOT / "search_progress.json"
COLUMNS = [
    "状态序号",
    "负载配置",
    "NMSE(NMSE_WITHOUTDPD)",
    "上下边带ACPR平均值withoutdpd",
    "A段建模精度",
    "AB段泛化精度",
    "C段建模精度",
    "CB段泛化精度",
]


def _load_log(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"missing persisted candidate log: {path}")
    frame = pd.read_csv(path)
    required = {
        "state_id",
        "candidate_id",
        "native_train_NMSE_dB",
        "native_B_NMSE_dB",
        "valid",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"candidate log {path.name} missing fields: {missing}")
    frame = frame.loc[frame["valid"].astype(bool)].copy()
    for name in ("state_id", "candidate_id"):
        frame[name] = frame[name].astype(np.int64)
    for name in ("native_train_NMSE_dB", "native_B_NMSE_dB"):
        frame[name] = frame[name].astype(float)
        if not np.all(np.isfinite(frame[name].to_numpy(dtype=float))):
            raise ValueError(f"candidate log {path.name} contains non-finite {name}")
    return frame


def _best_by_state(frame: pd.DataFrame, metric: str) -> tuple[pd.Series, pd.Series]:
    if metric not in frame.columns:
        raise ValueError(f"candidate log missing metric: {metric}")
    indices = frame.groupby("state_id", sort=True)[metric].idxmin()
    best = frame.loc[indices].sort_values("state_id").reset_index(drop=True)
    if best.shape[0] != STATE_COUNT or not np.array_equal(
        best["state_id"].to_numpy(dtype=np.int64), np.arange(STATE_COUNT)
    ):
        raise RuntimeError(f"best {metric} table does not cover all 425 states")
    return best[metric].astype(float), best["candidate_id"].astype(np.int64)


def _format_load_config(row: Any) -> str:
    def mismatch(value: Any) -> str:
        number = int(round(float(value)))
        return "0" if number == 0 else f"{number / 100:.2f}".rstrip("0").rstrip(".")

    return (
        f"funMng={mismatch(row.funMng)}, funAng={int(row.funAng)}°, "
        f"secMng={mismatch(row.secMng)}, secAng={int(row.secAng)}°"
    )


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def build_source_payload() -> dict[str, Any]:
    aend = _load_log(AEND_LOG)
    c2 = _load_log(C2_LOG)
    a_train, a_train_candidate = _best_by_state(aend, "native_train_NMSE_dB")
    a_b, a_b_candidate = _best_by_state(aend, "native_B_NMSE_dB")
    c_train, c_train_candidate = _best_by_state(c2, "native_train_NMSE_dB")
    c_b, c_b_candidate = _best_by_state(c2, "native_B_NMSE_dB")

    state_frame = pd.DataFrame(build_state_table()).sort_values("state_id").reset_index(drop=True)
    if state_frame.shape[0] != STATE_COUNT:
        raise RuntimeError("state table must contain 425 rows")
    raw_scalars = (
        load_raw_scalar_metrics(range(STATE_COUNT)).sort_values("state_id").reset_index(drop=True)
    )
    if raw_scalars.shape[0] != STATE_COUNT:
        raise RuntimeError("raw scalar table must contain 425 rows")
    rows: list[list[Any]] = []
    for index, row in enumerate(state_frame.itertuples(index=False)):
        rows.append(
            [
                int(row.state_id),
                _format_load_config(row),
                float(raw_scalars.iloc[index]["nmse_withoutdpd"]),
                float(raw_scalars.iloc[index]["acpr_withoutdpd_avg_dBc"]),
                float(a_train.iloc[index]),
                float(a_b.iloc[index]),
                float(c_train.iloc[index]),
                float(c_b.iloc[index]),
            ]
        )
    if len(rows) != STATE_COUNT or [row[0] for row in rows] != list(range(STATE_COUNT)):
        raise RuntimeError("Excel source rows must be state_id 0...424")

    checkpoint: dict[str, Any] = {}
    if CHECKPOINT.is_file():
        checkpoint = json.loads(CHECKPOINT.read_text(encoding="utf-8"))
    persisted_round = int(max(aend["search_round"].max(), c2["search_round"].max()))
    payload = {
        "columns": COLUMNS,
        "rows": rows,
        "metadata": {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "state_count": STATE_COUNT,
            "source_rule": "independent per-state minima over persisted valid candidate rows",
            "aend_train_metric": "min(candidate_search_Aend.native_train_NMSE_dB)",
            "aend_B_metric": "min(candidate_search_Aend.native_B_NMSE_dB)",
            "c2_train_metric": "min(candidate_search_C2.native_train_NMSE_dB)",
            "c2_B_metric": "min(candidate_search_C2.native_B_NMSE_dB)",
            "persisted_search_round_max": persisted_round,
            "checkpoint_active_search_round": checkpoint.get("active_search_round"),
            "checkpoint_completed_target_states": checkpoint.get("completed_target_states"),
            "checkpoint_target_states": checkpoint.get("target_states"),
            "checkpoint_search_stop_reason": checkpoint.get("search_stop_reason"),
            "candidate_log_rows": {"Aend": int(aend.shape[0]), "C2": int(c2.shape[0])},
            "selection_trace_candidate_ids": {
                "Aend_train": a_train_candidate.to_list(),
                "Aend_B": a_b_candidate.to_list(),
                "C2_train": c_train_candidate.to_list(),
                "C2_B": c_b_candidate.to_list(),
            },
            "source_files": {
                "Aend_log": str(AEND_LOG),
                "C2_log": str(C2_LOG),
                "checkpoint": str(CHECKPOINT),
            },
        },
    }
    return payload


def main() -> None:
    payload = build_source_payload()
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output": str(OUTPUT_JSON),
                "rows": len(payload["rows"]),
                "columns": len(payload["columns"]),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
