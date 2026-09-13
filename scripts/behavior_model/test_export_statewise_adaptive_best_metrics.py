"""Contract checks for the current statewise-best Excel source table."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

MODEL_ROOT = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "behavior_model"
    / "scenario_2_statewise_adaptive_Aend_C2_5B"
)
SOURCE_PATH = MODEL_ROOT / "statewise_adaptive_best_metrics_excel_source.json"
EXPECTED_COLUMNS = [
    "状态序号",
    "负载配置",
    "NMSE(NMSE_WITHOUTDPD)",
    "上下边带ACPR平均值withoutdpd",
    "A段建模精度",
    "AB段泛化精度",
    "C段建模精度",
    "CB段泛化精度",
]


def test_statewise_best_excel_source_contract() -> None:
    payload = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    assert payload["columns"] == EXPECTED_COLUMNS
    rows = payload["rows"]
    assert len(rows) == 425
    assert [row[0] for row in rows] == list(range(425))
    assert all(len(row) == len(EXPECTED_COLUMNS) for row in rows)
    for row in rows:
        assert isinstance(row[1], str) and row[1]
        assert np.all(np.isfinite(np.asarray(row[2:], dtype=float)))


def test_statewise_best_source_records_independent_metric_rule() -> None:
    payload = json.loads(SOURCE_PATH.read_text(encoding="utf-8"))
    metadata = payload["metadata"]
    assert (
        metadata["source_rule"]
        == "independent per-state minima over persisted valid candidate rows"
    )
    assert metadata["aend_train_metric"].endswith("native_train_NMSE_dB)")
    assert metadata["aend_B_metric"].endswith("native_B_NMSE_dB)")
    assert metadata["c2_train_metric"].endswith("native_train_NMSE_dB)")
    assert metadata["c2_B_metric"].endswith("native_B_NMSE_dB)")


def main() -> None:
    tests = [
        ("statewise best Excel source contract", test_statewise_best_excel_source_contract),
        (
            "independent metric selection rule",
            test_statewise_best_source_records_independent_metric_rule,
        ),
    ]
    for label, test in tests:
        test()
        print(f"{label}: PASS")


if __name__ == "__main__":
    main()
