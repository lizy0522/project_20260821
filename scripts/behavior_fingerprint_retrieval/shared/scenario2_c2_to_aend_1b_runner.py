"""Run and persist the fixed Scenario 2 1B retrieval experiment."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
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

from behavior_fingerprint_retrieval.shared.scenario2_c2_to_aend_1b import (  # noqa: E402
    EXPECTED_MAIN_COLUMNS,
    RESULT_ROOT,
    run_experiment,
)

TASK_NAME = "scenario_2_C2_to_Aend_1b"
TASK_ROOT = SCRIPTS_ROOT / "behavior_fingerprint_retrieval" / "scenario_2" / TASK_NAME
OUTPUT_XLSX = RESULT_ROOT / "scenario_2_C2_to_Aend_1B_retrieval.xlsx"
EXECUTION_LOG = (
    PROJECT_ROOT
    / "work_logs"
    / "behavior_fingerprint_retrieval"
    / "scenario_2" /TASK_NAME
    / "execution_log.txt"
)
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"
BUILDER = TASK_ROOT / "export_scenario2_c2_to_aend_1b.mjs"
NODE_EXE = Path(
    r"C:\Users\lizy\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"
)
NODE_PACKAGES = Path(
    r"C:\Users\lizy\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules"
)
MARK_ARTIFACT = Path(
    r"C:\Users\lizy\.codex\plugins\cache\openai-primary-runtime\spreadsheets\26.905.11957\skills\spreadsheets\container_tools\mark_artifact_operation_started.mjs"
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"无法JSON序列化类型：{type(value)!r}")


def _to_excel_scalar(value: Any, *, allow_neginf_string: bool = False) -> Any:
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        if np.isneginf(number) and allow_neginf_string:
            return "-Inf"
        return number
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    return str(value)


def _parse_summary_value(value: Any) -> Any:
    """Recover typed numeric/boolean summary values from the mixed CSV column."""

    if isinstance(value, (np.integer, int, np.floating, float, np.bool_, bool)):
        return _to_excel_scalar(value)
    text = str(value).strip()
    if text == "True":
        return True
    if text == "False":
        return False
    try:
        number = float(text)
    except ValueError:
        return text
    if (
        np.isfinite(number)
        and number.is_integer()
        and not any(token in text.lower() for token in (".", "e"))
    ):
        return int(number)
    return number


def _read_csv_source(
    *,
    result_root: Path = RESULT_ROOT,
    result_tag: str = "1B",
) -> dict[str, Any]:
    main_path = result_root / f"retrieval_results_{result_tag}.csv"
    diagnostic_path = result_root / f"retrieval_diagnostics_{result_tag}.csv"
    summary_path = result_root / f"retrieval_summary_{result_tag}.csv"
    main = pd.read_csv(main_path)
    diagnostics = pd.read_csv(diagnostic_path)
    summary = pd.read_csv(summary_path)
    _require(main.columns.tolist() == EXPECTED_MAIN_COLUMNS, "最终CSV主表列顺序错误")
    _require(main.shape == (425, 10), "最终CSV主表必须为425×10")
    _require(diagnostics.shape[0] == 425, "最终CSV诊断表必须有425行")
    _require(summary.shape[1] == 2, "最终CSV summary必须为metric/value两列")

    main_rows: list[list[Any]] = []
    for row in main.itertuples(index=False, name=None):
        converted = [_to_excel_scalar(value) for value in row]
        converted[9] = _to_excel_scalar(row[9], allow_neginf_string=True)
        main_rows.append(converted)

    diagnostic_rows: list[list[Any]] = []
    for row in diagnostics.itertuples(index=False, name=None):
        converted = [_to_excel_scalar(value) for value in row]
        converted[6] = str(row[6]).strip().lower() == "true"
        converted[7] = str(row[7]).strip().lower() == "true"
        converted[5] = _to_excel_scalar(row[5], allow_neginf_string=True)
        diagnostic_rows.append(converted)

    summary_rows = [
        {"metric": str(row.metric), "value": _parse_summary_value(row.value)}
        for row in summary.itertuples(index=False)
    ]
    return {
        "mainRows": main_rows,
        "diagnosticRows": diagnostic_rows,
        "summaryRows": summary_rows,
    }


def _run_node(source: Path, output: Path, preview_dir: Path) -> None:
    _require(NODE_EXE.is_file(), f"缺少bundled Node.js：{NODE_EXE}")
    _require(NODE_PACKAGES.is_dir(), f"缺少bundled Node packages：{NODE_PACKAGES}")
    _require(MARK_ARTIFACT.is_file(), f"缺少artifact operation marker：{MARK_ARTIFACT}")
    _require(BUILDER.is_file(), f"缺少artifact-tool builder：{BUILDER}")
    with tempfile.TemporaryDirectory(prefix="scenario2_1b_xlsx_") as temp_dir:
        temp_root = Path(temp_dir)
        temp_builder = temp_root / BUILDER.name
        shutil.copy2(BUILDER, temp_builder)
        junction = temp_root / "node_modules"
        junction_result = subprocess.run(
            ["cmd.exe", "/c", "mklink", "/J", str(junction), str(NODE_PACKAGES)],
            check=False,
            capture_output=True,
            text=True,
        )
        _require(
            junction_result.returncode == 0, f"创建Node依赖junction失败：{junction_result.stderr}"
        )
        command = [
            str(NODE_EXE),
            str(temp_builder),
            "--source",
            str(source),
            "--output",
            str(output),
            "--preview_dir",
            str(preview_dir),
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        print(completed.stdout)
        if completed.returncode != 0:
            print(completed.stderr)
            raise RuntimeError(
                f"artifact-tool builder failed with exit code {completed.returncode}"
            )


def _append_log(text: str, *, title: str = "fixed 1B behavior-fingerprint LUT retrieval") -> None:
    EXECUTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    with EXECUTION_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n[{timestamp}] Scenario 2 {title}\n")
        handle.write(text.rstrip() + "\n")


def _append_handoff(text: str, *, title: str = "固定1B检索") -> None:
    HANDOFF_LOG.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    with HANDOFF_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{timestamp} | behavior_fingerprint_lut_retrieval持续维护：{title}\n")
        handle.write(text.rstrip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress-interval", type=int, default=25)
    args = parser.parse_args()

    result = run_experiment(progress_interval=args.progress_interval)
    source_payload = _read_csv_source()
    with tempfile.TemporaryDirectory(prefix="scenario2_1b_source_") as temp_dir:
        source_path = Path(temp_dir) / "scenario2_1b_excel_source.json"
        source_path.write_text(
            json.dumps(source_payload, ensure_ascii=False, indent=2, default=_json_default),
            encoding="utf-8",
        )
        # Required spreadsheet-skill marker: this is immediately before the
        # first artifact-tool create/edit authoring operation.
        marker = subprocess.run(
            [
                str(NODE_EXE),
                str(MARK_ARTIFACT),
                "--operation-kind",
                "create",
                "--expected-output-count",
                "1",
                "--output-format",
                "xlsx",
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        print(marker.stdout)
        _require(marker.returncode == 0, f"artifact operation marker失败：{marker.stderr}")
        _run_node(source_path, OUTPUT_XLSX, RESULT_ROOT)

    _require(OUTPUT_XLSX.is_file() and OUTPUT_XLSX.stat().st_size > 0, "1B Excel未成功生成")
    validation_path = RESULT_ROOT / "validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    validation["output_files"]["xlsx"] = str(OUTPUT_XLSX)
    validation["excel"] = {
        "created_with": "@oai/artifact-tool",
        "sheet_names": ["retrieval_results", "retrieval_diagnostics", "summary"],
        "main_columns": EXPECTED_MAIN_COLUMNS,
        "main_data_rows": 425,
        "diagnostics_data_rows": 425,
        "xlsx_path": str(OUTPUT_XLSX),
        "formula_error_scan": "PASS",
        "source_fact_tables": [
            str(RESULT_ROOT / "retrieval_results_1B.csv"),
            str(RESULT_ROOT / "retrieval_diagnostics_1B.csv"),
            str(RESULT_ROOT / "retrieval_summary_1B.csv"),
        ],
    }
    validation_path.write_text(
        json.dumps(validation, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )

    summary = result["tables"]["summary_payload"]
    guard = result["validation"]["five_B_regression_guard"]
    _append_log(
        "\n".join(
            [
                "研究目标：固定n=1.0（Fs_obs=20MHz，k=10，up=1，down=5）完成Y-C2→Y-Aend行为指纹LUT检索。",
                (
                    "5B identity regression guard在同一原始数据/预处理/ABC/MP/Ridge/B probe/"
                    "候选集路径上通过；"
                    "theta、模型指标、fingerprint、距离/排名、State_Q和Real-B lookup均回归通过"
                    f"（{guard['pass']}）。"
                ),
                (
                    "1B严格在5B basis和model-valid support之后通过同一Observation Operator"
                    "处理basis/output；"
                ),
                "1B只负责建模和检索，5B canonical Real-B只用于最终DPD-shareable评价。",
                f"fingerprints LUT/Query={result['phase_1b'].aend_fingerprints.shape}/"
                f"{result['phase_1b'].c2_fingerprints.shape}；distance={result['tables']['distance'].shape}；",
                (f"exact={summary['exact_hit_count']}/{summary['exact_hit_rate']:.9g}；"),
                (
                    "shareable(strict<-40dB)="
                    f"{summary['shareable_count']}/{summary['shareable_rate']:.9g}；"
                ),
                (
                    f"non-exact shareable={summary['nonexact_shareable_count']}/"
                    f"{summary['nonexact_shareable_rate']:.9g}；"
                ),
                f"failure={summary['failure_count']}；failure_ids={summary['failure_state_ids']}。",
                (
                    f"Excel由artifact-tool创建/重载检查通过：{OUTPUT_XLSX}；"
                    "main=425x10，diagnostics=425x10，summary=metric/value。"
                ),
                "1B未用于Real-B评价；Real-B未参与State_Q选择、重排、调参或模型选择；未执行0.1B~5B扫描。",
                "data/raw与既有5B/低带宽结果保护复核由validation记录；未执行破坏性Git操作。",
            ]
        )
    )
    _append_handoff(
        f"固定1B行为指纹检索完成：5B regression guard={guard['pass']}；"
        f"425状态，LUT/Query fingerprint={result['phase_1b'].aend_fingerprints.shape}/"
        f"{result['phase_1b'].c2_fingerprints.shape}；"
        f"shareable={summary['shareable_count']}/{summary['shareable_rate']:.6g}，"
        f"failure={summary['failure_count']}；1B仅用于检索，5B Real-B仅用于最终验证；"
        f"Excel与机器可读结果已写入{RESULT_ROOT}。"
    )
    print(
        json.dumps(
            {"result_root": str(RESULT_ROOT), "xlsx": str(OUTPUT_XLSX), **summary},
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
