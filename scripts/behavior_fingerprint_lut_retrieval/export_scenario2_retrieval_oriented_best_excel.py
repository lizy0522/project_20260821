"""Create and verify the 10-column Excel summary for the Best candidate."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_lut_retrieval"
    / "scenario_2_C2_to_Aend_retrieval_oriented_model_scan"
)
OUTPUT_XLSX = RESULT_ROOT / "best_candidate_state_summary.xlsx"
METADATA_PATH = RESULT_ROOT / "best_candidate_excel_metadata.json"
BUILDER = (
    SCRIPTS_ROOT
    / "behavior_fingerprint_lut_retrieval"
    / "export_scenario2_retrieval_oriented_best_excel.mjs"
)
NODE_EXE = Path(
    r"C:\Users\lizy\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"
)
NODE_PACKAGES = Path(
    r"C:\Users\lizy\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules"
)
EXPECTED_COLUMNS = [
    "state_id_R",
    "load_config",
    "nmse_withoutdpd_dB",
    "acpr_withoutdpd_avg_dBc",
    "Y_Aend_train_NMSE_dB",
    "Y_Aend_B_NMSE_dB",
    "Y_C2_train_NMSE_dB",
    "Y_C2_B_NMSE_dB",
    "state_id_Q",
    "retrieved_real_B_CNMSE_dB",
]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _load_source() -> tuple[pd.DataFrame, dict[str, Any]]:
    source_path = RESULT_ROOT / "best_candidate_state_summary.csv"
    _require(source_path.is_file(), f"缺少Best Excel源表：{source_path}")
    frame = pd.read_csv(
        source_path,
        keep_default_na=False,
        dtype={"retrieved_real_B_CNMSE_dB": "string"},
    )
    _require(frame.shape == (425, 10), f"Best Excel源表shape错误：{frame.shape}")
    _require(frame.columns.tolist() == EXPECTED_COLUMNS, "Best Excel源表列顺序错误")
    _require(
        np.array_equal(frame["state_id_R"].to_numpy(dtype=np.int64), np.arange(425)),
        "state_id_R必须严格为0...424",
    )
    for column in EXPECTED_COLUMNS[2:8]:
        values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
        _require(np.all(np.isfinite(values)), f"{column}包含非finite")
    _require(frame["state_id_Q"].astype(int).between(0, 424).all(), "state_id_Q越界")
    exact = frame["retrieved_real_B_CNMSE_dB"].astype(str).str.lower().isin({"-inf", "-infinity"})
    frame["retrieved_real_B_CNMSE_dB"] = frame["retrieved_real_B_CNMSE_dB"].astype(str)
    frame.loc[exact, "retrieved_real_B_CNMSE_dB"] = "-Inf"
    finite = pd.to_numeric(frame.loc[~exact, "retrieved_real_B_CNMSE_dB"], errors="coerce")
    _require(np.all(np.isfinite(finite.to_numpy(dtype=float))), "非Exact Real-B包含非finite")
    expected = json.loads((RESULT_ROOT / "best_retrieval_model.json").read_text(encoding="utf-8"))
    exact_count = int(exact.sum())
    failure_count = int((~exact & (finite >= -40.0)).sum())
    shareable_count = exact_count + int((finite < -40.0).sum())
    _require(exact_count == int(expected["exact_hit_count"]), "Exact数量与Best摘要不一致")
    _require(failure_count == int(expected["failure_count"]), "Failure数量与Best摘要不一致")
    _require(shareable_count == int(expected["shareable_count"]), "Shareable数量与Best摘要不一致")
    metadata = {
        "columns": EXPECTED_COLUMNS,
        "row_count": 425,
        "exact_hit_count": exact_count,
        "failure_count": failure_count,
        "shareable_count": shareable_count,
        "best_candidate_id": int(expected["candidate_id"]),
        "threshold_dB": -40.0,
        "threshold_rule": "Real-B CNMSE < -40 dB",
        "source_csv": str(source_path),
    }
    return frame, metadata


def _source_payload(frame: pd.DataFrame, metadata: dict[str, Any]) -> dict[str, Any]:
    rows: list[list[Any]] = []
    for row in frame.itertuples(index=False, name=None):
        converted: list[Any] = []
        for value in row:
            if isinstance(value, (np.integer, int)):
                converted.append(int(value))
            elif isinstance(value, (np.floating, float)):
                converted.append(float(value))
            else:
                converted.append(value)
        rows.append(converted)
    return {"columns": EXPECTED_COLUMNS, "rows": rows, "metadata": metadata}


def _run_node(
    source: Path | None,
    output: Path,
    preview: Path | None,
    *,
    verify_only: bool,
    metadata: Path | None,
) -> None:
    _require(NODE_EXE.is_file(), f"缺少Bundled Node：{NODE_EXE}")
    _require(NODE_PACKAGES.is_dir(), f"缺少Node packages：{NODE_PACKAGES}")
    _require(BUILDER.is_file(), f"缺少artifact builder：{BUILDER}")
    with tempfile.TemporaryDirectory(prefix="retrieval_best_xlsx_") as temp_dir:
        temp_root = Path(temp_dir)
        temp_builder = temp_root / BUILDER.name
        shutil.copy2(BUILDER, temp_builder)
        junction = temp_root / "node_modules"
        subprocess.run(
            ["cmd.exe", "/c", "mklink", "/J", str(junction), str(NODE_PACKAGES)],
            check=True,
            capture_output=True,
            text=True,
        )
        command = [str(NODE_EXE), str(temp_builder)]
        if verify_only:
            _require(metadata is not None, "verify-only需要metadata")
            command.extend(["--verify-only", "--output", str(output), "--metadata", str(metadata)])
        else:
            _require(source is not None, "创建Excel需要source")
            command.extend(["--source", str(source), "--output", str(output)])
        if preview is not None:
            command.extend(["--preview", str(preview)])
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        print(completed.stdout)
        if completed.returncode != 0:
            print(completed.stderr)
            raise RuntimeError(f"artifact-tool builder exit={completed.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_XLSX)
    parser.add_argument("--preview", type=Path, default=None)
    args = parser.parse_args()
    frame, metadata = _load_source()
    if args.verify_only:
        _require(METADATA_PATH.is_file(), f"缺少Excel metadata：{METADATA_PATH}")
        _run_node(None, args.output, args.preview, verify_only=True, metadata=METADATA_PATH)
        print(f"verified: {args.output}")
        return
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="retrieval_best_source_") as temp_dir:
        source = Path(temp_dir) / "best_candidate_state_summary_source.json"
        source.write_text(
            json.dumps(_source_payload(frame, metadata), ensure_ascii=False), encoding="utf-8"
        )
        _run_node(
            source,
            args.output,
            Path(temp_dir) / "best_candidate_state_summary_preview.png",
            verify_only=False,
            metadata=None,
        )
    _require(args.output.is_file() and args.output.stat().st_size > 0, "Excel没有成功写出")
    METADATA_PATH.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), **metadata}, ensure_ascii=False))


if __name__ == "__main__":
    main()
