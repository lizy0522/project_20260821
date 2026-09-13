"""Build the frozen 10-column Excel summary for the equal-ABC ablation."""

from __future__ import annotations

import argparse
import hashlib
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
    / "scenario_2_C2_to_Aend_equal_ABC"
)
OUTPUT_XLSX = RESULT_ROOT / "scenario_2_C2_to_Aend_equal_ABC_state_summary.xlsx"
BUILDER = (
    SCRIPTS_ROOT
    / "behavior_fingerprint_lut_retrieval"
    / "export_scenario2_c2_to_aend_equal_ABC.mjs"
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


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_frame(path: Path, name: str) -> pd.DataFrame:
    _require(path.is_file(), f"缺少{name}：{path}")
    return pd.read_csv(path)


def _physical_mismatch(code: Any) -> str:
    value = int(round(float(code)))
    if value == 0:
        return "0"
    return f"{value / 100:.2f}".rstrip("0").rstrip(".")


def _validate_state_ids(frame: pd.DataFrame, column: str) -> None:
    values = frame[column].to_numpy(dtype=np.int64)
    _require(values.shape == (425,), f"{column}必须有425行")
    _require(np.array_equal(values, np.arange(425, dtype=np.int64)), f"{column}必须严格为0...424")


def build_state_summary_frame() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Assemble the exact 425x10 source table from equal-ABC result CSVs."""

    stage = _load_frame(RESULT_ROOT / "a_end_stage_map.csv", "equal-ABC Aend stage map")
    metrics = _load_frame(RESULT_ROOT / "equal_ABC_state_model_metrics.csv", "equal-ABC模型指标")
    retrieval = _load_frame(RESULT_ROOT / "retrieval_results.csv", "equal-ABC检索结果")
    diagnostics = _load_frame(RESULT_ROOT / "retrieved_real_B_cnmse.csv", "equal-ABC Real-B结果")
    summary = _load_frame(RESULT_ROOT / "retrieval_summary.csv", "equal-ABC检索汇总")
    for frame, column in (
        (stage, "state_id"),
        (metrics, "state_id"),
        (retrieval, "State_n_R"),
        (diagnostics, "State_n_R"),
    ):
        _validate_state_ids(frame, column)
    _require(stage["C2_available"].astype(bool).all(), "存在C2不可用状态")
    _require(
        stage["ilc_A_end"].value_counts().sort_index().to_dict() == {2: 1, 3: 416, 4: 7, 5: 1},
        "Aend stage分布错误",
    )
    required_metrics = {
        "state_id",
        "funMng",
        "funAng",
        "secMng",
        "secAng",
        "nmse_withoutdpd_dB",
        "acpr_withoutdpd_avg_dBc",
        "Y_Aend_train_NMSE_dB",
        "Y_Aend_B_NMSE_dB",
        "Y_C2_train_NMSE_dB",
        "Y_C2_B_NMSE_dB",
    }
    _require(
        required_metrics.issubset(metrics.columns),
        f"模型指标缺少字段：{sorted(required_metrics - set(metrics.columns))}",
    )
    _require(retrieval.shape[0] == 425 and diagnostics.shape[0] == 425, "检索表必须各有425行")
    _require(
        np.array_equal(retrieval["State_n_R"], diagnostics["State_n_R"]), "检索与诊断state_id不一致"
    )
    _require(
        np.array_equal(retrieval["State_n_Q"], diagnostics["State_n_Q"]),
        "检索与诊断state_id_Q不一致",
    )
    _require(
        np.array_equal(retrieval["exact_hit"], diagnostics["exact_hit"]), "检索与诊断exact不一致"
    )
    merged = metrics.merge(
        retrieval.loc[:, ["State_n_R", "State_n_Q", "exact_hit"]],
        left_on="state_id",
        right_on="State_n_R",
        how="inner",
        validate="one_to_one",
    ).merge(
        diagnostics.loc[:, ["State_n_R", "retrieved_real_B_CNMSE_dB"]],
        on="State_n_R",
        how="inner",
        validate="one_to_one",
    )
    _require(merged.shape[0] == 425, "合并后状态数不是425")
    _require(
        np.array_equal(merged["state_id"].to_numpy(dtype=np.int64), np.arange(425, dtype=np.int64)),
        "合并后的state_id不完整",
    )
    load_config = [
        f"funMng={_physical_mismatch(row.funMng)}, funAng={int(row.funAng)}°, "
        f"secMng={_physical_mismatch(row.secMng)}, secAng={int(row.secAng)}°"
        for row in merged.itertuples(index=False)
    ]
    retrieved_values: list[float | str] = []
    for row in merged.itertuples(index=False):
        if bool(row.exact_hit):
            retrieved_values.append("-Inf")
        else:
            value = float(row.retrieved_real_B_CNMSE_dB)
            _require(np.isfinite(value), f"state={row.state_id} non-exact Real-B非finite")
            retrieved_values.append(value)
    frame = pd.DataFrame(
        {
            "state_id_R": merged["state_id"].to_numpy(dtype=np.int64),
            "load_config": load_config,
            "nmse_withoutdpd_dB": merged["nmse_withoutdpd_dB"].to_numpy(dtype=float),
            "acpr_withoutdpd_avg_dBc": merged["acpr_withoutdpd_avg_dBc"].to_numpy(dtype=float),
            "Y_Aend_train_NMSE_dB": merged["Y_Aend_train_NMSE_dB"].to_numpy(dtype=float),
            "Y_Aend_B_NMSE_dB": merged["Y_Aend_B_NMSE_dB"].to_numpy(dtype=float),
            "Y_C2_train_NMSE_dB": merged["Y_C2_train_NMSE_dB"].to_numpy(dtype=float),
            "Y_C2_B_NMSE_dB": merged["Y_C2_B_NMSE_dB"].to_numpy(dtype=float),
            "state_id_Q": merged["State_n_Q"].to_numpy(dtype=np.int64),
            "retrieved_real_B_CNMSE_dB": retrieved_values,
        }
    )
    _require(frame.shape == (425, 10), f"Excel源表shape错误：{frame.shape}")
    _require(frame.columns.tolist() == EXPECTED_COLUMNS, "Excel源表列顺序错误")
    _require(frame["state_id_R"].is_unique, "state_id_R不唯一")
    for column in EXPECTED_COLUMNS[2:8]:
        _require(np.isfinite(frame[column].to_numpy(dtype=float)).all(), f"{column}包含非finite")
    _require(frame["state_id_Q"].between(0, 424).all(), "state_id_Q越界")
    exact_count = int((frame["retrieved_real_B_CNMSE_dB"] == "-Inf").sum())
    numeric_retrieved = pd.to_numeric(
        frame["retrieved_real_B_CNMSE_dB"].replace("-Inf", np.nan), errors="coerce"
    )
    failure_count = int((numeric_retrieved >= -40).sum())
    shareable_count = exact_count + int((numeric_retrieved < -40).sum())
    expected = summary.iloc[0]
    _require(exact_count == int(expected["exact_hit_count"]), "Exact数量与检索汇总不一致")
    _require(failure_count == int(expected["failure_count"]), "failure数量与检索汇总不一致")
    _require(
        shareable_count == int(expected["DPD_shareable_count"]), "shareable数量与检索汇总不一致"
    )
    metadata = {
        "columns": EXPECTED_COLUMNS,
        "row_count": 425,
        "exact_hit_count": exact_count,
        "failure_count": failure_count,
        "shareable_count": shareable_count,
        "segment_definition": json.loads(
            (RESULT_ROOT / "segment_definition.json").read_text(encoding="utf-8")
        ),
        "real_B_matrix_reused": False,
        "real_B_reference": "equal-ABC canonical OFF.B.output[2:]",
        "source_metrics": str(RESULT_ROOT / "equal_ABC_state_model_metrics.csv"),
        "source_retrieval": str(RESULT_ROOT / "retrieval_results.csv"),
        "source_real_B": str(RESULT_ROOT / "retrieved_real_B_cnmse.csv"),
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


def _remove_sidecar(path: Path) -> None:
    sidecar = Path(f"{path}.inspect.ndjson")
    if sidecar.is_file():
        sidecar.unlink()


def _run_node(
    source: Path | None,
    output: Path,
    preview: Path | None,
    *,
    verify_only: bool = False,
    metadata: Path | None = None,
) -> None:
    _require(NODE_EXE.is_file(), f"缺少Bundled Node：{NODE_EXE}")
    _require(NODE_PACKAGES.is_dir(), f"缺少Node packages：{NODE_PACKAGES}")
    _require(BUILDER.is_file(), f"缺少artifact builder：{BUILDER}")
    with tempfile.TemporaryDirectory(prefix="equal_abc_xlsx_") as temp_dir:
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
            if preview is not None:
                command.extend(["--preview", str(preview)])
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
    _remove_sidecar(args.output)
    frame, metadata = build_state_summary_frame()
    if args.verify_only:
        metadata_path = RESULT_ROOT / "equal_ABC_excel_metadata.json"
        _require(metadata_path.is_file(), "缺少Excel metadata")
        _run_node(
            None,
            args.output,
            args.preview,
            verify_only=True,
            metadata=metadata_path,
        )
        _remove_sidecar(args.output)
        print(f"verified: {args.output}")
        return
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="equal_abc_source_") as temp_dir:
        temp_root = Path(temp_dir)
        source = temp_root / "equal_abc_state_summary_source.json"
        source.write_text(
            json.dumps(_source_payload(frame, metadata), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        preview = temp_root / "equal_abc_state_summary_preview.png"
        _run_node(source, args.output, preview)
        _require(args.output.is_file() and args.output.stat().st_size > 0, "Excel没有成功写出")
        print(f"preview (temporary): {preview}")
    _remove_sidecar(args.output)
    (RESULT_ROOT / "equal_ABC_excel_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), **metadata}, ensure_ascii=False))


if __name__ == "__main__":
    main()
