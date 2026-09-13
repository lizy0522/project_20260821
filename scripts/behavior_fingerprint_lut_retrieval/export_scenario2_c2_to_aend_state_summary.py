"""Export the frozen Scenario 2 C2-to-A_end state summary to one Excel sheet."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
TASK_ROOT = SCRIPTS_ROOT / "behavior_fingerprint_lut_retrieval"
RESULT_ROOT = (
    PROJECT_ROOT / "results" / "behavior_fingerprint_lut_retrieval" / "scenario_2_C2_to_Aend"
)
OUTPUT_XLSX = RESULT_ROOT / "scenario_2_C2_to_Aend_state_summary.xlsx"
ALL_ILC_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2_all_ilc"
)
RETRIEVAL_ROOT = (
    PROJECT_ROOT / "results" / "behavior_fingerprint_lut_retrieval" / "scenario_2_C2_to_Aend"
)
OLD_RETRIEVAL_ROOT = (
    PROJECT_ROOT / "results" / "behavior_fingerprint_lut_retrieval" / "scenario_2_C3_to_A2"
)
WORK_LOG = PROJECT_ROOT / "work_logs" / "behavior_fingerprint_lut_retrieval" / "execution_log.txt"
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "codex_handoff" / "codex_handoff.txt"
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
NODE_EXE = Path(
    r"C:\Users\lizy\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"
)
NODE_PACKAGES = Path(
    r"C:\Users\lizy\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules"
)
BUILDER = TASK_ROOT / "export_scenario2_c2_to_aend_state_summary.mjs"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _read_csv(path: Path, name: str) -> pd.DataFrame:
    _require(path.is_file(), f"缺少{name}：{path}")
    return pd.read_csv(path)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _tree_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        (
            item
            for item in path.rglob("*")
            if item.is_file() and "__pycache__" not in item.parts and item.suffix.lower() != ".pyc"
        ),
        key=lambda item: str(item).lower(),
    )
    for file in files:
        content = file.read_bytes()
        digest.update(file.relative_to(path).as_posix().encode() + b"\0")
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def _raw_manifest() -> dict[str, Any]:
    raw_root = Path(r"\\?\{}".format(PROJECT_ROOT / "data" / "raw"))
    digest = hashlib.sha256()
    files = sorted(
        (item for item in raw_root.rglob("*") if item.is_file()),
        key=lambda item: str(item).lower(),
    )
    for file in files:
        digest.update(str(file.relative_to(raw_root)).replace("\\", "/").encode() + b"\0")
        digest.update(file.stat().st_size.to_bytes(8, "little"))
    return {
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "bytes": sum(file.stat().st_size for file in files),
    }


def _protection_snapshot() -> dict[str, Any]:
    existing = {
        file.name: _file_sha256(file)
        for file in sorted(RESULT_ROOT.iterdir(), key=lambda item: item.name.lower())
        if file.is_file() and file.name != OUTPUT_XLSX.name
    }
    return {
        "raw_manifest": _raw_manifest(),
        "c2_aend_existing_files": existing,
        "c3_to_a2_tree_sha256": _tree_sha256(OLD_RETRIEVAL_ROOT),
    }


def _remove_generated_inspect_sidecar(output: Path) -> None:
    sidecar = Path(f"{output}.inspect.ndjson")
    if sidecar.is_file():
        sidecar.unlink()


def _verify_protection(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    raw_before = before["raw_manifest"]
    raw_after = after["raw_manifest"]
    unchanged_files = {
        name: before["c2_aend_existing_files"].get(name)
        == after["c2_aend_existing_files"].get(name)
        for name in before["c2_aend_existing_files"]
    }
    result = {
        "raw_manifest_unchanged": raw_before == raw_after,
        "c2_aend_existing_files_unchanged": all(unchanged_files.values())
        and set(before["c2_aend_existing_files"]) == set(after["c2_aend_existing_files"]),
        "c2_aend_file_checks": unchanged_files,
        "c3_to_a2_unchanged": before["c3_to_a2_tree_sha256"] == after["c3_to_a2_tree_sha256"],
        "c3_to_a2_before": before["c3_to_a2_tree_sha256"],
        "c3_to_a2_after": after["c3_to_a2_tree_sha256"],
    }
    result["all_protected_unchanged"] = bool(
        result["raw_manifest_unchanged"]
        and result["c2_aend_existing_files_unchanged"]
        and result["c3_to_a2_unchanged"]
    )
    return result


def _validate_state_ids(values: pd.Series, name: str) -> np.ndarray:
    array = values.to_numpy(dtype=np.int64)
    _require(array.shape == (425,), f"{name}必须有425行")
    _require(np.array_equal(array, np.arange(425)), f"{name}必须严格为0...424")
    return array


def _physical_mismatch(code: Any) -> str:
    value = int(round(float(code)))
    if value == 0:
        return "0"
    text = f"{value / 100:.2f}".rstrip("0").rstrip(".")
    return text


def _load_raw_scalars() -> pd.DataFrame:
    # Only the three scalar variables are loaded; no IQ waveform is read here.
    import sys

    if str(SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_ROOT))
    from data_manager import load_variable_by_id

    rows: list[dict[str, float | int]] = []
    for state_id in range(425):
        values: dict[str, float | int] = {"state_id": state_id}
        for variable in (
            "nmse_withoutdpd",
            "acpr_low_withoutdpd",
            "acpr_upper_withoutdpd",
        ):
            array = np.asarray(load_variable_by_id(state_id, variable))
            _require(array.size == 1, f"state_id={state_id} {variable}不是标量")
            scalar = float(array.reshape(-1)[0])
            _require(np.isfinite(scalar), f"state_id={state_id} {variable}非finite")
            values[variable] = scalar
        rows.append(values)
        if (state_id + 1) % 100 == 0 or state_id == 424:
            print(f"raw scalar metrics: processed {state_id + 1} / 425")
    return pd.DataFrame(rows)


def _load_model_metrics(stage_map: pd.DataFrame) -> pd.DataFrame:
    metrics = _read_csv(ALL_ILC_ROOT / "all_ilc_model_metrics.csv", "all-ILC模型指标")
    required = {
        "state_id",
        "actual_ilc_n",
        "model_role",
        "train_NMSE_dB",
        "B_generalization_NMSE_dB",
    }
    _require(required.issubset(metrics.columns), "all_ilc_model_metrics缺少冻结指标列")
    keys = ["state_id", "actual_ilc_n", "model_role"]
    _require(not metrics.duplicated(keys).any(), "all_ilc_model_metrics存在重复模型键")
    stage_keys = stage_map.loc[:, ["state_id", "N_ilc_available"]].copy()
    stage_keys["actual_ilc_n"] = stage_keys["N_ilc_available"]
    stage_keys["model_role"] = "Y-A"
    aend = stage_keys.merge(
        metrics.loc[:, keys + ["train_NMSE_dB", "B_generalization_NMSE_dB"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    stage_keys_c = stage_map.loc[:, ["state_id"]].copy()
    stage_keys_c["actual_ilc_n"] = 2
    stage_keys_c["model_role"] = "Y-C"
    c2 = stage_keys_c.merge(
        metrics.loc[:, keys + ["train_NMSE_dB", "B_generalization_NMSE_dB"]],
        on=keys,
        how="left",
        validate="one_to_one",
    )
    _require(aend["train_NMSE_dB"].notna().all(), "A_end train NMSE存在缺失")
    _require(c2["train_NMSE_dB"].notna().all(), "C2 train NMSE存在缺失")
    for frame, label in ((aend, "A_end"), (c2, "C2")):
        for column in ("train_NMSE_dB", "B_generalization_NMSE_dB"):
            _require(
                np.isfinite(frame[column].to_numpy(dtype=float)).all(),
                f"{label} {column}非finite",
            )
    aend = aend.rename(
        columns={
            "train_NMSE_dB": "Y_Aend_train_NMSE_dB",
            "B_generalization_NMSE_dB": "Y_Aend_B_NMSE_dB",
        }
    )
    c2 = c2.rename(
        columns={
            "train_NMSE_dB": "Y_C2_train_NMSE_dB",
            "B_generalization_NMSE_dB": "Y_C2_B_NMSE_dB",
        }
    )
    return aend.loc[:, ["state_id", "Y_Aend_train_NMSE_dB", "Y_Aend_B_NMSE_dB"]].merge(
        c2.loc[:, ["state_id", "Y_C2_train_NMSE_dB", "Y_C2_B_NMSE_dB"]],
        on="state_id",
        how="inner",
        validate="one_to_one",
    )


def build_state_summary_frame() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the exactly 425×10 source table without authoring the workbook."""

    stage_map = _read_csv(RETRIEVAL_ROOT / "a_end_stage_map.csv", "A_end stage map")
    _validate_state_ids(stage_map["state_id"], "a_end_stage_map.state_id")
    _require(stage_map["C2_available"].astype(bool).all(), "存在C2不可用状态")
    _require(
        stage_map["ilc_A_end"].value_counts().sort_index().to_dict()
        == {2: 1, 3: 416, 4: 7, 5: 1},
        "A_end stage distribution不符合冻结结果",
    )
    retrieval = _read_csv(RETRIEVAL_ROOT / "retrieval_results.csv", "正式检索结果")
    diagnostics = _read_csv(RETRIEVAL_ROOT / "retrieved_real_B_cnmse.csv", "Real-B检索结果")
    _validate_state_ids(retrieval["State_n_R"], "retrieval_results.State_n_R")
    _validate_state_ids(diagnostics["State_n_R"], "retrieved_real_B_cnmse.State_n_R")
    _require(
        np.array_equal(
            retrieval["State_n_R"].to_numpy(dtype=np.int64),
            diagnostics["State_n_R"].to_numpy(dtype=np.int64),
        ),
        "检索表和Real-B表state_id不一致",
    )
    _require(
        np.array_equal(retrieval["State_n_Q"], diagnostics["State_n_Q"]),
        "检索表和Real-B表state_id_Q不一致",
    )
    _require(
        np.array_equal(retrieval["exact_hit"], diagnostics["exact_hit"]),
        "检索表和Real-B表exact_hit不一致",
    )
    acpr = _read_csv(
        PROJECT_ROOT
        / "results"
        / "pa_performance_observation"
        / "scenario_2"
        / "ilc_acpr_average_by_state.csv",
        "ACPR汇总",
    )
    _validate_state_ids(acpr["state_id"], "acpr_average.state_id")
    for column in ("funMng", "funAng", "secMng", "secAng"):
        _require(np.array_equal(stage_map[column], acpr[column]), f"{column} metadata mismatch")
    raw = _load_raw_scalars()
    models = _load_model_metrics(stage_map)
    merged = stage_map.merge(
        raw,
        left_on="state_id",
        right_on="state_id",
        how="inner",
        validate="one_to_one",
    )
    merged = merged.merge(models, on="state_id", how="inner", validate="one_to_one")
    merged = merged.merge(
        retrieval.loc[:, ["State_n_R", "State_n_Q", "exact_hit"]],
        left_on="state_id",
        right_on="State_n_R",
        how="inner",
        validate="one_to_one",
    )
    merged = merged.merge(
        diagnostics.loc[:, ["State_n_R", "retrieved_real_B_CNMSE_dB"]],
        left_on="state_id",
        right_on="State_n_R",
        how="inner",
        validate="one_to_one",
        suffixes=("", "_diag"),
    )
    _require(merged.shape[0] == 425, f"合并后状态数错误：{merged.shape[0]}")
    _require(np.array_equal(merged["state_id"], np.arange(425)), "合并后state_id不完整")
    avg_acpr = (merged["acpr_low_withoutdpd"] + merged["acpr_upper_withoutdpd"]) / 2.0
    sample_ids = [0, 1, 100, 340, 424]
    _require(
        np.allclose(
            avg_acpr.iloc[sample_ids].to_numpy(),
            (
                merged.loc[sample_ids, "acpr_low_withoutdpd"]
                + merged.loc[sample_ids, "acpr_upper_withoutdpd"]
            ).to_numpy()
            / 2.0,
            rtol=0,
            atol=1e-15,
        ),
        "ACPR算术平均复核失败",
    )
    load_config = [
        (
            f"funMng={_physical_mismatch(row.funMng)}, funAng={int(row.funAng)}°"
            f", secMng={_physical_mismatch(row.secMng)}, secAng={int(row.secAng)}°"
        )
        for row in merged.itertuples(index=False)
    ]
    retrieved_values: list[float | str] = []
    for row in merged.itertuples(index=False):
        if bool(row.exact_hit):
            retrieved_values.append("-Inf")
        else:
            value = float(row.retrieved_real_B_CNMSE_dB)
            _require(np.isfinite(value), f"state_id={row.state_id} non-exact Real-B CNMSE非finite")
            retrieved_values.append(value)
    frame = pd.DataFrame(
        {
            "state_id_R": merged["state_id"].to_numpy(dtype=np.int64),
            "load_config": load_config,
            "nmse_withoutdpd_dB": merged["nmse_withoutdpd"].to_numpy(dtype=float),
            "acpr_withoutdpd_avg_dBc": avg_acpr.to_numpy(dtype=float),
            "Y_Aend_train_NMSE_dB": merged["Y_Aend_train_NMSE_dB"].to_numpy(dtype=float),
            "Y_Aend_B_NMSE_dB": merged["Y_Aend_B_NMSE_dB"].to_numpy(dtype=float),
            "Y_C2_train_NMSE_dB": merged["Y_C2_train_NMSE_dB"].to_numpy(dtype=float),
            "Y_C2_B_NMSE_dB": merged["Y_C2_B_NMSE_dB"].to_numpy(dtype=float),
            "state_id_Q": merged["State_n_Q"].to_numpy(dtype=np.int64),
            "retrieved_real_B_CNMSE_dB": retrieved_values,
        }
    )
    _require(frame.shape == (425, 10), f"主表shape错误：{frame.shape}")
    _require(frame.columns.tolist() == EXPECTED_COLUMNS, "主表列顺序不符合冻结契约")
    _require(frame["state_id_R"].is_unique, "state_id_R存在重复")
    for column in EXPECTED_COLUMNS[2:8]:
        _require(np.isfinite(frame[column].to_numpy(dtype=float)).all(), f"{column}存在非finite")
    _require(frame["state_id_Q"].between(0, 424).all(), "state_id_Q存在越界")
    exact_count = int((frame["retrieved_real_B_CNMSE_dB"] == "-Inf").sum())
    finite_values = pd.to_numeric(
        frame["retrieved_real_B_CNMSE_dB"].replace("-Inf", np.nan), errors="coerce"
    )
    failure_count = int((finite_values >= -40).sum())
    shareable_count = exact_count + int((finite_values < -40).sum())
    _require(exact_count == 218, f"Excel源表-Inf数量错误：{exact_count}")
    _require(failure_count == 14, f"Excel源表failure数量错误：{failure_count}")
    _require(shareable_count == 411, f"Excel源表shareable数量错误：{shareable_count}")
    state340 = frame.iloc[340]
    _require(int(state340.state_id_Q) == 217, "state 340 的state_id_Q不是217")
    _require(
        abs(float(state340.retrieved_real_B_CNMSE_dB) - -37.10367236988057) < 1e-10,
        "state 340 CNMSE错误",
    )
    metadata = {
        "columns": EXPECTED_COLUMNS,
        "row_count": 425,
        "load_config_source": str(RETRIEVAL_ROOT / "a_end_stage_map.csv"),
        "raw_scalar_source": (
            "data_manager.load_variable_by_id: nmse_withoutdpd, "
            "acpr_low_withoutdpd, acpr_upper_withoutdpd"
        ),
        "acpr_formula": "(acpr_low_withoutdpd + acpr_upper_withoutdpd) / 2",
        "model_metrics_source": str(ALL_ILC_ROOT / "all_ilc_model_metrics.csv"),
        "model_evaluation": (
            "directly reused frozen train_NMSE_dB and "
            "B_generalization_NMSE_dB; no fit"
        ),
        "a_end_rule": "state-specific last actually available ILC column",
        "a_end_distribution": {
            str(stage): int((stage_map["ilc_A_end"] == stage).sum())
            for stage in (2, 3, 4, 5)
        },
        "c2_stage": 2,
        "c2_all_available": True,
        "state_id_q_source": str(RETRIEVAL_ROOT / "retrieval_results.csv"),
        "real_b_cnmse_source": str(RETRIEVAL_ROOT / "retrieved_real_B_cnmse.csv"),
        "exact_inf_count": exact_count,
        "failure_count": failure_count,
        "dpd_shareable_count": shareable_count,
        "models_retrained": False,
        "ridge_rescanned": False,
        "raw_waveform_reprocessed": False,
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


def _run_node(source: Path | None, output: Path, preview: Path | None, verify_only: bool) -> None:
    _require(NODE_EXE.is_file(), f"缺少Bundled Node.js：{NODE_EXE}")
    _require(NODE_PACKAGES.is_dir(), f"缺少Bundled Node packages：{NODE_PACKAGES}")
    _require(BUILDER.is_file(), f"缺少artifact-tool builder：{BUILDER}")
    with tempfile.TemporaryDirectory(prefix="c2_aend_xlsx_") as temp_dir:
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
            command.extend(["--verify-only", "--output", str(output)])
        else:
            _require(source is not None, "source JSON is required for creation")
            command.extend(["--source", str(source), "--output", str(output)])
            if preview is not None:
                command.extend(["--preview", str(preview)])
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        print(completed.stdout)
        if completed.returncode != 0:
            print(completed.stderr)
            raise RuntimeError(
                f"artifact-tool builder failed with exit code {completed.returncode}"
            )


def _append_log(text: str) -> None:
    WORK_LOG.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    with WORK_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n[{timestamp}] Scenario 2 state summary Excel export\n")
        handle.write(text.rstrip() + "\n")


def _append_handoff(text: str) -> None:
    HANDOFF_LOG.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    with HANDOFF_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{timestamp} | behavior_fingerprint_lut_retrieval持续维护：Excel状态汇总\n")
        handle.write(text.rstrip() + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT_XLSX)
    args = parser.parse_args()
    if args.verify_only:
        _remove_generated_inspect_sidecar(args.output)
        _run_node(None, args.output, None, True)
        _remove_generated_inspect_sidecar(args.output)
        print(f"verified: {args.output}")
        return
    _remove_generated_inspect_sidecar(args.output)
    before = _protection_snapshot()
    frame, metadata = build_state_summary_frame()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="c2_aend_source_") as temp_dir:
        source = Path(temp_dir) / "state_summary_source.json"
        source.write_text(
            json.dumps(_source_payload(frame, metadata), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        preview = Path(temp_dir) / "state_summary_preview.png"
        _run_node(source, args.output, preview, False)
        _require(args.output.is_file() and args.output.stat().st_size > 0, "xlsx没有成功写出")
        print(f"preview (temporary): {preview}")
    _remove_generated_inspect_sidecar(args.output)
    after = _protection_snapshot()
    protection = _verify_protection(before, after)
    if not protection["all_protected_unchanged"]:
        raise RuntimeError("Excel导出期间保护数据发生变化")
    _append_log(
        "\n".join(
            [
                f"生成固定10列表格：{args.output}；sheet=state_summary；data_rows=425。",
                "主表按state_id_R严格join a_end_stage_map/retrieval_results/"
                "retrieved_real_B_cnmse、all_ilc_model_metrics和data_manager三个原始标量。",
                "A_end train/B和C2 train/B直接复用冻结all_ilc_model_metrics中的train_NMSE_dB与"
                "B_generalization_NMSE_dB；未fit、未重新alignment/gain/normalization。",
                "ACPR平均=(acpr_low_withoutdpd+acpr_upper_withoutdpd)/2；Excel中Exact Real-B CNMSE"
                f"用文本-Inf表示，数量={metadata['exact_inf_count']}；failure={metadata['failure_count']}；"
                f"DPD-shareable={metadata['dpd_shareable_count']}。",
                "保护复核："
                f"{protection['all_protected_unchanged']}；raw manifest="
                f"{before['raw_manifest']}；"
                "C2→Aend既有文件逐项未变；旧C3→A2树未变；未执行破坏性Git操作。",
            ]
        )
    )
    _append_handoff(
        f"完成Scenario 2 C2→Aend 425状态Excel汇总：单sheet state_summary、10列、425行；"
        f"Exact -Inf={metadata['exact_inf_count']}、failure={metadata['failure_count']}、"
        f"DPD-shareable={metadata['dpd_shareable_count']}；冻结模型指标和PA标量均按state_id join，"
        f"保护复核通过，输出位于{args.output}。"
    )
    print(f"created: {args.output}")


if __name__ == "__main__":
    main()
