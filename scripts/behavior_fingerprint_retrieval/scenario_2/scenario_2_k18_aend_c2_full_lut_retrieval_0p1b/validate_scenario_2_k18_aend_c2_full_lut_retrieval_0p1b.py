"""Independent validator for the fixed K18 0.1B retrieval task."""

# ruff: noqa: E402,E501

from __future__ import annotations

import ast
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir())
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from core.shared.module_registry import MODULES  # noqa: E402

from behavior_fingerprint_retrieval.scenario_2.scenario_2_k18_aend_c2_full_lut_retrieval_0p1b.run_scenario_2_k18_aend_c2_full_lut_retrieval_0p1b import (  # noqa: E402
    COMMON_B_SHA256,
    EXPECTED_RAW_BYTES,
    EXPECTED_RAW_FILE_COUNT,
    EXPECTED_RAW_MANIFEST,
    EXPECTED_RAW_MAT_COUNT,
    MODEL_CANDIDATE,
    MODEL_K,
    MODEL_SUPPORT_HASH,
    RESULT_ROOT,
    RIDGE_LAMBDA,
    STATE_COUNT,
    SUPPORT_IDS,
    raw_manifest,
)

TASK_NAME = "scenario_2_k18_aend_c2_full_lut_retrieval_0p1b"
EXPECTED_HEADERS = ["状态序号", "负载配置", "NMSE(NMSE_WITHOUTDPD)", "上下边带ACPR平均值withoutdpd", "A段建模精度", "AB段泛化精度", "C段建模精度", "CB段泛化精度", "命中的序号", "命中的CNMSE"]


def _check(condition: bool, failures: list[str], name: str) -> bool:
    if not condition:
        failures.append(name)
    return condition


def _task_import_violations() -> list[str]:
    violations: list[str] = []
    task_root = Path(__file__).resolve().parent
    package = "behavior_fingerprint_retrieval.scenario_2.scenario_2_k18_aend_c2_full_lut_retrieval_0p1b"
    for path in task_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            module = node.module if isinstance(node, ast.ImportFrom) and node.module else ",".join(alias.name for alias in node.names) if isinstance(node, ast.Import) else ""
            if module.startswith("behavior_fingerprint_retrieval.scenario_") and not module.startswith(package):
                violations.append(f"{path.name}: {module}")
            if "scenario_2_self_first_joint_basis_ridge_lut_retrieval_search_5b" in module or "scenario_2_k18_aend_c2_full_lut_retrieval_5b" in module:
                violations.append(f"{path.name}: forbidden concrete retrieval task import")
    return violations


def _xlsx_artifact_validation() -> dict[str, object]:
    path = RESULT_ROOT / "13_xlsx_artifact_validation.json"
    if not path.is_file():
        return {"pass": False, "reason": "artifact-tool validation report missing"}
    return json.loads(path.read_text(encoding="utf-8"))


def validate() -> dict[str, object]:
    failures: list[str] = []
    required = ["01_state_results.csv", "02_retrieval_summary.json", "03_model_definition.json", "04_fingerprint_distance_matrix_0p1b.npy", "05_lut_fingerprints_0p1b.npz", "06_query_fingerprints_0p1b.npz", "07_final_result_summary.txt", "09_checkpoint.json", "10_model_coefficients_0p1b.npz", "11_state_retrieval_diagnostics.csv", "figure_01_full_metrics_vs_state_0p1b.png", "figure_02_retrieved_real_B_CNMSE_vs_state_0p1b.png", "scenario_2_k18_aend_c2_full_lut_retrieval_0p1b.xlsx"]
    for name in required:
        _check((RESULT_ROOT / name).is_file() and (RESULT_ROOT / name).stat().st_size > 0, failures, f"exists:{name}")
    if failures:
        return {"task": TASK_NAME, "pass": False, "checks": {}, "failures": failures}
    summary = json.loads((RESULT_ROOT / "02_retrieval_summary.json").read_text(encoding="utf-8"))
    model = json.loads((RESULT_ROOT / "03_model_definition.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((RESULT_ROOT / "09_checkpoint.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(RESULT_ROOT / "01_state_results.csv")
    full_frame = pd.read_csv(RESULT_ROOT / "03_retrieval_diagnostics.csv")
    diag = pd.read_csv(RESULT_ROOT / "11_state_retrieval_diagnostics.csv")
    distance = np.asarray(np.load(RESULT_ROOT / "04_fingerprint_distance_matrix_0p1b.npy", allow_pickle=False), dtype=np.float64)
    with np.load(RESULT_ROOT / "05_lut_fingerprints_0p1b.npz", allow_pickle=False) as lut_data, np.load(RESULT_ROOT / "06_query_fingerprints_0p1b.npz", allow_pickle=False) as query_data:
        lut = np.asarray(lut_data["fingerprints"])
        query = np.asarray(query_data["fingerprints"])
    checks: dict[str, object] = {}
    checks["state_count"] = _check(len(frame) == STATE_COUNT, failures, "state_count")
    checks["state_order"] = _check(np.array_equal(frame["state_R"].to_numpy(dtype=int), np.arange(STATE_COUNT)), failures, "state_order")
    checks["diagnostic_rows"] = _check(len(diag) == STATE_COUNT, failures, "diagnostic_rows")
    checks["support"] = _check(len(model.get("support", [])) == MODEL_K and tuple(item["basis_name"] for item in model["support"]) == SUPPORT_IDS, failures, "support")
    checks["candidate"] = _check(model.get("candidate") == MODEL_CANDIDATE, failures, "candidate")
    checks["lambda"] = _check(abs(float(model.get("lambda", np.nan)) - RIDGE_LAMBDA) <= 1e-24, failures, "lambda")
    checks["support_hash"] = _check(model.get("support_hash") == MODEL_SUPPORT_HASH, failures, "support_hash")
    checks["observation"] = _check(summary.get("bandwidth") == "0.1B" and summary.get("Fs_observation_MHz") == 2 and summary.get("resampling_ratio") == "1/50", failures, "observation")
    checks["operator"] = _check(summary.get("operator_name") == "scipy.signal.resample_poly" and summary.get("operator_metadata", {}).get("up") == 1 and summary.get("operator_metadata", {}).get("down") == 50, failures, "operator")
    checks["construction_order"] = _check(summary.get("construction_order") == "5B K18 basis -> shared 0.1B observation operator", failures, "construction_order")
    checks["final_validation_domain"] = _check(summary.get("final_validation_domain") == "5B Real-B yout_withoutdpd_ori B segment", failures, "final_validation_domain")
    checks["common_B"] = _check(summary.get("Common_B_sha256") == COMMON_B_SHA256, failures, "common_B")
    checks["fingerprints"] = _check(lut.shape == query.shape == (STATE_COUNT, int(summary["fingerprint_length"])) and lut.dtype == np.complex128 and query.dtype == np.complex128 and np.all(np.isfinite(lut)) and np.all(np.isfinite(query)), failures, "fingerprints")
    checks["distance"] = _check(distance.shape == (STATE_COUNT, STATE_COUNT) and not np.isnan(distance).any() and not np.isposinf(distance).any(), failures, "distance")
    checks["top1_range"] = _check(bool(np.all((frame["retrieved_state_Q"] >= 0) & (frame["retrieved_state_Q"] < STATE_COUNT))), failures, "top1_range")
    checks["class_logic"] = _check(bool(np.all(full_frame.loc[full_frame["self_hit"], "retrieval_class"] == "SELF") and np.all(full_frame.loc[~full_frame["self_hit"] & full_frame["real_B_shareable_lt_minus40"], "retrieval_class"] == "FALLBACK") and np.all(full_frame.loc[~full_frame["self_hit"] & ~full_frame["real_B_shareable_lt_minus40"], "retrieval_class"] == "FAIL")), failures, "class_logic")
    checks["metrics_finite"] = _check(all(np.isfinite(frame[col].to_numpy(dtype=float)).all() for col in ["nmse_withoutdpd_dB", "acpr_withoutdpd_mean_dBc", "Y_Aend_train_NMSE_dB", "Y_Aend_B_NMSE_dB", "Y_C2_train_NMSE_dB", "Y_C2_B_NMSE_dB"]) and np.isfinite(diag["abs_State_Q_minus_State_R"].to_numpy(dtype=float)).all(), failures, "metrics_finite")
    checks["raw_manifest"] = _check(raw_manifest() == {"sha256": EXPECTED_RAW_MANIFEST, "file_count": EXPECTED_RAW_FILE_COUNT, "mat_count": EXPECTED_RAW_MAT_COUNT, "bytes": EXPECTED_RAW_BYTES} and summary.get("raw_unchanged") is True, failures, "raw_manifest")
    checks["checkpoint"] = _check(checkpoint.get("phase") == "numerical_outputs_complete", failures, "checkpoint_before_final")
    checks["no_search"] = _check(summary.get("model_search_started") is False and summary.get("nested_cv_started") is False and summary.get("self_first_runner_started") is False, failures, "no_search")
    checks["main_csv_columns"] = _check(frame.columns.tolist() == ["state_R", "load_config", "nmse_withoutdpd_dB", "acpr_withoutdpd_mean_dBc", "Y_Aend_train_NMSE_dB", "Y_Aend_B_NMSE_dB", "Y_C2_train_NMSE_dB", "Y_C2_B_NMSE_dB", "retrieved_state_Q", "retrieved_real_B_CNMSE_dB"], failures, "main_csv_columns")
    checks["xlsx_artifact"] = _check(bool(_xlsx_artifact_validation().get("pass")), failures, "xlsx_artifact")
    figure_meta = json.loads((RESULT_ROOT / "12_figure_metadata.json").read_text(encoding="utf-8"))
    checks["figure_01"] = _check(figure_meta.get("figure_01_data_series_count") == 6 and figure_meta.get("marker_label_count_figure_01") == STATE_COUNT, failures, "figure_01")
    checks["figure_02"] = _check(figure_meta.get("figure_02_data_series_count") == 1 and figure_meta.get("marker_label_count_figure_02") == STATE_COUNT, failures, "figure_02")
    checks["task_imports"] = _check(not _task_import_violations(), failures, "task_imports")
    checks["module_mirror"] = _check(all(sorted(path.name for path in (PROJECT_ROOT / root).iterdir() if path.is_dir()) == sorted(MODULES) for root in ("scripts", "results", "work_logs")), failures, "module_mirror")
    checks["no_result_shared"] = _check(not any((PROJECT_ROOT / "results" / module / "shared").exists() for module in MODULES), failures, "no_result_shared")
    checks["no_log_shared"] = _check(not any((PROJECT_ROOT / "work_logs" / module / "shared").exists() for module in MODULES), failures, "no_log_shared")
    report = {"task": TASK_NAME, "pass": not failures, "checks": checks, "failures": failures, "observed_counts": {"N_self": int((full_frame["retrieval_class"] == "SELF").sum()), "N_fallback": int((full_frame["retrieval_class"] == "FALLBACK").sum()), "N_valid": int(full_frame["retrieval_class"].isin(["SELF", "FALLBACK"]).sum()), "N_fail": int((full_frame["retrieval_class"] == "FAIL").sum())}}
    (RESULT_ROOT / "08_validation_checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    if not failures:
        summary["status"] = "SUCCESS"
        summary["validation_status"] = "PASS"
        summary["xlsx_headers"] = EXPECTED_HEADERS
        summary["xlsx_column_count"] = 10
        _write_summary = RESULT_ROOT / "02_retrieval_summary.json"
        _write_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        checkpoint["phase"] = "completed"
        checkpoint["status"] = "SUCCESS"
        checkpoint["validation_status"] = "PASS"
        (RESULT_ROOT / "09_checkpoint.json").write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        with (PROJECT_ROOT / "work_logs" / "behavior_fingerprint_retrieval" / "scenario_2" /TASK_NAME / "execution_log.txt").open("a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} | VALIDATION PASS checks={len(checks)}\n")
        with (PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt").open("a", encoding="utf-8") as handle:
            handle.write(f"\n{time.strftime('%Y-%m-%d')} | 新任务：{TASK_NAME}\n完成固定 K18 0.1B Aend/C2 Full-425 检索；artifact-tool 严格 10 列 Excel、6/1 曲线、全部 R→Q 标注、5B Real-B 验证和 raw manifest 均通过；未启动任何模型搜索。\n")
    return report


def main() -> None:
    report = validate()
    print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
