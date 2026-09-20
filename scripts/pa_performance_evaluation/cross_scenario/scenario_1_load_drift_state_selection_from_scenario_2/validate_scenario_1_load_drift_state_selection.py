"""Validate Scenario 1 load-drift state selection outputs."""

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

from pa_performance_evaluation.cross_scenario.scenario_1_load_drift_state_selection_from_scenario_2.run_scenario_1_load_drift_state_selection import (  # noqa: E402
    EXPECTED_RAW,
    PHASES,
    RESULT_ROOT,
    TASK_NAME,
    _raw_manifest,
)

EXPECTED_RECOMMENDED_PHASES = [0, 90, 180]
EXPECTED_RECOMMENDED_STATES = [0, 17, 51, 85, 153, 187, 221]


def _check(condition: bool, failures: list[str], name: str) -> bool:
    if not condition:
        failures.append(name)
    return condition


def _task_import_violations() -> list[str]:
    violations: list[str] = []
    task_root = Path(__file__).resolve().parent
    package = "pa_performance_evaluation.cross_scenario.scenario_1_load_drift_state_selection_from_scenario_2"
    for path in task_root.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            module = node.module if isinstance(node, ast.ImportFrom) and node.module else ",".join(alias.name for alias in node.names) if isinstance(node, ast.Import) else ""
            if module.startswith("pa_performance_evaluation.scenario_") and not module.startswith(package):
                violations.append(f"{path.name}: {module}")
            if ".scenario_" in module and not module.startswith(package) and not module.startswith("pa_performance_evaluation.shared"):
                violations.append(f"{path.name}: {module}")
    return violations


def validate() -> dict[str, object]:
    failures: list[str] = []
    required = ["01_candidate_summary.csv", "02_phase_comparison.csv", "03_phase_triplet_evaluation.csv", "04_recommended_7.csv", "05_pairwise_B_CNMSE.csv", "05_pairwise_B_CNMSE.npy", "06_selection_summary.json", "07_final_result_summary.txt", "09_checkpoint.json", "10_xlsx_artifact_validation.json", "scenario_1_load_drift_candidate_selection.xlsx"]
    for name in required:
        _check((RESULT_ROOT / name).is_file() and (RESULT_ROOT / name).stat().st_size > 0, failures, f"exists:{name}")
    if failures:
        return {"task": TASK_NAME, "pass": False, "checks": {}, "failures": failures}
    candidate = pd.read_csv(RESULT_ROOT / "01_candidate_summary.csv")
    phase = pd.read_csv(RESULT_ROOT / "02_phase_comparison.csv")
    triplets = pd.read_csv(RESULT_ROOT / "03_phase_triplet_evaluation.csv")
    recommended = pd.read_csv(RESULT_ROOT / "04_recommended_7.csv")
    pairwise = np.asarray(np.load(RESULT_ROOT / "05_pairwise_B_CNMSE.npy", allow_pickle=False), dtype=float)
    summary = json.loads((RESULT_ROOT / "06_selection_summary.json").read_text(encoding="utf-8"))
    checkpoint = json.loads((RESULT_ROOT / "09_checkpoint.json").read_text(encoding="utf-8"))
    artifact = json.loads((RESULT_ROOT / "10_xlsx_artifact_validation.json").read_text(encoding="utf-8"))
    checks: dict[str, object] = {}
    checks["candidate_count"] = _check(len(candidate) == 17, failures, "candidate_count")
    checks["matched_count"] = _check(int((candidate["funMng"] == 0).sum()) == 1, failures, "matched_count")
    checks["funMng_01_count"] = _check(int((candidate["funMng"] == 10).sum()) == 8, failures, "funMng_01_count")
    checks["funMng_02_count"] = _check(int((candidate["funMng"] == 20).sum()) == 8, failures, "funMng_02_count")
    checks["sec_zero"] = _check(bool((candidate["secMng"] == 0).all() and (candidate["secAng_deg"] == 0).all()), failures, "sec_zero")
    checks["phase_set"] = _check(sorted(candidate.loc[candidate["funMng"] != 0, "funAng_deg"].unique().tolist()) == list(PHASES), failures, "phase_set")
    checks["state_mapping"] = _check(candidate["State"].astype(int).tolist() == [0, 17, 34, 51, 68, 85, 102, 119, 136, 153, 170, 187, 204, 221, 238, 255, 272], failures, "state_mapping")
    checks["pairwise_shape"] = _check(pairwise.shape == (17, 17), failures, "pairwise_shape")
    checks["pairwise_diagonal"] = _check(bool(np.all(np.isneginf(np.diag(pairwise)))), failures, "pairwise_diagonal")
    checks["phase_comparison_rows"] = _check(len(phase) == 8, failures, "phase_comparison_rows")
    checks["phase_triplets"] = _check(len(triplets) == 56, failures, "phase_triplets")
    checks["recommended_rows"] = _check(len(recommended) == 7, failures, "recommended_rows")
    checks["recommended_structure"] = _check(int((recommended["funMng"] == 0).sum()) == 1 and int((recommended["funMng"] == 10).sum()) == 3 and int((recommended["funMng"] == 20).sum()) == 3, failures, "recommended_structure")
    recommended_phases_01 = sorted(recommended.loc[recommended["funMng"] == 10, "funAng"].astype(int).tolist())
    recommended_phases_02 = sorted(recommended.loc[recommended["funMng"] == 20, "funAng"].astype(int).tolist())
    checks["same_phase_sets"] = _check(recommended_phases_01 == recommended_phases_02, failures, "same_phase_sets")
    checks["primary_recommendation"] = _check(summary.get("primary_recommended_phase_triplet") == EXPECTED_RECOMMENDED_PHASES and summary.get("primary_recommended_7_states") == EXPECTED_RECOMMENDED_STATES, failures, "primary_recommendation")
    checks["all_selected_own_dpd_recovered"] = _check(bool(recommended["own_DPD_target_reached"].all()), failures, "all_selected_own_dpd_recovered")
    checks["state_unique"] = _check(recommended["Scenario2_State"].is_unique, failures, "state_unique")
    checks["raw_unchanged"] = _check(_raw_manifest() == EXPECTED_RAW and summary.get("raw_unchanged") is True, failures, "raw_unchanged")
    checks["no_task_imports"] = _check(not _task_import_violations(), failures, "no_task_imports")
    checks["artifact_tool"] = _check(artifact.get("pass") is True and artifact.get("sheet_names") == ["candidate_summary", "pairwise_B_CNMSE", "phase_comparison", "phase_triplet_evaluation", "recommended_7"], failures, "artifact_tool")
    checks["artifact_row_counts"] = _check(artifact.get("sheet_row_counts") == {"candidate_summary": 17, "pairwise_B_CNMSE": 17, "phase_comparison": 8, "phase_triplet_evaluation": 56, "recommended_7": 7}, failures, "artifact_row_counts")
    checks["checkpoint_numerical"] = _check(checkpoint.get("phase") in {"numerical_outputs_complete", "completed"} and (checkpoint.get("phase") != "completed" or checkpoint.get("status") == "SUCCESS"), failures, "checkpoint_numerical")
    report = {"task": TASK_NAME, "pass": not failures, "checks": checks, "failures": failures, "recommended_phases": recommended_phases_01, "recommended_states": recommended["Scenario2_State"].astype(int).tolist()}
    (RESULT_ROOT / "08_validation_checks.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not failures:
        summary["status"] = "SUCCESS"
        summary["validation_status"] = "PASS"
        (RESULT_ROOT / "06_selection_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        checkpoint["phase"] = "completed"
        checkpoint["status"] = "SUCCESS"
        checkpoint["validation_status"] = "PASS"
        (RESULT_ROOT / "09_checkpoint.json").write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
        log_path = PROJECT_ROOT / "work_logs" / "pa_performance_evaluation" / "cross_scenario" /TASK_NAME / "execution_log.txt"
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} | VALIDATION PASS checks={len(checks)}\n")
        handoff = PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"
        with handoff.open("a", encoding="utf-8") as handle:
            handle.write(f"\n{time.strftime('%Y-%m-%d')} | 新任务：{TASK_NAME}\n完成 Scenario 1 load-drift candidate selection：17 candidates、17x17 B-CNMSE、8 phase comparisons、56 phase triplets、artifact-tool 5-sheet Excel；primary phases={recommended_phases_01}，states={recommended['Scenario2_State'].astype(int).tolist()}；raw unchanged，own-DPD recovery 与 validation 通过。\n")
    return report


def main() -> None:
    report = validate()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
