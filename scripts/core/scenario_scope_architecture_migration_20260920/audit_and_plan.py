"""Read-only inventory, scope classification, and dry-run migration plan."""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir())
MODULES = (
    "core",
    "data_management",
    "signal_segmentation",
    "pa_performance_evaluation",
    "behavior_modeling",
    "behavior_fingerprint_retrieval",
    "retrieval_oriented_model_selection",
    "behavior_fingerprint_ranking_consistency",
    "low_bandwidth_behavior_analysis",
    "lut_clustering_compression",
)
ROUTES = ("self_hit_oriented", "dpd_shareability_oriented")
PROTECTED_MODULES = {"retrieval_oriented_model_selection"}
SCOPES = {"scenario_1", "scenario_2", "cross_scenario"}
TASK_ROOT = PROJECT_ROOT / "scripts" / "core" / "scenario_scope_architecture_migration_20260920"
LOG_ROOT = PROJECT_ROOT / "work_logs" / "core" / "scenario_scope_architecture_migration_20260920"


def _now() -> str:
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    return value


def _write(name: str, value: Any) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    (LOG_ROOT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _manifest(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    records = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: str(item).lower()):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        records.append({"relative_path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": digest})
    return records


def _tree_summary(root: Path) -> dict[str, Any]:
    return {"path": str(root), "exists": root.exists(), "file_count": sum(1 for p in root.rglob("*") if p.is_file()) if root.exists() else 0, "directory_count": sum(1 for p in root.rglob("*") if p.is_dir()) if root.exists() else 0, "logical_bytes": sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.exists() else 0}


def _formal_tasks(root_name: str, module: str) -> list[dict[str, Any]]:
    root = PROJECT_ROOT / root_name / module
    entries: list[dict[str, Any]] = []
    if module == "retrieval_oriented_model_selection":
        return entries
    if not root.exists():
        return entries
    for task_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name not in {"shared", "__pycache__"}):
        if task_dir.name in {"engineering_tasks"}:
            continue
        entries.append({"module": module, "task_name": task_dir.name, "path": str(task_dir), "file_count": sum(1 for p in task_dir.rglob("*") if p.is_file())})
    return entries


def _scope_for(module: str, task: str, script_path: Path | None) -> tuple[str | None, str, str]:
    if module in PROTECTED_MODULES:
        return None, "DEFERRED_PROTECTED_MODULE", "entire retrieval_oriented_model_selection is frozen"
    if module == "core":
        return None, "NO_MOVE_NEEDED", "engineering/core task remains at module root"
    if module == "data_management":
        return None, "NO_MOVE_NEEDED", "no formal scientific task"
    if module == "pa_performance_evaluation":
        if task == "scenario_1_load_drift_state_selection_from_scenario_2":
            return "cross_scenario", "READY", "Scenario 2 data is used to select Scenario 1 load states"
        return "scenario_2", "READY", "task is explicitly Scenario 2 PA evaluation"
    if module in {"signal_segmentation", "behavior_modeling", "behavior_fingerprint_retrieval", "behavior_fingerprint_ranking_consistency", "low_bandwidth_behavior_analysis", "lut_clustering_compression"}:
        return "scenario_2", "READY", "task data/definition belongs to current Scenario 2 experiment"
    return None, "DEFERRED_AMBIGUOUS_SCENARIO", "no safe ownership rule"


def _protected_reference_audit(tasks: list[dict[str, Any]]) -> dict[str, Any]:
    protected_root = PROJECT_ROOT / "scripts" / "retrieval_oriented_model_selection"
    files = [p for p in protected_root.rglob("*") if p.is_file() and p.suffix in {".py", ".mjs", ".json", ".txt"}]
    findings: list[dict[str, Any]] = []
    for item in tasks:
        module = item["module"]
        task = item["task_name"]
        tokens = (f"{module}/{task}", f"{module}.{task}", f"scripts/{module}/{task}", f"results/{module}/{task}", f"work_logs/{module}/{task}")
        for path in files:
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            hits = [token for token in tokens if token in text]
            if hits:
                findings.append({"protected_file": str(path), "candidate_module": module, "candidate_task": task, "tokens": hits})
    return {"protected_root": str(protected_root), "protected_file_count_scanned": len(files), "findings": findings, "pass": not findings}


def _path_reference_audit(plan: list[dict[str, Any]]) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    patterns = (
        re.compile(r"Path\(__file__\)\.parents?\[\d+\]"),
        re.compile(r"PROJECT_ROOT|SCRIPTS_ROOT|RESULT_ROOT|WORK_LOG|LOG_ROOT"),
    )
    for record in plan:
        if record["migration_status"] != "READY":
            continue
        root = Path(record["script_path"])
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in {".py", ".mjs"}:
                continue
            text = path.read_text(encoding="utf-8", errors="ignore")
            hits = sorted({match.group(0) for pattern in patterns for match in pattern.finditer(text)})
            if hits:
                findings.append({"module": record["module"], "task_name": record["task_name"], "file": str(path), "patterns": hits})
    return {"findings": findings, "count": len(findings), "note": "Patterns are an audit inventory; each moved task is re-imported after migration."}


def run_audit() -> dict[str, Any]:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    modules = {root: sorted(p.name for p in (PROJECT_ROOT / root).iterdir() if p.is_dir()) for root in ("scripts", "results", "work_logs")}
    protected = {f"{root}/{PROTECTED_MODULES.__iter__().__next__()}": _tree_summary(PROJECT_ROOT / root / "retrieval_oriented_model_selection") for root in ("scripts", "results", "work_logs")}
    data_summary = _tree_summary(PROJECT_ROOT / "data")
    raw_manifest_path = PROJECT_ROOT / "data" / "raw"
    raw_records = _manifest(raw_manifest_path)
    raw_digest = hashlib.sha256()
    for record in raw_records:
        raw_digest.update(record["relative_path"].encode() + b"\0")
        raw_digest.update(int(record["bytes"]).to_bytes(8, "little"))
    raw_summary = {"sha256": raw_digest.hexdigest(), "file_count": len(raw_records), "mat_count": sum(str(record["relative_path"]).lower().endswith(".mat") for record in raw_records), "bytes": sum(int(record["bytes"]) for record in raw_records)}
    inventory: dict[str, Any] = {"timestamp": _now(), "modules": list(MODULES), "root_modules": modules, "protected": protected, "data": data_summary, "raw_manifest": raw_summary, "tasks": {}}
    plan: list[dict[str, Any]] = []
    all_nonprotected_script_tasks: list[dict[str, Any]] = []
    for module in MODULES:
        if module in PROTECTED_MODULES:
            inventory["tasks"][module] = {"status": "DEFERRED_PROTECTED_MODULE"}
            continue
        script_tasks = _formal_tasks("scripts", module)
        result_tasks = {item["task_name"]: item for item in _formal_tasks("results", module)}
        log_tasks = {item["task_name"]: item for item in _formal_tasks("work_logs", module)}
        inventory["tasks"][module] = {"scripts": script_tasks, "results": list(result_tasks.values()), "work_logs": list(log_tasks.values())}
        for item in script_tasks:
            scope, status, reason = _scope_for(module, item["task_name"], Path(item["path"]))
            source = Path(item["path"])
            target = PROJECT_ROOT / "scripts" / module / scope / item["task_name"] if scope else None
            result_source = PROJECT_ROOT / "results" / module / item["task_name"]
            log_source = PROJECT_ROOT / "work_logs" / module / item["task_name"]
            result_target = PROJECT_ROOT / "results" / module / scope / item["task_name"] if scope and result_source.exists() else None
            log_target = PROJECT_ROOT / "work_logs" / module / scope / item["task_name"] if scope and log_source.exists() else None
            collision = bool((target and target.exists()) or (result_target and result_target.exists()) or (log_target and log_target.exists()))
            record = {"module": module, "task_name": item["task_name"], "target_scope": scope, "migration_status": status, "classification_reason": reason, "script_path": str(source), "target_script_path": str(target) if target else None, "result_exists": result_source.exists(), "result_path": str(result_source) if result_source.exists() else None, "target_result_path": str(result_target) if result_target else None, "work_log_exists": log_source.exists(), "work_log_path": str(log_source) if log_source.exists() else None, "target_work_log_path": str(log_target) if log_target else None, "source_file_count": item["file_count"], "same_filesystem": bool(source.stat().st_dev == PROJECT_ROOT.stat().st_dev), "target_exists": bool(target and target.exists()), "collision": collision}
            if status == "READY" and record["collision"]:
                record["migration_status"] = "DEFERRED_PATH_COLLISION"
                record["classification_reason"] = "target path already exists"
            plan.append(record)
            all_nonprotected_script_tasks.append(record)
    protected_audit = _protected_reference_audit(all_nonprotected_script_tasks)
    path_audit = _path_reference_audit(plan)
    for record in plan:
        if record["migration_status"] == "READY" and protected_audit["findings"]:
            matching = [item for item in protected_audit["findings"] if item["candidate_module"] == record["module"] and item["candidate_task"] == record["task_name"]]
            if matching:
                record["migration_status"] = "DEFERRED_PROTECTED_REFERENCE"
                record["classification_reason"] = "protected retrieval source references this task/path"
    pre_manifest: list[dict[str, Any]] = []
    for record in plan:
        if record["migration_status"] != "READY":
            continue
        for artifact, key in (("scripts", "script_path"), ("results", "result_path"), ("work_logs", "work_log_path")):
            path = record.get(key)
            if path:
                pre_manifest.append({"module": record["module"], "task_name": record["task_name"], "artifact": artifact, "root": path, "files": _manifest(Path(path))})
    _write("pre_migration_inventory.json", inventory)
    _write("task_scope_classification.json", {"timestamp": _now(), "plan": plan})
    _write("protected_reference_audit.json", protected_audit)
    _write("path_reference_audit.json", path_audit)
    _write("pre_migration_manifest.json", {"timestamp": _now(), "entries": pre_manifest})
    _write("migration_plan.json", {"timestamp": _now(), "plan": plan, "protected_modules": sorted(PROTECTED_MODULES), "data_move": False, "config_write": False, "ready_count": sum(item["migration_status"] == "READY" for item in plan), "deferred_count": sum(item["migration_status"] != "READY" for item in plan)})
    (LOG_ROOT / "execution_log.txt").write_text(f"{_now()} | Phase 0-5 dry-run complete; ready={sum(item['migration_status']=='READY' for item in plan)} deferred={sum(item['migration_status']!='READY' for item in plan)} protected_reference_pass={protected_audit['pass']} raw={raw_summary}\n", encoding="utf-8")
    return {"inventory": inventory, "plan": plan, "protected_audit": protected_audit, "raw": raw_summary}


if __name__ == "__main__":
    result = run_audit()
    print(json.dumps({"ready": sum(item["migration_status"] == "READY" for item in result["plan"]), "deferred": sum(item["migration_status"] != "READY" for item in result["plan"]), "protected_reference_pass": result["protected_audit"]["pass"], "raw": result["raw"]}, ensure_ascii=False, indent=2))
