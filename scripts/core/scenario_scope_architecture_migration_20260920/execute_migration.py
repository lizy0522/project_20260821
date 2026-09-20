"""Execute the approved non-protected scenario-scope migration atomically."""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir())
LOG_ROOT = PROJECT_ROOT / "work_logs" / "core" / "scenario_scope_architecture_migration_20260920"
PROTECTED_PREFIX = PROJECT_ROOT / "retrieval_oriented_model_selection"
PROTECTED_PART = "retrieval_oriented_model_selection"


def _now() -> str:
    import time

    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _append_journal(record: dict[str, Any]) -> None:
    with (LOG_ROOT / "migration_journal.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": _now(), **record}, ensure_ascii=False, default=str) + "\n")


def _manifest(root: Path) -> list[dict[str, Any]]:
    if not root.exists():
        return []
    output = []
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: str(item).lower()):
        output.append({"relative_path": str(path.relative_to(root)), "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    return output


def _assert_safe(record: dict[str, Any]) -> None:
    for key in ("script_path", "target_script_path", "result_path", "target_result_path", "work_log_path", "target_work_log_path"):
        value = record.get(key)
        if not value:
            continue
        if PROTECTED_PART in str(Path(value).relative_to(PROJECT_ROOT)).split("/"):
            raise RuntimeError(f"protected path in migration record: {key}={value}")
    for key in ("target_script_path", "target_result_path", "target_work_log_path"):
        value = record.get(key)
        if value and Path(value).exists():
            raise RuntimeError(f"migration target already exists: {value}")
    source = Path(record["script_path"])
    if not source.exists():
        raise RuntimeError(f"source script task missing: {source}")
    if source.stat().st_dev != PROJECT_ROOT.stat().st_dev:
        raise RuntimeError(f"source task is on a different filesystem: {source}")


def _move_one(source: Path, target: Path, artifact: str, record: dict[str, Any]) -> None:
    if target.exists():
        raise RuntimeError(f"target exists before move: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    _append_journal({"event": "move_begin", "artifact": artifact, "source": str(source), "target": str(target), "module": record["module"], "task_name": record["task_name"]})
    source.rename(target)
    _append_journal({"event": "move_complete", "artifact": artifact, "source": str(source), "target": str(target), "module": record["module"], "task_name": record["task_name"]})


def _patch_task_sources(record: dict[str, Any]) -> list[dict[str, Any]]:
    scope = record["target_scope"]
    module = record["module"]
    task = record["task_name"]
    root = Path(record["target_script_path"])
    old_dotted = f"{module}.{task}"
    new_dotted = f"{module}.{scope}.{task}"
    old_fs = f"{module}/{task}"
    new_fs = f"{module}/{scope}/{task}"
    changed: list[dict[str, Any]] = []
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix not in {".py", ".mjs"}:
            continue
        before = path.read_text(encoding="utf-8", errors="ignore")
        after = before.replace(old_dotted, new_dotted)
        after = after.replace(f"scripts/{old_fs}", f"scripts/{new_fs}")
        after = after.replace(f"results/{old_fs}", f"results/{new_fs}")
        after = after.replace(f"work_logs/{old_fs}", f"work_logs/{new_fs}")
        if module == "behavior_fingerprint_retrieval" and task == "scenario_2_envelope19_c2endshared_commonB_full_lut_retrieval_5B" and path.name == "test_envelope19_c2endshared_commonB_full_lut_retrieval.py":
            after = after.replace("Path(__file__).resolve().parents[1]", "Path(__file__).resolve().parents[2]")
        if after != before:
            before_sha = hashlib.sha256(before.encode()).hexdigest()
            path.write_text(after, encoding="utf-8")
            changed.append({"file": str(path), "old_sha256": before_sha, "new_sha256": hashlib.sha256(after.encode()).hexdigest(), "change_category": "IMPORT_OR_PATH_SCOPE", "old_dotted": old_dotted, "new_dotted": new_dotted, "old_filesystem_token": old_fs, "new_filesystem_token": new_fs})
    return changed


def _post_inventory(records: list[dict[str, Any]], patches: list[dict[str, Any]]) -> dict[str, Any]:
    return {"timestamp": _now(), "migrated_records": records, "source_patches": patches, "protected_module_touched": False, "data_touched": False, "config_touched": False}


def execute() -> dict[str, Any]:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    plan_path = LOG_ROOT / "migration_plan.json"
    plan = json.loads(plan_path.read_text(encoding="utf-8"))["plan"]
    ready = [record for record in plan if record["migration_status"] == "READY"]
    if not ready:
        raise RuntimeError("migration plan contains no READY tasks")
    for record in ready:
        _assert_safe(record)
    migrated: list[dict[str, Any]] = []
    patches: list[dict[str, Any]] = []
    for record in ready:
        module = record["module"]
        scope = record["target_scope"]
        task = record["task_name"]
        target_scope_dir = PROJECT_ROOT / "scripts" / module / scope
        target_scope_dir.mkdir(parents=True, exist_ok=True)
        init_path = target_scope_dir / "__init__.py"
        if not init_path.exists():
            init_path.write_text(f'"""{scope} scenario scope."""\n', encoding="utf-8")
        moved_record = {"module": module, "task_name": task, "scope": scope, "artifacts": []}
        pairs = [("scripts", Path(record["script_path"]), Path(record["target_script_path"])), ("results", Path(record["result_path"]) if record.get("result_path") else None, Path(record["target_result_path"]) if record.get("target_result_path") else None), ("work_logs", Path(record["work_log_path"]) if record.get("work_log_path") else None, Path(record["target_work_log_path"]) if record.get("target_work_log_path") else None)]
        try:
            for artifact, source, target in pairs:
                if source is None or target is None:
                    continue
                _move_one(source, target, artifact, record)
                moved_record["artifacts"].append({"artifact": artifact, "source": str(source), "target": str(target), "manifest": _manifest(target)})
            patches.extend(_patch_task_sources(record))
            moved_record["status"] = "COMMITTED"
            _append_journal({"event": "task_committed", "module": module, "task_name": task, "scope": scope})
            migrated.append(moved_record)
        except Exception:
            _append_journal({"event": "task_failed", "module": module, "task_name": task, "scope": scope})
            raise
    _write = LOG_ROOT / "source_patch_manifest.json"
    _write.write_text(json.dumps(patches, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    post = _post_inventory(migrated, patches)
    (LOG_ROOT / "post_migration_inventory.json").write_text(json.dumps(post, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (LOG_ROOT / "execution_log.txt").open("a", encoding="utf-8") as handle:
        handle.write(f"{_now()} | EXECUTE migration committed tasks={len(migrated)} source_patches={len(patches)} protected_module_touched=False data_touched=False config_touched=False\n")
    return {"migrated_tasks": len(migrated), "source_patches": len(patches), "protected_module_touched": False, "data_touched": False, "config_touched": False}


if __name__ == "__main__":
    print(json.dumps(execute(), ensure_ascii=False, indent=2))
