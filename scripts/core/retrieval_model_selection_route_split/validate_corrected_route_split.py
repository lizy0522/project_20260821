"""Validate the corrected historical route ownership."""

# ruff: noqa: E501

from __future__ import annotations

import ast
import csv
import hashlib
import importlib
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

MODULE = "retrieval_oriented_model_selection"
SELF_ROUTE = "self_hit_oriented"
DPD_ROUTE = "dpd_shareability_oriented"
ROUTES = (SELF_ROUTE, DPD_ROUTE)
AUDIT_ROOT = PROJECT_ROOT / "results" / "core" / "retrieval_model_selection_route_split"
CONFIG_NAMES = ("AGENTS.md", "README.md", "environment.yml", "pyproject.toml", ".gitignore")
TASKS = tuple(
    row["task_name"]
    for row in csv.DictReader((AUDIT_ROOT / "08_corrected_task_route_classification.csv").open(encoding="utf-8"))
)


def _relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _config_hashes() -> dict[str, str]:
    return {name: _sha256(PROJECT_ROOT / name) for name in CONFIG_NAMES}


def _raw_manifest() -> dict[str, Any]:
    root = PROJECT_ROOT / "data" / "raw"
    files = sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.as_posix())
    digest = hashlib.sha256()
    total = 0
    mat_count = 0
    for path in files:
        size = path.stat().st_size
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        digest.update(size.to_bytes(8, "little"))
        total += size
        mat_count += path.suffix.lower() == ".mat"
    return {"sha256": digest.hexdigest(), "file_count": len(files), "mat_count": mat_count, "bytes": total}


def _files(path: Path) -> list[Path]:
    if not path.is_dir():
        return []
    return sorted(
        (p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"),
        key=lambda p: p.as_posix(),
    )


def _manifest(path: Path) -> dict[str, Any]:
    files = _files(path)
    critical_names = {
        "00_task_definition.txt",
        "execution_log.txt",
        "19_final_result_summary.txt",
        "21_final_result_summary.txt",
        "22_final_result_summary.txt",
        "22_final_result_summary.json",
        "24_validation_checks.json",
        "30_final_result_summary.txt",
        "31_checkpoint.json",
        "32_validation_checks.json",
        "36_final_result_summary.txt",
        "37_checkpoint.json",
        "38_validation_checks.json",
    }
    critical = {
        p.relative_to(path).as_posix(): _sha256(p)
        for p in files
        if p.name in critical_names
    }
    return {
        "file_count": len(files),
        "total_bytes": sum(p.stat().st_size for p in files),
        "relative_file_paths": [p.relative_to(path).as_posix() for p in files],
        "critical_sha256": critical,
    }


def _current_task_path(tree: str, task: str) -> Path:
    return PROJECT_ROOT / tree / MODULE / SELF_ROUTE / task


def _task_sets() -> dict[str, dict[str, list[str]]]:
    result: dict[str, dict[str, list[str]]] = {}
    for tree in ("scripts", "results", "work_logs"):
        result[tree] = {}
        for route in ROUTES:
            root = PROJECT_ROOT / tree / MODULE / route
            result[tree][route] = sorted(
                p.name for p in root.iterdir() if p.is_dir() and p.name != "__pycache__"
            ) if root.is_dir() else []
    return result


def _post_manifests() -> dict[str, Any]:
    result = {}
    for tree in ("scripts", "results", "work_logs"):
        for task in TASKS:
            path = _current_task_path(tree, task)
            if path.is_dir():
                result[f"{tree}/{task}"] = _manifest(path)
    return result


def _compare_manifests(pre: dict[str, Any], post: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for key, before in sorted(pre.items()):
        after = post.get(key)
        if after is None:
            rows[key] = {"pass": False, "reason": "missing post manifest"}
            continue
        tree = key.split("/", 1)[0]
        paths_same = before["relative_file_paths"] == after["relative_file_paths"]
        count_same = before["file_count"] == after["file_count"]
        bytes_same = before["total_bytes"] == after["total_bytes"]
        sha_same = before["critical_sha256"] == after["critical_sha256"]
        rows[key] = {
            "tree": tree,
            "file_count_before": before["file_count"],
            "file_count_after": after["file_count"],
            "bytes_before": before["total_bytes"],
            "bytes_after": after["total_bytes"],
            "relative_file_set_unchanged": paths_same,
            "critical_sha256_unchanged": sha_same,
            "total_bytes_unchanged": bytes_same,
            "strict_result_log_pass": bool(tree in {"results", "work_logs"} and paths_same and count_same and bytes_same and sha_same),
            "script_structure_pass": bool(tree == "scripts" and paths_same and count_same),
        }
    missing = sorted(set(pre) - set(post))
    result_log_rows = [row for row in rows.values() if row.get("tree") in {"results", "work_logs"}]
    script_rows = [row for row in rows.values() if row.get("tree") == "scripts"]
    return {
        "per_task": rows,
        "missing_post_keys": missing,
        "result_log_pass": bool(not missing and all(row["strict_result_log_pass"] for row in result_log_rows)),
        "script_structure_pass": bool(script_rows and all(row["script_structure_pass"] for row in script_rows)),
    }


def _stale_scan() -> dict[str, Any]:
    stale = []
    source_root = SCRIPTS_ROOT / MODULE
    for path in sorted(source_root.rglob("*"), key=lambda p: p.as_posix()):
        if not path.is_file() or path.suffix not in {".py", ".mjs"} or "__pycache__" in path.parts:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for task in TASKS:
            patterns = (
                re.compile(rf"retrieval_oriented_model_selection\.{DPD_ROUTE}\.{re.escape(task)}"),
                re.compile(rf'"{DPD_ROUTE}"\s*/\s*"{re.escape(task)}"'),
                re.compile(rf"{DPD_ROUTE}[\\/]\s*{re.escape(task)}"),
            )
            for pattern in patterns:
                if pattern.search(text):
                    stale.append({"path": _relative(path), "task": task, "pattern": pattern.pattern})
                    break
    result = {"historical_task_old_dpd_refs": stale, "pass": not stale}
    (AUDIT_ROOT / "12_corrected_active_stale_path_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def _task_dependencies() -> list[dict[str, str]]:
    violations = []
    prefix = f"retrieval_oriented_model_selection.{SELF_ROUTE}."
    source_root = SCRIPTS_ROOT / MODULE / SELF_ROUTE
    for path in sorted(source_root.rglob("*.py"), key=lambda p: p.as_posix()):
        current_task = path.relative_to(source_root).parts[0]
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            for name in names:
                if name.startswith(prefix):
                    target = name[len(prefix):].split(".", 1)[0]
                    if target in TASKS and target != current_task:
                        violations.append({"source": _relative(path), "target": name})
    return violations


def _import_smoke() -> dict[str, Any]:
    names = [
        f"{MODULE}.shared",
        f"{MODULE}.shared.route_paths",
        f"{MODULE}.{SELF_ROUTE}",
        f"{MODULE}.{DPD_ROUTE}",
    ]
    names.extend(f"{MODULE}.{SELF_ROUTE}.{task}" for task in TASKS)
    for name in names:
        importlib.import_module(name)
    return {"count": len(names), "packages": names, "pass": True}


def _run_check(command: list[str]) -> dict[str, Any]:
    result = subprocess.run(command, cwd=PROJECT_ROOT, capture_output=True, text=True, check=False)
    return {
        "command": command,
        "returncode": result.returncode,
        "pass": result.returncode == 0,
        "stdout_tail": result.stdout[-2000:],
        "stderr_tail": result.stderr[-2000:],
    }


def main() -> None:
    pre = json.loads((AUDIT_ROOT / "corrective_pre_migration_inventory.json").read_text())
    pre_manifests = json.loads((AUDIT_ROOT / "corrective_pre_task_manifests.json").read_text())
    post = _post_manifests()
    manifest_checks = _compare_manifests(pre_manifests, post)
    task_sets = _task_sets()
    expected_self = set(TASKS)
    expected_results = set(pre["inventory"]["task_sets"]["results"].get(DPD_ROUTE, [])) | set(
        pre["inventory"]["task_sets"]["results"].get(SELF_ROUTE, [])
    )
    expected_logs = set(pre["inventory"]["task_sets"]["work_logs"].get(DPD_ROUTE, [])) | set(
        pre["inventory"]["task_sets"]["work_logs"].get(SELF_ROUTE, [])
    )
    route_checks = {
        "scripts_self_exact": set(task_sets["scripts"][SELF_ROUTE]) == expected_self,
        "scripts_dpd_empty": not task_sets["scripts"][DPD_ROUTE],
        "results_self_expected": set(task_sets["results"][SELF_ROUTE]) == expected_results,
        "results_dpd_empty": not task_sets["results"][DPD_ROUTE],
        "logs_self_expected": set(task_sets["work_logs"][SELF_ROUTE]) == expected_logs,
        "logs_dpd_empty": not task_sets["work_logs"][DPD_ROUTE],
        "flat_tasks_empty": all(
            not [p for p in (PROJECT_ROOT / tree / MODULE).iterdir() if p.is_dir() and p.name not in {"shared", "__pycache__", *ROUTES}]
            for tree in ("scripts", "results", "work_logs")
        ),
        "route_level_shared_absent": all(
            not (PROJECT_ROOT / tree / MODULE / route / "shared").exists()
            for tree in ("scripts", "results", "work_logs")
            for route in ROUTES
        ),
        "module_shared_preserved": (SCRIPTS_ROOT / MODULE / "shared" / "route_paths.py").is_file(),
    }
    module_mirror = importlib.import_module("core.shared.module_registry").validate_module_root_alignment(PROJECT_ROOT)
    config_before = json.loads((AUDIT_ROOT / "corrective_root_config_hashes_before.json").read_text())
    config_after = _config_hashes()
    raw_before = pre["raw_manifest"]
    raw_after = _raw_manifest()
    checks = {
        "maintenance_mode": "continuous_corrective_migration",
        "previous_reports_preserved": all(
            (AUDIT_ROOT / name).exists()
            for name in (
                "03_task_route_classification.csv",
                "04_migration_map.csv",
                "06_route_split_validation.json",
                "migration_execution.json",
            )
        ),
        "corrected_task_count": len(TASKS),
        "corrected_route": SELF_ROUTE,
        "route_checks": route_checks,
        "manifest_checks": manifest_checks,
        "active_stale_path_scan": _stale_scan(),
        "task_to_task_dependencies": {"violations": _task_dependencies(), "pass": not _task_dependencies()},
        "module_root_mirror": module_mirror,
        "raw_manifest": {"before": raw_before, "after": raw_after, "unchanged": raw_before == raw_after},
        "root_config_hashes": {"before": config_before, "after": config_after, "unchanged": config_before == config_after},
        "route_paths_exists": (SCRIPTS_ROOT / MODULE / "shared" / "route_paths.py").is_file(),
        "import_smoke": _import_smoke(),
        "compileall": _run_check([sys.executable, "-m", "compileall", "-q", "scripts"]),
        "ruff": _run_check(["/opt/anaconda3/envs/project_20260821/bin/ruff", "check", "--no-cache", "scripts"]),
        "pip_check": _run_check(["/opt/anaconda3/envs/project_20260821/bin/python", "-m", "pip", "check"]),
        "scientific_computation_run": False,
        "historical_results_regenerated": False,
        "old_global_validator": "EXPECTED_SCHEMA_MISMATCH",
        "configuration_synchronization": "DEFERRED",
    }
    checks["pass"] = bool(
        all(route_checks.values())
        and manifest_checks["result_log_pass"]
        and manifest_checks["script_structure_pass"]
        and checks["active_stale_path_scan"]["pass"]
        and checks["task_to_task_dependencies"]["pass"]
        and module_mirror["pass"]
        and checks["raw_manifest"]["unchanged"]
        and checks["root_config_hashes"]["unchanged"]
        and checks["import_smoke"]["pass"]
        and checks["compileall"]["pass"]
        and checks["ruff"]["pass"]
        and checks["pip_check"]["pass"]
    )
    (AUDIT_ROOT / "13_corrected_route_split_validation.json").write_text(
        json.dumps(checks, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"pass": checks["pass"], "route_checks": route_checks, "manifest_checks": manifest_checks, "stale": checks["active_stale_path_scan"], "scientific_computation_run": False}, ensure_ascii=False, indent=2))
    if not checks["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
