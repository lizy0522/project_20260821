"""Validate the two-route migration without running research computations."""

# The validation report contains exact paths and machine-readable summaries.
# ruff: noqa: E501

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
AUDIT_ROOT = PROJECT_ROOT / "results" / "core" / "retrieval_model_selection_route_split"
MODULE = "retrieval_oriented_model_selection"
ROUTES = ("self_hit_oriented", "dpd_shareability_oriented")


def _read_json(name: str) -> Any:
    return json.loads((AUDIT_ROOT / name).read_text(encoding="utf-8"))


def _relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _source_files() -> list[Path]:
    return sorted(
        p
        for p in (SCRIPTS_ROOT / MODULE).rglob("*")
        if p.is_file() and p.suffix in {".py", ".mjs"} and "__pycache__" not in p.parts
    )


def _route_task_sets() -> dict[str, dict[str, set[str]]]:
    result: dict[str, dict[str, set[str]]] = {}
    for root_name in ("scripts", "results", "work_logs"):
        result[root_name] = {}
        for route in ROUTES:
            root = PROJECT_ROOT / root_name / MODULE / route
            result[root_name][route] = {
                p.name for p in root.iterdir() if p.is_dir() and p.name != "__pycache__"
            } if root.is_dir() else set()
    return result


def _compare_manifests(pre: dict[str, Any], post: dict[str, Any]) -> dict[str, Any]:
    rows = {}
    for key, before in sorted(pre.items()):
        after = post.get(key)
        if after is None:
            rows[key] = {"pass": False, "reason": "missing post manifest"}
            continue
        path_set_same = before["relative_file_paths"] == after["relative_file_paths"]
        count_same = before["file_count"] == after["file_count"]
        bytes_same = before["total_bytes"] == after["total_bytes"]
        critical_same = before["critical_sha256"] == after["critical_sha256"]
        tree = key.split("/", 1)[0]
        strict = tree in {"results", "work_logs"}
        rows[key] = {
            "tree": tree,
            "file_count_before": before["file_count"],
            "file_count_after": after["file_count"],
            "total_bytes_before": before["total_bytes"],
            "total_bytes_after": after["total_bytes"],
            "relative_file_set_unchanged": path_set_same,
            "critical_sha256_unchanged": critical_same,
            "total_bytes_unchanged": bytes_same,
            "strict_result_log_pass": bool(strict and path_set_same and count_same and bytes_same and critical_same),
            "script_structure_pass": bool(not strict and path_set_same and count_same),
            "script_bytes_changed_only_by_active_rewrite": bool(not strict and path_set_same and count_same and not bytes_same),
        }
    missing = sorted(set(post) - set(pre))
    result_log_rows = [row for row in rows.values() if row.get("tree") in {"results", "work_logs"}]
    return {
        "per_task": rows,
        "missing_post_keys": missing,
        "result_log_pass": bool(
            all(row.get("strict_result_log_pass", False) for row in result_log_rows) and not missing
        ),
        "script_structure_pass": bool(all(row.get("script_structure_pass", False) for row in rows.values() if row.get("tree") == "scripts")),
    }


def _active_stale_scan() -> dict[str, Any]:
    stale: list[dict[str, Any]] = []
    patterns = (
        re.compile(r"retrieval_oriented_model_selection\.scenario_2_"),
        re.compile(r'"(results|work_logs)"\s*/\s*"retrieval_oriented_model_selection"\s*/\s*"scenario_2_'),
        re.compile(r'"results"\s*/\s*"behavior_fingerprint_retrieval"\s*/\s*"scenario_2_C2_to_Aend_(retrieval_oriented_model_scan|statewise_best_round0_4_5B)'),
    )
    for path in _source_files():
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for line_number, line in enumerate(lines, 1):
            if any(pattern.search(line) for pattern in patterns):
                stale.append({"path": _relative(path), "line": line_number, "text": line.strip()})
    payload = {"stale_active_paths": stale, "pass": not stale}
    (AUDIT_ROOT / "07_active_stale_path_report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return payload


def _task_to_task_imports() -> list[dict[str, str]]:
    task_names = set(_read_json("01_pre_split_inventory.json")["tasks"])
    violations = []
    prefix = "retrieval_oriented_model_selection."
    for path in _source_files():
        relative = path.relative_to(SCRIPTS_ROOT / MODULE)
        current_task = relative.parts[1] if relative.parts and relative.parts[0] in ROUTES and len(relative.parts) > 1 else None
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            for name in names:
                if not name.startswith(prefix):
                    continue
                for task in task_names:
                    if f".{task}" not in name:
                        continue
                    if task != current_task:
                        violations.append({"source": _relative(path), "target": name})
    return violations


def _critical_route_checks() -> dict[str, Any]:
    pre = _read_json("01_pre_split_inventory.json")
    post = _read_json("05_post_split_inventory.json")
    classifications = {
        row["task_name"]: row["assigned_route"]
        for row in __import__("csv").DictReader(
            (AUDIT_ROOT / "03_task_route_classification.csv").open(encoding="utf-8")
        )
    }
    actual = _route_task_sets()
    expected = {route: {task for task, assigned in classifications.items() if assigned == route} for route in ROUTES}
    route_tree_checks = {}
    for root_name in ("scripts", "results", "work_logs"):
        for route in ROUTES:
            expected_present = {
                task for task in expected[route] if pre["inventory"][root_name]["tasks"].get(task)
            }
            route_tree_checks[f"{root_name}/{route}"] = {
                "expected": sorted(expected_present),
                "actual": sorted(actual[root_name][route]),
                "pass": expected_present == actual[root_name][route],
            }
    direct_children = {
        root_name: sorted(
            p.name
            for p in (PROJECT_ROOT / root_name / MODULE).iterdir()
            if p.is_dir() and p.name not in {"shared", "__pycache__", *ROUTES}
        )
        for root_name in ("scripts", "results", "work_logs")
    }
    module_mirror = __import__("core.shared.module_registry", fromlist=["validate_module_root_alignment"]).validate_module_root_alignment(PROJECT_ROOT)
    return {
        "task_count_pre": pre["task_count"],
        "task_count_post": post["task_count"],
        "classification_counts": pre["classification_counts"],
        "route_tree_checks": route_tree_checks,
        "direct_flat_tasks_remaining": direct_children,
        "flat_task_pass": all(not values for values in direct_children.values()),
        "module_root_mirror": module_mirror,
        "module_shared_preserved": all((PROJECT_ROOT / "scripts" / MODULE / route / "__init__.py").is_file() for route in ROUTES) and (PROJECT_ROOT / "scripts" / MODULE / "shared" / "__init__.py").is_file(),
        "legacy_internal_layout_status": "EXPECTED_SCHEMA_MISMATCH",
        "legacy_internal_layout_reason": "The old validator expects module/task; this migration intentionally introduces module/route/task before configuration synchronization.",
    }


def main() -> None:
    pre_manifests = _read_json("pre_task_manifests.json")
    post_manifests = _read_json("post_task_manifests.json")
    manifest_checks = _compare_manifests(pre_manifests, post_manifests)
    stale = _active_stale_scan()
    dependency_violations = _task_to_task_imports()
    route_checks = _critical_route_checks()
    config_before = _read_json("root_config_hashes_before.json")
    from core.retrieval_model_selection_route_split.build_inventory import (
        _config_hashes,
        _raw_manifest,
    )

    config_after = _config_hashes()
    raw_before = _read_json("01_pre_split_inventory.json")["raw_manifest"]
    raw_after = _raw_manifest()
    result = {
        "manifest_checks": manifest_checks,
        "route_checks": route_checks,
        "active_stale_path_scan": stale,
        "task_to_task_imports": {"violations": dependency_violations, "pass": not dependency_violations},
        "root_config_hashes": {"before": config_before, "after": config_after, "unchanged": config_before == config_after},
        "raw_manifest": {"before": raw_before, "after": raw_after, "unchanged": raw_before == raw_after},
        "research_computation_run": False,
        "large_result_copied": False,
    }
    result["pass"] = bool(
        manifest_checks["result_log_pass"]
        and manifest_checks["script_structure_pass"]
        and stale["pass"]
        and not dependency_violations
        and route_checks["flat_task_pass"]
        and route_checks["module_root_mirror"]["pass"]
        and config_before == config_after
        and raw_before == raw_after
    )
    (AUDIT_ROOT / "06_route_split_validation.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"pass": result["pass"], "manifest_checks": manifest_checks, "stale": stale, "task_to_task_imports": result["task_to_task_imports"]}, ensure_ascii=False, indent=2))
    if not result["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
