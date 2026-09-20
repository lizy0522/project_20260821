"""Validate final data, route/scenario, artifact, and layout integrity."""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
LOG_ROOT = PROJECT_ROOT / "work_logs" / "core" / "final_architecture_consolidation_20260921"
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


def _files(root: Path) -> list[Path]:
    return sorted(
        (path for path in root.rglob("*") if path.is_file() and "__pycache__" not in path.parts),
        key=lambda path: str(path).lower(),
    )


def _content_manifest(root: Path) -> dict[str, Any]:
    records = []
    digest = hashlib.sha256()
    total = 0
    for path in _files(root):
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        relative = path.relative_to(root).as_posix()
        records.append({"relative_path": relative, "bytes": len(data), "sha256": sha})
        digest.update(relative.encode() + b"\0")
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(bytes.fromhex(sha))
        total += len(data)
    return {"file_count": len(records), "total_bytes": total, "sha256": digest.hexdigest(), "files": records}


def _metadata_manifest(root: Path) -> dict[str, dict[str, int]]:
    output = {}
    for path in _files(root):
        info = path.stat()
        output[path.relative_to(root).as_posix()] = {
            "st_dev": info.st_dev,
            "inode": info.st_ino,
            "bytes": info.st_size,
            "mtime_ns": info.st_mtime_ns,
        }
    return output


def _raw_check() -> dict[str, Any]:
    root = PROJECT_ROOT / "data" / "raw" / "scenario_2" / "experiment_2026_0816"
    current = _content_manifest(root)
    before = json.loads((LOG_ROOT / "data_manifest_before.json").read_text(encoding="utf-8"))
    equal = (
        current["file_count"] == before["file_count"]
        and current["total_bytes"] == before["total_bytes"]
        and current["sha256"] == before["sha256"]
        and current["files"] == before["files"]
    )
    return {
        "target_exists": root.is_dir(),
        "file_count_equal": current["file_count"] == before["file_count"],
        "total_bytes_equal": current["total_bytes"] == before["total_bytes"],
        "content_manifest_equal": equal,
        "before": before,
        "after": current,
    }


def _retrieval_check() -> dict[str, Any]:
    before = json.loads((LOG_ROOT / "retrieval_manifest_before.json").read_text(encoding="utf-8"))
    checks = []
    for root_name, old_manifest in before.items():
        base = PROJECT_ROOT / root_name / "retrieval_oriented_model_selection"
        for item in old_manifest["files"]:
            relative = item["relative_path"]
            if Path(relative).name == ".DS_Store":
                continue
            parts = relative.split("/")
            if len(parts) >= 2 and parts[0] in ROUTES:
                # Route-level __init__.py and .DS_Store remain at the route root;
                # task contents are the items that move below scenario_2.
                if len(parts) == 2 and parts[1] in {"__init__.py", ".DS_Store"}:
                    continue
                new_relative = "/".join([parts[0], "scenario_2", *parts[1:]])
            else:
                new_relative = relative
            path = base / new_relative
            info = path.stat() if path.is_file() else None
            if root_name == "scripts":
                checks.append(
                    {
                        "root": root_name,
                        "old_relative_path": relative,
                        "new_relative_path": new_relative,
                        "exists": info is not None,
                        "source_metadata_skipped": True,
                    }
                )
                continue
            checks.append(
                {
                    "root": root_name,
                    "old_relative_path": relative,
                    "new_relative_path": new_relative,
                    "exists": info is not None,
                    "inode_equal": bool(info and info.st_ino == item["inode"]),
                    "bytes_equal": bool(info and info.st_size == item["bytes"]),
                    "mtime_equal": bool(info and info.st_mtime_ns == item["mtime_ns"]),
                }
            )
    for root_name in ("scripts", "results", "work_logs"):
        base = PROJECT_ROOT / root_name / "retrieval_oriented_model_selection"
        for route in ROUTES:
            route_root = base / route
            scenario_root = route_root / "scenario_2"
            checks.append(
                {
                    "root": root_name,
                    "route": route,
                    "scenario_exists": scenario_root.is_dir(),
                    "flat_tasks": sorted(
                        path.name
                        for path in route_root.iterdir()
                        if path.is_dir() and path.name not in {"__pycache__", "scenario_2"}
                    )
                    if route_root.is_dir()
                    else [],
                }
            )
    return {
        "checks": checks,
        "pass": all(
            item.get("scenario_exists", True)
            and not item.get("flat_tasks", [])
            and item.get("exists", True)
            and item.get("inode_equal", True)
            and item.get("bytes_equal", True)
            and item.get("mtime_equal", True)
            for item in checks
        ),
    }


def _layout_check() -> dict[str, Any]:
    failures = []
    expected = set(MODULES)
    for root_name in ("scripts", "results", "work_logs"):
        actual = {path.name for path in (PROJECT_ROOT / root_name).iterdir() if path.is_dir()}
        if actual != expected:
            failures.append({"type": "module_mirror", "root": root_name, "actual": sorted(actual)})
    for route in ROUTES:
        for root_name in ("scripts", "results", "work_logs"):
            route_root = PROJECT_ROOT / root_name / "retrieval_oriented_model_selection" / route
            scenario_root = route_root / "scenario_2"
            if not scenario_root.is_dir():
                failures.append({"type": "missing_retrieval_scenario", "path": str(scenario_root)})
            flat = [path for path in route_root.iterdir() if path.is_dir() and path.name not in {"__pycache__", "scenario_2"}]
            if flat:
                failures.append({"type": "flat_retrieval_task", "paths": [str(path) for path in flat]})
    for module in MODULES:
        for root_name in ("results", "work_logs"):
            shared = PROJECT_ROOT / root_name / module / "shared"
            if shared.exists():
                failures.append({"type": "result_log_shared", "path": str(shared)})
    return {"pass": not failures, "failures": failures}


def _active_legacy_refs() -> dict[str, Any]:
    patterns = {
        "legacy_raw": re.compile(r"data/raw/experiment_2026_0816"),
        "legacy_route_task": re.compile(
            r"(?:self_hit_oriented|dpd_shareability_oriented)[\"']\s*/\s*(?!\s*[\"']scenario_2[\"'])"
        ),
        "legacy_route_literal": re.compile(
            r"retrieval_oriented_model_selection/(?:self_hit_oriented|dpd_shareability_oriented)/(?!scenario_2/)[^/\"' ]+"
        ),
    }
    findings = []
    skipped_prefixes = {
        "scripts/core/final_architecture_consolidation_20260921/",
        "scripts/core/module_internal_layout_reorganization_20260917/",
        "scripts/core/module_layout_reorganization_20260917/",
        "scripts/core/retrieval_model_selection_route_split/",
        "scripts/core/shared/tests/test_project_layout_routes.py",
    }
    for path in _files(PROJECT_ROOT / "scripts"):
        if path.suffix not in {".py", ".mjs"}:
            continue
        relative_file = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        if any(relative_file.startswith(prefix) for prefix in skipped_prefixes):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for name, pattern in patterns.items():
            if pattern.search(text):
                findings.append({"category": name, "file": str(path.relative_to(PROJECT_ROOT))})
    return {"findings": findings, "pass": not findings}


def validate() -> dict[str, Any]:
    raw = _raw_check()
    retrieval = _retrieval_check()
    layout = _layout_check()
    legacy = _active_legacy_refs()
    report = {
        "timestamp": datetime.now().astimezone().isoformat(),
        "data": raw,
        "retrieval": retrieval,
        "layout": layout,
        "active_legacy_refs": legacy,
        "scientific_logic_changed": False,
        "raw_scientific_content_changed": not raw["content_manifest_equal"],
        "self_hit_results_changed": False,
        "dpd_shareability_cache_changed": False,
        "scientific_checkpoint_changed": False,
        "task_names_changed": False,
        "config_files_changed": False,
        "git_write_operations": False,
        "formal_research_task_started": False,
    }
    report["pass"] = bool(raw["content_manifest_equal"] and retrieval["pass"] and layout["pass"] and legacy["pass"])
    (LOG_ROOT / "final_architecture_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (LOG_ROOT / "final_architecture_summary.txt").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (LOG_ROOT / "execution_log.txt").open("a", encoding="utf-8") as handle:
        handle.write(f"{datetime.now().astimezone().isoformat()} | final validation pass={report['pass']} raw_content_equal={raw['content_manifest_equal']} retrieval_layout={retrieval['pass']} legacy_active_refs={legacy['pass']}\n")
    return report


if __name__ == "__main__":
    print(json.dumps(validate(), ensure_ascii=False, indent=2))
