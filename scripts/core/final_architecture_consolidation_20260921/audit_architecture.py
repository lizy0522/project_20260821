"""Build read-only manifests and a migration plan for the final architecture."""

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
SCOPES = ("scenario_1", "scenario_2", "cross_scenario")
ROUTED = "retrieval_oriented_model_selection"
CRITICAL_RETRIEVAL_NAMES = {
    "23_checkpoint.json",
    "execution_log.txt",
    "08_final_search_readiness.json",
    "01_full_k2_exhaustive_validation.json",
    "02_k2_screening_recall.json",
    "08_k3_screening_policy_validation.json",
    "k2_exhaustive_checkpoint.json",
    "k2_screening_checkpoint.json",
}


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat()


def _write(name: str, value: Any) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    (LOG_ROOT / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _files(root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in root.rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        ),
        key=lambda path: str(path).lower(),
    )


def _content_manifest(root: Path) -> dict[str, Any]:
    records = []
    digest = hashlib.sha256()
    total = 0
    for path in _files(root):
        relative = path.relative_to(root).as_posix()
        data = path.read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        size = len(data)
        records.append({"relative_path": relative, "bytes": size, "sha256": sha})
        digest.update(relative.encode() + b"\0")
        digest.update(size.to_bytes(8, "little"))
        digest.update(bytes.fromhex(sha))
        total += size
    return {
        "root": str(root),
        "file_count": len(records),
        "total_bytes": total,
        "sha256": digest.hexdigest(),
        "files": records,
    }


def _metadata_manifest(root: Path, *, critical_only: bool = False) -> dict[str, Any]:
    records = []
    total = 0
    directories = 0
    for path in sorted(root.rglob("*"), key=lambda item: str(item).lower()):
        if path.is_dir():
            directories += 1
            continue
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if critical_only and path.name not in CRITICAL_RETRIEVAL_NAMES:
            continue
        info = path.stat()
        records.append(
            {
                "relative_path": path.relative_to(root).as_posix(),
                "st_dev": info.st_dev,
                "inode": info.st_ino,
                "bytes": info.st_size,
                "mtime_ns": info.st_mtime_ns,
            }
        )
        total += info.st_size
    return {
        "root": str(root),
        "file_count": len(records),
        "directory_count": directories,
        "total_bytes": total,
        "critical_only": critical_only,
        "files": records,
    }


def _tree_inventory(root_name: str) -> dict[str, Any]:
    output = {}
    root = PROJECT_ROOT / root_name
    for module in MODULES:
        module_root = root / module
        output[module] = {
            "immediate_dirs": sorted(
                path.name
                for path in module_root.iterdir()
                if path.is_dir() and path.name != "__pycache__"
            ),
            "immediate_files": sorted(path.name for path in module_root.iterdir() if path.is_file()),
        }
    return output


def _active_source_reference_audit() -> dict[str, Any]:
    findings = []
    patterns = {
        "legacy_raw": re.compile(r"data/raw/experiment_2026_0816"),
        "legacy_self_hit": re.compile(r"self_hit_oriented/[^/\"' ]+"),
        "legacy_dpd": re.compile(r"dpd_shareability_oriented/[^/\"' ]+"),
        "retrieval_route_task_expression": re.compile(
            r"retrieval_oriented_model_selection[\"']\s*/\s*[\"'](?:self_hit_oriented|dpd_shareability_oriented)[\"']\s*/\s*(?![\"']scenario_2[\"'])"
        ),
    }
    root = PROJECT_ROOT / "scripts"
    skipped_prefixes = {
        "scripts/core/final_architecture_consolidation_20260921/",
        "scripts/core/module_internal_layout_reorganization_20260917/",
        "scripts/core/module_layout_reorganization_20260917/",
        "scripts/core/retrieval_model_selection_route_split/",
        "scripts/core/shared/tests/test_project_layout_routes.py",
    }
    for path in _files(root):
        if path.suffix not in {".py", ".mjs"}:
            continue
        relative_file = str(path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        if any(relative_file.startswith(prefix) for prefix in skipped_prefixes):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for category, pattern in patterns.items():
            matches = list(pattern.finditer(text))
            if matches:
                findings.append(
                    {
                        "category": category,
                        "file": str(path.relative_to(PROJECT_ROOT)),
                        "match_count": len(matches),
                        "samples": [match.group(0) for match in matches[:8]],
                    }
                )
    return {"findings": findings, "count": len(findings)}


def build_audit() -> dict[str, Any]:
    raw_experiment = PROJECT_ROOT / "data" / "raw" / "experiment_2026_0816"
    retrieval_roots = {
        root_name: PROJECT_ROOT / root_name / ROUTED
        for root_name in ("scripts", "results", "work_logs")
    }
    plan = {
        "data": {
            "source": str(raw_experiment),
            "target": str(PROJECT_ROOT / "data" / "raw" / "scenario_2" / raw_experiment.name),
            "processed_targets": [
                str(PROJECT_ROOT / "data" / "processed" / scope) for scope in ("scenario_1", "scenario_2")
            ],
        },
        "retrieval": [],
    }
    for route in ROUTES:
        source_root = retrieval_roots["scripts"] / route
        for task in sorted(
            path for path in source_root.iterdir() if path.is_dir() and path.name != "__pycache__"
        ):
            plan["retrieval"].append(
                {
                    "route": route,
                    "task": task.name,
                    "script_source": str(task),
                    "script_target": str(source_root / "scenario_2" / task.name),
                    "result_source": str(retrieval_roots["results"] / route / task.name),
                    "result_target": str(retrieval_roots["results"] / route / "scenario_2" / task.name),
                    "work_log_source": str(retrieval_roots["work_logs"] / route / task.name),
                    "work_log_target": str(retrieval_roots["work_logs"] / route / "scenario_2" / task.name),
                }
            )
    audit = {
        "timestamp": _timestamp(),
        "project_root": str(PROJECT_ROOT),
        "modules": list(MODULES),
        "routes": list(ROUTES),
        "scopes": list(SCOPES),
        "root_inventory": {root_name: _tree_inventory(root_name) for root_name in ("scripts", "results", "work_logs")},
        "data_raw_experiment_content_manifest": _content_manifest(raw_experiment),
        "retrieval_metadata_before": {
            root_name: _metadata_manifest(path)
            for root_name, path in retrieval_roots.items()
        },
        "retrieval_critical_content_before": {
            root_name: _metadata_manifest(path, critical_only=True)
            for root_name, path in retrieval_roots.items()
        },
        "active_source_reference_audit": _active_source_reference_audit(),
        "migration_plan": plan,
        "config_files_frozen": [
            "AGENTS.md",
            "README.md",
            "environment.yml",
            "pyproject.toml",
            ".gitignore",
            ".gitattributes",
            ".vscode/settings.json",
        ],
    }
    _write("pre_migration_inventory.json", audit)
    _write("data_manifest_before.json", audit["data_raw_experiment_content_manifest"])
    _write("retrieval_manifest_before.json", audit["retrieval_metadata_before"])
    _write("absolute_path_reference_audit.json", audit["active_source_reference_audit"])
    _write("migration_plan.json", plan)
    (LOG_ROOT / "execution_log.txt").open("a", encoding="utf-8").write(
        f"{_timestamp()} | Phase 0-3 audit complete; research_processes=0; raw_manifest_files={audit['data_raw_experiment_content_manifest']['file_count']}; retrieval_tasks={len(plan['retrieval'])}\n"
    )
    return audit


if __name__ == "__main__":
    result = build_audit()
    print(
        json.dumps(
            {
                "raw_files": result["data_raw_experiment_content_manifest"]["file_count"],
                "retrieval_tasks": len(result["migration_plan"]["retrieval"]),
                "active_reference_findings": result["active_source_reference_audit"]["count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
