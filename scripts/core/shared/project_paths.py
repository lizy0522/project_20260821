"""Canonical project, module, and task path helpers."""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .module_registry import MODULES, ROOT_NAMES

SCENARIO_IDS = ("scenario_1", "scenario_2")
TASK_SCOPES = SCENARIO_IDS + ("cross_scenario",)
ROUTED_MODULE = "retrieval_oriented_model_selection"

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
RESULTS_ROOT = PROJECT_ROOT / "results"
WORK_LOGS_ROOT = PROJECT_ROOT / "work_logs"
DATA_ROOT = PROJECT_ROOT / "data"
RAW_DATA_ROOT = DATA_ROOT / "raw"


@dataclass(frozen=True)
class TaskPaths:
    """The mirrored paths belonging to one module/scope/task pair."""

    module: str
    task: str
    scenario_scope: str | None
    scripts: Path
    results: Path
    work_logs: Path


def get_module_root(root_name: str, module_name: str) -> Path:
    """Return one canonical module root and reject unknown names."""

    if root_name not in ROOT_NAMES:
        raise ValueError(f"unknown project root: {root_name}")
    if module_name not in MODULES:
        raise ValueError(f"unknown module: {module_name}")
    return PROJECT_ROOT / root_name / module_name


def _validate_scope(scenario_scope: str | None) -> None:
    if scenario_scope is not None and scenario_scope not in TASK_SCOPES:
        raise ValueError(f"unknown scenario scope: {scenario_scope!r}")


def get_raw_scenario_root(scenario_id: str) -> Path:
    """Return the canonical raw-data root for one scenario."""

    if scenario_id not in SCENARIO_IDS:
        raise ValueError(f"unknown data scenario: {scenario_id!r}")
    return RAW_DATA_ROOT / scenario_id


def get_raw_experiment_root(scenario_id: str, experiment_name: str) -> Path:
    """Return one experiment path below the canonical raw scenario root."""

    if not experiment_name or "/" in experiment_name or "\\" in experiment_name:
        raise ValueError(f"invalid experiment name: {experiment_name!r}")
    return get_raw_scenario_root(scenario_id) / experiment_name


def get_processed_scenario_root(scenario_id: str) -> Path:
    """Return the canonical processed-data root for one scenario."""

    if scenario_id not in SCENARIO_IDS:
        raise ValueError(f"unknown data scenario: {scenario_id!r}")
    return DATA_ROOT / "processed" / scenario_id


def legacy_raw_manifest() -> dict[str, Any]:
    """Return the pre-scenario path-and-size manifest for resume compatibility."""

    digest = hashlib.sha256()
    files = sorted(
        (path for path in RAW_DATA_ROOT.rglob("*") if path.is_file()),
        key=lambda path: str(path).lower(),
    )
    total_bytes = 0
    mat_count = 0
    for path in files:
        relative = path.relative_to(RAW_DATA_ROOT).as_posix()
        if relative.startswith("scenario_2/"):
            relative = relative[len("scenario_2/") :]
        digest.update(relative.encode("utf-8") + b"\0")
        size = int(path.stat().st_size)
        digest.update(size.to_bytes(8, "little"))
        total_bytes += size
        mat_count += int(path.suffix.lower() == ".mat")
    return {
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "mat_count": mat_count,
        "bytes": total_bytes,
    }


def get_task_paths(
    module_name: str,
    task_name: str,
    scenario_scope: str | None = None,
) -> TaskPaths:
    """Return mirrored paths for a normal module task."""

    if not task_name or "/" in task_name or "\\" in task_name:
        raise ValueError(f"invalid task name: {task_name!r}")
    if module_name == ROUTED_MODULE:
        raise ValueError("use retrieval_oriented_model_selection.shared.route_paths for routed tasks")
    _validate_scope(scenario_scope)
    suffix = (scenario_scope, task_name) if scenario_scope else (task_name,)
    return TaskPaths(
        module=module_name,
        task=task_name,
        scenario_scope=scenario_scope,
        scripts=get_module_root("scripts", module_name).joinpath(*suffix),
        results=get_module_root("results", module_name).joinpath(*suffix),
        work_logs=get_module_root("work_logs", module_name).joinpath(*suffix),
    )


def find_project_root(file_path: Path | str) -> Path:
    """Find the project root from any source file below the checkout."""

    path = Path(file_path).resolve()
    for candidate in (path.parent, *path.parents):
        if (candidate / "environment.yml").is_file() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError(f"could not locate project root from {file_path}")


def get_task_paths_for_file(file_path: Path | str, task_name: str) -> TaskPaths:
    """Infer the module from a source file and return its task paths."""

    root = find_project_root(file_path)
    relative = Path(file_path).resolve().relative_to(root / "scripts")
    if not relative.parts:
        raise ValueError(f"source file is not below scripts/: {file_path}")
    module_name = relative.parts[0]
    if module_name not in MODULES:
        raise ValueError(f"source file is not below a registered module: {file_path}")
    if module_name == ROUTED_MODULE:
        raise ValueError("use retrieval_oriented_model_selection.shared.route_paths for routed files")
    scenario_scope = relative.parts[1] if len(relative.parts) >= 3 and relative.parts[1] in TASK_SCOPES else None
    _validate_scope(scenario_scope)
    suffix = (scenario_scope, task_name) if scenario_scope else (task_name,)
    return TaskPaths(
        module=module_name,
        task=task_name,
        scenario_scope=scenario_scope,
        scripts=(root / "scripts" / module_name).joinpath(*suffix),
        results=(root / "results" / module_name).joinpath(*suffix),
        work_logs=(root / "work_logs" / module_name).joinpath(*suffix),
    )


__all__ = [
    "DATA_ROOT",
    "ROUTED_MODULE",
    "SCENARIO_IDS",
    "TASK_SCOPES",
    "PROJECT_ROOT",
    "RAW_DATA_ROOT",
    "RESULTS_ROOT",
    "SCRIPTS_ROOT",
    "TaskPaths",
    "WORK_LOGS_ROOT",
    "find_project_root",
    "get_module_root",
    "get_processed_scenario_root",
    "get_raw_experiment_root",
    "get_raw_scenario_root",
    "get_task_paths",
    "get_task_paths_for_file",
    "legacy_raw_manifest",
]
