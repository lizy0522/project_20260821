"""Route-aware paths for the retrieval-oriented model-selection module."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MODULE_NAME = "retrieval_oriented_model_selection"
ROUTES = ("self_hit_oriented", "dpd_shareability_oriented")
SCENARIO_SCOPES = ("scenario_1", "scenario_2", "cross_scenario")

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)


@dataclass(frozen=True)
class RouteTaskPaths:
    """The mirrored paths for one route/scenario/task pair."""

    module: str
    route: str
    scenario_scope: str
    task: str
    scripts: Path
    results: Path
    work_logs: Path


def get_route_task_paths(
    route: str,
    scenario_scope: str,
    task: str | None = None,
) -> RouteTaskPaths:
    """Return the three paths for a route/scenario-scoped retrieval task."""

    # Backward-compatible interpretation for old callers during migration.
    if task is None:
        task = scenario_scope
        scenario_scope = "scenario_2"

    if route not in ROUTES:
        raise ValueError(f"unknown retrieval route: {route!r}")
    if scenario_scope not in SCENARIO_SCOPES:
        raise ValueError(f"unknown retrieval scenario scope: {scenario_scope!r}")
    if not task or "/" in task or "\\" in task:
        raise ValueError(f"invalid task name: {task!r}")
    return RouteTaskPaths(
        module=MODULE_NAME,
        route=route,
        scenario_scope=scenario_scope,
        task=task,
        scripts=PROJECT_ROOT / "scripts" / MODULE_NAME / route / scenario_scope / task,
        results=PROJECT_ROOT / "results" / MODULE_NAME / route / scenario_scope / task,
        work_logs=PROJECT_ROOT / "work_logs" / MODULE_NAME / route / scenario_scope / task,
    )


__all__ = [
    "MODULE_NAME",
    "PROJECT_ROOT",
    "ROUTES",
    "SCENARIO_SCOPES",
    "RouteTaskPaths",
    "get_route_task_paths",
]
