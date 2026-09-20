"""Validator for the ten-module scenario-aware project layout."""

# ruff: noqa: E501

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

from .module_registry import MODULES, ROOT_NAMES, validate_module_root_alignment

ROUTED_MODULE = "retrieval_oriented_model_selection"
ROUTES = ("self_hit_oriented", "dpd_shareability_oriented")
SCENARIO_SCOPES = ("scenario_1", "scenario_2", "cross_scenario")
ENGINEERING_ROOT_MODULES = {"core", "data_management"}


def _owner(parts: tuple[str, ...]) -> tuple[str, ...] | None:
    """Map an import path to its module/shared/task ownership."""

    if not parts or parts[0] not in MODULES or len(parts) < 2:
        return None
    module = parts[0]
    if parts[1] == "shared":
        return (module, "shared")
    if module == ROUTED_MODULE:
        if len(parts) >= 4 and parts[1] in ROUTES and parts[2] in SCENARIO_SCOPES:
            return (module, parts[1], parts[2], parts[3])
        return None
    if len(parts) >= 3 and parts[1] in SCENARIO_SCOPES:
        return (module, parts[1], parts[2])
    return (module, parts[1])


def _import_violations(scripts: Path) -> list[dict[str, str]]:
    violations = []
    for path in scripts.rglob("*.py"):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(scripts).with_suffix("")
        source = _owner(rel.parts)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:
                    prefix = rel.parts[:-1]
                    prefix = prefix[: len(prefix) - node.level + 1]
                    names = [".".join((*prefix, *(node.module or "").split(".")))]
                elif node.module:
                    names = [node.module]
            for name in names:
                target = _owner(tuple(name.split(".")))
                if target is None or source == target or target[-1] == "shared":
                    continue
                if source is None:
                    continue
                kind = "shared_to_task" if source[-1] == "shared" else "task_to_task"
                violations.append({"type": kind, "source": rel.as_posix(), "target": name})
    return sorted(violations, key=lambda row: (row["source"], row["target"]))


def _scenario_tasks(base: Path) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for scope in SCENARIO_SCOPES:
        scope_root = base / scope
        if not scope_root.is_dir():
            continue
        result[scope] = sorted(
            path.name
            for path in scope_root.iterdir()
            if path.is_dir() and path.name != "__pycache__"
        )
    return result


def validate_project_layout(root: Path | str) -> dict[str, Any]:
    """Validate mirrored roots, ordinary scenario scopes, routes, and dependencies."""

    project = Path(root).resolve()
    mirror = validate_module_root_alignment(project)
    violations: list[dict[str, str]] = []
    task_sets: dict[str, dict[str, list[str]]] = {name: {} for name in ROOT_NAMES}

    for tree in ROOT_NAMES:
        for module in MODULES:
            base = project / tree / module
            if tree != "scripts" and (base / "shared").exists():
                violations.append({"type": "result_log_shared", "path": str(base / "shared")})

            if module == ROUTED_MODULE:
                allowed = set(ROUTES) | ({"shared"} if tree == "scripts" else set())
                children = {
                    path.name
                    for path in base.iterdir()
                    if path.is_dir() and path.name != "__pycache__"
                }
                for extra in sorted(children - allowed):
                    violations.append({"type": "flat_or_extra_route", "path": str(base / extra)})
                for route in ROUTES:
                    route_root = base / route
                    if not route_root.is_dir():
                        violations.append({"type": "missing_route", "path": str(route_root)})
                        continue
                    if tree == "scripts" and not (route_root / "__init__.py").is_file():
                        violations.append({"type": "missing_route_init", "path": str(route_root)})
                    if (route_root / "shared").exists():
                        violations.append({"type": "route_shared", "path": str(route_root / "shared")})
                    flat_route_tasks = [
                        path
                        for path in route_root.iterdir()
                        if path.is_dir()
                        and path.name not in {"__pycache__", *SCENARIO_SCOPES}
                    ]
                    for path in flat_route_tasks:
                        violations.append({"type": "flat_task", "path": str(path)})
                    for scope in SCENARIO_SCOPES:
                        scope_root = route_root / scope
                        if not scope_root.is_dir():
                            continue
                        if tree == "scripts" and not (scope_root / "__init__.py").is_file():
                            violations.append({"type": "missing_scope_init", "path": str(scope_root)})
                        tasks = sorted(
                            path.name
                            for path in scope_root.iterdir()
                            if path.is_dir() and path.name != "__pycache__"
                        )
                        task_sets[tree][f"{module}/{route}/{scope}"] = tasks
                        if tree == "scripts":
                            for task in tasks:
                                if not (scope_root / task / "__init__.py").is_file():
                                    violations.append({"type": "missing_task_init", "path": str(scope_root / task)})
            elif module in ENGINEERING_ROOT_MODULES:
                tasks = sorted(
                    path.name
                    for path in base.iterdir()
                    if path.is_dir() and path.name not in {"__pycache__", "shared"}
                )
                task_sets[tree][module] = tasks
                if tree == "scripts":
                    for task in tasks:
                        if not (base / task / "__init__.py").is_file():
                            violations.append({"type": "missing_task_init", "path": str(base / task)})
            else:
                scoped_tasks = _scenario_tasks(base)
                for scope, tasks in scoped_tasks.items():
                    scope_root = base / scope
                    task_sets[tree][f"{module}/{scope}"] = tasks
                    if tree == "scripts":
                        if not (scope_root / "__init__.py").is_file():
                            violations.append({"type": "missing_scope_init", "path": str(scope_root)})
                        for task in tasks:
                            if not (scope_root / task / "__init__.py").is_file():
                                violations.append({"type": "missing_task_init", "path": str(scope_root / task)})
                if tree == "scripts":
                    flat = [
                        path
                        for path in base.iterdir()
                        if path.is_dir() and path.name not in {"__pycache__", "shared", *SCENARIO_SCOPES}
                    ]
                    for path in flat:
                        violations.append({"type": "flat_task", "path": str(path)})

            if tree == "scripts":
                if not (base / "shared" / "__init__.py").is_file():
                    violations.append({"type": "missing_module_shared", "path": str(base)})
                for file in base.iterdir():
                    if file.is_file() and file.name != "__init__.py":
                        violations.append({"type": "module_root_file", "path": str(file)})

    owners: dict[str, set[str]] = {}
    for tree in ROOT_NAMES:
        for key, tasks in task_sets[tree].items():
            if not key.startswith(f"{ROUTED_MODULE}/"):
                continue
            route_scope = key.split("/")
            for task in tasks:
                owners.setdefault(task, set()).add("/".join(route_scope[1:]))
    for task, locations in owners.items():
        if len(locations) != 1:
            violations.append({"type": "task_in_multiple_routes_or_scopes", "path": task})

    imports = _import_violations(project / "scripts")
    return {
        "module_count": len(MODULES),
        "module_mirror": mirror,
        "routes": list(ROUTES),
        "scenario_scopes": list(SCENARIO_SCOPES),
        "task_sets": task_sets,
        "layout_violations": violations,
        "dependency_violations": imports,
        "pass": bool(mirror["pass"] and not violations and not imports),
    }


__all__ = ["ROUTED_MODULE", "ROUTES", "SCENARIO_SCOPES", "validate_project_layout"]
