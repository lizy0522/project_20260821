"""Perform only the approved same-filesystem architecture renames."""

# ruff: noqa: E501

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
LOG_ROOT = PROJECT_ROOT / "work_logs" / "core" / "final_architecture_consolidation_20260921"
MODULE = "retrieval_oriented_model_selection"
ROUTES = ("self_hit_oriented", "dpd_shareability_oriented")


def _now() -> str:
    return datetime.now().astimezone().isoformat()


def _journal(record: dict[str, Any]) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    with (LOG_ROOT / "migration_journal.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp": _now(), **record}, ensure_ascii=False, default=str) + "\n")


def _rename(source: Path, target: Path, category: str) -> None:
    if source == target:
        return
    if source.exists() and target.exists():
        raise RuntimeError(f"both source and target exist: {source} -> {target}")
    if not source.exists() and target.exists():
        _journal({"event": "already_moved", "category": category, "source": str(source), "target": str(target)})
        return
    if not source.exists():
        raise FileNotFoundError(source)
    target.parent.mkdir(parents=True, exist_ok=True)
    if source.stat().st_dev != target.parent.stat().st_dev:
        raise RuntimeError(f"cross-filesystem rename refused: {source} -> {target}")
    _journal({"event": "rename_begin", "category": category, "source": str(source), "target": str(target)})
    source.rename(target)
    _journal({"event": "rename_complete", "category": category, "source": str(source), "target": str(target)})


def _ensure_init(path: Path, label: str) -> None:
    path.mkdir(parents=True, exist_ok=True)
    init = path / "__init__.py"
    if not init.exists():
        init.write_text(f'"""{label}."""\n', encoding="utf-8")
        _journal({"event": "create_scope_init", "path": str(init)})


def execute() -> dict[str, Any]:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    raw_source = PROJECT_ROOT / "data" / "raw" / "experiment_2026_0816"
    raw_target = PROJECT_ROOT / "data" / "raw" / "scenario_2" / "experiment_2026_0816"
    processed_root = PROJECT_ROOT / "data" / "processed"
    (PROJECT_ROOT / "data" / "raw" / "scenario_1").mkdir(parents=True, exist_ok=True)
    (PROJECT_ROOT / "data" / "raw" / "scenario_2").mkdir(parents=True, exist_ok=True)
    for scope in ("scenario_1", "scenario_2"):
        (processed_root / scope).mkdir(parents=True, exist_ok=True)
    _rename(raw_source, raw_target, "data_raw_experiment")

    moved = []
    for root_name in ("scripts", "results", "work_logs"):
        root = PROJECT_ROOT / root_name / MODULE
        for route in ROUTES:
            route_root = root / route
            if not route_root.exists():
                raise FileNotFoundError(route_root)
            _ensure_init(route_root / "scenario_2", f"{route} scenario_2 scope") if root_name == "scripts" else None
            tasks = sorted(
                path for path in route_root.iterdir()
                if path.is_dir() and path.name not in {"__pycache__", "scenario_2"}
            )
            for task in tasks:
                source = route_root / task.name
                target = route_root / "scenario_2" / task.name
                _rename(source, target, f"retrieval_{root_name}")
                moved.append({"root": root_name, "route": route, "task": task.name, "source": str(source), "target": str(target)})
    result = {
        "timestamp": _now(),
        "data_raw_source": str(raw_source),
        "data_raw_target": str(raw_target),
        "retrieval_moves": moved,
        "config_touched": False,
        "formal_research_started": False,
    }
    (LOG_ROOT / "migration_execution.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (LOG_ROOT / "execution_log.txt").open("a", encoding="utf-8") as handle:
        handle.write(f"{_now()} | Phase 5-8 atomic rename complete; retrieval_moves={len(moved)}; config_touched=False; formal_research_started=False\n")
    return result


if __name__ == "__main__":
    print(json.dumps(execute(), ensure_ascii=False, indent=2))
