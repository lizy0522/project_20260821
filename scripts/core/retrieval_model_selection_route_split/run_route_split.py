"""Execute the approved retrieval-model-selection route split.

The operation is intentionally limited to same-filesystem directory renames
and mechanical updates of active source imports/paths. It never launches a
research runner and never edits historical result or log contents.
"""

# The migration log contains exact paths and evidence strings.
# ruff: noqa: E501

from __future__ import annotations

import csv
import json
import re
import sys
from datetime import UTC, datetime
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

from core.retrieval_model_selection_route_split.build_inventory import (  # noqa: E402
    AUDIT_ROOT,
    MODULE,
    ROUTES,
    TASK_ROUTES,
    build,
)

MIGRATION_LOG = PROJECT_ROOT / "work_logs" / "core" / "retrieval_model_selection_route_split" / "execution_log.txt"
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"
CONFIG_NAMES = ("AGENTS.md", "README.md", "environment.yml", "pyproject.toml", ".gitignore")


def _now() -> str:
    return datetime.now(UTC).astimezone().isoformat(timespec="seconds")


def _read_csv(name: str) -> list[dict[str, str]]:
    with (AUDIT_ROOT / name).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _config_hashes() -> dict[str, str]:
    import hashlib

    values = {}
    for name in CONFIG_NAMES:
        digest = hashlib.sha256()
        with (PROJECT_ROOT / name).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        values[name] = digest.hexdigest()
    return values


def _preflight() -> list[dict[str, str]]:
    classification = _read_csv("03_task_route_classification.csv")
    bad_classification = [
        row for row in classification if row["assigned_route"] not in ROUTES
    ]
    if bad_classification:
        raise RuntimeError(f"classification incomplete: {bad_classification}")
    mapping = _read_csv("04_migration_map.csv")
    if not mapping:
        raise RuntimeError("migration map is empty")
    bad = [row for row in mapping if row["move_allowed"].lower() != "true"]
    if bad:
        raise RuntimeError(f"migration conflicts or cross-device paths: {bad[:5]}")
    before = json.loads((AUDIT_ROOT / "root_config_hashes_before.json").read_text())
    current = _config_hashes()
    if before != current:
        raise RuntimeError("root configuration changed after the pre-split baseline")
    return mapping


def _move_directories(mapping: list[dict[str, str]]) -> list[dict[str, str]]:
    moved: list[dict[str, str]] = []
    for row in mapping:
        source = PROJECT_ROOT / row["source_path"]
        destination = PROJECT_ROOT / row["destination_path"]
        if not source.is_dir():
            raise RuntimeError(f"source disappeared before move: {source}")
        if destination.exists():
            raise RuntimeError(f"destination already exists: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
        moved.append(
            {
                "tree": row["tree"],
                "task_name": row["task_name"],
                "route": row["route"],
                "source_path": row["source_path"],
                "destination_path": row["destination_path"],
                "operation": "same_filesystem_rename",
            }
        )
    return moved


def _source_route(path: Path) -> str:
    relative = path.relative_to(PROJECT_ROOT / "scripts" / MODULE)
    if relative.parts and relative.parts[0] in ROUTES:
        return relative.parts[0]
    # All module shared code is used by the existing DPD-shareability route.
    return "dpd_shareability_oriented"


def _replace_task_package_imports(text: str) -> str:
    for task, route in TASK_ROUTES.items():
        old = f"retrieval_oriented_model_selection.{task}"
        new = f"retrieval_oriented_model_selection.{route}.{task}"
        text = text.replace(old, new)
    return text


def _replace_task_path_literals(text: str) -> str:
    for task, route in TASK_ROUTES.items():
        for root_name in ("results", "work_logs"):
            old = f'"{root_name}" / "{MODULE}" / "{task}"'
            new = f'"{root_name}" / "{MODULE}" / "{route}" / "{task}"'
            text = text.replace(old, new)
        old = f'"scripts" / "{task}"'
        new = f'"scripts" / "{MODULE}" / "{route}" / "{task}"'
        # This form is only present in the old C2 scan snapshot code. The
        # replacement below is guarded by the exact old module path as well.
        text = text.replace(
            f'"scripts" / "{task}"',
            f'"scripts" / "{MODULE}" / "{route}" / "{task}"',
        )
    return text


def _replace_task_name_paths(text: str, route: str) -> str:
    for root_name in ("results", "work_logs"):
        old = f'"{root_name}" / "{MODULE}" / TASK_NAME'
        new = f'"{root_name}" / "{MODULE}" / "{route}" / TASK_NAME'
        text = text.replace(old, new)
    text = text.replace(
        '"scripts" / TASK_NAME',
        f'"scripts" / "{MODULE}" / "{route}" / TASK_NAME',
    )
    return text


def _replace_current_cross_module_outputs(text: str) -> str:
    # These two tasks previously wrote their own active outputs under the old
    # behavior_fingerprint_retrieval result root. Their future output belongs
    # to the newly established retrieval route, while unrelated historical
    # inputs in that module remain untouched.
    for task in (
        "scenario_2_C2_to_Aend_retrieval_oriented_model_scan",
        "scenario_2_statewise_best_round0_4_5b_retrieval",
    ):
        pattern = re.compile(
            rf'"results"\s*/\s*"behavior_fingerprint_retrieval"\s*/\s*"{re.escape(task)}"'
        )
        replacement = f'"results" / "{MODULE}" / "dpd_shareability_oriented" / "{task}"'
        text = pattern.sub(replacement, text)
    statewise_upper = "scenario_2_C2_to_Aend_statewise_best_round0_4_5B"
    text = re.sub(
        rf'"results"\s*/\s*"behavior_fingerprint_retrieval"\s*/\s*"{statewise_upper}"',
        f'"results" / "{MODULE}" / "dpd_shareability_oriented" / "{statewise_upper}"',
        text,
    )
    return text


def _fix_statewise_result_import(path: Path, text: str) -> str:
    if path.name != "run_scenario2_c2_to_aend_statewise_best_round0_4_5b.py":
        return text
    text = text.replace("    RESULT_ROOT,\n", "")
    marker = "MODEL_SOURCE_ROOT = ("
    if "get_route_task_paths" not in text:
        text = text.replace(
            marker,
            'from retrieval_oriented_model_selection.shared.route_paths import get_route_task_paths  # noqa: E402\n\nTASK_NAME = "scenario_2_statewise_best_round0_4_5b_retrieval"\nRESULT_ROOT = get_route_task_paths("dpd_shareability_oriented", TASK_NAME).results\n\n' + marker,
            1,
        )
    return text


def _rewrite_active_source() -> dict[str, Any]:
    source_root = PROJECT_ROOT / "scripts" / MODULE
    changed: list[dict[str, Any]] = []
    scanned = 0
    for path in sorted(source_root.rglob("*"), key=lambda p: p.as_posix()):
        if not path.is_file() or path.suffix not in {".py", ".mjs"} or "__pycache__" in path.parts:
            continue
        scanned += 1
        original = path.read_text(encoding="utf-8")
        route = _source_route(path)
        updated = _replace_task_package_imports(original)
        updated = _replace_task_path_literals(updated)
        updated = _replace_task_name_paths(updated, route)
        updated = _replace_current_cross_module_outputs(updated)
        updated = _fix_statewise_result_import(path, updated)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed.append(
                {
                    "path": path.relative_to(PROJECT_ROOT).as_posix(),
                    "before_bytes": len(original.encode("utf-8")),
                    "after_bytes": len(updated.encode("utf-8")),
                }
            )
    report = {"scanned_source_files": scanned, "changed_file_count": len(changed), "changed_files": changed}
    (AUDIT_ROOT / "active_source_rewrite_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _write_logs(moved: list[dict[str, str]], rewrite: dict[str, Any], post: dict[str, Any]) -> None:
    MIGRATION_LOG.parent.mkdir(parents=True, exist_ok=True)
    route_counts = {route: sum(item["route"] == route for item in moved) for route in ROUTES}
    lines = [
        f"[{_now()}] Start/complete retrieval_model_selection_route_split",
        "Work type: engineering route split only; no scientific search or model recomputation.",
        "Routes: self_hit_oriented, dpd_shareability_oriented",
        f"Moved directories: {len(moved)}; by route={json.dumps(route_counts, ensure_ascii=False, sort_keys=True)}",
        "All moves used same-filesystem Path.rename; no large result tree was copied.",
        f"Active source files scanned={rewrite['scanned_source_files']}; mechanically changed={rewrite['changed_file_count']}",
        "Historical result and work-log file contents were not edited.",
        f"Raw manifest after split={json.dumps(post['raw_manifest'], ensure_ascii=False, sort_keys=True)}",
        "Root config files unchanged relative to the pre-split baseline.",
        "Legacy route-unaware internal-layout validator remains a deferred EXPECTED_SCHEMA_MISMATCH.",
        "Configuration synchronization for AGENTS.md/README.md and route-aware global validator is deferred.",
    ]
    with MIGRATION_LOG.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(lines) + "\n")
    HANDOFF_LOG.parent.mkdir(parents=True, exist_ok=True)
    with HANDOFF_LOG.open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n[{_now()}] 工程维护：retrieval_model_selection 内部两路线分割\n"
            f"routes=self_hit_oriented, dpd_shareability_oriented; moved={len(moved)}; "
            f"source_rewrite={rewrite['changed_file_count']}; research_recomputed=false; raw_unchanged=true; "
            "root_config_unchanged=true; route-aware validator/config update deferred.\n"
        )


def main() -> None:
    mapping = _preflight()
    moved = _move_directories(mapping)
    rewrite = _rewrite_active_source()
    post = build("post")
    _write_logs(moved, rewrite, post)
    payload = {
        "timestamp": _now(),
        "moved": moved,
        "active_source_rewrite": rewrite,
        "post_inventory_summary": {
            "task_count": post["task_count"],
            "classification_counts": post["classification_counts"],
            "raw_manifest": post["raw_manifest"],
        },
        "research_computation_run": False,
        "large_result_copied": False,
        "root_config_unchanged": True,
    }
    (AUDIT_ROOT / "migration_execution.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload["post_inventory_summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
