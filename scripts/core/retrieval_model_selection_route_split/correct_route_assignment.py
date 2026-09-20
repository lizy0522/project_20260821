"""Correct the historical retrieval route ownership in-place.

This is a continuation of the existing route-split maintenance task. The
user-frozen rule is that all pre-existing retrieval-oriented model-selection
tasks belong to ``self_hit_oriented``; the DPD route remains reserved for
future shareability-first tasks.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from datetime import datetime
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
OLD_ROUTE = "dpd_shareability_oriented"
NEW_ROUTE = "self_hit_oriented"
ROUTES = (NEW_ROUTE, OLD_ROUTE)
AUDIT_ROOT = PROJECT_ROOT / "results" / "core" / "retrieval_model_selection_route_split"
LOG_PATH = PROJECT_ROOT / "work_logs" / "core" / "retrieval_model_selection_route_split" / "execution_log.txt"
HANDOFF_PATH = PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"
CONFIG_NAMES = ("AGENTS.md", "README.md", "environment.yml", "pyproject.toml", ".gitignore")

TASKS = (
    "scenario_2_C2_to_Aend_retrieval_oriented_model_scan",
    "scenario_2_mp10_k9_retrieval_oriented_basis_ablation_5b",
    "scenario_2_mp10_k11_retrieval_oriented_forward_scan_5b",
    "scenario_2_k11_backward_retrieval_oriented_basis_ablation_5b",
    "scenario_2_k12_retrieval_oriented_forward_scan_5b",
    "scenario_2_k12_backward_retrieval_oriented_basis_ablation_5b",
    "scenario_2_smallbeam3_k13_retrieval_oriented_forward_expansion_5b",
    "scenario_2_beam3_k13_floating_backward_retrieval_ablation_5b",
    "scenario_2_updated_beam3_iteration2_retrieval_oriented_forward_expansion_5b",
    "scenario_2_self_first_joint_basis_ridge_lut_retrieval_search_5b",
    "scenario_2_statewise_best_round0_4_5b_retrieval",
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
    return sorted(
        (p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"),
        key=lambda p: p.as_posix(),
    ) if path.is_dir() else []


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
        "exists": path.is_dir(),
        "file_count": len(files),
        "total_bytes": sum(p.stat().st_size for p in files),
        "relative_file_paths": [p.relative_to(path).as_posix() for p in files],
        "critical_sha256": critical,
    }


def _task_path(tree: str, route: str, task: str) -> Path:
    return PROJECT_ROOT / tree / MODULE / route / task


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


def _inventory() -> dict[str, Any]:
    trees = _task_sets()
    detail = {}
    for tree in ("scripts", "results", "work_logs"):
        for route in ROUTES:
            for task in trees[tree][route]:
                path = _task_path(tree, route, task)
                detail[f"{tree}/{route}/{task}"] = {
                    "path": _relative(path),
                    "file_count": _manifest(path)["file_count"],
                    "total_bytes": _manifest(path)["total_bytes"],
                }
    return {"task_sets": trees, "detail": detail}


def _task_manifests() -> dict[str, Any]:
    result = {}
    for tree in ("scripts", "results", "work_logs"):
        for task in TASKS:
            route = NEW_ROUTE if task == "scenario_2_self_first_joint_basis_ridge_lut_retrieval_search_5b" else OLD_ROUTE
            path = _task_path(tree, route, task)
            if path.is_dir():
                result[f"{tree}/{task}"] = _manifest(path)
    return result


def _write_json(name: str, payload: Any) -> None:
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    (AUDIT_ROOT / name).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _build_pre_inventory() -> None:
    inventory = _inventory()
    payload = {
        "maintenance_mode": "continuous_corrective_migration",
        "module": MODULE,
        "previous_route_classification": {task: (NEW_ROUTE if task == "scenario_2_self_first_joint_basis_ridge_lut_retrieval_search_5b" else OLD_ROUTE) for task in TASKS},
        "corrected_route_classification": {task: NEW_ROUTE for task in TASKS},
        "inventory": inventory,
        "raw_manifest": _raw_manifest(),
        "root_config_sha256": _config_hashes(),
        "routes": list(ROUTES),
        "task_count": len(TASKS),
    }
    _write_json("corrective_pre_migration_inventory.json", payload)
    _write_json("corrective_root_config_hashes_before.json", _config_hashes())
    _write_json("corrective_pre_task_manifests.json", _task_manifests())

    with (AUDIT_ROOT / "08_corrected_task_route_classification.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "task_name",
            "previous_route",
            "corrected_route",
            "correction_reason",
            "user_defined_research_objective",
            "historical_auxiliary_metrics",
            "status",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for task in TASKS:
            previous = NEW_ROUTE if task == "scenario_2_self_first_joint_basis_ridge_lut_retrieval_search_5b" else OLD_ROUTE
            writer.writerow(
                {
                    "task_name": task,
                    "previous_route": previous,
                    "corrected_route": NEW_ROUTE,
                    "correction_reason": "Explicit user research definition overrides inference from historical auxiliary metrics.",
                    "user_defined_research_objective": "Improve LUT self-hit State_Q == State_R success rate.",
                    "historical_auxiliary_metrics": "N_valid; Real-B pass; shareability margin; MRR; Top-k may remain auxiliary/constraint/diagnostic.",
                    "status": "FROZEN_CORRECTED_ASSIGNMENT",
                }
            )

    rows = []
    for tree in ("scripts", "results", "work_logs"):
        for task in TASKS:
            if task == "scenario_2_self_first_joint_basis_ridge_lut_retrieval_search_5b":
                continue
            source = _task_path(tree, OLD_ROUTE, task)
            destination = _task_path(tree, NEW_ROUTE, task)
            if not source.is_dir():
                continue
            parent = destination.parent
            same_device = source.stat().st_dev == parent.stat().st_dev
            rows.append(
                {
                    "tree": tree,
                    "task_name": task,
                    "source_path": _relative(source),
                    "destination_path": _relative(destination),
                    "source_exists": True,
                    "destination_exists": destination.exists(),
                    "same_filesystem": same_device,
                    "move_allowed": bool(same_device and not destination.exists()),
                }
            )
    with (AUDIT_ROOT / "09_corrective_migration_map.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["tree", "task_name", "source_path", "destination_path", "source_exists", "destination_exists", "same_filesystem", "move_allowed"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _read_map() -> list[dict[str, str]]:
    with (AUDIT_ROOT / "09_corrective_migration_map.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    bad = [row for row in rows if row["move_allowed"].lower() != "true"]
    if bad:
        raise RuntimeError(f"corrective migration conflict: {bad[:5]}")
    return rows


def _rewrite_active_source() -> dict[str, Any]:
    source_root = PROJECT_ROOT / "scripts" / MODULE
    changed = []
    scanned = 0
    for path in sorted(source_root.rglob("*"), key=lambda p: p.as_posix()):
        if not path.is_file() or path.suffix not in {".py", ".mjs"} or "__pycache__" in path.parts:
            continue
        scanned += 1
        original = path.read_text(encoding="utf-8")
        updated = original
        for task in TASKS:
            updated = updated.replace(
                f"retrieval_oriented_model_selection.{OLD_ROUTE}.{task}",
                f"retrieval_oriented_model_selection.{NEW_ROUTE}.{task}",
            )
            updated = re.sub(
                rf'"{OLD_ROUTE}"\s*/\s*"{re.escape(task)}"',
                f'"{NEW_ROUTE}" / "{task}"',
                updated,
            )
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed.append(
                {
                    "path": _relative(path),
                    "before_bytes": len(original.encode("utf-8")),
                    "after_bytes": len(updated.encode("utf-8")),
                }
            )
    report = {"files_scanned": scanned, "files_modified": len(changed), "changed_files": changed}
    _write_json("corrective_active_source_rewrite_report.json", report)
    return report


def _append_logs(moved: list[dict[str, str]], rewrite: dict[str, Any]) -> None:
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n[{timestamp}] Corrective route ownership maintenance\n"
            "Previous automatic classification corrected by explicit user research definition.\n"
            "Corrected assignment: all 11 historical retrieval-oriented tasks -> self_hit_oriented; "
            "dpd_shareability_oriented reserved and empty.\n"
            f"Moved directories={len(moved)}; active source files scanned={rewrite['files_scanned']}; "
            f"active source files modified={rewrite['files_modified']}; scientific_computation=false.\n"
            "Historical results, summaries, checkpoints, validation JSON and execution-log bodies were not edited.\n"
            "Root configuration synchronization deferred.\n"
        )
    HANDOFF_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HANDOFF_PATH.open("a", encoding="utf-8") as handle:
        handle.write(
            f"\n{timestamp} | 工程维护：纠正 retrieval_model_selection route ownership\n"
            "上一轮按历史辅助指标推断的 route 分类已按用户明确研究目的纠正：全部 11 个历史任务归入 self_hit_oriented；"
            "dpd_shareability_oriented 保留为空，留给未来 shareability-first 新任务。\n"
        )


def _execute() -> None:
    before = json.loads((AUDIT_ROOT / "corrective_pre_migration_inventory.json").read_text())
    current_hashes = _config_hashes()
    if current_hashes != before["root_config_sha256"]:
        raise RuntimeError("root configuration changed after corrective pre-inventory")
    moved = []
    for row in _read_map():
        source = PROJECT_ROOT / row["source_path"]
        destination = PROJECT_ROOT / row["destination_path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
        moved.append({**row, "operation": "same_filesystem_rename"})
    rewrite = _rewrite_active_source()
    post = {
        "inventory": _inventory(),
        "raw_manifest": _raw_manifest(),
        "root_config_sha256": _config_hashes(),
        "moved": moved,
    }
    _write_json("11_corrected_post_split_inventory.json", post)
    _write_json(
        "10_corrective_migration_execution.json",
        {
            "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
            "moved": moved,
            "active_source_rewrite": rewrite,
            "scientific_computation_run": False,
            "large_result_copied": False,
            "root_config_unchanged": post["root_config_sha256"] == before["root_config_sha256"],
        },
    )
    _append_logs(moved, rewrite)
    print(json.dumps({"moved": len(moved), "active_source_files_modified": rewrite["files_modified"]}, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pre", "execute"), required=True)
    args = parser.parse_args()
    if args.phase == "pre":
        _build_pre_inventory()
        print(json.dumps(_inventory(), ensure_ascii=False, indent=2))
    else:
        _execute()


if __name__ == "__main__":
    main()
