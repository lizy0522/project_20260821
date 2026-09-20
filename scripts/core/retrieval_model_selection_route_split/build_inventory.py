"""Build pre/post inventories for the retrieval-model-selection route split.

This task records filesystem facts and classification evidence. It never runs a
scientific model, reads MAT payloads, or modifies the source/result trees other
than writing its own audit artifacts under ``results/core``.
"""

# The audit embeds long evidence strings and exact historical paths.
# ruff: noqa: E501

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
MODULE = "retrieval_oriented_model_selection"
AUDIT_ROOT = PROJECT_ROOT / "results" / "core" / "retrieval_model_selection_route_split"
ROOT_NAMES = ("scripts", "results", "work_logs")
ROUTES = ("self_hit_oriented", "dpd_shareability_oriented")

TASK_ROUTES = {
    "scenario_2_C2_to_Aend_retrieval_oriented_model_scan": "dpd_shareability_oriented",
    "scenario_2_mp10_k9_retrieval_oriented_basis_ablation_5b": "dpd_shareability_oriented",
    "scenario_2_mp10_k11_retrieval_oriented_forward_scan_5b": "dpd_shareability_oriented",
    "scenario_2_k11_backward_retrieval_oriented_basis_ablation_5b": "dpd_shareability_oriented",
    "scenario_2_k12_retrieval_oriented_forward_scan_5b": "dpd_shareability_oriented",
    "scenario_2_k12_backward_retrieval_oriented_basis_ablation_5b": "dpd_shareability_oriented",
    "scenario_2_smallbeam3_k13_retrieval_oriented_forward_expansion_5b": "dpd_shareability_oriented",
    "scenario_2_beam3_k13_floating_backward_retrieval_ablation_5b": "dpd_shareability_oriented",
    "scenario_2_updated_beam3_iteration2_retrieval_oriented_forward_expansion_5b": "dpd_shareability_oriented",
    "scenario_2_statewise_best_round0_4_5b_retrieval": "dpd_shareability_oriented",
    "scenario_2_self_first_joint_basis_ridge_lut_retrieval_search_5b": "self_hit_oriented",
}

CLASSIFICATION_EVIDENCE = {
    "scenario_2_C2_to_Aend_retrieval_oriented_model_scan": (
        "dpd_shareability_oriented",
        "Real-B shareability is used for model ranking.",
        "scripts/retrieval_oriented_model_selection/scenario_2_C2_to_Aend_retrieval_oriented_model_scan/test_scenario2_retrieval_oriented_model_scan.py:123",
        'assert payload["real_B_used_for_ranking"] is True',
    ),
    "scenario_2_mp10_k9_retrieval_oriented_basis_ablation_5b": (
        "dpd_shareability_oriented",
        "Primary ranking is Top-1 Real-B pass, followed by shareability metrics.",
        "results/retrieval_oriented_model_selection/scenario_2_mp10_k9_retrieval_oriented_basis_ablation_5b/00_task_definition.txt",
        "Primary ranking: Top-1 Real-B pass, non-self pass rate, Q05 shareability margin, MRR, Top-3 oracle count, then smaller K.",
    ),
    "scenario_2_mp10_k11_retrieval_oriented_forward_scan_5b": (
        "dpd_shareability_oriented",
        "Primary ranking is Top-1 Real-B pass, not N_self.",
        "results/retrieval_oriented_model_selection/scenario_2_mp10_k11_retrieval_oriented_forward_scan_5b/21_final_result_summary.txt",
        "rank=1 MP10_plus_ENV_p04_m2_q0 ... Top1=416/425 ... recovered=5, regressed=0",
    ),
    "scenario_2_k11_backward_retrieval_oriented_basis_ablation_5b": (
        "dpd_shareability_oriented",
        "The ablation asks whether the Real-B retrieval pass count is preserved.",
        "results/retrieval_oriented_model_selection/scenario_2_k11_backward_retrieval_oriented_basis_ablation_5b/00_task_definition.txt",
        "Primary result: whether any single deletion improves or safely preserves 416/425.",
    ),
    "scenario_2_k12_retrieval_oriented_forward_scan_5b": (
        "dpd_shareability_oriented",
        "Primary ranking is Top-1 Real-B pass and shareability margin.",
        "results/retrieval_oriented_model_selection/scenario_2_k12_retrieval_oriented_forward_scan_5b/00_task_definition.txt",
        "Primary ranking: Top-1 Real-B pass, non-self pass rate, Q05 margin, MRR, Top-3 oracle, then K.",
    ),
    "scenario_2_k12_backward_retrieval_oriented_basis_ablation_5b": (
        "dpd_shareability_oriented",
        "The ablation asks whether the Real-B retrieval pass count is preserved.",
        "results/retrieval_oriented_model_selection/scenario_2_k12_backward_retrieval_oriented_basis_ablation_5b/00_task_definition.txt",
        "Primary question: whether any single deletion improves or safely preserves 419/425.",
    ),
    "scenario_2_smallbeam3_k13_retrieval_oriented_forward_expansion_5b": (
        "dpd_shareability_oriented",
        "Primary ranking is Top-1 Real-B pass and shareability margin.",
        "results/retrieval_oriented_model_selection/scenario_2_smallbeam3_k13_retrieval_oriented_forward_expansion_5b/00_task_definition.txt",
        "Primary ranking: Top-1 Real-B pass, non-self pass rate, Q05 shareability margin, MRR, Top-3 oracle, then K/model ID.",
    ),
    "scenario_2_beam3_k13_floating_backward_retrieval_ablation_5b": (
        "dpd_shareability_oriented",
        "The final comparison is driven by Top1 retrieval and shareability margin.",
        "results/retrieval_oriented_model_selection/scenario_2_beam3_k13_floating_backward_retrieval_ablation_5b/30_final_result_summary.txt",
        "best global support=child_K12_13 Top1=421/425; Q05 and MRR are shareability diagnostics.",
    ),
    "scenario_2_updated_beam3_iteration2_retrieval_oriented_forward_expansion_5b": (
        "dpd_shareability_oriented",
        "The global winner is selected by Top1 Real-B retrieval performance.",
        "results/retrieval_oriented_model_selection/scenario_2_updated_beam3_iteration2_retrieval_oriented_forward_expansion_5b/36_final_result_summary.txt",
        "Global best: F2_safe_K13__plus_ENV_p06_m1_q0, Top1=422/425.",
    ),
    "scenario_2_statewise_best_round0_4_5b_retrieval": (
        "dpd_shareability_oriented",
        "The upstream statewise model pool is selected using the joint -40 dB DPD-shareability feasibility rule; retrieval then evaluates shareability.",
        "scripts/retrieval_oriented_model_selection/scenario_2_statewise_best_round0_4_5b_retrieval/run_scenario2_c2_to_aend_statewise_best_round0_4_5b.py:668-680",
        'feasible_selection_rule="minimum_complexity_among_joint_lt_minus40"; top1_selection_source="fingerprint CNMSE only"',
    ),
    "scenario_2_self_first_joint_basis_ridge_lut_retrieval_search_5b": (
        "self_hit_oriented",
        "The task explicitly declares the Self-First objective and ranks N_self first.",
        "results/retrieval_oriented_model_selection/scenario_2_self_first_joint_basis_ridge_lut_retrieval_search_5b/00_task_definition.txt",
        "OBJECTIVE_VERSION=SELF_FIRST_V1; Primary objective: maximize Top-1 SELF retrieval; Ranking: N_self, N_valid, ...",
    ),
}


def _relative(path: Path) -> str:
    return path.relative_to(PROJECT_ROOT).as_posix()


def _is_ignored(path: Path) -> bool:
    return "__pycache__" in path.parts or path.suffix.lower() == ".pyc"


def _iter_files(root: Path) -> list[Path]:
    if not root.is_dir():
        return []
    return sorted((p for p in root.rglob("*") if p.is_file() and not _is_ignored(p)), key=lambda p: p.as_posix())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _manifest(path: Path) -> dict[str, Any]:
    files = _iter_files(path)
    critical_names = {
        "00_task_definition.txt",
        "execution_log.txt",
        "final_result_summary.txt",
        "22_final_result_summary.txt",
        "30_final_result_summary.txt",
        "31_final_result_summary.txt",
        "36_final_result_summary.txt",
        "19_final_result_summary.txt",
        "21_final_result_summary.txt",
        "26_final_result_summary.txt",
        "22_final_result_summary.json",
        "23_checkpoint.json",
        "24_validation_checks.json",
        "32_validation_checks.json",
        "33_validation_checks.json",
        "38_validation_checks.json",
    }
    critical = {}
    for file in files:
        if file.name in critical_names:
            critical[file.relative_to(path).as_posix()] = _sha256(file)
    return {
        "exists": path.is_dir(),
        "file_count": len(files),
        "total_bytes": sum(int(file.stat().st_size) for file in files),
        "relative_file_paths": [file.relative_to(path).as_posix() for file in files],
        "critical_sha256": critical,
    }


def _tree_inventory(root_name: str) -> dict[str, Any]:
    root = PROJECT_ROOT / root_name / MODULE
    tasks = {}
    if root.is_dir():
        candidates: list[tuple[str, Path, str | None]] = []
        for item in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if not item.is_dir() or item.name in {"shared", "__pycache__"}:
                continue
            if item.name in ROUTES:
                for task_dir in sorted(item.iterdir(), key=lambda p: p.name.lower()):
                    if task_dir.is_dir() and task_dir.name != "__pycache__":
                        candidates.append((task_dir.name, task_dir, item.name))
            else:
                candidates.append((item.name, item, None))
        for task_name, item, route in candidates:
            tasks[item.name] = {
                "path": _relative(item),
                "file_count": _manifest(item)["file_count"],
                "total_bytes": _manifest(item)["total_bytes"],
                "has_final_summary": any("final_result_summary" in p.name for p in item.iterdir()),
                "has_checkpoint": any("checkpoint" in p.name for p in item.iterdir()),
                "has_execution_log": (item / "execution_log.txt").is_file(),
                "route": route,
            }
    return {
        "root": _relative(root),
        "shared_exists": (root / "shared").is_dir(),
        "tasks": tasks,
    }


def _raw_manifest() -> dict[str, Any]:
    root = PROJECT_ROOT / "data" / "raw"
    files = sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.as_posix())
    digest = hashlib.sha256()
    total = 0
    mat_count = 0
    for file in files:
        rel = file.relative_to(root).as_posix().encode()
        size = int(file.stat().st_size)
        digest.update(rel + b"\0")
        digest.update(size.to_bytes(8, "little"))
        total += size
        mat_count += int(file.suffix.lower() == ".mat")
    return {
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "mat_count": mat_count,
        "bytes": total,
    }


def _config_hashes() -> dict[str, str]:
    names = ("AGENTS.md", "README.md", "environment.yml", "pyproject.toml", ".gitignore")
    return {name: _sha256(PROJECT_ROOT / name) for name in names}


def _git_status() -> dict[str, Any]:
    result = subprocess.run(
        ["git", "status", "--short", "--branch"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    return {"returncode": result.returncode, "stdout": result.stdout}


def _presence_rows(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    tasks = sorted({task for tree in inventory.values() for task in tree["tasks"]})
    rows = []
    for task in tasks:
        row = {"task_name": task}
        for tree in ROOT_NAMES:
            item = inventory[tree]["tasks"].get(task)
            row[f"{tree}_exists"] = bool(item)
            row[f"{tree}_file_count"] = item["file_count"] if item else 0
            row[f"{tree}_total_bytes"] = item["total_bytes"] if item else 0
            row[f"{tree}_final_summary_exists"] = bool(item and item["has_final_summary"])
            row[f"{tree}_checkpoint_exists"] = bool(item and item["has_checkpoint"])
            row[f"{tree}_execution_log_exists"] = bool(item and item["has_execution_log"])
        rows.append(row)
    return rows


def _classification_rows(tasks: list[str]) -> list[dict[str, Any]]:
    rows = []
    for task in tasks:
        if task not in CLASSIFICATION_EVIDENCE:
            rows.append(
                {
                    "task_name": task,
                    "primary_objective": "unclassified",
                    "objective_evidence": "",
                    "objective_metric": "",
                    "assigned_route": "unclassified",
                    "classification_confidence": "none",
                    "notes": "No evidence rule registered; migration must stop.",
                }
            )
            continue
        route, objective, evidence, quote = CLASSIFICATION_EVIDENCE[task]
        metric = "N_self / Exact" if route == "self_hit_oriented" else "Top-1 Real-B pass / DPD shareability"
        rows.append(
            {
                "task_name": task,
                "primary_objective": objective,
                "objective_evidence": evidence,
                "objective_metric": metric,
                "assigned_route": route,
                "classification_confidence": "high",
                "notes": quote,
            }
        )
    return rows


def _mapping_rows(inventory: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for tree in ROOT_NAMES:
        for task in sorted(inventory[tree]["tasks"]):
            route = TASK_ROUTES.get(task, "unclassified")
            source = PROJECT_ROOT / tree / MODULE / task
            destination = PROJECT_ROOT / tree / MODULE / route / task
            # The route container does not exist during the pre-split audit.
            # The module root is the nearest existing parent and is on the same
            # filesystem as any subsequently-created route container.
            destination_parent = destination.parent
            while not destination_parent.exists():
                destination_parent = destination_parent.parent
            same_device = destination_parent.stat().st_dev == source.stat().st_dev
            rows.append(
                {
                    "tree": tree,
                    "task_name": task,
                    "route": route,
                    "source_path": _relative(source),
                    "destination_path": _relative(destination),
                    "source_exists": source.is_dir(),
                    "destination_exists": destination.exists(),
                    "source_device": source.stat().st_dev,
                    "destination_parent_device": destination_parent.stat().st_dev,
                    "same_filesystem": same_device,
                    "move_allowed": bool(route in ROUTES and source.is_dir() and not destination.exists() and same_device),
                }
            )
    return rows


def _task_manifests() -> dict[str, Any]:
    payload = {}
    for tree in ROOT_NAMES:
        base = PROJECT_ROOT / tree / MODULE
        for task, route in TASK_ROUTES.items():
            source = base / task
            if not source.is_dir():
                source = base / route / task
            if source.is_dir():
                payload[f"{tree}/{task}"] = _manifest(source)
    return payload


def build(phase: str) -> dict[str, Any]:
    inventory = {tree: _tree_inventory(tree) for tree in ROOT_NAMES}
    tasks = sorted({task for tree in inventory.values() for task in tree["tasks"]})
    classifications = _classification_rows(tasks)
    payload = {
        "phase": phase,
        "project_root": str(PROJECT_ROOT),
        "module": MODULE,
        "routes": list(ROUTES),
        "tasks": tasks,
        "inventory": inventory,
        "task_count": len(tasks),
        "classification_complete": all(row["assigned_route"] in ROUTES for row in classifications),
        "classification_counts": {
            route: sum(row["assigned_route"] == route for row in classifications) for route in ROUTES
        },
        "raw_manifest": _raw_manifest(),
        "root_config_sha256": _config_hashes(),
        "git_status": _git_status(),
    }
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    inventory_name = "01_pre_split_inventory.json" if phase == "pre" else "05_post_split_inventory.json"
    manifest_name = "pre_task_manifests.json" if phase == "pre" else "post_task_manifests.json"
    (AUDIT_ROOT / inventory_name).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (AUDIT_ROOT / manifest_name).write_text(json.dumps(_task_manifests(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if phase == "pre":
        with (AUDIT_ROOT / "02_task_presence_matrix.csv").open("w", encoding="utf-8", newline="") as handle:
            rows = _presence_rows(inventory)
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["task_name"])
            writer.writeheader()
            writer.writerows(rows)
        with (AUDIT_ROOT / "03_task_route_classification.csv").open("w", encoding="utf-8", newline="") as handle:
            rows = _classification_rows(tasks)
            fields = ["task_name", "primary_objective", "objective_evidence", "objective_metric", "assigned_route", "classification_confidence", "notes"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        with (AUDIT_ROOT / "04_migration_map.csv").open("w", encoding="utf-8", newline="") as handle:
            rows = _mapping_rows(inventory)
            fields = list(rows[0]) if rows else ["tree", "task_name"]
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        (AUDIT_ROOT / "root_config_hashes_before.json").write_text(
            json.dumps(_config_hashes(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("pre", "post"), required=True)
    args = parser.parse_args()
    payload = build(args.phase)
    print(json.dumps({key: payload[key] for key in ("phase", "task_count", "classification_complete", "classification_counts", "raw_manifest")}, ensure_ascii=False, indent=2))
    if args.phase == "pre" and not payload["classification_complete"]:
        raise SystemExit("classification incomplete; migration not allowed")


if __name__ == "__main__":
    main()
