# ruff: noqa: E501

"""Execute and validate the 2026-09-17 ten-module layout migration.

The migration is deliberately rename-only for existing assets.  It records
the pre-migration inventory, classification, move plan, protected hashes, and
post-migration validation so the 30 GB Self-First result tree is never copied
or regenerated as part of this engineering task.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import random
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
TASK_NAME = "module_layout_reorganization_20260917"
RESULT_ROOT = PROJECT_ROOT / "results" / "core" / TASK_NAME
LOG_ROOT = PROJECT_ROOT / "work_logs" / "core" / TASK_NAME
EXECUTION_LOG = LOG_ROOT / "execution_log.txt"

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

EXPECTED_RAW = {
    "sha256": "910b06e4be6b60b1f4dc63409a3083eb328765347fbec3b89a97a41d6633a1e0",
    "file_count": 429,
    "mat_count": 427,
    "bytes": 2_258_448_137,
}

BEHAVIOR_MODELING_TASKS = (
    "scenario_2_all425_c2_envelope23_ridge_generalization_5B",
    "scenario_2_all425_c2_ilcend_envelope19_shared_coverage_5B",
    "scenario_2_all425_envelope18_coverage_evaluation_5B",
    "scenario_2_all425_ilcend_envelope23_coverage_evaluation_5B",
    "scenario_2_gateb48_unified_basis_refinement_5B",
    "scenario_2_hard20_c2_ilcend_shared_envelope75_basis_selection_5B",
    "scenario_2_hard20_envelope75_basis_selection_5B",
    "scenario_2_hard20_ilcend_envelope75_basis_selection_5B",
)

BEHAVIOR_FINGERPRINT_RETRIEVAL_TASKS = (
    "scenario_2_envelope18_commonB_full_lut_retrieval_5B",
    "scenario_2_envelope19_c2endshared_commonB_full_lut_retrieval_5B",
    "scenario_2_envelope23_ilcend_commonB_full_lut_retrieval_5B",
    "scenario_2_mp10_full_lut_retrieval_5b",
)

RETRIEVAL_SELECTION_TASKS = (
    "scenario_2_beam3_k13_floating_backward_retrieval_ablation_5b",
    "scenario_2_k11_backward_retrieval_oriented_basis_ablation_5b",
    "scenario_2_k12_backward_retrieval_oriented_basis_ablation_5b",
    "scenario_2_k12_retrieval_oriented_forward_scan_5b",
    "scenario_2_mp10_k11_retrieval_oriented_forward_scan_5b",
    "scenario_2_mp10_k9_retrieval_oriented_basis_ablation_5b",
    "scenario_2_self_first_joint_basis_ridge_lut_retrieval_search_5b",
    "scenario_2_smallbeam3_k13_retrieval_oriented_forward_expansion_5b",
    "scenario_2_updated_beam3_iteration2_retrieval_oriented_forward_expansion_5b",
)

RESULT_TASK_MODULE: dict[str, str] = {
    **{task: "behavior_modeling" for task in BEHAVIOR_MODELING_TASKS},
    **{task: "behavior_fingerprint_retrieval" for task in BEHAVIOR_FINGERPRINT_RETRIEVAL_TASKS},
    **{task: "retrieval_oriented_model_selection" for task in RETRIEVAL_SELECTION_TASKS},
}

SCRIPT_MODULE_RENAMES = {
    "_core": "core",
    "data_manager": "data_management",
    "pa_performance_observation": "pa_performance_evaluation",
    "behavior_model": "behavior_modeling",
    "behavior_fingerprint_lut_retrieval": "behavior_fingerprint_retrieval",
    "low_bandwidth_observation": "low_bandwidth_behavior_analysis",
    "clustering": "lut_clustering_compression",
}

SCRIPT_TASK_MODULE = {
    task: "retrieval_oriented_model_selection" for task in RETRIEVAL_SELECTION_TASKS
}
SCRIPT_TASK_MODULE["scenario_2_mp10_full_lut_retrieval_5b"] = "behavior_fingerprint_retrieval"

SCRIPT_TASK_SOURCE_NAMES = {
    "scenario_2_updated_beam3_iteration2_retrieval_oriented_forward_expansion_5b": "scenario_2_updated_beam3_k13_retrieval_oriented_forward_expansion_5b",
}

SPECIAL_SCRIPT_MOVES = {
    "retrieval_oriented_model_selection/scenario_2_C2_to_Aend_retrieval_oriented_model_scan": (
        "aggregate_scenario2_retrieval_oriented_model_scan.py",
        "analyze_scenario2_retrieval_oriented_model_scan.py",
        "export_scenario2_retrieval_oriented_best_excel.mjs",
        "export_scenario2_retrieval_oriented_best_excel.py",
        "plot_scenario2_retrieval_oriented_model_scan.py",
        "run_scenario2_retrieval_oriented_model_scan.py",
        "scenario2_retrieval_oriented_model_scan.py",
        "test_scenario2_retrieval_oriented_model_scan.py",
    ),
    "behavior_modeling/scenario_2_unified_model_capacity_scan": (
        "plot_scenario2_unified_model_capacity_scan.py",
        "run_scenario2_unified_model_capacity_scan.py",
        "scenario2_unified_model_capacity_scan.py",
        "test_scenario2_unified_model_capacity_scan.py",
    ),
    "retrieval_oriented_model_selection/scenario_2_statewise_best_round0_4_5b_retrieval": (
        "export_statewise_best_round0_4_5b_retrieval.mjs",
        "run_scenario2_c2_to_aend_statewise_best_round0_4_5b.py",
        "test_scenario2_c2_to_aend_statewise_best_round0_4_5b.py",
    ),
    "behavior_modeling/scenario_2_statewise_adaptive_aend_c2_5b": (
        "export_statewise_adaptive_best_metrics.mjs",
    ),
    "behavior_fingerprint_retrieval/scenario_2_C2_to_Aend_frozen_centered17_type3_commonB_control_5B": (
        "export_frozen_centered17_commonb_control_xlsx.mjs",
        "run_scenario2_C2_to_Aend_frozen_centered17_type3_commonB_control_5b.py",
    ),
    "lut_clustering_compression/scenario_2_C2_to_Aend_frozen_centered17_type3_compressed_5B": (
        "export_frozen_centered17_type3_xlsx.mjs",
        "export_type3_cluster_compressed_xlsx.mjs",
        "run_scenario2_C2_to_Aend_frozen_centered17_type3_compressed_5b.py",
        "run_scenario2_C2_to_Aend_type3_cluster_compressed_5b.py",
    ),
}

WORKLOG_MODULE_DIRS = {
    "codex_handoff": ("core", "codex_handoff"),
    "codex_session_export_project_20260821": ("core", "codex_session_export_project_20260821"),
    "core_module_build": ("core", "core_module_build"),
    "data_manager_build": ("data_management", "data_manager_build"),
    "signal_segmentation_build": ("signal_segmentation", "signal_segmentation_build"),
    "behavior_model": ("behavior_modeling", "behavior_model"),
    "behavior_fingerprint_lut_retrieval": (
        "behavior_fingerprint_retrieval",
        "behavior_fingerprint_lut_retrieval",
    ),
    "behavior_fingerprint_ranking_consistency": (
        "behavior_fingerprint_ranking_consistency",
        "behavior_fingerprint_ranking_consistency",
    ),
    "low_bandwidth_observation": ("low_bandwidth_behavior_analysis", "low_bandwidth_observation"),
    "pa_performance_observation": ("pa_performance_evaluation", "pa_performance_observation"),
    "clustering": ("lut_clustering_compression", "clustering"),
}

WORKLOG_SCENARIO_MODULES = {
    **{task: "behavior_modeling" for task in BEHAVIOR_MODELING_TASKS},
    **{task: "behavior_fingerprint_retrieval" for task in BEHAVIOR_FINGERPRINT_RETRIEVAL_TASKS},
    **{task: "retrieval_oriented_model_selection" for task in RETRIEVAL_SELECTION_TASKS},
    "scenario_2_C2_to_Aend_frozen_centered17_type3_commonB_control_5B": "behavior_fingerprint_retrieval",
    "scenario_2_C2_to_Aend_frozen_centered17_type3_compressed_5B": "lut_clustering_compression",
    "scenario_2_hard20_frozen10_reference_gate_5B": "behavior_modeling",
    "scenario_2_hard20_frozen_centered_basis_selection_5B": "behavior_modeling",
    "scenario_2_hard20_volterra_basis_selection_5B": "behavior_modeling",
}

OLD_MODULE_NAMES = (
    "_core",
    "data_manager",
    "pa_performance_observation",
    "behavior_model",
    "behavior_fingerprint_lut_retrieval",
    "basis_function_selection",
    "low_bandwidth_observation",
    "clustering",
)

MIGRATION_SCRIPT_BASENAME = Path(__file__).name


def _now() -> str:
    return datetime.now(UTC).astimezone().isoformat(timespec="seconds")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value.rstrip() + "\n", encoding="utf-8")


def _append_log(message: str) -> None:
    EXECUTION_LOG.parent.mkdir(parents=True, exist_ok=True)
    with EXECUTION_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"[{_now()}] {message.rstrip()}\n")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _raw_manifest() -> dict[str, Any]:
    raw_root = PROJECT_ROOT / "data" / "raw"
    files = sorted((item for item in raw_root.rglob("*") if item.is_file()), key=lambda item: str(item).lower())
    digest = hashlib.sha256()
    for item in files:
        digest.update(str(item.relative_to(raw_root)).replace("\\", "/").encode() + b"\0")
        digest.update(item.stat().st_size.to_bytes(8, "little"))
    return {
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "mat_count": sum(item.suffix.lower() == ".mat" for item in files),
        "bytes": sum(item.stat().st_size for item in files),
    }


def _git_status() -> dict[str, Any]:
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return {
        "branch": subprocess.run(
            ["git", "branch", "--show-current"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "origin_main": subprocess.run(
            ["git", "rev-parse", "origin/main"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
        ).stdout.strip(),
        "status_short": status,
        "unstaged_count": len(
            subprocess.run(
                ["git", "diff", "--name-only"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
            ).stdout.splitlines()
        ),
        "staged_count": len(
            subprocess.run(
                ["git", "diff", "--cached", "--name-only"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
        ),
        "untracked_count": len(
            subprocess.run(
                ["git", "ls-files", "--others", "--exclude-standard"],
                cwd=PROJECT_ROOT,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
        ),
    }


def _file_list(root: Path) -> list[Path]:
    return sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: str(item).lower())


def _tree_inventory(root: Path) -> dict[str, Any]:
    files = _file_list(root) if root.exists() else []
    return {
        "exists": root.exists(),
        "file_count": len(files),
        "bytes": sum(item.stat().st_size for item in files),
        "directories": sorted(str(item.relative_to(root)) for item in root.rglob("*") if item.is_dir())
        if root.exists()
        else [],
    }


def _self_first_snapshot(root: Path) -> dict[str, Any]:
    files = _file_list(root)
    distance_files = sorted(
        (item for item in files if "distance_matrices" in item.parts and item.suffix.lower() == ".npy"),
        key=lambda item: str(item).lower(),
    )
    critical_names = {
        "22_final_result_summary.json",
        "22_final_result_summary.txt",
        "23_checkpoint.json",
        "24_validation_checks.json",
        "12_final_dense_ridge_scan.csv",
        "13_final_basis_importance.csv",
        "15_final_single_add_validation.csv",
        "16_final_swap_validation.csv",
        "20_query_state_cv_results.csv",
        "21_query_state_cv_stability.csv",
    }
    critical = {}
    for item in files:
        if item.name in critical_names:
            critical[str(item.relative_to(root))] = _sha256(item)
    eligible = [item for item in files if item.name not in critical_names and item.suffix.lower() not in {".pyc", ".pyo"}]
    rng = random.Random(20260917)
    sample = rng.sample(eligible, min(100, len(eligible)))
    random_hashes = {str(item.relative_to(root)): _sha256(item) for item in sorted(sample, key=lambda x: str(x).lower())}
    return {
        "file_count": len(files),
        "bytes": sum(item.stat().st_size for item in files),
        "distance_matrix_count": len(distance_files),
        "critical_hashes": critical,
        "random_hashes": random_hashes,
    }


def _module_tree_lines(root: Path) -> list[str]:
    lines = [f"{root.name}/"]
    if not root.exists():
        return lines
    directories = sorted((item for item in root.rglob("*") if item.is_dir()), key=lambda item: str(item).lower())
    for directory in directories:
        depth = len(directory.relative_to(root).parts)
        lines.append(f"{'  ' * depth}{directory.name}/")
    return lines


def _pre_inventory() -> dict[str, Any]:
    return {
        "timestamp": _now(),
        "project_root": str(PROJECT_ROOT),
        "git": _git_status(),
        "raw_manifest": _raw_manifest(),
        "root_counts": {name: _tree_inventory(PROJECT_ROOT / name) for name in ("data", "scripts", "results", "work_logs")},
        "top_level_directories": {
            name: sorted(item.name for item in (PROJECT_ROOT / name).iterdir() if item.is_dir())
            for name in ("scripts", "results", "work_logs")
        },
        "self_first": _self_first_snapshot(
            PROJECT_ROOT / "results" / "scenario_2_self_first_joint_basis_ridge_lut_retrieval_search_5b"
        ),
    }


def _planned_moves() -> list[dict[str, str]]:
    moves: list[dict[str, str]] = []
    for source, destination in SCRIPT_MODULE_RENAMES.items():
        moves.append({"root": "scripts", "kind": "module_directory", "source": f"scripts/{source}", "destination": f"scripts/{destination}"})
    for task, module in RESULT_TASK_MODULE.items():
        moves.append({"root": "results", "kind": "task_directory", "source": f"results/{task}", "destination": f"results/{module}/{task}"})
    for source, (module, task) in WORKLOG_MODULE_DIRS.items():
        moves.append({"root": "work_logs", "kind": "log_directory", "source": f"work_logs/{source}", "destination": f"work_logs/{module}/{task}"})
    for task, module in WORKLOG_SCENARIO_MODULES.items():
        moves.append({
            "root": "work_logs",
            "kind": "task_log_directory",
            "source": f"work_logs/{task}",
            "destination": f"work_logs/{module}/{task}",
        })
    for task, module in SCRIPT_TASK_MODULE.items():
        source_task = SCRIPT_TASK_SOURCE_NAMES.get(task, task)
        moves.append({
            "root": "scripts",
            "kind": "task_directory",
            "source": f"scripts/{source_task}",
            "destination": f"scripts/{module}/{task}",
        })
    moves.append({
        "root": "scripts",
        "kind": "basis_package",
        "source": "scripts/basis_function_selection",
        "destination": "scripts/behavior_modeling/basis_function_selection",
    })
    for destination, names in SPECIAL_SCRIPT_MOVES.items():
        for name in names:
            moves.append({
                "root": "scripts",
                "kind": "special_script",
                "source": f"scripts/behavior_fingerprint_lut_retrieval/{name}",
                "destination": f"scripts/{destination}/{name}",
            })
    return moves


def _classification_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def add(**values: Any) -> None:
        values.setdefault("ambiguity", "")
        values.setdefault("status", "planned")
        rows.append(values)

    for source, destination in SCRIPT_MODULE_RENAMES.items():
        add(
            source_root="scripts",
            source_path=f"scripts/{source}",
            item_name=source,
            item_type="module_directory",
            destination_module=destination,
            destination_task="",
            classification_reason="existing public module renamed to the frozen registry name",
            has_script=True,
            has_result=False,
            has_log=False,
            move_required=True,
        )
    add(
        source_root="scripts",
        source_path="scripts/signal_segmentation",
        item_name="signal_segmentation",
        item_type="module_directory",
        destination_module="signal_segmentation",
        destination_task="",
        classification_reason="existing module name already matches the frozen registry",
        has_script=True,
        has_result=False,
        has_log=False,
        move_required=False,
    )
    add(
        source_root="scripts",
        source_path="scripts/behavior_fingerprint_ranking_consistency",
        item_name="behavior_fingerprint_ranking_consistency",
        item_type="module_directory",
        destination_module="behavior_fingerprint_ranking_consistency",
        destination_task="",
        classification_reason="existing module name already matches the frozen registry",
        has_script=True,
        has_result=False,
        has_log=False,
        move_required=False,
    )
    for task, module in SCRIPT_TASK_MODULE.items():
        source_task = SCRIPT_TASK_SOURCE_NAMES.get(task, task)
        add(
            source_root="scripts",
            source_path=f"scripts/{source_task}",
            item_name=source_task,
            item_type="task_directory",
            destination_module=module,
            destination_task=task,
            classification_reason="retrieval-oriented Forward/Backward/Beam/Floating/Self-First objective",
            has_script=True,
            has_result=task in RESULT_TASK_MODULE,
            has_log=task in WORKLOG_SCENARIO_MODULES,
            move_required=True,
        )
    add(
        source_root="scripts",
        source_path="scripts/basis_function_selection",
        item_name="basis_function_selection",
        item_type="package_split",
        destination_module="behavior_modeling",
        destination_task="basis_function_selection",
        classification_reason="all 43 source files are model Train/CV/Generalization/NMSE or basis-construction logic; no retrieval Top-1 objective found",
        has_script=True,
        has_result=False,
        has_log=True,
        move_required=True,
    )
    basis_source = PROJECT_ROOT / "scripts" / "basis_function_selection"
    for path in sorted((item for item in basis_source.iterdir() if item.is_file()), key=lambda item: item.name.lower()):
        add(
            source_root="scripts",
            source_path=str(path.relative_to(PROJECT_ROOT)),
            item_name=path.name,
            item_type="basis_source_file",
            destination_module="behavior_modeling",
            destination_task="basis_function_selection",
            classification_reason="file-level review: basis construction, model solver, Train/CV/Generalization, or NMSE-oriented selection support",
            has_script=True,
            has_result=False,
            has_log=True,
            move_required=True,
        )
    for destination, names in SPECIAL_SCRIPT_MOVES.items():
        module, task = destination.split("/", 1)
        for name in names:
            add(
                source_root="scripts",
                source_path=f"scripts/behavior_fingerprint_lut_retrieval/{name}",
                item_name=name,
                item_type="special_script",
                destination_module=module,
                destination_task=task,
                classification_reason="per-file objective review of mixed legacy module; assigned to the model, retrieval, or clustering task named by its operational purpose",
                has_script=True,
                has_result=False,
                has_log=False,
                move_required=True,
            )
    for task, module in RESULT_TASK_MODULE.items():
        add(
            source_root="results",
            source_path=f"results/{task}",
            item_name=task,
            item_type="task_directory",
            destination_module=module,
            destination_task=task,
            classification_reason="result task mapping reviewed against task names and execution summaries",
            has_script=task in SCRIPT_TASK_MODULE,
            has_result=True,
            has_log=task in WORKLOG_SCENARIO_MODULES,
            move_required=True,
        )
    for source, (module, task) in WORKLOG_MODULE_DIRS.items():
        add(
            source_root="work_logs",
            source_path=f"work_logs/{source}",
            item_name=source,
            item_type="module_or_task_log",
            destination_module=module,
            destination_task=task,
            classification_reason="existing module/build or engineering handoff log mapped to its public module",
            has_script=False,
            has_result=False,
            has_log=True,
            move_required=True,
        )
    for task, module in WORKLOG_SCENARIO_MODULES.items():
        add(
            source_root="work_logs",
            source_path=f"work_logs/{task}",
            item_name=task,
            item_type="task_log",
            destination_module=module,
            destination_task=task,
            classification_reason="scenario task log mapped by its primary scientific objective",
            has_script=task in SCRIPT_TASK_MODULE,
            has_result=task in RESULT_TASK_MODULE,
            has_log=True,
            move_required=True,
        )
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _move_manifest_rows(plan: list[dict[str, str]]) -> list[dict[str, Any]]:
    rows = []
    for item in plan:
        source = PROJECT_ROOT / item["source"]
        destination = PROJECT_ROOT / item["destination"]
        rows.append(
            {
                **item,
                "source_exists": source.exists(),
                "destination_exists_before": destination.exists(),
                "source_type": "directory" if source.is_dir() else "file" if source.is_file() else "missing",
                "source_file_count": len(_file_list(source)) if source.is_dir() else 1 if source.is_file() else 0,
                "source_bytes": sum(item.stat().st_size for item in _file_list(source)) if source.is_dir() else source.stat().st_size if source.is_file() else 0,
                "status": "planned",
            }
        )
    return rows


def _assert_preconditions(plan: list[dict[str, str]], pre: dict[str, Any]) -> None:
    if pre["raw_manifest"] != EXPECTED_RAW:
        raise RuntimeError(f"raw manifest HARD FAIL before migration: {pre['raw_manifest']}")
    devices = {os.stat(PROJECT_ROOT / name).st_dev for name in ("data", "scripts", "results", "work_logs")}
    if len(devices) != 1:
        raise RuntimeError(f"source roots are not on one filesystem: {devices}")
    for item in plan:
        source = PROJECT_ROOT / item["source"]
        destination = PROJECT_ROOT / item["destination"]
        if not source.exists():
            raise RuntimeError(f"planned source is missing: {source}")
        if destination.exists():
            if source.is_file() and destination.is_file() and _sha256(source) == _sha256(destination):
                continue
            raise RuntimeError(f"destination collision before migration: {destination}")
    unexpected_worklogs = set(item.name for item in (PROJECT_ROOT / "work_logs").iterdir() if item.is_dir()) - set(WORKLOG_MODULE_DIRS) - set(WORKLOG_SCENARIO_MODULES)
    if unexpected_worklogs:
        raise RuntimeError(f"unclassified work_log directories: {sorted(unexpected_worklogs)}")


def _rename(source: Path, destination: Path) -> str:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if source.is_file() and destination.is_file() and _sha256(source) == _sha256(destination):
            return "deduplicated_identical"
        raise RuntimeError(f"destination collision during migration: {destination}")
    os.rename(source, destination)
    return "renamed"


def _rename_resumable(source: Path, destination: Path, kind: str) -> str:
    """Perform one forward-only move and tolerate moves completed before a retry."""

    if source == destination:
        return "already_aligned"
    if destination.exists() and not source.exists():
        return "already_moved"
    if source.is_dir() and destination.parent == source:
        destination.mkdir(parents=True, exist_ok=True)
        for child in sorted(source.iterdir(), key=lambda item: item.name.lower()):
            if child == destination:
                continue
            os.rename(child, destination / child.name)
        return "nested_contents"
    if not source.exists():
        raise RuntimeError(f"planned source and destination are both missing: {source} -> {destination}")
    if destination.exists():
        if source.is_file() and destination.is_file() and _sha256(source) == _sha256(destination):
            return "deduplicated_identical"
        raise RuntimeError(f"destination collision during migration: {destination}")
    return _rename(source, destination)


def _execute_moves(plan: list[dict[str, str]], move_rows: list[dict[str, Any]]) -> None:
    def execute(source_rel: str, destination_rel: str, kind: str) -> str:
        return _rename_resumable(PROJECT_ROOT / source_rel, PROJECT_ROOT / destination_rel, kind)

    # Public module roots first.  The migration runner itself is inside _core
    # and remains valid while its parent directory is renamed.
    ordered = []
    ordered.extend(item for item in plan if item["kind"] == "module_directory")
    ordered.extend(item for item in plan if item["kind"] == "task_directory" and item["root"] == "scripts")
    ordered.extend(item for item in plan if item["kind"] == "basis_package")
    ordered.extend(item for item in plan if item["kind"] == "special_script")
    ordered.extend(item for item in plan if item["kind"] == "task_directory" and item["root"] == "results")
    ordered.extend(item for item in plan if item["kind"] == "log_directory")
    ordered.extend(item for item in plan if item["kind"] == "task_log_directory")
    done = set()
    for item in ordered:
        key = (item["source"], item["destination"])
        if key in done:
            continue
        done.add(key)
        source_rel = item["source"]
        if item["kind"] == "special_script":
            source_rel = source_rel.replace(
                "scripts/behavior_fingerprint_lut_retrieval/",
                "scripts/behavior_fingerprint_retrieval/",
            )
        status = execute(source_rel, item["destination"], item["kind"])
        for row in move_rows:
            if row["source"] == item["source"] and row["destination"] == item["destination"]:
                row["status"] = status
                row["destination_exists_after"] = (PROJECT_ROOT / item["destination"]).exists()
                break


def _replace_module_packages(text: str) -> str:
    text = text.replace("scripts._core", "scripts.core")
    text = text.replace("scripts/_core", "scripts/core")
    text = text.replace('PROJECT_ROOT / "scripts" / "_core"', 'PROJECT_ROOT / "scripts" / "core"')
    text = text.replace('"_core/', '"core/')
    text = text.replace("'_core/", "'core/")
    text = text.replace('/ "_core"', '/ "core"')
    text = text.replace('SCRIPTS_ROOT / "_core"', 'SCRIPTS_ROOT / "core"')
    text = re.sub(r"(?<![\w.])from _core\b", "from core", text)
    text = re.sub(r"(?<![\w.])import _core\b", "import core", text)
    text = re.sub(r"(?<![\w.])_core\.", "core.", text)
    text = re.sub(r"(?<![\w])data_manager(?![\w])", "data_management", text)
    text = re.sub(r"(?<![\w])pa_performance_observation(?![\w])", "pa_performance_evaluation", text)
    text = re.sub(r"(?<![\w])behavior_fingerprint_lut_retrieval(?![\w])", "behavior_fingerprint_retrieval", text)
    text = re.sub(r"(?<![\w])behavior_model(?![\w])", "behavior_modeling", text)
    text = re.sub(r"(?<![\w])low_bandwidth_observation(?![\w])", "low_bandwidth_behavior_analysis", text)
    text = re.sub(r"(?<![\w])from clustering\b", "from lut_clustering_compression", text)
    text = re.sub(r"(?<![\w])import clustering\b", "import lut_clustering_compression", text)
    text = text.replace('SCRIPTS_ROOT / "clustering"', 'SCRIPTS_ROOT / "lut_clustering_compression"')
    text = re.sub(r"(?<![\w.])from basis_function_selection\b", "from behavior_modeling.basis_function_selection", text)
    text = re.sub(r"(?<![\w.])import basis_function_selection\b", "import behavior_modeling.basis_function_selection", text)
    text = re.sub(r"(?<![\w.])basis_function_selection\.", "behavior_modeling.basis_function_selection.", text)
    text = text.replace(
        "scenario_2_updated_beam3_k13_retrieval_oriented_forward_expansion_5b",
        "scenario_2_updated_beam3_iteration2_retrieval_oriented_forward_expansion_5b",
    )
    text = text.replace('PROJECT_ROOT / "results" / "basis_function_selection"', 'PROJECT_ROOT / "results" / "behavior_modeling"')
    text = text.replace('PROJECT_ROOT / "work_logs" / "basis_function_selection"', 'PROJECT_ROOT / "work_logs" / "behavior_modeling"')
    text = text.replace('PROJECT_ROOT / "results" / "clustering"', 'PROJECT_ROOT / "results" / "lut_clustering_compression"')
    text = text.replace('PROJECT_ROOT / "work_logs" / "clustering"', 'PROJECT_ROOT / "work_logs" / "lut_clustering_compression"')
    return text


def _replace_task_paths(text: str, module: str) -> str:
    for task, task_module in sorted(RESULT_TASK_MODULE.items(), key=lambda item: len(item[0]), reverse=True):
        for root_name in ("results", "work_logs"):
            old = f'PROJECT_ROOT / "{root_name}" / "{task}"'
            new = f'PROJECT_ROOT / "{root_name}" / "{task_module}" / "{task}"'
            text = text.replace(old, new)
            text = text.replace(
                f'ROOT / "{root_name}" / "{task}"',
                f'ROOT / "{root_name}" / "{task_module}" / "{task}"',
            )
            text = text.replace(
                f'results/{task}',
                f'results/{task_module}/{task}',
            )
            text = text.replace(
                f'work_logs/{task}',
                f'work_logs/{task_module}/{task}',
            )
    for variable in ("PROJECT_ROOT", "ROOT"):
        text = text.replace(
            f'{variable} / "results" / TASK_NAME',
            f'{variable} / "results" / "{module}" / TASK_NAME',
        )
        text = text.replace(
            f'{variable} / "work_logs" / TASK_NAME',
            f'{variable} / "work_logs" / "{module}" / TASK_NAME',
        )
        text = text.replace(
            f'{variable} / "results" / EXPERIMENT_NAME',
            f'{variable} / "results" / "{module}" / EXPERIMENT_NAME',
        )
        text = text.replace(
            f'{variable} / "work_logs" / EXPERIMENT_NAME',
            f'{variable} / "work_logs" / "{module}" / EXPERIMENT_NAME',
        )
    text = text.replace(
        'PROJECT_ROOT / "work_logs" / "codex_handoff" / "codex_handoff.txt"',
        'PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"',
    )
    text = text.replace(
        'ROOT / "work_logs" / "codex_handoff" / "codex_handoff.txt"',
        'ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"',
    )
    return text


def _fix_nested_roots(text: str, nested: bool) -> str:
    if not nested:
        return text
    text = text.replace("SCRIPTS_ROOT = MODULE_ROOT.parent\n", "SCRIPTS_ROOT = MODULE_ROOT.parent.parent\n")
    text = text.replace(
        "SCRIPTS_ROOT = Path(__file__).resolve().parents[1]\n",
        "SCRIPTS_ROOT = Path(__file__).resolve().parents[2]\n",
    )
    text = text.replace(
        "ROOT = Path(__file__).resolve().parents[2]\n",
        'ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "environment.yml").is_file())\n',
    )
    text = text.replace(
        "PROJECT_ROOT = Path(__file__).resolve().parents[2]\n",
        'PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "environment.yml").is_file())\n',
    )
    if "sys.path.insert(0, str(SCRIPTS_ROOT))" in text and "sys.path.insert(0, str(Path(__file__).resolve().parent.parent))" not in text:
        text = text.replace(
            "sys.path.insert(0, str(SCRIPTS_ROOT))\n",
            "sys.path.insert(0, str(SCRIPTS_ROOT))\nif str(Path(__file__).resolve().parent.parent) not in sys.path:\n    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))\n",
            1,
        )
    return text


def _fix_special_package_imports(text: str) -> str:
    replacements = {
        "behavior_fingerprint_retrieval.plot_scenario2_retrieval_oriented_model_scan": "retrieval_oriented_model_selection.scenario_2_C2_to_Aend_retrieval_oriented_model_scan.plot_scenario2_retrieval_oriented_model_scan",
        "behavior_fingerprint_retrieval.scenario2_retrieval_oriented_model_scan": "retrieval_oriented_model_selection.scenario_2_C2_to_Aend_retrieval_oriented_model_scan.scenario2_retrieval_oriented_model_scan",
        "behavior_modeling.plot_scenario2_unified_model_capacity_scan": "behavior_modeling.scenario_2_unified_model_capacity_scan.plot_scenario2_unified_model_capacity_scan",
        "behavior_modeling.scenario2_unified_model_capacity_scan": "behavior_modeling.scenario_2_unified_model_capacity_scan.scenario2_unified_model_capacity_scan",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _rewrite_active_code() -> tuple[int, int]:
    changed_files = 0
    replacement_count = 0
    for path in sorted((PROJECT_ROOT / "scripts").rglob("*"), key=lambda item: str(item).lower()):
        if not path.is_file() or path.suffix.lower() not in {".py", ".mjs"} or path.name == MIGRATION_SCRIPT_BASENAME:
            continue
        original = path.read_text(encoding="utf-8")
        relative = path.relative_to(PROJECT_ROOT / "scripts")
        module = relative.parts[0] if relative.parts and relative.parts[0] in MODULES else "core"
        nested = len(relative.parts) >= 3 and relative.parts[1] != "basis_function_selection"
        if relative.parts[:2] == ("behavior_modeling", "basis_function_selection"):
            nested = True
        updated = _replace_module_packages(original)
        updated = _replace_task_paths(updated, module)
        updated = _fix_nested_roots(updated, nested)
        updated = _fix_special_package_imports(updated)
        if updated != original:
            path.write_text(updated, encoding="utf-8")
            changed_files += 1
            replacement_count += sum(1 for left, right in zip(original.splitlines(), updated.splitlines()) if left != right)
    return changed_files, replacement_count


STALE_PATTERNS = (
    re.compile(r"scripts\._core"),
    re.compile(r"(?<![\w.])_core(?:[/\"'])"),
    re.compile(r"(?<![\w])data_manager(?![\w])"),
    re.compile(r"(?<![\w])pa_performance_observation(?![\w])"),
    re.compile(r"(?<![\w])behavior_model(?![\w])"),
    re.compile(r"(?<![\w])behavior_fingerprint_lut_retrieval(?![\w])"),
    re.compile(r"(?<![\w])low_bandwidth_observation(?![\w])"),
    re.compile(r"(?<![\w.])from clustering\b"),
    re.compile(r"PROJECT_ROOT\s*/\s*[\"']results[\"']\s*/\s*[\"']clustering[\"']"),
    re.compile(r"PROJECT_ROOT\s*/\s*[\"']work_logs[\"']\s*/\s*[\"']clustering[\"']"),
    re.compile(r"PROJECT_ROOT\s*/\s*[\"']results[\"']\s*/\s*TASK_NAME"),
    re.compile(r"PROJECT_ROOT\s*/\s*[\"']work_logs[\"']\s*/\s*TASK_NAME"),
)


def _path_audit() -> list[dict[str, Any]]:
    findings = []
    for path in sorted((PROJECT_ROOT / "scripts").rglob("*"), key=lambda item: str(item).lower()):
        if not path.is_file() or path.suffix.lower() not in {".py", ".mjs"} or path.name == MIGRATION_SCRIPT_BASENAME:
            continue
        for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if any(pattern.search(line) for pattern in STALE_PATTERNS):
                findings.append({"path": str(path.relative_to(PROJECT_ROOT)), "line": line_number, "text": line})
    return findings


def _write_pre_artifacts(pre: dict[str, Any], plan: list[dict[str, str]]) -> list[dict[str, Any]]:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    _write_text(RESULT_ROOT / "01_pre_migration_inventory.txt", json.dumps(pre, ensure_ascii=False, indent=2, default=_json_default))
    registry = {"modules": list(MODULES), "module_count": len(MODULES), "root_names": ["scripts", "results", "work_logs"]}
    _write_json(RESULT_ROOT / "02_module_registry.json", registry)
    _write_csv(RESULT_ROOT / "03_task_classification_manifest.csv", _classification_rows())
    move_rows = _move_manifest_rows(plan)
    _write_csv(RESULT_ROOT / "04_move_manifest.csv", move_rows)
    _write_text(RESULT_ROOT / "05_path_reference_audit_before.txt", json.dumps(_path_audit(), ensure_ascii=False, indent=2))
    return move_rows


def _root_alignment() -> dict[str, Any]:
    expected = set(MODULES)
    roots = {}
    for root_name in ("scripts", "results", "work_logs"):
        actual = {item.name for item in (PROJECT_ROOT / root_name).iterdir() if item.is_dir()}
        roots[root_name] = {
            "actual": sorted(actual),
            "missing": sorted(expected - actual),
            "extra": sorted(actual - expected),
            "pass": actual == expected,
        }
    return {"modules": list(MODULES), "roots": roots, "pass": all(item["pass"] for item in roots.values())}


def _post_validation() -> dict[str, Any]:
    root_alignment = _root_alignment()
    root_task_dirs = {
        root_name: sorted(
            item.name
            for item in (PROJECT_ROOT / root_name).iterdir()
            if item.is_dir() and (item.name.startswith("scenario_2_") or item.name == "basis_function_selection")
        )
        for root_name in ("scripts", "results", "work_logs")
    }
    old_module_dirs = {
        root_name: sorted(set(OLD_MODULE_NAMES) & {item.name for item in (PROJECT_ROOT / root_name).iterdir() if item.is_dir()})
        for root_name in ("scripts", "results", "work_logs")
    }
    self_first_root = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "scenario_2_self_first_joint_basis_ridge_lut_retrieval_search_5b"
    state_path = RESULT_ROOT / "migration_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {}
    before_self = state.get("pre_inventory", {}).get("self_first", {})
    after_self = _self_first_snapshot(self_first_root) if self_first_root.exists() else {}
    critical_before = before_self.get("critical_hashes", {})
    critical_after = {
        name: _sha256(self_first_root / name)
        for name in critical_before
        if (self_first_root / name).is_file()
    }
    random_before = before_self.get("random_hashes", {})
    random_after = {
        name: _sha256(self_first_root / name)
        for name in random_before
        if (self_first_root / name).is_file()
    }
    critical_equal = critical_before == critical_after
    random_equal = random_before == random_after
    raw = _raw_manifest()
    path_findings = _path_audit()
    result = {
        "timestamp": _now(),
        "modules": list(MODULES),
        "root_alignment": root_alignment,
        "root_level_task_dirs": root_task_dirs,
        "old_module_dirs": old_module_dirs,
        "raw_manifest": raw,
        "raw_manifest_expected": EXPECTED_RAW,
        "raw_manifest_pass": raw == EXPECTED_RAW,
        "self_first": {
            "before": before_self,
            "after": after_self,
            "file_count_equal": before_self.get("file_count") == after_self.get("file_count"),
            "bytes_equal": before_self.get("bytes") == after_self.get("bytes"),
            "distance_matrix_count_equal": before_self.get("distance_matrix_count") == after_self.get("distance_matrix_count"),
            "critical_hashes_equal": critical_equal,
            "random_hashes_equal": random_equal,
            "critical_hash_mismatches": sorted(set(critical_before) ^ set(critical_after))
            + sorted(name for name in set(critical_before) & set(critical_after) if critical_before[name] != critical_after[name]),
            "random_hash_mismatch_count": sum(
                1 for name in set(random_before) & set(random_after) if random_before[name] != random_after[name]
            )
            + len(set(random_before) ^ set(random_after)),
        },
        "path_reference_audit_after": {"count": len(path_findings), "findings": path_findings},
        "collisions": [],
        "unresolved_tasks": [],
        "git": _git_status(),
    }
    result["pass"] = bool(
        root_alignment["pass"]
        and all(not values for values in root_task_dirs.values())
        and all(not values for values in old_module_dirs.values())
        and result["raw_manifest_pass"]
        and result["self_first"]["file_count_equal"]
        and result["self_first"]["bytes_equal"]
        and result["self_first"]["distance_matrix_count_equal"]
        and critical_equal
        and random_equal
        and not path_findings
    )
    _write_text(RESULT_ROOT / "06_path_reference_audit_after.txt", json.dumps(path_findings, ensure_ascii=False, indent=2))
    _write_text(RESULT_ROOT / "07_post_migration_inventory.txt", json.dumps({
        "timestamp": result["timestamp"],
        "root_directories": {
            root_name: sorted(item.name for item in (PROJECT_ROOT / root_name).iterdir() if item.is_dir())
            for root_name in ("scripts", "results", "work_logs")
        },
        "root_counts": {name: _tree_inventory(PROJECT_ROOT / name) for name in ("data", "scripts", "results", "work_logs")},
        "self_first": after_self,
    }, default=_json_default))
    _write_json(RESULT_ROOT / "08_validation_report.json", result)
    tree_lines = []
    for root_name in ("scripts", "results", "work_logs"):
        tree_lines.append(f"{root_name}/")
        tree_lines.extend(f"  {line}" for line in _module_tree_lines(PROJECT_ROOT / root_name)[1:])
    _write_text(RESULT_ROOT / "09_final_directory_tree.txt", "\n".join(tree_lines))
    summary = [
        f"Task: {TASK_NAME}",
        "Migration type: rename-only engineering reorganization; no research recomputation.",
        f"Modules: {', '.join(MODULES)}",
        f"Root alignment pass: {root_alignment['pass']}",
        f"Raw manifest pass: {result['raw_manifest_pass']}",
        f"Self-First file count/bytes/distance matrices preserved: {result['self_first']['file_count_equal']}/{result['self_first']['bytes_equal']}/{result['self_first']['distance_matrix_count_equal']}",
        f"Self-First critical hashes preserved: {critical_equal}",
        f"Self-First random hash sample preserved: {random_equal}",
        f"Stale active path references after rewrite: {len(path_findings)}",
        "Filename collisions: none",
        "Unresolved task classifications: none",
        f"Overall migration validation: {'PASS' if result['pass'] else 'FAIL'}",
        "No git add/commit/push was performed.",
    ]
    _write_text(RESULT_ROOT / "10_migration_summary.txt", "\n".join(summary))
    return result


def _run_migration() -> dict[str, Any]:
    plan = _planned_moves()
    pre = _pre_inventory()
    _assert_preconditions(plan, pre)
    move_rows = _write_pre_artifacts(pre, plan)
    _write_json(
        RESULT_ROOT / "migration_state.json",
        {"pre_inventory": pre, "planned_moves": plan, "expected_raw": EXPECTED_RAW},
    )
    for root_name in ("results", "work_logs"):
        for module in MODULES:
            (PROJECT_ROOT / root_name / module).mkdir(parents=True, exist_ok=True)
    _append_log("Phase 0/1 preflight, inventory, module registry, classification, and move manifest passed.")
    _execute_moves(plan, move_rows)
    _write_csv(RESULT_ROOT / "04_move_manifest.csv", move_rows)
    changed_files, replacement_count = _rewrite_active_code()
    _append_log(
        f"Renamed existing module/task/result/log directories in place; rewrote active code paths/imports in {changed_files} files ({replacement_count} changed lines)."
    )
    result = _post_validation()
    _append_log(f"Post-migration structural validation {'PASSED' if result['pass'] else 'FAILED'}.")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
    return result


def _continue_migration() -> dict[str, Any]:
    """Continue a migration interrupted after some forward-only renames."""

    state_path = RESULT_ROOT / "migration_state.json"
    if not state_path.is_file():
        raise RuntimeError("migration continuation requires migration_state.json")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    pre = state.get("pre_inventory")
    if not pre:
        raise RuntimeError("migration continuation is missing the pre-migration inventory")
    plan = _planned_moves()
    if _raw_manifest() != EXPECTED_RAW:
        raise RuntimeError(f"raw manifest HARD FAIL during migration continuation: {_raw_manifest()}")
    for root_name in ("results", "work_logs"):
        for module in MODULES:
            (PROJECT_ROOT / root_name / module).mkdir(parents=True, exist_ok=True)
    move_rows = _move_manifest_rows(plan)
    _execute_moves(plan, move_rows)
    _write_csv(RESULT_ROOT / "04_move_manifest.csv", move_rows)
    changed_files, replacement_count = _rewrite_active_code()
    _write_json(
        state_path,
        {"pre_inventory": pre, "planned_moves": plan, "expected_raw": EXPECTED_RAW},
    )
    _append_log(
        f"Continued interrupted migration forward-only; rewrote active code paths/imports in {changed_files} files ({replacement_count} changed lines)."
    )
    result = _post_validation()
    _append_log(f"Post-migration structural validation {'PASSED' if result['pass'] else 'FAILED'}.")
    print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
    return result


def main() -> None:
    if "--continue-migration" in sys.argv[1:]:
        result = _continue_migration()
        if not result["pass"]:
            raise SystemExit(1)
        return
    if "--validate-only" in sys.argv[1:]:
        result = _post_validation()
        _append_log(f"Validation-only pass {'PASSED' if result['pass'] else 'FAILED'}.")
        print(json.dumps(result, ensure_ascii=False, indent=2, default=_json_default))
        if not result["pass"]:
            raise SystemExit(1)
        return
    _run_migration()


if __name__ == "__main__":
    main()
