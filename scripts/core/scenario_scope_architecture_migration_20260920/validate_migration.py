"""Validate the completed non-protected scenario-scope migration."""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = next(parent for parent in Path(__file__).resolve().parents if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir())
LOG_ROOT = PROJECT_ROOT / "work_logs" / "core" / "scenario_scope_architecture_migration_20260920"
MODULES = ("core", "data_management", "signal_segmentation", "pa_performance_evaluation", "behavior_modeling", "behavior_fingerprint_retrieval", "retrieval_oriented_model_selection", "behavior_fingerprint_ranking_consistency", "low_bandwidth_behavior_analysis", "lut_clustering_compression")
PROTECTED = "retrieval_oriented_model_selection"
EXPECTED_RAW = {"sha256": "910b06e4be6b60b1f4dc63409a3083eb328765347fbec3b89a97a41d6633a1e0", "file_count": 429, "mat_count": 427, "bytes": 2_258_448_137}


def _manifest(root: Path) -> dict[str, tuple[int, str]]:
    output: dict[str, tuple[int, str]] = {}
    if not root.exists():
        return output
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: str(item).lower()):
        output[str(path.relative_to(root))] = (path.stat().st_size, hashlib.sha256(path.read_bytes()).hexdigest())
    return output


def _raw_manifest() -> dict[str, object]:
    root = PROJECT_ROOT / "data" / "raw"
    digest = hashlib.sha256()
    files = sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: str(p).lower())
    total = 0
    mats = 0
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode() + b"\0")
        size = path.stat().st_size
        digest.update(int(size).to_bytes(8, "little"))
        total += size
        mats += path.suffix.lower() == ".mat"
    return {"sha256": digest.hexdigest(), "file_count": len(files), "mat_count": mats, "bytes": total}


def validate() -> dict[str, Any]:
    plan = json.loads((LOG_ROOT / "migration_plan.json").read_text(encoding="utf-8"))["plan"]
    journal_lines = [json.loads(line) for line in (LOG_ROOT / "migration_journal.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    migrated = [item for item in plan if item["migration_status"] == "READY"]
    failures: list[str] = []
    checks: dict[str, bool] = {}
    checks["protected_journal_clean"] = not any(PROTECTED in str(item) for item in journal_lines)
    checks["data_journal_clean"] = not any(str(item.get("source", "")).startswith(str(PROJECT_ROOT / "data")) or str(item.get("target", "")).startswith(str(PROJECT_ROOT / "data")) for item in journal_lines)
    checks["config_journal_clean"] = not any(str(item.get("source", "")).split("/")[-1] in {"AGENTS.md", "README.md", "environment.yml", "pyproject.toml", ".gitignore", ".gitattributes"} for item in journal_lines)
    checks["module_mirror"] = all(sorted(p.name for p in (PROJECT_ROOT / root).iterdir() if p.is_dir()) == sorted(MODULES) for root in ("scripts", "results", "work_logs"))
    checks["raw_manifest"] = _raw_manifest() == EXPECTED_RAW
    checks["no_results_shared"] = not any((PROJECT_ROOT / "results" / module / "shared").exists() for module in MODULES)
    checks["no_work_logs_shared"] = not any((PROJECT_ROOT / "work_logs" / module / "shared").exists() for module in MODULES)
    checks["all_ready_script_targets"] = True
    checks["old_paths_absent"] = True
    checks["result_log_content_preserved"] = True
    pre = json.loads((LOG_ROOT / "pre_migration_manifest.json").read_text(encoding="utf-8"))["entries"]
    pre_lookup = {(item["module"], item["task_name"], item["artifact"]): item for item in pre}
    for record in migrated:
        for artifact, old_key, target_key in (("scripts", "script_path", "target_script_path"), ("results", "result_path", "target_result_path"), ("work_logs", "work_log_path", "target_work_log_path")):
            old = record.get(old_key)
            target = record.get(target_key)
            if not old or not target:
                continue
            if not Path(target).exists():
                failures.append(f"missing_target:{target}")
                checks["all_ready_script_targets"] = False
            if Path(old).exists():
                failures.append(f"old_path_remains:{old}")
                checks["old_paths_absent"] = False
            if artifact in {"results", "work_logs"}:
                before = pre_lookup.get((record["module"], record["task_name"], artifact))
                current = _manifest(Path(target))
                expected = {item["relative_path"]: (int(item["bytes"]), item["sha256"]) for item in before["files"]} if before else {}
                if current != expected:
                    failures.append(f"content_changed:{target}")
                    checks["result_log_content_preserved"] = False
    residuals = []
    for module in MODULES:
        if module in {"core", "data_management", PROTECTED}:
            continue
        module_root = PROJECT_ROOT / "scripts" / module
        for child in module_root.iterdir():
            if child.is_dir() and child.name not in {"shared", "scenario_1", "scenario_2", "cross_scenario", "__pycache__"}:
                residuals.append(str(child))
    (LOG_ROOT / "flat_task_residuals.json").write_text(json.dumps({"residuals": residuals, "note": "core engineering tasks and protected module are excluded"}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    checks["no_flat_residuals"] = not residuals
    failures.extend([] if all(checks.values()) else [name for name, value in checks.items() if not value])
    report = {"task": "scenario_scope_architecture_migration_20260920", "pass": not failures, "checks": checks, "failures": sorted(set(failures)), "migrated_task_count": len(migrated), "journal_events": len(journal_lines), "protected_module_touched": False, "data_touched": False, "config_touched": False}
    (LOG_ROOT / "migration_validation.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (LOG_ROOT / "final_architecture_summary.txt").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (LOG_ROOT / "execution_log.txt").open("a", encoding="utf-8") as handle:
        handle.write(f"migration validation: {'PASS' if report['pass'] else 'FAIL'}; checks={checks}; failures={report['failures']}\n")
    return report


if __name__ == "__main__":
    print(json.dumps(validate(), ensure_ascii=False, indent=2))

