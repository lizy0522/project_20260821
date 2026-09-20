# ruff: noqa: E501

"""Read-only final validator for the internal module layout migration."""

from __future__ import annotations

import ast
import csv
import importlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
AUDIT_ROOT = PROJECT_ROOT / "results" / "core" / "module_internal_layout_reorganization_20260917"
LOG_PATH = PROJECT_ROOT / "work_logs" / "core" / "module_internal_layout_reorganization_20260917" / "execution_log.txt"
HANDOFF_PATH = PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"
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


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


_write_text = _write


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_json(path: Path, value: Any) -> None:
    _write(path, json.dumps(value, ensure_ascii=False, indent=2, default=_json_default))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _run(command: list[str]) -> dict[str, Any]:
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(SCRIPTS_ROOT) + (
        os.pathsep + existing_pythonpath if existing_pythonpath else ""
    )
    result = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
        check=False,
        env=environment,
    )
    return {
        "command": command,
        "returncode": result.returncode,
        "pass": result.returncode == 0,
        "stdout_tail": result.stdout[-4000:],
        "stderr_tail": result.stderr[-4000:],
    }


def _module_files() -> list[Path]:
    return sorted(
        (
            item
            for item in SCRIPTS_ROOT.rglob("*.py")
            if item.is_file() and "__pycache__" not in item.parts
        ),
        key=lambda item: item.relative_to(PROJECT_ROOT).as_posix().lower(),
    )


def _module_name(path: Path) -> str | None:
    try:
        relative = path.relative_to(SCRIPTS_ROOT).with_suffix("")
    except ValueError:
        return None
    if not relative.parts or relative.parts[0] not in MODULES:
        return None
    if relative.name == "__init__":
        return ".".join(relative.parts[:-1])
    return ".".join(relative.parts)


def _available_modules() -> set[str]:
    return {name for path in _module_files() if (name := _module_name(path))}


def _resolve_absolute(name: str, available: set[str]) -> str | None:
    pieces = name.split(".")
    for end in range(len(pieces), 0, -1):
        candidate = ".".join(pieces[:end])
        if candidate in available:
            return candidate
    return None


def _resolve_relative(source: str, level: int, module: str | None, available: set[str]) -> str | None:
    package = source.rpartition(".")[0]
    if level > 1:
        package_parts = package.split(".") if package else []
        package = ".".join(package_parts[: max(0, len(package_parts) - level + 1)])
    target = ".".join(part for part in (package, module or "") if part)
    return _resolve_absolute(target, available)


def _dependency_graph() -> tuple[dict[str, set[str]], list[dict[str, str]]]:
    available = _available_modules()
    graph = {name: set() for name in available}
    stale: list[dict[str, str]] = []
    old_prefixes = (
        "core.metrics",
        "core.signal",
        "core.utils",
        "core.behavior_indexed_dpd",
        "data_manager",
        "behavior_model",
        "behavior_fingerprint_lut_retrieval",
        "low_bandwidth_observation",
        "pa_performance_observation",
        "clustering",
    )
    for path in _module_files():
        source = _module_name(path)
        if source is None:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            targets: list[tuple[str | None, int]] = []
            if isinstance(node, ast.Import):
                targets.extend((alias.name, 0) for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                targets.append((node.module, node.level))
            for name, level in targets:
                if not name:
                    continue
                if level:
                    target = _resolve_relative(source, level, name, available)
                else:
                    target = _resolve_absolute(name, available)
                    if any(name == old or name.startswith(old + ".") for old in old_prefixes):
                        stale.append({"source": str(path.relative_to(PROJECT_ROOT)), "target": name})
                if target is not None and target != source:
                    graph[source].add(target)
    return graph, stale


def _cycles(graph: dict[str, set[str]]) -> list[list[str]]:
    cycles: list[list[str]] = []
    active: list[str] = []
    visited: set[str] = set()

    def visit(node: str) -> None:
        if node in active:
            cycles.append(active[active.index(node) :] + [node])
            return
        if node in visited:
            return
        active.append(node)
        for target in sorted(graph[node]):
            visit(target)
        active.pop()
        visited.add(node)

    for node in sorted(graph):
        visit(node)
    unique: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for cycle in cycles:
        core = cycle[:-1]
        rotations = [tuple(core[i:] + core[:i]) for i in range(len(core))]
        key = min(rotations)
        if key not in seen:
            seen.add(key)
            unique.append(list(key) + [key[0]])
    return unique


def _layout_checks(migration: Any) -> tuple[dict[str, Any], list[dict[str, str]]]:
    from core.shared.project_layout import validate_project_layout

    current = validate_project_layout(PROJECT_ROOT)
    violations = [
        {**row, "detail": row.get("detail", "route-aware layout violation")}
        for row in current["layout_violations"]
    ]
    return current, violations


def _task_rows() -> list[dict[str, Any]]:
    rows = []
    for module in MODULES:
        root = SCRIPTS_ROOT / module
        if module == "retrieval_oriented_model_selection":
            task_dirs = [
                task for route in ("self_hit_oriented", "dpd_shareability_oriented")
                for task in (root / route).iterdir()
                if task.is_dir() and task.name != "__pycache__"
            ]
        else:
            task_dirs = [
                item for item in root.iterdir()
                if item.is_dir() and item.name not in {"shared", "__pycache__"}
            ]
        for task in sorted(task_dirs, key=lambda item: str(item).lower()):
            relative = task.relative_to(root)
            result = PROJECT_ROOT / "results" / module / relative
            log = PROJECT_ROOT / "work_logs" / module / relative
            rows.append(
                {
                    "module": module,
                    "task": relative.as_posix(),
                    "scripts_path": str(task.relative_to(PROJECT_ROOT)),
                    "results_path": str(result.relative_to(PROJECT_ROOT)),
                    "work_logs_path": str(log.relative_to(PROJECT_ROOT)),
                    "scripts_exists": task.is_dir(),
                    "results_exists": result.exists(),
                    "work_logs_exists": log.exists(),
                }
            )
    return rows


def _result_snapshot(path: Path) -> dict[str, Any]:
    files = [item for item in path.rglob("*") if item.is_file()] if path.exists() else []
    return {
        "path": str(path),
        "exists": path.exists(),
        "file_count": len(files),
        "bytes": sum(int(item.stat().st_size) for item in files),
    }


def _import_smoke() -> dict[str, Any]:
    sys.path.insert(0, str(SCRIPTS_ROOT))
    modules = (
        "core.shared",
        "data_management.shared",
        "signal_segmentation.shared",
        "pa_performance_evaluation.shared",
        "behavior_modeling.shared",
        "behavior_fingerprint_retrieval.shared",
        "retrieval_oriented_model_selection.shared",
        "behavior_fingerprint_ranking_consistency.shared",
        "low_bandwidth_behavior_analysis.shared",
        "lut_clustering_compression.shared",
    )
    imported = []
    for name in modules:
        importlib.import_module(name)
        imported.append(name)
    return {"modules": imported, "count": len(imported), "pass": True}


def _public_api_smoke() -> dict[str, Any]:
    checks = (
        ("core.shared.metrics", "cnmse"),
        ("core.shared.signal", "fine_align"),
        ("data_management.shared", "load_by_id"),
        ("signal_segmentation.shared", "build_partition_from_xin"),
        ("pa_performance_evaluation.shared", "collect_scenario2_ilc_acpr"),
        ("behavior_modeling.shared", "build_mp_basis"),
        ("behavior_fingerprint_retrieval.shared", "run_retrieval_analysis"),
        ("behavior_fingerprint_ranking_consistency.shared", "compute_spearman_by_state"),
        ("low_bandwidth_behavior_analysis.shared", "LowBandwidthObservationBank"),
        ("lut_clustering_compression.shared.clustering", "cluster_behavior_states"),
    )
    checked = []
    for module_name, attribute in checks:
        module = importlib.import_module(module_name)
        getattr(module, attribute)
        checked.append(f"{module_name}.{attribute}")
    return {"symbols": checked, "count": len(checked), "pass": True}


def _affected_regression_tests() -> dict[str, Any]:
    tests = {
        "core_shared": "scripts/core/shared/tests/test_core_modules.py",
        "data_management_shared": "scripts/data_management/shared/tests/test_data_manager.py",
        "signal_segmentation_shared": "scripts/signal_segmentation/shared/tests/test_signal_segmentation.py",
        "behavior_modeling_shared": "scripts/behavior_modeling/shared/tests/test_behavior_model.py",
        "odd_order_shared": "scripts/behavior_modeling/shared/tests/test_odd_order_variable_memory_scan.py",
        "envelope75_shared": "scripts/behavior_modeling/shared/tests/test_envelope75.py",
        "sparse_gmp_shared": "scripts/behavior_modeling/shared/tests/test_sparse_gmp.py",
        "retrieval_oriented_scan": "scripts/retrieval_oriented_model_selection/self_hit_oriented/scenario_2_C2_to_Aend_retrieval_oriented_model_scan/test_scenario2_retrieval_oriented_model_scan.py",
        "envelope18_retrieval": "scripts/behavior_fingerprint_retrieval/scenario_2_envelope18_commonB_full_lut_retrieval_5B/test_envelope18_commonB_full_lut.py",
        "envelope19_retrieval": "scripts/behavior_fingerprint_retrieval/scenario_2_envelope19_c2endshared_commonB_full_lut_retrieval_5B/test_envelope19_c2endshared_commonB_full_lut_retrieval.py",
    }
    results = {
        name: _run([sys.executable, "-B", path])
        for name, path in tests.items()
    }
    return {"tests": results, "count": len(results), "pass": all(item["pass"] for item in results.values())}


def _append_log(message: str) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[{__import__('datetime').datetime.now().astimezone().isoformat(timespec='seconds')}] {message.rstrip()}\n")


def _append_handoff(message: str) -> None:
    HANDOFF_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HANDOFF_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{__import__('datetime').datetime.now().astimezone().isoformat(timespec='seconds')} | 工程级维护：{message.rstrip()}\n")


def main() -> int:
    if str(SCRIPTS_ROOT) not in sys.path:
        sys.path.insert(0, str(SCRIPTS_ROOT))
    migration = importlib.import_module(
        "core.module_internal_layout_reorganization_20260917.run_module_internal_layout_reorganization_20260917"
    )
    pre_path = AUDIT_ROOT / "01_pre_reorganization_inventory.txt"
    pre = json.loads(pre_path.read_text(encoding="utf-8"))
    graph, stale_imports = _dependency_graph()
    cycles = _cycles(graph)
    from core.shared.project_layout import validate_project_layout

    dependency_violations = validate_project_layout(PROJECT_ROOT)["dependency_violations"]
    layout, layout_violations = _layout_checks(migration)
    compileall = _run([sys.executable, "-m", "compileall", "-q", "scripts"])
    ruff = _run([sys.executable, "-m", "ruff", "check", "scripts"])
    pip_check = _run([sys.executable, "-m", "pip", "check"])
    import_smoke = _import_smoke()
    api_smoke = _public_api_smoke()
    regression_tests = _affected_regression_tests()
    raw_after = migration._raw_manifest()
    self_after = migration._self_first_snapshot()
    raw_pass = raw_after == pre["raw_manifest"] == migration.EXPECTED_RAW
    self_first_pass = self_after == pre["self_first"]
    checks = {
        "compileall": compileall,
        "ruff": ruff,
        "pip_check": pip_check,
        "ten_module_shared_import_smoke": import_smoke,
        "public_api_smoke": api_smoke,
        "affected_regression_tests": regression_tests,
        "layout_validator": {"pass": layout["pass"], "violations": layout_violations},
        "dependency_validator": {"pass": not dependency_violations, "violations": dependency_violations},
        "cycle_validator": {"pass": not cycles, "cycles": cycles},
        "stale_import_validator": {"pass": not stale_imports, "imports": stale_imports},
        "raw_manifest_unchanged": {"pass": raw_pass, "before": pre["raw_manifest"], "after": raw_after},
        "self_first_tree_unchanged": {"pass": self_first_pass, "before": pre["self_first"], "after": self_after},
    }
    report = {
        "task": "module_internal_layout_reorganization_20260917",
        "mode": "engineering_only_read_only_validation",
        "checks": checks,
        "layout": layout,
        "dependency_violations": dependency_violations,
        "cycles": cycles,
        "stale_imports": stale_imports,
        "task_count": len(_task_rows()),
        "task_mapping": _task_rows(),
        "result_directory": _result_snapshot(AUDIT_ROOT),
        "git": migration._git_snapshot(),
    }
    report["pass"] = all(bool(value.get("pass")) for value in checks.values())
    _write_json(AUDIT_ROOT / "12_validation_report.json", report)
    _write_csv(
        AUDIT_ROOT / "05_task_mapping_manifest.csv",
        _task_rows(),
        ("module", "task", "scripts_path", "results_path", "work_logs_path", "scripts_exists", "results_exists", "work_logs_exists"),
    )
    _write_csv(
        AUDIT_ROOT / "08_dependency_violations.csv",
        dependency_violations,
        ("type", "source", "target", "detail"),
    )
    _write_csv(
        AUDIT_ROOT / "09_file_conflicts.csv",
        layout_violations,
        ("type", "path", "detail"),
    )
    migration._write_import_audit(AUDIT_ROOT / "10_import_dependency_audit_after.txt", "Import dependency audit after final validation")
    _write_text(AUDIT_ROOT / "07_shared_dependency_graph.txt", "\n".join(
        ["Project dependency graph (source -> internal target):", ""]
        + [f"{source} -> {target}" for source in sorted(graph) for target in sorted(graph[source])]
    ))
    post = migration._post_inventory()
    _write_text(AUDIT_ROOT / "11_post_reorganization_inventory.txt", json.dumps(post, ensure_ascii=False, indent=2, default=_json_default))
    tree_lines = ["scripts/"]
    for path in sorted(
        (item for item in SCRIPTS_ROOT.rglob("*") if item.is_dir() and "__pycache__" not in item.parts),
        key=lambda item: item.relative_to(SCRIPTS_ROOT).as_posix().lower(),
    ):
        tree_lines.append("  " * len(path.relative_to(SCRIPTS_ROOT).parts) + path.name + "/")
    _write(AUDIT_ROOT / "13_final_scripts_tree.txt", "\n".join(tree_lines))
    _write(
        AUDIT_ROOT / "14_reorganization_summary.txt",
        "\n".join(
            [
                "Task: module_internal_layout_reorganization_20260917",
                "Scope: source-only shared/task layout normalization.",
                f"Pass: {report['pass']}",
                f"Task directories: {report['task_count']}",
                f"Dependency violations: {len(dependency_violations)}",
                f"Cycles: {len(cycles)}",
                f"Raw manifest unchanged: {raw_pass}",
                f"Self-First tree unchanged: {self_first_pass}",
                "No research runner, model fitting, result recomputation, or data/raw mutation was performed.",
            ]
        ),
    )
    _append_log(
        f"Final validation: pass={report['pass']}; tasks={report['task_count']}; "
        f"dependency_violations={len(dependency_violations)}; cycles={len(cycles)}; "
        f"raw_unchanged={raw_pass}; self_first_unchanged={self_first_pass}; "
        "no research computation executed.",
    )
    _append_handoff(
        "module_internal_layout_reorganization_20260917 final validation "
        f"pass={report['pass']}：每个 scripts 模块均已分为 shared 与具体 task，" 
        f"task_count={report['task_count']}，dependency_violations={len(dependency_violations)}，cycles={len(cycles)}；"
        f"raw manifest unchanged={raw_pass}，Self-First tree unchanged={self_first_pass}；"
        "compileall/Ruff/pip check/10-module shared import/public API smoke 均纳入 12_validation_report.json，未执行科研计算。",
    )
    print(json.dumps({"task": report["task"], "pass": report["pass"], "task_count": report["task_count"], "dependency_violations": len(dependency_violations), "cycles": len(cycles), "raw_unchanged": raw_pass, "self_first_unchanged": self_first_pass}, ensure_ascii=False))
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
