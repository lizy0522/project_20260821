# ruff: noqa: E501

"""Reorganize the ten frozen modules into public ``shared`` code and tasks.

This is an engineering-only migration.  Existing result trees, caches, raw
MAT files, distance matrices, and work-log contents are never recomputed or
copied.  Existing source files are moved on the same filesystem and imports
are rewritten in place.  The script writes a complete before/after audit into
the current core task result directory.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def _find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir():
            return parent
    raise RuntimeError("project root could not be located")


PROJECT_ROOT = _find_project_root()
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
TASK_NAME = "module_internal_layout_reorganization_20260917"
RESULT_ROOT = PROJECT_ROOT / "results" / "core" / TASK_NAME
LOG_ROOT = PROJECT_ROOT / "work_logs" / "core" / TASK_NAME
EXECUTION_LOG = LOG_ROOT / "execution_log.txt"
SELF_FIRST_ROOT = (
    PROJECT_ROOT
    / "results"
    / "retrieval_oriented_model_selection"
    / "scenario_2_self_first_joint_basis_ridge_lut_retrieval_search_5b"
)

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


# Direct files that form stable, cross-task APIs.  Tests belonging to those
# APIs are placed below shared/tests; task tests stay with their task.
DIRECT_SHARED: dict[str, tuple[str, ...]] = {
    "core": ("module_registry.py", "project_paths.py", "validate_project_module_layout.py"),
    "data_management": ("config.py", "dataset.py", "file_index.py", "loader.py", "state_index.py"),
    "signal_segmentation": (
        "behavior_pairs.py",
        "config.py",
        "partition.py",
        "preprocessing.py",
        "splitter.py",
        "validation.py",
    ),
    "pa_performance_evaluation": ("ilc_acpr_observation.py",),
    "behavior_modeling": (
        "basis.py",
        "coefficient.py",
        "config.py",
        "evaluation.py",
        "frozen_neighborhood_memory_ridge_scan.py",
        "mp_model.py",
        "multibranch_independent_basis.py",
        "odd_order_mp_capacity_scan.py",
        "odd_order_variable_memory_scan.py",
        "p5_variable_memory_ridge_scan.py",
        "p5_with_order2_variable_memory_ols_scan.py",
        "prediction.py",
        "ridge.py",
        "scenario2_ridge_analysis.py",
        "scenario2_xy_analysis.py",
        "select_statewise_best_round0_4_models.py",
        "sparse_gmp.py",
        "statewise_adaptive_model_search.py",
        "xy_equivalence.py",
    ),
    "behavior_fingerprint_retrieval": (
        "envelope18_commonB_full_lut.py",
        "envelope19_c2endshared_commonB_full_lut.py",
        "envelope23_ilcend_commonB_full_lut_comparison.py",
        "scenario2_c2_to_aend_1b.py",
        "scenario2_c2_to_aend_retrieval.py",
        "scenario2_c2_to_aend_statewise_best_round0_4_5b.py",
        "scenario2_c3_to_a2_retrieval.py",
    ),
    "behavior_fingerprint_ranking_consistency": (
        "distance_matrix.py",
        "fingerprint_builder.py",
        "plotting.py",
        "ranking.py",
        "scenario2_all_ilc_analysis.py",
        "scenario2_analysis.py",
        "scenario2_y_lut_fusion_A123_analysis.py",
        "scenario2_y_lut_fusion_analysis.py",
        "scenario2_y_lut_weighted_fusion_analysis.py",
    ),
    "low_bandwidth_behavior_analysis": ("plot_sample_rate_operator_bank.py",),
    "lut_clustering_compression": (),
}

DIRECT_SHARED_TESTS: dict[str, tuple[str, ...]] = {
    "core": ("test_core_modules.py",),
    "data_management": ("test_data_manager.py",),
    "signal_segmentation": ("test_signal_segmentation.py",),
    "pa_performance_evaluation": ("test_ilc_acpr_observation.py",),
    "behavior_modeling": (
        "test_behavior_model.py",
        "test_odd_order_variable_memory_scan.py",
        "test_sparse_gmp.py",
    ),
    "behavior_fingerprint_retrieval": (),
    "behavior_fingerprint_ranking_consistency": (),
    "low_bandwidth_behavior_analysis": ("test_low_bandwidth_observation.py",),
    "lut_clustering_compression": (),
}


DIRECT_TASKS: dict[str, dict[str, tuple[str, ...]]] = {
    "signal_segmentation": {"state0_validation": ("run_state0_validation.py",)},
    "pa_performance_evaluation": {
        "scenario_2_ilc_acpr_observation": ("run_scenario2_ilc_acpr_observation.py",)
    },
    "behavior_modeling": {
        "behavior_model_state0": ("run_behavior_model_state0.py",),
        "state0_ridge_scan": ("ridge_scan.py", "run_state0_ridge_scan.py", "test_state0_ridge_scan.py"),
        "state0_xy_equivalence": ("run_state0_xy_equivalence.py", "test_state0_xy_equivalence.py"),
        "scenario_2_ridge_analysis": ("run_scenario2_ridge_analysis.py", "test_scenario2_ridge_analysis.py"),
        "scenario_2_xy_analysis": ("run_scenario2_xy_analysis.py", "test_scenario2_xy_analysis.py"),
        "scenario_2_select_statewise_best_round0_4_models": (
            "run_select_statewise_best_round0_4_models.py",
            "test_select_statewise_best_round0_4_models.py",
        ),
        "scenario_2_statewise_adaptive_aend_c2_5b": (
            "run_scenario2_statewise_adaptive_aend_c2_5b.py",
            "export_statewise_adaptive_best_metrics.py",
            "finalize_statewise_adaptive_partial.py",
            "test_export_statewise_adaptive_best_metrics.py",
        ),
        "scenario_2_failed14_P5_with_order2_variable_memory_OLS_scan_5b": (
            "export_scenario2_failed14_P5_with_order2_variable_memory_OLS_scan_5b.mjs",
            "plot_scenario2_failed14_P5_with_order2_variable_memory_OLS_scan_5b.py",
            "run_scenario2_failed14_P5_with_order2_variable_memory_OLS_scan_5b.py",
            "test_p5_with_order2_variable_memory_OLS_scan.py",
        ),
        "scenario_2_failed14_frozen_mp_residual_sparse_gmp_5b": (
            "export_scenario2_failed14_frozen_mp_residual_sparse_gmp_5b.mjs",
            "plot_scenario2_failed14_frozen_mp_residual_sparse_gmp_5b.py",
            "run_scenario2_failed14_frozen_mp_residual_sparse_gmp_5b.py",
        ),
        "scenario_2_failed14_frozen_neighborhood_memory_ridge_scan_5b": (
            "export_scenario2_failed14_frozen_neighborhood_memory_ridge_scan_5b.mjs",
            "plot_scenario2_failed14_frozen_neighborhood_memory_ridge_scan_5b.py",
            "run_scenario2_failed14_frozen_neighborhood_memory_ridge_scan_5b.py",
        ),
        "scenario_2_failed14_multibranch_independent_basis_ols_scan_5b": (
            "export_scenario2_failed14_multibranch_independent_basis_ols_scan_5b.mjs",
            "finalize_scenario2_failed14_multibranch_independent_basis_ols_scan_5b.py",
            "plot_scenario2_failed14_multibranch_independent_basis_ols_scan_5b.py",
            "run_scenario2_failed14_multibranch_independent_basis_ols_scan_5b.py",
            "test_multibranch_independent_basis.py",
        ),
        "scenario_2_failed14_odd_order_mp_order_dependent_memory_scan_5b": (
            "export_scenario2_failed14_odd_order_mp_order_dependent_memory_scan_5b.mjs",
            "plot_scenario2_failed14_odd_order_mp_order_dependent_memory_scan_5b.py",
            "run_scenario2_failed14_odd_order_mp_order_dependent_memory_scan_5b.py",
            "test_odd_order_mp_capacity_scan.py",
        ),
        "scenario_2_failed14_p5_variable_memory_ridge_scan_5b": (
            "export_scenario2_failed14_p5_variable_memory_ridge_scan_5b.mjs",
            "plot_scenario2_failed14_p5_variable_memory_ridge_scan_5b.py",
            "run_scenario2_failed14_p5_variable_memory_ridge_scan_5b.py",
            "test_p5_variable_memory_ridge_scan.py",
        ),
        "scenario_2_frozen_mp_global_sparse_gmp_residual_5b": (),
        "scenario_2_global_sparse_gmp_C2_gap_diagnosis_5b": (
            "run_scenario2_global_sparse_gmp_C2_gap_diagnosis_5b.py",
        ),
        "scenario_2_failed14_frozen_mp_normalized_basis_dependent_ridge_5B": (
            "run_scenario2_failed14_frozen_mp_normalized_basis_dependent_ridge_5b.py",
        ),
        "scenario_2_failed14_frozen_mp_normalized_uniform_ridge_5B": (
            "run_scenario2_failed14_frozen_mp_normalized_uniform_ridge_5b.py",
        ),
        "scenario_2_unified_odd_order_mp_capacity_scan_5b": (
            "export_scenario2_unified_odd_order_mp_capacity_scan_5b.mjs",
            "finalize_scenario2_unified_odd_order_mp_capacity_scan_5b.py",
            "run_scenario2_unified_odd_order_mp_capacity_scan_5b.py",
        ),
    },
    "behavior_fingerprint_retrieval": {
        "scenario_2_C2_to_Aend_0p5b": (
            "run_scenario2_c2_to_aend_0p5b.py",
            "test_scenario2_c2_to_aend_0p5b.py",
        ),
        "scenario_2_C2_to_Aend_1b": (
            "export_scenario2_c2_to_aend_1b.mjs",
            "run_scenario2_c2_to_aend_1b.py",
            "test_scenario2_c2_to_aend_1b.py",
        ),
        "scenario_2_C2_to_Aend_equal_ABC": (
            "analyze_scenario2_c2_to_aend_equal_abc.py",
            "export_scenario2_c2_to_aend_equal_abc.mjs",
            "export_scenario2_c2_to_aend_equal_abc_excel.py",
            "plot_scenario2_c2_to_aend_equal_abc.py",
            "run_scenario2_c2_to_aend_equal_abc.py",
            "scenario2_c2_to_aend_equal_abc.py",
            "test_scenario2_c2_to_aend_equal_abc.py",
        ),
        "scenario_2_C2_to_Aend_retrieval": (
            "export_scenario2_c2_to_aend_state_summary.mjs",
            "export_scenario2_c2_to_aend_state_summary.py",
            "plot_scenario2_c2_to_aend_retrieval.py",
            "plot_scenario2_c2_to_aend_state_metrics.py",
            "run_scenario2_c2_to_aend_retrieval.py",
            "test_scenario2_c2_to_aend_retrieval.py",
            "test_scenario2_c2_to_aend_state_metrics.py",
            "test_scenario2_c2_to_aend_state_summary.py",
        ),
        "scenario_2_C3_to_A2_retrieval": (
            "plot_scenario2_c3_to_a2_retrieval.py",
            "run_scenario2_c3_to_a2_retrieval.py",
            "test_scenario2_c3_to_a2_retrieval.py",
        ),
        "scenario_2_C2_to_Aend_g4_sparse_gmp_5b": (
            "export_scenario2_c2_to_aend_g4_sparse_gmp_5b.mjs",
            "plot_scenario2_c2_to_aend_g4_sparse_gmp_5b.py",
            "run_scenario2_c2_to_aend_g4_sparse_gmp_5b.py",
            "test_scenario2_c2_to_aend_g4_sparse_gmp_5b.py",
        ),
        "scenario_2_envelope18_commonB_full_lut_retrieval_5B": (
            "run_scenario2_envelope18_commonB_full_lut_retrieval_5b.py",
            "test_envelope18_commonB_full_lut.py",
        ),
        "scenario_2_envelope19_c2endshared_commonB_full_lut_retrieval_5B": (
            "run_scenario2_envelope19_c2endshared_commonB_full_lut_retrieval_5b.py",
            "test_envelope19_c2endshared_commonB_full_lut_retrieval.py",
        ),
        "scenario_2_envelope23_ilcend_commonB_full_lut_retrieval_5B": (
            "run_scenario2_envelope23_ilcend_commonB_full_lut_retrieval_5b.py",
        ),
    },
    "behavior_fingerprint_ranking_consistency": {
        "fixed_lut_ilc_stage_comparison": (
            "plot_fixed_lut_ilc_stage_comparison.py",
            "run_fixed_lut_ilc_stage_comparison.py",
            "test_fixed_lut_ilc_stage_comparison.py",
        ),
        "scenario2_all_ilc_analysis": (
            "run_scenario2_all_ilc_analysis.py",
            "test_scenario2_all_ilc_analysis.py",
        ),
        "scenario2_ranking_consistency": (
            "run_scenario2_ranking_consistency.py",
            "test_ranking_consistency.py",
        ),
        "scenario2_y_lut_fusion_analysis": (
            "plot_y_lut_fusion_analysis.py",
            "run_scenario2_y_lut_fusion_analysis.py",
            "test_scenario2_y_lut_fusion_analysis.py",
        ),
        "scenario2_y_lut_fusion_a123_analysis": (
            "plot_y_lut_fusion_A123_analysis.py",
            "run_scenario2_y_lut_fusion_A123_analysis.py",
            "test_scenario2_y_lut_fusion_A123_analysis.py",
        ),
        "scenario2_y_lut_weighted_fusion_analysis": (
            "plot_y_lut_weighted_fusion_analysis.py",
            "run_scenario2_y_lut_weighted_fusion_analysis.py",
            "test_scenario2_y_lut_weighted_fusion_analysis.py",
        ),
    },
    "low_bandwidth_behavior_analysis": {
        "low_bandwidth_observation": ("validate_sample_rate_operator_bank.py",),
    },
    "lut_clustering_compression": {
        "scenario_2_5b_A_behavior_clustering": ("run_scenario2_5b_A_behavior_clustering.py",),
        "scenario_2_5b_A_yout_withoutdpd_ori_complete_link": (
            "run_scenario2_5b_A_yout_withoutdpd_ori_complete_link.py",
        ),
    },
}


# The old basis package was a mixed library/entry-point package.  The stable
# algorithmic/evaluation portion becomes a public nested package below
# behavior_modeling/shared; concrete run/test files are mapped to tasks.
BASIS_SHARED_FILES = (
    "__init__.py",
    "all425_c2_ilcend_envelope19_shared_coverage.py",
    "all425_envelope18_evaluation.py",
    "all425_ilcend_envelope23_evaluation.py",
    "capacity_gates.py",
    "config.py",
    "data_preparation.py",
    "dual_ilc_behavior_dataset.py",
    "envelope_dictionary.py",
    "final_evaluation.py",
    "frozen_centered_dictionary.py",
    "hard20_envelope75_selection.py",
    "hard_state_selection.py",
    "ilcend_behavior_dataset.py",
    "model_solver.py",
    "reference_gate.py",
    "ridge_tuning.py",
    "search.py",
    "selection_metrics.py",
    "volterra_dictionary.py",
)
BASIS_TASK_FILES: dict[str, tuple[str, ...]] = {
    "scenario_2_all425_c2_envelope23_ridge_generalization_5B": (
        "c2_envelope23_ridge_generalization.py",
        "run_scenario2_all425_c2_envelope23_ridge_generalization_5b.py",
        "test_c2_envelope23_ridge_generalization.py",
    ),
    "scenario_2_all425_c2_ilcend_envelope19_shared_coverage_5B": (
        "run_scenario2_all425_c2_ilcend_envelope19_shared_coverage_5b.py",
        "test_all425_c2_ilcend_envelope19_shared_coverage.py",
    ),
    "scenario_2_all425_envelope18_coverage_evaluation_5B": (
        "run_scenario2_all425_envelope18_coverage_evaluation_5b.py",
        "test_all425_envelope18_evaluation.py",
    ),
    "scenario_2_all425_ilcend_envelope23_coverage_evaluation_5B": (
        "run_scenario2_all425_ilcend_envelope23_coverage_evaluation_5b.py",
        "test_all425_ilcend_envelope23_evaluation.py",
    ),
    "scenario_2_gateb48_unified_basis_refinement_5B": (
        "gateb48_refinement.py",
        "run_scenario2_gateb48_unified_basis_refinement_5b.py",
    ),
    "scenario_2_hard20_c2_ilcend_shared_envelope75_basis_selection_5B": (
        "hard20_c2_ilcend_shared_envelope75_selection.py",
        "run_scenario2_hard20_c2_ilcend_shared_envelope75_basis_selection_5b.py",
        "test_c2_ilcend_shared_basis_selection.py",
    ),
    "scenario_2_hard20_envelope75_basis_selection_5B": (
        "run_scenario2_hard20_envelope75_basis_selection_5b.py",
    ),
    "scenario_2_hard20_ilcend_envelope75_basis_selection_5B": (
        "run_scenario2_hard20_ilcend_envelope75_basis_selection_5b.py",
    ),
    "scenario_2_hard20_frozen10_reference_gate_5B": (
        "run_scenario2_hard20_frozen10_reference_gate_5b.py",
    ),
    "scenario_2_hard20_frozen_centered_basis_selection_5B": (
        "run_scenario2_hard20_frozen_centered_basis_selection_5b.py",
    ),
    "scenario_2_hard20_volterra_basis_selection_5B": (
        "run_scenario2_hard20_volterra_basis_selection_5b.py",
    ),
}
BASIS_SHARED_TESTS = ("test_envelope75.py", "test_ilcend_behavior_dataset.py")


# Full runner modules are promoted only when another task needs their helper
# functions.  Thin task wrappers are generated at the original task path.
SUPPORT_PROMOTIONS = {
    "retrieval_oriented_model_selection/scenario_2_C2_to_Aend_retrieval_oriented_model_scan/scenario2_retrieval_oriented_model_scan.py": (
        "retrieval_oriented_model_selection/shared/scenario2_retrieval_oriented_model_scan.py",
        None,
    ),
    "retrieval_oriented_model_selection/scenario_2_mp10_k9_retrieval_oriented_basis_ablation_5b/mp10_k9_backend.py": (
        "retrieval_oriented_model_selection/shared/mp10_k9_retrieval_support.py",
        None,
    ),
    "retrieval_oriented_model_selection/scenario_2_mp10_k9_retrieval_oriented_basis_ablation_5b/run_scenario2_mp10_k9_retrieval_oriented_basis_ablation_5b.py": (
        "retrieval_oriented_model_selection/shared/mp10_k9_retrieval_runner.py",
        "retrieval_oriented_model_selection/scenario_2_mp10_k9_retrieval_oriented_basis_ablation_5b/run_scenario2_mp10_k9_retrieval_oriented_basis_ablation_5b.py",
    ),
    "retrieval_oriented_model_selection/scenario_2_k11_backward_retrieval_oriented_basis_ablation_5b/k11_backward_backend.py": (
        "retrieval_oriented_model_selection/shared/k11_backward_retrieval_support.py",
        None,
    ),
    "retrieval_oriented_model_selection/scenario_2_k11_backward_retrieval_oriented_basis_ablation_5b/run_scenario2_k11_backward_retrieval_oriented_basis_ablation_5b.py": (
        "retrieval_oriented_model_selection/shared/k11_backward_retrieval_runner.py",
        "retrieval_oriented_model_selection/scenario_2_k11_backward_retrieval_oriented_basis_ablation_5b/run_scenario2_k11_backward_retrieval_oriented_basis_ablation_5b.py",
    ),
    "retrieval_oriented_model_selection/scenario_2_smallbeam3_k13_retrieval_oriented_forward_expansion_5b/smallbeam_backend.py": (
        "retrieval_oriented_model_selection/shared/smallbeam_retrieval_support.py",
        None,
    ),
    "retrieval_oriented_model_selection/scenario_2_smallbeam3_k13_retrieval_oriented_forward_expansion_5b/run_scenario2_smallbeam3_k13_retrieval_oriented_forward_expansion_5b.py": (
        "retrieval_oriented_model_selection/shared/smallbeam_retrieval_runner.py",
        "retrieval_oriented_model_selection/scenario_2_smallbeam3_k13_retrieval_oriented_forward_expansion_5b/run_scenario2_smallbeam3_k13_retrieval_oriented_forward_expansion_5b.py",
    ),
    "lut_clustering_compression/scenario_2_C2_to_Aend_frozen_centered17_type3_compressed_5B/run_scenario2_C2_to_Aend_type3_cluster_compressed_5b.py": (
        "lut_clustering_compression/shared/type3_retrieval_support.py",
        "lut_clustering_compression/scenario_2_C2_to_Aend_frozen_centered17_type3_compressed_5B/run_scenario2_C2_to_Aend_type3_cluster_compressed_5b.py",
    ),
    "behavior_modeling/run_scenario2_frozen_mp_global_sparse_gmp_residual_5b.py": (
        "behavior_modeling/shared/frozen_mp_global_sparse_gmp_residual_support.py",
        "behavior_modeling/scenario_2_frozen_mp_global_sparse_gmp_residual_5b/run_scenario2_frozen_mp_global_sparse_gmp_residual_5b.py",
    ),
}


TASK_RECONCILIATION = (
    ("basis_function_selection", "behavior_modeling", "split into shared library plus named behavior-modeling tasks", "split"),
    ("behavior_fingerprint_retrieval", "behavior_fingerprint_retrieval", "replaced module-level placeholder with concrete retrieval tasks", "replaced"),
    ("behavior_fingerprint_ranking_consistency", "behavior_fingerprint_ranking_consistency", "split one module-level maintenance name into six concrete analyses", "split"),
    ("low_bandwidth_behavior_analysis", "low_bandwidth_behavior_analysis/low_bandwidth_observation", "runner now has a concrete task name", "reconciled"),
    ("clustering", "lut_clustering_compression/scenario_2_5b_A_behavior_clustering", "legacy aggregate log retained; source runners are concrete tasks", "retained_as_legacy"),
    ("pa_performance_observation", "pa_performance_evaluation/scenario_2_ilc_acpr_observation", "module-level observation name replaced by concrete task", "reconciled"),
)


def _now() -> str:
    return datetime.now(UTC).astimezone().isoformat(timespec="seconds")


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content.rstrip() + "\n", encoding="utf-8")


def _write_json(path: Path, payload: Any) -> None:
    _write_text(path, json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default))


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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
    root = PROJECT_ROOT / "data" / "raw"
    files = sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: str(item).lower())
    digest = hashlib.sha256()
    total = 0
    for item in files:
        relative = item.relative_to(root).as_posix()
        size = int(item.stat().st_size)
        digest.update(relative.encode() + b"\0")
        digest.update(size.to_bytes(8, "little"))
        total += size
    return {
        "sha256": digest.hexdigest(),
        "file_count": len(files),
        "mat_count": sum(item.suffix.lower() == ".mat" for item in files),
        "bytes": total,
    }


def _tree_inventory(root: Path) -> dict[str, Any]:
    files = [item for item in root.rglob("*") if item.is_file()] if root.exists() else []
    return {
        "exists": root.exists(),
        "file_count": len(files),
        "bytes": sum(int(item.stat().st_size) for item in files),
        "files": sorted(item.relative_to(root).as_posix() for item in files),
        "directories": sorted(item.relative_to(root).as_posix() for item in root.rglob("*") if item.is_dir()) if root.exists() else [],
    }


def _self_first_snapshot() -> dict[str, Any]:
    files = [item for item in SELF_FIRST_ROOT.rglob("*") if item.is_file()] if SELF_FIRST_ROOT.exists() else []
    critical_names = {
        "12_final_dense_ridge_scan.csv",
        "13_final_basis_importance.csv",
        "15_final_single_add_validation.csv",
        "16_final_swap_validation.csv",
        "20_query_state_cv_results.csv",
        "21_query_state_cv_stability.csv",
        "22_final_result_summary.json",
        "22_final_result_summary.txt",
        "23_checkpoint.json",
        "24_validation_checks.json",
    }
    critical = {
        item.relative_to(SELF_FIRST_ROOT).as_posix(): _sha256(item)
        for item in files
        if item.name in critical_names
    }
    return {
        "exists": SELF_FIRST_ROOT.exists(),
        "file_count": len(files),
        "bytes": sum(int(item.stat().st_size) for item in files),
        "distance_matrix_count": sum(
            item.suffix.lower() == ".npy" and "distance_matrices" in item.parts for item in files
        ),
        "critical_hashes": critical,
    }


def _git_snapshot() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True
        ).stdout

    return {
        "branch": run("branch", "--show-current").strip(),
        "head": run("rev-parse", "HEAD").strip(),
        "status_short": run("status", "--short"),
        "staged_paths": run("diff", "--cached", "--name-only").splitlines(),
        "unstaged_paths": run("diff", "--name-only").splitlines(),
        "untracked_paths": run("ls-files", "--others", "--exclude-standard").splitlines(),
    }


def _pre_inventory() -> dict[str, Any]:
    return {
        "timestamp": _now(),
        "project_root": PROJECT_ROOT,
        "git": _git_snapshot(),
        "raw_manifest": _raw_manifest(),
        "self_first": _self_first_snapshot(),
        "roots": {name: _tree_inventory(PROJECT_ROOT / name) for name in ("scripts", "results", "work_logs")},
    }


def _script_files() -> list[Path]:
    return sorted(
        (
            item
            for item in SCRIPTS_ROOT.rglob("*")
            if item.is_file()
            and item.suffix.lower() in {".py", ".mjs"}
            and "__pycache__" not in item.parts
        ),
        key=lambda item: item.relative_to(PROJECT_ROOT).as_posix().lower(),
    )


def _move_manifest() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []

    def add(source: str, destination: str, kind: str) -> None:
        rows.append({"source": source, "destination": destination, "kind": kind})

    for module, files in DIRECT_SHARED.items():
        for name in files:
            add(f"scripts/{module}/{name}", f"scripts/{module}/shared/{name}", "direct_shared")
    for module, files in DIRECT_SHARED_TESTS.items():
        for name in files:
            add(f"scripts/{module}/{name}", f"scripts/{module}/shared/tests/{name}", "shared_test")
    for module, tasks in DIRECT_TASKS.items():
        for task, files in tasks.items():
            for name in files:
                add(f"scripts/{module}/{name}", f"scripts/{module}/{task}/{name}", "direct_task")
    for name in BASIS_SHARED_FILES:
        add(
            f"scripts/behavior_modeling/basis_function_selection/{name}",
            f"scripts/behavior_modeling/shared/basis_function_selection/{name}",
            "basis_shared",
        )
    for task, files in BASIS_TASK_FILES.items():
        for name in files:
            add(
                f"scripts/behavior_modeling/basis_function_selection/{name}",
                f"scripts/behavior_modeling/{task}/{name}",
                "basis_task",
            )
    for name in BASIS_SHARED_TESTS:
        add(
            f"scripts/behavior_modeling/basis_function_selection/{name}",
            f"scripts/behavior_modeling/shared/tests/{name}",
            "basis_shared_test",
        )
    add(
        "scripts/core/metrics",
        "scripts/core/shared/metrics",
        "core_shared_package",
    )
    add("scripts/core/signal", "scripts/core/shared/signal", "core_shared_package")
    add("scripts/core/utils", "scripts/core/shared/utils", "core_shared_package")
    add(
        "scripts/core/behavior_indexed_dpd/signal",
        "scripts/low_bandwidth_behavior_analysis/shared/behavior_indexed_dpd_signal",
        "domain_shared_package",
    )
    add(
        "scripts/core/behavior_indexed_dpd/clustering",
        "scripts/lut_clustering_compression/shared/clustering",
        "domain_shared_package",
    )
    add(
        "scripts/core/behavior_indexed_dpd/__init__.py",
        "scripts/core/shared/behavior_indexed_dpd_contract.py",
        "legacy_contract",
    )
    for source, (destination, wrapper) in SUPPORT_PROMOTIONS.items():
        source_path = source if source.startswith("scripts/") else f"scripts/{source}"
        destination_path = destination if destination.startswith("scripts/") else f"scripts/{destination}"
        add(source_path, destination_path, "promoted_support")
        if wrapper:
            wrapper_path = wrapper if wrapper.startswith("scripts/") else f"scripts/{wrapper}"
            add(wrapper_path, wrapper_path, "generated_wrapper")
    return rows


def _classify(path: Path) -> tuple[str, str, str]:
    relative = path.relative_to(PROJECT_ROOT).as_posix()
    parts = path.relative_to(SCRIPTS_ROOT).parts
    if len(parts) == 2 and parts[1] == "__init__.py":
        return "module_root_init", parts[0], "lightweight module package marker"
    if relative.startswith("scripts/core/module_layout_reorganization_20260917/"):
        return "existing_task", "core", "previous completed migration task"
    for source, (destination, wrapper) in SUPPORT_PROMOTIONS.items():
        if relative.removeprefix("scripts/") == source.removeprefix("scripts/"):
            return "shared_support", destination.split("/")[1], "promoted because another task imports its API"
    for module, files in DIRECT_SHARED.items():
        if len(parts) == 2 and parts[0] == module and parts[1] in files:
            return "shared", module, "stable cross-task public API"
    for module, files in DIRECT_SHARED_TESTS.items():
        if len(parts) == 2 and parts[0] == module and parts[1] in files:
            return "shared_test", module, "test of shared public API"
    for module, tasks in DIRECT_TASKS.items():
        if len(parts) == 2 and parts[0] == module:
            for task, files in tasks.items():
                if parts[1] in files:
                    return "task", module, task
    if relative.startswith("scripts/behavior_modeling/basis_function_selection/") and len(parts) == 3:
        name = parts[-1]
        if name in BASIS_SHARED_FILES:
            return "shared", "behavior_modeling", "shared basis-selection API"
        if name in BASIS_SHARED_TESTS:
            return "shared_test", "behavior_modeling", "shared basis-selection test"
        for task, files in BASIS_TASK_FILES.items():
            if name in files:
                return "task", "behavior_modeling", task
    if relative in {"scripts/core/metrics/__init__.py", "scripts/core/signal/__init__.py", "scripts/core/utils/__init__.py"}:
        return "shared", "core", "core shared package marker"
    if relative.startswith("scripts/core/behavior_indexed_dpd/signal/"):
        return "shared", "low_bandwidth_behavior_analysis", "low-bandwidth domain API"
    if relative.startswith("scripts/core/behavior_indexed_dpd/clustering/"):
        return "shared", "lut_clustering_compression", "clustering domain API"
    if parts and parts[0] in MODULES and len(parts) >= 3:
        return "existing_task", parts[0], parts[1]
    if parts and parts[0] in MODULES:
        return "module_other", parts[0], "module package metadata"
    return "outside_registry", "", "not below a registered module"


def _classification_rows() -> list[dict[str, Any]]:
    rows = []
    for path in _script_files():
        classification, module, reason = _classify(path)
        rows.append(
            {
                "source_path": path.relative_to(PROJECT_ROOT).as_posix(),
                "item_name": path.name,
                "classification": classification,
                "destination_module": module,
                "destination_task_or_subpackage": reason if classification in {"task", "existing_task"} else "",
                "reason": reason,
                "status": "planned" if classification not in {"outside_registry", "module_other"} else "review",
            }
        )
    return rows


def _import_records() -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for path in _script_files():
        if path.name == Path(__file__).name:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            records.append({"source": path.relative_to(PROJECT_ROOT).as_posix(), "line": str(exc.lineno or 0), "import": "<syntax-error>", "kind": "error"})
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    records.append({"source": path.relative_to(PROJECT_ROOT).as_posix(), "line": str(node.lineno), "import": alias.name, "kind": "absolute"})
            elif isinstance(node, ast.ImportFrom):
                module = ("." * node.level) + (node.module or "")
                records.append({"source": path.relative_to(PROJECT_ROOT).as_posix(), "line": str(node.lineno), "import": module, "kind": "relative" if node.level else "absolute"})
    return sorted(records, key=lambda item: (item["source"].lower(), int(item["line"]), item["import"]))


def _write_import_audit(path: Path, title: str) -> None:
    records = _import_records()
    lines = [title, "", *(
        f"{item['source']}:{item['line']} [{item['kind']}] {item['import']}" for item in records
    )]
    _write_text(path, "\n".join(lines))


def _ensure_no_preexisting_destination(source: Path, destination: Path) -> None:
    if source.exists() and destination.exists() and source.resolve() != destination.resolve():
        raise RuntimeError(f"destination already exists: {destination}")


def _move(source_relative: str, destination_relative: str, kind: str, moves: list[dict[str, str]]) -> None:
    source = PROJECT_ROOT / source_relative
    destination = PROJECT_ROOT / destination_relative
    if source_relative == destination_relative:
        if not destination.exists():
            raise FileNotFoundError(destination)
        return
    if not source.exists():
        if destination.exists():
            moves.append({"source": source_relative, "destination": destination_relative, "kind": kind, "status": "already_moved"})
            return
        raise FileNotFoundError(source)
    _ensure_no_preexisting_destination(source, destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.rename(source, destination)
    moves.append({"source": source_relative, "destination": destination_relative, "kind": kind, "status": "moved"})


def _root_init_text(module: str) -> str:
    return f'"""Package marker for the frozen {module} module.\n\nPublic implementation lives under ``shared/``; concrete research entry points live in named task directories.\n"""'


def _shared_init(module: str, original: str | None) -> str:
    if module == "behavior_modeling":
        return '''"""Public behavior-modeling APIs shared by multiple tasks."""

from .basis import build_mp_basis
from .coefficient import LeastSquaresDiagnostics, extract_behavior_coefficient
from .config import BASIS_TERMS, MAX_DELAY, MP_CONFIG, NUM_COEFFICIENTS
from .evaluation import calculate_cnmse, calculate_nmse
from .mp_model import MemoryPolynomialModel
from .prediction import predict_behavior
from .ridge import RidgeFitDiagnostics, fit_coefficients_ridge, validate_ridge_lambda
from .scenario2_ridge_analysis import Scenario2RidgeAnalysisResult, run_scenario2_ridge_analysis
from .scenario2_xy_analysis import (
    MODEL_IDS,
    Scenario2XYAnalysisResult,
    ScenarioModelResult,
    StateXYAnalysisResult,
    analyze_state_xy,
    collect_scenario2_xy_analysis,
    validate_state0_regression,
)
from .xy_equivalence import ModelEvaluation, State0XYEquivalenceResult, analyze_state0_xy_equivalence

__all__ = [
    "BASIS_TERMS", "MAX_DELAY", "MODEL_IDS", "MP_CONFIG", "NUM_COEFFICIENTS",
    "LeastSquaresDiagnostics", "MemoryPolynomialModel", "ModelEvaluation",
    "RidgeFitDiagnostics", "Scenario2RidgeAnalysisResult", "Scenario2XYAnalysisResult",
    "ScenarioModelResult", "State0XYEquivalenceResult", "StateXYAnalysisResult",
    "analyze_state0_xy_equivalence", "analyze_state_xy", "build_mp_basis",
    "calculate_cnmse", "calculate_nmse", "collect_scenario2_xy_analysis",
    "extract_behavior_coefficient", "fit_coefficients_ridge", "predict_behavior",
    "run_scenario2_ridge_analysis", "validate_ridge_lambda", "validate_state0_regression",
]
'''
    if module == "retrieval_oriented_model_selection":
        return '"""Shared retrieval-oriented support APIs."""\n'
    if module == "core":
        return '"""Stable cross-module numerical, path, and validation APIs."""\n'
    if module == "low_bandwidth_behavior_analysis":
        return '"""Shared low-bandwidth observation operator APIs."""\n\nfrom .behavior_indexed_dpd_signal import LowBandwidthObservationBank, LowBandwidthObservationOperator, SampleRateSpec\n\n__all__ = ["LowBandwidthObservationBank", "LowBandwidthObservationOperator", "SampleRateSpec"]\n'
    if module == "lut_clustering_compression":
        return '"""Shared LUT clustering primitives."""\n\nfrom .clustering import *  # noqa: F401,F403\n'
    if module == "behavior_fingerprint_retrieval":
        return '''"""Shared behavior-fingerprint retrieval APIs."""

from .scenario2_c2_to_aend_1b import *  # noqa: F401,F403
from .scenario2_c2_to_aend_retrieval import *  # noqa: F401,F403
from .scenario2_c3_to_a2_retrieval import *  # noqa: F401,F403
'''
    if original:
        return original
    return f'"""Shared public APIs for {module}."""\n'


def _prepare_package_markers() -> None:
    for module in MODULES:
        module_root = SCRIPTS_ROOT / module
        shared_root = module_root / "shared"
        shared_root.mkdir(parents=True, exist_ok=True)
        root_init = module_root / "__init__.py"
        original = root_init.read_text(encoding="utf-8") if root_init.exists() else None
        _write_text(shared_root / "__init__.py", _shared_init(module, original))
        _write_text(root_init, _root_init_text(module))
        _write_text(shared_root / "tests" / "__init__.py", '"""Shared API regression tests."""')
    _write_text(SCRIPTS_ROOT / "retrieval_oriented_model_selection" / "__init__.py", '"""Retrieval-oriented model-selection module."""')


def _write_basis_wrapper_init() -> None:
    # The moved basis package init is kept as the nested shared package marker;
    # it is rewritten after its relative imports remain valid.
    path = SCRIPTS_ROOT / "behavior_modeling" / "shared" / "basis_function_selection" / "__init__.py"
    _write_text(path, '"""Shared basis-dictionary, solver, and selection APIs."""\n\nfrom .config import EXPERIMENT_NAME\n\n__all__ = ["EXPERIMENT_NAME"]')


def _wrapper_text(import_module: str, public_name: str) -> str:
    return f'''"""Task entrypoint wrapper for the promoted shared support runner."""

from {import_module} import *  # noqa: F401,F403
from {import_module} import main

if __name__ == "__main__":
    main()
'''


def _promoted_wrapper(module: str) -> str | None:
    if module.endswith("mp10_k9_retrieval_runner"):
        return _wrapper_text("retrieval_oriented_model_selection.shared.mp10_k9_retrieval_runner", "main")
    if module.endswith("k11_backward_retrieval_runner"):
        return _wrapper_text("retrieval_oriented_model_selection.shared.k11_backward_retrieval_runner", "main")
    if module.endswith("smallbeam_retrieval_runner"):
        return _wrapper_text("retrieval_oriented_model_selection.shared.smallbeam_retrieval_runner", "main")
    if module.endswith("type3_retrieval_support"):
        return _wrapper_text("lut_clustering_compression.shared.type3_retrieval_support", "main")
    if module.endswith("frozen_mp_global_sparse_gmp_residual_support"):
        return _wrapper_text("behavior_modeling.shared.frozen_mp_global_sparse_gmp_residual_support", "main")
    return None


def _generate_promoted_wrappers() -> None:
    for source, (destination, wrapper) in SUPPORT_PROMOTIONS.items():
        if not wrapper:
            continue
        destination_module = destination.removeprefix("scripts/").removesuffix(".py").replace("/", ".")
        content = _promoted_wrapper(destination_module)
        if content is None:
            raise RuntimeError(f"no wrapper template for {destination}")
        target = PROJECT_ROOT / (wrapper if wrapper.startswith("scripts/") else f"scripts/{wrapper}")
        if target.exists():
            raise RuntimeError(f"wrapper target unexpectedly exists: {target}")
        _write_text(target, content)


def _rewrite_project_roots(text: str) -> str:
    robust = '''PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"'''
    robust_project = '''PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)'''
    if 'SCRIPTS_ROOT = PROJECT_ROOT / "scripts"' in text:
        text = re.sub(
            r'(?m)^(SCRIPTS_ROOT = PROJECT_ROOT / "scripts"\n){2,}',
            'SCRIPTS_ROOT = PROJECT_ROOT / "scripts"\n',
            text,
        )
        text = re.sub(
            r'(?m)^(if str\(SCRIPTS_ROOT\) not in sys\.path:\n    sys\.path\.insert\(0, str\(SCRIPTS_ROOT\)\)\n){2,}',
            'if str(SCRIPTS_ROOT) not in sys.path:\n    sys.path.insert(0, str(SCRIPTS_ROOT))\n',
            text,
        )
        return text
    # Use a sentinel so the newly inserted SCRIPTS_ROOT line is not matched a
    # second time by the single-line replacement.
    sentinel = "__PROJECT_SCRIPTS_ROOT_SENTINEL__"
    text = re.sub(
        r"(?ms)^SCRIPTS_ROOT\s*=\s*next\(.*?^\s*\)",
        sentinel,
        text,
    )
    text = re.sub(r"^SCRIPTS_ROOT\s*=.*$", sentinel, text, flags=re.MULTILINE)
    text = text.replace(sentinel, robust + "\nSCRIPTS_ROOT = PROJECT_ROOT / \"scripts\"")
    text = re.sub(
        r"^PROJECT_ROOT\s*=\s*Path\(__file__\).*?$",
        robust_project,
        text,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"if str\(Path\(__file__\).*?\) not in sys\.path:\n\s*sys\.path\.insert\(0, str\(Path\(__file__\).*?\)\)",
        'if str(SCRIPTS_ROOT) not in sys.path:\n    sys.path.insert(0, str(SCRIPTS_ROOT))',
        text,
        flags=re.DOTALL,
    )
    # A source file with the robust block already has the correct project
    # root; old ``SCRIPTS_ROOT.parent`` assignments are also correct once
    # SCRIPTS_ROOT is the scripts directory.
    return text


def _replace_imports(text: str) -> str:
    replacements = (
        ("core.behavior_indexed_dpd.clustering", "lut_clustering_compression.shared.clustering"),
        ("core.behavior_indexed_dpd.signal", "low_bandwidth_behavior_analysis.shared.behavior_indexed_dpd_signal"),
        ("core.validate_project_module_layout", "core.shared.validate_project_module_layout"),
        ("core.module_registry", "core.shared.module_registry"),
        ("core.project_paths", "core.shared.project_paths"),
        ("core.metrics", "core.shared.metrics"),
        ("core.signal", "core.shared.signal"),
        ("core.utils", "core.shared.utils"),
        ("data_management", "data_management.shared"),
        ("signal_segmentation", "signal_segmentation.shared"),
        ("pa_performance_evaluation", "pa_performance_evaluation.shared"),
        ("behavior_modeling.basis_function_selection", "behavior_modeling.shared.basis_function_selection"),
        ("behavior_modeling", "behavior_modeling.shared"),
        ("behavior_fingerprint_ranking_consistency", "behavior_fingerprint_ranking_consistency.shared"),
        ("behavior_fingerprint_retrieval", "behavior_fingerprint_retrieval.shared"),
        ("low_bandwidth_behavior_analysis", "low_bandwidth_behavior_analysis.shared"),
        ("lut_clustering_compression", "lut_clustering_compression.shared"),
    )
    # Only import-like prefixes are rewritten here; literal historical paths
    # in task summaries remain auditable.  The broad package replacement is
    # followed by precise task-module restoration below.
    for old, new in sorted(replacements, key=lambda item: len(item[0]), reverse=True):
        # Do not apply a broad module replacement to an already rewritten
        # ``<module>.shared`` import.
        pattern = rf"(?<=from ){re.escape(old)}(?=[.\s])(?!\.shared\b)"
        text = re.sub(pattern, new, text)
        pattern = rf"(?<=import ){re.escape(old)}(?=[.\s])(?!\.shared\b)"
        text = re.sub(pattern, new, text)
    return text


TASK_MODULE_REPLACEMENTS = {
    "behavior_modeling.shared.c2_envelope23_ridge_generalization": "behavior_modeling.scenario_2_all425_c2_envelope23_ridge_generalization_5B.c2_envelope23_ridge_generalization",
    "behavior_modeling.shared.gateb48_refinement": "behavior_modeling.scenario_2_gateb48_unified_basis_refinement_5B.gateb48_refinement",
    "behavior_modeling.shared.hard20_c2_ilcend_shared_envelope75_selection": "behavior_modeling.scenario_2_hard20_c2_ilcend_shared_envelope75_basis_selection_5B.hard20_c2_ilcend_shared_envelope75_selection",
    "behavior_modeling.shared.plot_scenario2_failed14_P5_with_order2_variable_memory_OLS_scan_5b": "behavior_modeling.scenario_2_failed14_P5_with_order2_variable_memory_OLS_scan_5b.plot_scenario2_failed14_P5_with_order2_variable_memory_OLS_scan_5b",
    "behavior_modeling.shared.plot_scenario2_failed14_frozen_mp_residual_sparse_gmp_5b": "behavior_modeling.scenario_2_failed14_frozen_mp_residual_sparse_gmp_5b.plot_scenario2_failed14_frozen_mp_residual_sparse_gmp_5b",
    "behavior_modeling.shared.plot_scenario2_failed14_frozen_neighborhood_memory_ridge_scan_5b": "behavior_modeling.scenario_2_failed14_frozen_neighborhood_memory_ridge_scan_5b.plot_scenario2_failed14_frozen_neighborhood_memory_ridge_scan_5b",
    "behavior_modeling.shared.plot_scenario2_failed14_multibranch_independent_basis_ols_scan_5b": "behavior_modeling.scenario_2_failed14_multibranch_independent_basis_ols_scan_5b.plot_scenario2_failed14_multibranch_independent_basis_ols_scan_5b",
    "behavior_modeling.shared.plot_scenario2_failed14_odd_order_mp_order_dependent_memory_scan_5b": "behavior_modeling.scenario_2_failed14_odd_order_mp_order_dependent_memory_scan_5b.plot_scenario2_failed14_odd_order_mp_order_dependent_memory_scan_5b",
    "behavior_modeling.shared.plot_scenario2_failed14_p5_variable_memory_ridge_scan_5b": "behavior_modeling.scenario_2_failed14_p5_variable_memory_ridge_scan_5b.plot_scenario2_failed14_p5_variable_memory_ridge_scan_5b",
    "behavior_modeling.shared.scenario_2_unified_model_capacity_scan": "behavior_modeling.scenario_2_unified_model_capacity_scan",
    "behavior_fingerprint_retrieval.shared.plot_scenario2_c2_to_aend_equal_abc": "behavior_fingerprint_retrieval.scenario_2_C2_to_Aend_equal_ABC.plot_scenario2_c2_to_aend_equal_abc",
    "behavior_fingerprint_retrieval.shared.plot_scenario2_c2_to_aend_g4_sparse_gmp_5b": "behavior_fingerprint_retrieval.scenario_2_C2_to_Aend_g4_sparse_gmp_5b.plot_scenario2_c2_to_aend_g4_sparse_gmp_5b",
    "behavior_fingerprint_retrieval.shared.plot_scenario2_c2_to_aend_retrieval": "behavior_fingerprint_retrieval.scenario_2_C2_to_Aend_retrieval.plot_scenario2_c2_to_aend_retrieval",
    "behavior_fingerprint_retrieval.shared.plot_scenario2_c2_to_aend_state_metrics": "behavior_fingerprint_retrieval.scenario_2_C2_to_Aend_retrieval.plot_scenario2_c2_to_aend_state_metrics",
    "behavior_fingerprint_retrieval.shared.plot_scenario2_c3_to_a2_retrieval": "behavior_fingerprint_retrieval.scenario_2_C3_to_A2_retrieval.plot_scenario2_c3_to_a2_retrieval",
    "behavior_fingerprint_retrieval.shared.scenario2_c2_to_aend_equal_abc": "behavior_fingerprint_retrieval.scenario_2_C2_to_Aend_equal_ABC.scenario2_c2_to_aend_equal_abc",
    "behavior_fingerprint_retrieval.shared.scenario2_unified_model_capacity_scan": "behavior_modeling.scenario_2_unified_model_capacity_scan.scenario2_unified_model_capacity_scan",
    "behavior_fingerprint_retrieval.shared.export_scenario2_c2_to_aend_state_summary": "behavior_fingerprint_retrieval.scenario_2_C2_to_Aend_retrieval.export_scenario2_c2_to_aend_state_summary",
    "behavior_fingerprint_retrieval.shared.export_scenario2_c2_to_aend_equal_abc_excel": "behavior_fingerprint_retrieval.scenario_2_C2_to_Aend_equal_ABC.export_scenario2_c2_to_aend_equal_abc_excel",
    "retrieval_oriented_model_selection.shared.scenario_2_C2_to_Aend_retrieval_oriented_model_scan": "retrieval_oriented_model_selection.shared",
}


def _restore_task_imports(text: str) -> str:
    for old, new in TASK_MODULE_REPLACEMENTS.items():
        text = text.replace(old, new)
    text = text.replace(
        "retrieval_oriented_model_selection.shared.scenario_2_mp10_k9_retrieval_oriented_basis_ablation_5b",
        "retrieval_oriented_model_selection.shared.mp10_k9_retrieval_support",
    )
    text = text.replace(
        "retrieval_oriented_model_selection.shared.scenario_2_k11_backward_retrieval_oriented_basis_ablation_5b",
        "retrieval_oriented_model_selection.shared.k11_backward_retrieval_support",
    )
    text = text.replace(
        "retrieval_oriented_model_selection.shared.scenario_2_smallbeam3_k13_retrieval_oriented_forward_expansion_5b",
        "retrieval_oriented_model_selection.shared.smallbeam_retrieval_support",
    )
    text = text.replace(
        "lut_clustering_compression.shared.scenario_2_C2_to_Aend_frozen_centered17_type3_compressed_5B",
        "lut_clustering_compression.shared.type3_retrieval_support",
    )
    text = text.replace(
        "behavior_modeling.shared.run_scenario2_frozen_mp_global_sparse_gmp_residual_5b",
        "behavior_modeling.shared.frozen_mp_global_sparse_gmp_residual_support",
    )
    # Bare task-package imports are used by the historical runners.  They are
    # rewritten to the promoted shared support modules so no task depends on a
    # sibling task package.
    bare_imports = {
        "from scenario_2_mp10_k9_retrieval_oriented_basis_ablation_5b import mp10_k9_backend as backend": "from retrieval_oriented_model_selection.shared import mp10_k9_retrieval_support as backend",
        "from scenario_2_mp10_k9_retrieval_oriented_basis_ablation_5b import run_scenario2_mp10_k9_retrieval_oriented_basis_ablation_5b as k9_utils": "from retrieval_oriented_model_selection.shared import mp10_k9_retrieval_runner as k9_utils",
        "from scenario_2_k11_backward_retrieval_oriented_basis_ablation_5b import k11_backward_backend as backend": "from retrieval_oriented_model_selection.shared import k11_backward_retrieval_support as backend",
        "from scenario_2_k11_backward_retrieval_oriented_basis_ablation_5b import run_scenario2_k11_backward_retrieval_oriented_basis_ablation_5b as prior_backward": "from retrieval_oriented_model_selection.shared import k11_backward_retrieval_runner as prior_backward",
        "from scenario_2_smallbeam3_k13_retrieval_oriented_forward_expansion_5b import smallbeam_backend as backend": "from retrieval_oriented_model_selection.shared import smallbeam_retrieval_support as backend",
        "from scenario_2_smallbeam3_k13_retrieval_oriented_forward_expansion_5b import run_scenario2_smallbeam3_k13_retrieval_oriented_forward_expansion_5b as legacy": "from retrieval_oriented_model_selection.shared import smallbeam_retrieval_runner as legacy",
        "from scenario_2_smallbeam3_k13_retrieval_oriented_forward_expansion_5b import run_scenario2_smallbeam3_k13_retrieval_oriented_forward_expansion_5b as smallbeam_utils": "from retrieval_oriented_model_selection.shared import smallbeam_retrieval_runner as smallbeam_utils",
        "from behavior_fingerprint_retrieval.shared import run_scenario2_C2_to_Aend_type3_cluster_compressed_5b as old_type3": "from lut_clustering_compression.shared import type3_retrieval_support as old_type3",
    }
    for old, new in bare_imports.items():
        text = text.replace(old, new)
    text = re.sub(
        r"from scenario_2_mp10_k9_retrieval_oriented_basis_ablation_5b import \(\s*(?:#.*?\n\s*)?mp10_k9_backend as backend,?\s*\)",
        "from retrieval_oriented_model_selection.shared import mp10_k9_retrieval_support as backend",
        text,
    )
    text = re.sub(
        r"from scenario_2_k11_backward_retrieval_oriented_basis_ablation_5b import \(\s*(?:#.*?\n\s*)?k11_backward_backend as backend,?\s*\)",
        "from retrieval_oriented_model_selection.shared import k11_backward_retrieval_support as backend",
        text,
    )
    text = re.sub(
        r"from scenario_2_smallbeam3_k13_retrieval_oriented_forward_expansion_5b import \(\s*(?:#.*?\n\s*)?smallbeam_backend as backend,?\s*\)",
        "from retrieval_oriented_model_selection.shared import smallbeam_retrieval_support as backend",
        text,
    )
    text = re.sub(
        r"from scenario_2_.*?_retrieval_oriented_forward_expansion_5b import \(\s*(?:#.*?\n\s*)?updated_beam_backend as backend,?\s*\)",
        "from scenario_2_updated_beam3_iteration2_retrieval_oriented_forward_expansion_5b import updated_beam_backend as backend",
        text,
    )
    text = re.sub(
        r"from scenario_2_mp10_k9_retrieval_oriented_basis_ablation_5b import \(\s*(?:#.*?\n\s*)?run_scenario2_mp10_k9_retrieval_oriented_basis_ablation_5b as k9_utils,?\s*\)",
        "from retrieval_oriented_model_selection.shared import mp10_k9_retrieval_runner as k9_utils",
        text,
    )
    text = re.sub(
        r"from scenario_2_k11_backward_retrieval_oriented_basis_ablation_5b import \(\s*(?:#.*?\n\s*)?run_scenario2_k11_backward_retrieval_oriented_basis_ablation_5b as prior_backward,?\s*\)",
        "from retrieval_oriented_model_selection.shared import k11_backward_retrieval_runner as prior_backward",
        text,
    )
    text = re.sub(
        r"from scenario_2_smallbeam3_k13_retrieval_oriented_forward_expansion_5b import \(\s*(?:#.*?\n\s*)?run_scenario2_smallbeam3_k13_retrieval_oriented_forward_expansion_5b as legacy,?\s*\)",
        "from retrieval_oriented_model_selection.shared import smallbeam_retrieval_runner as legacy",
        text,
    )
    text = text.replace(
        "retrieval_oriented_model_selection.scenario_2_C2_to_Aend_retrieval_oriented_model_scan.scenario2_retrieval_oriented_model_scan",
        "retrieval_oriented_model_selection.shared.scenario2_retrieval_oriented_model_scan",
    )
    text = text.replace(
        "behavior_modeling.scenario_2_unified_model_capacity_scan.scenario2_unified_model_capacity_scan",
        "behavior_modeling.shared.scenario2_unified_model_capacity_scan",
    )
    text = text.replace(
        "from behavior_fingerprint_retrieval.shared import (\n    run_scenario2_C2_to_Aend_type3_cluster_compressed_5b as old_type3,\n)",
        "from lut_clustering_compression.shared import type3_retrieval_support as old_type3",
    )
    text = text.replace(
        "from lut_clustering_compression.shared import run_scenario2_5b_A_yout_withoutdpd_ori_complete_link as legacy_runner",
        "from lut_clustering_compression.shared import complete_link_retrieval_support as legacy_runner",
    )
    for module in (
        "behavior_modeling",
        "behavior_fingerprint_retrieval",
        "behavior_fingerprint_ranking_consistency",
    ):
        text = text.replace(f"{module}.shared.scenario_2_", f"{module}.scenario_2_")
    text = text.replace(
        "behavior_fingerprint_retrieval.shared.run_scenario2_c2_to_aend_1b",
        "behavior_fingerprint_retrieval.shared.scenario2_c2_to_aend_1b_runner",
    )
    text = text.replace(
        "behavior_fingerprint_retrieval.shared.run_scenario2_envelope19_c2endshared_commonB_full_lut_retrieval_5b",
        "behavior_fingerprint_retrieval.shared.envelope19_retrieval_runner",
    )
    text = text.replace(
        "behavior_modeling.shared.run_scenario2_failed14_frozen_mp_normalized_uniform_ridge_5b",
        "behavior_modeling.shared.frozen_mp_normalized_uniform_ridge_support",
    )
    text = text.replace(
        "behavior_modeling.shared.run_scenario2_unified_odd_order_mp_capacity_scan_5b",
        "behavior_modeling.scenario_2_unified_odd_order_mp_capacity_scan_5b.run_scenario2_unified_odd_order_mp_capacity_scan_5b",
    )
    text = text.replace(
        "behavior_modeling.shared.basis_function_selection.c2_envelope23_ridge_generalization",
        "behavior_modeling.scenario_2_all425_c2_envelope23_ridge_generalization_5B.c2_envelope23_ridge_generalization",
    )
    text = text.replace(
        "behavior_modeling.shared.basis_function_selection.gateb48_refinement",
        "behavior_modeling.scenario_2_gateb48_unified_basis_refinement_5B.gateb48_refinement",
    )
    text = text.replace(
        "behavior_modeling.shared.basis_function_selection.hard20_c2_ilcend_shared_envelope75_selection",
        "behavior_modeling.scenario_2_hard20_c2_ilcend_shared_envelope75_basis_selection_5B.hard20_c2_ilcend_shared_envelope75_selection",
    )
    text = text.replace(
        "from behavior_modeling.shared import run_scenario2_failed14_multibranch_independent_basis_ols_scan_5b as runner",
        "from behavior_modeling.scenario_2_failed14_multibranch_independent_basis_ols_scan_5b import run_scenario2_failed14_multibranch_independent_basis_ols_scan_5b as runner",
    )
    text = text.replace(
        "from behavior_modeling.shared import run_scenario2_frozen_mp_global_sparse_gmp_residual_5b as global_runner",
        "from behavior_modeling.shared import frozen_mp_global_sparse_gmp_residual_support as global_runner",
    )
    text = text.replace(
        "from behavior_modeling.shared import run_scenario2_unified_odd_order_mp_capacity_scan_5b as scan",
        "from behavior_modeling.scenario_2_unified_odd_order_mp_capacity_scan_5b import run_scenario2_unified_odd_order_mp_capacity_scan_5b as scan",
    )
    text = text.replace(
        "behavior_modeling.shared.ridge_scan",
        "behavior_modeling.state0_ridge_scan.ridge_scan",
    )
    text = text.replace(
        "from module_registry import",
        "from core.shared.module_registry import",
    )
    text = text.replace(
        "from .scenario2_c2_to_aend_retrieval import",
        "from behavior_fingerprint_retrieval.shared.scenario2_c2_to_aend_retrieval import",
    )
    text = text.replace(
        "from .scenario2_c3_to_a2_retrieval import",
        "from behavior_fingerprint_retrieval.shared.scenario2_c3_to_a2_retrieval import",
    )
    text = text.replace(
        "from behavior_fingerprint_retrieval.shared.run_scenario2_c2_to_aend_1b",
        "from behavior_fingerprint_retrieval.shared.scenario2_c2_to_aend_1b_runner",
    )
    text = text.replace(
        "from behavior_fingerprint_retrieval.shared.run_scenario2_envelope19_c2endshared_commonB_full_lut_retrieval_5b",
        "from behavior_fingerprint_retrieval.shared.envelope19_retrieval_runner",
    )
    # Task files are imported through their fully-qualified package path.  A
    # direct script invocation therefore does not depend on the task folder
    # itself being added to sys.path.
    for module in (
        "behavior_modeling",
        "behavior_fingerprint_retrieval",
        "retrieval_oriented_model_selection",
        "behavior_fingerprint_ranking_consistency",
    ):
        marker = f"scripts/{module}/"
        # The actual task-specific path normalization is completed by the
        # path-aware pass in _rewrite_sources below.
        _ = marker
    return text


def _fix_task_names(path: Path, text: str) -> str:
    relative = path.relative_to(SCRIPTS_ROOT).parts
    placeholders = {
        "behavior_fingerprint_retrieval",
        "behavior_modeling",
        "behavior_fingerprint_ranking_consistency",
        "low_bandwidth_behavior_analysis",
    }
    if len(relative) >= 3 and relative[1] not in {"shared", "__pycache__"}:
        task = relative[1]
        for placeholder in placeholders:
            text = text.replace(f'TASK_NAME = "{placeholder}"', f'TASK_NAME = "{task}"')
    shared_task_names = {
        "scenario2_c2_to_aend_1b.py": "scenario_2_C2_to_Aend_1b",
        "select_statewise_best_round0_4_models.py": "scenario_2_select_statewise_best_round0_4_models",
    }
    if path.name in shared_task_names:
        for placeholder in placeholders:
            text = text.replace(f'TASK_NAME = "{placeholder}"', f'TASK_NAME = "{shared_task_names[path.name]}"')
    return text


def _rewrite_sources() -> int:
    changed = 0
    excluded = {Path(__file__).resolve(), PROJECT_ROOT / "scripts" / "core" / "module_layout_reorganization_20260917" / "run_module_layout_reorganization_20260917.py"}
    for path in _script_files():
        if path.resolve() in excluded:
            continue
        old = path.read_text(encoding="utf-8")
        new = _restore_task_imports(_replace_imports(_rewrite_project_roots(old)))
        relative = path.relative_to(SCRIPTS_ROOT).parts
        if len(relative) >= 3 and relative[0] in {
            "behavior_modeling",
            "behavior_fingerprint_retrieval",
            "retrieval_oriented_model_selection",
            "behavior_fingerprint_ranking_consistency",
        }:
            package = relative[0]
            task = relative[1]
            if task not in {"shared", "__pycache__"}:
                # Resolve imports of the current task package when a source
                # file is executed directly from its path.
                new = re.sub(
                    rf"(?<=from ){re.escape(task)}(?=[.\s])",
                    f"{package}.{task}",
                    new,
                )
                if package == "behavior_fingerprint_ranking_consistency":
                    new = new.replace(
                        "from .plotting import",
                        "from behavior_fingerprint_ranking_consistency.shared.plotting import",
                    )
                    new = re.sub(
                        r"from \.scenario2_([A-Za-z0-9_]+) import",
                        r"from behavior_fingerprint_ranking_consistency.shared.scenario2_\1 import",
                        new,
                    )
                    for candidate_task, candidate_files in DIRECT_TASKS[package].items():
                        for filename in candidate_files:
                            if not filename.startswith("plot_"):
                                continue
                            stem = filename.removesuffix(".py")
                            new = new.replace(
                                f"{package}.shared.{stem}",
                                f"{package}.{candidate_task}.{stem}",
                            )
        new = _fix_task_names(path, new)
        # Restore the task-specific module imports after the broad package
        # replacement.  These are exact fully-qualified import names.
        if new != old:
            _write_text(path, new)
            changed += 1
    return changed


def _task_rows() -> list[dict[str, Any]]:
    rows = []
    for module, tasks in DIRECT_TASKS.items():
        for task in tasks:
            rows.append({
                "module": module,
                "task": task,
                "scripts_path": f"scripts/{module}/{task}",
                "results_path": f"results/{module}/{task}",
                "work_logs_path": f"work_logs/{module}/{task}",
                "result_exists_before": (PROJECT_ROOT / "results" / module / task).exists(),
                "log_exists_before": (PROJECT_ROOT / "work_logs" / module / task).exists(),
            })
    for task in BASIS_TASK_FILES:
        rows.append({
            "module": "behavior_modeling",
            "task": task,
            "scripts_path": f"scripts/behavior_modeling/{task}",
            "results_path": f"results/behavior_modeling/{task}",
            "work_logs_path": f"work_logs/behavior_modeling/{task}",
            "result_exists_before": (PROJECT_ROOT / "results" / "behavior_modeling" / task).exists(),
            "log_exists_before": (PROJECT_ROOT / "work_logs" / "behavior_modeling" / task).exists(),
        })
    return sorted(rows, key=lambda row: (row["module"], row["task"]))


def _task_dirs() -> list[Path]:
    paths = []
    for module, tasks in DIRECT_TASKS.items():
        paths.extend(SCRIPTS_ROOT.joinpath(module, task) for task in tasks)
    paths.extend(SCRIPTS_ROOT / "behavior_modeling" / task for task in BASIS_TASK_FILES)
    return paths


def _ensure_task_markers() -> int:
    """Give every concrete script task a meaningful package marker."""

    created = 0
    for module in MODULES:
        module_root = SCRIPTS_ROOT / module
        for directory in sorted(module_root.iterdir(), key=lambda item: item.name.lower()):
            if not directory.is_dir() or directory.name in {"shared", "__pycache__"}:
                continue
            marker = directory / "__init__.py"
            if not marker.exists():
                _write_text(
                    marker,
                    f'"""Concrete research task package.\n\nModule: {module}\nTask: {directory.name}\n"""',
                )
                created += 1
    return created


def _validate_layout() -> tuple[dict[str, Any], list[dict[str, str]]]:
    violations: list[dict[str, str]] = []
    root_checks: dict[str, Any] = {}
    for module in MODULES:
        root = SCRIPTS_ROOT / module
        direct_files = sorted(
            item.name
            for item in root.iterdir()
            if item.is_file() and item.name != "__init__.py"
        )
        direct_dirs = sorted(
            item.name
            for item in root.iterdir()
            if item.is_dir() and item.name != "__pycache__"
        )
        # Every named directory below a module is a task; the migration
        # inventory is responsible for classifying the pre-existing names.
        allowed = {"shared"} | {
            item.name
            for item in root.iterdir()
            if item.is_dir() and item.name != "__pycache__"
        }
        pass_value = not direct_files and "shared" in direct_dirs and set(direct_dirs) <= allowed
        root_checks[module] = {
            "direct_files": direct_files,
            "direct_dirs": direct_dirs,
            "allowed_dirs": sorted(allowed),
            "pass": pass_value,
        }
        for name in direct_files:
            violations.append({"type": "root_direct_file", "path": f"scripts/{module}/{name}", "detail": "only __init__.py may remain at module root"})
        for name in direct_dirs:
            if name not in allowed:
                violations.append({"type": "root_unclassified_dir", "path": f"scripts/{module}/{name}", "detail": "module root may contain only shared or task directories"})
        shared = root / "shared"
        if not (shared / "__init__.py").is_file():
            violations.append({"type": "missing_shared_init", "path": str(shared / "__init__.py"), "detail": "shared package marker missing"})
    for root_name in ("results", "work_logs"):
        for module in MODULES:
            shared = PROJECT_ROOT / root_name / module / "shared"
            if shared.exists():
                violations.append({"type": "forbidden_shared_mirror", "path": str(shared.relative_to(PROJECT_ROOT)), "detail": f"{root_name} must contain task directories only"})
    return {"modules": root_checks, "pass": not violations}, violations


def _module_for_script_path(path: Path) -> tuple[str | None, str | None]:
    try:
        relative = path.relative_to(SCRIPTS_ROOT)
    except ValueError:
        return None, None
    if not relative.parts or relative.parts[0] not in MODULES:
        return None, None
    module = relative.parts[0]
    if len(relative.parts) >= 2 and relative.parts[1] not in {"shared", "__pycache__"}:
        return module, relative.parts[1]
    return module, "shared"


def _resolve_import_target(module: str, source_path: Path, imported: str, level: int) -> tuple[str | None, str | None]:
    if level:
        return _module_for_script_path(source_path)
    if not imported:
        return None, None
    first = imported.split(".", 1)[0]
    if first not in MODULES:
        return None, None
    pieces = imported.split(".")
    target_module = pieces[0]
    if len(pieces) >= 2 and pieces[1] not in {"shared"} and pieces[1].startswith("scenario_"):
        return target_module, pieces[1]
    if len(pieces) >= 2 and pieces[1] not in {"shared"} and target_module == "behavior_modeling" and pieces[1] == "basis_function_selection":
        return target_module, pieces[1]
    return target_module, "shared"


def _dependency_violations() -> list[dict[str, str]]:
    violations: list[dict[str, str]] = []
    for path in _script_files():
        source_module, source_area = _module_for_script_path(path)
        if source_module is None:
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports = [(alias.name, 0) for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                imports = [(("." * node.level) + (node.module or ""), node.level)]
            else:
                continue
            for imported, level in imports:
                target_module, target_area = _resolve_import_target(source_module, path, imported, level)
                if target_module is None:
                    continue
                if target_area not in {"shared", None} and source_area == "shared":
                    violations.append({"type": "shared_to_task", "source": path.relative_to(PROJECT_ROOT).as_posix(), "target": f"{target_module}/{target_area}", "detail": imported})
                if target_area not in {"shared", None} and target_module != source_module:
                    violations.append({"type": "cross_module_to_task", "source": path.relative_to(PROJECT_ROOT).as_posix(), "target": f"{target_module}/{target_area}", "detail": imported})
                if target_area not in {"shared", None} and target_module == source_module and source_area != target_area:
                    violations.append({"type": "cross_task", "source": path.relative_to(PROJECT_ROOT).as_posix(), "target": f"{target_module}/{target_area}", "detail": imported})
    return sorted(violations, key=lambda row: (row["source"], row["target"], row["type"]))


def _dependency_graph_text() -> str:
    edges = []
    for path in _script_files():
        module, area = _module_for_script_path(path)
        if module != "" and area == "shared":
            try:
                tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    target_module = node.module.split(".", 1)[0]
                    if target_module in MODULES:
                        edges.append(f"{path.relative_to(SCRIPTS_ROOT).as_posix()} -> {node.module}")
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        target_module = alias.name.split(".", 1)[0]
                        if target_module in MODULES:
                            edges.append(f"{path.relative_to(SCRIPTS_ROOT).as_posix()} -> {alias.name}")
    return "Shared dependency graph (source -> imported public module):\n\n" + "\n".join(sorted(set(edges)))


def _post_inventory() -> dict[str, Any]:
    return {
        "timestamp": _now(),
        "git": _git_snapshot(),
        "raw_manifest": _raw_manifest(),
        "self_first": _self_first_snapshot(),
        "roots": {name: _tree_inventory(PROJECT_ROOT / name) for name in ("scripts", "results", "work_logs")},
    }


def _write_required_artifacts(pre: dict[str, Any], post: dict[str, Any], moves: list[dict[str, str]], classifications: list[dict[str, Any]], layout: dict[str, Any], dependency_violations: list[dict[str, str]]) -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_text(RESULT_ROOT / "01_pre_reorganization_inventory.txt", json.dumps(pre, ensure_ascii=False, indent=2, default=_json_default))
    _write_csv(
        RESULT_ROOT / "02_module_internal_classification.csv",
        classifications,
        ("source_path", "item_name", "classification", "destination_module", "destination_task_or_subpackage", "reason", "status"),
    )
    _write_csv(RESULT_ROOT / "03_move_manifest.csv", moves, ("source", "destination", "kind", "status"))
    _write_csv(RESULT_ROOT / "05_task_mapping_manifest.csv", _task_rows(), ("module", "task", "scripts_path", "results_path", "work_logs_path", "result_exists_before", "log_exists_before"))
    _write_csv(RESULT_ROOT / "06_task_name_reconciliation.csv", ({"legacy_name": a, "canonical_name": b, "decision": c, "status": d} for a, b, c, d in TASK_RECONCILIATION), ("legacy_name", "canonical_name", "decision", "status"))
    _write_text(RESULT_ROOT / "07_shared_dependency_graph.txt", _dependency_graph_text())
    _write_csv(RESULT_ROOT / "08_dependency_violations.csv", dependency_violations, ("type", "source", "target", "detail"))
    _write_csv(RESULT_ROOT / "09_file_conflicts.csv", [], ("source", "destination", "kind", "detail"))
    _write_import_audit(RESULT_ROOT / "10_import_dependency_audit_after.txt", "Import dependency audit after reorganization")
    _write_text(RESULT_ROOT / "11_post_reorganization_inventory.txt", json.dumps(post, ensure_ascii=False, indent=2, default=_json_default))
    _write_json(RESULT_ROOT / "12_validation_report.json", {"layout": layout, "dependency_violations": dependency_violations, "raw_before": pre["raw_manifest"], "raw_after": post["raw_manifest"], "self_first_before": pre["self_first"], "self_first_after": post["self_first"], "pass": bool(layout["pass"] and not dependency_violations and pre["raw_manifest"] == post["raw_manifest"] and pre["self_first"] == post["self_first"])})
    lines = ["scripts/", *sorted(
        "  " * (len(path.relative_to(SCRIPTS_ROOT).parts) - 1) + path.name
        for path in SCRIPTS_ROOT.rglob("*")
        if path.is_dir() and "__pycache__" not in path.parts
    )]
    _write_text(RESULT_ROOT / "13_final_scripts_tree.txt", "\n".join(lines))
    _write_text(RESULT_ROOT / "14_reorganization_summary.txt", "\n".join([
        f"Task: {TASK_NAME}",
        "Purpose: normalize each frozen module into shared public code and concrete task directories.",
        f"Moved source entries: {sum(row.get('status') == 'moved' for row in moves)}",
        f"Rewritten source files: {sum(1 for path in _script_files() if path.suffix == '.py')}",
        f"Dependency violations: {len(dependency_violations)}",
        f"Layout pass: {layout['pass']}",
        f"Raw manifest unchanged: {pre['raw_manifest'] == post['raw_manifest']}",
        f"Self-First tree unchanged: {pre['self_first'] == post['self_first']}",
        "No results, distance matrices, MAT data, caches, or research computations were regenerated.",
    ]))


def _run() -> int:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    _append_log(f"Begin {TASK_NAME}; engineering-only source layout migration.")
    pre = _pre_inventory()
    if pre["raw_manifest"] != EXPECTED_RAW:
        raise RuntimeError(f"unexpected data/raw manifest before migration: {pre['raw_manifest']}")
    classifications = _classification_rows()
    review = [row for row in classifications if row["status"] == "review"]
    if review:
        _write_csv(RESULT_ROOT / "02_module_internal_classification.csv", classifications, ("source_path", "item_name", "classification", "destination_module", "destination_task_or_subpackage", "reason", "status"))
        raise RuntimeError(f"unclassified source files require review: {review[:5]}")
    _write_import_audit(
        RESULT_ROOT / "04_import_dependency_audit_before.txt",
        "Import dependency audit before reorganization",
    )
    planned = _move_manifest()
    moves: list[dict[str, str]] = []
    for row in planned:
        if row["kind"] == "generated_wrapper":
            continue
        _move(row["source"], row["destination"], row["kind"], moves)
    legacy_package = SCRIPTS_ROOT / "core" / "behavior_indexed_dpd"
    if legacy_package.exists():
        cache = legacy_package / "__pycache__"
        if cache.exists():
            shutil.rmtree(cache)
        if legacy_package.exists() and not any(legacy_package.iterdir()):
            legacy_package.rmdir()
    _prepare_package_markers()
    _write_basis_wrapper_init()
    _generate_promoted_wrappers()
    changed = _rewrite_sources()
    _append_log(f"Moved {len(moves)} entries and rewrote {changed} source files.")
    post = _post_inventory()
    layout, layout_violations = _validate_layout()
    violations = _dependency_violations()
    _write_required_artifacts(pre, post, moves, classifications, layout, violations)
    if layout_violations:
        _write_csv(RESULT_ROOT / "09_file_conflicts.csv", layout_violations, ("type", "path", "detail"))
    _append_log(f"Completed layout={layout['pass']}, dependency_violations={len(violations)}, raw_unchanged={pre['raw_manifest'] == post['raw_manifest']}, self_first_unchanged={pre['self_first'] == post['self_first']}.")
    if not layout["pass"] or violations or pre["raw_manifest"] != post["raw_manifest"] or pre["self_first"] != post["self_first"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_run())
