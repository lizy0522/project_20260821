"""Frozen ten-module registry and root-alignment checks for project_20260821."""

from __future__ import annotations

from pathlib import Path
from typing import Any

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

ROOT_NAMES = ("scripts", "results", "work_logs")


def module_roots(project_root: Path | str) -> dict[str, set[str]]:
    """Return the directory names directly below each mirrored project root."""

    root = Path(project_root).resolve()
    return {
        root_name: {
            item.name
            for item in (root / root_name).iterdir()
            if item.is_dir()
        }
        for root_name in ROOT_NAMES
    }


def validate_module_root_alignment(project_root: Path | str) -> dict[str, Any]:
    """Compare the three first-level module sets against the frozen registry."""

    expected = set(MODULES)
    actual = module_roots(project_root)
    details = {
        root_name: {
            "actual": sorted(actual[root_name]),
            "missing": sorted(expected - actual[root_name]),
            "extra": sorted(actual[root_name] - expected),
            "pass": actual[root_name] == expected,
        }
        for root_name in ROOT_NAMES
    }
    aligned = all(item["pass"] for item in details.values())
    return {
        "modules": list(MODULES),
        "root_names": list(ROOT_NAMES),
        "roots": details,
        "aligned": aligned,
        "pass": aligned,
    }


__all__ = ["MODULES", "ROOT_NAMES", "module_roots", "validate_module_root_alignment"]
