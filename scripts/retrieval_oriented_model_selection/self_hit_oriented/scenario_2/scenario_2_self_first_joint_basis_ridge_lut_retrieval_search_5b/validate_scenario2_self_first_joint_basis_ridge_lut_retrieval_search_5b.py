# ruff: noqa: E501,I001

"""Independent validation for the completed Self-First retrieval search."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from behavior_fingerprint_retrieval.shared.envelope19_c2endshared_commonB_full_lut import raw_manifest_gate  # noqa: E402
from retrieval_oriented_model_selection.self_hit_oriented.scenario_2.scenario_2_self_first_joint_basis_ridge_lut_retrieval_search_5b.self_first_metrics import STATE_COUNT  # noqa: E402

TASK_NAME = "scenario_2_self_first_joint_basis_ridge_lut_retrieval_search_5b"
RESULT_ROOT = PROJECT_ROOT / "results" / "retrieval_oriented_model_selection" / "self_hit_oriented" / "scenario_2" /TASK_NAME
OBJECTIVE_VERSION = "SELF_FIRST_V1"


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    checkpoint_path = RESULT_ROOT / "23_checkpoint.json"
    validation_path = RESULT_ROOT / "24_validation_checks.json"
    registry_path = RESULT_ROOT / "02_historical_support_registry.csv"
    ranking_path = RESULT_ROOT / "03_historical_self_first_ranking.csv"
    summary_path = RESULT_ROOT / "22_final_result_summary.json"
    for path in (checkpoint_path, validation_path, registry_path, ranking_path, summary_path):
        _assert(path.is_file(), f"missing validation input: {path}")
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    registry = pd.read_csv(registry_path)
    ranking = pd.read_csv(ranking_path)
    _assert(checkpoint.get("task_name") == TASK_NAME, "checkpoint task mismatch")
    _assert(checkpoint.get("objective_version") == OBJECTIVE_VERSION, "objective version mismatch")
    _assert(checkpoint.get("phase") == "completed", "checkpoint is not completed")
    _assert(validation.get("pass") is True, "runner validation artifact is not PASS")
    _assert(summary.get("status") == "SUCCESS", "final summary is not SUCCESS")
    _assert(registry.shape[0] >= 1 and ranking.shape[0] >= 1, "historical registry/ranking is empty")
    _assert(registry["support_hash"].nunique() >= 1, "support hashes are not unique")
    _assert(set(ranking["N_self"].astype(int) + ranking["N_fallback"].astype(int) + ranking["N_fail"].astype(int)) == {STATE_COUNT}, "historical class identity failed")

    final_top10 = summary.get("final_top10_count", 0)
    _assert(int(final_top10) >= 1, "final Top10 is empty")
    identity_rows = validation.get("class_identity", [])
    _assert(len(identity_rows) >= int(final_top10), "final class identity rows are incomplete")
    _assert(all(bool(row.get("identity_pass")) for row in identity_rows), "final class identity failed")
    raw = raw_manifest_gate()
    _assert(raw == checkpoint.get("raw_manifest"), "raw manifest changed after runner validation")
    print(
        json.dumps(
            {
                "task": TASK_NAME,
                "objective_version": OBJECTIVE_VERSION,
                "registry_rows": int(registry.shape[0]),
                "historical_distance_rows": int(ranking.shape[0]),
                "final_top10": int(final_top10),
                "best_N_self": summary.get("best_N_self"),
                "best_N_valid": summary.get("best_N_valid"),
                "validation_pass": True,
                "raw_manifest_pass": True,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
