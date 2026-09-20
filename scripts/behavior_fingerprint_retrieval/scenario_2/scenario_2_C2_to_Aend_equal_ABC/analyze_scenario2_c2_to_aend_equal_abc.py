"""Summarise already-generated equal-ABC results without touching raw data."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_retrieval"
    / "scenario_2" /"scenario_2_C2_to_Aend_equal_ABC"
)


def load_equal_abc_summary() -> dict[str, object]:
    """Read the persisted summaries and return a compact machine-readable report."""

    validation = json.loads((RESULT_ROOT / "validation.json").read_text(encoding="utf-8"))
    retrieval = pd.read_csv(RESULT_ROOT / "retrieval_summary.csv").iloc[0]
    gaps = pd.read_csv(RESULT_ROOT / "generalization_gap_summary.csv")
    comparison = pd.read_csv(RESULT_ROOT / "comparison_vs_original_ABC.csv")
    return {
        "segment_definition": validation["segment_definition"],
        "model": {
            "orders": validation["model_orders"],
            "memory": validation["memory"],
            "lambda": validation["lambda"],
        },
        "retrieval": retrieval.to_dict(),
        "generalization_gaps": gaps.to_dict("records"),
        "old_vs_equal_comparison": comparison.to_dict("records"),
        "protection_pass": validation["protection_verification"]["all_protected_unchanged"],
    }


def main() -> None:
    summary = load_equal_abc_summary()
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
