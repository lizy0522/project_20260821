"""Summarise the completed retrieval-oriented candidate scan."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_lut_retrieval"
    / "scenario_2_C2_to_Aend_retrieval_oriented_model_scan"
)


def build_analysis_summary() -> dict[str, object]:
    ranking = pd.read_csv(RESULT_ROOT / "retrieval_oriented_candidate_ranking.csv")
    best = json.loads((RESULT_ROOT / "best_retrieval_model.json").read_text(encoding="utf-8"))
    transitions = pd.read_csv(RESULT_ROOT / "baseline_vs_best_statewise.csv")
    summary: dict[str, object] = {
        "candidate_count": int(ranking.shape[0]),
        "valid_candidate_count": int(ranking["candidate_valid"].astype(bool).sum()),
        "best": best,
        "zero_failure_candidate_count": int(
            (ranking["candidate_valid"].astype(bool) & ranking["failure_count"].eq(0)).sum()
        ),
        "transition_counts": {
            str(key): int(value)
            for key, value in transitions["retrieval_transition"].value_counts().items()
        },
        "state_Q_changed_count": int(
            (transitions["baseline_state_Q"] != transitions["best_state_Q"]).sum()
        ),
    }
    (RESULT_ROOT / "analysis_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    validation_path = RESULT_ROOT / "validation.json"
    workbook_path = RESULT_ROOT / "best_candidate_state_summary.xlsx"
    metadata_path = RESULT_ROOT / "best_candidate_excel_metadata.json"
    progress_path = RESULT_ROOT / "candidate_progress.json"
    if progress_path.is_file():
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        progress["fit_cache_ready"] = (RESULT_ROOT / "candidate_fit_cache.npz").is_file()
        progress_path.write_text(
            json.dumps(progress, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    if validation_path.is_file() and workbook_path.is_file() and metadata_path.is_file():
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        validation["excel_verified"] = True
        validation["excel_artifact"] = json.loads(metadata_path.read_text(encoding="utf-8"))
        validation.setdefault("output_files", {})[workbook_path.name] = str(workbook_path)
        validation.setdefault("candidate_progress", {})["fit_cache_ready"] = True
        validation_path.write_text(
            json.dumps(validation, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    return summary


def main() -> None:
    print(json.dumps(build_analysis_summary(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
