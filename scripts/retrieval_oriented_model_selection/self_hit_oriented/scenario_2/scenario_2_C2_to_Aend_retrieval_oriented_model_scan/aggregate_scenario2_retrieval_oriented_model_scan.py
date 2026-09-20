"""Re-aggregate per-candidate cache files for the retrieval-oriented scan."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
PROJECT_ROOT = SCRIPTS_ROOT.parent
RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "retrieval_oriented_model_selection"
    / "self_hit_oriented" / "scenario_2" /"scenario_2_C2_to_Aend_retrieval_oriented_model_scan"
)
STATEWISE_CACHE_ROOT = RESULT_ROOT / "_candidate_cache" / "statewise"
SUMMARY_CACHE_ROOT = RESULT_ROOT / "_candidate_cache" / "summaries"
STATE_COUNT = 425


def aggregate() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read cache files and rewrite the two aggregate CSV outputs."""

    summary_paths = sorted(SUMMARY_CACHE_ROOT.glob("candidate_*.json"))
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in summary_paths]
    summary = pd.DataFrame(summaries).sort_values("candidate_id").reset_index(drop=True)
    state_paths = sorted(STATEWISE_CACHE_ROOT.glob("candidate_*.csv"))
    statewise = (
        pd.concat([pd.read_csv(path) for path in state_paths], ignore_index=True)
        if state_paths
        else pd.DataFrame()
    )
    valid_count = int(summary.get("candidate_valid", pd.Series(dtype=bool)).fillna(False).sum())
    if statewise.shape[0] != valid_count * STATE_COUNT:
        raise RuntimeError(
            f"statewise行数错误：{statewise.shape[0]} != {valid_count * STATE_COUNT}"
        )
    summary.to_csv(
        RESULT_ROOT / "retrieval_oriented_candidate_summary.csv", index=False, encoding="utf-8-sig"
    )
    statewise.to_csv(
        RESULT_ROOT / "retrieval_oriented_candidate_statewise.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return summary, statewise


def main() -> None:
    summary, statewise = aggregate()
    print(
        f"aggregated candidates={summary.shape[0]}, "
        f"valid={int(summary['candidate_valid'].sum())}, "
        f"statewise_rows={statewise.shape[0]}"
    )


if __name__ == "__main__":
    main()
