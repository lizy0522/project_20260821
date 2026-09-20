"""Finalize explicit partial diagnostics after a user-requested search stop."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
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

from behavior_modeling.shared.statewise_adaptive_model_search import (  # noqa: E402
    DPD_SHAREABLE_THRESHOLD_DB,
    STATE_COUNT,
    candidate_grid,
    expand_candidate_shell,
    initial_candidate_bank,
)

MODEL_ROOT = (
    PROJECT_ROOT / "results" / "behavior_modeling" / "scenario_2" /"scenario_2_statewise_adaptive_Aend_C2_5B"  # noqa: E501
)
PROGRESS_PATH = MODEL_ROOT / "search_progress.json"
PARTIAL_VALIDATION_PATH = MODEL_ROOT / "partial_validation.json"


def _load_log(name: str) -> pd.DataFrame:
    frame = pd.read_csv(MODEL_ROOT / name)
    frame = frame.loc[frame["valid"].astype(bool)].copy()
    return frame


def _state_summary(frame: pd.DataFrame) -> dict[str, Any]:
    feasible = frame[
        (frame["native_train_NMSE_dB"].astype(float) < DPD_SHAREABLE_THRESHOLD_DB)
        & (frame["native_B_NMSE_dB"].astype(float) < DPD_SHAREABLE_THRESHOLD_DB)
    ]
    unresolved = sorted(set(range(STATE_COUNT)) - set(feasible["state_id"].astype(int)))
    best_train = frame.groupby("state_id")["native_train_NMSE_dB"].min()
    best_b = frame.groupby("state_id")["native_B_NMSE_dB"].min()
    return {
        "candidate_row_count": int(frame.shape[0]),
        "evaluated_state_count": int(frame["state_id"].nunique()),
        "joint_native_feasible_state_count": int(feasible["state_id"].nunique()),
        "unresolved_state_ids": unresolved,
        "best_train_min_dB": float(best_train.min()),
        "best_train_median_dB": float(best_train.median()),
        "best_train_max_dB": float(best_train.max()),
        "best_B_min_dB": float(best_b.min()),
        "best_B_median_dB": float(best_b.median()),
        "best_B_max_dB": float(best_b.max()),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    return value


def main() -> None:
    candidates = list(initial_candidate_bank())
    for search_round in range(1, 5):
        candidates.extend(expand_candidate_shell(candidates, search_round))
    if len(candidates) != 1064:
        raise RuntimeError(
            f"persisted candidate bank must contain 1064 candidates, got {len(candidates)}"
        )
    candidate_grid(candidates).to_csv(
        MODEL_ROOT / "candidate_model_grid.csv.gz",
        index=False,
        compression="gzip",
        encoding="utf-8-sig",
    )
    aend = _load_log("candidate_search_Aend.csv.gz")
    c2 = _load_log("candidate_search_C2.csv.gz")
    progress = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
    payload = {
        "experiment": "scenario_2_C2_to_Aend_statewise_adaptive_model_5B",
        "observation_bandwidth": "5B",
        "low_bandwidth_operator_used": False,
        "state_count": STATE_COUNT,
        "ABC_changed": False,
        "persisted_search_round_max": 4,
        "candidate_count": len(candidates),
        "candidate_row_count_total": int(aend.shape[0] + c2.shape[0]),
        "candidate_row_count_by_side": {"Aend": int(aend.shape[0]), "C2": int(c2.shape[0])},
        "Aend": _state_summary(aend),
        "C2": _state_summary(c2),
        "formal_model_bank_complete": False,
        "common_support_validation_completed": False,
        "retrieval_started": False,
        "unresolved_Aend_state_ids": _state_summary(aend)["unresolved_state_ids"],
        "unresolved_C2_state_ids": _state_summary(c2)["unresolved_state_ids"],
        "search_stop_reason": "user_requested_stop_before_round_5_completion",
        "round_5_partial_attempt": {
            "generated_candidate_count": 276,
            "completed_target_states": 5,
            "target_states": 229,
            "persisted": False,
        },
        "checkpoint": progress,
        "model_selection_uses_Y_Aend_B_generalization": True,
        "model_selection_uses_Y_C2_B_generalization": True,
        "OFF_RealB_used_for_model_selection": False,
        "raw_data_modified": False,
        "next_step": (
            "Do not enter retrieval unless a separately authorized search resumes "
            "and common-support validation reaches 850/850."
        ),
    }
    PARTIAL_VALIDATION_PATH.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "candidate_count": len(candidates),
                "partial_validation": str(PARTIAL_VALIDATION_PATH),
                "formal_model_bank_complete": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
