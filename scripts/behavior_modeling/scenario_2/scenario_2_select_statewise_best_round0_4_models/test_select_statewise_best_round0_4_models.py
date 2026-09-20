"""Direct-run validation for frozen Round 0--4 model selection."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
RESULT_ROOT = ROOT / "results" / "behavior_modeling" / "scenario_2" /"scenario_2_statewise_best_round0_4_5B"  # noqa: E501


def test_selected_model_contract() -> None:
    validation = json.loads((RESULT_ROOT / "selection_validation.json").read_text(encoding="utf-8"))
    assert validation["persisted_search_round_max"] == 4
    assert validation["persisted_candidate_count"] == 1064
    assert validation["round_5_used"] is False
    assert validation["additional_model_search_performed"] is False
    assert validation["selected_Aend_count"] == 425
    assert validation["selected_C2_count"] == 425
    assert validation["all_850_refit"] is True
    for side in ("Aend", "C2"):
        frame = pd.read_csv(RESULT_ROOT / f"selected_{side}_models.csv")
        assert frame.shape[0] == 425
        assert np.array_equal(frame["state_id"].to_numpy(dtype=int), np.arange(425))
        assert np.all(frame["candidate_id"].to_numpy(dtype=int) >= 1)
        assert np.all(frame["candidate_id"].to_numpy(dtype=int) <= 1064)
        assert np.all(frame["search_round"].to_numpy(dtype=int) <= 4)
        assert np.all(
            np.isfinite(
                frame[["native_train_NMSE_dB", "native_B_NMSE_dB", "joint_score_dB"]].to_numpy(
                    dtype=float
                )
            )
        )


def test_feasible_and_fallback_counts() -> None:
    aend = pd.read_csv(RESULT_ROOT / "selected_Aend_models.csv")
    c2 = pd.read_csv(RESULT_ROOT / "selected_C2_models.csv")
    assert int(aend["joint_feasible_native"].sum()) == 344
    assert int(c2["joint_feasible_native"].sum()) == 196
    assert set(aend["selection_mode"]) == {"feasible_min_complexity", "best_available_minimax"}
    assert set(c2["selection_mode"]) == {"feasible_min_complexity", "best_available_minimax"}


def main() -> None:
    tests = [
        ("selected model contract", test_selected_model_contract),
        ("feasible/fallback counts", test_feasible_and_fallback_counts),
    ]
    for label, test in tests:
        test()
        print(f"{label}: PASS")


if __name__ == "__main__":
    main()
