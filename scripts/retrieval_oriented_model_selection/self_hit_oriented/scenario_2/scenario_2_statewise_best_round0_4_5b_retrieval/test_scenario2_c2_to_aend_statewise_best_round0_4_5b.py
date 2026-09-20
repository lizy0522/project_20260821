"""Direct-run validation for frozen Round 0--4 statewise-best retrieval."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = next(
    parent for parent in Path(__file__).resolve().parents if (parent / "environment.yml").is_file()
)
RESULT_ROOT = (
    ROOT
    / "results"
    / "retrieval_oriented_model_selection"
    / "self_hit_oriented"
    / "scenario_2"
    / "scenario_2_C2_to_Aend_statewise_best_round0_4_5B"
)


def test_retrieval_contract() -> None:
    summary = json.loads((RESULT_ROOT / "retrieval_summary.json").read_text(encoding="utf-8"))
    validation = json.loads((RESULT_ROOT / "validation.json").read_text(encoding="utf-8"))
    assert summary["candidate_count"] == 1064
    assert summary["selected_Aend_count"] == 425
    assert summary["selected_C2_count"] == 425
    assert summary["M_common"] == 9
    assert summary["fingerprint_length"] == 4906
    assert summary["Exact"] == 161
    assert summary["Shareable"] == 389
    assert summary["Failure"] == 36
    assert validation["round_5_used"] is False
    assert validation["additional_model_search_performed"] is False
    assert validation["OFF_RealB_used_for_model_selection"] is False
    assert validation["OFF_RealB_used_for_retrieval_selection"] is False
    assert validation["spot_checks"]["top1_recomputed_equal"] is True
    assert validation["spot_checks"]["real_B_lookup_equal"] is True
    assert validation["excel_verified"] is True


def test_matrix_and_table_shapes() -> None:
    result = pd.read_csv(RESULT_ROOT / "retrieval_results_statewise_best_round0_4_5B.csv")
    diagnostics = pd.read_csv(RESULT_ROOT / "top1_fingerprint_retrieval.csv")
    assert result.shape == (425, 10)
    assert diagnostics.shape == (425, 8)
    assert np.array_equal(result["state_id_R"].to_numpy(dtype=int), np.arange(425))
    assert np.array_equal(diagnostics["state_id_R"].to_numpy(dtype=int), np.arange(425))
    with np.load(RESULT_ROOT / "fingerprint_cnmse_matrix.npz", allow_pickle=False) as data:
        assert data["D_C2_Aend"].shape == (425, 425)
        assert data["R_C2_Aend"].shape == (425, 425)
    workbook = RESULT_ROOT / "scenario_2_C2_to_Aend_statewise_best_round0_4_5B_retrieval.xlsx"
    assert workbook.is_file() and workbook.stat().st_size > 0


def main() -> None:
    tests = [
        ("retrieval contract", test_retrieval_contract),
        ("matrix/table shapes", test_matrix_and_table_shapes),
    ]
    for label, test in tests:
        test()
        print(f"{label}: PASS")


if __name__ == "__main__":
    main()
