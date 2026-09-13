"""Contract tests for the retrieval-oriented unified MP scan."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from behavior_fingerprint_lut_retrieval.scenario2_retrieval_oriented_model_scan import (  # noqa: E402
    BASELINE_CANDIDATE,
    BASELINE_FAILURE_IDS,
    CANDIDATES,
    DPD_SHAREABLE_THRESHOLD_DB,
    LAMBDA_GRID,
    MEMORY_PROFILES,
    ORDER_PROFILES,
    build_candidate_retrieval,
    build_model_candidate_grid,
    compute_generic_cnmse_distance_matrix,
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_lut_retrieval"
    / "scenario_2_C2_to_Aend_retrieval_oriented_model_scan"
)


def test_candidate_grid_is_200_and_unique() -> None:
    frame = build_model_candidate_grid()
    assert frame.shape[0] == 200
    assert frame["candidate_id"].tolist() == list(range(1, 201))
    assert frame["is_baseline"].sum() == 1
    assert set(frame["order_profile"]) == set(ORDER_PROFILES)
    assert set(frame["memory_profile"]) == set(MEMORY_PROFILES)
    assert set(frame["lambda"].astype(float)) == set(LAMBDA_GRID)


def test_memory_horizons_are_2_to_5() -> None:
    assert sorted({candidate.max_delay for candidate in CANDIDATES}) == [2, 3, 4, 5]
    assert BASELINE_CANDIDATE.max_delay == 2


def test_baseline_anchor_and_failure_ids_are_frozen() -> None:
    assert BASELINE_CANDIDATE.order_profile == "P9"
    assert BASELINE_CANDIDATE.memory_profile == "M0"
    assert BASELINE_CANDIDATE.ridge_lambda == 1e-8
    assert BASELINE_FAILURE_IDS == (
        187,
        189,
        195,
        196,
        199,
        206,
        323,
        327,
        330,
        335,
        340,
        344,
        346,
        354,
    )


def test_generic_cnmse_exact_and_shape() -> None:
    values = np.arange(425 * 19, dtype=float).reshape(425, 19).astype(np.complex128)
    distance = compute_generic_cnmse_distance_matrix(values, values)
    assert distance.shape == (425, 425)
    assert distance.dtype == np.float64
    assert np.all(np.isneginf(np.diag(distance)))


def test_strict_shareable_threshold_contract() -> None:
    values = np.asarray([-40.000001, -40.0, -39.999999, -np.inf])
    assert (values < DPD_SHAREABLE_THRESHOLD_DB).tolist() == [True, False, False, True]


def test_full_candidate_set_and_self_match() -> None:
    rng = np.random.default_rng(20260906)
    query = rng.normal(size=(425, 7)) + 1j * rng.normal(size=(425, 7))
    candidate = query.copy()
    retrieval = build_candidate_retrieval(
        query.astype(np.complex128),
        candidate.astype(np.complex128),
        np.full((425, 425), -41.0, dtype=np.float64),
        candidate=BASELINE_CANDIDATE,
    )
    assert retrieval["distance"].shape == (425, 425)
    assert retrieval["distance"].size == 180625
    assert retrieval["results"].shape[0] == 425
    assert retrieval["results"]["state_id_Q"].between(0, 424).all()
    assert np.array_equal(retrieval["selected_q"], np.arange(425))


def test_result_contract_when_available() -> None:
    path = RESULT_ROOT / "validation.json"
    if not path.is_file():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["candidate_count"] == 200
    assert payload["query_count_per_candidate"] == 425
    assert payload["lut_entries_per_query"] == 425
    assert payload["distance_matrix_shape"] == [425, 425]
    assert payload["full_candidate_search"] is True
    assert payload["self_match"] is True
    assert payload["real_B_used_for_ranking"] is True
    assert payload["model_metrics_used_for_ranking"] is False


def test_summary_has_all_candidates_when_available() -> None:
    path = RESULT_ROOT / "retrieval_oriented_candidate_summary.csv"
    if not path.is_file():
        return
    frame = pd.read_csv(path)
    assert frame.shape[0] == 200
    assert frame["candidate_id"].nunique() == 200


def main() -> None:
    tests = [
        test_candidate_grid_is_200_and_unique,
        test_memory_horizons_are_2_to_5,
        test_baseline_anchor_and_failure_ids_are_frozen,
        test_generic_cnmse_exact_and_shape,
        test_strict_shareable_threshold_contract,
        test_full_candidate_set_and_self_match,
        test_result_contract_when_available,
        test_summary_has_all_candidates_when_available,
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All retrieval-oriented scan tests passed ({len(tests)} tests).")


if __name__ == "__main__":
    main()
