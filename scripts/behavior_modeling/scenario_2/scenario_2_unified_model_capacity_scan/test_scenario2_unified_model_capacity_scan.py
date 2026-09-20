"""Focused tests for the unified Scenario 2 model-capacity workflow."""

from __future__ import annotations

import json
import sys
from pathlib import Path

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

from behavior_modeling.shared.scenario2_unified_model_capacity_scan import (  # noqa: E402
    BASELINE_LAMBDA,
    BASELINE_MEMORY,
    BASELINE_ORDERS,
    CANDIDATES,
    DEVELOPMENT_COUNT,
    DPD_SHAREABLE_THRESHOLD_DB,
    LAMBDA_GRID,
    MAX_DELAY,
    MEMORY_PROFILES,
    ORDER_PROFILES,
    STATE_COUNT,
    VALID_LENGTHS,
    assign_pa_quartiles,
    build_model_candidate_grid,
    load_fixed_split,
    select_best_candidate,
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_retrieval"
    / "scenario_2" /"scenario_2_C2_to_Aend_unified_model_capacity"
)
SPLIT_SOURCE = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2" /"scenario_2_y_lut_weighted_fusion"
    / "split_definition.csv"
)


def _assert_raises(exception_type: type[BaseException], function) -> None:
    try:
        function()
    except exception_type:
        return
    raise AssertionError(f"expected {exception_type.__name__}")


def test_candidate_grid_contract() -> None:
    frame = build_model_candidate_grid()
    assert frame.shape[0] == 120
    assert frame["candidate_id"].is_unique
    assert set(frame["order_profile"]) == set(ORDER_PROFILES)
    assert set(frame["memory_profile"]) == set(MEMORY_PROFILES)
    assert set(frame["lambda"].astype(float)) == set(LAMBDA_GRID)
    assert frame["max_delay"].eq(MAX_DELAY).all()
    assert int(frame["is_baseline"].sum()) == 1
    baseline = frame.loc[frame["is_baseline"]].iloc[0]
    assert baseline["order_profile"] == "P9"
    assert baseline["memory_profile"] == "M0"
    assert float(baseline["lambda"]) == BASELINE_LAMBDA
    assert baseline["orders"] == "[1,2,3,5,7,9]"


def test_candidate_objects_use_uniform_architecture() -> None:
    assert len(CANDIDATES) == 120
    assert all(candidate.max_delay == 2 for candidate in CANDIDATES)
    grid = build_model_candidate_grid().set_index("candidate_id")
    for candidate in CANDIDATES:
        assert tuple(candidate.orders) == tuple(
            json.loads(grid.loc[candidate.candidate_id, "orders"])
        )
        assert candidate.n_complex_coefficients == len(candidate.basis_columns)
        assert max(candidate.memory_definition.values()) == 3
    baseline = next(candidate for candidate in CANDIDATES if candidate.is_baseline)
    assert baseline.orders == BASELINE_ORDERS
    assert baseline.memory_definition == BASELINE_MEMORY


def test_fixed_split_contract() -> None:
    split = load_fixed_split(SPLIT_SOURCE)
    assert split.shape[0] == STATE_COUNT
    assert split["state_id"].to_numpy().tolist() == list(range(STATE_COUNT))
    assert split["split"].value_counts().to_dict() == {
        "Development": DEVELOPMENT_COUNT,
        "Validation": 85,
    }


def test_valid_lengths_and_threshold() -> None:
    assert VALID_LENGTHS == {"A": 12286, "B": 4913, "C": 7371}
    values = np.asarray([-40.000001, -40.0, -39.999999, -np.inf])
    mask = values < DPD_SHAREABLE_THRESHOLD_DB
    assert mask.tolist() == [True, False, False, True]


def test_pa_quartile_direction() -> None:
    values = np.linspace(-50.0, -10.0, STATE_COUNT)
    quartiles = assign_pa_quartiles(values)
    assert quartiles[0] == "Q1"
    assert quartiles[-1] == "Q4"
    assert set(quartiles) == {"Q1", "Q2", "Q3", "Q4"}


def test_selector_rejects_validation_and_retrieval_fields() -> None:
    base = pd.DataFrame(
        {
            "candidate_id": [1],
            "rank": [1],
            "split": ["Validation"],
        }
    )
    _assert_raises(ValueError, lambda: select_best_candidate(base))
    retrieval_frame = base.assign(split="Development", retrieval_gain_dB=[0.0])
    _assert_raises(ValueError, lambda: select_best_candidate(retrieval_frame))


def test_completed_artifacts_contract() -> None:
    validation_path = RESULT_ROOT / "validation.json"
    if not validation_path.is_file():
        return
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    assert payload["candidate_count"] == 120
    assert payload["baseline_candidate_count"] == 1
    assert payload["Aend_and_C2_share_same_model"] is True
    assert payload["model_selection_uses_retrieval"] is False
    assert payload["model_selection_uses_real_B"] is False
    assert payload["model_selection_uses_failure_labels"] is False
    assert payload["model_frozen_before_retrieval"] is True
    assert payload["protection_verification"]["all_protected_unchanged"] is True
    assert payload["raw_data_modified"] is False


def test_completed_fingerprint_and_retrieval_shapes() -> None:
    coefficients_path = RESULT_ROOT / "selected_model_coefficients.npz"
    lut_path = RESULT_ROOT / "selected_lut_fingerprints_Aend.npz"
    query_path = RESULT_ROOT / "selected_query_fingerprints_C2.npz"
    distance_path = RESULT_ROOT / "selected_retrieval_distance_matrix.npz"
    if not all(path.is_file() for path in (coefficients_path, lut_path, query_path, distance_path)):
        return
    with np.load(coefficients_path, allow_pickle=False) as data:
        theta_a = np.asarray(data["theta_Aend"])
        theta_c = np.asarray(data["theta_C2"])
    with np.load(lut_path, allow_pickle=False) as data:
        lut = np.asarray(data["Y_Aend_fingerprints"])
    with np.load(query_path, allow_pickle=False) as data:
        query = np.asarray(data["Q_C2"])
    with np.load(distance_path, allow_pickle=False) as data:
        distance = np.asarray(data["D_C2_Aend"])
    assert theta_a.shape[0] == STATE_COUNT
    assert theta_c.shape == theta_a.shape
    assert theta_a.dtype == np.complex128
    assert theta_c.dtype == np.complex128
    assert lut.shape == (STATE_COUNT, 4913)
    assert query.shape == lut.shape
    assert lut.dtype == np.complex128
    assert query.dtype == np.complex128
    assert np.all(np.isfinite(lut)) and np.all(np.isfinite(query))
    assert distance.shape == (STATE_COUNT, STATE_COUNT)
    assert distance.dtype == np.float64


def test_completed_no_infinite_numeric_gain() -> None:
    path = RESULT_ROOT / "statewise_baseline_vs_selected.csv"
    if not path.is_file():
        return
    frame = pd.read_csv(path)
    finite_gain = frame["retrieval_gain_dB"].dropna().to_numpy(dtype=float)
    assert np.all(np.isfinite(finite_gain))
    assert frame["retrieval_gain_finite_finite"].dtype == bool


def main() -> None:
    tests = [
        test_candidate_grid_contract,
        test_candidate_objects_use_uniform_architecture,
        test_fixed_split_contract,
        test_valid_lengths_and_threshold,
        test_pa_quartile_direction,
        test_selector_rejects_validation_and_retrieval_fields,
        test_completed_artifacts_contract,
        test_completed_fingerprint_and_retrieval_shapes,
        test_completed_no_infinite_numeric_gain,
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All unified model-capacity tests passed ({len(tests)} tests).")


if __name__ == "__main__":
    main()
