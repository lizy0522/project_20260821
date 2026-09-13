"""Tests for the fixed 1B Scenario 2 retrieval experiment."""

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

from _core.behavior_indexed_dpd.signal import LowBandwidthObservationBank  # noqa: E402
from _core.metrics import cnmse  # noqa: E402

from behavior_fingerprint_lut_retrieval.scenario2_c2_to_aend_1b import (  # noqa: E402
    EXPECTED_MAIN_COLUMNS,
    EXPECTED_SEGMENT_LENGTHS_1B,
    EXPECTED_SEGMENT_LENGTHS_5B,
    RESULT_ROOT,
    STATE_COUNT,
    _compute_cnmse_distance_matrix,
    load_baseline_artifacts,
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_operator_contract() -> None:
    bank = LowBandwidthObservationBank()
    one_b = bank.get_by_index(10)
    five_b = bank.get_by_index(50)
    _require(one_b.spec.normalized_bandwidth_n == 1.0, "n=1.0 contract failed")
    _require(one_b.spec.fs_out_hz == 20_000_000, "1B Fs contract failed")
    _require(one_b.spec.up == 1 and one_b.spec.down == 5, "1B rational ratio failed")
    _require(five_b.spec.is_identity, "5B operator must be identity")
    for name, length in EXPECTED_SEGMENT_LENGTHS_5B.items():
        _require(
            one_b.expected_output_length(length) == EXPECTED_SEGMENT_LENGTHS_1B[name],
            f"{name} 1B length failed",
        )


def test_local_distance_matches_scalar_metric() -> None:
    rng = np.random.default_rng(20260907)
    query = rng.normal(size=(STATE_COUNT, 7)) + 1j * rng.normal(size=(STATE_COUNT, 7))
    candidate = rng.normal(size=(STATE_COUNT, 7)) + 1j * rng.normal(size=(STATE_COUNT, 7))
    query = query.astype(np.complex128)
    candidate = candidate.astype(np.complex128)
    matrix = _compute_cnmse_distance_matrix(query, candidate)
    for row, column in ((0, 0), (17, 3), (340, 217), (424, 424)):
        expected = cnmse(query[row], candidate[column])
        _require(
            np.isclose(matrix[row, column], expected, rtol=0, atol=1e-12),
            f"distance mismatch at {row},{column}",
        )


def test_baseline_distance_reference_is_self_consistent() -> None:
    baseline = load_baseline_artifacts()
    rebuilt = _compute_cnmse_distance_matrix(baseline.c2_fingerprints, baseline.aend_fingerprints)
    _require(
        np.array_equal(rebuilt, baseline.retrieval_distance),
        "frozen 5B distance reference mismatch",
    )


def test_saved_1b_outputs_contract() -> None:
    _require(RESULT_ROOT.is_dir(), f"missing 1B result root: {RESULT_ROOT}")
    main = pd.read_csv(RESULT_ROOT / "retrieval_results_1B.csv")
    _require(main.shape == (STATE_COUNT, 10), "main result shape mismatch")
    _require(main.columns.tolist() == EXPECTED_MAIN_COLUMNS, "main result columns mismatch")
    _require(
        np.array_equal(main["state_id_R"].to_numpy(dtype=int), np.arange(STATE_COUNT)),
        "state order mismatch",
    )
    _require(main["state_id_Q"].between(0, STATE_COUNT - 1).all(), "state_id_Q out of range")
    for column in EXPECTED_MAIN_COLUMNS[2:8]:
        _require(
            np.isfinite(main[column].to_numpy(dtype=float)).all(), f"{column} has nonfinite values"
        )
    retrieved = main["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float)
    exact = main["state_id_Q"].to_numpy(dtype=int) == np.arange(STATE_COUNT)
    _require(np.array_equal(exact, np.isneginf(retrieved)), "exact/self Real-B pattern mismatch")
    _require(
        np.all((retrieved < -40.0) | np.isneginf(retrieved) | (retrieved >= -40.0)),
        "invalid Real-B values",
    )

    lut = np.load(RESULT_ROOT / "lut_fingerprints_1B.npz")
    query = np.load(RESULT_ROOT / "query_fingerprints_1B.npz")
    _require(lut["fingerprints"].shape == (STATE_COUNT, 983), "LUT fingerprint shape mismatch")
    _require(query["fingerprints"].shape == (STATE_COUNT, 983), "Query fingerprint shape mismatch")
    _require(
        np.all(np.isfinite(lut["fingerprints"])) and np.all(np.isfinite(query["fingerprints"])),
        "fingerprint nonfinite",
    )
    matrix = np.load(RESULT_ROOT / "fingerprint_cnmse_matrix_1B.npy")
    _require(matrix.shape == (STATE_COUNT, STATE_COUNT), "1B distance shape mismatch")
    _require(
        not np.isnan(matrix).any() and not np.isposinf(matrix).any(),
        "1B distance has invalid values",
    )

    coefficients_a = np.load(RESULT_ROOT / "Aend_model_coefficients_1B.npy")
    coefficients_c = np.load(RESULT_ROOT / "C2_model_coefficients_1B.npy")
    _require(coefficients_a.shape == (STATE_COUNT, 10), "Aend coefficients shape mismatch")
    _require(coefficients_c.shape == (STATE_COUNT, 10), "C2 coefficients shape mismatch")
    _require(
        np.all(np.isfinite(coefficients_a)) and np.all(np.isfinite(coefficients_c)),
        "coefficients nonfinite",
    )

    validation = json.loads((RESULT_ROOT / "validation.json").read_text(encoding="utf-8"))
    _require(
        validation["five_B_regression_guard"]["pass"] is True, "5B regression guard did not pass"
    )
    _require(validation["one_B_used_for_retrieval"] is True, "1B retrieval flag missing")
    _require(
        validation["one_B_used_for_final_DPD_shareability_validation"] is False,
        "1B incorrectly used for Real-B",
    )
    _require(
        validation["retrieval_selection_uses_real_B"] is False,
        "Real-B incorrectly used for selection",
    )
    _require(validation["real_B_is_evaluation_only"] is True, "Real-B evaluation-only flag missing")
    _require(
        validation["Aend_fingerprint_shape"] == [STATE_COUNT, 983], "validation Aend shape mismatch"
    )
    _require(
        validation["C2_query_fingerprint_shape"] == [STATE_COUNT, 983],
        "validation C2 shape mismatch",
    )
    for filename in (
        "scenario_2_C2_to_Aend_1B_retrieval.xlsx",
        "figure_1B_statewise_model_and_retrieval_metrics.png",
        "figure_1B_retrieved_real_B_CNMSE.png",
    ):
        _require((RESULT_ROOT / filename).is_file(), f"missing output: {filename}")


def main() -> None:
    tests = [
        test_operator_contract,
        test_local_distance_matches_scalar_metric,
        test_baseline_distance_reference_is_self_consistent,
        test_saved_1b_outputs_contract,
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: PASS")


if __name__ == "__main__":
    main()
