# ruff: noqa: E402

"""Tests for the fixed 0.5B Scenario 2 retrieval experiment."""

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

from low_bandwidth_behavior_analysis.shared.behavior_indexed_dpd_signal import (
    LowBandwidthObservationBank,  # noqa: E402
)

from behavior_fingerprint_retrieval.shared.scenario2_c2_to_aend_1b import (  # noqa: E402
    BASELINE_ROOT,
    SPEC_0P5B,
    SPEC_1B,
)

RESULT_ROOT = SPEC_0P5B.result_root
ONE_B_ROOT = SPEC_1B.result_root
STATE_COUNT = 425


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _tree_digest(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    files = sorted(
        (
            item
            for item in path.rglob("*")
            if item.is_file()
            and "__pycache__" not in item.parts
            and item.suffix.lower() != ".pyc"
            and not item.name.startswith("~$")
        ),
        key=lambda item: str(item).lower(),
    )
    for file in files:
        content = file.read_bytes()
        digest.update(file.relative_to(path).as_posix().encode() + b"\0")
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def test_0p5b_operator_contract() -> None:
    bank = LowBandwidthObservationBank()
    operator = bank.get_by_index(SPEC_0P5B.observation_index_k)
    _require(operator.spec.normalized_bandwidth_n == 0.5, "n=0.5 contract failed")
    _require(operator.spec.fs_out_hz == 10_000_000, "0.5B Fs contract failed")
    _require(operator.spec.observation_bandwidth_hz == 10_000_000, "B_obs contract failed")
    _require(operator.spec.up == 1 and operator.spec.down == 10, "0.5B ratio failed")
    _require(operator.spec.tag == "nB_0p5", "0.5B tag failed")
    _require(
        operator.describe()["manual_group_delay_compensation_applied"] is False,
        "timing metadata changed",
    )


def test_0p5b_expected_lengths() -> None:
    operator = LowBandwidthObservationBank().get_by_index(5)
    for name, length in {"A": 12286, "B": 4913, "C": 7371}.items():
        _require(
            operator.expected_output_length(length) == SPEC_0P5B.expected_segment_lengths[name],
            f"{name} expected length failed",
        )


def test_fingerprints_coefficients_and_matrix_are_valid() -> None:
    _require(RESULT_ROOT.is_dir(), "missing 0.5B result root")
    lut = np.load(RESULT_ROOT / "lut_fingerprints_0p5B.npz")
    query = np.load(RESULT_ROOT / "query_fingerprints_0p5B.npz")
    for name, data in (("LUT", lut), ("Query", query)):
        values = np.asarray(data["fingerprints"])
        _require(values.shape == (STATE_COUNT, 492), f"{name} fingerprint shape mismatch")
        _require(
            values.dtype == np.complex128 and np.all(np.isfinite(values)),
            f"{name} fingerprint finite/dtype failure",
        )
        _require(
            np.all(np.linalg.norm(values, axis=1) > 0), f"{name} contains zero-norm fingerprint"
        )
    for filename in ("Aend_model_coefficients_0p5B.npy", "C2_model_coefficients_0p5B.npy"):
        values = np.load(RESULT_ROOT / filename)
        _require(values.shape == (STATE_COUNT, 10), f"{filename} shape mismatch")
        _require(
            values.dtype == np.complex128 and np.all(np.isfinite(values)),
            f"{filename} finite/dtype failure",
        )
    matrix = np.load(RESULT_ROOT / "fingerprint_cnmse_matrix_0p5B.npy")
    _require(matrix.shape == (STATE_COUNT, STATE_COUNT), "0.5B distance matrix shape mismatch")
    _require(
        not np.isnan(matrix).any() and not np.isposinf(matrix).any(), "0.5B distance matrix invalid"
    )
    _require(np.all(np.linalg.norm(lut["fingerprints"], axis=1) > 0), "LUT zero norm")
    _require(np.all(np.linalg.norm(query["fingerprints"], axis=1) > 0), "Query zero norm")


def test_top1_and_real_b_lookup_contract() -> None:
    matrix = np.load(RESULT_ROOT / "fingerprint_cnmse_matrix_0p5B.npy")
    main = pd.read_csv(RESULT_ROOT / "retrieval_results_0p5B.csv")
    _require(
        np.array_equal(main["state_id_R"].to_numpy(dtype=int), np.arange(STATE_COUNT)),
        "state order mismatch",
    )
    expected_q = np.argmin(matrix, axis=1)
    _require(
        np.array_equal(main["state_id_Q"].to_numpy(dtype=int), expected_q),
        "State_Q is not row argmin",
    )
    canonical = np.load(BASELINE_ROOT / "real_B_distance_matrix.npz")["D_B"]
    retrieved = main["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float)
    expected = canonical[np.arange(STATE_COUNT), expected_q]
    _require(
        np.array_equal(np.isneginf(retrieved), np.isneginf(expected)),
        "Real-B nonfinite pattern mismatch",
    )
    finite = np.isfinite(retrieved) & np.isfinite(expected)
    _require(
        np.allclose(retrieved[finite], expected[finite], rtol=0, atol=1e-12),
        "Real-B lookup values mismatch",
    )
    rng = np.random.default_rng(20260907)
    sample = np.sort(rng.choice(STATE_COUNT, size=20, replace=False))
    _require(
        all(int(main.iloc[i].state_id_Q) == int(np.argmin(matrix[i])) for i in sample),
        "sample Top1 argmin failed",
    )
    _require(
        all(
            np.isclose(retrieved[i], expected[i], rtol=0, atol=1e-12)
            or (np.isneginf(retrieved[i]) and np.isneginf(expected[i]))
            for i in sample
        ),
        "sample Real-B lookup failed",
    )


def test_summary_and_validation_scope() -> None:
    summary = json.loads((RESULT_ROOT / "retrieval_summary_0p5B.json").read_text(encoding="utf-8"))
    _require(
        summary["observation_n"] == 0.5 and summary["Fs_obs_MHz"] == 10.0,
        "summary observation mismatch",
    )
    _require(summary["exact_hit_count"] == 34, "exact count mismatch")
    _require(summary["shareable_count"] == 346, "shareable count mismatch")
    _require(summary["failure_count"] == 79, "failure count mismatch")
    _require(summary["nonexact_count"] == 391, "nonexact count mismatch")
    _require(summary["nonexact_shareable_count"] == 312, "nonexact shareable count mismatch")
    validation = json.loads((RESULT_ROOT / "validation.json").read_text(encoding="utf-8"))
    _require(validation["final_status"] == "completed", "final status missing")
    _require(validation["half_B_used_for_retrieval"] is True, "0.5B retrieval flag missing")
    _require(
        validation["half_B_used_for_final_DPD_shareability_validation"] is False,
        "0.5B Real-B flag incorrect",
    )
    _require(
        validation["five_B_real_B_used_as_ground_truth"] is True, "5B ground-truth flag missing"
    )
    _require(
        validation["retrieval_selection_uses_real_B"] is False, "Real-B selection flag incorrect"
    )
    _require(validation["real_B_is_evaluation_only"] is True, "Real-B evaluation flag missing")
    _require(
        validation["real_B_observation_operator_applied"] is False, "Real-B operator flag incorrect"
    )
    _require(validation["one_B_regression_guard"]["pass"] is True, "1B regression failed")
    _require(validation["five_B_regression_guard"]["pass"] is True, "5B regression failed")
    _require(
        validation["observation_length_checks"]["A"]["observed"] == [1229],
        "A length validation mismatch",
    )
    _require(
        validation["observation_length_checks"]["B"]["observed"] == [492],
        "B length validation mismatch",
    )
    _require(
        validation["observation_length_checks"]["C"]["observed"] == [738],
        "C length validation mismatch",
    )
    _require(
        validation["Aend_model_count"] == STATE_COUNT
        and validation["C2_model_count"] == STATE_COUNT,
        "model count mismatch",
    )
    _require(validation["c2_available_count"] == STATE_COUNT, "C2 availability mismatch")
    stability = validation["numerical_stability"]
    _require(
        stability["all_fingerprint_finite"] is True, "fingerprint finite stability flag missing"
    )
    _require(
        stability["all_fingerprint_norms_positive"] is True, "zero-norm stability flag missing"
    )
    _require(
        stability["all_coefficients_finite"] is True, "coefficient finite stability flag missing"
    )
    _require(
        stability["distance_nan_count"] == 0 and stability["distance_posinf_count"] == 0,
        "distance stability flag failed",
    )
    _require(stability["rank_collapse"] is False, "rank-collapse stability flag failed")


def test_protected_results_and_outputs_exist() -> None:
    validation = json.loads((RESULT_ROOT / "validation.json").read_text(encoding="utf-8"))
    _require(
        validation["protection_verification"]["all_protected_unchanged"] is True,
        "protected result verification failed",
    )
    _require(
        validation["protection_verification"]["result_dirs"]["scenario_2_C2_to_Aend_1B"][
            "sha256_unchanged"
        ]
        is True,
        "1B result changed",
    )
    _require(
        _tree_digest(ONE_B_ROOT)
        == validation["protection_before"]["result_dirs"]["scenario_2_C2_to_Aend_1B"]["sha256"],
        "1B result current SHA differs",
    )
    for filename in (
        "scenario_2_C2_to_Aend_0p5B_retrieval.xlsx",
        "figure_0p5B_statewise_model_and_retrieval_metrics.png",
        "figure_0p5B_statewise_model_and_retrieval_metrics.svg",
        "figure_0p5B_statewise_model_and_retrieval_metrics.pdf",
        "figure_0p5B_retrieved_real_B_CNMSE.png",
        "retrieval_failures_0p5B.csv",
        "validation.json",
    ):
        _require((RESULT_ROOT / filename).is_file(), f"missing output: {filename}")


def main() -> None:
    tests = [
        test_0p5b_operator_contract,
        test_0p5b_expected_lengths,
        test_fingerprints_coefficients_and_matrix_are_valid,
        test_top1_and_real_b_lookup_contract,
        test_summary_and_validation_scope,
        test_protected_results_and_outputs_exist,
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: PASS")


if __name__ == "__main__":
    main()
