"""Contract tests for the fixed target-sample-rate observation operator."""

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

from low_bandwidth_behavior_analysis.shared.behavior_indexed_dpd_signal import (  # noqa: E402
    LowBandwidthObservationBank,
    SampleRateSpec,
)
from low_bandwidth_behavior_analysis.shared.behavior_indexed_dpd_signal.config import (  # noqa: E402
    BASE_SIGNAL_BANDWIDTH_HZ,
    FS_INPUT_HZ,
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "low_bandwidth_behavior_analysis"
    / "scenario_2"
    / "low_bandwidth_observation"
    / "sample_rate_operator_bank"
)


def _complex_signal(length: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.normal(size=length) + 1j * rng.normal(size=length)).astype(np.complex128)


def _assert_raises(exception_type: type[BaseException], function) -> None:
    try:
        function()
    except exception_type:
        return
    raise AssertionError(f"expected {exception_type.__name__}")


def test_bank_grid_and_derived_bandwidth() -> None:
    bank = LowBandwidthObservationBank()
    assert len(bank.specs) == 50
    assert [spec.sample_rate_index_k for spec in bank.specs] == list(range(1, 51))
    assert bank.specs[0].tag == "nB_0p1"
    assert bank.specs[-1].tag == "nB_5p0"
    assert all(spec.observation_bandwidth_hz == spec.fs_out_hz for spec in bank.specs)
    assert bank.specs[0].fs_out_hz == 2_000_000
    assert bank.specs[-1].fs_out_hz == 100_000_000
    assert BASE_SIGNAL_BANDWIDTH_HZ == 20_000_000
    assert FS_INPUT_HZ == 100_000_000


def test_exact_rational_ratios() -> None:
    bank = LowBandwidthObservationBank()
    for spec in bank.specs:
        assert spec.up * 50 == spec.down * spec.sample_rate_index_k
        assert np.gcd(spec.up, spec.down) == 1
        assert spec.ratio_fraction.numerator == spec.up
        assert spec.ratio_fraction.denominator == spec.down


def test_sample_rate_and_n_input_validation() -> None:
    assert SampleRateSpec.from_n(0.1).sample_rate_index_k == 1
    assert SampleRateSpec.from_n(3.7).sample_rate_index_k == 37
    assert SampleRateSpec.from_sample_rate(34_000_000).sample_rate_index_k == 17
    for value in (0.0, 0.15, -0.1, 5.1, 1.234):
        _assert_raises(ValueError, lambda value=value: SampleRateSpec.from_n(value))
    for value in (1_000_000, 3_000_000, 101_000_000):
        _assert_raises(ValueError, lambda value=value: SampleRateSpec.from_sample_rate(value))
    _assert_raises(TypeError, lambda: SampleRateSpec(True))
    _assert_raises(ValueError, lambda: SampleRateSpec(0))
    _assert_raises(ValueError, lambda: SampleRateSpec(51))


def test_five_b_identity_vector_matrix_and_real() -> None:
    bank = LowBandwidthObservationBank()
    vector = _complex_signal(4913, 1)
    matrix = np.column_stack((vector, 2 * vector, -0.2j * vector))
    operator = bank.get_by_index(50)
    vector_output = operator.apply_vector(vector)
    matrix_output = operator.apply_matrix(matrix)
    assert vector_output.dtype == vector.dtype
    assert matrix_output.dtype == matrix.dtype
    assert np.array_equal(vector_output, vector)
    assert np.array_equal(matrix_output, matrix)
    assert np.max(np.abs(vector_output - vector)) == 0.0
    assert np.max(np.abs(matrix_output - matrix)) == 0.0


def test_timing_metadata_semantics_and_old_field_absence() -> None:
    bank = LowBandwidthObservationBank()
    description = bank.get_by_index(10).describe()
    assert "group_delay_compensated" not in description
    assert description["manual_group_delay_compensation_applied"] is False
    assert description["explicit_sample_shift_applied"] == 0
    assert description["input_time_origin_s"] == 0.0
    assert description["output_time_origin_s"] == 0.0
    assert description["output_time_origin_aligned"] is True
    assert description["additional_delay_compensation_required"] is False


def test_interior_impulse_matrix_and_pair_timing_all_fifty() -> None:
    bank = LowBandwidthObservationBank()
    input_length = 10_000
    impulse_index = 5_000
    vector = np.zeros(input_length, dtype=np.complex128)
    vector[impulse_index] = 1.0 + 0.0j
    matrix = np.zeros((input_length, 3), dtype=np.complex128)
    matrix[:, 1] = vector
    for k in range(1, 51):
        operator = bank.get_by_index(k)
        expected_index = impulse_index * k // 50
        vector_output = operator.apply_vector(vector)
        matrix_output = operator.apply_matrix(matrix)
        pair_e, pair_y = operator.apply_pair(matrix, vector)
        assert np.argmax(np.abs(vector_output)) == expected_index
        assert np.argmax(np.abs(matrix_output[:, 1])) == expected_index
        assert np.argmax(np.abs(pair_e[:, 1])) == np.argmax(np.abs(pair_y))


def test_saved_timing_validation_and_metadata_revision() -> None:
    validation_path = RESULT_ROOT / "validation.json"
    if not validation_path.is_file():
        return
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    assert payload["module_version"] == "1.0.1"
    assert payload["timing_validation"]["tested_operator_count"] == 50
    assert payload["timing_validation"]["max_peak_index_error_samples"] == 0
    assert payload["timing_validation"]["max_matrix_peak_index_error_samples"] == 0
    assert payload["timing_validation"]["max_pair_peak_index_error_samples"] == 0
    assert payload["tone_phase_alignment"]["pass"] is True
    assert payload["pre_post_numeric_regression"]["pass"] is True
    assert payload["pre_post_numeric_regression"]["max_vector_difference"] == 0.0
    assert payload["pre_post_numeric_regression"]["max_matrix_difference"] == 0.0
    assert "group_delay_compensated" not in (RESULT_ROOT / "operator_bank.json").read_text(
        encoding="utf-8"
    )
    assert "group_delay_compensated" not in (RESULT_ROOT / "operator_bank.csv").read_text(
        encoding="utf-8"
    )


def test_complex128_and_lengths_all_fifty() -> None:
    bank = LowBandwidthObservationBank()
    for k in range(1, 51):
        operator = bank.get_by_index(k)
        for length, seed in ((12286, 10), (4913, 11), (7371, 12)):
            output = operator.apply_vector(_complex_signal(length, seed + k))
            assert output.dtype == np.complex128
            assert output.shape == (operator.expected_output_length(length),)
            assert np.all(np.isfinite(output))
            assert operator.output_time_axis(length).shape == output.shape
            assert np.all(np.diff(operator.output_time_axis(length)) > 0) or output.size == 1


def test_matrix_is_columnwise_vector_equivalent_all_fifty() -> None:
    bank = LowBandwidthObservationBank()
    matrix = np.column_stack([_complex_signal(4913, 20 + column) for column in range(10)])
    for k in range(1, 51):
        operator = bank.get_by_index(k)
        observed = operator.apply_matrix(matrix)
        expected = np.column_stack(
            [operator.apply_vector(matrix[:, column]) for column in range(10)]
        )
        np.testing.assert_allclose(observed, expected, rtol=0.0, atol=1e-12)


def test_linearity_and_pair_synchronization() -> None:
    bank = LowBandwidthObservationBank()
    z1 = _complex_signal(1024, 30)
    z2 = _complex_signal(1024, 31)
    alpha, beta = 0.37 - 0.21j, -0.44 + 0.19j
    matrix = np.column_stack([_complex_signal(4913, 40 + column) for column in range(10)])
    output = _complex_signal(4913, 99)
    for k in range(1, 51):
        operator = bank.get_by_index(k)
        left = operator.apply_vector(alpha * z1 + beta * z2)
        right = alpha * operator.apply_vector(z1) + beta * operator.apply_vector(z2)
        np.testing.assert_allclose(left, right, rtol=0.0, atol=1e-11)
        e_output, y_output = operator.apply_pair(matrix, output)
        assert e_output.shape[0] == y_output.shape[0]
        assert e_output.shape[1] == matrix.shape[1]


def test_dc_and_alias_contracts() -> None:
    bank = LowBandwidthObservationBank()
    dc = np.ones(10000, dtype=np.complex128)
    for k in (1, 5, 10, 20, 30, 40, 50):
        operator = bank.get_by_index(k)
        observed = operator.apply_vector(dc)
        trim = max(1, observed.size // 20)
        central = observed[trim:-trim] if observed.size > 2 * trim else observed
        assert abs(float(np.median(np.abs(central))) - 1.0) < 0.02
    operator = bank.get_by_index(10)
    frequency = 13_000_000.0
    tone_length = 100003
    tone = np.exp(1j * 2 * np.pi * frequency * np.arange(tone_length) / FS_INPUT_HZ).astype(
        np.complex128
    )
    formal = operator.apply_vector(tone)
    alias_frequency = (
        (frequency + operator.spec.fs_out_hz / 2.0) % operator.spec.fs_out_hz
    ) - operator.spec.fs_out_hz / 2.0
    trim = formal.size // 20
    formal_amp = abs(
        np.mean(
            formal[trim:-trim]
            * np.exp(
                -1j
                * 2
                * np.pi
                * alias_frequency
                * np.arange(formal.size - 2 * trim)
                / operator.spec.fs_out_hz
            )
        )
    )
    naive = tone[::5]
    trim_naive = naive.size // 20
    naive_amp = abs(
        np.mean(
            naive[trim_naive:-trim_naive]
            * np.exp(
                -1j
                * 2
                * np.pi
                * alias_frequency
                * np.arange(naive.size - 2 * trim_naive)
                / operator.spec.fs_out_hz
            )
        )
    )
    assert 20 * np.log10(naive_amp / max(formal_amp, np.finfo(float).tiny)) > 20.0


def test_saved_outputs_if_available() -> None:
    validation_path = RESULT_ROOT / "validation.json"
    if not validation_path.is_file():
        return
    payload = json.loads(validation_path.read_text(encoding="utf-8"))
    assert payload["operator_count"] == 50
    assert payload["formal_control_variable"] == "target_sample_rate"
    assert payload["effective_bandwidth_parameter_exists"] is False
    assert payload["operator_is_controlled_only_by_target_sample_rate"] is True
    assert payload["lut_retrieval_executed"] is False
    frame = pd.read_csv(RESULT_ROOT / "operator_bank.csv")
    assert frame.shape[0] == 50
    assert np.allclose(frame["fs_out_hz"], frame["observation_bandwidth_hz"])
    smoke = pd.read_csv(RESULT_ROOT / "real_signal_smoke_test.csv")
    assert smoke.shape[0] == 7
    assert bool(smoke.loc[smoke["sample_rate_index_k"] == 50, "five_B_identity"].all())


def main() -> None:
    tests = [
        test_bank_grid_and_derived_bandwidth,
        test_exact_rational_ratios,
        test_sample_rate_and_n_input_validation,
        test_five_b_identity_vector_matrix_and_real,
        test_timing_metadata_semantics_and_old_field_absence,
        test_interior_impulse_matrix_and_pair_timing_all_fifty,
        test_complex128_and_lengths_all_fifty,
        test_matrix_is_columnwise_vector_equivalent_all_fifty,
        test_linearity_and_pair_synchronization,
        test_dc_and_alias_contracts,
        test_saved_timing_validation_and_metadata_revision,
        test_saved_outputs_if_available,
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All low-bandwidth observation tests passed ({len(tests)} tests).")


if __name__ == "__main__":
    main()
