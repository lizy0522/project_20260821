"""
功能说明：直接执行或由 pytest 收集的共享核心模块回归测试。
输入：脚本内部生成的确定性复信号，不读取科研原始数据。
输出：断言全部通过，并在直接执行时输出测试项目与 PASS 汇总。
数学定义：覆盖整数/分数延迟、复增益、NMSE、CNMSE、ACPR 和固定预处理链。
"""

from __future__ import annotations

import sys
from pathlib import Path

CORE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
# 直接执行本文件时，sys.path[0] 是 core；移除它，避免 core/signal 遮蔽标准库 signal。
sys.path[:] = [entry for entry in sys.path if Path(entry or ".").resolve() != CORE_ROOT]
sys.path.insert(0, str(SCRIPTS_ROOT))

import numpy as np  # noqa: E402

from core.shared.metrics import acpr, cnmse, nmse  # noqa: E402
from core.shared.signal import (  # noqa: E402
    adjust_complex_gain,
    fine_align,
    preprocess_pair,
    rough_align,
)


def _complex_test_signal(sample_count: int = 1024) -> np.ndarray:
    rng = np.random.default_rng(20260825)
    return rng.normal(size=sample_count) + 1j * rng.normal(size=sample_count)


def _apply_matlab_fraction_shift(x: np.ndarray, shift: float) -> np.ndarray:
    sample_count = x.size
    spectrum = np.fft.fftshift(np.fft.fft(x))
    phase_axis = np.arange(1, sample_count + 1, dtype=np.float64) / sample_count
    shifted = spectrum * np.exp(-1j * 2 * np.pi * phase_axis * shift)
    return np.fft.ifft(np.fft.fftshift(shifted))


def test_rough_align_recovers_integer_delay() -> None:
    x = _complex_test_signal()
    y = np.roll(x, 20)
    y_aligned, delay = rough_align(x, y)
    assert delay == -20
    np.testing.assert_allclose(y_aligned, x, rtol=0, atol=0)


def test_fine_align_recovers_fraction_delay() -> None:
    x = _complex_test_signal()
    expected_shift = 24 / 64
    y = _apply_matlab_fraction_shift(x, expected_shift)
    y_aligned, fraction_delay = fine_align(x, y, subtime=64)
    assert fraction_delay == expected_shift
    np.testing.assert_allclose(y_aligned, x, rtol=1e-12, atol=1e-12)


def test_adjust_complex_gain_recovers_reference() -> None:
    x = _complex_test_signal()
    applied_gain = 1.7 * np.exp(1j * 0.63)
    y = applied_gain * x
    y_adjusted, gain = adjust_complex_gain(x, y)
    np.testing.assert_allclose(y_adjusted, x, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(gain, 1 / applied_gain, rtol=1e-13, atol=1e-13)


def test_nmse_identical_is_negative_infinity() -> None:
    x = _complex_test_signal()
    assert np.isneginf(nmse(x, x))


def test_cnmse_identical_is_negative_infinity() -> None:
    y = _complex_test_signal()
    assert np.isneginf(cnmse(y, y))


def test_acpr_resolves_lower_and_upper_adjacent_tones() -> None:
    sample_count = 16384
    fs = 100e6
    time = np.arange(sample_count) / fs
    signal = (
        np.ones(sample_count, dtype=np.complex128)
        + 0.1 * np.exp(-1j * 2 * np.pi * 20e6 * time)
        + 0.2 * np.exp(1j * 2 * np.pi * 20e6 * time)
    )
    lower_db, upper_db = acpr(signal, fs=fs, bandwidth=10e6, spacing=20e6)
    np.testing.assert_allclose(lower_db, -20.0, atol=0.05)
    np.testing.assert_allclose(upper_db, 10 * np.log10(0.2**2), atol=0.05)


def test_preprocess_pair_runs_fixed_pipeline_without_mutating_inputs() -> None:
    x = _complex_test_signal()
    y = np.roll(_apply_matlab_fraction_shift(x, 16 / 64), 12)
    y *= 0.8 * np.exp(1j * 0.4)
    x_before = x.copy()
    y_before = y.copy()
    processed = preprocess_pair(x, y, subtime=64)
    np.testing.assert_allclose(processed, x, rtol=1e-11, atol=1e-11)
    np.testing.assert_array_equal(x, x_before)
    np.testing.assert_array_equal(y, y_before)


def main() -> None:
    tests = [
        test_rough_align_recovers_integer_delay,
        test_fine_align_recovers_fraction_delay,
        test_adjust_complex_gain_recovers_reference,
        test_nmse_identical_is_negative_infinity,
        test_cnmse_identical_is_negative_infinity,
        test_acpr_resolves_lower_and_upper_adjacent_tones,
        test_preprocess_pair_runs_fixed_pipeline_without_mutating_inputs,
    ]
    for test in tests:
        test()
        print(f"PASS: {test.__name__}")
    print(f"All {len(tests)} core module tests passed.")


if __name__ == "__main__":
    main()
