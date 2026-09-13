"""Low-bandwidth observation / low-speed ADC equivalent operator.

The formal control variable is the target complex-baseband ADC sampling rate
``Fs_obs``.  The nominal observation bandwidth is derived as ``B_obs = Fs_obs``
and is not an independent parameter.

The operator acts *after* nonlinear basis generation.  The correct future model
order is::

    x_5B -> Phi(x_5B) -> E_5B -> H_Fs -> E_nB

The incorrect order is::

    x_5B -> H_Fs -> x_nB -> Phi(x_nB)

This module only performs the fixed anti-aliasing polyphase rational resampling
operation.  It does not know about MP, LUT, ILC, Aend, C2, or retrieval logic.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from fractions import Fraction
from numbers import Integral, Real
from typing import Any

import numpy as np
from scipy.signal import resample_poly

from .config import (
    ADDITIONAL_DELAY_COMPENSATION_REQUIRED,
    BASE_SIGNAL_BANDWIDTH_HZ,
    EDGE_POLICY,
    EXPLICIT_SAMPLE_SHIFT_APPLIED,
    FS_INPUT_HZ,
    INPUT_TIME_ORIGIN_S,
    MANUAL_GROUP_DELAY_COMPENSATION_APPLIED,
    NORMALIZED_RATIO_DENOMINATOR,
    OUTPUT_TIME_ORIGIN_ALIGNED,
    OUTPUT_TIME_ORIGIN_S,
    RESAMPLER_CVAL,
    RESAMPLER_DELAY_HANDLING,
    RESAMPLER_NAME,
    RESAMPLER_PADTYPE,
    RESAMPLER_WINDOW,
    SAMPLE_RATE_INDEX_MAX,
    SAMPLE_RATE_INDEX_MIN,
    SAMPLE_RATE_STEP_HZ,
)


def _validate_index(value: Integral) -> int:
    if not isinstance(value, Integral) or isinstance(value, (bool, np.bool_)):
        raise TypeError("sample_rate_index_k必须是整数")
    normalized = int(value)
    if not SAMPLE_RATE_INDEX_MIN <= normalized <= SAMPLE_RATE_INDEX_MAX:
        raise ValueError(
            f"sample_rate_index_k必须位于{SAMPLE_RATE_INDEX_MIN}...{SAMPLE_RATE_INDEX_MAX}"
        )
    return normalized


def _index_from_real(value: Real, *, scale: float, name: str) -> int:
    if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name}必须是有限实数")
    normalized = float(value)
    if not np.isfinite(normalized):
        raise ValueError(f"{name}必须是有限实数")
    scaled = normalized * scale
    nearest = int(round(scaled))
    if not math.isclose(scaled, nearest, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{name}不对应0.1步长采样率档位：{value!r}")
    return _validate_index(nearest)


@dataclass(frozen=True, slots=True)
class SampleRateSpec:
    """One immutable sampling-rate definition in the 2--100 MHz grid.

    Only ``sample_rate_index_k`` is an input.  All other fields are derived
    from the fixed project facts and therefore cannot introduce a second
    observation-bandwidth control variable.
    """

    sample_rate_index_k: int
    normalized_bandwidth_n: float = field(init=False)
    base_signal_bandwidth_hz: int = field(init=False)
    fs_in_hz: int = field(init=False)
    fs_out_hz: int = field(init=False)
    observation_bandwidth_hz: int = field(init=False)
    up: int = field(init=False)
    down: int = field(init=False)
    tag: str = field(init=False)
    is_identity: bool = field(init=False)

    def __post_init__(self) -> None:
        k = _validate_index(self.sample_rate_index_k)
        object.__setattr__(self, "sample_rate_index_k", k)
        fs_out = SAMPLE_RATE_STEP_HZ * k
        gcd = math.gcd(k, 50)
        up = k // gcd
        down = 50 // gcd
        object.__setattr__(self, "normalized_bandwidth_n", k / NORMALIZED_RATIO_DENOMINATOR)
        object.__setattr__(self, "base_signal_bandwidth_hz", BASE_SIGNAL_BANDWIDTH_HZ)
        object.__setattr__(self, "fs_in_hz", FS_INPUT_HZ)
        object.__setattr__(self, "fs_out_hz", fs_out)
        object.__setattr__(self, "observation_bandwidth_hz", fs_out)
        object.__setattr__(self, "up", up)
        object.__setattr__(self, "down", down)
        object.__setattr__(self, "tag", f"nB_{k // 10}p{k % 10}")
        object.__setattr__(self, "is_identity", k == SAMPLE_RATE_INDEX_MAX)
        if up * 50 != down * k or math.gcd(up, down) != 1:
            raise RuntimeError("up/down不是k/50的最简整数比例")
        if fs_out != self.observation_bandwidth_hz:
            raise RuntimeError("observation_bandwidth_hz必须由fs_out_hz派生")

    @classmethod
    def from_n(cls, value: Real) -> SampleRateSpec:
        """Construct from n in the exact 0.1, ..., 5.0 grid."""

        return cls(_index_from_real(value, scale=NORMALIZED_RATIO_DENOMINATOR, name="n"))

    @classmethod
    def from_sample_rate(cls, value: Real) -> SampleRateSpec:
        """Construct from an exact 2 MHz-step target sampling rate."""

        if not isinstance(value, Real) or isinstance(value, (bool, np.bool_)):
            raise TypeError("fs_out_hz必须是有限实数")
        normalized = float(value)
        if not np.isfinite(normalized):
            raise ValueError("fs_out_hz必须是有限实数")
        scaled = normalized / SAMPLE_RATE_STEP_HZ
        nearest = int(round(scaled))
        if not math.isclose(scaled, nearest, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError("fs_out_hz必须是2 MHz的整数倍")
        return cls(nearest)

    @property
    def ratio_fraction(self) -> Fraction:
        """Return the exact input-to-output ratio ``up/down``."""

        return Fraction(self.up, self.down)

    def as_dict(self) -> dict[str, Any]:
        """Return JSON/CSV-ready derived metadata."""

        return asdict(self) | {
            "resampler": RESAMPLER_NAME,
            "window": list(RESAMPLER_WINDOW),
            "padtype": RESAMPLER_PADTYPE,
            "cval": RESAMPLER_CVAL,
            "edge_policy": EDGE_POLICY,
            "formal_control_variable": "target_sample_rate",
        }


def _validate_signal(value: Any, name: str, *, ndim: int) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim:
        raise ValueError(f"{name}必须是{ndim}维数组")
    if array.shape[0] <= 0:
        raise ValueError(f"{name}时间轴长度必须为正数")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name}必须是数值数组")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name}包含NaN或Inf")
    return array


class LowBandwidthObservationOperator:
    """Apply the one fixed anti-aliasing rational resampler for a spec.

    ``E_nb = H(E_5b)`` and ``y_nb = H(y_5b)`` must use this same instance in
    future model integration.  The operator has no mode, filter, or bandwidth
    argument: only the target sample rate encoded by ``SampleRateSpec`` varies.
    """

    def __init__(self, spec: SampleRateSpec) -> None:
        if not isinstance(spec, SampleRateSpec):
            raise TypeError("spec必须是SampleRateSpec")
        self._spec = spec

    @property
    def spec(self) -> SampleRateSpec:
        return self._spec

    def apply_vector(self, value: Any) -> np.ndarray:
        """Apply the operator along a one-dimensional time axis."""

        vector = _validate_signal(value, "vector", ndim=1)
        if self._spec.is_identity:
            return vector.copy()
        output = resample_poly(
            vector,
            self._spec.up,
            self._spec.down,
            axis=0,
            window=RESAMPLER_WINDOW,
            padtype=RESAMPLER_PADTYPE,
            cval=RESAMPLER_CVAL,
        )
        expected = self.expected_output_length(vector.shape[0])
        if output.shape != (expected,) or not np.all(np.isfinite(output)):
            raise RuntimeError(f"resample_poly输出长度或finite状态错误：{output.shape}")
        return np.asarray(output)

    def apply_matrix(self, value: Any) -> np.ndarray:
        """Apply the same operator independently to every time-series column."""

        matrix = _validate_signal(value, "matrix", ndim=2)
        if matrix.shape[1] <= 0:
            raise ValueError("matrix必须至少包含一列")
        if self._spec.is_identity:
            return matrix.copy()
        output = resample_poly(
            matrix,
            self._spec.up,
            self._spec.down,
            axis=0,
            window=RESAMPLER_WINDOW,
            padtype=RESAMPLER_PADTYPE,
            cval=RESAMPLER_CVAL,
        )
        expected = self.expected_output_length(matrix.shape[0])
        if output.shape != (expected, matrix.shape[1]) or not np.all(np.isfinite(output)):
            raise RuntimeError(f"resample_poly矩阵输出长度或finite状态错误：{output.shape}")
        return np.asarray(output)

    def apply(self, value: Any) -> np.ndarray:
        """Dispatch one- or two-dimensional time-series input."""

        array = np.asarray(value)
        if array.ndim == 1:
            return self.apply_vector(array)
        if array.ndim == 2:
            return self.apply_matrix(array)
        raise ValueError("输入必须是一维vector或二维(time, feature)矩阵")

    def apply_pair(self, e_5b: Any, y_5b: Any) -> tuple[np.ndarray, np.ndarray]:
        """Apply this same operator to a basis matrix and output vector.

        The same operator specification, resampling phase, filter, padding,
        and time-axis convention are used for both arrays.
        """

        e_nb = self.apply_matrix(e_5b)
        y_nb = self.apply_vector(y_5b)
        if e_nb.shape[0] != y_nb.shape[0]:
            raise RuntimeError("apply_pair输出的E/y时间长度不一致")
        return e_nb, y_nb

    def expected_output_length(self, input_length: Integral) -> int:
        """Return the exact current ``resample_poly`` output-length rule."""

        if not isinstance(input_length, Integral) or isinstance(input_length, (bool, np.bool_)):
            raise TypeError("input_length必须是正整数")
        n = int(input_length)
        if n <= 0:
            raise ValueError("input_length必须是正整数")
        if self._spec.is_identity:
            return n
        return (n * self._spec.up + self._spec.down - 1) // self._spec.down

    def output_time_axis(self, input_length: Integral) -> np.ndarray:
        """Return ``t_out[m] = m / Fs_obs`` for the output grid."""

        length = self.expected_output_length(input_length)
        return np.arange(length, dtype=np.float64) / float(self._spec.fs_out_hz)

    def describe(self) -> dict[str, Any]:
        """Return the fixed operator and time-axis metadata."""

        return {
            **self._spec.as_dict(),
            "module": "low-bandwidth observation / low-speed ADC equivalent operator",
            "resampler_variant": "polyphase_fir_fixed",
            "manual_group_delay_compensation_applied": (MANUAL_GROUP_DELAY_COMPENSATION_APPLIED),
            "explicit_sample_shift_applied": EXPLICIT_SAMPLE_SHIFT_APPLIED,
            "output_time_origin_aligned": OUTPUT_TIME_ORIGIN_ALIGNED,
            "input_time_origin_s": INPUT_TIME_ORIGIN_S,
            "output_time_origin_s": OUTPUT_TIME_ORIGIN_S,
            "additional_delay_compensation_required": (ADDITIONAL_DELAY_COMPENSATION_REQUIRED),
            "resampler_delay_handling": RESAMPLER_DELAY_HANDLING,
            "effective_bandwidth_parameter_exists": False,
            "operator_is_controlled_only_by_target_sample_rate": True,
            "adc_quantization_enabled": False,
            "adc_noise_enabled": False,
            "adc_jitter_enabled": False,
            "time_axis_policy": "t_out[m] = m / fs_out_hz",
        }


class LowBandwidthObservationBank:
    """Construct the complete 50-spec observation-rate bank on demand."""

    def __init__(self) -> None:
        self._specs = tuple(
            SampleRateSpec(index)
            for index in range(SAMPLE_RATE_INDEX_MIN, SAMPLE_RATE_INDEX_MAX + 1)
        )

    @property
    def specs(self) -> tuple[SampleRateSpec, ...]:
        return self._specs

    def get_spec_by_index(self, sample_rate_index_k: Integral) -> SampleRateSpec:
        index = _validate_index(sample_rate_index_k)
        return self._specs[index - SAMPLE_RATE_INDEX_MIN]

    def get_by_index(self, sample_rate_index_k: Integral) -> LowBandwidthObservationOperator:
        return LowBandwidthObservationOperator(self.get_spec_by_index(sample_rate_index_k))

    def get_by_sample_rate(self, fs_out_hz: Real) -> LowBandwidthObservationOperator:
        return self.get_by_index(SampleRateSpec.from_sample_rate(fs_out_hz).sample_rate_index_k)

    def get_by_n(self, normalized_ratio: Real) -> LowBandwidthObservationOperator:
        return self.get_by_index(SampleRateSpec.from_n(normalized_ratio).sample_rate_index_k)

    def describe(self) -> dict[str, Any]:
        return {
            "module": "low-bandwidth observation / low-speed ADC equivalent operator",
            "formal_control_variable": "target_sample_rate",
            "base_signal_bandwidth_hz": BASE_SIGNAL_BANDWIDTH_HZ,
            "fs_in_hz": FS_INPUT_HZ,
            "sample_rate_min_hz": SAMPLE_RATE_STEP_HZ * SAMPLE_RATE_INDEX_MIN,
            "sample_rate_max_hz": SAMPLE_RATE_STEP_HZ * SAMPLE_RATE_INDEX_MAX,
            "sample_rate_step_hz": SAMPLE_RATE_STEP_HZ,
            "normalized_n_min": SAMPLE_RATE_INDEX_MIN / 10,
            "normalized_n_max": SAMPLE_RATE_INDEX_MAX / 10,
            "normalized_n_step": 0.1,
            "operator_count": len(self._specs),
            "observation_bandwidth_rule": "B_obs = Fs_obs",
            "effective_bandwidth_parameter_exists": False,
            "operator_is_controlled_only_by_target_sample_rate": True,
            "resampler": RESAMPLER_NAME,
            "window": list(RESAMPLER_WINDOW),
            "padtype": RESAMPLER_PADTYPE,
            "cval": RESAMPLER_CVAL,
            "edge_policy": EDGE_POLICY,
            "manual_group_delay_compensation_applied": (MANUAL_GROUP_DELAY_COMPENSATION_APPLIED),
            "explicit_sample_shift_applied": EXPLICIT_SAMPLE_SHIFT_APPLIED,
            "output_time_origin_aligned": OUTPUT_TIME_ORIGIN_ALIGNED,
            "input_time_origin_s": INPUT_TIME_ORIGIN_S,
            "output_time_origin_s": OUTPUT_TIME_ORIGIN_S,
            "additional_delay_compensation_required": (ADDITIONAL_DELAY_COMPENSATION_REQUIRED),
            "resampler_delay_handling": RESAMPLER_DELAY_HANDLING,
            "five_B_identity": True,
            "complex_dtype_policy": (
                "complex128 input remains complex128; real input is preserved as real"
            ),
            "time_axis_policy": "t_out[m] = m / fs_out_hz",
            "adc_quantization_enabled": False,
            "adc_noise_enabled": False,
            "adc_jitter_enabled": False,
            "specs": [spec.as_dict() for spec in self._specs],
        }


__all__ = [
    "LowBandwidthObservationBank",
    "LowBandwidthObservationOperator",
    "SampleRateSpec",
]
