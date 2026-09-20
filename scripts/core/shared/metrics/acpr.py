"""
功能说明：迁移 MATLAB aclr.m、chan_pow.m 和 dbp.m，计算上下邻道功率比。
输入：时域 signal、采样率 fs、积分带宽 bandwidth、邻道中心间隔 spacing。
输出：下邻道和上邻道 ACPR，单位 dB。
数学定义：ACPR=10*log10(P_adjacent/P_main)，通道功率由加窗 FFT 频带积分得到。
"""

from __future__ import annotations

import math

import numpy as np

from ..utils import as_numeric_vector


def _channel_power(
    signal: np.ndarray,
    frequency_band: tuple[float, float],
    fs: float,
) -> float:
    """按 MATLAB chan_pow.m 的 Hann 窗、FFT 长度和 bin 边界计算通道功率。"""
    sample_count = signal.size
    fft_length = 1 << (sample_count.bit_length() - 1)
    window = np.hanning(fft_length)
    windowed = signal[:fft_length] * window
    spectrum = np.fft.fftshift(np.fft.fft(windowed, fft_length) / fft_length / fs)

    lower_frequency = min(frequency_band)
    upper_frequency = max(frequency_band)
    if not (-fs / 2 < lower_frequency < upper_frequency < fs / 2):
        raise ValueError(f"积分频带 {frequency_band} 必须严格位于 Nyquist 区间 (-fs/2, fs/2)")

    # MATLAB: ceil((f_low+fs/2)/fs*Nfft):floor((f_up+fs/2)/fs*Nfft)
    lower_matlab_index = math.ceil((lower_frequency + fs / 2) / fs * fft_length)
    upper_matlab_index = math.floor((upper_frequency + fs / 2) / fs * fft_length)
    start = lower_matlab_index - 1
    stop = upper_matlab_index
    if start < 0 or stop > fft_length or start >= stop:
        raise ValueError(f"积分频带 {frequency_band} 没有有效 FFT bin")

    inband = spectrum[start:stop]
    return float(np.vdot(inband, inband).real / (2 * np.pi))


def _db_power_ratio(ratio: float) -> float:
    """迁移 dbp.m：零功率比返回 -Inf，其余返回 10*log10(abs(ratio))。"""
    if ratio == 0:
        return float("-inf")
    return float(10 * np.log10(abs(ratio)))


def acpr(
    signal: np.ndarray,
    fs: float,
    bandwidth: float,
    spacing: float,
) -> tuple[float, float]:
    """返回 ``(acpr_lower_db, acpr_upper_db)``，不执行时域对齐或复增益调整。"""
    checked_signal = as_numeric_vector(signal, "signal")
    for value, name in ((fs, "fs"), (bandwidth, "bandwidth"), (spacing, "spacing")):
        if not np.isscalar(value) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} 必须是有限正数")

    main_band = (-bandwidth / 2, bandwidth / 2)
    lower_band = (-bandwidth / 2 - spacing, bandwidth / 2 - spacing)
    upper_band = (-bandwidth / 2 + spacing, bandwidth / 2 + spacing)

    main_power = _channel_power(checked_signal, main_band, float(fs))
    if main_power == 0:
        raise ValueError("主信道功率为零，ACPR 无定义")
    lower_power = _channel_power(checked_signal, lower_band, float(fs))
    upper_power = _channel_power(checked_signal, upper_band, float(fs))
    return (
        _db_power_ratio(lower_power / main_power),
        _db_power_ratio(upper_power / main_power),
    )


__all__ = ["acpr"]
