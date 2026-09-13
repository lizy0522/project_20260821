"""
功能说明：迁移 MATLAB Rough_Align.m 与 Fine_Align.m，实现整数和分数采样对齐。
输入：等长的一维 complex NumPy 数组 x、y；fine_align 可指定 subtime 和 ns。
输出：对齐后的 y，以及实际施加到 y 的整数或分数采样位移。
数学定义：通过最大互相关估计延迟；分数延迟用频域线性相位旋转补偿。
"""

from __future__ import annotations

import numpy as np

from ..utils import validate_pair


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def rough_align(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, int]:
    """估计整数延迟并用循环移位对齐 y。

    返回的 ``delay`` 是实际施加给 ``y`` 的位移；正值向右、负值向左。
    相关计算等价于 MATLAB 源函数构造的完整正、负时延互相关。
    """
    x_checked, y_checked = validate_pair(
        x,
        y,
        "x",
        "y",
        require_complex=True,
    )
    sample_count = x_checked.size
    full_length = 2 * sample_count - 1
    fft_length = _next_power_of_two(full_length)

    # 零填充 FFT 计算完整线性互相关，顺序对应 lags=-(N-1):N-1。
    correlation_circular = np.fft.ifft(
        np.fft.fft(x_checked, fft_length) * np.conj(np.fft.fft(y_checked, fft_length))
    )
    correlation = np.concatenate(
        (
            correlation_circular[-(sample_count - 1) :],
            correlation_circular[:sample_count],
        )
    )
    lags = np.arange(-(sample_count - 1), sample_count)
    best_index = int(np.argmax(np.abs(correlation)))
    delay = int(lags[best_index])
    return np.roll(y_checked, delay), delay


def fine_align(
    x: np.ndarray,
    y: np.ndarray,
    subtime: int = 256,
    ns: int | None = None,
) -> tuple[np.ndarray, float]:
    """在约 ``[-1, 1)`` 个采样内搜索并补偿分数延迟。

    ``subtime`` 与 MATLAB 参数一致，分数延迟网格步长为 ``1/subtime``。
    MATLAB 源函数返回网格整数；本接口返回实际施加的采样位移
    ``grid_delay/subtime``，便于后续直接解释。
    """
    x_checked, y_checked = validate_pair(
        x,
        y,
        "x",
        "y",
        require_complex=True,
    )
    if not isinstance(subtime, int) or isinstance(subtime, bool) or subtime <= 0:
        raise ValueError("subtime 必须是正整数")

    sample_count = x_checked.size
    if ns is None:
        ns = sample_count
    if not isinstance(ns, int) or isinstance(ns, bool) or ns < sample_count:
        raise ValueError("ns 必须是大于或等于输入长度的整数")

    spectrum_x = np.fft.fftshift(np.fft.fft(x_checked, ns))
    spectrum_y = np.fft.fftshift(np.fft.fft(y_checked, ns))
    phase_axis = np.arange(1, ns + 1, dtype=np.float64) / ns
    lag_grid = np.arange(-subtime, subtime, dtype=np.int64)

    # MATLAB 对复数 max 按幅度选取；逐网格计算以避免构造大型二维相位矩阵。
    correlation_magnitudes = np.empty(lag_grid.size, dtype=np.float64)
    for index, grid_delay in enumerate(lag_grid):
        phase = np.exp(1j * 2 * np.pi * phase_axis * grid_delay / subtime)
        correlation_magnitudes[index] = abs(np.vdot(spectrum_y * phase, spectrum_x))

    best_grid_delay = int(lag_grid[int(np.argmax(correlation_magnitudes))])
    fraction_delay = best_grid_delay / subtime
    phase = np.exp(1j * 2 * np.pi * phase_axis * fraction_delay)

    # 保留 MATLAB Fine_Align.m 中第二次 fftshift 的精确顺序。
    y_aligned = np.fft.ifft(np.fft.fftshift(spectrum_y * phase))
    return y_aligned, float(fraction_delay)


__all__ = ["fine_align", "rough_align"]
