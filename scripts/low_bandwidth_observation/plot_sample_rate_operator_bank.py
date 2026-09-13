"""Python/matplotlib figures for the fixed sample-rate observation operator.

Figure contract: show that the bank covers the complete 2--100 MHz grid, that
the fixed polyphase operator suppresses an out-of-band alias, and that real
5B spectra are plotted on their own output sampling-rate axes.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _core.behavior_indexed_dpd.signal import LowBandwidthObservationBank  # noqa: E402
from data_manager import load_variable_by_id  # noqa: E402


def _configure() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 8,
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.linewidth": 0.8,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def _save(figure: plt.Figure, path: Path) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    width, height = figure.get_size_inches()
    plt.close(figure)
    return {"path": str(path), "width_in": float(width), "height_in": float(height), "dpi": 300}


def _limits(values: np.ndarray, references: tuple[float, ...] = ()) -> tuple[float, float]:
    values = np.asarray(values, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    refs = np.asarray(references, dtype=float)
    finite = np.concatenate((finite, refs[np.isfinite(refs)]))
    if finite.size == 0:
        raise ValueError("图形至少需要一个finite值")
    low, high = float(np.min(finite)), float(np.max(finite))
    span = high - low
    pad = max(span * 0.1, 0.1)
    return low - pad, high + pad


def plot_sampling_rate_bank(bank_frame: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    """Figure 1: n, target sample rate, and derived observation bandwidth."""

    _configure()
    x = bank_frame["normalized_bandwidth_n"].to_numpy(dtype=float)
    fs = bank_frame["fs_out_hz"].to_numpy(dtype=float) / 1e6
    figure, axis = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    axis.plot(x, fs, color="#2166ac", marker="o", ms=3.2, lw=1.2, markevery=2, label="Fs_obs")
    axis.set_xlabel("Normalized rate n = Fs_obs / B")
    axis.set_ylabel("Fs_obs and derived B_obs (MHz)")
    axis.set_title("Sample-rate observation operator bank")
    axis.set_xlim(float(x.min()), float(x.max()))
    span = float(fs.max() - fs.min())
    axis.set_ylim(float(fs.min() - 0.05 * span), float(fs.max() + 0.05 * span))
    axis.text(
        0.02,
        0.94,
        "For complex-baseband IQ: B_obs = Fs_obs",
        transform=axis.transAxes,
        fontsize=7,
        color="#555555",
    )
    axis.legend(loc="best", fontsize=7)
    details = _save(figure, output_path)
    details.update(
        {
            "operator_count": int(bank_frame.shape[0]),
            "fs_min_MHz": float(fs.min()),
            "fs_max_MHz": float(fs.max()),
        }
    )
    return details


def plot_alias_suppression(alias_frame: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    """Figure 2: direct decimation versus the fixed anti-aliasing operator."""

    _configure()
    required = {"method", "alias_level_dB"}
    missing = sorted(required - set(alias_frame.columns))
    if missing:
        raise ValueError(f"alias验证缺少字段：{missing}")
    figure, axis = plt.subplots(figsize=(6.8, 4.6), constrained_layout=True)
    colors = ["#969696", "#1b7837"]
    bars = axis.bar(alias_frame["method"], alias_frame["alias_level_dB"], color=colors, width=0.55)
    for bar, value in zip(bars, alias_frame["alias_level_dB"], strict=True):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            f"{value:.1f} dB",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    axis.set_ylabel("Alias component level (dB)")
    axis.set_title("Anti-aliasing control at Fs_obs = 20 MHz")
    axis.grid(axis="y", color="#dddddd", lw=0.5, alpha=0.7)
    details = _save(figure, output_path)
    details.update(
        {
            "alias_suppression_improvement_dB": float(
                alias_frame.iloc[-1]["alias_suppression_improvement_dB"]
            )
        }
    )
    return details


def plot_impulse_time_alignment(impulse_frame: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    """Engineering Figure: peak-index error for all 50 sample-rate operators."""

    _configure()
    required = {"sample_rate_index_k", "peak_index_error_samples"}
    missing = sorted(required - set(impulse_frame.columns))
    if missing:
        raise ValueError(f"impulse验证缺少字段：{missing}")
    x = impulse_frame["sample_rate_index_k"].to_numpy(dtype=float)
    errors = impulse_frame["peak_index_error_samples"].to_numpy(dtype=float)
    figure, axis = plt.subplots(figsize=(8.2, 4.4), constrained_layout=True)
    axis.axhline(0.0, color="#555555", ls="--", lw=0.8)
    axis.plot(x, errors, color="#1b7837", marker="o", ms=3.0, lw=1.0)
    axis.set_xlabel("Sample-rate index k")
    axis.set_ylabel("Interior impulse peak error (output samples)")
    axis.set_title("Resample timing alignment (n_in = 5000)")
    axis.set_xlim(float(x.min()), float(x.max()))
    axis.set_ylim(*_limits(errors, references=(0.0,)))
    details = _save(figure, output_path)
    details.update(
        {
            "tested_operator_count": int(impulse_frame.shape[0]),
            "max_peak_index_error_samples": float(np.max(np.abs(errors))),
        }
    )
    return details


def _psd_db(signal: np.ndarray, fs_hz: float) -> tuple[np.ndarray, np.ndarray]:
    signal = np.asarray(signal, dtype=np.complex128)
    spectrum = np.fft.fftshift(np.fft.fft(signal))
    frequency = np.fft.fftshift(np.fft.fftfreq(signal.size, d=1.0 / fs_hz)) / 1e6
    magnitude = np.abs(spectrum)
    reference = float(np.max(magnitude))
    return frequency, 20.0 * np.log10(np.maximum(magnitude / reference, np.finfo(float).tiny))


def plot_real_signal_observation_psd(output_path: Path, *, state_id: int = 0) -> dict[str, Any]:
    """Figure 3: PSDs with each curve using its own output Fs."""

    _configure()
    raw = np.asarray(load_variable_by_id(state_id, "yout_withoutdpd_ori"))
    if raw.ndim == 2 and raw.shape[1] == 1:
        raw = raw[:, 0]
    if raw.ndim != 1 or not np.iscomplexobj(raw) or not np.all(np.isfinite(raw)):
        raise ValueError("真实5B waveform必须是一维有限复数数组")
    bank = LowBandwidthObservationBank()
    figure, axis = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)
    curves = (
        (50, "#2166ac", "-"),
        (20, "#1b7837", "--"),
        (10, "#b2182b", "-."),
        (5, "#762a83", ":"),
    )
    for k, color, linestyle in curves:
        operator = bank.get_by_index(k)
        observed = operator.apply_vector(raw)
        frequency, psd = _psd_db(observed, operator.spec.fs_out_hz)
        axis.plot(
            frequency,
            psd,
            color=color,
            lw=1.0,
            linestyle=linestyle,
            label=f"{operator.spec.tag} ({operator.spec.fs_out_hz / 1e6:.0f} MHz)",
        )
    axis.set_xlabel("Frequency (MHz; each curve uses its own Fs_obs)")
    axis.set_ylabel("Relative PSD (dB)")
    axis.set_title("Real 5B waveform under sample-rate observation")
    axis.set_ylim(-100.0, 2.0)
    axis.legend(loc="best", fontsize=7)
    details = _save(figure, output_path)
    details.update({"state_id": int(state_id), "curve_count": len(curves)})
    return details


__all__ = [
    "plot_alias_suppression",
    "plot_impulse_time_alignment",
    "plot_real_signal_observation_psd",
    "plot_sampling_rate_bank",
]
