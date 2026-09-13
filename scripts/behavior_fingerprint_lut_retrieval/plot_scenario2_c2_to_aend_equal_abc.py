"""Python/matplotlib figures for the equal-length ABC ablation.

The hero figure tests whether changing only the ownership window from
12288/4915/7373 to 8192/8192/8192 changes the four model metrics or the
retrieved Real-B CNMSE.  It keeps all 425 State_Q annotations and encodes each
series with a distinct marker so that the figure remains interpretable in
grayscale.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .scenario2_c2_to_aend_equal_abc import DPD_SHAREABLE_THRESHOLD_DB


def _configure_matplotlib() -> None:
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
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def _save_png(fig: plt.Figure, output_path: Path) -> dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    width, height = fig.get_size_inches()
    plt.close(fig)
    return {
        "path": str(output_path),
        "width_in": float(width),
        "height_in": float(height),
        "dpi": 300,
    }


def _finite_limits(values: np.ndarray, references: tuple[float, ...] = ()) -> tuple[float, float]:
    values = np.asarray(values, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("figure至少需要一个finite值")
    combined = np.concatenate([finite, np.asarray(references, dtype=float)])
    combined = combined[np.isfinite(combined)]
    low = float(np.min(combined))
    high = float(np.max(combined))
    if low == high:
        delta = max(abs(low) * 1e-6, np.finfo(float).eps * 10.0)
        return low - delta, high + delta
    return low, high


def _state_id_column(frame: pd.DataFrame) -> str:
    if "state_id_R" in frame.columns:
        return "state_id_R"
    if "state_id" in frame.columns:
        return "state_id"
    raise ValueError("statewise表缺少state_id_R/state_id")


def _metric_column(frame: pd.DataFrame, preferred: str, fallback: str | None = None) -> str:
    if preferred in frame.columns:
        return preferred
    if fallback is not None and fallback in frame.columns:
        return fallback
    raise ValueError(f"statewise表缺少{preferred}")


def plot_equal_abc_state_metrics(statewise: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    """Create the required six-curve 425-state figure."""

    _configure_matplotlib()
    state_column = _state_id_column(statewise)
    if "state_id_Q" not in statewise.columns:
        raise ValueError("statewise表缺少state_id_Q")
    metric_specs = [
        ("nmse_withoutdpd_dB", "nmse_withoutdpd", "PA without DPD NMSE", "o", "-"),
        ("Y_Aend_train_NMSE_dB", None, "Y-Aend train NMSE", "s", "-"),
        ("Y_Aend_B_NMSE_dB", None, "Y-Aend -> B NMSE", "^", "--"),
        ("Y_C2_train_NMSE_dB", None, "Y-C2 train NMSE", "D", "-"),
        ("Y_C2_B_NMSE_dB", None, "Y-C2 -> B NMSE", "P", "--"),
        ("retrieved_real_B_CNMSE_dB", None, "Retrieved Real-B CNMSE", "X", "-"),
    ]
    resolved = []
    for preferred, fallback, label, marker, linestyle in metric_specs:
        resolved.append(
            (
                _metric_column(statewise, preferred, fallback),
                label,
                marker,
                linestyle,
            )
        )
    x = statewise[state_column].to_numpy(dtype=float)
    if not np.array_equal(x.astype(np.int64), np.arange(425, dtype=np.int64)):
        raise ValueError("State_R必须严格为0...424")
    q_values = statewise["state_id_Q"].to_numpy(dtype=np.int64)
    retrieved_column = resolved[-1][0]
    retrieved = statewise[retrieved_column].to_numpy(dtype=float)
    finite_retrieved = np.isfinite(retrieved)
    exact = (
        statewise["exact_hit"].to_numpy(dtype=bool)
        if "exact_hit" in statewise.columns
        else ~finite_retrieved
    )
    finite_values = np.concatenate(
        [
            statewise[column].to_numpy(dtype=float)[
                np.isfinite(statewise[column].to_numpy(dtype=float))
            ]
            for column, *_ in resolved
        ]
    )
    finite_min = float(np.min(finite_values))
    exact_plot_y = finite_min - 1.0
    y_min, y_max = _finite_limits(
        finite_values, references=(DPD_SHAREABLE_THRESHOLD_DB, exact_plot_y)
    )

    figure, axis = plt.subplots(figsize=(30.0, 10.0), constrained_layout=True)
    line_colors: dict[str, Any] = {}
    for column, label, marker, linestyle in resolved[:-1]:
        mask = np.isfinite(statewise[column].to_numpy(dtype=float))
        line = axis.plot(
            x[mask],
            statewise.loc[mask, column],
            marker=marker,
            linestyle=linestyle,
            linewidth=0.75,
            markersize=3.0,
            markevery=10,
            alpha=0.82,
            label=label,
        )[0]
        line_colors[column] = line.get_color()
    finite_line = axis.plot(
        x[finite_retrieved],
        retrieved[finite_retrieved],
        marker="X",
        linestyle="-",
        linewidth=0.75,
        markersize=3.8,
        alpha=0.88,
        label="Retrieved Real-B CNMSE",
    )[0]
    line_colors[retrieved_column] = finite_line.get_color()
    axis.axhline(
        DPD_SHAREABLE_THRESHOLD_DB,
        color="#444444",
        linewidth=0.9,
        linestyle="--",
        label="DPD-shareable threshold (-40 dB)",
    )
    if exact.any():
        axis.scatter(
            x[exact],
            np.full(int(exact.sum()), exact_plot_y),
            marker="|",
            s=58,
            linewidths=1.0,
            color=line_colors.get(retrieved_column, "black"),
            label="Exact hit (-Inf, displayed at bottom)",
            zorder=5,
        )

    annotation_count = 0
    for index in range(425):
        if exact[index]:
            y_value = exact_plot_y
            offset = (0, 3 if index % 2 == 0 else -8)
        elif finite_retrieved[index]:
            y_value = retrieved[index]
            offset = (0, 3 if index % 2 == 0 else -8)
        else:
            continue
        axis.annotate(
            f"Q={int(q_values[index])}",
            (x[index], y_value),
            xytext=offset,
            textcoords="offset points",
            rotation=90,
            va="bottom" if offset[1] > 0 else "top",
            ha="center",
            fontsize=4.2,
            color=line_colors.get(retrieved_column, "black"),
            clip_on=False,
        )
        annotation_count += 1
    axis.set_xlim(float(x.min()), float(x.max()))
    axis.set_ylim(y_min, y_max)
    axis.set_xlabel(r"State$_R$")
    axis.set_ylabel("Metric (dB)")
    axis.set_title("Scenario 2 C2 -> Aend retrieval with equal-length ABC segments")
    axis.legend(loc="upper left", fontsize=7, ncol=2)
    axis.text(
        0.995,
        0.02,
        "Exact -Inf markers are placed at a display-only bottom sentinel; Q labels are State_Q.",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        color="#333333",
    )
    details = _save_png(figure, output_path)
    details.update(
        {
            "curve_count": 6,
            "state_count": 425,
            "q_annotation_count": annotation_count,
            "finite_retrieved_count": int(finite_retrieved.sum()),
            "exact_hit_count": int(exact.sum()),
            "threshold_dB": DPD_SHAREABLE_THRESHOLD_DB,
            "y_min": y_min,
            "y_max": y_max,
            "exact_display_sentinel_y": exact_plot_y,
            "marker_contract": {label: marker for _, label, marker, _ in resolved},
        }
    )
    return details


def plot_nonexact_failure_detail(statewise: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    """Create an auxiliary readable plot for finite non-exact retrievals."""

    _configure_matplotlib()
    state_column = _state_id_column(statewise)
    retrieved = statewise["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float)
    exact = statewise["exact_hit"].to_numpy(dtype=bool)
    finite_nonexact = (~exact) & np.isfinite(retrieved)
    failure = finite_nonexact & (retrieved >= DPD_SHAREABLE_THRESHOLD_DB)
    x = statewise[state_column].to_numpy(dtype=float)
    finite_values = retrieved[finite_nonexact]
    y_min, y_max = _finite_limits(finite_values, references=(DPD_SHAREABLE_THRESHOLD_DB,))
    figure, axis = plt.subplots(figsize=(12.0, 4.6), constrained_layout=True)
    axis.plot(
        x[finite_nonexact],
        retrieved[finite_nonexact],
        color="#4d4d4d",
        linewidth=0.7,
        marker="o",
        markersize=2.6,
        label="Finite non-exact Real-B",
    )
    if failure.any():
        axis.scatter(
            x[failure],
            retrieved[failure],
            color="#b2182b",
            marker="X",
            s=35,
            label="Failure (>= -40 dB)",
        )
        for index in np.flatnonzero(failure):
            axis.annotate(
                f"Q={int(statewise.iloc[index]['state_id_Q'])}",
                (x[index], retrieved[index]),
                xytext=(0, 4),
                textcoords="offset points",
                rotation=90,
                fontsize=5,
                ha="center",
            )
    axis.axhline(
        DPD_SHAREABLE_THRESHOLD_DB,
        color="#444444",
        linestyle="--",
        linewidth=0.9,
        label="Threshold (-40 dB)",
    )
    axis.set_xlim(float(x.min()), float(x.max()))
    axis.set_ylim(y_min, y_max)
    axis.set_xlabel(r"State$_R$")
    axis.set_ylabel("Retrieved Real-B CNMSE (dB)")
    axis.set_title("Equal-ABC finite non-exact retrieval detail")
    axis.legend(loc="best", fontsize=7)
    details = _save_png(figure, output_path)
    details.update(
        {
            "finite_nonexact_count": int(finite_nonexact.sum()),
            "failure_count": int(failure.sum()),
            "threshold_dB": DPD_SHAREABLE_THRESHOLD_DB,
            "y_min": y_min,
            "y_max": y_max,
        }
    )
    return details


__all__ = ["plot_equal_abc_state_metrics", "plot_nonexact_failure_detail"]
