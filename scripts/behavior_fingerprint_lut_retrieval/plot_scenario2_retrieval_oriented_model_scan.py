"""Python/matplotlib figures for the retrieval-oriented model scan."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

THRESHOLD_DB = -40.0


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


def _limits(values: np.ndarray, references: tuple[float, ...] = ()) -> tuple[float, float]:
    values = np.asarray(values, dtype=float).reshape(-1)
    finite = values[np.isfinite(values)]
    refs = np.asarray(references, dtype=float)
    finite = np.concatenate((finite, refs[np.isfinite(refs)]))
    if finite.size == 0:
        raise ValueError("图形至少需要一个finite值")
    low, high = float(np.min(finite)), float(np.max(finite))
    span = high - low
    pad = max(span * 0.06, 0.25)
    return low - pad, high + pad


def _save(figure: plt.Figure, output_path: Path) -> dict[str, Any]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    width, height = figure.get_size_inches()
    plt.close(figure)
    return {
        "path": str(output_path),
        "width_in": float(width),
        "height_in": float(height),
        "dpi": 300,
    }


def plot_candidate_failure_count(
    ranking: pd.DataFrame, best_candidate_id: int, output_path: Path
) -> dict[str, Any]:
    """Figure 1: failure count for every retrieval candidate."""

    _configure()
    required = {"candidate_id", "failure_count", "order_profile", "is_baseline", "candidate_valid"}
    missing = sorted(required - set(ranking.columns))
    if missing:
        raise ValueError(f"candidate ranking缺少字段：{missing}")
    figure, axis = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    markers = {"P7": "o", "P9": "s", "P11": "^", "P13": "D", "P15": "v"}
    colors = {
        "P7": "#2166ac",
        "P9": "#4d9221",
        "P11": "#b2182b",
        "P13": "#762a83",
        "P15": "#e08214",
    }
    valid = ranking.loc[ranking["candidate_valid"].astype(bool)]
    for profile in markers:
        group = valid.loc[valid["order_profile"] == profile]
        axis.scatter(
            group["candidate_id"],
            group["failure_count"],
            marker=markers[profile],
            color=colors[profile],
            s=25,
            alpha=0.75,
            linewidths=0.25,
            label=profile,
        )
    baseline = valid.loc[valid["is_baseline"].astype(bool)]
    if not baseline.empty:
        axis.axhline(float(baseline.iloc[0]["failure_count"]), color="#555555", ls="--", lw=0.8)
        axis.scatter(
            baseline["candidate_id"],
            baseline["failure_count"],
            marker="*",
            s=140,
            facecolor="white",
            edgecolor="black",
            linewidth=0.9,
            zorder=5,
        )
        axis.annotate(
            "Baseline",
            (baseline.iloc[0]["candidate_id"], baseline.iloc[0]["failure_count"]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
        )
    best = valid.loc[valid["candidate_id"] == int(best_candidate_id)]
    if not best.empty:
        axis.scatter(
            best["candidate_id"],
            best["failure_count"],
            marker="P",
            s=110,
            color="#d73027",
            edgecolor="black",
            linewidth=0.7,
            zorder=6,
        )
        axis.annotate(
            "Best",
            (best.iloc[0]["candidate_id"], best.iloc[0]["failure_count"]),
            xytext=(5, -12),
            textcoords="offset points",
            fontsize=7,
            color="#d73027",
        )
    axis.set_xlabel("Candidate ID")
    axis.set_ylabel("Failure count")
    axis.set_title("Retrieval-oriented candidate scan")
    axis.set_xlim(1, int(ranking["candidate_id"].max()))
    axis.set_ylim(*_limits(valid["failure_count"].to_numpy(dtype=float)))
    axis.legend(title="Order profile", ncol=5, fontsize=7, title_fontsize=7, loc="upper right")
    details = _save(figure, output_path)
    details.update(
        {"candidate_count": int(ranking.shape[0]), "valid_candidate_count": int(valid.shape[0])}
    )
    return details


def plot_candidate_shareable_rate(
    ranking: pd.DataFrame, best_candidate_id: int, output_path: Path
) -> dict[str, Any]:
    """Figure 2: DPD-shareable rate for every retrieval candidate."""

    _configure()
    required = {"candidate_id", "shareable_rate", "order_profile", "is_baseline", "candidate_valid"}
    missing = sorted(required - set(ranking.columns))
    if missing:
        raise ValueError(f"candidate ranking缺少字段：{missing}")
    figure, axis = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    markers = {"P7": "o", "P9": "s", "P11": "^", "P13": "D", "P15": "v"}
    colors = {
        "P7": "#2166ac",
        "P9": "#4d9221",
        "P11": "#b2182b",
        "P13": "#762a83",
        "P15": "#e08214",
    }
    valid = ranking.loc[ranking["candidate_valid"].astype(bool)].copy()
    for profile in markers:
        group = valid.loc[valid["order_profile"] == profile]
        axis.scatter(
            group["candidate_id"],
            100.0 * group["shareable_rate"],
            marker=markers[profile],
            color=colors[profile],
            s=25,
            alpha=0.75,
            linewidths=0.25,
            label=profile,
        )
    baseline = valid.loc[valid["is_baseline"].astype(bool)]
    if not baseline.empty:
        base_rate = 100.0 * float(baseline.iloc[0]["shareable_rate"])
        axis.axhline(base_rate, color="#555555", ls="--", lw=0.8)
        axis.scatter(
            baseline["candidate_id"],
            100.0 * baseline["shareable_rate"],
            marker="*",
            s=140,
            facecolor="white",
            edgecolor="black",
            linewidth=0.9,
            zorder=5,
        )
        axis.annotate(
            "Baseline",
            (baseline.iloc[0]["candidate_id"], base_rate),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=7,
        )
    best = valid.loc[valid["candidate_id"] == int(best_candidate_id)]
    if not best.empty:
        axis.scatter(
            best["candidate_id"],
            100.0 * best["shareable_rate"],
            marker="P",
            s=110,
            color="#d73027",
            edgecolor="black",
            linewidth=0.7,
            zorder=6,
        )
        axis.annotate(
            "Best",
            (best.iloc[0]["candidate_id"], 100.0 * best.iloc[0]["shareable_rate"]),
            xytext=(5, -12),
            textcoords="offset points",
            fontsize=7,
            color="#d73027",
        )
    axis.set_xlabel("Candidate ID")
    axis.set_ylabel("DPD-shareable rate (%)")
    axis.set_title("Final Real-B retrieval success across candidates")
    axis.set_xlim(1, int(ranking["candidate_id"].max()))
    axis.set_ylim(
        *_limits(100.0 * valid["shareable_rate"].to_numpy(dtype=float))
    )
    axis.legend(title="Order profile", ncol=5, fontsize=7, title_fontsize=7, loc="lower right")
    return _save(figure, output_path) | {
        "candidate_count": int(ranking.shape[0]),
        "valid_candidate_count": int(valid.shape[0]),
    }


def _finite_pair(values: pd.DataFrame, first: str, second: str) -> tuple[np.ndarray, np.ndarray]:
    first_values = values[first].to_numpy(dtype=float)
    second_values = values[second].to_numpy(dtype=float)
    return first_values, second_values


def plot_baseline_vs_best_real_b(
    statewise: pd.DataFrame, best_candidate_id: int, output_path: Path
) -> dict[str, Any]:
    """Figure 3: Baseline versus Best retrieved Real-B CNMSE."""

    _configure()
    x = statewise["state_id_R"].to_numpy(dtype=float)
    baseline = statewise["baseline_real_B"].to_numpy(dtype=float)
    best = statewise["best_real_B"].to_numpy(dtype=float)
    finite = np.concatenate((baseline[np.isfinite(baseline)], best[np.isfinite(best)]))
    y_min, y_max = _limits(finite, references=(THRESHOLD_DB,))
    strip_y = y_min - max((y_max - y_min) * 0.08, 0.4)
    figure, axis = plt.subplots(figsize=(12.0, 5.0), constrained_layout=True)
    for values, label, color, marker in (
        (baseline, "Baseline", "#2166ac", "o"),
        (best, "Best", "#d73027", "s"),
    ):
        mask = np.isfinite(values)
        axis.plot(
            x[mask],
            values[mask],
            color=color,
            lw=0.9,
            marker=marker,
            ms=2.5,
            markevery=15,
            alpha=0.85,
            label=label,
        )
        axis.scatter(
            x[~mask],
            np.full(np.count_nonzero(~mask), strip_y),
            color=color,
            marker=marker,
            s=16,
            alpha=0.65,
        )
    axis.axhline(THRESHOLD_DB, color="#555555", ls="--", lw=0.8, label="-40 dB threshold")
    axis.set_xlabel("State ID (State_R)")
    axis.set_ylabel("Retrieved Real-B CNMSE (dB)")
    axis.set_title("Baseline versus best retrieval candidate")
    axis.set_xlim(float(x.min()), float(x.max()))
    axis.set_ylim(strip_y - 0.25, y_max)
    axis.text(
        0.01,
        0.02,
        "Exact hits shown in the bottom strip",
        transform=axis.transAxes,
        fontsize=7,
        color="#555555",
    )
    axis.legend(loc="best", fontsize=7)
    details = _save(figure, output_path)
    details.update(
        {
            "state_count": int(statewise.shape[0]),
            "baseline_finite_count": int(np.isfinite(baseline).sum()),
            "best_finite_count": int(np.isfinite(best).sum()),
            "best_candidate_id": int(best_candidate_id),
            "threshold_dB": THRESHOLD_DB,
        }
    )
    return details


def plot_failure_detail(statewise: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    """Figure 4: baseline's original 14 failure states."""

    _configure()
    subset = statewise.loc[~statewise["baseline_shareable"].astype(bool)].sort_values("state_id_R")
    if subset.empty:
        raise ValueError("Baseline没有failure状态")
    x = subset["state_id_R"].to_numpy(dtype=float)
    baseline = subset["baseline_real_B"].to_numpy(dtype=float)
    best = subset["best_real_B"].to_numpy(dtype=float)
    finite = np.concatenate((baseline[np.isfinite(baseline)], best[np.isfinite(best)]))
    y_min, y_max = _limits(finite, references=(THRESHOLD_DB,))
    strip_y = y_min - max((y_max - y_min) * 0.10, 0.4)
    figure, axis = plt.subplots(figsize=(9.0, 4.8), constrained_layout=True)
    for values, label, color, marker in (
        (baseline, "Baseline", "#2166ac", "o"),
        (best, "Best", "#d73027", "s"),
    ):
        mask = np.isfinite(values)
        axis.plot(x[mask], values[mask], color=color, lw=1.0, marker=marker, ms=5, label=label)
        axis.scatter(
            x[~mask], np.full(np.count_nonzero(~mask), strip_y), color=color, marker=marker, s=28
        )
    axis.axhline(THRESHOLD_DB, color="#555555", ls="--", lw=0.8)
    axis.set_xticks(x, [str(int(value)) for value in x])
    axis.tick_params(axis="x", labelrotation=45)
    for label in axis.get_xticklabels():
        label.set_ha("right")
    axis.set_xlabel("Original Baseline Failure state ID")
    axis.set_ylabel("Retrieved Real-B CNMSE (dB)")
    axis.set_title("Baseline failures: rescue versus new failure")
    axis.set_ylim(strip_y - 0.25, y_max)
    axis.legend(loc="best", fontsize=7)
    details = _save(figure, output_path)
    details.update(
        {"baseline_failure_count": int(subset.shape[0]), "state_ids": [int(value) for value in x]}
    )
    return details


def plot_model_metric_comparison(statewise: pd.DataFrame, output_path: Path) -> dict[str, Any]:
    """Figure 5: four model metrics, Baseline versus Best."""

    _configure()
    specs = [
        ("baseline_Aend_train", "best_Aend_train", "Y-Aend train"),
        ("baseline_Aend_B", "best_Aend_B", "Y-Aend -> B"),
        ("baseline_C2_train", "best_C2_train", "Y-C2 train"),
        ("baseline_C2_B", "best_C2_B", "Y-C2 -> B"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 7.0), constrained_layout=True)
    for axis, (baseline_column, best_column, title) in zip(axes.flat, specs, strict=True):
        data = [
            statewise[baseline_column].to_numpy(dtype=float),
            statewise[best_column].to_numpy(dtype=float),
        ]
        box = axis.boxplot(
            data,
            tick_labels=["Baseline", "Best"],
            patch_artist=True,
            widths=0.55,
            showfliers=False,
        )
        box["boxes"][0].set_facecolor("#d9eaf7")
        box["boxes"][1].set_facecolor("#f6c7c7")
        for item in box["medians"]:
            item.set_color("#222222")
            item.set_linewidth(1.1)
        axis.set_title(title)
        axis.set_ylabel("NMSE (dB)")
        axis.set_ylim(*_limits(np.concatenate(data)))
        axis.grid(axis="y", color="#dddddd", lw=0.5, alpha=0.7)
    figure.suptitle("Baseline versus best model metrics", fontsize=10)
    details = _save(figure, output_path)
    details.update({"state_count": int(statewise.shape[0]), "panel_count": 4})
    return details


def plot_metric_change_vs_retrieval_change(
    statewise: pd.DataFrame, output_path: Path
) -> dict[str, Any]:
    """Figure 6: model-metric gain versus finite/finite Real-B gain."""

    _configure()
    finite = np.isfinite(statewise["baseline_real_B"]) & np.isfinite(statewise["best_real_B"])
    work = statewise.loc[finite].copy()
    work["retrieval_gain_dB"] = work["baseline_real_B"] - work["best_real_B"]
    specs = [
        ("baseline_Aend_train", "best_Aend_train", "Δ Y-Aend train"),
        ("baseline_Aend_B", "best_Aend_B", "Δ Y-Aend -> B"),
        ("baseline_C2_train", "best_C2_train", "Δ Y-C2 train"),
        ("baseline_C2_B", "best_C2_B", "Δ Y-C2 -> B"),
    ]
    figure, axes = plt.subplots(2, 2, figsize=(9.2, 7.0), constrained_layout=True)
    colors = {
        "success_to_success": "#4d9221",
        "success_to_failure": "#d73027",
        "failure_to_success": "#2166ac",
        "failure_to_failure": "#969696",
    }
    for axis, (baseline_column, best_column, title) in zip(axes.flat, specs, strict=True):
        gain = work[baseline_column].to_numpy(dtype=float) - work[best_column].to_numpy(dtype=float)
        for transition, color in colors.items():
            mask = work["retrieval_transition"].eq(transition).to_numpy()
            if np.any(mask):
                axis.scatter(
                    gain[mask],
                    work.loc[mask, "retrieval_gain_dB"],
                    s=13,
                    alpha=0.65,
                    color=color,
                    label=transition,
                )
        axis.axhline(0.0, color="#777777", lw=0.6, ls="--")
        axis.axvline(0.0, color="#777777", lw=0.6, ls="--")
        axis.set_xlabel(title + " (dB)")
        axis.set_ylabel("Δ retrieved Real-B (dB)")
        axis.set_title(title)
        axis.set_xlim(*_limits(gain, references=(0.0,)))
        axis.set_ylim(*_limits(work["retrieval_gain_dB"].to_numpy(dtype=float), references=(0.0,)))
    handles = [
        Line2D([0], [0], marker="o", color=color, linestyle="None", markersize=4, label=label)
        for label, color in colors.items()
    ]
    axes[0, 0].legend(handles=handles, fontsize=6, loc="best")
    figure.suptitle("Model-metric change versus retrieval change", fontsize=10)
    details = _save(figure, output_path)
    details.update({"finite_finite_state_count": int(work.shape[0]), "panel_count": 4})
    return details


__all__ = [
    "plot_baseline_vs_best_real_b",
    "plot_candidate_failure_count",
    "plot_candidate_shareable_rate",
    "plot_failure_detail",
    "plot_metric_change_vs_retrieval_change",
    "plot_model_metric_comparison",
]
