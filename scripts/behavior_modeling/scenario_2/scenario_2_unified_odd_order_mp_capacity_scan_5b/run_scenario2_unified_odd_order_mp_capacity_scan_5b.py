"""Run the 5B unified odd-order MP capacity scan in parallel by State.

The numerical scan is intentionally isolated from the retrieval experiments.
It uses ordinary complex least squares, a uniform memory depth per candidate,
the frozen 340/85 split, and a State-level ProcessPoolExecutor with BLAS
threads pinned to one per worker.
"""

# Imports intentionally follow the BLAS-thread environment setup below.
# ruff: noqa: E402,I001

from __future__ import annotations

import hashlib
import itertools
import json
import os
import sys
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

for _name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_name] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from behavior_modeling.shared.odd_order_mp_capacity_scan import (  # noqa: E402
    DEVELOPMENT_COUNT,
    FORMAL_A_LENGTH,
    FORMAL_B_LENGTH,
    FORMAL_C_LENGTH,
    OddOrderCandidate,
    STATE_COUNT,
    THRESHOLD_DB,
    VALIDATION_COUNT,
    aggregate_candidate_summary,
    candidate_grid_rows,
    choose_unified_candidate,
    direct_basis_for_candidate,
    evaluate_shell_parallel,
    evaluate_state_shell_serial,
    generate_shell,
    load_frozen_split,
    odd_orders,
    prepare_canonical_state,
    selected_metrics_for_candidate,
    target_worker_count,
    WAVEFORM_LENGTH,
)

TASK_NAME = "scenario_2_unified_odd_order_mp_capacity_scan_5b"
RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_modeling"
    / "scenario_2" /TASK_NAME
    / "scenario_2_unified_odd_order_mp_capacity_scan_5B"
)
CACHE_ROOT = RESULT_ROOT / "cache"
TABLE_ROOT = RESULT_ROOT / "tables"
FIGURE_ROOT = RESULT_ROOT / "figures"
VALIDATION_ROOT = RESULT_ROOT / "validation"
SPLIT_SOURCE = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_retrieval"
    / "scenario_2" /"scenario_2_C2_to_Aend_unified_model_capacity"
    / "split_definition.csv"
)
MODEL_LOG = PROJECT_ROOT / "work_logs" / "behavior_modeling" / "scenario_2" /TASK_NAME / "execution_log.txt"  # noqa: E501
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"

PROTECTED_RESULT_DIRS = {
    "statewise_adaptive": PROJECT_ROOT
    / "results"
    / TASK_NAME
    / "scenario_2_statewise_adaptive_Aend_C2_5B",
    "statewise_best_round0_4": PROJECT_ROOT
    / "results"
    / TASK_NAME
    / "scenario_2_statewise_best_round0_4_5B",
    "formal_5B_retrieval": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_retrieval"
    / "scenario_2" /"scenario_2_C2_to_Aend",
    "formal_1B_retrieval": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_retrieval"
    / "scenario_2" /"scenario_2_C2_to_Aend_1B",
    "formal_0p5B_retrieval": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_retrieval"
    / "scenario_2" /"scenario_2_C2_to_Aend_0p5B",
    "equal_ABC": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_retrieval"
    / "scenario_2" /"scenario_2_C2_to_Aend_equal_ABC",
    "unified_capacity_old": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_retrieval"
    / "scenario_2" /"scenario_2_C2_to_Aend_unified_model_capacity",
    "retrieval_oriented_old": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_retrieval"
    / "scenario_2" /"scenario_2_C2_to_Aend_retrieval_oriented_model_scan",
    "all_ilc": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2" /"scenario_2_all_ilc",
    "low_bandwidth": PROJECT_ROOT
    / "results"
    / "low_bandwidth_behavior_analysis"
    / "scenario_2" /"sample_rate_operator_bank",
}


def _tree_digest(path: Path) -> str | None:
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    for file in sorted(path.rglob("*"), key=lambda item: str(item).lower()):
        if not file.is_file() or "__pycache__" in file.parts or file.suffix.lower() == ".pyc":
            continue
        content = file.read_bytes()
        digest.update(file.relative_to(path).as_posix().encode() + b"\0")
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def _raw_manifest() -> dict[str, Any]:
    root = PROJECT_ROOT / "data" / "raw"
    digest = hashlib.sha256()
    files = sorted(
        (file for file in root.rglob("*") if file.is_file()), key=lambda item: str(item).lower()
    )
    total = 0
    for file in files:
        size = int(file.stat().st_size)
        digest.update(str(file.relative_to(root)).replace("\\", "/").encode() + b"\0")
        digest.update(size.to_bytes(8, "little"))
        total += size
    return {"sha256": digest.hexdigest(), "file_count": len(files), "bytes": total}


def _snapshot() -> dict[str, Any]:
    return {
        "data/raw": _raw_manifest(),
        "result_dirs": {
            name: {"path": str(path), "exists": path.is_dir(), "sha256": _tree_digest(path)}
            for name, path in PROTECTED_RESULT_DIRS.items()
        },
    }


def _verify_protection(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    raw = {
        "sha256_unchanged": before["data/raw"]["sha256"] == after["data/raw"]["sha256"],
        "file_count_unchanged": before["data/raw"]["file_count"] == after["data/raw"]["file_count"],
        "bytes_unchanged": before["data/raw"]["bytes"] == after["data/raw"]["bytes"],
    }
    result_dirs = {
        name: {
            "sha256_unchanged": item["sha256"] == after["result_dirs"][name]["sha256"],
            "before": item["sha256"],
            "after": after["result_dirs"][name]["sha256"],
        }
        for name, item in before["result_dirs"].items()
    }
    return {
        "data/raw": raw,
        "result_dirs": result_dirs,
        "all_protected_unchanged": bool(
            all(raw.values()) and all(item["sha256_unchanged"] for item in result_dirs.values())
        ),
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_frame(frame: pd.DataFrame, path: Path, *, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        na_rep="NaN",
        float_format="%.17g",
        compression=compression,
    )


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _state_chunk_path(round_index: int, state_id: int) -> Path:
    return CACHE_ROOT / f"round_{round_index:03d}" / "worker_chunks" / f"state_{state_id:03d}.json"


def _load_state_chunks(round_index: int) -> tuple[list[dict[str, Any]], set[int]]:
    directory = CACHE_ROOT / f"round_{round_index:03d}" / "worker_chunks"
    rows: list[dict[str, Any]] = []
    completed: set[int] = set()
    if not directory.is_dir():
        return rows, completed
    for path in sorted(directory.glob("state_*.json"), key=lambda item: item.name):
        payload = json.loads(path.read_text(encoding="utf-8"))
        state_id = int(payload["state_id"])
        if state_id in completed:
            raise RuntimeError(f"duplicate state chunk {state_id}")
        completed.add(state_id)
        rows.extend(payload["rows"])
    return rows, completed


def _save_state_chunk(round_index: int, payload: dict[str, Any]) -> None:
    path = _state_chunk_path(round_index, int(payload["state_id"]))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            _json_safe({"state_id": payload["state_id"], "rows": payload["rows"]}),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _parallel_serial_regression() -> dict[str, Any]:
    candidates = tuple(
        OddOrderCandidate(
            candidate_id=index,
            P=P,
            orders=odd_orders(P),
            M=M,
            max_delay=M - 1,
            coefficient_count=len(odd_orders(P)) * M,
            search_round=0,
            source="parallel_serial_regression",
        )
        for index, (P, M) in enumerate(((3, 1), (5, 2), (9, 3), (15, 5)), start=1)
    )
    states = (0, 1, 100, 200, 424)
    split = pd.DataFrame({"state_id": states, "split": ["Development"] * len(states)})
    serial = []
    serial_theta_a: dict[tuple[int, int], np.ndarray] = {}
    serial_theta_c: dict[tuple[int, int], np.ndarray] = {}
    for state_id in states:
        result = evaluate_state_shell_serial(state_id, candidates, "Development")
        serial.extend(result["rows"])
        serial_theta_a.update(result["theta_a"])
        serial_theta_c.update(result["theta_c"])
    parallel = evaluate_shell_parallel(candidates, split, worker_count=2)
    serial_frame = (
        pd.DataFrame(serial)
        .sort_values(["state_id", "side", "candidate_id"])
        .reset_index(drop=True)
    )
    parallel_frame = (
        pd.DataFrame(parallel.rows)
        .sort_values(["state_id", "side", "candidate_id"])
        .reset_index(drop=True)
    )
    metric_cols = ["train_NMSE_dB", "B_NMSE_dB"]
    if serial_frame.shape != parallel_frame.shape:
        return {
            "pass": False,
            "reason": "row_shape_mismatch",
            "serial_shape": list(serial_frame.shape),
            "parallel_shape": list(parallel_frame.shape),
        }
    metric_error = float(
        np.max(
            np.abs(
                serial_frame[metric_cols].to_numpy(dtype=float)
                - parallel_frame[metric_cols].to_numpy(dtype=float)
            )
        )
    )
    theta_error = 0.0
    for key, value in serial_theta_a.items():
        theta_error = max(theta_error, float(np.max(np.abs(value - parallel.theta_a[key]))))
    for key, value in serial_theta_c.items():
        theta_error = max(theta_error, float(np.max(np.abs(value - parallel.theta_c[key]))))
    return {
        "pass": bool(metric_error <= 1e-10 and theta_error <= 1e-10),
        "state_ids": list(states),
        "candidate_specs": [[3, 1], [5, 2], [9, 3], [15, 5]],
        "metric_max_abs_error_dB": metric_error,
        "theta_max_abs_error": theta_error,
        "serial_rows": int(serial_frame.shape[0]),
        "parallel_rows": int(parallel_frame.shape[0]),
        "valid_support_rule": "candidate-specific M-1 rows",
    }


def _run_shell(
    round_index: int,
    candidates: Sequence[OddOrderCandidate],
    split_frame: pd.DataFrame,
    logical_cpu_count: int,
    progress: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cached_rows, completed = _load_state_chunks(round_index)
    started = time.perf_counter()

    def on_state(result: dict[str, Any], done: int, elapsed: float) -> None:
        _save_state_chunk(round_index, result)
        progress.update(
            {
                "active_round": round_index,
                "completed_states": done,
                "target_states": STATE_COUNT,
                "completed_candidates": int(len(candidates)),
                "candidate_count": int(len(candidates)),
                "logical_cpu_count": logical_cpu_count,
                "cpu_target_fraction": 0.90,
                "parallel_worker_count": target_worker_count(logical_cpu_count),
                "blas_threads_per_worker": 1,
                "nested_parallelism": False,
                "elapsed_seconds": elapsed,
                "states_per_minute": float(done / elapsed * 60.0) if elapsed > 0 else None,
            }
        )
        _write_json(RESULT_ROOT / "search_progress.json", progress)

    result = evaluate_shell_parallel(
        candidates,
        split_frame,
        worker_count=target_worker_count(logical_cpu_count),
        completed_state_ids=completed,
        state_result_callback=on_state,
    )
    rows = cached_rows + list(result.rows)
    frame = pd.DataFrame(rows)
    if frame.empty or frame["state_id"].nunique() != STATE_COUNT:
        raise RuntimeError(f"round {round_index} did not cover all 425 states")
    frame = frame.sort_values(["candidate_id", "state_id", "side"]).reset_index(drop=True)
    expected_rows = STATE_COUNT * len(candidates) * 2
    if frame.shape[0] != expected_rows:
        raise RuntimeError(
            f"round {round_index} row count mismatch {frame.shape[0]} vs {expected_rows}"
        )
    elapsed = time.perf_counter() - started
    run_info = {
        "round": round_index,
        "candidate_count": len(candidates),
        "completed_state_count": STATE_COUNT,
        "row_count": int(frame.shape[0]),
        "elapsed_seconds": float(elapsed),
        "states_per_minute": float(STATE_COUNT / elapsed * 60.0) if elapsed > 0 else None,
        "worker_count": result.worker_count,
        "cached_state_count_before_run": len(completed),
    }
    _write_frame(
        frame, CACHE_ROOT / f"round_{round_index:03d}" / "state_metrics.csv.gz", compression="gzip"
    )
    return frame, run_info


def _best_candidate_summary(summary: pd.DataFrame) -> dict[str, Any]:
    chosen = choose_unified_candidate(summary)
    return {
        "candidate_id": int(chosen["candidate_id"]),
        "P": int(chosen["P"]),
        "M": int(chosen["M"]),
        "coefficient_count": int(chosen["coefficient_count"]),
        "dev_all_four_pass": int(chosen["dev_all_four_pass"]),
        "dev_C2_joint_pass": int(chosen["dev_C2_joint_pass"]),
        "dev_Aend_joint_pass": int(chosen["dev_Aend_joint_pass"]),
        "dev_worst4_median": float(chosen["dev_worst4_median"]),
        "dev_C2_B_median": float(chosen["dev_C2_B_median"]),
    }


def _selected_state_metrics(
    state_metrics: pd.DataFrame,
    candidate_id: int,
    split_frame: pd.DataFrame,
) -> pd.DataFrame:
    selected = state_metrics.loc[state_metrics["candidate_id"] == candidate_id].copy()
    selected = selected.merge(
        split_frame.loc[:, ["state_id", "split"]], on="state_id", how="left", validate="many_to_one"
    )
    a = selected.loc[
        selected["side"] == "Aend", ["state_id", "train_NMSE_dB", "B_NMSE_dB", "split"]
    ].rename(columns={"train_NMSE_dB": "Y_Aend_train_NMSE_dB", "B_NMSE_dB": "Y_Aend_B_NMSE_dB"})
    c = selected.loc[selected["side"] == "C2", ["state_id", "train_NMSE_dB", "B_NMSE_dB"]].rename(
        columns={"train_NMSE_dB": "Y_C2_train_NMSE_dB", "B_NMSE_dB": "Y_C2_B_NMSE_dB"}
    )
    result = a.merge(c, on="state_id", how="inner", validate="one_to_one")
    result["Aend_joint_lt_minus40"] = (result["Y_Aend_train_NMSE_dB"] < THRESHOLD_DB) & (
        result["Y_Aend_B_NMSE_dB"] < THRESHOLD_DB
    )
    result["C2_joint_lt_minus40"] = (result["Y_C2_train_NMSE_dB"] < THRESHOLD_DB) & (
        result["Y_C2_B_NMSE_dB"] < THRESHOLD_DB
    )
    result["all_four_lt_minus40"] = result["Aend_joint_lt_minus40"] & result["C2_joint_lt_minus40"]
    return result.sort_values("state_id").reset_index(drop=True)


def _refit_selected(
    selected: OddOrderCandidate,
    split_frame: pd.DataFrame,
    state_metrics: pd.DataFrame,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], pd.DataFrame, dict[str, Any]]:
    theta_a: dict[str, np.ndarray] = {}
    theta_c: dict[str, np.ndarray] = {}
    rows: list[dict[str, Any]] = []
    cached = state_metrics.loc[state_metrics["candidate_id"] == selected.candidate_id].set_index(
        ["state_id", "side"]
    )
    for state_id in range(STATE_COUNT):
        prepared = prepare_canonical_state(state_id)
        a_phi, a_y = direct_basis_for_candidate(prepared.a_end["A"], selected)
        ab_phi, ab_y = direct_basis_for_candidate(prepared.a_end["B"], selected)
        c_phi, c_y = direct_basis_for_candidate(prepared.c2["C"], selected)
        cb_phi, cb_y = direct_basis_for_candidate(prepared.c2["B"], selected)
        a_theta, _, a_rank, a_singular = np.linalg.lstsq(a_phi, a_y, rcond=None)
        c_theta, _, c_rank, c_singular = np.linalg.lstsq(c_phi, c_y, rcond=None)
        a_train = calculate_nmse(a_y, a_phi @ a_theta)
        a_b = calculate_nmse(ab_y, ab_phi @ a_theta)
        c_train = calculate_nmse(c_y, c_phi @ c_theta)
        c_b = calculate_nmse(cb_y, cb_phi @ c_theta)
        observed = {
            "Aend": (
                float(a_train),
                float(a_b),
                int(a_rank),
                float(a_singular[0] / a_singular[-1]),
            ),
            "C2": (float(c_train), float(c_b), int(c_rank), float(c_singular[0] / c_singular[-1])),
        }
        for side, values in observed.items():
            train_value, b_value, rank, condition = values
            cached_row = cached.loc[(state_id, side)]
            diff_train = abs(train_value - float(cached_row["train_NMSE_dB"]))
            diff_b = abs(b_value - float(cached_row["B_NMSE_dB"]))
            # CSV round-tripping and BLAS reduction order can introduce tiny
            # sub-nanodB differences on refit; retain a strict numerical check
            # while allowing the observed 1e-9 dB serialization noise.
            if diff_train > 1e-8 or diff_b > 1e-8:
                raise RuntimeError(
                    f"selected candidate regression mismatch state={state_id} "
                    f"side={side} diff={diff_train}/{diff_b}"
                )
            rows.append(
                {
                    "state_id": state_id,
                    "side": side,
                    "train_NMSE_dB": train_value,
                    "B_NMSE_dB": b_value,
                    "rank": rank,
                    "condition_number": condition,
                    "train_abs_diff_dB": diff_train,
                    "B_abs_diff_dB": diff_b,
                    "fit_rows": int((a_phi if side == "Aend" else c_phi).shape[0]),
                }
            )
        theta_a[f"state_{state_id:03d}"] = np.asarray(a_theta, dtype=np.complex128)
        theta_c[f"state_{state_id:03d}"] = np.asarray(c_theta, dtype=np.complex128)
        if (state_id + 1) % 50 == 0 or state_id == STATE_COUNT - 1:
            print(f"selected unified refit: processed {state_id + 1}/{STATE_COUNT}", flush=True)
    frame = pd.DataFrame(rows).sort_values(["side", "state_id"]).reset_index(drop=True)
    return (
        theta_a,
        theta_c,
        frame,
        {
            "all_850_refit": bool(frame.shape[0] == 2 * STATE_COUNT),
            "max_train_abs_diff_dB": float(frame["train_abs_diff_dB"].max()),
            "max_B_abs_diff_dB": float(frame["B_abs_diff_dB"].max()),
        },
    )


def calculate_nmse(y_ref: np.ndarray, y_pred: np.ndarray) -> float:
    from behavior_modeling.shared.evaluation import calculate_nmse as _calculate_nmse

    return _calculate_nmse(y_ref, y_pred)


def _save_figure(fig: plt.Figure, stem: Path) -> dict[str, str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs = {
        "png": stem.with_suffix(".png"),
        "svg": stem.with_suffix(".svg"),
        "pdf": stem.with_suffix(".pdf"),
    }
    fig.savefig(outputs["png"], dpi=300, bbox_inches="tight")
    fig.savefig(outputs["svg"], bbox_inches="tight")
    fig.savefig(outputs["pdf"], bbox_inches="tight")
    plt.close(fig)
    return {key: str(path) for key, path in outputs.items()}


def _heatmap(summary: pd.DataFrame, value: str, title: str, stem: Path, fmt: str) -> dict[str, str]:
    piv = summary.pivot(index="P", columns="M", values=value).sort_index().sort_index(axis=1)
    fig, ax = plt.subplots(figsize=(10, 7))
    image = ax.imshow(piv.to_numpy(dtype=float), aspect="auto", cmap="viridis")
    ax.set_xticks(np.arange(piv.shape[1]), labels=[str(value) for value in piv.columns])
    ax.set_yticks(np.arange(piv.shape[0]), labels=[str(value) for value in piv.index])
    ax.set_xlabel("Uniform memory depth M")
    ax.set_ylabel("Maximum odd nonlinear order P")
    ax.set_title(title)
    for row_index in range(piv.shape[0]):
        for col_index in range(piv.shape[1]):
            value_here = piv.iloc[row_index, col_index]
            if np.isfinite(value_here):
                ax.text(
                    col_index,
                    row_index,
                    format(value_here, fmt),
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white",
                )
    fig.colorbar(image, ax=ax, shrink=0.85)
    fig.tight_layout()
    return _save_figure(fig, stem)


def _plot_complexity(summary: pd.DataFrame, stem: Path) -> dict[str, str]:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    axes[0].scatter(
        summary["coefficient_count"],
        summary["dev_Aend_B_median"],
        c=summary["dev_all_four_pass"],
        cmap="viridis",
        s=25,
    )
    axes[0].set_xlabel("Complex coefficients K")
    axes[0].set_ylabel("Development A→B median (dB)")
    axes[1].scatter(
        summary["coefficient_count"],
        summary["dev_C2_B_median"],
        c=summary["dev_C2_joint_pass"],
        cmap="plasma",
        s=25,
    )
    axes[1].set_xlabel("Complex coefficients K")
    axes[1].set_ylabel("Development C2→B median (dB)")
    axes[2].scatter(
        summary["coefficient_count"],
        summary["dev_all_four_pass"],
        c=summary["M"],
        cmap="cividis",
        s=25,
    )
    axes[2].set_xlabel("Complex coefficients K")
    axes[2].set_ylabel("Development all-four pass count")
    fig.suptitle("Unified Odd-Order MP Complexity and Generalization")
    fig.tight_layout()
    return _save_figure(fig, stem)


def _plot_fixed_views(
    summary: pd.DataFrame, stem_p: Path, stem_m: Path
) -> dict[str, dict[str, str]]:
    figures: dict[str, dict[str, str]] = {}
    fig_p, ax_p = plt.subplots(figsize=(11, 6))
    for P in sorted(set(summary["P"]).intersection({5, 9, 13, 17, 21})):
        group = summary.loc[summary["P"] == P].sort_values("M")
        ax_p.plot(group["M"], group["dev_C2_B_median"], marker="o", label=f"P={P}")
    ax_p.set_xlabel("Uniform memory depth M")
    ax_p.set_ylabel("Development C2→B median NMSE (dB)")
    ax_p.set_title("Fixed P: Memory Depth vs C2→B Generalization")
    ax_p.legend(frameon=False, ncol=2)
    figures["fixed_P"] = _save_figure(fig_p, stem_p)

    fig_m, ax_m = plt.subplots(figsize=(11, 6))
    for M in sorted(set(summary["M"]).intersection({1, 2, 3, 5, 10})):
        group = summary.loc[summary["M"] == M].sort_values("P")
        ax_m.plot(group["P"], group["dev_C2_B_median"], marker="s", label=f"M={M}")
    ax_m.set_xlabel("Maximum odd nonlinear order P")
    ax_m.set_ylabel("Development C2→B median NMSE (dB)")
    ax_m.set_title("Fixed M: Nonlinear Order vs C2→B Generalization")
    ax_m.legend(frameon=False, ncol=2)
    figures["fixed_M"] = _save_figure(fig_m, stem_m)
    return figures


def _plot_selected_statewise(selected: pd.DataFrame, stem: Path) -> dict[str, str]:
    x = selected["state_id"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(20, 8))
    series = (
        ("Y_Aend_train_NMSE_dB", "Aend train", "#4C78A8", "o"),
        ("Y_Aend_B_NMSE_dB", "Aend → B", "#F58518", "s"),
        ("Y_C2_train_NMSE_dB", "C2 train", "#54A24B", "^"),
        ("Y_C2_B_NMSE_dB", "C2 → B", "#E45756", "D"),
    )
    for column, label, color, marker in series:
        ax.plot(
            x,
            selected[column],
            color=color,
            label=label,
            linewidth=0.9,
            marker=marker,
            markersize=2.2,
            markevery=12,
        )
    ax.axhline(
        THRESHOLD_DB, color="#666666", linestyle=":", linewidth=1.0, label="-40 dB threshold"
    )
    ax.set_xlabel("State ID")
    ax.set_ylabel("NMSE (dB)")
    ax.set_title("Selected Unified Odd-Order MP Statewise Metrics")
    ax.legend(frameon=False, ncol=3)
    ax.grid(False)
    fig.tight_layout()
    return _save_figure(fig, stem)


def _development_validation_summary(
    selected: pd.DataFrame, split_frame: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_name in ("Development", "Validation"):
        group = selected.loc[selected["split"] == split_name]
        row: dict[str, Any] = {"split": split_name, "state_count": int(group.shape[0])}
        for column in (
            "Y_Aend_train_NMSE_dB",
            "Y_Aend_B_NMSE_dB",
            "Y_C2_train_NMSE_dB",
            "Y_C2_B_NMSE_dB",
        ):
            values = group[column].to_numpy(dtype=float)
            key = column.replace("Y_", "").replace("_NMSE_dB", "")
            row[f"{key}_mean"] = float(np.mean(values))
            row[f"{key}_median"] = float(np.median(values))
            row[f"{key}_std"] = float(np.std(values))
            row[f"{key}_q05"] = float(np.quantile(values, 0.05))
            row[f"{key}_q95"] = float(np.quantile(values, 0.95))
            row[f"{key}_worst"] = float(np.max(values))
        row["Aend_joint_pass"] = int(group["Aend_joint_lt_minus40"].sum())
        row["C2_joint_pass"] = int(group["C2_joint_lt_minus40"].sum())
        row["all_four_pass"] = int(group["all_four_lt_minus40"].sum())
        rows.append(row)
    return pd.DataFrame(rows)


def _build_excel_source(
    summary: pd.DataFrame,
    selected: pd.DataFrame,
    dev_val_summary: pd.DataFrame,
    history: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "sheets": {
            "candidate_summary": {
                "columns": summary.columns.tolist(),
                "rows": summary.where(pd.notna(summary), None).to_dict("records"),
            },
            "selected_model_state_metrics": {
                "columns": selected.columns.tolist(),
                "rows": selected.where(pd.notna(selected), None).to_dict("records"),
            },
            "development_summary": {
                "columns": dev_val_summary.columns.tolist(),
                "rows": dev_val_summary.loc[dev_val_summary["split"] == "Development"]
                .where(pd.notna(dev_val_summary), None)
                .to_dict("records"),
            },
            "validation_summary": {
                "columns": dev_val_summary.columns.tolist(),
                "rows": dev_val_summary.loc[dev_val_summary["split"] == "Validation"]
                .where(pd.notna(dev_val_summary), None)
                .to_dict("records"),
            },
            "search_history": {
                "columns": [
                    "round",
                    "candidate_count",
                    "elapsed_seconds",
                    "states_per_minute",
                    "best_dev_candidate",
                    "capacity_saturation_detected",
                ],
                "rows": [
                    [
                        item.get("round"),
                        item.get("candidate_count"),
                        item.get("elapsed_seconds"),
                        item.get("states_per_minute"),
                        json.dumps(item.get("best_dev_candidate"), ensure_ascii=False),
                        item.get("capacity_saturation_detected"),
                    ]
                    for item in history
                ],
            },
        },
        "metadata": {
            "experiment": "scenario_2_unified_odd_order_mp_capacity_scan_5B",
            "ridge_used": False,
            "even_orders_used": False,
            "order_2_used": False,
            "development_used_for_selection": True,
            "validation_used_for_selection": False,
        },
    }


def main() -> None:
    print("Scenario 2 unified odd-order MP capacity scan started.", flush=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    before = _snapshot()
    split_frame = load_frozen_split(SPLIT_SOURCE)
    logical_cpu_count = int(os.cpu_count() or 1)
    workers = target_worker_count(logical_cpu_count)
    regression = _parallel_serial_regression()
    _write_json(VALIDATION_ROOT / "parallel_serial_regression.json", regression)
    if not regression["pass"]:
        raise RuntimeError(f"serial/parallel regression failed: {regression}")
    _write_json(
        RESULT_ROOT / "config" / "odd_order_mp_config.json",
        {
            "orders_rule": "all_consecutive_odd_orders_from_1_to_P",
            "uniform_memory_depth": True,
            "max_delay_rule": "M-1",
            "coefficient_count_rule": "((P+1)/2)*M",
            "ridge_used": False,
            "lambda": None,
            "5B_bandwidth": True,
            "ABC": {
                "A": [0, FORMAL_A_LENGTH],
                "B": [FORMAL_A_LENGTH, FORMAL_A_LENGTH + FORMAL_B_LENGTH],
                "C": [FORMAL_A_LENGTH + FORMAL_B_LENGTH, WAVEFORM_LENGTH],
            },
        },
    )
    progress: dict[str, Any] = {
        "experiment": "scenario_2_unified_odd_order_mp_capacity_scan_5B",
        "active_round": 0,
        "completed_states": 0,
        "target_states": STATE_COUNT,
        "completed_candidates": 0,
        "candidate_count": 0,
        "logical_cpu_count": logical_cpu_count,
        "cpu_target_fraction": 0.90,
        "parallel_worker_count": workers,
        "blas_threads_per_worker": 1,
        "nested_parallelism": False,
        "psutil_available": False,
        "parallel_serial_regression_pass": True,
        "search_stop_reason": None,
    }
    _write_json(RESULT_ROOT / "search_progress.json", progress)
    all_candidates: list[OddOrderCandidate] = []
    cumulative_frames: list[pd.DataFrame] = []
    history: list[dict[str, Any]] = []
    saturation_streak = 0
    confirmation_pending = False
    previous_best: dict[str, Any] | None = None
    selected_summary: pd.DataFrame | None = None
    for round_index in itertools.count(0):
        shell = generate_shell(all_candidates, round_index)
        usable_shell = [
            candidate
            for candidate in shell
            if candidate.coefficient_count < min(FORMAL_A_LENGTH, FORMAL_B_LENGTH, FORMAL_C_LENGTH)
        ]
        if not usable_shell:
            progress["search_stop_reason"] = "data_support_blocker_no_usable_candidate_shell"
            break
        all_candidates.extend(usable_shell)
        frame, run_info = _run_shell(
            round_index, usable_shell, split_frame, logical_cpu_count, progress
        )
        cumulative_frames.append(frame)
        cumulative = pd.concat(cumulative_frames, ignore_index=True)
        summary = aggregate_candidate_summary(cumulative, all_candidates, split_frame)
        _write_frame(candidate_grid_rows(all_candidates), TABLE_ROOT / "candidate_grid.csv")
        _write_frame(cumulative, TABLE_ROOT / "candidate_state_metrics.csv.gz", compression="gzip")
        _write_frame(summary, TABLE_ROOT / "candidate_summary.csv")
        best = _best_candidate_summary(summary)
        current_best = choose_unified_candidate(summary)
        current_key = {
            "candidate_id": int(current_best["candidate_id"]),
            "dev_all_four_pass": int(current_best["dev_all_four_pass"]),
            "dev_C2_joint_pass": int(current_best["dev_C2_joint_pass"]),
            "dev_Aend_joint_pass": int(current_best["dev_Aend_joint_pass"]),
            "dev_worst4_median": float(current_best["dev_worst4_median"]),
            "dev_C2_B_median": float(current_best["dev_C2_B_median"]),
        }
        if previous_best is not None:
            no_pass_gain = (
                current_key["dev_all_four_pass"] == previous_best["dev_all_four_pass"]
                and current_key["dev_C2_joint_pass"] == previous_best["dev_C2_joint_pass"]
            )
            small_generalization_gain = all(
                current_key[field] - previous_best[field] < 0.1
                for field in ("dev_worst4_median", "dev_C2_B_median")
            )
            saturation_streak = (
                saturation_streak + 1 if no_pass_gain and small_generalization_gain else 0
            )
        previous_best = current_key
        if int(current_best["dev_all_four_pass"]) == DEVELOPMENT_COUNT:
            if confirmation_pending:
                progress["search_stop_reason"] = (
                    "development_340_all_four_pass_with_confirmation_shell"
                )
                selected_summary = summary
                history.append(
                    {**run_info, "best_dev_candidate": best, "capacity_saturation_detected": False}
                )
                break
            confirmation_pending = True
        elif not confirmation_pending and saturation_streak >= 3:
            progress["search_stop_reason"] = "capacity_saturation_detected_after_three_shells"
            selected_summary = summary
            history.append(
                {**run_info, "best_dev_candidate": best, "capacity_saturation_detected": True}
            )
            break
        history.append(
            {**run_info, "best_dev_candidate": best, "capacity_saturation_detected": False}
        )
        progress.update(
            {
                "candidate_count": len(all_candidates),
                "completed_candidates": len(all_candidates),
                "active_round": round_index,
            }
        )
        _write_json(RESULT_ROOT / "search_history.json", history)
        _write_json(RESULT_ROOT / "search_progress.json", {**progress, "history": history})
        print(
            f"Round {round_index} complete: candidates={len(all_candidates)}, "
            f"best dev all-four={best['dev_all_four_pass']}, C2 joint={best['dev_C2_joint_pass']}, "
            f"elapsed={run_info['elapsed_seconds']:.1f}s.",
            flush=True,
        )
    if selected_summary is None:
        selected_summary = summary
    selected_payload = choose_unified_candidate(selected_summary)
    selected_candidate = next(
        candidate
        for candidate in all_candidates
        if candidate.candidate_id == int(selected_payload["candidate_id"])
    )
    selected_state_metrics = selected_metrics_for_candidate(
        cumulative, selected_candidate.candidate_id, split_frame
    )
    selected_state_metrics = selected_state_metrics.merge(
        split_frame.loc[:, ["state_id", "split"]],
        on="state_id",
        how="left",
        suffixes=("", "_split"),
        validate="one_to_one",
    )
    selected_state_metrics = selected_state_metrics.drop(columns=["split_split"], errors="ignore")
    _write_frame(selected_state_metrics, TABLE_ROOT / "selected_model_state_metrics.csv")
    dev_val_summary = _development_validation_summary(selected_state_metrics, split_frame)
    _write_frame(dev_val_summary, TABLE_ROOT / "development_validation_summary.csv")
    theta_a, theta_c, refit_diagnostics, refit_validation = _refit_selected(
        selected_candidate, split_frame, cumulative
    )
    _write_npz(RESULT_ROOT / "selected_unified_Aend_coefficients.npz", theta_a)
    _write_npz(RESULT_ROOT / "selected_unified_C2_coefficients.npz", theta_c)
    _write_frame(refit_diagnostics, TABLE_ROOT / "selected_refit_diagnostics.csv")
    selected_model_payload = {
        "P": selected_candidate.P,
        "orders": list(selected_candidate.orders),
        "M": selected_candidate.M,
        "max_delay": selected_candidate.max_delay,
        "coefficient_count": selected_candidate.coefficient_count,
        "ridge_used": False,
        "lambda": None,
        "selection_based_on": "Development",
        "development_count": DEVELOPMENT_COUNT,
        "validation_count": VALIDATION_COUNT,
        "candidate_id": selected_candidate.candidate_id,
        "selection_rule": selected_payload["selection_rule"],
        "search_stop_reason": progress["search_stop_reason"],
        "selected_candidate_summary": _json_safe(selected_payload),
    }
    _write_json(RESULT_ROOT / "selected_unified_odd_order_mp_model.json", selected_model_payload)
    # Diagnostic candidate metrics and matrices are kept separate from the
    # final selected state table so model selection remains auditable.
    matrix_candidates = sorted(all_candidates, key=lambda candidate: candidate.candidate_id)
    matrix_by_id = {
        candidate.candidate_id: index for index, candidate in enumerate(matrix_candidates)
    }
    matrices = {
        name: np.full((len(matrix_candidates), STATE_COUNT), np.nan, dtype=np.float64)
        for name in ("A_train", "A_B", "C_train", "C_B")
    }
    for row in cumulative.loc[cumulative["valid"].astype(bool)].to_dict("records"):
        index = matrix_by_id[int(row["candidate_id"])]
        state_id = int(row["state_id"])
        if row["side"] == "Aend":
            matrices["A_train"][index, state_id] = float(row["train_NMSE_dB"])
            matrices["A_B"][index, state_id] = float(row["B_NMSE_dB"])
        else:
            matrices["C_train"][index, state_id] = float(row["train_NMSE_dB"])
            matrices["C_B"][index, state_id] = float(row["B_NMSE_dB"])
    _write_npz(
        TABLE_ROOT / "candidate_metric_matrices.npz",
        {
            "candidate_ids": np.asarray(
                [candidate.candidate_id for candidate in matrix_candidates], dtype=np.int64
            ),
            "state_ids": np.arange(STATE_COUNT, dtype=np.int64),
            **matrices,
        },
    )
    figure_details = {
        "heatmap_dev_Aend_train": _heatmap(
            selected_summary,
            "dev_Aend_train_median",
            "Development Aend Train NMSE",
            FIGURE_ROOT / "heatmap_dev_Aend_train_median_vs_P_M",
            ".2f",
        ),
        "heatmap_dev_Aend_B": _heatmap(
            selected_summary,
            "dev_Aend_B_median",
            "Development Aend → B NMSE",
            FIGURE_ROOT / "heatmap_dev_Aend_B_median_vs_P_M",
            ".2f",
        ),
        "heatmap_dev_C2_train": _heatmap(
            selected_summary,
            "dev_C2_train_median",
            "Development C2 Train NMSE",
            FIGURE_ROOT / "heatmap_dev_C2_train_median_vs_P_M",
            ".2f",
        ),
        "heatmap_dev_C2_B": _heatmap(
            selected_summary,
            "dev_C2_B_median",
            "Development C2 → B NMSE",
            FIGURE_ROOT / "heatmap_dev_C2_B_median_vs_P_M",
            ".2f",
        ),
        "heatmap_dev_Aend_joint": _heatmap(
            selected_summary,
            "dev_Aend_joint_pass",
            "Development Aend Joint Pass Count",
            FIGURE_ROOT / "heatmap_dev_Aend_joint_pass_count",
            "d",
        ),
        "heatmap_dev_C2_joint": _heatmap(
            selected_summary,
            "dev_C2_joint_pass",
            "Development C2 Joint Pass Count",
            FIGURE_ROOT / "heatmap_dev_C2_joint_pass_count",
            "d",
        ),
        "heatmap_dev_all_four": _heatmap(
            selected_summary,
            "dev_all_four_pass",
            "Development All-Four Pass Count",
            FIGURE_ROOT / "heatmap_dev_all_four_pass_count",
            "d",
        ),
        "complexity_vs_generalization": _plot_complexity(
            selected_summary, FIGURE_ROOT / "figure_model_complexity_vs_generalization"
        ),
        **_plot_fixed_views(
            selected_summary,
            FIGURE_ROOT / "figure_fixed_P_memory_vs_generalization",
            FIGURE_ROOT / "figure_fixed_M_order_vs_generalization",
        ),
        "selected_statewise": _plot_selected_statewise(
            selected_state_metrics, FIGURE_ROOT / "figure_selected_model_statewise_metrics"
        ),
    }
    _write_json(RESULT_ROOT / "search_history.json", history)
    _write_json(
        RESULT_ROOT / "excel_source.json",
        _build_excel_source(selected_summary, selected_state_metrics, dev_val_summary, history),
    )
    final_validation = {
        "experiment": "scenario_2_unified_odd_order_mp_capacity_scan_5B",
        "bandwidth": "5B",
        "ridge_used": False,
        "even_orders_used": False,
        "order_2_used": False,
        "orders_rule": "all_consecutive_odd_orders_from_1_to_P",
        "uniform_memory_depth": True,
        "max_delay_rule": "M-1",
        "coefficient_count_rule": "((P+1)/2)*M",
        "Aend_and_C2_same_MP_structure": True,
        "all_states_same_MP_structure": True,
        "coefficients_shared": False,
        "low_bandwidth_operator_used": False,
        "retrieval_used": False,
        "real_B_used": False,
        "state_count": STATE_COUNT,
        "development_count": DEVELOPMENT_COUNT,
        "validation_count": VALIDATION_COUNT,
        "development_used_for_selection": True,
        "validation_used_for_selection": False,
        "retrieval_metrics_used_for_selection": False,
        "real_B_used_for_selection": False,
        "failure_labels_used_for_selection": False,
        "parallel_execution": True,
        "parallelization_axis": "state",
        "cpu_target_fraction": 0.90,
        "logical_cpu_count": logical_cpu_count,
        "worker_count_requested": workers,
        "blas_threads_per_worker": 1,
        "nested_parallelism": False,
        "parallel_serial_regression_pass": True,
        "parallel_serial_regression": regression,
        "selected_model": selected_model_payload,
        "search_stop_reason": progress["search_stop_reason"],
        "candidate_count": len(all_candidates),
        "valid_candidate_count": int(selected_summary["valid"].sum()),
        "search_history": history,
        "selected_refit_validation": refit_validation,
        "selected_development_validation_summary": dev_val_summary.to_dict("records"),
        "figure_backend": "Python/matplotlib",
        "figures": figure_details,
        "raw_data_modified": False,
    }
    after = _snapshot()
    protection = _verify_protection(before, after)
    if not protection["all_protected_unchanged"]:
        raise RuntimeError("raw or protected results changed")
    final_validation["protection_before"] = before
    final_validation["protection_after"] = after
    final_validation["protection_verification"] = protection
    _write_json(VALIDATION_ROOT / "validation.json", final_validation)
    _write_json(RESULT_ROOT / "validation.json", final_validation)
    _write_json(
        RESULT_ROOT / "search_progress.json",
        {
            **progress,
            "candidate_count": len(all_candidates),
            "completed_candidates": len(all_candidates),
            "final": True,
        },
    )
    log_body = "\n".join(
        [
            "完成 5B unified odd-order MP capacity scan；只使用普通complex "
            "least-squares，未调用Ridge、低带宽或retrieval。",
            f"parallel state workers={workers}/{logical_cpu_count}（target=0.90），"
            f"BLAS threads/worker=1；serial-parallel regression={regression['pass']}。",
            f"candidate_count={len(all_candidates)}；selected=(P{selected_candidate.P},"
            f"M{selected_candidate.M})，K={selected_candidate.coefficient_count}；"
            f"stop={progress['search_stop_reason']}。",
            f"Development/Validation={DEVELOPMENT_COUNT}/{VALIDATION_COUNT}；"
            f"selected refit={refit_validation}。",
            "raw/protected unchanged="
            f"{protection['all_protected_unchanged']}；结果目录={RESULT_ROOT}。",
        ]
    )
    for path, title in (
        (MODEL_LOG, "Scenario 2 unified odd-order MP capacity scan"),
        (HANDOFF_LOG, "Scenario 2 unified odd-order MP capacity scan"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n\n[{datetime.now(UTC).isoformat()}] {title}\n{log_body}\n")
    print("Scenario 2 unified odd-order MP capacity scan completed.", flush=True)
    print(
        "Selected P="
        f"{selected_candidate.P}, M={selected_candidate.M}, "
        f"K={selected_candidate.coefficient_count}",
        flush=True,
    )
    print(f"Results: {RESULT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
