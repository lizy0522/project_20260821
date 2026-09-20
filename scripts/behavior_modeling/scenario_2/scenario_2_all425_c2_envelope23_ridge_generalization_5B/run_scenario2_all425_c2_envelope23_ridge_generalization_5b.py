"""Run the frozen Envelope23-ilcEnd C2-only Ridge generalization task."""

# ruff: noqa: E402,E501

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any

# Set BLAS limits before NumPy/SciPy imports in spawned workers.
for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

MODULE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from behavior_modeling.scenario_2.scenario_2_all425_c2_envelope23_ridge_generalization_5B.c2_envelope23_ridge_generalization import (  # noqa: E402
    B_RAW_LENGTH,
    B_START,
    B_VALID_LENGTH,
    BLAS_THREADS_PER_WORKER,
    C2_COLUMN_INDEX,
    C2_STAGE_NUMBER,
    C_RAW_LENGTH,
    C_START,
    C_VALID_LENGTH,
    CV_BLOCK_RAW_LENGTHS,
    CV_BLOCK_VALID_LENGTHS,
    DMAX,
    EXPECTED_RAW_MANIFEST,
    EXPECTED_SUPPORT_HASH,
    EXPECTED_SUPPORT_IDS,
    FULL_LENGTH,
    HANDOFF_LOG,
    PREVIOUS_C2_OLS_METRICS,
    RESULT_ROOT,
    RIDGE_LAMBDA_GRID,
    STATE_COUNT,
    TASK_NAME,
    THRESHOLD_DB,
    WORK_LOG,
    WORKER_COUNT,
    build_c_block_slices,
    c_only_worker,
    evaluate_final_state,
    final_worker,
    load_previous_ols_reference,
    raw_manifest_gate,
    summary_stats,
    transition_label,
    verify_frozen_support,
    worker_init,
)

CHECKPOINT_PATH = RESULT_ROOT / "12_checkpoint.json"
MODEL_DEFINITION_PATH = (
    PROJECT_ROOT
    / "results"
    / "scenario_2_hard20_ilcend_envelope75_basis_selection_5B"
    / "08_final_model_definition.csv"
)
OLD_RESULT_ROOT = (
    PROJECT_ROOT / "results" / "behavior_fingerprint_retrieval" / "scenario_2" /"scenario_2_envelope23_ilcend_commonB_full_lut_retrieval_5B"
)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_checkpoint(
    *,
    phase: str,
    support_hash: str,
    raw_manifest: dict[str, object],
    selected_lambda: float | None,
    lambda_frozen: bool,
    b_evaluation_unlocked: bool,
    completed_cv_state_ids: list[int],
    completed_final_state_ids: list[int],
) -> None:
    CHECKPOINT_PATH.write_text(
        json.dumps(
            {
                "task_name": TASK_NAME,
                "phase": phase,
                "state_count": STATE_COUNT,
                "model_name": "Envelope23-ilcEnd",
                "K": 23,
                "dmax": DMAX,
                "support_hash": support_hash,
                "support_ids": list(EXPECTED_SUPPORT_IDS),
                "lambda_grid": list(RIDGE_LAMBDA_GRID),
                "selected_lambda": selected_lambda,
                "lambda_frozen": lambda_frozen,
                "lambda_selection_source": "C_only_contiguous_CV",
                "B_used_for_lambda_selection": False,
                "B_evaluation_unlocked": b_evaluation_unlocked,
                "C2_column_index": C2_COLUMN_INDEX,
                "C2_stage_number": C2_STAGE_NUMBER,
                "abc_bounds": [0, B_START, C_START, FULL_LENGTH],
                "C_block_raw_lengths": list(CV_BLOCK_RAW_LENGTHS),
                "C_block_valid_lengths": list(CV_BLOCK_VALID_LENGTHS),
                "completed_cv_state_ids": sorted(completed_cv_state_ids),
                "completed_final_state_ids": sorted(completed_final_state_ids),
                "worker_count": WORKER_COUNT,
                "blas_threads_per_worker": BLAS_THREADS_PER_WORKER,
                "raw_manifest": raw_manifest,
                "lut_retrieval_performed": False,
                "model_selection_performed": False,
                "basis_selection_performed": False,
                "ridge_scan_performed": True,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            default=_json_default,
        ),
        encoding="utf-8",
    )


def _validate_resume_checkpoint() -> None:
    if not CHECKPOINT_PATH.is_file():
        raise RuntimeError("--resume requires the task checkpoint")
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    expected = {
        "task_name": TASK_NAME,
        "state_count": STATE_COUNT,
        "model_name": "Envelope23-ilcEnd",
        "K": 23,
        "dmax": DMAX,
        "support_hash": EXPECTED_SUPPORT_HASH,
        "lambda_grid": list(RIDGE_LAMBDA_GRID),
        "C2_column_index": C2_COLUMN_INDEX,
        "abc_bounds": [0, B_START, C_START, FULL_LENGTH],
        "C_block_raw_lengths": list(CV_BLOCK_RAW_LENGTHS),
        "C_block_valid_lengths": list(CV_BLOCK_VALID_LENGTHS),
        "worker_count": WORKER_COUNT,
        "blas_threads_per_worker": BLAS_THREADS_PER_WORKER,
        "raw_manifest": EXPECTED_RAW_MANIFEST,
        "lut_retrieval_performed": False,
        "model_selection_performed": False,
        "basis_selection_performed": False,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise RuntimeError(f"resume checkpoint contract mismatch at {key}: {checkpoint.get(key)}")


def _write_task_definition(support_hash: str) -> None:
    (RESULT_ROOT / "00_task_definition.txt").write_text(
        "\n".join(
            [
                f"Task: {TASK_NAME}",
                "Purpose: Evaluate whether Ridge-stabilized C2 coefficient extraction improves C-segment modeling and C->B generalization.",
                "Frozen model: Envelope23-ilcEnd; K=23; dmax=2; support fixed.",
                f"Support hash: {support_hash}",
                "State range: 0...424.",
                "C2: second valid ILC stage selected through get_ilc_pair(..., iteration_index=1).",
                "Training segment: C=[17203,24576).",
                "External generalization segment: B=[12288,17203).",
                "C-only 3-fold contiguous CV selects one shared lambda for all 425 states.",
                "B is locked during lambda selection and is opened only after lambda is frozen.",
                "No LUT retrieval, fingerprint distance, Top-1, clustering, Type-III, DPD replay, low-bandwidth, basis search, or dmax scan.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_frozen_model_definition(terms: tuple[Any, ...], support: tuple[int, ...], digest: str) -> None:
    rows = []
    for index in support:
        term = terms[index]
        rows.append(
            {
                "term_index": int(term.index),
                "basis_id": term.basis_id,
                "p": int(term.order),
                "m": int(term.signal_delay),
                "q": "" if term.envelope_delay is None else int(term.envelope_delay),
                "K": 23,
                "dmax": DMAX,
                "support_hash": digest,
                "source_model_path": str(MODEL_DEFINITION_PATH),
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 23 or tuple(frame["basis_id"]) != EXPECTED_SUPPORT_IDS:
        raise RuntimeError("frozen model definition export is not the exact requested support")
    frame.to_csv(RESULT_ROOT / "01_frozen_model_definition.csv", index=False)


def _aggregate_cv_rows(cv_frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ridge_lambda in RIDGE_LAMBDA_GRID:
        group = cv_frame.loc[cv_frame["lambda"] == ridge_lambda]
        if len(group) != STATE_COUNT:
            raise RuntimeError(f"lambda={ridge_lambda} does not cover all 425 states")
        values = group["CV_W_NMSE_dB"].to_numpy(dtype=float)
        train_values = group["C_train_NMSE_dB"].to_numpy(dtype=float)
        theta_values = group["theta_norm"].to_numpy(dtype=float)
        if not np.all(np.isfinite(values)) or not np.all(group["finite"].astype(bool)):
            raise RuntimeError(f"lambda={ridge_lambda} has non-finite C-only CV values")
        rows.append(
            {
                "lambda": float(ridge_lambda),
                "CV_W_pass_count_lt_minus40": int(np.count_nonzero(values < THRESHOLD_DB)),
                "CV_W_pass_rate": float(np.mean(values < THRESHOLD_DB)),
                "CV_W_median_dB": float(np.median(values)),
                "CV_W_mean_dB": float(np.mean(values)),
                "CV_W_Q90_dB": float(np.quantile(values, 0.90)),
                "CV_W_Q95_dB": float(np.quantile(values, 0.95)),
                "CV_W_Q99_dB": float(np.quantile(values, 0.99)),
                "CV_W_worst_dB": float(np.max(values)),
                "C_train_pass_count": int(np.count_nonzero(train_values < THRESHOLD_DB)),
                "C_train_median_dB": float(np.median(train_values)),
                "C_train_Q95_dB": float(np.quantile(train_values, 0.95)),
                "C_train_worst_dB": float(np.max(train_values)),
                "theta_norm_median": float(np.median(theta_values)),
                "theta_norm_Q95": float(np.quantile(theta_values, 0.95)),
                "theta_norm_max": float(np.max(theta_values)),
                "selected": False,
            }
        )
    return pd.DataFrame(rows)


def _select_lambda(summary: pd.DataFrame) -> tuple[float, dict[str, object]]:
    if len(summary) != len(RIDGE_LAMBDA_GRID):
        raise RuntimeError("lambda summary does not cover the declared grid")
    row = min(
        summary.to_dict("records"),
        key=lambda item: (
            -int(item["CV_W_pass_count_lt_minus40"]),
            float(item["CV_W_worst_dB"]),
            float(item["CV_W_Q95_dB"]),
            float(item["CV_W_median_dB"]),
            float(item["lambda"]),
        ),
    )
    selected_lambda = float(row["lambda"])
    row["selected"] = True
    summary.loc[summary["lambda"] == selected_lambda, "selected"] = True
    return selected_lambda, row


def _write_lambda_selection_plot(summary: pd.DataFrame, path: Path) -> None:
    labels = ["OLS" if value == 0.0 else f"{value:.0e}" for value in summary["lambda"]]
    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(12, 6.5), dpi=220)
    ax.plot(x, summary["CV_W_median_dB"], marker="o", linewidth=1.5, label="CV-W median")
    ax.plot(x, summary["CV_W_Q95_dB"], marker="s", linewidth=1.5, label="CV-W Q95")
    ax.plot(x, summary["CV_W_worst_dB"], marker="^", linewidth=1.5, label="CV-W worst")
    ax.axhline(THRESHOLD_DB, color="#222222", linestyle="--", linewidth=1.0, label="-40 dB threshold")
    selected = summary[summary["selected"].astype(bool)]
    selected_index = int(selected.index[0])
    ax.axvline(selected_index, color="#d62728", linestyle=":", linewidth=1.2, label="selected lambda")
    ax.set_xticks(x, labels)
    ax.set_xlabel("Ridge lambda (0 shown as OLS)")
    ax.set_ylabel("C-only CV-W NMSE (dB)")
    ax.set_title("C-only contiguous CV for frozen Envelope23-ilcEnd C2")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    ax2 = ax.twinx()
    ax2.plot(x, summary["CV_W_pass_count_lt_minus40"], color="#2ca02c", linestyle="--", marker=".", label="CV-W pass count")
    ax2.set_ylabel("CV-W pass count (< -40 dB)")
    ax2.set_ylim(0, STATE_COUNT * 1.04)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_final_comparison_plot(frame: pd.DataFrame, path: Path) -> None:
    ordered = frame.sort_values("State_ID")
    x = ordered["State_ID"].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(18, 7), dpi=220)
    specs = (
        ("C_train_OLS_NMSE_dB", "OLS C Train", "#7f7f7f"),
        ("C_to_B_OLS_NMSE_dB", "OLS C→B", "#4d4d4d"),
        ("C_train_Ridge_NMSE_dB", "Ridge C Train", "#1f77b4"),
        ("C_to_B_Ridge_NMSE_dB", "Ridge C→B", "#d62728"),
    )
    for column, label, color in specs:
        ax.plot(x, ordered[column], linewidth=0.9, label=label, color=color)
    ax.axhline(THRESHOLD_DB, color="#111111", linestyle="--", linewidth=1.0, label="-40 dB threshold")
    ax.set_xlim(0, STATE_COUNT - 1)
    ax.set_xticks([0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 424])
    ax.set_xlabel("State ID")
    ax.set_ylabel("NMSE (dB)")
    ax.set_title("Frozen Envelope23-ilcEnd C2: OLS versus C-only-selected Ridge")
    ax.grid(alpha=0.23)
    ax.legend(frameon=False, ncol=3, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _quality_rows(frame: pd.DataFrame, selected_lambda: float) -> list[dict[str, object]]:
    rows = []
    for label, suffix, ridge_lambda in (
        ("OLS", "OLS", 0.0),
        ("Ridge", "Ridge", selected_lambda),
    ):
        train = frame[f"C_train_{suffix}_NMSE_dB"].to_numpy(dtype=float)
        generalization = frame[f"C_to_B_{suffix}_NMSE_dB"].to_numpy(dtype=float)
        gap = frame[f"generalization_gap_{suffix}_dB"].to_numpy(dtype=float)
        train_summary = summary_stats(train)
        b_summary = summary_stats(generalization)
        gap_summary = summary_stats(gap)
        rows.append(
            {
                "model": label,
                "lambda": float(ridge_lambda),
                "C_train_pass": train_summary["pass_count"],
                "C_train_median": train_summary["median_dB"],
                "C_train_mean": train_summary["mean_dB"],
                "C_train_Q90": train_summary["Q90_dB"],
                "C_train_Q95": train_summary["Q95_dB"],
                "C_train_Q99": train_summary["Q99_dB"],
                "C_train_worst": train_summary["worst_dB"],
                "C_to_B_pass": b_summary["pass_count"],
                "C_to_B_median": b_summary["median_dB"],
                "C_to_B_mean": b_summary["mean_dB"],
                "C_to_B_Q90": b_summary["Q90_dB"],
                "C_to_B_Q95": b_summary["Q95_dB"],
                "C_to_B_Q99": b_summary["Q99_dB"],
                "C_to_B_worst": b_summary["worst_dB"],
                "generalization_gap_median": gap_summary["median_dB"],
                "generalization_gap_Q95": gap_summary["Q95_dB"],
                "generalization_gap_worst": gap_summary["worst_dB"],
            }
        )
    return rows


def _compare_ols_regression(final_frame: pd.DataFrame) -> dict[str, object]:
    baseline = load_previous_ols_reference()
    baseline_train = baseline["Y_C2_train_NMSE_dB"].to_numpy(dtype=float)
    baseline_b = baseline["Y_C2_B_NMSE_dB"].to_numpy(dtype=float)
    current_train = final_frame.sort_values("State_ID")["C_train_OLS_NMSE_dB"].to_numpy(dtype=float)
    current_b = final_frame.sort_values("State_ID")["C_to_B_OLS_NMSE_dB"].to_numpy(dtype=float)
    train_error = float(np.max(np.abs(current_train - baseline_train)))
    b_error = float(np.max(np.abs(current_b - baseline_b)))
    result = {
        "pass": bool(train_error < 1e-8 and b_error < 1e-8),
        "max_abs_C_train_error_dB": train_error,
        "max_abs_C_to_B_error_dB": b_error,
        "previous_result": str(PREVIOUS_C2_OLS_METRICS),
        "previous_C_train_pass": int(np.count_nonzero(baseline_train < THRESHOLD_DB)),
        "previous_C_to_B_pass": int(np.count_nonzero(baseline_b < THRESHOLD_DB)),
    }
    if not result["pass"]:
        raise RuntimeError(f"lambda=0 OLS regression failed: {result}")
    if result["previous_C_train_pass"] != 425 or result["previous_C_to_B_pass"] != 57:
        raise RuntimeError(f"previous C2 OLS pass counts changed: {result}")
    return result


def _run(*, resume: bool = False) -> dict[str, object]:
    existing_output = RESULT_ROOT.exists() and any(RESULT_ROOT.iterdir())
    if existing_output and not resume:
        raise RuntimeError("Result directory is not empty; use --resume for the same task")
    if resume:
        _validate_resume_checkpoint()
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    raw_before = raw_manifest_gate()
    terms, support, support_hash = verify_frozen_support()
    blocks = build_c_block_slices()
    _write_task_definition(support_hash)
    _write_frozen_model_definition(terms, support, support_hash)
    _write_checkpoint(
        phase="setup_complete",
        support_hash=support_hash,
        raw_manifest=raw_before,
        selected_lambda=None,
        lambda_frozen=False,
        b_evaluation_unlocked=False,
        completed_cv_state_ids=[],
        completed_final_state_ids=[],
    )
    _append_log(
        WORK_LOG,
        "\n"
        f"[{_now()}] Start {TASK_NAME}\n"
        "Mode: new task; frozen Envelope23-ilcEnd, C2-only, C-only lambda selection, B locked until lambda freeze.\n"
        f"support_hash={support_hash}; C2_column_index={C2_COLUMN_INDEX}; C2_stage_number={C2_STAGE_NUMBER}; "
        f"ABC=[0,{B_START},{C_START},{FULL_LENGTH}]; C_blocks={[(item.start, item.stop) for item in blocks]}; "
        f"raw={json.dumps(raw_before, ensure_ascii=False, sort_keys=True)}\n"
        f"lambda_grid={list(RIDGE_LAMBDA_GRID)}; workers={WORKER_COUNT}; BLAS threads/worker={BLAS_THREADS_PER_WORKER}.\n"
        "B is not used for lambda selection; no LUT/fingerprint/retrieval path is imported.\n",
    )

    cv_rows: list[dict[str, object]] = []
    preflight_rows: list[dict[str, object]] = []
    completed_cv: list[int] = []
    context = get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=WORKER_COUNT,
        mp_context=context,
        initializer=worker_init,
        initargs=(tuple(support),),
    ) as executor:
        futures = {
            executor.submit(c_only_worker, state_id, tuple(RIDGE_LAMBDA_GRID)): state_id
            for state_id in range(STATE_COUNT)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            state_id = futures[future]
            result = future.result()
            preflight_rows.append(result["preflight"])
            cv_rows.extend(result["cv_rows"])
            completed_cv.append(int(state_id))
            if completed % 25 == 0 or completed == STATE_COUNT:
                _write_checkpoint(
                    phase="c_only_cv_progress",
                    support_hash=support_hash,
                    raw_manifest=raw_before,
                    selected_lambda=None,
                    lambda_frozen=False,
                    b_evaluation_unlocked=False,
                    completed_cv_state_ids=completed_cv,
                    completed_final_state_ids=[],
                )
                print(f"[C-ONLY CV] {completed}/{STATE_COUNT}", flush=True)

    if sorted(completed_cv) != list(range(STATE_COUNT)):
        raise RuntimeError("C-only CV did not cover canonical State IDs 0...424")
    preflight = pd.DataFrame(preflight_rows).sort_values("State_ID").reset_index(drop=True)
    if len(preflight) != STATE_COUNT or preflight["State_ID"].tolist() != list(range(STATE_COUNT)):
        raise RuntimeError("C2 preflight is not a canonical 425-row table")
    if not bool(preflight["C2_available"].all()) or not bool(preflight["finite"].all()):
        raise RuntimeError("C2 preflight availability/finite gate failed")
    if set(preflight["C2_column_index"].astype(int)) != {C2_COLUMN_INDEX}:
        raise RuntimeError("C2 column index changed")
    if set(preflight["B_raw_length"].astype(int)) != {B_RAW_LENGTH} or set(preflight["B_valid_length"].astype(int)) != {B_VALID_LENGTH}:
        raise RuntimeError("B length gate failed")
    if set(preflight["C_raw_length"].astype(int)) != {C_RAW_LENGTH} or set(preflight["C_valid_length"].astype(int)) != {C_VALID_LENGTH}:
        raise RuntimeError("C length gate failed")
    preflight.to_csv(RESULT_ROOT / "02_c2_data_preflight.csv", index=False)

    cv_frame = pd.DataFrame(cv_rows).sort_values(["lambda", "State_ID"]).reset_index(drop=True)
    expected_cv_rows = STATE_COUNT * len(RIDGE_LAMBDA_GRID)
    if len(cv_frame) != expected_cv_rows:
        raise RuntimeError(f"C-only per-state CV row count={len(cv_frame)}, expected={expected_cv_rows}")
    cv_frame.to_csv(RESULT_ROOT / "04_lambda_grid_cv_per_state.csv", index=False)
    cv_summary = _aggregate_cv_rows(cv_frame)
    selected_lambda, selected_row = _select_lambda(cv_summary)
    cv_summary.to_csv(RESULT_ROOT / "03_lambda_grid_cv_summary.csv", index=False)
    selection_payload = {
        "selected_lambda": selected_lambda,
        "selection_basis": "C-only 3-fold contiguous CV",
        "selection_objective": [
            "maximize CV_W_pass_count_lt_minus40",
            "minimize CV_W_worst_dB",
            "minimize CV_W_Q95_dB",
            "minimize CV_W_median_dB",
            "minimize lambda on complete tie",
        ],
        "selected_metrics": selected_row,
        "lambda_grid": list(RIDGE_LAMBDA_GRID),
        "B_used_for_selection": False,
        "lambda_frozen": True,
        "support_hash": support_hash,
        "K": 23,
        "dmax": DMAX,
    }
    (RESULT_ROOT / "05_selected_lambda.json").write_text(
        json.dumps(selection_payload, ensure_ascii=False, indent=2, default=_json_default),
        encoding="utf-8",
    )
    _write_lambda_selection_plot(cv_summary, RESULT_ROOT / "09_lambda_cv_selection.png")
    _write_checkpoint(
        phase="lambda_frozen",
        support_hash=support_hash,
        raw_manifest=raw_before,
        selected_lambda=selected_lambda,
        lambda_frozen=True,
        b_evaluation_unlocked=True,
        completed_cv_state_ids=completed_cv,
        completed_final_state_ids=[],
    )
    _append_log(
        WORK_LOG,
        f"[{_now()}] C-only lambda selection frozen: selected_lambda={selected_lambda:.12g}; "
        f"selected CV-W pass={selected_row['CV_W_pass_count_lt_minus40']}/425; "
        f"worst={selected_row['CV_W_worst_dB']:.9f} dB; Q95={selected_row['CV_W_Q95_dB']:.9f} dB; "
        f"median={selected_row['CV_W_median_dB']:.9f} dB. B used=False.\n",
    )

    final_results: list[dict[str, object]] = []
    completed_final: list[int] = []
    with ProcessPoolExecutor(
        max_workers=WORKER_COUNT,
        mp_context=context,
        initializer=worker_init,
        initargs=(tuple(support),),
    ) as executor:
        futures = {
            executor.submit(final_worker, state_id, selected_lambda): state_id
            for state_id in range(STATE_COUNT)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            state_id = futures[future]
            final_results.append(future.result())
            completed_final.append(int(state_id))
            if completed % 25 == 0 or completed == STATE_COUNT:
                _write_checkpoint(
                    phase="final_C_and_B_progress",
                    support_hash=support_hash,
                    raw_manifest=raw_before,
                    selected_lambda=selected_lambda,
                    lambda_frozen=True,
                    b_evaluation_unlocked=True,
                    completed_cv_state_ids=completed_cv,
                    completed_final_state_ids=completed_final,
                )
                print(f"[FINAL C/B] {completed}/{STATE_COUNT}", flush=True)

    if sorted(completed_final) != list(range(STATE_COUNT)):
        raise RuntimeError("Final C/B evaluation did not cover canonical State IDs 0...424")
    theta_ridge = np.empty((STATE_COUNT, 23), dtype=np.complex128)
    scalar_rows = []
    for result in sorted(final_results, key=lambda item: int(item["State_ID"])):
        state_id = int(result["State_ID"])
        theta_ridge[state_id] = result.pop("theta_ridge")
        result["generalization_transition"] = transition_label(
            bool(result["OLS_C_to_B_pass"]), bool(result["Ridge_C_to_B_pass"])
        )
        scalar_rows.append(result)
    final_frame = pd.DataFrame(scalar_rows).sort_values("State_ID").reset_index(drop=True)
    if len(final_frame) != STATE_COUNT or final_frame["State_ID"].tolist() != list(range(STATE_COUNT)):
        raise RuntimeError("final modeling/generalization table is not canonical")
    if theta_ridge.shape != (STATE_COUNT, 23) or not np.all(np.isfinite(theta_ridge)):
        raise RuntimeError(f"final Ridge coefficient shape/finite gate failed: {theta_ridge.shape}")

    ols_regression = _compare_ols_regression(final_frame)
    final_frame.to_csv(RESULT_ROOT / "07_final_all425_modeling_generalization.csv", index=False)
    np.savez(
        RESULT_ROOT / "06_final_ridge_coefficients.npz",
        state_ids=np.arange(STATE_COUNT, dtype=np.int64),
        theta_ridge=theta_ridge,
        basis_ids=np.asarray(EXPECTED_SUPPORT_IDS),
        selected_lambda=np.asarray(selected_lambda),
        K=np.asarray(23),
        dmax=np.asarray(DMAX),
        support_hash=np.asarray(support_hash),
    )

    spot_check_ids = (0, 187, 325, 424)
    for state_id in spot_check_ids:
        direct = evaluate_final_state(
            state_id,
            selected_lambda,
            terms=terms,
            support=support,
            lambda_frozen=True,
            b_evaluation_unlocked=True,
        )
        batch = final_frame.loc[final_frame["State_ID"] == state_id].iloc[0]
        for key in (
            "C_train_OLS_NMSE_dB",
            "C_to_B_OLS_NMSE_dB",
            "C_train_Ridge_NMSE_dB",
            "C_to_B_Ridge_NMSE_dB",
        ):
            if abs(float(direct[key]) - float(batch[key])) > 1e-12:
                raise RuntimeError(f"spot check mismatch at state {state_id}, {key}")

    comparison_rows = _quality_rows(final_frame, selected_lambda)
    pd.DataFrame(comparison_rows).to_csv(
        RESULT_ROOT / "08_ols_vs_ridge_paired_comparison.csv", index=False
    )
    _write_final_comparison_plot(
        final_frame,
        RESULT_ROOT / "10_ols_vs_ridge_modeling_generalization.png",
    )

    transition_counts = final_frame["generalization_transition"].value_counts().to_dict()
    transition_counts = {
        key: int(transition_counts.get(key, 0))
        for key in (
            "unchanged_pass",
            "OLS_fail_to_Ridge_pass",
            "OLS_pass_to_Ridge_fail",
            "unchanged_fail",
        )
    }
    ols_train = summary_stats(final_frame["C_train_OLS_NMSE_dB"])
    ols_b = summary_stats(final_frame["C_to_B_OLS_NMSE_dB"])
    ridge_train = summary_stats(final_frame["C_train_Ridge_NMSE_dB"])
    ridge_b = summary_stats(final_frame["C_to_B_Ridge_NMSE_dB"])
    gap_ols = summary_stats(final_frame["generalization_gap_OLS_dB"])
    gap_ridge = summary_stats(final_frame["generalization_gap_Ridge_dB"])
    if selected_lambda == 0.0:
        lambda_conclusion = "C-only CV does not support non-zero Ridge regularization."
    elif ridge_b["pass_count"] > ols_b["pass_count"]:
        lambda_conclusion = "Ridge selected without B information improves C->B generalization."
    else:
        lambda_conclusion = "Ridge improves C-internal validation but does not improve external B generalization."
    raw_after = raw_manifest_gate()
    if raw_after != raw_before:
        raise RuntimeError("raw manifest changed during task")
    summary_lines = [
        f"Task: {TASK_NAME}",
        "Frozen model: Envelope23-ilcEnd",
        "K = 23",
        f"dmax = {DMAX}",
        "OLS baseline lambda: 0",
        f"Selected Ridge lambda: {selected_lambda:.12g}",
        "Lambda selected using: C-only contiguous CV",
        "B used during lambda selection: No",
        f"C2 column index: {C2_COLUMN_INDEX} (stage {C2_STAGE_NUMBER})",
        f"Support hash: {support_hash}",
        "",
        "C Train",
        f"OLS: {ols_train['pass_count']}/425 < -40 dB; median={ols_train['median_dB']:.9f} dB; Q95={ols_train['Q95_dB']:.9f} dB; worst={ols_train['worst_dB']:.9f} dB",
        f"Ridge: {ridge_train['pass_count']}/425 < -40 dB; median={ridge_train['median_dB']:.9f} dB; Q95={ridge_train['Q95_dB']:.9f} dB; worst={ridge_train['worst_dB']:.9f} dB",
        "",
        "C->B Generalization",
        f"OLS: {ols_b['pass_count']}/425 < -40 dB; median={ols_b['median_dB']:.9f} dB; Q95={ols_b['Q95_dB']:.9f} dB; worst={ols_b['worst_dB']:.9f} dB",
        f"Ridge: {ridge_b['pass_count']}/425 < -40 dB; median={ridge_b['median_dB']:.9f} dB; Q95={ridge_b['Q95_dB']:.9f} dB; worst={ridge_b['worst_dB']:.9f} dB",
        "",
        "C->B transition",
        f"unchanged pass: {transition_counts['unchanged_pass']}",
        f"OLS fail -> Ridge pass: {transition_counts['OLS_fail_to_Ridge_pass']}",
        f"OLS pass -> Ridge fail: {transition_counts['OLS_pass_to_Ridge_fail']}",
        f"unchanged fail: {transition_counts['unchanged_fail']}",
        "",
        "Generalization gap (B - C Train)",
        f"OLS: median={gap_ols['median_dB']:.9f} dB; Q95={gap_ols['Q95_dB']:.9f} dB; worst={gap_ols['worst_dB']:.9f} dB",
        f"Ridge: median={gap_ridge['median_dB']:.9f} dB; Q95={gap_ridge['Q95_dB']:.9f} dB; worst={gap_ridge['worst_dB']:.9f} dB",
        "",
        f"lambda conclusion: {lambda_conclusion}",
        f"C-only selected metrics: {json.dumps(selected_row, ensure_ascii=False, sort_keys=True, default=_json_default)}",
        f"OLS regression: {json.dumps(ols_regression, ensure_ascii=False, sort_keys=True, default=_json_default)}",
        f"Independent spot checks: {list(spot_check_ids)} passed.",
        "No LUT retrieval, fingerprint distance, Top-1, Real-B retrieval, clustering, Type-III, DPD replay, low-bandwidth, Aend Ridge, basis search, support change, or dmax scan was run.",
        f"Raw snapshot before: {json.dumps(raw_before, ensure_ascii=False, sort_keys=True)}",
        f"Raw snapshot after: {json.dumps(raw_after, ensure_ascii=False, sort_keys=True)}",
        f"raw_data_modified: {raw_before != raw_after}",
    ]
    (RESULT_ROOT / "11_final_result_summary.txt").write_text(
        "\n".join(summary_lines) + "\n", encoding="utf-8"
    )
    _write_checkpoint(
        phase="completed",
        support_hash=support_hash,
        raw_manifest=raw_after,
        selected_lambda=selected_lambda,
        lambda_frozen=True,
        b_evaluation_unlocked=True,
        completed_cv_state_ids=completed_cv,
        completed_final_state_ids=completed_final,
    )
    return {
        "task": TASK_NAME,
        "selected_lambda": selected_lambda,
        "cv_W_pass_count": int(selected_row["CV_W_pass_count_lt_minus40"]),
        "ols_C_train_pass": int(ols_train["pass_count"]),
        "ridge_C_train_pass": int(ridge_train["pass_count"]),
        "ols_C_to_B_pass": int(ols_b["pass_count"]),
        "ridge_C_to_B_pass": int(ridge_b["pass_count"]),
        "transition": transition_counts,
        "ols_regression": ols_regression,
        "raw_data_modified": raw_before != raw_after,
        "result_root": str(RESULT_ROOT),
    }


def main() -> None:
    try:
        result = _run(resume="--resume" in sys.argv[1:])
    except Exception as exc:
        _append_log(WORK_LOG, f"[{_now()}] FAILED {TASK_NAME}: {type(exc).__name__}: {exc}\n")
        raise
    _append_log(
        WORK_LOG,
        f"[{_now()}] Completed {TASK_NAME}\n"
        f"Result: {json.dumps(result, ensure_ascii=False, sort_keys=True, default=_json_default)}\n"
        "B was opened only after C-only lambda freeze. No LUT/retrieval path was run.\n",
    )
    _append_log(
        HANDOFF_LOG,
        f"\n{_now()} | 新任务：{TASK_NAME}\n"
        f"完成 Frozen Envelope23-ilcEnd + C2-only Ridge 系数提取；结果目录：{RESULT_ROOT}\n"
        f"结果摘要：{json.dumps(result, ensure_ascii=False, sort_keys=True, default=_json_default)}\n"
        "lambda 仅由 C-only contiguous CV 选择，B 只在 lambda 冻结后用于 C→B 泛化评价；未运行 LUT、指纹、聚类、DPD 或 low-bandwidth。\n",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    __import__("multiprocessing").freeze_support()
    main()
