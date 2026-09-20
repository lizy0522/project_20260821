"""Evaluation-only coverage of the frozen ilc_end Envelope23 model."""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Set BLAS limits before NumPy/SciPy imports in spawned workers.
# ruff: noqa: E402
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

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from behavior_modeling.shared.basis_function_selection.all425_ilcend_envelope23_evaluation import (  # noqa: E402
    DMAX,
    OLS_CONDITION_HARD_LIMIT,
    PREVIOUS_HARD20_FINAL,
    RESULT_ROOT,
    STATE_COUNT,
    TARGET_NMSE_DB,
    _worker_init,
    raw_manifest_gate,
    verify_frozen_ilcend_support,
    worker_entry,
)

TASK_NAME = "scenario_2_all425_ilcend_envelope23_coverage_evaluation_5B"

WORK_LOG = (
    PROJECT_ROOT
    / "work_logs"
    / "scenario_2_all425_ilcend_envelope23_coverage_evaluation_5B"
    / "execution_log.txt"
)
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"
CHECKPOINT_PATH = RESULT_ROOT / "09_checkpoint.json"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message)


def _write_checkpoint(
    *,
    phase: str,
    completed_state_ids: list[int],
    support_hash: str,
    raw_manifest: dict[str, object],
    model_frozen: bool = True,
) -> None:
    CHECKPOINT_PATH.write_text(
        json.dumps(
            {
                "task_name": "scenario_2_all425_ilcend_envelope23_coverage_evaluation_5B",
                "phase": phase,
                "evaluation_only": True,
                "model_selection_performed": False,
                "model_name": "Envelope23-ilcEnd",
                "K": 23,
                "lambda": 0.0,
                "dmax": DMAX,
                "solver": "OLS",
                "support_hash": support_hash,
                "state_count": STATE_COUNT,
                "completed_state_ids": sorted(completed_state_ids),
                "train_bounds": [0, 16_384],
                "test_bounds": [16_384, 24_576],
                "ilc_end_policy": "last_valid_column_per_state",
                "worker_count": 10,
                "blas_threads_per_worker": 1,
                "raw_manifest": raw_manifest,
                "model_frozen": model_frozen,
                "test_unlocked": True,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _distribution(values: pd.Series) -> dict[str, float | int]:
    array = values.to_numpy(dtype=float)
    return {
        "count": int(array.size),
        "pass_count": int(np.count_nonzero(array < TARGET_NMSE_DB)),
        "median_dB": float(np.median(array)),
        "mean_dB": float(np.mean(array)),
        "Q90_dB": float(np.quantile(array, 0.90)),
        "Q95_dB": float(np.quantile(array, 0.95)),
        "Q99_dB": float(np.quantile(array, 0.99)),
        "worst_dB": float(np.max(array)),
    }


def _gap_summary(values: pd.Series) -> dict[str, float | int]:
    array = values.to_numpy(dtype=float)
    return {
        "median_dB": float(np.median(array)),
        "mean_dB": float(np.mean(array)),
        "Q90_dB": float(np.quantile(array, 0.90)),
        "Q95_dB": float(np.quantile(array, 0.95)),
        "Q99_dB": float(np.quantile(array, 0.99)),
        "max_dB": float(np.max(array)),
        "min_dB": float(np.min(array)),
        "positive_count": int(np.count_nonzero(array > 0.0)),
        "negative_count": int(np.count_nonzero(array < 0.0)),
        "abs_le_0p5_count": int(np.count_nonzero(np.abs(array) <= 0.5)),
        "abs_le_0p75_count": int(np.count_nonzero(np.abs(array) <= 0.75)),
        "abs_le_1p0_count": int(np.count_nonzero(np.abs(array) <= 1.0)),
    }


def _subset_summary(frame: pd.DataFrame) -> dict[str, object]:
    return {
        "count": int(len(frame)),
        "Train": _distribution(frame["Train_NMSE_dB"]),
        "Test": _distribution(frame["Test_NMSE_dB"]),
        "Both_pass_count": int(frame["Both_pass"].sum()),
        "Both_pass_rate": float(frame["Both_pass"].mean()),
        "gap": _gap_summary(frame["Generalization_gap_dB"]),
    }


def _load_type_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for load_type in (
        "matched",
        "fundamental_only",
        "second_harmonic_only",
        "joint_mismatch",
    ):
        subset = frame.loc[frame["load_type"] == load_type]
        summary = _subset_summary(subset)
        rows.append(
            {
                "load_type": load_type,
                "N": summary["count"],
                "Train_pass_count": summary["Train"]["pass_count"],
                "Train_pass_rate": summary["Train"]["pass_count"] / summary["count"],
                "Test_pass_count": summary["Test"]["pass_count"],
                "Test_pass_rate": summary["Test"]["pass_count"] / summary["count"],
                "Both_pass_count": summary["Both_pass_count"],
                "Both_pass_rate": summary["Both_pass_rate"],
                "Train_median_dB": summary["Train"]["median_dB"],
                "Train_Q95_dB": summary["Train"]["Q95_dB"],
                "Train_worst_dB": summary["Train"]["worst_dB"],
                "Test_median_dB": summary["Test"]["median_dB"],
                "Test_Q95_dB": summary["Test"]["Q95_dB"],
                "Test_worst_dB": summary["Test"]["worst_dB"],
                "Gap_median_dB": summary["gap"]["median_dB"],
                "Gap_Q95_dB": summary["gap"]["Q95_dB"],
                "Gap_max_dB": summary["gap"]["max_dB"],
            }
        )
    return pd.DataFrame(rows)


def _plot_main(frame: pd.DataFrame, path: Path) -> None:
    ordered = frame.sort_values("State_ID")
    x = ordered["State_ID"].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(18, 7), dpi=300)
    ax.plot(
        x, ordered["Train_NMSE_dB"], linewidth=1.0, color="#1f77b4", label="Train Modeling NMSE"
    )
    ax.plot(
        x, ordered["Test_NMSE_dB"], linewidth=1.0, color="#ff7f0e", label="Test Generalization NMSE"
    )
    ax.axhline(
        TARGET_NMSE_DB, color="#d62728", linestyle="--", linewidth=1.0, label="-40 dB Threshold"
    )
    ax.set_xlim(0, STATE_COUNT - 1)
    ax.set_xticks(
        [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 424]
    )
    lower = (
        min(
            float(ordered["Train_NMSE_dB"].min()),
            float(ordered["Test_NMSE_dB"].min()),
            TARGET_NMSE_DB,
        )
        - 0.5
    )
    upper = (
        max(
            float(ordered["Train_NMSE_dB"].max()),
            float(ordered["Test_NMSE_dB"].max()),
            TARGET_NMSE_DB,
        )
        + 0.5
    )
    ax.set_ylim(lower, upper)
    ax.set_xlabel("Load State")
    ax.set_ylabel("NMSE (dB)")
    ax.set_title("Frozen Envelope23-ilcEnd Modeling and Generalization Across 425 Load States")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_gap(frame: pd.DataFrame, path: Path) -> None:
    ordered = frame.sort_values("State_ID")
    fig, ax = plt.subplots(figsize=(18, 6), dpi=300)
    ax.plot(ordered["State_ID"], ordered["Generalization_gap_dB"], color="#2ca02c", linewidth=1.0)
    ax.axhline(0.0, color="#444444", linestyle="--", linewidth=0.9)
    ax.set_xlim(0, STATE_COUNT - 1)
    ax.set_xticks(
        [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 424]
    )
    ax.set_xlabel("Load State")
    ax.set_ylabel("Test - Train NMSE (dB)")
    ax.set_title("Frozen Envelope23-ilcEnd Generalization Gap Across 425 States")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _summary_text(payload: dict[str, object]) -> str:
    all_summary = payload["all_summary"]
    raw_before = json.dumps(payload["raw_before"], ensure_ascii=False, sort_keys=True)
    raw_after = json.dumps(payload["raw_after"], ensure_ascii=False, sort_keys=True)
    lines = [
        f"Task: {payload['task']}",
        "Mode: evaluation_only=true; model_selection_performed=false.",
        "Frozen model: Envelope23-ilcEnd; K=23; lambda=0; dmax=2; solver=OLS.",
        f"Support hash: {payload['support_hash']}",
        "Behavior chain: xin_pd_ori_ilc(:, ilc_end) -> yout_withdpd_ori_ilc(:, ilc_end).",
        "Full-record timing alignment precedes fixed Train/Test split; Train/Test gains "
        "are independent.",
        f"Train pass: {all_summary['Train']['pass_count']}/{STATE_COUNT}",
        f"Test pass: {all_summary['Test']['pass_count']}/{STATE_COUNT}",
        f"Both pass: {all_summary['Both_pass_count']}/{STATE_COUNT}",
        f"Pearson Train/Test correlation: {payload['train_test_correlation']}",
        "",
        "All425 summary:",
        json.dumps(all_summary, ensure_ascii=False, sort_keys=True),
        "Hard20 summary:",
        json.dumps(payload["hard20_summary"], ensure_ascii=False, sort_keys=True),
        "Remaining405 summary:",
        json.dumps(payload["remaining405_summary"], ensure_ascii=False, sort_keys=True),
        "Load-type summary:",
        json.dumps(payload["load_type_summary"], ensure_ascii=False, sort_keys=True),
        "Hard20 regression vs previous final ilc_end result:",
        json.dumps(payload["hard20_regression"], ensure_ascii=False, sort_keys=True),
        "",
        "No search, Ridge scan, support modification, Test-based retuning, LUT retrieval, "
        "Aend/C2, common-B, Type-III, DPD, low-bandwidth, or dmax3 was run.",
        f"Raw snapshot before: {raw_before}",
        f"Raw snapshot after: {raw_after}",
        f"raw_data_modified: {payload['raw_before'] != payload['raw_after']}",
    ]
    return "\n".join(lines) + "\n"


def _run(*, resume: bool = False) -> dict[str, object]:
    existing_output = RESULT_ROOT.exists() and any(RESULT_ROOT.iterdir())
    if existing_output and not resume:
        raise RuntimeError(
            "Result directory is not empty; use --resume only for this same evaluation task: "
            f"{RESULT_ROOT}"
        )
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    WORK_LOG.parent.mkdir(parents=True, exist_ok=True)
    raw_before = raw_manifest_gate()
    terms, support_indices, support_digest = verify_frozen_ilcend_support()
    task_definition = "\n".join(
        [
            f"Task: {TASK_NAME}",
            "Mode: evaluation_only=true; model_selection=false.",
            "Frozen model: Envelope23-ilcEnd; K=23; lambda=0; dmax=2; solver=OLS.",
            "Behavior chain: xin_pd_ori_ilc(:, ilc_end) -> yout_withdpd_ori_ilc(:, ilc_end).",
            "Train=[0,16384); Test=[16384,24576); full-record alignment before split.",
            "Train/Test gains are independent; theta is fitted on Train only.",
            f"Support hash: {support_digest}",
            "No search, Ridge scan, support modification, or downstream retrieval task.",
        ]
    )
    (RESULT_ROOT / "00_task_definition.txt").write_text(task_definition + "\n", encoding="utf-8")
    _append_log(
        WORK_LOG,
        "\n"
        f"[{_now()}] Start {TASK_NAME}\n"
        "Mode: evaluation_only=true; model_selection=false.\n"
        f"Raw snapshot before: {json.dumps(raw_before, ensure_ascii=False, sort_keys=True)}\n"
        f"Support hash: {support_digest}; workers=10; BLAS threads/worker=1.\n",
    )
    _write_checkpoint(
        phase="setup_complete",
        completed_state_ids=[],
        support_hash=support_digest,
        raw_manifest=raw_before,
    )

    rows: list[dict[str, object]] = []
    theta = np.empty((STATE_COUNT, 23), dtype=np.complex128)
    context = __import__("multiprocessing").get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=10,
        mp_context=context,
        initializer=_worker_init,
        initargs=(tuple(support_indices),),
    ) as executor:
        futures = {
            executor.submit(worker_entry, state_id): state_id for state_id in range(STATE_COUNT)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            theta[int(result["State_ID"])] = result.pop("theta")
            rows.append(result)
            if completed % 50 == 0 or completed == STATE_COUNT:
                _write_checkpoint(
                    phase="evaluation_progress",
                    completed_state_ids=sorted(int(row["State_ID"]) for row in rows),
                    support_hash=support_digest,
                    raw_manifest=raw_before,
                )
                print(f"[ALL425 ILCEND] {completed}/{STATE_COUNT}", flush=True)
    frame = pd.DataFrame(rows).sort_values("State_ID").reset_index(drop=True)
    if len(frame) != STATE_COUNT or frame["State_ID"].tolist() != list(range(STATE_COUNT)):
        raise RuntimeError("All425 ilc_end evaluation did not return canonical 0...424 rows")
    if not np.all(frame["rank"].to_numpy(dtype=int) == 23):
        raise RuntimeError("Envelope23 rank gate failed")
    if not frame[["finite"]].astype(bool).all().all():
        raise RuntimeError("Envelope23 finite gate failed")
    if np.max(frame["condition_number"].to_numpy(dtype=float)) > OLS_CONDITION_HARD_LIMIT:
        raise RuntimeError("Envelope23 condition-number gate failed")
    np.savez(
        RESULT_ROOT / "04_all425_train_coefficients.npz",
        state_ids=np.arange(STATE_COUNT, dtype=np.int64),
        theta=theta,
        basis_ids=np.asarray([term.basis_id for term in terms if term.index in support_indices]),
        K=np.asarray(23),
        lambda_value=np.asarray(0.0),
        dmax=np.asarray(DMAX),
        support_hash=np.asarray(support_digest),
        coefficient_source=np.asarray("Train only"),
    )
    frame["Train_pass"] = frame["Train_NMSE_dB"] < TARGET_NMSE_DB
    frame["Test_pass"] = frame["Test_NMSE_dB"] < TARGET_NMSE_DB
    frame["Both_pass"] = frame["Train_pass"] & frame["Test_pass"]
    frame.to_csv(RESULT_ROOT / "01_all425_ilcend_metrics.csv", index=False)
    failed = frame.loc[~frame["Both_pass"]].copy()
    failed.to_csv(RESULT_ROOT / "02_failed_states.csv", index=False)
    by_type = []
    for load_type in ("matched", "fundamental_only", "second_harmonic_only", "joint_mismatch"):
        subset = frame.loc[frame["load_type"] == load_type]
        by_type.append(
            {
                "load_type": load_type,
                "N": len(subset),
                "Train_pass_count": int(subset["Train_pass"].sum()),
                "Test_pass_count": int(subset["Test_pass"].sum()),
                "Both_pass_count": int(subset["Both_pass"].sum()),
                "Train_median_dB": float(subset["Train_NMSE_dB"].median()),
                "Test_median_dB": float(subset["Test_NMSE_dB"].median()),
                "Test_Q95_dB": float(subset["Test_NMSE_dB"].quantile(0.95)),
                "Test_worst_dB": float(subset["Test_NMSE_dB"].max()),
                "Gap_median_dB": float(subset["Generalization_gap_dB"].median()),
                "Gap_Q95_dB": float(subset["Generalization_gap_dB"].quantile(0.95)),
                "Gap_max_dB": float(subset["Generalization_gap_dB"].max()),
            }
        )
    by_type_frame = pd.DataFrame(by_type)
    by_type_frame.to_csv(RESULT_ROOT / "03_metrics_by_load_type.csv", index=False)
    subset_rows = []
    for name, subset in (
        ("Hard20", frame.loc[frame["is_Hard20"]]),
        ("Remaining405", frame.loc[~frame["is_Hard20"]]),
        ("All425", frame),
    ):
        subset_rows.append(
            {
                "subset": name,
                "N": len(subset),
                "Train_pass_count": int(subset["Train_pass"].sum()),
                "Test_pass_count": int(subset["Test_pass"].sum()),
                "Both_pass_count": int(subset["Both_pass"].sum()),
                "Train_median_dB": float(subset["Train_NMSE_dB"].median()),
                "Train_Q95_dB": float(subset["Train_NMSE_dB"].quantile(0.95)),
                "Train_worst_dB": float(subset["Train_NMSE_dB"].max()),
                "Test_median_dB": float(subset["Test_NMSE_dB"].median()),
                "Test_Q95_dB": float(subset["Test_NMSE_dB"].quantile(0.95)),
                "Test_worst_dB": float(subset["Test_NMSE_dB"].max()),
                "Gap_median_dB": float(subset["Generalization_gap_dB"].median()),
                "Gap_Q95_dB": float(subset["Generalization_gap_dB"].quantile(0.95)),
                "Gap_max_dB": float(subset["Generalization_gap_dB"].max()),
            }
        )
    pd.DataFrame(subset_rows).to_csv(RESULT_ROOT / "07_hard20_vs_remaining405.csv", index=False)
    previous = pd.read_csv(PREVIOUS_HARD20_FINAL).set_index("State_ID")
    current_hard = frame.loc[frame["is_Hard20"]].set_index("State_ID")
    train_error = np.max(
        np.abs(current_hard["Train_NMSE_dB"] - previous.loc[current_hard.index, "Train_NMSE_dB"])
    )
    test_error = np.max(
        np.abs(current_hard["Test_NMSE_dB"] - previous.loc[current_hard.index, "Test_NMSE_dB"])
    )
    hard20_regression = {
        "previous_result": str(PREVIOUS_HARD20_FINAL),
        "max_abs_train_error_dB": float(train_error),
        "max_abs_test_error_dB": float(test_error),
        "pass": bool(train_error < 1e-8 and test_error < 1e-8),
    }
    if not hard20_regression["pass"]:
        raise RuntimeError(f"Hard20 regression gate failed: {hard20_regression}")
    all_summary = {
        "N": STATE_COUNT,
        "Train": {
            "pass_count": int(frame["Train_pass"].sum()),
            "median_dB": float(frame["Train_NMSE_dB"].median()),
            "mean_dB": float(frame["Train_NMSE_dB"].mean()),
            "Q90_dB": float(frame["Train_NMSE_dB"].quantile(0.90)),
            "Q95_dB": float(frame["Train_NMSE_dB"].quantile(0.95)),
            "Q99_dB": float(frame["Train_NMSE_dB"].quantile(0.99)),
            "worst_dB": float(frame["Train_NMSE_dB"].max()),
        },
        "Test": {
            "pass_count": int(frame["Test_pass"].sum()),
            "median_dB": float(frame["Test_NMSE_dB"].median()),
            "mean_dB": float(frame["Test_NMSE_dB"].mean()),
            "Q90_dB": float(frame["Test_NMSE_dB"].quantile(0.90)),
            "Q95_dB": float(frame["Test_NMSE_dB"].quantile(0.95)),
            "Q99_dB": float(frame["Test_NMSE_dB"].quantile(0.99)),
            "worst_dB": float(frame["Test_NMSE_dB"].max()),
        },
        "Both_pass_count": int(frame["Both_pass"].sum()),
        "Gap": {
            "median_dB": float(frame["Generalization_gap_dB"].median()),
            "mean_dB": float(frame["Generalization_gap_dB"].mean()),
            "Q90_dB": float(frame["Generalization_gap_dB"].quantile(0.90)),
            "Q95_dB": float(frame["Generalization_gap_dB"].quantile(0.95)),
            "Q99_dB": float(frame["Generalization_gap_dB"].quantile(0.99)),
            "max_dB": float(frame["Generalization_gap_dB"].max()),
            "min_dB": float(frame["Generalization_gap_dB"].min()),
            "abs_le_0p5_count": int((frame["Generalization_gap_dB"].abs() <= 0.5).sum()),
            "abs_le_0p75_count": int((frame["Generalization_gap_dB"].abs() <= 0.75).sum()),
            "abs_le_1p0_count": int((frame["Generalization_gap_dB"].abs() <= 1.0).sum()),
        },
    }
    corr = float(np.corrcoef(frame["Train_NMSE_dB"], frame["Test_NMSE_dB"])[0, 1])
    raw_after = raw_manifest_gate()
    payload = {
        "task": TASK_NAME,
        "evaluation_only": True,
        "model_selection_performed": False,
        "support_hash": support_digest,
        "all_summary": all_summary,
        "hard20_summary": subset_rows[0],
        "remaining405_summary": subset_rows[1],
        "load_type_summary": by_type,
        "hard20_regression": hard20_regression,
        "train_test_correlation": corr,
        "raw_before": raw_before,
        "raw_after": raw_after,
    }
    (RESULT_ROOT / "08_final_result_summary.txt").write_text(
        _summary_text(payload), encoding="utf-8"
    )
    _plot_main(frame, RESULT_ROOT / "05_all425_modeling_generalization_nmse.png")
    _plot_gap(frame, RESULT_ROOT / "06_generalization_gap.png")
    _write_checkpoint(
        phase="completed",
        completed_state_ids=list(range(STATE_COUNT)),
        support_hash=support_digest,
        raw_manifest=raw_after,
    )
    return {
        "task": TASK_NAME,
        "Train_pass_count": int(frame["Train_pass"].sum()),
        "Test_pass_count": int(frame["Test_pass"].sum()),
        "Both_pass_count": int(frame["Both_pass"].sum()),
        "failed_count": int(len(failed)),
        "hard20_regression": hard20_regression,
        "raw_data_modified": raw_before != raw_after,
        "result_root": str(RESULT_ROOT),
        "support_hash": support_digest,
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
        f"Result: {json.dumps(result, ensure_ascii=False, sort_keys=True)}\n"
        "No search, Ridge scan, support modification, Test retuning, LUT retrieval, "
        "Aend/C2, common-B, Type-III, DPD, low-bandwidth, or dmax3 was run.\n",
    )
    _append_log(
        HANDOFF_LOG,
        f"\n{_now()} | 新任务：{TASK_NAME}\n"
        "完成冻结 ilc_end Envelope23 的 All-425 evaluation-only coverage；"
        f"结果目录：{RESULT_ROOT}\n"
        f"结果摘要：{json.dumps(result, ensure_ascii=False, sort_keys=True)}\n"
        "model_selection=false，Train-only theta，Test 使用同一 Train theta；"
        "未运行 search/Ridge/LUT retrieval/Aend-C2/common-B/Type-III/DPD/low-bandwidth/dmax3。\n",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    __import__("multiprocessing").freeze_support()
    main()
