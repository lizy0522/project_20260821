"""Run evaluation-only All425 coverage for frozen Envelope19-C2EndShared."""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

# ruff: noqa: E402,E501

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

from data_management.shared import build_state_table  # noqa: E402

from behavior_modeling.shared.basis_function_selection.all425_c2_ilcend_envelope19_shared_coverage import (  # noqa: E402
    BEHAVIORS,
    DMAX,
    EXPECTED_HARD20_IDS,
    EXPECTED_MODEL_NAME,
    EXPECTED_RAW_MANIFEST,
    EXPECTED_SUPPORT_HASH,
    HANDOFF_LOG,
    ILC_COL2,
    ILC_END,
    PREVIOUS_HARD20_RESULT,
    RESULT_ROOT,
    STATE_COUNT,
    TARGET_NMSE_DB,
    TASK_NAME,
    WORK_LOG,
    WORKER_COUNT,
    _worker_entry,
    _worker_init,
    load_frozen_model,
    raw_manifest_gate,
    summary_stats,
)

CHECKPOINT_PATH = RESULT_ROOT / "11_checkpoint.json"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message)


def _write_checkpoint(**payload: object) -> None:
    CHECKPOINT_PATH.write_text(
        json.dumps(
            {
                "task_name": TASK_NAME,
                "mode": "evaluation_only",
                "model_name": EXPECTED_MODEL_NAME,
                "K": 19,
                "dmax": 2,
                "lambda": 0.0,
                "solver": "OLS",
                "state_start": 0,
                "state_end": 424,
                "state_count": STATE_COUNT,
                "behaviors": list(BEHAVIORS),
                "train_bounds": [0, 16_384],
                "test_bounds": [16_384, 24_576],
                "worker_count": WORKER_COUNT,
                "blas_threads_per_worker": 1,
                "basis_selection_performed": False,
                "ridge_scan_performed": False,
                "cv_performed": False,
                "lut_retrieval_performed": False,
                **payload,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _validate_resume_checkpoint() -> None:
    if not CHECKPOINT_PATH.is_file():
        raise RuntimeError("--resume requires an existing checkpoint")
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    expected = {
        "task_name": TASK_NAME,
        "mode": "evaluation_only",
        "model_name": EXPECTED_MODEL_NAME,
        "K": 19,
        "dmax": 2,
        "lambda": 0.0,
        "solver": "OLS",
        "state_start": 0,
        "state_end": 424,
        "state_count": STATE_COUNT,
        "behaviors": list(BEHAVIORS),
        "train_bounds": [0, 16_384],
        "test_bounds": [16_384, 24_576],
        "worker_count": WORKER_COUNT,
        "blas_threads_per_worker": 1,
        "basis_selection_performed": False,
        "ridge_scan_performed": False,
        "cv_performed": False,
        "lut_retrieval_performed": False,
    }
    for key, value in expected.items():
        if checkpoint.get(key) != value:
            raise RuntimeError(f"resume checkpoint mismatch at {key}")
    if checkpoint.get("support_hash") != EXPECTED_SUPPORT_HASH:
        raise RuntimeError("resume checkpoint support hash mismatch")
    if checkpoint.get("raw_manifest_before") != EXPECTED_RAW_MANIFEST:
        raise RuntimeError("resume checkpoint raw manifest mismatch")


def _write_task_definition(model_frame: pd.DataFrame) -> None:
    (RESULT_ROOT / "00_task_definition.txt").write_text(
        "\n".join(
            [
                f"Task: {TASK_NAME}",
                "Mode: evaluation_only.",
                f"Frozen model: {EXPECTED_MODEL_NAME}; K=19; dmax=2; lambda=0; solver=OLS.",
                f"Support hash: {EXPECTED_SUPPORT_HASH}",
                "State range: 0...424.",
                "Behavior 1: ILC_END, last valid ILC column per state.",
                "Behavior 2: ILC_COL2, literal MATLAB column 2 / Python index 1.",
                "Full record: 24576; Train=[0,16384); Test=[16384,24576).",
                "Full-record alignment precedes split; Train/Test gains are independent per behavior.",
                "No basis selection, hyperparameter search, CV, Ridge, ABC, common-B, LUT, fingerprint, or retrieval.",
                f"Frozen model definition rows: {len(model_frame)}.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_frozen_definition(frame: pd.DataFrame) -> None:
    output = frame.copy()
    output["source_task"] = "scenario_2_hard20_c2_ilcend_shared_envelope75_basis_selection_5B"
    output["source_model_definition"] = str(
        PROJECT_ROOT
        / "results"
        / "scenario_2_hard20_c2_ilcend_shared_envelope75_basis_selection_5B"
        / "07_final_model_definition.csv"
    )
    output.to_csv(RESULT_ROOT / "01_frozen_model_definition.csv", index=False)


def _combine_preflight(rows: list[dict[str, object]], state_table: dict[int, dict[str, int | float]]) -> pd.DataFrame:
    long_frame = pd.DataFrame(rows).sort_values(["State_ID", "behavior"]).reset_index(drop=True)
    if len(long_frame) != STATE_COUNT * 2:
        raise RuntimeError("All425 preflight does not contain 850 behavior rows")
    wide_rows = []
    for state_id in range(STATE_COUNT):
        state_rows = long_frame.loc[long_frame["State_ID"] == state_id].set_index("behavior")
        end = state_rows.loc[ILC_END]
        col2 = state_rows.loc[ILC_COL2]
        valid_count = int(end["valid_ilc_count"])
        wide_rows.append(
            {
                "State_ID": state_id,
                "funMng": int(state_table[state_id]["funMng"]),
                "funAng": int(state_table[state_id]["funAng"]),
                "secMng": int(state_table[state_id]["secMng"]),
                "secAng": int(state_table[state_id]["secAng"]),
                "valid_ilc_count": valid_count,
                "ilc_end_matlab_column": valid_count,
                "ilc_end_python_index": valid_count - 1,
                "ilc_col2_available": bool(int(col2["ILC_column_index"]) == 1),
                "col2_equals_ilc_end": bool(valid_count - 1 == 1),
                "full_length": int(end["full_length"]),
                "end_delay_rough": int(end["full_delay_rough"]),
                "end_delay_fine": float(end["full_delay_fine"]),
                "col2_delay_rough": int(col2["full_delay_rough"]),
                "col2_delay_fine": float(col2["full_delay_fine"]),
                "end_train_gain_real": float(end["Train_gain_real"]),
                "end_train_gain_imag": float(end["Train_gain_imag"]),
                "end_test_gain_real": float(end["Test_gain_real"]),
                "end_test_gain_imag": float(end["Test_gain_imag"]),
                "col2_train_gain_real": float(col2["Train_gain_real"]),
                "col2_train_gain_imag": float(col2["Train_gain_imag"]),
                "col2_test_gain_real": float(col2["Test_gain_real"]),
                "col2_test_gain_imag": float(col2["Test_gain_imag"]),
                "train_raw_length": int(end["Train_raw_length"]),
                "train_valid_length": int(end["Train_valid_length"]),
                "test_raw_length": int(end["Test_raw_length"]),
                "test_valid_length": int(end["Test_valid_length"]),
                "finite": bool(end["finite"] and col2["finite"]),
            }
        )
    frame = pd.DataFrame(wide_rows)
    if frame["State_ID"].tolist() != list(range(STATE_COUNT)):
        raise RuntimeError("All425 preflight state order is not canonical")
    return frame


def _build_coverage_frame(results: list[dict[str, object]]) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    rows = []
    theta_end = np.empty((STATE_COUNT, 19), dtype=np.complex128)
    theta_col2 = np.empty_like(theta_end)
    for result in sorted(results, key=lambda item: int(item["State_ID"])):
        state_id = int(result["State_ID"])
        end = result["behaviors"][ILC_END]
        col2 = result["behaviors"][ILC_COL2]
        theta_end[state_id] = end.pop("theta")
        theta_col2[state_id] = col2.pop("theta")
        joint = bool(end["Both_pass"] and col2["Both_pass"])
        rows.append(
            {
                "State_ID": state_id,
                "funMng": int(result["funMng"]),
                "funAng": int(result["funAng"]),
                "secMng": int(result["secMng"]),
                "secAng": int(result["secAng"]),
                "ILC_END_Train_NMSE_dB": end["Train_NMSE_dB"],
                "ILC_END_Test_NMSE_dB": end["Test_NMSE_dB"],
                "ILC_END_Gap_dB": end["Generalization_gap_dB"],
                "ILC_END_Train_pass": end["Train_pass"],
                "ILC_END_Test_pass": end["Test_pass"],
                "ILC_END_Both_pass": end["Both_pass"],
                "ILC_END_rank": end["rank"],
                "ILC_END_condition_number": end["condition_number"],
                "ILC_COL2_Train_NMSE_dB": col2["Train_NMSE_dB"],
                "ILC_COL2_Test_NMSE_dB": col2["Test_NMSE_dB"],
                "ILC_COL2_Gap_dB": col2["Generalization_gap_dB"],
                "ILC_COL2_Train_pass": col2["Train_pass"],
                "ILC_COL2_Test_pass": col2["Test_pass"],
                "ILC_COL2_Both_pass": col2["Both_pass"],
                "ILC_COL2_rank": col2["rank"],
                "ILC_COL2_condition_number": col2["condition_number"],
                "Joint_Final_pass": joint,
                "finite": bool(end["finite"] and col2["finite"]),
            }
        )
    frame = pd.DataFrame(rows).sort_values("State_ID").reset_index(drop=True)
    if len(frame) != STATE_COUNT or frame["State_ID"].tolist() != list(range(STATE_COUNT)):
        raise RuntimeError("All425 coverage frame is not canonical")
    if not np.all(np.isfinite(theta_end)) or not np.all(np.isfinite(theta_col2)):
        raise RuntimeError("All425 coefficient arrays contain non-finite values")
    return frame, theta_end, theta_col2


def _summary_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for behavior, prefix in ((ILC_END, "ILC_END"), (ILC_COL2, "ILC_COL2")):
        train = frame[f"{prefix}_Train_NMSE_dB"].to_numpy(dtype=float)
        test = frame[f"{prefix}_Test_NMSE_dB"].to_numpy(dtype=float)
        gap = frame[f"{prefix}_Gap_dB"].to_numpy(dtype=float)
        train_stats = summary_stats(train)
        test_stats = summary_stats(test)
        rows.append(
            {
                "record_type": "behavior",
                "behavior": behavior,
                **{f"Train_{key}": value for key, value in train_stats.items()},
                **{f"Test_{key}": value for key, value in test_stats.items()},
                "Both_pass_count": int(frame[f"{prefix}_Both_pass"].sum()),
                "Both_pass_rate": float(frame[f"{prefix}_Both_pass"].mean()),
                "Gap_median_dB": float(np.median(gap)),
                "Gap_mean_dB": float(np.mean(gap)),
                "Gap_Q90_dB": float(np.quantile(gap, 0.90)),
                "Gap_Q95_dB": float(np.quantile(gap, 0.95)),
                "Gap_Q99_dB": float(np.quantile(gap, 0.99)),
                "Gap_min_dB": float(np.min(gap)),
                "Gap_max_dB": float(np.max(gap)),
            }
        )
    rows.append(
        {
            "record_type": "joint",
            "behavior": "JOINT_FINAL",
            "Joint_Final_pass_count": int(frame["Joint_Final_pass"].sum()),
            "Joint_Final_pass_rate": float(frame["Joint_Final_pass"].mean()),
        }
    )
    return pd.DataFrame(rows)


def _write_plot(frame: pd.DataFrame, behavior: str, path: Path, ylim: tuple[float, float]) -> None:
    prefix = "ILC_END" if behavior == ILC_END else "ILC_COL2"
    x = frame["State_ID"].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=300)
    ax.plot(x, frame[f"{prefix}_Train_NMSE_dB"], linewidth=0.9, label="Train", color="#1f77b4")
    ax.plot(x, frame[f"{prefix}_Test_NMSE_dB"], linewidth=0.9, label="Test", color="#d62728")
    ax.axhline(TARGET_NMSE_DB, color="#222222", linestyle="--", linewidth=1.0, label="-40 dB threshold")
    ax.set_xlim(0, STATE_COUNT - 1)
    ax.set_ylim(*ylim)
    ax.set_xticks([0, 50, 100, 150, 200, 250, 300, 350, 400, 424])
    ax.set_xlabel("Load State Index")
    ax.set_ylabel("NMSE (dB)")
    ax.set_title(f"Envelope19-C2EndShared: {behavior} Modeling and Generalization")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_summary(
    frame: pd.DataFrame,
    hard20_regression: dict[str, object],
    preflight: pd.DataFrame,
    raw_before: dict[str, object],
    raw_after: dict[str, object],
) -> None:
    lines = [
        f"Task: {TASK_NAME}",
        "Mode: evaluation_only",
        f"Frozen model: {EXPECTED_MODEL_NAME}",
        "K = 19",
        "dmax = 2",
        "lambda = 0",
        "solver = OLS",
        "State range: 0...424",
        f"support_hash: {EXPECTED_SUPPORT_HASH}",
        "",
    ]
    summary = _summary_rows(frame)
    for _, row in summary.loc[summary["record_type"] == "behavior"].iterrows():
        behavior = row["behavior"]
        lines.extend(
            [
                str(behavior),
                f"Train: {int(row['Train_pass_count'])}/425; median={row['Train_median_dB']:.9f} dB; mean={row['Train_mean_dB']:.9f} dB; Q90={row['Train_Q90_dB']:.9f} dB; Q95={row['Train_Q95_dB']:.9f} dB; Q99={row['Train_Q99_dB']:.9f} dB; worst={row['Train_worst_dB']:.9f} dB",
                f"Test: {int(row['Test_pass_count'])}/425; median={row['Test_median_dB']:.9f} dB; mean={row['Test_mean_dB']:.9f} dB; Q90={row['Test_Q90_dB']:.9f} dB; Q95={row['Test_Q95_dB']:.9f} dB; Q99={row['Test_Q99_dB']:.9f} dB; worst={row['Test_worst_dB']:.9f} dB",
                f"Train+Test both pass: {int(row['Both_pass_count'])}/425",
                f"Test-Train gap: median={row['Gap_median_dB']:.9f} dB; Q95={row['Gap_Q95_dB']:.9f} dB; min={row['Gap_min_dB']:.9f} dB; max={row['Gap_max_dB']:.9f} dB",
                "",
            ]
        )
    joint = summary.loc[summary["record_type"] == "joint"].iloc[0]
    lines.extend(
        [
            f"Joint Final Pass: {int(joint['Joint_Final_pass_count'])}/425",
            "A State counts as Joint Final Pass only if ILC_END Train/Test and ILC_COL2 Train/Test are all strictly below -40 dB.",
            "",
            f"valid_ilc_count distribution: {preflight.groupby('valid_ilc_count').size().to_dict()}",
            f"ilc_end column distribution: {preflight['ilc_end_matlab_column'].value_counts().sort_index().to_dict()}",
            f"col2_equals_ilc_end count: {int(preflight['ilc_end_python_index'].eq(1).sum())}",
            "",
            f"Hard20 regression pass: {hard20_regression['pass']}",
            f"Hard20 maximum absolute deltas: {json.dumps(hard20_regression, ensure_ascii=False, sort_keys=True)}",
            "No basis selection, CV, Ridge, ABC, common-B, LUT, fingerprint, Top-1, clustering, DPD replay, low-bandwidth, or downstream retrieval was run.",
            f"Raw snapshot before: {json.dumps(raw_before, ensure_ascii=False, sort_keys=True)}",
            f"Raw snapshot after: {json.dumps(raw_after, ensure_ascii=False, sort_keys=True)}",
            f"raw_data_modified: {raw_before != raw_after}",
        ]
    )
    (RESULT_ROOT / "10_final_result_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run(*, resume: bool = False) -> dict[str, object]:
    existing = RESULT_ROOT.exists() and any(RESULT_ROOT.iterdir())
    if existing and not resume:
        raise RuntimeError("result directory is not empty; use --resume for this task")
    if resume:
        if not CHECKPOINT_PATH.is_file():
            raise RuntimeError("--resume requires an existing checkpoint")
        checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        for key, expected in {
            "task_name": TASK_NAME,
            "mode": "evaluation_only",
            "model_name": EXPECTED_MODEL_NAME,
            "K": 19,
            "dmax": 2,
            "lambda": 0.0,
            "solver": "OLS",
            "state_count": STATE_COUNT,
            "worker_count": WORKER_COUNT,
            "blas_threads_per_worker": 1,
            "basis_selection_performed": False,
            "ridge_scan_performed": False,
            "cv_performed": False,
            "lut_retrieval_performed": False,
        }.items():
            if checkpoint.get(key) != expected:
                raise RuntimeError(f"resume checkpoint mismatch at {key}")
        if checkpoint.get("support_hash") != EXPECTED_SUPPORT_HASH:
            raise RuntimeError("resume checkpoint support hash mismatch")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    raw_before = raw_manifest_gate()
    terms, support, support_ids, support_digest, model_frame = load_frozen_model()
    if support_digest != EXPECTED_SUPPORT_HASH or len(support) != 19:
        raise RuntimeError("frozen support contract failed")
    _write_task_definition(model_frame)
    _write_frozen_definition(model_frame)
    _write_checkpoint(
        phase="setup_complete",
        support_hash=EXPECTED_SUPPORT_HASH,
        raw_manifest_before=raw_before,
        raw_manifest_after=None,
        completed_state_ids=[],
        hard20_regression_pass=None,
        model_frozen=True,
        test_unlocked=True,
    )
    _append_log(
        WORK_LOG,
        "\n"
        f"[{_now()}] Start {TASK_NAME}\n"
        f"Frozen model={EXPECTED_MODEL_NAME}; support_hash={EXPECTED_SUPPORT_HASH}; states=0...424; workers={WORKER_COUNT}; raw={json.dumps(raw_before, sort_keys=True)}\n"
        "Mode: evaluation_only; no selection/CV/Ridge/ABC/LUT path.\n",
    )
    context = get_context("spawn")
    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(
        max_workers=WORKER_COUNT,
        mp_context=context,
        initializer=_worker_init,
        initargs=(tuple(support),),
    ) as executor:
        futures = {
            executor.submit(_worker_entry, state_id): state_id
            for state_id in range(STATE_COUNT)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed % 50 == 0 or completed == STATE_COUNT:
                _write_checkpoint(
                    phase="evaluation_progress",
                    support_hash=EXPECTED_SUPPORT_HASH,
                    raw_manifest_before=raw_before,
                    raw_manifest_after=None,
                    completed_state_ids=sorted(int(item["State_ID"]) for item in results),
                    hard20_regression_pass=None,
                    model_frozen=True,
                    test_unlocked=True,
                )
                print(f"[ALL425 EVALUATION] {completed}/{STATE_COUNT}", flush=True)
    results.sort(key=lambda item: int(item["State_ID"]))
    state_table = {int(row["state_id"]): row for row in build_state_table()}
    preflight = _combine_preflight(
        [row for item in results for row in item["preflight"]],
        state_table,
    )
    preflight.to_csv(RESULT_ROOT / "02_all425_data_preflight.csv", index=False)
    coverage, theta_end, theta_col2 = _build_coverage_frame(results)
    coverage.to_csv(RESULT_ROOT / "03_all425_dual_behavior_coverage.csv", index=False)
    summary = _summary_rows(coverage)
    summary.to_csv(RESULT_ROOT / "04_all425_summary.csv", index=False)
    failed_mask = ~coverage[[
        "ILC_END_Train_pass",
        "ILC_END_Test_pass",
        "ILC_COL2_Train_pass",
        "ILC_COL2_Test_pass",
    ]].all(axis=1)
    failed = coverage.loc[failed_mask].copy()
    failed["failed_conditions"] = failed.apply(
        lambda row: ";".join(
            condition
            for condition, passed in (
                ("ILC_END_Train", row["ILC_END_Train_pass"]),
                ("ILC_END_Test", row["ILC_END_Test_pass"]),
                ("ILC_COL2_Train", row["ILC_COL2_Train_pass"]),
                ("ILC_COL2_Test", row["ILC_COL2_Test_pass"]),
            )
            if not bool(passed)
        ),
        axis=1,
    )
    failed.to_csv(RESULT_ROOT / "05_failed_states.csv", index=False)
    np.savez(
        RESULT_ROOT / "06_final_dual_behavior_coefficients.npz",
        state_ids=np.arange(STATE_COUNT, dtype=np.int64),
        theta_end=theta_end,
        theta_col2=theta_col2,
        basis_ids=np.asarray(support_ids),
        K=np.asarray(19),
        dmax=np.asarray(DMAX),
        lambda_value=np.asarray(0.0),
        support_hash=np.asarray(EXPECTED_SUPPORT_HASH),
        ilc_end_columns=preflight["ilc_end_matlab_column"].to_numpy(dtype=np.int64),
    )

    if not PREVIOUS_HARD20_RESULT.is_file():
        raise FileNotFoundError(f"previous Hard20 dual-behavior result is missing: {PREVIOUS_HARD20_RESULT}")
    previous = pd.read_csv(PREVIOUS_HARD20_RESULT)
    old_hard = previous.loc[previous["State_ID"].isin(EXPECTED_HARD20_IDS)].copy()
    new_hard = coverage.loc[coverage["State_ID"].isin(EXPECTED_HARD20_IDS)].copy()
    regression_rows = []
    errors = []
    for state_id in EXPECTED_HARD20_IDS:
        old_rows = old_hard.loc[old_hard["State_ID"] == state_id].set_index("behavior")
        new_row = new_hard.loc[new_hard["State_ID"] == state_id].iloc[0]
        for behavior, prefix in ((ILC_END, "ILC_END"), (ILC_COL2, "ILC_COL2")):
            old = old_rows.loc[behavior]
            for old_col, new_col, label in (
                ("Train_NMSE_dB", f"{prefix}_Train_NMSE_dB", "train"),
                ("Test_NMSE_dB", f"{prefix}_Test_NMSE_dB", "test"),
            ):
                delta = float(new_row[new_col] - old[old_col])
                errors.append(abs(delta))
                regression_rows.append(
                    {
                        "State_ID": state_id,
                        "behavior": behavior,
                        "metric": label,
                        "old_value_dB": float(old[old_col]),
                        "new_value_dB": float(new_row[new_col]),
                        "delta_dB": delta,
                        "regression_pass": abs(delta) < 1e-8,
                    }
                )
    regression = pd.DataFrame(regression_rows)
    regression.to_csv(RESULT_ROOT / "07_hard20_regression_check.csv", index=False)
    regression_gate = {
        "pass": bool(regression["regression_pass"].all()),
        "max_abs_delta_dB": float(max(errors)),
        "row_count": len(regression),
    }
    if not regression_gate["pass"]:
        raise RuntimeError(f"Hard20 regression failed: {regression_gate}")

    all_values = np.concatenate(
        [coverage[f"{prefix}_{metric}_NMSE_dB"].to_numpy(dtype=float) for prefix in ("ILC_END", "ILC_COL2") for metric in ("Train", "Test")]
    )
    y_limits = (float(np.min(all_values) - 0.5), float(np.max(all_values) + 0.5))
    _write_plot(coverage, ILC_END, RESULT_ROOT / "08_ilcend_all425_train_test.png", y_limits)
    _write_plot(coverage, ILC_COL2, RESULT_ROOT / "09_ilccol2_all425_train_test.png", y_limits)
    raw_after = raw_manifest_gate()
    if raw_after != raw_before:
        raise RuntimeError("raw manifest changed during All425 evaluation")
    _write_summary(coverage, regression_gate, preflight, raw_before, raw_after)
    _write_checkpoint(
        phase="completed",
        support_hash=EXPECTED_SUPPORT_HASH,
        raw_manifest_before=raw_before,
        raw_manifest_after=raw_after,
        completed_state_ids=list(range(STATE_COUNT)),
        hard20_regression_pass=regression_gate,
        model_frozen=True,
        test_unlocked=True,
    )
    return {
        "task": TASK_NAME,
        "status": "SUCCESS",
        "state_count": STATE_COUNT,
        "ILC_END_Train_pass": int(coverage["ILC_END_Train_pass"].sum()),
        "ILC_END_Test_pass": int(coverage["ILC_END_Test_pass"].sum()),
        "ILC_COL2_Train_pass": int(coverage["ILC_COL2_Train_pass"].sum()),
        "ILC_COL2_Test_pass": int(coverage["ILC_COL2_Test_pass"].sum()),
        "Joint_Final_pass": int(coverage["Joint_Final_pass"].sum()),
        "failed_state_count": int(len(failed)),
        "hard20_regression": regression_gate,
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
        f"Result: {json.dumps(result, ensure_ascii=False, sort_keys=True)}\n"
        "Evaluation-only boundary preserved; no selection, CV, Ridge, ABC, LUT, or retrieval was run.\n",
    )
    _append_log(
        HANDOFF_LOG,
        f"\n{_now()} | 新任务：{TASK_NAME}\n"
        f"完成 Frozen Envelope19-C2EndShared 的 All425 双行为 coverage evaluation；结果目录：{RESULT_ROOT}\n"
        f"结果摘要：{json.dumps(result, ensure_ascii=False, sort_keys=True)}\n"
        "仅评价 ILC_END/ILC_COL2 的 Train/Test；未重新选 support、未运行 CV/Ridge/ABC/LUT/retrieval。\n",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    __import__("multiprocessing").freeze_support()
    main()
