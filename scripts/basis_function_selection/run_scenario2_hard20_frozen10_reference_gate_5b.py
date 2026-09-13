"""Run the Train-only Frozen10 reference gate on the frozen Hard-20 states."""

# ruff: noqa: E402

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from multiprocessing import freeze_support
from pathlib import Path
from typing import Any

for _thread_variable in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_variable] = "1"

import numpy as np
import pandas as pd

MODULE_ROOT = Path(__file__).resolve().parent
SCRIPTS_ROOT = MODULE_ROOT.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from basis_function_selection.config import TARGET_NMSE_DB
from basis_function_selection.data_preparation import prepare_train_only_state
from basis_function_selection.reference_gate import (
    FROZEN_BASES,
    REFERENCE_MODELS,
    evaluate_reference_state,
    formal_ridge_gate,
    frozen135_subset_gate,
    validate_reference_definitions,
)

EXPERIMENT_NAME = "scenario_2_hard20_frozen10_reference_gate_5B"
RESULT_PARENT = PROJECT_ROOT / "results" / "basis_function_selection"
PRIMARY_RESULT_ROOT = RESULT_PARENT / EXPERIMENT_NAME
PRIOR_RESULT_ROOT = RESULT_PARENT / "scenario_2_hard20_volterra_basis_selection_5B"
PRIOR_RANKING = PRIOR_RESULT_ROOT / "01_hard20_state_ranking.csv"
PRIOR_FULL81 = PRIOR_RESULT_ROOT / "05_hard20_inner_cv_metrics.csv"
PRIOR_SUMMARY = PRIOR_RESULT_ROOT / "07_final_result_summary.txt"
WORK_LOG = PROJECT_ROOT / "work_logs" / EXPERIMENT_NAME / "execution_log.txt"
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "codex_handoff" / "codex_handoff.txt"
WORKER_COUNT = 12
EXPECTED_HARD20 = (
    325,
    332,
    333,
    334,
    336,
    353,
    324,
    338,
    350,
    326,
    349,
    339,
    351,
    342,
    328,
    323,
    355,
    331,
    327,
    356,
)
EXPECTED_OUTPUTS = (
    "01_reference_model_definitions.csv",
    "02_hard20_reference_metrics.csv",
    "03_reference_model_comparison.csv",
    "04_final_result_summary.txt",
)
PROTECTED_PATHS = (
    PROJECT_ROOT / "data" / "raw",
    PRIOR_RESULT_ROOT,
    PROJECT_ROOT / "results" / "behavior_model",
    PROJECT_ROOT / "results" / "behavior_fingerprint_lut_retrieval",
    PROJECT_ROOT / "results" / "clustering",
)


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def _fast_manifest(path: Path) -> dict[str, object]:
    if not path.is_dir():
        return {"exists": False, "file_count": 0, "bytes": 0, "latest_mtime_ns": None}
    files = [item for item in path.rglob("*") if item.is_file()]
    stats = [item.stat() for item in files]
    return {
        "exists": True,
        "file_count": len(files),
        "bytes": int(sum(item.st_size for item in stats)),
        "latest_mtime_ns": max((item.st_mtime_ns for item in stats), default=None),
    }


def _protection_snapshot() -> dict[str, dict[str, object]]:
    return {str(path): _fast_manifest(path) for path in PROTECTED_PATHS}


def _select_result_root() -> Path:
    RESULT_PARENT.mkdir(parents=True, exist_ok=True)
    if not PRIMARY_RESULT_ROOT.exists() or not any(PRIMARY_RESULT_ROOT.iterdir()):
        PRIMARY_RESULT_ROOT.mkdir(parents=True, exist_ok=True)
        return PRIMARY_RESULT_ROOT
    suffix = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    rerun = RESULT_PARENT / f"{EXPERIMENT_NAME}_rerun_{suffix}"
    rerun.mkdir(parents=True, exist_ok=False)
    return rerun


def _load_prior_inputs() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    for path in (PRIOR_RANKING, PRIOR_FULL81, PRIOR_SUMMARY):
        if not path.is_file():
            raise FileNotFoundError(f"Required prior result is missing: {path}")
    ranking = pd.read_csv(PRIOR_RANKING)
    full81 = pd.read_csv(PRIOR_FULL81)
    summary = PRIOR_SUMMARY.read_text(encoding="utf-8")
    required_ranking = {
        "state_id",
        "train_noDPD_NMSE_dB",
        "rank",
        "is_hard20",
    }
    required_full81 = {
        "state_id",
        "train_nmse_db",
        "cv1_nmse_db",
        "cv2_nmse_db",
        "cv3_nmse_db",
        "W_dB",
    }
    if not required_ranking.issubset(ranking.columns):
        raise RuntimeError("Prior ranking schema is incomplete")
    if not required_full81.issubset(full81.columns):
        raise RuntimeError("Prior Full81 schema is incomplete")
    hard_mask = ranking["is_hard20"].map(
        lambda value: value is True or str(value).strip().lower() == "true"
    )
    hard_ids = tuple(ranking.loc[hard_mask].sort_values("rank")["state_id"].astype(int).tolist())
    if hard_ids != EXPECTED_HARD20:
        raise RuntimeError(f"Frozen Hard-20 mismatch: {hard_ids}")
    if len(full81) != 20 or set(full81["state_id"].astype(int)) != set(EXPECTED_HARD20):
        raise RuntimeError("Prior Full81 metrics do not cover the frozen Hard-20 exactly")
    if "Candidate-space capacity failure" not in summary:
        raise RuntimeError("Prior summary does not contain the frozen capacity conclusion")
    return ranking, full81, summary


def _definitions_frame() -> pd.DataFrame:
    rows = []
    for model in REFERENCE_MODELS:
        bases = [FROZEN_BASES[index] for index in model.support]
        rows.append(
            {
                "model_id": model.model_id,
                "model_name": model.model_name,
                "solver": model.solver,
                "lambda": model.ridge_lambda,
                "K": model.k,
                "basis_ids": ";".join(item.basis_id for item in bases),
                "basis_formulas": ";".join(item.formula for item in bases),
                "purpose": model.purpose,
            }
        )
    return pd.DataFrame(rows)


def _reference_worker(payload: tuple[int, int, float, dict[str, object]]) -> dict[str, object]:
    state_id, hard20_rank, saved_ranking_nmse_db, saved_full81 = payload
    return evaluate_reference_state(state_id, hard20_rank, saved_ranking_nmse_db, saved_full81)


def _run_reference_models(
    ranking: pd.DataFrame,
    full81: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    ranking_by_state = ranking.set_index("state_id").to_dict("index")
    full81_by_state = full81.set_index("state_id").to_dict("index")
    payloads = [
        (
            state_id,
            int(ranking_by_state[state_id]["rank"]),
            float(ranking_by_state[state_id]["train_noDPD_NMSE_dB"]),
            dict(full81_by_state[state_id]),
        )
        for state_id in EXPECTED_HARD20
    ]
    results: dict[int, dict[str, object]] = {}
    with ProcessPoolExecutor(max_workers=WORKER_COUNT) as executor:
        futures = {executor.submit(_reference_worker, payload): payload[0] for payload in payloads}
        for done, future in enumerate(as_completed(futures), start=1):
            state_id = futures[future]
            results[state_id] = future.result()
            print(f"[REFERENCE] {done}/20", flush=True)
    ordered = [results[state_id] for state_id in EXPECTED_HARD20]
    rows = [row for result in ordered for row in result["rows"]]
    frame = pd.DataFrame(rows).sort_values(["hard20_rank", "model_id"]).reset_index(drop=True)
    if len(frame) != 100:
        raise RuntimeError(f"Reference metric row count must be 100, got {len(frame)}")
    return frame, ordered


def _aggregate_reference_metrics(
    metrics: pd.DataFrame,
    full81: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    full81_train = full81["train_nmse_db"].to_numpy(dtype=np.float64)
    full81_w = full81["W_dB"].to_numpy(dtype=np.float64)
    rows.append(
        {
            "model_id": "FULL81",
            "K": 81,
            "lambda": 0.0,
            "train_mean_db": float(np.mean(full81_train)),
            "train_median_db": float(np.median(full81_train)),
            "train_best_db": float(np.min(full81_train)),
            "train_worst_db": float(np.max(full81_train)),
            "W_mean_db": float(np.mean(full81_w)),
            "W_median_db": float(np.median(full81_w)),
            "W_best_db": float(np.min(full81_w)),
            "W_worst_db": float(np.max(full81_w)),
            "N_train40": int(np.count_nonzero(full81_train < TARGET_NMSE_DB)),
            "N_W40": int(np.count_nonzero(full81_w < TARGET_NMSE_DB)),
            "median_delta_train_vs_full81_db": 0.0,
            "mean_delta_train_vs_full81_db": 0.0,
            "min_delta_train_vs_full81_db": 0.0,
            "max_delta_train_vs_full81_db": 0.0,
            "states_better_train_than_full81": 0,
            "median_delta_W_vs_full81_db": 0.0,
            "states_better_W_than_full81": 0,
        }
    )
    for model in REFERENCE_MODELS:
        group = metrics.loc[metrics["model_id"] == model.model_id]
        if len(group) != 20:
            raise RuntimeError(f"{model.model_id} does not contain 20 states")
        train = group["train_nmse_db"].to_numpy(dtype=np.float64)
        w_values = group["W_db"].to_numpy(dtype=np.float64)
        delta_train = group["delta_train_vs_full81_db"].to_numpy(dtype=np.float64)
        delta_w = group["delta_W_vs_full81_db"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "model_id": model.model_id,
                "K": model.k,
                "lambda": model.ridge_lambda,
                "train_mean_db": float(np.mean(train)),
                "train_median_db": float(np.median(train)),
                "train_best_db": float(np.min(train)),
                "train_worst_db": float(np.max(train)),
                "W_mean_db": float(np.mean(w_values)),
                "W_median_db": float(np.median(w_values)),
                "W_best_db": float(np.min(w_values)),
                "W_worst_db": float(np.max(w_values)),
                "N_train40": int(np.count_nonzero(train < TARGET_NMSE_DB)),
                "N_W40": int(np.count_nonzero(w_values < TARGET_NMSE_DB)),
                "median_delta_train_vs_full81_db": float(np.median(delta_train)),
                "mean_delta_train_vs_full81_db": float(np.mean(delta_train)),
                "min_delta_train_vs_full81_db": float(np.min(delta_train)),
                "max_delta_train_vs_full81_db": float(np.max(delta_train)),
                "states_better_train_than_full81": int(np.count_nonzero(delta_train < 0.0)),
                "median_delta_W_vs_full81_db": float(np.median(delta_w)),
                "states_better_W_than_full81": int(np.count_nonzero(delta_w < 0.0)),
            }
        )
    frame = pd.DataFrame(rows)
    baseline = metrics.loc[metrics["model_id"] == "M1", ["state_id", "train_nmse_db", "W_db"]]
    baseline = baseline.rename(columns={"train_nmse_db": "m1_train", "W_db": "m1_w"})
    for model_id in ("M2", "M3", "M4"):
        selected = metrics.loc[
            metrics["model_id"] == model_id,
            ["state_id", "train_nmse_db", "W_db"],
        ]
        joined = selected.merge(baseline, on="state_id", validate="one_to_one")
        frame.loc[frame["model_id"] == model_id, "median_delta_train_vs_M1_db"] = float(
            np.median(joined["train_nmse_db"] - joined["m1_train"])
        )
        frame.loc[frame["model_id"] == model_id, "median_delta_W_vs_M1_db"] = float(
            np.median(joined["W_db"] - joined["m1_w"])
        )
    return frame


def _ablation_effect(metrics: pd.DataFrame, model_id: str) -> dict[str, float]:
    baseline = metrics.loc[metrics["model_id"] == "M1", ["state_id", "train_nmse_db", "W_db"]]
    selected = metrics.loc[
        metrics["model_id"] == model_id,
        ["state_id", "train_nmse_db", "W_db"],
    ]
    joined = selected.merge(
        baseline, on="state_id", suffixes=("_model", "_m1"), validate="one_to_one"
    )
    return {
        "median_train_delta_db": float(
            np.median(joined["train_nmse_db_model"] - joined["train_nmse_db_m1"])
        ),
        "mean_train_delta_db": float(
            np.mean(joined["train_nmse_db_model"] - joined["train_nmse_db_m1"])
        ),
        "median_W_delta_db": float(np.median(joined["W_db_model"] - joined["W_db_m1"])),
        "mean_W_delta_db": float(np.mean(joined["W_db_model"] - joined["W_db_m1"])),
    }


def _summary_text(
    comparison: pd.DataFrame,
    diagnostics: list[dict[str, object]],
    gates: dict[str, object],
    elapsed_sec: float,
    result_root: Path,
) -> str:
    by_model = comparison.set_index("model_id")
    effects = {
        "P2_M2_minus_M1": _ablation_effect(gates["metrics_frame"], "M2"),
        "P79_M3_minus_M1": _ablation_effect(gates["metrics_frame"], "M3"),
        "P2_P79_M4_minus_M1": _ablation_effect(gates["metrics_frame"], "M4"),
    }
    m4 = by_model.loc["M4"]
    m4_train_delta = float(m4["median_delta_train_vs_full81_db"])
    m4_w_delta = float(m4["median_delta_W_vs_full81_db"])
    if m4_train_delta < 0.0:
        interpretation = (
            "Frozen10 OLS is better than Full81 on median Train under the same data definition. "
            "The ablation values identify whether P2, P7/P9, or both provide the advantage."
        )
    elif m4_train_delta > 0.0:
        interpretation = (
            "Full81 remains better than Frozen10 OLS on median Train. The current failure cannot "
            "be attributed to omission of P2/P7/P9 alone."
        )
    else:
        interpretation = (
            "Frozen10 OLS and Full81 have equal median Train performance at saved precision."
        )
    lines = [
        f"Experiment: {EXPERIMENT_NAME}",
        "Purpose: Compare the formal Frozen MP basis family with the strict Full81 1/3/5-order "
        "Volterra space under the exact same Hard-20 and Train definition.",
        "Data: xin -> yout_withoutdpd_ori",
        "Preprocessing: FULL rough align -> FULL fine align -> Train [0:16384) -> Train-local gain",
        "Hard-20: " + ", ".join(str(value) for value in EXPECTED_HARD20),
        "Test evaluated: False",
        "Worker mode: 12 processes; BLAS threads per worker=1; GPU unused",
        "Reference models: M1 Frozen135 OLS; M2 M1+P2 OLS; M3 M1+P7/P9 OLS; "
        "M4 Frozen10 OLS; M5 Frozen10 raw-basis Ridge lambda=1e-8",
        "",
    ]
    for model_id in ("FULL81", "M1", "M2", "M3", "M4", "M5"):
        row = by_model.loc[model_id]
        lines.extend(
            [
                f"[{model_id}]",
                f"K={int(row['K'])}, lambda={float(row['lambda']):.17g}",
                f"Train range=[{float(row['train_best_db']):.9f}, "
                f"{float(row['train_worst_db']):.9f}] dB",
                f"Train mean={float(row['train_mean_db']):.9f} dB",
                f"Train median={float(row['train_median_db']):.9f} dB",
                f"W range=[{float(row['W_best_db']):.9f}, {float(row['W_worst_db']):.9f}] dB",
                f"W mean={float(row['W_mean_db']):.9f} dB",
                f"W median={float(row['W_median_db']):.9f} dB",
                f"Train pass40={int(row['N_train40'])}/20",
                f"W pass40={int(row['N_W40'])}/20",
                "",
            ]
        )
    lines.extend(
        [
            "[Ablation effects; negative delta means improvement versus M1]",
            json.dumps(effects, ensure_ascii=False),
            "",
            "[Frozen10 OLS vs Full81]",
            f"Median Train delta={m4_train_delta:.9f} dB",
            f"Median W delta={m4_w_delta:.9f} dB",
            f"States with better Frozen10 Train={int(m4['states_better_train_than_full81'])}/20",
            f"States with better Frozen10 W={int(m4['states_better_W_than_full81'])}/20",
            "",
            "[Gates]",
            "Hard-20 exact match: True",
            "Train definition reproduction max absolute delta: "
            f"{max(float(item['ranking_abs_delta_db']) for item in diagnostics):.3e} dB",
            "Full81 reproduction max absolute delta: "
            f"{max(float(item['full81_max_abs_delta_db']) for item in diagnostics):.3e} dB",
            "Frozen135 subset relative SSE excess maximum: "
            f"{max(float(item['full81_subset_relative_sse_excess']) for item in diagnostics):.3e}",
            f"Reference definition gate: {gates['reference_definitions']}",
            f"Frozen135 subset mapping gate: {gates['subset_mapping']}",
            f"Formal Ridge gate: {gates['ridge']}",
            "Test access in reference runner: False",
            "",
            "[Conclusion]",
            interpretation,
            "Correlation or ablation differences are descriptive and do not prove "
            "physical causality.",
            "No basis selection, Forward, Backward, Pair Rescue, Swap, lambda scan, Test, "
            "405-state validation, LUT retrieval, Real-B, or DPD replay was performed.",
            f"Runtime seconds: {elapsed_sec:.3f}",
            f"Result root: {result_root}",
        ]
    )
    return "\n".join(lines) + "\n"


def _engineering_checks() -> dict[str, object]:
    python = sys.executable
    commands = {
        "compileall": [python, "-m", "compileall", "-q", str(MODULE_ROOT)],
        "ruff": [python, "-m", "ruff", "check", "--no-cache", str(MODULE_ROOT)],
        "pip_check": [python, "-m", "pip", "check"],
        "git_diff_check": ["git", "diff", "--check"],
        "core_regression": [python, "-B", str(SCRIPTS_ROOT / "_core" / "test_core_modules.py")],
        "data_manager_regression": [
            python,
            "-B",
            str(SCRIPTS_ROOT / "data_manager" / "test_data_manager.py"),
        ],
        "signal_segmentation_regression": [
            python,
            "-B",
            str(SCRIPTS_ROOT / "signal_segmentation" / "test_signal_segmentation.py"),
        ],
    }
    results: dict[str, object] = {}
    for name, command in commands.items():
        completed = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        results[name] = {
            "returncode": completed.returncode,
            "stdout_tail": completed.stdout[-800:],
            "stderr_tail": completed.stderr[-800:],
        }
        if completed.returncode != 0:
            raise RuntimeError(f"Engineering check failed: {name}: {results[name]}")
    return results


def main() -> None:
    freeze_support()
    started = time.perf_counter()
    before = _protection_snapshot()
    result_root = _select_result_root()
    _append_log(
        WORK_LOG,
        f"\n\n[{_timestamp()}] START {EXPERIMENT_NAME}\n"
        f"Result root: {result_root}\nPython: {sys.executable}\n"
        "Prior Full81 result is read-only. Test is forbidden.\n",
    )
    try:
        ranking, full81, _ = _load_prior_inputs()
        reference_definitions = validate_reference_definitions()
        subset_mapping = frozen135_subset_gate()
        ranking_by_state = ranking.set_index("state_id")
        for state_id in (325, 340, 356):
            prepared = prepare_train_only_state(state_id)
            direct_nmse = float(
                10
                * np.log10(
                    np.sum(np.abs(prepared.x_train - prepared.y_train_adjusted) ** 2)
                    / np.sum(np.abs(prepared.x_train) ** 2)
                )
            )
            saved_nmse = float(ranking_by_state.loc[state_id, "train_noDPD_NMSE_dB"])
            if abs(direct_nmse - saved_nmse) > 1e-9:
                raise RuntimeError(
                    f"Representative Train definition gate failed for state {state_id}"
                )
        ridge_gate = formal_ridge_gate(325)

        metrics, diagnostics = _run_reference_models(ranking, full81)
        comparison = _aggregate_reference_metrics(metrics, full81)
        definitions = _definitions_frame()

        definitions.to_csv(result_root / EXPECTED_OUTPUTS[0], index=False, encoding="utf-8-sig")
        metrics.to_csv(result_root / EXPECTED_OUTPUTS[1], index=False, encoding="utf-8-sig")
        comparison.to_csv(result_root / EXPECTED_OUTPUTS[2], index=False, encoding="utf-8-sig")

        gates: dict[str, Any] = {
            "reference_definitions": reference_definitions,
            "subset_mapping": subset_mapping,
            "ridge": ridge_gate,
            "metrics_frame": metrics,
        }
        elapsed = time.perf_counter() - started
        summary = _summary_text(comparison, diagnostics, gates, elapsed, result_root)
        (result_root / EXPECTED_OUTPUTS[3]).write_text(summary, encoding="utf-8")

        engineering = _engineering_checks()
        after = _protection_snapshot()
        if before != after:
            raise RuntimeError("Protected raw/history metadata changed during the reference gate")
        files = tuple(sorted(item.name for item in result_root.iterdir() if item.is_file()))
        if files != tuple(sorted(EXPECTED_OUTPUTS)):
            raise RuntimeError(f"Formal output-file gate failed: {files}")
        indexed_comparison = comparison.set_index("model_id")
        m4_median = float(indexed_comparison.loc["M4", "train_median_db"])
        full81_median = float(indexed_comparison.loc["FULL81", "train_median_db"])
        log_summary = (
            f"Completed {EXPERIMENT_NAME}: Hard20 exact, models=5, rows={len(metrics)}, "
            f"M4 median Train={m4_median:.9f} dB, "
            f"Full81 median Train={full81_median:.9f} dB, "
            f"Test not accessed, workers={WORKER_COUNT}, result={result_root}, "
            f"elapsed={elapsed:.3f}s, engineering={engineering}.\n"
        )
        _append_log(WORK_LOG, f"[{_timestamp()}] COMPLETE\n{log_summary}")
        _append_log(
            HANDOFF_LOG,
            f"\n\n{_timestamp()} | basis_function_selection Frozen10 reference gate\n{log_summary}",
        )
        print("[COMPLETE] Frozen10 reference gate passed", flush=True)
        print(f"[COMPLETE] Result: {result_root}", flush=True)
    except Exception as exc:
        _append_log(
            WORK_LOG,
            f"[{_timestamp()}] FAILED\n{type(exc).__name__}: {exc}\n"
            "No Test, raw-data mutation, history overwrite, or Git cleanup was performed.\n",
        )
        raise


if __name__ == "__main__":
    main()
