"""Run the frozen Hard-20 unified sparse strict-Volterra basis-selection task."""

# ruff: noqa: E402

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Sequence
from datetime import datetime
from multiprocessing import freeze_support
from pathlib import Path

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
PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from behavior_modeling.shared.basis_function_selection.config import (
    CANDIDATE_COUNT,
    DMAX,
    EXPECTED_OUTPUT_FILENAMES,
    EXPERIMENT_NAME,
    HANDOFF_LOG,
    HARD_STATE_COUNT,
    OLS_CONDITION_HARD_LIMIT,
    ORDERS,
    PREPROCESS_WORKERS,
    PRIMARY_RESULT_ROOT,
    RESULT_PARENT,
    TARGET_NMSE_DB,
    TEST_END,
    TEST_LENGTH,
    TEST_START,
    TRAIN_END,
    TRAIN_LENGTH,
    WORK_LOG,
    WORKER_NEAR_TIE_FRACTION,
    available_benchmark_workers,
    logical_cpu_count,
)
from behavior_modeling.shared.basis_function_selection.data_preparation import (
    HardCacheSpec,
    create_hard_cache,
    load_hard_cache_arrays,
    populate_hard_cache,
    preprocess_all_for_ranking,
)
from behavior_modeling.shared.basis_function_selection.final_evaluation import (
    assert_test_access_allowed,
    evaluate_final_model,
)
from behavior_modeling.shared.basis_function_selection.hard_state_selection import (
    select_hard_states,
)
from behavior_modeling.shared.basis_function_selection.model_solver import (
    fit_ridge,
    ols_scaling_equivalence_gate,
)
from behavior_modeling.shared.basis_function_selection.ridge_tuning import (
    RidgeTuningResult,
    tune_ridge,
)
from behavior_modeling.shared.basis_function_selection.search import (
    PersistentStateEvaluator,
    SearchResult,
    run_sparse_search,
)
from behavior_modeling.shared.basis_function_selection.selection_metrics import SupportScore
from behavior_modeling.shared.basis_function_selection.volterra_dictionary import (
    VolterraBasis,
    build_dictionary,
    dictionary_gate,
    support_ids,
)

PROTECTED_RESULT_DIRS = (
    PROJECT_ROOT / "results" / "behavior_modeling",
    PROJECT_ROOT / "results" / "behavior_fingerprint_retrieval",
    PROJECT_ROOT / "results" / "lut_clustering_compression",
    PROJECT_ROOT / "results" / "low_bandwidth_behavior_analysis",
)
OLD_RUNNER_TOKEN = "scenario_2_unified_sparse_volterra_basis_discovery_5b"


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def _check_old_process() -> None:
    command = (
        "Get-CimInstance Win32_Process | "
        "Where-Object { $_.Name -match '^python(\\.exe)?$' -and "
        f"$_.CommandLine -match '{OLD_RUNNER_TOKEN}' }} | "
        'ForEach-Object { "$($_.ProcessId)`t$($_.CommandLine)" }'
    )
    completed = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if lines:
        raise RuntimeError(
            "Old Sparse-Volterra task process is still running; it was not terminated:\n"
            + "\n".join(lines)
        )


def _fast_tree_manifest(path: Path) -> dict[str, object]:
    if not path.is_dir():
        return {"exists": False, "file_count": 0, "bytes": 0, "latest_mtime_ns": None}
    files = [item for item in path.rglob("*") if item.is_file()]
    stats = [item.stat() for item in files]
    return {
        "exists": True,
        "file_count": len(files),
        "bytes": int(sum(stat.st_size for stat in stats)),
        "latest_mtime_ns": max((stat.st_mtime_ns for stat in stats), default=None),
    }


def _protection_snapshot() -> dict[str, object]:
    return {
        "data_raw": _fast_tree_manifest(PROJECT_ROOT / "data" / "raw"),
        "protected_results": {
            str(path): _fast_tree_manifest(path) for path in PROTECTED_RESULT_DIRS
        },
    }


def _select_result_root() -> Path:
    RESULT_PARENT.mkdir(parents=True, exist_ok=True)
    if not PRIMARY_RESULT_ROOT.exists() or not any(PRIMARY_RESULT_ROOT.iterdir()):
        PRIMARY_RESULT_ROOT.mkdir(parents=True, exist_ok=True)
        return PRIMARY_RESULT_ROOT
    suffix = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    rerun = RESULT_PARENT / f"{EXPERIMENT_NAME}_rerun_{suffix}"
    rerun.mkdir(parents=True, exist_ok=False)
    return rerun


def _candidate_frame(terms: Sequence[VolterraBasis]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "basis_id": term.basis_id,
                "order": term.order,
                "nonconjugate_delays": json.dumps(term.nonconjugate_delays),
                "conjugate_delays": json.dumps(term.conjugate_delays),
                "formula": term.formula,
                "mandatory": term.mandatory,
            }
            for term in terms
        ]
    )


def _representative_alignment_gate(ranking_rows: Sequence[dict[str, object]]) -> dict[str, object]:
    by_state = {int(row["state_id"]): row for row in ranking_rows}
    states = (0, 100, 187, 340, 424)
    checks = []
    for state_id in states:
        row = by_state[state_id]
        finite = all(
            np.isfinite(float(row[key]))
            for key in (
                "train_noDPD_NMSE_dB",
                "rough_delay",
                "fine_delay",
                "train_gain_real",
                "train_gain_imag",
            )
        )
        checks.append({"state_id": state_id, "finite": bool(finite)})
    passed = all(item["finite"] for item in checks)
    if not passed:
        raise RuntimeError("Representative alignment gate failed")
    return {"states": states, "checks": checks, "pass": True}


def _solver_gates(spec: HardCacheSpec) -> dict[str, object]:
    arrays = load_hard_cache_arrays(spec)
    support = (0, 1, 3, 5, 21)
    equivalence = []
    for state_index in range(5):
        phi = np.asarray(arrays["full_bank"][state_index][:, support], dtype=np.complex128)
        y = np.asarray(arrays["full_target"][state_index], dtype=np.complex128)
        equivalence.append(ols_scaling_equivalence_gate(phi, y))
    phi = np.asarray(arrays["full_bank"][0][:, support], dtype=np.complex128)
    y = np.asarray(arrays["full_target"][0], dtype=np.complex128)
    ridge_lambda = 1e-8
    actual = fit_ridge(phi, y, ridge_lambda)
    n_samples, k = phi.shape
    augmented = np.vstack([phi, np.sqrt(n_samples * ridge_lambda) * np.eye(k, dtype=np.complex128)])
    augmented_y = np.concatenate([y, np.zeros(k, dtype=np.complex128)])
    expected_theta, *_ = np.linalg.lstsq(augmented, augmented_y, rcond=None)
    ridge_theta_error = float(np.max(np.abs(actual.theta - expected_theta)))
    if ridge_theta_error > 1e-12:
        raise RuntimeError(f"Ridge augmented-LS gate failed: {ridge_theta_error:.3e}")
    leakage_gate = False
    try:
        assert_test_access_allowed(model_frozen=False)
    except AssertionError:
        leakage_gate = True
    if not leakage_gate:
        raise RuntimeError("Test leakage gate failed")
    return {
        "ols_scaling_max_nmse_difference_db": max(
            float(item["nmse_abs_difference_db"]) for item in equivalence
        ),
        "ols_scaling_max_prediction_relative_l2": max(
            float(item["prediction_relative_l2"]) for item in equivalence
        ),
        "ridge_theta_max_abs_error": ridge_theta_error,
        "test_access_before_freeze_asserted": leakage_gate,
        "pass": True,
    }


def _benchmark_workers(spec: HardCacheSpec) -> tuple[int, list[dict[str, float | int]]]:
    supports = (tuple(range(5)), tuple(range(15)), tuple(range(25)))
    results: list[dict[str, float | int]] = []
    for workers in available_benchmark_workers():
        started = time.perf_counter()
        with PersistentStateEvaluator(spec, workers) as evaluator:
            evaluator.evaluate_supports(supports)
        elapsed = time.perf_counter() - started
        evaluations = len(spec.state_ids) * len(supports)
        throughput = evaluations / elapsed
        row = {
            "workers": workers,
            "elapsed_sec": elapsed,
            "evaluations": evaluations,
            "throughput": throughput,
        }
        results.append(row)
        print(
            f"[WORKER BENCH] {workers} workers: {throughput:.4f} support-state eval/s", flush=True
        )
    best = max(results, key=lambda row: float(row["throughput"]))
    selected = int(best["workers"])
    by_workers = {int(row["workers"]): row for row in results}
    if 14 in by_workers and 16 in by_workers:
        t14 = float(by_workers[14]["throughput"])
        t16 = float(by_workers[16]["throughput"])
        relative = abs(t14 - t16) / max(t14, t16)
        t12 = float(by_workers.get(12, {"throughput": -np.inf})["throughput"])
        if relative < WORKER_NEAR_TIE_FRACTION and max(t14, t16) >= t12:
            selected = 14
    print(f"[WORKER BENCH] selected: {selected}", flush=True)
    return selected, results


def _capacity_gate(full81: SupportScore) -> dict[str, object]:
    train_failures = [
        item.state_id for item in full81.state_metrics if item.full_train_nmse_db >= TARGET_NMSE_DB
    ]
    numerically_reliable = bool(
        full81.all_full_rank
        and np.isfinite(full81.q99_condition)
        and full81.q99_condition <= OLS_CONDITION_HARD_LIMIT
    )
    capacity_failure = bool(numerically_reliable and train_failures)
    return {
        "full81_all_full_rank": full81.all_full_rank,
        "full81_q99_condition": full81.q99_condition,
        "numerically_reliable": numerically_reliable,
        "train_failure_state_ids": train_failures,
        "capacity_failure": capacity_failure,
        "inconclusive": bool(not numerically_reliable),
    }


def _score_state_rows(score: SupportScore, model_id: str) -> list[dict[str, object]]:
    return [
        {
            "state_id": item.state_id,
            "model_id": model_id,
            "train_nmse_db": item.full_train_nmse_db,
            "cv1_nmse_db": item.cv_nmse_db[0],
            "cv2_nmse_db": item.cv_nmse_db[1],
            "cv3_nmse_db": item.cv_nmse_db[2],
            "W_dB": item.worst_nmse_db,
            "pass40": item.worst_nmse_db < TARGET_NMSE_DB,
        }
        for item in score.state_metrics
    ]


def _map_search_path(
    rows: Sequence[dict[str, object]], terms: Sequence[VolterraBasis]
) -> pd.DataFrame:
    mapped = []
    for row in rows:
        item = dict(row)
        support = tuple(
            int(value) for value in str(item.pop("support_indices")).split(";") if value
        )
        added = tuple(int(value) for value in str(item.pop("added_indices")).split(";") if value)
        removed = tuple(
            int(value) for value in str(item.pop("removed_indices")).split(";") if value
        )
        item["added_basis"] = ";".join(support_ids(terms, added))
        item["removed_basis"] = ";".join(support_ids(terms, removed))
        item["support_ids"] = ";".join(support_ids(terms, support))
        mapped.append(item)
    return pd.DataFrame(mapped)


def _selected_models_frame(
    terms: Sequence[VolterraBasis],
    search: SearchResult,
    ridge: RidgeTuningResult,
) -> pd.DataFrame:
    rows = []
    for index, score in enumerate(ridge.pareto_ols_scores, start=1):
        rows.append(
            {
                "model_id": f"OLS_PARETO_{index}",
                "stage": "pareto_ols",
                "K": score.k,
                "lambda": 0.0,
                "N40_inner": score.n40,
                "worst_W_dB": score.worst_w_db,
                "mean_deficit_dB": score.mean_deficit_db,
                "support_ids": ";".join(support_ids(terms, score.support)),
                "is_final": False,
            }
        )
    for index, (score, ridge_lambda) in enumerate(ridge.evaluated, start=1):
        rows.append(
            {
                "model_id": f"RIDGE_{index}",
                "stage": "ridge_scan",
                "K": score.k,
                "lambda": ridge_lambda,
                "N40_inner": score.n40,
                "worst_W_dB": score.worst_w_db,
                "mean_deficit_dB": score.mean_deficit_db,
                "support_ids": ";".join(support_ids(terms, score.support)),
                "is_final": bool(
                    score.support == ridge.final_score.support
                    and ridge_lambda == ridge.final_lambda
                ),
            }
        )
    rows.append(
        {
            "model_id": "OLS_SELECTED",
            "stage": "selected_ols",
            "K": search.selected_ols.k,
            "lambda": 0.0,
            "N40_inner": search.selected_ols.n40,
            "worst_W_dB": search.selected_ols.worst_w_db,
            "mean_deficit_dB": search.selected_ols.mean_deficit_db,
            "support_ids": ";".join(support_ids(terms, search.selected_ols.support)),
            "is_final": False,
        }
    )
    return pd.DataFrame(rows)


def _summary_text(
    *,
    result_root: Path,
    hard_ids: Sequence[int],
    worker_count: int,
    benchmark: Sequence[dict[str, float | int]],
    capacity: dict[str, object],
    dictionary_validation: dict[str, object],
    solver_validation: dict[str, object],
    elapsed_sec: float,
    search: SearchResult | None,
    ridge: RidgeTuningResult | None,
    final_rows: Sequence[dict[str, object]],
    engineering_checks: dict[str, object],
) -> str:
    lines = [
        f"Experiment: {EXPERIMENT_NAME}",
        "Module: scripts/basis_function_selection",
        "Research mode: new independent basis-selection task",
        "Signal pair: xin -> yout_withoutdpd_ori",
        "Bandwidth: 5B (100 MHz sample rate / 20 MHz base bandwidth)",
        "Preprocessing: FULL rough align -> FULL fine align -> fixed Train/Test split "
        "-> segment-local complex gain",
        f"Train: [{0}:{TRAIN_END}) ({TRAIN_LENGTH} samples)",
        f"Test: [{TEST_START}:{TEST_END}) ({TEST_LENGTH} samples)",
        "Test basis history is segment-local; no history is borrowed from Train.",
        "Hard-20 rule: descending aligned/gain-adjusted Train no-DPD NMSE, "
        "state_id ascending on exact ties",
        "Hard-20 state IDs: " + ", ".join(str(value) for value in hard_ids),
        f"Volterra orders: {list(ORDERS)}",
        f"dmax: {DMAX}",
        f"Candidate count: {CANDIDATE_COUNT}",
        "Mandatory basis: V1_u0 = x[n]",
        f"Dictionary gate: {dictionary_validation}",
        f"Solver/leakage gates: {solver_validation}",
        f"Logical CPUs: {logical_cpu_count()}",
        f"Selected workers: {worker_count}",
        "BLAS threads per worker: 1",
        "Nested parallelism: false",
        "Worker benchmark: " + json.dumps(list(benchmark), ensure_ascii=False),
        f"Full81 numerical-reliability condition limit: {OLS_CONDITION_HARD_LIMIT:.17g}",
        "Full81 capacity result: " + json.dumps(capacity, ensure_ascii=False),
    ]
    if search is None or ridge is None:
        lines.extend(
            [
                "OLS sparse search: not run because a numerically reliable Full81 "
                "capacity failure was detected.",
                "Ridge scan: not run.",
                "Final Test: not evaluated.",
                "Test gain computed: false",
                "Conclusion: Candidate-space capacity failure for strict orders 1/3/5 with dmax=2.",
            ]
        )
    else:
        train_count = sum(bool(row["pass_train40"]) for row in final_rows)
        test_count = sum(bool(row["pass_test40"]) for row in final_rows)
        both_count = sum(bool(row["pass_both40"]) for row in final_rows)
        type_b = sum(
            row["failure_type"] == "B_train_success_test_generalization_failure"
            for row in final_rows
        )
        type_c = sum(row["failure_type"] == "C_train_modeling_failure" for row in final_rows)
        worst_train = max(final_rows, key=lambda row: float(row["final_train_nmse_db"]))
        worst_test = max(final_rows, key=lambda row: float(row["final_test_nmse_db"]))
        lines.extend(
            [
                f"OLS stop reason: {search.stop_reason}",
                f"OLS final K: {search.selected_ols.k}",
                "OLS final support: "
                + ", ".join(support_ids(build_dictionary(), search.selected_ols.support)),
                f"OLS inner N40: {search.selected_ols.n40}/20",
                f"Ridge final K: {ridge.final_score.k}",
                f"Ridge final lambda: {ridge.final_lambda:.17g}",
                "Ridge final support: "
                + ", ".join(support_ids(build_dictionary(), ridge.final_score.support)),
                f"Ridge inner N40: {ridge.final_score.n40}/20",
                f"Final Train: {train_count}/20 < -40 dB",
                f"Final Test: {test_count}/20 < -40 dB",
                f"Final Both: {both_count}/20",
                "Test gain computed: true",
                f"Worst Train state: {worst_train['state_id']} "
                f"({float(worst_train['final_train_nmse_db']):.9f} dB)",
                f"Worst Test state: {worst_test['state_id']} "
                f"({float(worst_test['final_test_nmse_db']):.9f} dB)",
                f"Target 20/20 achieved: {both_count == HARD_STATE_COUNT}",
                f"Failure type B count: {type_b}",
                f"Failure type C count: {type_c}",
                "Conclusion: "
                + (
                    "The frozen strict-Volterra support achieved 20/20 Train/Test."
                    if both_count == HARD_STATE_COUNT
                    else (
                        "Train capacity reached 20/20 but Train-to-Test generalization "
                        "remains the bottleneck."
                        if train_count == HARD_STATE_COUNT
                        else "The selected orders 1/3/5, dmax=2 support still has "
                        "Train-capacity failures."
                    )
                ),
            ]
        )
    lines.extend(
        [
            "B/Test used during support selection: false",
            "B/Test used during Ridge tuning: false",
            "Test gain policy: compute only after model freeze",
            "Test target-side gain calibration is part of this experiment's evaluation definition.",
            f"Engineering checks: {json.dumps(engineering_checks, ensure_ascii=False)}",
            f"Result root: {result_root}",
            f"Total runtime seconds: {elapsed_sec:.3f}",
            "No 405-state validation, LUT retrieval, Real-B, DPD replay, low-bandwidth, "
            "order-7/9, or dmax>2 work was performed.",
        ]
    )
    return "\n".join(lines) + "\n"


def _run_engineering_checks() -> dict[str, object]:
    python = Path(sys.executable)
    commands = {
        "py_compile": [
            str(python),
            "-m",
            "compileall",
            "-q",
            str(MODULE_ROOT),
        ],
        "ruff": [
            str(python),
            "-m",
            "ruff",
            "check",
            "--no-cache",
            str(MODULE_ROOT),
        ],
        "pip_check": [str(python), "-m", "pip", "check"],
        "git_diff_check": ["git", "diff", "--check"],
        "core_regression": [
            str(python),
            "-B",
            str(SCRIPTS_ROOT / "core" / "test_core_modules.py"),
        ],
        "data_manager_regression": [
            str(python),
            "-B",
            str(SCRIPTS_ROOT / "data_management" / "test_data_manager.py"),
        ],
        "signal_segmentation_regression": [
            str(python),
            "-B",
            str(
                SCRIPTS_ROOT
                / "signal_segmentation"
                / "shared"
                / "tests"
                / "test_signal_segmentation.py"
            ),
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
            "stdout_tail": completed.stdout[-1000:],
            "stderr_tail": completed.stderr[-1000:],
        }
        if completed.returncode != 0:
            raise RuntimeError(f"Engineering check {name} failed: {results[name]}")
    return results


def _write_capacity_failure_outputs(
    result_root: Path,
    ranking_frame: pd.DataFrame,
    candidate_frame: pd.DataFrame,
    terms: Sequence[VolterraBasis],
    full81: SupportScore,
    capacity: dict[str, object],
) -> None:
    ranking_frame.to_csv(
        result_root / EXPECTED_OUTPUT_FILENAMES[0], index=False, encoding="utf-8-sig"
    )
    candidate_frame.to_csv(
        result_root / EXPECTED_OUTPUT_FILENAMES[1], index=False, encoding="utf-8-sig"
    )
    path = pd.DataFrame(
        [
            {
                "step": 0,
                "action": "full81_capacity_failure",
                "added_basis": "",
                "removed_basis": "",
                "K": full81.k,
                "N40": full81.n40,
                "worst_W_dB": full81.worst_w_db,
                "worst_deficit_dB": full81.worst_deficit_db,
                "Q90_deficit_dB": full81.q90_deficit_db,
                "mean_deficit_dB": full81.mean_deficit_db,
                "median_W_dB": full81.median_w_db,
                "support_ids": ";".join(support_ids(terms, full81.support)),
                "elapsed_sec": 0.0,
            }
        ]
    )
    path.to_csv(result_root / EXPECTED_OUTPUT_FILENAMES[2], index=False, encoding="utf-8-sig")
    pd.DataFrame(
        [
            {
                "model_id": "FULL81_CAPACITY",
                "stage": "capacity_gate",
                "K": full81.k,
                "lambda": 0.0,
                "N40_inner": full81.n40,
                "worst_W_dB": full81.worst_w_db,
                "mean_deficit_dB": full81.mean_deficit_db,
                "support_ids": ";".join(support_ids(terms, full81.support)),
                "is_final": False,
            }
        ]
    ).to_csv(result_root / EXPECTED_OUTPUT_FILENAMES[3], index=False, encoding="utf-8-sig")
    pd.DataFrame(_score_state_rows(full81, "FULL81_CAPACITY")).to_csv(
        result_root / EXPECTED_OUTPUT_FILENAMES[4], index=False, encoding="utf-8-sig"
    )
    final_rows = []
    ranking_by_state = ranking_frame.set_index("state_id").to_dict("index")
    metrics_by_state = {item.state_id: item for item in full81.state_metrics}
    for actual_state_id, metric in metrics_by_state.items():
        rank = ranking_by_state[actual_state_id]
        final_rows.append(
            {
                "state_id": actual_state_id,
                "funMng": rank["funMng"],
                "funAng": rank["funAng"],
                "secMng": rank["secMng"],
                "secAng": rank["secAng"],
                "Vm": rank["Vm"],
                "Pin": rank["Pin"],
                "original_train_noDPD_NMSE_dB": rank["train_noDPD_NMSE_dB"],
                "final_train_nmse_db": metric.full_train_nmse_db,
                "final_test_nmse_db": np.nan,
                "test_evaluated": False,
                "pass_train40": metric.full_train_nmse_db < TARGET_NMSE_DB,
                "pass_test40": None,
                "pass_both40": None,
                "failure_type": "capacity_gate_stopped_before_test",
            }
        )
    pd.DataFrame(final_rows).to_csv(
        result_root / EXPECTED_OUTPUT_FILENAMES[5], index=False, encoding="utf-8-sig"
    )


def main() -> None:
    freeze_support()
    started = time.perf_counter()
    _check_old_process()
    before_protection = _protection_snapshot()
    result_root = _select_result_root()
    scratch_root = result_root / "_scratch"
    _append_log(
        WORK_LOG,
        f"\n\n[{_timestamp()}] START {EXPERIMENT_NAME}\n"
        f"Result root: {result_root}\nPython: {sys.executable}\n"
        "Mode: new independent research task; raw data and historical results are read-only.\n",
    )
    completed_successfully = False
    try:
        terms = build_dictionary()
        dictionary_validation = dictionary_gate()
        ranking_rows = preprocess_all_for_ranking(PREPROCESS_WORKERS)
        alignment_validation = _representative_alignment_gate(ranking_rows)
        ranked_rows, hard_ids = select_hard_states(ranking_rows)
        print(
            "[HARD20] selected states: " + ", ".join(str(value) for value in hard_ids), flush=True
        )
        ranking_frame = pd.DataFrame(ranked_rows)
        candidate_frame = _candidate_frame(terms)
        ranking_frame.to_csv(
            result_root / EXPECTED_OUTPUT_FILENAMES[0], index=False, encoding="utf-8-sig"
        )
        candidate_frame.to_csv(
            result_root / EXPECTED_OUTPUT_FILENAMES[1], index=False, encoding="utf-8-sig"
        )

        spec = create_hard_cache(scratch_root, hard_ids)
        hard_cache_diagnostics = populate_hard_cache(spec, PREPROCESS_WORKERS)
        solver_validation = _solver_gates(spec)
        selected_workers, benchmark = _benchmark_workers(spec)

        with PersistentStateEvaluator(spec, selected_workers) as evaluator:
            full81 = evaluator.evaluate_supports([tuple(range(CANDIDATE_COUNT))])[0]
            capacity = _capacity_gate(full81)
            print(
                f"[CAPACITY] Full81 full-rank={full81.all_full_rank} "
                f"q99_cond={full81.q99_condition:.6g} "
                f"Train failures={capacity['train_failure_state_ids']}",
                flush=True,
            )
            if bool(capacity["capacity_failure"]):
                _write_capacity_failure_outputs(
                    result_root, ranking_frame, candidate_frame, terms, full81, capacity
                )
                search_result = None
                ridge_result = None
                final_rows: list[dict[str, object]] = []
            else:
                search_result = run_sparse_search(evaluator, terms)
                ridge_result = tune_ridge(
                    evaluator,
                    search_result.accepted_scores,
                    search_result.selected_ols,
                )
                model_frozen = True
                ranking_by_state = {int(row["state_id"]): row for row in ranked_rows}
                final_rows = evaluate_final_model(
                    spec,
                    terms,
                    ridge_result.final_score.support,
                    ridge_result.final_lambda,
                    ranking_by_state,
                    model_frozen=model_frozen,
                )
                _map_search_path(search_result.path_rows, terms).to_csv(
                    result_root / EXPECTED_OUTPUT_FILENAMES[2], index=False, encoding="utf-8-sig"
                )
                _selected_models_frame(terms, search_result, ridge_result).to_csv(
                    result_root / EXPECTED_OUTPUT_FILENAMES[3], index=False, encoding="utf-8-sig"
                )
                inner_rows = []
                inner_rows.extend(_score_state_rows(search_result.selected_ols, "OLS_SELECTED"))
                inner_rows.extend(_score_state_rows(ridge_result.final_score, "FINAL_RIDGE_OR_OLS"))
                pd.DataFrame(inner_rows).to_csv(
                    result_root / EXPECTED_OUTPUT_FILENAMES[4], index=False, encoding="utf-8-sig"
                )
                pd.DataFrame(final_rows).to_csv(
                    result_root / EXPECTED_OUTPUT_FILENAMES[5], index=False, encoding="utf-8-sig"
                )

        engineering_checks = _run_engineering_checks()
        after_protection = _protection_snapshot()
        protection_pass = before_protection == after_protection
        if not protection_pass:
            raise RuntimeError(
                "data/raw or protected historical result metadata changed during this task"
            )
        elapsed = time.perf_counter() - started
        summary = _summary_text(
            result_root=result_root,
            hard_ids=hard_ids,
            worker_count=selected_workers,
            benchmark=benchmark,
            capacity=capacity,
            dictionary_validation=dictionary_validation,
            solver_validation=solver_validation,
            elapsed_sec=elapsed,
            search=search_result,
            ridge=ridge_result,
            final_rows=final_rows,
            engineering_checks={
                **engineering_checks,
                "alignment_gate": alignment_validation,
                "hard_cache_diagnostics_count": len(hard_cache_diagnostics),
                "raw_and_protected_metadata_unchanged": protection_pass,
            },
        )
        (result_root / EXPECTED_OUTPUT_FILENAMES[6]).write_text(summary, encoding="utf-8")
        if scratch_root.exists():
            shutil.rmtree(scratch_root)
        actual_files = tuple(sorted(path.name for path in result_root.iterdir() if path.is_file()))
        if actual_files != tuple(sorted(EXPECTED_OUTPUT_FILENAMES)):
            raise RuntimeError(f"Formal result-file gate failed: {actual_files}")
        completed_successfully = True
        log_summary = (
            f"Completed {EXPERIMENT_NAME}. Hard20={list(hard_ids)}; workers={selected_workers}; "
            f"capacity={capacity}; result={result_root}; elapsed={elapsed:.3f}s; "
            "data/raw and protected historical results unchanged.\n"
        )
        _append_log(WORK_LOG, f"[{_timestamp()}] COMPLETE\n{log_summary}")
        _append_log(
            HANDOFF_LOG,
            f"\n\n{_timestamp()} | basis_function_selection: {EXPERIMENT_NAME}\n{log_summary}",
        )
        print("[COMPLETE] All task gates passed", flush=True)
        print(f"[COMPLETE] Result: {result_root}", flush=True)
    except Exception as exc:
        _append_log(
            WORK_LOG,
            f"[{_timestamp()}] FAILED\n{type(exc).__name__}: {exc}\n"
            "No raw-data or historical-result cleanup/reset operation was performed.\n",
        )
        raise
    finally:
        if completed_successfully and scratch_root.exists():
            shutil.rmtree(scratch_root)


if __name__ == "__main__":
    main()
