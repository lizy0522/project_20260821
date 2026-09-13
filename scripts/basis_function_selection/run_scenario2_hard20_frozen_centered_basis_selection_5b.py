"""Run nested Frozen-centered capacity gates and sparse selection on Hard-20."""

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

import pandas as pd

MODULE_ROOT = Path(__file__).resolve().parent
SCRIPTS_ROOT = MODULE_ROOT.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from basis_function_selection.capacity_gates import (
    FrozenCenteredEvaluator,
    evaluate_capacity_state,
    summarize_capacity_gate,
)
from basis_function_selection.final_evaluation import (
    assert_test_access_allowed,
    evaluate_frozen_centered_final_model,
)
from basis_function_selection.frozen_centered_dictionary import (
    FrozenCenteredBasis,
    build_frozen_centered_dictionary,
    dictionary_numerical_gate,
    gate_indices,
)
from basis_function_selection.ridge_tuning import tune_frozen_centered_ridge
from basis_function_selection.search import run_frozen_centered_search

EXPERIMENT_NAME = "scenario_2_hard20_frozen_centered_basis_selection_5B"
RESULT_PARENT = PROJECT_ROOT / "results" / "basis_function_selection"
PRIMARY_RESULT_ROOT = RESULT_PARENT / EXPERIMENT_NAME
FULL81_ROOT = RESULT_PARENT / "scenario_2_hard20_volterra_basis_selection_5B"
REFERENCE_ROOT = RESULT_PARENT / "scenario_2_hard20_frozen10_reference_gate_5B"
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
OUTPUT_NAMES = (
    "01_frozen_centered_candidate_dictionary.csv",
    "02_capacity_gate_per_state_metrics.csv",
    "03_capacity_gate_summary.csv",
    "04_selection_path.csv",
    "05_selected_models.csv",
    "06_hard20_final_metrics.csv",
    "07_final_result_summary.txt",
)
PROTECTED_PATHS = (
    PROJECT_ROOT / "data" / "raw",
    FULL81_ROOT,
    REFERENCE_ROOT,
    PROJECT_ROOT / "results" / "behavior_model",
    PROJECT_ROOT / "results" / "behavior_fingerprint_lut_retrieval",
    PROJECT_ROOT / "results" / "clustering",
    PROJECT_ROOT / "results" / "low_bandwidth_observation",
)


def _timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def _manifest(path: Path) -> dict[str, object]:
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
    return {str(path): _manifest(path) for path in PROTECTED_PATHS}


def _result_root() -> Path:
    RESULT_PARENT.mkdir(parents=True, exist_ok=True)
    if not PRIMARY_RESULT_ROOT.exists() or not any(PRIMARY_RESULT_ROOT.iterdir()):
        PRIMARY_RESULT_ROOT.mkdir(parents=True, exist_ok=True)
        return PRIMARY_RESULT_ROOT
    suffix = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    rerun = RESULT_PARENT / f"{EXPERIMENT_NAME}_rerun_{suffix}"
    rerun.mkdir(parents=True, exist_ok=False)
    return rerun


def _load_history() -> tuple[pd.DataFrame, pd.DataFrame]:
    ranking_path = FULL81_ROOT / "01_hard20_state_ranking.csv"
    reference_path = REFERENCE_ROOT / "02_hard20_reference_metrics.csv"
    if not ranking_path.is_file() or not reference_path.is_file():
        raise FileNotFoundError("Required Full81/Frozen10 reference inputs are missing")
    ranking = pd.read_csv(ranking_path)
    reference = pd.read_csv(reference_path)
    hard_mask = ranking["is_hard20"].map(
        lambda value: value is True or str(value).strip().lower() == "true"
    )
    hard_ids = tuple(ranking.loc[hard_mask].sort_values("rank")["state_id"].astype(int).tolist())
    if hard_ids != EXPECTED_HARD20:
        raise RuntimeError(f"Frozen Hard-20 mismatch: {hard_ids}")
    frozen = reference.loc[reference["model_id"] == "M4"].copy()
    if len(frozen) != 20 or set(frozen["state_id"].astype(int)) != set(EXPECTED_HARD20):
        raise RuntimeError("Frozen10 reference M4 does not cover Hard-20 exactly")
    return ranking, frozen


def _dictionary_frame(terms: tuple[FrozenCenteredBasis, ...]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "basis_id": term.basis_id,
                "formula": term.formula,
                "family": term.family,
                "order": term.order,
                "signal_delay": term.signal_delay,
                "envelope_delay": term.envelope_delay,
                "nonconjugate_delays": json.dumps(term.nonconjugate_delays),
                "conjugate_delays": json.dumps(term.conjugate_delays),
                "algebraic_signature": json.dumps(term.algebraic_signature),
                "in_frozen10": term.in_frozen10,
                "in_gate_A": term.in_gate_a,
                "in_gate_B": term.in_gate_b,
                "in_gate_C": term.in_gate_c,
            }
            for term in terms
        ]
    )


def _capacity_worker(
    payload: tuple[int, int, str, tuple[FrozenCenteredBasis, ...], dict[str, object]],
) -> dict[str, object]:
    return evaluate_capacity_state(*payload)


def _run_capacity_gate(
    executor: ProcessPoolExecutor,
    gate_id: str,
    terms: tuple[FrozenCenteredBasis, ...],
    ranking_by_state: dict[int, dict[str, object]],
    frozen_by_state: dict[int, dict[str, object]],
) -> list[dict[str, object]]:
    payloads = [
        (
            state_id,
            int(ranking_by_state[state_id]["rank"]),
            gate_id,
            terms,
            frozen_by_state[state_id],
        )
        for state_id in EXPECTED_HARD20
    ]
    rows: dict[int, dict[str, object]] = {}
    futures = {executor.submit(_capacity_worker, payload): payload[0] for payload in payloads}
    for done, future in enumerate(as_completed(futures), start=1):
        state_id = futures[future]
        rows[state_id] = future.result()
        if done % 5 == 0 or done == 20:
            print(f"[CAPACITY {gate_id}] {done}/20", flush=True)
    return [rows[state_id] for state_id in EXPECTED_HARD20]


def _verify_frozen_reproduction(
    frozen_rows: list[dict[str, object]],
    frozen_by_state: dict[int, dict[str, object]],
) -> float:
    keys = ("train_nmse_db", "cv1_nmse_db", "cv2_nmse_db", "cv3_nmse_db", "W_db")
    maximum = 0.0
    for row in frozen_rows:
        saved = frozen_by_state[int(row["state_id"])]
        maximum = max(maximum, *(abs(float(row[key]) - float(saved[key])) for key in keys))
    if maximum > 1e-6:
        raise RuntimeError(f"Frozen10 reproduction failed by {maximum:.3e} dB")
    return maximum


def _path_frame(
    rows: tuple[dict[str, object], ...], terms: tuple[FrozenCenteredBasis, ...]
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
        item["added_basis"] = ";".join(terms[index].basis_id for index in added)
        item["removed_basis"] = ";".join(terms[index].basis_id for index in removed)
        item["support_ids"] = ";".join(terms[index].basis_id for index in support)
        mapped.append(item)
    return pd.DataFrame(mapped)


def _family_composition(
    support: tuple[int, ...],
    terms: tuple[FrozenCenteredBasis, ...],
) -> dict[str, int]:
    composition: dict[str, int] = {}
    for index in support:
        term = terms[index]
        if term.family == "ALIGNED_ENVELOPE":
            key = f"ALIGNED_P{term.order}"
        else:
            key = term.family
        composition[key] = composition.get(key, 0) + 1
    return dict(sorted(composition.items()))


def _selected_models_frame(
    selected_gate: str,
    terms: tuple[FrozenCenteredBasis, ...],
    search_result: object,
    ridge_result: object,
) -> pd.DataFrame:
    rows = []
    for index, score in enumerate(ridge_result.pareto_ols_scores, start=1):
        rows.append(
            {
                "model_id": f"OLS_PARETO_{index}",
                "candidate_gate": selected_gate,
                "stage": "pareto_ols",
                "K": score.k,
                "lambda": 0.0,
                "N_train40": score.n_train40,
                "N_W40": score.n_w40,
                "worst_W_db": score.worst_w_db,
                "median_W_db": score.median_w_db,
                "support_ids": ";".join(terms[i].basis_id for i in score.support),
                "support_families": json.dumps(
                    _family_composition(score.support, terms), ensure_ascii=False
                ),
                "is_final": False,
            }
        )
    for index, (score, ridge_lambda) in enumerate(ridge_result.evaluated, start=1):
        is_final = bool(
            score.support == ridge_result.final_score.support
            and ridge_lambda == ridge_result.final_lambda
        )
        rows.append(
            {
                "model_id": f"RIDGE_{index}",
                "candidate_gate": selected_gate,
                "stage": "ridge_scan",
                "K": score.k,
                "lambda": ridge_lambda,
                "N_train40": score.n_train40,
                "N_W40": score.n_w40,
                "worst_W_db": score.worst_w_db,
                "median_W_db": score.median_w_db,
                "support_ids": ";".join(terms[i].basis_id for i in score.support),
                "support_families": json.dumps(
                    _family_composition(score.support, terms), ensure_ascii=False
                ),
                "is_final": is_final,
            }
        )
    rows.append(
        {
            "model_id": "OLS_SELECTED",
            "candidate_gate": selected_gate,
            "stage": "selected_ols",
            "K": search_result.selected_ols.k,
            "lambda": 0.0,
            "N_train40": search_result.selected_ols.n_train40,
            "N_W40": search_result.selected_ols.n_w40,
            "worst_W_db": search_result.selected_ols.worst_w_db,
            "median_W_db": search_result.selected_ols.median_w_db,
            "support_ids": ";".join(terms[i].basis_id for i in search_result.selected_ols.support),
            "support_families": json.dumps(
                _family_composition(search_result.selected_ols.support, terms),
                ensure_ascii=False,
            ),
            "is_final": False,
        }
    )
    return pd.DataFrame(rows)


def _summary_text(
    gate_summary: pd.DataFrame,
    selected_gate: str | None,
    terms: tuple[FrozenCenteredBasis, ...],
    search_result: object | None,
    ridge_result: object | None,
    final_rows: list[dict[str, object]],
    gates: dict[str, object],
    elapsed: float,
    result_root: Path,
) -> str:
    lines = [
        f"Experiment: {EXPERIMENT_NAME}",
        "Data: xin -> yout_withoutdpd_ori",
        "Preprocessing: FULL rough align -> FULL fine align -> Train/Test split "
        "-> segment-local gain",
        "Hard-20: " + ", ".join(str(value) for value in EXPECTED_HARD20),
        "Candidate nesting: Frozen10(10) < Gate A(18) < Gate B(48) < Gate C(108)",
        "Capacity success criterion: 20/20 full-Train NMSE < -40 dB",
        "Numerical reliability: all states full rank and condition Q99 <= 1e10",
        "Workers: 12 processes; BLAS threads per worker=1; GPU unused",
        "",
        "[Capacity gates]",
    ]
    for _, row in gate_summary.iterrows():
        delta_text = ""
        if pd.notna(row["median_delta_train_vs_previous_gate"]):
            delta_text = (
                f", median Train delta vs previous="
                f"{float(row['median_delta_train_vs_previous_gate']):.9f} dB, "
                f"median W delta vs previous="
                f"{float(row['median_delta_W_vs_previous_gate']):.9f} dB"
            )
        lines.append(
            f"{row['gate_id']}: K={int(row['K'])}, "
            f"Train median={float(row['train_median']):.9f} dB, "
            f"W median={float(row['W_median']):.9f} dB, "
            f"Train pass={int(row['N_train40'])}/20, W pass={int(row['N_W40'])}/20, "
            f"reliable={bool(row['numerically_reliable'])}, "
            f"success={bool(row['capacity_success'])}{delta_text}"
        )
    ran = set(gate_summary["gate_id"].astype(str))
    for gate_id in ("GATE_A", "GATE_B", "GATE_C"):
        if gate_id not in ran:
            lines.append(f"{gate_id}: not run")
    lines.extend(
        [
            f"Smallest Train-capacity-sufficient family: {selected_gate or 'none'}",
            "",
            "[Gates]",
            json.dumps(gates, ensure_ascii=False),
        ]
    )
    if selected_gate is None or search_result is None or ridge_result is None:
        lines.extend(
            [
                "Sparse search: not run",
                "Ridge: not run",
                "Test: not evaluated",
                "Conclusion: Frozen-centered dmax=2 candidate-space capacity failure "
                "or inconclusive.",
            ]
        )
    else:
        final = ridge_result.final_score
        composition = _family_composition(final.support, terms)
        train_pass = sum(bool(row["pass_train40"]) for row in final_rows)
        inner_pass = sum(bool(row["pass_inner_W40"]) for row in final_rows)
        test_pass = sum(bool(row["pass_test40"]) for row in final_rows)
        both_pass = sum(bool(row["pass_both40"]) for row in final_rows)
        type_b = sum(
            row["failure_type"] == "B_train_success_test_generalization_failure"
            for row in final_rows
        )
        type_c = sum(row["failure_type"] == "C_train_modeling_failure" for row in final_rows)
        failed_rows = [row for row in final_rows if not bool(row["pass_both40"])]
        lines.extend(
            [
                "",
                "[Sparse selection]",
                f"Search stop reason: {search_result.stop_reason}",
                f"OLS selected K={search_result.selected_ols.k}",
                f"OLS Train pass={search_result.selected_ols.n_train40}/20",
                f"OLS inner W pass={search_result.selected_ols.n_w40}/20",
                f"Final K={final.k}",
                f"Final lambda={ridge_result.final_lambda:.17g}",
                "Final support: " + ", ".join(terms[i].basis_id for i in final.support),
                "Family composition: " + json.dumps(composition, ensure_ascii=False),
                "",
                "[Final Hard-20]",
                f"Train pass={train_pass}/20",
                f"Inner W pass={inner_pass}/20",
                f"Test pass={test_pass}/20",
                f"Both pass={both_pass}/20",
                f"Failure type B={type_b}",
                f"Failure type C={type_c}",
                "Failed states: "
                + json.dumps(
                    [
                        {
                            "state_id": row["state_id"],
                            "train_nmse_db": row["train_nmse_db"],
                            "inner_W_db": row["inner_W_db"],
                            "test_nmse_db": row["test_nmse_db"],
                            "failure_type": row["failure_type"],
                        }
                        for row in failed_rows
                    ],
                    ensure_ascii=False,
                ),
                "Conclusion: "
                + (
                    "Hard-20 Train/Test target achieved."
                    if both_pass == 20
                    else (
                        "Train capacity is sufficient, but Train-to-Test generalization "
                        "remains limiting."
                        if train_pass == 20
                        else "The selected sparse structure retains Train-capacity failures."
                    )
                ),
            ]
        )
    lines.extend(
        [
            "Test was hidden until support and lambda were frozen.",
            "No 405-state validation, LUT retrieval, Real-B, DPD replay, nB, dmax>2, "
            "or 11th-order work was performed.",
            f"Runtime seconds: {elapsed:.3f}",
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
    results = {}
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
            "stdout_tail": completed.stdout[-600:],
            "stderr_tail": completed.stderr[-600:],
        }
        if completed.returncode != 0:
            raise RuntimeError(f"Engineering check failed: {name}: {results[name]}")
    return results


def main() -> None:
    freeze_support()
    started = time.perf_counter()
    before = _protection_snapshot()
    result_root = _result_root()
    _append_log(
        WORK_LOG,
        f"\n\n[{_timestamp()}] START {EXPERIMENT_NAME}\n"
        f"Result root: {result_root}\nPython: {sys.executable}\n"
        "Full81 and Frozen10 Reference Gate inputs are read-only; Test is hidden "
        "until model freeze.\n",
    )
    try:
        ranking, frozen_reference = _load_history()
        terms = build_frozen_centered_dictionary()
        dictionary_gate = dictionary_numerical_gate()
        ranking_by_state = ranking.set_index("state_id").to_dict("index")
        frozen_by_state = frozen_reference.set_index("state_id").to_dict("index")
        assert_test_access_allowed(model_frozen=False)
    except AssertionError:
        leakage_gate = True
    else:
        raise RuntimeError("Test leakage assertion did not fail before model freeze")

    try:
        capacity_rows: list[dict[str, object]] = []
        capacity_summaries: list[dict[str, object]] = []
        selected_gate: str | None = None
        with ProcessPoolExecutor(max_workers=WORKER_COUNT) as executor:
            frozen_rows = _run_capacity_gate(
                executor,
                "FROZEN10",
                terms,
                ranking_by_state,
                frozen_by_state,
            )
            reproduction_max = _verify_frozen_reproduction(frozen_rows, frozen_by_state)
            frozen_summary = summarize_capacity_gate(frozen_rows, None)
            capacity_rows.extend(frozen_rows)
            capacity_summaries.append(frozen_summary)
            previous_rows = frozen_rows
            for gate_id in ("GATE_A", "GATE_B", "GATE_C"):
                rows = _run_capacity_gate(
                    executor,
                    gate_id,
                    terms,
                    ranking_by_state,
                    frozen_by_state,
                )
                summary = summarize_capacity_gate(rows, previous_rows)
                capacity_rows.extend(rows)
                capacity_summaries.append(summary)
                print(
                    f"[CAPACITY {gate_id}] Train={summary['N_train40']}/20 "
                    f"W={summary['N_W40']}/20 median={summary['train_median']:.6f} dB",
                    flush=True,
                )
                if summary["capacity_inconclusive"]:
                    break
                if summary["capacity_success"]:
                    selected_gate = gate_id
                    break
                previous_rows = rows

        dictionary_frame = _dictionary_frame(terms)
        capacity_frame = pd.DataFrame(capacity_rows).drop(columns=["train_sse"])
        capacity_summary_frame = pd.DataFrame(capacity_summaries)
        dictionary_frame.to_csv(result_root / OUTPUT_NAMES[0], index=False, encoding="utf-8-sig")
        capacity_frame.to_csv(result_root / OUTPUT_NAMES[1], index=False, encoding="utf-8-sig")
        capacity_summary_frame.to_csv(
            result_root / OUTPUT_NAMES[2], index=False, encoding="utf-8-sig"
        )

        search_result = None
        ridge_result = None
        final_rows: list[dict[str, object]] = []
        if selected_gate is not None:
            dictionary_indices = gate_indices(terms, selected_gate)
            frozen_support = gate_indices(terms, "FROZEN10")
            mandatory = next(
                term.index for term in terms if term.family == "LINEAR" and term.signal_delay == 0
            )
            with FrozenCenteredEvaluator(
                terms,
                selected_gate,
                EXPECTED_HARD20,
                WORKER_COUNT,
            ) as evaluator:
                full_space_score = evaluator.evaluate_supports([dictionary_indices])[0]
                matching_summary = next(
                    row for row in capacity_summaries if row["gate_id"] == selected_gate
                )
                if (
                    full_space_score.n_train40 != matching_summary["N_train40"]
                    or full_space_score.n_w40 != matching_summary["N_W40"]
                ):
                    raise RuntimeError("Search evaluator does not reproduce capacity gate counts")
                search_result = run_frozen_centered_search(
                    evaluator,
                    dictionary_indices,
                    frozen_support,
                    mandatory,
                    full_space_score=full_space_score,
                    k_max=min(30, len(dictionary_indices)),
                )
                ridge_result = tune_frozen_centered_ridge(
                    evaluator,
                    search_result.accepted_scores,
                    search_result.selected_ols,
                )
            model_frozen = True
            inner_w_by_state = {
                metric.state_id: metric.worst_nmse_db
                for metric in ridge_result.final_score.state_metrics
            }
            rank_by_state = {
                state_id: int(ranking_by_state[state_id]["rank"]) for state_id in EXPECTED_HARD20
            }
            final_rows = evaluate_frozen_centered_final_model(
                EXPECTED_HARD20,
                terms,
                ridge_result.final_score.support,
                ridge_result.final_lambda,
                rank_by_state,
                inner_w_by_state,
                model_frozen=model_frozen,
            )
            _path_frame(search_result.path_rows, terms).to_csv(
                result_root / OUTPUT_NAMES[3], index=False, encoding="utf-8-sig"
            )
            _selected_models_frame(selected_gate, terms, search_result, ridge_result).to_csv(
                result_root / OUTPUT_NAMES[4], index=False, encoding="utf-8-sig"
            )
            pd.DataFrame(final_rows).to_csv(
                result_root / OUTPUT_NAMES[5], index=False, encoding="utf-8-sig"
            )

        gates: dict[str, Any] = {
            "dictionary": dictionary_gate,
            "frozen_reproduction_max_abs_delta_db": reproduction_max,
            "test_leakage_assertion": leakage_gate,
            "nested_sse_monotonicity": True,
            "selected_gate": selected_gate,
        }
        elapsed = time.perf_counter() - started
        summary = _summary_text(
            capacity_summary_frame,
            selected_gate,
            terms,
            search_result,
            ridge_result,
            final_rows,
            gates,
            elapsed,
            result_root,
        )
        (result_root / OUTPUT_NAMES[6]).write_text(summary, encoding="utf-8")
        engineering = _engineering_checks()
        after = _protection_snapshot()
        if before != after:
            raise RuntimeError("Raw data or protected historical result metadata changed")
        expected = (
            OUTPUT_NAMES
            if selected_gate is not None
            else (
                OUTPUT_NAMES[0],
                OUTPUT_NAMES[1],
                OUTPUT_NAMES[2],
                OUTPUT_NAMES[6],
            )
        )
        actual = tuple(sorted(item.name for item in result_root.iterdir() if item.is_file()))
        if actual != tuple(sorted(expected)):
            raise RuntimeError(f"Formal output-file gate failed: {actual}")
        final_both = sum(bool(row["pass_both40"]) for row in final_rows)
        log_summary = (
            f"Completed {EXPERIMENT_NAME}: selected_gate={selected_gate}, "
            f"capacity={capacity_summaries}, final_both={final_both}/20, "
            f"Test_accessed={selected_gate is not None}, workers={WORKER_COUNT}, "
            f"result={result_root}, elapsed={elapsed:.3f}s, engineering={engineering}.\n"
        )
        _append_log(WORK_LOG, f"[{_timestamp()}] COMPLETE\n{log_summary}")
        _append_log(
            HANDOFF_LOG,
            f"\n\n{_timestamp()} | basis_function_selection Frozen-centered task\n{log_summary}",
        )
        print("[COMPLETE] Frozen-centered task passed all gates", flush=True)
        print(f"[COMPLETE] Result: {result_root}", flush=True)
    except Exception as exc:
        _append_log(
            WORK_LOG,
            f"[{_timestamp()}] FAILED\n{type(exc).__name__}: {exc}\n"
            "No Git cleanup, raw mutation, or history overwrite was performed.\n",
        )
        raise


if __name__ == "__main__":
    main()
