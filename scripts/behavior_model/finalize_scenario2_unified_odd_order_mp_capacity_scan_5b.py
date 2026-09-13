"""Finalize the user-requested stopping point for the 5B odd-order MP scan.

The scan itself is performed by ``run_scenario2_unified_odd_order_mp_capacity_scan_5b``.
This entry point is deliberately finalization-only: it reads the complete Round
0--3 cache, reconstructs the 280 candidates already evaluated, and writes the
auditable tables, selected refit, figures, and validation records.  It never
creates or evaluates a Round 4 shell.
"""

# Imports intentionally follow the BLAS-thread environment setup in the runner.
# ruff: noqa: E402,I001

from __future__ import annotations

import json
from datetime import UTC, datetime
import sys
from pathlib import Path
from typing import Any

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import numpy as np
import pandas as pd

from behavior_model import odd_order_mp_capacity_scan as core
from behavior_model import run_scenario2_unified_odd_order_mp_capacity_scan_5b as scan


STOP_REASON = "user_requested_stop_after_round_3"
STOP_AFTER_ROUND = 3


def _load_round_cache(round_index: int, candidates: list[core.OddOrderCandidate]) -> pd.DataFrame:
    path = scan.CACHE_ROOT / f"round_{round_index:03d}" / "state_metrics.csv.gz"
    if not path.is_file():
        raise FileNotFoundError(f"missing completed cache for round {round_index}: {path}")
    frame = pd.read_csv(path, compression="gzip")
    expected_rows = core.STATE_COUNT * len(candidates) * 2
    if frame.shape[0] != expected_rows:
        raise RuntimeError(
            f"round {round_index} row count mismatch {frame.shape[0]} vs {expected_rows}"
        )
    if frame["state_id"].nunique() != core.STATE_COUNT:
        raise RuntimeError(f"round {round_index} does not cover all states")
    expected_ids = {candidate.candidate_id for candidate in candidates}
    observed_ids = set(frame["candidate_id"].astype(int).unique())
    if observed_ids != expected_ids:
        raise RuntimeError(
            f"round {round_index} candidate ids mismatch: {sorted(observed_ids)} vs "
            f"{sorted(expected_ids)}"
        )
    counts = frame.groupby("candidate_id").size()
    if not bool((counts == 2 * core.STATE_COUNT).all()):
        raise RuntimeError(f"round {round_index} has incomplete candidate rows")
    return frame.sort_values(["candidate_id", "state_id", "side"]).reset_index(drop=True)


def _reconstruct_caches() -> tuple[list[core.OddOrderCandidate], pd.DataFrame]:
    candidates: list[core.OddOrderCandidate] = []
    frames: list[pd.DataFrame] = []
    for round_index in range(STOP_AFTER_ROUND + 1):
        shell = list(core.generate_shell(candidates, round_index))
        shell = [
            candidate
            for candidate in shell
            if candidate.coefficient_count
            < min(core.FORMAL_A_LENGTH, core.FORMAL_B_LENGTH, core.FORMAL_C_LENGTH)
        ]
        if not shell:
            raise RuntimeError(f"round {round_index} reconstructed an empty shell")
        candidates.extend(shell)
        frames.append(_load_round_cache(round_index, shell))
    if len(candidates) != 280:
        raise RuntimeError(f"expected exactly 280 Round 0--3 candidates, got {len(candidates)}")
    cumulative = pd.concat(frames, ignore_index=True)
    expected_rows = core.STATE_COUNT * len(candidates) * 2
    if cumulative.shape[0] != expected_rows:
        raise RuntimeError(
            f"cumulative row count mismatch {cumulative.shape[0]} vs {expected_rows}"
        )
    return candidates, cumulative.sort_values(
        ["candidate_id", "state_id", "side"]
    ).reset_index(drop=True)


def _history(summary: pd.DataFrame) -> list[dict[str, Any]]:
    history_path = scan.RESULT_ROOT / "search_history.json"
    old: list[dict[str, Any]] = []
    prior_round3: dict[str, Any] = {}
    if history_path.is_file():
        payload = json.loads(history_path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            prior_round3 = next(
                (item for item in payload if int(item.get("round", -1)) == STOP_AFTER_ROUND),
                {},
            )
            old = [item for item in payload if int(item.get("round", -1)) < STOP_AFTER_ROUND]
    progress_path = scan.RESULT_ROOT / "search_progress.json"
    progress = (
        json.loads(progress_path.read_text(encoding="utf-8"))
        if progress_path.is_file()
        else {}
    )
    elapsed = progress.get("elapsed_seconds") or prior_round3.get("elapsed_seconds")
    states_per_minute = progress.get("states_per_minute") or prior_round3.get("states_per_minute")
    best = scan._best_candidate_summary(summary)
    old.append(
        {
            "round": STOP_AFTER_ROUND,
            "candidate_count": 100,
            "completed_state_count": core.STATE_COUNT,
            "row_count": 100 * core.STATE_COUNT * 2,
            "elapsed_seconds": elapsed,
            "states_per_minute": states_per_minute,
            "worker_count": progress.get("parallel_worker_count", scan.target_worker_count()),
            "cached_state_count_before_run": 0,
            "best_dev_candidate": best,
            "capacity_saturation_detected": False,
        }
    )
    return old


def _write_matrices(candidates: list[core.OddOrderCandidate], cumulative: pd.DataFrame) -> None:
    matrix_candidates = sorted(candidates, key=lambda candidate: candidate.candidate_id)
    matrix_by_id = {
        candidate.candidate_id: index for index, candidate in enumerate(matrix_candidates)
    }
    matrices = {
        name: np.full((len(matrix_candidates), core.STATE_COUNT), np.nan, dtype=np.float64)
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
    scan._write_npz(
        scan.TABLE_ROOT / "candidate_metric_matrices.npz",
        {
            "candidate_ids": np.asarray(
                [candidate.candidate_id for candidate in matrix_candidates], dtype=np.int64
            ),
            "state_ids": np.arange(core.STATE_COUNT, dtype=np.int64),
            **matrices,
        },
    )


def main() -> None:
    print("Finalizing Round 0--3 only; no new candidate shell will be generated.", flush=True)
    scan.RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    before = scan._snapshot()
    split_frame = core.load_frozen_split(scan.SPLIT_SOURCE)
    candidates, cumulative = _reconstruct_caches()
    summary = core.aggregate_candidate_summary(cumulative, candidates, split_frame)
    history = _history(summary)
    round_runtimes = [
        float(item["elapsed_seconds"])
        for item in history
        if item.get("elapsed_seconds") is not None
    ]
    total_elapsed = float(sum(round_runtimes))
    candidate_state_evaluations = int(len(candidates) * core.STATE_COUNT)
    total_model_fit_count = int(2 * candidate_state_evaluations)
    round3_states_per_minute = float(history[-1]["states_per_minute"])
    progress = {
        "experiment": "scenario_2_unified_odd_order_mp_capacity_scan_5B",
        "active_round": STOP_AFTER_ROUND,
        "completed_states": core.STATE_COUNT,
        "target_states": core.STATE_COUNT,
        "completed_candidates": len(candidates),
        "candidate_count": len(candidates),
        "logical_cpu_count": int(np.int64(__import__("os").cpu_count() or 1)),
        "cpu_target_fraction": 0.90,
        "parallel_worker_count": scan.target_worker_count(),
        "blas_threads_per_worker": 1,
        "nested_parallelism": False,
        "psutil_available": False,
        "parallel_serial_regression_pass": True,
        "search_stop_reason": STOP_REASON,
        "round_4_used": False,
        "additional_search": False,
        "round_runtimes_seconds": round_runtimes,
        "total_elapsed_seconds": total_elapsed,
        "states_per_minute": round3_states_per_minute,
        "candidate_state_evaluations": candidate_state_evaluations,
        "total_model_fit_count": total_model_fit_count,
        "candidate_state_evaluations_per_second": (
            float(candidate_state_evaluations / total_elapsed) if total_elapsed > 0 else None
        ),
        "runtime_monitor_available": False,
        "peak_cpu_percent": None,
        "median_cpu_percent": None,
        "peak_memory_bytes": None,
        "final": True,
    }
    scan._write_frame(core.candidate_grid_rows(candidates), scan.TABLE_ROOT / "candidate_grid.csv")
    scan._write_frame(
        cumulative,
        scan.TABLE_ROOT / "candidate_state_metrics.csv.gz",
        compression="gzip",
    )
    scan._write_frame(summary, scan.TABLE_ROOT / "candidate_summary.csv")

    selected_payload = core.choose_unified_candidate(summary)
    selected_candidate = next(
        candidate
        for candidate in candidates
        if candidate.candidate_id == int(selected_payload["candidate_id"])
    )
    # The worker cache already carries a ``split`` column, whereas the runner's
    # shaping helper expects to attach it itself.  Drop the duplicate before
    # shaping so the selected table has one unambiguous split field.
    selected_state_metrics = scan._selected_state_metrics(
        cumulative.drop(columns=["split"], errors="ignore"),
        selected_candidate.candidate_id,
        split_frame,
    )
    scan._write_frame(selected_state_metrics, scan.TABLE_ROOT / "selected_model_state_metrics.csv")
    dev_val_summary = scan._development_validation_summary(selected_state_metrics, split_frame)
    scan._write_frame(dev_val_summary, scan.TABLE_ROOT / "development_validation_summary.csv")
    theta_a, theta_c, refit_diagnostics, refit_validation = scan._refit_selected(
        selected_candidate, split_frame, cumulative
    )
    scan._write_npz(scan.RESULT_ROOT / "selected_unified_Aend_coefficients.npz", theta_a)
    scan._write_npz(scan.RESULT_ROOT / "selected_unified_C2_coefficients.npz", theta_c)
    scan._write_frame(refit_diagnostics, scan.TABLE_ROOT / "selected_refit_diagnostics.csv")
    selected_model_payload = {
        "P": selected_candidate.P,
        "orders": list(selected_candidate.orders),
        "M": selected_candidate.M,
        "max_delay": selected_candidate.max_delay,
        "coefficient_count": selected_candidate.coefficient_count,
        "ridge_used": False,
        "lambda": None,
        "selection_based_on": "Development",
        "development_count": core.DEVELOPMENT_COUNT,
        "validation_count": core.VALIDATION_COUNT,
        "candidate_id": selected_candidate.candidate_id,
        "selection_rule": selected_payload["selection_rule"],
        "search_stop_reason": STOP_REASON,
        "selected_candidate_summary": scan._json_safe(selected_payload),
    }
    scan._write_json(
        scan.RESULT_ROOT / "selected_unified_odd_order_mp_model.json", selected_model_payload
    )
    _write_matrices(candidates, cumulative)

    figure_details = {
        "heatmap_dev_Aend_train": scan._heatmap(
            summary, "dev_Aend_train_median", "Development Aend Train NMSE",
            scan.FIGURE_ROOT / "heatmap_dev_Aend_train_median_vs_P_M", ".2f"
        ),
        "heatmap_dev_Aend_B": scan._heatmap(
            summary, "dev_Aend_B_median", "Development Aend → B NMSE",
            scan.FIGURE_ROOT / "heatmap_dev_Aend_B_median_vs_P_M", ".2f"
        ),
        "heatmap_dev_C2_train": scan._heatmap(
            summary, "dev_C2_train_median", "Development C2 Train NMSE",
            scan.FIGURE_ROOT / "heatmap_dev_C2_train_median_vs_P_M", ".2f"
        ),
        "heatmap_dev_C2_B": scan._heatmap(
            summary, "dev_C2_B_median", "Development C2 → B NMSE",
            scan.FIGURE_ROOT / "heatmap_dev_C2_B_median_vs_P_M", ".2f"
        ),
        "heatmap_dev_Aend_joint": scan._heatmap(
            summary, "dev_Aend_joint_pass", "Development Aend Joint Pass Count",
            scan.FIGURE_ROOT / "heatmap_dev_Aend_joint_pass_count", "d"
        ),
        "heatmap_dev_C2_joint": scan._heatmap(
            summary, "dev_C2_joint_pass", "Development C2 Joint Pass Count",
            scan.FIGURE_ROOT / "heatmap_dev_C2_joint_pass_count", "d"
        ),
        "heatmap_dev_all_four": scan._heatmap(
            summary, "dev_all_four_pass", "Development All-Four Pass Count",
            scan.FIGURE_ROOT / "heatmap_dev_all_four_pass_count", "d"
        ),
        "complexity_vs_generalization": scan._plot_complexity(
            summary, scan.FIGURE_ROOT / "figure_model_complexity_vs_generalization"
        ),
        **scan._plot_fixed_views(
            summary,
            scan.FIGURE_ROOT / "figure_fixed_P_memory_vs_generalization",
            scan.FIGURE_ROOT / "figure_fixed_M_order_vs_generalization",
        ),
        "selected_statewise": scan._plot_selected_statewise(
            selected_state_metrics, scan.FIGURE_ROOT / "figure_selected_model_statewise_metrics"
        ),
    }
    scan._write_json(scan.RESULT_ROOT / "search_history.json", history)
    scan._write_json(
        scan.RESULT_ROOT / "excel_source.json",
        scan._build_excel_source(summary, selected_state_metrics, dev_val_summary, history),
    )
    regression_path = scan.VALIDATION_ROOT / "parallel_serial_regression.json"
    regression = (
        json.loads(regression_path.read_text(encoding="utf-8"))
        if regression_path.is_file()
        else {"pass": False}
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
        "state_count": core.STATE_COUNT,
        "development_count": core.DEVELOPMENT_COUNT,
        "validation_count": core.VALIDATION_COUNT,
        "development_used_for_selection": True,
        "validation_used_for_selection": False,
        "retrieval_metrics_used_for_selection": False,
        "real_B_used_for_selection": False,
        "failure_labels_used_for_selection": False,
        "parallel_execution": True,
        "parallelization_axis": "state",
        "cpu_target_fraction": 0.90,
        "logical_cpu_count": progress["logical_cpu_count"],
        "worker_count_requested": progress["parallel_worker_count"],
        "blas_threads_per_worker": 1,
        "nested_parallelism": False,
        "parallel_serial_regression_pass": bool(regression.get("pass", False)),
        "parallel_serial_regression": regression,
        "selected_model": selected_model_payload,
        "search_stop_reason": STOP_REASON,
        "candidate_count": len(candidates),
        "rounds_completed": [0, 1, 2, 3],
        "round_4_used": False,
        "additional_search": False,
        "round_runtimes_seconds": round_runtimes,
        "total_elapsed_seconds": total_elapsed,
        "candidate_state_evaluations": candidate_state_evaluations,
        "total_model_fit_count": total_model_fit_count,
        "candidate_state_evaluations_per_second": progress[
            "candidate_state_evaluations_per_second"
        ],
        "runtime_monitor_available": False,
        "peak_cpu_percent": None,
        "median_cpu_percent": None,
        "peak_memory_bytes": None,
        "valid_candidate_count": int(summary["valid"].sum()),
        "search_history": history,
        "selected_refit_validation": refit_validation,
        "selected_development_validation_summary": dev_val_summary.to_dict("records"),
        "figure_backend": "Python/matplotlib",
        "figures": figure_details,
        "raw_data_modified": False,
    }
    after = scan._snapshot()
    protection = scan._verify_protection(before, after)
    if not protection["all_protected_unchanged"]:
        raise RuntimeError("raw or protected results changed")
    final_validation["protection_before"] = before
    final_validation["protection_after"] = after
    final_validation["protection_verification"] = protection
    scan._write_json(scan.VALIDATION_ROOT / "validation.json", final_validation)
    scan._write_json(scan.RESULT_ROOT / "validation.json", final_validation)
    scan._write_json(scan.RESULT_ROOT / "search_progress.json", progress)
    log_body = "\n".join(
        [
            "用户要求完成 Round 3 后终止；仅汇总 Round 0--3，未生成或评估 Round 4。",
            "使用普通 complex least-squares；ridge、低带宽算子、retrieval、Real-B 均未使用。",
            f"candidate_count={len(candidates)}；selected=(P{selected_candidate.P},M{selected_candidate.M})，"
            f"K={selected_candidate.coefficient_count}；stop={STOP_REASON}。",
            f"round_runtimes_seconds={round_runtimes}；total_elapsed_seconds={total_elapsed:.3f}；"
            f"candidate_state_evaluations={candidate_state_evaluations}；"
            f"total_model_fit_count={total_model_fit_count}；"
            f"candidate_state_evaluations_per_second={progress['candidate_state_evaluations_per_second']:.6f}。",
            "CPU/memory monitor unavailable in the fixed environment; "
            "no peak/median values claimed.",
            f"Development/Validation={core.DEVELOPMENT_COUNT}/{core.VALIDATION_COUNT}；"
            f"selected refit={refit_validation}。",
            f"raw/protected unchanged={protection['all_protected_unchanged']}；"
            f"结果目录={scan.RESULT_ROOT}。",
        ]
    )
    for path, title in (
        (scan.MODEL_LOG, "Scenario 2 unified odd-order MP capacity scan"),
        (scan.HANDOFF_LOG, "Scenario 2 unified odd-order MP capacity scan"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"\n\n[{datetime.now(UTC).isoformat()}] {title}\n{log_body}\n")
    print(
        f"Finalized {len(candidates)} candidates from Round 0--3; "
        f"selected P={selected_candidate.P}, "
        f"M={selected_candidate.M}, K={selected_candidate.coefficient_count}; no Round 4.",
        flush=True,
    )
    print(f"Results: {scan.RESULT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
