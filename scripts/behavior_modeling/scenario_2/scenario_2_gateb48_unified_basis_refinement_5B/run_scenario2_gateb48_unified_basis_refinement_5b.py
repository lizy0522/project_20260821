"""Run Scenario 2 Gate-B48 unified-support basis refinement."""

# ruff: noqa: E402

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
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

from behavior_modeling.scenario_2.scenario_2_gateb48_unified_basis_refinement_5B.gateb48_refinement import (  # noqa: E402, E501
    BASELINE_SUPPORT_IDS,
    BENCHMARK_CANDIDATE_BATCH_SIZE,
    BENCHMARK_SAMPLE_SIZE,
    DEVELOPMENT_COUNT,
    DMAX,
    EXTRA_RIDGE_LAMBDA,
    K_SAFETY_CAP,
    RESULT_ROOT,
    RIDGE_GRID,
    RIDGE_LAMBDA,
    SPLIT_SEED,
    STATE_COUNT,
    TASK_NAME,
    THRESHOLD_DB,
    VALIDATION_COUNT,
    acceptance_test,
    aggregate_metrics,
    baseline_support_indices,
    benchmark_worker_counts,
    build_dictionary_frame,
    build_split_definition,
    build_summary_text,
    candidate_supports,
    checkpoint,
    evaluate_supports,
    metadata_frame,
    prune_acceptance,
    rank_summary,
    support_id_from_indices,
)
from behavior_modeling.shared.basis_function_selection.frozen_centered_dictionary import (  # noqa: E402
    build_frozen_centered_dictionary,
    gate_indices,
    validate_frozen_centered_dictionary,
)


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _append_log(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)


def _raw_snapshot() -> dict[str, Any]:
    raw_root = PROJECT_ROOT / "data" / "raw"
    files = [path for path in raw_root.rglob("*") if path.is_file()]
    stats = [path.stat() for path in files]
    return {
        "exists": raw_root.is_dir(),
        "file_count": len(files),
        "mat_count": sum(path.suffix.lower() == ".mat" for path in files),
        "bytes": int(sum(stat.st_size for stat in stats)),
        "latest_mtime_ns": max((stat.st_mtime_ns for stat in stats), default=None),
    }


def _support_payload(terms: tuple[Any, ...], support: tuple[int, ...]) -> dict[str, Any]:
    return {
        "support_id": support_id_from_indices(terms, support),
        "support_indices": list(map(int, support)),
        "basis_ids": [terms[int(index)].basis_id for index in support],
        "K": len(support),
    }


def _add_summary_columns(
    summary: dict[str, Any], *, stage: str, split: str, support: tuple[int, ...]
) -> dict[str, Any]:
    return {
        "stage": stage,
        "split": split,
        **summary,
        "support_indices": ";".join(str(int(index)) for index in support),
    }


def _write_plot_benchmark(frame: pd.DataFrame, path: Path) -> None:
    median = frame.groupby("worker_count", as_index=False)["throughput_states_per_second"].median()
    selected = int(frame["selected_worker_count"].iloc[0])
    fig, ax = plt.subplots(figsize=(7.6, 4.6), dpi=160)
    ax.plot(
        median["worker_count"],
        median["throughput_states_per_second"],
        marker="o",
        linewidth=1.8,
        color="#1f77b4",
    )
    selected_row = median.loc[median["worker_count"] == selected]
    if not selected_row.empty:
        ax.scatter(
            selected_row["worker_count"],
            selected_row["throughput_states_per_second"],
            color="#d62728",
            zorder=3,
            label=f"selected={selected}",
        )
    ax.set_xlabel("Worker processes")
    ax.set_ylabel("States per second (median)")
    ax.set_title("Gate-B48 worker benchmark, BLAS threads/worker=1")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_plot_pareto(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=160)
    if not frame.empty:
        scatter = ax.scatter(
            frame["K"],
            frame["joint_Q95_dB"],
            c=frame["pass_count"],
            cmap="viridis",
            s=58,
            edgecolor="black",
            linewidth=0.35,
        )
        for _, row in frame.iterrows():
            raw_label = str(row["label"])
            label_parts = []
            if "Baseline17" in raw_label:
                label_parts.append("B17")
            if "best_forward" in raw_label:
                label_parts.append("forward")
            if "best_pruned" in raw_label:
                label_parts.append("pruned")
            if "best_swap" in raw_label:
                label_parts.append("swap")
            plot_label = "/".join(label_parts) or "support"
            ax.annotate(
                plot_label,
                (row["K"], row["joint_Q95_dB"]),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
            )
        fig.colorbar(scatter, ax=ax, label="Development pass count")
    ax.set_xlabel("Support size K")
    ax.set_ylabel("Development joint B Q95 (dB)")
    ax.set_title("Gate-B48 support Pareto view")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _write_plot_ridge(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=160)
    for support_number, (_, group) in enumerate(frame.groupby("support_id"), start=1):
        group = group.sort_values("lambda")
        ax.plot(
            group["lambda"].replace(0.0, np.nan),
            group["joint_Q95_dB"],
            marker="o",
            linewidth=1.2,
            label=f"Pareto-{support_number} (K={int(group['K'].iloc[0])})",
        )
    ax.set_xscale("log")
    ax.set_xlabel("Ridge lambda (zero-OLS point omitted on log axis)")
    ax.set_ylabel("Development joint B Q95 (dB)")
    ax.set_title("Ridge scan on Pareto supports")
    ax.grid(alpha=0.25, which="both")
    if frame["support_id"].nunique() <= 8:
        ax.legend(frameon=False, fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _rank_prune_candidates(
    candidate_summaries: list[dict[str, Any]],
    removed_order: dict[str, int],
) -> list[dict[str, Any]]:
    return sorted(
        candidate_summaries,
        key=lambda item: (
            float(item["joint_Q95_dB"]),
            float(item["joint_worst_dB"]),
            -int(removed_order[item["support_id"]]),
            str(item["support_id"]),
        ),
    )


def main() -> None:
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    work_log = PROJECT_ROOT / "work_logs" / "behavior_modeling" / "scenario_2" /TASK_NAME / "execution_log.txt"  # noqa: E501
    checkpoint_path = RESULT_ROOT / "search_checkpoint.json"
    raw_before = _raw_snapshot()
    _append_log(
        work_log,
        "\n"
        f"[{_now()}] Start new task {TASK_NAME}\n"
        "Mode: new task; Scenario 2 Gate-B48 unified-support refinement.\n"
        "Contract: Development-only selection, Validation after freeze; "
        "no retrieval/DPD/clustering/low-bandwidth.\n"
        f"Raw snapshot before: {json.dumps(raw_before, ensure_ascii=False, sort_keys=True)}\n",
    )

    terms = tuple(build_frozen_centered_dictionary())
    dictionary_gate = validate_frozen_centered_dictionary(terms)
    gate_b = tuple(gate_indices(terms, "GATE_B"))
    if len(gate_b) != 48:
        raise RuntimeError(f"Gate-B count mismatch: {len(gate_b)}")
    dictionary_frame = build_dictionary_frame(terms, gate_b)
    dictionary_frame.to_csv(RESULT_ROOT / "01_gateb48_dictionary.csv", index=False)

    split_frame, development_ids, validation_ids = build_split_definition()
    split_frame.to_csv(RESULT_ROOT / "02_development_validation_split.csv", index=False)
    if len(development_ids) != DEVELOPMENT_COUNT or len(validation_ids) != VALIDATION_COUNT:
        raise RuntimeError("split count gate failed")

    baseline_support = baseline_support_indices(terms, gate_b)
    baseline_id = support_id_from_indices(terms, baseline_support)
    if tuple(terms[index].basis_id for index in baseline_support) != BASELINE_SUPPORT_IDS:
        raise RuntimeError("Baseline17 ordering does not match the frozen contract")

    remaining_nonlinear = [
        int(index)
        for index in gate_b
        if index not in set(baseline_support) and terms[index].order > 1
    ]
    if len(remaining_nonlinear) != 31:
        raise RuntimeError(
            f"Gate-B remaining candidate count must be 31, got {len(remaining_nonlinear)}"
        )

    benchmark_supports = [(baseline_id, baseline_support)]
    for index in remaining_nonlinear[:BENCHMARK_CANDIDATE_BATCH_SIZE]:
        support = tuple([*baseline_support, index])
        benchmark_supports.append((support_id_from_indices(terms, support), support))
    benchmark_sample = tuple(int(state_id) for state_id in development_ids[:BENCHMARK_SAMPLE_SIZE])
    benchmark_frame, worker_count = benchmark_worker_counts(
        terms,
        gate_b,
        benchmark_sample,
        benchmark_supports,
    )
    benchmark_frame.to_csv(RESULT_ROOT / "00_parallel_benchmark.csv", index=False)
    _write_plot_benchmark(benchmark_frame, RESULT_ROOT / "11_parallel_benchmark.png")
    checkpoint(
        checkpoint_path,
        phase="benchmark_complete",
        selected_worker_count=worker_count,
        benchmark_sample_state_ids=list(benchmark_sample),
        benchmark_candidate_batch_size=len(benchmark_supports),
    )

    support_payloads: dict[str, tuple[int, ...]] = {
        baseline_id: baseline_support,
    }
    support_frames: dict[str, pd.DataFrame] = {}
    support_summaries: dict[str, dict[str, Any]] = {}
    forward_history: list[dict[str, Any]] = []
    backward_history: list[dict[str, Any]] = []
    swap_history: list[dict[str, Any]] = []

    def evaluate_fixed_supports(
        executor: ProcessPoolExecutor,
        supports: list[tuple[str, tuple[int, ...]]],
        ridge_lambda: float,
        *,
        label: str,
        state_ids: np.ndarray,
        exact_condition: bool = False,
    ) -> list[dict[str, Any]]:
        frame = evaluate_supports(
            executor,
            state_ids,
            supports,
            ridge_lambda,
            exact_condition=exact_condition,
            progress_label=label,
        )
        summaries = []
        for support_id, support in supports:
            support_payloads[support_id] = tuple(support)
            selected = frame.loc[frame["support_id"] == support_id].copy()
            support_frames[support_id] = selected
            summary = aggregate_metrics(frame, support_id)
            support_summaries[support_id] = summary
            summaries.append(summary)
        return summaries

    context = __import__("multiprocessing").get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=worker_count,
        mp_context=context,
        initializer=__import__(
            "behavior_modeling.basis_function_selection.gateb48_refinement",
            fromlist=["_worker_init"],
        )._worker_init,
        initargs=(terms, gate_b),
    ) as executor:
        baseline_summary = evaluate_fixed_supports(
            executor,
            [(baseline_id, baseline_support)],
            RIDGE_LAMBDA,
            label="baseline-development",
            state_ids=development_ids,
        )[0]

        current_support = baseline_support
        current_id = baseline_id
        current_summary = baseline_summary
        accepted_forward_ids = [baseline_id]
        remaining = set(remaining_nonlinear)
        round_number = 0
        while remaining and len(current_support) < K_SAFETY_CAP:
            round_number += 1
            candidate_specs = []
            candidate_indices: dict[str, int] = {}
            for candidate_id, support, index in candidate_supports(terms, gate_b, current_support):
                if index in remaining:
                    full_id = support_id_from_indices(terms, support)
                    candidate_specs.append((full_id, support))
                    candidate_indices[full_id] = index
            if not candidate_specs:
                break
            candidate_summaries = evaluate_fixed_supports(
                executor,
                candidate_specs,
                RIDGE_LAMBDA,
                label=f"forward-round-{round_number}",
                state_ids=development_ids,
            )
            candidate_summaries = sorted(candidate_summaries, key=rank_summary)
            ranking = {
                item["support_id"]: position + 1
                for position, item in enumerate(candidate_summaries)
            }
            winner = candidate_summaries[0]
            winner_eligible, _ = acceptance_test(current_summary, winner)
            for candidate in candidate_summaries:
                eligible, reason = acceptance_test(current_summary, candidate)
                forward_history.append(
                    {
                        "phase": "forward",
                        "round": round_number,
                        "current_support_id": current_id,
                        "candidate_support_id": candidate["support_id"],
                        "added_basis_id": terms[
                            candidate_indices[candidate["support_id"]]
                        ].basis_id,
                        "added_order": terms[candidate_indices[candidate["support_id"]]].order,
                        "rank": ranking[candidate["support_id"]],
                        "eligible": eligible,
                        "accepted": bool(
                            winner_eligible and candidate["support_id"] == winner["support_id"]
                        ),
                        "acceptance_reason": reason,
                        **candidate,
                    }
                )
            if not winner_eligible:
                break
            added_index = candidate_indices[winner["support_id"]]
            current_support = tuple([*current_support, added_index])
            current_id = winner["support_id"]
            current_summary = support_summaries[current_id]
            accepted_forward_ids.append(current_id)
            remaining.remove(added_index)
            checkpoint(
                checkpoint_path,
                phase="forward_round_complete",
                round=round_number,
                current_support=_support_payload(terms, current_support),
                current_summary=current_summary,
            )

        best_forward_id = current_id
        best_forward_summary = current_summary

        prune_round = 0
        last_prune_candidates: list[tuple[dict[str, Any], int]] = []
        while True:
            prune_round += 1
            deletable = [int(index) for index in current_support if terms[int(index)].order > 1]
            if not deletable:
                break
            deletion_specs: list[tuple[str, tuple[int, ...]]] = []
            removed_order: dict[str, int] = {}
            removed_index: dict[str, int] = {}
            for index in deletable:
                support = tuple(value for value in current_support if value != index)
                candidate_id = support_id_from_indices(terms, support)
                deletion_specs.append((candidate_id, support))
                removed_order[candidate_id] = int(terms[index].order)
                removed_index[candidate_id] = index
            deletion_summaries = evaluate_fixed_supports(
                executor,
                deletion_specs,
                RIDGE_LAMBDA,
                label=f"backward-round-{prune_round}",
                state_ids=development_ids,
            )
            last_prune_candidates = [
                (summary, removed_index[summary["support_id"]]) for summary in deletion_summaries
            ]
            eligible = []
            candidate_history: list[dict[str, Any]] = []
            for summary in deletion_summaries:
                candidate_eligible, reason = prune_acceptance(current_summary, summary)
                candidate_history.append(
                    {
                        "phase": "backward",
                        "round": prune_round,
                        "current_support_id": current_id,
                        "candidate_support_id": summary["support_id"],
                        "removed_basis_id": terms[removed_index[summary["support_id"]]].basis_id,
                        "removed_order": terms[removed_index[summary["support_id"]]].order,
                        "eligible": candidate_eligible,
                        "accepted": False,
                        "acceptance_reason": reason,
                        **summary,
                    }
                )
                if candidate_eligible:
                    eligible.append(summary)
            if not eligible:
                backward_history.extend(candidate_history)
                break
            ranked_eligible = _rank_prune_candidates(eligible, removed_order)
            winner = ranked_eligible[0]
            for item in candidate_history:
                if item["candidate_support_id"] == winner["support_id"]:
                    item["accepted"] = True
            backward_history.extend(candidate_history)
            removed = removed_index[winner["support_id"]]
            current_support = tuple(value for value in current_support if value != removed)
            current_id = winner["support_id"]
            current_summary = support_summaries[current_id]
            remaining.add(removed)
            checkpoint(
                checkpoint_path,
                phase="backward_round_complete",
                round=prune_round,
                current_support=_support_payload(terms, current_support),
                current_summary=current_summary,
            )

        best_pruned_id = current_id
        best_pruned_summary = current_summary

        # One local 1-swap rescue after the backward plateau.
        swap_candidates = []
        if last_prune_candidates:
            sorted_weak = sorted(
                last_prune_candidates,
                key=lambda item: (
                    float(item[0]["joint_Q95_dB"]),
                    float(item[0]["joint_worst_dB"]),
                    -int(terms[item[1]].order),
                    str(item[0]["support_id"]),
                ),
            )
            swap_candidates = sorted_weak[:8]
        add_specs = []
        add_index: dict[str, int] = {}
        for candidate_id, support, index in candidate_supports(terms, gate_b, current_support):
            if index in remaining:
                full_id = support_id_from_indices(terms, support)
                add_specs.append((full_id, support))
                add_index[full_id] = index
        add_summaries = (
            evaluate_fixed_supports(
                executor,
                add_specs[:8],
                RIDGE_LAMBDA,
                label="swap-addition-probe",
                state_ids=development_ids,
            )
            if add_specs
            else []
        )
        add_summaries = sorted(add_summaries, key=rank_summary)
        top_adds = add_summaries[:8]
        swap_specs: list[tuple[str, tuple[int, ...]]] = []
        swap_map: dict[str, tuple[int, int]] = {}
        for _weak_summary, remove_index in swap_candidates:
            weak_support = tuple(value for value in current_support if value != remove_index)
            for add_summary in top_adds:
                add_index_value = add_index[add_summary["support_id"]]
                if add_index_value in weak_support:
                    continue
                support = tuple([*weak_support, add_index_value])
                support = tuple(
                    sorted(
                        support,
                        key=lambda value: next(
                            i for i, item in enumerate(terms) if item.index == value
                        ),
                    )
                )
                candidate_id = support_id_from_indices(terms, support)
                if candidate_id not in {name for name, _ in swap_specs}:
                    swap_specs.append((candidate_id, support))
                    swap_map[candidate_id] = (remove_index, add_index_value)
                if len(swap_specs) >= 64:
                    break
            if len(swap_specs) >= 64:
                break
        if swap_specs:
            swap_summaries = evaluate_fixed_supports(
                executor,
                swap_specs,
                RIDGE_LAMBDA,
                label="local-swap",
                state_ids=development_ids,
            )
            ranked_swaps = sorted(swap_summaries, key=rank_summary)
            winner = ranked_swaps[0] if ranked_swaps else None
            winner_eligible = bool(
                winner is not None and acceptance_test(current_summary, winner)[0]
            )
            for summary in swap_summaries:
                eligible, reason = acceptance_test(current_summary, summary)
                remove_index, add_index_value = swap_map[summary["support_id"]]
                swap_history.append(
                    {
                        "phase": "local_swap",
                        "current_support_id": current_id,
                        "candidate_support_id": summary["support_id"],
                        "removed_basis_id": terms[remove_index].basis_id,
                        "added_basis_id": terms[add_index_value].basis_id,
                        "eligible": eligible,
                        "accepted": bool(
                            winner_eligible and summary["support_id"] == winner["support_id"]
                        ),
                        "acceptance_reason": reason,
                        **summary,
                    }
                )
            if ranked_swaps:
                if winner_eligible:
                    current_id = winner["support_id"]
                    current_support = support_payloads[current_id]
                    current_summary = support_summaries[current_id]

        best_swap_id = current_id
        best_swap_summary = current_summary

        milestone_ids = []
        for support_id in [
            baseline_id,
            best_forward_id,
            best_pruned_id,
            best_swap_id,
            *accepted_forward_ids,
        ]:
            if support_id not in milestone_ids:
                milestone_ids.append(support_id)
        milestone_summaries = [support_summaries[support_id] for support_id in milestone_ids]
        nondominated_ids = []
        for summary in milestone_summaries:
            dominated = False
            for other in milestone_summaries:
                if other["support_id"] == summary["support_id"]:
                    continue
                no_worse = (
                    int(other["pass_count"]) >= int(summary["pass_count"])
                    and float(other["joint_Q95_dB"]) <= float(summary["joint_Q95_dB"])
                    and float(other["joint_worst_dB"]) <= float(summary["joint_worst_dB"])
                    and int(other["K"]) <= int(summary["K"])
                )
                strict = (
                    int(other["pass_count"]) > int(summary["pass_count"])
                    or float(other["joint_Q95_dB"]) < float(summary["joint_Q95_dB"])
                    or float(other["joint_worst_dB"]) < float(summary["joint_worst_dB"])
                    or int(other["K"]) < int(summary["K"])
                )
                if no_worse and strict:
                    dominated = True
                    break
            if not dominated:
                nondominated_ids.append(summary["support_id"])
        if baseline_id not in nondominated_ids:
            nondominated_ids.append(baseline_id)
        pareto_ids = list(dict.fromkeys(nondominated_ids))
        pareto_rows = []
        for support_id in pareto_ids:
            summary = support_summaries[support_id]
            roles = []
            if support_id == baseline_id:
                roles.append("Baseline17")
            if support_id == best_forward_id:
                roles.append("best_forward")
            if support_id == best_pruned_id:
                roles.append("best_pruned")
            if support_id == best_swap_id:
                roles.append("best_swap")
            if support_id in nondominated_ids:
                roles.append("non_dominated")
            pareto_rows.append(
                {
                    "support_id": support_id,
                    "label": "+".join(roles),
                    "support_indices": ";".join(map(str, support_payloads[support_id])),
                    "basis_ids": ";".join(
                        terms[index].basis_id for index in support_payloads[support_id]
                    ),
                    **summary,
                }
            )
        pareto_frame = pd.DataFrame(pareto_rows).sort_values(
            ["K", "joint_Q95_dB", "joint_worst_dB", "support_id"]
        )
        pareto_frame.to_csv(RESULT_ROOT / "06_pareto_supports.csv", index=False)
        _write_plot_pareto(pareto_frame, RESULT_ROOT / "12_support_pareto.png")

        ridge_rows: list[dict[str, Any]] = []
        ridge_frames: dict[str, pd.DataFrame] = {}
        ridge_lambdas = list(RIDGE_GRID)
        for ridge_lambda in ridge_lambdas:
            ridge_specs = [
                (f"{support_id}@lambda={ridge_lambda:.12g}", support_payloads[support_id])
                for support_id in pareto_ids
            ]
            evaluated = evaluate_supports(
                executor,
                development_ids,
                ridge_specs,
                ridge_lambda,
                progress_label=f"ridge-{ridge_lambda:.0e}",
            )
            for support_id in pareto_ids:
                eval_id = f"{support_id}@lambda={ridge_lambda:.12g}"
                frame = evaluated.loc[evaluated["support_id"] == eval_id].copy()
                frame["support_id"] = support_id
                ridge_frames[eval_id] = frame
                summary = aggregate_metrics(
                    evaluated.assign(
                        support_id=evaluated["support_id"].str.replace(
                            f"@lambda={ridge_lambda:.12g}", "", regex=False
                        )
                    ),
                    support_id,
                )
                ridge_rows.append(
                    {
                        "support_id": support_id,
                        "lambda": ridge_lambda,
                        "selection_split": "Development",
                        **summary,
                    }
                )
        ridge_frame = pd.DataFrame(ridge_rows)
        prelim_best = ridge_frame.sort_values(
            ["pass_count", "joint_Q95_dB", "joint_worst_dB", "K", "lambda", "support_id"],
            ascending=[False, True, True, True, True, True],
        ).iloc[0]
        if float(prelim_best["lambda"]) == 1e-6:
            ridge_lambda = EXTRA_RIDGE_LAMBDA
            ridge_specs = [
                (f"{support_id}@lambda={ridge_lambda:.12g}", support_payloads[support_id])
                for support_id in pareto_ids
            ]
            evaluated = evaluate_supports(
                executor,
                development_ids,
                ridge_specs,
                ridge_lambda,
                progress_label="ridge-extra-1e-5",
            )
            for support_id in pareto_ids:
                eval_id = f"{support_id}@lambda={ridge_lambda:.12g}"
                normalized = evaluated.assign(
                    support_id=evaluated["support_id"].str.replace(
                        f"@lambda={ridge_lambda:.12g}", "", regex=False
                    )
                )
                summary = aggregate_metrics(normalized, support_id)
                ridge_rows.append(
                    {
                        "support_id": support_id,
                        "lambda": ridge_lambda,
                        "selection_split": "Development",
                        **summary,
                    }
                )
        ridge_frame = pd.DataFrame(ridge_rows)
        ridge_frame = ridge_frame.drop_duplicates(subset=["support_id", "lambda"], keep="last")
        ridge_order = ridge_frame.sort_values(
            ["pass_count", "joint_Q95_dB", "joint_worst_dB", "K", "lambda", "support_id"],
            ascending=[False, True, True, True, True, True],
        ).index
        ridge_frame["rank"] = 0
        ridge_frame.loc[ridge_order, "rank"] = np.arange(1, len(ridge_order) + 1)
        ridge_frame.to_csv(RESULT_ROOT / "07_ridge_scan.csv", index=False)
        _write_plot_ridge(ridge_frame, RESULT_ROOT / "13_ridge_scan.png")

        selected_row = ridge_frame.sort_values(
            ["pass_count", "joint_Q95_dB", "joint_worst_dB", "K", "lambda", "support_id"],
            ascending=[False, True, True, True, True, True],
        ).iloc[0]
        selected_support_id = str(selected_row["support_id"])
        selected_lambda = float(selected_row["lambda"])
        selected_support = support_payloads[selected_support_id]

        # Exact condition-number metrics are computed only after the support and
        # lambda are frozen, keeping the search itself state-major and cheap.
        baseline_dev_exact = evaluate_supports(
            executor,
            development_ids,
            [(baseline_id, baseline_support)],
            RIDGE_LAMBDA,
            exact_condition=True,
            progress_label="baseline-development-exact-condition",
        )
        selected_dev_exact = evaluate_supports(
            executor,
            development_ids,
            [(selected_support_id, selected_support)],
            selected_lambda,
            exact_condition=True,
            progress_label="selected-development-exact-condition",
        )
        baseline_dev_summary_exact = aggregate_metrics(baseline_dev_exact, baseline_id)
        selected_dev_summary_exact = aggregate_metrics(selected_dev_exact, selected_support_id)

        baseline_validation = evaluate_supports(
            executor,
            validation_ids,
            [(baseline_id, baseline_support)],
            RIDGE_LAMBDA,
            exact_condition=True,
            progress_label="baseline-validation",
        )
        selected_validation = evaluate_supports(
            executor,
            validation_ids,
            [(selected_support_id, selected_support)],
            selected_lambda,
            exact_condition=True,
            progress_label="selected-validation",
        )

    # The exact baseline per-state Development/Validation table is useful for
    # later audit and does not involve any LUT or retrieval calculation.
    baseline_dev_exact["split"] = "Development"
    baseline_validation["split"] = "Validation"
    baseline_all = pd.concat([baseline_dev_exact, baseline_validation], ignore_index=True)
    baseline_all = metadata_frame().merge(baseline_all, on="state_id", how="left")
    baseline_all.to_csv(RESULT_ROOT / "03_baseline17_per_state_metrics.csv", index=False)

    forward_frame = pd.DataFrame(forward_history)
    backward_frame = pd.DataFrame(backward_history)
    if forward_frame.empty:
        forward_frame = pd.DataFrame(
            [{"phase": "forward", "round": 0, "accepted": False, "support_id": baseline_id}]
        )
    if backward_frame.empty:
        backward_frame = pd.DataFrame(
            [{"phase": "backward", "round": 0, "accepted": False, "support_id": baseline_id}]
        )
    if swap_history:
        forward_frame = pd.concat([forward_frame, pd.DataFrame(swap_history)], ignore_index=True)
    forward_frame.to_csv(RESULT_ROOT / "04_forward_search_history.csv", index=False)
    backward_frame.to_csv(RESULT_ROOT / "05_backward_pruning_history.csv", index=False)

    validation_baseline = aggregate_metrics(baseline_validation, baseline_id)
    validation_selected = aggregate_metrics(selected_validation, selected_support_id)
    selected_pass_delta = int(validation_selected["pass_count"]) - int(
        validation_baseline["pass_count"]
    )
    selected_q95_gain = float(validation_baseline["joint_Q95_dB"]) - float(
        validation_selected["joint_Q95_dB"]
    )
    selected_side_ok = (
        float(validation_selected["Aend_B_median_dB"])
        - float(validation_baseline["Aend_B_median_dB"])
        <= 0.05
        and float(validation_selected["C2_B_median_dB"])
        - float(validation_baseline["C2_B_median_dB"])
        <= 0.05
    )
    adopted = bool(
        int(validation_selected["pass_count"]) >= int(validation_baseline["pass_count"])
        and (selected_q95_gain >= 0.05 or selected_pass_delta >= 2)
        and selected_side_ok
    )
    validation_rows = []
    for variant, frame, summary, support_id, ridge_lambda in (
        (
            "Baseline17",
            baseline_validation,
            validation_baseline,
            baseline_id,
            RIDGE_LAMBDA,
        ),
        (
            "DevelopmentSelected",
            selected_validation,
            validation_selected,
            selected_support_id,
            selected_lambda,
        ),
    ):
        value = frame.copy()
        value["split"] = "Validation"
        value["model_variant"] = variant
        value["support_id"] = support_id
        value["ridge_lambda"] = ridge_lambda
        value["selected_adopted"] = adopted and variant == "DevelopmentSelected"
        validation_rows.append(value)
    validation_comparison = pd.concat(validation_rows, ignore_index=True)
    validation_comparison = metadata_frame().merge(
        validation_comparison, on="state_id", how="inner", suffixes=("", "_metric")
    )
    validation_comparison.to_csv(RESULT_ROOT / "08_validation_comparison.csv", index=False)

    final_support_id = selected_support_id if adopted else baseline_id
    final_support = selected_support if adopted else baseline_support
    final_lambda = selected_lambda if adopted else RIDGE_LAMBDA
    final_role = "adopted_selected" if adopted else "retained_Baseline17"
    final_rows = []
    for position, index in enumerate(final_support, start=1):
        term = terms[index]
        final_rows.append(
            {
                "model_role": final_role,
                "support_id": final_support_id,
                "ridge_lambda": final_lambda,
                "support_size_K": len(final_support),
                "term_position": position,
                "global_index": int(index),
                "basis_id": term.basis_id,
                "formula": term.formula,
                "family": term.family,
                "order": term.order,
                "signal_delay": term.signal_delay,
                "envelope_delay": term.envelope_delay,
                "selection_split": "Development",
                "validation_adopted": adopted,
            }
        )
    pd.DataFrame(final_rows).to_csv(RESULT_ROOT / "09_final_model_definition.csv", index=False)

    raw_after = _raw_snapshot()
    raw_unchanged = raw_before == raw_after
    summary_payload = {
        "task": TASK_NAME,
        "dictionary_gate": dictionary_gate,
        "state_count": STATE_COUNT,
        "development_count": DEVELOPMENT_COUNT,
        "validation_count": VALIDATION_COUNT,
        "split_seed": SPLIT_SEED,
        "split_source": (
            "scripts/behavior_fingerprint_ranking_consistency/"
            "scenario2_y_lut_weighted_fusion_analysis.py::build_query_split"
        ),
        "abc_bounds": [0, 12288, 17203, 24576],
        "valid_lengths": {"A": 12286, "B": 4913, "C": 7371},
        "dmax": DMAX,
        "baseline_support_ids": list(BASELINE_SUPPORT_IDS),
        "baseline_support_id": baseline_id,
        "baseline_development": baseline_dev_summary_exact,
        "best_forward_support_id": best_forward_id,
        "best_forward": best_forward_summary,
        "best_pruned_support_id": best_pruned_id,
        "best_pruned": best_pruned_summary,
        "best_swap_support_id": best_swap_id,
        "best_swap": best_swap_summary,
        "development_selected_support_id": selected_support_id,
        "development_selected_lambda": selected_lambda,
        "development_selected": selected_dev_summary_exact,
        "validation_baseline": validation_baseline,
        "validation_selected": validation_selected,
        "validation_pass_delta_selected_minus_baseline": selected_pass_delta,
        "validation_joint_Q95_gain_dB": selected_q95_gain,
        "validation_side_median_guard_pass": selected_side_ok,
        "adopted": adopted,
        "final_model_role": final_role,
        "final_support_id": final_support_id,
        "final_lambda": final_lambda,
        "final_support_size_K": len(final_support),
        "selected_worker_count": worker_count,
        "blas_threads_per_worker": 1,
        "gpu_used": False,
        "retrieval_run": False,
        "clustering_run": False,
        "dpd_run": False,
        "low_bandwidth_run": False,
        "raw_snapshot_before": raw_before,
        "raw_snapshot_after": raw_after,
        "raw_data_modified": not raw_unchanged,
        "development_improvement_not_confirmed": not adopted,
        "threshold_db": THRESHOLD_DB,
    }
    (RESULT_ROOT / "10_final_result_summary.txt").write_text(
        build_summary_text(summary_payload), encoding="utf-8"
    )
    checkpoint(
        checkpoint_path,
        phase="completed",
        final_model_role=final_role,
        final_support=_support_payload(terms, final_support),
        final_lambda=final_lambda,
        adopted=adopted,
        raw_data_modified=not raw_unchanged,
    )
    _append_log(
        work_log,
        f"[{_now()}] Completed {TASK_NAME}\n"
        f"Worker count={worker_count}; raw unchanged={raw_unchanged}; adopted={adopted}; "
        f"final_support_K={len(final_support)}; final_lambda={final_lambda:.12g}.\n"
        "Results: 00_parallel_benchmark.csv through 13_ridge_scan.png plus "
        "search_checkpoint.json.\n"
        "No LUT retrieval, clustering, DPD, or low-bandwidth calculation was run.\n",
    )
    print(
        json.dumps(
            {
                "task": TASK_NAME,
                "result_root": str(RESULT_ROOT),
                "selected_worker_count": worker_count,
                "development_selected_support_id": selected_support_id,
                "development_selected_lambda": selected_lambda,
                "adopted": adopted,
                "final_support_id": final_support_id,
                "final_lambda": final_lambda,
                "raw_data_modified": not raw_unchanged,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    __import__("multiprocessing").freeze_support()
    main()
