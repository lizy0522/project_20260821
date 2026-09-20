"""Run the independent Scenario 2 Hard-20 Envelope75 basis experiment."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
from pathlib import Path

# Keep every numerical worker CPU-only and single-threaded at the BLAS layer.
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

from behavior_modeling.shared.basis_function_selection.envelope_dictionary import (  # noqa: E402
    EnvelopeBasis,
    build_envelope_dictionary,
    dictionary_gate,
)
from behavior_modeling.shared.basis_function_selection.hard20_envelope75_selection import (  # noqa: E402
    BASELINE_SUPPORT_IDS,
    EXPECTED_RAW_BYTES,
    EXPECTED_RAW_FILE_COUNT,
    EXPECTED_RAW_MANIFEST,
    EXPECTED_RAW_MAT_COUNT,
    FULL_LENGTH,
    HARD_STATE_COUNT,
    K_SAFETY_CAP,
    OLS_CONDITION_HARD_LIMIT,
    RESULT_ROOT,
    RIDGE_BOUNDARY_HIGH,
    RIDGE_BOUNDARY_LOW,
    RIDGE_EXTRA_HIGH,
    RIDGE_GRID,
    TARGET_NMSE_DB,
    TASK_NAME,
    TEST_END,
    TRAIN_END,
    TrainSelectionEvaluator,
    _raw_manifest,
    _write_task_definition,
    baseline_support,
    checkpoint,
    create_train_cache,
    deletion_allowed,
    pareto_filter,
    populate_train_cache,
    run_final_evaluation,
    run_ranking,
    support_ids,
    support_improves,
    support_sort_key,
)

WORK_LOG = PROJECT_ROOT / "work_logs" / "behavior_modeling" / "scenario_2" /TASK_NAME / "execution_log.txt"  # noqa: E501
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"
CHECKPOINT_PATH = RESULT_ROOT / "12_search_checkpoint.json"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message)


def _dictionary_hash(terms: tuple[EnvelopeBasis, ...]) -> str:
    payload = [
        {
            "index": term.index,
            "basis_id": term.basis_id,
            "formula": term.formula,
            "family": term.family,
            "order": term.order,
            "signal_delay": term.signal_delay,
            "envelope_delay": term.envelope_delay,
            "mandatory": term.mandatory,
        }
        for term in terms
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _support_label(terms: tuple[EnvelopeBasis, ...], support: tuple[int, ...]) -> str:
    return ";".join(support_ids(terms, support))


def _dictionary_frame(
    terms: tuple[EnvelopeBasis, ...],
    gate_b_ids: set[str],
    baseline_ids: set[str],
) -> pd.DataFrame:
    rows = []
    for term in terms:
        rows.append(
            {
                "global_index": term.index,
                "basis_id": term.basis_id,
                "formula": term.formula,
                "family": term.family,
                "order": term.order,
                "signal_delay": term.signal_delay,
                "envelope_delay": term.envelope_delay,
                "mandatory": term.mandatory,
                "in_gate_b48": term.basis_id in gate_b_ids,
                "in_baseline17": term.basis_id in baseline_ids,
                "new_even_order_term": term.order in {4, 6, 8},
            }
        )
    frame = pd.DataFrame(rows)
    if len(frame) != 75 or frame["basis_id"].nunique() != 75:
        raise RuntimeError("Envelope75 dictionary output contract failed")
    if int(frame["in_gate_b48"].sum()) != 48:
        raise RuntimeError("Gate-B48 is not a 48-term Envelope75 subset")
    if int(frame["in_baseline17"].sum()) != 17:
        raise RuntimeError("Baseline17 is not a 17-term Envelope75 subset")
    return frame


def _summary_from_capacity(
    frame: pd.DataFrame,
    support_id: str,
) -> dict[str, object]:
    selected = frame.loc[frame["support_id"] == support_id].copy()
    if len(selected) != HARD_STATE_COUNT:
        raise RuntimeError(f"Capacity support {support_id} has {len(selected)} rows")
    train = selected["train_nmse_db"].to_numpy(dtype=float)
    condition = selected["condition_number"].to_numpy(dtype=float)
    return {
        "support_id": support_id,
        "K": int(selected["K"].iloc[0]),
        "N_train40": int(np.count_nonzero(train < TARGET_NMSE_DB)),
        "train_median_dB": float(np.median(train)),
        "train_worst_dB": float(np.max(train)),
        "condition_q99": float(np.quantile(condition, 0.99)),
        "condition_max": float(np.max(condition)),
        "all_full_rank": bool(
            np.all(selected["rank"].to_numpy(dtype=int) == int(selected["K"].iloc[0]))
        ),
        "all_finite": bool(selected["finite"].all()),
        "numerically_reliable": bool(
            np.all(selected["rank"].to_numpy(dtype=int) == int(selected["K"].iloc[0]))
            and np.all(np.isfinite(condition))
            and np.quantile(condition, 0.99) <= OLS_CONDITION_HARD_LIMIT
            and bool(selected["finite"].all())
        ),
    }


def _history_row(
    phase: str,
    round_number: int,
    summary: dict[str, object],
    *,
    accepted: bool,
    added: str = "",
    removed: str = "",
    reason: str = "",
) -> dict[str, object]:
    return {
        "phase": phase,
        "round": round_number,
        "accepted": accepted,
        "added_basis_ids": added,
        "removed_basis_ids": removed,
        "acceptance_reason": reason,
        **summary,
    }


def _plot_ranking(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.8), dpi=160)
    ranks = frame["rank"].to_numpy()
    values = frame["Train_noDPD_NMSE_dB"].to_numpy(dtype=float)
    ax.plot(ranks, values, color="#1f77b4", linewidth=1.0)
    hard = frame["selected_Hard20"].to_numpy(dtype=bool)
    ax.scatter(ranks[hard], values[hard], color="#d62728", s=22, label="Hard-20")
    ax.axvline(HARD_STATE_COUNT + 0.5, color="#d62728", linestyle="--", linewidth=0.9)
    ax.set_xlabel("Train no-DPD NMSE rank")
    ax.set_ylabel("NMSE (dB)")
    ax.set_title("Scenario 2 no-DPD Train ranking")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_capacity(
    capacity: pd.DataFrame,
    hard20: pd.DataFrame,
    path: Path,
) -> None:
    merged = capacity.merge(hard20[["state_id", "hard20_rank"]], on="state_id", how="left")
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=160)
    for support_id, color in (("Full48", "#7f7f7f"), ("Full75", "#1f77b4")):
        selected = merged.loc[merged["support_id"] == support_id].sort_values("hard20_rank")
        ax.plot(
            selected["hard20_rank"],
            selected["train_nmse_db"],
            marker="o",
            markersize=3.5,
            linewidth=1.2,
            color=color,
            label=support_id,
        )
    ax.axhline(TARGET_NMSE_DB, color="#d62728", linestyle="--", linewidth=1.0, label="-40 dB")
    ax.set_xlabel("Hard-20 rank")
    ax.set_ylabel("Full OLS Train NMSE (dB)")
    ax.set_title("Full48 vs Full75 Train capacity")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_search(history: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=160)
    if not history.empty:
        accepted = history.loc[history["accepted"]].copy()
        if not accepted.empty:
            accepted = accepted.reset_index(drop=True)
            ax.plot(
                np.arange(1, len(accepted) + 1),
                accepted["pass_count_W40"],
                marker="o",
                linewidth=1.5,
                color="#2ca02c",
                label="accepted support W-pass count",
            )
            ax.axhline(HARD_STATE_COUNT, color="#d62728", linestyle="--", linewidth=0.9)
            ax.set_xlabel("Accepted sparse-search step")
            ax.set_ylabel("Hard-20 W<-40 dB count")
            ax.legend(frameon=False)
        else:
            ax.text(0.5, 0.5, "No accepted sparse-search step", ha="center", va="center")
            ax.set_axis_off()
    ax.set_title("Envelope75 sparse-search progress")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_final(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=160)
    ordered = frame.sort_values("hard20_rank")
    x = ordered["hard20_rank"].to_numpy()
    ax.plot(x, ordered["train_nmse_db"], marker="o", markersize=4, label="Train", linewidth=1.2)
    ax.plot(x, ordered["test_nmse_db"], marker="s", markersize=4, label="Test", linewidth=1.2)
    ax.axhline(TARGET_NMSE_DB, color="#d62728", linestyle="--", linewidth=1.0, label="-40 dB")
    ax.set_xlabel("Hard-20 rank")
    ax.set_ylabel("Final model NMSE (dB)")
    ax.set_title("Frozen Envelope75-support Hard-20 Train/Test")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _build_summary_text(payload: dict[str, object]) -> str:
    capacity = payload["capacity_summaries"]
    full75 = capacity["Full75"]
    full48 = capacity["Full48"]
    final = payload.get("final_metrics")
    selected = payload.get("selected_summary")
    final_lines = []
    if isinstance(final, dict):
        final_lines = [
            f"Final Train pass: {final['train_pass_count']}/{HARD_STATE_COUNT}",
            f"Final Test pass: {final['test_pass_count']}/{HARD_STATE_COUNT}",
            f"Train failures: {final['train_failures']}",
            f"Test failures: {final['test_failures']}",
            f"Final failure classification: {final['failure_classification']}",
        ]
    else:
        final_lines = [
            "Final Train/Test evaluation: not run because Full75 capacity gate failed.",
            "Final failure classification: candidate-space capacity failure.",
        ]
    selected_lines = []
    if isinstance(selected, dict):
        selected_lines = [
            f"Selected support: {selected['support_id']}",
            f"Selected K: {selected['K']}",
            f"Selected lambda: {selected['ridge_lambda']:.12g}",
            f"Selected Train/CV W pass: {selected['pass_count_W40']}/{HARD_STATE_COUNT}",
            f"Selected W Q95: {selected['W_Q95_dB']:.6f} dB",
            f"Selected W worst: {selected['W_worst_dB']:.6f} dB",
        ]
    else:
        selected_lines = [
            "Selected support: none; sparse search was not permitted after capacity failure.",
            "Selected K: n/a",
            "Selected lambda: n/a",
        ]
    lines = [
        f"Task: {TASK_NAME}",
        "Mode: new independent research task.",
        "Scope: Scenario 2 / 425 states / xin -> yout_withoutdpd_ori / Fs=100 MHz / B=20 MHz / 5B.",
        f"Record: N={FULL_LENGTH}; Train=[0,{TRAIN_END}); Test=[{TRAIN_END},{TEST_END}); dmax=2.",
        "Preprocessing: full-record rough/fine alignment once; independent "
        "Train/Test complex gains.",
        f"Hard-20 state IDs in Train ranking order: {payload['hard20_state_ids']}",
        "Candidate family: 3 linear terms plus p=2..9, m/q=0..2 (Envelope75).",
        f"Dictionary hash: {payload['dictionary_hash']}",
        f"Full48 capacity: Train pass {full48['N_train40']}/{HARD_STATE_COUNT}, "
        f"median={full48['train_median_dB']:.6f} dB, "
        f"worst={full48['train_worst_dB']:.6f} dB.",
        f"Full75 capacity: Train pass {full75['N_train40']}/{HARD_STATE_COUNT}, "
        f"median={full75['train_median_dB']:.6f} dB, "
        f"worst={full75['train_worst_dB']:.6f} dB.",
        f"Full75 capacity status: {payload['capacity_status']}",
        *selected_lines,
        *final_lines,
        f"Selected p=4 count: {payload['selected_even_order_counts'].get('p4', 0)}",
        f"Selected p=6 count: {payload['selected_even_order_counts'].get('p6', 0)}",
        f"Selected p=8 count: {payload['selected_even_order_counts'].get('p8', 0)}",
        f"Any new even-order term selected: {payload['selected_even_order_present']}",
        f"dmax=3 next step: {payload['dmax3_recommendation']}",
        "Test policy: Test metrics were unlocked only after support and lambda freeze; "
        "no Test-based retuning was performed.",
        "Workers: 10 spawned CPU worker processes, one BLAS thread per worker; "
        "GPU unused; no worker benchmark.",
        "Excluded: Aend/C2, common-B, LUT retrieval, Type-III clustering, DPD replay, "
        "low-bandwidth, and automatic dmax=3.",
        "Raw snapshot before: "
        f"{json.dumps(payload['raw_before'], ensure_ascii=False, sort_keys=True)}",
        "Raw snapshot after: "
        f"{json.dumps(payload['raw_after'], ensure_ascii=False, sort_keys=True)}",
        f"raw_data_modified: {payload['raw_data_modified']}",
    ]
    return "\n".join(lines) + "\n"


def _capacity_failure_outputs(
    terms: tuple[EnvelopeBasis, ...],
    capacity: pd.DataFrame,
    capacity_summaries: dict[str, dict[str, object]],
    raw_before: dict[str, object],
    raw_after: dict[str, object],
    hard20_ids: list[int],
    dictionary_hash: str,
) -> None:
    pd.DataFrame(
        [
            {
                "phase": "capacity_failure",
                "reason": "Full75 OLS Train pass count below 20/20",
                "support_id": "Full75",
                "pass_count_W40": capacity_summaries["Full75"]["N_train40"],
            }
        ]
    ).to_csv(RESULT_ROOT / "06_search_history.csv", index=False)
    pd.DataFrame(columns=["support_id", "K", "pass_count_W40"]).to_csv(
        RESULT_ROOT / "07_pareto_supports.csv", index=False
    )
    pd.DataFrame(columns=["support_id", "ridge_lambda", "pass_count_W40"]).to_csv(
        RESULT_ROOT / "08_ridge_scan.csv", index=False
    )
    pd.DataFrame(
        columns=["basis_position", "basis_id", "formula", "family", "p", "m", "q", "mandatory"]
    ).to_csv(RESULT_ROOT / "09_final_model_definition.csv", index=False)
    pd.DataFrame(columns=["state_id", "hard20_rank", "train_nmse_db", "test_nmse_db"]).to_csv(
        RESULT_ROOT / "10_final_hard20_train_test_metrics.csv", index=False
    )
    raw_modified = raw_before != raw_after
    payload = {
        "hard20_state_ids": hard20_ids,
        "dictionary_hash": dictionary_hash,
        "capacity_summaries": capacity_summaries,
        "capacity_status": "Envelope75 candidate-space capacity failure",
        "selected_even_order_counts": {"p4": 0, "p6": 0, "p8": 0},
        "selected_even_order_present": False,
        "dmax3_recommendation": "Investigate in a separate next task; this task stops at dmax=2.",
        "raw_before": raw_before,
        "raw_after": raw_after,
        "raw_data_modified": raw_modified,
    }
    (RESULT_ROOT / "11_final_result_summary.txt").write_text(
        _build_summary_text(payload), encoding="utf-8"
    )
    checkpoint(
        CHECKPOINT_PATH,
        phase="capacity_failure",
        hard20_state_ids=hard20_ids,
        dictionary_hash=dictionary_hash,
        current_support=None,
        completed_round=0,
        lambda_scan_progress=0,
        model_frozen=False,
        test_unlocked=False,
        raw_data_modified=raw_modified,
    )


def _run(*, resume: bool = False) -> dict[str, object]:
    existing_output = any(RESULT_ROOT.iterdir())
    if existing_output and not resume:
        raise RuntimeError(
            "Result directory is not empty; refusing to overwrite existing task output: "
            f"{RESULT_ROOT}"
        )
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    WORK_LOG.parent.mkdir(parents=True, exist_ok=True)
    _write_task_definition(RESULT_ROOT / "00_task_definition.txt")
    if existing_output:
        _append_log(
            WORK_LOG,
            f"[{_now()}] Resume same incomplete {TASK_NAME}; preserving prior artifacts.\n",
        )
    raw_before = _raw_manifest()
    expected_raw = {
        "sha256": EXPECTED_RAW_MANIFEST,
        "file_count": EXPECTED_RAW_FILE_COUNT,
        "mat_count": EXPECTED_RAW_MAT_COUNT,
        "bytes": EXPECTED_RAW_BYTES,
    }
    if raw_before != expected_raw:
        raise RuntimeError(f"data/raw manifest differs from frozen baseline: {raw_before}")
    _append_log(
        WORK_LOG,
        "\n"
        f"[{_now()}] Start {TASK_NAME}\n"
        "Mode: new independent Hard-20 forward-behavior basis-selection task.\n"
        f"Raw snapshot before: {json.dumps(raw_before, ensure_ascii=False, sort_keys=True)}\n"
        "Workers: 10 spawn processes, BLAS threads per worker=1, no benchmark, no GPU.\n",
    )

    terms = tuple(build_envelope_dictionary())
    gate = dictionary_gate()
    dictionary_hash = _dictionary_hash(terms)
    # The exact Gate-B ID set is reconstructed from the validated frozen gate,
    # not inferred from order prefixes.  This also checks that the count remains 48.
    from behavior_modeling.shared.basis_function_selection.frozen_centered_dictionary import (  # noqa: PLC0415
        build_frozen_centered_dictionary,
        gate_indices,
    )

    frozen_terms = build_frozen_centered_dictionary()
    gate_b_ids = {frozen_terms[index].basis_id for index in gate_indices(frozen_terms, "GATE_B")}
    baseline_ids = set(BASELINE_SUPPORT_IDS)
    dictionary_frame = _dictionary_frame(terms, gate_b_ids, baseline_ids)
    dictionary_frame.to_csv(RESULT_ROOT / "03_envelope75_dictionary.csv", index=False)

    checkpoint(
        CHECKPOINT_PATH,
        phase="dictionary_complete",
        hard20_state_ids=None,
        dictionary_hash=dictionary_hash,
        dictionary_gate=gate,
        current_support=None,
        completed_round=0,
        lambda_scan_progress=0,
        model_frozen=False,
        test_unlocked=False,
        raw_data_modified=False,
    )

    ranking = run_ranking()
    ranking.to_csv(RESULT_ROOT / "01_all425_nodpd_train_nmse_ranking.csv", index=False)
    _plot_ranking(ranking, RESULT_ROOT / "13_all425_nodpd_nmse_ranking.png")
    hard20 = ranking.loc[ranking["selected_Hard20"]].copy().reset_index(drop=True)
    hard20.insert(1, "hard20_rank", np.arange(1, HARD_STATE_COUNT + 1, dtype=np.int64))
    hard20.to_csv(RESULT_ROOT / "02_hard20_states.csv", index=False)
    hard20_ids = [int(value) for value in hard20["state_id"]]
    checkpoint(
        CHECKPOINT_PATH,
        phase="hard20_frozen",
        hard20_state_ids=hard20_ids,
        dictionary_hash=dictionary_hash,
        current_support=None,
        completed_round=0,
        lambda_scan_progress=0,
        model_frozen=False,
        test_unlocked=False,
        raw_data_modified=False,
    )

    full75_support = tuple(range(len(terms)))
    full48_support = tuple(term.index for term in terms if term.basis_id in gate_b_ids)
    if len(full48_support) != 48 or len(full75_support) != 75:
        raise RuntimeError("Full48/Full75 support construction failed")
    if not set(full48_support) < set(full75_support):
        raise RuntimeError("Gate-B48 is not a strict Envelope75 subset")
    baseline = baseline_support(terms)
    if len(baseline) != 17:
        raise RuntimeError("Baseline17 count is not 17")

    temp_root = Path(tempfile.mkdtemp(prefix=f"{TASK_NAME}_"))
    try:
        cache = create_train_cache(temp_root / "train_cache", hard20_ids)
        cache_metadata = populate_train_cache(cache)
        metadata_by_state = {int(row["state_id"]): row for row in cache_metadata}
        for _, row in hard20.iterrows():
            cached = metadata_by_state[int(row["state_id"])]
            for name in (
                "rough_delay",
                "fine_delay",
                "Train_gain_real",
                "Train_gain_imag",
                "Test_gain_real",
                "Test_gain_imag",
            ):
                if not np.isclose(float(row[name]), float(cached[name]), rtol=0.0, atol=1e-12):
                    raise RuntimeError(
                        f"Hard-20 cache metadata mismatch at state {row['state_id']}: {name}"
                    )
        checkpoint(
            CHECKPOINT_PATH,
            phase="train_cache_complete",
            hard20_state_ids=hard20_ids,
            dictionary_hash=dictionary_hash,
            current_support=None,
            completed_round=0,
            lambda_scan_progress=0,
            model_frozen=False,
            test_unlocked=False,
            raw_data_modified=False,
        )

        with TrainSelectionEvaluator(cache, terms) as evaluator:
            capacity = evaluator.evaluate_capacity(
                (("Full48", full48_support), ("Full75", full75_support)),
                progress_label="capacity",
            )
            capacity["train_pass40"] = capacity["train_nmse_db"] < TARGET_NMSE_DB
            capacity.to_csv(RESULT_ROOT / "05_full75_capacity_gate.csv", index=False)
            capacity_summaries = {
                support_id: _summary_from_capacity(capacity, support_id)
                for support_id in ("Full48", "Full75")
            }
            _plot_capacity(capacity, hard20, RESULT_ROOT / "14_full75_capacity.png")
            for summary in capacity_summaries.values():
                if not bool(summary["numerically_reliable"]):
                    raise RuntimeError(f"Capacity numerical hard gate failed: {summary}")
            full75_capacity_success = capacity_summaries["Full75"]["N_train40"] == HARD_STATE_COUNT
            if not full75_capacity_success:
                raw_after = _raw_manifest()
                _capacity_failure_outputs(
                    terms,
                    capacity,
                    capacity_summaries,
                    raw_before,
                    raw_after,
                    hard20_ids,
                    dictionary_hash,
                )
                return {
                    "capacity_status": "Envelope75 candidate-space capacity failure",
                    "final_test_pass": None,
                    "raw_data_modified": raw_before != raw_after,
                }

            support_payloads: dict[str, tuple[int, ...]] = {"Baseline17": baseline}
            support_summaries: dict[str, dict[str, object]] = {}
            support_frames: dict[str, pd.DataFrame] = {}
            history_rows: list[dict[str, object]] = []

            def evaluate_specs(
                specs: list[tuple[str, tuple[int, ...]]],
                ridge_lambda: float,
                label: str,
                *,
                exact_condition: bool = False,
            ) -> None:
                summaries, frames = evaluator.evaluate(
                    specs,
                    ridge_lambda,
                    exact_condition=exact_condition,
                    progress_label=label,
                )
                for support_id, summary in summaries.items():
                    support_summaries[support_id] = summary
                    support_frames[support_id] = frames[support_id]
                    support_payloads[support_id] = tuple(
                        int(index)
                        for index in specs[[item[0] for item in specs].index(support_id)][1]
                    )

            evaluate_specs([("Baseline17", baseline)], 0.0, "baseline-ols")
            baseline_ols_summary = support_summaries["Baseline17"]
            evaluate_specs([("Baseline17_Ridge_1e-8", baseline)], 1e-8, "baseline-ridge")
            baseline_reference = pd.concat(
                [
                    support_frames["Baseline17"].assign(model_variant="Baseline17_OLS"),
                    support_frames["Baseline17_Ridge_1e-8"].assign(
                        model_variant="Baseline17_Ridge_1e-8"
                    ),
                ],
                ignore_index=True,
            )
            baseline_reference.to_csv(RESULT_ROOT / "04_baseline17_reference.csv", index=False)
            history_rows.append(
                _history_row(
                    "baseline",
                    0,
                    baseline_ols_summary,
                    accepted=True,
                    reason="fixed Baseline17 anchor",
                )
            )

            current_support = baseline
            current_id = "Baseline17"
            current_summary = baseline_ols_summary
            accepted_ids = [current_id]
            remaining = {
                term.index for term in terms if term.order > 1 and term.index not in set(baseline)
            }
            last_single_summaries: list[tuple[int, str, dict[str, object]]] = []
            best_forward_id = current_id
            best_pruned_id = current_id
            best_pair_id: str | None = None
            best_swap_id: str | None = None
            forward_round = 0
            forward_plateau = False

            while remaining and len(current_support) < K_SAFETY_CAP:
                forward_round += 1
                candidates: list[tuple[str, tuple[int, ...], int]] = []
                for index in sorted(remaining):
                    if len(current_support) + 1 > K_SAFETY_CAP:
                        break
                    candidate_support = tuple(sorted((*current_support, index)))
                    label = f"Forward_r{forward_round}_{terms[index].basis_id}"
                    candidates.append((label, candidate_support, index))
                if not candidates:
                    break
                specs = [(label, support) for label, support, _ in candidates]
                evaluate_specs(specs, 0.0, f"forward-{forward_round}")
                candidate_rows = [
                    (index, label, support_summaries[label]) for label, _, index in candidates
                ]
                last_single_summaries = candidate_rows
                ordered = sorted(candidate_rows, key=lambda item: support_sort_key(item[2]))
                best_index, best_id, best_summary = ordered[0]
                for index, label, summary in candidate_rows:
                    accepted = label == best_id and support_improves(current_summary, summary)
                    history_rows.append(
                        _history_row(
                            "forward",
                            forward_round,
                            summary,
                            accepted=accepted,
                            added=terms[index].basis_id,
                            reason="best candidate accepted" if accepted else "candidate rejected",
                        )
                    )
                if not support_improves(current_summary, best_summary):
                    forward_plateau = True
                    break
                current_support = tuple(sorted((*current_support, best_index)))
                current_id = best_id
                current_summary = best_summary
                accepted_ids.append(current_id)
                best_forward_id = current_id
                remaining.remove(best_index)

                # Backward cleanup after every accepted forward step.
                while True:
                    removable = [index for index in current_support if terms[index].order > 1]
                    deletion_specs = []
                    for removed in removable:
                        candidate_support = tuple(
                            index for index in current_support if index != removed
                        )
                        label = f"Backward_after_{current_id}_{terms[removed].basis_id}"
                        deletion_specs.append((label, candidate_support, removed))
                    if not deletion_specs:
                        break
                    evaluate_specs(
                        [(label, support) for label, support, _ in deletion_specs],
                        0.0,
                        f"backward-{forward_round}",
                    )
                    allowed = [
                        item
                        for item in deletion_specs
                        if deletion_allowed(current_summary, support_summaries[item[0]])
                    ]
                    if not allowed:
                        break
                    removed_label, removed_support, removed_index = min(
                        allowed,
                        key=lambda item: support_sort_key(support_summaries[item[0]]),
                    )
                    removed_summary = support_summaries[removed_label]
                    history_rows.append(
                        _history_row(
                            "backward",
                            forward_round,
                            removed_summary,
                            accepted=True,
                            removed=terms[removed_index].basis_id,
                            reason="deletion preserved W-pass and stayed within 0.02 dB tolerance",
                        )
                    )
                    current_support = tuple(removed_support)
                    current_id = removed_label
                    current_summary = removed_summary
                    accepted_ids.append(current_id)
                    best_pruned_id = current_id
                    current_set = set(current_support)
                    remaining = {
                        term.index
                        for term in terms
                        if term.order > 1 and term.index not in current_set
                    }

            # Pair rescue is intentionally bounded to the best single-add pool.
            if (forward_plateau or not remaining) and current_summary[
                "pass_count_W40"
            ] < HARD_STATE_COUNT:
                top_single = sorted(
                    last_single_summaries, key=lambda item: support_sort_key(item[2])
                )[:12]
                pair_specs: list[tuple[str, tuple[int, ...], tuple[int, int]]] = []
                for left_position, left in enumerate(top_single):
                    for right in top_single[left_position + 1 :]:
                        left_index, _, _ = left
                        right_index, _, _ = right
                        if left_index == right_index:
                            continue
                        if left_index in current_support or right_index in current_support:
                            continue
                        pair_support = tuple(sorted((*current_support, left_index, right_index)))
                        if len(pair_support) > K_SAFETY_CAP:
                            continue
                        label = (
                            f"Pair_r{forward_round}_{terms[left_index].basis_id}_"
                            f"{terms[right_index].basis_id}"
                        )
                        pair_specs.append((label, pair_support, (left_index, right_index)))
                if pair_specs:
                    evaluate_specs(
                        [(label, support) for label, support, _ in pair_specs],
                        0.0,
                        "pair-rescue",
                    )
                    pair_order = sorted(
                        pair_specs,
                        key=lambda item: support_sort_key(support_summaries[item[0]]),
                    )
                    pair_label, pair_support, pair_indices = pair_order[0]
                    pair_summary = support_summaries[pair_label]
                    if support_improves(current_summary, pair_summary):
                        history_rows.append(
                            _history_row(
                                "pair_rescue",
                                forward_round,
                                pair_summary,
                                accepted=True,
                                added=";".join(terms[index].basis_id for index in pair_indices),
                                reason="bounded top-12 pair rescue accepted",
                            )
                        )
                        current_support = tuple(pair_support)
                        current_id = pair_label
                        current_summary = pair_summary
                        accepted_ids.append(current_id)
                        best_pair_id = current_id
                    else:
                        history_rows.append(
                            _history_row(
                                "pair_rescue",
                                forward_round,
                                pair_summary,
                                accepted=False,
                                added=";".join(terms[index].basis_id for index in pair_indices),
                                reason="best bounded pair did not improve current support",
                            )
                        )

            # One bounded local swap stage follows Forward/Backward/Pair.
            current_nonlinear = [index for index in current_support if terms[index].order > 1]
            add_pool = [
                item[0]
                for item in sorted(
                    last_single_summaries, key=lambda item: support_sort_key(item[2])
                )[:12]
            ]
            add_indices = []
            for index, _, _ in last_single_summaries:
                if index not in current_support and index not in add_indices:
                    add_indices.append(index)
            if add_pool:
                add_indices = [
                    index
                    for label in add_pool
                    for index, candidate_label, _ in last_single_summaries
                    if candidate_label == label
                ]
            swap_specs: list[tuple[str, tuple[int, ...], tuple[int, int]]] = []
            for removed in current_nonlinear:
                for added in add_indices:
                    if added in current_support or added == removed:
                        continue
                    swap_support = tuple(sorted((set(current_support) - {removed}) | {added}))
                    label = f"Swap_{terms[removed].basis_id}_to_{terms[added].basis_id}"
                    swap_specs.append((label, swap_support, (removed, added)))
            if swap_specs:
                evaluate_specs(
                    [(label, support) for label, support, _ in swap_specs],
                    0.0,
                    "local-swap",
                )
                swap_order = sorted(
                    swap_specs,
                    key=lambda item: support_sort_key(support_summaries[item[0]]),
                )
                swap_label, swap_support, swap_indices = swap_order[0]
                swap_summary = support_summaries[swap_label]
                if support_improves(current_summary, swap_summary):
                    history_rows.append(
                        _history_row(
                            "local_swap",
                            forward_round,
                            swap_summary,
                            accepted=True,
                            added=terms[swap_indices[1]].basis_id,
                            removed=terms[swap_indices[0]].basis_id,
                            reason="bounded local swap accepted",
                        )
                    )
                    current_support = tuple(swap_support)
                    current_id = swap_label
                    current_summary = swap_summary
                    accepted_ids.append(current_id)
                    best_swap_id = current_id
                else:
                    history_rows.append(
                        _history_row(
                            "local_swap",
                            forward_round,
                            swap_summary,
                            accepted=False,
                            added=terms[swap_indices[1]].basis_id,
                            removed=terms[swap_indices[0]].basis_id,
                            reason="best bounded local swap did not improve current support",
                        )
                    )

            milestone_ids = []
            for label in (
                "Baseline17",
                best_forward_id,
                best_pruned_id,
                best_pair_id,
                best_swap_id,
                current_id,
            ):
                if label is not None and label in support_summaries and label not in milestone_ids:
                    milestone_ids.append(label)
            milestone_summaries = [support_summaries[label] for label in milestone_ids]
            pareto_summaries = pareto_filter(milestone_summaries)
            if not any(item["support_id"] == "Baseline17" for item in pareto_summaries):
                pareto_summaries.append(support_summaries["Baseline17"])
            pareto_summaries = sorted(pareto_summaries, key=support_sort_key)[:6]
            pareto_ids = [str(item["support_id"]) for item in pareto_summaries]
            pareto_frame = pd.DataFrame(pareto_summaries)
            pareto_frame.to_csv(RESULT_ROOT / "07_pareto_supports.csv", index=False)

            ridge_rows: list[dict[str, object]] = []
            ridge_frame_by_key: dict[tuple[str, float], pd.DataFrame] = {}
            ridge_lambdas = list(RIDGE_GRID)
            for ridge_lambda in ridge_lambdas:
                specs = [
                    (
                        f"{support_id}@lambda={ridge_lambda:.12g}",
                        support_payloads[support_id],
                    )
                    for support_id in pareto_ids
                ]
                evaluate_specs(specs, ridge_lambda, f"ridge-{ridge_lambda:.0e}")
                for support_id in pareto_ids:
                    eval_id = f"{support_id}@lambda={ridge_lambda:.12g}"
                    row = dict(support_summaries[eval_id])
                    row["basis_support_id"] = support_id
                    ridge_rows.append(row)
                    ridge_frame_by_key[(support_id, float(ridge_lambda))] = support_frames[eval_id]
            prelim = sorted(
                ridge_rows,
                key=lambda row: (
                    -int(row["pass_count_W40"]),
                    float(row["W_Q95_deficit_dB"]),
                    float(row["W_worst_dB"]),
                    int(row["K"]),
                    float(row["ridge_lambda"]),
                    str(row["basis_support_id"]),
                ),
            )[0]
            boundary_lambdas: list[float] = []
            if float(prelim["ridge_lambda"]) == RIDGE_GRID[-1]:
                boundary_lambdas.extend([RIDGE_BOUNDARY_HIGH, RIDGE_EXTRA_HIGH])
            if float(prelim["ridge_lambda"]) == 1e-10:
                boundary_lambdas.extend([RIDGE_BOUNDARY_LOW])
            for ridge_lambda in dict.fromkeys(boundary_lambdas):
                specs = [
                    (
                        f"{support_id}@lambda={ridge_lambda:.12g}",
                        support_payloads[support_id],
                    )
                    for support_id in pareto_ids
                ]
                evaluate_specs(specs, ridge_lambda, f"ridge-extra-{ridge_lambda:.0e}")
                for support_id in pareto_ids:
                    eval_id = f"{support_id}@lambda={ridge_lambda:.12g}"
                    row = dict(support_summaries[eval_id])
                    row["basis_support_id"] = support_id
                    ridge_rows.append(row)
                    ridge_frame_by_key[(support_id, float(ridge_lambda))] = support_frames[eval_id]
            ridge_frame = pd.DataFrame(ridge_rows).drop_duplicates(
                subset=["basis_support_id", "ridge_lambda"], keep="last"
            )
            ridge_frame = ridge_frame.sort_values(
                [
                    "pass_count_W40",
                    "W_Q95_deficit_dB",
                    "W_worst_dB",
                    "K",
                    "ridge_lambda",
                    "basis_support_id",
                ],
                ascending=[False, True, True, True, True, True],
            ).reset_index(drop=True)
            ridge_frame.insert(0, "rank", np.arange(1, len(ridge_frame) + 1))
            ridge_frame.to_csv(RESULT_ROOT / "08_ridge_scan.csv", index=False)

            selected_row = ridge_frame.iloc[0]
            selected_basis_support_id = str(selected_row["basis_support_id"])
            selected_lambda = float(selected_row["ridge_lambda"])
            selected_support = support_payloads[selected_basis_support_id]
            selected_eval_id = f"{selected_basis_support_id}@lambda={selected_lambda:.12g}"
            selected_summary = support_summaries[selected_eval_id]
            selected_ids = support_ids(terms, selected_support)
            even_counts = {
                f"p{order}": sum(terms[index].order == order for index in selected_support)
                for order in (4, 6, 8)
            }
            final_definition = pd.DataFrame(
                [
                    {
                        "basis_position": position,
                        "basis_id": term.basis_id,
                        "family": term.family,
                        "p": term.order,
                        "m": term.signal_delay,
                        "q": term.envelope_delay,
                        "formula": term.formula,
                        "mandatory": term.mandatory,
                        "global_index": term.index,
                        "K": len(selected_support),
                        "lambda": selected_lambda,
                        "dmax": 2,
                        "candidate_family": "Envelope75",
                    }
                    for position, term in enumerate(
                        (terms[index] for index in selected_support), start=1
                    )
                ]
            )
            final_definition.to_csv(RESULT_ROOT / "09_final_model_definition.csv", index=False)
            checkpoint(
                CHECKPOINT_PATH,
                phase="model_frozen",
                hard20_state_ids=hard20_ids,
                dictionary_hash=dictionary_hash,
                current_support=list(selected_ids),
                completed_round=forward_round,
                lambda_scan_progress=len(ridge_rows),
                final_lambda=selected_lambda,
                model_frozen=True,
                test_unlocked=True,
                raw_data_modified=False,
            )

            # This is the first and only Test score call.
            final_train_summary, final_train_frames = evaluator.evaluate(
                [("FinalFrozen", selected_support)],
                selected_lambda,
                exact_condition=True,
                progress_label="final-train-exact",
            )
            final_train_frame = final_train_frames["FinalFrozen"].copy()
            final_train_frame = final_train_frame.rename(
                columns={
                    "train_nmse_db": "selection_train_nmse_db",
                    "W_db": "selection_W_db",
                }
            )
            final_test_frame = run_final_evaluation(
                cache,
                terms,
                selected_support,
                selected_lambda,
            )
            hard20_meta = hard20[
                ["state_id", "hard20_rank", "funMng", "funAng", "secMng", "secAng", "Vm", "Pin"]
            ]
            final_metrics = hard20_meta.merge(final_test_frame, on="state_id", how="inner")
            final_metrics = final_metrics.merge(
                final_train_frame[
                    [
                        "state_id",
                        "selection_train_nmse_db",
                        "cv1_nmse_db",
                        "cv2_nmse_db",
                        "cv3_nmse_db",
                        "selection_W_db",
                        "min_rank_ratio",
                    ]
                ],
                on="state_id",
                how="inner",
            )
            final_metrics["support_id"] = selected_basis_support_id
            final_metrics["K"] = len(selected_support)
            final_metrics["ridge_lambda"] = selected_lambda
            final_metrics["train_pass40"] = final_metrics["train_nmse_db"] < TARGET_NMSE_DB
            final_metrics["test_pass40"] = final_metrics["test_nmse_db"] < TARGET_NMSE_DB
            final_metrics.to_csv(
                RESULT_ROOT / "10_final_hard20_train_test_metrics.csv", index=False
            )
            _plot_ranking(ranking, RESULT_ROOT / "13_all425_nodpd_nmse_ranking.png")
            history_frame = pd.DataFrame(history_rows)
            history_frame.to_csv(RESULT_ROOT / "06_search_history.csv", index=False)
            _plot_search(history_frame, RESULT_ROOT / "15_sparse_search_progress.png")
            _plot_final(final_metrics, RESULT_ROOT / "16_final_hard20_train_test.png")

            train_pass = int(final_metrics["train_pass40"].sum())
            test_pass = int(final_metrics["test_pass40"].sum())
            failure_classification = (
                "Envelope75 goal achieved"
                if train_pass == HARD_STATE_COUNT and test_pass == HARD_STATE_COUNT
                else "Training capacity solved; cross-segment generalization remains"
            )
            raw_after = _raw_manifest()
            raw_modified = raw_before != raw_after
            final_summary_payload = {
                "hard20_state_ids": hard20_ids,
                "dictionary_hash": dictionary_hash,
                "capacity_summaries": capacity_summaries,
                "capacity_status": "Envelope75 Train capacity solved",
                "selected_summary": selected_summary,
                "final_metrics": {
                    "train_pass_count": train_pass,
                    "test_pass_count": test_pass,
                    "train_failures": final_metrics.loc[
                        ~final_metrics["train_pass40"], "state_id"
                    ].tolist(),
                    "test_failures": final_metrics.loc[
                        ~final_metrics["test_pass40"], "state_id"
                    ].tolist(),
                    "failure_classification": failure_classification,
                },
                "selected_even_order_counts": even_counts,
                "selected_even_order_present": bool(sum(even_counts.values())),
                "dmax3_recommendation": (
                    "Not automatically; consider only as a separate task if the remaining "
                    "generalization gap requires it."
                    if test_pass == HARD_STATE_COUNT
                    else "May be investigated in a separate next task; this task stops at dmax=2."
                ),
                "raw_before": raw_before,
                "raw_after": raw_after,
                "raw_data_modified": raw_modified,
            }
            (RESULT_ROOT / "11_final_result_summary.txt").write_text(
                _build_summary_text(final_summary_payload), encoding="utf-8"
            )
            history_frame.to_csv(RESULT_ROOT / "06_search_history.csv", index=False)
            checkpoint(
                CHECKPOINT_PATH,
                phase="completed",
                hard20_state_ids=hard20_ids,
                dictionary_hash=dictionary_hash,
                current_support=list(selected_ids),
                completed_round=forward_round,
                lambda_scan_progress=len(ridge_rows),
                final_lambda=selected_lambda,
                model_frozen=True,
                test_unlocked=True,
                raw_data_modified=raw_modified,
            )
            return {
                "capacity_status": "Envelope75 Train capacity solved",
                "final_test_pass": test_pass,
                "raw_data_modified": raw_modified,
                "selected_support": selected_ids,
                "selected_lambda": selected_lambda,
            }
    finally:
        # The cache contains large reproducible design banks and is deliberately
        # outside results/.  It is removed after all result artifacts are written.
        import shutil

        shutil.rmtree(temp_root, ignore_errors=True)


def main() -> None:
    try:
        result = _run(resume="--resume" in sys.argv[1:])
    except Exception as exc:
        _append_log(
            WORK_LOG,
            f"[{_now()}] FAILED {TASK_NAME}: {type(exc).__name__}: {exc}\n",
        )
        raise
    _append_log(
        WORK_LOG,
        f"[{_now()}] Completed {TASK_NAME}\n"
        f"Result: {json.dumps(result, ensure_ascii=False, sort_keys=True)}\n"
        "No Aend/C2, LUT retrieval, clustering, DPD, low-bandwidth, or automatic dmax=3 was run.\n",
    )
    _append_log(
        HANDOFF_LOG,
        f"\n{_now()} | 新任务：{TASK_NAME}\n"
        f"完成 Envelope75 Hard-20 正向基函数选择主线；结果目录：{RESULT_ROOT}\n"
        f"结果摘要：{json.dumps(result, ensure_ascii=False, sort_keys=True)}\n"
        "严格使用 10 个 spawn worker、每 worker 1 个 BLAS 线程；"
        "Test 仅在 support/lambda 冻结后解锁；"
        "未运行 Aend/C2、LUT retrieval、clustering、DPD、low-bandwidth 或自动 dmax=3。\n",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    __import__("multiprocessing").freeze_support()
    main()
