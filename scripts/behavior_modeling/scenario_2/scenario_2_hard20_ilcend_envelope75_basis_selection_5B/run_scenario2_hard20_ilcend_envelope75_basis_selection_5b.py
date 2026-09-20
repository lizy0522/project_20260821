"""Run independent Hard-20 ilc_end Envelope75 basis selection."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import tempfile
import time
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

from behavior_modeling.shared.basis_function_selection.all425_envelope18_evaluation import (  # noqa: E402
    FINAL_SUPPORT_IDS,
    verify_frozen_support,
)
from behavior_modeling.shared.basis_function_selection.envelope_dictionary import (
    DMAX,
    EnvelopeBasis,  # noqa: E402
    build_envelope_dictionary,  # noqa: E402
)
from behavior_modeling.shared.basis_function_selection.hard20_envelope75_selection import (  # noqa: E402
    HARD_STATE_COUNT,
    K_SAFETY_CAP,
    RIDGE_GRID,
    TARGET_NMSE_DB,
    TrainSelectionEvaluator,
    _raw_manifest,
    checkpoint,
    create_train_cache,
    deletion_allowed,
    pareto_filter,
    run_final_evaluation,
    run_ranking,
    support_ids,
    support_improves,
    support_sort_key,
)
from behavior_modeling.shared.basis_function_selection.ilcend_behavior_dataset import (  # noqa: E402
    populate_ilcend_cache,
    run_ilcend_preflight,
)

TASK_NAME = "scenario_2_hard20_ilcend_envelope75_basis_selection_5B"
RESULT_ROOT = PROJECT_ROOT / "results" / "behavior_modeling" / "scenario_2" /TASK_NAME
WORK_LOG = PROJECT_ROOT / "work_logs" / "behavior_modeling" / "scenario_2" /TASK_NAME / "execution_log.txt"  # noqa: E501
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"
CHECKPOINT_PATH = RESULT_ROOT / "13_checkpoint.json"
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


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message)


def _support_hash(terms: tuple[EnvelopeBasis, ...], support: tuple[int, ...]) -> str:
    payload = [terms[index].basis_id for index in support]
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def _task_definition(terms: tuple[EnvelopeBasis, ...], candidate_hash: str) -> str:
    return (
        "\n".join(
            [
                f"Task: {TASK_NAME}",
                "Behavior chain: xin_pd_ori_ilc(:, ilc_end) -> yout_withdpd_ori_ilc(:, ilc_end).",
                "ilc_end: last valid ILC column per State; same column for input/output.",
                "Full record: 24576 samples; full-record timing alignment before split.",
                "Train=[0,16384); Test=[16384,24576); independent Train/Test complex gains.",
                "Hard20: selected only by formal NMSE_withoutdpd ranking.",
                "Candidate dictionary: canonical Envelope75; dmax=2.",
                "Search: Envelope18 baseline, Full75 Train capacity, Train-only contiguous "
                "CV, sparse search, Ridge.",
                "Final Test: unlocked only after support/lambda/dmax freeze.",
                f"Candidate dictionary hash: {candidate_hash}",
                f"Candidate count: {len(terms)}",
                "Downstream LUT/Aend/C2/common-B/Type-III/DPD/low-bandwidth tasks are excluded.",
            ]
        )
        + "\n"
    )


def _history_row(
    phase: str,
    iteration: int,
    summary: dict[str, object],
    *,
    accepted: bool,
    operation: str,
    added: str = "",
    removed: str = "",
    reason: str = "",
) -> dict[str, object]:
    return {
        "iteration": iteration,
        "phase": phase,
        "operation": operation,
        "added_terms": added,
        "removed_terms": removed,
        "accepted": accepted,
        "reason": reason,
        **summary,
    }


def _plot_final(frame: pd.DataFrame, path: Path) -> None:
    ordered = frame.sort_values("hard_rank")
    x = ordered["hard_rank"].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(10.0, 5.6), dpi=300)
    ax.plot(x, ordered["Train_NMSE_dB"], marker="o", linewidth=1.2, label="Train NMSE")
    ax.plot(x, ordered["CV_W_NMSE_dB"], marker="s", linewidth=1.2, label="CV-W NMSE")
    ax.plot(x, ordered["Test_NMSE_dB"], marker="^", linewidth=1.2, label="Test NMSE")
    ax.axhline(TARGET_NMSE_DB, color="#d62728", linestyle="--", linewidth=1.0, label="-40 dB")
    ax.set_xticks(x)
    ax.set_xticklabels(ordered["State_ID"].astype(int).tolist(), rotation=60, fontsize=7)
    ax.set_xlabel("Hard20 rank (tick label = State ID)")
    ax.set_ylabel("NMSE (dB)")
    ax.set_title("ilc_end Envelope75 Hard20 Modeling and Generalization")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _summary_text(payload: dict[str, object]) -> str:
    full75_text = json.dumps(payload["full75_capacity"], ensure_ascii=False, sort_keys=True)
    baseline_text = json.dumps(payload["baseline_summary"], ensure_ascii=False, sort_keys=True)
    train_text = json.dumps(payload.get("train_summary"), ensure_ascii=False, sort_keys=True)
    cv_text = json.dumps(payload.get("cv_summary"), ensure_ascii=False, sort_keys=True)
    test_text = json.dumps(payload.get("test_summary"), ensure_ascii=False, sort_keys=True)
    comparison_text = json.dumps(
        payload.get("support_comparison"), ensure_ascii=False, sort_keys=True
    )
    raw_before_text = json.dumps(payload["raw_before"], ensure_ascii=False, sort_keys=True)
    raw_after_text = json.dumps(payload["raw_after"], ensure_ascii=False, sort_keys=True)
    lines = [
        f"Task: {TASK_NAME}",
        "Behavior chain: xin_pd_ori_ilc(:, ilc_end) -> yout_withdpd_ori_ilc(:, ilc_end).",
        "ilc_end: per-State last valid ILC column; not ABC segment A.",
        "Train=[0,16384); Test=[16384,24576); full-record timing alignment first.",
        "Train/Test complex gains are estimated independently; Test is not used for "
        "selection or fitting.",
        f"Hard20 State IDs: {payload['hard20_state_ids']}",
        f"Full75 capacity Train: {payload['full75_capacity']['pass_count']}/{HARD_STATE_COUNT}",
        f"Full75 capacity summary: {full75_text}",
        f"Baseline Envelope18 summary: {baseline_text}",
        f"Final model status: {payload['final_status']}",
        f"Final K: {payload.get('final_K')}",
        f"Final lambda: {payload.get('final_lambda')}",
        f"Final support hash: {payload.get('final_support_hash')}",
        f"Final support IDs: {payload.get('final_support_ids')}",
        f"Train summary: {train_text}",
        f"CV-W summary: {cv_text}",
        f"Test summary: {test_text}",
        f"Support comparison: {comparison_text}",
        "Full75 is a Train-capacity diagnostic, not the final model.",
        "No LUT retrieval, common-B, Aend/C2, Type-III, DPD, low-bandwidth, dmax3, "
        "or expanded dictionary task was run.",
        f"Raw snapshot before: {raw_before_text}",
        f"Raw snapshot after: {raw_after_text}",
        f"raw_data_modified: {payload['raw_before'] != payload['raw_after']}",
    ]
    return "\n".join(lines) + "\n"


def _aggregate_final(values: pd.Series) -> dict[str, float | int]:
    array = values.to_numpy(dtype=float)
    return {
        "pass_count": int(np.count_nonzero(array < TARGET_NMSE_DB)),
        "median_dB": float(np.median(array)),
        "Q95_dB": float(np.quantile(array, 0.95)),
        "worst_dB": float(np.max(array)),
    }


def _write_empty_after_capacity_failure() -> None:
    pd.DataFrame(columns=["iteration", "phase", "operation"]).to_csv(
        RESULT_ROOT / "05_sparse_search_history.csv", index=False
    )
    pd.DataFrame(columns=["candidate_id", "K", "support_hash"]).to_csv(
        RESULT_ROOT / "06_pareto_supports.csv", index=False
    )
    pd.DataFrame(columns=["candidate_id", "lambda", "N40"]).to_csv(
        RESULT_ROOT / "07_ridge_scan.csv", index=False
    )
    pd.DataFrame(columns=["term_index", "basis_id", "p", "m", "q", "mandatory"]).to_csv(
        RESULT_ROOT / "08_final_model_definition.csv", index=False
    )
    pd.DataFrame(columns=["hard_rank", "State_ID", "Train_NMSE_dB", "Test_NMSE_dB"]).to_csv(
        RESULT_ROOT / "09_final_hard20_train_cv_test.csv", index=False
    )


def _run(*, resume: bool = False) -> dict[str, object]:
    existing_output = RESULT_ROOT.exists() and any(RESULT_ROOT.iterdir())
    if existing_output and not resume:
        raise RuntimeError(
            "Result directory is not empty; use --resume only for this same incomplete task: "
            f"{RESULT_ROOT}"
        )
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    WORK_LOG.parent.mkdir(parents=True, exist_ok=True)
    raw_before = _raw_manifest()
    expected_raw = {
        "sha256": "910b06e4be6b60b1f4dc63409a3083eb328765347fbec3b89a97a41d6633a1e0",
        "file_count": 429,
        "mat_count": 427,
        "bytes": 2_258_448_137,
    }
    if raw_before != expected_raw:
        raise RuntimeError(f"raw manifest differs from frozen baseline: {raw_before}")
    terms, _, envelope18_hash = verify_frozen_support()
    terms = tuple(build_envelope_dictionary())
    candidate_hash = hashlib.sha256(
        json.dumps([term.basis_id for term in terms], separators=(",", ":")).encode()
    ).hexdigest()
    if len(terms) != 75:
        raise RuntimeError("Envelope75 candidate dictionary count is not 75")
    RESULT_ROOT.joinpath("00_task_definition.txt").write_text(
        _task_definition(terms, candidate_hash), encoding="utf-8"
    )
    _append_log(
        WORK_LOG,
        "\n"
        f"[{_now()}] Start {TASK_NAME}\n"
        "Mode: new independent ilc_end behavior basis-selection task.\n"
        f"Raw snapshot before: {json.dumps(raw_before, ensure_ascii=False, sort_keys=True)}\n"
        f"Envelope18 baseline hash={envelope18_hash}; candidate_hash={candidate_hash}; "
        "workers=10; BLAS threads/worker=1.\n"
        + (
            "Resume note: previous same-task output used the legacy 17-term baseline; "
            "this run corrects it to the frozen 18-term Envelope18 support.\n"
            if existing_output
            else ""
        ),
    )

    ranking = run_ranking()
    preflight_all = run_ilcend_preflight(range(425))
    count_distribution = preflight_all["valid_ilc_count"].value_counts().sort_index().to_dict()
    if count_distribution != {2: 1, 3: 416, 4: 7, 5: 1}:
        raise RuntimeError(f"ilc_end valid-column distribution changed: {count_distribution}")
    hard = ranking.loc[ranking["selected_Hard20"]].copy().reset_index(drop=True)
    hard_ids = tuple(int(value) for value in hard["state_id"])
    if hard_ids != EXPECTED_HARD20:
        raise RuntimeError(f"Hard20 differs from frozen expected set: {hard_ids}")
    hard = hard.rename(columns={"rank": "hard_rank", "state_id": "State_ID"})
    preflight_hard = preflight_all.loc[preflight_all["State_ID"].isin(hard_ids)].copy()
    preflight_hard = preflight_hard.set_index("State_ID").loc[list(hard_ids)].reset_index()
    hard_states = hard.merge(preflight_hard, on="State_ID", how="inner", validate="one_to_one")
    hard_states.to_csv(RESULT_ROOT / "01_hard20_states.csv", index=False)
    preflight_hard.to_csv(RESULT_ROOT / "02_ilcend_data_preflight.csv", index=False)
    checkpoint(
        CHECKPOINT_PATH,
        phase="hard20_frozen",
        hard20_state_ids=list(hard_ids),
        dictionary_hash=candidate_hash,
        current_support=None,
        completed_round=0,
        lambda_scan_progress=0,
        model_frozen=False,
        test_unlocked=False,
        raw_data_modified=False,
    )

    baseline_by_id = {term.basis_id: term.index for term in terms}
    missing_baseline = [
        basis_id for basis_id in FINAL_SUPPORT_IDS if basis_id not in baseline_by_id
    ]
    if missing_baseline:
        raise RuntimeError(
            f"Frozen Envelope18 baseline IDs missing from Envelope75: {missing_baseline}"
        )
    baseline = tuple(baseline_by_id[basis_id] for basis_id in FINAL_SUPPORT_IDS)
    if len(baseline) != 18 or len(set(baseline)) != 18:
        raise RuntimeError("Frozen Envelope18 baseline must contain 18 unique terms")
    full75 = tuple(range(len(terms)))
    temp_root = Path(tempfile.mkdtemp(prefix=f"{TASK_NAME}_"))
    try:
        cache = create_train_cache(temp_root / "train_cache", hard_ids)
        cache_metadata = populate_ilcend_cache(cache)
        metadata_by_state = {int(row["state_id"]): row for row in cache_metadata}
        for _, row in preflight_hard.iterrows():
            cached = metadata_by_state[int(row["State_ID"])]
            if int(cached["ilc_end_index"]) != int(row["ilc_end_index"]):
                raise RuntimeError(f"ilc_end cache mismatch at State {row['State_ID']}")
        with TrainSelectionEvaluator(cache, terms) as evaluator:
            capacity = evaluator.evaluate_capacity(
                (("Full75", full75),), progress_label="full75-capacity"
            )
            capacity["pass_lt_minus40"] = capacity["train_nmse_db"] < TARGET_NMSE_DB
            capacity.to_csv(RESULT_ROOT / "04_full75_train_capacity.csv", index=False)
            capacity_summary = {
                "pass_count": int(capacity["pass_lt_minus40"].sum()),
                "median_dB": float(capacity["train_nmse_db"].median()),
                "Q95_dB": float(capacity["train_nmse_db"].quantile(0.95)),
                "worst_dB": float(capacity["train_nmse_db"].max()),
                "rank_min": int(capacity["rank"].min()),
                "rank_max": int(capacity["rank"].max()),
                "condition_q99": float(capacity["condition_number"].quantile(0.99)),
                "condition_max": float(capacity["condition_number"].max()),
                "finite": bool(capacity["finite"].all()),
            }
            if capacity_summary["rank_min"] != 75 or not capacity_summary["finite"]:
                raise RuntimeError(f"Full75 numerical capacity gate failed: {capacity_summary}")
            if capacity_summary["pass_count"] != HARD_STATE_COUNT:
                raw_after = _raw_manifest()
                _write_empty_after_capacity_failure()
                payload = {
                    "hard20_state_ids": list(hard_ids),
                    "full75_capacity": capacity_summary,
                    "baseline_summary": None,
                    "final_status": "Envelope75+dmax2 candidate-space capacity insufficient",
                    "train_summary": None,
                    "cv_summary": None,
                    "test_summary": None,
                    "support_comparison": None,
                    "raw_before": raw_before,
                    "raw_after": raw_after,
                }
                (RESULT_ROOT / "12_final_result_summary.txt").write_text(
                    _summary_text(payload), encoding="utf-8"
                )
                return {"status": payload["final_status"], "test_unlocked": False}

            support_payloads: dict[str, tuple[int, ...]] = {"Envelope18": baseline}
            support_summaries: dict[str, dict[str, object]] = {}
            support_frames: dict[str, pd.DataFrame] = {}
            history: list[dict[str, object]] = []

            def evaluate_specs(
                specs: list[tuple[str, tuple[int, ...]]], ridge: float, label: str
            ) -> None:
                summaries, frames = evaluator.evaluate(specs, ridge, progress_label=label)
                for support_id, summary in summaries.items():
                    support_summaries[support_id] = summary
                    support_frames[support_id] = frames[support_id]
                    support_payloads[support_id] = tuple(
                        int(index)
                        for name, support in specs
                        if name == support_id
                        for index in support
                    )

            evaluate_specs([("Envelope18", baseline)], 0.0, "ilcend-envelope18-baseline")
            baseline_summary = support_summaries["Envelope18"]
            support_frames["Envelope18"].assign(model_variant="Envelope18").to_csv(
                RESULT_ROOT / "03_envelope18_baseline_train_cv.csv", index=False
            )
            history.append(
                _history_row(
                    "baseline",
                    0,
                    baseline_summary,
                    accepted=True,
                    operation="fixed_baseline",
                    reason="ilc_end Envelope18 baseline",
                )
            )
            current_support = baseline
            current_id = "Envelope18"
            current_summary = baseline_summary
            remaining = {
                term.index for term in terms if term.order > 1 and term.index not in set(baseline)
            }
            last_candidates: list[tuple[int, str, dict[str, object]]] = []
            best_forward_id = current_id
            best_pruned_id = current_id
            best_pair_id: str | None = None
            best_swap_id: str | None = None
            round_number = 0
            plateau = False

            while remaining and len(current_support) < K_SAFETY_CAP:
                round_number += 1
                candidates = []
                for index in sorted(remaining):
                    if len(current_support) + 1 > K_SAFETY_CAP:
                        break
                    candidate = tuple(sorted((*current_support, index)))
                    candidates.append(
                        (f"Forward_r{round_number}_{terms[index].basis_id}", candidate, index)
                    )
                if not candidates:
                    break
                evaluate_specs(
                    [(label, support) for label, support, _ in candidates],
                    0.0,
                    f"ilcend-forward-{round_number}",
                )
                candidate_rows = [
                    (index, label, support_summaries[label]) for label, _, index in candidates
                ]
                last_candidates = candidate_rows
                ordered = sorted(candidate_rows, key=lambda item: support_sort_key(item[2]))
                best_index, best_id, best_summary = ordered[0]
                for index, label, summary in candidate_rows:
                    accepted = label == best_id and support_improves(current_summary, summary)
                    history.append(
                        _history_row(
                            "forward",
                            round_number,
                            summary,
                            accepted=accepted,
                            operation="add",
                            added=terms[index].basis_id,
                            reason="best improvement" if accepted else "not accepted",
                        )
                    )
                if not support_improves(current_summary, best_summary):
                    plateau = True
                    break
                current_support = tuple(sorted((*current_support, best_index)))
                current_id = best_id
                current_summary = best_summary
                best_forward_id = current_id
                remaining.remove(best_index)
                while True:
                    removable = [index for index in current_support if terms[index].order > 1]
                    deletion_specs = [
                        (
                            f"Backward_{current_id}_{terms[removed].basis_id}",
                            tuple(index for index in current_support if index != removed),
                            removed,
                        )
                        for removed in removable
                    ]
                    if not deletion_specs:
                        break
                    evaluate_specs(
                        [(label, support) for label, support, _ in deletion_specs],
                        0.0,
                        f"ilcend-backward-{round_number}",
                    )
                    allowed = [
                        item
                        for item in deletion_specs
                        if deletion_allowed(current_summary, support_summaries[item[0]])
                    ]
                    if not allowed:
                        break
                    label, support, removed = min(
                        allowed, key=lambda item: support_sort_key(support_summaries[item[0]])
                    )
                    current_support = tuple(support)
                    current_id = label
                    current_summary = support_summaries[label]
                    best_pruned_id = current_id
                    history.append(
                        _history_row(
                            "backward",
                            round_number,
                            current_summary,
                            accepted=True,
                            operation="delete",
                            removed=terms[removed].basis_id,
                            reason="within backward tolerance",
                        )
                    )
                    current_set = set(current_support)
                    remaining = {
                        term.index
                        for term in terms
                        if term.order > 1 and term.index not in current_set
                    }

            if (plateau or not remaining) and current_summary["pass_count_W40"] < HARD_STATE_COUNT:
                top_single = sorted(last_candidates, key=lambda item: support_sort_key(item[2]))[
                    :12
                ]
                pair_specs = []
                for left_pos, left in enumerate(top_single):
                    for right in top_single[left_pos + 1 :]:
                        i, _, _ = left
                        j, _, _ = right
                        if i in current_support or j in current_support or i == j:
                            continue
                        support = tuple(sorted((*current_support, i, j)))
                        if len(support) > K_SAFETY_CAP:
                            continue
                        pair_specs.append(
                            (f"Pair_{terms[i].basis_id}_{terms[j].basis_id}", support, (i, j))
                        )
                if pair_specs:
                    evaluate_specs(
                        [(label, support) for label, support, _ in pair_specs],
                        0.0,
                        "ilcend-pair-rescue",
                    )
                    label, support, added = min(
                        pair_specs, key=lambda item: support_sort_key(support_summaries[item[0]])
                    )
                    summary = support_summaries[label]
                    accepted = support_improves(current_summary, summary)
                    history.append(
                        _history_row(
                            "pair_rescue",
                            round_number,
                            summary,
                            accepted=accepted,
                            operation="add_pair",
                            added=";".join(terms[index].basis_id for index in added),
                            reason="bounded pair rescue",
                        )
                    )
                    if accepted:
                        current_support, current_id, current_summary = (
                            tuple(support),
                            label,
                            summary,
                        )
                        best_pair_id = current_id

            swap_add_indices = [
                item[0]
                for item in sorted(last_candidates, key=lambda item: support_sort_key(item[2]))[:12]
            ]
            swap_specs = []
            for removed in [index for index in current_support if terms[index].order > 1]:
                for added in swap_add_indices:
                    if added in current_support or added == removed:
                        continue
                    support = tuple(sorted((set(current_support) - {removed}) | {added}))
                    swap_specs.append(
                        (
                            f"Swap_{terms[removed].basis_id}_to_{terms[added].basis_id}",
                            support,
                            (removed, added),
                        )
                    )
            if swap_specs:
                evaluate_specs(
                    [(label, support) for label, support, _ in swap_specs], 0.0, "ilcend-local-swap"
                )
                label, support, pair = min(
                    swap_specs, key=lambda item: support_sort_key(support_summaries[item[0]])
                )
                summary = support_summaries[label]
                accepted = support_improves(current_summary, summary)
                history.append(
                    _history_row(
                        "local_swap",
                        round_number,
                        summary,
                        accepted=accepted,
                        operation="swap",
                        added=terms[pair[1]].basis_id,
                        removed=terms[pair[0]].basis_id,
                        reason="bounded local swap",
                    )
                )
                if accepted:
                    current_support, current_id, current_summary = tuple(support), label, summary
                    best_swap_id = current_id

            milestone_ids = []
            for label in (
                "Envelope18",
                best_forward_id,
                best_pruned_id,
                best_pair_id,
                best_swap_id,
                current_id,
            ):
                if label is not None and label in support_summaries and label not in milestone_ids:
                    milestone_ids.append(label)
            pareto = pareto_filter([support_summaries[label] for label in milestone_ids])
            if not any(item["support_id"] == "Envelope18" for item in pareto):
                pareto.append(support_summaries["Envelope18"])
            pareto = sorted(pareto, key=support_sort_key)[:6]
            pd.DataFrame(pareto).to_csv(RESULT_ROOT / "06_pareto_supports.csv", index=False)

            history_frame = pd.DataFrame(history)
            history_frame.to_csv(RESULT_ROOT / "05_sparse_search_history.csv", index=False)

            ridge_rows: list[dict[str, object]] = []
            for ridge in RIDGE_GRID:
                specs = [
                    (f"{label}@lambda={ridge:.12g}", support_payloads[label])
                    for label in [item["support_id"] for item in pareto]
                ]
                evaluate_specs(specs, ridge, f"ilcend-ridge-{ridge:.0e}")
                for label in [item["support_id"] for item in pareto]:
                    eval_id = f"{label}@lambda={ridge:.12g}"
                    row = dict(support_summaries[eval_id])
                    row["candidate_id"] = label
                    row["support_hash"] = _support_hash(terms, support_payloads[label])
                    ridge_rows.append(row)
            ridge_frame = pd.DataFrame(ridge_rows)
            ridge_frame = ridge_frame.sort_values(
                [
                    "pass_count_W40",
                    "W_worst_dB",
                    "W_Q95_deficit_dB",
                    "K",
                    "ridge_lambda",
                    "candidate_id",
                ],
                ascending=[False, True, True, True, True, True],
            ).reset_index(drop=True)
            ridge_frame.insert(0, "rank", np.arange(1, len(ridge_frame) + 1))
            ridge_frame.to_csv(RESULT_ROOT / "07_ridge_scan.csv", index=False)
            selected_row = ridge_frame.iloc[0]
            selected_candidate = str(selected_row["candidate_id"])
            selected_lambda = float(selected_row["ridge_lambda"])
            selected_support = support_payloads[selected_candidate]
            selected_summary = support_summaries[
                f"{selected_candidate}@lambda={selected_lambda:.12g}"
            ]
            if int(selected_summary["pass_count_W40"]) != HARD_STATE_COUNT:
                raw_after = _raw_manifest()
                _write_empty_after_capacity_failure()
                summary_payload = {
                    "hard20_state_ids": list(hard_ids),
                    "full75_capacity": capacity_summary,
                    "baseline_summary": baseline_summary,
                    "final_status": "ilc_end sparse support did not reach CV-W 20/20",
                    "final_K": len(selected_support),
                    "final_lambda": selected_lambda,
                    "final_support_hash": _support_hash(terms, selected_support),
                    "final_support_ids": list(support_ids(terms, selected_support)),
                    "train_summary": None,
                    "cv_summary": selected_summary,
                    "test_summary": None,
                    "support_comparison": None,
                    "raw_before": raw_before,
                    "raw_after": raw_after,
                }
                (RESULT_ROOT / "12_final_result_summary.txt").write_text(
                    _summary_text(summary_payload), encoding="utf-8"
                )
                return {"status": summary_payload["final_status"], "test_unlocked": False}

            final_ids = support_ids(terms, selected_support)
            final_hash = _support_hash(terms, selected_support)
            final_definition = pd.DataFrame(
                [
                    {
                        "term_index": int(index),
                        "basis_id": terms[index].basis_id,
                        "p": terms[index].order,
                        "m": terms[index].signal_delay,
                        "q": terms[index].envelope_delay,
                        "mandatory": terms[index].mandatory,
                        "K": len(selected_support),
                        "lambda": selected_lambda,
                        "dmax": DMAX,
                        "support_hash": final_hash,
                    }
                    for index in selected_support
                ]
            )
            final_definition.to_csv(RESULT_ROOT / "08_final_model_definition.csv", index=False)
            checkpoint(
                CHECKPOINT_PATH,
                phase="model_frozen",
                hard20_state_ids=list(hard_ids),
                dictionary_hash=candidate_hash,
                current_support=list(final_ids),
                completed_round=round_number,
                lambda_scan_progress=len(ridge_rows),
                model_frozen=True,
                test_unlocked=True,
                raw_data_modified=False,
            )
            final_train_summaries, final_train_frames = evaluator.evaluate(
                [("FinalFrozen", selected_support)],
                selected_lambda,
                exact_condition=True,
                progress_label="final-ilcend-train",
            )
            final_train_frame = final_train_frames["FinalFrozen"]
            final_test_frame = run_final_evaluation(cache, terms, selected_support, selected_lambda)
            final = hard_states[["hard_rank", "State_ID", "Train_noDPD_NMSE_dB"]].rename(
                columns={"State_ID": "State_ID", "Train_noDPD_NMSE_dB": "NMSE_withoutdpd_dB"}
            )
            final = final.merge(
                final_train_frame.rename(
                    columns={
                        "state_id": "State_ID",
                        "train_nmse_db": "Train_NMSE_dB",
                        "W_db": "CV_W_NMSE_dB",
                    }
                )[
                    [
                        "State_ID",
                        "Train_NMSE_dB",
                        "cv1_nmse_db",
                        "cv2_nmse_db",
                        "cv3_nmse_db",
                        "CV_W_NMSE_dB",
                    ]
                ],
                on="State_ID",
                how="inner",
            )
            final = final.merge(
                final_test_frame.rename(
                    columns={"state_id": "State_ID", "test_nmse_db": "Test_NMSE_dB"}
                )[["State_ID", "Test_NMSE_dB"]],
                on="State_ID",
                how="inner",
            )
            final["Train_pass"] = final["Train_NMSE_dB"] < TARGET_NMSE_DB
            final["CV_pass"] = final["CV_W_NMSE_dB"] < TARGET_NMSE_DB
            final["Test_pass"] = final["Test_NMSE_dB"] < TARGET_NMSE_DB
            final = final.rename(
                columns={
                    "cv1_nmse_db": "CV1_NMSE_dB",
                    "cv2_nmse_db": "CV2_NMSE_dB",
                    "cv3_nmse_db": "CV3_NMSE_dB",
                }
            )
            final.to_csv(RESULT_ROOT / "09_final_hard20_train_cv_test.csv", index=False)
            support_comparison = []
            no_dpd_set = set(FINAL_SUPPORT_IDS)
            ilc_set = set(final_ids)
            for term in terms:
                if term.basis_id in no_dpd_set and term.basis_id in ilc_set:
                    category = "shared"
                elif term.basis_id in ilc_set:
                    category = "ilc_end_only"
                elif term.basis_id in no_dpd_set:
                    category = "noDPD_only"
                else:
                    category = "neither_final_support"
                support_comparison.append(
                    {"basis_id": term.basis_id, "order": term.order, "category": category}
                )
            comparison_frame = pd.DataFrame(support_comparison)
            comparison_frame.to_csv(
                RESULT_ROOT / "10_support_comparison_vs_nodpd_envelope18.csv", index=False
            )
            _plot_final(final, RESULT_ROOT / "11_final_modeling_generalization.png")
            train_summary = _aggregate_final(final["Train_NMSE_dB"])
            cv_summary = _aggregate_final(final["CV_W_NMSE_dB"])
            test_summary = _aggregate_final(final["Test_NMSE_dB"])
            final_status = (
                "SUCCESS"
                if train_summary["pass_count"] == HARD_STATE_COUNT
                and cv_summary["pass_count"] == HARD_STATE_COUNT
                and test_summary["pass_count"] == HARD_STATE_COUNT
                else "NOT FULLY SUCCESSFUL"
            )
            raw_after = _raw_manifest()
            summary_payload = {
                "hard20_state_ids": list(hard_ids),
                "full75_capacity": capacity_summary,
                "baseline_summary": baseline_summary,
                "final_status": final_status,
                "final_K": len(selected_support),
                "final_lambda": selected_lambda,
                "final_support_hash": final_hash,
                "final_support_ids": list(final_ids),
                "train_summary": train_summary,
                "cv_summary": cv_summary,
                "test_summary": test_summary,
                "support_comparison": {
                    "noDPD_K": len(no_dpd_set),
                    "ilc_end_K": len(ilc_set),
                    "shared_count": len(no_dpd_set & ilc_set),
                    "ilc_end_only": sorted(ilc_set - no_dpd_set),
                    "noDPD_only": sorted(no_dpd_set - ilc_set),
                },
                "raw_before": raw_before,
                "raw_after": raw_after,
            }
            (RESULT_ROOT / "12_final_result_summary.txt").write_text(
                _summary_text(summary_payload), encoding="utf-8"
            )
            checkpoint(
                CHECKPOINT_PATH,
                phase="completed",
                hard20_state_ids=list(hard_ids),
                dictionary_hash=candidate_hash,
                current_support=list(final_ids),
                completed_round=round_number,
                lambda_scan_progress=len(ridge_rows),
                model_frozen=True,
                test_unlocked=True,
                raw_data_modified=raw_before != raw_after,
            )
            return {
                "status": final_status,
                "final_K": len(selected_support),
                "final_lambda": selected_lambda,
                "train_pass_count": train_summary["pass_count"],
                "cv_pass_count": cv_summary["pass_count"],
                "test_pass_count": test_summary["pass_count"],
                "raw_data_modified": raw_before != raw_after,
                "result_root": str(RESULT_ROOT),
            }
    finally:
        import shutil

        shutil.rmtree(temp_root, ignore_errors=True)


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
        "No LUT retrieval, Aend/C2, common-B, Type-III, DPD, low-bandwidth, dmax3, "
        "or expanded dictionary task was run.\n",
    )
    _append_log(
        HANDOFF_LOG,
        f"\n{_now()} | 新任务：{TASK_NAME}\n"
        f"完成 ilc_end Envelope75 Hard20 basis selection；结果目录：{RESULT_ROOT}\n"
        f"结果摘要：{json.dumps(result, ensure_ascii=False, sort_keys=True)}\n"
        "固定使用 10 个 spawn worker、每 worker 1 个 BLAS 线程；"
        "Test 仅在 support/lambda/dmax freeze 后解锁；"
        "未运行 LUT retrieval、Aend/C2、common-B、Type-III、DPD 或 low-bandwidth。\n",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    __import__("multiprocessing").freeze_support()
    main()
