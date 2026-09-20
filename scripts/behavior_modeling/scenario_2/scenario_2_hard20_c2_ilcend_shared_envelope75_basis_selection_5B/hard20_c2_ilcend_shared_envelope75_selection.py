"""Shared Envelope75 support selection for ILC_END and literal ILC column 2."""

from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any

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

from core.shared.metrics import nmse  # noqa: E402
from data_management.shared import build_state_table  # noqa: E402

from behavior_modeling.shared.basis_function_selection.all425_envelope18_evaluation import (  # noqa: E402
    verify_frozen_support as verify_envelope18_support,
)
from behavior_modeling.shared.basis_function_selection.all425_ilcend_envelope23_evaluation import (  # noqa: E402
    verify_frozen_ilcend_support as verify_envelope23_support,
)
from behavior_modeling.shared.basis_function_selection.data_preparation import (
    prepare_train_only_state,  # noqa: E402
)
from behavior_modeling.shared.basis_function_selection.dual_ilc_behavior_dataset import (  # noqa: E402
    BEHAVIORS,
    ILC_COL2,
    ILC_END,
    PreparedDualBehaviorState,
    preflight_rows,
    prepare_dual_state,
)
from behavior_modeling.shared.basis_function_selection.envelope_dictionary import (  # noqa: E402
    DMAX,
    EnvelopeBasis,
    build_envelope_bank,
    build_envelope_dictionary,
)
from behavior_modeling.shared.basis_function_selection.hard20_envelope75_selection import (  # noqa: E402
    _raw_manifest,
)
from behavior_modeling.shared.basis_function_selection.model_solver import fit_ols  # noqa: E402

TASK_NAME = "scenario_2_hard20_c2_ilcend_shared_envelope75_basis_selection_5B"
RESULT_ROOT = PROJECT_ROOT / "results" / "behavior_modeling" / "scenario_2" /TASK_NAME
WORK_LOG = PROJECT_ROOT / "work_logs" / "behavior_modeling" / "scenario_2" /TASK_NAME / "execution_log.txt"
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"
CHECKPOINT_PATH = RESULT_ROOT / "12_checkpoint.json"

STATE_COUNT = 425
HARD_STATE_COUNT = 20
FULL_LENGTH = 24_576
TRAIN_START = 0
TRAIN_END = 16_384
TEST_START = 16_384
TEST_END = 24_576
TRAIN_LENGTH = TRAIN_END - TRAIN_START
TEST_LENGTH = TEST_END - TEST_START
TARGET_NMSE_DB = -40.0
WORKER_COUNT = 10
BLAS_THREADS_PER_WORKER = 1
FINE_ALIGN_SUBTIME = 256
ILC_COL2_COLUMN_INDEX = 1
CV_BLOCK_RAW_LENGTHS = (5_461, 5_462, 5_461)
CV_BLOCK_VALID_LENGTHS = (5_459, 5_460, 5_459)
MANDATORY_BASIS_IDS = ("LIN_d0", "LIN_d1", "LIN_d2")
OLD_E18_HASH = "e6a8405ba5b9417b169e64acff2d8c2426139ae199ce7b5a45ca92dbc035cdbc"
OLD_E23_HASH = "7c8348beaaa9db183c26bb8adfd29f226c11f36bf54ad587e9ac0baa6f14ea14"
EXPECTED_RAW_MANIFEST = {
    "sha256": "910b06e4be6b60b1f4dc63409a3083eb328765347fbec3b89a97a41d6633a1e0",
    "file_count": 429,
    "mat_count": 427,
    "bytes": 2_258_448_137,
}
EXPECTED_HARD20_IDS = (
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

E18_MODEL_DEFINITION = (
    PROJECT_ROOT
    / "results"
    / "scenario_2_hard20_envelope75_basis_selection_5B"
    / "09_final_model_definition.csv"
)
E23_MODEL_DEFINITION = (
    PROJECT_ROOT
    / "results"
    / "scenario_2_hard20_ilcend_envelope75_basis_selection_5B"
    / "08_final_model_definition.csv"
)

_WORKER_TERMS: tuple[EnvelopeBasis, ...] | None = None
_WORKER_STATE_TABLE: dict[int, dict[str, int | float]] | None = None
_WORKER_CACHE: dict[int, PreparedDualBehaviorState] = {}


def support_hash(ids: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(ids), ensure_ascii=False, separators=(",", ":")).encode()
    ).hexdigest()


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message)


def raw_manifest_gate() -> dict[str, object]:
    value = _raw_manifest()
    if value != EXPECTED_RAW_MANIFEST:
        raise RuntimeError(f"raw manifest differs from frozen baseline: {value}")
    return value


def assert_test_access_allowed(*, model_frozen: bool, test_unlocked: bool) -> None:
    if not model_frozen:
        raise RuntimeError("Final Test evaluation is forbidden before shared support is frozen.")
    if not test_unlocked:
        raise RuntimeError("Final Test evaluation is still locked.")


def _state_table() -> dict[int, dict[str, int | float]]:
    return {int(row["state_id"]): row for row in build_state_table()}


def _hard20_ranking_worker(state_id: int) -> dict[str, object]:
    prepared = prepare_train_only_state(int(state_id))
    return {
        "State_ID": int(state_id),
        "NMSE_withoutdpd_dB": float(nmse(prepared.x_train, prepared.y_train_adjusted)),
        "rough_delay": int(prepared.rough_delay),
        "fine_delay": float(prepared.fine_delay),
    }


def run_hard20_ranking() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank all 425 states from no-DPD Train only, without reading Test."""

    rows: list[dict[str, object]] = []
    context = get_context("spawn")
    with ProcessPoolExecutor(max_workers=WORKER_COUNT, mp_context=context) as executor:
        futures = {
            executor.submit(_hard20_ranking_worker, state_id): state_id
            for state_id in range(STATE_COUNT)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            if completed % 50 == 0 or completed == STATE_COUNT:
                print(f"[HARD20 RANK] {completed}/{STATE_COUNT}", flush=True)
    ranking = pd.DataFrame(rows).sort_values(
        ["NMSE_withoutdpd_dB", "State_ID"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranking.insert(0, "hard_rank", np.arange(1, STATE_COUNT + 1, dtype=np.int64))
    hard20 = ranking.head(HARD_STATE_COUNT).copy()
    actual_ids = tuple(hard20["State_ID"].astype(int).tolist())
    if actual_ids != EXPECTED_HARD20_IDS:
        raise RuntimeError(f"Hard20 no-DPD ranking changed: {actual_ids}")
    state_table = _state_table()
    hard20["funMng"] = [int(state_table[state_id]["funMng"]) for state_id in actual_ids]
    hard20["funAng"] = [int(state_table[state_id]["funAng"]) for state_id in actual_ids]
    hard20["secMng"] = [int(state_table[state_id]["secMng"]) for state_id in actual_ids]
    hard20["secAng"] = [int(state_table[state_id]["secAng"]) for state_id in actual_ids]
    return ranking, hard20


def load_supports() -> tuple[tuple[EnvelopeBasis, ...], dict[str, tuple[int, ...]], dict[str, str]]:
    """Load E18/E23 from their frozen definitions and derive canonical Union24."""

    terms, e18_indices, e18_hash = verify_envelope18_support()
    terms23, e23_indices, e23_hash = verify_envelope23_support()
    if tuple(term.basis_id for term in terms) != tuple(term.basis_id for term in terms23):
        raise RuntimeError("Envelope18 and Envelope23 dictionaries do not share canonical order")
    e18_ids = tuple(terms[index].basis_id for index in e18_indices)
    e23_ids = tuple(terms[index].basis_id for index in e23_indices)
    if e18_hash != OLD_E18_HASH or e23_hash != OLD_E23_HASH:
        raise RuntimeError(f"Frozen baseline hash changed: E18={e18_hash}, E23={e23_hash}")
    e18_set = set(e18_ids)
    e23_set = set(e23_ids)
    union_ids = tuple(term.basis_id for term in terms if term.basis_id in e18_set | e23_set)
    union_indices = tuple(term.index for term in terms if term.basis_id in e18_set | e23_set)
    if len(e18_ids) != 18 or len(e23_ids) != 23 or len(union_ids) != 24:
        raise RuntimeError("E18/E23/Union24 support cardinality gate failed")
    if set(union_ids) != e18_set | e23_set:
        raise RuntimeError("Union24 support set gate failed")
    support_indices = {
        "Envelope18": tuple(e18_indices),
        "Envelope23-ilcEnd": tuple(e23_indices),
        "Union24": union_indices,
    }
    hashes = {
        "Envelope18": e18_hash,
        "Envelope23-ilcEnd": e23_hash,
        "Union24": support_hash(union_ids),
    }
    return tuple(terms), support_indices, hashes


def _worker_init() -> None:
    global _WORKER_TERMS, _WORKER_STATE_TABLE, _WORKER_CACHE
    _WORKER_TERMS = tuple(build_envelope_dictionary())
    _WORKER_STATE_TABLE = _state_table()
    _WORKER_CACHE = {}


def _get_prepared_train(state_id: int) -> PreparedDualBehaviorState:
    if _WORKER_TERMS is None:
        raise RuntimeError("dual-behavior worker terms are not initialized")
    prepared = _WORKER_CACHE.get(int(state_id))
    if prepared is None:
        prepared = prepare_dual_state(int(state_id), _WORKER_TERMS, include_test=False)
        _WORKER_CACHE[int(state_id)] = prepared
    return prepared


def _fit_behavior_support(
    prepared: Any,
    support: tuple[int, ...],
) -> dict[str, object]:
    indices = np.asarray(support, dtype=np.int64)
    full_phi = np.asarray(prepared.full_bank[:, indices], dtype=np.complex128)
    full_fit = fit_ols(full_phi, prepared.full_target, scale_columns=True)
    cv_values: list[float] = []
    ranks = [int(full_fit.rank)]
    conditions = [float(full_fit.condition_number)]
    finite = bool(
        np.all(np.isfinite(full_fit.theta))
        and np.all(np.isfinite(full_fit.prediction))
        and np.isfinite(full_fit.condition_number)
    )
    for validation in range(3):
        train_phi = np.concatenate(
            [
                prepared.block_banks[index][:, indices]
                for index in range(3)
                if index != validation
            ],
            axis=0,
        )
        train_target = np.concatenate(
            [
                prepared.block_targets[index]
                for index in range(3)
                if index != validation
            ],
            axis=0,
        )
        fit = fit_ols(train_phi, train_target, scale_columns=True)
        prediction = prepared.block_banks[validation][:, indices] @ fit.theta
        value = float(nmse(prepared.block_targets[validation], prediction))
        cv_values.append(value)
        ranks.append(int(fit.rank))
        conditions.append(float(fit.condition_number))
        finite = bool(
            finite
            and np.all(np.isfinite(fit.theta))
            and np.all(np.isfinite(prediction))
            and np.isfinite(value)
        )
    w_value = float(max([full_fit.nmse_db, *cv_values]))
    return {
        "Train_NMSE_dB": float(full_fit.nmse_db),
        "CV1_NMSE_dB": cv_values[0],
        "CV2_NMSE_dB": cv_values[1],
        "CV3_NMSE_dB": cv_values[2],
        "CV_W_NMSE_dB": w_value,
        "behavior_pass": bool(finite and w_value < TARGET_NMSE_DB),
        "rank": int(min(ranks)),
        "condition_number": float(max(conditions)),
        "finite": finite,
    }


def _state_support_worker(
    payload: tuple[int, tuple[tuple[str, tuple[int, ...]], ...]]
) -> dict[str, object]:
    state_id, specs = payload
    prepared = _get_prepared_train(int(state_id))
    support_results: dict[str, dict[str, dict[str, object]]] = {}
    for support_name, support in specs:
        support_results[support_name] = {
            behavior: _fit_behavior_support(prepared.behavior(behavior), support)
            for behavior in BEHAVIORS
        }
    return {"State_ID": int(state_id), "support_results": support_results}


def _capacity_worker(state_id: int) -> dict[str, object]:
    prepared = _get_prepared_train(int(state_id))
    result: dict[str, object] = {"State_ID": int(state_id), "preflight": preflight_rows(prepared)}
    capacity: dict[str, dict[str, object]] = {}
    full_support = tuple(range(len(_WORKER_TERMS or ())))
    for behavior in BEHAVIORS:
        item = prepared.behavior(behavior)
        fit = fit_ols(item.full_bank, item.full_target, scale_columns=True)
        capacity[behavior] = {
            "Train_NMSE_dB": float(fit.nmse_db),
            "rank": int(fit.rank),
            "condition_number": float(fit.condition_number),
            "finite": bool(
                np.all(np.isfinite(item.full_bank))
                and np.all(np.isfinite(item.full_target))
                and np.all(np.isfinite(fit.theta))
                and np.all(np.isfinite(fit.prediction))
                and np.isfinite(fit.condition_number)
            ),
            "pass_lt_minus40": bool(fit.nmse_db < TARGET_NMSE_DB),
            "K": len(full_support),
        }
    result["capacity"] = capacity
    return result


def _final_worker(payload: tuple[int, tuple[int, ...]]) -> dict[str, object]:
    state_id, support = payload
    assert_test_access_allowed(model_frozen=True, test_unlocked=True)
    terms = _WORKER_TERMS or tuple(build_envelope_dictionary())
    prepared = prepare_dual_state(int(state_id), terms, include_test=True)
    result: dict[str, object] = {
        "State_ID": int(state_id),
        "preflight": preflight_rows(prepared),
        "behaviors": {},
    }
    for behavior in BEHAVIORS:
        item = prepared.behavior(behavior)
        train_metrics = _fit_behavior_support(item, support)
        if item.x_test is None or item.y_test_adjusted is None or item.test_gain is None:
            raise RuntimeError("final Test data was not unlocked")
        test_bank = build_envelope_bank(item.x_test, terms)
        indices = np.asarray(support, dtype=np.int64)
        test_phi = np.asarray(test_bank[:, indices], dtype=np.complex128)
        full_phi = np.asarray(item.full_bank[:, indices], dtype=np.complex128)
        fit = fit_ols(full_phi, item.full_target, scale_columns=True)
        test_target = np.asarray(item.y_test_adjusted[DMAX:], dtype=np.complex128)
        test_prediction = test_phi @ fit.theta
        test_value = float(nmse(test_target, test_prediction))
        train_metrics["Test_NMSE_dB"] = test_value
        train_metrics["Test_pass"] = bool(test_value < TARGET_NMSE_DB)
        train_metrics["Train_pass"] = bool(train_metrics["Train_NMSE_dB"] < TARGET_NMSE_DB)
        train_metrics["CV_pass"] = bool(train_metrics["CV_W_NMSE_dB"] < TARGET_NMSE_DB)
        train_metrics["theta"] = np.asarray(fit.theta, dtype=np.complex128)
        result["behaviors"][behavior] = train_metrics
    return result


def evaluate_capacity_and_preflight(
    executor: ProcessPoolExecutor,
    hard_ids: Sequence[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    futures = {executor.submit(_capacity_worker, int(state_id)): int(state_id) for state_id in hard_ids}
    preflight_rows_all: list[dict[str, object]] = []
    capacity_rows: list[dict[str, object]] = []
    for completed, future in enumerate(as_completed(futures), start=1):
        result = future.result()
        preflight_rows_all.extend(result["preflight"])
        state_id = int(result["State_ID"])
        for behavior, values in result["capacity"].items():
            capacity_rows.append({"State_ID": state_id, "behavior": behavior, **values})
        if completed % 5 == 0 or completed == len(futures):
            print(f"[FULL75 CAPACITY] {completed}/{len(futures)}", flush=True)
    preflight = pd.DataFrame(preflight_rows_all).sort_values(["State_ID", "behavior"]).reset_index(drop=True)
    capacity = pd.DataFrame(capacity_rows).sort_values(["State_ID", "behavior"]).reset_index(drop=True)
    if len(preflight) != HARD_STATE_COUNT * 2 or len(capacity) != HARD_STATE_COUNT * 2:
        raise RuntimeError("dual preflight/capacity row count is not 40")
    return preflight, capacity


def evaluate_support_batch(
    executor: ProcessPoolExecutor,
    hard_ids: Sequence[int],
    specs: Sequence[tuple[str, tuple[int, ...]]],
    label: str,
) -> tuple[dict[str, dict[str, object]], dict[str, pd.DataFrame]]:
    normalized_specs = tuple((str(name), tuple(int(index) for index in support)) for name, support in specs)
    futures = {
        executor.submit(_state_support_worker, (int(state_id), normalized_specs)): int(state_id)
        for state_id in hard_ids
    }
    state_results: list[dict[str, object]] = []
    for completed, future in enumerate(as_completed(futures), start=1):
        state_results.append(future.result())
        if completed % 5 == 0 or completed == len(futures):
            print(f"[{label}] {completed}/{len(futures)}", flush=True)
    state_results.sort(key=lambda item: int(item["State_ID"]))
    summaries: dict[str, dict[str, object]] = {}
    frames: dict[str, pd.DataFrame] = {}
    canonical_terms = tuple(build_envelope_dictionary())
    for support_name, support in normalized_specs:
        rows: list[dict[str, object]] = []
        for state_result in state_results:
            state_id = int(state_result["State_ID"])
            by_behavior = state_result["support_results"][support_name]
            for behavior in BEHAVIORS:
                rows.append(
                    {
                        "support_name": support_name,
                        "K": len(support),
                        "State_ID": state_id,
                        "behavior": behavior,
                        **by_behavior[behavior],
                    }
                )
        frame = pd.DataFrame(rows).sort_values(["State_ID", "behavior"]).reset_index(drop=True)
        end_w = frame.loc[frame["behavior"] == ILC_END, "CV_W_NMSE_dB"].to_numpy(dtype=float)
        col2_w = frame.loc[frame["behavior"] == ILC_COL2, "CV_W_NMSE_dB"].to_numpy(dtype=float)
        joint_w = np.maximum(end_w, col2_w)
        if len(end_w) != HARD_STATE_COUNT or len(col2_w) != HARD_STATE_COUNT:
            raise RuntimeError(f"{support_name} does not cover both behaviors for all Hard20 states")
        behavior_pass_count = int(np.count_nonzero(frame["CV_W_NMSE_dB"].to_numpy(dtype=float) < TARGET_NMSE_DB))
        summaries[support_name] = {
            "support_name": support_name,
            "K": len(support),
            "support_hash": support_hash(
                tuple(canonical_terms[index].basis_id for index in support)
            ),
            "basis_ids": ";".join(canonical_terms[index].basis_id for index in support),
            "joint_pass_count": int(np.count_nonzero(joint_w < TARGET_NMSE_DB)),
            "behavior_pass_count": behavior_pass_count,
            "joint_worst_dB": float(np.max(joint_w)),
            "joint_Q95_dB": float(np.quantile(joint_w, 0.95)),
            "joint_median_dB": float(np.median(joint_w)),
            "end_W_median_dB": float(np.median(end_w)),
            "C2_W_median_dB": float(np.median(col2_w)),
        }
        frames[support_name] = frame
    return summaries, frames


def score_key(summary: dict[str, object]) -> tuple[object, ...]:
    """Lower is better for the declared joint-first lexicographic objective."""

    return (
        -int(summary["joint_pass_count"]),
        -int(summary["behavior_pass_count"]),
        float(summary["joint_worst_dB"]),
        float(summary["joint_Q95_dB"]),
        float(summary["joint_median_dB"]),
        int(summary["K"]),
        str(summary["basis_ids"]),
    )


def _write_behavior_plot(frame: pd.DataFrame, behavior: str, path: Path) -> None:
    ordered = frame.loc[frame["behavior"] == behavior].sort_values("hard_rank")
    x = ordered["hard_rank"].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(12, 6), dpi=220)
    for column, label, color in (
        ("Train_NMSE_dB", "Train", "#1f77b4"),
        ("CV_W_NMSE_dB", "CV-W", "#2ca02c"),
        ("Test_NMSE_dB", "Test", "#d62728"),
    ):
        ax.plot(x, ordered[column].to_numpy(dtype=float), marker="o", linewidth=1.3, label=label, color=color)
    ax.axhline(TARGET_NMSE_DB, color="#222222", linestyle="--", linewidth=1.0, label="-40 dB threshold")
    ax.set_xticks(x, [f"State{value}" for value in ordered["State_ID"].to_numpy(dtype=int)], rotation=45, ha="right")
    ax.set_xlabel("Fixed Hard20 rank order")
    ax.set_ylabel("NMSE (dB)")
    ax.set_title(f"{behavior}: final shared-support modeling and generalization")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def objective_improves(current: dict[str, object], candidate: dict[str, object]) -> bool:
    return score_key(candidate)[:5] < score_key(current)[:5]


def history_row(
    summary: dict[str, object],
    *,
    iteration: int,
    phase: str,
    operation: str,
    added_terms: str = "",
    removed_terms: str = "",
    accepted: bool,
    reason: str,
) -> dict[str, object]:
    return {
        "iteration": iteration,
        "phase": phase,
        "operation": operation,
        "added_terms": added_terms,
        "removed_terms": removed_terms,
        "K": summary["K"],
        "joint_pass_count_20": summary["joint_pass_count"],
        "behavior_pass_count_40": summary["behavior_pass_count"],
        "joint_worst_dB": summary["joint_worst_dB"],
        "joint_Q95_dB": summary["joint_Q95_dB"],
        "joint_median_dB": summary["joint_median_dB"],
        "support_hash": summary["support_hash"],
        "basis_ids": summary["basis_ids"],
        "accepted": bool(accepted),
        "reason": reason,
    }


def _build_baseline_table(
    summaries: dict[str, dict[str, object]],
    frames: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    columns = [
        "record_type",
        "support_name",
        "K",
        "support_hash",
        "State_ID",
        "behavior",
        "Train_NMSE_dB",
        "CV1_NMSE_dB",
        "CV2_NMSE_dB",
        "CV3_NMSE_dB",
        "CV_W_NMSE_dB",
        "behavior_pass",
        "joint_pass_count",
        "behavior_pass_count",
        "joint_worst_dB",
        "joint_Q95_dB",
        "joint_median_dB",
    ]
    for name, frame in frames.items():
        summary = summaries[name]
        for row in frame.to_dict("records"):
            records.append(
                {
                    "record_type": "per_state",
                    "support_name": name,
                    "K": summary["K"],
                    "support_hash": summary["support_hash"],
                    "State_ID": row["State_ID"],
                    "behavior": row["behavior"],
                    "Train_NMSE_dB": row["Train_NMSE_dB"],
                    "CV1_NMSE_dB": row["CV1_NMSE_dB"],
                    "CV2_NMSE_dB": row["CV2_NMSE_dB"],
                    "CV3_NMSE_dB": row["CV3_NMSE_dB"],
                    "CV_W_NMSE_dB": row["CV_W_NMSE_dB"],
                    "behavior_pass": row["behavior_pass"],
                }
            )
        records.append(
            {
                "record_type": "support_summary",
                "support_name": name,
                "K": summary["K"],
                "support_hash": summary["support_hash"],
                "State_ID": "",
                "behavior": "JOINT",
                "joint_pass_count": summary["joint_pass_count"],
                "behavior_pass_count": summary["behavior_pass_count"],
                "joint_worst_dB": summary["joint_worst_dB"],
                "joint_Q95_dB": summary["joint_Q95_dB"],
                "joint_median_dB": summary["joint_median_dB"],
            }
        )
    return pd.DataFrame(records, columns=columns)


def _pareto_rows(
    summaries: dict[str, dict[str, object]],
    selected_name: str | None,
) -> pd.DataFrame:
    robust = [summary for summary in summaries.values() if int(summary["joint_pass_count"]) == HARD_STATE_COUNT]
    chosen: list[dict[str, object]] = []
    if robust:
        best_worst = min(robust, key=lambda item: float(item["joint_worst_dB"]))
        best_q95 = min(robust, key=lambda item: float(item["joint_Q95_dB"]))
        smallest_k = min(robust, key=lambda item: (int(item["K"]), score_key(item)))
        for item in (best_worst, best_q95, smallest_k):
            if item["support_hash"] not in {row["support_hash"] for row in chosen}:
                chosen.append(item)
    else:
        chosen = sorted(summaries.values(), key=score_key)[:3]
    if selected_name is not None and selected_name in summaries:
        selected = summaries[selected_name]
        if selected["support_hash"] not in {row["support_hash"] for row in chosen}:
            chosen.append(selected)
    rows = []
    for item in chosen:
        rows.append(
            {
                "candidate_id": item["support_name"],
                "K": item["K"],
                "basis_ids": item["basis_ids"],
                "support_hash": item["support_hash"],
                "joint_pass_count": item["joint_pass_count"],
                "behavior_pass_count": item["behavior_pass_count"],
                "joint_worst_dB": item["joint_worst_dB"],
                "joint_Q95_dB": item["joint_Q95_dB"],
                "joint_median_dB": item["joint_median_dB"],
                "selected": item["support_name"] == selected_name,
            }
        )
    return pd.DataFrame(rows)


def _support_comparison_frame(
    terms: tuple[EnvelopeBasis, ...],
    final_ids: tuple[str, ...],
    baseline_ids: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    final_set = set(final_ids)
    final_terms = [term for term in terms if term.basis_id in final_set]

    def counts(items: Sequence[str]) -> dict[str, int]:
        selected = {str(value) for value in items}
        selected_terms = [term for term in terms if term.basis_id in selected and term.order > 1]
        return {
            **{f"p{order}_count": sum(term.order == order for term in selected_terms) for order in range(2, 10)},
            "memoryless_count": sum(
                term.signal_delay == 0 and term.envelope_delay == 0 for term in selected_terms
            ),
            "same_delay_count": sum(
                term.signal_delay == term.envelope_delay for term in selected_terms
            ),
            "cross_delay_count": sum(
                term.signal_delay != term.envelope_delay for term in selected_terms
            ),
        }

    rows = []
    for name, reference_ids in baseline_ids.items():
        reference_set = set(reference_ids)
        rows.append(
            {
                "reference": name,
                "reference_K": len(reference_ids),
                "reference_support_hash": support_hash(reference_ids),
                "final_K": len(final_ids),
                "final_support_hash": support_hash(final_ids),
                "shared_count": len(final_set & reference_set),
                "shared_terms": ";".join(term.basis_id for term in terms if term.basis_id in final_set & reference_set),
                "final_only_count": len(final_set - reference_set),
                "final_only_terms": ";".join(term.basis_id for term in terms if term.basis_id in final_set - reference_set),
                "reference_only_count": len(reference_set - final_set),
                "reference_only_terms": ";".join(term.basis_id for term in terms if term.basis_id in reference_set - final_set),
                **{f"final_{key}": value for key, value in counts(final_ids).items()},
                **{f"reference_{key}": value for key, value in counts(reference_ids).items()},
            }
        )
    if len(final_terms) < 3 or any(term.basis_id not in final_set for term in terms[:3]):
        raise RuntimeError("final support lost mandatory linear terms")
    return pd.DataFrame(rows)


def _write_final_model_definition(
    terms: tuple[EnvelopeBasis, ...],
    final_ids: tuple[str, ...],
    final_hash: str,
) -> None:
    final_set = set(final_ids)
    final_terms = [term for term in terms if term.basis_id in final_set]
    model_name = f"Envelope{len(final_ids)}-C2EndShared"
    frame = pd.DataFrame(
        [
            {
                "term_index": int(term.index),
                "basis_id": term.basis_id,
                "p": int(term.order),
                "m": int(term.signal_delay),
                "q": "" if term.envelope_delay is None else int(term.envelope_delay),
                "mandatory": bool(term.basis_id in MANDATORY_BASIS_IDS),
                "model_name": model_name,
                "K": len(final_ids),
                "dmax": DMAX,
                "lambda": 0.0,
                "solver": "OLS",
                "support_hash": final_hash,
            }
            for term in final_terms
        ]
    )
    frame.to_csv(RESULT_ROOT / "07_final_model_definition.csv", index=False)


def _write_summary(
    *,
    final_support: tuple[str, ...] | None,
    final_hash: str | None,
    selected_summary: dict[str, object] | None,
    capacity: pd.DataFrame,
    final_frame: pd.DataFrame | None,
    baseline_summaries: dict[str, dict[str, object]],
    raw_before: dict[str, object],
    raw_after: dict[str, object],
    test_unlocked: bool,
) -> None:
    lines = [
        f"Task: {TASK_NAME}",
        "Goal: select one shared sparse Envelope75 support for ILC_END and literal ILC column 2.",
        "",
        "Shared support selection successful? "
        + ("YES" if selected_summary and int(selected_summary["joint_pass_count"]) == HARD_STATE_COUNT else "NO"),
        f"Test unlocked: {test_unlocked}",
        f"Final support: {selected_summary['support_name'] if selected_summary else 'NONE'}",
        f"Final K: {selected_summary['K'] if selected_summary else 'NONE'}",
        f"Final model name: Envelope{selected_summary['K']}-C2EndShared" if selected_summary else "Final model name: NONE",
        f"dmax: {DMAX}",
        "lambda: 0",
        "solver: OLS",
        f"support hash: {final_hash or 'NONE'}",
        f"basis IDs: {selected_summary['basis_ids'] if selected_summary else 'NONE'}",
        "",
        "Full75 Train capacity (40 behavior instances)",
        f"ILC_END Train: {int(capacity.loc[capacity['behavior'] == ILC_END, 'pass_lt_minus40'].sum())}/20",
        f"ILC_COL2 Train: {int(capacity.loc[capacity['behavior'] == ILC_COL2, 'pass_lt_minus40'].sum())}/20",
        f"Combined behavior Train: {int(capacity['pass_lt_minus40'].sum())}/40",
    ]
    for behavior, label in ((ILC_END, "ILC_END"), (ILC_COL2, "ILC_COL2")):
        capacity_values = capacity.loc[
            capacity["behavior"] == behavior, "Train_NMSE_dB"
        ].to_numpy(dtype=float)
        lines.append(
            f"{label} Full75 Train summary: median={np.median(capacity_values):.9f} dB; "
            f"Q95={np.quantile(capacity_values, 0.95):.9f} dB; "
            f"worst={np.max(capacity_values):.9f} dB"
        )
    if final_frame is not None:
        for behavior, label in ((ILC_END, "ILC_END"), (ILC_COL2, "ILC_COL2")):
            part = final_frame.loc[final_frame["behavior"] == behavior]
            for metric, display in (
                ("Train_NMSE_dB", "Train"),
                ("CV_W_NMSE_dB", "CV-W"),
                ("Test_NMSE_dB", "Test"),
            ):
                values = part[metric].to_numpy(dtype=float)
                lines.append(
                    f"{label} {display}: {int(np.count_nonzero(values < TARGET_NMSE_DB))}/20; "
                    f"median={np.median(values):.9f} dB; Q95={np.quantile(values, 0.95):.9f} dB; "
                    f"worst={np.max(values):.9f} dB"
                )
        joint_final = final_frame.groupby("State_ID")["joint_final_pass"].first()
        lines.extend(
            [
                "",
                f"Joint CV pass: {int(final_frame.groupby('State_ID')['CV_pass'].all().sum())}/20",
                f"Joint final pass: {int(joint_final.sum())}/20",
                "",
                "Baseline support comparison (Train/CV only)",
            ]
        )
        for name in ("Envelope18", "Envelope23-ilcEnd", "Union24"):
            item = baseline_summaries[name]
            lines.append(
                f"{name}: K={item['K']}; joint CV={item['joint_pass_count']}/20; "
                f"behavior CV={item['behavior_pass_count']}/40; "
                f"joint worst={item['joint_worst_dB']:.9f} dB; "
                f"joint Q95={item['joint_Q95_dB']:.9f} dB; "
                f"joint median={item['joint_median_dB']:.9f} dB"
            )
    else:
        lines.append("Final Test evaluation was not unlocked because no Train/CV support reached joint 20/20.")
    if selected_summary:
        lines.extend(
            [
                "",
                f"Final joint Train/CV summary: {json.dumps(selected_summary, ensure_ascii=False, sort_keys=True)}",
            ]
        )
    lines.extend(
        [
            "",
            "Hard20 source: NMSE_withoutdpd only.",
            "Both behaviors use independent coefficients; C2 and ILC_END were never concatenated into one coefficient vector.",
            "Full-record alignment preceded fixed Train/Test split; Train/Test gains were independent per behavior.",
            "No ABC segmentation, common-B, LUT, fingerprint, Top-1, clustering, Ridge, dmax scan, DPD replay, or low-bandwidth task was run.",
            f"Raw snapshot before: {json.dumps(raw_before, ensure_ascii=False, sort_keys=True)}",
            f"Raw snapshot after: {json.dumps(raw_after, ensure_ascii=False, sort_keys=True)}",
            f"raw_data_modified: {raw_before != raw_after}",
        ]
    )
    (RESULT_ROOT / "11_final_result_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run(*, resume: bool = False) -> dict[str, object]:
    existing = RESULT_ROOT.exists() and any(RESULT_ROOT.iterdir())
    if existing and not resume:
        raise RuntimeError("result directory is not empty; use --resume for the same task")
    if resume:
        if not CHECKPOINT_PATH.is_file():
            raise RuntimeError("--resume requires an existing checkpoint")
        checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
        for key, expected in {
            "task_name": TASK_NAME,
            "state_count": HARD_STATE_COUNT,
            "candidate_dictionary": "Envelope75",
            "candidate_count": 75,
            "dmax": DMAX,
            "lambda": 0.0,
            "worker_count": WORKER_COUNT,
            "blas_threads_per_worker": BLAS_THREADS_PER_WORKER,
            "raw_manifest": EXPECTED_RAW_MANIFEST,
        }.items():
            if checkpoint.get(key) != expected:
                raise RuntimeError(f"resume checkpoint mismatch at {key}")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    raw_before = raw_manifest_gate()
    ranking, hard20 = run_hard20_ranking()
    terms, supports, hashes = load_supports()
    support_ids = {
        name: tuple(terms[index].basis_id for index in indices)
        for name, indices in supports.items()
    }
    hard_ids = tuple(hard20["State_ID"].astype(int).tolist())
    _write_task_definition_placeholder = RESULT_ROOT / "00_task_definition.txt"
    _write_task_definition_placeholder.write_text(
        "\n".join(
            [
                f"Task: {TASK_NAME}",
                "Goal: select one shared sparse Envelope75 support for ILC_END and literal ILC column 2.",
                "Behavior 1: xin_pd_ori_ilc(:, ilc_end) -> yout_withdpd_ori_ilc(:, ilc_end).",
                "Behavior 2: xin_pd_ori_ilc(:, 2) -> yout_withdpd_ori_ilc(:, 2).",
                "Full record: 24576 samples; alignment precedes Train/Test split.",
                "Train=[0,16384), Test=[16384,24576), independent Train/Test gains per behavior.",
                "Hard20 selected only by NMSE_withoutdpd; candidate Envelope75; dmax=2; lambda=0; solver=OLS.",
                "Search uses Train and Train-only contiguous CV. Test is locked until shared support freeze.",
                "No ABC, common-B, C->B/A->B, LUT, fingerprint, clustering, Ridge, DPD replay, or low-bandwidth.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_checkpoint(
        phase="hard20_and_supports_loaded",
        hard20_ids=list(hard_ids),
        support_hashes=hashes,
        selected_support_hash=None,
        selected_K=None,
        model_frozen=False,
        test_unlocked=False,
        completed_train_states=[],
        completed_test_states=[],
        union24_hash=hashes["Union24"],
        raw_manifest=raw_before,
    )
    append_log(
        WORK_LOG,
        "\n"
        f"[{_now()}] Start {TASK_NAME}\n"
        f"Hard20={list(hard_ids)}; support hashes={json.dumps(hashes, sort_keys=True)}; "
        f"raw={json.dumps(raw_before, sort_keys=True)}\n"
        "Mode: new task; dmax=2, lambda=0, OLS; dual independent behaviors; Test locked during search.\n",
    )
    context = get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=WORKER_COUNT,
        mp_context=context,
        initializer=_worker_init,
    ) as executor:
        preflight, capacity = evaluate_capacity_and_preflight(executor, hard_ids)
        end_preflight = preflight.loc[preflight["behavior"] == ILC_END].set_index("State_ID")
        col2_preflight = preflight.loc[preflight["behavior"] == ILC_COL2].set_index("State_ID")
        hard20["valid_ilc_count"] = [int(end_preflight.loc[state_id, "valid_ilc_count"]) for state_id in hard_ids]
        hard20["ilc_end_index"] = hard20["valid_ilc_count"] - 1
        hard20["ilc_col2_available"] = [bool(state_id in col2_preflight.index) for state_id in hard_ids]
        hard20["col2_equals_ilc_end"] = hard20["ilc_end_index"] == ILC_COL2_COLUMN_INDEX
        hard20.to_csv(RESULT_ROOT / "01_hard20_states.csv", index=False)
        capacity.to_csv(RESULT_ROOT / "04_full75_joint_train_capacity.csv", index=False)
        if not bool(capacity["finite"].all()) or set(capacity["rank"].astype(int)) != {75}:
            raise RuntimeError("Full75 dual-behavior numerical capacity gate failed")
        if int(capacity.loc[capacity["behavior"] == ILC_END, "pass_lt_minus40"].sum()) != 20:
            raise RuntimeError("ILC_END Full75 Train capacity is not 20/20")
        if int(capacity.loc[capacity["behavior"] == ILC_COL2, "pass_lt_minus40"].sum()) != 20:
            raise RuntimeError("ILC_COL2 Full75 Train capacity is not 20/20")

        baseline_specs = [(name, supports[name]) for name in ("Envelope18", "Envelope23-ilcEnd", "Union24")]
        baseline_summaries, baseline_frames = evaluate_support_batch(
            executor, hard_ids, baseline_specs, "BASELINE TRAIN/CV"
        )
        preflight.to_csv(RESULT_ROOT / "02_dual_behavior_data_preflight.csv", index=False)
        baseline_table = _build_baseline_table(baseline_summaries, baseline_frames)
        baseline_table.to_csv(RESULT_ROOT / "03_existing_support_baselines_train_cv.csv", index=False)
        history: list[dict[str, object]] = []
        for name in ("Envelope18", "Envelope23-ilcEnd", "Union24"):
            history.append(
                history_row(
                    baseline_summaries[name],
                    iteration=0,
                    phase="baseline",
                    operation="baseline",
                    accepted=name == "Union24",
                    reason="primary Union24 search start" if name == "Union24" else "baseline comparison",
                )
            )

        all_summaries = dict(baseline_summaries)
        support_payloads = dict(supports)
        current_name = "Union24"
        current_support = tuple(supports[current_name])
        current_summary = baseline_summaries[current_name]
        remaining = {
            term.index
            for term in terms
            if term.order > 1 and term.index not in set(current_support)
        }
        last_single: list[tuple[int, str, tuple[int, ...], dict[str, object]]] = []
        iteration = 0
        while int(current_summary["joint_pass_count"]) < HARD_STATE_COUNT and remaining:
            iteration += 1
            candidates = []
            for index in sorted(remaining):
                candidate_support = tuple(
                    term.index for term in terms if term.index in set((*current_support, index))
                )
                label = f"Forward_r{iteration}_{terms[index].basis_id}"
                candidates.append((label, candidate_support, index))
            candidate_summaries, _ = evaluate_support_batch(
                executor,
                hard_ids,
                [(label, support) for label, support, _ in candidates],
                f"FORWARD {iteration}",
            )
            all_summaries.update(candidate_summaries)
            for label, support, index in candidates:
                support_payloads[label] = support
            ordered = sorted(
                [(index, label, support, candidate_summaries[label]) for label, support, index in candidates],
                key=lambda item: score_key(item[3]),
            )
            last_single = ordered
            best_index, best_name, best_support, best_summary = ordered[0]
            for index, label, support, summary in ordered:
                history.append(
                    history_row(
                        summary,
                        iteration=iteration,
                        phase="forward",
                        operation="add",
                        added_terms=terms[index].basis_id,
                        accepted=label == best_name and objective_improves(current_summary, summary),
                        reason="best forward candidate accepted" if label == best_name and objective_improves(current_summary, summary) else "candidate rejected",
                    )
                )
            if not objective_improves(current_summary, best_summary):
                break
            current_name = best_name
            current_support = best_support
            current_summary = best_summary
            remaining.discard(best_index)

        if int(current_summary["joint_pass_count"]) < HARD_STATE_COUNT and last_single:
            top_single = last_single[:12]
            pair_specs: list[tuple[str, tuple[int, ...], tuple[int, int]]] = []
            for left_pos, left in enumerate(top_single):
                for right in top_single[left_pos + 1 :]:
                    left_index, _, _, _ = left
                    right_index, _, _, _ = right
                    pair_set = set((*current_support, left_index, right_index))
                    pair_support = tuple(term.index for term in terms if term.index in pair_set)
                    label = f"Pair_r{iteration}_{terms[left_index].basis_id}_{terms[right_index].basis_id}"
                    pair_specs.append((label, pair_support, (left_index, right_index)))
            if pair_specs:
                pair_summaries, _ = evaluate_support_batch(
                    executor,
                    hard_ids,
                    [(label, support) for label, support, _ in pair_specs],
                    "PAIR RESCUE",
                )
                all_summaries.update(pair_summaries)
                for label, support, _ in pair_specs:
                    support_payloads[label] = support
                ordered_pairs = sorted(
                    [(label, support, pair, pair_summaries[label]) for label, support, pair in pair_specs],
                    key=lambda item: score_key(item[3]),
                )
                best_label, best_support, best_pair, best_summary = ordered_pairs[0]
                history.append(
                    history_row(
                        best_summary,
                        iteration=iteration,
                        phase="pair_rescue",
                        operation="add_pair",
                        added_terms=";".join(terms[index].basis_id for index in best_pair),
                        accepted=objective_improves(current_summary, best_summary),
                        reason="bounded top-12 pair rescue accepted" if objective_improves(current_summary, best_summary) else "best bounded pair did not improve current support",
                    )
                )
                if objective_improves(current_summary, best_summary):
                    current_name = best_label
                    current_support = best_support
                    current_summary = best_summary
                    support_payloads[current_name] = current_support

        if int(current_summary["joint_pass_count"]) == HARD_STATE_COUNT:
            while True:
                removable = [index for index in current_support if terms[index].order > 1]
                if not removable:
                    break
                deletion_specs = []
                for removed in removable:
                    deletion_support = tuple(
                        term.index for term in terms if term.index in set(current_support) - {removed}
                    )
                    label = f"Backward_{current_name}_minus_{terms[removed].basis_id}"
                    deletion_specs.append((label, deletion_support, removed))
                deletion_summaries, _ = evaluate_support_batch(
                    executor,
                    hard_ids,
                    [(label, support) for label, support, _ in deletion_specs],
                    "BACKWARD CLEANUP",
                )
                all_summaries.update(deletion_summaries)
                for label, support, _ in deletion_specs:
                    support_payloads[label] = support
                allowed = [
                    (label, support, removed, deletion_summaries[label])
                    for label, support, removed in deletion_specs
                    if int(deletion_summaries[label]["joint_pass_count"]) == HARD_STATE_COUNT
                    and float(deletion_summaries[label]["joint_worst_dB"]) <= float(current_summary["joint_worst_dB"]) + 0.10
                    and float(deletion_summaries[label]["joint_Q95_dB"]) <= float(current_summary["joint_Q95_dB"]) + 0.10
                ]
                if not allowed:
                    break
                best_label, best_support, removed, best_summary = min(
                    allowed, key=lambda item: score_key(item[3])
                )
                history.append(
                    history_row(
                        best_summary,
                        iteration=iteration,
                        phase="backward",
                        operation="delete",
                        removed_terms=terms[removed].basis_id,
                        accepted=True,
                        reason="joint 20/20 preserved within 0.10 dB worst/Q95 tolerance",
                    )
                )
                current_name = best_label
                current_support = best_support
                current_summary = best_summary
                support_payloads[current_name] = current_support

        # One bounded swap pass uses the last single-add shortlist when available.
        if int(current_summary["joint_pass_count"]) == HARD_STATE_COUNT and last_single:
            add_indices = [item[0] for item in last_single[:12] if item[0] not in current_support]
            swap_specs = []
            for removed in [index for index in current_support if terms[index].order > 1]:
                for added in add_indices:
                    if added in current_support or added == removed:
                        continue
                    swap_set = (set(current_support) - {removed}) | {added}
                    swap_support = tuple(term.index for term in terms if term.index in swap_set)
                    label = f"Swap_{terms[removed].basis_id}_to_{terms[added].basis_id}"
                    swap_specs.append((label, swap_support, (removed, added)))
            if swap_specs:
                swap_summaries, _ = evaluate_support_batch(
                    executor,
                    hard_ids,
                    [(label, support) for label, support, _ in swap_specs],
                    "LOCAL SWAP",
                )
                all_summaries.update(swap_summaries)
                for label, support, _ in swap_specs:
                    support_payloads[label] = support
                best_label, best_support, pair, best_summary = min(
                    [(label, support, pair, swap_summaries[label]) for label, support, pair in swap_specs],
                    key=lambda item: score_key(item[3]),
                )
                accepted = objective_improves(current_summary, best_summary)
                history.append(
                    history_row(
                        best_summary,
                        iteration=iteration,
                        phase="local_swap",
                        operation="swap",
                        added_terms=terms[pair[1]].basis_id,
                        removed_terms=terms[pair[0]].basis_id,
                        accepted=accepted,
                        reason="bounded local swap improved joint objective" if accepted else "best bounded local swap did not improve current support",
                    )
                )
                if accepted:
                    current_name = best_label
                    current_support = best_support
                    current_summary = best_summary
                    support_payloads[current_name] = current_support

        robust = [item for item in all_summaries.values() if int(item["joint_pass_count"]) == HARD_STATE_COUNT]
        selected_summary = None
        selected_support = None
        selected_name = None
        if robust:
            best_worst = min(float(item["joint_worst_dB"]) for item in robust)
            close = [item for item in robust if float(item["joint_worst_dB"]) <= best_worst + 0.10]
            selected_summary = min(
                close,
                key=lambda item: (
                    int(item["K"]),
                    float(item["joint_worst_dB"]),
                    float(item["joint_Q95_dB"]),
                    float(item["joint_median_dB"]),
                    str(item["basis_ids"]),
                ),
            )
            selected_name = str(selected_summary["support_name"])
            selected_support = tuple(support_payloads[selected_name])
        pd.DataFrame(history).to_csv(RESULT_ROOT / "05_sparse_search_history.csv", index=False)
        pareto = _pareto_rows(all_summaries, selected_name)
        pareto.to_csv(RESULT_ROOT / "06_pareto_supports.csv", index=False)

        raw_after_search = raw_manifest_gate()
        if raw_after_search != raw_before:
            raise RuntimeError("raw manifest changed during Train-only search")

        if selected_summary is None or selected_support is None:
            _write_summary(
                final_support=None,
                final_hash=None,
                selected_summary=None,
                capacity=capacity,
                final_frame=None,
                baseline_summaries=baseline_summaries,
                raw_before=raw_before,
                raw_after=raw_after_search,
                test_unlocked=False,
            )
            _write_checkpoint(
                phase="completed_without_joint20",
                hard20_ids=list(hard_ids),
                support_hashes=hashes,
                selected_support_hash=None,
                selected_K=None,
                model_frozen=False,
                test_unlocked=False,
                completed_train_states=list(hard_ids),
                completed_test_states=[],
                union24_hash=hashes["Union24"],
                raw_manifest=raw_after_search,
            )
            result = {
                "task": TASK_NAME,
                "status": "NOT_FULLY_SUCCESSFUL",
                "joint_train_cv_pass": max(int(item["joint_pass_count"]) for item in all_summaries.values()),
                "test_unlocked": False,
                "raw_data_modified": False,
                "result_root": str(RESULT_ROOT),
            }
            return result

        final_ids = tuple(terms[index].basis_id for index in selected_support)
        final_hash = support_hash(final_ids)
        _write_final_model_definition(terms, final_ids, final_hash)
        baseline_ids = {name: support_ids[name] for name in ("Envelope18", "Envelope23-ilcEnd", "Union24")}
        _support_comparison_frame(terms, final_ids, baseline_ids).to_csv(
            RESULT_ROOT / "09_support_comparison_vs_e18_e23.csv", index=False
        )
        _write_checkpoint(
            phase="support_frozen_test_unlocked",
            hard20_ids=list(hard_ids),
            support_hashes=hashes,
            selected_support_hash=final_hash,
            selected_K=len(final_ids),
            model_frozen=True,
            test_unlocked=True,
            completed_train_states=list(hard_ids),
            completed_test_states=[],
            union24_hash=hashes["Union24"],
            raw_manifest=raw_after_search,
        )

        final_futures = {
            executor.submit(_final_worker, (int(state_id), selected_support)): int(state_id)
            for state_id in hard_ids
        }
        final_results: list[dict[str, object]] = []
        for completed, future in enumerate(as_completed(final_futures), start=1):
            final_results.append(future.result())
            if completed % 5 == 0 or completed == len(final_futures):
                _write_checkpoint(
                    phase="final_test_progress",
                    hard20_ids=list(hard_ids),
                    support_hashes=hashes,
                    selected_support_hash=final_hash,
                    selected_K=len(final_ids),
                    model_frozen=True,
                    test_unlocked=True,
                    completed_train_states=list(hard_ids),
                    completed_test_states=[int(item["State_ID"]) for item in final_results],
                    union24_hash=hashes["Union24"],
                    raw_manifest=raw_after_search,
                )
                print(f"[FINAL TEST] {completed}/{len(final_futures)}", flush=True)
        final_results.sort(key=lambda item: int(item["State_ID"]))
        final_preflight = pd.DataFrame(
            [row for item in final_results for row in item["preflight"]]
        ).sort_values(["State_ID", "behavior"]).reset_index(drop=True)
        final_preflight.to_csv(RESULT_ROOT / "02_dual_behavior_data_preflight.csv", index=False)
        hard_rank_map = {int(row.State_ID): int(row.hard_rank) for row in hard20.itertuples()}
        nodpd_map = {int(row.State_ID): float(row.NMSE_withoutdpd_dB) for row in hard20.itertuples()}
        final_rows: list[dict[str, object]] = []
        theta_end = np.empty((HARD_STATE_COUNT, len(final_ids)), dtype=np.complex128)
        theta_col2 = np.empty_like(theta_end)
        for position, item in enumerate(final_results):
            state_id = int(item["State_ID"])
            state_joint = all(
                bool(item["behaviors"][behavior]["Train_pass"])
                and bool(item["behaviors"][behavior]["Test_pass"])
                for behavior in BEHAVIORS
            )
            for behavior in BEHAVIORS:
                values = item["behaviors"][behavior]
                if behavior == ILC_END:
                    theta_end[position] = values.pop("theta")
                else:
                    theta_col2[position] = values.pop("theta")
                final_rows.append(
                    {
                        "hard_rank": hard_rank_map[state_id],
                        "State_ID": state_id,
                        "behavior": behavior,
                        "NMSE_withoutdpd_dB": nodpd_map[state_id],
                        **values,
                        "joint_final_pass": state_joint,
                    }
                )
        final_frame = pd.DataFrame(final_rows).sort_values(["hard_rank", "behavior"]).reset_index(drop=True)
        if len(final_frame) != HARD_STATE_COUNT * 2:
            raise RuntimeError("final dual-behavior table is not 40 rows")
        if not np.all(final_frame["finite"].astype(bool)):
            raise RuntimeError("final dual-behavior finite gate failed")
        np.savez(
            RESULT_ROOT / "13_final_dual_behavior_coefficients.npz",
            state_ids=np.asarray(hard_ids, dtype=np.int64),
            basis_ids=np.asarray(final_ids),
            theta_end=theta_end,
            theta_col2=theta_col2,
            K=np.asarray(len(final_ids)),
            dmax=np.asarray(DMAX),
            lambda_value=np.asarray(0.0),
            support_hash=np.asarray(final_hash),
        )
        final_frame.to_csv(RESULT_ROOT / "08_final_hard20_dual_behavior_train_cv_test.csv", index=False)
        _write_behavior_plot(
            final_frame,
            ILC_END,
            RESULT_ROOT / "10a_ilcend_final_modeling_generalization.png",
        )
        _write_behavior_plot(
            final_frame,
            ILC_COL2,
            RESULT_ROOT / "10b_ilccol2_final_modeling_generalization.png",
        )

        for state_id in (325, 356):
            direct = _final_worker((state_id, selected_support))
            batch = final_frame.loc[final_frame["State_ID"] == state_id].set_index("behavior")
            for behavior in BEHAVIORS:
                for key in ("Train_NMSE_dB", "CV_W_NMSE_dB", "Test_NMSE_dB"):
                    if abs(float(direct["behaviors"][behavior][key]) - float(batch.loc[behavior, key])) > 1e-12:
                        raise RuntimeError(f"final support spot check mismatch: state={state_id}, behavior={behavior}, key={key}")

        raw_after = raw_manifest_gate()
        if raw_after != raw_before:
            raise RuntimeError("raw manifest changed during final evaluation")
        _write_summary(
            final_support=final_ids,
            final_hash=final_hash,
            selected_summary=selected_summary,
            capacity=capacity,
            final_frame=final_frame,
            baseline_summaries=baseline_summaries,
            raw_before=raw_before,
            raw_after=raw_after,
            test_unlocked=True,
        )
        _write_checkpoint(
            phase="completed",
            hard20_ids=list(hard_ids),
            support_hashes=hashes,
            selected_support_hash=final_hash,
            selected_K=len(final_ids),
            model_frozen=True,
            test_unlocked=True,
            completed_train_states=list(hard_ids),
            completed_test_states=list(hard_ids),
            union24_hash=hashes["Union24"],
            raw_manifest=raw_after,
        )
        final_counts = {
            behavior: int(final_frame.loc[final_frame["behavior"] == behavior, "Test_pass"].sum())
            for behavior in BEHAVIORS
        }
        return {
            "task": TASK_NAME,
            "status": "SUCCESS" if bool(final_frame.groupby("State_ID")["joint_final_pass"].first().all()) else "NOT_FULLY_SUCCESSFUL",
            "selected_support": selected_summary["support_name"],
            "selected_K": len(final_ids),
            "selected_support_hash": final_hash,
            "joint_train_cv_pass": int(selected_summary["joint_pass_count"]),
            "ILC_END_test_pass": final_counts[ILC_END],
            "ILC_COL2_test_pass": final_counts[ILC_COL2],
            "joint_final_pass": int(final_frame.groupby("State_ID")["joint_final_pass"].first().sum()),
            "raw_data_modified": raw_before != raw_after,
            "result_root": str(RESULT_ROOT),
        }


def _write_checkpoint(**payload: object) -> None:
    CHECKPOINT_PATH.write_text(
        json.dumps(
            {
                "task_name": TASK_NAME,
                "state_count": HARD_STATE_COUNT,
                "behavior_count": 2,
                "candidate_dictionary": "Envelope75",
                "candidate_count": 75,
                "lambda": 0.0,
                "dmax": DMAX,
                "hard20_state_ids": payload.get("hard20_ids", []),
                "worker_count": WORKER_COUNT,
                "blas_threads_per_worker": BLAS_THREADS_PER_WORKER,
                "lut_retrieval_performed": False,
                "ridge_scan_performed": False,
                "basis_selection_mode": "shared_support_only",
                **payload,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def main() -> None:
    try:
        result = _run(resume="--resume" in sys.argv[1:])
    except Exception as exc:
        append_log(WORK_LOG, f"[{_now()}] FAILED {TASK_NAME}: {type(exc).__name__}: {exc}\n")
        raise
    append_log(
        WORK_LOG,
        f"[{_now()}] Completed {TASK_NAME}\n"
        f"Result: {json.dumps(result, ensure_ascii=False, sort_keys=True)}\n"
        "Search stopped at shared C2+ILC_END basis selection; no downstream retrieval task was run.\n",
    )
    append_log(
        HANDOFF_LOG,
        f"\n{_now()} | 新任务：{TASK_NAME}\n"
        f"完成 C2 + ilc_end 双行为统一基函数选择；结果目录：{RESULT_ROOT}\n"
        f"结果摘要：{json.dumps(result, ensure_ascii=False, sort_keys=True)}\n"
        "固定 OLS/dmax=2，仅搜索 shared Envelope75 support；Test 仅在 support freeze 后解锁，未运行 Ridge、LUT、聚类、DPD 或 low-bandwidth。\n",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    __import__("multiprocessing").freeze_support()
    main()
