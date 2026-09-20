"""Strict paired comparison: frozen Envelope18 versus frozen ilc_end Envelope23."""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
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

from behavior_fingerprint_retrieval.shared.envelope18_commonB_full_lut import (  # noqa: E402
    top1_retrieval,
)
from behavior_fingerprint_retrieval.shared.envelope23_ilcend_commonB_full_lut_comparison import (  # noqa: E402
    DMAX,
    E18_MODEL_DEFINITION,
    E18_ROOT,
    OLD_E18_FAILURE_IDS,
    RESULT_ROOT,
    STATE_COUNT,
    TASK_NAME,
    VALID_B_LENGTH,
    _load_support,
    _worker_init,
    build_common_b_probe,
    compute_cnmse_matrix,
    raw_manifest_gate,
    verify_old_baseline,
    worker_entry,
)

E23_MODEL_DEFINITION = (
    PROJECT_ROOT
    / "results"
    / "scenario_2_hard20_ilcend_envelope75_basis_selection_5B"
    / "08_final_model_definition.csv"
)
WORK_LOG = PROJECT_ROOT / "work_logs" / "behavior_fingerprint_retrieval" / "scenario_2" /TASK_NAME / "execution_log.txt"
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"
CHECKPOINT_PATH = RESULT_ROOT / "15_checkpoint.json"
THRESHOLD_DB = -40.0


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message)


def _support_hash(ids: tuple[str, ...]) -> str:
    return hashlib.sha256(json.dumps(list(ids), separators=(",", ":")).encode()).hexdigest()


def _checkpoint(
    *,
    phase: str,
    support_hash: str,
    common_sha: str,
    raw: dict[str, object],
    completed_model_state_ids: list[int],
    completed_retrieval_rows: int,
) -> None:
    CHECKPOINT_PATH.write_text(
        json.dumps(
            {
                "task_name": TASK_NAME,
                "phase": phase,
                "evaluation_only": True,
                "model_selection_performed": False,
                "ridge_scan_performed": False,
                "clustering_performed": False,
                "model_name": "Envelope23-ilcEnd",
                "baseline_model": "Envelope18",
                "K": 23,
                "lambda": 0.0,
                "dmax": DMAX,
                "support_hash": support_hash,
                "baseline_result_path": str(E18_ROOT),
                "common_B_sha256": common_sha,
                "common_B_source_state": 0,
                "state_count": STATE_COUNT,
                "completed_model_state_ids": sorted(completed_model_state_ids),
                "completed_retrieval_rows": completed_retrieval_rows,
                "abc_bounds": [0, 12_288, 17_203, 24_576],
                "allow_self_retrieval": True,
                "worker_count": 10,
                "blas_threads_per_worker": 1,
                "raw_manifest": raw,
                "model_frozen": True,
                "test_unlocked": True,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _finite_summary(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    return {
        "count": int(finite.size),
        "median_dB": float(np.median(finite)) if finite.size else float("nan"),
        "mean_dB": float(np.mean(finite)) if finite.size else float("nan"),
        "Q90_dB": float(np.quantile(finite, 0.90)) if finite.size else float("nan"),
        "Q95_dB": float(np.quantile(finite, 0.95)) if finite.size else float("nan"),
        "worst_dB": float(np.max(finite)) if finite.size else float("nan"),
    }


def _plot_six_metrics(frame: pd.DataFrame, path: Path) -> None:
    ordered = frame.sort_values("State_n_R")
    x = ordered["State_n_R"].to_numpy(dtype=int)
    specs = (
        ("nmse_withoutdpd_dB", "NMSE without DPD", "#7f7f7f"),
        ("Y_Aend_train_NMSE_dB", "E23 Y-Aend train", "#1f77b4"),
        ("Y_Aend_B_NMSE_dB", "E23 Y-Aend → B", "#2ca02c"),
        ("Y_C2_train_NMSE_dB", "E23 Y-C2 train", "#9467bd"),
        ("Y_C2_B_NMSE_dB", "E23 Y-C2 → B", "#8c564b"),
        ("retrieved_real_B_CNMSE_dB", "E23 Retrieved Real-B", "#d62728"),
    )
    values_list = []
    for column, _, _ in specs:
        values = ordered[column].to_numpy(dtype=float)
        if column == "retrieved_real_B_CNMSE_dB":
            values = np.where(np.isfinite(values), np.maximum(values, -55.0), -55.0)
        values_list.append(values)
    fig, ax = plt.subplots(figsize=(36, 12), dpi=300)
    for (column, label, color), values in zip(specs, values_list, strict=True):
        marker = "o" if column == "retrieved_real_B_CNMSE_dB" else None
        ax.plot(x, values, color=color, linewidth=0.9, marker=marker, markersize=2.4, label=label)
    ax.axhline(
        THRESHOLD_DB, color="#111111", linestyle="--", linewidth=1.0, label="-40 dB Threshold"
    )
    ax.set_xlim(0, STATE_COUNT - 1)
    ax.set_xticks(
        [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 424]
    )
    ax.set_xlabel("Load State")
    ax.set_ylabel("NMSE / CNMSE (dB)")
    ax.set_title("Frozen Envelope23-ilcEnd Full-LUT Retrieval Across 425 States")
    ax.grid(alpha=0.22)
    ax.legend(frameon=False, ncol=3, loc="best")
    ax.text(
        0.01,
        0.01,
        "Values below -55 dB are clipped for visualization only.",
        transform=ax.transAxes,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_real_b_comparison(frame: pd.DataFrame, path: Path) -> None:
    ordered = frame.sort_values("State_n_R")
    x = ordered["State_n_R"].to_numpy(dtype=int)
    fig, ax = plt.subplots(figsize=(18, 7), dpi=300)
    for column, label, color in (
        ("E18_real_B_CNMSE_dB", "Envelope18 retrieved Real-B", "#7f7f7f"),
        ("E23_real_B_CNMSE_dB", "Envelope23 retrieved Real-B", "#d62728"),
    ):
        values = ordered[column].to_numpy(dtype=float)
        plot_values = np.where(np.isfinite(values), np.maximum(values, -55.0), -55.0)
        ax.plot(x, plot_values, linewidth=1.0, marker="o", markersize=2.0, label=label, color=color)
    ax.axhline(
        THRESHOLD_DB, color="#111111", linestyle="--", linewidth=1.0, label="-40 dB Threshold"
    )
    ax.set_xlim(0, STATE_COUNT - 1)
    ax.set_xticks(
        [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 424]
    )
    ax.set_xlabel("Load State")
    ax.set_ylabel("Retrieved Real-B CNMSE (dB)")
    ax.set_title("Envelope18 versus Envelope23 Retrieved Real-B CNMSE")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    ax.text(
        0.01,
        0.01,
        "Values below -55 dB are clipped for visualization only.",
        transform=ax.transAxes,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_transitions(summary: pd.DataFrame, path: Path) -> None:
    categories = ["unchanged_pass", "recovered_by_E23", "regressed_under_E23", "unchanged_fail"]
    values = [
        int(summary.loc[summary["comparison_class"] == category, "count"].iloc[0])
        for category in categories
    ]
    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    colors = ["#2ca02c", "#1f77b4", "#d62728", "#7f7f7f"]
    ax.bar(categories, values, color=colors)
    ax.set_ylabel("State count")
    ax.set_title("Envelope18 → Envelope23 Retrieval Transitions")
    ax.grid(axis="y", alpha=0.25)
    for index, value in enumerate(values):
        ax.text(index, value + 2, str(value), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _run(*, resume: bool = False) -> dict[str, object]:
    existing_output = RESULT_ROOT.exists() and any(RESULT_ROOT.iterdir())
    if existing_output and not resume:
        raise RuntimeError("Result directory is not empty; use --resume for the same task")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    WORK_LOG.parent.mkdir(parents=True, exist_ok=True)
    raw_before = raw_manifest_gate()
    old_frame, old_common_sha = verify_old_baseline()
    terms, support_e18, e18_hash = _load_support(E18_MODEL_DEFINITION, 18)
    _, support_e23, e23_hash = _load_support(E23_MODEL_DEFINITION, 23)
    common_b, phi_e18, common_meta = build_common_b_probe()
    phi_e23 = np.asarray(common_meta.pop("phi_e23"), dtype=np.complex128)
    _write_common_b_metadata = RESULT_ROOT / "01_common_B_probe_metadata.txt"
    _write_common_b_metadata.write_text(
        "\n".join(
            [
                "Common-B comparison metadata",
                "source_state_id: 0",
                "raw_length: 4915",
                "valid_length: 4913",
                f"sha256: {common_meta['sha256']}",
                f"old_Envelope18_sha256: {old_common_sha}",
                f"Envelope18_support_hash: {e18_hash}",
                f"Envelope23_support_hash: {e23_hash}",
                "same_common_B_contract: True",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (RESULT_ROOT / "00_task_definition.txt").write_text(
        "\n".join(
            [
                f"Task: {TASK_NAME}",
                "Mode: frozen-model paired comparison; evaluation_only=true.",
                "Only variable: Envelope18 -> Frozen Envelope23-ilcEnd.",
                "Aend/C2, ABC, common-B, Full425, Top-1, Real-B, threshold are frozen from Envelope18 task.",
                f"E18 support hash: {e18_hash}",
                f"E23 support hash: {e23_hash}",
                f"Old baseline: {E18_ROOT}",
                "No search, Ridge scan, support modification, clustering, Type-III, DPD, or low-bandwidth.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _append_log(
        WORK_LOG,
        "\n"
        f"[{_now()}] Start {TASK_NAME}\n"
        "Mode: frozen-model paired comparison; only Envelope18 -> Envelope23 changes.\n"
        f"Old baseline exact/pass=170/403; common-B sha={common_meta['sha256']}; raw={json.dumps(raw_before, ensure_ascii=False, sort_keys=True)}\n"
        "Workers=10; BLAS threads/worker=1; self retrieval allowed.\n",
    )
    _checkpoint(
        phase="setup_complete",
        support_hash=e23_hash,
        common_sha=str(common_meta["sha256"]),
        raw=raw_before,
        completed_model_state_ids=[],
        completed_retrieval_rows=0,
    )
    model_results: dict[int, dict[str, object]] = {}
    context = __import__("multiprocessing").get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=10,
        mp_context=context,
        initializer=_worker_init,
        initargs=(tuple(support_e18), tuple(support_e23), phi_e18, phi_e23),
    ) as executor:
        futures = {
            executor.submit(worker_entry, state_id): state_id for state_id in range(STATE_COUNT)
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            state_id = futures[future]
            model_results[state_id] = future.result()
            if completed % 50 == 0 or completed == STATE_COUNT:
                _checkpoint(
                    phase="model_progress",
                    support_hash=e23_hash,
                    common_sha=str(common_meta["sha256"]),
                    raw=raw_before,
                    completed_model_state_ids=sorted(model_results),
                    completed_retrieval_rows=0,
                )
                print(f"[PAIRED MODEL] {completed}/{STATE_COUNT}", flush=True)
    if sorted(model_results) != list(range(STATE_COUNT)):
        raise RuntimeError("Paired Aend/C2 model evaluation did not cover 425 states")
    a18 = np.empty((STATE_COUNT, 18), dtype=np.complex128)
    c18 = np.empty_like(a18)
    a23 = np.empty((STATE_COUNT, 23), dtype=np.complex128)
    c23 = np.empty_like(a23)
    real_b = np.empty((STATE_COUNT, VALID_B_LENGTH), dtype=np.complex128)
    rows = []
    for state_id in range(STATE_COUNT):
        result = model_results[state_id]
        a18[state_id] = result["theta_Aend_E18"]
        c18[state_id] = result["theta_C2_E18"]
        a23[state_id] = result["theta_Aend_E23"]
        c23[state_id] = result["theta_C2_E23"]
        real_b[state_id] = result["real_B"]
        rows.append(
            {
                key: value
                for key, value in result.items()
                if not key.startswith("theta_") and key != "real_B"
            }
        )
    model_frame = pd.DataFrame(rows).sort_values("State_n_R").reset_index(drop=True)
    if not np.all(np.isfinite(real_b)):
        raise RuntimeError("Real-B library is non-finite")
    model_quality_rows = []
    for version, prefix, k in ((18, "E18", 18), (23, "E23", 23)):
        for metric in ("Aend_train", "Aend_B", "C2_train", "C2_B"):
            if metric in ("Aend_train", "C2_train"):
                finite_values = np.asarray(
                    [float(row[f"{prefix}_{metric}"]["train_NMSE_dB"]) for row in rows]
                )
                ranks = np.asarray([int(row[f"{prefix}_{metric}"]["rank"]) for row in rows])
                conditions = np.asarray(
                    [float(row[f"{prefix}_{metric}"]["condition_number"]) for row in rows]
                )
            else:
                finite_values = np.asarray([float(row[f"{prefix}_{metric}"]) for row in rows])
                ranks = np.full(STATE_COUNT, k, dtype=int)
                conditions = np.zeros(STATE_COUNT)
            model_quality_rows.append(
                {
                    "model": prefix,
                    "metric": metric,
                    "pass_count_lt_minus40": int(np.count_nonzero(finite_values < THRESHOLD_DB)),
                    "median_dB": float(np.median(finite_values)),
                    "Q95_dB": float(np.quantile(finite_values, 0.95)),
                    "worst_dB": float(np.max(finite_values)),
                    "rank_min": int(np.min(ranks)),
                    "condition_max": float(np.max(conditions)),
                }
            )
    # Rebuild a compact per-state model-quality table with flattened columns.
    flat_rows = []
    for row in rows:
        flat_rows.append(
            {
                "State_n_R": row["State_n_R"],
                "funMng": row["funMng"],
                "funAng": row["funAng"],
                "secMng": row["secMng"],
                "secAng": row["secAng"],
                "E18_Y_Aend_train_NMSE_dB": row["E18_Aend_train"]["train_NMSE_dB"],
                "E18_Y_Aend_B_NMSE_dB": row["E18_Aend_B"],
                "E18_Y_C2_train_NMSE_dB": row["E18_C2_train"]["train_NMSE_dB"],
                "E18_Y_C2_B_NMSE_dB": row["E18_C2_B"],
                "E23_Y_Aend_train_NMSE_dB": row["E23_Aend_train"]["train_NMSE_dB"],
                "E23_Y_Aend_B_NMSE_dB": row["E23_Aend_B"],
                "E23_Y_C2_train_NMSE_dB": row["E23_C2_train"]["train_NMSE_dB"],
                "E23_Y_C2_B_NMSE_dB": row["E23_C2_B"],
                "E18_Aend_rank": row["E18_Aend_train"]["rank"],
                "E18_C2_rank": row["E18_C2_train"]["rank"],
                "E23_Aend_rank": row["E23_Aend_train"]["rank"],
                "E23_C2_rank": row["E23_C2_train"]["rank"],
                "E18_Aend_condition": row["E18_Aend_train"]["condition_number"],
                "E18_C2_condition": row["E18_C2_train"]["condition_number"],
                "E23_Aend_condition": row["E23_Aend_train"]["condition_number"],
                "E23_C2_condition": row["E23_C2_train"]["condition_number"],
                "finite": bool(
                    row["E18_Aend_train"]["finite"]
                    and row["E18_C2_train"]["finite"]
                    and row["E23_Aend_train"]["finite"]
                    and row["E23_C2_train"]["finite"]
                ),
            }
        )
    quality_frame = pd.DataFrame(model_quality_rows)
    quality_frame.to_csv(RESULT_ROOT / "09_model_quality_comparison.csv", index=False)
    for model_name, expected_rank in (("E18", 18), ("E23", 23)):
        rank_values = quality_frame.loc[
            (quality_frame["model"] == model_name)
            & quality_frame["metric"].isin(["Aend_train", "C2_train"]),
            "rank_min",
        ]
        if not bool((rank_values == expected_rank).all()):
            raise RuntimeError(f"Paired model rank gate failed for {model_name}")
    lut18 = (phi_e18 @ a18.T).T.astype(np.complex128)
    query18 = (phi_e18 @ c18.T).T.astype(np.complex128)
    lut23 = (phi_e23 @ a23.T).T.astype(np.complex128)
    query23 = (phi_e23 @ c23.T).T.astype(np.complex128)
    d18 = compute_cnmse_matrix(query18, lut18)
    d23 = compute_cnmse_matrix(query23, lut23)
    np.save(RESULT_ROOT / "05_envelope18_fingerprint_cnmse_matrix.npy", d18)
    np.save(RESULT_ROOT / "05_envelope23_fingerprint_cnmse_matrix.npy", d23)
    q18, qd18, _, _ = top1_retrieval(d18)
    q23, qd23, _, _ = top1_retrieval(d23)
    real_distance = compute_cnmse_matrix(real_b, real_b)
    rb18 = real_distance[np.arange(STATE_COUNT), q18]
    rb23 = real_distance[np.arange(STATE_COUNT), q23]
    if not np.array_equal(q18, old_frame["State_n_Q"].to_numpy(dtype=int)):
        raise RuntimeError("Recomputed Envelope18 Top-1 differs from frozen baseline")
    old_rb = old_frame["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float)
    finite_old = np.isfinite(old_rb) & np.isfinite(rb18)
    if np.any(np.abs(old_rb[finite_old] - rb18[finite_old]) > 1e-9):
        raise RuntimeError("Recomputed Envelope18 Real-B values differ from frozen baseline")
    if int(np.isneginf(old_rb).sum()) != int(np.isneginf(rb18).sum()):
        raise RuntimeError("Recomputed Envelope18 self-hit set differs from frozen baseline")
    e23_core = pd.DataFrame(
        {
            "State_n_R": np.arange(STATE_COUNT),
            "funMng": model_frame["funMng"],
            "funAng": model_frame["funAng"],
            "secMng": model_frame["secMng"],
            "secAng": model_frame["secAng"],
            "nmse_withoutdpd_dB": model_frame["nmse_withoutdpd_dB"],
            "ACPR_withoutdpd_mean_dBc": model_frame["ACPR_withoutdpd_mean_dBc"],
            "Y_Aend_train_NMSE_dB": [row["E23_Aend_train"]["train_NMSE_dB"] for row in rows],
            "Y_Aend_B_NMSE_dB": model_frame["E23_Aend_B"],
            "Y_C2_train_NMSE_dB": [row["E23_C2_train"]["train_NMSE_dB"] for row in rows],
            "Y_C2_B_NMSE_dB": model_frame["E23_C2_B"],
            "State_n_Q": q23,
            "retrieved_real_B_CNMSE_dB": rb23,
            "retrieval_fingerprint_CNMSE_dB": qd23,
            "State_index_delta": q23 - np.arange(STATE_COUNT),
            "State_index_abs_delta": np.abs(q23 - np.arange(STATE_COUNT)),
            "is_exact_state_hit": q23 == np.arange(STATE_COUNT),
            "retrieved_real_B_pass": rb23 < THRESHOLD_DB,
        }
    )
    e23_core.to_csv(RESULT_ROOT / "03_envelope23_all425_retrieval_metrics.csv", index=False)
    old_pass = old_frame["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float) < THRESHOLD_DB
    new_pass = e23_core["retrieved_real_B_pass"].to_numpy(dtype=bool)
    transition = np.where(
        old_pass & new_pass,
        "unchanged_pass",
        np.where(
            ~old_pass & new_pass,
            "recovered_by_E23",
            np.where(old_pass & ~new_pass, "regressed_under_E23", "unchanged_fail"),
        ),
    )
    paired = pd.DataFrame(
        {
            "State_n_R": np.arange(STATE_COUNT),
            "E18_State_n_Q": q18,
            "E23_State_n_Q": q23,
            "E18_exact_hit": q18 == np.arange(STATE_COUNT),
            "E23_exact_hit": q23 == np.arange(STATE_COUNT),
            "E18_fingerprint_CNMSE_dB": qd18,
            "E23_fingerprint_CNMSE_dB": qd23,
            "E18_real_B_CNMSE_dB": rb18,
            "E23_real_B_CNMSE_dB": rb23,
            "E18_pass": old_pass,
            "E23_pass": new_pass,
            "comparison_class": transition,
        }
    )
    paired.to_csv(RESULT_ROOT / "07_envelope18_vs_envelope23_paired_comparison.csv", index=False)
    transition_summary = pd.DataFrame(
        {
            "comparison_class": [
                "unchanged_pass",
                "recovered_by_E23",
                "regressed_under_E23",
                "unchanged_fail",
            ],
            "count": [
                int(np.count_nonzero(transition == value))
                for value in (
                    "unchanged_pass",
                    "recovered_by_E23",
                    "regressed_under_E23",
                    "unchanged_fail",
                )
            ],
        }
    )
    transition_summary.to_csv(RESULT_ROOT / "08_comparison_summary.csv", index=False)
    old_failures = paired.loc[~paired["E18_pass"]].copy()
    old_failures["E23_recovered"] = old_failures["E23_pass"]
    old_failures.to_csv(
        RESULT_ROOT / "10_old_envelope18_failures_under_envelope23.csv", index=False
    )
    e23_failed = e23_core.loc[~e23_core["retrieved_real_B_pass"]].copy()
    e23_failed.to_csv(RESULT_ROOT / "06_failed_real_B_states.csv", index=False)
    np.savez(
        RESULT_ROOT / "04_envelope23_aend_c2_coefficients.npz",
        state_ids=np.arange(STATE_COUNT, dtype=np.int64),
        theta_Aend=a23,
        theta_C2=c23,
        basis_ids=np.asarray([term.basis_id for term in terms if term.index in support_e23]),
        K=np.asarray(23),
        lambda_value=np.asarray(0.0),
        dmax=np.asarray(DMAX),
        support_hash=np.asarray(e23_hash),
    )
    _write_six_metric_plot = _plot_six_metrics
    _write_six_metric_plot(e23_core, RESULT_ROOT / "11_envelope23_six_metrics.png")
    _plot_real_b_comparison(
        pd.DataFrame(
            {
                "State_n_R": np.arange(STATE_COUNT),
                "E18_real_B_CNMSE_dB": rb18,
                "E23_real_B_CNMSE_dB": rb23,
            }
        ),
        RESULT_ROOT / "12_envelope18_vs_envelope23_realB.png",
    )
    _plot_transitions(transition_summary, RESULT_ROOT / "13_retrieval_transition_summary.png")
    e23_summary = {
        "exact_count": int((q23 == np.arange(STATE_COUNT)).sum()),
        "real_B_pass_count": int(new_pass.sum()),
        "failure_count": int((~new_pass).sum()),
        "nonself_behavior_valid_count": int((new_pass & (q23 != np.arange(STATE_COUNT))).sum()),
        "fingerprint_CNMSE": _finite_summary(qd23),
        "real_B_CNMSE_finite_only": _finite_summary(rb23),
        "exact_real_B_negative_infinity_count": int(np.isneginf(rb23).sum()),
        "state_index_abs_delta_median": float(np.median(np.abs(q23 - np.arange(STATE_COUNT)))),
        "state_index_abs_delta_Q90": float(np.quantile(np.abs(q23 - np.arange(STATE_COUNT)), 0.90)),
        "state_index_abs_delta_max": int(np.max(np.abs(q23 - np.arange(STATE_COUNT)))),
    }
    quality = []
    for label, prefix in (("Envelope18", "E18"), ("Envelope23", "E23")):
        for metric in ("Aend_train", "Aend_B", "C2_train", "C2_B"):
            values = (
                np.asarray([row[f"{prefix}_{metric}"]["train_NMSE_dB"] for row in rows])
                if metric in ("Aend_train", "C2_train")
                else model_frame[f"{prefix}_{metric}"].to_numpy(dtype=float)
            )
            quality.append(
                {
                    "model": label,
                    "metric": metric,
                    "pass_count_lt_minus40": int(np.count_nonzero(values < THRESHOLD_DB)),
                    "median_dB": float(np.median(values)),
                    "Q95_dB": float(np.quantile(values, 0.95)),
                    "worst_dB": float(np.max(values)),
                }
            )
    pd.DataFrame(quality).to_csv(RESULT_ROOT / "09_model_quality_comparison.csv", index=False)
    paired_classes = transition_summary.set_index("comparison_class")["count"].to_dict()
    raw_after = raw_manifest_gate()
    summary_text = (
        "\n".join(
            [
                f"Task: {TASK_NAME}",
                "Mode: frozen-model paired comparison; evaluation_only=true; model_selection=false.",
                "Only experimental change: Envelope18 -> Frozen Envelope23-ilcEnd.",
                "Envelope18 exact/Real-B pass: 170/425, 403/425.",
                f"Envelope23 exact/Real-B pass: {e23_summary['exact_count']}/425, {e23_summary['real_B_pass_count']}/425.",
                f"Recovered: {paired_classes['recovered_by_E23']}",
                f"Regressed: {paired_classes['regressed_under_E23']}",
                f"Unchanged pass: {paired_classes['unchanged_pass']}",
                f"Unchanged fail: {paired_classes['unchanged_fail']}",
                f"Net change: {paired_classes['recovered_by_E23'] - paired_classes['regressed_under_E23']}",
                f"E23 retrieval summary: {json.dumps(e23_summary, ensure_ascii=False, sort_keys=True)}",
                f"Model-quality comparison: {json.dumps(quality, ensure_ascii=False, sort_keys=True)}",
                f"Old Envelope18 failure IDs: {sorted(OLD_E18_FAILURE_IDS)}",
                "Real-B is the primary retrieval success criterion; fingerprint CNMSE is a ranking diagnostic.",
                "No Top-k fallback, model retuning, clustering, Type-III, DPD, or low-bandwidth task was run.",
                f"Raw snapshot before: {json.dumps(raw_before, ensure_ascii=False, sort_keys=True)}",
                f"Raw snapshot after: {json.dumps(raw_after, ensure_ascii=False, sort_keys=True)}",
                f"raw_data_modified: {raw_before != raw_after}",
            ]
        )
        + "\n"
    )
    (RESULT_ROOT / "14_final_result_summary.txt").write_text(summary_text, encoding="utf-8")
    _checkpoint(
        phase="completed",
        support_hash=e23_hash,
        common_sha=str(common_meta["sha256"]),
        raw=raw_after,
        completed_model_state_ids=list(range(STATE_COUNT)),
        completed_retrieval_rows=STATE_COUNT,
    )
    return {
        "task": TASK_NAME,
        "envelope18_real_B_pass": 403,
        "envelope23_real_B_pass": int(new_pass.sum()),
        "exact_state_hit_count_E18": 170,
        "exact_state_hit_count_E23": int((q23 == np.arange(STATE_COUNT)).sum()),
        "recovered": paired_classes["recovered_by_E23"],
        "regressed": paired_classes["regressed_under_E23"],
        "net_change": paired_classes["recovered_by_E23"] - paired_classes["regressed_under_E23"],
        "failed_count": int((~new_pass).sum()),
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
        "No re-selection, Ridge scan, Top-k fallback, clustering, Type-III, DPD, or low-bandwidth task was run.\n",
    )
    _append_log(
        HANDOFF_LOG,
        f"\n{_now()} | 新任务：{TASK_NAME}\n"
        f"完成 Envelope18 vs Frozen Envelope23-ilcEnd 严格单变量 paired retrieval comparison；结果目录：{RESULT_ROOT}\n"
        f"结果摘要：{json.dumps(result, ensure_ascii=False, sort_keys=True)}\n"
        "Aend/C2、ABC、common-B、Full425、Top-1、Real-B 和 -40 dB 判据保持旧任务合同；未运行重新选模、Ridge、Top-k、聚类、Type-III、DPD 或 low-bandwidth。\n",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    __import__("multiprocessing").freeze_support()
    main()
