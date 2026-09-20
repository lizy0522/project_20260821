"""Run the frozen Envelope18 coverage evaluation over all 425 states."""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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

from behavior_modeling.shared.basis_function_selection.all425_envelope18_evaluation import (  # noqa: E402
    DMAX,
    FINAL_SUPPORT_IDS,
    RESULT_ROOT,
    STATE_COUNT,
    TARGET_NMSE_DB,
    TASK_NAME,
    WORKER_COUNT,
    _worker_entry,
    _worker_init,
    add_failure_type,
    empty_metrics_frame,
    load_type_summary,
    raw_manifest_gate,
    summarize_subset,
    task_definition_text,
    verify_frozen_support,
)

WORK_LOG = PROJECT_ROOT / "work_logs" / "behavior_modeling" / "scenario_2" /TASK_NAME / "execution_log.txt"  # noqa: E501
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"
CHECKPOINT_PATH = RESULT_ROOT / "08_checkpoint.json"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(message)


def _checkpoint(
    completed_state_ids: list[int],
    *,
    support_digest: str,
    raw_manifest: dict[str, object],
    phase: str,
    model_frozen: bool,
    test_unlocked: bool,
) -> None:
    CHECKPOINT_PATH.write_text(
        json.dumps(
            {
                "phase": phase,
                "completed_state_ids": sorted(int(value) for value in completed_state_ids),
                "support_hash": support_digest,
                "K": len(FINAL_SUPPORT_IDS),
                "lambda": 0.0,
                "dmax": DMAX,
                "train_bounds": [0, 16_384],
                "test_bounds": [16_384, 24_576],
                "worker_count": WORKER_COUNT,
                "blas_threads_per_worker": 1,
                "model_frozen": model_frozen,
                "test_unlocked": test_unlocked,
                "raw_manifest": raw_manifest,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def _validate_resume_checkpoint(support_digest: str, raw_manifest: dict[str, object]) -> None:
    if not CHECKPOINT_PATH.is_file():
        return
    checkpoint = json.loads(CHECKPOINT_PATH.read_text(encoding="utf-8"))
    if checkpoint.get("support_hash") != support_digest:
        raise RuntimeError("Resume checkpoint support hash does not match frozen Envelope18")
    if int(checkpoint.get("K", -1)) != 18 or float(checkpoint.get("lambda", -1.0)) != 0.0:
        raise RuntimeError("Resume checkpoint K/lambda does not match frozen Envelope18")
    if int(checkpoint.get("dmax", -1)) != DMAX:
        raise RuntimeError("Resume checkpoint dmax does not match frozen Envelope18")
    if checkpoint.get("train_bounds") != [0, 16_384] or checkpoint.get("test_bounds") != [
        16_384,
        24_576,
    ]:
        raise RuntimeError("Resume checkpoint Train/Test bounds changed")
    if checkpoint.get("raw_manifest") != raw_manifest:
        raise RuntimeError("Resume checkpoint raw manifest changed")


def _plot_main(frame: pd.DataFrame, path: Path) -> None:
    ordered = frame.sort_values("state_id")
    x = ordered["state_id"].to_numpy(dtype=int)
    train = ordered["modeling_NMSE_dB"].to_numpy(dtype=float)
    test = ordered["generalization_NMSE_dB"].to_numpy(dtype=float)
    lower = float(min(np.min(train), np.min(test), TARGET_NMSE_DB) - 0.5)
    upper = float(max(np.max(train), np.max(test), TARGET_NMSE_DB) + 0.5)
    fig, ax = plt.subplots(figsize=(12.0, 5.6), dpi=320)
    ax.plot(x, train, linewidth=1.0, color="#1f77b4", label="Modeling NMSE (Train)")
    ax.plot(x, test, linewidth=1.0, color="#ff7f0e", label="Generalization NMSE (Test)")
    ax.axhline(
        TARGET_NMSE_DB, color="#d62728", linestyle="--", linewidth=1.1, label="-40 dB Threshold"
    )
    ax.set_xlim(0, STATE_COUNT - 1)
    ax.set_ylim(lower, upper)
    ax.set_xticks(
        [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 424]
    )
    ax.set_xlabel("Load State")
    ax.set_ylabel("NMSE (dB)")
    ax.set_title("Envelope18 Modeling and Generalization Accuracy Across 425 Load States")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_gap(frame: pd.DataFrame, path: Path) -> None:
    ordered = frame.sort_values("state_id")
    x = ordered["state_id"].to_numpy(dtype=int)
    gap = ordered["generalization_gap_dB"].to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(12.0, 4.8), dpi=320)
    ax.plot(x, gap, linewidth=1.0, color="#2ca02c")
    ax.axhline(0.0, color="#444444", linestyle="--", linewidth=0.9)
    ax.set_xlim(0, STATE_COUNT - 1)
    ax.set_xticks(
        [0, 25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 424]
    )
    ax.set_xlabel("Load State")
    ax.set_ylabel("Test - Train NMSE (dB)")
    ax.set_title("Envelope18 Generalization Gap Across 425 Load States")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _summary_text(
    frame: pd.DataFrame,
    *,
    support_digest: str,
    raw_before: dict[str, object],
    raw_after: dict[str, object],
    numerical_gate_pass: bool,
) -> str:
    all_summary = summarize_subset(frame)
    hard20_summary = summarize_subset(frame.loc[frame["is_hard20"]])
    remaining_summary = summarize_subset(frame.loc[~frame["is_hard20"]])
    modeling_count = int(all_summary["modeling_pass_count"])
    generalization_count = int(all_summary["generalization_pass_count"])
    both_count = int(all_summary["both_pass_count"])
    if modeling_count == STATE_COUNT and generalization_count == STATE_COUNT:
        conclusion = (
            "Envelope18 achieved full 425-state modeling and generalization coverage "
            "under the -40 dB criterion."
        )
    elif modeling_count == STATE_COUNT:
        conclusion = (
            "Envelope18 achieved full modeling coverage, but generalization coverage "
            "was incomplete."
        )
    else:
        conclusion = "Envelope18 did not achieve full modeling capacity across all 425 states."
    lines = [
        f"Task: {TASK_NAME}",
        "Frozen model: K = 18; lambda = 0; dmax = 2; solver = OLS.",
        f"Support hash: {support_digest}",
        f"All425: Modeling pass (<-40 dB): {modeling_count}/{STATE_COUNT}",
        f"All425: Generalization pass (<-40 dB): {generalization_count}/{STATE_COUNT}",
        f"All425: Both pass: {both_count}/{STATE_COUNT}",
        f"Numerical gate (rank=18, finite, condition<=1e10): {numerical_gate_pass}",
        f"Conclusion: {conclusion}",
        "",
        "All425 modeling:",
        json.dumps(all_summary["modeling"], ensure_ascii=False, sort_keys=True),
        "All425 generalization:",
        json.dumps(all_summary["generalization"], ensure_ascii=False, sort_keys=True),
        "All425 gap:",
        json.dumps(all_summary["gap"], ensure_ascii=False, sort_keys=True),
        "",
        "Hard20 subset:",
        json.dumps(hard20_summary, ensure_ascii=False, sort_keys=True),
        "Remaining405 subset:",
        json.dumps(remaining_summary, ensure_ascii=False, sort_keys=True),
        "",
        f"Support IDs: {list(FINAL_SUPPORT_IDS)}",
        "Preprocessing: full-record rough/fine alignment, fixed Train/Test split, "
        "independent gains.",
        "State coefficients: independently fitted per State on Train; Test uses the "
        "Train-fitted coefficients.",
        "No re-selection, Ridge scan, lambda change, dmax/p-order expansion, or "
        "downstream LUT/DPD task was run.",
        "Workers: 10 spawned CPU processes, one BLAS thread per worker; GPU unused.",
        f"Raw snapshot before: {json.dumps(raw_before, ensure_ascii=False, sort_keys=True)}",
        f"Raw snapshot after: {json.dumps(raw_after, ensure_ascii=False, sort_keys=True)}",
        f"raw_data_modified: {raw_before != raw_after}",
    ]
    return "\n".join(lines) + "\n"


def _run(*, resume: bool = False) -> dict[str, object]:
    existing_output = any(RESULT_ROOT.iterdir())
    if existing_output and not resume:
        raise RuntimeError(
            "Result directory is not empty; use --resume only for this same incomplete task: "
            f"{RESULT_ROOT}"
        )
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    WORK_LOG.parent.mkdir(parents=True, exist_ok=True)

    raw_before = raw_manifest_gate()
    terms, support_indices, support_digest = verify_frozen_support()
    if len(support_indices) != 18:
        raise RuntimeError("Frozen Envelope18 support count is not 18")
    (RESULT_ROOT / "00_task_definition.txt").write_text(
        task_definition_text(FINAL_SUPPORT_IDS), encoding="utf-8"
    )
    _validate_resume_checkpoint(support_digest, raw_before)
    _append_log(
        WORK_LOG,
        "\n"
        f"[{_now()}] Start {TASK_NAME}\n"
        "Mode: frozen Envelope18 All-425 coverage evaluation; no model selection.\n"
        f"Raw snapshot before: {json.dumps(raw_before, ensure_ascii=False, sort_keys=True)}\n"
        f"Support hash: {support_digest}; workers={WORKER_COUNT}; BLAS threads/worker=1.\n",
    )

    partial_path = RESULT_ROOT / "01_all425_envelope18_metrics.csv"
    if resume and partial_path.is_file():
        partial = pd.read_csv(partial_path)
        if partial.empty:
            frame = empty_metrics_frame()
        else:
            frame = partial
        completed_state_ids = set(int(value) for value in frame["state_id"].tolist())
    else:
        frame = empty_metrics_frame()
        completed_state_ids = set()
    expected_ids = set(range(STATE_COUNT))
    if not completed_state_ids <= expected_ids:
        raise RuntimeError("Partial metrics contain an invalid state_id")

    pending = sorted(expected_ids - completed_state_ids)
    context = __import__("multiprocessing").get_context("spawn")
    rows = frame.to_dict("records")
    with ProcessPoolExecutor(
        max_workers=WORKER_COUNT,
        mp_context=context,
        initializer=_worker_init,
        initargs=(tuple(support_indices),),
    ) as executor:
        futures = {executor.submit(_worker_entry, state_id): state_id for state_id in pending}
        for completed, future in enumerate(as_completed(futures), start=1):
            rows.append(future.result())
            total_completed = len(completed_state_ids) + completed
            if total_completed % 50 == 0 or total_completed == STATE_COUNT:
                partial_frame = pd.DataFrame(rows).sort_values("state_id").reset_index(drop=True)
                partial_frame.to_csv(partial_path, index=False)
                _checkpoint(
                    partial_frame["state_id"].astype(int).tolist(),
                    support_digest=support_digest,
                    raw_manifest=raw_before,
                    phase="coverage_progress",
                    model_frozen=True,
                    test_unlocked=True,
                )
                print(f"[ALL425] {total_completed}/{STATE_COUNT}", flush=True)

    frame = pd.DataFrame(rows).sort_values("state_id").reset_index(drop=True)
    if set(frame["state_id"].astype(int)) != expected_ids or len(frame) != STATE_COUNT:
        raise RuntimeError(f"All-425 evaluation returned {len(frame)} rows with incomplete IDs")
    if not frame["state_id"].astype(int).to_numpy().tolist() == list(range(STATE_COUNT)):
        raise RuntimeError("All-425 results are not sorted in canonical State ID order")
    if not bool(frame["numerical_gate_pass"].astype(bool).all()):
        frame.to_csv(partial_path, index=False)
        raise RuntimeError(
            "At least one State failed the formal rank/finite/condition numerical gate"
        )
    frame.to_csv(partial_path, index=False)

    with_failure_type = add_failure_type(frame)
    failed = with_failure_type.loc[with_failure_type["failure_type"] != ""].copy()
    failed.to_csv(RESULT_ROOT / "02_failed_states.csv", index=False)
    load_type_frame = load_type_summary(frame)
    load_type_frame.to_csv(RESULT_ROOT / "03_metrics_by_load_type.csv", index=False)
    _plot_main(frame, RESULT_ROOT / "04_all425_modeling_generalization_nmse.png")
    _plot_gap(frame, RESULT_ROOT / "05_generalization_gap.png")

    raw_after = raw_manifest_gate()
    numerical_gate_pass = bool(frame["numerical_gate_pass"].astype(bool).all())
    summary = _summary_text(
        frame,
        support_digest=support_digest,
        raw_before=raw_before,
        raw_after=raw_after,
        numerical_gate_pass=numerical_gate_pass,
    )
    (RESULT_ROOT / "06_final_result_summary.txt").write_text(summary, encoding="utf-8")
    _checkpoint(
        frame["state_id"].astype(int).tolist(),
        support_digest=support_digest,
        raw_manifest=raw_after,
        phase="completed",
        model_frozen=True,
        test_unlocked=True,
    )
    return {
        "task": TASK_NAME,
        "modeling_pass_count": int(frame["modeling_pass"].sum()),
        "generalization_pass_count": int(frame["generalization_pass"].sum()),
        "both_pass_count": int(frame["both_pass"].sum()),
        "failed_count": int(len(failed)),
        "raw_data_modified": raw_before != raw_after,
        "support_hash": support_digest,
        "result_root": str(RESULT_ROOT),
    }


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
        "No re-selection, Ridge scan, dmax/p-order expansion, LUT retrieval, "
        "clustering, DPD, or low-bandwidth task was run.\n",
    )
    _append_log(
        HANDOFF_LOG,
        f"\n{_now()} | 新任务：{TASK_NAME}\n"
        f"完成冻结 Envelope18 的 All-425 coverage evaluation；结果目录：{RESULT_ROOT}\n"
        f"结果摘要：{json.dumps(result, ensure_ascii=False, sort_keys=True)}\n"
        "使用 10 个 spawn worker、每 worker 1 个 BLAS 线程；不重新选择 support、lambda 或 dmax；"
        "Test 只使用 Train 拟合系数；未运行 LUT retrieval、clustering、DPD 或 low-bandwidth。\n",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    __import__("multiprocessing").freeze_support()
    main()
