# ruff: noqa: E402,E501,I001

"""Run historical MP10 through the current E19 Full425 common-B protocol."""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path
from typing import Any

# Keep one numerical thread in each of the ten state workers.
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

from behavior_fingerprint_retrieval.shared.envelope18_commonB_full_lut import top1_retrieval  # noqa: E402
from behavior_fingerprint_retrieval.shared.envelope19_retrieval_runner import (  # noqa: E402
    _format_quality,
    _write_excel,
    _write_plot,
)
from behavior_fingerprint_retrieval.shared.envelope19_c2endshared_commonB_full_lut import (  # noqa: E402
    EXPECTED_COMMON_B_SHA,
    compute_cnmse_matrix,
    raw_manifest_gate,
    verify_common_b_contract,
)

from behavior_fingerprint_retrieval.scenario_2.scenario_2_mp10_full_lut_retrieval_5b import mp10_c2endshared_commonB_full_lut as mp10  # noqa: E402

TASK_NAME = "scenario_2_mp10_full_lut_retrieval_5b"
RESULT_ROOT = PROJECT_ROOT / "results" / "behavior_fingerprint_retrieval" / "scenario_2" /TASK_NAME
WORK_LOG = PROJECT_ROOT / "work_logs" / "behavior_fingerprint_retrieval" / "scenario_2" /TASK_NAME / "execution_log.txt"
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "core" / "codex_handoff" / "codex_handoff.txt"
E19_ROOT = PROJECT_ROOT / "results" / "behavior_fingerprint_retrieval" / "scenario_2" /"scenario_2_envelope19_c2endshared_commonB_full_lut_retrieval_5B"
E19_METRICS = E19_ROOT / "06_retrieval_results_all425.csv"
E19_MODEL_METRICS = E19_ROOT / "03_all425_model_quality.csv"
STATE_COUNT = mp10.STATE_COUNT
WORKER_COUNT = 10
BLAS_THREADS_PER_WORKER = 1
THRESHOLD_DB = -40.0
DRY_RUN_IDS = (0, 187, 325, 424)
CHECKPOINT_PATH = RESULT_ROOT / "17_checkpoint.json"


def _now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def _append_log(message: str) -> None:
    WORK_LOG.parent.mkdir(parents=True, exist_ok=True)
    with WORK_LOG.open("a", encoding="utf-8") as handle:
        handle.write(message)


def _append_handoff(message: str) -> None:
    HANDOFF_LOG.parent.mkdir(parents=True, exist_ok=True)
    with HANDOFF_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{_now()} | 新任务：{TASK_NAME}\n{message.rstrip()}\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot JSON serialize {type(value)!r}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _checkpoint(**payload: Any) -> None:
    _write_json(
        CHECKPOINT_PATH,
        {
            "task_name": TASK_NAME,
            "mode": "new_task",
            "model_name": mp10.MODEL_NAME,
            "state_count": STATE_COUNT,
            "workers": WORKER_COUNT,
            "blas_threads_per_worker": BLAS_THREADS_PER_WORKER,
            "abc_bounds": [0, 12_288, 17_203, 24_576],
            "allow_self_retrieval": True,
            "common_B_source_state": 0,
            "common_B_sha256": EXPECTED_COMMON_B_SHA,
            "retrieval_protocol_source": "scenario_2_envelope19_c2endshared_commonB_full_lut_retrieval_5B",
            "basis_selection_performed": False,
            "model_selection_performed": False,
            "ridge_scan_performed": False,
            "dmax_scan_performed": False,
            "top_k_or_fusion_performed": False,
            "lut_retrieval_performed": True,
            **payload,
        },
    )


def _write_audit(common_meta: dict[str, Any]) -> None:
    contract = mp10.historical_contract()
    lines = [
        f"Task: {TASK_NAME}",
        "Purpose: reuse the historical MP10 backend inside the current E19 Full425 protocol.",
        "",
        "Historical MP10 source files:",
        "- scripts/behavior_modeling/config.py: frozen MP_CONFIG, K=10, dmax=2.",
        "- scripts/behavior_modeling/basis.py: canonical build_mp_basis implementation.",
        "- scripts/behavior_modeling/sparse_gmp.py: historical build_frozen_mp_basis wrapper and fit_ridge path.",
        "- scripts/behavior_fingerprint_retrieval/run_scenario2_C2_to_Aend_type3_cluster_compressed_5b.py: historical MP10 Aend/C2 fitting call chain.",
        "",
        "Historical MP10 functions:",
        "- feature builder: behavior_modeling.sparse_gmp.build_frozen_mp_basis.",
        "- underlying feature builder: behavior_modeling.basis.build_mp_basis.",
        "- Ridge solver: behavior_modeling.sparse_gmp.fit_ridge.",
        "- historical call: _fit_frozen_side -> build_frozen_mp_basis -> fit_ridge -> Phi @ theta.",
        "",
        "Historical MP10 contract:",
        f"- orders={contract['orders']}; memory={contract['memory']}; K={contract['K']}; dmax={contract['dmax']}; lambda={contract['ridge_lambda']}.",
        "- basis expression: x[n-m] * abs(x[n-m]) ** (p-1), with order-major columns; p=2 is included.",
        f"- basis normalization: {contract['basis_normalization']}.",
        f"- ILC normalization: {contract['ilc_input_normalization']}.",
        f"- Ridge convention: {contract['ridge_solver']}; all K coefficients are penalized through sqrt(N*lambda) I_K.",
        "- separate Hnorm function/step: not found in the current Mac checkout; no Hnorm was invented or added.",
        "- State0 direct comparison: sparse_gmp and behavior_modeling.ridge produce exactly equal Phi and theta at lambda=1e-8.",
        "",
        "Current E19 Full-LUT protocol source files:",
        "- scripts/behavior_fingerprint_retrieval/envelope19_c2endshared_commonB_full_lut.py: current state worker contract/common-B gate/CNMSE helper.",
        "- scripts/behavior_fingerprint_retrieval/run_scenario2_envelope19_c2endshared_commonB_full_lut_retrieval_5b.py: current Full425 orchestration, Excel writer and plotter.",
        "- behavior_fingerprint_retrieval.envelope18_commonB_full_lut.top1_retrieval: current stable Top-1 and tie rule.",
        "",
        "Reused modules and semantics:",
        "- data_management: current State table and raw MAT loading.",
        "- signal_segmentation: current canonical A/B/C, final-ILC Aend, ILC2 C2, and OFF Real-B.",
        "- E19 common-B: canonical State 0 B probe, raw length 4915, valid length 4913, frozen hash.",
        "- E19/CNMSE: existing vectorized 425x425 CNMSE engine.",
        "- E19 Top-1: existing allow-self retrieval with stable smallest-state tie handling.",
        "- E19 writer/plotter: existing ten-column Excel layout and six-metric plotter, with only the title parameterized for this model.",
        "",
        "Protocol freeze:",
        "- LUT fingerprint = MP10 Y-Aend(final ILC) response on the shared common-B probe.",
        "- Online query fingerprint = MP10 Y-C2(ILC2) response on the same common-B probe.",
        "- State_Q is selected from fingerprint CNMSE before Real-B evaluation.",
        "- Real-B is canonical OFF yout_withoutdpd_ori B and is evaluation-only.",
        "- No clustering, compressed LUT, Type-III, Top-k, ensemble, fusion, DPD replay, bandwidth scan, order scan, memory scan, or lambda scan.",
        "",
        "Historical numerical result availability:",
        "- The old Windows MP10/Type-III result directories are not present in the current Mac checkout; this run therefore records source-level reuse and fresh representative-state regression rather than claiming an old artifact comparison.",
        "",
        f"Common-B metadata: {json.dumps(common_meta, ensure_ascii=False, sort_keys=True, default=_json_default)}",
        "Newly added code: one task-local MP10 protocol adapter, one runner, one direct-contract test; no existing retrieval core was copied.",
    ]
    (RESULT_ROOT / "01_reuse_audit.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_dry_run(common_b: np.ndarray) -> pd.DataFrame:
    mp10._worker_init(common_b)
    rows: list[dict[str, Any]] = []
    for state_id in DRY_RUN_IDS:
        item = mp10._state_worker(state_id)
        for role, segment in (("Aend", "A"), ("C2", "C")):
            train_shape = {
                "Aend": (12_286, mp10.MP_K),
                "C2": (7_371, mp10.MP_K),
            }[role]
            rows.append(
                {
                    "State_n_R": int(state_id),
                    "role": role,
                    "train_segment": segment,
                    "Phi_shape": str(train_shape),
                    "Phi_B_shape": str((mp10.FINGERPRINT_LENGTH, mp10.MP_K)),
                    "theta_shape": str(np.asarray(item["theta_Aend" if role == "Aend" else "theta_C2"]).shape),
                    "train_NMSE_dB": float(item[f"Y_{role}_train_NMSE_dB"]),
                    "B_NMSE_dB": float(item[f"Y_{role}_B_NMSE_dB"]),
                    "rank": int(item[f"Y_{role}_rank"]),
                    "rank_augmented": int(item[f"Y_{role}_rank_augmented"]),
                    "lambda": mp10.RIDGE_LAMBDA,
                    "finite": bool(item[f"Y_{role}_finite"]),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.shape != (8, 12) or not frame["finite"].all():
        raise RuntimeError("MP10 representative-state dry-run contract failed")
    if set(frame["Phi_shape"]) != {"(12286, 10)", "(7371, 10)"}:
        raise RuntimeError("MP10 dry-run Phi shapes failed")
    if set(frame["Phi_B_shape"]) != {"(4913, 10)"} or set(frame["theta_shape"]) != {"(10,)"}:
        raise RuntimeError("MP10 dry-run B/theta shapes failed")
    frame.to_csv(RESULT_ROOT / "03_preflight_dry_run.csv", index=False)
    return frame


def _run_all425(common_b: np.ndarray) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    results: list[dict[str, Any]] = []
    context = get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=WORKER_COUNT,
        mp_context=context,
        initializer=mp10._worker_init,
        initargs=(np.asarray(common_b, dtype=np.complex128),),
    ) as executor:
        futures = {executor.submit(mp10.model_worker_entry, state_id): state_id for state_id in range(STATE_COUNT)}
        for completed, future in enumerate(as_completed(futures), start=1):
            results.append(future.result())
            if completed % 50 == 0 or completed == STATE_COUNT:
                _checkpoint(
                    phase="model_progress",
                    completed_model_states=completed,
                    completed_model_state_ids=sorted(int(item["State_n_R"]) for item in results),
                )
                print(f"[MP10 MODEL] {completed}/{STATE_COUNT}", flush=True)
    results.sort(key=lambda item: int(item["State_n_R"]))
    if [int(item["State_n_R"]) for item in results] != list(range(STATE_COUNT)):
        raise RuntimeError("MP10 All425 state order is not 0...424")
    model_frame = pd.DataFrame(
        [
            {key: value for key, value in item.items() if key not in {"theta_Aend", "theta_C2", "real_B_valid"}}
            for item in results
        ]
    )
    theta_a = np.stack([item["theta_Aend"] for item in results], axis=0).astype(np.complex128)
    theta_c = np.stack([item["theta_C2"] for item in results], axis=0).astype(np.complex128)
    real_b = np.stack([item["real_B_valid"] for item in results], axis=0).astype(np.complex128)
    if theta_a.shape != (STATE_COUNT, mp10.MP_K) or theta_c.shape != (STATE_COUNT, mp10.MP_K):
        raise RuntimeError("MP10 All425 coefficient shape failed")
    if real_b.shape != (STATE_COUNT, mp10.FINGERPRINT_LENGTH) or not np.all(np.isfinite(real_b)):
        raise RuntimeError("MP10 All425 Real-B shape/finite failed")
    model_frame.to_csv(RESULT_ROOT / "04_all425_model_quality.csv", index=False)
    return model_frame, theta_a, theta_c, real_b


def _build_paired_comparison(metrics: pd.DataFrame) -> pd.DataFrame:
    if not E19_METRICS.is_file() or not E19_MODEL_METRICS.is_file():
        raise FileNotFoundError(f"E19 reference metrics missing under {E19_ROOT}")
    e19 = pd.read_csv(E19_METRICS).sort_values("State_n_R").reset_index(drop=True)
    e19_model = pd.read_csv(E19_MODEL_METRICS).sort_values("State_n_R").reset_index(drop=True)
    if len(e19) != STATE_COUNT or len(e19_model) != STATE_COUNT:
        raise RuntimeError("E19 reference metrics are not canonical 425 rows")
    if not np.array_equal(e19["State_n_R"].to_numpy(dtype=int), np.arange(STATE_COUNT)):
        raise RuntimeError("E19 State_R order changed")
    transition = np.where(
        e19["retrieved_real_B_pass"].to_numpy(dtype=bool) & metrics["retrieved_real_B_pass"].to_numpy(dtype=bool),
        "unchanged_pass",
        np.where(
            ~e19["retrieved_real_B_pass"].to_numpy(dtype=bool) & metrics["retrieved_real_B_pass"].to_numpy(dtype=bool),
            "E19_fail_to_MP10_pass",
            np.where(
                e19["retrieved_real_B_pass"].to_numpy(dtype=bool) & ~metrics["retrieved_real_B_pass"].to_numpy(dtype=bool),
                "E19_pass_to_MP10_fail",
                "unchanged_fail",
            ),
        ),
    )
    paired = pd.DataFrame(
        {
            "State_R": np.arange(STATE_COUNT),
            "E19_State_Q": e19["State_n_Q"].to_numpy(dtype=int),
            "MP10_State_Q": metrics["State_n_Q"].to_numpy(dtype=int),
            "E19_fingerprint_CNMSE_dB": e19["retrieval_fingerprint_CNMSE_dB"].to_numpy(dtype=float),
            "MP10_fingerprint_CNMSE_dB": metrics["retrieval_fingerprint_CNMSE_dB"].to_numpy(dtype=float),
            "E19_real_B_CNMSE_dB": e19["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float),
            "MP10_real_B_CNMSE_dB": metrics["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float),
            "E19_pass": e19["retrieved_real_B_pass"].to_numpy(dtype=bool),
            "MP10_pass": metrics["retrieved_real_B_pass"].to_numpy(dtype=bool),
            "transition": transition,
        }
    )
    paired.to_csv(RESULT_ROOT / "12_mp10_vs_e19_paired_comparison.csv", index=False)
    quality_rows = _format_quality(metrics, mp10.MODEL_NAME, "MP10")
    quality_rows.extend(_format_quality(e19_model, "Envelope19-C2EndShared", "E19"))
    pd.DataFrame(quality_rows).to_csv(RESULT_ROOT / "13_model_quality_comparison.csv", index=False)
    return paired


def _run(*, resume: bool = False) -> dict[str, Any]:
    if RESULT_ROOT.exists() and any(RESULT_ROOT.iterdir()) and not resume:
        raise RuntimeError(f"result directory is not empty; use --resume: {RESULT_ROOT}")
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    raw_before = raw_manifest_gate()
    contract = mp10.historical_contract()
    common_b, common_meta = verify_common_b_contract()
    common_b = np.asarray(common_b, dtype=np.complex128)
    common_meta_clean = {
        key: value for key, value in common_meta.items() if not isinstance(value, np.ndarray)
    }
    if common_meta["sha256"] != EXPECTED_COMMON_B_SHA:
        raise RuntimeError("common-B hash regression failed")
    backend_meta = mp10.verify_backend_contract(common_b)
    _write_audit({**common_meta_clean, "backend": backend_meta})
    (RESULT_ROOT / "00_task_definition.txt").write_text(
        "\n".join(
            [
                f"Task: {TASK_NAME}",
                "Mode: new_task; historical MP10 backend plus current E19 Full425 common-B retrieval protocol.",
                f"Model: {mp10.MODEL_NAME}; orders={contract['orders']}; memory={contract['memory']}; K={contract['K']}; dmax={contract['dmax']}; lambda={contract['ridge_lambda']}.",
                "LUT: final-ILC Aend model response on common-B; Query: ILC2 C2 model response on common-B.",
                "ABC: A=[0,12288), B=[12288,17203), C=[17203,24576).",
                "Top-1 allow-self; State_Q is frozen before canonical OFF Real-B evaluation.",
                "No model optimization, order/memory/lambda scan, clustering, Top-k, ensemble, fusion, DPD replay, or low-bandwidth processing.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    _write_json(
        RESULT_ROOT / "02_protocol_contract.json",
        {
            "task_name": TASK_NAME,
            "model": contract,
            "abc_bounds": [0, 12_288, 17_203, 24_576],
            "state_count": STATE_COUNT,
            "allow_self": True,
            "common_B_source_state": 0,
            "common_B_raw_length": int(common_b.size),
            "common_B_valid_length": mp10.FINGERPRINT_LENGTH,
            "common_B_sha256": EXPECTED_COMMON_B_SHA,
            "common_B_metadata": common_meta_clean,
            "retrieval_source": "E19 Full-LUT pipeline",
            "real_B_selection_used": False,
        },
    )
    _checkpoint(
        phase="setup_complete",
        contract=contract,
        raw_manifest_before=raw_before,
        common_B_hash=EXPECTED_COMMON_B_SHA,
        completed_model_state_ids=[],
        completed_retrieval_rows=0,
    )
    _append_log(
        f"\n[{_now()}] Start {TASK_NAME}\n"
        f"Historical MP10 contract={json.dumps(contract, ensure_ascii=False, sort_keys=True)}\n"
        f"E19 common-B metadata={json.dumps(common_meta_clean, ensure_ascii=False, sort_keys=True, default=_json_default)}\n"
        f"Raw manifest before={json.dumps(raw_before, ensure_ascii=False, sort_keys=True)}\n"
        "Reuse audit written before All425; separate Hnorm function was not found, so none was added.\n"
    )
    _run_dry_run(common_b)
    _checkpoint(phase="dry_run_complete", dry_run_ids=list(DRY_RUN_IDS), completed_model_state_ids=[])
    model_frame, theta_a, theta_c, real_b = _run_all425(common_b)
    np.savez(
        RESULT_ROOT / "05_mp10_aend_c2_coefficients.npz",
        state_ids=np.arange(STATE_COUNT, dtype=np.int64),
        theta_aend=theta_a,
        theta_c2=theta_c,
        orders=np.asarray(mp10.MP_ORDERS, dtype=np.int64),
        memory=np.asarray([mp10.MP_MEMORY[order] for order in mp10.MP_ORDERS], dtype=np.int64),
        K=np.asarray(mp10.MP_K),
        dmax=np.asarray(mp10.MP_DMAX),
        lambda_value=np.asarray(mp10.RIDGE_LAMBDA),
        common_B_sha256=np.asarray(EXPECTED_COMMON_B_SHA),
    )
    common_phi = mp10.historical_mp10.build_frozen_mp_basis(common_b)
    lut_fingerprints = (common_phi @ theta_a.T).T.astype(np.complex128, copy=False)
    query_fingerprints = (common_phi @ theta_c.T).T.astype(np.complex128, copy=False)
    if lut_fingerprints.shape != (STATE_COUNT, mp10.FINGERPRINT_LENGTH) or query_fingerprints.shape != (STATE_COUNT, mp10.FINGERPRINT_LENGTH):
        raise RuntimeError("MP10 LUT/query fingerprint shape failed")
    if not np.all(np.isfinite(lut_fingerprints)) or not np.all(np.isfinite(query_fingerprints)):
        raise RuntimeError("MP10 LUT/query fingerprints are not finite")
    np.save(RESULT_ROOT / "06_mp10_lut_fingerprints.npy", lut_fingerprints)
    np.save(RESULT_ROOT / "07_mp10_query_fingerprints.npy", query_fingerprints)
    distance = compute_cnmse_matrix(query_fingerprints, lut_fingerprints)
    if distance.shape != (STATE_COUNT, STATE_COUNT) or np.isnan(distance).any() or np.isposinf(distance).any():
        raise RuntimeError("MP10 fingerprint CNMSE matrix contract failed")
    np.save(RESULT_ROOT / "08_mp10_fingerprint_cnmse_matrix.npy", distance)
    _checkpoint(
        phase="fingerprints_complete",
        completed_model_state_ids=list(range(STATE_COUNT)),
        completed_retrieval_rows=0,
        coefficients_completed=True,
        fingerprints_completed=True,
    )
    selected, selected_distance, true_rank, tie_count = top1_retrieval(distance)
    real_distance = compute_cnmse_matrix(real_b, real_b)
    retrieved_real_b = real_distance[np.arange(STATE_COUNT), selected]
    metrics = model_frame.copy()
    metrics["State_n_Q"] = selected.astype(int)
    metrics["retrieval_fingerprint_CNMSE_dB"] = selected_distance
    metrics["retrieved_real_B_CNMSE_dB"] = retrieved_real_b
    metrics["true_state_rank"] = true_rank.astype(int)
    metrics["minimum_tie_count"] = tie_count.astype(int)
    metrics["State_index_delta"] = selected.astype(int) - np.arange(STATE_COUNT)
    metrics["State_index_abs_delta"] = np.abs(metrics["State_index_delta"].to_numpy(dtype=int))
    metrics["is_exact_state_hit"] = selected == np.arange(STATE_COUNT)
    metrics["retrieved_real_B_pass"] = retrieved_real_b < THRESHOLD_DB
    metrics.to_csv(RESULT_ROOT / "09_retrieval_results_all425.csv", index=False)
    metrics[
        [
            "State_n_R",
            "State_n_Q",
            "retrieval_fingerprint_CNMSE_dB",
            "true_state_rank",
            "minimum_tie_count",
            "State_index_delta",
            "State_index_abs_delta",
            "is_exact_state_hit",
            "retrieved_real_B_CNMSE_dB",
            "retrieved_real_B_pass",
            "Y_Aend_train_NMSE_dB",
            "Y_Aend_B_NMSE_dB",
            "Y_C2_train_NMSE_dB",
            "Y_C2_B_NMSE_dB",
        ]
    ].to_csv(RESULT_ROOT / "10_retrieval_diagnostics_all425.csv", index=False)
    metrics.loc[~metrics["retrieved_real_B_pass"]].to_csv(RESULT_ROOT / "11_failed_states.csv", index=False)
    paired = _build_paired_comparison(metrics)
    _write_excel(metrics, RESULT_ROOT / "scenario_2_mp10_full_lut_state_summary.xlsx")
    _write_plot(
        metrics,
        RESULT_ROOT / "scenario_2_mp10_full_lut_retrieval.png",
        title="Historical Frozen MP10 Full425 common-B LUT retrieval",
    )
    transitions = paired["transition"].value_counts().to_dict()
    result_summary = {
        "exact_count": int(metrics["is_exact_state_hit"].sum()),
        "real_B_pass_count": int(metrics["retrieved_real_B_pass"].sum()),
        "failure_count": int((~metrics["retrieved_real_B_pass"]).sum()),
        "nonself_count": int((metrics["State_n_Q"] != np.arange(STATE_COUNT)).sum()),
        "nonself_pass_count": int(((metrics["State_n_Q"] != np.arange(STATE_COUNT)) & metrics["retrieved_real_B_pass"]).sum()),
        "fingerprint_CNMSE_median_finite_dB": float(np.median(selected_distance[np.isfinite(selected_distance)])),
        "retrieved_real_B_median_finite_dB": float(np.median(retrieved_real_b[np.isfinite(retrieved_real_b)])),
        "retrieved_real_B_worst_finite_dB": float(np.max(retrieved_real_b[np.isfinite(retrieved_real_b)])),
        "exact_real_B_negative_infinity_count": int(np.isneginf(retrieved_real_b).sum()),
        "state_index_abs_delta_median": float(np.median(metrics["State_index_abs_delta"])),
        "state_index_abs_delta_Q90": float(np.quantile(metrics["State_index_abs_delta"], 0.90)),
        "state_index_abs_delta_max": int(np.max(metrics["State_index_abs_delta"])),
        "transitions_vs_E19": {key: int(transitions.get(key, 0)) for key in ("unchanged_pass", "E19_fail_to_MP10_pass", "E19_pass_to_MP10_fail", "unchanged_fail")},
        "recovered_state_ids": paired.loc[paired["transition"] == "E19_fail_to_MP10_pass", "State_R"].astype(int).tolist(),
        "regressed_state_ids": paired.loc[paired["transition"] == "E19_pass_to_MP10_fail", "State_R"].astype(int).tolist(),
        "raw_manifest_before": raw_before,
        "raw_manifest_after": raw_manifest_gate(),
    }
    _write_json(RESULT_ROOT / "16_final_result_summary.json", result_summary)
    (RESULT_ROOT / "16_final_result_summary.txt").write_text(
        "\n".join(
            [
                f"Task: {TASK_NAME}",
                "Historical Frozen MP10 backend + E19 Full425 common-B LUT retrieval.",
                f"Contract: orders={contract['orders']}; memory={contract['memory']}; K={contract['K']}; dmax={contract['dmax']}; lambda={contract['ridge_lambda']}.",
                "ABC bounds: A=[0,12288), B=[12288,17203), C=[17203,24576).",
                "LUT fingerprint=final-ILC Aend response on common-B; query=ILC2 C2 response on common-B.",
                f"MP10 exact hit: {result_summary['exact_count']}/425.",
                f"MP10 Real-B strict pass (<-40 dB): {result_summary['real_B_pass_count']}/425; failures={result_summary['failure_count']}.",
                f"E19 transitions: {json.dumps(result_summary['transitions_vs_E19'], ensure_ascii=False, sort_keys=True)}.",
                f"Recovered states: {result_summary['recovered_state_ids']}",
                f"Regressed states: {result_summary['regressed_state_ids']}",
                "State_Q was frozen before Real-B evaluation. Real-B was not used for Top-1 selection.",
                "No Hnorm step was found in the current historical source; no new normalization was introduced beyond get_ilc_pair peak normalization.",
                "No model optimization, order/memory/lambda scan, clustering, Top-k, ensemble, fusion, DPD replay, or low-bandwidth task was run.",
                f"Raw data unchanged: {result_summary['raw_manifest_before'] == result_summary['raw_manifest_after']}.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    raw_after = raw_manifest_gate()
    _checkpoint(
        phase="completed",
        completed_model_state_ids=list(range(STATE_COUNT)),
        completed_retrieval_rows=STATE_COUNT,
        coefficients_completed=True,
        fingerprints_completed=True,
        retrieval_completed=True,
        excel_completed=True,
        plot_completed=True,
        raw_manifest_before=raw_before,
        raw_manifest_after=raw_after,
    )
    _append_log(
        f"[{_now()}] Completed {TASK_NAME}\n"
        f"Summary={json.dumps(result_summary, ensure_ascii=False, sort_keys=True)}\n"
        "Historical MP10 backend reused; E19 Full-LUT retrieval/writer/plotter reused; raw data unchanged.\n"
    )
    _append_handoff(
        f"完成历史 Frozen MP10 + 当前 E19 Full425 common-B LUT retrieval；结果目录：{RESULT_ROOT}\n"
        f"摘要：{json.dumps(result_summary, ensure_ascii=False, sort_keys=True)}\n"
        "复用 historical sparse_gmp MP10、现有 signal_segmentation、E19 CNMSE/Top-1/Real-B/Excel/plot；未进行模型优化或扫描。"
    )
    return {"task": TASK_NAME, "status": "SUCCESS", **result_summary, "result_root": str(RESULT_ROOT)}


def main() -> None:
    result = _run(resume="--resume" in sys.argv[1:])
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    __import__("multiprocessing").freeze_support()
    main()
