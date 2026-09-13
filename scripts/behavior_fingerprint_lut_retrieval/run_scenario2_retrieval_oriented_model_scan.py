"""Run the full retrieval-oriented Scenario 2 MP candidate scan."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from behavior_fingerprint_ranking_consistency.scenario2_all_ilc_analysis import (  # noqa: E402
    load_frozen_scenario2_artifacts,
)
from data_manager import build_state_table  # noqa: E402

from behavior_fingerprint_lut_retrieval.plot_scenario2_retrieval_oriented_model_scan import (  # noqa: E402
    plot_baseline_vs_best_real_b,
    plot_candidate_failure_count,
    plot_candidate_shareable_rate,
    plot_failure_detail,
    plot_metric_change_vs_retrieval_change,
    plot_model_metric_comparison,
)
from behavior_fingerprint_lut_retrieval.scenario2_retrieval_oriented_model_scan import (  # noqa: E402
    BASELINE_CANDIDATE,
    BASELINE_FAILURE_IDS,
    BASELINE_LAMBDA,
    BASELINE_MAX_DELAY,
    BASELINE_MEMORY,
    BASELINE_ORDERS,
    C2_STAGE,
    CANDIDATES,
    COMMON_B_INPUT_LENGTH,
    DPD_SHAREABLE_THRESHOLD_DB,
    FORMAL_A_LENGTH,
    FORMAL_B_LENGTH,
    FORMAL_C_LENGTH,
    LAMBDA_GRID,
    MAX_COEFFICIENTS,
    MEMORY_PROFILES,
    NUMERIC_TOLERANCE_DB,
    ORDER_PROFILES,
    STATE_COUNT,
    build_candidate_retrieval,
    build_model_candidate_grid,
    candidate_from_id,
    fit_structure_for_state,
    load_baseline_metrics_reference,
    load_raw_scalar_metrics,
    load_real_b_reference,
    prepare_canonical_state,
    safe_correlation,
)

TASK_NAME = "behavior_fingerprint_lut_retrieval"
REFERENCE_ROOT = (
    PROJECT_ROOT / "results" / "behavior_fingerprint_ranking_consistency" / "scenario_2"
)
ALL_ILC_ROOT = (
    PROJECT_ROOT / "results" / "behavior_fingerprint_ranking_consistency" / "scenario_2_all_ilc"
)
BASELINE_ROOT = PROJECT_ROOT / "results" / TASK_NAME / "scenario_2_C2_to_Aend"
RESULT_ROOT = (
    PROJECT_ROOT / "results" / TASK_NAME / "scenario_2_C2_to_Aend_retrieval_oriented_model_scan"
)
LOG_ROOT = PROJECT_ROOT / "work_logs" / TASK_NAME
EXECUTION_LOG = LOG_ROOT / "execution_log.txt"
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "codex_handoff" / "codex_handoff.txt"
FIT_CACHE_PATH = RESULT_ROOT / "candidate_fit_cache.npz"
FIT_PROGRESS_PATH = RESULT_ROOT / "candidate_fit_progress.json"
PROGRESS_PATH = RESULT_ROOT / "candidate_progress.json"
STATEWISE_CACHE_ROOT = RESULT_ROOT / "_candidate_cache" / "statewise"
SUMMARY_CACHE_ROOT = RESULT_ROOT / "_candidate_cache" / "summaries"
EXPECTED_RAW_MANIFEST = "910b06e4be6b60b1f4dc63409a3083eb328765347fbec3b89a97a41d6633a1e0"

PROTECTED_RESULT_DIRS = {
    "scenario_2": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2",
    "scenario_2_all_ilc": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2_all_ilc",
    "scenario_2_y_lut_fusion": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2_y_lut_fusion",
    "scenario_2_y_lut_fusion_A123": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2_y_lut_fusion_A123",
    "scenario_2_y_lut_weighted_fusion": PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2_y_lut_weighted_fusion",
    "scenario_2_C3_to_A2": PROJECT_ROOT / "results" / TASK_NAME / "scenario_2_C3_to_A2",
    "scenario_2_C2_to_Aend": BASELINE_ROOT,
    "scenario_2_C2_to_Aend_unified_model_capacity": PROJECT_ROOT
    / "results"
    / TASK_NAME
    / "scenario_2_C2_to_Aend_unified_model_capacity",
    "scenario_2_C2_to_Aend_equal_ABC": PROJECT_ROOT
    / "results"
    / TASK_NAME
    / "scenario_2_C2_to_Aend_equal_ABC",
    "scenario_2_ridge_analysis": PROJECT_ROOT
    / "results"
    / "behavior_model"
    / "scenario_2_ridge_analysis",
    "scenario_2_xy_analysis": PROJECT_ROOT
    / "results"
    / "behavior_model"
    / "scenario_2_xy_analysis",
}
PROTECTED_SCRIPT_DIRS = {
    "behavior_fingerprint_ranking_consistency": PROJECT_ROOT
    / "scripts"
    / "behavior_fingerprint_ranking_consistency",
    "behavior_model": PROJECT_ROOT / "scripts" / "behavior_model",
    "signal_segmentation": PROJECT_ROOT / "scripts" / "signal_segmentation",
}


def _tree_digest(path: Path) -> str | None:
    path = Path(path)
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    files = (
        file
        for file in path.rglob("*")
        if file.is_file()
        and "__pycache__" not in file.parts
        and file.suffix.lower() != ".pyc"
        and not file.name.startswith("~$")
    )
    for file in sorted(files, key=lambda item: str(item).lower()):
        content = file.read_bytes()
        digest.update(file.relative_to(path).as_posix().encode() + b"\0")
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def _raw_manifest() -> dict[str, Any]:
    raw_root = Path(r"\\?\{}".format(PROJECT_ROOT / "data" / "raw"))
    digest = hashlib.sha256()
    files = sorted(
        (file for file in raw_root.rglob("*") if file.is_file()),
        key=lambda item: str(item).lower(),
    )
    total_bytes = 0
    for file in files:
        relative = str(file.relative_to(raw_root)).replace("\\", "/")
        size = int(file.stat().st_size)
        digest.update(relative.encode() + b"\0")
        digest.update(size.to_bytes(8, "little"))
        total_bytes += size
    return {"sha256": digest.hexdigest(), "file_count": len(files), "bytes": total_bytes}


def _file_snapshot(path: Path) -> dict[str, str]:
    path = Path(path)
    return {
        file.relative_to(path).as_posix(): hashlib.sha256(file.read_bytes()).hexdigest()
        for file in sorted(path.rglob("*"), key=lambda item: str(item).lower())
        if file.is_file()
        and "__pycache__" not in file.parts
        and file.suffix.lower() != ".pyc"
        and not file.name.startswith("~$")
    }


def _protection_snapshot() -> dict[str, Any]:
    return {
        "data/raw": _raw_manifest(),
        "result_dirs": {
            name: {
                "path": str(path),
                "exists": path.is_dir(),
                "sha256": _tree_digest(path),
            }
            for name, path in PROTECTED_RESULT_DIRS.items()
        },
        "script_dirs": {
            name: {
                "path": str(path),
                "exists": path.is_dir(),
                "sha256": _tree_digest(path),
            }
            for name, path in PROTECTED_SCRIPT_DIRS.items()
        },
        "retrieval_script_files": _file_snapshot(PROJECT_ROOT / "scripts" / TASK_NAME),
    }


def _verify_protection(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "data/raw": {},
        "result_dirs": {},
        "script_dirs": {},
        "retrieval_script_files": {},
    }
    result["data/raw"] = {
        "sha256_unchanged": before["data/raw"]["sha256"] == after["data/raw"]["sha256"],
        "file_count_unchanged": before["data/raw"]["file_count"] == after["data/raw"]["file_count"],
        "bytes_unchanged": before["data/raw"]["bytes"] == after["data/raw"]["bytes"],
    }
    for group in ("result_dirs", "script_dirs"):
        for name, item in before[group].items():
            after_item = after[group][name]
            result[group][name] = {
                "sha256_unchanged": item["sha256"] == after_item["sha256"],
                "before": item["sha256"],
                "after": after_item["sha256"],
            }
    for name, digest in before["retrieval_script_files"].items():
        result["retrieval_script_files"][name] = {
            "sha256_unchanged": after["retrieval_script_files"].get(name) == digest,
            "before": digest,
            "after": after["retrieval_script_files"].get(name),
        }
    result["all_protected_unchanged"] = bool(
        all(result["data/raw"].values())
        and all(
            item["sha256_unchanged"]
            for group in ("result_dirs", "script_dirs", "retrieval_script_files")
            for item in result[group].values()
        )
    )
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int, np.bool_, bool)):
        return value.item() if isinstance(value, np.generic) else value
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", na_rep="NaN", float_format="%.17g")


def _write_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _append_log(text: str) -> None:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    with EXECUTION_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n[{timestamp}] Scenario 2 retrieval-oriented unified MP model scan\n")
        handle.write(text.rstrip() + "\n")


def _append_handoff(text: str) -> None:
    HANDOFF_LOG.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    with HANDOFF_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n{timestamp} | {TASK_NAME}持续维护：retrieval-oriented统一MP扫描\n")
        handle.write(text.rstrip() + "\n")


def _load_formal_baseline_retrieval() -> pd.DataFrame:
    results = pd.read_csv(BASELINE_ROOT / "retrieval_results.csv")
    diagnostics = pd.read_csv(BASELINE_ROOT / "retrieved_real_B_cnmse.csv")
    merged = results.loc[:, ["State_n_R", "State_n_Q", "exact_hit"]].merge(
        diagnostics.loc[:, ["State_n_R", "retrieved_real_B_CNMSE_dB", "dpd_shareable", "failure"]],
        on="State_n_R",
        how="left",
        validate="one_to_one",
    )
    if merged.shape[0] != STATE_COUNT:
        raise RuntimeError("正式C2→Aend Baseline检索结果必须有425行")
    return merged.rename(
        columns={
            "State_n_R": "state_id_R",
            "State_n_Q": "state_id_Q",
            "dpd_shareable": "baseline_shareable",
        }
    )


def _distance_error(current: np.ndarray, reference: np.ndarray) -> dict[str, Any]:
    current = np.asarray(current)
    reference = np.asarray(reference)
    if current.shape != reference.shape:
        return {
            "shape_equal": False,
            "max_abs_error": None,
            "nonfinite_pattern_mismatch": current.size,
        }
    current_neg_inf = np.isneginf(current)
    reference_neg_inf = np.isneginf(reference)
    mismatch = int(np.count_nonzero(current_neg_inf != reference_neg_inf))
    current_finite = np.isfinite(current)
    reference_finite = np.isfinite(reference)
    mismatch += int(np.count_nonzero(current_finite != reference_finite))
    finite = current_finite & reference_finite
    error = float(np.max(np.abs(current[finite] - reference[finite]))) if np.any(finite) else 0.0
    return {
        "shape_equal": True,
        "max_abs_error": error,
        "nonfinite_pattern_mismatch": mismatch,
        "pass": bool(mismatch == 0 and error <= 1e-12),
    }


def _load_or_initialize_progress() -> dict[str, Any]:
    if PROGRESS_PATH.is_file():
        payload = json.loads(PROGRESS_PATH.read_text(encoding="utf-8"))
        if int(payload.get("candidate_count", -1)) == len(CANDIDATES):
            return payload
    return {
        "candidate_count": len(CANDIDATES),
        "completed_candidate_ids": [],
        "invalid_candidate_ids": [],
        "fit_cache_ready": FIT_CACHE_PATH.is_file(),
        "resume_scope": "candidate retrieval artifacts after complete fit cache",
    }


def _save_progress(progress: dict[str, Any]) -> None:
    progress["completed_candidate_ids"] = sorted(
        {int(value) for value in progress.get("completed_candidate_ids", [])}
    )
    progress["invalid_candidate_ids"] = sorted(
        {int(value) for value in progress.get("invalid_candidate_ids", [])}
    )
    _write_json(PROGRESS_PATH, progress)


def _load_or_fit_all_candidates(
    candidate_grid: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return padded theta, four metrics, two norm metrics and ILC stage counts."""

    if FIT_CACHE_PATH.is_file():
        with np.load(FIT_CACHE_PATH, allow_pickle=False) as data:
            required = {
                "candidate_ids",
                "state_ids",
                "theta_Aend",
                "theta_C2",
                "metrics",
                "theta_norms",
                "ilc_A_end",
            }
            if required.issubset(data.files):
                candidate_ids = np.asarray(data["candidate_ids"], dtype=np.int64)
                state_ids = np.asarray(data["state_ids"], dtype=np.int64)
                theta_a = np.asarray(data["theta_Aend"])
                theta_c = np.asarray(data["theta_C2"])
                metrics = np.asarray(data["metrics"])
                norms = np.asarray(data["theta_norms"])
                ilc_counts = np.asarray(data["ilc_A_end"], dtype=np.int64)
            else:
                candidate_ids = np.asarray([], dtype=np.int64)
        if (
            np.array_equal(candidate_ids, np.arange(1, len(CANDIDATES) + 1, dtype=np.int64))
            and np.array_equal(state_ids, np.arange(STATE_COUNT, dtype=np.int64))
            and theta_a.shape == (len(CANDIDATES), STATE_COUNT, MAX_COEFFICIENTS)
            and theta_c.shape == theta_a.shape
            and metrics.shape == (len(CANDIDATES), STATE_COUNT, 4)
            and norms.shape == (len(CANDIDATES), STATE_COUNT, 2)
            and ilc_counts.shape == (STATE_COUNT,)
            and np.all(np.isfinite(theta_a))
            and np.all(np.isfinite(theta_c))
            and np.all(np.isfinite(metrics))
            and np.all(np.isfinite(norms))
        ):
            print("Loaded complete candidate fit cache.", flush=True)
            return theta_a, theta_c, metrics, norms, ilc_counts

    # Unused padded coefficient slots are zero-filled.  Candidate-specific
    # multiplications always slice to the exact coefficient count.
    theta_a = np.zeros((len(CANDIDATES), STATE_COUNT, MAX_COEFFICIENTS), dtype=np.complex128)
    theta_c = np.zeros_like(theta_a)
    metrics = np.full((len(CANDIDATES), STATE_COUNT, 4), np.nan, dtype=np.float64)
    norms = np.full((len(CANDIDATES), STATE_COUNT, 2), np.nan, dtype=np.float64)
    ilc_counts = np.full(STATE_COUNT, -1, dtype=np.int64)
    candidate_by_structure: dict[tuple[str, str], dict[int, Any]] = {}
    for candidate in CANDIDATES:
        candidate_by_structure.setdefault((candidate.order_profile, candidate.memory_profile), {})[
            candidate.candidate_id
        ] = candidate
    lambda_index = {float(value): index for index, value in enumerate(LAMBDA_GRID)}
    fit_progress: dict[str, Any] = {
        "state_count": STATE_COUNT,
        "completed_state_ids": [],
        "candidate_count": len(CANDIDATES),
        "model_metrics_do_not_prune": True,
    }
    for position, state_id in enumerate(range(STATE_COUNT), start=1):
        prepared = prepare_canonical_state(state_id)
        ilc_counts[state_id] = prepared.ilc_column_count
        for memory_profile in MEMORY_PROFILES:
            fits = fit_structure_for_state(prepared, memory_profile)
            for order_profile, fit in fits.items():
                candidates = candidate_by_structure[(order_profile, memory_profile)]
                for candidate_id, candidate in candidates.items():
                    lam_idx = lambda_index[candidate.ridge_lambda]
                    index = candidate_id - 1
                    n_coefficients = candidate.n_complex_coefficients
                    metrics[index, state_id] = fit.metrics[lam_idx]
                    theta_a[index, state_id, :n_coefficients] = fit.theta_a[lam_idx]
                    theta_c[index, state_id, :n_coefficients] = fit.theta_c[lam_idx]
                    norms[index, state_id] = (
                        fit.theta_norm_a[lam_idx],
                        fit.theta_norm_c[lam_idx],
                    )
        fit_progress["completed_state_ids"].append(state_id)
        if position % 5 == 0 or position == STATE_COUNT:
            _write_json(FIT_PROGRESS_PATH, fit_progress)
            print(f"fit cache: processed {position} / {STATE_COUNT} states", flush=True)
    if not np.all(ilc_counts >= C2_STAGE):
        raise RuntimeError("存在缺少真实C2的状态")
    if not np.all(np.isfinite(theta_a)) or not np.all(np.isfinite(theta_c)):
        raise RuntimeError("Candidate theta包含非有限值")
    if not np.all(np.isfinite(metrics)) or not np.all(np.isfinite(norms)):
        raise RuntimeError("Candidate模型指标或系数范数包含非有限值")
    _write_npz(
        FIT_CACHE_PATH,
        {
            "candidate_ids": np.arange(1, len(CANDIDATES) + 1, dtype=np.int64),
            "state_ids": np.arange(STATE_COUNT, dtype=np.int64),
            "theta_Aend": theta_a,
            "theta_C2": theta_c,
            "metrics": metrics,
            "theta_norms": norms,
            "ilc_A_end": ilc_counts,
        },
    )
    _write_json(FIT_PROGRESS_PATH, {**fit_progress, "fit_cache_ready": True})
    print(f"Saved complete candidate fit cache: {FIT_CACHE_PATH}", flush=True)
    return theta_a, theta_c, metrics, norms, ilc_counts


def _candidate_fingerprints(
    candidate: Any,
    theta_a: np.ndarray,
    theta_c: np.ndarray,
    common_b_input: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    from behavior_model.basis import build_mp_basis

    phi_common = build_mp_basis(common_b_input, candidate.orders, candidate.memory_definition)
    expected_length = COMMON_B_INPUT_LENGTH - candidate.max_delay
    if phi_common.shape != (expected_length, candidate.n_complex_coefficients):
        raise RuntimeError(
            f"candidate={candidate.candidate_id} common-B Phi shape错误：{phi_common.shape}"
        )
    fp_a = (phi_common @ theta_a[:, : candidate.n_complex_coefficients].T).T.astype(
        np.complex128, copy=False
    )
    fp_c = (phi_common @ theta_c[:, : candidate.n_complex_coefficients].T).T.astype(
        np.complex128, copy=False
    )
    if not np.all(np.isfinite(fp_a)) or not np.all(np.isfinite(fp_c)):
        raise RuntimeError(f"candidate={candidate.candidate_id} fingerprint包含非有限值")
    return fp_a, fp_c


def _statewise_with_metrics(
    candidate: Any,
    metrics: np.ndarray,
    retrieval: dict[str, Any],
    ilc_counts: np.ndarray,
) -> pd.DataFrame:
    result = retrieval["results"].copy()
    model = pd.DataFrame(
        {
            "candidate_id": candidate.candidate_id,
            "state_id_R": np.arange(STATE_COUNT, dtype=np.int64),
            "Y_Aend_train_NMSE_dB": metrics[:, 0],
            "Y_Aend_B_NMSE_dB": metrics[:, 1],
            "Y_C2_train_NMSE_dB": metrics[:, 2],
            "Y_C2_B_NMSE_dB": metrics[:, 3],
            "ilc_A_end": ilc_counts,
        }
    )
    merged = model.merge(
        result, on=["candidate_id", "state_id_R"], how="inner", validate="one_to_one"
    )
    if merged.shape[0] != STATE_COUNT:
        raise RuntimeError("Candidate statewise合并后不是425行")
    return merged


def _transition_counts(
    baseline_shareable: np.ndarray, candidate_shareable: np.ndarray
) -> dict[str, int]:
    return {
        "success_to_success": int(np.count_nonzero(baseline_shareable & candidate_shareable)),
        "success_to_failure": int(np.count_nonzero(baseline_shareable & ~candidate_shareable)),
        "failure_to_success": int(np.count_nonzero(~baseline_shareable & candidate_shareable)),
        "failure_to_failure": int(np.count_nonzero(~baseline_shareable & ~candidate_shareable)),
    }


def _candidate_summary(
    candidate: Any,
    metrics: np.ndarray,
    norms: np.ndarray,
    retrieval: dict[str, Any],
    baseline_shareable: np.ndarray,
) -> dict[str, Any]:
    summary = dict(retrieval["summary"])
    shareable = retrieval["results"]["dpd_shareable"].to_numpy(dtype=bool)
    transition = _transition_counts(baseline_shareable, shareable)
    finite_nonexact = retrieval["real_b_values"][
        (~retrieval["results"]["exact_hit"].to_numpy(dtype=bool))
        & np.isfinite(retrieval["real_b_values"])
    ]
    return {
        "candidate_id": candidate.candidate_id,
        "order_profile": candidate.order_profile,
        "orders": json.dumps(list(candidate.orders), separators=(",", ":")),
        "memory_profile": candidate.memory_profile,
        "memory_definition": json.dumps(
            candidate.memory_definition, sort_keys=True, separators=(",", ":")
        ),
        "lambda": candidate.ridge_lambda,
        "max_delay": candidate.max_delay,
        "n_complex_coefficients": candidate.n_complex_coefficients,
        "structure_id": candidate.structure_id,
        "is_baseline": candidate.is_baseline,
        "candidate_valid": True,
        "invalid_reason": "",
        "Aend_train_median_dB": float(np.median(metrics[:, 0])),
        "Aend_B_median_dB": float(np.median(metrics[:, 1])),
        "C2_train_median_dB": float(np.median(metrics[:, 2])),
        "C2_B_median_dB": float(np.median(metrics[:, 3])),
        "Aend_theta_norm_median": float(np.median(norms[:, 0])),
        "Aend_theta_norm_q95": float(np.quantile(norms[:, 0], 0.95)),
        "C2_theta_norm_median": float(np.median(norms[:, 1])),
        "C2_theta_norm_q95": float(np.quantile(norms[:, 1], 0.95)),
        "exact_hit_count": summary["exact_hit_count"],
        "exact_hit_rate": summary["exact_hit_rate"],
        "nonexact_count": summary["nonexact_count"],
        "shareable_count": summary["shareable_count"],
        "shareable_rate": summary["shareable_rate"],
        "failure_count": summary["failure_count"],
        "old_failure_rescued_count": transition["failure_to_success"],
        "new_failure_created_count": transition["success_to_failure"],
        "net_success_gain": int(summary["shareable_count"] - int(baseline_shareable.sum())),
        "nonexact_real_B_median_dB": float(np.median(finite_nonexact))
        if finite_nonexact.size
        else np.nan,
        "real_B_q95_dB": summary["retrieved_real_B_q95_dB"],
        "retrieved_real_B_q95_dB": summary["retrieved_real_B_q95_dB"],
        "worst_finite_real_B_CNMSE_dB": summary["worst_finite_retrieved_real_B_CNMSE_dB"],
        "true_state_rank_median": summary["true_state_rank_median"],
        "true_state_rank_max": summary["true_state_rank_max"],
        "top3_true_state_hit_count": summary["top3_true_state_hit_count"],
        "top5_true_state_hit_count": summary["top5_true_state_hit_count"],
        "top10_true_state_hit_count": summary["top10_true_state_hit_count"],
        "minimum_tie_rows": summary["minimum_tie_rows"],
        "query_count": summary["query_count"],
        "lut_entry_count": summary["lut_entry_count"],
        "distance_count": summary["distance_count"],
        "distance_matrix_shape": json.dumps(summary["distance_matrix_shape"]),
        "candidate_ranking_uses_real_B": True,
        "model_metrics_used_for_ranking": False,
        **{f"transition_{key}": value for key, value in transition.items()},
    }


def _build_best_excel_source(
    metrics: pd.DataFrame,
    retrieval: pd.DataFrame,
    raw_scalars: pd.DataFrame,
    state_frame: pd.DataFrame,
) -> pd.DataFrame:
    merged = (
        metrics.merge(state_frame, on="state_id", how="inner", validate="one_to_one")
        .merge(
            raw_scalars.loc[:, ["state_id", "nmse_withoutdpd", "acpr_withoutdpd_avg_dBc"]],
            on="state_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            retrieval.loc[
                :, ["state_id_R", "state_id_Q", "exact_hit", "retrieved_real_B_CNMSE_dB"]
            ],
            left_on="state_id",
            right_on="state_id_R",
            how="inner",
            validate="one_to_one",
        )
    )
    if merged.shape[0] != STATE_COUNT:
        raise RuntimeError("Best Excel合并后不是425行")

    def _mismatch(value: Any) -> str:
        number = int(round(float(value)))
        return "0" if number == 0 else f"{number / 100:.2f}".rstrip("0").rstrip(".")

    load_config = [
        f"funMng={_mismatch(row.funMng)}, funAng={int(row.funAng)}°, "
        f"secMng={_mismatch(row.secMng)}, secAng={int(row.secAng)}°"
        for row in merged.itertuples(index=False)
    ]
    retrieved: list[Any] = []
    for row in merged.itertuples(index=False):
        if bool(row.exact_hit):
            retrieved.append("-Inf")
        else:
            value = float(row.retrieved_real_B_CNMSE_dB)
            if not np.isfinite(value):
                raise RuntimeError(f"Best state={row.state_id} non-exact Real-B非finite")
            retrieved.append(value)
    frame = pd.DataFrame(
        {
            "state_id_R": merged["state_id"].to_numpy(dtype=np.int64),
            "load_config": load_config,
            "nmse_withoutdpd_dB": merged["nmse_withoutdpd"].to_numpy(dtype=float),
            "acpr_withoutdpd_avg_dBc": merged["acpr_withoutdpd_avg_dBc"].to_numpy(dtype=float),
            "Y_Aend_train_NMSE_dB": merged["Y_Aend_train_NMSE_dB"].to_numpy(dtype=float),
            "Y_Aend_B_NMSE_dB": merged["Y_Aend_B_NMSE_dB"].to_numpy(dtype=float),
            "Y_C2_train_NMSE_dB": merged["Y_C2_train_NMSE_dB"].to_numpy(dtype=float),
            "Y_C2_B_NMSE_dB": merged["Y_C2_B_NMSE_dB"].to_numpy(dtype=float),
            "state_id_Q": merged["state_id_Q"].to_numpy(dtype=np.int64),
            "retrieved_real_B_CNMSE_dB": retrieved,
        }
    )
    return frame.sort_values("state_id_R").reset_index(drop=True)


def _support_control(
    baseline_fingerprints: tuple[np.ndarray, np.ndarray],
    real_b_distance: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    base_a, base_c = baseline_fingerprints
    for support_delay in (2, 3, 4, 5):
        length = 4913 - (support_delay - BASELINE_MAX_DELAY)
        retrieval = build_candidate_retrieval(
            base_c[:, :length],
            base_a[:, :length],
            real_b_distance,
            candidate=BASELINE_CANDIDATE,
        )
        summary = retrieval["summary"]
        rows.append(
            {
                "support_delay": support_delay,
                "common_support_length": length,
                "query_count": STATE_COUNT,
                "lut_entry_count": STATE_COUNT,
                "distance_matrix_shape": json.dumps([STATE_COUNT, STATE_COUNT]),
                "exact_hit_count": summary["exact_hit_count"],
                "shareable_count": summary["shareable_count"],
                "shareable_rate": summary["shareable_rate"],
                "failure_count": summary["failure_count"],
                "nonexact_real_B_median_dB": summary["nonexact_real_B_median_dB"],
                "retrieved_real_B_q95_dB": summary["retrieved_real_B_q95_dB"],
                "worst_finite_retrieved_real_B_CNMSE_dB": summary[
                    "worst_finite_retrieved_real_B_CNMSE_dB"
                ],
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    print("Scenario 2 retrieval-oriented unified MP model scan started.", flush=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    before = _protection_snapshot()
    if (
        before["data/raw"]["sha256"] != EXPECTED_RAW_MANIFEST
        or before["data/raw"]["file_count"] != 429
    ):
        raise RuntimeError(f"data/raw manifest不符合冻结基线：{before['data/raw']}")

    candidate_grid = build_model_candidate_grid()
    _write_frame(candidate_grid, RESULT_ROOT / "model_candidate_grid.csv")
    state_frame = pd.DataFrame(build_state_table()).sort_values("state_id").reset_index(drop=True)
    if state_frame.shape[0] != STATE_COUNT:
        raise RuntimeError("state table必须包含425状态")
    frozen = load_frozen_scenario2_artifacts(REFERENCE_ROOT)
    real_b_distance, real_b_ranking = load_real_b_reference(
        BASELINE_ROOT / "real_B_distance_matrix.npz",
        REFERENCE_ROOT / "distance_matrices.npz",
        REFERENCE_ROOT / "ranking_matrices.npz",
    )
    baseline_retrieval_reference = _load_formal_baseline_retrieval()
    if (
        int(baseline_retrieval_reference["exact_hit"].sum()),
        int(baseline_retrieval_reference["baseline_shareable"].sum()),
        int(baseline_retrieval_reference["failure"].sum()),
    ) != (218, 411, 14):
        raise RuntimeError("正式Baseline不是218 exact/411 shareable/14 failure")
    baseline_shareable = baseline_retrieval_reference["baseline_shareable"].to_numpy(dtype=bool)
    _write_json(
        RESULT_ROOT / "candidate_progress.json",
        _load_or_initialize_progress(),
    )

    print("Step 1/7: fit or load all 200 candidates", flush=True)
    theta_a, theta_c, all_metrics, theta_norms, ilc_counts = _load_or_fit_all_candidates(
        candidate_grid
    )
    ilc_counts = np.asarray(ilc_counts, dtype=np.int64)
    if not np.array_equal(
        ilc_counts,
        pd.read_csv(BASELINE_ROOT / "a_end_stage_map.csv")["ilc_A_end"].to_numpy(dtype=np.int64),
    ):
        raise RuntimeError("重新读取的Aend stage map与正式结果不一致")
    stage_map = state_frame.copy()
    stage_map["N_ilc_available"] = ilc_counts
    stage_map["ilc_A_end"] = ilc_counts
    stage_map["C2_available"] = ilc_counts >= C2_STAGE
    _write_frame(stage_map, RESULT_ROOT / "a_end_stage_map.csv")

    print("Step 2/7: strict Baseline regression before scanning retrieval candidates", flush=True)
    baseline_index = BASELINE_CANDIDATE.candidate_id - 1
    baseline_fp_a, baseline_fp_c = _candidate_fingerprints(
        BASELINE_CANDIDATE,
        theta_a[baseline_index],
        theta_c[baseline_index],
        frozen.common_B_input,
    )
    baseline_retrieval = build_candidate_retrieval(
        baseline_fp_c,
        baseline_fp_a,
        real_b_distance,
        candidate=BASELINE_CANDIDATE,
        baseline_retrieval=None,
    )
    with np.load(BASELINE_ROOT / "retrieval_distance_matrix.npz", allow_pickle=False) as data:
        old_distance = np.asarray(data["D_C2_Aend"])
    with np.load(BASELINE_ROOT / "retrieval_ranking_matrix.npz", allow_pickle=False) as data:
        old_ranking = np.asarray(data["R_C2_Aend"])
    baseline_distance_check = _distance_error(baseline_retrieval["distance"], old_distance)
    baseline_ranking_check = {
        "equal": bool(np.array_equal(baseline_retrieval["ranking"], old_ranking)),
        "max_abs_error": float(np.max(np.abs(baseline_retrieval["ranking"] - old_ranking))),
    }
    baseline_selected_check = bool(
        np.array_equal(
            baseline_retrieval["selected_q"],
            baseline_retrieval_reference.sort_values("state_id_R")["state_id_Q"].to_numpy(
                dtype=np.int64
            ),
        )
    )
    baseline_failure_ids = tuple(
        baseline_retrieval["results"]
        .loc[baseline_retrieval["results"]["failure"], "state_id_R"]
        .tolist()
    )
    metric_reference = load_baseline_metrics_reference(
        ALL_ILC_ROOT / "all_ilc_model_metrics.csv",
        ALL_ILC_ROOT / "ilc_availability.csv",
    )
    observed_metrics = all_metrics[baseline_index]
    metric_errors = {
        name: float(
            np.max(
                np.abs(
                    observed_metrics[:, idx]
                    - metric_reference.sort_values("state_id")[name].to_numpy(dtype=float)
                )
            )
        )
        for idx, name in enumerate(
            [
                "Y_Aend_train_NMSE_dB",
                "Y_Aend_B_NMSE_dB",
                "Y_C2_train_NMSE_dB",
                "Y_C2_B_NMSE_dB",
            ]
        )
    }
    baseline_regression = {
        "candidate_id": BASELINE_CANDIDATE.candidate_id,
        "orders": list(BASELINE_ORDERS),
        "memory": BASELINE_MEMORY,
        "lambda": BASELINE_LAMBDA,
        "retrieval_distance": baseline_distance_check,
        "retrieval_ranking": baseline_ranking_check,
        "selected_state_Q_equal": baseline_selected_check,
        "metric_max_abs_error_dB": metric_errors,
        "metric_tolerance_dB": 1e-10,
        "expected_exact": 218,
        "expected_shareable": 411,
        "expected_failure": 14,
        "observed_exact": int(baseline_retrieval["summary"]["exact_hit_count"]),
        "observed_shareable": int(baseline_retrieval["summary"]["shareable_count"]),
        "observed_failure": int(baseline_retrieval["summary"]["failure_count"]),
        "failure_ids": list(baseline_failure_ids),
        "expected_failure_ids": list(BASELINE_FAILURE_IDS),
        "pass": bool(
            baseline_distance_check.get("pass", False)
            and baseline_ranking_check["equal"]
            and baseline_selected_check
            and max(metric_errors.values()) <= 1e-10
            and baseline_failure_ids == BASELINE_FAILURE_IDS
            and baseline_retrieval["summary"]["exact_hit_count"] == 218
            and baseline_retrieval["summary"]["shareable_count"] == 411
            and baseline_retrieval["summary"]["failure_count"] == 14
        ),
    }
    _write_json(RESULT_ROOT / "baseline_regression.json", baseline_regression)
    if not baseline_regression["pass"]:
        raise RuntimeError(f"Baseline回归失败：{baseline_regression}")
    _write_npz(
        RESULT_ROOT / "baseline_candidate_coefficients.npz",
        {
            "state_ids": np.arange(STATE_COUNT, dtype=np.int64),
            "theta_Aend": theta_a[baseline_index, :, : BASELINE_CANDIDATE.n_complex_coefficients],
            "theta_C2": theta_c[baseline_index, :, : BASELINE_CANDIDATE.n_complex_coefficients],
            "orders": np.asarray(BASELINE_CANDIDATE.orders, dtype=np.int64),
            "memory_orders": np.asarray(
                sorted(BASELINE_CANDIDATE.memory_definition), dtype=np.int64
            ),
            "memory_taps": np.asarray(
                [
                    BASELINE_CANDIDATE.memory_definition[key]
                    for key in sorted(BASELINE_CANDIDATE.memory_definition)
                ],
                dtype=np.int64,
            ),
            "ridge_lambda": np.asarray([BASELINE_LAMBDA], dtype=np.float64),
            "Aend_stage": ilc_counts,
        },
    )
    _write_npz(
        RESULT_ROOT / "baseline_candidate_lut_fingerprints_Aend.npz",
        {
            "state_ids": np.arange(STATE_COUNT),
            "common_B_input": frozen.common_B_input,
            "Y_Aend_fingerprints": baseline_fp_a,
            "Aend_stage": ilc_counts,
        },
    )
    _write_npz(
        RESULT_ROOT / "baseline_candidate_query_fingerprints_C2.npz",
        {
            "state_ids": np.arange(STATE_COUNT),
            "common_B_input": frozen.common_B_input,
            "Q_C2": baseline_fp_c,
            "effective_C_stage": np.full(STATE_COUNT, C2_STAGE, dtype=np.int64),
        },
    )
    _write_npz(
        RESULT_ROOT / "baseline_candidate_retrieval_distance_matrix.npz",
        {"state_ids": np.arange(STATE_COUNT), "D_C2_Aend": baseline_retrieval["distance"]},
    )
    _write_npz(
        RESULT_ROOT / "baseline_candidate_retrieval_ranking_matrix.npz",
        {"state_ids": np.arange(STATE_COUNT), "R_C2_Aend": baseline_retrieval["ranking"]},
    )

    print("Step 3/7: full 425-query/full-425-LUT retrieval for every candidate", flush=True)
    progress = _load_or_initialize_progress()
    baseline_for_transition = (
        baseline_retrieval["results"]
        .loc[:, ["state_id_R", "dpd_shareable"]]
        .rename(columns={"dpd_shareable": "baseline_shareable"})
    )
    summary_rows: list[dict[str, Any]] = []
    completed_ids = set(int(value) for value in progress.get("completed_candidate_ids", []))
    invalid_ids = set(int(value) for value in progress.get("invalid_candidate_ids", []))
    for candidate in CANDIDATES:
        statewise_path = STATEWISE_CACHE_ROOT / f"candidate_{candidate.candidate_id:03d}.csv"
        summary_path = SUMMARY_CACHE_ROOT / f"candidate_{candidate.candidate_id:03d}.json"
        if (
            candidate.candidate_id in completed_ids
            and statewise_path.is_file()
            and summary_path.is_file()
        ):
            summary_rows.append(json.loads(summary_path.read_text(encoding="utf-8")))
            continue
        index = candidate.candidate_id - 1
        try:
            fp_a, fp_c = _candidate_fingerprints(
                candidate,
                theta_a[index],
                theta_c[index],
                frozen.common_B_input,
            )
            retrieval = build_candidate_retrieval(
                fp_c,
                fp_a,
                real_b_distance,
                candidate=candidate,
                baseline_retrieval=baseline_for_transition,
            )
            statewise = _statewise_with_metrics(
                candidate,
                all_metrics[index],
                retrieval,
                ilc_counts,
            )
            summary = _candidate_summary(
                candidate,
                all_metrics[index],
                theta_norms[index],
                retrieval,
                baseline_shareable,
            )
            _write_frame(statewise, statewise_path)
            _write_json(summary_path, summary)
            summary_rows.append(summary)
            completed_ids.add(candidate.candidate_id)
            progress["completed_candidate_ids"] = sorted(completed_ids)
            progress["invalid_candidate_ids"] = sorted(invalid_ids)
            progress["last_candidate_id"] = candidate.candidate_id
            _save_progress(progress)
            if candidate.candidate_id % 10 == 0 or candidate.candidate_id == len(CANDIDATES):
                print(
                    f"retrieval candidates: completed {candidate.candidate_id} / {len(CANDIDATES)}",
                    flush=True,
                )
        except Exception as exc:
            invalid_ids.add(candidate.candidate_id)
            summary = {
                "candidate_id": candidate.candidate_id,
                "order_profile": candidate.order_profile,
                "orders": json.dumps(list(candidate.orders), separators=(",", ":")),
                "memory_profile": candidate.memory_profile,
                "memory_definition": json.dumps(
                    candidate.memory_definition, sort_keys=True, separators=(",", ":")
                ),
                "lambda": candidate.ridge_lambda,
                "max_delay": candidate.max_delay,
                "n_complex_coefficients": candidate.n_complex_coefficients,
                "structure_id": candidate.structure_id,
                "is_baseline": candidate.is_baseline,
                "candidate_valid": False,
                "invalid_reason": repr(exc),
                "failure_count": np.nan,
                "shareable_count": np.nan,
                "shareable_rate": np.nan,
                "exact_hit_count": np.nan,
            }
            _write_json(summary_path, summary)
            summary_rows.append(summary)
            _save_progress(
                {
                    **progress,
                    "completed_candidate_ids": sorted(completed_ids),
                    "invalid_candidate_ids": sorted(invalid_ids),
                    "last_candidate_id": candidate.candidate_id,
                }
            )
            print(f"WARNING candidate={candidate.candidate_id} invalid: {exc}", flush=True)

    progress["completed_candidate_ids"] = sorted(completed_ids)
    progress["invalid_candidate_ids"] = sorted(invalid_ids)
    progress["valid_candidate_count"] = len(completed_ids)
    progress["statewise_rows_expected"] = len(completed_ids) * STATE_COUNT
    _save_progress(progress)
    summary_frame = (
        pd.DataFrame(summary_rows).drop_duplicates("candidate_id").sort_values("candidate_id")
    )
    if summary_frame.shape[0] != len(CANDIDATES):
        raise RuntimeError("Candidate summary必须包含200行")
    statewise_parts = [
        pd.read_csv(STATEWISE_CACHE_ROOT / f"candidate_{candidate_id:03d}.csv")
        for candidate_id in sorted(completed_ids)
        if (STATEWISE_CACHE_ROOT / f"candidate_{candidate_id:03d}.csv").is_file()
    ]
    if statewise_parts:
        statewise_all = pd.concat(statewise_parts, ignore_index=True)
    else:
        statewise_all = pd.DataFrame()
    if statewise_all.shape[0] != len(completed_ids) * STATE_COUNT:
        raise RuntimeError("retrieval_oriented_candidate_statewise行数错误")
    _write_frame(statewise_all, RESULT_ROOT / "retrieval_oriented_candidate_statewise.csv")
    _write_frame(summary_frame, RESULT_ROOT / "retrieval_oriented_candidate_summary.csv")

    print("Step 4/7: rank candidates by final Real-B retrieval outcome", flush=True)
    rank_columns = [
        "failure_count",
        "worst_finite_real_B_CNMSE_dB",
        "real_B_q95_dB",
        "nonexact_real_B_median_dB",
        "exact_hit_count",
        "n_complex_coefficients",
        "candidate_id",
    ]
    ranking_frame = summary_frame.sort_values(
        rank_columns,
        ascending=[True, True, True, True, False, True, True],
        na_position="last",
        kind="mergesort",
    ).reset_index(drop=True)
    ranking_frame["rank"] = np.arange(1, ranking_frame.shape[0] + 1, dtype=np.int64)
    _write_frame(ranking_frame, RESULT_ROOT / "retrieval_oriented_candidate_ranking.csv")
    valid_ranking = ranking_frame.loc[ranking_frame["candidate_valid"].astype(bool)]
    if valid_ranking.empty:
        raise RuntimeError("没有有效Candidate")
    best_row = valid_ranking.iloc[0]
    best_candidate = candidate_from_id(int(best_row["candidate_id"]))
    best_payload = {
        "candidate_id": best_candidate.candidate_id,
        "orders": list(best_candidate.orders),
        "order_profile": best_candidate.order_profile,
        "memory_profile": best_candidate.memory_profile,
        "memory": best_candidate.memory_definition,
        "max_delay": best_candidate.max_delay,
        "lambda": best_candidate.ridge_lambda,
        "n_complex_coefficients": best_candidate.n_complex_coefficients,
        "failure_count": int(best_row["failure_count"]),
        "shareable_count": int(best_row["shareable_count"]),
        "shareable_rate": float(best_row["shareable_rate"]),
        "exact_hit_count": int(best_row["exact_hit_count"]),
        "worst_real_B": float(best_row["worst_finite_real_B_CNMSE_dB"]),
        "q95": float(best_row["real_B_q95_dB"]),
        "median": float(best_row["nonexact_real_B_median_dB"]),
        "old_failure_rescued_count": int(best_row["old_failure_rescued_count"]),
        "new_failure_created_count": int(best_row["new_failure_created_count"]),
        "rank": int(best_row["rank"]),
        "candidate_ranking_uses_real_B": True,
        "study_type": "retrieval-oriented oracle/post-hoc scan",
    }
    _write_json(RESULT_ROOT / "best_retrieval_model.json", best_payload)
    zero_failure = ranking_frame.loc[
        ranking_frame["candidate_valid"].astype(bool) & ranking_frame["failure_count"].eq(0)
    ].copy()
    _write_frame(zero_failure, RESULT_ROOT / "zero_failure_candidates.csv")

    print("Step 5/7: Best artifacts, statewise comparisons and diagnostics", flush=True)
    best_index = best_candidate.candidate_id - 1
    best_fp_a, best_fp_c = _candidate_fingerprints(
        best_candidate,
        theta_a[best_index],
        theta_c[best_index],
        frozen.common_B_input,
    )
    best_retrieval = build_candidate_retrieval(
        best_fp_c,
        best_fp_a,
        real_b_distance,
        candidate=best_candidate,
        baseline_retrieval=baseline_for_transition,
    )
    _write_npz(
        RESULT_ROOT / "best_candidate_coefficients.npz",
        {
            "state_ids": np.arange(STATE_COUNT, dtype=np.int64),
            "theta_Aend": theta_a[best_index, :, : best_candidate.n_complex_coefficients],
            "theta_C2": theta_c[best_index, :, : best_candidate.n_complex_coefficients],
            "orders": np.asarray(best_candidate.orders, dtype=np.int64),
            "memory_orders": np.asarray(sorted(best_candidate.memory_definition), dtype=np.int64),
            "memory_taps": np.asarray(
                [
                    best_candidate.memory_definition[key]
                    for key in sorted(best_candidate.memory_definition)
                ],
                dtype=np.int64,
            ),
            "ridge_lambda": np.asarray([best_candidate.ridge_lambda], dtype=np.float64),
            "Aend_stage": ilc_counts,
        },
    )
    _write_npz(
        RESULT_ROOT / "best_candidate_lut_fingerprints_Aend.npz",
        {
            "state_ids": np.arange(STATE_COUNT),
            "common_B_input": frozen.common_B_input,
            "Y_Aend_fingerprints": best_fp_a,
            "Aend_stage": ilc_counts,
        },
    )
    _write_npz(
        RESULT_ROOT / "best_candidate_query_fingerprints_C2.npz",
        {
            "state_ids": np.arange(STATE_COUNT),
            "common_B_input": frozen.common_B_input,
            "Q_C2": best_fp_c,
            "effective_C_stage": np.full(STATE_COUNT, C2_STAGE, dtype=np.int64),
        },
    )
    _write_npz(
        RESULT_ROOT / "best_candidate_retrieval_distance_matrix.npz",
        {"state_ids": np.arange(STATE_COUNT), "D_C2_Aend": best_retrieval["distance"]},
    )
    _write_npz(
        RESULT_ROOT / "best_candidate_retrieval_ranking_matrix.npz",
        {"state_ids": np.arange(STATE_COUNT), "R_C2_Aend": best_retrieval["ranking"]},
    )

    baseline_statewise = _statewise_with_metrics(
        BASELINE_CANDIDATE,
        all_metrics[baseline_index],
        baseline_retrieval,
        ilc_counts,
    )
    best_statewise = _statewise_with_metrics(
        best_candidate,
        all_metrics[best_index],
        best_retrieval,
        ilc_counts,
    )
    baseline_best = baseline_statewise.loc[
        :,
        [
            "state_id_R",
            "state_id_Q",
            "retrieved_real_B_CNMSE_dB",
            "dpd_shareable",
            "Y_Aend_train_NMSE_dB",
            "Y_Aend_B_NMSE_dB",
            "Y_C2_train_NMSE_dB",
            "Y_C2_B_NMSE_dB",
        ],
    ].rename(
        columns={
            "state_id_Q": "baseline_state_Q",
            "retrieved_real_B_CNMSE_dB": "baseline_real_B",
            "dpd_shareable": "baseline_shareable",
            "Y_Aend_train_NMSE_dB": "baseline_Aend_train",
            "Y_Aend_B_NMSE_dB": "baseline_Aend_B",
            "Y_C2_train_NMSE_dB": "baseline_C2_train",
            "Y_C2_B_NMSE_dB": "baseline_C2_B",
        }
    )
    best_for_compare = best_statewise.loc[
        :,
        [
            "state_id_R",
            "state_id_Q",
            "retrieved_real_B_CNMSE_dB",
            "dpd_shareable",
            "Y_Aend_train_NMSE_dB",
            "Y_Aend_B_NMSE_dB",
            "Y_C2_train_NMSE_dB",
            "Y_C2_B_NMSE_dB",
        ],
    ].rename(
        columns={
            "state_id_Q": "best_state_Q",
            "retrieved_real_B_CNMSE_dB": "best_real_B",
            "dpd_shareable": "best_shareable",
            "Y_Aend_train_NMSE_dB": "best_Aend_train",
            "Y_Aend_B_NMSE_dB": "best_Aend_B",
            "Y_C2_train_NMSE_dB": "best_C2_train",
            "Y_C2_B_NMSE_dB": "best_C2_B",
        }
    )
    baseline_vs_best = baseline_best.merge(
        best_for_compare, on="state_id_R", how="inner", validate="one_to_one"
    )
    baseline_vs_best["retrieval_transition"] = np.select(
        [
            baseline_vs_best["baseline_shareable"] & baseline_vs_best["best_shareable"],
            baseline_vs_best["baseline_shareable"] & ~baseline_vs_best["best_shareable"],
            ~baseline_vs_best["baseline_shareable"] & baseline_vs_best["best_shareable"],
        ],
        ["success_to_success", "success_to_failure", "failure_to_success"],
        default="failure_to_failure",
    )
    baseline_vs_best["delta_real_B_dB"] = np.where(
        np.isfinite(baseline_vs_best["baseline_real_B"])
        & np.isfinite(baseline_vs_best["best_real_B"]),
        baseline_vs_best["baseline_real_B"] - baseline_vs_best["best_real_B"],
        np.nan,
    )
    _write_frame(baseline_vs_best, RESULT_ROOT / "baseline_vs_best_statewise.csv")

    failure_detail = baseline_vs_best.loc[
        baseline_vs_best["state_id_R"].isin(BASELINE_FAILURE_IDS)
    ].copy()
    _write_frame(failure_detail, RESULT_ROOT / "baseline_failure_rescue_detail.csv")
    transition_detail = baseline_vs_best.loc[
        baseline_vs_best["retrieval_transition"].isin(["failure_to_success", "success_to_failure"])
    ].copy()
    for name in ("Aend_train", "Aend_B", "C2_train", "C2_B"):
        transition_detail[f"delta_{name}"] = (
            transition_detail[f"baseline_{name}"] - transition_detail[f"best_{name}"]
        )
        transition_detail[f"model_{name}_worsened"] = (
            transition_detail[f"delta_{name}"] < -NUMERIC_TOLERANCE_DB
        )
    _write_frame(transition_detail, RESULT_ROOT / "best_failure_transition_detail.csv")

    rank_detail_rows: list[dict[str, Any]] = []
    for state_id in BASELINE_FAILURE_IDS:
        for label, candidate, result in (
            ("Baseline", BASELINE_CANDIDATE, baseline_retrieval),
            ("Best", best_candidate, best_retrieval),
        ):
            order = result["ranking_order"][state_id]
            for rank in (1, 2, 3, 5):
                state_q = int(order[rank - 1])
                real_b = float(real_b_distance[state_id, state_q])
                rank_detail_rows.append(
                    {
                        "state_id_R": state_id,
                        "model": label,
                        "candidate_id": candidate.candidate_id,
                        "lut_rank": rank,
                        "state_id_Q": state_q,
                        "query_to_lut_CNMSE_dB": float(result["distance"][state_id, state_q]),
                        "real_B_CNMSE_dB": real_b,
                        "dpd_shareable": bool(real_b < DPD_SHAREABLE_THRESHOLD_DB),
                    }
                )
    _write_frame(pd.DataFrame(rank_detail_rows), RESULT_ROOT / "baseline_failure_topk_detail.csv")

    support_control = _support_control((baseline_fp_a, baseline_fp_c), real_b_distance)
    _write_frame(support_control, RESULT_ROOT / "support_control_summary.csv")

    analysis_rows: list[dict[str, Any]] = []
    for model_metric, base_column, best_column in (
        ("Y_Aend_train_NMSE_dB", "baseline_Aend_train", "best_Aend_train"),
        ("Y_Aend_B_NMSE_dB", "baseline_Aend_B", "best_Aend_B"),
        ("Y_C2_train_NMSE_dB", "baseline_C2_train", "best_C2_train"),
        ("Y_C2_B_NMSE_dB", "baseline_C2_B", "best_C2_B"),
    ):
        mask = np.isfinite(baseline_vs_best["baseline_real_B"]) & np.isfinite(
            baseline_vs_best["best_real_B"]
        )
        stats = safe_correlation(
            (
                baseline_vs_best.loc[mask, base_column] - baseline_vs_best.loc[mask, best_column]
            ).to_numpy(dtype=float),
            baseline_vs_best.loc[mask, "delta_real_B_dB"].to_numpy(dtype=float),
        )
        analysis_rows.append({"model_metric": model_metric, "subset": "finite_finite", **stats})
    _write_frame(pd.DataFrame(analysis_rows), RESULT_ROOT / "model_change_vs_retrieval_change.csv")

    raw_scalars = load_raw_scalar_metrics(range(STATE_COUNT))
    best_excel_frame = _build_best_excel_source(
        best_statewise.loc[
            :,
            [
                "state_id_R",
                "Y_Aend_train_NMSE_dB",
                "Y_Aend_B_NMSE_dB",
                "Y_C2_train_NMSE_dB",
                "Y_C2_B_NMSE_dB",
            ],
        ].rename(columns={"state_id_R": "state_id"}),
        best_statewise.loc[
            :, ["state_id_R", "state_id_Q", "exact_hit", "retrieved_real_B_CNMSE_dB"]
        ],
        raw_scalars,
        state_frame,
    )
    _write_frame(best_excel_frame, RESULT_ROOT / "best_candidate_state_summary.csv")

    print("Step 6/7: generate six Python/matplotlib evidence figures", flush=True)
    figure_details = {
        "figure1": plot_candidate_failure_count(
            ranking_frame,
            best_candidate.candidate_id,
            RESULT_ROOT / "figure1_candidate_failure_count.png",
        ),
        "figure2": plot_candidate_shareable_rate(
            ranking_frame,
            best_candidate.candidate_id,
            RESULT_ROOT / "figure2_candidate_shareable_rate.png",
        ),
        "figure3": plot_baseline_vs_best_real_b(
            baseline_vs_best,
            best_candidate.candidate_id,
            RESULT_ROOT / "figure3_baseline_vs_best_real_B.png",
        ),
        "figure4": plot_failure_detail(
            baseline_vs_best, RESULT_ROOT / "figure4_original_failure_detail.png"
        ),
        "figure5": plot_model_metric_comparison(
            baseline_vs_best, RESULT_ROOT / "figure5_model_metric_comparison.png"
        ),
        "figure6": plot_metric_change_vs_retrieval_change(
            baseline_vs_best, RESULT_ROOT / "figure6_model_change_vs_retrieval_change.png"
        ),
    }

    print("Step 7/7: validation, protection and logs", flush=True)
    after = _protection_snapshot()
    protection = _verify_protection(before, after)
    if not protection["all_protected_unchanged"]:
        raise RuntimeError("raw或受保护旧结果/脚本发生变化，拒绝完成扫描")
    transition_counts = baseline_vs_best["retrieval_transition"].value_counts().to_dict()
    best_failure_ids = (
        best_statewise.loc[best_statewise["failure"], "state_id_R"].astype(int).tolist()
    )
    failure_to_success = transition_detail.loc[
        transition_detail["retrieval_transition"] == "failure_to_success"
    ]
    worsened_counts = {
        column: int(failure_to_success[column].sum())
        for column in [
            "model_Aend_train_worsened",
            "model_Aend_B_worsened",
            "model_C2_train_worsened",
            "model_C2_B_worsened",
        ]
    }
    validation_payload = {
        "study": "retrieval-oriented unified MP model scan",
        "study_type": "retrieval-oriented oracle/post-hoc scan",
        "scenario": 2,
        "ABC": "original unequal ABC",
        "A_length": FORMAL_A_LENGTH,
        "B_length": FORMAL_B_LENGTH,
        "C_length": FORMAL_C_LENGTH,
        "query": "Y-C2",
        "lut": "Y-Aend",
        "state_count": STATE_COUNT,
        "query_count_per_candidate": STATE_COUNT,
        "lut_entries_per_query": STATE_COUNT,
        "full_candidate_search": True,
        "self_match": True,
        "distance_matrix_shape": [STATE_COUNT, STATE_COUNT],
        "distance_count_per_candidate": STATE_COUNT * STATE_COUNT,
        "candidate_count": len(CANDIDATES),
        "valid_candidate_count": len(completed_ids),
        "invalid_candidate_ids": sorted(invalid_ids),
        "order_profiles": {name: list(values) for name, values in ORDER_PROFILES.items()},
        "memory_profiles": MEMORY_PROFILES,
        "max_delays": sorted({candidate.max_delay for candidate in CANDIDATES}),
        "lambda_grid": list(LAMBDA_GRID),
        "Aend_and_C2_share_same_model": True,
        "model_metrics_used_for_ranking": False,
        "real_B_used_for_ranking": True,
        "candidate_ranking_rule": [
            "failure_count ascending",
            "worst_finite_retrieved_real_B_CNMSE_dB ascending",
            "retrieved_real_B_q95_dB ascending",
            "nonexact_real_B_median_dB ascending",
            "exact_hit_count descending",
            "n_complex_coefficients ascending",
            "candidate_id ascending",
        ],
        "threshold_dB": DPD_SHAREABLE_THRESHOLD_DB,
        "threshold_rule": "Real-B CNMSE < -40 dB",
        "baseline": baseline_regression,
        "best_retrieval_candidate": best_payload,
        "zero_failure_candidate_count": int(zero_failure.shape[0]),
        "best_failure_ids": best_failure_ids,
        "baseline_failure_ids": list(BASELINE_FAILURE_IDS),
        "old_failure_rescued_count": int(best_row["old_failure_rescued_count"]),
        "new_failure_created_count": int(best_row["new_failure_created_count"]),
        "net_success_gain": int(best_row["net_success_gain"]),
        "transition_counts": {str(key): int(value) for key, value in transition_counts.items()},
        "state_Q_changed_count": int(
            np.count_nonzero(
                baseline_vs_best["baseline_state_Q"] != baseline_vs_best["best_state_Q"]
            )
        ),
        "failure_to_success_model_worsened_counts": worsened_counts,
        "failure_to_success_count": int(failure_to_success.shape[0]),
        "model_change_vs_retrieval_change": analysis_rows,
        "support_control": support_control.to_dict("records"),
        "real_B_matrix_reused": True,
        "real_B_matrix_source": str(BASELINE_ROOT / "real_B_distance_matrix.npz"),
        "real_B_matrix_shape": list(real_b_distance.shape),
        "real_B_diagonal_neg_inf": bool(np.all(np.isneginf(np.diag(real_b_distance)))),
        "c2_available_count": int(np.count_nonzero(ilc_counts >= C2_STAGE)),
        "Aend_stage_distribution": {
            str(int(key)): int(value)
            for key, value in pd.Series(ilc_counts).value_counts().sort_index().items()
        },
        "statewise_rows": int(statewise_all.shape[0]),
        "candidate_progress": progress,
        "raw_data_modified": False,
        "protected_results_unchanged": protection["all_protected_unchanged"],
        "protection_before": before,
        "protection_after": after,
        "protection_verification": protection,
        "figures": figure_details,
        "output_files": {
            path.name: str(path) for path in sorted(RESULT_ROOT.iterdir()) if path.is_file()
        },
    }
    _write_json(RESULT_ROOT / "validation.json", validation_payload)
    _write_json(
        RESULT_ROOT / "analysis_summary.json",
        {
            "best_retrieval_candidate": best_payload,
            "baseline": baseline_regression,
            "zero_failure_candidate_count": int(zero_failure.shape[0]),
            "best_failure_ids": best_failure_ids,
            "transition_counts": validation_payload["transition_counts"],
            "support_control": validation_payload["support_control"],
        },
    )
    _append_log(
        "\n".join(
            [
                "研究类型：retrieval-oriented oracle/post-hoc scan；"
                "按最终Real-B检索结果排名，模型NMSE不参与pruning或排名。",
                f"原正式ABC={FORMAL_A_LENGTH}/{FORMAL_B_LENGTH}/{FORMAL_C_LENGTH}；候选={len(CANDIDATES)}；有效={len(completed_ids)}；"
                f"每候选Query/LUT={STATE_COUNT}/{STATE_COUNT}；距离矩阵={STATE_COUNT}x{STATE_COUNT}。",
                "Baseline regression="
                + json.dumps(baseline_regression, ensure_ascii=False, separators=(",", ":")),
                "Best=" + json.dumps(best_payload, ensure_ascii=False, separators=(",", ":")),
                f"Best failure IDs={best_failure_ids}；"
                f"transition={validation_payload['transition_counts']}；"
                f"State_Q changed={validation_payload['state_Q_changed_count']}。",
                f"Support-control failure counts={support_control['failure_count'].tolist()}；"
                f"raw/受保护结果验证={protection['all_protected_unchanged']}。",
            ]
        )
    )
    _append_handoff(
        f"完成Scenario 2 retrieval-oriented统一MP扫描：200 candidates，完整425×425 Query-LUT检索，"
        f"Best candidate={best_candidate.order_profile}/{best_candidate.memory_profile}/"
        f"lambda={best_candidate.ridge_lambda}，"
        f"shareable={best_payload['shareable_count']}/{STATE_COUNT}，failure={best_payload['failure_count']}，"
        f"zero-failure candidates={zero_failure.shape[0]}；Real-B严格用于oracle/post-hoc排名；"
        f"raw及既有正式结果保护通过；结果写入{RESULT_ROOT}。"
    )
    print("Scenario 2 retrieval-oriented unified MP model scan completed.", flush=True)
    print(f"Results: {RESULT_ROOT}", flush=True)


if __name__ == "__main__":
    main()
