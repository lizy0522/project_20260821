"""Train-only RMS-normalized uniform-Ridge scan around the frozen MP model.

The model basis is fixed to the ten-term Frozen MP (orders [1, 2, 3, 5, 7,
9], memory [3, 2, 2, 1, 1, 1]).  Only one scalar Ridge value is selected by
contiguous A/C blocked cross-validation on the fixed fourteen failure states.
The discovery payload passed to workers contains A/C arrays only.  B targets
are loaded again only after the selected lambda has been frozen.

The task intentionally writes only the three requested result files.  All
other diagnostics are kept in memory and summarized in ``final_result_summary``.
"""

# ruff: noqa: E402,E501,I001

from __future__ import annotations

import hashlib
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from multiprocessing import freeze_support
from pathlib import Path
from typing import Any

for _thread_name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_name] = "1"

import numpy as np
import pandas as pd

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from behavior_model import frozen_neighborhood_memory_ridge_scan as frozen_model
from behavior_model import sparse_gmp
from behavior_model.evaluation import calculate_nmse


TASK_NAME = "scenario_2_failed14_frozen_mp_normalized_uniform_ridge_5B"
RESULT_ROOT = PROJECT_ROOT / "results" / "behavior_model" / TASK_NAME
WORK_LOG = PROJECT_ROOT / "work_logs" / "behavior_model" / "execution_log.txt"
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "codex_handoff" / "codex_handoff.txt"

FAILED_STATE_IDS = (
    187,
    189,
    195,
    196,
    199,
    206,
    323,
    327,
    330,
    335,
    340,
    344,
    346,
    354,
)
ORDERS = (1, 2, 3, 5, 7, 9)
MEMORY = {1: 3, 2: 2, 3: 2, 5: 1, 7: 1, 9: 1}
K = 10
DMAX = 2
CV_FOLDS = 3
CPU_TARGET = 0.90
FROZEN_LAMBDA = 1e-8
TIE_TOLERANCE_DB = 1e-9
FROZEN_REFERENCE_ROOT = PROJECT_ROOT / "results" / "behavior_model" / "scenario_2_failed14_frozen_neighborhood_memory_ridge_scan_5B"
FROZEN_REFERENCE_FILE = FROZEN_REFERENCE_ROOT / "tables" / "frozen_baseline_regression.csv"
PROTECTED_RESULT_DIRS = {
    "formal_retrieval": PROJECT_ROOT / "results" / "behavior_fingerprint_lut_retrieval" / "scenario_2_C2_to_Aend",
    "all_ilc_frozen": PROJECT_ROOT / "results" / "behavior_fingerprint_ranking_consistency" / "scenario_2_all_ilc",
    "unified_capacity": PROJECT_ROOT / "results" / "behavior_model" / "scenario_2_unified_odd_order_mp_capacity_scan_5B",
    "odd_order_scan": PROJECT_ROOT / "results" / "behavior_model" / "scenario_2_failed14_odd_order_mp_order_dependent_memory_scan_5B",
    "p5_ridge": PROJECT_ROOT / "results" / "behavior_model" / "scenario_2_failed14_P5_variable_memory_ridge_scan_5B",
    "p5_order2": PROJECT_ROOT / "results" / "behavior_model" / "scenario_2_failed14_P5_with_order2_variable_memory_OLS_scan_5B",
    "frozen_neighborhood": FROZEN_REFERENCE_ROOT,
    "sparse_gmp": PROJECT_ROOT / "results" / "behavior_model" / "scenario_2_failed14_frozen_mp_residual_sparse_gmp_5B",
    "multibranch": PROJECT_ROOT / "results" / "behavior_model" / "scenario_2_failed14_multibranch_independent_basis_OLS_scan_5B",
}

LAMBDA_GRID = np.concatenate(
    [
        np.asarray([0.0], dtype=np.float64),
        10.0 ** np.arange(-12.0, -2.0 + 1e-12, 0.25, dtype=np.float64),
    ]
)

_WORKER_DISCOVERY: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.17g", na_rep="NaN")


def _tree_digest(path: Path) -> str | None:
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    for file in sorted(path.rglob("*"), key=lambda item: str(item).lower()):
        if not file.is_file() or "__pycache__" in file.parts or file.suffix.lower() == ".pyc":
            continue
        content = file.read_bytes()
        digest.update(file.relative_to(path).as_posix().encode("utf-8") + b"\0")
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def _raw_manifest() -> dict[str, Any]:
    raw_root = PROJECT_ROOT / "data" / "raw"
    if not raw_root.is_dir():
        return {"exists": False}
    files = [file for file in raw_root.rglob("*") if file.is_file()]
    return {
        "exists": True,
        "file_count": len(files),
        "bytes": int(sum(file.stat().st_size for file in files)),
        "sha256": _tree_digest(raw_root),
    }


def _snapshot() -> dict[str, Any]:
    return {
        "data/raw": _raw_manifest(),
        "result_dirs": {name: _tree_digest(path) for name, path in PROTECTED_RESULT_DIRS.items()},
    }


def _protection_check(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    raw_before = before["data/raw"]
    raw_after = after["data/raw"]
    raw_ok = raw_before == raw_after
    result_checks = {}
    for name in PROTECTED_RESULT_DIRS:
        result_checks[name] = {
            "sha256_unchanged": before["result_dirs"].get(name) == after["result_dirs"].get(name),
            "before": before["result_dirs"].get(name),
            "after": after["result_dirs"].get(name),
        }
    return {
        "data_raw_unchanged": raw_ok,
        "protected_result_dirs_unchanged": bool(all(item["sha256_unchanged"] for item in result_checks.values())),
        "all_protected_unchanged": bool(raw_ok and all(item["sha256_unchanged"] for item in result_checks.values())),
        "data_raw": {"before": raw_before, "after": raw_after},
        "result_dirs": result_checks,
    }


def _fit_augmented(phi: np.ndarray, y: np.ndarray, lam: float) -> tuple[np.ndarray, np.ndarray, int, np.ndarray, float]:
    phi = np.asarray(phi, dtype=np.complex128)
    y = np.asarray(y, dtype=np.complex128).reshape(-1)
    if phi.ndim != 2 or y.ndim != 1 or phi.shape[0] != y.size:
        raise ValueError("invalid least-squares dimensions")
    if not np.all(np.isfinite(phi)) or not np.all(np.isfinite(y)):
        raise ValueError("non-finite least-squares inputs")
    n_samples, n_columns = phi.shape
    lam = float(lam)
    if lam < 0 or not np.isfinite(lam):
        raise ValueError("lambda must be finite and non-negative")
    if lam == 0.0:
        theta, _, rank, singular = np.linalg.lstsq(phi, y, rcond=None)
        singular = np.asarray(singular, dtype=np.float64)
        objective_penalty = 0.0
    else:
        phi_aug = np.vstack((phi, np.sqrt(n_samples * lam) * np.eye(n_columns, dtype=np.complex128)))
        y_aug = np.concatenate((y, np.zeros(n_columns, dtype=np.complex128)))
        theta, _, rank, singular_aug = np.linalg.lstsq(phi_aug, y_aug, rcond=None)
        singular_aug = np.asarray(singular_aug, dtype=np.float64)
        singular = np.asarray(np.linalg.svd(phi, compute_uv=False), dtype=np.float64)
        objective_penalty = float(lam * np.linalg.norm(theta) ** 2)
    theta = np.asarray(theta, dtype=np.complex128)
    return theta, singular, int(rank), np.asarray(singular, dtype=np.float64), objective_penalty


def _fit_normalized_ridge(phi_train: np.ndarray, y_train: np.ndarray, lam: float) -> dict[str, Any]:
    phi_train = np.asarray(phi_train, dtype=np.complex128)
    y_train = np.asarray(y_train, dtype=np.complex128).reshape(-1)
    scale = np.sqrt(np.mean(np.abs(phi_train) ** 2, axis=0))
    if not np.all(np.isfinite(scale)) or np.any(scale <= 0):
        raise RuntimeError("Invalid basis RMS scale.")
    phi_norm = phi_train / scale[None, :]
    theta, singular, rank, _, penalty = _fit_augmented(phi_norm, y_train, lam)
    singular_norm = np.asarray(np.linalg.svd(phi_norm, compute_uv=False), dtype=np.float64)
    return {
        "beta": theta,
        "scale": scale,
        "rank": int(rank),
        "singular": singular_norm,
        "ridge_penalty": penalty,
    }


def _fit_raw_ridge(phi_train: np.ndarray, y_train: np.ndarray, lam: float) -> dict[str, Any]:
    theta, singular, rank, _, penalty = _fit_augmented(phi_train, y_train, lam)
    return {
        "theta": theta,
        "rank": int(rank),
        "singular": np.asarray(singular, dtype=np.float64),
        "ridge_penalty": penalty,
    }


def _condition_number(singular: np.ndarray) -> float:
    singular = np.asarray(singular, dtype=np.float64)
    if singular.size == 0 or singular[-1] <= 0:
        return float("inf")
    return float(singular[0] / singular[-1])


def _load_reference() -> pd.DataFrame:
    if not FROZEN_REFERENCE_FILE.is_file():
        raise FileNotFoundError(f"Frozen reference file missing: {FROZEN_REFERENCE_FILE}")
    reference = pd.read_csv(FROZEN_REFERENCE_FILE)
    reference["state_id"] = reference["state_id"].astype(int)
    expected = set(FAILED_STATE_IDS)
    if set(reference["state_id"]) != expected:
        raise RuntimeError("Frozen reference does not contain exactly the fixed fourteen states")
    return reference.set_index("state_id").loc[list(FAILED_STATE_IDS)].reset_index()


def _build_basis(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    y = np.asarray(y, dtype=np.complex128).reshape(-1)
    phi = sparse_gmp.build_frozen_mp_basis(x)
    y_valid = y[DMAX:]
    if phi.shape != (y_valid.size, K):
        raise RuntimeError(f"Frozen basis/support mismatch: phi={phi.shape}, y={y_valid.shape}")
    return phi, y_valid


def _prepare_discovery_payload() -> tuple[dict[int, dict[str, tuple[np.ndarray, np.ndarray]]], list[dict[str, Any]]]:
    payload: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    records: list[dict[str, Any]] = []
    for state_id in FAILED_STATE_IDS:
        prepared = frozen_model.prepare_state(state_id)
        a_phi, a_y = _build_basis(prepared.aend_A_input, prepared.aend_A_output)
        c_phi, c_y = _build_basis(prepared.c2_C_input, prepared.c2_C_output)
        payload[state_id] = {"Aend": (a_phi, a_y), "C2": (c_phi, c_y)}
        records.append(
            {
                "state_id": state_id,
                "ilc_A_end": int(prepared.ilc_A_end),
                "A_length": int(prepared.aend_A_input.size),
                "C_length": int(prepared.c2_C_input.size),
                "A_valid_length": int(a_phi.shape[0]),
                "C_valid_length": int(c_phi.shape[0]),
                "B_target_retained": False,
            }
        )
        del prepared
    return payload, records


def _worker_init(payload: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]]) -> None:
    global _WORKER_DISCOVERY
    _WORKER_DISCOVERY = payload


def _blocked_indices(length: int) -> list[np.ndarray]:
    if length <= 0:
        raise ValueError("segment must contain samples")
    return [np.asarray(block, dtype=np.int64) for block in np.array_split(np.arange(length, dtype=np.int64), CV_FOLDS)]


def _cv_task(task: tuple[int, float, int, str]) -> dict[str, Any]:
    lambda_id, lam, state_id, side = task
    phi, y = _WORKER_DISCOVERY[state_id][side]
    blocks = _blocked_indices(phi.shape[0])
    fold_values: list[float] = []
    rank_values: list[int] = []
    for validation_block in range(CV_FOLDS):
        validation_index = blocks[validation_block]
        train_index = np.concatenate([blocks[index] for index in range(CV_FOLDS) if index != validation_block])
        fit = _fit_normalized_ridge(phi[train_index], y[train_index], lam)
        prediction = (phi[validation_index] / fit["scale"][None, :]) @ fit["beta"]
        fold_values.append(float(calculate_nmse(y[validation_index], prediction)))
        rank_values.append(int(fit["rank"]))
    return {
        "lambda_id": int(lambda_id),
        "lambda": float(lam),
        "state_id": int(state_id),
        "side": side,
        "cv_NMSE_dB": float(np.mean(fold_values)),
        "fold_count": CV_FOLDS,
        "rank_min": int(min(rank_values)),
    }


def _scan_lambda_grid(payload: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]], workers: int) -> tuple[pd.DataFrame, dict[str, Any]]:
    tasks = [
        (lambda_id, float(lam), state_id, side)
        for lambda_id, lam in enumerate(LAMBDA_GRID)
        for state_id in FAILED_STATE_IDS
        for side in ("Aend", "C2")
    ]
    rows: list[dict[str, Any]] = []
    errors: list[str] = []
    print(f"normalized Ridge CV tasks: {len(tasks)}; workers={workers}", flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_init, initargs=(payload,)) as executor:
        futures = [executor.submit(_cv_task, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), start=1):
            try:
                rows.append(future.result())
            except Exception as exc:  # pragma: no cover - exercised only on worker failure
                errors.append(f"{type(exc).__name__}: {exc}")
            if done % 200 == 0 or done == len(futures):
                print(f"normalized Ridge CV tasks: {done}/{len(futures)}", flush=True)
    if errors:
        raise RuntimeError("normalized Ridge worker failure: " + " | ".join(errors[:5]))
    detail = pd.DataFrame(rows).sort_values(["lambda_id", "state_id", "side"]).reset_index(drop=True)
    summary_rows: list[dict[str, Any]] = []
    for lambda_id, lam in enumerate(LAMBDA_GRID):
        current = detail.loc[detail["lambda_id"] == lambda_id]
        a = current.loc[current["side"] == "Aend", "cv_NMSE_dB"].to_numpy(dtype=float)
        c = current.loc[current["side"] == "C2", "cv_NMSE_dB"].to_numpy(dtype=float)
        if a.size != len(FAILED_STATE_IDS) or c.size != len(FAILED_STATE_IDS):
            raise RuntimeError(f"lambda {lam} does not have 14 Aend and 14 C2 values")
        a_median, c_median = float(np.median(a)), float(np.median(c))
        a_q75, c_q75 = float(np.quantile(a, 0.75)), float(np.quantile(c, 0.75))
        a_worst, c_worst = float(np.max(a)), float(np.max(c))
        summary_rows.append(
            {
                "lambda": float(lam),
                "A_CV_median_dB": a_median,
                "A_CV_q75_dB": a_q75,
                "A_CV_worst_dB": a_worst,
                "C_CV_median_dB": c_median,
                "C_CV_q75_dB": c_q75,
                "C_CV_worst_dB": c_worst,
                "BalancedMedian_dB": max(a_median, c_median),
                "BalancedQ75_dB": max(a_q75, c_q75),
                "BalancedWorst_dB": max(a_worst, c_worst),
                "rank_min": int(current["rank_min"].min()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    best = float(summary["BalancedMedian_dB"].min())
    ties = summary.loc[np.abs(summary["BalancedMedian_dB"] - best) < TIE_TOLERANCE_DB].copy()
    ties = ties.sort_values(["BalancedQ75_dB", "BalancedWorst_dB", "lambda"], ascending=[True, True, True], kind="mergesort")
    selected_lambda = float(ties.iloc[0]["lambda"])
    summary["is_selected"] = np.isclose(summary["lambda"].to_numpy(dtype=float), selected_lambda, rtol=0.0, atol=0.0)
    selection_info = {
        "selected_lambda": selected_lambda,
        "best_balanced_median_dB": best,
        "tie_count_within_1e-9_dB": int(ties.shape[0]),
        "task_count": len(tasks),
        "fit_count": len(tasks) * CV_FOLDS,
        "detail_rows_in_memory": int(detail.shape[0]),
        "B_target_used_for_selection": False,
    }
    return summary, selection_info


def _normalized_ols_equivalence(payload: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]]) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for state_id in (187, 340):
        for side in ("Aend", "C2"):
            phi, y = payload[state_id][side]
            raw_theta, _, raw_rank, _, _ = _fit_augmented(phi, y, 0.0)
            normalized = _fit_normalized_ridge(phi, y, 0.0)
            pred_raw = phi @ raw_theta
            pred_norm = (phi / normalized["scale"][None, :]) @ normalized["beta"]
            raw_nmse = float(calculate_nmse(y, pred_raw))
            norm_nmse = float(calculate_nmse(y, pred_norm))
            relative_prediction_error = float(np.linalg.norm(pred_raw - pred_norm) / max(np.linalg.norm(pred_raw), 1e-30))
            checks.append(
                {
                    "state_id": state_id,
                    "side": side,
                    "raw_rank": int(raw_rank),
                    "normalized_rank": int(normalized["rank"]),
                    "raw_NMSE_dB": raw_nmse,
                    "normalized_OLS_NMSE_dB": norm_nmse,
                    "NMSE_difference_dB": norm_nmse - raw_nmse,
                    "prediction_relative_error": relative_prediction_error,
                    "pass": bool(abs(norm_nmse - raw_nmse) < 1e-9 or relative_prediction_error < 1e-10),
                }
            )
    return {"pass": bool(all(item["pass"] for item in checks)), "checks": checks}


def _frozen_reproduction_gate(reference: pd.DataFrame) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for state_id in FAILED_STATE_IDS:
        prepared = frozen_model.prepare_state(state_id)
        a_phi, a_y = _build_basis(prepared.aend_A_input, prepared.aend_A_output)
        c_phi, c_y = _build_basis(prepared.c2_C_input, prepared.c2_C_output)
        a_b_phi, a_b_y = _build_basis(prepared.aend_B_input, prepared.aend_B_output)
        c_b_phi, c_b_y = _build_basis(prepared.c2_B_input, prepared.c2_B_output)
        a_fit = _fit_raw_ridge(a_phi, a_y, FROZEN_LAMBDA)
        c_fit = _fit_raw_ridge(c_phi, c_y, FROZEN_LAMBDA)
        current = {
            "state_id": state_id,
            "ilc_A_end_current": int(prepared.ilc_A_end),
            "ilc_A_end_formal": int(reference.loc[reference["state_id"] == state_id, "ilc_A_end_formal"].iloc[0]),
            "Aend_train_current": float(calculate_nmse(a_y, a_phi @ a_fit["theta"])),
            "Aend_B_current": float(calculate_nmse(a_b_y, a_b_phi @ a_fit["theta"])),
            "C2_train_current": float(calculate_nmse(c_y, c_phi @ c_fit["theta"])),
            "C2_B_current": float(calculate_nmse(c_b_y, c_b_phi @ c_fit["theta"])),
        }
        formal = reference.loc[reference["state_id"] == state_id].iloc[0]
        for metric in ("Aend_train", "Aend_B", "C2_train", "C2_B"):
            current[f"{metric}_formal"] = float(formal[f"{metric}_formal"])
            current[f"{metric}_error_dB"] = float(current[f"{metric}_current"] - current[f"{metric}_formal"])
        current["state_pass"] = bool(
            current["ilc_A_end_current"] == current["ilc_A_end_formal"]
            and max(abs(current[f"{metric}_error_dB"]) for metric in ("Aend_train", "Aend_B", "C2_train", "C2_B")) <= 1e-6
        )
        rows.append(current)
        del prepared
    frame = pd.DataFrame(rows)
    return {
        "pass": bool(frame["state_pass"].all()),
        "maximum_abs_error_dB": float(frame[[f"{metric}_error_dB" for metric in ("Aend_train", "Aend_B", "C2_train", "C2_B")]].abs().to_numpy().max()),
        "rows": frame,
    }


def _fit_model_set(prepared: Any, lam_selected: float) -> tuple[dict[str, Any], dict[str, float]]:
    a_phi, a_y = _build_basis(prepared.aend_A_input, prepared.aend_A_output)
    c_phi, c_y = _build_basis(prepared.c2_C_input, prepared.c2_C_output)
    a_b_phi, a_b_y = _build_basis(prepared.aend_B_input, prepared.aend_B_output)
    c_b_phi, c_b_y = _build_basis(prepared.c2_B_input, prepared.c2_B_output)
    metric_rows: dict[str, Any] = {"state_id": int(prepared.state_id), "ilc_A_end": int(prepared.ilc_A_end)}
    condition: dict[str, float] = {}
    for label, train_phi, train_y, b_phi, b_y in (
        ("A", a_phi, a_y, a_b_phi, a_b_y),
        ("C", c_phi, c_y, c_b_phi, c_b_y),
    ):
        raw_fit = _fit_raw_ridge(train_phi, train_y, FROZEN_LAMBDA)
        raw_train_pred = train_phi @ raw_fit["theta"]
        raw_b_pred = b_phi @ raw_fit["theta"]
        norm_ols = _fit_normalized_ridge(train_phi, train_y, 0.0)
        norm_ridge = _fit_normalized_ridge(train_phi, train_y, FROZEN_LAMBDA)
        selected = _fit_normalized_ridge(train_phi, train_y, lam_selected)
        raw_condition = _condition_number(np.asarray(np.linalg.svd(train_phi, compute_uv=False), dtype=np.float64))
        norm_condition = _condition_number(norm_ols["singular"])
        condition[f"{label}_raw_condition"] = raw_condition
        condition[f"{label}_normalized_condition"] = norm_condition
        for model, fit, prefix in (
            ("Frozen", raw_fit, "Frozen"),
            ("NormOLS", norm_ols, "NormOLS"),
            ("NormRidge1e8", norm_ridge, "NormRidge1e8"),
            ("Selected", selected, "Selected"),
        ):
            if model == "Frozen":
                train_prediction = raw_train_pred
                b_prediction = raw_b_pred
            else:
                train_prediction = (train_phi / fit["scale"][None, :]) @ fit["beta"]
                b_prediction = (b_phi / fit["scale"][None, :]) @ fit["beta"]
            metric_rows[f"{prefix}_{label}_train"] = float(calculate_nmse(train_y, train_prediction))
            metric_rows[f"{prefix}_{label}_B"] = float(calculate_nmse(b_y, b_prediction))
            metric_rows[f"{prefix}_{label}_gap"] = metric_rows[f"{prefix}_{label}_B"] - metric_rows[f"{prefix}_{label}_train"]
            metric_rows[f"{prefix}_{label}_rank"] = int(raw_fit["rank"] if model == "Frozen" else fit["rank"])
        metric_rows[f"{label}_raw_basis_condition_number"] = raw_condition
        metric_rows[f"{label}_normalized_basis_condition_number"] = norm_condition
    for model in ("Frozen", "NormOLS", "NormRidge1e8", "Selected"):
        metric_rows[f"{model}_worst4"] = max(metric_rows[f"{model}_A_train"], metric_rows[f"{model}_A_B"], metric_rows[f"{model}_C_train"], metric_rows[f"{model}_C_B"])
    for suffix in ("A_train", "A_B", "C_train", "C_B", "worst4"):
        metric_rows[f"Selected_minus_Frozen_{suffix}"] = metric_rows[f"Selected_{suffix}"] - metric_rows[f"Frozen_{suffix}"]
    return metric_rows, condition


def _final_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        output_rows.append(
            {
                "state_id": row["state_id"],
                "Frozen_A_train": row["Frozen_A_train"],
                "Frozen_A_B": row["Frozen_A_B"],
                "Frozen_C_train": row["Frozen_C_train"],
                "Frozen_C_B": row["Frozen_C_B"],
                "Frozen_worst4": row["Frozen_worst4"],
                "NormOLS_A_train": row["NormOLS_A_train"],
                "NormOLS_A_B": row["NormOLS_A_B"],
                "NormOLS_C_train": row["NormOLS_C_train"],
                "NormOLS_C_B": row["NormOLS_C_B"],
                "NormOLS_worst4": row["NormOLS_worst4"],
                "NormRidge1e8_A_train": row["NormRidge1e8_A_train"],
                "NormRidge1e8_A_B": row["NormRidge1e8_A_B"],
                "NormRidge1e8_C_train": row["NormRidge1e8_C_train"],
                "NormRidge1e8_C_B": row["NormRidge1e8_C_B"],
                "NormRidge1e8_worst4": row["NormRidge1e8_worst4"],
                "Selected_A_train": row["Selected_A_train"],
                "Selected_A_B": row["Selected_A_B"],
                "Selected_C_train": row["Selected_C_train"],
                "Selected_C_B": row["Selected_C_B"],
                "Selected_worst4": row["Selected_worst4"],
                "Selected_minus_Frozen_A_train": row["Selected_minus_Frozen_A_train"],
                "Selected_minus_Frozen_A_B": row["Selected_minus_Frozen_A_B"],
                "Selected_minus_Frozen_C_train": row["Selected_minus_Frozen_C_train"],
                "Selected_minus_Frozen_C_B": row["Selected_minus_Frozen_C_B"],
                "Selected_minus_Frozen_worst4": row["Selected_minus_Frozen_worst4"],
            }
        )
    return pd.DataFrame(output_rows).sort_values("state_id").reset_index(drop=True)


def _median_metrics(frame: pd.DataFrame, model: str) -> dict[str, float]:
    return {metric: float(frame[f"{model}_{metric}"].median()) for metric in ("A_train", "A_B", "C_train", "C_B", "worst4")}


def _summary_text(
    lambda_summary: pd.DataFrame,
    selection_info: dict[str, Any],
    final_rows: pd.DataFrame,
    condition_rows: list[dict[str, float]],
    normalized_ols_gate: dict[str, Any],
    frozen_gate: dict[str, Any],
    protection: dict[str, Any],
    worker_count: int,
    discovery_records: list[dict[str, Any]],
) -> str:
    medians = {model: _median_metrics(final_rows, model) for model in ("Frozen", "NormOLS", "NormRidge1e8", "Selected")}
    selected_delta = {metric: float(final_rows[f"Selected_minus_Frozen_{metric}"].median()) for metric in ("A_train", "A_B", "C_train", "C_B", "worst4")}
    selected_worst = final_rows["Selected_minus_Frozen_worst4"]
    selected_condition_raw = [row[key] for row in condition_rows for key in ("A_raw_condition", "C_raw_condition")]
    selected_condition_norm = [row[key] for row in condition_rows for key in ("A_normalized_condition", "C_normalized_condition")]
    lines = [
        TASK_NAME,
        "Train-only RMS-normalized Frozen MP with one shared scalar Ridge lambda.",
        "",
        "Experiment",
        f"- States: {list(FAILED_STATE_IDS)} (exactly 14 Frozen failure states).",
        f"- Frozen orders: {list(ORDERS)}; memory=[3,2,2,1,1,1]; K={K}; dmax={DMAX}.",
        "- Canonical preprocessing: actual input -> full-record Rough/Fine alignment -> fixed ABC ownership -> segment-wise complex-gain adjustment.",
        "- ABC ownership: A=[0:12288), B=[12288:17203), C=[17203:24576); valid lengths A=12286, B=4913, C=7371.",
        "",
        "Normalization and lambda selection",
        "- Basis normalization: train-only per-column RMS scaling; no centering and no target normalization.",
        "- Ridge penalty: one shared scalar lambda for all ten basis columns, all states, and both Aend/C2 sides.",
        f"- Lambda grid: 0 plus 10^(-12:0.25:-2), total {len(LAMBDA_GRID)} values.",
        f"- Selected lambda: {selection_info['selected_lambda']:.17g}; best BalancedMedian={selection_info['best_balanced_median_dB']:.9f} dB; tie_count={selection_info['tie_count_within_1e-9_dB']}.",
        f"- CV workload: {selection_info['task_count']} lambda/state/side tasks and {selection_info['fit_count']} blocked-CV fits; folds=3 contiguous, no shuffle.",
        f"- Discovery records retained only A/C: {len(discovery_records)} states; B target used for lambda selection=False.",
        "",
        "CPU",
        f"- logical_cpu_count={os.cpu_count() or 1}; worker_count={worker_count}; target={CPU_TARGET:.2f}; BLAS threads/worker=1; nested_parallelism=False.",
        "- Parallel task grain: (lambda_id, state_id, side); worker folds executed sequentially.",
        "",
        "Frozen and normalized model medians (dB)",
        "model | A train | A→B | C train | C→B | worst-four",
    ]
    for model in ("Frozen", "NormOLS", "NormRidge1e8", "Selected"):
        item = medians[model]
        lines.append(f"{model} | {item['A_train']:.6f} | {item['A_B']:.6f} | {item['C_train']:.6f} | {item['C_B']:.6f} | {item['worst4']:.6f}")
    lines.extend(
        [
            "",
            "Selected minus Frozen (negative means improvement)",
            *[f"- {metric}: {value:+.6f} dB" for metric, value in selected_delta.items()],
            f"- worst-four improved states: {int((selected_worst < 0).sum())}/{len(selected_worst)}; degraded states: {int((selected_worst > 0).sum())}/{len(selected_worst)}.",
            f"- Aend gap median: Frozen={float(final_rows['Frozen_A_gap'].median()):.6f}, Selected={float(final_rows['Selected_A_gap'].median()):.6f}, delta={float((final_rows['Selected_A_gap']-final_rows['Frozen_A_gap']).median()):+.6f} dB.",
            f"- C2 gap median: Frozen={float(final_rows['Frozen_C_gap'].median()):.6f}, Selected={float(final_rows['Selected_C_gap'].median()):.6f}, delta={float((final_rows['Selected_C_gap']-final_rows['Frozen_C_gap']).median()):+.6f} dB.",
            "",
            "Condition-number diagnostics",
            f"- raw basis condition number median={float(np.median(selected_condition_raw)):.6g}, max={float(np.max(selected_condition_raw)):.6g}.",
            f"- normalized basis condition number median={float(np.median(selected_condition_norm)):.6g}, max={float(np.max(selected_condition_norm)):.6g}.",
            "- These are diagnostics only; condition-number reduction is not treated as model success without A/C→B improvement.",
            "",
            "Gates",
            f"- normalized OLS equivalence gate={normalized_ols_gate['pass']}; Frozen raw Ridge reproduction gate={frozen_gate['pass']} (max abs error={frozen_gate['maximum_abs_error_dB']:.3g} dB).",
            "- B target used during lambda selection=False; B opened only after selected lambda was frozen=True.",
            f"- raw data unchanged={protection['data_raw_unchanged']}; protected historical results unchanged={protection['protected_result_dirs_unchanged']}.",
            "",
            "Conclusion",
            "- Normalized OLS is expected to reproduce raw OLS predictions; the equivalence gate verifies the coordinate-change implementation.",
            "- The final decision is based on cross-segment Aend→B and C2→B behavior, not on condition-number reduction alone.",
            "- This task does not alter the Frozen MP structure, does not save coefficients, and does not perform weighted Ridge, GMP, LUT retrieval, Real-B shareability, low-bandwidth processing, or 425-state validation.",
            "- Whether the normalized Ridge candidate should replace the Frozen baseline remains a separate decision; no automatic replacement was made.",
            "",
            "Result files",
            f"- {RESULT_ROOT / 'lambda_scan_summary.csv'}",
            f"- {RESULT_ROOT / 'final_per_state_metrics.csv'}",
            f"- {RESULT_ROOT / 'final_result_summary.txt'}",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    freeze_support()
    print(f"Starting {TASK_NAME}", flush=True)
    if len(FAILED_STATE_IDS) != 14 or K != 10 or DMAX != 2 or len(LAMBDA_GRID) != 42:
        raise RuntimeError("fixed task definition gate failed")
    before = _snapshot()
    reference = _load_reference()
    logical_cpu_count = int(os.cpu_count() or 1)
    worker_count = max(1, int(math.floor(CPU_TARGET * logical_cpu_count)))
    print(f"logical_cpu_count={logical_cpu_count}; worker_count={worker_count}; target={CPU_TARGET:.2f}", flush=True)

    # Baseline gate is deliberately completed before the normalized lambda scan.
    frozen_gate = _frozen_reproduction_gate(reference)
    if not frozen_gate["pass"]:
        raise RuntimeError(f"Frozen reproduction gate failed: max error={frozen_gate['maximum_abs_error_dB']}")
    print(f"Frozen reproduction gate PASS; max_abs_error_dB={frozen_gate['maximum_abs_error_dB']:.3g}", flush=True)

    discovery_payload, discovery_records = _prepare_discovery_payload()
    normalized_ols_gate = _normalized_ols_equivalence(discovery_payload)
    if not normalized_ols_gate["pass"]:
        raise RuntimeError("normalized OLS equivalence gate failed")
    print("Normalized OLS equivalence gate PASS", flush=True)

    lambda_summary, selection_info = _scan_lambda_grid(discovery_payload, worker_count)
    selected_lambda = float(selection_info["selected_lambda"])
    print(f"Selected normalized lambda={selected_lambda:.17g}; BalancedMedian={selection_info['best_balanced_median_dB']:.6f} dB", flush=True)
    del discovery_payload

    # B is first loaded here, after lambda selection has been frozen.
    final_rows: list[dict[str, Any]] = []
    condition_rows: list[dict[str, float]] = []
    for state_id in FAILED_STATE_IDS:
        prepared = frozen_model.prepare_state(state_id)
        row, condition = _fit_model_set(prepared, selected_lambda)
        final_rows.append(row)
        condition_rows.append(condition)
        del prepared
    final_raw = pd.DataFrame(final_rows).sort_values("state_id").reset_index(drop=True)
    output_frame = _final_frame(final_raw.to_dict("records"))

    # Keep the required output directory minimal: exactly three task result files.
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_frame(lambda_summary.drop(columns=["rank_min"]), RESULT_ROOT / "lambda_scan_summary.csv")
    _write_frame(output_frame, RESULT_ROOT / "final_per_state_metrics.csv")

    after = _snapshot()
    protection = _protection_check(before, after)
    if not protection["all_protected_unchanged"]:
        raise RuntimeError("raw data or protected historical result changed")
    summary = _summary_text(lambda_summary, selection_info, final_raw, condition_rows, normalized_ols_gate, frozen_gate, protection, worker_count, discovery_records)
    (RESULT_ROOT / "final_result_summary.txt").write_text(summary, encoding="utf-8")

    timestamp = datetime.now(UTC).isoformat()
    log = (
        f"{TASK_NAME} completed: states=14, lambda_count={len(LAMBDA_GRID)}, selected_lambda={selected_lambda:.17g}, "
        f"CV_fits={selection_info['fit_count']}, workers={worker_count}, normalized_OLS_gate={normalized_ols_gate['pass']}, "
        f"frozen_reproduction_gate={frozen_gate['pass']}, raw/protected unchanged={protection['all_protected_unchanged']}. "
        f"Only three result files were written under {RESULT_ROOT}; no figures, Excel, cache, coefficients, or retrieval were generated."
    )
    WORK_LOG.parent.mkdir(parents=True, exist_ok=True)
    HANDOFF_LOG.parent.mkdir(parents=True, exist_ok=True)
    with WORK_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n[{timestamp}] {TASK_NAME}\n{log}\n")
    with HANDOFF_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"\n\n{timestamp} | behavior_model 新派生任务：Frozen MP basis RMS归一化 + uniform Ridge\n{log}\n")
    print(log, flush=True)


if __name__ == "__main__":
    main()
