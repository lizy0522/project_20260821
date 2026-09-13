"""Post-hoc diagnosis of the Global-G4 C2-to-B improvement gap.

This task does not select a model.  It reuses the already frozen Global-G4
support and the frozen MP, refits both models for the existing 85-state
post-hoc Validation set, and diagnoses correction transfer and C/B feature
shift.  No LUT retrieval, 425-state characterization, or configuration
replacement is performed.
"""

# Imports intentionally follow the BLAS-thread environment setup below.
# ruff: noqa: E402,I001,E501

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
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from behavior_model import frozen_neighborhood_memory_ridge_scan as frozen_model  # noqa: E402
from behavior_model import run_scenario2_frozen_mp_global_sparse_gmp_residual_5b as global_runner  # noqa: E402
from behavior_model import sparse_gmp as gmp  # noqa: E402


TASK_NAME = "scenario_2_global_sparse_gmp_C2_gap_diagnosis_5B"
RESULT_ROOT = PROJECT_ROOT / "results" / "behavior_model" / TASK_NAME
PREVIOUS_ROOT = PROJECT_ROOT / "results" / "behavior_model" / "scenario_2_frozen_mp_global_sparse_gmp_residual_5B"
PREVIOUS_FORWARD = PREVIOUS_ROOT / "forward_selection_summary.csv"
PREVIOUS_VALIDATION = PREVIOUS_ROOT / "validation_per_state_metrics.csv"
SPLIT_PATH = PROJECT_ROOT / "results" / "behavior_model" / "scenario_2_ridge_analysis" / "state_split.csv"
MODEL_LOG = PROJECT_ROOT / "work_logs" / "behavior_model" / "execution_log.txt"
HANDOFF_LOG = PROJECT_ROOT / "work_logs" / "codex_handoff" / "codex_handoff.txt"

STATE_COUNT = 425
VALIDATION_COUNT = 85
ORDERS = (1, 2, 3, 5, 7, 9)
MEMORY = (3, 2, 2, 1, 1, 1)
FROZEN_K = 10
G4_K = 14
DMAX = 2
RIDGE_LAMBDA = 1e-8
WORKER_TARGET = 0.90
REPRODUCTION_IDS_COUNT = 5
METRIC_TOLERANCE_DB = 1e-6
IDENTITY_TOLERANCE_DB = 1e-9
DECOMPOSITION_TOLERANCE = 1e-10
VALIDATION_GAP_LIMIT_DB = 0.10

EXPECTED_G4_TERM_IDS = (
    "GMP_p02_m00_q01",
    "GMP_p02_m01_q00",
    "GMP_p03_m01_q00",
    "GMP_p02_m00_q02",
)


def _term_lookup() -> dict[str, Any]:
    return {term.term_id: term for term in global_runner.GLOBAL_TERMS}


def _load_frozen_g4_definition() -> dict[str, Any]:
    if not PREVIOUS_FORWARD.is_file() or not PREVIOUS_VALIDATION.is_file():
        raise FileNotFoundError("previous Global Sparse-GMP result files are missing")
    forward = pd.read_csv(PREVIOUS_FORWARD)
    selected_rows = forward.loc[forward["is_final_selected_G"].astype(bool)]
    if selected_rows.shape[0] != 1:
        raise RuntimeError("previous forward summary must contain exactly one selected G")
    selected_row = selected_rows.iloc[0]
    selected_ids = tuple(str(value) for value in str(selected_row["selected_support_ids"]).split(";") if value)
    if int(selected_row["stage_G"]) != 4 or int(selected_row["K_total"]) != G4_K or selected_ids != EXPECTED_G4_TERM_IDS:
        raise RuntimeError(
            f"previous Global G4 definition mismatch: G={selected_row['stage_G']}, "
            f"K={selected_row['K_total']}, support={selected_ids}"
        )
    lookup = _term_lookup()
    if any(term_id not in lookup for term_id in selected_ids):
        raise RuntimeError("previous selected G4 term is absent from current GMP dictionary")
    return {
        "selected_G": 4,
        "K_total": G4_K,
        "term_ids": selected_ids,
        "terms": tuple(lookup[term_id] for term_id in selected_ids),
        "previous_best_balanced_median": float(forward["balanced_median"].min()),
    }


def _load_posthoc_validation_ids() -> tuple[tuple[int, ...], dict[str, Any]]:
    previous = pd.read_csv(PREVIOUS_VALIDATION)
    if previous.shape[0] != VALIDATION_COUNT or "state_id" not in previous.columns:
        raise RuntimeError("previous validation_per_state_metrics.csv must contain 85 rows and state_id")
    previous_ids = tuple(int(value) for value in previous["state_id"])
    split = pd.read_csv(SPLIT_PATH)
    if not {"state_id", "split"}.issubset(split.columns):
        raise RuntimeError("existing frozen state split is missing state_id/split")
    split = split.sort_values("state_id").reset_index(drop=True)
    split_ids = tuple(int(value) for value in split.loc[split["split"].astype(str).str.lower() == "validation", "state_id"])
    if previous_ids != split_ids:
        raise RuntimeError("previous validation state IDs do not match the frozen split Validation IDs")
    if len(set(previous_ids)) != VALIDATION_COUNT:
        raise RuntimeError("post-hoc Validation state IDs are not unique")
    split_hash = hashlib.sha256(SPLIT_PATH.read_bytes()).hexdigest()
    return previous_ids, {
        "path": str(SPLIT_PATH),
        "sha256": split_hash,
        "state_count": VALIDATION_COUNT,
        "ids_match_previous_csv": True,
        "untouched_validation_claim": False,
    }


def _nmse_db(reference: np.ndarray, prediction: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.complex128).reshape(-1)
    prediction = np.asarray(prediction, dtype=np.complex128).reshape(-1)
    if reference.shape != prediction.shape or not np.all(np.isfinite(reference)) or not np.all(np.isfinite(prediction)):
        raise ValueError("NMSE inputs must be finite complex vectors with equal shape")
    denominator = float(np.sum(np.abs(reference) ** 2))
    numerator = float(np.sum(np.abs(reference - prediction) ** 2))
    if denominator <= 0:
        raise ValueError("NMSE reference energy must be positive")
    if numerator == 0.0:
        return float("-inf")
    return float(10.0 * np.log10(numerator / denominator))


def _combined_basis(x: np.ndarray, selected_terms: tuple[Any, ...]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mp = gmp.build_frozen_mp_basis(x)
    if not selected_terms:
        return mp, mp, np.empty((mp.shape[0], 0), dtype=np.complex128)
    all_gmp = global_runner._build_global_gmp_basis(x)
    by_index = {term.term_id: int(term.term_index - 1) for term in global_runner.GLOBAL_TERMS}
    indices = [by_index[term.term_id] for term in selected_terms]
    gmp_selected = all_gmp[:, indices]
    return np.column_stack((mp, gmp_selected)), mp, gmp_selected


def _fit_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_b: np.ndarray,
    y_b: np.ndarray,
    selected_terms: tuple[Any, ...],
) -> dict[str, Any]:
    phi, mp_phi, gmp_phi = _combined_basis(x_train, selected_terms)
    phi_b, _, _ = _combined_basis(x_b, selected_terms)
    y_train_valid = np.asarray(y_train, dtype=np.complex128).reshape(-1)[DMAX:]
    y_b_valid = np.asarray(y_b, dtype=np.complex128).reshape(-1)[DMAX:]
    if phi.shape[0] != y_train_valid.size or phi_b.shape[0] != y_b_valid.size:
        raise RuntimeError("C/B design matrix and target support mismatch")
    theta, diagnostics = gmp.fit_ridge(phi, y_train_valid, RIDGE_LAMBDA)
    pred_train = phi @ theta
    pred_b = phi_b @ theta
    return {
        "phi": phi,
        "phi_b": phi_b,
        "mp_phi": mp_phi,
        "gmp_phi": gmp_phi,
        "theta": theta,
        "target_train": y_train_valid,
        "target_b": y_b_valid,
        "prediction_train": pred_train,
        "prediction_b": pred_b,
        "train_nmse": _nmse_db(y_train_valid, pred_train),
        "b_nmse": _nmse_db(y_b_valid, pred_b),
        "rank": int(diagnostics.rank_phi),
        "condition": float(diagnostics.condition_number_phi),
    }


def _alignment(residual: np.ndarray, correction: np.ndarray) -> tuple[float, float]:
    residual = np.asarray(residual, dtype=np.complex128).reshape(-1)
    correction = np.asarray(correction, dtype=np.complex128).reshape(-1)
    denominator = float(np.linalg.norm(residual) * np.linalg.norm(correction))
    if denominator <= 0:
        return float("nan"), float("nan")
    raw = float(np.real(np.vdot(residual, correction)) / denominator)
    clipped = float(np.clip(raw, -1.0, 1.0))
    return clipped, abs(raw - clipped)


def _strength(residual: np.ndarray, correction: np.ndarray) -> float:
    denominator = float(np.linalg.norm(residual))
    if denominator <= 0:
        return float("nan")
    return float(np.linalg.norm(correction) / denominator)


def _residual_reduction(residual_frozen: np.ndarray, residual_new: np.ndarray) -> tuple[float, float]:
    denominator = float(np.linalg.norm(residual_frozen) ** 2)
    if denominator <= 0:
        return float("nan"), float("nan")
    direct = float((np.linalg.norm(residual_frozen) ** 2 - np.linalg.norm(residual_new) ** 2) / denominator)
    return direct, direct


def _correction_metrics(frozen: dict[str, Any], g4: dict[str, Any]) -> dict[str, float]:
    residual_frozen_train = frozen["target_train"] - frozen["prediction_train"]
    residual_frozen_b = frozen["target_b"] - frozen["prediction_b"]
    residual_g4_train = g4["target_train"] - g4["prediction_train"]
    residual_g4_b = g4["target_b"] - g4["prediction_b"]
    correction_train = g4["prediction_train"] - frozen["prediction_train"]
    correction_b = g4["prediction_b"] - frozen["prediction_b"]
    alignment_train, train_clip_error = _alignment(residual_frozen_train, correction_train)
    alignment_b, b_clip_error = _alignment(residual_frozen_b, correction_b)
    strength_train = _strength(residual_frozen_train, correction_train)
    strength_b = _strength(residual_frozen_b, correction_b)
    reduction_train, _ = _residual_reduction(residual_frozen_train, residual_g4_train)
    reduction_b, _ = _residual_reduction(residual_frozen_b, residual_g4_b)
    strength_ratio = strength_b / strength_train if strength_train > 0 else float("nan")
    strength_transfer = float(20.0 * np.log10(strength_ratio)) if strength_ratio > 0 else float("nan")
    decomposition_train = 2.0 * strength_train * alignment_train - strength_train**2
    decomposition_b = 2.0 * strength_b * alignment_b - strength_b**2
    identity_train = float(10.0 * np.log10(1.0 - reduction_train))
    identity_b = float(10.0 * np.log10(1.0 - reduction_b))
    return {
        "correction_alignment_train": alignment_train,
        "correction_alignment_B": alignment_b,
        "alignment_drop_train_minus_B": alignment_train - alignment_b,
        "correction_strength_train": strength_train,
        "correction_strength_B": strength_b,
        "strength_ratio_B_over_C": strength_ratio,
        "strength_transfer_dB": strength_transfer,
        "residual_reduction_train": reduction_train,
        "residual_reduction_B": reduction_b,
        "residual_reduction_drop_train_minus_B": reduction_train - reduction_b,
        "residual_reduction_decomposition_train": decomposition_train,
        "residual_reduction_decomposition_B": decomposition_b,
        "residual_reduction_decomposition_error_train": abs(reduction_train - decomposition_train),
        "residual_reduction_decomposition_error_B": abs(reduction_b - decomposition_b),
        "nmse_identity_error_train_dB": abs((g4["train_nmse"] - frozen["train_nmse"]) - identity_train),
        "nmse_identity_error_B_dB": abs((g4["b_nmse"] - frozen["b_nmse"]) - identity_b),
        "alignment_clip_error_max": max(train_clip_error, b_clip_error),
    }


def _gram_shift(phi_train: np.ndarray, phi_b: np.ndarray) -> float:
    def gram(phi: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(phi, axis=0)
        if np.any(norms <= 0) or not np.all(np.isfinite(norms)):
            raise RuntimeError("basis column has invalid norm")
        normalized = phi / norms[None, :]
        return normalized.conj().T @ normalized

    gram_c = gram(phi_train)
    gram_b = gram(phi_b)
    return float(np.linalg.norm(gram_b - gram_c, ord="fro") / G4_K)


def _condition_log10(phi: np.ndarray) -> float:
    singular = np.linalg.svd(phi, compute_uv=False)
    if singular[-1] <= 0 or not np.all(np.isfinite(singular)):
        return float("inf")
    return float(np.log10(singular[0] / singular[-1]))


def _gmp_rms_db_ratio(gmp_c: np.ndarray, gmp_b: np.ndarray) -> list[float]:
    values: list[float] = []
    for column in range(gmp_c.shape[1]):
        rms_c = float(np.sqrt(np.mean(np.abs(gmp_c[:, column]) ** 2)))
        rms_b = float(np.sqrt(np.mean(np.abs(gmp_b[:, column]) ** 2)))
        if rms_c <= 0 or rms_b <= 0:
            raise RuntimeError("selected GMP RMS excitation must be positive")
        values.append(float(20.0 * np.log10(rms_b / rms_c)))
    return values


def _max_frozen_correlation(mp_phi: np.ndarray, gmp_phi: np.ndarray) -> list[float]:
    result: list[float] = []
    mp_norms = np.linalg.norm(mp_phi, axis=0)
    for column in range(gmp_phi.shape[1]):
        gmp_norm = float(np.linalg.norm(gmp_phi[:, column]))
        values = [abs(np.vdot(mp_phi[:, index], gmp_phi[:, column])) / (float(mp_norms[index]) * gmp_norm) for index in range(FROZEN_K)]
        result.append(float(max(values)))
    return result


def _diagnostic_worker(state_id: int, selected_terms: tuple[Any, ...]) -> dict[str, Any]:
    prepared = frozen_model.prepare_state_any(int(state_id))
    c_frozen = _fit_predict(prepared.c2_C_input, prepared.c2_C_output, prepared.c2_B_input, prepared.c2_B_output, ())
    c_g4 = _fit_predict(prepared.c2_C_input, prepared.c2_C_output, prepared.c2_B_input, prepared.c2_B_output, selected_terms)
    a_frozen = _fit_predict(prepared.aend_A_input, prepared.aend_A_output, prepared.aend_B_input, prepared.aend_B_output, ())
    a_g4 = _fit_predict(prepared.aend_A_input, prepared.aend_A_output, prepared.aend_B_input, prepared.aend_B_output, selected_terms)
    c_diag = _correction_metrics(c_frozen, c_g4)
    a_diag = _correction_metrics(a_frozen, a_g4)
    c_rms = _gmp_rms_db_ratio(c_g4["gmp_phi"], _combined_basis(prepared.c2_B_input, selected_terms)[2])
    gram_shift = _gram_shift(c_g4["phi"], c_g4["phi_b"])
    corr_c = _max_frozen_correlation(c_g4["mp_phi"], c_g4["gmp_phi"])
    corr_b = _max_frozen_correlation(c_g4["phi_b"][:, :FROZEN_K], c_g4["phi_b"][:, FROZEN_K:])
    row: dict[str, Any] = {
        "state_id": int(state_id),
        "ilc_A_end": int(prepared.ilc_A_end),
        "Frozen_C2_train_NMSE_dB": c_frozen["train_nmse"],
        "G4_C2_train_NMSE_dB": c_g4["train_nmse"],
        "delta_C2_train_dB": c_g4["train_nmse"] - c_frozen["train_nmse"],
        "Frozen_C2_B_NMSE_dB": c_frozen["b_nmse"],
        "G4_C2_B_NMSE_dB": c_g4["b_nmse"],
        "delta_C2_B_dB": c_g4["b_nmse"] - c_frozen["b_nmse"],
        "delta_gap_C_dB": (c_g4["b_nmse"] - c_g4["train_nmse"]) - (c_frozen["b_nmse"] - c_frozen["train_nmse"]),
        "train_improvement_dB": -(c_g4["train_nmse"] - c_frozen["train_nmse"]),
        "B_improvement_dB": -(c_g4["b_nmse"] - c_frozen["b_nmse"]),
        "Frozen_Aend_train_NMSE_dB": a_frozen["train_nmse"],
        "G4_Aend_train_NMSE_dB": a_g4["train_nmse"],
        "delta_Aend_train_dB": a_g4["train_nmse"] - a_frozen["train_nmse"],
        "Frozen_Aend_B_NMSE_dB": a_frozen["b_nmse"],
        "G4_Aend_B_NMSE_dB": a_g4["b_nmse"],
        "delta_Aend_B_dB": a_g4["b_nmse"] - a_frozen["b_nmse"],
        "delta_gap_A_dB": (a_g4["b_nmse"] - a_g4["train_nmse"]) - (a_frozen["b_nmse"] - a_frozen["train_nmse"]),
        "A_alignment_train": a_diag["correction_alignment_train"],
        "A_alignment_B": a_diag["correction_alignment_B"],
        "A_alignment_drop_A_to_B": a_diag["alignment_drop_train_minus_B"],
        "A_strength_train": a_diag["correction_strength_train"],
        "A_strength_B": a_diag["correction_strength_B"],
        "A_strength_transfer_dB": a_diag["strength_transfer_dB"],
        "A_residual_reduction_train": a_diag["residual_reduction_train"],
        "A_residual_reduction_B": a_diag["residual_reduction_B"],
        "A_residual_reduction_drop": a_diag["residual_reduction_drop_train_minus_B"],
        "correction_alignment_C": c_diag["correction_alignment_train"],
        "correction_alignment_B": c_diag["correction_alignment_B"],
        "alignment_drop_C_to_B": c_diag["alignment_drop_train_minus_B"],
        "correction_strength_C": c_diag["correction_strength_train"],
        "correction_strength_B": c_diag["correction_strength_B"],
        "strength_ratio_B_over_C": c_diag["strength_ratio_B_over_C"],
        "strength_transfer_dB": c_diag["strength_transfer_dB"],
        "residual_reduction_C": c_diag["residual_reduction_train"],
        "residual_reduction_B": c_diag["residual_reduction_B"],
        "residual_reduction_drop_C_minus_B": c_diag["residual_reduction_drop_train_minus_B"],
        "residual_reduction_decomposition_error_C": c_diag["residual_reduction_decomposition_error_train"],
        "residual_reduction_decomposition_error_B": c_diag["residual_reduction_decomposition_error_B"],
        "nmse_identity_error_C_dB": c_diag["nmse_identity_error_train_dB"],
        "nmse_identity_error_B_dB": c_diag["nmse_identity_error_B_dB"],
        "gram_shift_C_to_B": gram_shift,
        "log10_condition_C": _condition_log10(c_g4["phi"]),
        "log10_condition_B": _condition_log10(c_g4["phi_b"]),
        "condition_log10_delta_B_minus_C": _condition_log10(c_g4["phi_b"]) - _condition_log10(c_g4["phi"]),
        "rank_C": c_g4["rank"],
        "rank_B": int(np.linalg.matrix_rank(c_g4["phi_b"])),
        "A_correction_identity_error_max": max(a_diag["residual_reduction_decomposition_error_train"], a_diag["residual_reduction_decomposition_error_B"]),
        "A_nmse_identity_error_max_dB": max(a_diag["nmse_identity_error_train_dB"], a_diag["nmse_identity_error_B_dB"]),
    }
    for index in range(4):
        row[f"GMP0{index + 1}_RMS_ratio_B_over_C_dB"] = c_rms[index]
        row[f"GMP0{index + 1}_maxcorr_Frozen_C"] = corr_c[index]
        row[f"GMP0{index + 1}_maxcorr_Frozen_B"] = corr_b[index]
        row[f"GMP0{index + 1}_maxcorr_Frozen_delta"] = corr_b[index] - corr_c[index]
    return row


def _run_workers(state_ids: tuple[int, ...], selected_terms: tuple[Any, ...], workers: int) -> pd.DataFrame:
    values: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_diagnostic_worker, state_id, selected_terms): state_id for state_id in state_ids}
        checkpoints = {math.ceil(len(state_ids) * fraction) for fraction in (0.2, 0.4, 0.6, 0.8, 1.0)}
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            values[int(result["state_id"])] = result
            if done in checkpoints:
                print(f"diagnostic states: {done}/{len(state_ids)}", flush=True)
    if tuple(sorted(values)) != tuple(sorted(state_ids)):
        raise RuntimeError("diagnostic state set is incomplete")
    frame = pd.DataFrame([values[state_id] for state_id in state_ids]).sort_values("state_id").reset_index(drop=True)
    if frame.shape[0] != len(state_ids):
        raise RuntimeError("diagnostic frame row count mismatch")
    return frame


def _reproduction_gate(frame: pd.DataFrame) -> dict[str, Any]:
    previous = pd.read_csv(PREVIOUS_VALIDATION).set_index("state_id")
    check_ids = tuple(int(value) for value in frame["state_id"].head(REPRODUCTION_IDS_COUNT))
    rows: list[dict[str, Any]] = []
    max_error = 0.0
    for state_id in check_ids:
        current = frame.loc[frame["state_id"] == state_id].iloc[0]
        old = previous.loc[state_id]
        errors = {
            "Frozen_C2_train": abs(float(current["Frozen_C2_train_NMSE_dB"]) - float(old["Frozen_C2_train_NMSE_dB"])),
            "G4_C2_train": abs(float(current["G4_C2_train_NMSE_dB"]) - float(old["Selected_C2_train_NMSE_dB"])),
            "Frozen_C2_B": abs(float(current["Frozen_C2_B_NMSE_dB"]) - float(old["Frozen_C2_B_NMSE_dB"])),
            "G4_C2_B": abs(float(current["G4_C2_B_NMSE_dB"]) - float(old["Selected_C2_B_NMSE_dB"])),
        }
        maximum = max(errors.values())
        max_error = max(max_error, maximum)
        rows.append({"state_id": state_id, **errors, "pass": bool(maximum <= METRIC_TOLERANCE_DB)})
    return {"pass": bool(all(row["pass"] for row in rows)), "maximum_abs_error_dB": max_error, "rows": rows}


def _correlation(value_x: pd.Series, value_y: pd.Series) -> dict[str, float | int | str]:
    x = value_x.to_numpy(dtype=float)
    y = value_y.to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if int(finite.sum()) < 3 or np.ptp(x[finite]) == 0 or np.ptp(y[finite]) == 0:
        return {"n": int(finite.sum()), "pearson_r": float("nan"), "spearman_rho": float("nan"), "p_value": float("nan"), "status": "undefined_constant_or_insufficient"}
    pearson = pearsonr(x[finite], y[finite])
    spearman = spearmanr(x[finite], y[finite])
    return {"n": int(finite.sum()), "pearson_r": float(pearson.statistic), "spearman_rho": float(spearman.statistic), "p_value": float(spearman.pvalue), "status": "finite"}


def _posthoc_correlations(frame: pd.DataFrame) -> dict[str, Any]:
    metrics = [
        "alignment_drop_C_to_B",
        "strength_transfer_dB",
        "residual_reduction_drop_C_minus_B",
        "gram_shift_C_to_B",
        "condition_log10_delta_B_minus_C",
        "GMP01_RMS_ratio_B_over_C_dB",
        "GMP02_RMS_ratio_B_over_C_dB",
        "GMP03_RMS_ratio_B_over_C_dB",
        "GMP04_RMS_ratio_B_over_C_dB",
        "GMP01_maxcorr_Frozen_delta",
        "GMP02_maxcorr_Frozen_delta",
        "GMP03_maxcorr_Frozen_delta",
        "GMP04_maxcorr_Frozen_delta",
    ]
    result = {metric: _correlation(frame[metric], frame["delta_gap_C_dB"]) for metric in metrics}
    result["improvement_transfer"] = _correlation(frame["train_improvement_dB"], frame["B_improvement_dB"])
    return result


def _write_figure(frame: pd.DataFrame, output_path: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )
    figure, axes = plt.subplots(2, 2, figsize=(10.0, 7.2), constrained_layout=True)
    ax = axes[0, 0]
    ax.scatter(frame["train_improvement_dB"], frame["B_improvement_dB"], s=22, color="#2C6E9B", edgecolor="white", linewidth=0.35, alpha=0.9)
    limits = [0.0, float(max(frame["train_improvement_dB"].max(), frame["B_improvement_dB"].max()) * 1.05)]
    ax.plot(limits, limits, "--", color="#777777", linewidth=0.9, label="y=x")
    ax.set_xlabel("C2 train improvement (dB)")
    ax.set_ylabel("C2 → B improvement (dB)")
    ax.set_title("Improvement transfer")
    ax.legend(loc="best", fontsize=7)

    ax = axes[0, 1]
    ordered = frame.sort_values("state_id")
    ax.plot(ordered["state_id"], ordered["delta_gap_C_dB"], color="#B04A47", linewidth=1.1, marker="o", markersize=2.5, markeredgecolor="white", markeredgewidth=0.25)
    ax.axhline(VALIDATION_GAP_LIMIT_DB, color="#777777", linestyle="--", linewidth=0.9, label="+0.10 dB limit")
    top_ids = frame.nlargest(5, "delta_gap_C_dB")["state_id"].tolist()
    for state_id in top_ids:
        row = frame.loc[frame["state_id"] == state_id].iloc[0]
        ax.annotate(str(state_id), (state_id, row["delta_gap_C_dB"]), xytext=(0, 6), textcoords="offset points", ha="center", fontsize=7, color="#8C2F2B")
    ax.set_xlabel("State ID")
    ax.set_ylabel("ΔGap_C (dB)")
    ax.set_title("C2 generalization-gap change")
    ax.legend(loc="best", fontsize=7)

    ax = axes[1, 0]
    ax.scatter(frame["correction_alignment_C"], frame["correction_alignment_B"], s=22, color="#5D8C65", edgecolor="white", linewidth=0.35, alpha=0.9)
    rho_min = float(min(frame["correction_alignment_C"].min(), frame["correction_alignment_B"].min()) - 0.03)
    rho_max = float(max(frame["correction_alignment_C"].max(), frame["correction_alignment_B"].max()) + 0.03)
    ax.plot([rho_min, rho_max], [rho_min, rho_max], "--", color="#777777", linewidth=0.9)
    ax.set_xlim(rho_min, rho_max)
    ax.set_ylim(rho_min, rho_max)
    ax.set_xlabel("Correction alignment on C")
    ax.set_ylabel("Correction alignment on B")
    ax.set_title("Correction-direction transfer")

    ax = axes[1, 1]
    ax.scatter(frame["alignment_drop_C_to_B"], frame["delta_gap_C_dB"], s=22, color="#8A5A9A", edgecolor="white", linewidth=0.35, alpha=0.9)
    ax.axhline(VALIDATION_GAP_LIMIT_DB, color="#777777", linestyle="--", linewidth=0.9)
    ax.axvline(0.0, color="#B0B0B0", linewidth=0.7)
    ax.set_xlabel("Alignment drop, ρ_C − ρ_B")
    ax.set_ylabel("ΔGap_C (dB)")
    ax.set_title("Direction mismatch versus gap")

    for panel, axis in zip(("a", "b", "c", "d"), axes.flat, strict=True):
        axis.text(
            0.01,
            0.99,
            panel,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            fontweight="bold",
        )

    figure.suptitle("Global G4 C2 gap diagnosis (85 post-hoc Validation states)", fontsize=12)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _summary_text(
    frame: pd.DataFrame,
    previous_definition: dict[str, Any],
    split_info: dict[str, Any],
    reproduction: dict[str, Any],
    correlations: dict[str, Any],
    logical_cpus: int,
    workers: int,
    protection: dict[str, Any],
) -> str:
    medians = frame.median(numeric_only=True)
    alignment_drop = float(medians["alignment_drop_C_to_B"])
    strength_transfer = float(medians["strength_transfer_dB"])
    if alignment_drop > 0.0 and strength_transfer < 0.0:
        interpretation = "Both correction weakening and correction-direction mismatch contribute to the reduced C-to-B transfer."
    elif alignment_drop > 0.0:
        interpretation = "The G4 correction direction is less aligned with the Frozen residual on B than on C."
    elif strength_transfer < 0.0:
        interpretation = "The relative G4 correction magnitude becomes weaker on B."
    else:
        interpretation = "Neither median correction weakening nor median alignment drop is positive; other factors require follow-up."
    lines = [
        TASK_NAME,
        "Post-hoc diagnosis only; no model selection or configuration update.",
        "",
        "Frozen models",
        f"Frozen orders={list(ORDERS)}; memory={list(MEMORY)}; K={FROZEN_K}; dmax={DMAX}; lambda={RIDGE_LAMBDA}",
        f"Global G4 K={G4_K}; fixed_support={list(previous_definition['term_ids'])}",
        "Frozen and G4 coefficients were independently jointly refit on each state/side.",
        "",
        "State scope",
        f"post_hoc_validation_state_count={VALIDATION_COUNT}",
        f"split_source={split_info['path']}",
        f"split_sha256={split_info['sha256']}",
        "validation_set_is_untouched_for_future_selection=False",
        "",
        "Previous Global-G4 result reproduction",
        f"previous_selected_G={previous_definition['selected_G']}; previous_validation_pass=False",
        "previous_failure_reason=median delta_gap_C exceeded +0.10 dB",
        f"reproduction_pass={reproduction['pass']}; maximum_abs_error_dB={reproduction['maximum_abs_error_dB']:.17g}",
        "",
        "Core C2 gap diagnosis medians",
        f"Frozen_C2_train={float(medians['Frozen_C2_train_NMSE_dB']):.9f} dB; G4_C2_train={float(medians['G4_C2_train_NMSE_dB']):.9f} dB; delta={float(medians['delta_C2_train_dB']):.9f} dB",
        f"Frozen_C2_B={float(medians['Frozen_C2_B_NMSE_dB']):.9f} dB; G4_C2_B={float(medians['G4_C2_B_NMSE_dB']):.9f} dB; delta={float(medians['delta_C2_B_dB']):.9f} dB",
        f"delta_gap_C={float(medians['delta_gap_C_dB']):.9f} dB; train_improvement={float(medians['train_improvement_dB']):.9f} dB; B_improvement={float(medians['B_improvement_dB']):.9f} dB",
        f"Frozen_gap_C={float(medians['Frozen_C2_B_NMSE_dB'] - medians['Frozen_C2_train_NMSE_dB']):.9f} dB; G4_gap_C={float(medians['G4_C2_B_NMSE_dB'] - medians['G4_C2_train_NMSE_dB']):.9f} dB",
        f"C2_B_improved_count={(frame['delta_C2_B_dB'] < 0).sum()}/{VALIDATION_COUNT}; all four C2 metrics are finite",
        "",
        "Diagnostic A: improvement transfer",
        f"train_vs_B_correlation={correlations['improvement_transfer']}",
        f"B_improvement_less_than_train_count={int((frame['B_improvement_dB'] < frame['train_improvement_dB']).sum())}",
        f"B_improvement_at_least_train_count={int((frame['B_improvement_dB'] >= frame['train_improvement_dB']).sum())}",
        "",
        "Diagnostic B: Frozen→G4 correction transfer",
        f"median_alignment_C={float(medians['correction_alignment_C']):.9f}",
        f"median_alignment_B={float(medians['correction_alignment_B']):.9f}",
        f"median_alignment_drop_C_to_B={alignment_drop:.9f}",
        f"median_strength_C={float(medians['correction_strength_C']):.9f}",
        f"median_strength_B={float(medians['correction_strength_B']):.9f}",
        f"median_strength_transfer_dB={strength_transfer:.9f}",
        f"median_residual_reduction_C={float(medians['residual_reduction_C']):.9f}",
        f"median_residual_reduction_B={float(medians['residual_reduction_B']):.9f}",
        f"median_residual_reduction_drop_C_minus_B={float(medians['residual_reduction_drop_C_minus_B']):.9f}",
        f"correction_identity_max_error={float(max(frame['residual_reduction_decomposition_error_C'].max(), frame['residual_reduction_decomposition_error_B'].max())):.3g}",
        f"NMSE_identity_max_error_dB={float(max(frame['nmse_identity_error_C_dB'].max(), frame['nmse_identity_error_B_dB'].max())):.3g}",
        f"interpretation={interpretation}",
        "",
        "Diagnostic C: C/B feature excitation and geometry",
    ]
    for index in range(4):
        lines.append(f"GMP0{index + 1} median RMS ratio B/C={float(medians[f'GMP0{index + 1}_RMS_ratio_B_over_C_dB']):.9f} dB; median maxcorr delta={float(medians[f'GMP0{index + 1}_maxcorr_Frozen_delta']):.9f}")
    lines.extend(
        [
            f"median_GramShift={float(medians['gram_shift_C_to_B']):.9f}",
            f"median_log10_condition_C={float(medians['log10_condition_C']):.9f}",
            f"median_log10_condition_B={float(medians['log10_condition_B']):.9f}",
            f"median_condition_log10_delta_B_minus_C={float(medians['condition_log10_delta_B_minus_C']):.9f}",
            "",
            "Aend control",
            f"median_A_delta_gap={float(medians['delta_gap_A_dB']):.9f} dB",
            f"median_A_alignment_drop={float(medians['A_alignment_drop_A_to_B']):.9f}",
            f"median_A_strength_transfer={float(medians['A_strength_transfer_dB']):.9f} dB",
            f"median_A_residual_reduction_drop={float(medians['A_residual_reduction_drop']):.9f}",
            "",
            "Post-hoc Spearman correlations with delta_gap_C",
        ]
    )
    for metric, result in sorted(
        ((key, value) for key, value in correlations.items() if key != "improvement_transfer"),
        key=lambda item: -(abs(float(item[1]["spearman_rho"])) if np.isfinite(item[1]["spearman_rho"]) else -1.0),
    ):
        lines.append(f"{metric}: rho={result['spearman_rho']}, p={result['p_value']}, n={result['n']}, status={result['status']}")
    lines.extend(
        [
            "Correlation is descriptive and does not establish causality.",
            "",
            "Stopping boundary",
            "No model selection, no support change, no lambda change, no Frozen/G4 modification, no 425-state characterization, no LUT retrieval, no Real-B retrieval validation, and no DPD replay.",
            "These 85 states are post-hoc diagnostics only and must not be presented as a new untouched validation set for future selection.",
            "",
            "Execution and protection",
            f"logical_cpus={logical_cpus}; workers={workers}; cpu_target={WORKER_TARGET}; BLAS_threads_per_worker=1; nested_parallelism=False",
            f"protection={protection}",
            "",
            "Output files",
            f"{RESULT_ROOT / 'c2_gap_diagnosis_per_state.csv'}",
            f"{RESULT_ROOT / 'c2_gap_diagnosis.png'}",
            f"{RESULT_ROOT / 'final_result_summary.txt'}",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, encoding="utf-8-sig", float_format="%.17g", na_rep="NaN")


def _snapshot(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    files = [file for file in path.rglob("*") if file.is_file() and "__pycache__" not in file.parts and file.suffix.lower() != ".pyc"]
    for file in sorted(files, key=lambda item: str(item).lower()):
        digest.update(file.relative_to(path).as_posix().encode("utf-8") + b"\0")
        data = file.read_bytes()
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(hashlib.sha256(data).digest())
    return digest.hexdigest()


def _raw_manifest() -> dict[str, Any]:
    raw_root = PROJECT_ROOT / "data" / "raw"
    digest = hashlib.sha256()
    files = [file for file in raw_root.rglob("*") if file.is_file()]
    total = 0
    for file in sorted(files, key=lambda item: str(item).lower()):
        data = file.read_bytes()
        digest.update(file.relative_to(raw_root).as_posix().encode("utf-8") + b"\0")
        digest.update(len(data).to_bytes(8, "little"))
        digest.update(hashlib.sha256(data).digest())
        total += len(data)
    return {"sha256": digest.hexdigest(), "file_count": len(files), "bytes": total}


def _protection_before_after(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    result_unchanged = {name: before["results"][name] == after["results"][name] for name in before["results"]}
    return {
        "data_raw_unchanged": before["raw"] == after["raw"],
        "protected_results_unchanged": bool(all(result_unchanged.values())),
        "all_protected_unchanged": bool(before["raw"] == after["raw"] and all(result_unchanged.values())),
        "results": result_unchanged,
    }


def main() -> None:
    freeze_support()
    previous_definition = _load_frozen_g4_definition()
    validation_ids, split_info = _load_posthoc_validation_ids()
    expected_output_names = {
        "c2_gap_diagnosis_per_state.csv",
        "c2_gap_diagnosis.png",
        "final_result_summary.txt",
    }
    if RESULT_ROOT.exists():
        existing_names = {path.name for path in RESULT_ROOT.iterdir()}
        if not existing_names.issubset(expected_output_names):
            raise FileExistsError(
                f"refusing to overwrite unexpected result files: {sorted(existing_names - expected_output_names)}"
            )
    protected_dirs = {
        "previous_global_result": PREVIOUS_ROOT,
        "formal_retrieval": PROJECT_ROOT / "results" / "behavior_fingerprint_lut_retrieval" / "scenario_2_C2_to_Aend",
        "g4_retrieval": PROJECT_ROOT / "results" / "behavior_fingerprint_lut_retrieval" / "scenario_2_C2_to_Aend_G4_sparse_gmp_5B",
        "all_ilc": PROJECT_ROOT / "results" / "behavior_fingerprint_ranking_consistency" / "scenario_2_all_ilc",
    }
    before = {"raw": _raw_manifest(), "results": {name: _snapshot(path) for name, path in protected_dirs.items()}}
    logical_cpus = int(os.cpu_count() or 1)
    workers = max(1, int(math.floor(WORKER_TARGET * logical_cpus)))
    print(
        f"{TASK_NAME}: states={len(validation_ids)}, workers={workers}, "
        f"selected_G={previous_definition['selected_G']}, terms={list(previous_definition['term_ids'])}",
        flush=True,
    )
    frame = _run_workers(validation_ids, previous_definition["terms"], workers)
    reproduction = _reproduction_gate(frame)
    if not reproduction["pass"]:
        raise RuntimeError(f"previous Global G4 metrics reproduction failed: {reproduction}")
    print(f"Previous Global G4 C2 metric reproduction: PASS; max error={reproduction['maximum_abs_error_dB']:.3g} dB", flush=True)
    correlations = _posthoc_correlations(frame)
    max_decomposition_error = float(max(frame["residual_reduction_decomposition_error_C"].max(), frame["residual_reduction_decomposition_error_B"].max()))
    max_identity_error = float(max(frame["nmse_identity_error_C_dB"].max(), frame["nmse_identity_error_B_dB"].max()))
    if max_decomposition_error >= DECOMPOSITION_TOLERANCE or max_identity_error >= IDENTITY_TOLERANCE_DB:
        raise RuntimeError(
            f"diagnostic identity gate failed: decomposition={max_decomposition_error}, "
            f"nmse_identity_dB={max_identity_error}"
        )
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    _write_figure(frame, RESULT_ROOT / "c2_gap_diagnosis.png")
    _write_frame(frame, RESULT_ROOT / "c2_gap_diagnosis_per_state.csv")
    after = {"raw": _raw_manifest(), "results": {name: _snapshot(path) for name, path in protected_dirs.items()}}
    protection = _protection_before_after(before, after)
    if not protection["all_protected_unchanged"]:
        raise RuntimeError(f"raw/protected results changed: {protection}")
    summary = _summary_text(frame, previous_definition, split_info, reproduction, correlations, logical_cpus, workers, protection)
    (RESULT_ROOT / "final_result_summary.txt").write_text(summary, encoding="utf-8")
    if sorted(path.name for path in RESULT_ROOT.iterdir() if path.is_file()) != [
        "c2_gap_diagnosis.png",
        "c2_gap_diagnosis_per_state.csv",
        "final_result_summary.txt",
    ]:
        raise RuntimeError("diagnostic result directory must contain exactly three formal files")
    log_stamp = datetime.now(UTC).isoformat()
    with MODEL_LOG.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(f"\n[{log_stamp}] {TASK_NAME}\n{summary}\n")
    with HANDOFF_LOG.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            f"\n[{log_stamp}] {TASK_NAME}\n"
            f"post-hoc C2 gap diagnosis completed: fixed Global G4 support={list(previous_definition['term_ids'])}, "
            f"85-state reproduction=True, max delta_gap_C={float(frame['delta_gap_C_dB'].median()):.9f} dB, "
            f"diagnostic identities=True, no model selection/425 characterization/LUT retrieval; "
            f"raw/protected unchanged={protection['all_protected_unchanged']}; results={RESULT_ROOT}\n"
        )
    print(f"Result directory: {RESULT_ROOT}", flush=True)
    print(f"Diagnostics completed; output files=3; median delta_gap_C={float(frame['delta_gap_C_dB'].median()):.9f} dB", flush=True)


if __name__ == "__main__":
    main()
