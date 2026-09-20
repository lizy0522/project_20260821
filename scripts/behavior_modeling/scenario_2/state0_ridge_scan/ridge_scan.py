"""
功能说明：在一次State0 canonical数据准备上执行16个统一lambda、每点11模型的Ridge诊断扫描。
输入：State0数据字典和冻结lambda列表；不使用STALE、不重新生成每个lambda的canonical数据。
输出：176行模型结果、16行聚合结果、theta(16,11,10)、候选区域和自动验收信息。
用途：严格比较OLS与Ridge的Train、真实B泛化、公共B-Probe CNMSE和theta范数。
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from data_management.shared import load_by_id
from signal_segmentation.shared import (
    CanonicalSegment,
    build_ilc_segments,
    build_off_segments,
    build_partition_from_xin,
    get_common_probe,
)

from behavior_modeling.shared.basis import build_mp_basis
from behavior_modeling.shared.config import MAX_DELAY, MP_CONFIG, NUM_COEFFICIENTS
from behavior_modeling.shared.evaluation import calculate_cnmse, calculate_nmse
from behavior_modeling.shared.ridge import fit_coefficients_ridge

RIDGE_LAMBDAS = np.array(
    [
        0.0,
        1e-12,
        1e-11,
        1e-10,
        1e-9,
        1e-8,
        1e-7,
        1e-6,
        1e-5,
        1e-4,
        1e-3,
        1e-2,
        1e-1,
        1e0,
        1e1,
        1e2,
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class PreparedModelData:
    """一个模型在lambda扫描中不变的训练、B泛化设计矩阵和参考。"""

    model_id: str
    behavior_class: str
    train_segment: str
    ilc_iteration: int | None
    phi_train: np.ndarray
    y_train: np.ndarray
    phi_b: np.ndarray
    y_b: np.ndarray
    b_generalization_source: str


@dataclass
class RidgeScanResult:
    """完整Ridge扫描内存结果。"""

    lambdas: np.ndarray
    model_ids: tuple[str, ...]
    theta: np.ndarray
    model_rows: list[dict[str, Any]]
    summary_rows: list[dict[str, Any]]
    diagnostic: dict[str, Any]
    common_probe: np.ndarray
    real_b_valid: np.ndarray


def _basis(x: np.ndarray) -> np.ndarray:
    return build_mp_basis(x, MP_CONFIG["orders"], MP_CONFIG["memory_depth"])


def _prepare_model(
    *,
    model_id: str,
    behavior_class: str,
    train_segment_name: str,
    ilc_iteration: int | None,
    train_segment: CanonicalSegment,
    b_segment: CanonicalSegment,
    b_source: str,
) -> PreparedModelData:
    return PreparedModelData(
        model_id=model_id,
        behavior_class=behavior_class,
        train_segment=train_segment_name,
        ilc_iteration=ilc_iteration,
        phi_train=_basis(train_segment.input),
        y_train=train_segment.output[MAX_DELAY:],
        phi_b=_basis(b_segment.input),
        y_b=b_segment.output[MAX_DELAY:],
        b_generalization_source=b_source,
    )


def _prepare_state0(
    state_data: dict[str, Any],
) -> tuple[tuple[PreparedModelData, ...], np.ndarray, np.ndarray, np.ndarray]:
    partition = build_partition_from_xin(np.asarray(state_data["xin"]))
    off = build_off_segments(state_data, partition)
    ilc = build_ilc_segments(state_data, partition)
    if len(ilc) != 5:
        raise RuntimeError(f"State0 ILC数量必须为5，实际为{len(ilc)}")
    common_probe = get_common_probe(state_data, partition)
    phi_common = _basis(common_probe)
    real_b_valid = off["B"].output[MAX_DELAY:]

    prepared: list[PreparedModelData] = [
        _prepare_model(
            model_id="X-A",
            behavior_class="X",
            train_segment_name="A",
            ilc_iteration=None,
            train_segment=off["A"],
            b_segment=off["B"],
            b_source="OFF.B",
        )
    ]
    for iteration_index, canonical in enumerate(ilc):
        iteration = iteration_index + 1
        prepared.append(
            _prepare_model(
                model_id=f"Y-A-{iteration}",
                behavior_class="Y",
                train_segment_name="A",
                ilc_iteration=iteration,
                train_segment=canonical["A"],
                b_segment=canonical["B"],
                b_source=f"ILC[{iteration}].B",
            )
        )
    for iteration_index, canonical in enumerate(ilc):
        iteration = iteration_index + 1
        prepared.append(
            _prepare_model(
                model_id=f"Y-C-{iteration}",
                behavior_class="Y",
                train_segment_name="C",
                ilc_iteration=iteration,
                train_segment=canonical["C"],
                b_segment=canonical["B"],
                b_source=f"ILC[{iteration}].B",
            )
        )
    return tuple(prepared), common_probe, phi_common, real_b_valid


def _aggregate(values: list[float]) -> tuple[float, float]:
    return float(np.median(values)), float(np.mean(values))


def _summary_for_lambda(
    ridge_lambda: float,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    x_row = next(row for row in rows if row["model_id"] == "X-A")
    y_a = [row for row in rows if row["model_id"].startswith("Y-A-")]
    y_c = [row for row in rows if row["model_id"].startswith("Y-C-")]

    def stats(group: list[dict[str, Any]], field: str) -> tuple[float, float]:
        return _aggregate([float(row[field]) for row in group])

    ya_train_med, ya_train_mean = stats(y_a, "train_NMSE_dB")
    ya_b_med, ya_b_mean = stats(y_a, "B_generalization_NMSE_dB")
    ya_x_med, ya_x_mean = stats(y_a, "commonB_vs_X_CNMSE_dB")
    ya_real_med, ya_real_mean = stats(y_a, "commonB_vs_realB_CNMSE_dB")
    ya_norm_med, _ = stats(y_a, "theta_l2_norm")
    yc_train_med, yc_train_mean = stats(y_c, "train_NMSE_dB")
    yc_b_med, yc_b_mean = stats(y_c, "B_generalization_NMSE_dB")
    yc_x_med, yc_x_mean = stats(y_c, "commonB_vs_X_CNMSE_dB")
    yc_real_med, yc_real_mean = stats(y_c, "commonB_vs_realB_CNMSE_dB")
    yc_norm_med, _ = stats(y_c, "theta_l2_norm")
    return {
        "ridge_lambda": ridge_lambda,
        "X_train_NMSE_dB": x_row["train_NMSE_dB"],
        "X_B_generalization_NMSE_dB": x_row["B_generalization_NMSE_dB"],
        "X_vs_realB_CNMSE_dB": x_row["commonB_vs_realB_CNMSE_dB"],
        "X_theta_l2_norm": x_row["theta_l2_norm"],
        "Y_A_train_median_dB": ya_train_med,
        "Y_A_train_mean_dB": ya_train_mean,
        "Y_A_B_generalization_median_dB": ya_b_med,
        "Y_A_B_generalization_mean_dB": ya_b_mean,
        "Y_A_vs_X_median_dB": ya_x_med,
        "Y_A_vs_X_mean_dB": ya_x_mean,
        "Y_A_vs_realB_median_dB": ya_real_med,
        "Y_A_vs_realB_mean_dB": ya_real_mean,
        "Y_A_theta_norm_median": ya_norm_med,
        "Y_C_train_median_dB": yc_train_med,
        "Y_C_train_mean_dB": yc_train_mean,
        "Y_C_B_generalization_median_dB": yc_b_med,
        "Y_C_B_generalization_mean_dB": yc_b_mean,
        "Y_C_vs_X_median_dB": yc_x_med,
        "Y_C_vs_X_mean_dB": yc_x_mean,
        "Y_C_vs_realB_median_dB": yc_real_med,
        "Y_C_vs_realB_mean_dB": yc_real_mean,
        "Y_C_theta_norm_median": yc_norm_med,
    }


def _add_improvements(summary_rows: list[dict[str, Any]]) -> None:
    baseline = summary_rows[0]
    metric_pairs = {
        "X_B_gen_improvement_dB": "X_B_generalization_NMSE_dB",
        "Y_A_B_gen_improvement_dB": "Y_A_B_generalization_median_dB",
        "Y_A_vs_X_improvement_dB": "Y_A_vs_X_median_dB",
        "Y_C_train_improvement_dB": "Y_C_train_median_dB",
        "Y_C_B_gen_improvement_dB": "Y_C_B_generalization_median_dB",
        "Y_C_vs_X_improvement_dB": "Y_C_vs_X_median_dB",
        "Y_C_vs_real_improvement_dB": "Y_C_vs_realB_median_dB",
    }
    for row in summary_rows:
        for improvement_name, metric_name in metric_pairs.items():
            row[improvement_name] = float(baseline[metric_name]) - float(row[metric_name])


def _diagnostic_candidate(summary_rows: list[dict[str, Any]]) -> dict[str, Any]:
    positive_rows = [row for row in summary_rows if row["ridge_lambda"] > 0]
    constrained = [
        row
        for row in positive_rows
        if row["X_B_gen_improvement_dB"] >= -1.0
        and row["Y_A_B_gen_improvement_dB"] >= -1.0
        and row["Y_C_train_improvement_dB"] >= -1.0
    ]
    simultaneous = [
        row
        for row in constrained
        if row["Y_C_B_gen_improvement_dB"] > 0
        and row["Y_C_vs_X_improvement_dB"] > 0
        and row["Y_C_vs_real_improvement_dB"] > 0
    ]
    metric_names = (
        "Y_C_B_gen_improvement_dB",
        "Y_C_vs_X_improvement_dB",
        "Y_C_vs_real_improvement_dB",
    )
    best_by_metric: dict[str, dict[str, float | bool]] = {}
    minimum_lambda = float(RIDGE_LAMBDAS[1])
    maximum_lambda = float(RIDGE_LAMBDAS[-1])
    for metric_name in metric_names:
        pool = constrained or positive_rows
        best = max(pool, key=lambda row: row[metric_name])
        best_lambda = float(best["ridge_lambda"])
        best_by_metric[metric_name] = {
            "ridge_lambda": best_lambda,
            "improvement_dB": float(best[metric_name]),
            "on_scan_boundary": best_lambda in (minimum_lambda, maximum_lambda),
        }
    candidate_lambdas = [float(row["ridge_lambda"]) for row in simultaneous]
    touches_boundary = any(
        ridge_lambda in (minimum_lambda, maximum_lambda) for ridge_lambda in candidate_lambdas
    )
    return {
        "selection_role": "diagnostic_only_not_final_hyperparameter_selection",
        "constraint_rule": {
            "X_B_gen_improvement_min_dB": -1.0,
            "Y_A_B_gen_improvement_min_dB": -1.0,
            "Y_C_train_improvement_min_dB": -1.0,
        },
        "constrained_lambdas": [float(row["ridge_lambda"]) for row in constrained],
        "simultaneous_Y_C_improvement_lambdas": candidate_lambdas,
        "candidate_region": (
            None if not candidate_lambdas else [min(candidate_lambdas), max(candidate_lambdas)]
        ),
        "candidate_region_touches_scan_boundary": touches_boundary,
        "best_by_metric": best_by_metric,
        "classification": (
            "RIDGE_SIMULTANEOUS_Y_C_IMPROVEMENT_OBSERVED"
            if candidate_lambdas
            else "NO_SIMULTANEOUS_Y_C_IMPROVEMENT"
        ),
    }


def run_state0_ridge_scan(
    state_data: dict[str, Any] | None = None,
    lambdas: np.ndarray = RIDGE_LAMBDAS,
) -> RidgeScanResult:
    """执行统一lambda扫描；canonical数据和各设计矩阵仅准备一次。"""
    if state_data is None:
        state_data = load_by_id(0)
    normalized_lambdas = np.asarray(lambdas, dtype=np.float64)
    if normalized_lambdas.ndim != 1 or normalized_lambdas.size == 0:
        raise ValueError("lambdas必须是一维非空数组")
    if normalized_lambdas[0] != 0.0:
        raise ValueError("扫描首点必须是lambda=0 OLS基线")
    if not np.all(np.isfinite(normalized_lambdas)) or np.any(normalized_lambdas < 0):
        raise ValueError("lambdas必须全部有限且非负")

    prepared, common_probe, phi_common, real_b_valid = _prepare_state0(state_data)
    model_ids = tuple(item.model_id for item in prepared)
    theta_grid = np.empty(
        (normalized_lambdas.size, len(prepared), NUM_COEFFICIENTS),
        dtype=np.complex128,
    )
    model_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for lambda_index, ridge_lambda in enumerate(normalized_lambdas):
        lambda_rows: list[dict[str, Any]] = []
        common_responses: list[np.ndarray] = []
        for model_index, item in enumerate(prepared):
            theta, diagnostics = fit_coefficients_ridge(
                item.phi_train,
                item.y_train,
                float(ridge_lambda),
            )
            theta_grid[lambda_index, model_index] = theta
            train_prediction = item.phi_train @ theta
            b_prediction = item.phi_b @ theta
            common_response = phi_common @ theta
            common_responses.append(common_response)
            lambda_rows.append(
                {
                    "ridge_lambda": float(ridge_lambda),
                    "model_id": item.model_id,
                    "behavior_class": item.behavior_class,
                    "train_segment": item.train_segment,
                    "ilc_iteration": ("" if item.ilc_iteration is None else item.ilc_iteration),
                    "train_NMSE_dB": calculate_nmse(item.y_train, train_prediction),
                    "B_generalization_NMSE_dB": calculate_nmse(item.y_b, b_prediction),
                    "commonB_vs_X_CNMSE_dB": float("nan"),
                    "commonB_vs_realB_CNMSE_dB": float("nan"),
                    "theta_l2_norm": diagnostics.theta_l2_norm,
                    "rank": diagnostics.rank_phi,
                    "rank_augmented": diagnostics.rank_augmented,
                    "condition_number_phi": diagnostics.condition_number_phi,
                    "condition_number_augmented": diagnostics.condition_number_augmented,
                    "n_train_samples": diagnostics.n_train_samples,
                    "coefficient_count": diagnostics.coefficient_count,
                    "solver_path": diagnostics.solver_path,
                    "B_generalization_source": item.b_generalization_source,
                    "common_B_probe_source": "OFF.B.input",
                    "common_B_response_length": common_response.size,
                }
            )
        x_response = common_responses[0]
        for row, response in zip(lambda_rows, common_responses, strict=True):
            if row["behavior_class"] == "X":
                row["commonB_vs_realB_CNMSE_dB"] = calculate_cnmse(
                    real_b_valid,
                    response,
                )
            else:
                row["commonB_vs_X_CNMSE_dB"] = calculate_cnmse(x_response, response)
                row["commonB_vs_realB_CNMSE_dB"] = calculate_cnmse(
                    real_b_valid,
                    response,
                )
        model_rows.extend(lambda_rows)
        summary_rows.append(_summary_for_lambda(float(ridge_lambda), lambda_rows))

    _add_improvements(summary_rows)
    diagnostic = _diagnostic_candidate(summary_rows)
    diagnostic.update(
        {
            "state_id": 0,
            "lambda_count": normalized_lambdas.size,
            "model_count_per_lambda": len(prepared),
            "total_model_fits": normalized_lambdas.size * len(prepared),
            "train_nmse_count_per_lambda": 11,
            "B_gen_count_per_lambda": 11,
            "X_vs_real_count_per_lambda": 1,
            "Y_vs_X_count_per_lambda": 10,
            "Y_vs_real_count_per_lambda": 10,
            "cnmse_count_per_lambda": 21,
            "A_ownership": 12288,
            "B_ownership": 4915,
            "C_ownership": 7373,
            "A_valid": 12286,
            "B_valid": 4913,
            "C_valid": 7371,
            "used_stale": False,
            "common_probe_source": "OFF.B.input",
            "all_theta_finite": bool(np.all(np.isfinite(theta_grid))),
            "all_rank_full": all(row["rank"] == 10 for row in model_rows),
            "all_theta_norm_nonincreasing_by_model": all(
                np.all(np.diff(np.linalg.norm(theta_grid[:, index, :], axis=1)) <= 1e-10)
                for index in range(len(prepared))
            ),
        }
    )
    return RidgeScanResult(
        lambdas=normalized_lambdas,
        model_ids=model_ids,
        theta=theta_grid,
        model_rows=model_rows,
        summary_rows=summary_rows,
        diagnostic=diagnostic,
        common_probe=common_probe,
        real_b_valid=real_b_valid,
    )


def validate_ols_reproduction(
    result: RidgeScanResult,
    baseline_root: Path,
    *,
    atol: float = 1e-10,
) -> dict[str, Any]:
    """比较lambda=0的11组theta和全部CSV指标与既有OLS xy_equivalence。"""
    with (baseline_root / "state0_xy_equivalence_summary.csv").open(encoding="utf-8-sig") as handle:
        baseline_rows = {row["model_id"]: row for row in csv.DictReader(handle)}
    scan_rows = {row["model_id"]: row for row in result.model_rows if row["ridge_lambda"] == 0.0}
    theta_files = {
        "X-A": "theta_X_A.npy",
        **{f"Y-A-{index}": f"theta_Y_A_iter_{index:03d}.npy" for index in range(1, 6)},
        **{f"Y-C-{index}": f"theta_Y_C_iter_{index:03d}.npy" for index in range(1, 6)},
    }
    metric_fields = (
        "train_NMSE_dB",
        "B_generalization_NMSE_dB",
        "commonB_vs_X_CNMSE_dB",
        "commonB_vs_realB_CNMSE_dB",
    )
    maximum_metric_error = 0.0
    maximum_theta_error = 0.0
    for model_index, model_id in enumerate(result.model_ids):
        baseline_row = baseline_rows[model_id]
        scan_row = scan_rows[model_id]
        for field in metric_fields:
            baseline_value = float(baseline_row[field])
            scan_value = float(scan_row[field])
            if np.isnan(baseline_value) and np.isnan(scan_value):
                continue
            maximum_metric_error = max(
                maximum_metric_error,
                abs(baseline_value - scan_value),
            )
        baseline_theta = np.load(baseline_root / "models" / theta_files[model_id])
        maximum_theta_error = max(
            maximum_theta_error,
            float(np.max(np.abs(baseline_theta - result.theta[0, model_index]))),
        )
    reproduced = maximum_metric_error <= atol and maximum_theta_error <= atol
    if not reproduced:
        raise RuntimeError(
            "lambda=0未复现OLS基线："
            f"metric_error={maximum_metric_error}, theta_error={maximum_theta_error}"
        )
    return {
        "ols_baseline_reproduced": True,
        "ols_comparison_atol": atol,
        "ols_maximum_metric_absolute_error": maximum_metric_error,
        "ols_maximum_theta_absolute_error": maximum_theta_error,
    }


__all__ = [
    "RIDGE_LAMBDAS",
    "PreparedModelData",
    "RidgeScanResult",
    "run_state0_ridge_scan",
    "validate_ols_reproduction",
]
