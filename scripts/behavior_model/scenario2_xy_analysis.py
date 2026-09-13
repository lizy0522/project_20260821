"""
功能说明：按论文正式定义对Scenario 2每状态仅提取X-A、Y-A-1、Y-C-1三个canonical OLS模型。
输入：state_id；只使用OFF原始对和第一次ILC原始输入输出对。
输出：单状态11个有效指标、三组theta/诊断，或425状态长表、宽表、聚合表和验证信息。
用途：建立全425状态正式三模型OLS基线；不使用ILC2以后、STALE、Ridge或其它模型结构。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from data_manager import get_state_info, load_by_id
from signal_segmentation import (
    CanonicalSegment,
    build_off_segments,
    build_partition_from_xin,
    get_common_probe,
    get_ilc_pair,
    preprocess_full_pair,
)

from .basis import build_mp_basis
from .coefficient import _solve_basis_ols
from .config import MAX_DELAY, MP_CONFIG, NUM_COEFFICIENTS
from .evaluation import calculate_cnmse, calculate_nmse

MODEL_IDS = ("X-A", "Y-A-1", "Y-C-1")
METRIC_FIELDS = (
    "train_nmse_db",
    "B_generalization_nmse_db",
    "commonB_vs_X_cnmse_db",
    "commonB_vs_realB_cnmse_db",
)


@dataclass(frozen=True)
class ScenarioModelResult:
    """单状态单模型的OLS系数、四类指标、诊断和数据来源。"""

    model_id: str
    behavior_class: str
    train_segment: str
    ilc_iteration_used: int | None
    theta: np.ndarray
    train_nmse_db: float
    B_generalization_nmse_db: float
    commonB_vs_X_cnmse_db: float
    commonB_vs_realB_cnmse_db: float
    rank: int
    condition_number_phi: float
    theta_l2_norm: float
    n_train: int
    coefficient_count: int
    train_source: str
    B_generalization_source: str
    common_probe_source: str
    common_B_response: np.ndarray


@dataclass(frozen=True)
class StateXYAnalysisResult:
    """一个状态的正式三模型分析结果。"""

    state_id: int
    state_info: dict[str, int | float]
    models: tuple[ScenarioModelResult, ...]
    theta: np.ndarray
    common_probe: np.ndarray
    real_B_valid: np.ndarray
    only_ilc_iteration_1_used: bool


@dataclass
class Scenario2XYAnalysisResult:
    """全425状态正式分析的结果表、theta和机器验收。"""

    state_results: tuple[StateXYAnalysisResult, ...]
    model_metrics: pd.DataFrame
    state_summary: pd.DataFrame
    aggregate_summary: pd.DataFrame
    theta: np.ndarray
    validation: dict[str, Any]


def _basis(x: np.ndarray) -> np.ndarray:
    return build_mp_basis(x, MP_CONFIG["orders"], MP_CONFIG["memory_depth"])


def _fit_model(
    *,
    model_id: str,
    behavior_class: str,
    train_segment_name: str,
    ilc_iteration_used: int | None,
    train_segment: CanonicalSegment,
    B_segment: CanonicalSegment,
    common_probe: np.ndarray,
    train_source: str,
    B_source: str,
) -> ScenarioModelResult:
    phi_train = _basis(train_segment.input)
    y_train = train_segment.output[MAX_DELAY:]
    theta, _, rank, singular_values = _solve_basis_ols(phi_train, y_train)
    condition_number = (
        float("inf")
        if singular_values[-1] == 0
        else float(singular_values[0] / singular_values[-1])
    )
    train_prediction = phi_train @ theta
    B_prediction = _basis(B_segment.input) @ theta
    common_B_response = _basis(common_probe) @ theta
    return ScenarioModelResult(
        model_id=model_id,
        behavior_class=behavior_class,
        train_segment=train_segment_name,
        ilc_iteration_used=ilc_iteration_used,
        theta=theta,
        train_nmse_db=calculate_nmse(y_train, train_prediction),
        B_generalization_nmse_db=calculate_nmse(
            B_segment.output[MAX_DELAY:],
            B_prediction,
        ),
        commonB_vs_X_cnmse_db=float("nan"),
        commonB_vs_realB_cnmse_db=float("nan"),
        rank=int(rank),
        condition_number_phi=condition_number,
        theta_l2_norm=float(np.linalg.norm(theta)),
        n_train=phi_train.shape[0],
        coefficient_count=phi_train.shape[1],
        train_source=train_source,
        B_generalization_source=B_source,
        common_probe_source="OFF.B.input",
        common_B_response=common_B_response,
    )


def analyze_state_xy(
    state_id: int,
    state_data: dict[str, Any] | None = None,
) -> StateXYAnalysisResult:
    """分析一个状态；严格只提取第一次ILC并始终返回三个模型。"""
    if state_data is None:
        state_data = load_by_id(state_id)
    input_history = np.asarray(state_data["xin_pd_ori_ilc"])
    output_history = np.asarray(state_data["yout_withdpd_ori_ilc"])
    if input_history.ndim != 2 or output_history.ndim != 2:
        raise ValueError(f"state_id={state_id} ILC输入输出历史必须为二维")
    if input_history.shape[1] < 1 or output_history.shape[1] < 1:
        raise ValueError(f"state_id={state_id}缺少正式ILC第1列")

    partition = build_partition_from_xin(np.asarray(state_data["xin"]))
    off = build_off_segments(state_data, partition)
    ilc1_pair = get_ilc_pair(state_data, 0)
    ilc1 = preprocess_full_pair(
        ilc1_pair.input_full,
        ilc1_pair.output_raw_full,
        partition,
        pair_type="ILC",
        iteration_index=0,
        input_peak_normalization_factor=ilc1_pair.input_peak_normalization_factor,
    )
    common_probe = get_common_probe(state_data, partition)
    if not np.array_equal(common_probe, off["B"].input):
        raise RuntimeError(f"state_id={state_id}公共probe不是OFF.B.input")
    real_B_valid = off["B"].output[MAX_DELAY:]

    provisional = [
        _fit_model(
            model_id="X-A",
            behavior_class="X",
            train_segment_name="A",
            ilc_iteration_used=None,
            train_segment=off["A"],
            B_segment=off["B"],
            common_probe=common_probe,
            train_source="OFF.A",
            B_source="OFF.B",
        ),
        _fit_model(
            model_id="Y-A-1",
            behavior_class="Y",
            train_segment_name="A",
            ilc_iteration_used=1,
            train_segment=ilc1["A"],
            B_segment=ilc1["B"],
            common_probe=common_probe,
            train_source="ILC1.A",
            B_source="ILC1.B",
        ),
        _fit_model(
            model_id="Y-C-1",
            behavior_class="Y",
            train_segment_name="C",
            ilc_iteration_used=1,
            train_segment=ilc1["C"],
            B_segment=ilc1["B"],
            common_probe=common_probe,
            train_source="ILC1.C",
            B_source="ILC1.B",
        ),
    ]
    x_response = provisional[0].common_B_response
    models: list[ScenarioModelResult] = []
    for model in provisional:
        models.append(
            replace(
                model,
                commonB_vs_X_cnmse_db=(
                    float("nan")
                    if model.behavior_class == "X"
                    else calculate_cnmse(x_response, model.common_B_response)
                ),
                commonB_vs_realB_cnmse_db=calculate_cnmse(
                    real_B_valid,
                    model.common_B_response,
                ),
            )
        )

    theta = np.stack([model.theta for model in models], axis=0)
    result = StateXYAnalysisResult(
        state_id=state_id,
        state_info=get_state_info(state_id),
        models=tuple(models),
        theta=theta,
        common_probe=common_probe,
        real_B_valid=real_B_valid,
        only_ilc_iteration_1_used=all(model.ilc_iteration_used in (None, 1) for model in models),
    )
    _validate_state_result(result)
    return result


def _validate_state_result(result: StateXYAnalysisResult) -> None:
    if tuple(model.model_id for model in result.models) != MODEL_IDS:
        raise RuntimeError(f"state_id={result.state_id}模型ID或顺序错误")
    if result.theta.shape != (3, NUM_COEFFICIENTS):
        raise RuntimeError(f"state_id={result.state_id} theta shape错误：{result.theta.shape}")
    if result.common_probe.shape != (4915,) or result.real_B_valid.shape != (4913,):
        raise RuntimeError(f"state_id={result.state_id}公共B ownership/valid长度错误")
    if not result.only_ilc_iteration_1_used:
        raise RuntimeError(f"state_id={result.state_id}使用了ILC第2次或以后数据")
    for model in result.models:
        if model.theta.shape != (10,) or not np.all(np.isfinite(model.theta)):
            raise RuntimeError(f"state_id={result.state_id} {model.model_id} theta无效")
        if model.common_B_response.shape != (4913,):
            raise RuntimeError(f"state_id={result.state_id}公共B响应长度错误")


def _model_row(
    state_result: StateXYAnalysisResult,
    model: ScenarioModelResult,
) -> dict[str, Any]:
    return {
        "state_id": state_result.state_id,
        **state_result.state_info,
        "model_id": model.model_id,
        "behavior_class": model.behavior_class,
        "train_segment": model.train_segment,
        "ilc_iteration_used": (
            "" if model.ilc_iteration_used is None else model.ilc_iteration_used
        ),
        "train_nmse_db": model.train_nmse_db,
        "B_generalization_nmse_db": model.B_generalization_nmse_db,
        "commonB_vs_X_cnmse_db": model.commonB_vs_X_cnmse_db,
        "commonB_vs_realB_cnmse_db": model.commonB_vs_realB_cnmse_db,
        "rank": model.rank,
        "condition_number_phi": model.condition_number_phi,
        "theta_l2_norm": model.theta_l2_norm,
        "n_train": model.n_train,
        "coefficient_count": model.coefficient_count,
        "train_source": model.train_source,
        "B_generalization_source": model.B_generalization_source,
        "common_probe_source": model.common_probe_source,
    }


def _state_row(result: StateXYAnalysisResult) -> dict[str, Any]:
    models = {model.model_id: model for model in result.models}
    return {
        "state_id": result.state_id,
        **result.state_info,
        "X_train_nmse_db": models["X-A"].train_nmse_db,
        "X_B_gen_nmse_db": models["X-A"].B_generalization_nmse_db,
        "X_vs_realB_cnmse_db": models["X-A"].commonB_vs_realB_cnmse_db,
        "Y_A_train_nmse_db": models["Y-A-1"].train_nmse_db,
        "Y_A_B_gen_nmse_db": models["Y-A-1"].B_generalization_nmse_db,
        "Y_A_vs_X_cnmse_db": models["Y-A-1"].commonB_vs_X_cnmse_db,
        "Y_A_vs_realB_cnmse_db": models["Y-A-1"].commonB_vs_realB_cnmse_db,
        "Y_C_train_nmse_db": models["Y-C-1"].train_nmse_db,
        "Y_C_B_gen_nmse_db": models["Y-C-1"].B_generalization_nmse_db,
        "Y_C_vs_X_cnmse_db": models["Y-C-1"].commonB_vs_X_cnmse_db,
        "Y_C_vs_realB_cnmse_db": models["Y-C-1"].commonB_vs_realB_cnmse_db,
    }


def _aggregate_summary(model_metrics: pd.DataFrame) -> pd.DataFrame:
    statistics = ("count", "mean", "median", "std", "min", "q25", "q75", "max")
    rows: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        group = model_metrics[model_metrics["model_id"] == model_id]
        row: dict[str, Any] = {"model_id": model_id}
        for metric in METRIC_FIELDS:
            values = group[metric].dropna().to_numpy(dtype=float)
            if values.size == 0:
                for statistic in statistics:
                    row[f"{metric}_{statistic}"] = 0 if statistic == "count" else np.nan
                continue
            summary_values = {
                "count": int(values.size),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                "min": float(np.min(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
                "max": float(np.max(values)),
            }
            for statistic, value in summary_values.items():
                row[f"{metric}_{statistic}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _extremes(model_metrics: pd.DataFrame) -> dict[str, Any]:
    extremes: dict[str, Any] = {}
    for model_id in MODEL_IDS:
        group = model_metrics[model_metrics["model_id"] == model_id]
        model_extremes: dict[str, Any] = {}
        for metric in METRIC_FIELDS:
            valid = group[np.isfinite(group[metric].to_numpy(dtype=float))]
            if valid.empty:
                continue
            best = valid.loc[valid[metric].idxmin()]
            worst = valid.loc[valid[metric].idxmax()]
            model_extremes[metric] = {
                "best": {
                    "state_id": int(best["state_id"]),
                    "value_dB": float(best[metric]),
                    "condition": {
                        name: float(best[name]) if name in ("Vm", "Pin") else int(best[name])
                        for name in ("funMng", "funAng", "secMng", "secAng", "Vm", "Pin")
                    },
                },
                "worst": {
                    "state_id": int(worst["state_id"]),
                    "value_dB": float(worst[metric]),
                    "condition": {
                        name: float(worst[name]) if name in ("Vm", "Pin") else int(worst[name])
                        for name in ("funMng", "funAng", "secMng", "secAng", "Vm", "Pin")
                    },
                },
            }
        extremes[model_id] = model_extremes
    return extremes


def collect_scenario2_xy_analysis(
    *,
    state0_result: StateXYAnalysisResult | None = None,
    progress_interval: int = 50,
) -> Scenario2XYAnalysisResult:
    """批量分析state_id 0...424；State0可复用回归结果，避免重复计算。"""
    state_results: list[StateXYAnalysisResult] = []
    for state_id in range(425):
        result = (
            state0_result
            if state_id == 0 and state0_result is not None
            else analyze_state_xy(state_id)
        )
        state_results.append(result)
        if progress_interval > 0 and ((state_id + 1) % progress_interval == 0 or state_id == 424):
            print(f"Processed {state_id + 1} / 425")

    model_rows = [
        _model_row(state_result, model)
        for state_result in state_results
        for model in state_result.models
    ]
    state_rows = [_state_row(result) for result in state_results]
    model_metrics = pd.DataFrame(model_rows)
    state_summary = pd.DataFrame(state_rows)
    aggregate_summary = _aggregate_summary(model_metrics)
    theta = np.stack([result.theta for result in state_results], axis=0)
    rank_deficient = [
        {"state_id": int(row.state_id), "model_id": str(row.model_id), "rank": int(row.rank)}
        for row in model_metrics.itertuples(index=False)
        if row.rank < NUM_COEFFICIENTS
    ]
    validation = {
        "state_count": len(state_results),
        "model_count": model_metrics.shape[0],
        "X_model_count": int(np.count_nonzero(model_metrics["model_id"] == "X-A")),
        "Y_A_model_count": int(np.count_nonzero(model_metrics["model_id"] == "Y-A-1")),
        "Y_C_model_count": int(np.count_nonzero(model_metrics["model_id"] == "Y-C-1")),
        "train_nmse_count": int(np.count_nonzero(np.isfinite(model_metrics["train_nmse_db"]))),
        "B_generalization_nmse_count": int(
            np.count_nonzero(np.isfinite(model_metrics["B_generalization_nmse_db"]))
        ),
        "commonB_vs_X_non_nan_count": int(
            np.count_nonzero(np.isfinite(model_metrics["commonB_vs_X_cnmse_db"]))
        ),
        "commonB_vs_X_expected_nan_count": int(model_metrics["commonB_vs_X_cnmse_db"].isna().sum()),
        "commonB_vs_realB_count": int(
            np.count_nonzero(np.isfinite(model_metrics["commonB_vs_realB_cnmse_db"]))
        ),
        "theta_shape": list(theta.shape),
        "theta_dtype": str(theta.dtype),
        "all_theta_finite": bool(np.all(np.isfinite(theta))),
        "rank_10_count": int(np.count_nonzero(model_metrics["rank"] == NUM_COEFFICIENTS)),
        "rank_deficient_count": len(rank_deficient),
        "rank_deficient_models": rank_deficient,
        "A_ownership": 12288,
        "B_ownership": 4915,
        "C_ownership": 7373,
        "A_valid": 12286,
        "B_valid": 4913,
        "C_valid": 7371,
        "only_ilc_iteration_1_used": all(
            result.only_ilc_iteration_1_used for result in state_results
        ),
        "model_ids": list(MODEL_IDS),
        "used_stale": False,
        "used_ridge": False,
        "extremes": _extremes(model_metrics),
    }
    return Scenario2XYAnalysisResult(
        state_results=tuple(state_results),
        model_metrics=model_metrics,
        state_summary=state_summary,
        aggregate_summary=aggregate_summary,
        theta=theta,
        validation=validation,
    )


def validate_state0_regression(
    state0_result: StateXYAnalysisResult,
    baseline_root: Path,
    *,
    atol: float = 1e-10,
) -> dict[str, float | bool]:
    """对比State0旧正式三模型的theta和四类指标，失败则抛错。"""
    baseline = pd.read_csv(baseline_root / "state0_xy_equivalence_summary.csv")
    baseline = baseline.set_index("model_id")
    theta_files = {
        "X-A": "theta_X_A.npy",
        "Y-A-1": "theta_Y_A_iter_001.npy",
        "Y-C-1": "theta_Y_C_iter_001.npy",
    }
    metric_mapping = {
        "train_nmse_db": "train_NMSE_dB",
        "B_generalization_nmse_db": "B_generalization_NMSE_dB",
        "commonB_vs_X_cnmse_db": "commonB_vs_X_CNMSE_dB",
        "commonB_vs_realB_cnmse_db": "commonB_vs_realB_CNMSE_dB",
    }
    maximum_metric_error = 0.0
    maximum_theta_error = 0.0
    for model in state0_result.models:
        baseline_row = baseline.loc[model.model_id]
        for current_field, baseline_field in metric_mapping.items():
            current_value = float(getattr(model, current_field))
            baseline_value = float(baseline_row[baseline_field])
            if np.isnan(current_value) and np.isnan(baseline_value):
                continue
            maximum_metric_error = max(
                maximum_metric_error,
                abs(current_value - baseline_value),
            )
        baseline_theta = np.load(baseline_root / "models" / theta_files[model.model_id])
        maximum_theta_error = max(
            maximum_theta_error,
            float(np.max(np.abs(model.theta - baseline_theta))),
        )
    passed = maximum_metric_error <= atol and maximum_theta_error <= atol
    if not passed:
        raise RuntimeError(
            "State0正式三模型回归失败："
            f"metric_error={maximum_metric_error}, theta_error={maximum_theta_error}"
        )
    return {
        "state0_regression_passed": True,
        "state0_regression_atol": atol,
        "state0_metric_max_abs_error": maximum_metric_error,
        "state0_theta_max_abs_error": maximum_theta_error,
    }


__all__ = [
    "MODEL_IDS",
    "Scenario2XYAnalysisResult",
    "ScenarioModelResult",
    "StateXYAnalysisResult",
    "analyze_state_xy",
    "collect_scenario2_xy_analysis",
    "validate_state0_regression",
]
