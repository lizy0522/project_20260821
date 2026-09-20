"""
Scenario 2 全状态统一 Ridge 超参数开发、独立验证和最终模型提取。

本模块只改变 10 系数复数 Memory Polynomial 的系数求解器；正式行为定义
仍然是 X-A、Y-A-1 和 Y-C-1，且 Y 只使用 ILC 第一次输入/输出对。所有
canonical 波形均由 signal_segmentation 的冻结接口生成，不在本模块内再次
对齐、复增益或归一化。
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from data_management.shared import build_state_table, get_state_info, load_by_id
from signal_segmentation.shared import (
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
from .ridge import fit_coefficients_ridge, validate_ridge_lambda

MODEL_IDS = ("X-A", "Y-A-1", "Y-C-1")
METRIC_FIELDS = (
    "train_nmse_db",
    "B_generalization_nmse_db",
    "commonB_vs_X_cnmse_db",
    "commonB_vs_realB_cnmse_db",
)
STATE_COUNT = 425
RANDOM_SEED = 20260827
DEVELOPMENT_COUNT = 340
VALIDATION_COUNT = 85

COARSE_LAMBDAS = np.asarray(
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
        1.0,
        10.0,
        100.0,
    ],
    dtype=np.float64,
)

SPLIT_COLUMNS = (
    "state_id",
    "funMng",
    "funAng",
    "secMng",
    "secAng",
    "Vm",
    "Pin",
    "state_type",
    "stratum_key",
    "split",
)

SELECTION_CONSTRAINTS = {
    "median_X_B_gen_degradation_dB_max": 0.5,
    "median_YA_B_gen_degradation_dB_max": 0.5,
    "median_YC_train_degradation_dB_max": 1.0,
    "median_YC_B_gen_gain_dB_min_exclusive": 0.0,
    "median_YC_vs_X_gain_dB_min": 0.0,
    "median_YC_vs_real_gain_dB_min": 0.0,
}

_FOCUS = {
    "YC_B_gen": ("Y-C-1", "B_generalization_nmse_db"),
    "YC_vs_X": ("Y-C-1", "commonB_vs_X_cnmse_db"),
    "YC_vs_real": ("Y-C-1", "commonB_vs_realB_cnmse_db"),
}
_PROTECTION = {
    "X_B_gen": ("X-A", "B_generalization_nmse_db"),
    "YA_B_gen": ("Y-A-1", "B_generalization_nmse_db"),
    "YC_train": ("Y-C-1", "train_nmse_db"),
}


@dataclass(frozen=True)
class PreparedModelData:
    """一个模型在所有 lambda 下不变的设计矩阵和参考数据。"""

    model_id: str
    behavior_class: str
    train_segment: str
    ilc_iteration: int | None
    phi_train: np.ndarray
    y_train: np.ndarray
    phi_b: np.ndarray
    y_b: np.ndarray
    train_source: str
    b_generalization_source: str
    ols_theta: np.ndarray
    ols_rank: int
    ols_singular_values: np.ndarray


@dataclass(frozen=True)
class PreparedStateData:
    """一个状态的正式三模型设计矩阵、公共 probe 和真实 B 参考。"""

    state_id: int
    state_info: dict[str, int | float]
    models: tuple[PreparedModelData, ...]
    phi_common_b: np.ndarray
    common_probe: np.ndarray
    real_b_valid: np.ndarray


@dataclass(frozen=True)
class StateLambdaEvaluation:
    """单状态单 lambda 的三模型结果和系数。"""

    state_id: int
    ridge_lambda: float
    rows: tuple[dict[str, Any], ...]
    theta: np.ndarray


@dataclass(frozen=True)
class FullComparisonResult:
    """一组状态的 OLS 与冻结 Ridge 配对结果。"""

    state_ids: tuple[int, ...]
    ols_metrics: pd.DataFrame
    ridge_metrics: pd.DataFrame
    state_summary: pd.DataFrame
    aggregate_summary: pd.DataFrame
    theta_ols: np.ndarray
    theta_ridge: np.ndarray


def _basis(x: np.ndarray) -> np.ndarray:
    return build_mp_basis(x, MP_CONFIG["orders"], MP_CONFIG["memory_depth"])


def _slug(value: str) -> str:
    return value.replace("-", "_")


def _prepare_model(
    *,
    model_id: str,
    behavior_class: str,
    train_segment_name: str,
    ilc_iteration: int | None,
    train_segment: CanonicalSegment,
    b_segment: CanonicalSegment,
    train_source: str,
    b_source: str,
) -> PreparedModelData:
    phi_train = _basis(train_segment.input)
    phi_b = _basis(b_segment.input)
    y_train = np.asarray(train_segment.output[MAX_DELAY:])
    y_b = np.asarray(b_segment.output[MAX_DELAY:])
    if phi_train.shape[0] != y_train.size or phi_b.shape[0] != y_b.size:
        raise RuntimeError(f"{model_id}设计矩阵与输出长度不一致")
    ols_theta, _, ols_rank, ols_singular_values = _solve_basis_ols(phi_train, y_train)
    return PreparedModelData(
        model_id=model_id,
        behavior_class=behavior_class,
        train_segment=train_segment_name,
        ilc_iteration=ilc_iteration,
        phi_train=phi_train,
        y_train=y_train,
        phi_b=phi_b,
        y_b=y_b,
        train_source=train_source,
        b_generalization_source=b_source,
        ols_theta=ols_theta,
        ols_rank=int(ols_rank),
        ols_singular_values=ols_singular_values,
    )


def prepare_state_ridge_data(
    state_id: int,
    state_data: dict[str, Any] | None = None,
) -> PreparedStateData:
    """准备一个状态的 OFF/ILC1 canonical 设计矩阵；不使用其它 ILC 或 STALE。"""

    if not isinstance(state_id, Integral) or isinstance(state_id, bool):
        raise TypeError("state_id必须是整数")
    normalized_id = int(state_id)
    if not 0 <= normalized_id < STATE_COUNT:
        raise IndexError(f"state_id必须位于[0, {STATE_COUNT - 1}]")
    if state_data is None:
        state_data = load_by_id(normalized_id)

    input_history = np.asarray(state_data["xin_pd_ori_ilc"])
    output_history = np.asarray(state_data["yout_withdpd_ori_ilc"])
    if input_history.ndim != 2 or output_history.ndim != 2:
        raise ValueError(f"state_id={normalized_id} ILC历史必须是二维数组")
    if input_history.shape[1] < 1 or output_history.shape[1] < 1:
        raise ValueError(f"state_id={normalized_id}缺少ILC第1列")

    partition = build_partition_from_xin(np.asarray(state_data["xin"]))
    off = build_off_segments(state_data, partition)
    ilc1_pair = get_ilc_pair(state_data, 0)
    if ilc1_pair.iteration_index != 0:
        raise RuntimeError("正式Y行为必须显式使用ILC第1列")
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
        raise RuntimeError(f"state_id={normalized_id}公共probe不是OFF.B.input")
    real_b_valid = np.asarray(off["B"].output[MAX_DELAY:])
    phi_common_b = _basis(common_probe)
    if phi_common_b.shape[0] != real_b_valid.size:
        raise RuntimeError(f"state_id={normalized_id}公共B参考长度错误")

    models = (
        _prepare_model(
            model_id="X-A",
            behavior_class="X",
            train_segment_name="A",
            ilc_iteration=None,
            train_segment=off["A"],
            b_segment=off["B"],
            train_source="OFF.A",
            b_source="OFF.B",
        ),
        _prepare_model(
            model_id="Y-A-1",
            behavior_class="Y",
            train_segment_name="A",
            ilc_iteration=1,
            train_segment=ilc1["A"],
            b_segment=ilc1["B"],
            train_source="ILC1.A",
            b_source="ILC1.B",
        ),
        _prepare_model(
            model_id="Y-C-1",
            behavior_class="Y",
            train_segment_name="C",
            ilc_iteration=1,
            train_segment=ilc1["C"],
            b_segment=ilc1["B"],
            train_source="ILC1.C",
            b_source="ILC1.B",
        ),
    )
    return PreparedStateData(
        state_id=normalized_id,
        state_info=get_state_info(normalized_id),
        models=models,
        phi_common_b=phi_common_b,
        common_probe=common_probe,
        real_b_valid=real_b_valid,
    )


def evaluate_prepared_state(
    prepared: PreparedStateData,
    ridge_lambda: Real = 0.0,
) -> StateLambdaEvaluation:
    """在已准备状态上拟合统一 lambda，并计算四类正式指标。"""

    normalized_lambda = validate_ridge_lambda(ridge_lambda)
    fitted: list[tuple[PreparedModelData, np.ndarray, Any]] = []
    for model in prepared.models:
        theta, diagnostics = fit_coefficients_ridge(
            model.phi_train,
            model.y_train,
            normalized_lambda,
            ols_solution=(model.ols_theta, model.ols_rank, model.ols_singular_values),
        )
        fitted.append((model, theta, diagnostics))

    responses = {
        model.model_id: prepared.phi_common_b @ theta
        for model, theta, _ in fitted
    }
    x_response = responses["X-A"]
    rows: list[dict[str, Any]] = []
    for model, theta, diagnostics in fitted:
        train_prediction = model.phi_train @ theta
        b_prediction = model.phi_b @ theta
        common_prediction = responses[model.model_id]
        row: dict[str, Any] = {
            "state_id": prepared.state_id,
            **prepared.state_info,
            "ridge_lambda": normalized_lambda,
            "model_id": model.model_id,
            "behavior_class": model.behavior_class,
            "train_segment": model.train_segment,
            "ilc_iteration_used": model.ilc_iteration,
            "train_nmse_db": calculate_nmse(model.y_train, train_prediction),
            "B_generalization_nmse_db": calculate_nmse(model.y_b, b_prediction),
            "commonB_vs_X_cnmse_db": (
                float("nan")
                if model.model_id == "X-A"
                else calculate_cnmse(x_response, common_prediction)
            ),
            "commonB_vs_realB_cnmse_db": calculate_cnmse(
                prepared.real_b_valid,
                common_prediction,
            ),
            "theta_l2_norm": diagnostics.theta_l2_norm,
            "rank": diagnostics.rank_phi,
            "rank_augmented": diagnostics.rank_augmented,
            "condition_number_phi": diagnostics.condition_number_phi,
            "condition_number_augmented": diagnostics.condition_number_augmented,
            "residual_l2": diagnostics.residual_l2,
            "n_train": diagnostics.n_train_samples,
            "coefficient_count": diagnostics.coefficient_count,
            "solver_path": diagnostics.solver_path,
            "train_source": model.train_source,
            "B_generalization_source": model.b_generalization_source,
            "common_probe_source": "OFF.B.input",
        }
        rows.append(row)

    theta = np.stack([theta for _, theta, _ in fitted], axis=0)
    if theta.shape != (len(MODEL_IDS), NUM_COEFFICIENTS):
        raise RuntimeError(f"state_id={prepared.state_id} theta shape错误：{theta.shape}")
    if not np.all(np.isfinite(theta)):
        raise RuntimeError(f"state_id={prepared.state_id} theta包含非有限值")
    return StateLambdaEvaluation(
        state_id=prepared.state_id,
        ridge_lambda=normalized_lambda,
        rows=tuple(rows),
        theta=theta,
    )


def _normalize_state_ids(state_ids: Iterable[int]) -> tuple[int, ...]:
    normalized: list[int] = []
    for value in state_ids:
        if not isinstance(value, Integral) or isinstance(value, bool):
            raise TypeError("state_ids必须只包含整数")
        state_id = int(value)
        if not 0 <= state_id < STATE_COUNT:
            raise IndexError(f"state_id必须位于[0, {STATE_COUNT - 1}]")
        normalized.append(state_id)
    if not normalized:
        raise ValueError("state_ids不能为空")
    if len(set(normalized)) != len(normalized):
        raise ValueError("state_ids不能重复")
    return tuple(normalized)


def _normalize_lambda_grid(lambdas: Sequence[Real]) -> tuple[float, ...]:
    if len(lambdas) == 0:
        raise ValueError("lambda网格不能为空")
    normalized = tuple(validate_ridge_lambda(value) for value in lambdas)
    if len(set(normalized)) != len(normalized):
        raise ValueError("lambda网格不能包含重复值")
    return normalized


def validate_development_state_ids(
    state_ids: Iterable[int],
    development_state_ids: Iterable[int],
    validation_state_ids: Iterable[int],
) -> tuple[int, ...]:
    """验证扫描输入只属于 Development，并显式阻止 Validation 泄漏。"""

    normalized = _normalize_state_ids(state_ids)
    development = set(_normalize_state_ids(development_state_ids))
    validation = set(_normalize_state_ids(validation_state_ids))
    if development & validation:
        raise ValueError("Development 与 Validation 不能相交")
    if set(normalized) - development:
        raise ValueError("lambda扫描只能接收development_state_ids")
    if set(normalized) & validation:
        raise ValueError("Validation状态不能进入lambda扫描")
    return normalized


def run_lambda_scan(
    state_ids: Iterable[int],
    lambdas: Sequence[Real],
    *,
    development_state_ids: Iterable[int],
    validation_state_ids: Iterable[int],
    scan_stage: str,
    progress_interval: int = 25,
) -> pd.DataFrame:
    """逐状态执行 Development 粗/细扫描；不会同时驻留全部状态设计矩阵。"""

    normalized_states = validate_development_state_ids(
        state_ids,
        development_state_ids,
        validation_state_ids,
    )
    normalized_lambdas = _normalize_lambda_grid(lambdas)
    rows: list[dict[str, Any]] = []
    for position, state_id in enumerate(normalized_states, start=1):
        prepared = prepare_state_ridge_data(state_id)
        for ridge_lambda in normalized_lambdas:
            evaluation = evaluate_prepared_state(prepared, ridge_lambda)
            for row in evaluation.rows:
                row["scan_stage"] = scan_stage
            rows.extend(evaluation.rows)
        if progress_interval > 0 and (
            position % progress_interval == 0 or position == len(normalized_states)
        ):
            print(f"{scan_stage}: processed {position} / {len(normalized_states)} states")
    return pd.DataFrame(rows)


def run_full_lambda_zero_regression(
    state_ids: Iterable[int] | None = None,
    *,
    progress_interval: int = 50,
) -> tuple[pd.DataFrame, np.ndarray]:
    """对指定全状态集合执行新 pipeline 的 lambda=0 OLS 路径。"""

    normalized_states = _normalize_state_ids(
        tuple(range(STATE_COUNT)) if state_ids is None else state_ids
    )
    rows: list[dict[str, Any]] = []
    theta_blocks: list[np.ndarray] = []
    for position, state_id in enumerate(normalized_states, start=1):
        prepared = prepare_state_ridge_data(state_id)
        evaluation = evaluate_prepared_state(prepared, 0.0)
        rows.extend(evaluation.rows)
        theta_blocks.append(evaluation.theta)
        if progress_interval > 0 and (
            position % progress_interval == 0 or position == len(normalized_states)
        ):
            print(f"lambda=0 regression: processed {position} / {len(normalized_states)} states")
    return pd.DataFrame(rows), np.stack(theta_blocks, axis=0)


def compare_lambda_zero_to_baseline(
    current_metrics: pd.DataFrame,
    current_theta: np.ndarray,
    baseline_root: Path,
    *,
    atol: float = 1e-10,
) -> dict[str, Any]:
    """逐状态比较新 lambda=0 结果与正式 Scenario 2 OLS oracle。"""

    baseline_metrics_path = baseline_root / "scenario2_xy_model_metrics.csv"
    baseline_theta_path = baseline_root / "scenario2_xy_theta.npz"
    if not baseline_metrics_path.is_file() or not baseline_theta_path.is_file():
        raise FileNotFoundError("缺少Scenario 2正式OLS基线文件")
    baseline_metrics = pd.read_csv(baseline_metrics_path)
    metric_keys = ["state_id", "model_id", *METRIC_FIELDS]
    current = current_metrics[current_metrics["ridge_lambda"] == 0.0].copy()
    if current.shape[0] != STATE_COUNT * len(MODEL_IDS):
        raise RuntimeError("lambda=0回归模型数量不是1275")
    if baseline_metrics.shape[0] != STATE_COUNT * len(MODEL_IDS):
        raise RuntimeError("正式OLS基线模型数量不是1275")
    merged = baseline_metrics[metric_keys].merge(
        current[metric_keys],
        on=["state_id", "model_id"],
        how="outer",
        suffixes=("_baseline", "_current"),
        indicator=True,
        validate="one_to_one",
    )
    if not (merged["_merge"] == "both").all():
        raise RuntimeError("lambda=0回归与正式OLS基线的状态/模型键不一致")

    maximum_metric_error = 0.0
    for metric in METRIC_FIELDS:
        baseline_values = merged[f"{metric}_baseline"].to_numpy(dtype=float)
        current_values = merged[f"{metric}_current"].to_numpy(dtype=float)
        finite_match = np.isfinite(baseline_values) == np.isfinite(current_values)
        if not np.all(finite_match):
            maximum_metric_error = float("inf")
            break
        finite = np.isfinite(baseline_values)
        if np.any(finite):
            maximum_metric_error = max(
                maximum_metric_error,
                float(np.max(np.abs(baseline_values[finite] - current_values[finite]))),
            )

    baseline_npz = np.load(baseline_theta_path)
    baseline_theta = np.asarray(baseline_npz["theta"])
    if current_theta.shape != baseline_theta.shape:
        maximum_theta_error = float("inf")
    else:
        maximum_theta_error = float(np.max(np.abs(current_theta - baseline_theta)))
    passed = bool(maximum_metric_error <= atol and maximum_theta_error <= atol)
    return {
        "lambda_zero_regression_passed": passed,
        "state_count": STATE_COUNT,
        "model_count": STATE_COUNT * len(MODEL_IDS),
        "theta_shape": list(current_theta.shape),
        "baseline_theta_shape": list(baseline_theta.shape),
        "theta_max_abs_error": maximum_theta_error,
        "metric_max_abs_error": maximum_metric_error,
        "atol": atol,
    }


def _state_type_and_stratum(state: dict[str, int | float]) -> tuple[str, str]:
    state_id = int(state["state_id"])
    fun_mng = int(state["funMng"])
    sec_mng = int(state["secMng"])
    if state_id == 0:
        return "reference", "state0"
    if fun_mng != 0 and sec_mng == 0:
        return "fundamental_only", f"fundamental_funMng_{fun_mng}"
    if fun_mng == 0 and sec_mng != 0:
        return "second_harmonic_only", f"second_harmonic_secMng_{sec_mng}"
    if fun_mng != 0 and sec_mng != 0:
        return "joint", f"joint_funMng_{fun_mng}_secMng_{sec_mng}"
    raise RuntimeError(f"state_id={state_id}不属于预期失配状态")


def _validate_state_split(frame: pd.DataFrame) -> pd.DataFrame:
    missing = [column for column in SPLIT_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"state_split.csv缺少字段：{missing}")
    normalized = frame.copy()
    normalized["state_id"] = normalized["state_id"].astype(int)
    normalized = normalized.sort_values("state_id").reset_index(drop=True)
    if tuple(normalized["state_id"]) != tuple(range(STATE_COUNT)):
        raise ValueError("state_split必须包含0...424且不能重复")
    expected = pd.DataFrame(build_state_table())
    for column in ("funMng", "funAng", "secMng", "secAng", "Vm", "Pin"):
        if not np.allclose(
            normalized[column].to_numpy(dtype=float),
            expected[column].to_numpy(dtype=float),
            rtol=0,
            atol=0,
        ):
            raise ValueError(f"state_split中的{column}与状态表不一致")
    expected_types = [_state_type_and_stratum(row) for row in expected.to_dict("records")]
    actual_types = list(
        zip(normalized["state_type"], normalized["stratum_key"], strict=True)
    )
    if actual_types != expected_types:
        raise ValueError("state_split的state_type/stratum_key不一致")
    if set(normalized["split"]) != {"development", "validation"}:
        raise ValueError("state_split.split必须同时包含development和validation")
    if int((normalized["split"] == "development").sum()) != DEVELOPMENT_COUNT:
        raise ValueError("Development状态数必须为340")
    if int((normalized["split"] == "validation").sum()) != VALIDATION_COUNT:
        raise ValueError("Validation状态数必须为85")
    if normalized.loc[normalized["state_id"] == 0, "split"].item() != "development":
        raise ValueError("state0必须固定放入Development")
    nonzero_validation_count = normalized.loc[
        normalized["state_id"] != 0, "split"
    ].eq("validation").sum()
    if nonzero_validation_count != VALIDATION_COUNT:
        raise ValueError("非State0 Validation状态数必须为85")
    nonzero_strata = normalized.loc[normalized["state_id"] != 0, "stratum_key"]
    if nonzero_strata.nunique() != 11:
        raise ValueError("失配状态必须形成11个strata")
    return normalized[list(SPLIT_COLUMNS)]


def create_state_split(
    *,
    seed: int = RANDOM_SEED,
    development_count: int = DEVELOPMENT_COUNT,
    validation_count: int = VALIDATION_COUNT,
) -> pd.DataFrame:
    """按11个strata和largest-remainder规则确定性生成状态划分。"""

    if development_count + validation_count != STATE_COUNT:
        raise ValueError("Development与Validation数量必须合计425")
    state_rows = build_state_table()
    rows: list[dict[str, Any]] = []
    groups: dict[str, list[int]] = {}
    for state in state_rows:
        state_type, stratum_key = _state_type_and_stratum(state)
        if int(state["state_id"]) != 0:
            groups.setdefault(stratum_key, []).append(int(state["state_id"]))
        rows.append({**state, "state_type": state_type, "stratum_key": stratum_key})

    target_validation_nonzero = validation_count
    stratum_keys = sorted(groups)
    sizes = np.asarray([len(groups[key]) for key in stratum_keys], dtype=float)
    quotas = target_validation_nonzero * sizes / float(np.sum(sizes))
    allocation = np.floor(quotas).astype(int)
    remainder = target_validation_nonzero - int(np.sum(allocation))
    rng = np.random.default_rng(seed)
    tie_order = rng.permutation(len(stratum_keys))
    tie_rank = {int(index): rank for rank, index in enumerate(tie_order)}
    order = sorted(
        range(len(stratum_keys)),
        key=lambda index: (-float(quotas[index] - allocation[index]), tie_rank[index]),
    )
    for index in order[:remainder]:
        allocation[index] += 1
    if int(np.sum(allocation)) != target_validation_nonzero:
        raise RuntimeError("largest-remainder分配未达到Validation目标数")

    validation_ids: set[int] = set()
    for key, count in zip(stratum_keys, allocation, strict=True):
        shuffled = rng.permutation(groups[key])
        validation_ids.update(int(value) for value in shuffled[: int(count)])
    if len(validation_ids) != validation_count:
        raise RuntimeError("Validation状态数生成错误")

    manifest_rows: list[dict[str, Any]] = []
    for row in rows:
        state_id = int(row["state_id"])
        manifest_rows.append({
            **row,
            "split": "validation" if state_id in validation_ids else "development",
        })
    return _validate_state_split(pd.DataFrame(manifest_rows))


def load_or_create_state_split(manifest_path: Path) -> pd.DataFrame:
    """首次生成manifest，后续只读取并验证，不重新划分。"""

    if manifest_path.exists():
        return _validate_state_split(pd.read_csv(manifest_path))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    frame = create_state_split()
    frame.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    return frame


def _finite_values(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=float)
    return array[np.isfinite(array)]


def _stats(values: Iterable[float]) -> dict[str, float | int]:
    finite = _finite_values(values)
    if finite.size == 0:
        return {
            "count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
        }
    return {
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std": float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0,
        "q25": float(np.quantile(finite, 0.25)),
        "q75": float(np.quantile(finite, 0.75)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
    }


def add_paired_gain_columns(metrics: pd.DataFrame) -> pd.DataFrame:
    """以每状态 lambda=0 为基线添加 gain/degradation 列。"""

    required = {"ridge_lambda", "state_id", "model_id", *METRIC_FIELDS}
    missing = sorted(required - set(metrics.columns))
    if missing:
        raise ValueError(f"扫描结果缺少配对字段：{missing}")
    baseline = metrics.loc[metrics["ridge_lambda"] == 0.0, [
        "state_id",
        "model_id",
        *METRIC_FIELDS,
    ]].copy()
    if baseline.duplicated(["state_id", "model_id"]).any():
        raise ValueError("lambda=0基线每状态每模型必须唯一")
    baseline = baseline.rename(columns={metric: f"{metric}_ols" for metric in METRIC_FIELDS})
    paired = metrics.merge(
        baseline,
        on=["state_id", "model_id"],
        how="left",
        sort=False,
        validate="many_to_one",
    )
    if paired[[f"{metric}_ols" for metric in METRIC_FIELDS]].isna().all(axis=1).any():
        raise ValueError("存在没有lambda=0配对基线的扫描行")
    for metric in METRIC_FIELDS:
        current = paired[metric].to_numpy(dtype=float)
        reference = paired[f"{metric}_ols"].to_numpy(dtype=float)
        current_finite = np.isfinite(current)
        reference_finite = np.isfinite(reference)
        if not np.array_equal(current_finite, reference_finite):
            raise ValueError(f"{metric}的OLS/Ridge有限性掩码不一致")
        gain = np.full(current.shape, np.nan, dtype=float)
        degradation = np.full(current.shape, np.nan, dtype=float)
        gain[current_finite] = reference[current_finite] - current[current_finite]
        degradation[current_finite] = current[current_finite] - reference[current_finite]
        paired[f"gain_{metric}"] = gain
        paired[f"degradation_{metric}"] = degradation
        paired.drop(columns=[f"{metric}_ols"], inplace=True)
    return paired


def _summary_metric_stats(
    row: dict[str, Any],
    group: pd.DataFrame,
    *,
    value_column: str,
    prefix: str,
) -> None:
    for statistic, value in _stats(group[value_column].to_numpy(dtype=float)).items():
        row[f"{prefix}_{statistic}"] = value


def _summary_for_lambda(group: pd.DataFrame, ridge_lambda: float) -> dict[str, Any]:
    row: dict[str, Any] = {
        "ridge_lambda": float(ridge_lambda),
        "state_count": int(group["state_id"].nunique()),
        "model_count": int(group.shape[0]),
        "rank_min": int(group["rank"].min()),
        "rank_deficient_count": int(np.count_nonzero(group["rank"].to_numpy() < NUM_COEFFICIENTS)),
        "all_theta_finite": bool(np.all(np.isfinite(group["theta_l2_norm"].to_numpy(dtype=float)))),
    }
    for model_id in MODEL_IDS:
        model_group = group[group["model_id"] == model_id]
        model_slug = _slug(model_id)
        for metric in METRIC_FIELDS:
            _summary_metric_stats(
                row,
                model_group,
                value_column=metric,
                prefix=f"{model_slug}_{metric}",
            )
        for kind in ("gain", "degradation"):
            for metric in METRIC_FIELDS:
                _summary_metric_stats(
                    row,
                    model_group,
                    value_column=f"{kind}_{metric}",
                    prefix=f"{model_slug}_{kind}_{metric}",
                )
        _summary_metric_stats(
            row,
            model_group,
            value_column="theta_l2_norm",
            prefix=f"{model_slug}_theta_l2_norm",
        )

    for label, (model_id, metric) in _FOCUS.items():
        model_group = group[group["model_id"] == model_id]
        _summary_metric_stats(
            row,
            model_group,
            value_column=f"gain_{metric}",
            prefix=f"{label}_gain_dB",
        )
        row[f"{label}_improved_count"] = int(
            np.count_nonzero(model_group[f"gain_{metric}"].to_numpy(dtype=float) > 0)
        )
        row[f"{label}_degraded_count"] = int(
            np.count_nonzero(model_group[f"gain_{metric}"].to_numpy(dtype=float) < 0)
        )
        valid_count = int(
            np.count_nonzero(
                np.isfinite(model_group[f"gain_{metric}"].to_numpy(dtype=float))
            )
        )
        row[f"{label}_improved_fraction"] = (
            float(row[f"{label}_improved_count"] / valid_count) if valid_count else float("nan")
        )
    for label, (model_id, metric) in _PROTECTION.items():
        model_group = group[group["model_id"] == model_id]
        _summary_metric_stats(
            row,
            model_group,
            value_column=f"degradation_{metric}",
            prefix=f"{label}_degradation_dB",
        )

    row["median_X_B_gen_degradation_dB"] = row["X_B_gen_degradation_dB_median"]
    row["median_YA_B_gen_degradation_dB"] = row["YA_B_gen_degradation_dB_median"]
    row["median_YC_train_degradation_dB"] = row["YC_train_degradation_dB_median"]
    row["median_YC_B_gen_gain_dB"] = row["YC_B_gen_gain_dB_median"]
    row["median_YC_vs_X_gain_dB"] = row["YC_vs_X_gain_dB_median"]
    row["median_YC_vs_real_gain_dB"] = row["YC_vs_real_gain_dB_median"]
    row["feasible"] = _is_feasible_nonzero_lambda(row)
    return row


def _is_feasible_nonzero_lambda(row: dict[str, Any]) -> bool:
    if float(row["ridge_lambda"]) <= 0:
        return False
    required = (
        "median_X_B_gen_degradation_dB",
        "median_YA_B_gen_degradation_dB",
        "median_YC_train_degradation_dB",
        "median_YC_B_gen_gain_dB",
        "median_YC_vs_X_gain_dB",
        "median_YC_vs_real_gain_dB",
    )
    if not all(np.isfinite(float(row[name])) for name in required):
        return False
    return bool(
        float(row["median_X_B_gen_degradation_dB"])
        <= SELECTION_CONSTRAINTS["median_X_B_gen_degradation_dB_max"]
        and float(row["median_YA_B_gen_degradation_dB"])
        <= SELECTION_CONSTRAINTS["median_YA_B_gen_degradation_dB_max"]
        and float(row["median_YC_train_degradation_dB"])
        <= SELECTION_CONSTRAINTS["median_YC_train_degradation_dB_max"]
        and float(row["median_YC_B_gen_gain_dB"])
        > SELECTION_CONSTRAINTS["median_YC_B_gen_gain_dB_min_exclusive"]
        and float(row["median_YC_vs_X_gain_dB"])
        >= SELECTION_CONSTRAINTS["median_YC_vs_X_gain_dB_min"]
        and float(row["median_YC_vs_real_gain_dB"])
        >= SELECTION_CONSTRAINTS["median_YC_vs_real_gain_dB_min"]
    )


def build_scan_summary(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """添加paired列并返回逐模型长表和每lambda聚合表。"""

    paired = add_paired_gain_columns(metrics)
    summary_rows = [
        _summary_for_lambda(
            paired[paired["ridge_lambda"] == ridge_lambda],
            float(ridge_lambda),
        )
        for ridge_lambda in sorted(paired["ridge_lambda"].unique())
    ]
    return paired, pd.DataFrame(summary_rows)


def select_lambda(
    summary: pd.DataFrame,
    *,
    stage: str,
    coarse_winner: float | None = None,
) -> dict[str, Any]:
    """按预声明约束和字典序选择统一非零 lambda。"""

    if "feasible" not in summary.columns:
        raise ValueError("summary缺少feasible字段")
    candidates = summary[
        (summary["ridge_lambda"] > 0) & summary["feasible"].astype(bool)
    ].copy()
    payload: dict[str, Any] = {
        "stage": stage,
        "coarse_winner": None if coarse_winner is None else float(coarse_winner),
        "selection_constraints": dict(SELECTION_CONSTRAINTS),
        "candidate_count": int(candidates.shape[0]),
    }
    if candidates.empty:
        payload.update({
            "status": "NO_FEASIBLE_NONZERO_LAMBDA",
            "selected_lambda": None,
        })
        return payload

    ordered = sorted(
        candidates.to_dict("records"),
        key=lambda row: (
            -float(row["median_YC_B_gen_gain_dB"]),
            -float(row["median_YC_vs_X_gain_dB"]),
            -float(row["median_YC_vs_real_gain_dB"]),
            float(row["ridge_lambda"]),
        ),
    )
    winner = ordered[0]
    payload.update({
        "status": "SELECTED",
        "selected_lambda": float(winner["ridge_lambda"]),
        "selected_summary": {
            key: _json_scalar(value) for key, value in winner.items()
        },
    })
    return payload


def make_fine_lambda_grid(coarse_winner: Real) -> np.ndarray:
    """围绕粗赢家上下各一个decade生成17点细网格并加入lambda=0。"""

    center = validate_ridge_lambda(coarse_winner)
    if center <= 0:
        raise ValueError("细扫描中心必须是非零正lambda")
    grid = np.logspace(np.log10(center) - 1.0, np.log10(center) + 1.0, 17)
    return np.unique(np.concatenate(([0.0], grid.astype(np.float64))))


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, Integral)):
        return int(value)
    if isinstance(value, (np.floating, Real)):
        value_float = float(value)
        return value_float if np.isfinite(value_float) else None
    if pd.isna(value):
        return None
    return value


def _merge_comparison_rows(
    ols_metrics: pd.DataFrame,
    ridge_metrics: pd.DataFrame,
) -> pd.DataFrame:
    keys = ["state_id", "model_id"]
    fields = [*METRIC_FIELDS, "theta_l2_norm", "rank", "condition_number_phi"]
    ols = ols_metrics[keys + fields].rename(
        columns={field: f"{field}_ols" for field in fields}
    )
    ridge = ridge_metrics[keys + fields].rename(
        columns={field: f"{field}_ridge" for field in fields}
    )
    merged = ridge.merge(ols, on=keys, how="outer", validate="one_to_one")
    if merged.shape[0] != ols_metrics.shape[0] or merged.isna().all(axis=1).any():
        raise ValueError("OLS/Ridge配对行数或键不一致")
    for metric in METRIC_FIELDS:
        ridge_values = merged[f"{metric}_ridge"].to_numpy(dtype=float)
        ols_values = merged[f"{metric}_ols"].to_numpy(dtype=float)
        ridge_finite = np.isfinite(ridge_values)
        ols_finite = np.isfinite(ols_values)
        if not np.array_equal(ridge_finite, ols_finite):
            raise ValueError(f"{metric}的OLS/Ridge有限性掩码不一致")
        gain = np.full(ridge_values.shape, np.nan, dtype=float)
        degradation = np.full(ridge_values.shape, np.nan, dtype=float)
        gain[ridge_finite] = ols_values[ridge_finite] - ridge_values[ridge_finite]
        degradation[ridge_finite] = ridge_values[ridge_finite] - ols_values[ridge_finite]
        merged[f"{metric}_gain_db"] = gain
        merged[f"{metric}_degradation_db"] = degradation
    theta_ols = merged["theta_l2_norm_ols"].to_numpy(dtype=float)
    theta_ridge = merged["theta_l2_norm_ridge"].to_numpy(dtype=float)
    ratio = np.full(theta_ols.shape, np.nan, dtype=float)
    nonzero = np.isfinite(theta_ols) & np.isfinite(theta_ridge) & (theta_ols != 0)
    ratio[nonzero] = theta_ridge[nonzero] / theta_ols[nonzero]
    merged["theta_l2_norm_ratio"] = ratio
    return merged


def build_comparison_state_summary(
    ols_metrics: pd.DataFrame,
    ridge_metrics: pd.DataFrame,
    *,
    split_lookup: dict[int, str] | None = None,
) -> pd.DataFrame:
    """把OLS/Ridge逐模型配对结果整理为每状态一行。"""

    merged = _merge_comparison_rows(ols_metrics, ridge_metrics)
    state_rows: list[dict[str, Any]] = []
    for state_id, group in merged.groupby("state_id", sort=True):
        row: dict[str, Any] = {
            "state_id": int(state_id),
            **get_state_info(int(state_id)),
        }
        if split_lookup is not None:
            row["split"] = split_lookup[int(state_id)]
        for model_id in MODEL_IDS:
            model_group = group[group["model_id"] == model_id]
            if model_group.shape[0] != 1:
                raise ValueError(f"state_id={state_id}缺少或重复正式模型")
            model_row = model_group.iloc[0]
            model_slug = _slug(model_id)
            for metric in METRIC_FIELDS:
                row[f"{model_slug}_{metric}_ols_db"] = float(model_row[f"{metric}_ols"])
                row[f"{model_slug}_{metric}_ridge_db"] = float(model_row[f"{metric}_ridge"])
                row[f"{model_slug}_{metric}_gain_db"] = float(model_row[f"{metric}_gain_db"])
                row[f"{model_slug}_{metric}_degradation_db"] = float(
                    model_row[f"{metric}_degradation_db"]
                )
            row[f"{model_slug}_theta_norm_ols"] = float(model_row["theta_l2_norm_ols"])
            row[f"{model_slug}_theta_norm_ridge"] = float(model_row["theta_l2_norm_ridge"])
            row[f"{model_slug}_theta_norm_ratio"] = float(model_row["theta_l2_norm_ratio"])
        row["X_B_gen_degradation_db"] = row["X_A_B_generalization_nmse_db_degradation_db"]
        row["Y_A_B_gen_degradation_db"] = row["Y_A_1_B_generalization_nmse_db_degradation_db"]
        row["Y_C_train_degradation_db"] = row["Y_C_1_train_nmse_db_degradation_db"]
        row["Y_C_B_gen_gain_db"] = row["Y_C_1_B_generalization_nmse_db_gain_db"]
        row["Y_C_vs_X_gain_db"] = row["Y_C_1_commonB_vs_X_cnmse_db_gain_db"]
        row["Y_C_vs_realB_gain_db"] = row["Y_C_1_commonB_vs_realB_cnmse_db_gain_db"]
        row["Y_C_vs_X_ols_db"] = row["Y_C_1_commonB_vs_X_cnmse_db_ols_db"]
        row["Y_C_vs_X_ridge_db"] = row["Y_C_1_commonB_vs_X_cnmse_db_ridge_db"]
        row["Y_C_vs_realB_ols_db"] = row["Y_C_1_commonB_vs_realB_cnmse_db_ols_db"]
        row["Y_C_vs_realB_ridge_db"] = row["Y_C_1_commonB_vs_realB_cnmse_db_ridge_db"]
        state_rows.append(row)
    return pd.DataFrame(state_rows)


def build_comparison_aggregate_summary(
    ols_metrics: pd.DataFrame,
    ridge_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """生成每模型一行的OLS、Ridge和paired统计。"""

    merged = _merge_comparison_rows(ols_metrics, ridge_metrics)
    rows: list[dict[str, Any]] = []
    for model_id in MODEL_IDS:
        group = merged[merged["model_id"] == model_id]
        row: dict[str, Any] = {"model_id": model_id, "state_count": int(group.shape[0])}
        for metric in METRIC_FIELDS:
            for source, suffix in (("ols", "ols"), ("ridge", "ridge")):
                for statistic, value in _stats(
                    group[f"{metric}_{source}"].to_numpy(dtype=float)
                ).items():
                    row[f"{metric}_{suffix}_{statistic}"] = value
            for kind in ("gain_db", "degradation_db"):
                for statistic, value in _stats(
                    group[f"{metric}_{kind}"].to_numpy(dtype=float)
                ).items():
                    row[f"{metric}_{kind}_{statistic}"] = value
        for statistic, value in _stats(group["theta_l2_norm_ols"].to_numpy(dtype=float)).items():
            row[f"theta_l2_norm_ols_{statistic}"] = value
        for statistic, value in _stats(group["theta_l2_norm_ridge"].to_numpy(dtype=float)).items():
            row[f"theta_l2_norm_ridge_{statistic}"] = value
        for statistic, value in _stats(group["theta_l2_norm_ratio"].to_numpy(dtype=float)).items():
            row[f"theta_l2_norm_ratio_{statistic}"] = value
        rows.append(row)
    return pd.DataFrame(rows)


def _worst_state(state_summary: pd.DataFrame, column: str, *, maximize: bool) -> dict[str, Any]:
    values = state_summary[column].to_numpy(dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return {"state_id": None, "value_dB": None}
    indices = np.flatnonzero(finite)
    index = indices[np.argmax(values[finite]) if maximize else np.argmin(values[finite])]
    row = state_summary.iloc[int(index)]
    return {
        "state_id": int(row["state_id"]),
        "value_dB": float(row[column]),
        "condition": {
            key: _json_scalar(row[key])
            for key in ("funMng", "funAng", "secMng", "secAng", "Vm", "Pin")
        },
    }


def build_validation_result(
    state_summary: pd.DataFrame,
    *,
    selected_lambda: float,
    validation_state_count: int,
) -> dict[str, Any]:
    """执行预声明 Validation PASS/FAIL 规则，并保留改善状态数和最差状态。"""

    focus_columns = {
        "Y_C_B_gen": "Y_C_B_gen_gain_db",
        "Y_C_vs_X": "Y_C_vs_X_gain_db",
        "Y_C_vs_real": "Y_C_vs_realB_gain_db",
    }
    protection_columns = {
        "X_B_gen": "X_B_gen_degradation_db",
        "Y_A_B_gen": "Y_A_B_gen_degradation_db",
        "Y_C_train": "Y_C_train_degradation_db",
    }
    medians = {
        name: float(np.median(_finite_values(state_summary[column])))
        for name, column in {**focus_columns, **protection_columns}.items()
    }
    criteria = {
        "Y_C_B_gen_gain_positive": medians["Y_C_B_gen"] > 0,
        "Y_C_vs_X_gain_positive": medians["Y_C_vs_X"] > 0,
        "Y_C_vs_real_gain_positive": medians["Y_C_vs_real"] > 0,
        "X_B_gen_degradation_at_most_0_5_dB": medians["X_B_gen"] <= 0.5,
        "Y_A_B_gen_degradation_at_most_0_5_dB": medians["Y_A_B_gen"] <= 0.5,
        "Y_C_train_degradation_at_most_1_dB": medians["Y_C_train"] <= 1.0,
    }
    improved_counts = {
        name: int(np.count_nonzero(state_summary[column].to_numpy(dtype=float) > 0))
        for name, column in focus_columns.items()
    }
    degraded_counts = {
        name: int(np.count_nonzero(state_summary[column].to_numpy(dtype=float) < 0))
        for name, column in focus_columns.items()
    }
    result = {
        "selected_lambda": float(selected_lambda),
        "validation_state_count": int(validation_state_count),
        "validation_model_pair_count": int(validation_state_count * len(MODEL_IDS)),
        "pass": bool(all(criteria.values())),
        "criteria": criteria,
        "observed_medians_dB": medians,
        "improved_state_counts": improved_counts,
        "degraded_state_counts": degraded_counts,
        "improved_state_fractions": {
            name: float(count / validation_state_count)
            for name, count in improved_counts.items()
        },
        "max_improvement": {
            name: _worst_state(state_summary, column, maximize=True)
            for name, column in focus_columns.items()
        },
        "max_degradation": {
            name: _worst_state(state_summary, column, maximize=False)
            for name, column in focus_columns.items()
        },
        "worst_protection_degradation": {
            name: _worst_state(state_summary, column, maximize=True)
            for name, column in protection_columns.items()
        },
    }
    return result


def run_ols_vs_ridge_comparison(
    state_ids: Iterable[int],
    selected_lambda: Real,
    *,
    split_lookup: dict[int, str] | None = None,
    progress_interval: int = 25,
) -> FullComparisonResult:
    """对一组状态只计算 OLS 与已冻结 Ridge 两套模型。"""

    normalized_states = _normalize_state_ids(state_ids)
    normalized_lambda = validate_ridge_lambda(selected_lambda)
    if normalized_lambda <= 0:
        raise ValueError("独立验证和最终比较的Ridge lambda必须大于0")
    ols_rows: list[dict[str, Any]] = []
    ridge_rows: list[dict[str, Any]] = []
    ols_theta: list[np.ndarray] = []
    ridge_theta: list[np.ndarray] = []
    for position, state_id in enumerate(normalized_states, start=1):
        prepared = prepare_state_ridge_data(state_id)
        ols = evaluate_prepared_state(prepared, 0.0)
        ridge = evaluate_prepared_state(prepared, normalized_lambda)
        ols_rows.extend(ols.rows)
        ridge_rows.extend(ridge.rows)
        ols_theta.append(ols.theta)
        ridge_theta.append(ridge.theta)
        if progress_interval > 0 and (
            position % progress_interval == 0 or position == len(normalized_states)
        ):
            print(f"OLS/Ridge comparison: processed {position} / {len(normalized_states)} states")
    ols_metrics = pd.DataFrame(ols_rows)
    ridge_metrics = pd.DataFrame(ridge_rows)
    state_summary = build_comparison_state_summary(
        ols_metrics,
        ridge_metrics,
        split_lookup=split_lookup,
    )
    aggregate_summary = build_comparison_aggregate_summary(ols_metrics, ridge_metrics)
    return FullComparisonResult(
        state_ids=normalized_states,
        ols_metrics=ols_metrics,
        ridge_metrics=ridge_metrics,
        state_summary=state_summary,
        aggregate_summary=aggregate_summary,
        theta_ols=np.stack(ols_theta, axis=0),
        theta_ridge=np.stack(ridge_theta, axis=0),
    )


def augment_final_model_metrics(comparison: FullComparisonResult) -> pd.DataFrame:
    """将 Ridge 行扩展为最终1275行表，并附OLS参考与paired gain。"""

    merged = _merge_comparison_rows(comparison.ols_metrics, comparison.ridge_metrics)
    ridge = comparison.ridge_metrics.copy()
    for metric in METRIC_FIELDS:
        ridge[f"{metric}_ols_db"] = merged[f"{metric}_ols"].to_numpy(dtype=float)
        ridge[f"{metric}_gain_db"] = merged[f"{metric}_gain_db"].to_numpy(dtype=float)
        ridge[f"{metric}_degradation_db"] = merged[
            f"{metric}_degradation_db"
        ].to_numpy(dtype=float)
    ridge["theta_l2_norm_ols"] = merged["theta_l2_norm_ols"].to_numpy(dtype=float)
    ridge["theta_l2_norm_ratio"] = merged["theta_l2_norm_ratio"].to_numpy(dtype=float)
    return ridge


def threshold_counts(
    state_summary: pd.DataFrame,
    *,
    source: str,
    metric_column: str,
) -> dict[str, int]:
    values = state_summary[f"{metric_column}_{source}_db"].to_numpy(dtype=float)
    return {
        "le_minus30_dB": int(np.count_nonzero(values <= -30.0)),
        "le_minus35_dB": int(np.count_nonzero(values <= -35.0)),
        "le_minus40_dB": int(np.count_nonzero(values <= -40.0)),
    }


def build_selection_payload(
    *,
    baseline_check: dict[str, Any],
    split_frame: pd.DataFrame,
    coarse_lambdas: Sequence[float],
    coarse_selection: dict[str, Any],
    fine_lambdas: Sequence[float] | None,
    fine_selection: dict[str, Any] | None,
) -> dict[str, Any]:
    """生成在Validation前冻结的lambda_selection.json内容。"""

    selected = None
    selected_summary: dict[str, Any] | None = None
    if fine_selection is not None and fine_selection.get("status") == "SELECTED":
        selected = float(fine_selection["selected_lambda"])
        selected_summary = fine_selection.get("selected_summary")
    elif fine_selection is None and coarse_selection.get("status") == "SELECTED":
        selected = float(coarse_selection["selected_lambda"])
        selected_summary = coarse_selection.get("selected_summary")
    payload: dict[str, Any] = {
        "selected_lambda": selected,
        "selection_stage": (
            "fine" if fine_selection is not None else "coarse"
        ),
        "status": (
            fine_selection.get("status")
            if fine_selection is not None
            else coarse_selection.get("status")
        ),
        "coarse_winner": coarse_selection.get("selected_lambda"),
        "coarse_grid": [float(value) for value in coarse_lambdas],
        "fine_grid": None if fine_lambdas is None else [float(value) for value in fine_lambdas],
        "development_state_count": int(np.count_nonzero(split_frame["split"] == "development")),
        "validation_state_count": int(np.count_nonzero(split_frame["split"] == "validation")),
        "random_seed": RANDOM_SEED,
        "selection_constraints": dict(SELECTION_CONSTRAINTS),
        "coarse_selection": coarse_selection,
        "fine_selection": fine_selection,
        "selected_summary": selected_summary,
        "lambda_zero_regression": baseline_check,
        "timestamp_utc": pd.Timestamp.now(tz="UTC").isoformat(),
    }
    if selected_summary is not None:
        payload["median_YC_B_gain_dB"] = selected_summary.get("median_YC_B_gen_gain_dB")
        payload["median_YC_vsX_gain_dB"] = selected_summary.get("median_YC_vs_X_gain_dB")
        payload["median_YC_vsReal_gain_dB"] = selected_summary.get("median_YC_vs_real_gain_dB")
        payload["median_X_B_degradation_dB"] = selected_summary.get(
            "median_X_B_gen_degradation_dB"
        )
        payload["median_YA_B_degradation_dB"] = selected_summary.get(
            "median_YA_B_gen_degradation_dB"
        )
        payload["median_YC_train_degradation_dB"] = selected_summary.get(
            "median_YC_train_degradation_dB"
        )
        payload["improved_state_counts"] = {
            key: selected_summary.get(f"{key}_improved_count")
            for key in _FOCUS
        }
    return _jsonify(payload)


def _jsonify(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonify(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonify(item) for item in value]
    if isinstance(value, np.ndarray):
        return [_jsonify(item) for item in value.tolist()]
    return _json_scalar(value)


__all__ = [
    "COARSE_LAMBDAS",
    "DEVELOPMENT_COUNT",
    "FullComparisonResult",
    "METRIC_FIELDS",
    "MODEL_IDS",
    "RANDOM_SEED",
    "STATE_COUNT",
    "VALIDATION_COUNT",
    "add_paired_gain_columns",
    "augment_final_model_metrics",
    "build_comparison_aggregate_summary",
    "build_comparison_state_summary",
    "build_scan_summary",
    "build_selection_payload",
    "build_validation_result",
    "compare_lambda_zero_to_baseline",
    "create_state_split",
    "load_or_create_state_split",
    "make_fine_lambda_grid",
    "prepare_state_ridge_data",
    "run_full_lambda_zero_regression",
    "run_lambda_scan",
    "run_ols_vs_ridge_comparison",
    "select_lambda",
    "threshold_counts",
    "validate_development_state_ids",
]
