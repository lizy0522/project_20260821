"""
功能说明：基于State0 canonical A/B/C拟合11个正向MP模型并计算Train、B泛化和公共B-Probe指标。
输入：data_manager状态字典；只消费signal_segmentation提供的OFF和5次ILC canonical数据。
输出：11个模型评价、11条公共B响应、11个Train NMSE、11个B泛化NMSE和21个CNMSE。
用途：严格区分各链路真实B泛化与统一xin_B下的X/Y行为等效性，不使用STALE或多状态数据。
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Any

import numpy as np
from data_manager import load_by_id
from signal_segmentation import (
    CanonicalSegment,
    build_ilc_segments,
    build_off_segments,
    build_partition_from_xin,
    get_common_probe,
)

from .config import MAX_DELAY, NUM_COEFFICIENTS
from .evaluation import calculate_cnmse, calculate_nmse
from .mp_model import MemoryPolynomialModel
from .prediction import predict_behavior


@dataclass(frozen=True)
class ModelEvaluation:
    """一个模型系数向量及其训练、真实B泛化和公共B响应诊断。"""

    model_id: str
    behavior_class: str
    train_segment: str
    ilc_iteration: int | None
    theta: np.ndarray
    rank: int
    condition_number: float
    n_train_samples: int
    coefficient_count: int
    train_nmse_db: float
    b_generalization_nmse_db: float
    b_generalization_source: str
    common_b_response: np.ndarray
    common_b_vs_x_cnmse_db: float
    common_b_vs_real_b_cnmse_db: float


@dataclass(frozen=True)
class State0XYEquivalenceResult:
    """State0 11模型和统一B-Probe验证的完整内存结果。"""

    state_id: int
    evaluations: tuple[ModelEvaluation, ...]
    common_probe: np.ndarray
    common_probe_valid: np.ndarray
    real_b_valid: np.ndarray
    common_responses: dict[str, np.ndarray]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "common_responses",
            MappingProxyType(dict(self.common_responses)),
        )


def _fit_and_evaluate(
    *,
    model_id: str,
    behavior_class: str,
    train_segment_name: str,
    ilc_iteration: int | None,
    train_segment: CanonicalSegment,
    b_segment: CanonicalSegment,
    b_generalization_source: str,
    common_probe: np.ndarray,
) -> ModelEvaluation:
    model = MemoryPolynomialModel().fit(train_segment.input, train_segment.output)
    if model.theta is None or model.fit_diagnostics is None:
        raise RuntimeError(f"{model_id}未生成模型系数或LS诊断")
    train_prediction = model.predict(train_segment.input)
    train_nmse_db = calculate_nmse(
        train_segment.output[MAX_DELAY:],
        train_prediction,
    )
    b_prediction = model.predict(b_segment.input)
    b_generalization_nmse_db = calculate_nmse(
        b_segment.output[MAX_DELAY:],
        b_prediction,
    )
    common_b_response = model.predict(common_probe)
    return ModelEvaluation(
        model_id=model_id,
        behavior_class=behavior_class,
        train_segment=train_segment_name,
        ilc_iteration=ilc_iteration,
        theta=model.theta,
        rank=model.fit_diagnostics.rank,
        condition_number=model.fit_diagnostics.condition_number,
        n_train_samples=model.fit_diagnostics.sample_count,
        coefficient_count=model.fit_diagnostics.coefficient_count,
        train_nmse_db=train_nmse_db,
        b_generalization_nmse_db=b_generalization_nmse_db,
        b_generalization_source=b_generalization_source,
        common_b_response=common_b_response,
        common_b_vs_x_cnmse_db=float("nan"),
        common_b_vs_real_b_cnmse_db=float("nan"),
    )


def _validate_result(result: State0XYEquivalenceResult) -> None:
    evaluations = result.evaluations
    if len(evaluations) != 11:
        raise RuntimeError(f"模型总数必须为11，实际为{len(evaluations)}")
    if sum(item.model_id == "X-A" for item in evaluations) != 1:
        raise RuntimeError("X-A模型数量必须为1")
    if sum(item.model_id.startswith("Y-A-") for item in evaluations) != 5:
        raise RuntimeError("Y-A模型数量必须为5")
    if sum(item.model_id.startswith("Y-C-") for item in evaluations) != 5:
        raise RuntimeError("Y-C模型数量必须为5")

    for item in evaluations:
        if item.theta.shape != (NUM_COEFFICIENTS,):
            raise RuntimeError(f"{item.model_id} theta shape错误：{item.theta.shape}")
        if not np.all(np.isfinite(item.theta)):
            raise RuntimeError(f"{item.model_id} theta包含NaN或Inf")
        if item.rank != NUM_COEFFICIENTS:
            raise RuntimeError(f"{item.model_id} rank={item.rank}，预期为10")
        if item.common_b_response.shape != (4913,):
            raise RuntimeError(f"{item.model_id}公共B响应长度错误：{item.common_b_response.shape}")
        if not np.isfinite(item.train_nmse_db):
            raise RuntimeError(f"{item.model_id} Train NMSE非有限")
        if not np.isfinite(item.b_generalization_nmse_db):
            raise RuntimeError(f"{item.model_id} B泛化NMSE非有限")

    if result.common_probe.shape != (4915,):
        raise RuntimeError("公共xin_B ownership长度必须为4915")
    if result.common_probe_valid.shape != (4913,):
        raise RuntimeError("公共xin_B valid长度必须为4913")
    if result.real_b_valid.shape != (4913,):
        raise RuntimeError("真实B参考长度必须为4913")

    y_rows = [item for item in evaluations if item.behavior_class == "Y"]
    if len(y_rows) != 10:
        raise RuntimeError("Y模型总数必须为10")
    y_vs_x_count = sum(not np.isnan(item.common_b_vs_x_cnmse_db) for item in y_rows)
    y_vs_real_count = sum(not np.isnan(item.common_b_vs_real_b_cnmse_db) for item in y_rows)
    x_vs_real_count = sum(
        item.behavior_class == "X" and not np.isnan(item.common_b_vs_real_b_cnmse_db)
        for item in evaluations
    )
    total_cnmse_count = sum(
        not np.isnan(value)
        for item in evaluations
        for value in (
            item.common_b_vs_x_cnmse_db,
            item.common_b_vs_real_b_cnmse_db,
        )
    )
    if (y_vs_x_count, y_vs_real_count, x_vs_real_count, total_cnmse_count) != (
        10,
        10,
        1,
        21,
    ):
        raise RuntimeError(
            "CNMSE数量错误："
            f"YvsX={y_vs_x_count}, YvsReal={y_vs_real_count}, "
            f"XvsReal={x_vs_real_count}, total={total_cnmse_count}"
        )


def analyze_state0_xy_equivalence(
    state_data: dict[str, Any] | None = None,
) -> State0XYEquivalenceResult:
    """执行State0的11模型训练、真实B泛化和统一xin_B等效分析。"""
    state_id = 0
    if state_data is None:
        state_data = load_by_id(state_id)
    partition = build_partition_from_xin(np.asarray(state_data["xin"]))
    off = build_off_segments(state_data, partition)
    ilc = build_ilc_segments(state_data, partition)
    if len(ilc) != 5:
        raise RuntimeError(f"State0 ILC迭代数必须为5，实际为{len(ilc)}")

    common_probe = get_common_probe(state_data, partition)
    common_probe_valid = common_probe[MAX_DELAY:]
    real_b_valid = off["B"].output[MAX_DELAY:]

    provisional: list[ModelEvaluation] = []
    provisional.append(
        _fit_and_evaluate(
            model_id="X-A",
            behavior_class="X",
            train_segment_name="A",
            ilc_iteration=None,
            train_segment=off["A"],
            b_segment=off["B"],
            b_generalization_source="OFF.B",
            common_probe=common_probe,
        )
    )
    for iteration_index, canonical in enumerate(ilc):
        iteration = iteration_index + 1
        provisional.append(
            _fit_and_evaluate(
                model_id=f"Y-A-{iteration}",
                behavior_class="Y",
                train_segment_name="A",
                ilc_iteration=iteration,
                train_segment=canonical["A"],
                b_segment=canonical["B"],
                b_generalization_source=f"ILC[{iteration}].B",
                common_probe=common_probe,
            )
        )
    for iteration_index, canonical in enumerate(ilc):
        iteration = iteration_index + 1
        provisional.append(
            _fit_and_evaluate(
                model_id=f"Y-C-{iteration}",
                behavior_class="Y",
                train_segment_name="C",
                ilc_iteration=iteration,
                train_segment=canonical["C"],
                b_segment=canonical["B"],
                b_generalization_source=f"ILC[{iteration}].B",
                common_probe=common_probe,
            )
        )

    x_response = provisional[0].common_b_response
    evaluations: list[ModelEvaluation] = []
    for item in provisional:
        if item.behavior_class == "X":
            evaluations.append(
                replace(
                    item,
                    common_b_vs_x_cnmse_db=float("nan"),
                    common_b_vs_real_b_cnmse_db=calculate_cnmse(
                        real_b_valid,
                        item.common_b_response,
                    ),
                )
            )
        else:
            evaluations.append(
                replace(
                    item,
                    common_b_vs_x_cnmse_db=calculate_cnmse(
                        x_response,
                        item.common_b_response,
                    ),
                    common_b_vs_real_b_cnmse_db=calculate_cnmse(
                        real_b_valid,
                        item.common_b_response,
                    ),
                )
            )

    common_responses = {item.model_id: item.common_b_response for item in evaluations}
    result = State0XYEquivalenceResult(
        state_id=state_id,
        evaluations=tuple(evaluations),
        common_probe=common_probe,
        common_probe_valid=common_probe_valid,
        real_b_valid=real_b_valid,
        common_responses=common_responses,
    )
    _validate_result(result)
    return result


def predict_with_theta(x: np.ndarray, theta: np.ndarray) -> np.ndarray:
    """测试和验收辅助：使用冻结结构从指定系数生成模型响应。"""
    return predict_behavior(x, theta)


__all__ = [
    "ModelEvaluation",
    "State0XYEquivalenceResult",
    "analyze_state0_xy_equivalence",
    "predict_with_theta",
]
