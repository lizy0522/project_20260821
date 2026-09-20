"""
从冻结 Scenario 2 Ridge theta 和统一公共 B probe 生成四类行为指纹。

X-A、Y-A-1、Y-C-1 均使用同一个公共 B 输入构造模型响应；Real-B 使用每个状态
已有 canonical OFF.B.output[2:]。本模块不重新训练、不重新对齐、不重新 adjust。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
from behavior_modeling.shared.basis import build_mp_basis
from behavior_modeling.shared.config import MAX_DELAY, MP_CONFIG, NUM_COEFFICIENTS
from data_management.shared import load_by_id
from signal_segmentation.shared import (
    build_off_segments,
    build_partition_from_xin,
    get_common_probe,
)

EXPECTED_MODEL_IDS = ("X-A", "Y-A-1", "Y-C-1")
STATE_COUNT = 425
FINGERPRINT_LENGTH = 4913
COMMON_B_ATOL = 1e-12
COMMON_B_RTOL = 1e-12


@dataclass(frozen=True)
class FrozenRidgeModels:
    """从 final NPZ 加载的冻结三模型 theta。"""

    state_ids: np.ndarray
    model_ids: tuple[str, ...]
    model_index: dict[str, int]
    theta: np.ndarray
    ridge_lambda: float
    source_path: Path


@dataclass(frozen=True)
class FingerprintResult:
    """四类 Scenario 2 指纹及公共 B 设计矩阵。"""

    state_ids: np.ndarray
    common_B_input: np.ndarray
    phi_B: np.ndarray
    X_A_fingerprints: np.ndarray
    Y_A_fingerprints: np.ndarray
    Y_C_query_fingerprints: np.ndarray
    real_B_fingerprints: np.ndarray
    common_B_consistent: bool
    common_B_max_abs_difference: float


def _decode_model_id(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def _validate_state_ids(state_ids: np.ndarray) -> np.ndarray:
    normalized = np.asarray(state_ids)
    if normalized.ndim != 1 or normalized.size != STATE_COUNT:
        raise ValueError("冻结模型state_ids必须是长度425的一维数组")
    if not np.issubdtype(normalized.dtype, np.integer):
        raise ValueError("冻结模型state_ids必须是整数数组")
    normalized = normalized.astype(np.int64, copy=False)
    if not np.array_equal(normalized, np.arange(STATE_COUNT, dtype=np.int64)):
        raise ValueError("冻结模型state_ids必须严格为0...424")
    return normalized


def load_frozen_ridge_models(
    theta_path: Path,
    *,
    validation_path: Path | None = None,
    expected_lambda: Real = 1e-8,
) -> FrozenRidgeModels:
    """加载并验证已通过独立Validation的正式 Ridge theta。"""

    theta_path = Path(theta_path)
    if not theta_path.is_file():
        raise FileNotFoundError(f"缺少冻结Ridge theta：{theta_path}")
    with np.load(theta_path, allow_pickle=False) as data:
        required = {"state_ids", "model_ids", "theta"}
        if not required.issubset(data.files):
            raise ValueError(f"冻结theta缺少字段：{sorted(required - set(data.files))}")
        state_ids = _validate_state_ids(np.asarray(data["state_ids"]))
        model_ids = tuple(_decode_model_id(value) for value in np.asarray(data["model_ids"]))
        theta = np.asarray(data["theta"])

    if model_ids != EXPECTED_MODEL_IDS:
        raise ValueError(f"冻结模型ID必须为{EXPECTED_MODEL_IDS}，实际为{model_ids}")
    if theta.shape != (STATE_COUNT, len(EXPECTED_MODEL_IDS), NUM_COEFFICIENTS):
        raise ValueError(f"冻结theta shape错误：{theta.shape}")
    if theta.dtype != np.complex128:
        raise ValueError(f"冻结theta dtype必须为complex128，实际为{theta.dtype}")
    if not np.iscomplexobj(theta) or not np.all(np.isfinite(theta)):
        raise ValueError("冻结theta必须是有限复数数组")

    normalized_lambda = float(expected_lambda)
    if not np.isfinite(normalized_lambda) or normalized_lambda <= 0:
        raise ValueError("expected_lambda必须是有限正数")
    if validation_path is None:
        validation_path = theta_path.with_name("scenario2_ridge_validation.json")
    if validation_path.is_file():
        with validation_path.open("r", encoding="utf-8-sig") as handle:
            validation = json.load(handle)
        saved_lambda = float(validation.get("selected_lambda", np.nan))
        if not np.isclose(saved_lambda, normalized_lambda, rtol=0, atol=0):
            raise ValueError(
                f"冻结模型lambda与validation不一致：{saved_lambda} != {normalized_lambda}"
            )
        if validation.get("used_ridge") is not True:
            raise ValueError("冻结模型validation未标记used_ridge=true")
        if validation.get("used_ilc_iteration_1_only") is not True:
            raise ValueError("冻结模型validation未标记ILC1-only")

    return FrozenRidgeModels(
        state_ids=state_ids,
        model_ids=model_ids,
        model_index={model_id: index for index, model_id in enumerate(model_ids)},
        theta=theta,
        ridge_lambda=normalized_lambda,
        source_path=theta_path,
    )


def _validate_fingerprint_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (STATE_COUNT, FINGERPRINT_LENGTH):
        raise ValueError(f"{name} shape错误：{array.shape}")
    if array.dtype != np.complex128 or not np.iscomplexobj(array):
        raise ValueError(f"{name}必须是complex128")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name}包含NaN或Inf")
    return array


def build_scenario2_fingerprints(
    frozen_models: FrozenRidgeModels,
    *,
    progress_interval: int = 50,
) -> FingerprintResult:
    """遍历425状态，验证公共B输入并生成四类4913点指纹。"""

    common_B_input: np.ndarray | None = None
    real_B_rows: list[np.ndarray] = []
    maximum_difference = 0.0
    for position, state_id_value in enumerate(frozen_models.state_ids, start=1):
        state_id = int(state_id_value)
        state_data = load_by_id(state_id)
        partition = build_partition_from_xin(np.asarray(state_data["xin"]))
        off = build_off_segments(state_data, partition)
        state_probe = np.asarray(get_common_probe(state_data, partition), dtype=np.complex128)
        if state_probe.shape != (FINGERPRINT_LENGTH + MAX_DELAY,):
            raise RuntimeError(f"state_id={state_id}公共B输入长度错误：{state_probe.shape}")
        if not np.all(np.isfinite(state_probe)):
            raise RuntimeError(f"state_id={state_id}公共B输入包含非有限值")
        if common_B_input is None:
            common_B_input = state_probe.copy()
        else:
            difference = float(np.max(np.abs(state_probe - common_B_input)))
            maximum_difference = max(maximum_difference, difference)
            if not np.allclose(
                state_probe,
                common_B_input,
                rtol=COMMON_B_RTOL,
                atol=COMMON_B_ATOL,
            ):
                raise RuntimeError(
                    f"公共B probe不一致，state_id={state_id}，"
                    f"max_abs_difference={difference}"
                )
        real_B = np.asarray(off["B"].output[MAX_DELAY:], dtype=np.complex128)
        if real_B.shape != (FINGERPRINT_LENGTH,) or not np.all(np.isfinite(real_B)):
            raise RuntimeError(f"state_id={state_id} Real-B指纹无效：{real_B.shape}")
        real_B_rows.append(real_B)
        if progress_interval > 0 and (
            position % progress_interval == 0 or position == len(frozen_models.state_ids)
        ):
            print(f"fingerprints: processed {position} / {len(frozen_models.state_ids)} states")

    if common_B_input is None:
        raise RuntimeError("未生成公共B输入")
    phi_B = build_mp_basis(
        common_B_input,
        MP_CONFIG["orders"],
        MP_CONFIG["memory_depth"],
    )
    if phi_B.shape != (FINGERPRINT_LENGTH, NUM_COEFFICIENTS):
        raise RuntimeError(f"公共B basis shape错误：{phi_B.shape}")

    X_A = (phi_B @ frozen_models.theta[:, frozen_models.model_index["X-A"], :].T).T
    Y_A = (phi_B @ frozen_models.theta[:, frozen_models.model_index["Y-A-1"], :].T).T
    Y_C = (phi_B @ frozen_models.theta[:, frozen_models.model_index["Y-C-1"], :].T).T
    real_B_matrix = np.stack(real_B_rows, axis=0)
    X_A = _validate_fingerprint_array(X_A.astype(np.complex128, copy=False), "X-A指纹")
    Y_A = _validate_fingerprint_array(Y_A.astype(np.complex128, copy=False), "Y-A指纹")
    Y_C = _validate_fingerprint_array(Y_C.astype(np.complex128, copy=False), "Y-C指纹")
    real_B_matrix = _validate_fingerprint_array(real_B_matrix, "Real-B指纹")

    return FingerprintResult(
        state_ids=frozen_models.state_ids.copy(),
        common_B_input=common_B_input,
        phi_B=phi_B,
        X_A_fingerprints=X_A,
        Y_A_fingerprints=Y_A,
        Y_C_query_fingerprints=Y_C,
        real_B_fingerprints=real_B_matrix,
        common_B_consistent=True,
        common_B_max_abs_difference=maximum_difference,
    )


__all__ = [
    "COMMON_B_ATOL",
    "COMMON_B_RTOL",
    "EXPECTED_MODEL_IDS",
    "FINGERPRINT_LENGTH",
    "FrozenRidgeModels",
    "FingerprintResult",
    "build_scenario2_fingerprints",
    "load_frozen_ridge_models",
]
