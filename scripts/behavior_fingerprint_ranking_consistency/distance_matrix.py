"""用统一 Query→Candidate 方向计算四个 CNMSE 距离矩阵。"""

from __future__ import annotations

import numpy as np
from _core.metrics import cnmse

from .fingerprint_builder import FINGERPRINT_LENGTH, FingerprintResult

STATE_COUNT = 425


def _validate_fingerprint_matrix(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.shape[1] != FINGERPRINT_LENGTH:
        raise ValueError(f"{name}必须是(N,4913)二维数组")
    if array.dtype != np.complex128 or not np.iscomplexobj(array):
        raise ValueError(f"{name}必须是complex128数组")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name}包含NaN或Inf")
    return array


def compute_cnmse_distance_matrix(
    query_fingerprints: np.ndarray,
    candidate_fingerprints: np.ndarray,
) -> np.ndarray:
    """返回 ``CNMSE(query_i, candidate_j)``，行是Query、列是Candidate。

    使用平方范数和复内积的等价形式进行批量计算，数值定义与
    ``_core.metrics.cnmse`` 一致；零误差保留为 ``-inf``。
    """

    query = _validate_fingerprint_matrix(query_fingerprints, "query_fingerprints")
    candidate = _validate_fingerprint_matrix(candidate_fingerprints, "candidate_fingerprints")
    query_energy = np.sum(np.abs(query) ** 2, axis=1, dtype=np.float64)
    candidate_energy = np.sum(np.abs(candidate) ** 2, axis=1, dtype=np.float64)
    if np.any(query_energy <= 0) or np.any(candidate_energy < 0):
        raise ValueError("CNMSE指纹能量必须为正")

    with np.errstate(over="ignore", invalid="ignore"):
        squared_error = (
            query_energy[:, None]
            + candidate_energy[None, :]
            - 2.0 * np.real(query @ candidate.conj().T)
        )
    scale = query_energy[:, None] + candidate_energy[None, :] + 1.0
    tolerance = 32.0 * np.finfo(np.float64).eps * scale
    squared_error[np.abs(squared_error) <= tolerance] = 0.0
    squared_error = np.maximum(squared_error, 0.0)

    if query.shape == candidate.shape and np.array_equal(query, candidate):
        np.fill_diagonal(squared_error, 0.0)

    ratio = np.divide(
        squared_error,
        query_energy[:, None],
        out=np.zeros_like(squared_error),
        where=query_energy[:, None] > 0,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = 10.0 * np.log10(ratio)
    distance[squared_error == 0.0] = -np.inf
    if np.isnan(distance).any() or np.isposinf(distance).any():
        raise RuntimeError("CNMSE距离矩阵出现NaN或+Inf")
    return distance.astype(np.float64, copy=False)


def build_distance_matrices(fingerprints: FingerprintResult) -> dict[str, np.ndarray]:
    """构造 ``D_RR``、``D_CX``、``D_CY`` 和 ``D_CR``。"""

    real_B = fingerprints.real_B_fingerprints
    query = fingerprints.Y_C_query_fingerprints
    matrices = {
        "D_RR": compute_cnmse_distance_matrix(real_B, real_B),
        "D_CX": compute_cnmse_distance_matrix(
            query,
            fingerprints.X_A_fingerprints,
        ),
        "D_CY": compute_cnmse_distance_matrix(
            query,
            fingerprints.Y_A_fingerprints,
        ),
        "D_CR": compute_cnmse_distance_matrix(query, real_B),
    }
    for name, matrix in matrices.items():
        if matrix.shape != (STATE_COUNT, STATE_COUNT):
            raise RuntimeError(f"{name} shape错误：{matrix.shape}")
        if np.isnan(matrix).any() or np.isposinf(matrix).any():
            raise RuntimeError(f"{name}包含NaN或+Inf")
    return matrices


def validate_distance_direction(
    query: np.ndarray,
    candidates: np.ndarray,
) -> bool:
    """小型辅助验收：完全相同候选应比不同候选更相似。"""

    distance = compute_cnmse_distance_matrix(query, candidates)
    reference = cnmse(query[0], candidates[0])
    return bool(distance[0, 0] == reference)


__all__ = [
    "build_distance_matrices",
    "compute_cnmse_distance_matrix",
    "validate_distance_direction",
]
