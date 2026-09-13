"""
Scenario 2 retrieval-oriented unified MP model scan.

This module implements the post-hoc/oracle experiment in which every candidate
Memory Polynomial model is judged by the final C2 -> Aend LUT retrieval result
on the frozen Real-B ground truth.  It deliberately keeps the formal unequal
ABC partition and the canonical full-record preprocessing pipeline unchanged.

The implementation is independent of the earlier Development-only model scan:
model NMSE values are retained for interpretation, but never prune candidates or
enter the ranking.  For every valid candidate, all 425 C2 queries search all 425
Y-Aend LUT entries, producing a complete 425 x 425 CNMSE matrix.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from behavior_fingerprint_ranking_consistency.distance_matrix import (
    compute_cnmse_distance_matrix,
)
from behavior_fingerprint_ranking_consistency.ranking import distance_to_ranks
from behavior_model.basis import build_mp_basis
from behavior_model.coefficient import _solve_basis_ols
from behavior_model.evaluation import calculate_nmse
from behavior_model.ridge import fit_coefficients_ridge
from data_manager import load_by_id, load_variable_by_id
from signal_segmentation import build_partition_from_xin, get_ilc_pair, preprocess_full_pair

STATE_COUNT = 425
WAVEFORM_LENGTH = 24576
COMMON_B_INPUT_LENGTH = 4915
FORMAL_A_LENGTH = 12288
FORMAL_B_LENGTH = 4915
FORMAL_C_LENGTH = 7373
BASELINE_MAX_DELAY = 2
MAX_DELAY_MAX = 5
BASELINE_VALID_LENGTHS = {"A": 12286, "B": 4913, "C": 7371}
DPD_SHAREABLE_THRESHOLD_DB = -40.0
C2_STAGE = 2
NUMERIC_TOLERANCE_DB = 1e-12
LAMBDA_GRID = (0.0, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4)

BASELINE_ORDERS = (1, 2, 3, 5, 7, 9)
BASELINE_MEMORY = {1: 3, 2: 2, 3: 2, 5: 1, 7: 1, 9: 1}
BASELINE_LAMBDA = 1e-8
BASELINE_FAILURE_IDS = (187, 189, 195, 196, 199, 206, 323, 327, 330, 335, 340, 344, 346, 354)

ORDER_PROFILES: dict[str, tuple[int, ...]] = {
    "P7": (1, 2, 3, 5, 7),
    "P9": (1, 2, 3, 5, 7, 9),
    "P11": (1, 2, 3, 5, 7, 9, 11),
    "P13": (1, 2, 3, 5, 7, 9, 11, 13),
    "P15": (1, 2, 3, 5, 7, 9, 11, 13, 15),
}

# The templates intentionally define memory for every P15 term.  A candidate
# order profile selects only its active orders from the template.
MEMORY_PROFILES: dict[str, dict[int, int]] = {
    "M0": {1: 3, 2: 2, 3: 2, 5: 1, 7: 1, 9: 1, 11: 1, 13: 1, 15: 1},
    "M1": {1: 4, 2: 3, 3: 3, 5: 2, 7: 2, 9: 2, 11: 1, 13: 1, 15: 1},
    "M2": {1: 5, 2: 4, 3: 4, 5: 3, 7: 3, 9: 2, 11: 2, 13: 1, 15: 1},
    "M3": {1: 6, 2: 5, 3: 5, 5: 4, 7: 3, 9: 3, 11: 2, 13: 2, 15: 1},
}
FULL_ORDERS = (1, 2, 3, 5, 7, 9, 11, 13, 15)


@dataclass(frozen=True)
class ModelCandidate:
    """One deterministic row in the 5 x 4 x 10 candidate grid."""

    candidate_id: int
    order_profile: str
    orders: tuple[int, ...]
    memory_profile: str
    memory_definition: dict[int, int]
    ridge_lambda: float
    max_delay: int
    n_complex_coefficients: int
    is_baseline: bool
    structure_id: str
    basis_columns: tuple[int, ...]


@dataclass(frozen=True)
class PreparedCanonicalState:
    """Canonical Aend and C2 pairs for one state, before model-specific bases."""

    state_id: int
    ilc_column_count: int
    a_end: Any
    c2: Any


@dataclass(frozen=True)
class StructureBases:
    """Full P15 basis blocks for one memory template and one state."""

    a_train: np.ndarray
    a_y: np.ndarray
    a_b: np.ndarray
    a_b_y: np.ndarray
    c_train: np.ndarray
    c_y: np.ndarray
    c_b: np.ndarray
    c_b_y: np.ndarray
    max_delay: int


@dataclass(frozen=True)
class StructureFit:
    """All ten lambda fits for one order profile at one state."""

    metrics: np.ndarray  # (10, 4): Aend train/B, C2 train/B
    theta_a: np.ndarray  # (10, n_coeff)
    theta_c: np.ndarray  # (10, n_coeff)
    theta_norm_a: np.ndarray
    theta_norm_c: np.ndarray


def _json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _candidate_memory(orders: Iterable[int], profile: Mapping[int, int]) -> dict[int, int]:
    return {int(order): int(profile[int(order)]) for order in orders}


def _basis_columns(
    orders: tuple[int, ...], memory: Mapping[int, int], full_memory: Mapping[int, int]
) -> tuple[int, ...]:
    full_terms = tuple(
        (order, delay) for order in FULL_ORDERS for delay in range(int(full_memory[order]))
    )
    term_to_column = {term: index for index, term in enumerate(full_terms)}
    selected_terms = tuple(
        (order, delay) for order in orders for delay in range(int(memory[order]))
    )
    return tuple(term_to_column[term] for term in selected_terms)


def _make_candidate_objects() -> tuple[ModelCandidate, ...]:
    candidates: list[ModelCandidate] = []
    candidate_id = 1
    for order_profile, orders in ORDER_PROFILES.items():
        for memory_profile, memory_template in MEMORY_PROFILES.items():
            memory = _candidate_memory(orders, memory_template)
            basis_columns = _basis_columns(orders, memory, memory_template)
            max_delay = max(memory_template.values()) - 1
            for ridge_lambda in LAMBDA_GRID:
                candidates.append(
                    ModelCandidate(
                        candidate_id=candidate_id,
                        order_profile=order_profile,
                        orders=orders,
                        memory_profile=memory_profile,
                        memory_definition=memory,
                        ridge_lambda=float(ridge_lambda),
                        max_delay=max_delay,
                        n_complex_coefficients=len(basis_columns),
                        is_baseline=(
                            orders == BASELINE_ORDERS
                            and memory == BASELINE_MEMORY
                            and float(ridge_lambda) == BASELINE_LAMBDA
                        ),
                        structure_id=f"{order_profile}_{memory_profile}",
                        basis_columns=basis_columns,
                    )
                )
                candidate_id += 1
    return tuple(candidates)


CANDIDATES = _make_candidate_objects()
CANDIDATE_BY_ID = {candidate.candidate_id: candidate for candidate in CANDIDATES}
BASELINE_CANDIDATE = next(candidate for candidate in CANDIDATES if candidate.is_baseline)
MAX_COEFFICIENTS = max(candidate.n_complex_coefficients for candidate in CANDIDATES)


def build_model_candidate_grid() -> pd.DataFrame:
    """Return the complete deterministic 200-row candidate table."""

    rows = [
        {
            "candidate_id": candidate.candidate_id,
            "order_profile": candidate.order_profile,
            "orders": _json_compact(list(candidate.orders)),
            "memory_profile": candidate.memory_profile,
            "memory_definition": _json_compact(candidate.memory_definition),
            "lambda": candidate.ridge_lambda,
            "max_delay": candidate.max_delay,
            "n_complex_coefficients": candidate.n_complex_coefficients,
            "is_baseline": candidate.is_baseline,
            "structure_id": candidate.structure_id,
        }
        for candidate in CANDIDATES
    ]
    frame = pd.DataFrame(rows)
    validate_candidate_grid(frame)
    return frame


def validate_candidate_grid(frame: pd.DataFrame) -> None:
    """Validate the frozen 5 x 4 x 10 grid and unique formal Baseline."""

    required = {
        "candidate_id",
        "order_profile",
        "orders",
        "memory_profile",
        "memory_definition",
        "lambda",
        "max_delay",
        "n_complex_coefficients",
        "is_baseline",
        "structure_id",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"candidate grid缺少字段：{missing}")
    if frame.shape[0] != 200:
        raise ValueError(f"candidate_count必须为200，实际为{frame.shape[0]}")
    ids = frame["candidate_id"].to_numpy(dtype=np.int64)
    if not np.array_equal(ids, np.arange(1, 201, dtype=np.int64)):
        raise ValueError("candidate_id必须严格为1...200且唯一")
    if set(frame["order_profile"]) != set(ORDER_PROFILES):
        raise ValueError("order profile集合错误")
    if set(frame["memory_profile"]) != set(MEMORY_PROFILES):
        raise ValueError("memory profile集合错误")
    if set(frame["lambda"].astype(float)) != set(LAMBDA_GRID):
        raise ValueError("lambda网格错误")
    if set(frame["max_delay"].astype(int)) != {2, 3, 4, 5}:
        raise ValueError("max_delay必须覆盖2/3/4/5")
    baseline = frame.loc[frame["is_baseline"].astype(bool)]
    if baseline.shape[0] != 1:
        raise ValueError("Baseline必须在candidate grid中唯一存在")
    row = baseline.iloc[0]
    if not (
        row["order_profile"] == "P9"
        and row["memory_profile"] == "M0"
        and float(row["lambda"]) == BASELINE_LAMBDA
    ):
        raise ValueError("Baseline anchor必须是P9/M0/lambda=1e-8")


def _as_vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or not np.iscomplexobj(array):
        raise ValueError(f"{name}必须是一维复数数组")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name}包含NaN或Inf")
    return array.astype(np.complex128, copy=False)


def prepare_canonical_state(state_id: int) -> PreparedCanonicalState:
    """Read and canonicalise one state using full-record synchronization."""

    if not isinstance(state_id, (int, np.integer)) or not 0 <= int(state_id) < STATE_COUNT:
        raise ValueError(f"state_id必须位于0...424：{state_id!r}")
    state_id = int(state_id)
    state_data = load_by_id(state_id)
    input_history = np.asarray(state_data["xin_pd_ori_ilc"])
    output_history = np.asarray(state_data["yout_withdpd_ori_ilc"])
    if input_history.ndim != 2 or output_history.ndim != 2:
        raise ValueError(f"state_id={state_id} ILC历史必须为二维")
    if input_history.shape != output_history.shape:
        raise ValueError(f"state_id={state_id} ILC输入输出shape不一致")
    if input_history.shape[0] != WAVEFORM_LENGTH:
        raise ValueError(f"state_id={state_id}波形长度必须为{WAVEFORM_LENGTH}")
    if input_history.shape[1] < C2_STAGE or input_history.shape[1] > 5:
        raise RuntimeError(f"state_id={state_id}缺少真实C2或ILC列数超过5")
    if not np.iscomplexobj(input_history) or not np.iscomplexobj(output_history):
        raise ValueError(f"state_id={state_id} ILC历史必须为复数")
    if not np.all(np.isfinite(input_history)) or not np.all(np.isfinite(output_history)):
        raise ValueError(f"state_id={state_id} ILC历史包含非有限值")
    partition = build_partition_from_xin(_as_vector(state_data["xin"], "xin"))
    if (partition.n_a, partition.n_b, partition.n_c) != (
        FORMAL_A_LENGTH,
        FORMAL_B_LENGTH,
        FORMAL_C_LENGTH,
    ):
        raise RuntimeError("正式ABC分段不是12288/4915/7373")
    ilc_count = int(input_history.shape[1])
    a_end_pair = get_ilc_pair(state_data, ilc_count - 1)
    a_end = preprocess_full_pair(
        a_end_pair.input_full,
        a_end_pair.output_raw_full,
        partition,
        pair_type="ILC",
        iteration_index=ilc_count - 1,
        input_peak_normalization_factor=a_end_pair.input_peak_normalization_factor,
    )
    c2_pair = get_ilc_pair(state_data, C2_STAGE - 1)
    c2 = (
        a_end
        if ilc_count == C2_STAGE
        else preprocess_full_pair(
            c2_pair.input_full,
            c2_pair.output_raw_full,
            partition,
            pair_type="ILC",
            iteration_index=C2_STAGE - 1,
            input_peak_normalization_factor=c2_pair.input_peak_normalization_factor,
        )
    )
    return PreparedCanonicalState(
        state_id=state_id,
        ilc_column_count=ilc_count,
        a_end=a_end,
        c2=c2,
    )


def _basis_and_target(
    canonical: Any,
    segment_name: str,
    full_memory: Mapping[int, int],
    max_delay: int,
) -> tuple[np.ndarray, np.ndarray]:
    segment = canonical[segment_name]
    phi = build_mp_basis(segment.input, FULL_ORDERS, full_memory)
    y = np.asarray(segment.output[max_delay:], dtype=np.complex128)
    expected = segment.ownership_length - max_delay
    if phi.shape != (expected, sum(int(full_memory[order]) for order in FULL_ORDERS)):
        raise RuntimeError(f"{segment_name} basis shape错误：{phi.shape}")
    if y.shape != (expected,) or not np.all(np.isfinite(phi)) or not np.all(np.isfinite(y)):
        raise RuntimeError(f"{segment_name} basis/output包含非法值")
    return phi, y


def build_structure_bases(prepared: PreparedCanonicalState, memory_profile: str) -> StructureBases:
    """Build one full P15 basis per canonical segment for a memory template."""

    if memory_profile not in MEMORY_PROFILES:
        raise KeyError(f"未知memory profile：{memory_profile}")
    full_memory = MEMORY_PROFILES[memory_profile]
    max_delay = max(full_memory.values()) - 1
    a_train, a_y = _basis_and_target(prepared.a_end, "A", full_memory, max_delay)
    a_b, a_b_y = _basis_and_target(prepared.a_end, "B", full_memory, max_delay)
    c_train, c_y = _basis_and_target(prepared.c2, "C", full_memory, max_delay)
    c_b, c_b_y = _basis_and_target(prepared.c2, "B", full_memory, max_delay)
    return StructureBases(
        a_train=a_train,
        a_y=a_y,
        a_b=a_b,
        a_b_y=a_b_y,
        c_train=c_train,
        c_y=c_y,
        c_b=c_b,
        c_b_y=c_b_y,
        max_delay=max_delay,
    )


def _fit_role_lambdas(
    phi_train: np.ndarray,
    y_train: np.ndarray,
    phi_b: np.ndarray,
    y_b: np.ndarray,
    lambdas: Iterable[Real] = LAMBDA_GRID,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit all lambdas through the existing OLS/augmented-LS Ridge path."""

    theta_ols, _, rank, singular = _solve_basis_ols(phi_train, y_train)
    if rank != phi_train.shape[1]:
        raise RuntimeError(f"设计矩阵秩不足：{rank}/{phi_train.shape[1]}")
    ols_solution = (theta_ols, int(rank), np.asarray(singular))
    # Materialise once because callers may pass a generator.
    lambda_values = tuple(float(value) for value in lambdas)
    metrics = np.empty((len(lambda_values), 2), dtype=np.float64)
    theta_values = np.empty((len(lambda_values), phi_train.shape[1]), dtype=np.complex128)
    norms = np.empty(len(lambda_values), dtype=np.float64)
    for index, ridge_lambda in enumerate(lambda_values):
        theta, diagnostics = fit_coefficients_ridge(
            phi_train,
            y_train,
            ridge_lambda,
            ols_solution=ols_solution,
        )
        theta_values[index] = theta
        norms[index] = diagnostics.theta_l2_norm
        metrics[index] = (
            calculate_nmse(y_train, phi_train @ theta),
            calculate_nmse(y_b, phi_b @ theta),
        )
    if not np.all(np.isfinite(metrics)) or not np.all(np.isfinite(theta_values)):
        raise RuntimeError("模型拟合产生非有限值")
    return metrics, theta_values, norms


def fit_structure_for_state(
    prepared: PreparedCanonicalState, memory_profile: str
) -> dict[str, StructureFit]:
    """Fit all five order profiles and ten lambdas for one memory profile/state."""

    bases = build_structure_bases(prepared, memory_profile)
    result: dict[str, StructureFit] = {}
    full_memory = MEMORY_PROFILES[memory_profile]
    for order_profile, orders in ORDER_PROFILES.items():
        memory = _candidate_memory(orders, full_memory)
        columns = _basis_columns(orders, memory, full_memory)
        a_metrics, theta_a, norms_a = _fit_role_lambdas(
            bases.a_train[:, columns],
            bases.a_y,
            bases.a_b[:, columns],
            bases.a_b_y,
            LAMBDA_GRID,
        )
        c_metrics, theta_c, norms_c = _fit_role_lambdas(
            bases.c_train[:, columns],
            bases.c_y,
            bases.c_b[:, columns],
            bases.c_b_y,
            LAMBDA_GRID,
        )
        result[order_profile] = StructureFit(
            metrics=np.column_stack(
                (a_metrics[:, 0], a_metrics[:, 1], c_metrics[:, 0], c_metrics[:, 1])
            ),
            theta_a=theta_a,
            theta_c=theta_c,
            theta_norm_a=norms_a,
            theta_norm_c=norms_c,
        )
    return result


def _validate_fingerprint_matrix(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.dtype != np.complex128 or not np.iscomplexobj(array):
        raise ValueError(f"{name}必须是二维complex128数组")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name}包含NaN或Inf")
    return array


def compute_generic_cnmse_distance_matrix(
    query_fingerprints: np.ndarray,
    candidate_fingerprints: np.ndarray,
) -> np.ndarray:
    """Compute Query-row/Candidate-column CNMSE for any common length."""

    query = _validate_fingerprint_matrix(query_fingerprints, "query_fingerprints")
    candidate = _validate_fingerprint_matrix(candidate_fingerprints, "candidate_fingerprints")
    if query.shape[1] != candidate.shape[1]:
        raise ValueError("Query与Candidate指纹长度必须一致")
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


def compute_candidate_distance_matrix(
    query_fingerprints: np.ndarray, candidate_fingerprints: np.ndarray
) -> np.ndarray:
    """Use the frozen shared implementation at d=2, generic path otherwise."""

    query = np.asarray(query_fingerprints)
    candidate = np.asarray(candidate_fingerprints)
    if query.ndim == 2 and candidate.ndim == 2 and query.shape[1] == 4913:
        return compute_cnmse_distance_matrix(query, candidate)
    return compute_generic_cnmse_distance_matrix(query, candidate)


def _stable_top1(
    distance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if distance.shape != (STATE_COUNT, STATE_COUNT):
        raise ValueError(f"distance必须是(425,425)，实际为{distance.shape}")
    if np.isnan(distance).any() or np.isposinf(distance).any():
        raise ValueError("distance包含NaN或+Inf")
    state_ids = np.arange(STATE_COUNT, dtype=np.int64)
    selected = np.empty(STATE_COUNT, dtype=np.int64)
    selected_distance = np.empty(STATE_COUNT, dtype=np.float64)
    true_rank = np.empty(STATE_COUNT, dtype=np.int64)
    tie_count = np.empty(STATE_COUNT, dtype=np.int64)
    order_matrix = np.empty((STATE_COUNT, STATE_COUNT), dtype=np.int64)
    for state_id in range(STATE_COUNT):
        order = np.lexsort((state_ids, distance[state_id]))
        order_matrix[state_id] = order
        selected[state_id] = int(order[0])
        selected_distance[state_id] = float(distance[state_id, order[0]])
        true_rank[state_id] = int(np.flatnonzero(order == state_id)[0] + 1)
        tie_count[state_id] = int(
            np.count_nonzero(distance[state_id] == distance[state_id, order[0]])
        )
    return selected, selected_distance, true_rank, tie_count, order_matrix


def dpd_shareable_mask(values: np.ndarray) -> np.ndarray:
    """Apply the strict Real-B CNMSE < -40 dB rule."""

    return np.asarray(values, dtype=float) < DPD_SHAREABLE_THRESHOLD_DB


def build_candidate_retrieval(
    query_fingerprints: np.ndarray,
    lut_fingerprints: np.ndarray,
    real_b_distance: np.ndarray,
    *,
    candidate: ModelCandidate,
    baseline_retrieval: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Perform full 425-query/full-425-LUT Top-1 retrieval for one candidate."""

    distance = compute_candidate_distance_matrix(query_fingerprints, lut_fingerprints)
    ranking = distance_to_ranks(distance)
    selected, selected_distance, true_rank, tie_count, order_matrix = _stable_top1(distance)
    state_ids = np.arange(STATE_COUNT, dtype=np.int64)
    real_b = np.asarray(real_b_distance, dtype=np.float64)[state_ids, selected]
    exact = selected == state_ids
    shareable = dpd_shareable_mask(real_b)
    sorted_distance = np.take_along_axis(distance, order_matrix, axis=1)
    margin12 = sorted_distance[:, 1] - sorted_distance[:, 0]
    margin13 = sorted_distance[:, 2] - sorted_distance[:, 0]
    rows: list[dict[str, Any]] = []
    baseline_mask = None
    if baseline_retrieval is not None:
        baseline_lookup = baseline_retrieval.set_index("state_id_R")
        baseline_mask = baseline_lookup.loc[state_ids, "baseline_shareable"].to_numpy(dtype=bool)
    for state_id in state_ids:
        transition = "not_compared"
        if baseline_mask is not None:
            if baseline_mask[state_id] and shareable[state_id]:
                transition = "success_to_success"
            elif baseline_mask[state_id] and not shareable[state_id]:
                transition = "success_to_failure"
            elif not baseline_mask[state_id] and shareable[state_id]:
                transition = "failure_to_success"
            else:
                transition = "failure_to_failure"
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "state_id_R": int(state_id),
                "state_id_Q": int(selected[state_id]),
                "exact_hit": bool(exact[state_id]),
                "true_state_rank": int(true_rank[state_id]),
                "top1_true_state_hit": bool(true_rank[state_id] <= 1),
                "top3_true_state_hit": bool(true_rank[state_id] <= 3),
                "top5_true_state_hit": bool(true_rank[state_id] <= 5),
                "top10_true_state_hit": bool(true_rank[state_id] <= 10),
                "query_to_selected_LUT_CNMSE_dB": float(selected_distance[state_id]),
                "fingerprint_margin_12_dB": float(margin12[state_id]),
                "fingerprint_margin_13_dB": float(margin13[state_id]),
                "minimum_tie_count": int(tie_count[state_id]),
                "state_id_delta": int(selected[state_id] - state_id),
                "abs_state_id_delta": int(abs(selected[state_id] - state_id)),
                "retrieved_real_B_CNMSE_dB": float(real_b[state_id]),
                "dpd_shareable": bool(shareable[state_id]),
                "failure": bool(not shareable[state_id]),
                "transition_vs_baseline": transition,
            }
        )
    results = pd.DataFrame(rows)
    finite_nonexact = real_b[(~exact) & np.isfinite(real_b)]
    if finite_nonexact.size == 0:
        median = q95 = worst = np.nan
    else:
        median = float(np.median(finite_nonexact))
        q95 = float(np.quantile(finite_nonexact, 0.95))
        worst = float(np.max(finite_nonexact))
    summary = {
        "candidate_id": candidate.candidate_id,
        "candidate_valid": True,
        "invalid_reason": "",
        "query_count": STATE_COUNT,
        "lut_entry_count": STATE_COUNT,
        "distance_matrix_shape": list(distance.shape),
        "distance_count": int(distance.size),
        "self_match": True,
        "minimum_tie_rows": int(np.count_nonzero(tie_count > 1)),
        "exact_hit_count": int(exact.sum()),
        "exact_hit_rate": float(exact.mean()),
        "nonexact_count": int((~exact).sum()),
        "nonexact_finite_count": int(finite_nonexact.size),
        "shareable_count": int(shareable.sum()),
        "shareable_rate": float(shareable.mean()),
        "failure_count": int((~shareable).sum()),
        "nonexact_real_B_median_dB": median,
        "retrieved_real_B_q95_dB": q95,
        "worst_finite_retrieved_real_B_CNMSE_dB": worst,
        "true_state_rank_median": float(np.median(true_rank)),
        "true_state_rank_max": int(np.max(true_rank)),
        "top1_true_state_hit_count": int(np.count_nonzero(true_rank <= 1)),
        "top3_true_state_hit_count": int(np.count_nonzero(true_rank <= 3)),
        "top5_true_state_hit_count": int(np.count_nonzero(true_rank <= 5)),
        "top10_true_state_hit_count": int(np.count_nonzero(true_rank <= 10)),
    }
    return {
        "distance": distance,
        "ranking": ranking,
        "ranking_order": order_matrix,
        "selected_q": selected,
        "selected_distance": selected_distance,
        "true_rank": true_rank,
        "tie_count": tie_count,
        "real_b_values": real_b,
        "results": results,
        "summary": summary,
    }


def load_real_b_reference(
    real_b_path: Path,
    reference_distance_path: Path,
    reference_ranking_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load the formal unequal-ABC Real-B matrix and compare it to D_RR/R_RR."""

    with np.load(real_b_path, allow_pickle=False) as data:
        required = {"state_ids", "D_B", "R_B"}
        if not required.issubset(data.files):
            raise ValueError(f"Real-B矩阵缺少字段：{sorted(required - set(data.files))}")
        state_ids = np.asarray(data["state_ids"], dtype=np.int64)
        distance = np.asarray(data["D_B"])
        ranking = np.asarray(data["R_B"])
    if not np.array_equal(state_ids, np.arange(STATE_COUNT, dtype=np.int64)):
        raise ValueError("Real-B state_ids错误")
    if distance.shape != (STATE_COUNT, STATE_COUNT) or distance.dtype != np.float64:
        raise ValueError("Real-B distance必须为float64(425,425)")
    if ranking.shape != distance.shape or ranking.dtype != np.float64:
        raise ValueError("Real-B ranking必须为float64(425,425)")
    if np.isnan(distance).any() or np.isposinf(distance).any() or not np.all(np.isfinite(ranking)):
        raise ValueError("Real-B矩阵包含非法值")
    with np.load(reference_distance_path, allow_pickle=False) as data:
        reference_distance = np.asarray(data["D_RR"])
    with np.load(reference_ranking_path, allow_pickle=False) as data:
        reference_ranking = np.asarray(data["R_RR"])
    if not np.array_equal(np.isneginf(distance), np.isneginf(reference_distance)):
        raise RuntimeError("Real-B矩阵非有限模式与冻结D_RR不一致")
    finite = np.isfinite(distance) & np.isfinite(reference_distance)
    if not np.array_equal(distance[finite], reference_distance[finite]):
        raise RuntimeError("Real-B矩阵与冻结D_RR不一致")
    if not np.array_equal(ranking, reference_ranking):
        raise RuntimeError("Real-B排名与冻结R_RR不一致")
    if not np.all(np.isneginf(np.diag(distance))):
        raise RuntimeError("Real-B对角线必须全部为-Inf")
    return distance, ranking


def load_baseline_metrics_reference(
    all_ilc_metrics_path: Path, availability_path: Path
) -> pd.DataFrame:
    """Extract formal P9/M0/1e-8 Aend/C2 metrics for all 425 states."""

    metrics = pd.read_csv(all_ilc_metrics_path)
    availability = pd.read_csv(availability_path).sort_values("state_id").reset_index(drop=True)
    if availability.shape[0] != STATE_COUNT:
        raise ValueError("ILC availability必须包含425状态")
    rows: list[dict[str, Any]] = []
    for state_id, count in enumerate(availability["ilc_column_count"].to_numpy(dtype=int)):
        a = metrics.loc[
            (metrics["state_id"] == state_id)
            & (metrics["model_role"] == "Y-A")
            & (metrics["actual_ilc_n"] == count)
            & np.isclose(metrics["ridge_lambda"].astype(float), BASELINE_LAMBDA)
        ]
        c = metrics.loc[
            (metrics["state_id"] == state_id)
            & (metrics["model_role"] == "Y-C")
            & (metrics["actual_ilc_n"] == C2_STAGE)
            & np.isclose(metrics["ridge_lambda"].astype(float), BASELINE_LAMBDA)
        ]
        if a.shape[0] != 1 or c.shape[0] != 1:
            raise ValueError(f"state={state_id} Baseline指标行数错误：{a.shape[0]}/{c.shape[0]}")
        rows.append(
            {
                "state_id": state_id,
                "ilc_A_end": count,
                "Y_Aend_train_NMSE_dB": float(a.iloc[0]["train_nmse_db"]),
                "Y_Aend_B_NMSE_dB": float(a.iloc[0]["B_generalization_nmse_db"]),
                "Y_C2_train_NMSE_dB": float(c.iloc[0]["train_nmse_db"]),
                "Y_C2_B_NMSE_dB": float(c.iloc[0]["B_generalization_nmse_db"]),
            }
        )
    result = pd.DataFrame(rows)
    if not np.all(np.isfinite(result.iloc[:, 2:].to_numpy(dtype=float))):
        raise ValueError("Baseline指标包含非有限值")
    return result


def load_raw_scalar_metrics(state_ids: Iterable[int]) -> pd.DataFrame:
    """Read the three frozen raw scalar metrics used by the Best Excel table."""

    ids = [int(value) for value in state_ids]
    if ids != list(range(STATE_COUNT)):
        raise ValueError("raw scalar state ids必须严格为0...424")
    rows: list[dict[str, float | int]] = []
    for state_id in ids:
        row: dict[str, float | int] = {"state_id": state_id}
        for variable in ("nmse_withoutdpd", "acpr_low_withoutdpd", "acpr_upper_withoutdpd"):
            value = np.asarray(load_variable_by_id(state_id, variable), dtype=float).reshape(-1)
            if value.size != 1 or not np.isfinite(value[0]):
                raise ValueError(f"state={state_id} {variable}不是finite scalar")
            row[variable] = float(value[0])
        row["acpr_withoutdpd_avg_dBc"] = (
            float(row["acpr_low_withoutdpd"]) + float(row["acpr_upper_withoutdpd"])
        ) / 2.0
        rows.append(row)
    return pd.DataFrame(rows)


def assign_pa_quartiles(values: np.ndarray) -> np.ndarray:
    """Assign Q1 (most negative) through Q4 (closest to zero) deterministically."""

    values = np.asarray(values, dtype=float)
    if values.shape != (STATE_COUNT,) or not np.all(np.isfinite(values)):
        raise ValueError("PA nonlinearity values必须为425个finite值")
    order = np.argsort(values, kind="mergesort")
    labels = np.empty(STATE_COUNT, dtype=object)
    names = np.asarray(["Q1", "Q2", "Q3", "Q4"], dtype=object)
    labels[order] = names[np.minimum((np.arange(STATE_COUNT) * 4) // STATE_COUNT, 3)]
    return labels


def safe_correlation(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Return finite correlation statistics or an explicit undefined flag."""

    from scipy.stats import pearsonr, spearmanr

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x_valid = x[mask]
    y_valid = y[mask]
    if x_valid.size < 3 or np.ptp(x_valid) == 0.0 or np.ptp(y_valid) == 0.0:
        return {
            "count": int(x_valid.size),
            "pearson_r": np.nan,
            "spearman_rho": np.nan,
            "defined": False,
        }
    return {
        "count": int(x_valid.size),
        "pearson_r": float(pearsonr(x_valid, y_valid).statistic),
        "spearman_rho": float(spearmanr(x_valid, y_valid).statistic),
        "defined": True,
    }


def candidate_from_id(candidate_id: int) -> ModelCandidate:
    """Return a candidate by its one-based grid id."""

    try:
        return CANDIDATE_BY_ID[int(candidate_id)]
    except KeyError as exc:
        raise ValueError(f"candidate_id不存在：{candidate_id}") from exc


__all__ = [
    "BASELINE_CANDIDATE",
    "BASELINE_FAILURE_IDS",
    "BASELINE_LAMBDA",
    "BASELINE_MEMORY",
    "BASELINE_ORDERS",
    "BASELINE_VALID_LENGTHS",
    "C2_STAGE",
    "CANDIDATE_BY_ID",
    "CANDIDATES",
    "COMMON_B_INPUT_LENGTH",
    "DPD_SHAREABLE_THRESHOLD_DB",
    "FORMAL_A_LENGTH",
    "FORMAL_B_LENGTH",
    "FORMAL_C_LENGTH",
    "FULL_ORDERS",
    "LAMBDA_GRID",
    "MAX_COEFFICIENTS",
    "MAX_DELAY_MAX",
    "MEMORY_PROFILES",
    "ModelCandidate",
    "ORDER_PROFILES",
    "PreparedCanonicalState",
    "StructureBases",
    "StructureFit",
    "STATE_COUNT",
    "WAVEFORM_LENGTH",
    "assign_pa_quartiles",
    "build_candidate_retrieval",
    "build_model_candidate_grid",
    "build_structure_bases",
    "candidate_from_id",
    "compute_candidate_distance_matrix",
    "compute_generic_cnmse_distance_matrix",
    "dpd_shareable_mask",
    "fit_structure_for_state",
    "load_baseline_metrics_reference",
    "load_raw_scalar_metrics",
    "load_real_b_reference",
    "prepare_canonical_state",
    "safe_correlation",
    "validate_candidate_grid",
]
