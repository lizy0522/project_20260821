"""
Scenario 2 unified Y-Aend/Y-C2 model-capacity scan and retrieval comparison.

This module implements the frozen experimental contract described in the task
brief.  It deliberately separates the Development-only model scan from the
post-freeze Validation/Full-425 retrieval evaluation:

* Y-Aend and Y-C2 always receive the same order profile, memory profile and
  Ridge lambda, while their coefficients are fitted independently.
* Development model selection uses only the four train/B-generalisation NMSE
  values.  Real-B distances, retrieval outputs and failure labels are never
  passed to the selector.
* After the selected candidate is frozen, the same candidate is fitted on all
  425 states and its common-B fingerprints are used for C2 -> Aend retrieval.

The raw MAT corpus is read through the existing read-only ``data_manager`` and
the existing canonical ``actual input -> full Rough -> full Fine -> ABC ->
segment-wise complex-gain adjustment`` pipeline.  No raw file is written.
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
from scipy.stats import pearsonr, spearmanr
from signal_segmentation import build_partition_from_xin, get_ilc_pair, preprocess_full_pair

STATE_COUNT = 425
FINGERPRINT_LENGTH = 4913
COMMON_B_INPUT_LENGTH = 4915
MAX_DELAY = 2
VALID_LENGTHS = {"A": 12286, "B": 4913, "C": 7371}
DEVELOPMENT_COUNT = 340
VALIDATION_COUNT = 85
SPLIT_SEED = 20260827
DPD_SHAREABLE_THRESHOLD_DB = -40.0
NUMERIC_GAIN_TOLERANCE_DB = 1e-12
C2_STAGE = 2

BASELINE_ORDERS = (1, 2, 3, 5, 7, 9)
BASELINE_MEMORY = {1: 3, 2: 2, 3: 2, 5: 1, 7: 1, 9: 1}
BASELINE_LAMBDA = 1e-8

ORDER_PROFILES: dict[str, tuple[int, ...]] = {
    "P7": (1, 2, 3, 5, 7),
    "P9": (1, 2, 3, 5, 7, 9),
    "P11": (1, 2, 3, 5, 7, 9, 11),
    "P13": (1, 2, 3, 5, 7, 9, 11, 13),
}
MEMORY_PROFILES: dict[str, dict[int, int]] = {
    "M0": {1: 3, 2: 2, 3: 2, 5: 1, 7: 1, 9: 1, 11: 1, 13: 1},
    "M1": {1: 3, 2: 3, 3: 3, 5: 2, 7: 2, 9: 2, 11: 1, 13: 1},
    "M2": {1: 3, 2: 3, 3: 3, 5: 3, 7: 3, 9: 3, 11: 2, 13: 2},
}
LAMBDA_GRID = (0.0, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4)

FULL_ORDERS = (1, 2, 3, 5, 7, 9, 11, 13)
FULL_MEMORY = MEMORY_PROFILES["M2"]


@dataclass(frozen=True)
class ModelCandidate:
    """One row in the frozen candidate grid."""

    candidate_id: int
    order_profile: str
    orders: tuple[int, ...]
    memory_profile: str
    memory_definition: dict[int, int]
    ridge_lambda: float
    n_complex_coefficients: int
    max_delay: int
    is_baseline: bool
    structure_id: str
    basis_columns: tuple[int, ...]


@dataclass(frozen=True)
class PreparedPair:
    """Canonical segments and a full P13/M2 basis for one ILC column."""

    actual_ilc_n: int
    input_peak_normalization_factor: float
    phi_full: Mapping[str, np.ndarray]
    y_valid: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class PreparedState:
    """The A-end and C2 canonical pairs needed by this study."""

    state_id: int
    ilc_column_count: int
    a_end: PreparedPair
    c2: PreparedPair


def _json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _candidate_memory(orders: Iterable[int], profile: Mapping[int, int]) -> dict[int, int]:
    return {int(order): int(profile[int(order)]) for order in orders}


def _basis_columns(orders: tuple[int, ...], memory: Mapping[int, int]) -> tuple[int, ...]:
    full_terms = tuple(
        (order, delay) for order in FULL_ORDERS for delay in range(FULL_MEMORY[order])
    )
    term_to_column = {term: index for index, term in enumerate(full_terms)}
    selected_terms = tuple((order, delay) for order in orders for delay in range(memory[order]))
    return tuple(term_to_column[term] for term in selected_terms)


def _make_candidate_objects() -> tuple[ModelCandidate, ...]:
    candidates: list[ModelCandidate] = []
    candidate_id = 1
    for order_profile, orders in ORDER_PROFILES.items():
        for memory_profile, memory_template in MEMORY_PROFILES.items():
            memory = _candidate_memory(orders, memory_template)
            columns = _basis_columns(orders, memory)
            max_delay = max(memory.values()) - 1
            structure_id = f"{order_profile}_{memory_profile}"
            for ridge_lambda in LAMBDA_GRID:
                candidates.append(
                    ModelCandidate(
                        candidate_id=candidate_id,
                        order_profile=order_profile,
                        orders=orders,
                        memory_profile=memory_profile,
                        memory_definition=memory,
                        ridge_lambda=float(ridge_lambda),
                        n_complex_coefficients=len(columns),
                        max_delay=max_delay,
                        is_baseline=(
                            orders == BASELINE_ORDERS
                            and memory == BASELINE_MEMORY
                            and ridge_lambda == BASELINE_LAMBDA
                        ),
                        structure_id=structure_id,
                        basis_columns=columns,
                    )
                )
                candidate_id += 1
    return tuple(candidates)


CANDIDATES = _make_candidate_objects()
CANDIDATE_BY_ID = {candidate.candidate_id: candidate for candidate in CANDIDATES}


def build_model_candidate_grid() -> pd.DataFrame:
    """Return the deterministic 4 x 3 x 10 candidate table."""

    rows = []
    for candidate in CANDIDATES:
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "order_profile": candidate.order_profile,
                "orders": _json_compact(list(candidate.orders)),
                "memory_profile": candidate.memory_profile,
                "memory_definition": _json_compact(candidate.memory_definition),
                "lambda": candidate.ridge_lambda,
                "n_complex_coefficients": candidate.n_complex_coefficients,
                "max_delay": candidate.max_delay,
                "is_baseline": candidate.is_baseline,
                "structure_id": candidate.structure_id,
            }
        )
    frame = pd.DataFrame(rows)
    validate_candidate_grid(frame)
    return frame


def validate_candidate_grid(frame: pd.DataFrame) -> None:
    """Validate count, uniqueness, common delay and the unique Baseline anchor."""

    required = {
        "candidate_id",
        "order_profile",
        "orders",
        "memory_profile",
        "memory_definition",
        "lambda",
        "n_complex_coefficients",
        "max_delay",
        "is_baseline",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"candidate grid缺少字段：{missing}")
    if frame.shape[0] != 120:
        raise ValueError(f"candidate_count必须为120，实际为{frame.shape[0]}")
    if frame["candidate_id"].nunique() != 120:
        raise ValueError("candidate_id必须唯一")
    if set(frame["order_profile"]) != set(ORDER_PROFILES):
        raise ValueError("order profile集合错误")
    if set(frame["memory_profile"]) != set(MEMORY_PROFILES):
        raise ValueError("memory profile集合错误")
    if set(frame["lambda"].astype(float)) != set(LAMBDA_GRID):
        raise ValueError("lambda网格错误")
    if not np.all(frame["max_delay"].to_numpy(dtype=int) == MAX_DELAY):
        raise ValueError("第一轮所有candidate的max_delay必须为2")
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


def load_fixed_split(split_path: Path) -> pd.DataFrame:
    """Read the pre-existing deterministic 340/85 Development/Validation split."""

    split_path = Path(split_path)
    if not split_path.is_file():
        raise FileNotFoundError(f"缺少固定split：{split_path}")
    frame = pd.read_csv(split_path)
    required = {"state_id", "split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"split_definition缺少字段：{missing}")
    frame = frame.sort_values("state_id").reset_index(drop=True)
    state_ids = frame["state_id"].to_numpy(dtype=np.int64)
    if state_ids.shape != (STATE_COUNT,) or not np.array_equal(
        state_ids, np.arange(STATE_COUNT, dtype=np.int64)
    ):
        raise ValueError("split的state_id必须严格为0...424")
    normalized = frame["split"].astype(str).str.strip().str.lower()
    normalized = normalized.replace({"development": "Development", "validation": "Validation"})
    if set(normalized) != {"Development", "Validation"}:
        raise ValueError(f"split标签错误：{sorted(set(normalized))}")
    frame = frame.copy()
    frame["split"] = normalized
    if int((frame["split"] == "Development").sum()) != DEVELOPMENT_COUNT:
        raise ValueError("Development必须恰好340个状态")
    if int((frame["split"] == "Validation").sum()) != VALIDATION_COUNT:
        raise ValueError("Validation必须恰好85个状态")
    return frame


def _as_vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or not np.iscomplexobj(array):
        raise ValueError(f"{name}必须是一维复数数组")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name}包含NaN或Inf")
    return array.astype(np.complex128, copy=False)


def _prepare_pair(state_data: dict[str, Any], partition: Any, actual_n: int) -> PreparedPair:
    pair = get_ilc_pair(state_data, actual_n - 1)
    canonical = preprocess_full_pair(
        pair.input_full,
        pair.output_raw_full,
        partition,
        pair_type="ILC",
        iteration_index=actual_n - 1,
        input_peak_normalization_factor=pair.input_peak_normalization_factor,
    )
    phi_full: dict[str, np.ndarray] = {}
    y_valid: dict[str, np.ndarray] = {}
    for segment_name in ("A", "B", "C"):
        segment = canonical[segment_name]
        phi = build_mp_basis(segment.input, FULL_ORDERS, FULL_MEMORY)
        y = np.asarray(segment.output[MAX_DELAY:], dtype=np.complex128)
        expected_length = VALID_LENGTHS[segment_name]
        if phi.shape[0] != expected_length or y.shape != (expected_length,):
            raise RuntimeError(
                f"state={actual_n} segment={segment_name}有效长度错误："
                f"{phi.shape[0]}/{y.shape} != {expected_length}"
            )
        if phi.shape[1] != 22 or not np.all(np.isfinite(phi)) or not np.all(np.isfinite(y)):
            raise RuntimeError(f"ILC{actual_n} {segment_name} basis/output无效")
        phi_full[segment_name] = phi
        y_valid[segment_name] = y
    return PreparedPair(
        actual_ilc_n=actual_n,
        input_peak_normalization_factor=float(pair.input_peak_normalization_factor),
        phi_full=phi_full,
        y_valid=y_valid,
    )


def prepare_state(state_id: int) -> PreparedState:
    """Read one state and prepare only its A-end and C2 canonical pairs."""

    if not isinstance(state_id, (int, np.integer)) or not 0 <= int(state_id) < STATE_COUNT:
        raise ValueError(f"state_id必须位于0...424：{state_id!r}")
    state_id = int(state_id)
    state_data = load_by_id(state_id)
    input_history = np.asarray(state_data["xin_pd_ori_ilc"])
    output_history = np.asarray(state_data["yout_withdpd_ori_ilc"])
    if input_history.ndim != 2 or output_history.ndim != 2:
        raise ValueError(f"state_id={state_id} ILC历史必须为二维")
    if input_history.shape != output_history.shape or input_history.shape[0] != 24576:
        raise ValueError(
            f"state_id={state_id} ILC历史shape错误：{input_history.shape}/{output_history.shape}"
        )
    if input_history.shape[1] < C2_STAGE or input_history.shape[1] > 5:
        raise ValueError(f"state_id={state_id}不存在可用C2或超过Nmax=5")
    partition = build_partition_from_xin(_as_vector(state_data["xin"], "xin"))
    ilc_count = int(input_history.shape[1])
    a_end = _prepare_pair(state_data, partition, ilc_count)
    c2 = a_end if ilc_count == C2_STAGE else _prepare_pair(state_data, partition, C2_STAGE)
    return PreparedState(state_id=state_id, ilc_column_count=ilc_count, a_end=a_end, c2=c2)


def _select_phi(
    pair: PreparedPair, segment: str, candidate: ModelCandidate
) -> tuple[np.ndarray, np.ndarray]:
    phi_full = np.asarray(pair.phi_full[segment])
    phi = phi_full[:, candidate.basis_columns]
    y = np.asarray(pair.y_valid[segment])
    if phi.shape[1] != candidate.n_complex_coefficients:
        raise RuntimeError("candidate基函数列数与Phi不一致")
    if phi.shape[0] != y.size or not np.all(np.isfinite(phi)):
        raise RuntimeError("candidate Phi/y无效")
    return phi, y


def _evaluate_phi_lambdas(
    phi_train: np.ndarray,
    y_train: np.ndarray,
    phi_b: np.ndarray,
    y_b: np.ndarray,
    candidate: ModelCandidate,
    lambdas: Iterable[Real],
    *,
    return_theta: bool = False,
) -> tuple[dict[float, tuple[float, float]], dict[float, np.ndarray]]:
    """Fit one role for all lambdas using the existing augmented-LS Ridge path."""

    theta_ols, _, rank_phi, singular_phi = _solve_basis_ols(phi_train, y_train)
    if rank_phi != candidate.n_complex_coefficients:
        raise RuntimeError(
            f"candidate={candidate.candidate_id} rank deficient："
            f"{rank_phi}/{candidate.n_complex_coefficients}"
        )
    ols_solution = (theta_ols, int(rank_phi), np.asarray(singular_phi))
    metrics: dict[float, tuple[float, float]] = {}
    thetas: dict[float, np.ndarray] = {}
    for ridge_lambda in lambdas:
        normalized = float(ridge_lambda)
        theta, _ = fit_coefficients_ridge(
            phi_train,
            y_train,
            normalized,
            ols_solution=ols_solution,
        )
        train_prediction = phi_train @ theta
        b_prediction = phi_b @ theta
        metrics[normalized] = (
            float(calculate_nmse(y_train, train_prediction)),
            float(calculate_nmse(y_b, b_prediction)),
        )
        if return_theta:
            thetas[normalized] = theta.astype(np.complex128, copy=True)
    return metrics, thetas


def _role_metrics_for_candidate(
    prepared: PreparedState, candidate: ModelCandidate, *, return_theta: bool = False
) -> tuple[
    dict[float, tuple[float, float]],
    dict[float, tuple[float, float]],
    dict[float, np.ndarray],
    dict[float, np.ndarray],
]:
    a_phi, a_y = _select_phi(prepared.a_end, "A", candidate)
    a_b_phi, a_b_y = _select_phi(prepared.a_end, "B", candidate)
    c_phi, c_y = _select_phi(prepared.c2, "C", candidate)
    c_b_phi, c_b_y = _select_phi(prepared.c2, "B", candidate)
    a_metrics, a_thetas = _evaluate_phi_lambdas(
        a_phi, a_y, a_b_phi, a_b_y, candidate, LAMBDA_GRID, return_theta=return_theta
    )
    c_metrics, c_thetas = _evaluate_phi_lambdas(
        c_phi, c_y, c_b_phi, c_b_y, candidate, LAMBDA_GRID, return_theta=return_theta
    )
    return a_metrics, c_metrics, a_thetas, c_thetas


def evaluate_single_candidate(
    state_ids: Iterable[int], candidate: ModelCandidate, *, progress_interval: int = 25
) -> pd.DataFrame:
    """Evaluate one candidate on a state set, used first for Baseline regression."""

    rows: list[dict[str, Any]] = []
    state_ids = [int(value) for value in state_ids]
    for position, state_id in enumerate(state_ids, start=1):
        prepared = prepare_state(state_id)
        a_metrics, c_metrics, _, _ = _role_metrics_for_candidate(prepared, candidate)
        values_a = a_metrics[candidate.ridge_lambda]
        values_c = c_metrics[candidate.ridge_lambda]
        rows.append(
            {
                "state_id": state_id,
                "candidate_id": candidate.candidate_id,
                "order_profile": candidate.order_profile,
                "memory_profile": candidate.memory_profile,
                "lambda": candidate.ridge_lambda,
                "Y_Aend_train_NMSE_dB": values_a[0],
                "Y_Aend_B_NMSE_dB": values_a[1],
                "Y_C2_train_NMSE_dB": values_c[0],
                "Y_C2_B_NMSE_dB": values_c[1],
            }
        )
        if progress_interval > 0 and (
            position % progress_interval == 0 or position == len(state_ids)
        ):
            print(f"baseline candidate: processed {position} / {len(state_ids)} states", flush=True)
    return pd.DataFrame(rows)


def scan_development(
    development_state_ids: Iterable[int],
    *,
    progress_interval: int = 10,
) -> pd.DataFrame:
    """Scan all 120 candidates using Development states only."""

    state_ids = [int(value) for value in development_state_ids]
    if len(state_ids) != DEVELOPMENT_COUNT or len(set(state_ids)) != DEVELOPMENT_COUNT:
        raise ValueError("Development state ids必须恰好包含340个唯一状态")
    candidate_rows: list[dict[str, Any]] = []
    structure_candidates: dict[str, list[ModelCandidate]] = {}
    for candidate in CANDIDATES:
        structure_candidates.setdefault(candidate.structure_id, []).append(candidate)

    for position, state_id in enumerate(state_ids, start=1):
        prepared = prepare_state(state_id)
        for structure_id, candidates in structure_candidates.items():
            representative = candidates[0]
            a_phi, a_y = _select_phi(prepared.a_end, "A", representative)
            a_b_phi, a_b_y = _select_phi(prepared.a_end, "B", representative)
            c_phi, c_y = _select_phi(prepared.c2, "C", representative)
            c_b_phi, c_b_y = _select_phi(prepared.c2, "B", representative)
            theta_ols_a, _, rank_a, singular_a = _solve_basis_ols(a_phi, a_y)
            theta_ols_c, _, rank_c, singular_c = _solve_basis_ols(c_phi, c_y)
            if (
                rank_a != representative.n_complex_coefficients
                or rank_c != representative.n_complex_coefficients
            ):
                raise RuntimeError(f"state={state_id} structure={structure_id} rank deficient")
            ols_a = (theta_ols_a, int(rank_a), np.asarray(singular_a))
            ols_c = (theta_ols_c, int(rank_c), np.asarray(singular_c))
            for candidate in candidates:
                if candidate.basis_columns != representative.basis_columns:
                    a_phi_candidate = np.asarray(prepared.a_end.phi_full["A"])[
                        :, candidate.basis_columns
                    ]
                    a_b_phi_candidate = np.asarray(prepared.a_end.phi_full["B"])[
                        :, candidate.basis_columns
                    ]
                    c_phi_candidate = np.asarray(prepared.c2.phi_full["C"])[
                        :, candidate.basis_columns
                    ]
                    c_b_phi_candidate = np.asarray(prepared.c2.phi_full["B"])[
                        :, candidate.basis_columns
                    ]
                    theta_ols_a, _, rank_a, singular_a = _solve_basis_ols(a_phi_candidate, a_y)
                    theta_ols_c, _, rank_c, singular_c = _solve_basis_ols(c_phi_candidate, c_y)
                    if (
                        rank_a != candidate.n_complex_coefficients
                        or rank_c != candidate.n_complex_coefficients
                    ):
                        raise RuntimeError(
                            f"state={state_id} candidate={candidate.candidate_id} rank deficient"
                        )
                    ols_a = (theta_ols_a, int(rank_a), np.asarray(singular_a))
                    ols_c = (theta_ols_c, int(rank_c), np.asarray(singular_c))
                    a_phi_use, a_b_phi_use = a_phi_candidate, a_b_phi_candidate
                    c_phi_use, c_b_phi_use = c_phi_candidate, c_b_phi_candidate
                else:
                    a_phi_use, a_b_phi_use = a_phi, a_b_phi
                    c_phi_use, c_b_phi_use = c_phi, c_b_phi
                theta_a, _ = fit_coefficients_ridge(
                    a_phi_use,
                    a_y,
                    candidate.ridge_lambda,
                    ols_solution=ols_a,
                )
                theta_c, _ = fit_coefficients_ridge(
                    c_phi_use,
                    c_y,
                    candidate.ridge_lambda,
                    ols_solution=ols_c,
                )
                candidate_rows.append(
                    {
                        "state_id": state_id,
                        "candidate_id": candidate.candidate_id,
                        "order_profile": candidate.order_profile,
                        "orders": _json_compact(list(candidate.orders)),
                        "memory_profile": candidate.memory_profile,
                        "memory_definition": _json_compact(candidate.memory_definition),
                        "lambda": candidate.ridge_lambda,
                        "n_complex_coefficients": candidate.n_complex_coefficients,
                        "max_delay": candidate.max_delay,
                        "is_baseline": candidate.is_baseline,
                        "Y_Aend_train_NMSE_dB": calculate_nmse(a_y, a_phi_use @ theta_a),
                        "Y_Aend_B_NMSE_dB": calculate_nmse(a_b_y, a_b_phi_use @ theta_a),
                        "Y_C2_train_NMSE_dB": calculate_nmse(c_y, c_phi_use @ theta_c),
                        "Y_C2_B_NMSE_dB": calculate_nmse(c_b_y, c_b_phi_use @ theta_c),
                    }
                )
        if progress_interval > 0 and (
            position % progress_interval == 0 or position == len(state_ids)
        ):
            print(f"development scan: processed {position} / {len(state_ids)} states", flush=True)
    result = pd.DataFrame(candidate_rows)
    expected_rows = DEVELOPMENT_COUNT * len(CANDIDATES)
    if result.shape[0] != expected_rows:
        raise RuntimeError(f"development scan行数错误：{result.shape[0]} != {expected_rows}")
    if not np.all(
        np.isfinite(
            result[
                [
                    "Y_Aend_train_NMSE_dB",
                    "Y_Aend_B_NMSE_dB",
                    "Y_C2_train_NMSE_dB",
                    "Y_C2_B_NMSE_dB",
                ]
            ].to_numpy(dtype=float)
        )
    ):
        raise RuntimeError("development scan指标包含非有限值")
    return result


def _assert_development_only(frame: pd.DataFrame) -> None:
    if "split" in frame.columns and set(frame["split"].astype(str)) != {"Development"}:
        raise ValueError("模型选择输入包含Validation状态，违反VALIDATION_LOCKED")
    forbidden_tokens = ("retrieval", "real_b", "failure", "shareable")
    forbidden = [
        column
        for column in frame.columns
        if any(token in column.lower() for token in forbidden_tokens)
    ]
    if forbidden:
        raise ValueError(f"candidate score不得接收retrieval/Real-B/failure字段：{forbidden}")


def aggregate_development_scan(scan: pd.DataFrame, candidate_grid: pd.DataFrame) -> pd.DataFrame:
    """Aggregate statewise Development NMSE and apply the frozen ranking rule."""

    _assert_development_only(scan)
    metric_columns = [
        "Y_Aend_train_NMSE_dB",
        "Y_Aend_B_NMSE_dB",
        "Y_C2_train_NMSE_dB",
        "Y_C2_B_NMSE_dB",
    ]
    if scan["state_id"].nunique() != DEVELOPMENT_COUNT:
        raise ValueError("Development scan状态数错误")
    rows: list[dict[str, Any]] = []
    for candidate_id, group in scan.groupby("candidate_id", sort=True):
        if group.shape[0] != DEVELOPMENT_COUNT:
            raise ValueError(f"candidate={candidate_id}未覆盖全部Development状态")
        values = {column: group[column].to_numpy(dtype=float) for column in metric_columns}
        a_gap = values["Y_Aend_B_NMSE_dB"] - values["Y_Aend_train_NMSE_dB"]
        c_gap = values["Y_C2_B_NMSE_dB"] - values["Y_C2_train_NMSE_dB"]
        a_median = float(np.median(values["Y_Aend_B_NMSE_dB"]))
        c_median = float(np.median(values["Y_C2_B_NMSE_dB"]))
        a_q95 = float(np.quantile(values["Y_Aend_B_NMSE_dB"], 0.95))
        c_q95 = float(np.quantile(values["Y_C2_B_NMSE_dB"], 0.95))
        rows.append(
            {
                "candidate_id": int(candidate_id),
                "Aend_train_median_dB": float(np.median(values["Y_Aend_train_NMSE_dB"])),
                "Aend_B_median_dB": a_median,
                "Aend_B_q95_dB": a_q95,
                "C2_train_median_dB": float(np.median(values["Y_C2_train_NMSE_dB"])),
                "C2_B_median_dB": c_median,
                "C2_B_q95_dB": c_q95,
                "Aend_gap_median_dB": float(np.median(a_gap)),
                "C2_gap_median_dB": float(np.median(c_gap)),
                "worst_side_B_median_dB": max(a_median, c_median),
                "worst_side_B_q95_dB": max(a_q95, c_q95),
                "mean_side_B_median_dB": (a_median + c_median) / 2.0,
            }
        )
    summary = pd.DataFrame(rows).merge(
        candidate_grid, on="candidate_id", how="left", validate="one_to_one"
    )
    summary = summary.sort_values(
        [
            "worst_side_B_median_dB",
            "worst_side_B_q95_dB",
            "mean_side_B_median_dB",
            "n_complex_coefficients",
            "candidate_id",
        ],
        ascending=[True, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    summary["rank"] = np.arange(1, summary.shape[0] + 1, dtype=np.int64)
    summary["is_top20"] = summary["rank"] <= 20
    return summary


def select_best_candidate(development_summary: pd.DataFrame) -> dict[str, Any]:
    """Select the first row after the frozen Development-only lexicographic sort."""

    _assert_development_only(development_summary)
    if "rank" not in development_summary.columns or development_summary.empty:
        raise ValueError("development summary为空或缺少rank")
    best = development_summary.sort_values("rank", kind="mergesort").iloc[0]
    candidate = CANDIDATE_BY_ID[int(best["candidate_id"])]
    return {
        "candidate_id": candidate.candidate_id,
        "order_profile": candidate.order_profile,
        "orders": list(candidate.orders),
        "memory_profile": candidate.memory_profile,
        "memory_definition": candidate.memory_definition,
        "lambda": candidate.ridge_lambda,
        "selected_lambda": candidate.ridge_lambda,
        "n_complex_coefficients": candidate.n_complex_coefficients,
        "max_delay": candidate.max_delay,
        "is_baseline": candidate.is_baseline,
        "structure_id": candidate.structure_id,
        "selection_rank": int(best["rank"]),
        "selection_rule": [
            "worst_side_B_median_dB ascending",
            "worst_side_B_q95_dB ascending",
            "mean_side_B_median_dB ascending",
            "n_complex_coefficients ascending",
            "candidate_id ascending",
        ],
    }


def candidate_from_payload(payload: Mapping[str, Any]) -> ModelCandidate:
    candidate_id = int(payload["candidate_id"])
    if candidate_id not in CANDIDATE_BY_ID:
        raise ValueError(f"selected candidate_id不存在：{candidate_id}")
    candidate = CANDIDATE_BY_ID[candidate_id]
    if (
        list(candidate.orders) != [int(value) for value in payload["orders"]]
        or candidate.memory_profile != str(payload["memory_profile"])
        or float(candidate.ridge_lambda) != float(payload["lambda"])
    ):
        raise ValueError("selected_model.json与candidate grid不一致")
    return candidate


def fit_selected_model_all_states(
    candidate: ModelCandidate,
    state_ids: Iterable[int],
    common_b_input: np.ndarray,
    *,
    split_frame: pd.DataFrame,
    progress_interval: int = 10,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fit the frozen candidate on all states and construct common-B fingerprints."""

    common_b_input = np.asarray(common_b_input)
    if common_b_input.shape != (COMMON_B_INPUT_LENGTH,) or common_b_input.dtype != np.complex128:
        raise ValueError("common_B_input必须是complex128(4915,)")
    phi_common_full = build_mp_basis(common_b_input, FULL_ORDERS, FULL_MEMORY)
    phi_common = phi_common_full[:, candidate.basis_columns]
    if phi_common.shape != (FINGERPRINT_LENGTH, candidate.n_complex_coefficients):
        raise RuntimeError(f"selected Phi_common shape错误：{phi_common.shape}")

    split_lookup = split_frame.set_index("state_id")["split"].to_dict()
    state_ids = [int(value) for value in state_ids]
    if state_ids != list(range(STATE_COUNT)):
        raise ValueError("Full-425 state ids必须严格为0...424")
    theta_a = np.empty((STATE_COUNT, candidate.n_complex_coefficients), dtype=np.complex128)
    theta_c = np.empty_like(theta_a)
    rows: list[dict[str, Any]] = []
    for position, state_id in enumerate(state_ids, start=1):
        prepared = prepare_state(state_id)
        a_phi, a_y = _select_phi(prepared.a_end, "A", candidate)
        a_b_phi, a_b_y = _select_phi(prepared.a_end, "B", candidate)
        c_phi, c_y = _select_phi(prepared.c2, "C", candidate)
        c_b_phi, c_b_y = _select_phi(prepared.c2, "B", candidate)
        theta_a_result, _, rank_a, singular_a = _solve_basis_ols(a_phi, a_y)
        theta_c_result, _, rank_c, singular_c = _solve_basis_ols(c_phi, c_y)
        if rank_a != candidate.n_complex_coefficients or rank_c != candidate.n_complex_coefficients:
            raise RuntimeError(f"selected candidate在state={state_id} rank deficient")
        theta_a_result, diag_a = fit_coefficients_ridge(
            a_phi,
            a_y,
            candidate.ridge_lambda,
            ols_solution=(theta_a_result, int(rank_a), np.asarray(singular_a)),
        )
        theta_c_result, diag_c = fit_coefficients_ridge(
            c_phi,
            c_y,
            candidate.ridge_lambda,
            ols_solution=(theta_c_result, int(rank_c), np.asarray(singular_c)),
        )
        theta_a[state_id] = theta_a_result
        theta_c[state_id] = theta_c_result
        rows.append(
            {
                "state_id": state_id,
                "split": split_lookup[state_id],
                "ilc_A_end": prepared.ilc_column_count,
                "effective_C2": C2_STAGE,
                "Y_Aend_train_NMSE_dB": calculate_nmse(a_y, a_phi @ theta_a_result),
                "Y_Aend_B_NMSE_dB": calculate_nmse(a_b_y, a_b_phi @ theta_a_result),
                "Y_C2_train_NMSE_dB": calculate_nmse(c_y, c_phi @ theta_c_result),
                "Y_C2_B_NMSE_dB": calculate_nmse(c_b_y, c_b_phi @ theta_c_result),
                "Y_Aend_theta_l2_norm": diag_a.theta_l2_norm,
                "Y_C2_theta_l2_norm": diag_c.theta_l2_norm,
                "Y_Aend_rank": diag_a.rank_phi,
                "Y_C2_rank": diag_c.rank_phi,
                "n_train_Aend": diag_a.n_train_samples,
                "n_train_C2": diag_c.n_train_samples,
            }
        )
        if progress_interval > 0 and (
            position % progress_interval == 0 or position == len(state_ids)
        ):
            print(f"selected model: processed {position} / {len(state_ids)} states", flush=True)
    metrics = pd.DataFrame(rows)
    if metrics.shape[0] != STATE_COUNT:
        raise RuntimeError("selected_model_state_metrics_all425必须有425行")
    fingerprints_a = (phi_common @ theta_a.T).T.astype(np.complex128, copy=False)
    fingerprints_c = (phi_common @ theta_c.T).T.astype(np.complex128, copy=False)
    if (
        fingerprints_a.shape != (STATE_COUNT, FINGERPRINT_LENGTH)
        or fingerprints_c.shape != fingerprints_a.shape
    ):
        raise RuntimeError("selected fingerprint shape错误")
    if not np.all(np.isfinite(theta_a)) or not np.all(np.isfinite(theta_c)):
        raise RuntimeError("selected theta包含非有限值")
    if not np.all(np.isfinite(fingerprints_a)) or not np.all(np.isfinite(fingerprints_c)):
        raise RuntimeError("selected fingerprint包含非有限值")
    return metrics, theta_a, theta_c, fingerprints_a, fingerprints_c


def load_real_b_reference(
    real_b_path: Path,
    reference_distance_path: Path,
    reference_ranking_path: Path,
) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate the frozen Real-B matrix without recomputing it."""

    real_b_path = Path(real_b_path)
    with np.load(real_b_path, allow_pickle=False) as data:
        if not {"state_ids", "D_B", "R_B"}.issubset(data.files):
            raise ValueError("Real-B matrix缺少state_ids/D_B/R_B")
        state_ids = np.asarray(data["state_ids"], dtype=np.int64)
        distance = np.asarray(data["D_B"])
        ranking = np.asarray(data["R_B"])
    if not np.array_equal(state_ids, np.arange(STATE_COUNT, dtype=np.int64)):
        raise ValueError("Real-B state_ids错误")
    if distance.shape != (STATE_COUNT, STATE_COUNT) or distance.dtype != np.float64:
        raise ValueError("Real-B distance shape/dtype错误")
    if ranking.shape != distance.shape or ranking.dtype != np.float64:
        raise ValueError("Real-B ranking shape/dtype错误")
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


def _stable_retrieval(
    distance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if (
        distance.shape != (STATE_COUNT, STATE_COUNT)
        or np.isnan(distance).any()
        or np.isposinf(distance).any()
    ):
        raise ValueError("retrieval distance非法")
    state_ids = np.arange(STATE_COUNT, dtype=np.int64)
    selected = np.empty(STATE_COUNT, dtype=np.int64)
    selected_distance = np.empty(STATE_COUNT, dtype=np.float64)
    true_rank = np.empty(STATE_COUNT, dtype=np.int64)
    tie_count = np.empty(STATE_COUNT, dtype=np.int64)
    ranking_order = np.empty((STATE_COUNT, STATE_COUNT), dtype=np.int64)
    for state_id in range(STATE_COUNT):
        order = np.lexsort((state_ids, distance[state_id]))
        ranking_order[state_id] = order
        selected_index = int(order[0])
        selected[state_id] = selected_index
        selected_distance[state_id] = float(distance[state_id, selected_index])
        minimum = distance[state_id, selected_index]
        tie_count[state_id] = int(np.count_nonzero(distance[state_id] == minimum))
        true_rank[state_id] = int(np.flatnonzero(order == state_id)[0] + 1)
    return selected, selected_distance, true_rank, tie_count, ranking_order


def _shareable(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=float) < DPD_SHAREABLE_THRESHOLD_DB


def build_selected_retrieval(
    selected_query_fingerprints: np.ndarray,
    selected_lut_fingerprints: np.ndarray,
    real_b_distance: np.ndarray,
    state_frame: pd.DataFrame,
) -> dict[str, Any]:
    distance = compute_cnmse_distance_matrix(selected_query_fingerprints, selected_lut_fingerprints)
    ranking = distance_to_ranks(distance)
    selected_q, selected_distance, true_rank, tie_count, ranking_order = _stable_retrieval(distance)
    state_ids = np.arange(STATE_COUNT, dtype=np.int64)
    real_b_values = real_b_distance[state_ids, selected_q]
    exact = selected_q == state_ids
    shareable = _shareable(real_b_values)
    lookup = state_frame.set_index("state_id")
    rows = []
    for state_id in state_ids:
        query_id = int(selected_q[state_id])
        rows.append(
            {
                "State_n_R": int(state_id),
                "State_n_Q": query_id,
                "exact_hit": bool(exact[state_id]),
                "true_state_rank": int(true_rank[state_id]),
                "top3_true_state_hit": bool(true_rank[state_id] <= 3),
                "top5_true_state_hit": bool(true_rank[state_id] <= 5),
                "query_to_selected_LUT_CNMSE_dB": float(selected_distance[state_id]),
                "minimum_tie_count": int(tie_count[state_id]),
                "Aend_stage_R": int(lookup.loc[int(state_id), "ilc_A_end"]),
                "Aend_stage_Q": int(lookup.loc[query_id, "ilc_A_end"]),
                "funMng_R": int(lookup.loc[int(state_id), "funMng"]),
                "funAng_R": int(lookup.loc[int(state_id), "funAng"]),
                "secMng_R": int(lookup.loc[int(state_id), "secMng"]),
                "secAng_R": int(lookup.loc[int(state_id), "secAng"]),
                "funMng_Q": int(lookup.loc[query_id, "funMng"]),
                "funAng_Q": int(lookup.loc[query_id, "funAng"]),
                "secMng_Q": int(lookup.loc[query_id, "secMng"]),
                "secAng_Q": int(lookup.loc[query_id, "secAng"]),
                "retrieved_real_B_CNMSE_dB": float(real_b_values[state_id]),
                "dpd_shareable": bool(shareable[state_id]),
                "failure": bool(not shareable[state_id]),
            }
        )
    results = pd.DataFrame(rows)
    diagnostics = results.loc[
        :,
        [
            "State_n_R",
            "State_n_Q",
            "exact_hit",
            "retrieved_real_B_CNMSE_dB",
            "dpd_shareable",
            "failure",
        ],
    ].copy()
    values_nonexact = real_b_values[(~exact) & np.isfinite(real_b_values)]
    summary = {
        "state_count": STATE_COUNT,
        "candidate_count": STATE_COUNT,
        "candidate_count_per_query": STATE_COUNT,
        "query": "Y-C2",
        "lut": "Y-Aend",
        "self_match_included": True,
        "exact_hit_count": int(exact.sum()),
        "exact_hit_rate": float(exact.mean()),
        "non_exact_count": int((~exact).sum()),
        "non_exact_finite_count": int(values_nonexact.size),
        "true_state_rank_median": float(np.median(true_rank)),
        "true_state_rank_max": int(np.max(true_rank)),
        "minimum_tie_count_rows": int(np.count_nonzero(tie_count > 1)),
        "top3_true_state_hit_count": int(np.count_nonzero(true_rank <= 3)),
        "top3_true_state_hit_rate": float(np.mean(true_rank <= 3)),
        "top5_true_state_hit_count": int(np.count_nonzero(true_rank <= 5)),
        "top5_true_state_hit_rate": float(np.mean(true_rank <= 5)),
        "DPD_shareable_threshold_dB": DPD_SHAREABLE_THRESHOLD_DB,
        "DPD_shareable_threshold_rule": "Real-B CNMSE < -40 dB",
        "DPD_shareable_count": int(shareable.sum()),
        "DPD_shareable_rate": float(shareable.mean()),
        "failure_count": int((~shareable).sum()),
        "non_exact_B_CNMSE_count": int(values_nonexact.size),
        "non_exact_B_CNMSE_median": float(np.median(values_nonexact)),
        "non_exact_B_CNMSE_q95": float(np.quantile(values_nonexact, 0.95)),
        "non_exact_B_CNMSE_min": float(np.min(values_nonexact)),
        "non_exact_B_CNMSE_max": float(np.max(values_nonexact)),
    }
    return {
        "distance": distance,
        "ranking": ranking,
        "ranking_order": ranking_order,
        "selected_q": selected_q,
        "selected_distance": selected_distance,
        "true_rank": true_rank,
        "tie_count": tie_count,
        "real_b_values": real_b_values,
        "results": results,
        "diagnostics": diagnostics,
        "summary": summary,
    }


def load_baseline_state_metrics(
    all_ilc_metrics_path: Path,
    availability_path: Path,
) -> pd.DataFrame:
    """Extract the frozen P9/M0/1e-8 Aend and C2 rows state-wise."""

    metrics = pd.read_csv(all_ilc_metrics_path)
    availability = pd.read_csv(availability_path).sort_values("state_id").reset_index(drop=True)
    if metrics.empty or availability.shape[0] != STATE_COUNT:
        raise ValueError("Baseline模型指标或availability不完整")
    counts = availability["ilc_column_count"].to_numpy(dtype=np.int64)
    rows = []
    for state_id, a_end_stage in enumerate(counts):
        a = metrics.loc[
            (metrics["state_id"] == state_id)
            & (metrics["model_role"] == "Y-A")
            & (metrics["actual_ilc_n"] == int(a_end_stage))
            & (np.isclose(metrics["ridge_lambda"].astype(float), BASELINE_LAMBDA))
        ]
        c = metrics.loc[
            (metrics["state_id"] == state_id)
            & (metrics["model_role"] == "Y-C")
            & (metrics["actual_ilc_n"] == C2_STAGE)
            & (np.isclose(metrics["ridge_lambda"].astype(float), BASELINE_LAMBDA))
        ]
        if a.shape[0] != 1 or c.shape[0] != 1:
            raise ValueError(
                f"state={state_id} Baseline Aend/C2行数错误：{a.shape[0]}/{c.shape[0]}"
            )
        rows.append(
            {
                "state_id": state_id,
                "ilc_A_end": int(a_end_stage),
                "Y_Aend_train_NMSE_dB": float(a.iloc[0]["train_nmse_db"]),
                "Y_Aend_B_NMSE_dB": float(a.iloc[0]["B_generalization_nmse_db"]),
                "Y_C2_train_NMSE_dB": float(c.iloc[0]["train_nmse_db"]),
                "Y_C2_B_NMSE_dB": float(c.iloc[0]["B_generalization_nmse_db"]),
            }
        )
    frame = pd.DataFrame(rows)
    if not np.all(np.isfinite(frame.iloc[:, 2:].to_numpy(dtype=float))):
        raise ValueError("Baseline state metrics包含非有限值")
    return frame


def baseline_regression_report(
    observed: pd.DataFrame, baseline_reference: pd.DataFrame
) -> dict[str, Any]:
    """Compare the re-fit Baseline against frozen all-ILC metrics."""

    merged = observed.merge(baseline_reference, on="state_id", suffixes=("_observed", "_reference"))
    if merged.shape[0] != DEVELOPMENT_COUNT:
        raise ValueError("Baseline regression必须在340个Development状态上比较")
    columns = [
        "Y_Aend_train_NMSE_dB",
        "Y_Aend_B_NMSE_dB",
        "Y_C2_train_NMSE_dB",
        "Y_C2_B_NMSE_dB",
    ]
    errors = {}
    for column in columns:
        errors[column] = float(
            np.max(
                np.abs(
                    merged[f"{column}_observed"].to_numpy(dtype=float)
                    - merged[f"{column}_reference"].to_numpy(dtype=float)
                )
            )
        )
    maximum = max(errors.values())
    return {
        "state_count": DEVELOPMENT_COUNT,
        "metric_max_abs_error_dB": errors,
        "max_abs_error_dB": float(maximum),
        "tolerance_dB": 1e-10,
        "pass": bool(maximum <= 1e-10),
    }


def load_pa_nonlinearity_values(
    state_ids: Iterable[int], *, progress_interval: int = 50
) -> np.ndarray:
    """Read only the scalar ``nmse_withoutdpd`` values for quartile diagnosis."""

    values = np.empty(STATE_COUNT, dtype=np.float64)
    state_ids = [int(value) for value in state_ids]
    if state_ids != list(range(STATE_COUNT)):
        raise ValueError("PA nonlinearity state ids必须严格为0...424")
    for position, state_id in enumerate(state_ids, start=1):
        value = np.asarray(load_variable_by_id(state_id, "nmse_withoutdpd"), dtype=float).reshape(
            -1
        )
        if value.size != 1 or not np.isfinite(value[0]):
            raise ValueError(f"state={state_id} nmse_withoutdpd无效")
        values[state_id] = float(value[0])
        if progress_interval > 0 and (
            position % progress_interval == 0 or position == len(state_ids)
        ):
            print(f"PA nonlinearity: processed {position} / {len(state_ids)} states", flush=True)
    return values


def assign_pa_quartiles(values: np.ndarray) -> np.ndarray:
    """Assign Q1 (most linear) through Q4 (closest to zero) deterministically."""

    values = np.asarray(values, dtype=float)
    if values.shape != (STATE_COUNT,) or not np.all(np.isfinite(values)):
        raise ValueError("nmse_withoutdpd values必须是425个finite值")
    order = np.argsort(values, kind="mergesort")
    labels = np.empty(STATE_COUNT, dtype=object)
    names = np.asarray(["Q1", "Q2", "Q3", "Q4"], dtype=object)
    quartile_index = np.minimum((np.arange(STATE_COUNT) * 4) // STATE_COUNT, 3)
    labels[order] = names[quartile_index]
    return labels


def _safe_correlation(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    mask = np.isfinite(x) & np.isfinite(y)
    x_valid = np.asarray(x[mask], dtype=float)
    y_valid = np.asarray(y[mask], dtype=float)
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


def build_statewise_comparison(
    baseline_metrics: pd.DataFrame,
    selected_metrics: pd.DataFrame,
    baseline_retrieval: pd.DataFrame,
    selected_retrieval: pd.DataFrame,
    nmse_withoutdpd: np.ndarray,
    split_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build statewise gains, failure transitions and gain correlations."""

    selected_metrics = selected_metrics.copy()
    baseline_metrics = baseline_metrics.copy()
    selected_metrics = selected_metrics.rename(
        columns={
            "Y_Aend_train_NMSE_dB": "selected_Y_Aend_train_NMSE_dB",
            "Y_Aend_B_NMSE_dB": "selected_Y_Aend_B_NMSE_dB",
            "Y_C2_train_NMSE_dB": "selected_Y_C2_train_NMSE_dB",
            "Y_C2_B_NMSE_dB": "selected_Y_C2_B_NMSE_dB",
        }
    )
    baseline_metrics = baseline_metrics.rename(
        columns={
            "Y_Aend_train_NMSE_dB": "baseline_Y_Aend_train_NMSE_dB",
            "Y_Aend_B_NMSE_dB": "baseline_Y_Aend_B_NMSE_dB",
            "Y_C2_train_NMSE_dB": "baseline_Y_C2_train_NMSE_dB",
            "Y_C2_B_NMSE_dB": "baseline_Y_C2_B_NMSE_dB",
        }
    )
    baseline_retrieval = baseline_retrieval.rename(
        columns={
            "State_n_R": "state_id",
            "State_n_Q": "baseline_state_id_Q",
            "exact_hit": "baseline_exact_hit",
            "retrieved_real_B_CNMSE_dB": "baseline_retrieved_real_B_CNMSE_dB",
            "dpd_shareable": "baseline_shareable",
            "failure": "baseline_failure",
        }
    )
    selected_retrieval = selected_retrieval.rename(
        columns={
            "State_n_R": "state_id",
            "State_n_Q": "selected_state_id_Q",
            "exact_hit": "selected_exact_hit",
            "retrieved_real_B_CNMSE_dB": "selected_retrieved_real_B_CNMSE_dB",
            "dpd_shareable": "selected_shareable",
            "failure": "selected_failure",
        }
    )
    columns_retrieval = [
        "state_id",
        "baseline_state_id_Q",
        "baseline_exact_hit",
        "baseline_retrieved_real_B_CNMSE_dB",
        "baseline_shareable",
        "baseline_failure",
    ]
    selected_columns_retrieval = [
        "state_id",
        "selected_state_id_Q",
        "selected_exact_hit",
        "selected_retrieved_real_B_CNMSE_dB",
        "selected_shareable",
        "selected_failure",
    ]
    frame = (
        baseline_metrics.merge(selected_metrics, on="state_id", how="inner", validate="one_to_one")
        .merge(
            baseline_retrieval.loc[:, columns_retrieval],
            on="state_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            selected_retrieval.loc[:, selected_columns_retrieval],
            on="state_id",
            how="inner",
            validate="one_to_one",
        )
        .merge(
            split_frame.loc[:, ["state_id", "split"]],
            on="state_id",
            how="left",
            validate="one_to_one",
        )
    )
    frame["nmse_withoutdpd_dB"] = np.asarray(nmse_withoutdpd, dtype=float)[
        frame["state_id"].to_numpy(dtype=int)
    ]
    gain_pairs = {
        "gain_Y_Aend_train_dB": ("baseline_Y_Aend_train_NMSE_dB", "selected_Y_Aend_train_NMSE_dB"),
        "gain_Y_Aend_B_dB": ("baseline_Y_Aend_B_NMSE_dB", "selected_Y_Aend_B_NMSE_dB"),
        "gain_Y_C2_train_dB": ("baseline_Y_C2_train_NMSE_dB", "selected_Y_C2_train_NMSE_dB"),
        "gain_Y_C2_B_dB": ("baseline_Y_C2_B_NMSE_dB", "selected_Y_C2_B_NMSE_dB"),
    }
    for name, (baseline_column, selected_column) in gain_pairs.items():
        raw_gain = frame[baseline_column] - frame[selected_column]
        frame[name] = raw_gain.mask(raw_gain.abs() <= NUMERIC_GAIN_TOLERANCE_DB, 0.0)
    frame["baseline_worst_side_B_dB"] = np.maximum(
        frame["baseline_Y_Aend_B_NMSE_dB"], frame["baseline_Y_C2_B_NMSE_dB"]
    )
    frame["selected_worst_side_B_dB"] = np.maximum(
        frame["selected_Y_Aend_B_NMSE_dB"], frame["selected_Y_C2_B_NMSE_dB"]
    )
    raw_worst_gain = frame["baseline_worst_side_B_dB"] - frame["selected_worst_side_B_dB"]
    frame["gain_worst_side_B_dB"] = raw_worst_gain.mask(
        raw_worst_gain.abs() <= NUMERIC_GAIN_TOLERANCE_DB,
        0.0,
    )
    finite_retrieval = np.isfinite(frame["baseline_retrieved_real_B_CNMSE_dB"]) & np.isfinite(
        frame["selected_retrieved_real_B_CNMSE_dB"]
    )
    raw_retrieval_gain = (
        frame["baseline_retrieved_real_B_CNMSE_dB"] - frame["selected_retrieved_real_B_CNMSE_dB"]
    )
    raw_retrieval_gain = raw_retrieval_gain.mask(
        raw_retrieval_gain.abs() <= NUMERIC_GAIN_TOLERANCE_DB,
        0.0,
    )
    frame["retrieval_gain_dB"] = np.where(
        finite_retrieval,
        raw_retrieval_gain,
        np.nan,
    )
    frame["retrieval_gain_finite_finite"] = finite_retrieval
    frame["transition_class"] = np.select(
        [
            frame["baseline_failure"] & ~frame["selected_failure"],
            frame["baseline_failure"] & frame["selected_failure"],
            ~frame["baseline_failure"] & ~frame["selected_failure"],
            ~frame["baseline_failure"] & frame["selected_failure"],
        ],
        ["failure_to_success", "failure_to_failure", "success_to_success", "success_to_failure"],
        default="unknown",
    )
    frame["exact_transition_class"] = np.select(
        [
            frame["baseline_exact_hit"] & frame["selected_exact_hit"],
            ~frame["baseline_exact_hit"] & frame["selected_exact_hit"],
            frame["baseline_exact_hit"] & ~frame["selected_exact_hit"],
        ],
        ["exact_to_exact", "nonexact_to_exact", "exact_to_nonexact"],
        default="nonexact_to_nonexact",
    )
    frame["pa_nonlinearity_quartile"] = assign_pa_quartiles(nmse_withoutdpd)[
        frame["state_id"].to_numpy(dtype=int)
    ]
    frame = frame.sort_values("state_id").reset_index(drop=True)

    quartile_rows = []
    for quartile in ("Q1", "Q2", "Q3", "Q4"):
        group = frame.loc[frame["pa_nonlinearity_quartile"] == quartile]
        quartile_rows.append(
            {
                "pa_nonlinearity_quartile": quartile,
                "state_count": int(group.shape[0]),
                "nmse_withoutdpd_median_dB": float(group["nmse_withoutdpd_dB"].median()),
                "gain_Y_Aend_B_median_dB": float(group["gain_Y_Aend_B_dB"].median()),
                "gain_Y_C2_B_median_dB": float(group["gain_Y_C2_B_dB"].median()),
                "gain_worst_side_B_median_dB": float(group["gain_worst_side_B_dB"].median()),
            }
        )
    quartile_summary = pd.DataFrame(quartile_rows)

    baseline_failure_transition = (
        frame.loc[frame["baseline_failure"]]
        .loc[
            :,
            [
                "state_id",
                "baseline_Y_Aend_B_NMSE_dB",
                "selected_Y_Aend_B_NMSE_dB",
                "gain_Y_Aend_B_dB",
                "baseline_Y_C2_B_NMSE_dB",
                "selected_Y_C2_B_NMSE_dB",
                "gain_Y_C2_B_dB",
                "baseline_state_id_Q",
                "selected_state_id_Q",
                "baseline_retrieved_real_B_CNMSE_dB",
                "selected_retrieved_real_B_CNMSE_dB",
                "baseline_failure",
                "selected_failure",
            ],
        ]
        .copy()
    )

    correlations = {
        "gain_Y_Aend_B_vs_retrieval_gain": _safe_correlation(
            frame["gain_Y_Aend_B_dB"].to_numpy(dtype=float),
            frame["retrieval_gain_dB"].to_numpy(dtype=float),
        ),
        "gain_Y_C2_B_vs_retrieval_gain": _safe_correlation(
            frame["gain_Y_C2_B_dB"].to_numpy(dtype=float),
            frame["retrieval_gain_dB"].to_numpy(dtype=float),
        ),
        "gain_worst_side_B_vs_retrieval_gain": _safe_correlation(
            frame["gain_worst_side_B_dB"].to_numpy(dtype=float),
            frame["retrieval_gain_dB"].to_numpy(dtype=float),
        ),
    }
    transition_counts = frame["transition_class"].value_counts().to_dict()
    transition_summary = pd.DataFrame(
        [
            {"transition_class": key, "state_count": int(value)}
            for key, value in sorted(transition_counts.items())
        ]
    )
    comparison = {
        "transition_counts": {str(key): int(value) for key, value in transition_counts.items()},
        "exact_transition_counts": {
            str(key): int(value)
            for key, value in frame["exact_transition_class"].value_counts().to_dict().items()
        },
        "finite_finite_retrieval_gain_count": int(finite_retrieval.sum()),
        "correlations": correlations,
        "baseline_failure_count": int(frame["baseline_failure"].sum()),
        "selected_failure_count": int(frame["selected_failure"].sum()),
        "new_success_to_failure_count": int(
            np.count_nonzero(frame["transition_class"] == "success_to_failure")
        ),
        "old_failure_to_success_count": int(
            np.count_nonzero(frame["transition_class"] == "failure_to_success")
        ),
    }
    return (
        frame,
        quartile_summary,
        baseline_failure_transition,
        {
            "comparison": comparison,
            "transition_summary": transition_summary,
        },
    )


def retrieval_validation_summary(
    baseline_retrieval: pd.DataFrame,
    selected_retrieval: pd.DataFrame,
    validation_state_ids: Iterable[int],
) -> dict[str, Any]:
    ids = np.asarray([int(value) for value in validation_state_ids], dtype=np.int64)
    baseline = baseline_retrieval.set_index("State_n_R").loc[ids]
    selected = selected_retrieval.set_index("State_n_R").loc[ids]
    baseline_count = int(baseline["dpd_shareable"].sum())
    selected_count = int(selected["dpd_shareable"].sum())
    baseline_rate = baseline_count / VALIDATION_COUNT
    selected_rate = selected_count / VALIDATION_COUNT
    return {
        "validation_count": VALIDATION_COUNT,
        "baseline_shareable_count": baseline_count,
        "baseline_shareable_rate": baseline_rate,
        "baseline_failure_count": VALIDATION_COUNT - baseline_count,
        "selected_shareable_count": selected_count,
        "selected_shareable_rate": selected_rate,
        "selected_failure_count": VALIDATION_COUNT - selected_count,
        "selected_minus_baseline_rate": selected_rate - baseline_rate,
        "validation_retrieval_improvement": bool(selected_rate > baseline_rate),
        "strict_success_rule": (
            "selected_validation_shareable_rate > baseline_validation_shareable_rate"
        ),
    }


def full_retrieval_summary(
    baseline_retrieval: pd.DataFrame,
    selected_retrieval: pd.DataFrame,
) -> dict[str, Any]:
    def _summary(frame: pd.DataFrame) -> dict[str, Any]:
        values = frame["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float)
        exact = frame["exact_hit"].to_numpy(dtype=bool)
        finite_nonexact = values[(~exact) & np.isfinite(values)]
        return {
            "exact_hit_count": int(exact.sum()),
            "exact_hit_rate": float(exact.mean()),
            "shareable_count": int(frame["dpd_shareable"].sum()),
            "shareable_rate": float(frame["dpd_shareable"].mean()),
            "failure_count": int(frame["failure"].sum()),
            "nonexact_finite_count": int(finite_nonexact.size),
            "nonexact_real_B_median_dB": float(np.median(finite_nonexact)),
            "nonexact_real_B_q95_dB": float(np.quantile(finite_nonexact, 0.95)),
            "nonexact_real_B_worst_dB": float(np.max(finite_nonexact)),
        }

    baseline = _summary(baseline_retrieval)
    selected = _summary(selected_retrieval)
    return {
        "baseline": baseline,
        "selected": selected,
        "selected_minus_baseline_shareable_count": selected["shareable_count"]
        - baseline["shareable_count"],
        "selected_minus_baseline_shareable_rate": selected["shareable_rate"]
        - baseline["shareable_rate"],
    }


__all__ = [
    "BASELINE_LAMBDA",
    "BASELINE_MEMORY",
    "BASELINE_ORDERS",
    "CANDIDATES",
    "CANDIDATE_BY_ID",
    "COMMON_B_INPUT_LENGTH",
    "DPD_SHAREABLE_THRESHOLD_DB",
    "DEVELOPMENT_COUNT",
    "FINGERPRINT_LENGTH",
    "LAMBDA_GRID",
    "MAX_DELAY",
    "MEMORY_PROFILES",
    "NUMERIC_GAIN_TOLERANCE_DB",
    "ORDER_PROFILES",
    "STATE_COUNT",
    "VALIDATION_COUNT",
    "VALID_LENGTHS",
    "ModelCandidate",
    "PreparedPair",
    "PreparedState",
    "aggregate_development_scan",
    "assign_pa_quartiles",
    "baseline_regression_report",
    "build_model_candidate_grid",
    "build_selected_retrieval",
    "build_statewise_comparison",
    "candidate_from_payload",
    "evaluate_single_candidate",
    "fit_selected_model_all_states",
    "full_retrieval_summary",
    "load_baseline_state_metrics",
    "load_fixed_split",
    "load_pa_nonlinearity_values",
    "load_real_b_reference",
    "prepare_state",
    "retrieval_validation_summary",
    "scan_development",
    "select_best_candidate",
    "validate_candidate_grid",
]
