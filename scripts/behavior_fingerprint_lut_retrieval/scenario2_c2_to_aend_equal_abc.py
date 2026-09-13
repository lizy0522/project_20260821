"""
Scenario 2 equal-length ABC segmentation ablation.

Only the A/B/C ownership partition changes relative to the frozen C2 -> Aend
workflow.  The canonical full-record synchronization, segment-local MP basis,
Y-Aend/Y-C2 definitions, Ridge implementation, CNMSE retrieval rule and
strict Real-B threshold remain unchanged.  Because the B window changes, both
the common-B fingerprints and the Real-B distance matrix are rebuilt from the
raw corpus through the read-only data manager.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from behavior_fingerprint_ranking_consistency.ranking import distance_to_ranks
from behavior_model.basis import build_mp_basis
from behavior_model.evaluation import calculate_nmse
from behavior_model.ridge import fit_coefficients_ridge
from data_manager import load_by_id, load_variable_by_id
from scipy.stats import pearsonr, spearmanr
from signal_segmentation import (
    SegmentPartition,
    get_ilc_pair,
    get_off_pair,
    preprocess_full_pair,
)

STATE_COUNT = 425
WAVEFORM_LENGTH = 24576
SEGMENT_LENGTH = 8192
MAX_DELAY = 2
VALID_LENGTH = SEGMENT_LENGTH - MAX_DELAY
FINGERPRINT_LENGTH = VALID_LENGTH
COMMON_B_INPUT_LENGTH = SEGMENT_LENGTH
C2_STAGE = 2
DPD_SHAREABLE_THRESHOLD_DB = -40.0
RIDGE_LAMBDA = 1e-8
ORDERS = (1, 2, 3, 5, 7, 9)
MEMORY = {1: 3, 2: 2, 3: 2, 5: 1, 7: 1, 9: 1}
N_COMPLEX_COEFFICIENTS = 10
SEGMENT_NAMES = ("A", "B", "C")
VALID_LENGTHS = {name: VALID_LENGTH for name in SEGMENT_NAMES}
NUMERIC_TOLERANCE = 1e-12


@dataclass(frozen=True)
class EqualABCPair:
    """One canonical pair represented by local MP bases and valid outputs."""

    pair_type: str
    actual_ilc_n: int | None
    input_peak_normalization_factor: float | None
    phi: Mapping[str, np.ndarray]
    y_valid: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class EqualABCPreparedState:
    """Aend, C2 and OFF canonical data for one state."""

    state_id: int
    ilc_column_count: int
    common_b_input: np.ndarray
    a_end: EqualABCPair
    c2: EqualABCPair
    off: EqualABCPair


def build_equal_abc_partition(signal_length: int = WAVEFORM_LENGTH) -> SegmentPartition:
    """Construct the only partition allowed by this ablation."""

    if int(signal_length) != WAVEFORM_LENGTH:
        raise ValueError(f"equal ABC仅支持总长{WAVEFORM_LENGTH}，实际为{signal_length}")
    partition = SegmentPartition(
        total_length=WAVEFORM_LENGTH,
        a_slice=slice(0, SEGMENT_LENGTH),
        b_slice=slice(SEGMENT_LENGTH, 2 * SEGMENT_LENGTH),
        c_slice=slice(2 * SEGMENT_LENGTH, WAVEFORM_LENGTH),
    )
    validate_equal_abc_partition(partition)
    return partition


def validate_equal_abc_partition(partition: SegmentPartition) -> None:
    """Validate exact boundaries, equal lengths, no overlap and no gaps."""

    if partition.total_length != WAVEFORM_LENGTH:
        raise ValueError("partition总长度错误")
    slices = (partition.a_slice, partition.b_slice, partition.c_slice)
    expected = (
        (0, SEGMENT_LENGTH),
        (SEGMENT_LENGTH, 2 * SEGMENT_LENGTH),
        (2 * SEGMENT_LENGTH, WAVEFORM_LENGTH),
    )
    for current, (start, stop) in zip(slices, expected, strict=True):
        if (current.start, current.stop) != (start, stop):
            raise ValueError(f"equal ABC边界错误：{current}")
        if current.stop - current.start != SEGMENT_LENGTH:
            raise ValueError("equal ABC三段长度必须均为8192")
    if slices[0].stop != slices[1].start or slices[1].stop != slices[2].start:
        raise ValueError("equal ABC存在重叠或间隙")


def segment_definition() -> dict[str, Any]:
    """Return the JSON-serialisable segmentation contract."""

    return {
        "waveform_length": WAVEFORM_LENGTH,
        "A_start": 0,
        "A_stop": SEGMENT_LENGTH,
        "B_start": SEGMENT_LENGTH,
        "B_stop": 2 * SEGMENT_LENGTH,
        "C_start": 2 * SEGMENT_LENGTH,
        "C_stop": WAVEFORM_LENGTH,
        "A_length": SEGMENT_LENGTH,
        "B_length": SEGMENT_LENGTH,
        "C_length": SEGMENT_LENGTH,
        "max_delay": MAX_DELAY,
        "A_valid_length": VALID_LENGTH,
        "B_valid_length": VALID_LENGTH,
        "C_valid_length": VALID_LENGTH,
        "basis_is_segment_local": True,
        "valid_index_rule": "A[2:], B[2:], C[2:]",
    }


def _as_vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or not np.iscomplexobj(array):
        raise ValueError(f"{name}必须是一维复数波形")
    if array.size != WAVEFORM_LENGTH or not np.all(np.isfinite(array)):
        raise ValueError(f"{name}长度或有限性错误")
    return array.astype(np.complex128, copy=False)


def _canonical_pair_to_bases(
    canonical: Any,
    *,
    pair_type: str,
    actual_ilc_n: int | None,
    input_peak_normalization_factor: float | None,
) -> EqualABCPair:
    phi: dict[str, np.ndarray] = {}
    y_valid: dict[str, np.ndarray] = {}
    for segment_name in SEGMENT_NAMES:
        segment = canonical[segment_name]
        segment_input = np.asarray(segment.input, dtype=np.complex128)
        segment_output = np.asarray(segment.output, dtype=np.complex128)
        if segment_input.shape != (SEGMENT_LENGTH,) or segment_output.shape != (SEGMENT_LENGTH,):
            raise RuntimeError(f"{pair_type} {segment_name} ownership长度不是8192")
        basis = build_mp_basis(segment_input, ORDERS, MEMORY)
        output_valid = segment_output[MAX_DELAY:]
        if basis.shape != (VALID_LENGTH, N_COMPLEX_COEFFICIENTS):
            raise RuntimeError(f"{pair_type} {segment_name} Phi shape错误：{basis.shape}")
        if output_valid.shape != (VALID_LENGTH,):
            raise RuntimeError(
                f"{pair_type} {segment_name} y_valid shape错误：{output_valid.shape}"
            )
        if not np.all(np.isfinite(basis)) or not np.all(np.isfinite(output_valid)):
            raise RuntimeError(f"{pair_type} {segment_name}包含非有限值")
        phi[segment_name] = basis
        y_valid[segment_name] = output_valid
    return EqualABCPair(
        pair_type=pair_type,
        actual_ilc_n=actual_ilc_n,
        input_peak_normalization_factor=input_peak_normalization_factor,
        phi=phi,
        y_valid=y_valid,
    )


def prepare_equal_abc_state(state_id: int) -> EqualABCPreparedState:
    """Read one state and rebuild its equal-ABC canonical data."""

    if not isinstance(state_id, (int, np.integer)) or not 0 <= int(state_id) < STATE_COUNT:
        raise ValueError(f"state_id必须位于0...424：{state_id!r}")
    state_id = int(state_id)
    state_data = load_by_id(state_id)
    xin = _as_vector(state_data["xin"], "xin")
    input_history = np.asarray(state_data["xin_pd_ori_ilc"])
    output_history = np.asarray(state_data["yout_withdpd_ori_ilc"])
    if input_history.ndim != 2 or output_history.ndim != 2:
        raise ValueError(f"state={state_id} ILC history必须为二维")
    if input_history.shape != output_history.shape or input_history.shape[0] != WAVEFORM_LENGTH:
        raise ValueError(f"state={state_id} ILC history shape错误")
    ilc_count = int(input_history.shape[1])
    if ilc_count < C2_STAGE or ilc_count > 5:
        raise ValueError(f"state={state_id}真实C2不可用或ILC列数越界：{ilc_count}")
    partition = build_equal_abc_partition()
    common_b_input = xin[partition.b_slice].copy()
    if common_b_input.shape != (COMMON_B_INPUT_LENGTH,):
        raise RuntimeError("new common-B input长度错误")

    a_end_pair = get_ilc_pair(state_data, ilc_count - 1)
    canonical_a_end = preprocess_full_pair(
        a_end_pair.input_full,
        a_end_pair.output_raw_full,
        partition,
        pair_type="ILC",
        iteration_index=ilc_count - 1,
        input_peak_normalization_factor=a_end_pair.input_peak_normalization_factor,
    )
    a_end = _canonical_pair_to_bases(
        canonical_a_end,
        pair_type="Y-Aend",
        actual_ilc_n=ilc_count,
        input_peak_normalization_factor=float(a_end_pair.input_peak_normalization_factor),
    )
    if ilc_count == C2_STAGE:
        c2 = a_end
    else:
        c2_pair = get_ilc_pair(state_data, C2_STAGE - 1)
        canonical_c2 = preprocess_full_pair(
            c2_pair.input_full,
            c2_pair.output_raw_full,
            partition,
            pair_type="ILC",
            iteration_index=C2_STAGE - 1,
            input_peak_normalization_factor=c2_pair.input_peak_normalization_factor,
        )
        c2 = _canonical_pair_to_bases(
            canonical_c2,
            pair_type="Y-C2",
            actual_ilc_n=C2_STAGE,
            input_peak_normalization_factor=float(c2_pair.input_peak_normalization_factor),
        )
    off_pair = get_off_pair(state_data)
    canonical_off = preprocess_full_pair(
        off_pair.input_full,
        off_pair.output_raw_full,
        partition,
        pair_type="OFF",
    )
    off = _canonical_pair_to_bases(
        canonical_off,
        pair_type="Real-B",
        actual_ilc_n=None,
        input_peak_normalization_factor=None,
    )
    return EqualABCPreparedState(
        state_id=state_id,
        ilc_column_count=ilc_count,
        common_b_input=common_b_input,
        a_end=a_end,
        c2=c2,
        off=off,
    )


def fit_equal_abc_state(
    prepared: EqualABCPreparedState,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray]:
    """Fit frozen P9/M0/lambda on Aend-A and C2-C, then evaluate B and OFF-B."""

    a_phi = prepared.a_end.phi["A"]
    a_y = prepared.a_end.y_valid["A"]
    a_b_phi = prepared.a_end.phi["B"]
    a_b_y = prepared.a_end.y_valid["B"]
    c_phi = prepared.c2.phi["C"]
    c_y = prepared.c2.y_valid["C"]
    c_b_phi = prepared.c2.phi["B"]
    c_b_y = prepared.c2.y_valid["B"]
    theta_a, diag_a = fit_coefficients_ridge(a_phi, a_y, RIDGE_LAMBDA)
    theta_c, diag_c = fit_coefficients_ridge(c_phi, c_y, RIDGE_LAMBDA)
    real_b = prepared.off.y_valid["B"]
    row = {
        "state_id": prepared.state_id,
        "ilc_A_end": prepared.ilc_column_count,
        "effective_C2": C2_STAGE,
        "Y_Aend_train_NMSE_dB": float(calculate_nmse(a_y, a_phi @ theta_a)),
        "Y_Aend_B_NMSE_dB": float(calculate_nmse(a_b_y, a_b_phi @ theta_a)),
        "Y_C2_train_NMSE_dB": float(calculate_nmse(c_y, c_phi @ theta_c)),
        "Y_C2_B_NMSE_dB": float(calculate_nmse(c_b_y, c_b_phi @ theta_c)),
        "Y_Aend_theta_l2_norm": diag_a.theta_l2_norm,
        "Y_C2_theta_l2_norm": diag_c.theta_l2_norm,
        "Y_Aend_rank": diag_a.rank_phi,
        "Y_C2_rank": diag_c.rank_phi,
        "n_train_Aend": diag_a.n_train_samples,
        "n_train_C2": diag_c.n_train_samples,
        "A_valid_length": int(a_y.size),
        "B_valid_length_Aend": int(a_b_y.size),
        "C_valid_length_C2": int(c_y.size),
        "B_valid_length_C2": int(c_b_y.size),
        "real_B_valid_length": int(real_b.size),
    }
    numeric = np.asarray(
        [
            row["Y_Aend_train_NMSE_dB"],
            row["Y_Aend_B_NMSE_dB"],
            row["Y_C2_train_NMSE_dB"],
            row["Y_C2_B_NMSE_dB"],
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(numeric)):
        raise RuntimeError(f"state={prepared.state_id}模型指标包含非有限值")
    return row, theta_a, theta_c, real_b


def compute_equal_cnmse_distance_matrix(
    query_fingerprints: np.ndarray,
    candidate_fingerprints: np.ndarray,
) -> np.ndarray:
    """Compute CNMSE for arbitrary equal-ABC fingerprint length."""

    query = np.asarray(query_fingerprints)
    candidate = np.asarray(candidate_fingerprints)
    if (
        query.ndim != 2
        or candidate.ndim != 2
        or query.shape[1] != candidate.shape[1]
        or query.dtype != np.complex128
        or candidate.dtype != np.complex128
    ):
        raise ValueError("等长CNMSE输入必须是同长度complex128二维数组")
    if not np.all(np.isfinite(query)) or not np.all(np.isfinite(candidate)):
        raise ValueError("CNMSE输入不能包含NaN/Inf")
    query_energy = np.sum(np.abs(query) ** 2, axis=1, dtype=np.float64)
    candidate_energy = np.sum(np.abs(candidate) ** 2, axis=1, dtype=np.float64)
    if np.any(query_energy <= 0) or np.any(candidate_energy <= 0):
        raise ValueError("CNMSE输入能量必须为正")
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
        raise RuntimeError("equal ABC CNMSE矩阵出现NaN或+Inf")
    return distance.astype(np.float64, copy=False)


def stable_top1_retrieval(
    distance: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return stable State_R -> State_Q, ranks, ties and ranking order."""

    distance = np.asarray(distance)
    if distance.shape != (STATE_COUNT, STATE_COUNT):
        raise ValueError("retrieval distance必须为(425,425)")
    if np.isnan(distance).any() or np.isposinf(distance).any():
        raise ValueError("retrieval distance不能含NaN或+Inf")
    state_ids = np.arange(STATE_COUNT, dtype=np.int64)
    selected = np.empty(STATE_COUNT, dtype=np.int64)
    selected_distance = np.empty(STATE_COUNT, dtype=np.float64)
    true_rank = np.empty(STATE_COUNT, dtype=np.int64)
    tie_count = np.empty(STATE_COUNT, dtype=np.int64)
    orders = np.empty((STATE_COUNT, STATE_COUNT), dtype=np.int64)
    for state_id in state_ids:
        order = np.lexsort((state_ids, distance[state_id]))
        orders[state_id] = order
        selected[state_id] = int(order[0])
        selected_distance[state_id] = float(distance[state_id, order[0]])
        tie_count[state_id] = int(
            np.count_nonzero(distance[state_id] == distance[state_id, order[0]])
        )
        true_rank[state_id] = int(np.flatnonzero(order == state_id)[0] + 1)
    return selected, selected_distance, true_rank, tie_count, orders


def build_equal_retrieval_results(
    distance: np.ndarray,
    real_b_distance: np.ndarray,
    state_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build retrieval table and Real-B diagnostics for the equal-ABC study."""

    distance_to_ranks(distance)
    selected, selected_distance, true_rank, tie_count, _ = stable_top1_retrieval(distance)
    state_ids = np.arange(STATE_COUNT, dtype=np.int64)
    retrieved = real_b_distance[state_ids, selected]
    exact = selected == state_ids
    shareable = retrieved < DPD_SHAREABLE_THRESHOLD_DB
    lookup = state_frame.set_index("state_id")
    rows: list[dict[str, Any]] = []
    for state_id in state_ids:
        query_id = int(selected[state_id])
        rows.append(
            {
                "State_n_R": int(state_id),
                "State_n_Q": query_id,
                "exact_hit": bool(exact[state_id]),
                "true_state_rank": int(true_rank[state_id]),
                "top1_true_state_hit": bool(true_rank[state_id] <= 1),
                "top3_true_state_hit": bool(true_rank[state_id] <= 3),
                "top5_true_state_hit": bool(true_rank[state_id] <= 5),
                "top10_true_state_hit": bool(true_rank[state_id] <= 10),
                "query_to_selected_LUT_CNMSE_dB": float(selected_distance[state_id]),
                "state_id_delta": query_id - int(state_id),
                "abs_state_id_delta": abs(query_id - int(state_id)),
                "Aend_stage_R": int(lookup.loc[int(state_id), "ilc_A_end"]),
                "Aend_stage_Q": int(lookup.loc[query_id, "ilc_A_end"]),
                "minimum_tie_count": int(tie_count[state_id]),
                "funMng_R": int(lookup.loc[int(state_id), "funMng"]),
                "funAng_R": int(lookup.loc[int(state_id), "funAng"]),
                "secMng_R": int(lookup.loc[int(state_id), "secMng"]),
                "secAng_R": int(lookup.loc[int(state_id), "secAng"]),
                "funMng_Q": int(lookup.loc[query_id, "funMng"]),
                "funAng_Q": int(lookup.loc[query_id, "funAng"]),
                "secMng_Q": int(lookup.loc[query_id, "secMng"]),
                "secAng_Q": int(lookup.loc[query_id, "secAng"]),
                "retrieved_real_B_CNMSE_dB": float(retrieved[state_id]),
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
    finite_nonexact = retrieved[(~exact) & np.isfinite(retrieved)]
    summary = {
        "state_count": STATE_COUNT,
        "candidate_count": STATE_COUNT,
        "candidate_count_per_query": STATE_COUNT,
        "query": "Y-C2",
        "lut": "Y-Aend",
        "self_match_included": True,
        "exact_hit_count": int(exact.sum()),
        "exact_hit_rate": float(exact.mean()),
        "nonexact_count": int((~exact).sum()),
        "nonexact_finite_count": int(finite_nonexact.size),
        "top1_true_state_hit_count": int(np.count_nonzero(true_rank <= 1)),
        "top3_true_state_hit_count": int(np.count_nonzero(true_rank <= 3)),
        "top3_true_state_hit_rate": float(np.mean(true_rank <= 3)),
        "top5_true_state_hit_count": int(np.count_nonzero(true_rank <= 5)),
        "top5_true_state_hit_rate": float(np.mean(true_rank <= 5)),
        "top10_true_state_hit_count": int(np.count_nonzero(true_rank <= 10)),
        "top10_true_state_hit_rate": float(np.mean(true_rank <= 10)),
        "minimum_tie_count_rows": int(np.count_nonzero(tie_count > 1)),
        "DPD_shareable_threshold_dB": DPD_SHAREABLE_THRESHOLD_DB,
        "DPD_shareable_threshold_rule": "Real-B CNMSE < -40 dB",
        "DPD_shareable_count": int(shareable.sum()),
        "DPD_shareable_rate": float(shareable.mean()),
        "failure_count": int((~shareable).sum()),
        "nonexact_shareable_count": int(np.count_nonzero((~exact) & shareable)),
        "nonexact_failure_count": int(np.count_nonzero((~exact) & ~shareable)),
        "nonexact_B_CNMSE_median": float(np.median(finite_nonexact)),
        "nonexact_B_CNMSE_q95": float(np.quantile(finite_nonexact, 0.95)),
        "nonexact_B_CNMSE_min": float(np.min(finite_nonexact)),
        "nonexact_B_CNMSE_max": float(np.max(finite_nonexact)),
    }
    return results, diagnostics, summary


def load_raw_scalar_metrics(
    state_ids: Iterable[int] = range(STATE_COUNT),
    *,
    progress_interval: int = 50,
) -> pd.DataFrame:
    """Read the three saved scalar PA metrics without recomputing from IQ."""

    ids = [int(value) for value in state_ids]
    if ids != list(range(STATE_COUNT)):
        raise ValueError("raw scalar state ids必须严格为0...424")
    rows: list[dict[str, float | int]] = []
    for position, state_id in enumerate(ids, start=1):
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
        if progress_interval > 0 and (position % progress_interval == 0 or position == len(ids)):
            print(f"raw scalar metrics: processed {position} / {len(ids)}", flush=True)
    return pd.DataFrame(rows)


def assign_pa_nonlinearity_quartiles(values: np.ndarray) -> np.ndarray:
    """Q1 is most negative (most linear); Q4 is closest to zero."""

    values = np.asarray(values, dtype=float)
    if values.shape != (STATE_COUNT,) or not np.all(np.isfinite(values)):
        raise ValueError("PA nonlinearity values必须是425个finite值")
    order = np.argsort(values, kind="mergesort")
    labels = np.empty(STATE_COUNT, dtype=object)
    names = np.asarray(["Q1", "Q2", "Q3", "Q4"], dtype=object)
    positions = np.minimum((np.arange(STATE_COUNT) * 4) // STATE_COUNT, 3)
    labels[order] = names[positions]
    return labels


def _safe_corr(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
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


def build_correlation_summary(statewise: pd.DataFrame) -> pd.DataFrame:
    """Return Pearson/Spearman rows for all requested PA/model relationships."""

    rows: list[dict[str, Any]] = []
    model_columns = (
        "Y_Aend_train_NMSE_dB",
        "Y_Aend_B_NMSE_dB",
        "Y_C2_train_NMSE_dB",
        "Y_C2_B_NMSE_dB",
    )
    for x_name, x_column in (
        ("nmse_withoutdpd_dB", "nmse_withoutdpd_dB"),
        ("acpr_withoutdpd_avg_dBc", "acpr_withoutdpd_avg_dBc"),
    ):
        for y_column in model_columns:
            stats = _safe_corr(
                statewise[x_column].to_numpy(dtype=float),
                statewise[y_column].to_numpy(dtype=float),
            )
            rows.append({"x_metric": x_name, "y_metric": y_column, "subset": "all425", **stats})
    finite_nonexact = (~statewise["exact_hit"].to_numpy(dtype=bool)) & np.isfinite(
        statewise["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float)
    )
    for x_column in ("nmse_withoutdpd_dB", "acpr_withoutdpd_avg_dBc", *model_columns):
        stats = _safe_corr(
            statewise.loc[finite_nonexact, x_column].to_numpy(dtype=float),
            statewise.loc[finite_nonexact, "retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float),
        )
        rows.append(
            {
                "x_metric": x_column,
                "y_metric": "retrieved_real_B_CNMSE_dB",
                "subset": "finite_nonexact",
                **stats,
            }
        )
    return pd.DataFrame(rows)


def build_quartile_summary(statewise: pd.DataFrame) -> pd.DataFrame:
    """Summarise PA quartiles, model quality, Real-B quality and failures."""

    quartiles = assign_pa_nonlinearity_quartiles(
        statewise.sort_values("state_id")["nmse_withoutdpd_dB"].to_numpy(dtype=float)
    )
    work = statewise.sort_values("state_id").copy()
    work["pa_nonlinearity_quartile"] = quartiles
    rows: list[dict[str, Any]] = []
    for quartile in ("Q1", "Q2", "Q3", "Q4"):
        group = work.loc[work["pa_nonlinearity_quartile"] == quartile]
        nonexact = group.loc[~group["exact_hit"]]
        finite_nonexact = nonexact.loc[np.isfinite(nonexact["retrieved_real_B_CNMSE_dB"])]
        rows.append(
            {
                "pa_nonlinearity_quartile": quartile,
                "state_count": int(group.shape[0]),
                "nmse_withoutdpd_median_dB": float(group["nmse_withoutdpd_dB"].median()),
                "Y_Aend_train_median_dB": float(group["Y_Aend_train_NMSE_dB"].median()),
                "Y_Aend_B_median_dB": float(group["Y_Aend_B_NMSE_dB"].median()),
                "Y_C2_train_median_dB": float(group["Y_C2_train_NMSE_dB"].median()),
                "Y_C2_B_median_dB": float(group["Y_C2_B_NMSE_dB"].median()),
                "nonexact_count": int(nonexact.shape[0]),
                "nonexact_real_B_median_dB": (
                    float(finite_nonexact["retrieved_real_B_CNMSE_dB"].median())
                    if not finite_nonexact.empty
                    else np.nan
                ),
                "failure_count": int(group["failure"].sum()),
                "failure_rate": float(group["failure"].mean()),
            }
        )
    return pd.DataFrame(rows)


def build_retrieval_class_summary(statewise: pd.DataFrame) -> pd.DataFrame:
    """Compare exact, nonexact-shareable and failure model/PA metrics."""

    exact = statewise["exact_hit"].to_numpy(dtype=bool)
    shareable = statewise["dpd_shareable"].to_numpy(dtype=bool)
    classes = {
        "exact_hit": exact,
        "nonexact_shareable": (~exact) & shareable,
        "failure": ~shareable,
    }
    metrics = (
        "nmse_withoutdpd_dB",
        "acpr_withoutdpd_avg_dBc",
        "Y_Aend_train_NMSE_dB",
        "Y_Aend_B_NMSE_dB",
        "Y_C2_train_NMSE_dB",
        "Y_C2_B_NMSE_dB",
    )
    rows: list[dict[str, Any]] = []
    for class_name, mask in classes.items():
        group = statewise.loc[mask]
        row: dict[str, Any] = {
            "retrieval_class": class_name,
            "state_count": int(group.shape[0]),
        }
        for metric in metrics:
            values = group[metric].to_numpy(dtype=float)
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_q25"] = float(np.quantile(values, 0.25))
            row[f"{metric}_q75"] = float(np.quantile(values, 0.75))
        rows.append(row)
    return pd.DataFrame(rows)


def build_generalization_gap_summary(statewise: pd.DataFrame) -> pd.DataFrame:
    """Summarise A->B, C->B gaps and their asymmetry."""

    work = statewise.copy()
    work["Gap_A_dB"] = work["Y_Aend_B_NMSE_dB"] - work["Y_Aend_train_NMSE_dB"]
    work["Gap_C_dB"] = work["Y_C2_B_NMSE_dB"] - work["Y_C2_train_NMSE_dB"]
    work["Delta_AC_B_dB"] = work["Y_C2_B_NMSE_dB"] - work["Y_Aend_B_NMSE_dB"]
    rows: list[dict[str, Any]] = []
    for metric, values in (
        ("Gap_A_dB", work["Gap_A_dB"].to_numpy(dtype=float)),
        ("Gap_C_dB", work["Gap_C_dB"].to_numpy(dtype=float)),
        ("Delta_AC_B_dB", work["Delta_AC_B_dB"].to_numpy(dtype=float)),
    ):
        rows.append(
            {
                "metric": metric,
                "count": int(values.size),
                "median_dB": float(np.median(values)),
                "mean_dB": float(np.mean(values)),
                "q25_dB": float(np.quantile(values, 0.25)),
                "q75_dB": float(np.quantile(values, 0.75)),
                "q95_dB": float(np.quantile(values, 0.95)),
                "fraction_gt_zero": float(np.mean(values > 0.0)),
            }
        )
    return pd.DataFrame(rows)


__all__ = [
    "C2_STAGE",
    "COMMON_B_INPUT_LENGTH",
    "DPD_SHAREABLE_THRESHOLD_DB",
    "FINGERPRINT_LENGTH",
    "MAX_DELAY",
    "MEMORY",
    "N_COMPLEX_COEFFICIENTS",
    "NUMERIC_TOLERANCE",
    "ORDERS",
    "RIDGE_LAMBDA",
    "SEGMENT_LENGTH",
    "STATE_COUNT",
    "VALID_LENGTH",
    "VALID_LENGTHS",
    "WAVEFORM_LENGTH",
    "EqualABCPair",
    "EqualABCPreparedState",
    "assign_pa_nonlinearity_quartiles",
    "build_correlation_summary",
    "build_equal_abc_partition",
    "build_equal_retrieval_results",
    "build_generalization_gap_summary",
    "build_retrieval_class_summary",
    "build_quartile_summary",
    "calculate_nmse",
    "compute_equal_cnmse_distance_matrix",
    "fit_equal_abc_state",
    "load_raw_scalar_metrics",
    "prepare_equal_abc_state",
    "segment_definition",
    "stable_top1_retrieval",
    "validate_equal_abc_partition",
]
