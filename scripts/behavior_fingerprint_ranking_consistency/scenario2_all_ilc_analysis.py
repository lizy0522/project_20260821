"""
Scenario 2 全 ILC requested n1×n2 行为指纹排序保持性扫描。

本模块复用已冻结的 scenario_2 X-A、Real-B、D_RR 和 R_RR，仅对每个状态实际存在的
ILC 列提取 Y-A-k/Y-C-k Ridge 模型；requested iteration 通过 state-wise saturation
``effective_n = min(requested_n, state_actual_max_n)`` 引用已有模型。
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from behavior_model.basis import build_mp_basis
from behavior_model.config import MAX_DELAY, MP_CONFIG, NUM_COEFFICIENTS
from behavior_model.evaluation import calculate_nmse
from behavior_model.ridge import fit_coefficients_ridge
from data_manager import build_state_table, get_state_info, load_by_id
from signal_segmentation import (
    build_partition_from_xin,
    get_ilc_pair,
    preprocess_full_pair,
)

from .distance_matrix import compute_cnmse_distance_matrix
from .fingerprint_builder import FINGERPRINT_LENGTH
from .ranking import distance_to_ranks, spearman_statistic

STATE_COUNT = 425
B_OWNERSHIP_LENGTH = FINGERPRINT_LENGTH + MAX_DELAY
RIDGE_LAMBDA = 1e-8
MODEL_ROLES = ("Y-A", "Y-C")


@dataclass(frozen=True)
class FrozenScenario2Artifacts:
    """此前 scenario_2 结果中本轮必须冻结复用的对象。"""

    state_ids: np.ndarray
    common_B_input: np.ndarray
    phi_B: np.ndarray
    X_A_fingerprints: np.ndarray
    Y_A_fingerprints: np.ndarray
    Y_C_query_fingerprints: np.ndarray
    real_B_fingerprints: np.ndarray
    D_RR: np.ndarray
    R_RR: np.ndarray
    source_root: Path


@dataclass(frozen=True)
class ActualILCModels:
    """按每状态实际ILC列保存的Y-A/Y-C theta和模型诊断。"""

    theta_Y_A_actual: np.ndarray
    theta_Y_C_actual: np.ndarray
    available_mask: np.ndarray
    ilc_column_counts: np.ndarray
    model_metrics: pd.DataFrame


@dataclass(frozen=True)
class Scenario2AllILCResult:
    """全 ILC 扫描的内存结果和验证信息。"""

    frozen: FrozenScenario2Artifacts
    availability: pd.DataFrame
    availability_summary: pd.DataFrame
    nmax: int
    effective_iteration_map: pd.DataFrame
    effective_iteration_summary: pd.DataFrame
    actual_models: ActualILCModels
    distance_matrices: dict[str, np.ndarray]
    ranking_matrices: dict[str, np.ndarray]
    spearman_long: pd.DataFrame
    spearman_summary: pd.DataFrame
    spearman_by_n1: dict[int, pd.DataFrame]
    validation: dict[str, Any]


def _decode(value: Any) -> str:
    return value.decode("utf-8") if isinstance(value, bytes) else str(value)


def _validate_state_ids(state_ids: np.ndarray) -> np.ndarray:
    values = np.asarray(state_ids)
    if values.shape != (STATE_COUNT,) or not np.issubdtype(values.dtype, np.integer):
        raise ValueError("state_ids必须是长度425的整数数组")
    values = values.astype(np.int64, copy=False)
    if not np.array_equal(values, np.arange(STATE_COUNT, dtype=np.int64)):
        raise ValueError("state_ids必须严格为0...424")
    return values


def _validate_matrix(
    value: np.ndarray,
    shape: tuple[int, int],
    name: str,
    *,
    complex_required: bool = False,
) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape:
        raise ValueError(f"{name} shape错误：{array.shape}")
    if complex_required and array.dtype != np.complex128:
        raise ValueError(f"{name} dtype必须为complex128")
    if complex_required:
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name}包含NaN或Inf")
    elif np.isnan(array).any() or np.isposinf(array).any():
        raise ValueError(f"{name}包含NaN或+Inf")
    return array


def load_frozen_scenario2_artifacts(source_root: Path) -> FrozenScenario2Artifacts:
    """只读加载旧 scenario_2 的公共probe、X/Real指纹和D/R reference。"""

    source_root = Path(source_root)
    fingerprints_path = source_root / "fingerprints.npz"
    distances_path = source_root / "distance_matrices.npz"
    rankings_path = source_root / "ranking_matrices.npz"
    for path in (fingerprints_path, distances_path, rankings_path):
        if not path.is_file():
            raise FileNotFoundError(f"缺少旧scenario_2输入：{path}")
    with np.load(fingerprints_path, allow_pickle=False) as data:
        required = {
            "state_ids",
            "common_B_input",
            "X_A_fingerprints",
            "Y_A_fingerprints",
            "Y_C_query_fingerprints",
            "real_B_fingerprints",
        }
        if not required.issubset(data.files):
            raise ValueError(f"fingerprints.npz缺少字段：{sorted(required - set(data.files))}")
        state_ids = _validate_state_ids(data["state_ids"])
        common_B_input = np.asarray(data["common_B_input"])
        X_A = np.asarray(data["X_A_fingerprints"])
        Y_A = np.asarray(data["Y_A_fingerprints"])
        Y_C = np.asarray(data["Y_C_query_fingerprints"])
        real_B = np.asarray(data["real_B_fingerprints"])
    if common_B_input.shape != (B_OWNERSHIP_LENGTH,) or common_B_input.dtype != np.complex128:
        raise ValueError(f"旧common_B_input必须是complex128(4915,)，实际{common_B_input.shape}")
    if not np.all(np.isfinite(common_B_input)):
        raise ValueError("旧common_B_input包含非有限值")
    for name, value in (
        ("X_A_fingerprints", X_A),
        ("Y_A_fingerprints", Y_A),
        ("Y_C_query_fingerprints", Y_C),
        ("real_B_fingerprints", real_B),
    ):
        _validate_matrix(value, (STATE_COUNT, FINGERPRINT_LENGTH), name, complex_required=True)

    with np.load(distances_path, allow_pickle=False) as data:
        if not {"state_ids", "D_RR"}.issubset(data.files):
            raise ValueError("distance_matrices.npz缺少state_ids或D_RR")
        _validate_state_ids(data["state_ids"])
        D_RR = _validate_matrix(data["D_RR"], (STATE_COUNT, STATE_COUNT), "D_RR")
    with np.load(rankings_path, allow_pickle=False) as data:
        if not {"state_ids", "R_RR"}.issubset(data.files):
            raise ValueError("ranking_matrices.npz缺少state_ids或R_RR")
        _validate_state_ids(data["state_ids"])
        R_RR = _validate_matrix(data["R_RR"], (STATE_COUNT, STATE_COUNT), "R_RR")
    if not np.all(np.isfinite(R_RR)):
        raise ValueError("旧R_RR包含非有限值")
    phi_B = build_mp_basis(common_B_input, MP_CONFIG["orders"], MP_CONFIG["memory_depth"])
    if phi_B.shape != (FINGERPRINT_LENGTH, NUM_COEFFICIENTS):
        raise ValueError(f"公共Phi_B shape错误：{phi_B.shape}")
    return FrozenScenario2Artifacts(
        state_ids=state_ids,
        common_B_input=common_B_input,
        phi_B=phi_B,
        X_A_fingerprints=X_A,
        Y_A_fingerprints=Y_A,
        Y_C_query_fingerprints=Y_C,
        real_B_fingerprints=real_B,
        D_RR=D_RR,
        R_RR=R_RR,
        source_root=source_root,
    )


def _state_frame_with_counts(ilc_counts: dict[int, int]) -> pd.DataFrame:
    rows = []
    state_table = build_state_table()
    for row in state_table:
        state_id = int(row["state_id"])
        rows.append({**row, "ilc_column_count": int(ilc_counts[state_id])})
    return pd.DataFrame(rows)


def scan_ilc_availability(*, progress_interval: int = 50) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    """读取所有状态的ILC输入/输出列数并动态确定Nmax。"""

    counts: dict[int, int] = {}
    for position, state_id in enumerate(range(STATE_COUNT), start=1):
        data = load_by_id(state_id)
        input_history = np.asarray(data["xin_pd_ori_ilc"])
        output_history = np.asarray(data["yout_withdpd_ori_ilc"])
        if input_history.ndim != 2 or output_history.ndim != 2:
            raise ValueError(f"state_id={state_id} ILC输入/输出必须为二维")
        if input_history.shape != output_history.shape:
            raise ValueError(
                f"state_id={state_id} ILC输入输出列数不一致："
                f"{input_history.shape} != {output_history.shape}"
            )
        if input_history.shape[1] < 1:
            raise ValueError(f"state_id={state_id} ILC至少需要一列")
        if not np.iscomplexobj(input_history) or not np.iscomplexobj(output_history):
            raise ValueError(f"state_id={state_id} ILC输入/输出必须为复数")
        if not np.all(np.isfinite(input_history)) or not np.all(np.isfinite(output_history)):
            raise ValueError(f"state_id={state_id} ILC输入/输出包含非有限值")
        counts[state_id] = int(input_history.shape[1])
        if progress_interval > 0 and (position % progress_interval == 0 or position == STATE_COUNT):
            print(f"availability: processed {position} / {STATE_COUNT} states")
    availability = _state_frame_with_counts(counts)
    nmax = int(availability["ilc_column_count"].max())
    grouped = (
        availability.groupby("ilc_column_count", as_index=False)
        .size()
        .rename(columns={"ilc_column_count": "actual_ilc_column_count", "size": "state_count"})
    )
    grouped["fraction"] = grouped["state_count"] / STATE_COUNT
    return availability, grouped, nmax


def effective_iteration(requested_n: Integral, actual_max_n: Integral) -> tuple[int, int, bool]:
    """返回effective_n、Python零基index和是否发生state-wise saturation。"""

    if not isinstance(requested_n, Integral) or isinstance(requested_n, bool):
        raise TypeError("requested_n必须是整数")
    if not isinstance(actual_max_n, Integral) or isinstance(actual_max_n, bool):
        raise TypeError("actual_max_n必须是整数")
    requested = int(requested_n)
    actual = int(actual_max_n)
    if requested < 1 or actual < 1:
        raise ValueError("requested_n和actual_max_n必须大于0")
    effective = min(requested, actual)
    return effective, effective - 1, requested > actual


def build_effective_iteration_map(availability: pd.DataFrame, nmax: int) -> pd.DataFrame:
    """生成425×Nmax的requested→effective映射。"""

    if nmax < 1:
        raise ValueError("nmax必须大于0")
    rows: list[dict[str, Any]] = []
    for item in availability.itertuples(index=False):
        for requested_n in range(1, nmax + 1):
            effective_n, python_index, saturated = effective_iteration(
                requested_n,
                int(item.ilc_column_count),
            )
            rows.append({
                "state_id": int(item.state_id),
                "funMng": int(item.funMng),
                "funAng": int(item.funAng),
                "secMng": int(item.secMng),
                "secAng": int(item.secAng),
                "Vm": float(item.Vm),
                "Pin": float(item.Pin),
                "requested_n": requested_n,
                "actual_max_n": int(item.ilc_column_count),
                "effective_n": effective_n,
                "python_index": python_index,
                "is_saturated": saturated,
            })
    return pd.DataFrame(rows)


def build_effective_iteration_summary(
    effective_map: pd.DataFrame,
    nmax: int,
) -> pd.DataFrame:
    """统计每个requested_n由哪些effective_n组成。"""

    rows: list[dict[str, Any]] = []
    for requested_n in range(1, nmax + 1):
        group = effective_map[effective_map["requested_n"] == requested_n]
        for effective_n in range(1, nmax + 1):
            selected = group[group["effective_n"] == effective_n]
            count = int(selected.shape[0])
            rows.append({
                "requested_n": requested_n,
                "effective_n": effective_n,
                "state_count": count,
                "fraction": count / STATE_COUNT,
                "saturated_state_count": int(selected["is_saturated"].sum()),
            })
    return pd.DataFrame(rows)


def _fit_actual_model(
    *,
    state_id: int,
    actual_n: int,
    model_role: str,
    train_segment: Any,
    b_segment: Any,
    input_peak_normalization_factor: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    phi_train = build_mp_basis(train_segment.input, MP_CONFIG["orders"], MP_CONFIG["memory_depth"])
    y_train = np.asarray(train_segment.output[MAX_DELAY:])
    phi_b = build_mp_basis(b_segment.input, MP_CONFIG["orders"], MP_CONFIG["memory_depth"])
    y_b = np.asarray(b_segment.output[MAX_DELAY:])
    theta, diagnostics = fit_coefficients_ridge(phi_train, y_train, RIDGE_LAMBDA)
    train_prediction = phi_train @ theta
    b_prediction = phi_b @ theta
    row = {
        "state_id": state_id,
        **get_state_info(state_id),
        "actual_ilc_n": actual_n,
        "model_role": model_role,
        "train_nmse_db": calculate_nmse(y_train, train_prediction),
        "B_generalization_nmse_db": calculate_nmse(y_b, b_prediction),
        "theta_norm": diagnostics.theta_l2_norm,
        "rank": diagnostics.rank_phi,
        "rank_augmented": diagnostics.rank_augmented,
        "condition_number_phi": diagnostics.condition_number_phi,
        "condition_number_augmented": diagnostics.condition_number_augmented,
        "n_train": diagnostics.n_train_samples,
        "coefficient_count": diagnostics.coefficient_count,
        "input_peak_normalization_factor": input_peak_normalization_factor,
        "train_source": f"ILC{actual_n}.{model_role[-1]}",
        "B_generalization_source": f"ILC{actual_n}.B",
        "ridge_lambda": RIDGE_LAMBDA,
        "solver_path": diagnostics.solver_path,
    }
    row["train_NMSE_dB"] = row["train_nmse_db"]
    row["B_generalization_NMSE_dB"] = row["B_generalization_nmse_db"]
    return theta, row


def extract_actual_ilc_models(
    availability: pd.DataFrame,
    nmax: int,
    *,
    progress_interval: int = 25,
) -> ActualILCModels:
    """每状态每个真实ILC列只做一次canonical，并同时提取Y-A-k/Y-C-k。"""

    theta_y_a = np.full(
        (STATE_COUNT, nmax, NUM_COEFFICIENTS),
        np.nan + 0j,
        dtype=np.complex128,
    )
    theta_y_c = np.full_like(theta_y_a, np.nan + 0j)
    available_mask = np.zeros((STATE_COUNT, nmax), dtype=bool)
    counts = availability.sort_values("state_id")["ilc_column_count"].to_numpy(dtype=np.int64)
    metric_rows: list[dict[str, Any]] = []
    sorted_availability = availability.sort_values("state_id")
    for position, item in enumerate(
        sorted_availability.itertuples(index=False),
        start=1,
    ):
        state_id = int(item.state_id)
        data = load_by_id(state_id)
        input_history = np.asarray(data["xin_pd_ori_ilc"])
        if input_history.shape != (24576, int(item.ilc_column_count)):
            raise ValueError(f"state_id={state_id} ILC输入shape与availability不一致")
        partition = build_partition_from_xin(np.asarray(data["xin"]))
        for actual_n in range(1, int(item.ilc_column_count) + 1):
            pair = get_ilc_pair(data, actual_n - 1)
            canonical = preprocess_full_pair(
                pair.input_full,
                pair.output_raw_full,
                partition,
                pair_type="ILC",
                iteration_index=actual_n - 1,
                input_peak_normalization_factor=pair.input_peak_normalization_factor,
            )
            theta_a, row_a = _fit_actual_model(
                state_id=state_id,
                actual_n=actual_n,
                model_role="Y-A",
                train_segment=canonical["A"],
                b_segment=canonical["B"],
                input_peak_normalization_factor=float(pair.input_peak_normalization_factor),
            )
            theta_c, row_c = _fit_actual_model(
                state_id=state_id,
                actual_n=actual_n,
                model_role="Y-C",
                train_segment=canonical["C"],
                b_segment=canonical["B"],
                input_peak_normalization_factor=float(pair.input_peak_normalization_factor),
            )
            slot = actual_n - 1
            theta_y_a[state_id, slot, :] = theta_a
            theta_y_c[state_id, slot, :] = theta_c
            available_mask[state_id, slot] = True
            metric_rows.extend((row_a, row_c))
        if progress_interval > 0 and (position % progress_interval == 0 or position == STATE_COUNT):
            print(f"actual models: processed {position} / {STATE_COUNT} states")

    model_metrics = pd.DataFrame(metric_rows)
    available_theta = np.concatenate(
        [theta_y_a[available_mask], theta_y_c[available_mask]],
        axis=0,
    )
    if not np.all(np.isfinite(available_theta)):
        raise RuntimeError("实际可用Y-A/Y-C theta包含非有限值")
    rank_deficient = model_metrics[model_metrics["rank"] < NUM_COEFFICIENTS]
    if not rank_deficient.empty:
        columns = ["state_id", "actual_ilc_n", "model_role", "rank"]
        details = rank_deficient[columns].to_dict("records")
        raise RuntimeError(f"实际Y模型存在rank deficient：{details}")
    expected_available = int(np.sum(counts))
    if int(np.count_nonzero(available_mask)) != expected_available:
        raise RuntimeError("available_mask与实际ILC列数不一致")
    return ActualILCModels(
        theta_Y_A_actual=theta_y_a,
        theta_Y_C_actual=theta_y_c,
        available_mask=available_mask,
        ilc_column_counts=counts,
        model_metrics=model_metrics,
    )


def _fingerprints_for_requested_iteration(
    theta_actual: np.ndarray,
    ilc_column_counts: np.ndarray,
    phi_B: np.ndarray,
    requested_n: int,
) -> np.ndarray:
    """按每个状态effective_n从实际theta中生成 requested 指纹。"""

    nmax = theta_actual.shape[1]
    if requested_n < 1 or requested_n > nmax:
        raise ValueError("requested_n超出全局Nmax")
    effective_indices = np.minimum(requested_n, ilc_column_counts) - 1
    fingerprints = np.empty((STATE_COUNT, FINGERPRINT_LENGTH), dtype=np.complex128)
    for actual_index in range(nmax):
        state_indices = np.flatnonzero(effective_indices == actual_index)
        if state_indices.size == 0:
            continue
        selected_theta = theta_actual[state_indices, actual_index, :]
        if not np.all(np.isfinite(selected_theta)):
            raise RuntimeError("requested effective slot引用了不可用theta")
        fingerprints[state_indices, :] = (phi_B @ selected_theta.T).T
    if not np.all(np.isfinite(fingerprints)):
        raise RuntimeError("requested指纹包含非有限值")
    return fingerprints


def _spearman_for_ranking_rows(reference: np.ndarray, target: np.ndarray) -> np.ndarray:
    if reference.shape != (STATE_COUNT, STATE_COUNT) or target.shape != reference.shape:
        raise ValueError("Spearman排名矩阵必须都是(425,425)")
    values = np.empty(STATE_COUNT, dtype=np.float64)
    for state_id in range(STATE_COUNT):
        values[state_id] = spearman_statistic(reference[state_id], target[state_id])
    return values


def _spearman_summary(long_table: pd.DataFrame, nmax: int) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    lut_order = {"X-A": 0, "Y-A": 1, "Real-B": 2}
    for requested_n1 in range(1, nmax + 1):
        n1_group = long_table[long_table["requested_n1"] == requested_n1]
        for lut_type in ("X-A", "Y-A", "Real-B"):
            lut_group = n1_group[n1_group["lut_type"] == lut_type]
            n2_values = [None] if lut_type != "Y-A" else list(range(1, nmax + 1))
            for requested_n2 in n2_values:
                group = (
                    lut_group[lut_group["requested_n2"].isna()]
                    if requested_n2 is None
                    else lut_group[lut_group["requested_n2"] == requested_n2]
                )
                values = group["spearman"].to_numpy(dtype=float)
                if values.size != STATE_COUNT or not np.all(np.isfinite(values)):
                    raise RuntimeError("每组Spearman必须恰好有425个finite值")
                rows.append({
                    "requested_n1": requested_n1,
                    "lut_type": lut_type,
                    "requested_n2": requested_n2,
                    "count": int(values.size),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values, ddof=1)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                    "lut_order": lut_order[lut_type],
                })
    summary = pd.DataFrame(rows)
    return summary.sort_values(
        ["requested_n1", "lut_order", "requested_n2"],
        na_position="first",
    ).drop(columns="lut_order").reset_index(drop=True)


def run_requested_scan(
    frozen: FrozenScenario2Artifacts,
    actual_models: ActualILCModels,
    nmax: int,
    *,
    progress_interval: int = 1,
) -> tuple[
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    pd.DataFrame,
    pd.DataFrame,
    dict[int, pd.DataFrame],
]:
    """执行所有requested n1×n2并缓存距离、排名和Spearman结果。"""

    requested_y_a = {
        n2: _fingerprints_for_requested_iteration(
            actual_models.theta_Y_A_actual,
            actual_models.ilc_column_counts,
            frozen.phi_B,
            n2,
        )
        for n2 in range(1, nmax + 1)
    }
    distances: dict[str, np.ndarray] = {"D_RR": frozen.D_RR.copy()}
    rankings: dict[str, np.ndarray] = {"R_RR": frozen.R_RR.copy()}
    long_rows: list[dict[str, Any]] = []
    wide_by_n1: dict[int, pd.DataFrame] = {}
    state_table = pd.DataFrame(build_state_table()).sort_values("state_id")
    for n1 in range(1, nmax + 1):
        query = _fingerprints_for_requested_iteration(
            actual_models.theta_Y_C_actual,
            actual_models.ilc_column_counts,
            frozen.phi_B,
            n1,
        )
        effective_query = np.minimum(n1, actual_models.ilc_column_counts)
        D_CX = compute_cnmse_distance_matrix(query, frozen.X_A_fingerprints)
        D_CR = compute_cnmse_distance_matrix(query, frozen.real_B_fingerprints)
        R_CX = distance_to_ranks(D_CX)
        R_CR = distance_to_ranks(D_CR)
        distances[f"D_CX_n1_{n1:02d}"] = D_CX
        distances[f"D_CR_n1_{n1:02d}"] = D_CR
        rankings[f"R_CX_n1_{n1:02d}"] = R_CX
        rankings[f"R_CR_n1_{n1:02d}"] = R_CR
        spearman_x = _spearman_for_ranking_rows(frozen.R_RR, R_CX)
        spearman_real = _spearman_for_ranking_rows(frozen.R_RR, R_CR)
        wide = state_table.loc[
            :, ["state_id", "funMng", "funAng", "secMng", "secAng", "Vm", "Pin"]
        ].copy()
        wide["query_effective_n1"] = effective_query.astype(int)
        wide["spearman_X_A"] = spearman_x
        wide["spearman_Real_B"] = spearman_real
        for state_id in range(STATE_COUNT):
            info = get_state_info(state_id)
            long_rows.append({
                "state_id": state_id,
                **info,
                "requested_n1": n1,
                "query_effective_n1": int(effective_query[state_id]),
                "lut_type": "X-A",
                "requested_n2": np.nan,
                "spearman": float(spearman_x[state_id]),
            })
            long_rows.append({
                "state_id": state_id,
                **info,
                "requested_n1": n1,
                "query_effective_n1": int(effective_query[state_id]),
                "lut_type": "Real-B",
                "requested_n2": np.nan,
                "spearman": float(spearman_real[state_id]),
            })

        for n2 in range(1, nmax + 1):
            D_CY = compute_cnmse_distance_matrix(query, requested_y_a[n2])
            R_CY = distance_to_ranks(D_CY)
            distances[f"D_CY_n1_{n1:02d}_n2_{n2:02d}"] = D_CY
            rankings[f"R_CY_n1_{n1:02d}_n2_{n2:02d}"] = R_CY
            spearman_y = _spearman_for_ranking_rows(frozen.R_RR, R_CY)
            wide[f"spearman_Y_A_n2_{n2:02d}"] = spearman_y
            for state_id in range(STATE_COUNT):
                info = get_state_info(state_id)
                long_rows.append({
                    "state_id": state_id,
                    **info,
                    "requested_n1": n1,
                    "query_effective_n1": int(effective_query[state_id]),
                    "lut_type": "Y-A",
                    "requested_n2": n2,
                    "spearman": float(spearman_y[state_id]),
                })
        wide_by_n1[n1] = wide
        if progress_interval > 0:
            print(f"requested scan: completed n1={n1} / {nmax}")

    long_table = pd.DataFrame(long_rows)
    expected_rows = STATE_COUNT * nmax * (nmax + 2)
    if long_table.shape[0] != expected_rows:
        raise RuntimeError(f"Spearman长表行数错误：{long_table.shape[0]} != {expected_rows}")
    summary = _spearman_summary(long_table, nmax)
    return distances, rankings, long_table, summary, wide_by_n1


def _matrix_validity(matrices: dict[str, np.ndarray]) -> bool:
    return all(
        matrix.shape == (STATE_COUNT, STATE_COUNT)
        and not np.isnan(matrix).any()
        and not np.isposinf(matrix).any()
        for matrix in matrices.values()
    )


def run_scenario2_all_ilc_analysis(
    source_root: Path,
    *,
    progress_interval: int = 25,
) -> Scenario2AllILCResult:
    """执行全 ILC 模型提取、ILC1 gate 和 requested n1×n2 扫描。"""

    frozen = load_frozen_scenario2_artifacts(source_root)
    availability, availability_summary, nmax = scan_ilc_availability(
        progress_interval=progress_interval
    )
    effective_map = build_effective_iteration_map(availability, nmax)
    effective_summary = build_effective_iteration_summary(effective_map, nmax)
    actual_models = extract_actual_ilc_models(
        availability,
        nmax,
        progress_interval=progress_interval,
    )

    Y_A_1 = (frozen.phi_B @ actual_models.theta_Y_A_actual[:, 0, :].T).T
    Y_C_1 = (frozen.phi_B @ actual_models.theta_Y_C_actual[:, 0, :].T).T
    ya_error = float(np.max(np.abs(Y_A_1 - frozen.Y_A_fingerprints)))
    yc_error = float(np.max(np.abs(Y_C_1 - frozen.Y_C_query_fingerprints)))
    if ya_error > 1e-10 or yc_error > 1e-10:
        raise RuntimeError(
            f"ILC1回归失败：Y-A max_error={ya_error}, Y-C max_error={yc_error}"
        )

    distances, rankings, spearman_long, spearman_summary, wide_by_n1 = run_requested_scan(
        frozen,
        actual_models,
        nmax,
    )
    available_theta_count = int(np.count_nonzero(actual_models.available_mask))
    rank_deficient_count = int(
        np.count_nonzero(actual_models.model_metrics["rank"].to_numpy() < NUM_COEFFICIENTS)
    )
    expected_spearman = STATE_COUNT * nmax * (nmax + 2)
    diagonal_index = np.arange(STATE_COUNT)
    d_rr_diagonal = frozen.D_RR[diagonal_index, diagonal_index]
    d_rr_row_min = np.min(frozen.D_RR, axis=1)
    r_rr_diagonal = frozen.R_RR[diagonal_index, diagonal_index]
    validation: dict[str, Any] = {
        "scenario": 2,
        "study": "all ILC iteration ranking consistency scan",
        "state_count": STATE_COUNT,
        "global_ilc_max": nmax,
        "requested_n1_values": list(range(1, nmax + 1)),
        "requested_n2_values": list(range(1, nmax + 1)),
        "statewise_iteration_capping": True,
        "effective_iteration_rule": "min(requested_n, state_max_n)",
        "ilc_input_output_columns_equal": True,
        "ilc_minimum_column_count": int(availability["ilc_column_count"].min()),
        "ilc_availability_summary": {
            str(int(row.actual_ilc_column_count)): int(row.state_count)
            for row in availability_summary.itertuples(index=False)
        },
        "self_match_included": True,
        "candidate_count_per_query": STATE_COUNT,
        "ridge_lambda": RIDGE_LAMBDA,
        "ridge_lambda_rescanned": False,
        "mp_structure_changed": False,
        "canonical_changed": False,
        "common_B_source": "frozen scenario_2",
        "X_A_LUT_source": "frozen scenario_2",
        "Real_B_LUT_source": "frozen scenario_2",
        "reference_source": "frozen Real-B -> Real-B",
        "fingerprint_length": FINGERPRINT_LENGTH,
        "common_B_input_shape": list(frozen.common_B_input.shape),
        "phi_B_shape": list(frozen.phi_B.shape),
        "actual_ilc_available_slot_count": available_theta_count,
        "Y_A_actual_model_count": int(
            np.count_nonzero(actual_models.available_mask)
        ),
        "Y_C_actual_model_count": int(
            np.count_nonzero(actual_models.available_mask)
        ),
        "all_available_theta_finite": bool(
            np.all(np.isfinite(actual_models.theta_Y_A_actual[actual_models.available_mask]))
            and np.all(np.isfinite(actual_models.theta_Y_C_actual[actual_models.available_mask]))
        ),
        "rank_deficient_available_models": rank_deficient_count,
        "ILC1_regression_pass": True,
        "ILC1_Y_A_max_abs_error": ya_error,
        "ILC1_Y_C_max_abs_error": yc_error,
        "all_distance_matrices_valid": _matrix_validity(distances),
        "all_ranking_matrices_valid": _matrix_validity(rankings),
        "D_RR_self_row_min_count": int(np.count_nonzero(d_rr_diagonal <= d_rr_row_min)),
        "R_RR_self_rank1_count": int(np.count_nonzero(r_rr_diagonal == 1.0)),
        "D_RR_negative_infinity_count": int(np.count_nonzero(np.isneginf(frozen.D_RR))),
        "distance_matrix_count": len(distances),
        "ranking_matrix_count": len(rankings),
        "spearman_total_count": int(spearman_long.shape[0]),
        "expected_spearman_total_count": expected_spearman,
        "spearman_X_A_count": int(
            np.count_nonzero(spearman_long["lut_type"] == "X-A")
        ),
        "spearman_Y_A_count": int(
            np.count_nonzero(spearman_long["lut_type"] == "Y-A")
        ),
        "spearman_Real_B_count": int(
            np.count_nonzero(spearman_long["lut_type"] == "Real-B")
        ),
        "all_spearman_finite": bool(np.all(np.isfinite(spearman_long["spearman"]))),
        "spearman_range_min": float(spearman_long["spearman"].min()),
        "spearman_range_max": float(spearman_long["spearman"].max()),
        "formal_png_count": 2 * nmax,
        "figure_y_axis_mode": "per-figure actual minimum-to-maximum",
        "model_roles": list(MODEL_ROLES),
    }
    if not validation["all_available_theta_finite"] or rank_deficient_count != 0:
        raise RuntimeError("实际可用theta合法性验证失败")
    if (
        not validation["all_distance_matrices_valid"]
        or not validation["all_ranking_matrices_valid"]
    ):
        raise RuntimeError("距离或排名矩阵验证失败")
    if not validation["all_spearman_finite"]:
        raise RuntimeError("Spearman存在非有限值")
    return Scenario2AllILCResult(
        frozen=frozen,
        availability=availability,
        availability_summary=availability_summary,
        nmax=nmax,
        effective_iteration_map=effective_map,
        effective_iteration_summary=effective_summary,
        actual_models=actual_models,
        distance_matrices=distances,
        ranking_matrices=rankings,
        spearman_long=spearman_long,
        spearman_summary=spearman_summary,
        spearman_by_n1=wide_by_n1,
        validation=validation,
    )


__all__ = [
    "ActualILCModels",
    "B_OWNERSHIP_LENGTH",
    "FrozenScenario2Artifacts",
    "RIDGE_LAMBDA",
    "Scenario2AllILCResult",
    "build_effective_iteration_map",
    "build_effective_iteration_summary",
    "effective_iteration",
    "extract_actual_ilc_models",
    "load_frozen_scenario2_artifacts",
    "run_scenario2_all_ilc_analysis",
    "run_requested_scan",
    "scan_ilc_availability",
]
