"""Scenario 2 多阶段 Y-A LUT 指纹融合与跨在线 C 阶段鲁棒性分析。

本模块只读取已经冻结的 Scenario 2 reference 和 ``scenario_2_all_ilc`` 的实际
Y-A/Y-C Ridge theta。固定 LUT 候选为 Y-A1、Y-A2 以及两个 common-B 响应逐复数
采样点的算术平均 Y-A12-Mean；Real-B 只作为 ``R_RR`` 的状态排序 reference，
不作为本轮 LUT 候选。模块不读取 raw waveform，也不重新训练模型。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .distance_matrix import compute_cnmse_distance_matrix
from .fingerprint_builder import FINGERPRINT_LENGTH
from .ranking import distance_to_ranks, spearman_statistic
from .scenario2_all_ilc_analysis import load_frozen_scenario2_artifacts

STATE_COUNT = 425
NMAX = 5
NUM_COEFFICIENTS = 10
RIDGE_LAMBDA = 1e-8
LUT_TYPES = ("Y-A1", "Y-A2", "Y-A12-Mean")
LUT_SUFFIXES = {"Y-A1": "Y_A1", "Y-A2": "Y_A2", "Y-A12-Mean": "Y_A12_mean"}
STATE_COLUMNS = ("state_id", "funMng", "funAng", "secMng", "secAng", "Vm", "Pin")
EFFECTIVE_MAP_COLUMNS = (
    *STATE_COLUMNS,
    "requested_n",
    "actual_max_n",
    "effective_n",
    "python_index",
    "is_saturated",
)
SPEARMAN_COLUMNS = (
    *STATE_COLUMNS,
    "requested_C_n",
    "effective_C_n",
    "lut_fingerprint_type",
    "spearman",
)


@dataclass(frozen=True)
class LoadedAllILC:
    """冻结的 all-ILC theta、可用性和 state-wise saturation 映射。"""

    source_root: Path
    state_ids: np.ndarray
    theta_Y_A_actual: np.ndarray
    theta_Y_C_actual: np.ndarray
    available_mask: np.ndarray
    ilc_column_counts: np.ndarray
    effective_iteration_map: pd.DataFrame


@dataclass(frozen=True)
class YLUTFusionResult:
    """分析所需数组、表格和验证信息。"""

    loaded: LoadedAllILC
    state_ids: np.ndarray
    common_B_input: np.ndarray
    phi_B: np.ndarray
    lut_fingerprints: dict[str, np.ndarray]
    query_fingerprints: dict[int, np.ndarray]
    effective_C: np.ndarray
    distance_matrices: dict[str, np.ndarray]
    ranking_matrices: dict[str, np.ndarray]
    spearman_by_combination: dict[tuple[int, str], np.ndarray]
    spearman_long: pd.DataFrame
    spearman_summary: pd.DataFrame
    lut_robustness_summary: pd.DataFrame
    statewise_worst_over_C: pd.DataFrame
    fusion_paired_comparison: pd.DataFrame
    fusion_paired_summary: pd.DataFrame
    baseline_regression: dict[str, Any]
    validation: dict[str, Any]


def _validate_state_ids(values: np.ndarray, name: str = "state_ids") -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (STATE_COUNT,) or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name}必须是长度425的整数数组")
    array = array.astype(np.int64, copy=False)
    if not np.array_equal(array, np.arange(STATE_COUNT, dtype=np.int64)):
        raise ValueError(f"{name}必须严格为0...424")
    return array


def _validate_complex_array(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype != np.complex128:
        raise ValueError(f"{name}必须是complex128且shape={shape}，实际{array.shape}/{array.dtype}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name}包含NaN或Inf")
    return array


def _load_effective_iteration_map(path: Path, counts: np.ndarray) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"缺少冻结effective_iteration_map：{path}")
    frame = pd.read_csv(path)
    missing = [column for column in EFFECTIVE_MAP_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"effective_iteration_map缺少字段：{missing}")
    frame = frame.loc[:, list(EFFECTIVE_MAP_COLUMNS)].copy()
    int_columns = (
        "state_id",
        "funMng",
        "funAng",
        "secMng",
        "secAng",
        "requested_n",
        "actual_max_n",
        "effective_n",
        "python_index",
    )
    for column in int_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(int)
    frame["Vm"] = pd.to_numeric(frame["Vm"], errors="raise").astype(float)
    frame["Pin"] = pd.to_numeric(frame["Pin"], errors="raise").astype(float)
    frame["is_saturated"] = frame["is_saturated"].astype(bool)
    if frame.shape[0] != STATE_COUNT * NMAX:
        raise ValueError(f"effective_iteration_map行数错误：{frame.shape[0]}")
    for state_id in range(STATE_COUNT):
        state_rows = frame[frame["state_id"] == state_id]
        if state_rows.shape[0] != NMAX:
            raise ValueError(f"state_id={state_id}的effective map不是5行")
        expected_count = int(counts[state_id])
        expected = np.minimum(np.arange(1, NMAX + 1), expected_count)
        observed = state_rows.sort_values("requested_n")["effective_n"].to_numpy(dtype=int)
        if not np.array_equal(observed, expected):
            raise ValueError(f"state_id={state_id}的state-wise saturation映射错误")
        if not np.array_equal(
            state_rows.sort_values("requested_n")["python_index"].to_numpy(dtype=int),
            expected - 1,
        ):
            raise ValueError(f"state_id={state_id}的python_index错误")
        if not np.array_equal(
            state_rows.sort_values("requested_n")["is_saturated"].to_numpy(dtype=bool),
            np.arange(1, NMAX + 1) > expected_count,
        ):
            raise ValueError(f"state_id={state_id}的saturation标志错误")
    return frame.sort_values(["state_id", "requested_n"]).reset_index(drop=True)


def load_all_ilc_theta(source_root: Path) -> LoadedAllILC:
    """读取已完成的 all-ILC theta，不重新训练任何模型。"""

    source_root = Path(source_root)
    theta_path = source_root / "all_ilc_theta.npz"
    if not theta_path.is_file():
        raise FileNotFoundError(f"缺少all_ilc_theta.npz：{theta_path}")
    with np.load(theta_path, allow_pickle=False) as data:
        required = {
            "state_ids",
            "theta_Y_A_actual",
            "theta_Y_C_actual",
            "available_mask",
            "ilc_column_counts",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"all_ilc_theta.npz缺少字段：{missing}")
        state_ids = _validate_state_ids(data["state_ids"])
        theta_y_a = np.asarray(data["theta_Y_A_actual"])
        theta_y_c = np.asarray(data["theta_Y_C_actual"])
        mask = np.asarray(data["available_mask"])
        counts = np.asarray(data["ilc_column_counts"])
    theta_shape = (STATE_COUNT, NMAX, NUM_COEFFICIENTS)
    if theta_y_a.shape != theta_shape or theta_y_c.shape != theta_shape:
        raise ValueError(f"Y-A/Y-C theta必须是{theta_shape}")
    if theta_y_a.dtype != np.complex128 or theta_y_c.dtype != np.complex128:
        raise ValueError("Y-A/Y-C theta必须是complex128")
    if mask.shape != (STATE_COUNT, NMAX) or mask.dtype != np.bool_:
        raise ValueError("available_mask必须是bool且shape=(425,5)")
    if counts.shape != (STATE_COUNT,) or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("ilc_column_counts必须是长度425的整数数组")
    counts = counts.astype(np.int64, copy=False)
    if np.any(counts < 1) or np.any(counts > NMAX):
        raise ValueError("ilc_column_counts必须位于1...5")
    expected_mask = np.arange(NMAX)[None, :] < counts[:, None]
    if not np.array_equal(mask, expected_mask):
        raise ValueError("available_mask与ilc_column_counts不一致")
    if not np.all(np.isfinite(theta_y_a[mask])) or not np.all(np.isfinite(theta_y_c[mask])):
        raise ValueError("可用Y-A/Y-C theta包含非有限值")
    if np.any(~np.isfinite(theta_y_a[~mask])) or np.any(~np.isfinite(theta_y_c[~mask])):
        # Missing slots are expected to be NaN sentinels, not usable model data.
        if not (np.all(np.isnan(theta_y_a[~mask])) and np.all(np.isnan(theta_y_c[~mask]))):
            raise ValueError("不可用theta槽位必须全部为NaN")
    effective_map = _load_effective_iteration_map(
        source_root / "effective_iteration_map.csv", counts
    )
    return LoadedAllILC(
        source_root=source_root,
        state_ids=state_ids,
        theta_Y_A_actual=theta_y_a,
        theta_Y_C_actual=theta_y_c,
        available_mask=mask,
        ilc_column_counts=counts,
        effective_iteration_map=effective_map,
    )


def build_fingerprint_from_theta(theta: np.ndarray, phi_B: np.ndarray) -> np.ndarray:
    """用一次冻结的公共 B basis 生成逐状态复数行为指纹。"""

    coefficients = _validate_complex_array(theta, (STATE_COUNT, NUM_COEFFICIENTS), "theta")
    basis = np.asarray(phi_B)
    if basis.shape != (FINGERPRINT_LENGTH, NUM_COEFFICIENTS):
        raise ValueError(f"phi_B必须是({FINGERPRINT_LENGTH},{NUM_COEFFICIENTS})")
    if basis.dtype != np.complex128 or not np.all(np.isfinite(basis)):
        raise ValueError("phi_B必须是finite complex128")
    fingerprints = (basis @ coefficients.T).T.astype(np.complex128, copy=False)
    if fingerprints.shape != (STATE_COUNT, FINGERPRINT_LENGTH) or not np.all(
        np.isfinite(fingerprints)
    ):
        raise RuntimeError("生成的指纹shape或finite性错误")
    return fingerprints


def build_y_lut_fingerprints(
    loaded: LoadedAllILC,
    phi_B: np.ndarray,
) -> dict[str, np.ndarray]:
    """构造 Y-A1、Y-A2 以及 fingerprint-level Y-A12-Mean。"""

    if not np.all(loaded.available_mask[:, 0]) or not np.all(loaded.available_mask[:, 1]):
        raise RuntimeError("正式融合要求A1和A2在425个状态全部真实可用")
    fingerprint_a1 = build_fingerprint_from_theta(loaded.theta_Y_A_actual[:, 0, :], phi_B)
    fingerprint_a2 = build_fingerprint_from_theta(loaded.theta_Y_A_actual[:, 1, :], phi_B)
    fingerprint_a12 = (0.5 * (fingerprint_a1 + fingerprint_a2)).astype(np.complex128, copy=False)
    if not np.all(np.isfinite(fingerprint_a12)):
        raise RuntimeError("Y-A12-Mean指纹包含非有限值")
    return {"Y-A1": fingerprint_a1, "Y-A2": fingerprint_a2, "Y-A12-Mean": fingerprint_a12}


def build_query_fingerprints(
    loaded: LoadedAllILC,
    phi_B: np.ndarray,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """按 ``effective_C=min(requested_C,state_max_n)`` 生成五套 Y-C Query。"""

    query_fingerprints: dict[int, np.ndarray] = {}
    effective_c = np.empty((STATE_COUNT, NMAX), dtype=np.int64)
    state_indices = np.arange(STATE_COUNT, dtype=np.int64)
    for requested_c in range(1, NMAX + 1):
        effective = np.minimum(requested_c, loaded.ilc_column_counts)
        effective_c[:, requested_c - 1] = effective
        theta = loaded.theta_Y_C_actual[state_indices, effective - 1, :]
        query_fingerprints[requested_c] = build_fingerprint_from_theta(theta, phi_B)
    return query_fingerprints, effective_c


def _combination_key(requested_c: int, lut_type: str, prefix: str = "D") -> str:
    if requested_c not in range(1, NMAX + 1):
        raise ValueError("requested_C_n必须为1...5")
    if lut_type not in LUT_TYPES:
        raise ValueError(f"未知LUT类型：{lut_type}")
    return f"{prefix}_C{requested_c}_{LUT_SUFFIXES[lut_type]}"


def build_distance_and_ranking_matrices(
    query_fingerprints: dict[int, np.ndarray],
    lut_fingerprints: dict[str, np.ndarray],
    reference_ranking: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[tuple[int, str], np.ndarray]]:
    """计算15个 Query→LUT 距离、排名及逐状态 Spearman。"""

    reference = np.asarray(reference_ranking)
    if reference.shape != (STATE_COUNT, STATE_COUNT) or not np.all(np.isfinite(reference)):
        raise ValueError("R_RR必须是finite的(425,425)矩阵")
    distances: dict[str, np.ndarray] = {}
    rankings: dict[str, np.ndarray] = {}
    spearman_values: dict[tuple[int, str], np.ndarray] = {}
    for requested_c in range(1, NMAX + 1):
        query = query_fingerprints[requested_c]
        for lut_type in LUT_TYPES:
            distance_key = _combination_key(requested_c, lut_type, "D")
            ranking_key = _combination_key(requested_c, lut_type, "R")
            distance = compute_cnmse_distance_matrix(query, lut_fingerprints[lut_type])
            ranking = distance_to_ranks(distance)
            values = np.asarray(
                [
                    spearman_statistic(reference[state_id], ranking[state_id])
                    for state_id in range(STATE_COUNT)
                ],
                dtype=np.float64,
            )
            if (
                distance.shape != (STATE_COUNT, STATE_COUNT)
                or np.isnan(distance).any()
                or np.isposinf(distance).any()
            ):
                raise RuntimeError(f"{distance_key}矩阵无效")
            if ranking.shape != (STATE_COUNT, STATE_COUNT) or not np.all(np.isfinite(ranking)):
                raise RuntimeError(f"{ranking_key}矩阵无效")
            if not np.all(np.isfinite(values)) or np.any(values < -1) or np.any(values > 1):
                raise RuntimeError(f"{requested_c}->{lut_type} Spearman无效")
            distances[distance_key] = distance
            rankings[ranking_key] = ranking
            spearman_values[(requested_c, lut_type)] = values
    return distances, rankings, spearman_values


def _state_frame_from_effective_map(effective_map: pd.DataFrame) -> pd.DataFrame:
    frame = (
        effective_map.loc[:, list(STATE_COLUMNS)]
        .drop_duplicates("state_id")
        .sort_values("state_id")
        .reset_index(drop=True)
    )
    _validate_state_ids(frame["state_id"].to_numpy(dtype=np.int64), "state_frame.state_id")
    return frame


def build_spearman_tables(
    loaded: LoadedAllILC,
    effective_c: np.ndarray,
    spearman_values: dict[tuple[int, str], np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """输出逐状态长表和15行组合聚合表。"""

    if effective_c.shape != (STATE_COUNT, NMAX):
        raise ValueError("effective_C必须是(425,5)")
    state_frame = _state_frame_from_effective_map(loaded.effective_iteration_map)
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for requested_c in range(1, NMAX + 1):
        for lut_type in LUT_TYPES:
            values = np.asarray(spearman_values[(requested_c, lut_type)], dtype=float)
            if values.shape != (STATE_COUNT,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{requested_c}->{lut_type}不是425个finite Spearman")
            for state_id, metadata in state_frame.set_index("state_id").iterrows():
                rows.append(
                    {
                        "state_id": int(state_id),
                        "funMng": int(metadata["funMng"]),
                        "funAng": int(metadata["funAng"]),
                        "secMng": int(metadata["secMng"]),
                        "secAng": int(metadata["secAng"]),
                        "Vm": float(metadata["Vm"]),
                        "Pin": float(metadata["Pin"]),
                        "requested_C_n": requested_c,
                        "effective_C_n": effective_c[:, requested_c - 1].astype(int)[state_id],
                        "lut_fingerprint_type": lut_type,
                        "spearman": float(values[state_id]),
                    }
                )
            summary_rows.append(
                {
                    "requested_C_n": requested_c,
                    "lut_fingerprint_type": lut_type,
                    "count": int(values.size),
                    "mean": float(np.mean(values)),
                    "median": float(np.median(values)),
                    "std": float(np.std(values, ddof=1)),
                    "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values)),
                }
            )
    long_table = pd.DataFrame(rows, columns=list(SPEARMAN_COLUMNS))
    summary = pd.DataFrame(summary_rows)
    expected = STATE_COUNT * NMAX * len(LUT_TYPES)
    if long_table.shape[0] != expected or summary.shape[0] != NMAX * len(LUT_TYPES):
        raise RuntimeError("Spearman长表或summary行数错误")
    return long_table, summary


def build_robustness_tables(
    state_frame: pd.DataFrame,
    spearman_values: dict[tuple[int, str], np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """计算固定 LUT 的跨 C 中位数、Worst-C 和逐状态 Worst-C。"""

    medians = {
        lut_type: np.asarray(
            [
                np.median(spearman_values[(requested_c, lut_type)])
                for requested_c in range(1, NMAX + 1)
            ],
            dtype=float,
        )
        for lut_type in LUT_TYPES
    }
    robustness_rows: list[dict[str, Any]] = []
    statewise = state_frame.copy().sort_values("state_id").reset_index(drop=True)
    for lut_type in LUT_TYPES:
        values_by_c = np.vstack(
            [spearman_values[(requested_c, lut_type)] for requested_c in range(1, NMAX + 1)]
        )
        worst_by_state = np.min(values_by_c, axis=0)
        worst_stage = np.argmin(values_by_c, axis=0) + 1
        robustness_rows.append(
            {
                "lut_type": lut_type,
                **{
                    f"C{stage}_median": float(medians[lut_type][stage - 1])
                    for stage in range(1, NMAX + 1)
                },
                "worst_C_median": float(np.min(medians[lut_type])),
                "mean_C_median": float(np.mean(medians[lut_type])),
                "std_C_median": float(np.std(medians[lut_type], ddof=0)),
                "range_C_median": float(np.max(medians[lut_type]) - np.min(medians[lut_type])),
                "statewise_worst_mean": float(np.mean(worst_by_state)),
                "statewise_worst_median": float(np.median(worst_by_state)),
                "statewise_worst_std": float(np.std(worst_by_state, ddof=1)),
                "statewise_worst_min": float(np.min(worst_by_state)),
                "statewise_worst_q25": float(np.quantile(worst_by_state, 0.25)),
                "statewise_worst_q75": float(np.quantile(worst_by_state, 0.75)),
                "statewise_worst_max": float(np.max(worst_by_state)),
            }
        )
        statewise[f"worst_C_{LUT_SUFFIXES[lut_type]}"] = worst_by_state
        statewise[f"worst_C_stage_{LUT_SUFFIXES[lut_type]}"] = worst_stage.astype(int)
    return pd.DataFrame(robustness_rows), statewise


def build_fusion_paired_tables(
    state_frame: pd.DataFrame,
    spearman_values: dict[tuple[int, str], np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """计算 A12 相对 A1/A2 的逐状态配对增益及汇总。"""

    state_frame = state_frame.sort_values("state_id").reset_index(drop=True)
    state_ids = state_frame["state_id"].to_numpy(dtype=np.int64)
    state_count = state_ids.size
    rows: list[dict[str, Any]] = []
    for requested_c in range(1, NMAX + 1):
        a1 = spearman_values[(requested_c, "Y-A1")]
        a2 = spearman_values[(requested_c, "Y-A2")]
        a12 = spearman_values[(requested_c, "Y-A12-Mean")]
        if a1.shape != (state_count,) or a2.shape != (state_count,) or a12.shape != (state_count,):
            raise ValueError("配对统计输入长度必须与state_frame一致")
        for index, state_id in enumerate(state_ids):
            rows.append(
                {
                    "state_id": int(state_id),
                    "C_stage": requested_c,
                    "rho_A1": float(a1[index]),
                    "rho_A2": float(a2[index]),
                    "rho_A12": float(a12[index]),
                    "delta_A12_vs_A1": float(a12[index] - a1[index]),
                    "delta_A12_vs_A2": float(a12[index] - a2[index]),
                }
            )
    comparison = pd.DataFrame(rows)
    summary_rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, int | None, pd.DataFrame]] = [
        ("C_stage", stage, comparison[comparison["C_stage"] == stage])
        for stage in range(1, NMAX + 1)
    ]
    scopes.append(("all_C", None, comparison))
    for scope, stage, group in scopes:
        delta_a1 = group["delta_A12_vs_A1"].to_numpy(dtype=float)
        delta_a2 = group["delta_A12_vs_A2"].to_numpy(dtype=float)
        summary_rows.append(
            {
                "scope": scope,
                "C_stage": stage,
                "count": int(group.shape[0]),
                "A12_gt_A1_state_count": int(np.count_nonzero(delta_a1 > 0)),
                "A12_lt_A1_state_count": int(np.count_nonzero(delta_a1 < 0)),
                "A12_equal_A1_state_count": int(np.count_nonzero(delta_a1 == 0)),
                "A12_gt_A2_state_count": int(np.count_nonzero(delta_a2 > 0)),
                "A12_lt_A2_state_count": int(np.count_nonzero(delta_a2 < 0)),
                "A12_equal_A2_state_count": int(np.count_nonzero(delta_a2 == 0)),
                "median_delta_vs_A1": float(np.median(delta_a1)),
                "median_delta_vs_A2": float(np.median(delta_a2)),
                "mean_delta_vs_A1": float(np.mean(delta_a1)),
                "mean_delta_vs_A2": float(np.mean(delta_a2)),
            }
        )
    return comparison, pd.DataFrame(summary_rows)


def _max_abs_difference(left: np.ndarray, right: np.ndarray) -> tuple[float, int]:
    """比较允许 -Inf 的距离矩阵，返回有限元素最大误差和非有限模式差异数。"""

    a = np.asarray(left)
    b = np.asarray(right)
    if a.shape != b.shape:
        return float("inf"), int(max(a.size, b.size))
    neg_inf_mismatch = int(np.count_nonzero(np.isneginf(a) != np.isneginf(b)))
    finite = np.isfinite(a) & np.isfinite(b)
    if np.any(np.isfinite(a) != np.isfinite(b)):
        neg_inf_mismatch += int(np.count_nonzero(np.isfinite(a) != np.isfinite(b)))
    if not np.any(finite):
        return 0.0, neg_inf_mismatch
    return float(np.max(np.abs(a[finite] - b[finite]))), neg_inf_mismatch


def baseline_regression_against_all_ilc(
    source_root: Path,
    new_distances: dict[str, np.ndarray],
    new_rankings: dict[str, np.ndarray],
    new_spearman: dict[tuple[int, str], np.ndarray],
) -> dict[str, Any]:
    """验证新 A1/A2 pipeline 与旧 all-ILC Y-A 排序逐项一致。"""

    source_root = Path(source_root)
    distance_path = source_root / "distance_matrices_all_ilc.npz"
    ranking_path = source_root / "ranking_matrices_all_ilc.npz"
    long_path = source_root / "spearman_all_ilc_long.csv"
    for path in (distance_path, ranking_path, long_path):
        if not path.is_file():
            raise FileNotFoundError(f"缺少A1/A2 baseline回归输入：{path}")
    distance_errors: list[float] = []
    ranking_errors: list[float] = []
    spearman_errors: list[float] = []
    distance_inf_mismatches = 0
    ranking_inf_mismatches = 0
    with (
        np.load(distance_path, allow_pickle=False) as old_distance,
        np.load(ranking_path, allow_pickle=False) as old_ranking,
    ):
        for requested_c in range(1, NMAX + 1):
            for lut_type, a_stage in (("Y-A1", 1), ("Y-A2", 2)):
                old_d_key = f"D_CY_n1_{requested_c:02d}_n2_{a_stage:02d}"
                old_r_key = f"R_CY_n1_{requested_c:02d}_n2_{a_stage:02d}"
                new_d_key = _combination_key(requested_c, lut_type, "D")
                new_r_key = _combination_key(requested_c, lut_type, "R")
                if old_d_key not in old_distance.files or old_r_key not in old_ranking.files:
                    raise ValueError(f"旧all-ILC缺少baseline键：{old_d_key}/{old_r_key}")
                distance_error, inf_mismatch = _max_abs_difference(
                    new_distances[new_d_key], old_distance[old_d_key]
                )
                ranking_error, ranking_inf_mismatch = _max_abs_difference(
                    new_rankings[new_r_key], old_ranking[old_r_key]
                )
                distance_errors.append(distance_error)
                ranking_errors.append(ranking_error)
                distance_inf_mismatches += inf_mismatch
                ranking_inf_mismatches += ranking_inf_mismatch
    old_long = pd.read_csv(long_path)
    for requested_c in range(1, NMAX + 1):
        for lut_type, a_stage in (("Y-A1", 1), ("Y-A2", 2)):
            old_values = (
                old_long[
                    (old_long["requested_n1"] == requested_c)
                    & (old_long["requested_n2"] == a_stage)
                    & (old_long["lut_type"] == "Y-A")
                ]
                .sort_values("state_id")["spearman"]
                .to_numpy(dtype=float)
            )
            if old_values.shape != (STATE_COUNT,):
                raise ValueError(f"旧all-ILC baseline C{requested_c}->A{a_stage}不是425行")
            spearman_errors.append(
                float(np.max(np.abs(new_spearman[(requested_c, lut_type)] - old_values)))
            )
    result = {
        "distance_max_abs_error": float(max(distance_errors)),
        "ranking_max_abs_error": float(max(ranking_errors)),
        "spearman_max_abs_error": float(max(spearman_errors)),
        "distance_nonfinite_pattern_mismatch_count": distance_inf_mismatches,
        "ranking_nonfinite_pattern_mismatch_count": ranking_inf_mismatches,
        "distance_diagnostic_tolerance": 1e-6,
        "ranking_tolerance": 1e-10,
        "spearman_tolerance": 1e-10,
    }
    result["pass"] = bool(
        result["ranking_max_abs_error"] <= result["ranking_tolerance"]
        and result["spearman_max_abs_error"] <= result["spearman_tolerance"]
        and distance_inf_mismatches == 0
        and ranking_inf_mismatches == 0
    )
    if not result["pass"]:
        raise RuntimeError(f"A1/A2 baseline回归失败：{result}")
    return result


def choose_robust_lut(robustness_summary: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """按预声明的 Worst-C→statewise median→mean→std 规则选择 LUT。"""

    required = {
        "lut_type",
        "worst_C_median",
        "statewise_worst_median",
        "mean_C_median",
        "std_C_median",
    }
    if not required.issubset(robustness_summary.columns):
        raise ValueError("robustness summary缺少LUT选择字段")
    ranked = robustness_summary.copy()
    ranked["_simple_order"] = ranked["lut_type"].map({"Y-A1": 0, "Y-A2": 1, "Y-A12-Mean": 2})
    ranked = ranked.sort_values(
        [
            "worst_C_median",
            "statewise_worst_median",
            "mean_C_median",
            "std_C_median",
            "_simple_order",
        ],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return str(ranked.iloc[0]["lut_type"]), ranked.drop(columns="_simple_order")


def run_y_lut_fusion_analysis(
    reference_root: Path,
    all_ilc_root: Path,
) -> YLUTFusionResult:
    """执行完整的固定 Y-A LUT 融合分析，但不接触 raw 或重新训练模型。"""

    frozen = load_frozen_scenario2_artifacts(Path(reference_root))
    loaded = load_all_ilc_theta(Path(all_ilc_root))
    if not np.array_equal(frozen.state_ids, loaded.state_ids):
        raise ValueError("Scenario 2 reference与all-ILC state_ids不一致")
    lut_fingerprints = build_y_lut_fingerprints(loaded, frozen.phi_B)
    theta_mean = 0.5 * (loaded.theta_Y_A_actual[:, 0, :] + loaded.theta_Y_A_actual[:, 1, :])
    fingerprint_from_theta_mean = build_fingerprint_from_theta(theta_mean, frozen.phi_B)
    fingerprint_mean_error = float(
        np.max(np.abs(lut_fingerprints["Y-A12-Mean"] - fingerprint_from_theta_mean))
    )
    if fingerprint_mean_error > 1e-10:
        raise RuntimeError(f"fingerprint mean与theta mean不一致：{fingerprint_mean_error}")
    query_fingerprints, effective_c = build_query_fingerprints(loaded, frozen.phi_B)
    distances, rankings, spearman_values = build_distance_and_ranking_matrices(
        query_fingerprints,
        lut_fingerprints,
        frozen.R_RR,
    )
    baseline = baseline_regression_against_all_ilc(
        Path(all_ilc_root), distances, rankings, spearman_values
    )
    spearman_long, spearman_summary = build_spearman_tables(loaded, effective_c, spearman_values)
    state_frame = _state_frame_from_effective_map(loaded.effective_iteration_map)
    robustness_summary, statewise_worst = build_robustness_tables(state_frame, spearman_values)
    selected_lut, robustness_ranked = choose_robust_lut(robustness_summary)
    paired_comparison, paired_summary = build_fusion_paired_tables(state_frame, spearman_values)
    validation: dict[str, Any] = {
        "scenario": 2,
        "study": "robust Y-A LUT fingerprint fusion",
        "state_count": STATE_COUNT,
        "lut_types": list(LUT_TYPES),
        "fusion_definition": "sample-wise arithmetic mean of common-B Y-A1 and Y-A2 responses",
        "raw_waveform_averaging": False,
        "common_B_probe_changed": False,
        "X_A_used_as_LUT": False,
        "Real_B_used_as_LUT": False,
        "Real_B_used_as_reference": True,
        "reference_ranking": "R_RR = frozen Real-B to Real-B",
        "requested_C_stages": list(range(1, NMAX + 1)),
        "statewise_C_saturation": True,
        "effective_C_rule": "min(requested_C_n, state_max_n)",
        "A1_available_state_count": int(np.count_nonzero(loaded.available_mask[:, 0])),
        "A2_available_state_count": int(np.count_nonzero(loaded.available_mask[:, 1])),
        "candidate_count_per_query": STATE_COUNT,
        "self_match_included": True,
        "ridge_lambda": RIDGE_LAMBDA,
        "ridge_rescanned": False,
        "models_retrained": False,
        "mp_changed": False,
        "canonical_changed": False,
        "common_B_input_shape": list(frozen.common_B_input.shape),
        "phi_B_shape": list(frozen.phi_B.shape),
        "fingerprint_length": FINGERPRINT_LENGTH,
        "Y_A1_fingerprint_shape": list(lut_fingerprints["Y-A1"].shape),
        "Y_A2_fingerprint_shape": list(lut_fingerprints["Y-A2"].shape),
        "Y_A12_mean_fingerprint_shape": list(lut_fingerprints["Y-A12-Mean"].shape),
        "all_lut_fingerprints_complex128_finite": bool(
            all(
                value.dtype == np.complex128 and np.all(np.isfinite(value))
                for value in lut_fingerprints.values()
            )
        ),
        "fingerprint_mean_theta_mean_max_abs_error": fingerprint_mean_error,
        "distance_matrix_count": len(distances),
        "expected_distance_matrix_count": NMAX * len(LUT_TYPES),
        "ranking_matrix_count": len(rankings),
        "expected_ranking_matrix_count": NMAX * len(LUT_TYPES),
        "all_distance_matrices_valid": bool(
            all(
                value.shape == (STATE_COUNT, STATE_COUNT)
                and not np.isnan(value).any()
                and not np.isposinf(value).any()
                for value in distances.values()
            )
        ),
        "all_ranking_matrices_valid": bool(
            all(
                value.shape == (STATE_COUNT, STATE_COUNT) and np.all(np.isfinite(value))
                for value in rankings.values()
            )
        ),
        "spearman_total_count": int(spearman_long.shape[0]),
        "expected_spearman_total_count": STATE_COUNT * NMAX * len(LUT_TYPES),
        "all_spearman_finite": bool(
            np.all(np.isfinite(spearman_long["spearman"].to_numpy(dtype=float)))
        ),
        "spearman_range_min": float(spearman_long["spearman"].min()),
        "spearman_range_max": float(spearman_long["spearman"].max()),
        "baseline_regression": baseline,
        "baseline_regression_pass": bool(baseline["pass"]),
        "fusion_selection_rule": [
            "maximize worst_C_median",
            "then maximize statewise_worst_median",
            "then maximize mean_C_median",
            "then minimize std_C_median",
            "exact tie prefers simpler single-stage LUT",
        ],
        "selected_robust_lut_type": selected_lut,
        "robustness_ranking": robustness_ranked.to_dict("records"),
        "cnmse_recomputed": True,
        "spearman_recomputed": True,
        "D_RR_recomputed": False,
        "figure_count": 2,
    }
    if (
        validation["A1_available_state_count"] != STATE_COUNT
        or validation["A2_available_state_count"] != STATE_COUNT
    ):
        raise RuntimeError("A1/A2不是425/425可用，不能继续正式融合")
    if validation["distance_matrix_count"] != validation["expected_distance_matrix_count"]:
        raise RuntimeError("距离矩阵数量错误")
    if validation["ranking_matrix_count"] != validation["expected_ranking_matrix_count"]:
        raise RuntimeError("排名矩阵数量错误")
    if validation["spearman_total_count"] != validation["expected_spearman_total_count"]:
        raise RuntimeError("Spearman总数错误")
    return YLUTFusionResult(
        loaded=loaded,
        state_ids=loaded.state_ids,
        common_B_input=frozen.common_B_input,
        phi_B=frozen.phi_B,
        lut_fingerprints=lut_fingerprints,
        query_fingerprints=query_fingerprints,
        effective_C=effective_c,
        distance_matrices=distances,
        ranking_matrices=rankings,
        spearman_by_combination=spearman_values,
        spearman_long=spearman_long,
        spearman_summary=spearman_summary,
        lut_robustness_summary=robustness_summary,
        statewise_worst_over_C=statewise_worst,
        fusion_paired_comparison=paired_comparison,
        fusion_paired_summary=paired_summary,
        baseline_regression=baseline,
        validation=validation,
    )


__all__ = [
    "EFFECTIVE_MAP_COLUMNS",
    "FINGERPRINT_LENGTH",
    "LUT_TYPES",
    "NMAX",
    "NUM_COEFFICIENTS",
    "RIDGE_LAMBDA",
    "STATE_COLUMNS",
    "STATE_COUNT",
    "LoadedAllILC",
    "YLUTFusionResult",
    "baseline_regression_against_all_ilc",
    "build_distance_and_ranking_matrices",
    "build_fingerprint_from_theta",
    "build_fusion_paired_tables",
    "build_query_fingerprints",
    "build_robustness_tables",
    "build_spearman_tables",
    "build_y_lut_fingerprints",
    "choose_robust_lut",
    "load_all_ilc_theta",
    "run_y_lut_fusion_analysis",
]
