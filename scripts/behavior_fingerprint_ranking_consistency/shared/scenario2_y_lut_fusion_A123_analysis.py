"""Scenario 2 的 A1/A2/A3 及等权多阶段 Y-A LUT 指纹鲁棒性分析。

本模块在已冻结的 common-B 行为指纹层扩展上一轮 Y-A1/Y-A2/Y-A12 研究，
固定比较七种 LUT：Y-A1、Y-A2、Y-A3、Y-A12-Mean、Y-A13-Mean、
Y-A23-Mean 和 Y-A123-Mean。所有融合都由配置表驱动并在最终复数指纹层
完成；Real-B 只作为冻结的 R_RR reference，不作为 LUT candidate。
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
from .scenario2_y_lut_fusion_analysis import (
    LoadedAllILC,
    _max_abs_difference,
    _state_frame_from_effective_map,
    build_fingerprint_from_theta,
    build_query_fingerprints,
    load_all_ilc_theta,
)

STATE_COUNT = 425
NMAX = 5
NUM_COEFFICIENTS = 10
RIDGE_LAMBDA = 1e-8
LUT_TYPES = (
    "Y-A1",
    "Y-A2",
    "Y-A3",
    "Y-A12-Mean",
    "Y-A13-Mean",
    "Y-A23-Mean",
    "Y-A123-Mean",
)
LUT_SUFFIXES = {
    "Y-A1": "A1",
    "Y-A2": "A2",
    "Y-A3": "A3",
    "Y-A12-Mean": "A12",
    "Y-A13-Mean": "A13",
    "Y-A23-Mean": "A23",
    "Y-A123-Mean": "A123",
}
LUT_DEFINITIONS: dict[str, tuple[tuple[int, float], ...]] = {
    "Y-A1": ((1, 1.0),),
    "Y-A2": ((2, 1.0),),
    "Y-A3": ((3, 1.0),),
    "Y-A12-Mean": ((1, 0.5), (2, 0.5)),
    "Y-A13-Mean": ((1, 0.5), (3, 0.5)),
    "Y-A23-Mean": ((2, 0.5), (3, 0.5)),
    "Y-A123-Mean": ((1, 1.0 / 3.0), (2, 1.0 / 3.0), (3, 1.0 / 3.0)),
}
NEW_LUT_TYPES = ("Y-A3", "Y-A12-Mean", "Y-A13-Mean", "Y-A23-Mean", "Y-A123-Mean")
STATE_COLUMNS = ("state_id", "funMng", "funAng", "secMng", "secAng", "Vm", "Pin")
SPEARMAN_COLUMNS = (
    *STATE_COLUMNS,
    "requested_C_n",
    "effective_C_n",
    "lut_fingerprint_type",
    "spearman",
)


@dataclass(frozen=True)
class A123FusionResult:
    """A123 研究的数组、表格、回归和验证信息。"""

    loaded: LoadedAllILC
    state_ids: np.ndarray
    common_B_input: np.ndarray
    phi_B: np.ndarray
    theta_effective_A: dict[int, np.ndarray]
    A3_effective_stage_per_state: np.ndarray
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
    paired_vs_A2_by_state: pd.DataFrame
    paired_vs_A2_summary: pd.DataFrame
    saturation_sensitivity_summary: pd.DataFrame
    fusion_equivalence_errors: dict[str, float]
    baseline_regression: dict[str, Any]
    sensitivity_validation: dict[str, Any]
    validation: dict[str, Any]


def identify_missing_a3_state(loaded: LoadedAllILC) -> dict[str, Any]:
    """验证 A1/A2/A3 availability 并返回唯一缺失 A3 状态的元数据。"""

    counts = {
        f"A{stage}_available_state_count": int(
            np.count_nonzero(loaded.available_mask[:, stage - 1])
        )
        for stage in range(1, 4)
    }
    if counts["A1_available_state_count"] != STATE_COUNT:
        raise RuntimeError("A1必须在425个状态全部真实可用")
    if counts["A2_available_state_count"] != STATE_COUNT:
        raise RuntimeError("A2必须在425个状态全部真实可用")
    if counts["A3_available_state_count"] != STATE_COUNT - 1:
        raise RuntimeError(
            f"本轮预期A3恰好缺失1个状态，实际可用{counts['A3_available_state_count']}"
        )
    missing = np.flatnonzero(~loaded.available_mask[:, 2])
    if missing.size != 1:
        raise RuntimeError("A3缺失状态数量不是1")
    state_frame = _state_frame_from_effective_map(loaded.effective_iteration_map)
    row = state_frame[state_frame["state_id"] == int(missing[0])].iloc[0]
    return {
        **counts,
        "missing_A3_state_id": int(missing[0]),
        "missing_A3_state": {
            column: (
                int(row[column])
                if column in {"state_id", "funMng", "funAng", "secMng", "secAng"}
                else float(row[column])
            )
            for column in STATE_COLUMNS
        },
    }


def build_effective_a_thetas(
    loaded: LoadedAllILC,
) -> tuple[dict[int, np.ndarray], np.ndarray]:
    """提取 A1/A2，并按 state-wise saturation 生成 effective A3 theta。"""

    state_indices = np.arange(STATE_COUNT, dtype=np.int64)
    a3_effective_stage = np.minimum(3, loaded.ilc_column_counts).astype(np.int64)
    theta_effective = {
        1: np.asarray(loaded.theta_Y_A_actual[:, 0, :]),
        2: np.asarray(loaded.theta_Y_A_actual[:, 1, :]),
        3: np.asarray(loaded.theta_Y_A_actual[state_indices, a3_effective_stage - 1, :]),
    }
    for stage, theta in theta_effective.items():
        if theta.shape != (STATE_COUNT, NUM_COEFFICIENTS):
            raise ValueError(f"A{stage} theta shape错误：{theta.shape}")
        if theta.dtype != np.complex128 or not np.all(np.isfinite(theta)):
            raise ValueError(f"effective A{stage} theta必须是finite complex128")
    return theta_effective, a3_effective_stage


def build_a123_lut_fingerprints(
    loaded: LoadedAllILC,
    phi_B: np.ndarray,
) -> tuple[dict[int, np.ndarray], np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    """按 LUT_DEFINITIONS 构造七类指纹和 fingerprint/theta 等价性误差。"""

    theta_effective, a3_effective_stage = build_effective_a_thetas(loaded)
    base_fingerprints = {
        stage: build_fingerprint_from_theta(theta, phi_B)
        for stage, theta in theta_effective.items()
    }
    lut_fingerprints: dict[str, np.ndarray] = {}
    fusion_errors: dict[str, float] = {}
    for lut_type, definition in LUT_DEFINITIONS.items():
        fused_fingerprint = np.zeros_like(base_fingerprints[1])
        fused_theta = np.zeros_like(theta_effective[1])
        for stage, weight in definition:
            fused_fingerprint = fused_fingerprint + weight * base_fingerprints[stage]
            fused_theta = fused_theta + weight * theta_effective[stage]
        fused_fingerprint = fused_fingerprint.astype(np.complex128, copy=False)
        if fused_fingerprint.shape != (STATE_COUNT, FINGERPRINT_LENGTH) or not np.all(
            np.isfinite(fused_fingerprint)
        ):
            raise RuntimeError(f"{lut_type}指纹shape或finite性错误")
        from_theta = build_fingerprint_from_theta(fused_theta, phi_B)
        error = float(np.max(np.abs(fused_fingerprint - from_theta)))
        if error > 1e-10:
            raise RuntimeError(f"{lut_type} fingerprint/theta mean不一致：{error}")
        lut_fingerprints[lut_type] = fused_fingerprint
        fusion_errors[lut_type] = error
    return theta_effective, a3_effective_stage, lut_fingerprints, fusion_errors


def _combination_key(requested_c: int, lut_type: str, prefix: str) -> str:
    if requested_c not in range(1, NMAX + 1):
        raise ValueError("requested_C_n必须为1...5")
    if lut_type not in LUT_TYPES:
        raise ValueError(f"未知LUT类型：{lut_type}")
    if prefix not in {"D", "R"}:
        raise ValueError("矩阵前缀必须是D或R")
    return f"{prefix}_C{requested_c}_{LUT_SUFFIXES[lut_type]}"


def build_a123_distance_ranking(
    query_fingerprints: dict[int, np.ndarray],
    lut_fingerprints: dict[str, np.ndarray],
    reference_ranking: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[tuple[int, str], np.ndarray]]:
    """计算 35 个 Query→LUT 距离、排名和逐状态 Spearman。"""

    reference = np.asarray(reference_ranking)
    if reference.shape != (STATE_COUNT, STATE_COUNT) or not np.all(np.isfinite(reference)):
        raise ValueError("R_RR必须是finite的(425,425)矩阵")
    if set(query_fingerprints) != set(range(1, NMAX + 1)):
        raise ValueError("Query必须覆盖C1...C5")
    if set(lut_fingerprints) != set(LUT_TYPES):
        raise ValueError("LUT指纹必须覆盖七类固定定义")
    distances: dict[str, np.ndarray] = {}
    rankings: dict[str, np.ndarray] = {}
    spearman_values: dict[tuple[int, str], np.ndarray] = {}
    for requested_c in range(1, NMAX + 1):
        for lut_type in LUT_TYPES:
            distance_key = _combination_key(requested_c, lut_type, "D")
            ranking_key = _combination_key(requested_c, lut_type, "R")
            distance = compute_cnmse_distance_matrix(
                query_fingerprints[requested_c], lut_fingerprints[lut_type]
            )
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
                raise RuntimeError(f"{distance_key}无效")
            if ranking.shape != (STATE_COUNT, STATE_COUNT) or not np.all(np.isfinite(ranking)):
                raise RuntimeError(f"{ranking_key}无效")
            if not np.all(np.isfinite(values)) or np.any(values < -1) or np.any(values > 1):
                raise RuntimeError(f"{requested_c}->{lut_type} Spearman无效")
            distances[distance_key] = distance
            rankings[ranking_key] = ranking
            spearman_values[(requested_c, lut_type)] = values
    return distances, rankings, spearman_values


def build_a123_spearman_tables(
    loaded: LoadedAllILC,
    effective_c: np.ndarray,
    spearman_values: dict[tuple[int, str], np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成 14875 行逐状态长表和 35 行聚合表。"""

    if effective_c.shape != (STATE_COUNT, NMAX):
        raise ValueError("effective_C必须是(425,5)")
    state_frame = _state_frame_from_effective_map(loaded.effective_iteration_map)
    metadata = state_frame.set_index("state_id").to_dict("index")
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for requested_c in range(1, NMAX + 1):
        for lut_type in LUT_TYPES:
            values = np.asarray(spearman_values[(requested_c, lut_type)], dtype=float)
            if values.shape != (STATE_COUNT,) or not np.all(np.isfinite(values)):
                raise ValueError(f"{requested_c}->{lut_type}不是425个finite Spearman")
            for state_id in range(STATE_COUNT):
                item = metadata[state_id]
                rows.append(
                    {
                        "state_id": state_id,
                        "funMng": int(item["funMng"]),
                        "funAng": int(item["funAng"]),
                        "secMng": int(item["secMng"]),
                        "secAng": int(item["secAng"]),
                        "Vm": float(item["Vm"]),
                        "Pin": float(item["Pin"]),
                        "requested_C_n": requested_c,
                        "effective_C_n": int(effective_c[state_id, requested_c - 1]),
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
    if long_table.shape[0] != STATE_COUNT * NMAX * len(LUT_TYPES):
        raise RuntimeError("Spearman长表行数错误")
    if summary.shape[0] != NMAX * len(LUT_TYPES):
        raise RuntimeError("Spearman summary行数错误")
    return long_table, summary


def _robustness_from_values(
    state_frame: pd.DataFrame,
    spearman_values: dict[tuple[int, str], np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """对任意状态子集计算 LUT 鲁棒性表。"""

    state_frame = state_frame.sort_values("state_id").reset_index(drop=True)
    state_count = state_frame.shape[0]
    rows: list[dict[str, Any]] = []
    statewise = state_frame.copy()
    for lut_type in LUT_TYPES:
        values_by_c = np.vstack(
            [
                np.asarray(spearman_values[(stage, lut_type)], dtype=float)
                for stage in range(1, NMAX + 1)
            ]
        )
        if values_by_c.shape != (NMAX, state_count):
            raise ValueError("鲁棒性输入状态数不一致")
        medians = np.median(values_by_c, axis=1)
        worst_by_state = np.min(values_by_c, axis=0)
        worst_stage = np.argmin(values_by_c, axis=0) + 1
        rows.append(
            {
                "lut_type": lut_type,
                **{f"C{stage}_median": float(medians[stage - 1]) for stage in range(1, NMAX + 1)},
                "worst_C_median": float(np.min(medians)),
                "statewise_worst_C_median": float(np.median(worst_by_state)),
                "mean_C_median": float(np.mean(medians)),
                "std_C_median": float(np.std(medians, ddof=0)),
                "range_C_median": float(np.max(medians) - np.min(medians)),
                "statewise_worst_mean": float(np.mean(worst_by_state)),
                "statewise_worst_q25": float(np.quantile(worst_by_state, 0.25)),
                "statewise_worst_min": float(np.min(worst_by_state)),
                "statewise_worst_q75": float(np.quantile(worst_by_state, 0.75)),
                "statewise_worst_max": float(np.max(worst_by_state)),
            }
        )
        suffix = LUT_SUFFIXES[lut_type]
        statewise[f"worst_C_{suffix}"] = worst_by_state
        statewise[f"worst_C_stage_{suffix}"] = worst_stage.astype(int)
    return pd.DataFrame(rows), statewise


def build_a123_robustness_tables(
    state_frame: pd.DataFrame,
    spearman_values: dict[tuple[int, str], np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """计算完整 425 状态的七类 LUT 鲁棒性统计。"""

    return _robustness_from_values(state_frame, spearman_values)


def choose_a123_robust_lut(robustness_summary: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    """按冻结的 Worst-C→statewise median→mean→std 规则选择 LUT。"""

    required = {
        "lut_type",
        "worst_C_median",
        "statewise_worst_C_median",
        "mean_C_median",
        "std_C_median",
    }
    if not required.issubset(robustness_summary.columns):
        raise ValueError("鲁棒性表缺少 winner 选择字段")
    ranked = robustness_summary.copy()
    ranked["_simple_order"] = ranked["lut_type"].map(
        {lut_type: index for index, lut_type in enumerate(LUT_TYPES)}
    )
    ranked = ranked.sort_values(
        [
            "worst_C_median",
            "statewise_worst_C_median",
            "mean_C_median",
            "std_C_median",
            "_simple_order",
        ],
        ascending=[False, False, False, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    return str(ranked.iloc[0]["lut_type"]), ranked.drop(columns="_simple_order")


def build_paired_vs_a2(
    state_frame: pd.DataFrame,
    spearman_values: dict[tuple[int, str], np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """生成五种新候选相对于 Y-A2 的逐状态 paired 表和汇总。"""

    state_ids = state_frame.sort_values("state_id")["state_id"].to_numpy(dtype=np.int64)
    state_count = state_ids.size
    rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for requested_c in range(1, NMAX + 1):
        baseline = spearman_values[(requested_c, "Y-A2")]
        values_by_lut = {
            lut_type: spearman_values[(requested_c, lut_type)] for lut_type in NEW_LUT_TYPES
        }
        for lut_type in NEW_LUT_TYPES:
            values = values_by_lut[lut_type]
            if baseline.shape != (state_count,) or values.shape != (state_count,):
                raise ValueError("paired输入长度必须与state_frame一致")
            delta = values - baseline
            summary_rows.append(
                {
                    "C_stage": requested_c,
                    "lut_type": lut_type,
                    "count": state_count,
                    "median_delta": float(np.median(delta)),
                    "mean_delta": float(np.mean(delta)),
                    "better_count": int(np.count_nonzero(delta > 0)),
                    "equal_count": int(np.count_nonzero(delta == 0)),
                    "worse_count": int(np.count_nonzero(delta < 0)),
                    "q25_delta": float(np.quantile(delta, 0.25)),
                    "q75_delta": float(np.quantile(delta, 0.75)),
                    "min_delta": float(np.min(delta)),
                    "max_delta": float(np.max(delta)),
                }
            )
        for index, state_id in enumerate(state_ids):
            row: dict[str, Any] = {
                "state_id": int(state_id),
                "C_stage": requested_c,
                "rho_A2": float(baseline[index]),
            }
            for lut_type, values in values_by_lut.items():
                suffix = LUT_SUFFIXES[lut_type]
                row[f"rho_{suffix}"] = float(values[index])
                row[f"delta_{suffix}_vs_A2"] = float(values[index] - baseline[index])
            rows.append(row)
    comparison = pd.DataFrame(rows)
    summary = pd.DataFrame(summary_rows)
    if comparison.shape[0] != STATE_COUNT * NMAX:
        raise RuntimeError("paired_vs_A2_by_state行数错误")
    if summary.shape[0] != NMAX * len(NEW_LUT_TYPES):
        raise RuntimeError("paired_vs_A2_summary行数错误")
    return comparison, summary


def build_saturation_sensitivity(
    state_frame: pd.DataFrame,
    spearman_values: dict[tuple[int, str], np.ndarray],
    missing_state_id: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """比较全425状态与排除唯一 Ns=2 状态的 winner 和鲁棒指标。"""

    full_summary, _ = _robustness_from_values(state_frame, spearman_values)
    full_winner, _ = choose_a123_robust_lut(full_summary)
    kept = state_frame[state_frame["state_id"] != missing_state_id].copy()
    keep_mask = state_frame["state_id"].to_numpy(dtype=int) != missing_state_id
    excluded_values = {key: np.asarray(value)[keep_mask] for key, value in spearman_values.items()}
    excluded_summary, _ = _robustness_from_values(kept, excluded_values)
    excluded_winner, _ = choose_a123_robust_lut(excluded_summary)
    rows: list[dict[str, Any]] = []
    for sample_scope, sample_count, summary in (
        ("all_425_states", STATE_COUNT, full_summary),
        ("exclude_Ns2_state", STATE_COUNT - 1, excluded_summary),
    ):
        winner = full_winner if sample_scope == "all_425_states" else excluded_winner
        for row in summary.to_dict("records"):
            rows.append(
                {
                    "sample_scope": sample_scope,
                    "state_count": sample_count,
                    "winner": winner,
                    **row,
                }
            )
    sensitivity = pd.DataFrame(rows)
    return sensitivity, {
        "missing_A3_state_id": int(missing_state_id),
        "full_425_winner": full_winner,
        "exclude_Ns2_winner": excluded_winner,
        "winner_unchanged": bool(full_winner == excluded_winner),
        "full_state_count": STATE_COUNT,
        "exclude_state_count": STATE_COUNT - 1,
    }


def _max_finite_abs_difference(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left)
    b = np.asarray(right)
    if a.shape != b.shape:
        return float("inf")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("此比较仅允许finite数组")
    return float(np.max(np.abs(a - b)))


def baseline_regression_against_previous_fusion(
    previous_root: Path,
    current_lut_fingerprints: dict[str, np.ndarray],
    current_query_fingerprints: dict[int, np.ndarray],
    current_distances: dict[str, np.ndarray],
    current_rankings: dict[str, np.ndarray],
    current_spearman: dict[tuple[int, str], np.ndarray],
    current_robustness: pd.DataFrame,
) -> dict[str, Any]:
    """回归上一轮 A1/A2/A12 的指纹、Query、矩阵、Spearman 和鲁棒性。"""

    previous_root = Path(previous_root)
    paths = {
        "lut": previous_root / "lut_fingerprints.npz",
        "query": previous_root / "query_fingerprints.npz",
        "distance": previous_root / "distance_matrices.npz",
        "ranking": previous_root / "ranking_matrices.npz",
        "long": previous_root / "spearman_by_state_long.csv",
        "summary": previous_root / "spearman_summary.csv",
        "robustness": previous_root / "lut_robustness_summary.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"缺少上一轮fusion baseline文件：{missing}")
    old_lut = np.load(paths["lut"], allow_pickle=False)
    old_query = np.load(paths["query"], allow_pickle=False)
    old_distance = np.load(paths["distance"], allow_pickle=False)
    old_ranking = np.load(paths["ranking"], allow_pickle=False)
    try:
        fingerprint_errors = []
        for lut_type, key in (
            ("Y-A1", "Y_A1_fingerprints"),
            ("Y-A2", "Y_A2_fingerprints"),
            ("Y-A12-Mean", "Y_A12_mean_fingerprints"),
        ):
            fingerprint_errors.append(
                _max_finite_abs_difference(current_lut_fingerprints[lut_type], old_lut[key])
            )
        query_errors = [
            _max_finite_abs_difference(current_query_fingerprints[stage], old_query[f"Q_C{stage}"])
            for stage in range(1, NMAX + 1)
        ]
        distance_errors: list[float] = []
        ranking_errors: list[float] = []
        distance_inf_mismatches = 0
        ranking_inf_mismatches = 0
        for requested_c in range(1, NMAX + 1):
            for lut_type, old_suffix in (
                ("Y-A1", "Y_A1"),
                ("Y-A2", "Y_A2"),
                ("Y-A12-Mean", "Y_A12_mean"),
            ):
                current_d = current_distances[_combination_key(requested_c, lut_type, "D")]
                current_r = current_rankings[_combination_key(requested_c, lut_type, "R")]
                old_d = old_distance[f"D_C{requested_c}_{old_suffix}"]
                old_r = old_ranking[f"R_C{requested_c}_{old_suffix}"]
                distance_error, distance_inf = _max_abs_difference(current_d, old_d)
                ranking_error, ranking_inf = _max_abs_difference(current_r, old_r)
                distance_errors.append(distance_error)
                ranking_errors.append(ranking_error)
                distance_inf_mismatches += distance_inf
                ranking_inf_mismatches += ranking_inf
        old_long = pd.read_csv(paths["long"])
        spearman_errors: list[float] = []
        for requested_c in range(1, NMAX + 1):
            for lut_type in ("Y-A1", "Y-A2", "Y-A12-Mean"):
                old_values = (
                    old_long[
                        (old_long["requested_C_n"] == requested_c)
                        & (old_long["lut_fingerprint_type"] == lut_type)
                    ]
                    .sort_values("state_id")["spearman"]
                    .to_numpy(dtype=float)
                )
                if old_values.shape != (STATE_COUNT,):
                    raise ValueError(f"上一轮{requested_c}->{lut_type}不是425行")
                spearman_errors.append(
                    float(np.max(np.abs(current_spearman[(requested_c, lut_type)] - old_values)))
                )
        old_robustness = pd.read_csv(paths["robustness"])
        robustness_errors: list[float] = []
        robust_field_pairs = (
            ("C1_median", "C1_median"),
            ("C2_median", "C2_median"),
            ("C3_median", "C3_median"),
            ("C4_median", "C4_median"),
            ("C5_median", "C5_median"),
            ("worst_C_median", "worst_C_median"),
            ("mean_C_median", "mean_C_median"),
            ("std_C_median", "std_C_median"),
            ("range_C_median", "range_C_median"),
            ("statewise_worst_median", "statewise_worst_C_median"),
            ("statewise_worst_mean", "statewise_worst_mean"),
            ("statewise_worst_q25", "statewise_worst_q25"),
            ("statewise_worst_min", "statewise_worst_min"),
        )
        for lut_type in ("Y-A1", "Y-A2", "Y-A12-Mean"):
            old_row = old_robustness[old_robustness["lut_type"] == lut_type].iloc[0]
            current_row = current_robustness[current_robustness["lut_type"] == lut_type].iloc[0]
            robustness_errors.extend(
                [
                    abs(float(current_row[current_field]) - float(old_row[old_field]))
                    for old_field, current_field in robust_field_pairs
                ]
            )
    finally:
        old_lut.close()
        old_query.close()
        old_distance.close()
        old_ranking.close()
    result = {
        "fingerprint_max_abs_error": float(max(fingerprint_errors)),
        "query_fingerprint_max_abs_error": float(max(query_errors)),
        "distance_max_abs_error": float(max(distance_errors)),
        "ranking_max_abs_error": float(max(ranking_errors)),
        "spearman_max_abs_error": float(max(spearman_errors)),
        "robustness_max_abs_error": float(max(robustness_errors)),
        "distance_nonfinite_pattern_mismatch_count": distance_inf_mismatches,
        "ranking_nonfinite_pattern_mismatch_count": ranking_inf_mismatches,
        "fingerprint_tolerance": 1e-12,
        "query_fingerprint_tolerance": 1e-12,
        "distance_diagnostic_tolerance": 1e-6,
        "ranking_tolerance": 1e-12,
        "spearman_tolerance": 1e-12,
        "robustness_tolerance": 1e-12,
    }
    result["pass"] = bool(
        result["fingerprint_max_abs_error"] <= result["fingerprint_tolerance"]
        and result["query_fingerprint_max_abs_error"] <= result["query_fingerprint_tolerance"]
        and result["ranking_max_abs_error"] <= result["ranking_tolerance"]
        and result["spearman_max_abs_error"] <= result["spearman_tolerance"]
        and result["robustness_max_abs_error"] <= result["robustness_tolerance"]
        and distance_inf_mismatches == 0
        and ranking_inf_mismatches == 0
    )
    if not result["pass"]:
        raise RuntimeError(f"上一轮A1/A2/A12回归失败：{result}")
    return result


def run_a123_y_lut_fusion_analysis(
    reference_root: Path,
    all_ilc_root: Path,
    previous_fusion_root: Path,
) -> A123FusionResult:
    """执行完整 A1/A2/A3/融合研究，且不读取 raw 或重新训练模型。"""

    frozen = load_frozen_scenario2_artifacts(Path(reference_root))
    loaded = load_all_ilc_theta(Path(all_ilc_root))
    if not np.array_equal(frozen.state_ids, loaded.state_ids):
        raise ValueError("reference与all-ILC state_ids不一致")
    availability = identify_missing_a3_state(loaded)
    theta_effective, a3_effective_stage, lut_fingerprints, fusion_errors = (
        build_a123_lut_fingerprints(loaded, frozen.phi_B)
    )
    query_fingerprints, effective_c = build_query_fingerprints(loaded, frozen.phi_B)
    distances, rankings, spearman_values = build_a123_distance_ranking(
        query_fingerprints, lut_fingerprints, frozen.R_RR
    )
    spearman_long, spearman_summary = build_a123_spearman_tables(
        loaded, effective_c, spearman_values
    )
    state_frame = _state_frame_from_effective_map(loaded.effective_iteration_map)
    robustness_summary, statewise_worst = build_a123_robustness_tables(state_frame, spearman_values)
    selected_lut, robustness_ranked = choose_a123_robust_lut(robustness_summary)
    baseline = baseline_regression_against_previous_fusion(
        Path(previous_fusion_root),
        lut_fingerprints,
        query_fingerprints,
        distances,
        rankings,
        spearman_values,
        robustness_summary,
    )
    paired_comparison, paired_summary = build_paired_vs_a2(state_frame, spearman_values)
    sensitivity_summary, sensitivity_validation = build_saturation_sensitivity(
        state_frame,
        spearman_values,
        int(availability["missing_A3_state_id"]),
    )
    effective_stage_distribution = {
        str(int(stage)): int(np.count_nonzero(a3_effective_stage == stage))
        for stage in sorted(np.unique(a3_effective_stage))
    }
    validation: dict[str, Any] = {
        "scenario": 2,
        "study": "A1 A2 A3 and equal-weight fingerprint fusion",
        "state_count": STATE_COUNT,
        "lut_types": list(LUT_TYPES),
        "lut_definitions": {
            lut_type: [[int(stage), float(weight)] for stage, weight in definition]
            for lut_type, definition in LUT_DEFINITIONS.items()
        },
        "fusion_layer": "common-B fingerprint layer",
        "raw_waveform_averaging": False,
        "continuous_weight_search": False,
        "A1_available": availability["A1_available_state_count"],
        "A2_available": availability["A2_available_state_count"],
        "A3_available": availability["A3_available_state_count"],
        "missing_A3_state_id": availability["missing_A3_state_id"],
        "missing_A3_state": availability["missing_A3_state"],
        "A3_saturation_rule": "effective_A3 = min(3, Ns)",
        "A3_effective_stage_distribution": effective_stage_distribution,
        "fusion_deduplication": False,
        "X_A_used_as_LUT": False,
        "Real_B_used_as_LUT": False,
        "Real_B_used_as_reference": True,
        "reference_ranking": "R_RR = frozen Real-B to Real-B",
        "requested_C_stages": list(range(1, NMAX + 1)),
        "C_statewise_saturation": True,
        "C_effective_rule": "effective_C_n = min(requested_C_n, Ns)",
        "ridge_lambda": RIDGE_LAMBDA,
        "ridge_rescanned": False,
        "mp_changed": False,
        "canonical_changed": False,
        "common_B_changed": False,
        "D_RR_changed": False,
        "common_B_input_shape": list(frozen.common_B_input.shape),
        "phi_B_shape": list(frozen.phi_B.shape),
        "fingerprint_length": FINGERPRINT_LENGTH,
        "fingerprint_shapes": {
            lut_type: list(value.shape) for lut_type, value in lut_fingerprints.items()
        },
        "all_lut_fingerprints_complex128_finite": bool(
            all(
                value.dtype == np.complex128 and np.all(np.isfinite(value))
                for value in lut_fingerprints.values()
            )
        ),
        "fusion_fingerprint_theta_max_abs_errors": fusion_errors,
        "fusion_fingerprint_theta_equivalence_pass": bool(max(fusion_errors.values()) <= 1e-10),
        "query_shapes": {
            f"C{stage}": list(value.shape) for stage, value in query_fingerprints.items()
        },
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
        "candidate_count": STATE_COUNT,
        "candidate_count_per_query": STATE_COUNT,
        "self_match": True,
        "self_match_included": True,
        "spearman_count": int(spearman_long.shape[0]),
        "spearman_total_count": int(spearman_long.shape[0]),
        "expected_spearman_total_count": STATE_COUNT * NMAX * len(LUT_TYPES),
        "all_spearman_finite": bool(np.all(np.isfinite(spearman_long["spearman"]))),
        "spearman_range_min": float(spearman_long["spearman"].min()),
        "spearman_range_max": float(spearman_long["spearman"].max()),
        "previous_fusion_baseline_regression": baseline,
        "baseline_regression_pass": bool(baseline["pass"]),
        "winner_rule": [
            "worst_C_median",
            "statewise_worst_C_median",
            "mean_C_median",
            "std_C_median",
        ],
        "selected_robust_lut_type": selected_lut,
        "robustness_ranking": robustness_ranked.to_dict("records"),
        "saturation_sensitivity": sensitivity_validation,
        "figure_count": 2,
        "figure3_generated": False,
        "models_retrained": False,
        "raw_data_read": False,
    }
    if validation["distance_matrix_count"] != 35 or validation["ranking_matrix_count"] != 35:
        raise RuntimeError("35个距离/排名矩阵数量错误")
    if validation["spearman_total_count"] != 14875:
        raise RuntimeError("14875个Spearman数量错误")
    if not validation["all_spearman_finite"]:
        raise RuntimeError("Spearman包含非有限值")
    return A123FusionResult(
        loaded=loaded,
        state_ids=loaded.state_ids,
        common_B_input=frozen.common_B_input,
        phi_B=frozen.phi_B,
        theta_effective_A=theta_effective,
        A3_effective_stage_per_state=a3_effective_stage,
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
        paired_vs_A2_by_state=paired_comparison,
        paired_vs_A2_summary=paired_summary,
        saturation_sensitivity_summary=sensitivity_summary,
        fusion_equivalence_errors=fusion_errors,
        baseline_regression=baseline,
        sensitivity_validation=sensitivity_validation,
        validation=validation,
    )


__all__ = [
    "A123FusionResult",
    "LUT_DEFINITIONS",
    "LUT_SUFFIXES",
    "LUT_TYPES",
    "NEW_LUT_TYPES",
    "NMAX",
    "RIDGE_LAMBDA",
    "STATE_COLUMNS",
    "STATE_COUNT",
    "build_a123_distance_ranking",
    "build_a123_lut_fingerprints",
    "build_a123_robustness_tables",
    "build_a123_spearman_tables",
    "build_effective_a_thetas",
    "build_paired_vs_a2",
    "build_saturation_sensitivity",
    "choose_a123_robust_lut",
    "identify_missing_a3_state",
    "run_a123_y_lut_fusion_analysis",
]
