"""Scenario 2 非等权全局 Y-A LUT 指纹融合与独立 Validation 分析。

本模块复用已完成的 A123 common-B 指纹和 C1...C5 Query，先在 340 个
Development Query state 上扫描全局 simplex 权重，再冻结唯一权重，最后只用
85 个 Validation Query state 比较 Y-A2 与 Weighted Fusion。LUT candidate 始终
保留全部 425 个状态；Real-B 仅作为冻结的 R_RR reference。
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from functools import cmp_to_key
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .distance_matrix import compute_cnmse_distance_matrix
from .fingerprint_builder import FINGERPRINT_LENGTH
from .ranking import spearman_statistic
from .scenario2_all_ilc_analysis import load_frozen_scenario2_artifacts

STATE_COUNT = 425
NMAX = 5
BASE_LUT_TYPES = ("Y-A1", "Y-A2", "Y-A3")
FORMAL_LUT_TYPES = ("Y-A2", "Y-A-Weighted")
NEW_LUT_TYPES = ("Y-A3", "Y-A12", "Y-A13", "Y-A23", "Y-A123")
SPLIT_SEED = 20260827
DEVELOPMENT_COUNT = 340
VALIDATION_COUNT = 85
COARSE_STEP = 0.05
FINE_STEP = 0.01
METRIC_TOLERANCE = 1e-6
Q05_DEGRADATION_LIMIT = 0.002
STATE_COLUMNS = ("state_id", "funMng", "funAng", "secMng", "secAng", "Vm", "Pin")
WEIGHT_COLUMNS = ("w1_A1", "w2_A2", "w3_A3")
METRIC_COLUMNS = (
    "C1_median",
    "C2_median",
    "C3_median",
    "C4_median",
    "C5_median",
    "worst_C_median",
    "statewise_worst_C_median",
    "mean_C_median",
    "std_C_median",
    "range_C_median",
    "statewise_worst_q05",
    "statewise_worst_q25",
    "statewise_worst_mean",
    "statewise_worst_min",
)


@dataclass(frozen=True)
class A123Inputs:
    """来自上一轮 A123 输出的冻结指纹、Query 和 reference。"""

    source_root: Path
    state_ids: np.ndarray
    state_frame: pd.DataFrame
    base_fingerprints: np.ndarray
    query_fingerprints: dict[int, np.ndarray]
    effective_C_n: np.ndarray
    reference_ranking: np.ndarray
    previous_distance_matrices: dict[str, np.ndarray]
    previous_ranking_matrices: dict[str, np.ndarray]
    previous_spearman_long: pd.DataFrame
    previous_robustness: pd.DataFrame
    common_B_input: np.ndarray


@dataclass(frozen=True)
class ScanTerms:
    """Development Query 与三类基础 LUT 的预计算内积项。"""

    dev_ids: np.ndarray
    query_energy: dict[int, np.ndarray]
    query_base_inner: dict[int, np.ndarray]
    candidate_gram: np.ndarray
    reference_dev: np.ndarray


@dataclass(frozen=True)
class WeightedFusionResult:
    """完整 weighted fusion 运行结果。"""

    inputs: A123Inputs
    split_definition: pd.DataFrame
    development_ids: np.ndarray
    validation_ids: np.ndarray
    weight_grid: pd.DataFrame
    development_scan: pd.DataFrame
    development_ranking: pd.DataFrame
    selected_weight: dict[str, Any]
    selected_weights: np.ndarray
    selected_weight_fingerprint: np.ndarray
    validation_distance_matrices: dict[str, np.ndarray]
    validation_ranking_matrices: dict[str, np.ndarray]
    validation_spearman: dict[tuple[int, str], np.ndarray]
    validation_long: pd.DataFrame
    validation_summary: pd.DataFrame
    validation_paired: pd.DataFrame
    tail_analysis: pd.DataFrame
    condition_region_analysis: pd.DataFrame
    saturation_sensitivity: pd.DataFrame
    validation_success: dict[str, Any]
    baseline_regression: dict[str, Any]
    validation: dict[str, Any]


def _validate_state_ids(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (STATE_COUNT,) or not np.issubdtype(array.dtype, np.integer):
        raise ValueError("state_ids必须是425个整数")
    array = array.astype(np.int64, copy=False)
    if not np.array_equal(array, np.arange(STATE_COUNT, dtype=np.int64)):
        raise ValueError("state_ids必须严格为0...424")
    return array


def _validate_fingerprint_array(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    expected = (STATE_COUNT, FINGERPRINT_LENGTH)
    if array.shape != expected or array.dtype != np.complex128:
        raise ValueError(
            f"{name}必须是complex128且shape={expected}，实际{array.shape}/{array.dtype}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name}包含非有限值")
    return array


def _load_state_frame(source_root: Path) -> pd.DataFrame:
    path = source_root / "spearman_by_state_long.csv"
    if not path.is_file():
        raise FileNotFoundError(f"缺少A123 state metadata：{path}")
    frame = pd.read_csv(path)
    missing = [column for column in STATE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"A123长表缺少state字段：{missing}")
    state_frame = (
        frame.loc[:, list(STATE_COLUMNS)]
        .drop_duplicates("state_id")
        .sort_values("state_id")
        .reset_index(drop=True)
    )
    _validate_state_ids(state_frame["state_id"].to_numpy(dtype=np.int64))
    return state_frame


def _load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少npz输入：{path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def load_a123_inputs(
    a123_root: Path,
    reference_root: Path,
) -> A123Inputs:
    """只读取 A123 结果和冻结 Real-B reference，不读取 raw。"""

    a123_root = Path(a123_root)
    lut = _load_npz_dict(a123_root / "lut_fingerprints.npz")
    query = _load_npz_dict(a123_root / "query_fingerprints.npz")
    required_lut = {
        "state_ids",
        "common_B_input",
        "Y_A1_fingerprints",
        "Y_A2_fingerprints",
        "Y_A3_fingerprints",
        "A3_effective_stage_per_state",
    }
    missing = sorted(required_lut - set(lut))
    if missing:
        raise ValueError(f"A123 lut_fingerprints缺少字段：{missing}")
    state_ids = _validate_state_ids(lut["state_ids"])
    common_B_input = np.asarray(lut["common_B_input"])
    if common_B_input.shape != (4915,) or common_B_input.dtype != np.complex128:
        raise ValueError("A123 common_B_input必须是complex128(4915,)")
    base = np.stack(
        [
            _validate_fingerprint_array(lut["Y_A1_fingerprints"], "Y_A1"),
            _validate_fingerprint_array(lut["Y_A2_fingerprints"], "Y_A2"),
            _validate_fingerprint_array(lut["Y_A3_fingerprints"], "Y_A3"),
        ],
        axis=0,
    )
    query_fingerprints: dict[int, np.ndarray] = {}
    for stage in range(1, NMAX + 1):
        key = f"Q_C{stage}"
        if key not in query:
            raise ValueError(f"A123 query_fingerprints缺少{key}")
        query_fingerprints[stage] = _validate_fingerprint_array(query[key], key)
    effective_C_n = np.asarray(query.get("effective_C_n", np.empty((0, 0))))
    if effective_C_n.shape != (STATE_COUNT, NMAX) or not np.issubdtype(
        effective_C_n.dtype, np.integer
    ):
        raise ValueError("A123 effective_C_n必须是(425,5)整数数组")
    effective_C_n = effective_C_n.astype(np.int64, copy=False)
    frozen = load_frozen_scenario2_artifacts(Path(reference_root))
    if not np.array_equal(frozen.state_ids, state_ids):
        raise ValueError("A123 state_ids与冻结reference不一致")
    if not np.array_equal(frozen.common_B_input, common_B_input):
        raise ValueError("A123 common_B_input与冻结reference不一致")
    previous_distances = _load_npz_dict(a123_root / "distance_matrices.npz")
    previous_rankings = _load_npz_dict(a123_root / "ranking_matrices.npz")
    previous_long = pd.read_csv(a123_root / "spearman_by_state_long.csv")
    previous_robustness = pd.read_csv(a123_root / "lut_robustness_summary.csv")
    return A123Inputs(
        source_root=a123_root,
        state_ids=state_ids,
        state_frame=_load_state_frame(a123_root),
        base_fingerprints=base,
        query_fingerprints=query_fingerprints,
        effective_C_n=effective_C_n,
        reference_ranking=np.asarray(frozen.R_RR, dtype=np.float64),
        previous_distance_matrices=previous_distances,
        previous_ranking_matrices=previous_rankings,
        previous_spearman_long=previous_long,
        previous_robustness=previous_robustness,
        common_B_input=common_B_input,
    )


def build_query_split(
    state_frame: pd.DataFrame,
    *,
    seed: int = SPLIT_SEED,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """按 (funMng, funAng) 分层并在组内确定性随机划分 340/85 Query states。"""

    if state_frame.shape[0] != STATE_COUNT:
        raise ValueError("state_frame必须有425个状态")
    if not np.array_equal(state_frame["state_id"].to_numpy(dtype=np.int64), np.arange(STATE_COUNT)):
        raise ValueError("state_frame必须按state_id=0...424排列")
    angle_counts = state_frame.groupby("funAng").size().sort_index().to_dict()
    if set(angle_counts) != {0, 45, 90, 135, 180, 225, 270, 315}:
        raise ValueError(f"funAng分层不符合预期：{angle_counts}")
    # 20% largest-remainder allocation, with the extra state assigned to 135°
    # so the three previously identified sensitive angles remain represented.
    validation_targets = {
        int(angle): int(round(count * 0.2)) for angle, count in angle_counts.items()
    }
    current_total = sum(validation_targets.values())
    if current_total != VALIDATION_COUNT:
        validation_targets[135] += VALIDATION_COUNT - current_total
    validation_ids: list[int] = []
    for angle, target in validation_targets.items():
        angle_groups = [
            (int(fun_mng), group)
            for (fun_mng, _), group in state_frame[state_frame["funAng"] == angle].groupby(
                ["funMng", "funAng"], sort=True
            )
        ]
        if not angle_groups:
            raise RuntimeError(f"funAng={angle}没有可分层状态")
        base_count, remainder = divmod(target, len(angle_groups))
        group_rng = np.random.default_rng(int(seed) + int(angle) * 1009)
        extra_group_positions = set(group_rng.permutation(len(angle_groups))[:remainder].tolist())
        for position, (fun_mng, group) in enumerate(angle_groups):
            group_target = base_count + int(position in extra_group_positions)
            group = group.sort_values(["secMng", "secAng", "state_id"])
            state_rng = np.random.default_rng(int(seed) + int(angle) * 1009 + int(fun_mng) * 9176)
            order = state_rng.permutation(group.shape[0])
            validation_ids.extend(group.iloc[order[:group_target]]["state_id"].astype(int).tolist())
    validation = np.sort(np.asarray(validation_ids, dtype=np.int64))
    development = np.setdiff1d(np.arange(STATE_COUNT, dtype=np.int64), validation)
    if development.size != DEVELOPMENT_COUNT or validation.size != VALIDATION_COUNT:
        raise RuntimeError("Development/Validation数量错误")
    if np.intersect1d(development, validation).size != 0:
        raise RuntimeError("Development与Validation存在重叠")
    split = state_frame.copy()
    split["split"] = np.where(
        np.isin(split["state_id"].to_numpy(dtype=np.int64), validation),
        "Validation",
        "Development",
    )
    return split, development, validation


def _canonical_weight(weight: Iterable[float]) -> tuple[float, float, float]:
    values = tuple(round(float(value), 12) for value in weight)
    if len(values) != 3:
        raise ValueError("权重必须有三个分量")
    if any(value < -1e-12 for value in values) or abs(sum(values) - 1.0) > 1e-10:
        raise ValueError(f"非法simplex权重：{values}")
    values = tuple(0.0 if abs(value) < 1e-12 else value for value in values)
    return values


def build_coarse_weight_grid(step: float = COARSE_STEP) -> pd.DataFrame:
    """生成 0.05 simplex 网格并加入 A123 anchor。"""

    if step <= 0 or abs(step * 20 - 1.0) > 1e-12:
        raise ValueError("本轮coarse step必须严格为0.05")
    weights: list[tuple[float, float, float]] = []
    for i in range(21):
        for j in range(21 - i):
            weights.append(_canonical_weight((i * step, j * step, 1.0 - (i + j) * step)))
    weights.append(_canonical_weight((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)))
    unique = sorted(set(weights))
    rows: list[dict[str, Any]] = []
    anchors = {
        (1.0, 0.0, 0.0): "A1",
        (0.0, 1.0, 0.0): "A2",
        (0.0, 0.0, 1.0): "A3",
        (0.5, 0.5, 0.0): "A12",
        (0.5, 0.0, 0.5): "A13",
        (0.0, 0.5, 0.5): "A23",
        _canonical_weight((1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0)): "A123",
    }
    for index, weight in enumerate(unique):
        rows.append(
            {
                "weight_id": f"coarse_{index:04d}",
                "grid_phase": "coarse",
                "w1_A1": weight[0],
                "w2_A2": weight[1],
                "w3_A3": weight[2],
                "is_anchor": weight in anchors,
                "anchor_name": anchors.get(weight, ""),
            }
        )
    return pd.DataFrame(rows)


def build_fine_weight_grid(
    coarse_winner: np.ndarray,
    coarse_weights: pd.DataFrame,
    step: float = FINE_STEP,
) -> pd.DataFrame:
    """围绕 coarse winner 生成 A2 邻域或局部 simplex fine 网格。"""

    if step <= 0 or abs(step * 100 - 1.0) > 1e-12:
        raise ValueError("本轮fine step必须严格为0.01")
    winner = np.asarray(coarse_winner, dtype=float)
    if winner.shape != (3,):
        raise ValueError("coarse winner必须有三个权重")
    if np.allclose(winner, [0.0, 1.0, 0.0], rtol=0, atol=1e-12):
        candidates = [
            _canonical_weight((i * step, j * step, 1.0 - (i + j) * step))
            for i in range(11)
            for j in range(11 - i)
        ]
    else:
        w1_values = np.arange(max(0.0, winner[0] - 0.05), min(1.0, winner[0] + 0.05) + 0.0001, step)
        w2_values = np.arange(max(0.0, winner[1] - 0.05), min(1.0, winner[1] + 0.05) + 0.0001, step)
        candidates = []
        for w1 in w1_values:
            for w2 in w2_values:
                w3 = 1.0 - float(w1) - float(w2)
                if w3 >= -1e-10:
                    candidates.append(_canonical_weight((w1, w2, w3)))
    existing = {
        _canonical_weight(row[list(WEIGHT_COLUMNS)].to_numpy(dtype=float))
        for _, row in coarse_weights.iterrows()
    }
    unique = sorted(set(candidates) - existing)
    rows = [
        {
            "weight_id": f"fine_{index:04d}",
            "grid_phase": "fine",
            "w1_A1": weight[0],
            "w2_A2": weight[1],
            "w3_A3": weight[2],
            "is_anchor": False,
            "anchor_name": "",
        }
        for index, weight in enumerate(unique)
    ]
    return pd.DataFrame(
        rows, columns=["weight_id", "grid_phase", *WEIGHT_COLUMNS, "is_anchor", "anchor_name"]
    )


def _precompute_scan_terms(
    base_fingerprints: np.ndarray,
    query_fingerprints: dict[int, np.ndarray],
    reference_ranking: np.ndarray,
    development_ids: np.ndarray,
) -> ScanTerms:
    if base_fingerprints.shape != (3, STATE_COUNT, FINGERPRINT_LENGTH):
        raise ValueError("base_fingerprints必须是(3,425,4913)")
    candidate_gram = np.empty((STATE_COUNT, 3, 3), dtype=np.complex128)
    for left in range(3):
        for right in range(3):
            candidate_gram[:, left, right] = np.sum(
                base_fingerprints[left] * np.conj(base_fingerprints[right]),
                axis=1,
                dtype=np.complex128,
            )
    query_energy: dict[int, np.ndarray] = {}
    query_base_inner: dict[int, np.ndarray] = {}
    for stage in range(1, NMAX + 1):
        query = query_fingerprints[stage][development_ids]
        query_energy[stage] = np.sum(np.abs(query) ** 2, axis=1, dtype=np.float64)
        query_base_inner[stage] = np.stack(
            [query @ base_fingerprints[index].conj().T for index in range(3)],
            axis=0,
        )
    return ScanTerms(
        dev_ids=development_ids,
        query_energy=query_energy,
        query_base_inner=query_base_inner,
        candidate_gram=candidate_gram,
        reference_dev=reference_ranking[development_ids],
    )


def _distance_from_scan_terms(
    query_energy: np.ndarray,
    query_base_inner: np.ndarray,
    candidate_gram: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    weighted_inner = np.einsum("a,aij->ij", weights, query_base_inner, optimize=True)
    candidate_energy = np.real(
        np.einsum("a,jab,b->j", weights, candidate_gram, weights, optimize=True)
    )
    if np.any(query_energy <= 0) or np.any(candidate_energy < 0):
        raise ValueError("指纹能量必须为正")
    with np.errstate(over="ignore", invalid="ignore"):
        squared_error = (
            query_energy[:, None] + candidate_energy[None, :] - 2.0 * np.real(weighted_inner)
        )
    scale = query_energy[:, None] + candidate_energy[None, :] + 1.0
    tolerance = 32.0 * np.finfo(np.float64).eps * scale
    squared_error[np.abs(squared_error) <= tolerance] = 0.0
    squared_error = np.maximum(squared_error, 0.0)
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
        raise RuntimeError("优化CNMSE矩阵包含NaN或+Inf")
    return distance.astype(np.float64, copy=False)


def _aggregate_spearman_metrics(
    values_by_c: dict[int, np.ndarray],
    c_stages: tuple[int, ...] = tuple(range(1, NMAX + 1)),
) -> dict[str, float]:
    values = np.vstack([values_by_c[stage] for stage in c_stages])
    if values.ndim != 2 or not np.all(np.isfinite(values)):
        raise ValueError("Spearman聚合输入必须全部finite")
    medians = np.median(values, axis=1)
    statewise_worst = np.min(values, axis=0)
    return {
        **{f"C{stage}_median": float(medians[index]) for index, stage in enumerate(c_stages)},
        "worst_C_median": float(np.min(medians)),
        "statewise_worst_C_median": float(np.median(statewise_worst)),
        "mean_C_median": float(np.mean(medians)),
        "std_C_median": float(np.std(medians, ddof=0)),
        "range_C_median": float(np.max(medians) - np.min(medians)),
        "statewise_worst_q05": float(np.quantile(statewise_worst, 0.05)),
        "statewise_worst_q25": float(np.quantile(statewise_worst, 0.25)),
        "statewise_worst_mean": float(np.mean(statewise_worst)),
        "statewise_worst_min": float(np.min(statewise_worst)),
    }


def evaluate_development_weight(weights: np.ndarray, terms: ScanTerms) -> dict[str, Any]:
    """只用 Development Query 计算一个权重的五阶段鲁棒指标。"""

    weight = np.asarray(weights, dtype=np.float64)
    if weight.shape != (3,) or np.any(weight < -1e-12) or abs(float(weight.sum()) - 1.0) > 1e-10:
        raise ValueError("Development权重不满足simplex约束")
    values_by_c: dict[int, np.ndarray] = {}
    for stage in range(1, NMAX + 1):
        distance = _distance_from_scan_terms(
            terms.query_energy[stage],
            terms.query_base_inner[stage],
            terms.candidate_gram,
            weight,
        )
        ranks = rankdata(distance, axis=1, method="average")
        values_by_c[stage] = np.asarray(
            [
                spearman_statistic(terms.reference_dev[index], ranks[index])
                for index in range(terms.dev_ids.size)
            ],
            dtype=np.float64,
        )
    metrics = _aggregate_spearman_metrics(values_by_c)
    return {"weights": weight.copy(), **metrics}


def scan_development_weights(weight_grid: pd.DataFrame, terms: ScanTerms) -> pd.DataFrame:
    """扫描全部 Development-only 权重网格，释放每个权重的大矩阵。"""

    rows: list[dict[str, Any]] = []
    for row in weight_grid.itertuples(index=False):
        weights = np.asarray([row.w1_A1, row.w2_A2, row.w3_A3], dtype=np.float64)
        result = evaluate_development_weight(weights, terms)
        rows.append(
            {
                "weight_id": row.weight_id,
                "grid_phase": row.grid_phase,
                "w1": float(weights[0]),
                "w2": float(weights[1]),
                "w3": float(weights[2]),
                "is_anchor": bool(row.is_anchor),
                "anchor_name": str(row.anchor_name),
                "development_state_count": int(terms.dev_ids.size),
                **{column: result[column] for column in METRIC_COLUMNS},
            }
        )
    return pd.DataFrame(rows)


def compare_weight_records(
    left: pd.Series | dict[str, Any], right: pd.Series | dict[str, Any]
) -> int:
    """按冻结优先级比较两个 Development 权重；返回负数表示left优先。"""

    def value(record: pd.Series | dict[str, Any], key: str) -> float:
        return float(record[key])

    descending = ("worst_C_median", "statewise_worst_C_median", "mean_C_median")
    for key in descending:
        difference = value(right, key) - value(left, key)
        if abs(difference) > METRIC_TOLERANCE:
            return 1 if difference > 0 else -1
    difference = value(left, "std_C_median") - value(right, "std_C_median")
    if abs(difference) > METRIC_TOLERANCE:
        return -1 if difference < 0 else 1
    left_distance = float(
        np.linalg.norm(np.asarray([value(left, "w1"), value(left, "w2") - 1.0, value(left, "w3")]))
    )
    right_distance = float(
        np.linalg.norm(
            np.asarray([value(right, "w1"), value(right, "w2") - 1.0, value(right, "w3")])
        )
    )
    if abs(left_distance - right_distance) > 1e-12:
        return -1 if left_distance < right_distance else 1
    left_id = str(left["weight_id"])
    right_id = str(right["weight_id"])
    if left_id < right_id:
        return -1
    if left_id > right_id:
        return 1
    return 0


def rank_development_weights(scan: pd.DataFrame) -> pd.DataFrame:
    """按冻结 winner rule 排序 Development 扫描表。"""

    records = scan.to_dict("records")
    records.sort(key=cmp_to_key(compare_weight_records))
    ranked = pd.DataFrame(records)
    ranked.insert(0, "development_rank", np.arange(1, ranked.shape[0] + 1))
    return ranked


def select_development_weight(
    scan: pd.DataFrame, *, phase: str | None = None
) -> tuple[pd.Series, pd.DataFrame]:
    subset = scan if phase is None else scan[scan["grid_phase"] == phase]
    if subset.empty:
        raise ValueError("没有可选择的Development权重")
    ranked = rank_development_weights(subset.reset_index(drop=True))
    return ranked.iloc[0], ranked


def _max_difference_allowing_neginf(left: np.ndarray, right: np.ndarray) -> tuple[float, int]:
    a = np.asarray(left)
    b = np.asarray(right)
    if a.shape != b.shape:
        return float("inf"), max(a.size, b.size)
    pattern_mismatch = int(np.count_nonzero(np.isneginf(a) != np.isneginf(b)))
    finite_a = np.isfinite(a)
    finite_b = np.isfinite(b)
    pattern_mismatch += int(np.count_nonzero(finite_a != finite_b))
    finite = finite_a & finite_b
    if not np.any(finite):
        return 0.0, pattern_mismatch
    return float(np.max(np.abs(a[finite] - b[finite]))), pattern_mismatch


def regression_against_a123(inputs: A123Inputs) -> dict[str, Any]:
    """完整回归 A1/A2/A12 指纹、矩阵、排名和 Spearman。"""

    candidate_fingerprints = {
        "Y-A1": inputs.base_fingerprints[0],
        "Y-A2": inputs.base_fingerprints[1],
        "Y-A12-Mean": 0.5 * (inputs.base_fingerprints[0] + inputs.base_fingerprints[1]),
    }
    old_suffix = {"Y-A1": "A1", "Y-A2": "A2", "Y-A12-Mean": "A12"}
    fingerprint_errors: list[float] = []
    distance_errors: list[float] = []
    ranking_errors: list[float] = []
    spearman_errors: list[float] = []
    distance_mismatch = 0
    ranking_mismatch = 0
    for lut_type, fingerprint in candidate_fingerprints.items():
        old_key = {
            "Y-A1": "Y_A1_fingerprints",
            "Y-A2": "Y_A2_fingerprints",
            "Y-A12-Mean": "Y_A12_Mean_fingerprints",
        }[lut_type]
        # A123 stores the corresponding arrays under these stable names.
        old_npz = _load_npz_dict(inputs.source_root / "lut_fingerprints.npz")
        fingerprint_errors.append(float(np.max(np.abs(fingerprint - old_npz[old_key]))))
        for stage in range(1, NMAX + 1):
            current_distance = compute_cnmse_distance_matrix(
                inputs.query_fingerprints[stage], fingerprint
            )
            current_ranking = rankdata(current_distance, axis=1, method="average")
            old_distance = inputs.previous_distance_matrices[f"D_C{stage}_{old_suffix[lut_type]}"]
            old_ranking = inputs.previous_ranking_matrices[f"R_C{stage}_{old_suffix[lut_type]}"]
            distance_error, mismatch = _max_difference_allowing_neginf(
                current_distance, old_distance
            )
            ranking_error, ranking_pattern = _max_difference_allowing_neginf(
                current_ranking, old_ranking
            )
            distance_errors.append(distance_error)
            ranking_errors.append(ranking_error)
            distance_mismatch += mismatch
            ranking_mismatch += ranking_pattern
            current_spearman = np.asarray(
                [
                    spearman_statistic(
                        inputs.reference_ranking[state_id], current_ranking[state_id]
                    )
                    for state_id in range(STATE_COUNT)
                ],
                dtype=float,
            )
            old_spearman = (
                inputs.previous_spearman_long[
                    (inputs.previous_spearman_long["requested_C_n"] == stage)
                    & (inputs.previous_spearman_long["lut_fingerprint_type"] == lut_type)
                ]
                .sort_values("state_id")["spearman"]
                .to_numpy(dtype=float)
            )
            if old_spearman.shape != (STATE_COUNT,):
                raise ValueError(f"上一轮{stage}->{lut_type} Spearman不是425行")
            spearman_errors.append(float(np.max(np.abs(current_spearman - old_spearman))))
    old_robustness = inputs.previous_robustness
    current_robustness: dict[str, dict[str, float]] = {}
    for lut_type in ("Y-A1", "Y-A2", "Y-A12-Mean"):
        group = inputs.previous_spearman_long[
            inputs.previous_spearman_long["lut_fingerprint_type"] == lut_type
        ]
        values_by_c = {
            stage: group[group["requested_C_n"] == stage]
            .sort_values("state_id")["spearman"]
            .to_numpy(dtype=float)
            for stage in range(1, NMAX + 1)
        }
        current_robustness[lut_type] = _aggregate_spearman_metrics(values_by_c)
    robustness_errors: list[float] = []
    for lut_type, current in current_robustness.items():
        old = old_robustness[old_robustness["lut_type"] == lut_type].iloc[0]
        for field in (
            "C1_median",
            "C2_median",
            "C3_median",
            "C4_median",
            "C5_median",
            "worst_C_median",
            "mean_C_median",
            "std_C_median",
            "range_C_median",
        ):
            robustness_errors.append(abs(float(current[field]) - float(old[field])))
    result = {
        "fingerprint_max_abs_error": max(fingerprint_errors),
        "distance_max_abs_error": max(distance_errors),
        "ranking_max_abs_error": max(ranking_errors),
        "spearman_max_abs_error": max(spearman_errors),
        "robustness_max_abs_error": max(robustness_errors),
        "distance_nonfinite_pattern_mismatch_count": distance_mismatch,
        "ranking_nonfinite_pattern_mismatch_count": ranking_mismatch,
        "fingerprint_tolerance": 1e-12,
        "distance_diagnostic_tolerance": 1e-6,
        "ranking_tolerance": 1e-12,
        "spearman_tolerance": 1e-12,
        "robustness_tolerance": 1e-12,
    }
    result["pass"] = bool(
        result["fingerprint_max_abs_error"] <= result["fingerprint_tolerance"]
        and result["distance_max_abs_error"] <= result["distance_diagnostic_tolerance"]
        and result["ranking_max_abs_error"] <= result["ranking_tolerance"]
        and result["spearman_max_abs_error"] <= result["spearman_tolerance"]
        and result["robustness_max_abs_error"] <= result["robustness_tolerance"]
        and distance_mismatch == 0
        and ranking_mismatch == 0
    )
    if not result["pass"]:
        raise RuntimeError(f"A123 A1/A2/A12回归失败：{result}")
    return result


def build_weighted_fingerprint(base_fingerprints: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weight = np.asarray(weights, dtype=np.float64)
    if weight.shape != (3,) or np.any(weight < -1e-12) or abs(float(weight.sum()) - 1.0) > 1e-10:
        raise ValueError("Weighted权重不满足simplex约束")
    fingerprint = np.einsum("a,ajk->jk", weight, base_fingerprints, optimize=True)
    fingerprint = np.asarray(fingerprint, dtype=np.complex128)
    if fingerprint.shape != (STATE_COUNT, FINGERPRINT_LENGTH) or not np.all(
        np.isfinite(fingerprint)
    ):
        raise RuntimeError("Weighted fingerprint shape或finite性错误")
    return fingerprint


def compute_validation_results(
    inputs: A123Inputs,
    validation_ids: np.ndarray,
    weighted_fingerprint: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[tuple[int, str], np.ndarray]]:
    """冻结权重后，只对 Validation Query 计算 A2/Weighted 的10个矩阵。"""

    distances, rankings, spearman_values = compute_fixed_pair_results(
        inputs, validation_ids, weighted_fingerprint
    )
    for key, value in distances.items():
        if value.shape != (VALIDATION_COUNT, STATE_COUNT):
            raise RuntimeError(f"{key} shape错误：{value.shape}")
    return distances, rankings, spearman_values


def compute_fixed_pair_results(
    inputs: A123Inputs,
    query_ids: np.ndarray,
    weighted_fingerprint: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[tuple[int, str], np.ndarray]]:
    """对给定 Query state 集合计算冻结 A2/Weighted 的距离、排名和Spearman。"""

    query_ids = np.asarray(query_ids, dtype=np.int64)
    if (
        query_ids.ndim != 1
        or query_ids.size < 1
        or np.any(query_ids < 0)
        or np.any(query_ids >= STATE_COUNT)
    ):
        raise ValueError("query_ids必须是合法的非空state id一维数组")
    candidates = {"Y-A2": inputs.base_fingerprints[1], "Y-A-Weighted": weighted_fingerprint}
    distances: dict[str, np.ndarray] = {}
    rankings: dict[str, np.ndarray] = {}
    spearman_values: dict[tuple[int, str], np.ndarray] = {}
    for stage in range(1, NMAX + 1):
        query = inputs.query_fingerprints[stage][query_ids]
        reference = inputs.reference_ranking[query_ids]
        for lut_type, candidate in candidates.items():
            suffix = "A2" if lut_type == "Y-A2" else "Weighted"
            distance_key = f"D_C{stage}_{suffix}"
            ranking_key = f"R_C{stage}_{suffix}"
            distance = compute_cnmse_distance_matrix(query, candidate)
            ranking = rankdata(distance, axis=1, method="average").astype(np.float64)
            values = np.asarray(
                [
                    spearman_statistic(reference[index], ranking[index])
                    for index in range(query_ids.size)
                ],
                dtype=np.float64,
            )
            if (
                np.isnan(distance).any()
                or np.isposinf(distance).any()
                or not np.all(np.isfinite(ranking))
            ):
                raise RuntimeError(f"{distance_key}/{ranking_key}包含非法值")
            distances[distance_key] = distance
            rankings[ranking_key] = ranking
            spearman_values[(stage, lut_type)] = values
    return distances, rankings, spearman_values


def build_validation_tables(
    inputs: A123Inputs,
    validation_ids: np.ndarray,
    spearman_values: dict[tuple[int, str], np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    state_frame = inputs.state_frame.set_index("state_id").to_dict("index")
    long_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    for stage in range(1, NMAX + 1):
        for lut_type in FORMAL_LUT_TYPES:
            values = spearman_values[(stage, lut_type)]
            for index, state_id in enumerate(validation_ids):
                item = state_frame[int(state_id)]
                long_rows.append(
                    {
                        "state_id": int(state_id),
                        "funMng": int(item["funMng"]),
                        "funAng": int(item["funAng"]),
                        "secMng": int(item["secMng"]),
                        "secAng": int(item["secAng"]),
                        "Vm": float(item["Vm"]),
                        "Pin": float(item["Pin"]),
                        "C_stage": stage,
                        "lut_type": lut_type,
                        "spearman": float(values[index]),
                    }
                )
            summary_rows.append(
                {
                    "C_stage": stage,
                    "lut_type": lut_type,
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
    long_table = pd.DataFrame(long_rows)
    summary = pd.DataFrame(summary_rows)
    robust_rows: list[dict[str, Any]] = []
    for lut_type in FORMAL_LUT_TYPES:
        metrics = _aggregate_spearman_metrics(
            {stage: spearman_values[(stage, lut_type)] for stage in range(1, NMAX + 1)}
        )
        robust_rows.append({"lut_type": lut_type, "state_count": VALIDATION_COUNT, **metrics})
    return long_table, summary, pd.DataFrame(robust_rows)


def build_validation_paired_table(
    validation_ids: np.ndarray,
    spearman_values: dict[tuple[int, str], np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stage in range(1, NMAX + 1):
        a2 = spearman_values[(stage, "Y-A2")]
        weighted = spearman_values[(stage, "Y-A-Weighted")]
        delta = weighted - a2
        for index, state_id in enumerate(validation_ids):
            rows.append(
                {
                    "state_id": int(state_id),
                    "C_stage": stage,
                    "rho_A2": float(a2[index]),
                    "rho_weighted": float(weighted[index]),
                    "delta": float(delta[index]),
                }
            )
    return pd.DataFrame(rows)


def _metrics_for_subset(
    spearman_values: dict[tuple[int, str], np.ndarray],
    lut_type: str,
    state_mask: np.ndarray,
    c_stages: tuple[int, ...],
) -> dict[str, float]:
    values = {stage: spearman_values[(stage, lut_type)][state_mask] for stage in c_stages}
    return _aggregate_spearman_metrics(values, c_stages)


def build_saturation_sensitivity_table(
    inputs: A123Inputs,
    validation_ids: np.ndarray,
    spearman_values: dict[tuple[int, str], np.ndarray],
    missing_state_id: int,
    full_spearman_values: dict[tuple[int, str], np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    local_state_ids = np.asarray(validation_ids, dtype=np.int64)
    full_state_ids = np.arange(STATE_COUNT, dtype=np.int64)
    full_mask = np.ones(full_state_ids.size, dtype=bool)
    exclude_mask = full_state_ids != int(missing_state_id)
    cases = [
        (
            "state374_sensitivity",
            "full_425_states",
            full_state_ids,
            full_mask,
            full_spearman_values,
            tuple(range(1, NMAX + 1)),
        ),
        (
            "state374_sensitivity",
            "exclude_state374",
            full_state_ids,
            exclude_mask,
            full_spearman_values,
            tuple(range(1, NMAX + 1)),
        ),
        (
            "C1_C3_sensitivity",
            "validation_C1_C3",
            local_state_ids,
            np.ones(local_state_ids.size, dtype=bool),
            spearman_values,
            (1, 2, 3),
        ),
    ]
    for analysis, scope, case_state_ids, state_mask, case_values, c_stages in cases:
        metric_rows: dict[str, dict[str, float]] = {}
        for lut_type in FORMAL_LUT_TYPES:
            metric_rows[lut_type] = _metrics_for_subset(case_values, lut_type, state_mask, c_stages)
        winner = max(
            FORMAL_LUT_TYPES,
            key=lambda lut_type: (
                metric_rows[lut_type]["worst_C_median"],
                metric_rows[lut_type]["statewise_worst_C_median"],
                metric_rows[lut_type]["mean_C_median"],
                -metric_rows[lut_type]["std_C_median"],
            ),
        )
        for lut_type, metrics in metric_rows.items():
            rows.append(
                {
                    "analysis": analysis,
                    "sample_scope": scope,
                    "state_count": int(case_state_ids[state_mask].size),
                    "C_stage_set": ",".join(f"C{stage}" for stage in c_stages),
                    "lut_type": lut_type,
                    "winner_by_worst_C": winner,
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def build_tail_analysis(
    validation_ids: np.ndarray,
    spearman_values: dict[tuple[int, str], np.ndarray],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for lut_type in FORMAL_LUT_TYPES:
        values = np.vstack([spearman_values[(stage, lut_type)] for stage in range(1, NMAX + 1)])
        worst = np.min(values, axis=0)
        order = np.argsort(worst)
        for fraction in (0.05, 0.10):
            count = max(1, int(np.ceil(validation_ids.size * fraction)))
            selected = order[:count]
            rows.append(
                {
                    "lut_type": lut_type,
                    "tail_fraction": fraction,
                    "tail_count": count,
                    "state_ids": ",".join(str(int(value)) for value in validation_ids[selected]),
                    "worst_over_C_quantile": float(np.quantile(worst, fraction)),
                    "tail_mean": float(np.mean(worst[selected])),
                    "tail_median": float(np.median(worst[selected])),
                    "tail_min": float(np.min(worst[selected])),
                    "tail_max": float(np.max(worst[selected])),
                }
            )
    return pd.DataFrame(rows)


def build_condition_region_analysis(
    inputs: A123Inputs,
    validation_ids: np.ndarray,
    paired: pd.DataFrame,
) -> pd.DataFrame:
    metadata = inputs.state_frame.set_index("state_id")
    paired = paired.copy()
    paired = paired.join(metadata, on="state_id", rsuffix="_metadata")
    rows: list[dict[str, Any]] = []
    for grouping in ("funAng", "funMng", "secMng", "secAng"):
        for condition, group in paired.groupby(grouping, sort=True):
            delta = group["delta"].to_numpy(dtype=float)
            rows.append(
                {
                    "grouping": grouping,
                    "condition": condition,
                    "count": int(group.shape[0]),
                    "median_rho_A2": float(group["rho_A2"].median()),
                    "median_rho_weighted": float(group["rho_weighted"].median()),
                    "median_delta": float(np.median(delta)),
                    "better_fraction": float(np.mean(delta > 0)),
                    "better_count": int(np.count_nonzero(delta > 0)),
                    "worse_count": int(np.count_nonzero(delta < 0)),
                    "equal_count": int(np.count_nonzero(delta == 0)),
                }
            )
    return pd.DataFrame(rows)


def evaluate_validation_success(
    validation_summary: pd.DataFrame,
) -> dict[str, Any]:
    indexed = validation_summary.set_index("lut_type")
    a2 = indexed.loc["Y-A2"]
    weighted = indexed.loc["Y-A-Weighted"]
    worst_gain = float(weighted["worst_C_median"] - a2["worst_C_median"])
    statewise_gain = float(weighted["statewise_worst_C_median"] - a2["statewise_worst_C_median"])
    q05_gain = float(weighted["statewise_worst_q05"] - a2["statewise_worst_q05"])
    conditions = {
        "worst_C_strictly_better": bool(worst_gain > 0.0),
        "statewise_worst_not_degraded": bool(statewise_gain >= -1e-6),
        "q05_not_degraded_beyond_limit": bool(q05_gain >= -Q05_DEGRADATION_LIMIT),
    }
    return {
        "worst_C_gain": worst_gain,
        "statewise_worst_C_median_gain": statewise_gain,
        "statewise_worst_q05_gain": q05_gain,
        "success_rule": {
            "worst_C_weighted_gt_A2": True,
            "statewise_worst_C_median_weighted_ge_A2_minus": 1e-6,
            "statewise_worst_q05_degradation_limit": Q05_DEGRADATION_LIMIT,
        },
        "conditions": conditions,
        "success": bool(all(conditions.values())),
    }


def run_weighted_fusion_analysis(
    a123_root: Path,
    reference_root: Path,
    *,
    seed: int = SPLIT_SEED,
    on_weights_frozen: Callable[[dict[str, Any]], None] | None = None,
) -> WeightedFusionResult:
    """执行 Development-only 权重选择和冻结后 Validation 比较。"""

    inputs = load_a123_inputs(a123_root, reference_root)
    baseline = regression_against_a123(inputs)
    split_definition, development_ids, validation_ids = build_query_split(
        inputs.state_frame, seed=seed
    )
    terms = _precompute_scan_terms(
        inputs.base_fingerprints,
        inputs.query_fingerprints,
        inputs.reference_ranking,
        development_ids,
    )
    coarse_grid = build_coarse_weight_grid()
    coarse_scan = scan_development_weights(coarse_grid, terms)
    coarse_selected, _coarse_ranked = select_development_weight(coarse_scan)
    coarse_weights = coarse_selected[["w1", "w2", "w3"]].to_numpy(dtype=float)
    fine_grid = build_fine_weight_grid(coarse_weights, coarse_grid)
    fine_scan = (
        scan_development_weights(fine_grid, terms) if not fine_grid.empty else pd.DataFrame()
    )
    combined_scan = pd.concat([coarse_scan, fine_scan], ignore_index=True)
    final_selected, development_ranking = select_development_weight(combined_scan)
    selected_weights = final_selected[["w1", "w2", "w3"]].to_numpy(dtype=np.float64)
    selected_fingerprint = build_weighted_fingerprint(inputs.base_fingerprints, selected_weights)
    selected_weight = {
        "selection_split": "Development",
        "split_seed": seed,
        "split_stratification_keys": ["funMng", "funAng"],
        "coarse_grid_step": COARSE_STEP,
        "fine_grid_step": FINE_STEP,
        "metric_tolerance": METRIC_TOLERANCE,
        "coarse_candidate_count": int(coarse_grid.shape[0]),
        "fine_candidate_count": int(fine_grid.shape[0]),
        "coarse_winner": {
            "weight_id": str(coarse_selected["weight_id"]),
            "w1": float(coarse_selected["w1"]),
            "w2": float(coarse_selected["w2"]),
            "w3": float(coarse_selected["w3"]),
            **{column: float(coarse_selected[column]) for column in METRIC_COLUMNS},
        },
        "selected_weight_id": str(final_selected["weight_id"]),
        "w1": float(selected_weights[0]),
        "w2": float(selected_weights[1]),
        "w3": float(selected_weights[2]),
        "selected_phase": str(final_selected["grid_phase"]),
        "development_selection_metrics": {
            column: float(final_selected[column]) for column in METRIC_COLUMNS
        },
        "winner_rule": [
            "worst_C_median descending",
            "statewise_worst_C_median descending",
            "mean_C_median descending",
            "std_C_median ascending",
            "exact tie prefers distance to A2",
        ],
        "weights_frozen_marker": "WEIGHTS_FROZEN",
        "validation_unlocked_marker": "VALIDATION_UNLOCKED_AFTER_WEIGHT_FREEZE",
    }
    if on_weights_frozen is not None:
        on_weights_frozen(selected_weight)
    validation_distances, validation_rankings, validation_spearman = compute_validation_results(
        inputs, validation_ids, selected_fingerprint
    )
    validation_long, _validation_summary_by_c, validation_summary = build_validation_tables(
        inputs, validation_ids, validation_spearman
    )
    validation_paired = build_validation_paired_table(validation_ids, validation_spearman)
    tail_analysis = build_tail_analysis(validation_ids, validation_spearman)
    condition_region = build_condition_region_analysis(inputs, validation_ids, validation_paired)
    _, _, full_spearman = compute_fixed_pair_results(
        inputs,
        np.arange(STATE_COUNT, dtype=np.int64),
        selected_fingerprint,
    )
    sensitivity = build_saturation_sensitivity_table(
        inputs,
        validation_ids,
        validation_spearman,
        missing_state_id=374,
        full_spearman_values=full_spearman,
    )
    validation_success = evaluate_validation_success(validation_summary)
    weighted_is_a2 = bool(
        np.max(np.abs(selected_fingerprint - inputs.base_fingerprints[1])) <= 1e-12
    )
    validation: dict[str, Any] = {
        "study": "non-equal global weighted Y-A LUT fusion",
        "scenario": 2,
        "state_count": STATE_COUNT,
        "base_fingerprints": ["A1", "A2", "A3"],
        "fusion_layer": "common-B fingerprint",
        "nonnegative_weights": True,
        "sum_to_one": True,
        "raw_waveform_averaging": False,
        "development_count": DEVELOPMENT_COUNT,
        "validation_count": VALIDATION_COUNT,
        "split_seed": seed,
        "split_overlap_count": int(np.intersect1d(development_ids, validation_ids).size),
        "split_union_count": int(np.union1d(development_ids, validation_ids).size),
        "split_balance_by_funAng": {
            str(int(angle)): {
                "Development": int(
                    split_definition[
                        (split_definition["funAng"] == angle)
                        & (split_definition["split"] == "Development")
                    ].shape[0]
                ),
                "Validation": int(
                    split_definition[
                        (split_definition["funAng"] == angle)
                        & (split_definition["split"] == "Validation")
                    ].shape[0]
                ),
            }
            for angle in sorted(split_definition["funAng"].unique())
        },
        "candidate_count": STATE_COUNT,
        "query_split_only": True,
        "coarse_step": COARSE_STEP,
        "fine_step": FINE_STEP,
        "coarse_candidate_count": int(coarse_grid.shape[0]),
        "fine_candidate_count": int(fine_grid.shape[0]),
        "validation_used_for_weight_selection": False,
        "validation_unlocked_marker": "VALIDATION_UNLOCKED_AFTER_WEIGHT_FREEZE",
        "weights_frozen_marker": "WEIGHTS_FROZEN",
        "baseline": "Y-A2",
        "selected_weights": {
            "w1": float(selected_weights[0]),
            "w2": float(selected_weights[1]),
            "w3": float(selected_weights[2]),
        },
        "selected_weight_is_A2": weighted_is_a2,
        "selected_weight_is_A2_dominant": bool(selected_weights[1] > 0.8),
        "winner_rule": [
            "worst_C_median",
            "statewise_worst_C_median",
            "mean_C_median",
            "std_C_median",
        ],
        "validation_success_rule": validation_success["success_rule"],
        "validation_success": validation_success["success"],
        "validation_success_detail": validation_success,
        "A3_saturation_rule": "effective_A3 = min(3, Ns)",
        "A3_missing_state_id": 374,
        "A3_missing_state_effective_stage": 2,
        "C_saturation_rule": "effective_C_n = min(requested_C_n, Ns)",
        "C_requested_stages": list(range(1, NMAX + 1)),
        "ridge_lambda": 1e-8,
        "models_retrained": False,
        "lambda_rescanned": False,
        "mp_changed": False,
        "canonical_changed": False,
        "common_B_changed": False,
        "D_RR_changed": False,
        "X_A_used_as_LUT": False,
        "Real_B_used_as_LUT": False,
        "Real_B_used_as_reference": True,
        "common_B_input_shape": list(inputs.common_B_input.shape),
        "fingerprint_length": FINGERPRINT_LENGTH,
        "selected_weight_fingerprint_shape": list(selected_fingerprint.shape),
        "selected_weight_fingerprint_complex128_finite": bool(
            selected_fingerprint.dtype == np.complex128
            and np.all(np.isfinite(selected_fingerprint))
        ),
        "validation_distance_matrix_count": len(validation_distances),
        "validation_ranking_matrix_count": len(validation_rankings),
        "validation_matrix_shape": [VALIDATION_COUNT, STATE_COUNT],
        "validation_spearman_count": int(validation_long.shape[0]),
        "expected_validation_spearman_count": VALIDATION_COUNT * NMAX * len(FORMAL_LUT_TYPES),
        "validation_all_spearman_finite": bool(np.all(np.isfinite(validation_long["spearman"]))),
        "baseline_regression": baseline,
        "baseline_regression_pass": bool(baseline["pass"]),
        "saturation_sensitivity_winner_unchanged": bool(
            sensitivity[
                (sensitivity["analysis"] == "state374_sensitivity")
                & (sensitivity["sample_scope"] == "full_425_states")
            ]["winner_by_worst_C"].iloc[0]
            == sensitivity[
                (sensitivity["analysis"] == "state374_sensitivity")
                & (sensitivity["sample_scope"] == "exclude_state374")
            ]["winner_by_worst_C"].iloc[0]
        ),
        "figure_count": 3,
        "figure4_generated": False,
        "raw_data_read": False,
    }
    return WeightedFusionResult(
        inputs=inputs,
        split_definition=split_definition,
        development_ids=development_ids,
        validation_ids=validation_ids,
        weight_grid=pd.concat([coarse_grid, fine_grid], ignore_index=True),
        development_scan=combined_scan,
        development_ranking=development_ranking,
        selected_weight=selected_weight,
        selected_weights=selected_weights,
        selected_weight_fingerprint=selected_fingerprint,
        validation_distance_matrices=validation_distances,
        validation_ranking_matrices=validation_rankings,
        validation_spearman=validation_spearman,
        validation_long=validation_long,
        validation_summary=validation_summary,
        validation_paired=validation_paired,
        tail_analysis=tail_analysis,
        condition_region_analysis=condition_region,
        saturation_sensitivity=sensitivity,
        validation_success=validation_success,
        baseline_regression=baseline,
        validation=validation,
    )


__all__ = [
    "A123Inputs",
    "BASE_LUT_TYPES",
    "COARSE_STEP",
    "DEVELOPMENT_COUNT",
    "FINE_STEP",
    "FORMAL_LUT_TYPES",
    "METRIC_COLUMNS",
    "METRIC_TOLERANCE",
    "NEW_LUT_TYPES",
    "NMAX",
    "Q05_DEGRADATION_LIMIT",
    "SPLIT_SEED",
    "STATE_COUNT",
    "VALIDATION_COUNT",
    "WeightedFusionResult",
    "build_coarse_weight_grid",
    "build_condition_region_analysis",
    "build_fine_weight_grid",
    "build_query_split",
    "build_saturation_sensitivity_table",
    "build_tail_analysis",
    "build_validation_paired_table",
    "build_validation_tables",
    "build_weighted_fingerprint",
    "compare_weight_records",
    "compute_validation_results",
    "evaluate_development_weight",
    "evaluate_validation_success",
    "load_a123_inputs",
    "rank_development_weights",
    "regression_against_a123",
    "run_weighted_fusion_analysis",
    "scan_development_weights",
    "select_development_weight",
]
