"""
Scenario 2: Y-C3 Query -> fixed Y-A2 LUT -> Real-B behavior validation.

The module consumes only the already frozen A123 fingerprint outputs and the frozen
Scenario 2 Real-B reference.  It does not retrain models, read raw waveforms, change
canonical preprocessing, or define a new CNMSE implementation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from behavior_fingerprint_ranking_consistency.distance_matrix import (
    compute_cnmse_distance_matrix,
)
from behavior_fingerprint_ranking_consistency.ranking import distance_to_ranks
from behavior_fingerprint_ranking_consistency.scenario2_all_ilc_analysis import (
    load_frozen_scenario2_artifacts,
)
from data_manager import build_state_table

STATE_COUNT = 425
FINGERPRINT_LENGTH = 4913
REQUESTED_C_STAGE = 3
NMAX = 5
THRESHOLD_DB_VALUES = (-20.0, -25.0, -30.0, -35.0, -40.0)
STATE_COLUMNS = ("state_id", "funMng", "funAng", "secMng", "secAng", "Vm", "Pin")


@dataclass(frozen=True)
class RetrievalInputs:
    """Frozen arrays and metadata required for the retrieval experiment."""

    source_root: Path
    reference_root: Path
    all_ilc_root: Path
    state_ids: np.ndarray
    state_frame: pd.DataFrame
    query_c3: np.ndarray
    lut_a2: np.ndarray
    real_b: np.ndarray
    common_b_input: np.ndarray
    effective_c_n: np.ndarray
    ilc_column_counts: np.ndarray
    previous_distance_c3_a2: np.ndarray
    previous_ranking_c3_a2: np.ndarray
    previous_real_b_distance: np.ndarray
    previous_real_b_ranking: np.ndarray
    a123_validation: dict[str, Any]


@dataclass(frozen=True)
class RetrievalResult:
    """In-memory outputs of the complete C3-to-A2 retrieval analysis."""

    inputs: RetrievalInputs
    query_to_a2_distance: np.ndarray
    real_b_distance: np.ndarray
    query_to_a2_ranking: np.ndarray
    real_b_ranking: np.ndarray
    retrieval_results: pd.DataFrame
    real_b_diagnostics: pd.DataFrame
    threshold_sweep: pd.DataFrame
    retrieval_summary: pd.DataFrame
    worst_retrieval_states: pd.DataFrame
    validation: dict[str, Any]


def _load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少NPZ输入：{path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _validate_state_ids(values: np.ndarray, name: str = "state_ids") -> np.ndarray:
    array = np.asarray(values)
    if array.shape != (STATE_COUNT,) or not np.issubdtype(array.dtype, np.integer):
        raise ValueError(f"{name}必须是长度425的整数数组")
    array = array.astype(np.int64, copy=False)
    if not np.array_equal(array, np.arange(STATE_COUNT, dtype=np.int64)):
        raise ValueError(f"{name}必须严格为0...424")
    return array


def _validate_fingerprint(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    expected = (STATE_COUNT, FINGERPRINT_LENGTH)
    if array.shape != expected or array.dtype != np.complex128:
        raise ValueError(
            f"{name}必须是complex128且shape={expected}，"
            f"实际{array.shape}/{array.dtype}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name}包含NaN或Inf")
    return array


def _validate_distance(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    expected = (STATE_COUNT, STATE_COUNT)
    if array.shape != expected or array.dtype != np.float64:
        raise ValueError(f"{name}必须是float64且shape={expected}，实际{array.shape}/{array.dtype}")
    if np.isnan(array).any() or np.isposinf(array).any():
        raise ValueError(f"{name}包含NaN或+Inf")
    return array


def _validate_ranking(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    expected = (STATE_COUNT, STATE_COUNT)
    if array.shape != expected or array.dtype != np.float64:
        raise ValueError(f"{name}必须是float64且shape={expected}，实际{array.shape}/{array.dtype}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name}包含非有限值")
    return array


def _load_state_frame(all_ilc_root: Path) -> tuple[pd.DataFrame, np.ndarray]:
    """Load frozen condition metadata and actual ILC column counts."""

    path = Path(all_ilc_root) / "ilc_availability.csv"
    if not path.is_file():
        raise FileNotFoundError(f"缺少ILC availability metadata：{path}")
    frame = pd.read_csv(path)
    required = (*STATE_COLUMNS, "ilc_column_count")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"availability缺少字段：{missing}")
    frame = frame.loc[:, list(required)].sort_values("state_id").reset_index(drop=True)
    _validate_state_ids(frame["state_id"].to_numpy(dtype=np.int64), "availability.state_id")
    counts = frame["ilc_column_count"].to_numpy(dtype=np.int64)
    if counts.shape != (STATE_COUNT,) or np.any(counts < 1) or np.any(counts > NMAX):
        raise ValueError("ilc_column_count必须是1...5之间的425个正整数")
    state_frame = frame.loc[:, list(STATE_COLUMNS)].copy()
    table = pd.DataFrame(build_state_table()).loc[:, list(STATE_COLUMNS)]
    pd.testing.assert_frame_equal(
        state_frame.reset_index(drop=True),
        table.reset_index(drop=True),
        check_dtype=False,
        check_exact=False,
        atol=0.0,
        rtol=0.0,
    )
    return state_frame, counts


def load_retrieval_inputs(
    source_root: Path,
    reference_root: Path,
    all_ilc_root: Path,
) -> RetrievalInputs:
    """Load and validate frozen Y-A2, Y-C3, Real-B and regression references."""

    source_root = Path(source_root)
    reference_root = Path(reference_root)
    all_ilc_root = Path(all_ilc_root)
    lut = _load_npz_dict(source_root / "lut_fingerprints.npz")
    query = _load_npz_dict(source_root / "query_fingerprints.npz")
    required_lut = {"state_ids", "common_B_input", "Y_A2_fingerprints"}
    missing = sorted(required_lut - set(lut))
    if missing:
        raise ValueError(f"A123 lut_fingerprints缺少字段：{missing}")
    required_query = {"state_ids", "effective_C_n", "Q_C3"}
    missing = sorted(required_query - set(query))
    if missing:
        raise ValueError(f"A123 query_fingerprints缺少字段：{missing}")

    state_ids = _validate_state_ids(lut["state_ids"])
    query_state_ids = _validate_state_ids(query["state_ids"], "query.state_ids")
    if not np.array_equal(state_ids, query_state_ids):
        raise ValueError("LUT与Query的state_ids不一致")
    common_b_input = np.asarray(lut["common_B_input"])
    if common_b_input.shape != (4915,) or common_b_input.dtype != np.complex128:
        raise ValueError("common_B_input必须是complex128(4915,)")
    if not np.all(np.isfinite(common_b_input)):
        raise ValueError("common_B_input包含非有限值")
    lut_a2 = _validate_fingerprint(lut["Y_A2_fingerprints"], "Y-A2 LUT")
    query_c3 = _validate_fingerprint(query["Q_C3"], "Y-C3 Query")
    effective_c_n = np.asarray(query["effective_C_n"])
    if effective_c_n.shape != (STATE_COUNT, NMAX) or not np.issubdtype(
        effective_c_n.dtype, np.integer
    ):
        raise ValueError("effective_C_n必须是(425,5)整数数组")
    effective_c_n = effective_c_n.astype(np.int64, copy=False)

    state_frame, ilc_counts = _load_state_frame(all_ilc_root)
    expected_effective_c3 = np.minimum(REQUESTED_C_STAGE, ilc_counts)
    if not np.array_equal(effective_c_n[:, REQUESTED_C_STAGE - 1], expected_effective_c3):
        raise ValueError("Q_C3未遵循effective_C_n=min(3,Ns)规则")

    frozen = load_frozen_scenario2_artifacts(reference_root)
    if not np.array_equal(frozen.state_ids, state_ids):
        raise ValueError("A123 state_ids与Scenario 2冻结reference不一致")
    if not np.array_equal(frozen.common_B_input, common_b_input):
        raise ValueError("A123 common_B_input与冻结reference不一致")
    previous_distances = _load_npz_dict(source_root / "distance_matrices.npz")
    previous_rankings = _load_npz_dict(source_root / "ranking_matrices.npz")
    if "D_C3_A2" not in previous_distances or "R_C3_A2" not in previous_rankings:
        raise ValueError("A123冻结结果缺少D_C3_A2或R_C3_A2")

    validation_path = source_root / "validation.json"
    if not validation_path.is_file():
        raise FileNotFoundError(f"缺少A123 validation.json：{validation_path}")
    with validation_path.open("r", encoding="utf-8-sig") as handle:
        a123_validation = json.load(handle)
    definitions = a123_validation.get("lut_definitions", {})
    if definitions.get("Y-A2") != [[2, 1.0]]:
        raise ValueError("冻结A123结果中的Y-A2定义不是A段ILC2")
    if a123_validation.get("raw_waveform_averaging") is not False:
        raise ValueError("冻结A123结果不满足fingerprint-level A2定义")
    if a123_validation.get("Real_B_used_as_LUT") is not False:
        raise ValueError("Real-B不能作为A2 LUT")

    return RetrievalInputs(
        source_root=source_root,
        reference_root=reference_root,
        all_ilc_root=all_ilc_root,
        state_ids=state_ids,
        state_frame=state_frame,
        query_c3=query_c3,
        lut_a2=lut_a2,
        real_b=_validate_fingerprint(frozen.real_B_fingerprints, "Real-B B段"),
        common_b_input=common_b_input,
        effective_c_n=effective_c_n,
        ilc_column_counts=ilc_counts,
        previous_distance_c3_a2=_validate_distance(
            previous_distances["D_C3_A2"], "旧D_C3_A2"
        ),
        previous_ranking_c3_a2=_validate_ranking(
            previous_rankings["R_C3_A2"], "旧R_C3_A2"
        ),
        previous_real_b_distance=_validate_distance(frozen.D_RR, "旧D_RR"),
        previous_real_b_ranking=_validate_ranking(frozen.R_RR, "旧R_RR"),
        a123_validation=a123_validation,
    )


def _max_difference_allowing_neginf(
    current: np.ndarray, previous: np.ndarray
) -> tuple[float, int]:
    current = np.asarray(current)
    previous = np.asarray(previous)
    if current.shape != previous.shape:
        return float("inf"), max(current.size, previous.size)
    pattern_mismatch = int(np.count_nonzero(np.isneginf(current) != np.isneginf(previous)))
    finite_current = np.isfinite(current)
    finite_previous = np.isfinite(previous)
    pattern_mismatch += int(np.count_nonzero(finite_current != finite_previous))
    finite = finite_current & finite_previous
    if not np.any(finite):
        return 0.0, pattern_mismatch
    return float(np.max(np.abs(current[finite] - previous[finite]))), pattern_mismatch


def _regression_report(
    current_distance: np.ndarray,
    previous_distance: np.ndarray,
    previous_ranking: np.ndarray,
    name: str,
) -> dict[str, Any]:
    current_ranking = distance_to_ranks(current_distance)
    distance_error, distance_pattern_mismatch = _max_difference_allowing_neginf(
        current_distance, previous_distance
    )
    ranking_error, ranking_pattern_mismatch = _max_difference_allowing_neginf(
        current_ranking, previous_ranking
    )
    return {
        "name": name,
        "distance_max_abs_error": distance_error,
        "distance_nonfinite_pattern_mismatch_count": distance_pattern_mismatch,
        "ranking_max_abs_error": ranking_error,
        "ranking_nonfinite_pattern_mismatch_count": ranking_pattern_mismatch,
        "ranking_identical": bool(np.array_equal(current_ranking, previous_ranking)),
        "pass": bool(
            distance_pattern_mismatch == 0
            and ranking_pattern_mismatch == 0
            and np.array_equal(current_ranking, previous_ranking)
        ),
    }


def compute_retrieval_distance_matrix(inputs: RetrievalInputs) -> np.ndarray:
    """Compute ``D_QA2(i,j)=CNMSE(Y-C3_i,Y-A2_j)``."""

    return compute_cnmse_distance_matrix(inputs.query_c3, inputs.lut_a2)


def compute_real_b_distance_matrix(inputs: RetrievalInputs) -> np.ndarray:
    """Compute the frozen Real-B to Real-B behavioral distance matrix."""

    return compute_cnmse_distance_matrix(inputs.real_b, inputs.real_b)


def _stable_order(row: np.ndarray, state_ids: np.ndarray) -> np.ndarray:
    if row.shape != (STATE_COUNT,) or state_ids.shape != (STATE_COUNT,):
        raise ValueError("距离行和state_ids必须长度为425")
    if np.isnan(row).any() or np.isposinf(row).any():
        raise ValueError("距离行包含NaN或+Inf")
    return np.lexsort((state_ids, row))


def _topk_retrieval(
    distance_matrix: np.ndarray, state_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    distance_matrix = _validate_distance(distance_matrix, "Query到LUT距离")
    state_ids = _validate_state_ids(state_ids)
    selected = np.empty(STATE_COUNT, dtype=np.int64)
    selected_distance = np.empty(STATE_COUNT, dtype=np.float64)
    true_rank = np.empty(STATE_COUNT, dtype=np.int64)
    minimum_tie_count = np.empty(STATE_COUNT, dtype=np.int64)
    top_order = np.empty((STATE_COUNT, STATE_COUNT), dtype=np.int64)
    for state_index in range(STATE_COUNT):
        row = distance_matrix[state_index]
        order = _stable_order(row, state_ids)
        top_order[state_index] = order
        minimum = row[order[0]]
        minimum_tie_count[state_index] = int(np.count_nonzero(row == minimum))
        selected_index = int(order[0])
        selected[state_index] = state_ids[selected_index]
        selected_distance[state_index] = row[selected_index]
        true_positions = np.flatnonzero(order == state_index)
        if true_positions.size != 1:
            raise RuntimeError(f"state_id={state_index}无法定位真实candidate排名")
        true_rank[state_index] = int(true_positions[0] + 1)
    return selected, selected_distance, true_rank, minimum_tie_count, top_order


def circular_difference_deg(left: float, right: float) -> float:
    """Return the smallest absolute circular difference in degrees."""

    left_value = float(left) % 360.0
    right_value = float(right) % 360.0
    difference = abs(left_value - right_value)
    return float(min(difference, 360.0 - difference))


def _condition_differences(
    real_row: pd.Series, query_row: pd.Series
) -> dict[str, Any]:
    return {
        "same_funMng": bool(real_row.funMng == query_row.funMng),
        "same_funAng": bool(real_row.funAng == query_row.funAng),
        "same_secMng": bool(real_row.secMng == query_row.secMng),
        "same_secAng": bool(real_row.secAng == query_row.secAng),
        "funMng_diff": int(query_row.funMng - real_row.funMng),
        "funAng_diff_deg": circular_difference_deg(real_row.funAng, query_row.funAng),
        "secMng_diff": int(query_row.secMng - real_row.secMng),
        "secAng_diff_deg": circular_difference_deg(real_row.secAng, query_row.secAng),
    }


def build_retrieval_results(
    inputs: RetrievalInputs,
    query_to_a2_distance: np.ndarray,
) -> pd.DataFrame:
    """Select deterministic Top-1 candidates and build one row per real state."""

    selected, selected_distance, true_rank, tie_count, _ = _topk_retrieval(
        query_to_a2_distance, inputs.state_ids
    )
    state_lookup = inputs.state_frame.set_index("state_id")
    rows: list[dict[str, Any]] = []
    for state_id in inputs.state_ids:
        real_id = int(state_id)
        query_id = int(selected[real_id])
        real_row = state_lookup.loc[real_id]
        query_row = state_lookup.loc[query_id]
        condition = _condition_differences(real_row, query_row)
        rows.append(
            {
                "State_n_R": real_id,
                "State_n_Q": query_id,
                "exact_hit": bool(query_id == real_id),
                "state_id_delta": query_id - real_id,
                "abs_state_id_delta": abs(query_id - real_id),
                "query_to_selected_lut_CNMSE_dB": float(selected_distance[real_id]),
                "true_state_rank_in_A2_LUT": int(true_rank[real_id]),
                "top1_hit": bool(true_rank[real_id] <= 1),
                "top3_hit": bool(true_rank[real_id] <= 3),
                "top5_hit": bool(true_rank[real_id] <= 5),
                "minimum_tie_count": int(tie_count[real_id]),
                **{
                    f"{column}_R": real_row[column]
                    for column in ("funMng", "funAng", "secMng", "secAng")
                },
                **{
                    f"{column}_Q": query_row[column]
                    for column in ("funMng", "funAng", "secMng", "secAng")
                },
                **condition,
            }
        )
    result = pd.DataFrame(rows)
    if result.shape[0] != STATE_COUNT:
        raise RuntimeError("retrieval_results必须恰好有425行")
    return result


def _nearest_nonself(
    real_b_distance: np.ndarray, state_id: int, state_ids: np.ndarray
) -> tuple[int, float]:
    row = np.asarray(real_b_distance[state_id], dtype=np.float64).copy()
    candidate_mask = np.ones(STATE_COUNT, dtype=bool)
    candidate_mask[state_id] = False
    candidate_ids = state_ids[candidate_mask]
    candidate_distances = row[candidate_mask]
    order = np.lexsort((candidate_ids, candidate_distances))
    candidate_index = int(order[0])
    return int(candidate_ids[candidate_index]), float(candidate_distances[candidate_index])


def build_real_b_diagnostics(
    inputs: RetrievalInputs,
    real_b_distance: np.ndarray,
    retrieval_results: pd.DataFrame,
) -> pd.DataFrame:
    """Attach retrieved Real-B CNMSE, non-self oracle and behavioral regret."""

    real_b_distance = _validate_distance(real_b_distance, "Real-B距离")
    rows: list[dict[str, Any]] = []
    for record in retrieval_results.itertuples(index=False):
        real_id = int(record.State_n_R)
        query_id = int(record.State_n_Q)
        retrieved_value = float(real_b_distance[real_id, query_id])
        oracle_state, oracle_value = _nearest_nonself(real_b_distance, real_id, inputs.state_ids)
        if bool(record.exact_hit):
            regret = float("nan")
        elif np.isfinite(retrieved_value) and np.isfinite(oracle_value):
            regret = float(retrieved_value - oracle_value)
        else:
            # Duplicate Real-B behavior makes an extended-real regret undefined;
            # preserve the raw distances and expose the duplicate count in summary.
            regret = float("nan")
        rows.append(
            {
                "State_n_R": real_id,
                "State_n_Q": query_id,
                "exact_hit": bool(record.exact_hit),
                "retrieved_real_B_CNMSE_dB": retrieved_value,
                "nearest_nonself_B_state": oracle_state,
                "nearest_nonself_B_CNMSE_dB": oracle_value,
                "nonself_behavioral_regret_dB": regret,
                "nonexact_retrieval_nonfinite": bool(
                    not bool(record.exact_hit) and not np.isfinite(retrieved_value)
                ),
            }
        )
    result = pd.DataFrame(rows)
    if result.shape[0] != STATE_COUNT:
        raise RuntimeError("retrieved_real_B_cnmse必须恰好有425行")
    return result


def _finite_summary(values: np.ndarray, prefix: str) -> dict[str, Any]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": float("nan"),
            f"{prefix}_median": float("nan"),
            f"{prefix}_std": float("nan"),
            f"{prefix}_q05": float("nan"),
            f"{prefix}_q25": float("nan"),
            f"{prefix}_q75": float("nan"),
            f"{prefix}_q95": float("nan"),
            f"{prefix}_min": float("nan"),
            f"{prefix}_max": float("nan"),
        }
    return {
        f"{prefix}_count": int(finite.size),
        f"{prefix}_mean": float(np.mean(finite)),
        f"{prefix}_median": float(np.median(finite)),
        f"{prefix}_std": float(np.std(finite)),
        f"{prefix}_q05": float(np.quantile(finite, 0.05)),
        f"{prefix}_q25": float(np.quantile(finite, 0.25)),
        f"{prefix}_q75": float(np.quantile(finite, 0.75)),
        f"{prefix}_q95": float(np.quantile(finite, 0.95)),
        f"{prefix}_min": float(np.min(finite)),
        f"{prefix}_max": float(np.max(finite)),
    }


def build_threshold_sweep(
    real_b_values: np.ndarray, exact_hit: np.ndarray | None = None
) -> pd.DataFrame:
    values = np.asarray(real_b_values, dtype=np.float64)
    if exact_hit is None:
        exact = np.isneginf(values)
    else:
        exact = np.asarray(exact_hit, dtype=bool)
        if exact.shape != values.shape:
            raise ValueError("exact_hit必须与real_b_values等长")
    rows: list[dict[str, Any]] = []
    for threshold in THRESHOLD_DB_VALUES:
        all_success = values <= threshold
        nonexact = ~exact
        nonexact_success = nonexact & (values <= threshold)
        rows.append(
            {
                "threshold_dB": threshold,
                "all_retrieval_success_count_including_exact": int(np.count_nonzero(all_success)),
                "all_retrieval_success_rate": float(np.mean(all_success)),
                "non_exact_success_count": int(np.count_nonzero(nonexact_success)),
                "non_exact_success_rate": float(
                    np.count_nonzero(nonexact_success) / max(np.count_nonzero(nonexact), 1)
                ),
            }
        )
    return pd.DataFrame(rows)


def _build_summary(
    retrieval_results: pd.DataFrame,
    real_b_diagnostics: pd.DataFrame,
    threshold_sweep: pd.DataFrame,
) -> pd.DataFrame:
    exact = retrieval_results["exact_hit"].to_numpy(dtype=bool)
    real_b_values = real_b_diagnostics["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float)
    nonexact_values = real_b_values[~exact]
    finite_nonexact = nonexact_values[np.isfinite(nonexact_values)]
    regret_values = real_b_diagnostics.loc[
        ~exact, "nonself_behavioral_regret_dB"
    ].to_numpy(dtype=float)
    finite_regret = regret_values[np.isfinite(regret_values)]
    summary: dict[str, Any] = {
        "state_count": STATE_COUNT,
        "candidate_count": STATE_COUNT,
        "candidate_count_per_query": STATE_COUNT,
        "self_candidate_included": True,
        "exact_hit_count": int(np.count_nonzero(exact)),
        "exact_hit_rate": float(np.mean(exact)),
        "non_exact_count": int(np.count_nonzero(~exact)),
        "non_exact_finite_count": int(finite_nonexact.size),
        "non_exact_nonfinite_count": int(nonexact_values.size - finite_nonexact.size),
        "top3_hit_count": int(retrieval_results["top3_hit"].sum()),
        "top3_hit_rate": float(retrieval_results["top3_hit"].mean()),
        "top5_hit_count": int(retrieval_results["top5_hit"].sum()),
        "top5_hit_rate": float(retrieval_results["top5_hit"].mean()),
        "true_state_rank_median": float(retrieval_results["true_state_rank_in_A2_LUT"].median()),
        "true_state_rank_max": int(retrieval_results["true_state_rank_in_A2_LUT"].max()),
        "minimum_tie_count_rows": int(
            np.count_nonzero(retrieval_results["minimum_tie_count"].to_numpy(dtype=int) > 1)
        ),
        "real_B_nonexact_duplicate_count": int(
            real_b_diagnostics.loc[
                ~exact, "nonexact_retrieval_nonfinite"
            ].to_numpy(dtype=bool).sum()
        ),
        "non_exact_regret_count": int(finite_regret.size),
        "non_exact_regret_mean": (
            float(np.mean(finite_regret)) if finite_regret.size else float("nan")
        ),
        "non_exact_regret_median": (
            float(np.median(finite_regret)) if finite_regret.size else float("nan")
        ),
        "non_exact_regret_max": (
            float(np.max(finite_regret)) if finite_regret.size else float("nan")
        ),
        "threshold_sweep_is_descriptive_only": True,
        "behavioral_success_threshold_frozen": False,
    }
    summary.update(_finite_summary(finite_nonexact, "non_exact_B_CNMSE"))
    for row in threshold_sweep.itertuples(index=False):
        key = str(int(row.threshold_dB))
        summary[f"threshold_{key}_all_rate"] = float(row.all_retrieval_success_rate)
        summary[f"threshold_{key}_non_exact_rate"] = float(row.non_exact_success_rate)
    return pd.DataFrame([summary])


def _build_worst_table(
    retrieval_results: pd.DataFrame,
    real_b_diagnostics: pd.DataFrame,
    limit: int = 20,
) -> pd.DataFrame:
    merged = retrieval_results.merge(
        real_b_diagnostics,
        on=["State_n_R", "State_n_Q", "exact_hit"],
        how="left",
        validate="one_to_one",
        suffixes=("", "_B"),
    )
    nonexact = merged.loc[~merged["exact_hit"]].copy()
    nonexact = nonexact.sort_values(
        ["retrieved_real_B_CNMSE_dB", "State_n_R"],
        ascending=[False, True],
        na_position="last",
    )
    return nonexact.head(limit).reset_index(drop=True)


def run_retrieval_analysis(
    source_root: Path,
    reference_root: Path,
    all_ilc_root: Path,
) -> RetrievalResult:
    """Run the complete frozen-input retrieval and Real-B diagnostic pipeline."""

    inputs = load_retrieval_inputs(source_root, reference_root, all_ilc_root)
    query_to_a2_distance = compute_retrieval_distance_matrix(inputs)
    real_b_distance = compute_real_b_distance_matrix(inputs)
    query_to_a2_ranking = distance_to_ranks(query_to_a2_distance)
    real_b_ranking = distance_to_ranks(real_b_distance)
    retrieval_results = build_retrieval_results(inputs, query_to_a2_distance)
    real_b_diagnostics = build_real_b_diagnostics(inputs, real_b_distance, retrieval_results)
    threshold_sweep = build_threshold_sweep(
        real_b_diagnostics["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float),
        retrieval_results["exact_hit"].to_numpy(dtype=bool),
    )
    retrieval_summary = _build_summary(
        retrieval_results, real_b_diagnostics, threshold_sweep
    )
    worst_retrieval_states = _build_worst_table(retrieval_results, real_b_diagnostics)
    baseline_regression = _regression_report(
        query_to_a2_distance,
        inputs.previous_distance_c3_a2,
        inputs.previous_ranking_c3_a2,
        "C3-to-A2",
    )
    real_b_regression = _regression_report(
        real_b_distance,
        inputs.previous_real_b_distance,
        inputs.previous_real_b_ranking,
        "Real-B-to-Real-B",
    )
    validation = {
        "scenario": 2,
        "study": "C3 online behavior fingerprint to fixed A2 LUT retrieval with Real-B validation",
        "state_count": STATE_COUNT,
        "query_fingerprint_type": "Y-C3",
        "lut_fingerprint_type": "Y-A2",
        "query_segment": "C",
        "lut_model_segment": "A",
        "reference_segment": "real B",
        "query_effective_rule": "effective_C_n = min(3, Ns)",
        "query_effective_rule_valid": bool(
            np.array_equal(
                inputs.effective_c_n[:, REQUESTED_C_STAGE - 1],
                np.minimum(REQUESTED_C_STAGE, inputs.ilc_column_counts),
            )
        ),
        "candidate_count": STATE_COUNT,
        "candidate_count_per_query": STATE_COUNT,
        "self_match_included": True,
        "retrieval_metric": "CNMSE",
        "retrieval_direction": "minimum; more negative is more similar",
        "real_B_used_as_LUT": False,
        "real_B_used_as_reference": True,
        "real_B_signal": "canonical OFF B segment from yout_withoutdpd_ori",
        "exact_hit_definition": "State_n_Q == State_n_R",
        "behavioral_distance_definition": "CNMSE(Real_B_R, Real_B_Q)",
        "behavioral_success_threshold_frozen": False,
        "threshold_sweep_values_dB": list(THRESHOLD_DB_VALUES),
        "models_retrained": False,
        "ridge_rescanned": False,
        "mp_changed": False,
        "canonical_changed": False,
        "raw_data_modified": False,
        "raw_data_read": False,
        "a2_definition_regression": {
            "definition": inputs.a123_validation.get("lut_definitions", {}).get("Y-A2"),
            "source": str(inputs.source_root / "lut_fingerprints.npz"),
            "pass": True,
        },
        "query_shape": list(inputs.query_c3.shape),
        "lut_shape": list(inputs.lut_a2.shape),
        "real_B_shape": list(inputs.real_b.shape),
        "query_dtype": str(inputs.query_c3.dtype),
        "lut_dtype": str(inputs.lut_a2.dtype),
        "real_B_dtype": str(inputs.real_b.dtype),
        "query_lut_all_finite": bool(
            np.all(np.isfinite(inputs.query_c3)) and np.all(np.isfinite(inputs.lut_a2))
        ),
        "distance_matrix_shape": list(query_to_a2_distance.shape),
        "real_B_distance_matrix_shape": list(real_b_distance.shape),
        "real_B_exact_diagonal_all_negative_infinity": bool(
            np.all(np.isneginf(np.diag(real_b_distance)))
        ),
        "minimum_tie_count_rows": int(
            retrieval_summary.loc[0, "minimum_tie_count_rows"]
        ),
        "c3_to_a2_ranking_regression": baseline_regression,
        "real_B_ranking_regression": real_b_regression,
        "summary": retrieval_summary.iloc[0].to_dict(),
    }
    if not baseline_regression["pass"] or not real_b_regression["pass"]:
        raise RuntimeError("冻结C3→A2或Real-B排名回归失败，停止正式分析")
    return RetrievalResult(
        inputs=inputs,
        query_to_a2_distance=query_to_a2_distance,
        real_b_distance=real_b_distance,
        query_to_a2_ranking=query_to_a2_ranking,
        real_b_ranking=real_b_ranking,
        retrieval_results=retrieval_results,
        real_b_diagnostics=real_b_diagnostics,
        threshold_sweep=threshold_sweep,
        retrieval_summary=retrieval_summary,
        worst_retrieval_states=worst_retrieval_states,
        validation=validation,
    )


__all__ = [
    "FINGERPRINT_LENGTH",
    "NMAX",
    "REQUESTED_C_STAGE",
    "STATE_COUNT",
    "THRESHOLD_DB_VALUES",
    "RetrievalInputs",
    "RetrievalResult",
    "build_retrieval_results",
    "build_real_b_diagnostics",
    "build_threshold_sweep",
    "circular_difference_deg",
    "compute_real_b_distance_matrix",
    "compute_retrieval_distance_matrix",
    "load_retrieval_inputs",
    "run_retrieval_analysis",
]
