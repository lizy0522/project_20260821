"""
Scenario 2 formal retrieval: Y-C2 Query -> state-specific Y-A_end LUT.

The implementation consumes the frozen all-ILC theta, common-B probe and Real-B
artifacts.  ``A_end`` is selected independently for every state from its actual
ILC column count.  The formal DPD-shareable rule is strictly ``Real-B CNMSE <
-40 dB``.
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
from behavior_model.basis import build_mp_basis
from behavior_model.config import MP_CONFIG, NUM_COEFFICIENTS
from data_manager import build_state_table

STATE_COUNT = 425
FINGERPRINT_LENGTH = 4913
COMMON_B_INPUT_LENGTH = 4915
NMAX = 5
C2_STAGE = 2
DPD_SHAREABLE_THRESHOLD_DB = -40.0
AEND_STAGES = (2, 3, 4, 5)
STATE_COLUMNS = ("state_id", "funMng", "funAng", "secMng", "secAng", "Vm", "Pin")


@dataclass(frozen=True)
class C2AendInputs:
    """Frozen inputs for the C2-to-A_end study."""

    all_ilc_root: Path
    a123_root: Path
    previous_c3a2_root: Path
    reference_root: Path
    state_ids: np.ndarray
    state_frame: pd.DataFrame
    ilc_column_counts: np.ndarray
    available_mask: np.ndarray
    theta_y_a_actual: np.ndarray
    theta_y_c_actual: np.ndarray
    common_b_input: np.ndarray
    phi_b: np.ndarray
    frozen_a2_fingerprint: np.ndarray
    frozen_a3_fingerprint: np.ndarray
    frozen_c2_query: np.ndarray
    previous_real_b_distance: np.ndarray
    previous_real_b_ranking: np.ndarray
    previous_c3a2_distance: np.ndarray
    previous_c3a2_ranking: np.ndarray
    previous_c3a2_results: pd.DataFrame
    previous_c3a2_diagnostics: pd.DataFrame
    previous_c3a2_summary: pd.DataFrame
    a123_validation: dict[str, Any]


@dataclass(frozen=True)
class C2AendResult:
    """Complete in-memory result of the formal C2-to-A_end retrieval."""

    inputs: C2AendInputs
    a_end_stage_map: pd.DataFrame
    a_end_fingerprints: np.ndarray
    c2_query_fingerprint: np.ndarray
    retrieval_distance: np.ndarray
    retrieval_ranking: np.ndarray
    real_b_distance: np.ndarray
    real_b_ranking: np.ndarray
    retrieval_results: pd.DataFrame
    real_b_diagnostics: pd.DataFrame
    oracle_shareable_rank: pd.DataFrame
    failed_retrieval_states: pd.DataFrame
    condition_region_summary: pd.DataFrame
    retrieval_summary: pd.DataFrame
    dpd_shareable_summary: pd.DataFrame
    comparison_vs_c3_a2: pd.DataFrame
    validation: dict[str, Any]


def _load_npz_dict(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(f"缺少NPZ输入：{path}")
    with np.load(path, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _validate_state_ids(value: np.ndarray, name: str = "state_ids") -> np.ndarray:
    array = np.asarray(value)
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
            f"{name}必须为complex128且shape={expected}，"
            f"实际{array.shape}/{array.dtype}"
        )
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name}包含NaN或Inf")
    return array


def _validate_theta(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    expected = (STATE_COUNT, NMAX, NUM_COEFFICIENTS)
    if array.shape != expected or array.dtype != np.complex128:
        raise ValueError(
            f"{name}必须为complex128且shape={expected}，"
            f"实际{array.shape}/{array.dtype}"
        )
    return array


def _validate_distance(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (STATE_COUNT, STATE_COUNT) or array.dtype != np.float64:
        raise ValueError(f"{name}必须为float64且shape=(425,425)，实际{array.shape}/{array.dtype}")
    if np.isnan(array).any() or np.isposinf(array).any():
        raise ValueError(f"{name}包含NaN或+Inf")
    return array


def _validate_ranking(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != (STATE_COUNT, STATE_COUNT) or array.dtype != np.float64:
        raise ValueError(f"{name}必须为float64且shape=(425,425)，实际{array.shape}/{array.dtype}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name}包含非有限值")
    return array


def _max_difference_allowing_neginf(
    current: np.ndarray, previous: np.ndarray
) -> tuple[float, int]:
    current = np.asarray(current)
    previous = np.asarray(previous)
    if current.shape != previous.shape:
        return float("inf"), max(current.size, previous.size)
    mismatch = int(np.count_nonzero(np.isneginf(current) != np.isneginf(previous)))
    current_finite = np.isfinite(current)
    previous_finite = np.isfinite(previous)
    mismatch += int(np.count_nonzero(current_finite != previous_finite))
    finite = current_finite & previous_finite
    if not np.any(finite):
        return 0.0, mismatch
    return float(np.max(np.abs(current[finite] - previous[finite]))), mismatch


def _load_state_metadata(all_ilc_root: Path) -> tuple[pd.DataFrame, np.ndarray]:
    path = Path(all_ilc_root) / "ilc_availability.csv"
    if not path.is_file():
        raise FileNotFoundError(f"缺少ILC availability：{path}")
    frame = pd.read_csv(path)
    required = (*STATE_COLUMNS, "ilc_column_count")
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"ILC availability缺少字段：{missing}")
    frame = frame.loc[:, list(required)].sort_values("state_id").reset_index(drop=True)
    _validate_state_ids(frame["state_id"].to_numpy(dtype=np.int64), "availability.state_id")
    counts = frame["ilc_column_count"].to_numpy(dtype=np.int64)
    if counts.shape != (STATE_COUNT,) or np.any(counts < 1) or np.any(counts > NMAX):
        raise ValueError("ilc_column_count必须为1...5之间的425个整数")
    state_frame = frame.loc[:, list(STATE_COLUMNS)].copy()
    reference_frame = pd.DataFrame(build_state_table()).loc[:, list(STATE_COLUMNS)]
    pd.testing.assert_frame_equal(
        state_frame,
        reference_frame,
        check_dtype=False,
        check_exact=False,
        atol=0.0,
        rtol=0.0,
    )
    return state_frame, counts


def load_c2_aend_inputs(
    all_ilc_root: Path,
    a123_root: Path,
    previous_c3a2_root: Path,
    reference_root: Path,
) -> C2AendInputs:
    """Load all frozen inputs and fail before formal output if C2 is unavailable."""

    all_ilc_root = Path(all_ilc_root)
    a123_root = Path(a123_root)
    previous_c3a2_root = Path(previous_c3a2_root)
    reference_root = Path(reference_root)
    state_frame, counts = _load_state_metadata(all_ilc_root)
    validate_c2_availability(counts)

    theta_data = _load_npz_dict(all_ilc_root / "all_ilc_theta.npz")
    required_theta = {
        "state_ids",
        "theta_Y_A_actual",
        "theta_Y_C_actual",
        "available_mask",
        "ilc_column_counts",
    }
    missing = sorted(required_theta - set(theta_data))
    if missing:
        raise ValueError(f"all_ilc_theta缺少字段：{missing}")
    state_ids = _validate_state_ids(theta_data["state_ids"], "theta.state_ids")
    theta_a = _validate_theta(theta_data["theta_Y_A_actual"], "theta_Y_A_actual")
    theta_c = _validate_theta(theta_data["theta_Y_C_actual"], "theta_Y_C_actual")
    available_mask = np.asarray(theta_data["available_mask"])
    if available_mask.shape != (STATE_COUNT, NMAX) or available_mask.dtype != bool:
        raise ValueError("available_mask必须为(425,5) bool")
    theta_counts = np.asarray(theta_data["ilc_column_counts"], dtype=np.int64)
    if not np.array_equal(theta_counts, counts):
        raise ValueError("theta中的ilc_column_counts与availability不一致")
    expected_mask = np.arange(NMAX, dtype=np.int64)[None, :] < counts[:, None]
    if not np.array_equal(available_mask, expected_mask):
        raise ValueError("available_mask与实际ILC列数不一致")
    active_theta = np.concatenate((theta_a[available_mask], theta_c[available_mask]), axis=0)
    if not np.all(np.isfinite(active_theta)):
        raise ValueError("可用冻结theta包含NaN或Inf")

    lut_data = _load_npz_dict(a123_root / "lut_fingerprints.npz")
    query_data = _load_npz_dict(a123_root / "query_fingerprints.npz")
    required_lut = {"state_ids", "common_B_input", "Y_A2_fingerprints", "Y_A3_fingerprints"}
    missing = sorted(required_lut - set(lut_data))
    if missing:
        raise ValueError(f"A123 lut_fingerprints缺少字段：{missing}")
    if not np.array_equal(state_ids, _validate_state_ids(lut_data["state_ids"], "lut.state_ids")):
        raise ValueError("all-ILC与A123 state_ids不一致")
    common_b_input = np.asarray(lut_data["common_B_input"])
    if common_b_input.shape != (COMMON_B_INPUT_LENGTH,) or common_b_input.dtype != np.complex128:
        raise ValueError("common_B_input必须为complex128(4915,)")
    if not np.all(np.isfinite(common_b_input)):
        raise ValueError("common_B_input包含非有限值")
    frozen_a2 = _validate_fingerprint(lut_data["Y_A2_fingerprints"], "冻结Y-A2")
    frozen_a3 = _validate_fingerprint(lut_data["Y_A3_fingerprints"], "冻结Y-A3")
    if "Q_C2" not in query_data:
        raise ValueError("A123 query_fingerprints缺少Q_C2")
    frozen_c2 = _validate_fingerprint(query_data["Q_C2"], "冻结Y-C2")
    query_state_ids = _validate_state_ids(query_data["state_ids"], "query.state_ids")
    if not np.array_equal(state_ids, query_state_ids):
        raise ValueError("A123 LUT与Query state_ids不一致")

    with (a123_root / "validation.json").open("r", encoding="utf-8-sig") as handle:
        a123_validation = json.load(handle)
    if a123_validation.get("ridge_lambda") != 1e-8:
        raise ValueError("A123冻结结果的Ridge lambda不是1e-8")
    if a123_validation.get("C_effective_rule") != "effective_C_n = min(requested_C_n, Ns)":
        raise ValueError("A123 C有效阶段规则与冻结定义不一致")
    if a123_validation.get("lut_definitions", {}).get("Y-A2") != [[2, 1.0]]:
        raise ValueError("冻结Y-A2定义不是ILC2")

    phi_b = build_mp_basis(common_b_input, MP_CONFIG["orders"], MP_CONFIG["memory_depth"])
    if phi_b.shape != (FINGERPRINT_LENGTH, NUM_COEFFICIENTS):
        raise ValueError(f"Phi_B shape错误：{phi_b.shape}")
    if not np.all(np.isfinite(phi_b)):
        raise ValueError("Phi_B包含非有限值")

    frozen = load_frozen_scenario2_artifacts(reference_root)
    if not np.array_equal(frozen.state_ids, state_ids):
        raise ValueError("Scenario 2 reference与all-ILC state_ids不一致")
    if not np.array_equal(frozen.common_B_input, common_b_input):
        raise ValueError("Scenario 2 reference与A123 common_B_input不一致")

    previous_distance_data = _load_npz_dict(previous_c3a2_root / "retrieval_distance_matrix.npz")
    previous_ranking_data = _load_npz_dict(previous_c3a2_root / "real_B_distance_matrix.npz")
    previous_c3_distance = _validate_distance(
        previous_distance_data["D_QA2"], "旧C3→A2距离"
    )
    previous_c3_ranking = _validate_ranking(
        previous_distance_data["R_QA2"], "旧C3→A2排名"
    )
    previous_real_b_distance = _validate_distance(
        previous_ranking_data["D_B"], "旧Real-B距离"
    )
    previous_real_b_ranking = _validate_ranking(previous_ranking_data["R_B"], "旧Real-B排名")
    real_distance_error, real_distance_mismatch = _max_difference_allowing_neginf(
        previous_real_b_distance, frozen.D_RR
    )
    real_ranking_error, real_ranking_mismatch = _max_difference_allowing_neginf(
        previous_real_b_ranking, frozen.R_RR
    )
    if (
        real_distance_mismatch
        or real_ranking_mismatch
        or real_distance_error != 0.0
        or real_ranking_error != 0.0
    ):
        raise RuntimeError("旧C3→A2 Real-B矩阵与冻结R_RR不一致")

    previous_results = pd.read_csv(previous_c3a2_root / "retrieval_results.csv")
    previous_diagnostics = pd.read_csv(previous_c3a2_root / "retrieved_real_B_cnmse.csv")
    previous_summary = pd.read_csv(previous_c3a2_root / "retrieval_summary.csv")
    if previous_results.shape[0] != STATE_COUNT or previous_diagnostics.shape[0] != STATE_COUNT:
        raise ValueError("旧C3→A2检索表必须各有425行")
    return C2AendInputs(
        all_ilc_root=all_ilc_root,
        a123_root=a123_root,
        previous_c3a2_root=previous_c3a2_root,
        reference_root=reference_root,
        state_ids=state_ids,
        state_frame=state_frame,
        ilc_column_counts=counts,
        available_mask=available_mask,
        theta_y_a_actual=theta_a,
        theta_y_c_actual=theta_c,
        common_b_input=common_b_input,
        phi_b=phi_b,
        frozen_a2_fingerprint=frozen_a2,
        frozen_a3_fingerprint=frozen_a3,
        frozen_c2_query=frozen_c2,
        previous_real_b_distance=previous_real_b_distance,
        previous_real_b_ranking=previous_real_b_ranking,
        previous_c3a2_distance=previous_c3_distance,
        previous_c3a2_ranking=previous_c3_ranking,
        previous_c3a2_results=previous_results,
        previous_c3a2_diagnostics=previous_diagnostics,
        previous_c3a2_summary=previous_summary,
        a123_validation=a123_validation,
    )


def build_a_end_stage_map(inputs: C2AendInputs) -> pd.DataFrame:
    frame = inputs.state_frame.copy()
    frame["N_ilc_available"] = inputs.ilc_column_counts
    frame["ilc_A_end"] = inputs.ilc_column_counts
    frame["C2_available"] = inputs.ilc_column_counts >= C2_STAGE
    return frame


def validate_c2_availability(ilc_column_counts: np.ndarray) -> bool:
    """Require a real second ILC column for every formal Query state."""

    counts = np.asarray(ilc_column_counts)
    if counts.ndim != 1 or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("ilc_column_counts必须是一维整数数组")
    missing_states = np.flatnonzero(counts < C2_STAGE).tolist()
    if missing_states:
        raise RuntimeError(f"正式C2不可用，存在Ns<2的状态：{missing_states}")
    return True


def select_a_end_stage_indices(ilc_column_counts: np.ndarray) -> np.ndarray:
    """Map each actual ILC count to its zero-based final-stage theta index."""

    counts = np.asarray(ilc_column_counts)
    if counts.ndim != 1 or not np.issubdtype(counts.dtype, np.integer):
        raise ValueError("ilc_column_counts必须是一维整数数组")
    if np.any(counts < C2_STAGE) or np.any(counts > NMAX):
        raise ValueError("A_end要求每个状态的实际ILC列数位于2...5")
    return counts.astype(np.int64, copy=False) - 1


def dpd_shareable_mask(real_b_cnmse: np.ndarray) -> np.ndarray:
    """Apply the formal strict ``Real-B CNMSE < -40 dB`` rule."""

    return np.asarray(real_b_cnmse, dtype=float) < DPD_SHAREABLE_THRESHOLD_DB


def _fingerprints_from_theta(
    inputs: C2AendInputs, theta: np.ndarray, stage_index: int | None = None
) -> np.ndarray:
    if stage_index is None:
        stage_indices = select_a_end_stage_indices(inputs.ilc_column_counts)
        selected = theta[np.arange(STATE_COUNT), stage_indices, :]
    else:
        selected = theta[:, stage_index, :]
    if not np.all(np.isfinite(selected)):
        raise ValueError("选中的冻结theta包含非有限值")
    fingerprints = (inputs.phi_b @ selected.T).T.astype(np.complex128, copy=False)
    return _validate_fingerprint(fingerprints, "重建fingerprint")


def build_a_end_fingerprint(inputs: C2AendInputs) -> np.ndarray:
    return _fingerprints_from_theta(inputs, inputs.theta_y_a_actual)


def build_c2_query_fingerprint(inputs: C2AendInputs) -> np.ndarray:
    return _fingerprints_from_theta(inputs, inputs.theta_y_c_actual, C2_STAGE - 1)


def a_end_regression(
    inputs: C2AendInputs, a_end_fingerprints: np.ndarray
) -> dict[str, Any]:
    stage_errors: dict[str, float] = {}
    stage_counts: dict[str, int] = {}
    maximum = 0.0
    for stage in AEND_STAGES:
        mask = inputs.ilc_column_counts == stage
        stage_counts[str(stage)] = int(np.count_nonzero(mask))
        if not np.any(mask):
            stage_errors[str(stage)] = 0.0
            continue
        stage_theta = inputs.theta_y_a_actual[mask, stage - 1, :]
        stage_fingerprints = (
            inputs.phi_b @ stage_theta.T
        ).T.astype(np.complex128, copy=False)
        error = float(np.max(np.abs(a_end_fingerprints[mask] - stage_fingerprints)))
        stage_errors[str(stage)] = error
        maximum = max(maximum, error)
    a123_stage_errors = {
        "A2": float(
            np.max(
                np.abs(
                    a_end_fingerprints[inputs.ilc_column_counts == 2]
                    - inputs.frozen_a2_fingerprint[inputs.ilc_column_counts == 2]
                )
            )
        ),
        "A3": float(
            np.max(
                np.abs(
                    a_end_fingerprints[inputs.ilc_column_counts == 3]
                    - inputs.frozen_a3_fingerprint[inputs.ilc_column_counts == 3]
                )
            )
        ),
    }
    maximum = max(maximum, *a123_stage_errors.values())
    return {
        "stage_max_abs_error": stage_errors,
        "stage_state_count": stage_counts,
        "a123_A2_A3_max_abs_error": a123_stage_errors,
        "max_abs_error": maximum,
        "pass": bool(maximum < 1e-12),
    }


def c2_regression(inputs: C2AendInputs, c2_query: np.ndarray) -> dict[str, Any]:
    error = float(np.max(np.abs(c2_query - inputs.frozen_c2_query)))
    return {"max_abs_error": error, "pass": bool(error < 1e-12)}


def _stable_order(row: np.ndarray, state_ids: np.ndarray) -> np.ndarray:
    if row.shape != (STATE_COUNT,):
        raise ValueError("距离行必须长度为425")
    if np.isnan(row).any() or np.isposinf(row).any():
        raise ValueError("距离行包含NaN或+Inf")
    return np.lexsort((state_ids, row))


def _retrieve_top1(
    distance: np.ndarray, state_ids: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    distance = _validate_distance(distance, "C2→Aend距离")
    state_ids = _validate_state_ids(state_ids)
    selected = np.empty(STATE_COUNT, dtype=np.int64)
    selected_distance = np.empty(STATE_COUNT, dtype=np.float64)
    true_rank = np.empty(STATE_COUNT, dtype=np.int64)
    tie_count = np.empty(STATE_COUNT, dtype=np.int64)
    orders = np.empty((STATE_COUNT, STATE_COUNT), dtype=np.int64)
    for state_id in range(STATE_COUNT):
        order = _stable_order(distance[state_id], state_ids)
        orders[state_id] = order
        minimum = distance[state_id, order[0]]
        tie_count[state_id] = int(np.count_nonzero(distance[state_id] == minimum))
        selected_index = int(order[0])
        selected[state_id] = int(state_ids[selected_index])
        selected_distance[state_id] = distance[state_id, selected_index]
        positions = np.flatnonzero(order == state_id)
        if positions.size != 1:
            raise RuntimeError(f"无法定位state_id={state_id}在Aend LUT中的排名")
        true_rank[state_id] = int(positions[0] + 1)
    return selected, selected_distance, true_rank, tie_count, orders


def circular_difference_deg(left: float, right: float) -> float:
    left_value = float(left) % 360.0
    right_value = float(right) % 360.0
    difference = abs(left_value - right_value)
    return float(min(difference, 360.0 - difference))


def _condition_differences(real_row: pd.Series, query_row: pd.Series) -> dict[str, Any]:
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
    inputs: C2AendInputs,
    a_end_stage_map: pd.DataFrame,
    distance: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    selected, selected_distance, true_rank, tie_count, orders = _retrieve_top1(
        distance, inputs.state_ids
    )
    lookup = a_end_stage_map.set_index("state_id")
    rows: list[dict[str, Any]] = []
    for real_id in inputs.state_ids:
        real_id = int(real_id)
        query_id = int(selected[real_id])
        real_row = lookup.loc[real_id]
        query_row = lookup.loc[query_id]
        condition = _condition_differences(real_row, query_row)
        rows.append(
            {
                "State_n_R": real_id,
                "State_n_Q": query_id,
                "exact_hit": bool(query_id == real_id),
                "true_state_rank": int(true_rank[real_id]),
                "top1_true_state_hit": bool(true_rank[real_id] <= 1),
                "top3_true_state_hit": bool(true_rank[real_id] <= 3),
                "top5_true_state_hit": bool(true_rank[real_id] <= 5),
                "query_to_selected_LUT_CNMSE_dB": float(selected_distance[real_id]),
                "state_id_delta": query_id - real_id,
                "abs_state_id_delta": abs(query_id - real_id),
                "Aend_stage_R": int(real_row.ilc_A_end),
                "Aend_stage_Q": int(query_row.ilc_A_end),
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
    return result, selected, selected_distance, true_rank, orders


def _nearest_nonself(
    distance: np.ndarray, state_id: int, state_ids: np.ndarray
) -> tuple[int, float]:
    mask = np.ones(STATE_COUNT, dtype=bool)
    mask[state_id] = False
    candidates = state_ids[mask]
    candidate_distance = distance[state_id, mask]
    order = np.lexsort((candidates, candidate_distance))
    index = int(order[0])
    return int(candidates[index]), float(candidate_distance[index])


def build_real_b_diagnostics(
    inputs: C2AendInputs,
    real_b_distance: np.ndarray,
    retrieval_results: pd.DataFrame,
) -> pd.DataFrame:
    real_b_distance = _validate_distance(real_b_distance, "Real-B距离")
    rows: list[dict[str, Any]] = []
    for record in retrieval_results.itertuples(index=False):
        real_id = int(record.State_n_R)
        query_id = int(record.State_n_Q)
        retrieved = float(real_b_distance[real_id, query_id])
        shareable = bool(dpd_shareable_mask(np.asarray([retrieved]))[0])
        oracle_state, oracle_value = _nearest_nonself(real_b_distance, real_id, inputs.state_ids)
        if bool(record.exact_hit):
            regret = float("nan")
            margin = float("nan")
        elif np.isfinite(retrieved) and np.isfinite(oracle_value):
            regret = float(retrieved - oracle_value)
            margin = float(retrieved - DPD_SHAREABLE_THRESHOLD_DB)
        else:
            regret = float("nan")
            margin = float("nan")
        rows.append(
            {
                "State_n_R": real_id,
                "State_n_Q": query_id,
                "exact_hit": bool(record.exact_hit),
                "retrieved_real_B_CNMSE_dB": retrieved,
                "real_B_retrieval_CNMSE_dB": retrieved,
                "dpd_shareable": shareable,
                "failure": bool(not shareable),
                "distance_above_threshold_dB": margin,
                "nearest_nonself_B_state": oracle_state,
                "nearest_nonself_B_CNMSE_dB": oracle_value,
                "nonself_behavioral_regret_dB": regret,
                "behavioral_regret_dB": regret,
                "nonexact_retrieval_nonfinite": bool(
                    not bool(record.exact_hit) and not np.isfinite(retrieved)
                ),
            }
        )
    return pd.DataFrame(rows)


def build_oracle_shareable_rank(
    inputs: C2AendInputs,
    retrieval_results: pd.DataFrame,
    real_b_distance: np.ndarray,
    retrieval_orders: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in retrieval_results.itertuples(index=False):
        real_id = int(record.State_n_R)
        order = retrieval_orders[real_id]
        ordered_distance = real_b_distance[real_id, order]
        shareable_positions = np.flatnonzero(dpd_shareable_mask(ordered_distance))
        if shareable_positions.size:
            position = int(shareable_positions[0])
            first_rank = position + 1
            first_state = int(inputs.state_ids[order[position]])
            first_distance = float(ordered_distance[position])
        else:
            first_rank = np.nan
            first_state = np.nan
            first_distance = np.nan
        top_shareable = ordered_distance < DPD_SHAREABLE_THRESHOLD_DB
        rows.append(
            {
                "State_n_R": real_id,
                "top1_State_n_Q": int(record.State_n_Q),
                "top1_shareable": bool(top_shareable[:1].any()),
                "first_shareable_rank": first_rank,
                "first_shareable_candidate_rank": first_rank,
                "first_shareable_state": first_state,
                "first_shareable_candidate_state": first_state,
                "first_shareable_real_B_CNMSE_dB": first_distance,
                "shareable_in_top3": bool(top_shareable[:3].any()),
                "shareable_in_top5": bool(top_shareable[:5].any()),
                "shareable_in_top10": bool(top_shareable[:10].any()),
            }
        )
    return pd.DataFrame(rows)


def _finite_stats(values: np.ndarray, prefix: str) -> dict[str, Any]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_mean": np.nan,
            f"{prefix}_median": np.nan,
            f"{prefix}_std": np.nan,
            f"{prefix}_q05": np.nan,
            f"{prefix}_q25": np.nan,
            f"{prefix}_q75": np.nan,
            f"{prefix}_q95": np.nan,
            f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan,
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


def _finite_nonexact(values: np.ndarray, exact: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    exact = np.asarray(exact, dtype=bool)
    return values[(~exact) & np.isfinite(values)]


def build_condition_region_summary(
    retrieval_results: pd.DataFrame,
    real_b_diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    if "retrieved_real_B_CNMSE_dB" in retrieval_results.columns:
        merged = retrieval_results.copy()
    else:
        merged = retrieval_results.merge(
            real_b_diagnostics.loc[
                :,
                [
                    "State_n_R",
                    "State_n_Q",
                    "exact_hit",
                    "retrieved_real_B_CNMSE_dB",
                    "dpd_shareable",
                    "failure",
                ],
            ],
            on=["State_n_R", "State_n_Q", "exact_hit"],
            how="left",
            validate="one_to_one",
        )
    rows: list[dict[str, Any]] = []
    group_columns = ["funMng_R", "funAng_R", "secMng_R", "secAng_R"]
    for keys, group in merged.groupby(group_columns, sort=True, dropna=False):
        values = group["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float)
        exact = group["exact_hit"].to_numpy(dtype=bool)
        nonexact = _finite_nonexact(values, exact)
        rows.append(
            {
                **dict(zip(group_columns, keys, strict=True)),
                "state_count": int(group.shape[0]),
                "exact_hit_count": int(exact.sum()),
                "exact_hit_rate": float(exact.mean()),
                "dpd_shareable_count": int(group["dpd_shareable"].sum()),
                "dpd_shareable_rate": float(group["dpd_shareable"].mean()),
                "failure_count": int(group["failure"].sum()),
                "non_exact_median_real_B_CNMSE": (
                    float(np.median(nonexact)) if nonexact.size else np.nan
                ),
                "median_true_state_rank": float(group["true_state_rank"].median()),
            }
        )
    return pd.DataFrame(rows)


def _build_failed_states(
    retrieval_results: pd.DataFrame,
    diagnostics: pd.DataFrame,
    oracle: pd.DataFrame,
) -> pd.DataFrame:
    diagnostics_extra = diagnostics.loc[
        :,
        [
            "State_n_R",
            "State_n_Q",
            "exact_hit",
            "nearest_nonself_B_state",
            "nearest_nonself_B_CNMSE_dB",
            "nonself_behavioral_regret_dB",
            "behavioral_regret_dB",
            "nonexact_retrieval_nonfinite",
        ],
    ]
    merged = retrieval_results.merge(
        diagnostics_extra,
        on=["State_n_R", "State_n_Q", "exact_hit"],
        how="left",
        validate="one_to_one",
    ).merge(oracle, on="State_n_R", how="left", validate="one_to_one")
    failed = merged.loc[merged["failure"]].copy()
    failed = failed.sort_values(
        ["retrieved_real_B_CNMSE_dB", "State_n_R"],
        ascending=[False, True],
        na_position="last",
    )
    return failed.reset_index(drop=True)


def _build_retrieval_summary(
    inputs: C2AendInputs,
    stage_map: pd.DataFrame,
    retrieval_results: pd.DataFrame,
    diagnostics: pd.DataFrame,
    oracle: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    exact = retrieval_results["exact_hit"].to_numpy(dtype=bool)
    values = diagnostics["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float)
    nonexact_values = _finite_nonexact(values, exact)
    regrets = diagnostics.loc[~exact, "behavioral_regret_dB"].to_numpy(dtype=float)
    finite_regrets = regrets[np.isfinite(regrets)]
    shareable = diagnostics["dpd_shareable"].to_numpy(dtype=bool)
    summary: dict[str, Any] = {
        "state_count": STATE_COUNT,
        "candidate_count": STATE_COUNT,
        "candidate_count_per_query": STATE_COUNT,
        "query": "Y-C2",
        "lut": "Y-Aend",
        "self_match_included": True,
        "C2_available_count": int(stage_map["C2_available"].sum()),
        "exact_hit_count": int(exact.sum()),
        "exact_hit_rate": float(exact.mean()),
        "non_exact_count": int((~exact).sum()),
        "non_exact_finite_count": int(nonexact_values.size),
        "non_exact_nonfinite_count": int((~exact).sum() - nonexact_values.size),
        "true_state_rank_median": float(retrieval_results["true_state_rank"].median()),
        "true_state_rank_max": int(retrieval_results["true_state_rank"].max()),
        "minimum_tie_count_rows": int(
            (retrieval_results["minimum_tie_count"].to_numpy(dtype=int) > 1).sum()
        ),
        "top3_true_state_hit_count": int(retrieval_results["top3_true_state_hit"].sum()),
        "top3_true_state_hit_rate": float(retrieval_results["top3_true_state_hit"].mean()),
        "top5_true_state_hit_count": int(retrieval_results["top5_true_state_hit"].sum()),
        "top5_true_state_hit_rate": float(retrieval_results["top5_true_state_hit"].mean()),
        "DPD_shareable_threshold_dB": DPD_SHAREABLE_THRESHOLD_DB,
        "DPD_shareable_threshold_rule": "Real-B CNMSE < -40 dB",
        "DPD_shareable_count": int(shareable.sum()),
        "DPD_shareable_rate": float(shareable.mean()),
        "non_exact_shareable_count": int((shareable & ~exact).sum()),
        "non_exact_shareable_rate": float((shareable & ~exact).sum() / max((~exact).sum(), 1)),
        "non_exact_nonshareable_count": int((~shareable & ~exact).sum()),
        "failure_count": int((~shareable).sum()),
        "top1_shareable_rate": float(oracle["top1_shareable"].mean()),
        "top3_shareable_oracle_coverage": float(oracle["shareable_in_top3"].mean()),
        "top5_shareable_oracle_coverage": float(oracle["shareable_in_top5"].mean()),
        "top10_shareable_oracle_coverage": float(oracle["shareable_in_top10"].mean()),
        "regret_count": int(finite_regrets.size),
        "regret_mean": float(np.mean(finite_regrets)) if finite_regrets.size else np.nan,
        "regret_median": float(np.median(finite_regrets)) if finite_regrets.size else np.nan,
        "regret_max": float(np.max(finite_regrets)) if finite_regrets.size else np.nan,
    }
    summary.update(_finite_stats(nonexact_values, "non_exact_B_CNMSE"))
    for stage in AEND_STAGES:
        summary[f"A{stage}_end_count"] = int((stage_map["ilc_A_end"] == stage).sum())
    summary_frame = pd.DataFrame([summary])

    shareable_rows = [
        ("Total states", STATE_COUNT),
        ("Exact hits", int(exact.sum())),
        ("Exact Hit Rate", float(exact.mean())),
        ("Non-exact states", int((~exact).sum())),
        ("Non-exact but DPD-shareable", int((shareable & ~exact).sum())),
        ("Non-exact non-shareable", int((~shareable & ~exact).sum())),
        ("Total DPD-shareable retrievals", int(shareable.sum())),
        ("DPD-shareable Retrieval Rate", float(shareable.mean())),
        ("Formal threshold", "Real-B CNMSE < -40 dB"),
        ("Strict threshold", True),
    ]
    shareable_summary = pd.DataFrame(shareable_rows, columns=["metric", "value"])
    return summary_frame, shareable_summary


def _comparison_metrics(
    results: pd.DataFrame, diagnostics: pd.DataFrame
) -> dict[str, float]:
    exact = results["exact_hit"].to_numpy(dtype=bool)
    values = diagnostics["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float)
    shareable = values < DPD_SHAREABLE_THRESHOLD_DB
    nonexact = _finite_nonexact(values, exact)
    regrets = diagnostics.loc[~exact, "nonself_behavioral_regret_dB"].to_numpy(dtype=float)
    regrets = regrets[np.isfinite(regrets)]
    return {
        "exact_hit_count": float(exact.sum()),
        "exact_hit_rate": float(exact.mean()),
        "top3_true_state_hit_rate": float(results["top3_hit"].mean())
        if "top3_hit" in results
        else float(results["top3_true_state_hit"].mean()),
        "top5_true_state_hit_rate": float(results["top5_hit"].mean())
        if "top5_hit" in results
        else float(results["top5_true_state_hit"].mean()),
        "DPD_shareable_count": float(shareable.sum()),
        "DPD_shareable_rate": float(shareable.mean()),
        "failure_count": float((~shareable).sum()),
        "non_exact_B_CNMSE_median": float(np.median(nonexact)),
        "non_exact_B_CNMSE_q95": float(np.quantile(nonexact, 0.95)),
        "non_exact_B_CNMSE_max": float(np.max(nonexact)),
        "regret_median": float(np.median(regrets)) if regrets.size else np.nan,
        "regret_max": float(np.max(regrets)) if regrets.size else np.nan,
    }


def build_comparison_vs_c3_a2(
    inputs: C2AendInputs,
    retrieval_results: pd.DataFrame,
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    old = _comparison_metrics(inputs.previous_c3a2_results, inputs.previous_c3a2_diagnostics)
    new = _comparison_metrics(retrieval_results, diagnostics)
    rows = [
        {
            "metric": metric,
            "C3_to_A2": old[metric],
            "C2_to_Aend": new[metric],
            "delta": new[metric] - old[metric],
        }
        for metric in old
    ]
    return pd.DataFrame(rows)


def run_c2_aend_retrieval_analysis(
    all_ilc_root: Path,
    a123_root: Path,
    previous_c3a2_root: Path,
    reference_root: Path,
) -> C2AendResult:
    """Run the complete fixed-method, all-425-state C2-to-A_end analysis."""

    inputs = load_c2_aend_inputs(all_ilc_root, a123_root, previous_c3a2_root, reference_root)
    stage_map = build_a_end_stage_map(inputs)
    a_end = build_a_end_fingerprint(inputs)
    c2_query = build_c2_query_fingerprint(inputs)
    a_end_validation = a_end_regression(inputs, a_end)
    c2_validation = c2_regression(inputs, c2_query)
    if not a_end_validation["pass"] or not c2_validation["pass"]:
        raise RuntimeError("A_end或C2 fingerprint冻结回归失败")
    retrieval_distance = compute_cnmse_distance_matrix(c2_query, a_end)
    retrieval_ranking = distance_to_ranks(retrieval_distance)
    real_b_distance = inputs.previous_real_b_distance.copy()
    real_b_ranking = inputs.previous_real_b_ranking.copy()
    retrieval_results, _, _, _, orders = build_retrieval_results(
        inputs, stage_map, retrieval_distance
    )
    diagnostics = build_real_b_diagnostics(inputs, real_b_distance, retrieval_results)
    retrieval_results = retrieval_results.merge(
        diagnostics.loc[
            :, [
                "State_n_R",
                "State_n_Q",
                "exact_hit",
                "retrieved_real_B_CNMSE_dB",
                "dpd_shareable",
                "failure",
                "distance_above_threshold_dB",
            ]
        ],
        on=["State_n_R", "State_n_Q", "exact_hit"],
        how="left",
        validate="one_to_one",
    )
    oracle = build_oracle_shareable_rank(inputs, retrieval_results, real_b_distance, orders)
    failed = _build_failed_states(retrieval_results, diagnostics, oracle)
    condition = build_condition_region_summary(retrieval_results, diagnostics)
    summary, shareable_summary = _build_retrieval_summary(
        inputs, stage_map, retrieval_results, diagnostics, oracle
    )
    comparison = build_comparison_vs_c3_a2(inputs, retrieval_results, diagnostics)
    real_b_regression = {
        "distance_max_abs_error_vs_R_RR": 0.0,
        "ranking_max_abs_error_vs_R_RR": 0.0,
        "ranking_identical_vs_R_RR": True,
        "diagonal_all_negative_infinity": bool(np.all(np.isneginf(np.diag(real_b_distance)))),
        "pass": True,
    }
    validation = {
        "scenario": 2,
        "study": (
            "C2 online Y behavior fingerprint to state-specific A-end Y LUT retrieval "
            "with Real-B DPD-shareability validation"
        ),
        "query_fingerprint_type": "Y-C2",
        "lut_fingerprint_type": "Y-Aend",
        "query_ILC_stage": C2_STAGE,
        "query_stage_must_exist": True,
        "query_saturation_allowed": False,
        "lut_stage_rule": "A_end = last actually available ILC column for each state",
        "state_count": STATE_COUNT,
        "candidate_count": STATE_COUNT,
        "candidate_count_per_query": STATE_COUNT,
        "self_match_included": True,
        "retrieval_metric": "CNMSE",
        "retrieval_direction": "minimum; more negative is more similar",
        "real_B_used_as_LUT": False,
        "real_B_used_as_reference": True,
        "real_B_signal": "canonical OFF B segment from yout_withoutdpd_ori",
        "real_B_matrix_reused_from": str(inputs.previous_c3a2_root / "real_B_distance_matrix.npz"),
        "dpd_shareable_threshold_frozen": True,
        "dpd_shareable_threshold_dB": DPD_SHAREABLE_THRESHOLD_DB,
        "dpd_shareable_rule": "Real-B CNMSE < -40 dB",
        "threshold_is_strict": True,
        "exact_hit_definition": "State_n_Q == State_n_R",
        "models_retrained": False,
        "ridge_rescanned": False,
        "mp_changed": False,
        "ABC_changed": False,
        "canonical_changed": False,
        "common_B_probe_changed": False,
        "raw_data_modified": False,
        "raw_data_read": False,
        "C2_available_count": int((inputs.ilc_column_counts >= C2_STAGE).sum()),
        "C2_all_states_available": bool(np.all(inputs.ilc_column_counts >= C2_STAGE)),
        "N_ilc_distribution": {
            str(stage): int(np.count_nonzero(inputs.ilc_column_counts == stage))
            for stage in range(1, NMAX + 1)
        },
        "A_end_stage_distribution": {
            f"A{stage}": int(np.count_nonzero(inputs.ilc_column_counts == stage))
            for stage in AEND_STAGES
        },
        "A_end_fingerprint_shape": list(a_end.shape),
        "C2_query_shape": list(c2_query.shape),
        "A_end_fingerprint_dtype": str(a_end.dtype),
        "C2_query_dtype": str(c2_query.dtype),
        "A_end_all_finite": bool(np.all(np.isfinite(a_end))),
        "C2_all_finite": bool(np.all(np.isfinite(c2_query))),
        "a_end_fingerprint_regression": a_end_validation,
        "c2_fingerprint_regression": c2_validation,
        "retrieval_distance_shape": list(retrieval_distance.shape),
        "retrieval_ranking_shape": list(retrieval_ranking.shape),
        "real_B_distance_shape": list(real_b_distance.shape),
        "real_B_ranking_shape": list(real_b_ranking.shape),
        "minimum_tie_count_rows": int(
            (retrieval_results["minimum_tie_count"].to_numpy(dtype=int) > 1).sum()
        ),
        "real_B_regression": real_b_regression,
        "summary": summary.iloc[0].to_dict(),
        "comparison_vs_C3_to_A2": comparison.to_dict("records"),
    }
    return C2AendResult(
        inputs=inputs,
        a_end_stage_map=stage_map,
        a_end_fingerprints=a_end,
        c2_query_fingerprint=c2_query,
        retrieval_distance=retrieval_distance,
        retrieval_ranking=retrieval_ranking,
        real_b_distance=real_b_distance,
        real_b_ranking=real_b_ranking,
        retrieval_results=retrieval_results,
        real_b_diagnostics=diagnostics,
        oracle_shareable_rank=oracle,
        failed_retrieval_states=failed,
        condition_region_summary=condition,
        retrieval_summary=summary,
        dpd_shareable_summary=shareable_summary,
        comparison_vs_c3_a2=comparison,
        validation=validation,
    )


__all__ = [
    "AEND_STAGES",
    "C2AendInputs",
    "C2AendResult",
    "C2_STAGE",
    "DPD_SHAREABLE_THRESHOLD_DB",
    "FINGERPRINT_LENGTH",
    "STATE_COUNT",
    "a_end_regression",
    "build_a_end_fingerprint",
    "build_a_end_stage_map",
    "build_c2_query_fingerprint",
    "build_oracle_shareable_rank",
    "build_retrieval_results",
    "build_real_b_diagnostics",
    "circular_difference_deg",
    "load_c2_aend_inputs",
    "run_c2_aend_retrieval_analysis",
]
