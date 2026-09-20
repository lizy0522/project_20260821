"""Scenario 2 fixed 1B behavior-fingerprint LUT retrieval.

This module intentionally keeps the formal Scenario 2 model, preprocessing,
ABC ownership, candidate set and Real-B ground truth unchanged.  It inserts
only the frozen low-bandwidth observation operator after 5B basis construction
and model-valid support cropping:

    x_5B -> Phi(x_5B) -> valid support -> H_1B -> Phi_1B

The 1B operator is used for Aend/C2 model fitting and fingerprint retrieval.
The final retrieved Real-B CNMSE is always looked up from the canonical 5B
OFF-B matrix and is never used to select the candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from behavior_fingerprint_ranking_consistency.shared.ranking import distance_to_ranks  # noqa: E402
from behavior_modeling.shared.basis import build_mp_basis  # noqa: E402
from behavior_modeling.shared.config import MAX_DELAY, MP_CONFIG, NUM_COEFFICIENTS  # noqa: E402
from behavior_modeling.shared.evaluation import calculate_nmse  # noqa: E402
from behavior_modeling.shared.ridge import fit_coefficients_ridge  # noqa: E402
from core.shared.project_paths import legacy_raw_manifest  # noqa: E402
from data_management.shared import build_state_table, get_state_info, load_by_id  # noqa: E402
from low_bandwidth_behavior_analysis.shared.behavior_indexed_dpd_signal import (  # noqa: E402
    LowBandwidthObservationBank,
    LowBandwidthObservationOperator,
)
from signal_segmentation.shared import (  # noqa: E402
    build_partition_from_xin,
    get_ilc_pair,
    preprocess_full_pair,
)

# Nature-figure contract: the figure defends the claim that a fixed 1B
# behavior fingerprint can choose a LUT candidate whose final full-rate 5B
# Real-B behavior remains DPD-shareable.  The archetype is a quantitative
# statewise grid/trend figure.  Python/matplotlib is the exclusive backend.
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["legend.frameon"] = False


TASK_NAME = "scenario_2_C2_to_Aend_1b"
RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_retrieval"
    / "scenario_2" /TASK_NAME
    / "scenario_2_C2_to_Aend_1B"
)
ALL_ILC_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2" /"scenario2_all_ilc_analysis"
    / "scenario_2"
)
BASELINE_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_retrieval"
    / "scenario_2" /"scenario_2_C2_to_Aend_retrieval"
    / "scenario_2_C2_to_Aend"
)
LOW_BANDWIDTH_ROOT = (
    PROJECT_ROOT
    / "results"
    / "low_bandwidth_behavior_analysis"
    / "scenario_2" /"low_bandwidth_observation"
    / "sample_rate_operator_bank"
)

STATE_COUNT = 425
FINGERPRINT_LENGTH_5B = 4913
C2_STAGE = 2
RIDGE_LAMBDA = 1e-8
SHAREABLE_THRESHOLD_DB = -40.0
IDENTITY_INDEX_K = 50
RANDOM_SEED = 20260907

EXPECTED_SEGMENT_LENGTHS_5B = {"A": 12286, "B": 4913, "C": 7371}
EXPECTED_SEGMENT_LENGTHS_1B = {"A": 2458, "B": 983, "C": 1475}
ONE_B_RESULT_ROOT = RESULT_ROOT


@dataclass(frozen=True)
class ObservationExperimentSpec:
    """One fixed observation point using the shared operator bank."""

    result_tag: str
    bandwidth_label: str
    normalized_n: float
    observation_index_k: int
    fs_obs_hz: int
    expected_segment_lengths: dict[str, int]
    result_root: Path


SPEC_1B = ObservationExperimentSpec(
    result_tag="1B",
    bandwidth_label="1B",
    normalized_n=1.0,
    observation_index_k=10,
    fs_obs_hz=20_000_000,
    expected_segment_lengths=EXPECTED_SEGMENT_LENGTHS_1B,
    result_root=ONE_B_RESULT_ROOT,
)
SPEC_0P5B = ObservationExperimentSpec(
    result_tag="0p5B",
    bandwidth_label="0.5B",
    normalized_n=0.5,
    observation_index_k=5,
    fs_obs_hz=10_000_000,
    expected_segment_lengths={"A": 1229, "B": 492, "C": 738},
    result_root=PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_retrieval"
    / "scenario_2" /TASK_NAME
    / "scenario_2_C2_to_Aend_0p5B",
)

# Backward-compatible aliases retained for the completed 1B tests and callers.
OBSERVATION_N = SPEC_1B.normalized_n
OBSERVATION_INDEX_K = SPEC_1B.observation_index_k
OBSERVATION_FS_HZ = SPEC_1B.fs_obs_hz

EXPECTED_MAIN_COLUMNS = [
    "state_id_R",
    "load_config",
    "nmse_withoutdpd_dB",
    "acpr_withoutdpd_avg_dBc",
    "Y_Aend_train_NMSE_dB",
    "Y_Aend_B_NMSE_dB",
    "Y_C2_train_NMSE_dB",
    "Y_C2_B_NMSE_dB",
    "state_id_Q",
    "retrieved_real_B_CNMSE_dB",
]


@dataclass(frozen=True)
class BaselineArtifacts:
    """Read-only formal 5B references used by the regression guard."""

    state_ids: np.ndarray
    common_b_input: np.ndarray
    theta_y_a_actual: np.ndarray
    theta_y_c_actual: np.ndarray
    ilc_column_counts: np.ndarray
    aend_fingerprints: np.ndarray
    c2_fingerprints: np.ndarray
    retrieval_distance: np.ndarray
    retrieval_ranking: np.ndarray
    real_b_distance: np.ndarray
    real_b_ranking: np.ndarray
    retrieval_results: pd.DataFrame
    model_metrics: pd.DataFrame


@dataclass(frozen=True)
class ModelPhase:
    """Per-state models and metrics for one observation operator."""

    operator_label: str
    operator: LowBandwidthObservationOperator
    aend_coefficients: np.ndarray
    c2_coefficients: np.ndarray
    model_metrics: pd.DataFrame
    aend_fingerprints: np.ndarray
    c2_fingerprints: np.ndarray
    length_checks: dict[str, Any]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"无法JSON序列化类型：{type(value)!r}")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=_json_default)
        handle.write("\n")


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        na_rep="NaN",
        float_format="%.17g",
    )


def _tree_digest(path: Path) -> str | None:
    if not path.is_dir():
        return None
    digest = hashlib.sha256()
    files = sorted(
        (
            item
            for item in path.rglob("*")
            if item.is_file()
            and "__pycache__" not in item.parts
            and item.suffix.lower() != ".pyc"
            and not item.name.startswith("~$")
        ),
        key=lambda item: str(item).lower(),
    )
    for file in files:
        content = file.read_bytes()
        digest.update(file.relative_to(path).as_posix().encode() + b"\0")
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def _raw_manifest() -> dict[str, Any]:
    value = legacy_raw_manifest()
    return {
        "sha256": value["sha256"],
        "file_count": value["file_count"],
        "bytes": value["bytes"],
    }


def protection_snapshot() -> dict[str, Any]:
    protected = {
        "scenario_2_C2_to_Aend": BASELINE_ROOT,
        "scenario_2_C2_to_Aend_unified_model_capacity": (
            PROJECT_ROOT
            / "results"
            / "behavior_fingerprint_retrieval"
            / "scenario_2" /TASK_NAME
            / "scenario_2_C2_to_Aend_unified_model_capacity"
        ),
        "scenario_2_C2_to_Aend_equal_ABC": (
            PROJECT_ROOT
            / "results"
            / "behavior_fingerprint_retrieval"
            / "scenario_2" /TASK_NAME
            / "scenario_2_C2_to_Aend_equal_ABC"
        ),
        "scenario_2_C2_to_Aend_retrieval_oriented_model_scan": (
            PROJECT_ROOT
            / "results"
            / TASK_NAME
            / "scenario_2_C2_to_Aend_retrieval_oriented_model_scan"
        ),
        "scenario_2_all_ilc": ALL_ILC_ROOT,
        "low_bandwidth_operator": LOW_BANDWIDTH_ROOT,
        "scenario_2_C2_to_Aend_1B": ONE_B_RESULT_ROOT,
    }
    return {
        "data/raw": _raw_manifest(),
        "result_dirs": {
            name: {"path": str(path), "exists": path.is_dir(), "sha256": _tree_digest(path)}
            for name, path in protected.items()
        },
    }


def verify_protection(before: dict[str, Any], after: dict[str, Any]) -> dict[str, Any]:
    raw_before = before["data/raw"]
    raw_after = after["data/raw"]
    raw = {
        "sha256_unchanged": raw_before["sha256"] == raw_after["sha256"],
        "file_count_unchanged": raw_before["file_count"] == raw_after["file_count"],
        "bytes_unchanged": raw_before["bytes"] == raw_after["bytes"],
    }
    dirs: dict[str, Any] = {}
    for name, item in before["result_dirs"].items():
        after_item = after["result_dirs"][name]
        dirs[name] = {
            "sha256_unchanged": item["sha256"] == after_item["sha256"],
            "before": item["sha256"],
            "after": after_item["sha256"],
        }
    return {
        "data/raw": raw,
        "result_dirs": dirs,
        "all_protected_unchanged": bool(
            all(raw.values()) and all(item["sha256_unchanged"] for item in dirs.values())
        ),
    }


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    _require(path.is_file(), f"缺少NPZ输入：{path}")
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def _validate_state_ids(values: Iterable[Any], name: str) -> np.ndarray:
    array = np.asarray(list(values) if not isinstance(values, np.ndarray) else values)
    _require(array.shape == (STATE_COUNT,), f"{name}必须是长度425的一维数组")
    _require(np.issubdtype(array.dtype, np.integer), f"{name}必须是整数数组")
    array = array.astype(np.int64, copy=False)
    _require(np.array_equal(array, np.arange(STATE_COUNT)), f"{name}必须严格为0...424")
    return array


def _validate_complex_matrix(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    _require(array.shape == shape, f"{name} shape错误：{array.shape} != {shape}")
    _require(array.dtype == np.complex128 and np.iscomplexobj(array), f"{name}必须是complex128")
    _require(np.all(np.isfinite(array)), f"{name}包含NaN或Inf")
    return array


def _validate_theta_history(value: np.ndarray, counts: np.ndarray, name: str) -> np.ndarray:
    """Validate active ILC theta slots while permitting inactive NaN padding."""

    array = np.asarray(value)
    _require(
        array.shape == (STATE_COUNT, 5, NUM_COEFFICIENTS),
        f"{name} shape错误：{array.shape}",
    )
    _require(
        array.dtype == np.complex128 and np.iscomplexobj(array),
        f"{name}必须是complex128",
    )
    active = np.concatenate(
        [array[state_id, : int(counts[state_id]), :] for state_id in range(STATE_COUNT)],
        axis=0,
    )
    _require(np.all(np.isfinite(active)), f"{name}的可用ILC槽位包含NaN或Inf")
    _require(not np.isinf(array).any(), f"{name}包含Inf")
    return array


def _validate_distance_matrix(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    _require(array.shape == (STATE_COUNT, STATE_COUNT), f"{name} shape错误：{array.shape}")
    _require(array.dtype == np.float64, f"{name}必须是float64")
    _require(not np.isnan(array).any() and not np.isposinf(array).any(), f"{name}包含NaN或+Inf")
    return array


def _max_difference_allowing_neginf(left: np.ndarray, right: np.ndarray) -> tuple[float, int]:
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        return float("inf"), max(left.size, right.size)
    if np.iscomplexobj(left) or np.iscomplexobj(right):
        left_finite = np.isfinite(left)
        right_finite = np.isfinite(right)
        mismatch = int(np.count_nonzero(left_finite != right_finite))
        finite = left_finite & right_finite
        if not np.any(finite):
            return 0.0, mismatch
        return float(np.max(np.abs(left[finite] - right[finite]))), mismatch
    mismatch = int(np.count_nonzero(np.isneginf(left) != np.isneginf(right)))
    left_finite = np.isfinite(left)
    right_finite = np.isfinite(right)
    mismatch += int(np.count_nonzero(left_finite != right_finite))
    finite = left_finite & right_finite
    if not np.any(finite):
        return 0.0, mismatch
    return float(np.max(np.abs(left[finite] - right[finite]))), mismatch


def load_baseline_artifacts() -> BaselineArtifacts:
    """Load only frozen 5B references; no old result is rewritten."""

    theta_data = _load_npz(ALL_ILC_ROOT / "all_ilc_theta.npz")
    required_theta = {
        "state_ids",
        "theta_Y_A_actual",
        "theta_Y_C_actual",
        "ilc_column_counts",
    }
    _require(required_theta.issubset(theta_data), "all_ilc_theta缺少冻结字段")
    state_ids = _validate_state_ids(theta_data["state_ids"], "all_ilc_theta.state_ids")
    counts = np.asarray(theta_data["ilc_column_counts"], dtype=np.int64)
    _require(
        counts.shape == (STATE_COUNT,) and np.all((counts >= 2) & (counts <= 5)), "ILC列数契约错误"
    )
    theta_a = _validate_theta_history(theta_data["theta_Y_A_actual"], counts, "theta_Y_A_actual")
    theta_c = _validate_theta_history(theta_data["theta_Y_C_actual"], counts, "theta_Y_C_actual")

    lut_data = _load_npz(BASELINE_ROOT / "lut_fingerprints_Aend.npz")
    query_data = _load_npz(BASELINE_ROOT / "query_fingerprints_C2.npz")
    common_b = np.asarray(lut_data["common_B_input"])
    _validate_complex_matrix(common_b, (4915,), "common_B_input")
    _require(
        np.array_equal(
            state_ids, _validate_state_ids(lut_data["state_ids"], "baseline LUT state_ids")
        ),
        "LUT state顺序不一致",
    )
    _require(
        np.array_equal(
            state_ids, _validate_state_ids(query_data["state_ids"], "baseline Query state_ids")
        ),
        "Query state顺序不一致",
    )
    aend_fp = _validate_complex_matrix(
        lut_data["Y_Aend_fingerprints"],
        (STATE_COUNT, FINGERPRINT_LENGTH_5B),
        "baseline Aend fingerprints",
    )
    c2_fp = _validate_complex_matrix(
        query_data["Q_C2"], (STATE_COUNT, FINGERPRINT_LENGTH_5B), "baseline C2 fingerprints"
    )

    retrieval_data = _load_npz(BASELINE_ROOT / "retrieval_distance_matrix.npz")
    ranking_data = _load_npz(BASELINE_ROOT / "retrieval_ranking_matrix.npz")
    real_b_data = _load_npz(BASELINE_ROOT / "real_B_distance_matrix.npz")
    _validate_state_ids(retrieval_data["state_ids"], "baseline retrieval state_ids")
    _validate_state_ids(ranking_data["state_ids"], "baseline ranking state_ids")
    _validate_state_ids(real_b_data["state_ids"], "baseline Real-B state_ids")
    d_retrieval = _validate_distance_matrix(retrieval_data["D_C2_Aend"], "baseline D_C2_Aend")
    r_retrieval = _validate_distance_matrix(ranking_data["R_C2_Aend"], "baseline R_C2_Aend")
    d_real_b = _validate_distance_matrix(real_b_data["D_B"], "baseline D_B")
    r_real_b = _validate_distance_matrix(real_b_data["R_B"], "baseline R_B")
    results = pd.read_csv(BASELINE_ROOT / "retrieval_results.csv")
    _require(results.shape[0] == STATE_COUNT, "baseline retrieval_results必须有425行")
    metrics = pd.read_csv(ALL_ILC_ROOT / "all_ilc_model_metrics.csv")
    return BaselineArtifacts(
        state_ids=state_ids,
        common_b_input=common_b,
        theta_y_a_actual=theta_a,
        theta_y_c_actual=theta_c,
        ilc_column_counts=counts,
        aend_fingerprints=aend_fp,
        c2_fingerprints=c2_fp,
        retrieval_distance=d_retrieval,
        retrieval_ranking=r_retrieval,
        real_b_distance=d_real_b,
        real_b_ranking=r_real_b,
        retrieval_results=results,
        model_metrics=metrics,
    )


def _validate_partition_lengths(partition: Any) -> None:
    _require(partition.total_length == 24576, f"总长度必须为24576，实际{partition.total_length}")
    _require(
        partition.n_a == 12288 and partition.n_b == 4915 and partition.n_c == 7373,
        "ABC ownership不符合正式非等长分段",
    )


def _fit_one_segment(
    *,
    segment_name: str,
    canonical: Any,
    operator: LowBandwidthObservationOperator,
    expected_observed_lengths: dict[str, int],
    observation_index_k: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    segment = canonical[segment_name]
    phi_5b = build_mp_basis(segment.input, MP_CONFIG["orders"], MP_CONFIG["memory_depth"])
    y_5b = np.asarray(segment.output[MAX_DELAY:], dtype=np.complex128)
    _require(
        phi_5b.shape[0] == EXPECTED_SEGMENT_LENGTHS_5B[segment_name],
        f"{segment_name} 5B basis长度错误",
    )
    _require(
        y_5b.shape == (EXPECTED_SEGMENT_LENGTHS_5B[segment_name],), f"{segment_name} 5B输出长度错误"
    )
    phi_obs, y_obs = operator.apply_pair(phi_5b, y_5b)
    expected_obs = operator.expected_output_length(phi_5b.shape[0])
    _require(
        phi_obs.shape[0] == expected_obs == expected_observed_lengths[segment_name]
        if operator.spec.sample_rate_index_k == observation_index_k
        else phi_obs.shape[0] == expected_obs,
        f"{segment_name}观测后长度错误",
    )
    if operator.spec.sample_rate_index_k == observation_index_k:
        _require(
            phi_obs.shape[0] == expected_observed_lengths[segment_name],
            f"{segment_name}观测长度与spec不一致",
        )
    _require(y_obs.shape[0] == phi_obs.shape[0], f"{segment_name} E/y观测长度不一致")
    theta, diagnostics = fit_coefficients_ridge(phi_obs, y_obs, RIDGE_LAMBDA)
    return (
        phi_obs,
        y_obs,
        {
            "theta": theta,
            "train_nmse_db": calculate_nmse(y_obs, phi_obs @ theta),
            "diagnostics": diagnostics,
            "basis_5b": phi_5b,
            "output_5b": y_5b,
        },
    )


def _fit_role(
    *,
    role: str,
    train_segment_name: str,
    canonical: Any,
    operator: LowBandwidthObservationOperator,
    expected_observed_lengths: dict[str, int],
    observation_index_k: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    train = _fit_one_segment(
        segment_name=train_segment_name,
        canonical=canonical,
        operator=operator,
        expected_observed_lengths=expected_observed_lengths,
        observation_index_k=observation_index_k,
    )
    b = canonical["B"]
    phi_b_5b = build_mp_basis(b.input, MP_CONFIG["orders"], MP_CONFIG["memory_depth"])
    y_b_5b = np.asarray(b.output[MAX_DELAY:], dtype=np.complex128)
    phi_b, y_b = operator.apply_pair(phi_b_5b, y_b_5b)
    _require(
        phi_b.shape[0] == expected_observed_lengths["B"]
        if operator.spec.sample_rate_index_k == observation_index_k
        else phi_b.shape[0] == phi_b_5b.shape[0],
        "B泛化观测长度错误",
    )
    if operator.spec.sample_rate_index_k == observation_index_k:
        _require(
            phi_b.shape[0] == expected_observed_lengths["B"],
            "B泛化观测长度与spec不一致",
        )
    _require(y_b.shape[0] == phi_b.shape[0], "B泛化E/y长度不一致")
    theta = np.asarray(train[2]["theta"], dtype=np.complex128)
    prediction_b = phi_b @ theta
    diagnostics = train[2]["diagnostics"]
    row = {
        "train_NMSE_dB": float(train[2]["train_nmse_db"]),
        "B_generalization_NMSE_dB": float(calculate_nmse(y_b, prediction_b)),
        "theta_norm": float(diagnostics.theta_l2_norm),
        "rank": int(diagnostics.rank_phi),
        "rank_augmented": int(diagnostics.rank_augmented),
        "condition_number_phi": float(diagnostics.condition_number_phi),
        "condition_number_augmented": float(diagnostics.condition_number_augmented),
        "n_train": int(train[0].shape[0]),
        "coefficient_count": int(train[0].shape[1]),
        "operator_output_length": int(train[0].shape[0]),
        "B_output_length": int(phi_b.shape[0]),
        "model_role": role,
    }
    return theta, row


def _load_and_validate_raw_state(state_id: int) -> tuple[dict[str, Any], Any, int]:
    data = load_by_id(state_id)
    xin = np.asarray(data["xin"])
    _require(xin.shape == (24576, 1), f"state_id={state_id} xin shape错误：{xin.shape}")
    partition = build_partition_from_xin(xin)
    _validate_partition_lengths(partition)
    input_history = np.asarray(data["xin_pd_ori_ilc"])
    output_history = np.asarray(data["yout_withdpd_ori_ilc"])
    _require(
        input_history.ndim == 2 and output_history.ndim == 2, f"state_id={state_id} ILC必须为二维"
    )
    _require(
        input_history.shape == output_history.shape, f"state_id={state_id} ILC输入输出shape不一致"
    )
    _require(
        input_history.shape[0] == 24576 and 2 <= input_history.shape[1] <= 5,
        f"state_id={state_id} ILC列数不在2...5",
    )
    _require(
        np.iscomplexobj(input_history) and np.iscomplexobj(output_history),
        f"state_id={state_id} ILC必须为复数",
    )
    _require(
        np.all(np.isfinite(input_history)) and np.all(np.isfinite(output_history)),
        f"state_id={state_id} ILC包含非finite",
    )
    return data, partition, int(input_history.shape[1])


def run_model_phase(
    operator: LowBandwidthObservationOperator,
    *,
    label: str,
    common_b_input: np.ndarray,
    expected_observed_lengths: dict[str, int],
    observation_index_k: int,
    progress_interval: int = 25,
) -> ModelPhase:
    """Fit all 425 Aend/C2 models through one fixed operator."""

    aend_coefficients = np.empty((STATE_COUNT, NUM_COEFFICIENTS), dtype=np.complex128)
    c2_coefficients = np.empty_like(aend_coefficients)
    rows: list[dict[str, Any]] = []
    stage_values = np.empty(STATE_COUNT, dtype=np.int64)
    length_seen: dict[str, set[int]] = {name: set() for name in ("A", "B", "C")}

    for position, state_id in enumerate(range(STATE_COUNT), start=1):
        data, partition, count = _load_and_validate_raw_state(state_id)
        stage_values[state_id] = count

        aend_pair = get_ilc_pair(data, count - 1)
        c2_pair = get_ilc_pair(data, C2_STAGE - 1)
        canonical_a = preprocess_full_pair(
            aend_pair.input_full,
            aend_pair.output_raw_full,
            partition,
            pair_type="ILC",
            iteration_index=count - 1,
            input_peak_normalization_factor=aend_pair.input_peak_normalization_factor,
        )
        canonical_c = preprocess_full_pair(
            c2_pair.input_full,
            c2_pair.output_raw_full,
            partition,
            pair_type="ILC",
            iteration_index=C2_STAGE - 1,
            input_peak_normalization_factor=c2_pair.input_peak_normalization_factor,
        )
        for canonical in (canonical_a, canonical_c):
            for segment_name in ("A", "B", "C"):
                segment = canonical[segment_name]
                basis_length = build_mp_basis(
                    segment.input, MP_CONFIG["orders"], MP_CONFIG["memory_depth"]
                ).shape[0]
                length_seen[segment_name].add(int(operator.expected_output_length(basis_length)))

        theta_a, metric_a = _fit_role(
            role="Y-Aend",
            train_segment_name="A",
            canonical=canonical_a,
            operator=operator,
            expected_observed_lengths=expected_observed_lengths,
            observation_index_k=observation_index_k,
        )
        theta_c, metric_c = _fit_role(
            role="Y-C2",
            train_segment_name="C",
            canonical=canonical_c,
            operator=operator,
            expected_observed_lengths=expected_observed_lengths,
            observation_index_k=observation_index_k,
        )
        aend_coefficients[state_id] = theta_a
        c2_coefficients[state_id] = theta_c
        info = get_state_info(state_id)
        rows.append(
            {
                "state_id": state_id,
                **info,
                "ilc_A_end": count,
                "ilc_C_stage": C2_STAGE,
                "Y_Aend_train_NMSE_dB": metric_a["train_NMSE_dB"],
                "Y_Aend_B_NMSE_dB": metric_a["B_generalization_NMSE_dB"],
                "Y_C2_train_NMSE_dB": metric_c["train_NMSE_dB"],
                "Y_C2_B_NMSE_dB": metric_c["B_generalization_NMSE_dB"],
                "Y_Aend_theta_norm": metric_a["theta_norm"],
                "Y_C2_theta_norm": metric_c["theta_norm"],
                "Y_Aend_rank": metric_a["rank"],
                "Y_C2_rank": metric_c["rank"],
                "Y_Aend_n_train": metric_a["n_train"],
                "Y_C2_n_train": metric_c["n_train"],
                "operator_label": label,
            }
        )
        if progress_interval > 0 and (position % progress_interval == 0 or position == STATE_COUNT):
            print(f"{label} model phase: processed {position} / {STATE_COUNT} states", flush=True)

    phi_probe_5b = build_mp_basis(common_b_input, MP_CONFIG["orders"], MP_CONFIG["memory_depth"])
    _require(
        phi_probe_5b.shape == (FINGERPRINT_LENGTH_5B, NUM_COEFFICIENTS),
        "公共B probe 5B basis shape错误",
    )
    phi_probe_obs = operator.apply_matrix(phi_probe_5b)
    expected_probe = operator.expected_output_length(phi_probe_5b.shape[0])
    _require(phi_probe_obs.shape[0] == expected_probe, "公共B probe观测长度错误")
    aend_fingerprints = (phi_probe_obs @ aend_coefficients.T).T.astype(np.complex128, copy=False)
    c2_fingerprints = (phi_probe_obs @ c2_coefficients.T).T.astype(np.complex128, copy=False)
    _validate_complex_matrix(
        aend_fingerprints,
        (STATE_COUNT, expected_probe),
        f"{label} Aend fingerprints",
    )
    _validate_complex_matrix(
        c2_fingerprints, (STATE_COUNT, expected_probe), f"{label} C2 fingerprints"
    )
    model_metrics = pd.DataFrame(rows).sort_values("state_id").reset_index(drop=True)
    _require(
        np.array_equal(model_metrics["state_id"].to_numpy(), np.arange(STATE_COUNT)),
        f"{label} model state顺序错误",
    )
    length_checks = {
        name: {
            "observed": sorted(length_seen[name]),
            "expected_5B": EXPECTED_SEGMENT_LENGTHS_5B[name],
            "expected_observed": expected_observed_lengths[name],
        }
        for name in ("A", "B", "C")
    }
    length_checks["probe_length"] = int(expected_probe)
    return ModelPhase(
        operator_label=label,
        operator=operator,
        aend_coefficients=aend_coefficients,
        c2_coefficients=c2_coefficients,
        model_metrics=model_metrics,
        aend_fingerprints=aend_fingerprints,
        c2_fingerprints=c2_fingerprints,
        length_checks=length_checks,
    )


def regression_guard_5b(phase: ModelPhase, baseline: BaselineArtifacts) -> dict[str, Any]:
    """Compare the new integration path against the frozen 5B result."""

    counts = baseline.ilc_column_counts
    old_a = baseline.theta_y_a_actual[np.arange(STATE_COUNT), counts - 1, :]
    old_c = baseline.theta_y_c_actual[:, C2_STAGE - 1, :]
    theta_a_error = float(np.max(np.abs(phase.aend_coefficients - old_a)))
    theta_c_error = float(np.max(np.abs(phase.c2_coefficients - old_c)))

    metrics = phase.model_metrics
    baseline_metrics = baseline.model_metrics
    key = ["state_id", "ilc_A_end"]
    expected_a = baseline_metrics.loc[baseline_metrics["model_role"].eq("Y-A")].rename(
        columns={"actual_ilc_n": "ilc_A_end"}
    )
    expected_a = expected_a[key + ["train_NMSE_dB", "B_generalization_NMSE_dB"]]
    expected_a = expected_a.rename(
        columns={
            "train_NMSE_dB": "Y_Aend_train_NMSE_dB",
            "B_generalization_NMSE_dB": "Y_Aend_B_NMSE_dB",
        }
    )
    expected_c = baseline_metrics.loc[
        baseline_metrics["model_role"].eq("Y-C") & baseline_metrics["actual_ilc_n"].eq(C2_STAGE)
    ][["state_id", "train_NMSE_dB", "B_generalization_NMSE_dB"]].rename(
        columns={
            "train_NMSE_dB": "Y_C2_train_NMSE_dB",
            "B_generalization_NMSE_dB": "Y_C2_B_NMSE_dB",
        }
    )
    merged = metrics.merge(
        expected_a, on=key, how="inner", validate="one_to_one", suffixes=("", "_expected")
    )
    merged = merged.merge(
        expected_c, on="state_id", how="inner", validate="one_to_one", suffixes=("", "_expected")
    )
    metric_errors = {
        name: float(
            np.max(
                np.abs(
                    merged[name].to_numpy(dtype=float)
                    - merged[f"{name}_expected"].to_numpy(dtype=float)
                )
            )
        )
        for name in (
            "Y_Aend_train_NMSE_dB",
            "Y_Aend_B_NMSE_dB",
            "Y_C2_train_NMSE_dB",
            "Y_C2_B_NMSE_dB",
        )
    }

    lut_error = float(np.max(np.abs(phase.aend_fingerprints - baseline.aend_fingerprints)))
    query_error = float(np.max(np.abs(phase.c2_fingerprints - baseline.c2_fingerprints)))
    d_new = _compute_cnmse_distance_matrix(phase.c2_fingerprints, phase.aend_fingerprints)
    r_new = distance_to_ranks(d_new)
    d_error, d_mismatch = _max_difference_allowing_neginf(d_new, baseline.retrieval_distance)
    r_error, r_mismatch = _max_difference_allowing_neginf(r_new, baseline.retrieval_ranking)
    q_new = np.argmin(d_new, axis=1).astype(np.int64)
    q_old = baseline.retrieval_results["State_n_Q"].to_numpy(dtype=np.int64)
    q_equal = bool(np.array_equal(q_new, q_old))
    real_new = baseline.real_b_distance[np.arange(STATE_COUNT), q_new]
    real_old = baseline.real_b_distance[np.arange(STATE_COUNT), q_old]
    real_error, real_mismatch = _max_difference_allowing_neginf(real_new, real_old)
    exact_new = q_new == np.arange(STATE_COUNT)
    exact_old = q_old == np.arange(STATE_COUNT)
    share_new = real_new < SHAREABLE_THRESHOLD_DB
    share_old = real_old < SHAREABLE_THRESHOLD_DB
    result = {
        "operator": phase.operator.describe(),
        "theta_max_abs_error": {"Aend": theta_a_error, "C2": theta_c_error},
        "metric_max_abs_error_dB": metric_errors,
        "fingerprint_max_abs_error": {"Aend": lut_error, "C2": query_error},
        "retrieval_distance_max_abs_error": d_error,
        "retrieval_distance_nonfinite_pattern_mismatch": d_mismatch,
        "retrieval_ranking_max_abs_error": r_error,
        "retrieval_ranking_nonfinite_pattern_mismatch": r_mismatch,
        "state_Q_identical": q_equal,
        "real_B_lookup_max_abs_error": real_error,
        "real_B_lookup_nonfinite_pattern_mismatch": real_mismatch,
        "exact_count_new": int(exact_new.sum()),
        "exact_count_baseline": int(exact_old.sum()),
        "shareable_count_new": int(share_new.sum()),
        "shareable_count_baseline": int(share_old.sum()),
    }
    result["pass"] = bool(
        theta_a_error < 1e-12
        and theta_c_error < 1e-12
        and max(metric_errors.values()) < 1e-10
        and lut_error < 1e-12
        and query_error < 1e-12
        and d_error < 1e-12
        and d_mismatch == 0
        and r_error < 1e-12
        and r_mismatch == 0
        and q_equal
        and real_error < 1e-12
        and real_mismatch == 0
        and np.array_equal(exact_new, exact_old)
        and np.array_equal(share_new, share_old)
    )
    if not result["pass"]:
        raise RuntimeError(f"5B regression guard failed: {result}")
    return result


def _compute_cnmse_distance_matrix(query: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    """Batch form of the frozen core.metrics.cnmse formula for any length."""

    query = np.asarray(query)
    candidate = np.asarray(candidate)
    _require(query.ndim == 2 and candidate.ndim == 2, "fingerprint必须为二维数组")
    _require(
        query.shape[0] == STATE_COUNT and candidate.shape[0] == STATE_COUNT,
        "fingerprint状态数必须为425",
    )
    _require(
        query.dtype == np.complex128 and candidate.dtype == np.complex128,
        "fingerprint必须是complex128",
    )
    _require(
        np.all(np.isfinite(query)) and np.all(np.isfinite(candidate)), "fingerprint包含NaN或Inf"
    )
    q_energy = np.sum(np.abs(query) ** 2, axis=1, dtype=np.float64)
    c_energy = np.sum(np.abs(candidate) ** 2, axis=1, dtype=np.float64)
    _require(np.all(q_energy > 0) and np.all(c_energy > 0), "fingerprint能量必须为正")
    with np.errstate(over="ignore", invalid="ignore"):
        squared_error = (
            q_energy[:, None] + c_energy[None, :] - 2.0 * np.real(query @ candidate.conj().T)
        )
    scale = q_energy[:, None] + c_energy[None, :] + 1.0
    tolerance = 32.0 * np.finfo(np.float64).eps * scale
    squared_error[np.abs(squared_error) <= tolerance] = 0.0
    squared_error = np.maximum(squared_error, 0.0)
    if query.shape == candidate.shape and np.array_equal(query, candidate):
        np.fill_diagonal(squared_error, 0.0)
    ratio = np.divide(
        squared_error,
        q_energy[:, None],
        out=np.zeros_like(squared_error),
        where=q_energy[:, None] > 0,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = 10.0 * np.log10(ratio)
    distance[squared_error == 0.0] = -np.inf
    _require(
        not np.isnan(distance).any() and not np.isposinf(distance).any(),
        "CNMSE距离矩阵出现NaN或+Inf",
    )
    return distance.astype(np.float64, copy=False)


def _physical_mismatch(code: Any) -> str:
    value = int(round(float(code)))
    if value == 0:
        return "0"
    return f"{value / 100:.2f}".rstrip("0").rstrip(".")


def _load_config(info: dict[str, Any]) -> str:
    return (
        f"funMng={_physical_mismatch(info['funMng'])}, funAng={int(info['funAng'])}deg, "
        f"secMng={_physical_mismatch(info['secMng'])}, secAng={int(info['secAng'])}deg"
    )


def _random_consistency_checks(
    distance: np.ndarray,
    state_q: np.ndarray,
    real_b: np.ndarray,
    baseline: BaselineArtifacts,
) -> dict[str, Any]:
    rng = np.random.default_rng(RANDOM_SEED)
    ids = np.sort(rng.choice(STATE_COUNT, size=20, replace=False)).astype(np.int64)
    top1_ok = [int(state_q[i]) == int(np.argmin(distance[i])) for i in ids]
    real_ok = [
        (
            (np.isneginf(real_b[i]) and np.isneginf(baseline.real_b_distance[i, state_q[i]]))
            or np.isclose(real_b[i], baseline.real_b_distance[i, state_q[i]], rtol=0, atol=0)
        )
        for i in ids
    ]
    return {
        "seed": RANDOM_SEED,
        "state_ids": ids.tolist(),
        "top1_argmin_all_pass": bool(all(top1_ok)),
        "real_B_lookup_all_pass": bool(all(real_ok)),
        "top1_argmin_pass_count": int(sum(top1_ok)),
        "real_B_lookup_pass_count": int(sum(real_ok)),
    }


def build_final_tables(
    *,
    phase_observation: ModelPhase,
    spec: ObservationExperimentSpec,
    baseline: BaselineArtifacts,
    raw_scalar_rows: list[dict[str, Any]],
    aend_stage_map: pd.DataFrame,
) -> dict[str, Any]:
    distance = _compute_cnmse_distance_matrix(
        phase_observation.c2_fingerprints,
        phase_observation.aend_fingerprints,
    )
    ranking = distance_to_ranks(distance)
    state_q = np.argmin(distance, axis=1).astype(np.int64)
    top1_fingerprint = distance[np.arange(STATE_COUNT), state_q]
    retrieved_real_b = baseline.real_b_distance[np.arange(STATE_COUNT), state_q]
    exact = state_q == np.arange(STATE_COUNT)
    shareable = retrieved_real_b < SHAREABLE_THRESHOLD_DB
    failure = ~shareable
    _require(np.all(exact[retrieved_real_b == -np.inf]), "Exact self-hit必须对应Real-B=-Inf")
    _require(
        np.array_equal(np.flatnonzero(exact), np.flatnonzero(np.isneginf(retrieved_real_b))),
        "exact_hit与Real-B -Inf pattern不一致",
    )

    raw = pd.DataFrame(raw_scalar_rows).sort_values("state_id").reset_index(drop=True)
    metrics = phase_observation.model_metrics
    merged = aend_stage_map.merge(raw, on="state_id", validate="one_to_one")
    merged = merged.merge(metrics, on="state_id", validate="one_to_one")
    _require(
        np.array_equal(merged["state_id"].to_numpy(), np.arange(STATE_COUNT)), "最终表state顺序错误"
    )
    load_config = [_load_config(get_state_info(i)) for i in range(STATE_COUNT)]
    main = pd.DataFrame(
        {
            "state_id_R": np.arange(STATE_COUNT, dtype=np.int64),
            "load_config": load_config,
            "nmse_withoutdpd_dB": merged["nmse_withoutdpd_dB"].to_numpy(dtype=float),
            "acpr_withoutdpd_avg_dBc": merged["acpr_withoutdpd_avg_dBc"].to_numpy(dtype=float),
            "Y_Aend_train_NMSE_dB": merged["Y_Aend_train_NMSE_dB"].to_numpy(dtype=float),
            "Y_Aend_B_NMSE_dB": merged["Y_Aend_B_NMSE_dB"].to_numpy(dtype=float),
            "Y_C2_train_NMSE_dB": merged["Y_C2_train_NMSE_dB"].to_numpy(dtype=float),
            "Y_C2_B_NMSE_dB": merged["Y_C2_B_NMSE_dB"].to_numpy(dtype=float),
            "state_id_Q": state_q,
            "retrieved_real_B_CNMSE_dB": retrieved_real_b,
        }
    )
    _require(
        main.columns.tolist() == EXPECTED_MAIN_COLUMNS and main.shape == (STATE_COUNT, 10),
        "主表shape/列顺序错误",
    )

    diagnostics = pd.DataFrame(
        {
            "state_id_R": np.arange(STATE_COUNT, dtype=np.int64),
            "state_id_Q": state_q,
            "state_id_delta_signed": state_q - np.arange(STATE_COUNT, dtype=np.int64),
            "state_id_delta_abs": np.abs(state_q - np.arange(STATE_COUNT, dtype=np.int64)),
            "query_top1_fingerprint_CNMSE_dB": top1_fingerprint,
            "retrieved_real_B_CNMSE_dB": retrieved_real_b,
            "exact_hit": exact,
            "dpd_shareable": shareable,
            "ilc_A_end": aend_stage_map["ilc_A_end"].to_numpy(dtype=np.int64),
            "ilc_C_stage": np.full(STATE_COUNT, C2_STAGE, dtype=np.int64),
        }
    )
    top1 = diagnostics.loc[
        :,
        [
            "state_id_R",
            "state_id_Q",
            "query_top1_fingerprint_CNMSE_dB",
            "retrieved_real_B_CNMSE_dB",
            "exact_hit",
            "dpd_shareable",
        ],
    ].rename(columns={"query_top1_fingerprint_CNMSE_dB": "top1_fingerprint_CNMSE_dB"})

    q_load = [_load_config(get_state_info(int(q))) for q in state_q]
    failures = diagnostics.loc[failure].copy()
    failures.insert(6, "load_config_R", [load_config[i] for i in failures["state_id_R"]])
    failures.insert(7, "load_config_Q", [q_load[i] for i in failures["state_id_R"]])
    finite_nonexact = retrieved_real_b[(~exact) & np.isfinite(retrieved_real_b)]
    finite_top1 = top1_fingerprint[np.isfinite(top1_fingerprint)]
    summary_payload: dict[str, Any] = {
        "observation_n": spec.normalized_n,
        "Fs_obs_MHz": spec.fs_obs_hz / 1e6,
        "retrieval_bandwidth": spec.bandwidth_label,
        "validation_bandwidth": "5B",
        "state_count": STATE_COUNT,
        "candidate_count": STATE_COUNT,
        "self_candidate_allowed": True,
        "exact_hit_count": int(exact.sum()),
        "exact_hit_rate": float(np.mean(exact)),
        "shareable_count": int(shareable.sum()),
        "shareable_rate": float(np.mean(shareable)),
        "nonexact_count": int((~exact).sum()),
        "nonexact_shareable_count": int((~exact & shareable).sum()),
        "nonexact_shareable_rate": float(np.mean(shareable[~exact])),
        "failure_count": int(failure.sum()),
        "failure_state_ids": np.flatnonzero(failure).tolist(),
        "shareable_threshold_dB": SHAREABLE_THRESHOLD_DB,
        "shareable_rule": "retrieved 5B Real-B CNMSE < -40 dB",
        "top1_fingerprint_CNMSE_finite_count": int(finite_top1.size),
        "top1_fingerprint_CNMSE_min_dB": float(np.min(finite_top1)),
        "top1_fingerprint_CNMSE_max_dB": float(np.max(finite_top1)),
        "retrieved_real_B_finite_nonexact_count": int(finite_nonexact.size),
        "retrieved_real_B_best_nonexact_dB": float(np.min(finite_nonexact)),
        "retrieved_real_B_worst_nonexact_dB": float(np.max(finite_nonexact)),
        "retrieved_real_B_nonexact_median_dB": float(np.median(finite_nonexact)),
        "retrieved_real_B_nonexact_q95_dB": float(np.quantile(finite_nonexact, 0.95)),
        "nmse_withoutdpd_min_dB": float(main["nmse_withoutdpd_dB"].min()),
        "nmse_withoutdpd_max_dB": float(main["nmse_withoutdpd_dB"].max()),
        "acpr_withoutdpd_avg_min_dBc": float(main["acpr_withoutdpd_avg_dBc"].min()),
        "acpr_withoutdpd_avg_max_dBc": float(main["acpr_withoutdpd_avg_dBc"].max()),
    }
    summary = pd.DataFrame(
        {"metric": list(summary_payload), "value": list(summary_payload.values())}
    )
    return {
        "distance": distance,
        "ranking": ranking,
        "state_q": state_q,
        "top1_fingerprint": top1_fingerprint,
        "retrieved_real_b": retrieved_real_b,
        "exact": exact,
        "shareable": shareable,
        "failure": failure,
        "main": main,
        "diagnostics": diagnostics,
        "top1": top1,
        "failures": failures,
        "summary_payload": summary_payload,
        "summary": summary,
    }


def _figure_contract(*, bandwidth_label: str, source: str) -> dict[str, Any]:
    return {
        "core_conclusion": (
            f"A fixed {bandwidth_label} behavior fingerprint selects LUT candidates "
            "whose full-rate 5B Real-B behavior can be evaluated for DPD sharing."
        ),
        "archetype": "quantitative grid/statewise trend",
        "backend": "Python/matplotlib",
        "final_size_inches": [24, 10],
        "panel_map": {"single_axes": "six statewise dB curves and the -40 dB Real-B threshold"},
        "evidence_hierarchy": {
            "hero_evidence": f"retrieved 5B Real-B CNMSE after {bandwidth_label} Top1 selection",
            "model_evidence": f"four {bandwidth_label} train/generalization NMSE curves",
            "control": "per-state NMSE without DPD and strict -40 dB threshold",
        },
        "statistics": {
            "n": "425 states",
            "metric": "NMSE/CNMSE in dB",
            "threshold": "retrieved 5B Real-B CNMSE < -40 dB",
            "variability": "statewise values; no error bars",
        },
        "source_data": source,
        "image_integrity": (
            "No image processing; figure is generated from the final CSV fact source."
        ),
        "reviewer_risk": (
            "Exact self-hit Real-B values are -Inf in data and shown only at a labeled "
            "plotting strip."
        ),
    }


def _plot_main_figure(
    main: pd.DataFrame,
    output_base: Path,
    *,
    bandwidth_label: str,
    source: str,
) -> dict[str, Any]:
    x = main["state_id_R"].to_numpy(dtype=int)
    y_names = [
        ("nmse_withoutdpd_dB", "NMSE without DPD", "#4D4D4D", "o"),
        (
            "Y_Aend_train_NMSE_dB",
            f"Y-Aend Train NMSE ({bandwidth_label})",
            "#0F4D92",
            "s",
        ),
        (
            "Y_Aend_B_NMSE_dB",
            f"Y-Aend -> B NMSE ({bandwidth_label})",
            "#3775BA",
            "^",
        ),
        (
            "Y_C2_train_NMSE_dB",
            f"Y-C2 Train NMSE ({bandwidth_label})",
            "#9A4D8E",
            "D",
        ),
        (
            "Y_C2_B_NMSE_dB",
            f"Y-C2 -> B NMSE ({bandwidth_label})",
            "#42949E",
            "v",
        ),
    ]
    finite_values = [main[name].to_numpy(dtype=float) for name, *_ in y_names]
    retrieved = main["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float)
    finite_retrieved = retrieved[np.isfinite(retrieved)]
    all_finite = np.concatenate([*finite_values, finite_retrieved])
    plot_min = float(np.min(all_finite) - 4.0)
    plot_max = float(np.max(all_finite) + 2.0)
    retrieved_plot = np.where(np.isneginf(retrieved), plot_min + 0.5, retrieved)

    fig, ax = plt.subplots(figsize=(24, 10), constrained_layout=False)
    for index, (name, label, color, marker) in enumerate(y_names):
        ax.plot(
            x,
            main[name].to_numpy(dtype=float),
            color=color,
            linewidth=1.2,
            marker=marker,
            markersize=3.0,
            markevery=10,
            alpha=0.9,
            label=label,
        )
    ax.plot(
        x,
        retrieved_plot,
        color="#B64342",
        linewidth=1.4,
        marker="o",
        markersize=3.2,
        markerfacecolor="white",
        markeredgewidth=0.8,
        label="Retrieved Real-B CNMSE (5B)",
        zorder=5,
    )
    ax.axhline(
        SHAREABLE_THRESHOLD_DB,
        color="#B64342",
        linestyle="--",
        linewidth=1.0,
        alpha=0.8,
        label="DPD-shareable threshold (-40 dB)",
    )
    ax.axhline(plot_min + 0.5, color="#767676", linestyle=":", linewidth=0.8, alpha=0.8)
    q_values = main["state_id_Q"].to_numpy(dtype=int)
    exact = np.isneginf(retrieved)
    for i, (xi, yi, q, is_exact) in enumerate(zip(x, retrieved_plot, q_values, exact)):
        ax.annotate(
            f"Q={q}",
            xy=(xi, yi),
            xytext=(0, 4 if i % 2 == 0 else -8),
            textcoords="offset points",
            rotation=90,
            ha="center",
            va="bottom" if i % 2 == 0 else "top",
            fontsize=4.2,
            color="#262626",
            clip_on=True,
        )
        if is_exact:
            ax.plot(
                xi,
                yi,
                marker="o",
                markersize=3.3,
                color="#1F4E78",
                markerfacecolor="#D9EAF7",
                markeredgewidth=0.7,
                zorder=6,
            )
    ax.set_xlim(-2, STATE_COUNT + 1)
    ax.set_ylim(plot_min - 1.0, plot_max)
    ax.set_xlabel("State_R")
    ax.set_ylabel("Metric (dB)")
    ax.set_title(
        f"{bandwidth_label} Behavior-Fingerprint LUT Retrieval: "
        "Statewise Model Accuracy and 5B Real-B Validation",
        pad=12,
    )
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.65)
    ax.legend(loc="upper left", ncol=3, fontsize=8, frameon=False)
    ax.text(
        0.995,
        0.985,
        "Exact self-hit Real-B = -Inf; displayed at the lower strip only for plotting.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="#4D4D4D",
    )
    fig.tight_layout(pad=1.3)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return {
        "path_png": str(output_base.with_suffix(".png")),
        "path_svg": str(output_base.with_suffix(".svg")),
        "path_pdf": str(output_base.with_suffix(".pdf")),
        "dpi": 300,
        "state_count": STATE_COUNT,
        "curve_count": 6,
        "q_annotation_count": STATE_COUNT,
        "finite_retrieved_count": int(np.isfinite(retrieved).sum()),
        "exact_display_count": int(exact.sum()),
        "plot_y_min": plot_min,
        "plot_y_max": plot_max,
        "threshold_dB": SHAREABLE_THRESHOLD_DB,
        "source": source,
    }


def _plot_real_b_figure(
    main: pd.DataFrame,
    output_base: Path,
    *,
    bandwidth_label: str,
    source: str,
) -> dict[str, Any]:
    x = main["state_id_R"].to_numpy(dtype=int)
    values = main["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float)
    finite = values[np.isfinite(values)]
    y_min = float(np.min(finite) - 4.0)
    display = np.where(np.isneginf(values), y_min + 0.5, values)
    fig, ax = plt.subplots(figsize=(20, 8))
    ax.plot(
        x,
        display,
        color="#B64342",
        marker="o",
        markersize=3.2,
        markerfacecolor="white",
        linewidth=1.1,
        label="Retrieved Real-B CNMSE (5B)",
    )
    ax.axhline(
        SHAREABLE_THRESHOLD_DB,
        color="#B64342",
        linestyle="--",
        linewidth=1.0,
        label="DPD-shareable threshold (-40 dB)",
    )
    for i, (xi, yi, q) in enumerate(zip(x, display, main["state_id_Q"].to_numpy(dtype=int))):
        ax.annotate(
            f"Q={q}",
            (xi, yi),
            xytext=(0, 4 if i % 2 == 0 else -8),
            textcoords="offset points",
            rotation=90,
            ha="center",
            fontsize=4.2,
            clip_on=True,
        )
    ax.set_xlim(-2, STATE_COUNT + 1)
    ax.set_ylim(y_min - 1, float(np.max(finite) + 2))
    ax.set_xlabel("State_R")
    ax.set_ylabel("Retrieved Real-B CNMSE (dB)")
    ax.set_title(f"{bandwidth_label} Retrieval with Full-rate 5B Real-B Validation")
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.45, alpha=0.65)
    ax.legend(loc="upper left", frameon=False)
    ax.text(
        0.995,
        0.02,
        "-Inf self-hits are displayed at a plotting-only lower strip.",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
    )
    fig.tight_layout(pad=1.2)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    return {
        "path_png": str(output_base.with_suffix(".png")),
        "path_svg": str(output_base.with_suffix(".svg")),
        "path_pdf": str(output_base.with_suffix(".pdf")),
        "q_annotation_count": STATE_COUNT,
        "dpi": 300,
        "source": source,
    }


def run_experiment(
    *,
    spec: ObservationExperimentSpec = SPEC_1B,
    result_root: Path | None = None,
    progress_interval: int = 25,
    append_logs: bool = False,
    run_5b_guard: bool = True,
) -> dict[str, Any]:
    """Run the 5B guard, then one fixed observation retrieval and persist artifacts."""

    if not isinstance(spec, ObservationExperimentSpec):
        raise TypeError("spec必须是ObservationExperimentSpec")
    root = Path(result_root) if result_root is not None else spec.result_root
    protection_before = protection_snapshot()
    baseline = load_baseline_artifacts()
    bank = LowBandwidthObservationBank()
    identity = bank.get_by_index(IDENTITY_INDEX_K)
    observation = bank.get_by_index(spec.observation_index_k)
    _require(
        observation.spec.normalized_bandwidth_n == spec.normalized_n,
        f"{spec.bandwidth_label} observation n不一致",
    )
    _require(
        observation.spec.fs_out_hz == spec.fs_obs_hz,
        f"{spec.bandwidth_label} observation Fs不一致",
    )
    _require(identity.spec.is_identity, "k=50必须是5B identity operator")

    phase_5b: ModelPhase | None = None
    if run_5b_guard:
        print("Step 1/4: run 5B identity integration regression guard", flush=True)
        phase_5b = run_model_phase(
            identity,
            label="5B identity",
            common_b_input=baseline.common_b_input,
            expected_observed_lengths=EXPECTED_SEGMENT_LENGTHS_5B,
            observation_index_k=IDENTITY_INDEX_K,
            progress_interval=progress_interval,
        )
        guard = regression_guard_5b(phase_5b, baseline)
        print("5B regression guard: PASS", flush=True)
    else:
        guard = {"pass": None, "skipped": True, "reason": "1B regression guard run before 5B guard"}
        print("Step 1/4: skip 5B guard for the requested preliminary 1B regression", flush=True)

    print(f"Step 2/4: run fixed {spec.bandwidth_label} model and fingerprint phases", flush=True)
    phase_observation = run_model_phase(
        observation,
        label=spec.bandwidth_label,
        common_b_input=baseline.common_b_input,
        expected_observed_lengths=spec.expected_segment_lengths,
        observation_index_k=spec.observation_index_k,
        progress_interval=progress_interval,
    )
    root.mkdir(parents=True, exist_ok=True)

    state_frame = pd.DataFrame(build_state_table()).sort_values("state_id").reset_index(drop=True)
    stage_map = state_frame.copy()
    stage_map["N_ilc_available"] = baseline.ilc_column_counts
    stage_map["ilc_A_end"] = baseline.ilc_column_counts
    stage_map["C2_available"] = baseline.ilc_column_counts >= C2_STAGE
    _require(stage_map["C2_available"].all(), "存在C2不可用状态")
    _require(
        stage_map["ilc_A_end"].value_counts().sort_index().to_dict() == {2: 1, 3: 416, 4: 7, 5: 1},
        "Aend stage分布错误",
    )

    print("Step 3/4: load per-state raw scalar metrics and build retrieval tables", flush=True)
    scalar_rows: list[dict[str, Any]] = []
    for state_id in range(STATE_COUNT):
        data = load_by_id(state_id)
        nmse = float(np.asarray(data["nmse_withoutdpd"]).reshape(-1)[0])
        acpr_low = float(np.asarray(data["acpr_low_withoutdpd"]).reshape(-1)[0])
        acpr_upper = float(np.asarray(data["acpr_upper_withoutdpd"]).reshape(-1)[0])
        _require(
            np.isfinite(nmse) and np.isfinite(acpr_low) and np.isfinite(acpr_upper),
            f"state_id={state_id} raw scalar非finite",
        )
        scalar_rows.append(
            {
                "state_id": state_id,
                "nmse_withoutdpd_dB": nmse,
                "acpr_low_withoutdpd": acpr_low,
                "acpr_upper_withoutdpd": acpr_upper,
                "acpr_withoutdpd_avg_dBc": (acpr_low + acpr_upper) / 2.0,
            }
        )
        if progress_interval > 0 and (
            (state_id + 1) % (progress_interval * 4) == 0 or state_id == STATE_COUNT - 1
        ):
            print(
                f"raw scalar metrics: processed {state_id + 1} / {STATE_COUNT} states", flush=True
            )
    tables = build_final_tables(
        phase_observation=phase_observation,
        spec=spec,
        baseline=baseline,
        raw_scalar_rows=scalar_rows,
        aend_stage_map=stage_map,
    )
    random_checks = _random_consistency_checks(
        tables["distance"], tables["state_q"], tables["retrieved_real_b"], baseline
    )
    _require(
        random_checks["top1_argmin_all_pass"] and random_checks["real_B_lookup_all_pass"],
        "随机Top1/Real-B自洽检查失败",
    )

    _require(
        tables["distance"].shape == (STATE_COUNT, STATE_COUNT),
        f"{spec.bandwidth_label}距离矩阵shape错误",
    )
    expected_fingerprint_shape = (
        STATE_COUNT,
        spec.expected_segment_lengths["B"],
    )
    _require(
        phase_observation.aend_fingerprints.shape == expected_fingerprint_shape,
        f"{spec.bandwidth_label} LUT fingerprint shape错误",
    )
    _require(
        phase_observation.c2_fingerprints.shape == expected_fingerprint_shape,
        f"{spec.bandwidth_label} Query fingerprint shape错误",
    )
    _require(
        np.all(np.isfinite(phase_observation.aend_fingerprints))
        and np.all(np.isfinite(phase_observation.c2_fingerprints)),
        f"{spec.bandwidth_label} fingerprint非finite",
    )
    _require(
        np.all(np.isfinite(phase_observation.aend_coefficients))
        and np.all(np.isfinite(phase_observation.c2_coefficients)),
        f"{spec.bandwidth_label}系数非finite",
    )

    print("Step 4/4: persist machine-readable outputs and Python figures", flush=True)
    _write_frame(stage_map, root / "a_end_stage_map.csv")
    _write_frame(phase_observation.model_metrics, root / f"model_metrics_{spec.result_tag}.csv")
    _write_frame(tables["main"], root / f"retrieval_results_{spec.result_tag}.csv")
    _write_frame(tables["top1"], root / f"top1_retrieval_{spec.result_tag}.csv")
    _write_frame(tables["diagnostics"], root / f"retrieval_diagnostics_{spec.result_tag}.csv")
    _write_frame(tables["failures"], root / f"retrieval_failures_{spec.result_tag}.csv")
    _write_frame(tables["summary"], root / f"retrieval_summary_{spec.result_tag}.csv")
    matrix_frame = pd.DataFrame(
        tables["distance"], columns=[f"candidate_state_{i}" for i in range(STATE_COUNT)]
    )
    matrix_frame.insert(0, "state_id_R", np.arange(STATE_COUNT, dtype=np.int64))
    _write_frame(matrix_frame, root / f"fingerprint_cnmse_matrix_{spec.result_tag}.csv")
    np.save(root / f"fingerprint_cnmse_matrix_{spec.result_tag}.npy", tables["distance"])
    np.save(root / f"query_to_lut_cnmse_{spec.result_tag}.npy", tables["distance"])
    np.savez_compressed(
        root / f"lut_fingerprints_{spec.result_tag}.npz",
        state_ids=np.arange(STATE_COUNT, dtype=np.int64),
        common_B_input=baseline.common_b_input,
        fingerprints=phase_observation.aend_fingerprints,
        Y_Aend_fingerprints=phase_observation.aend_fingerprints,
        observation_index_k=np.array(spec.observation_index_k, dtype=np.int64),
    )
    np.savez_compressed(
        root / f"query_fingerprints_{spec.result_tag}.npz",
        state_ids=np.arange(STATE_COUNT, dtype=np.int64),
        common_B_input=baseline.common_b_input,
        fingerprints=phase_observation.c2_fingerprints,
        Q_C2=phase_observation.c2_fingerprints,
        observation_index_k=np.array(spec.observation_index_k, dtype=np.int64),
    )
    np.save(
        root / f"Aend_model_coefficients_{spec.result_tag}.npy", phase_observation.aend_coefficients
    )
    np.save(
        root / f"C2_model_coefficients_{spec.result_tag}.npy", phase_observation.c2_coefficients
    )

    main_figure = _plot_main_figure(
        tables["main"],
        root / f"figure_{spec.result_tag}_statewise_model_and_retrieval_metrics",
        bandwidth_label=spec.bandwidth_label,
        source=f"retrieval_results_{spec.result_tag}.csv",
    )
    real_b_figure = _plot_real_b_figure(
        tables["main"],
        root / f"figure_{spec.result_tag}_retrieved_real_B_CNMSE",
        bandwidth_label=spec.bandwidth_label,
        source=f"retrieval_results_{spec.result_tag}.csv",
    )

    protection_after = protection_snapshot()
    protection = verify_protection(protection_before, protection_after)
    _require(protection["all_protected_unchanged"], "保护输入/旧结果SHA发生变化")

    tag = spec.result_tag
    bandwidth_label = spec.bandwidth_label
    validation = {
        "experiment": f"scenario_2_C2_to_Aend_{tag}",
        "state_count": STATE_COUNT,
        "retrieval_observation_n": spec.normalized_n,
        "retrieval_fs_obs_hz": spec.fs_obs_hz,
        "retrieval_observation_bandwidth": bandwidth_label,
        "final_validation_bandwidth": "5B",
        "model": {
            "orders": MP_CONFIG["orders"],
            "memory_depth": MP_CONFIG["memory_depth"],
            "max_delay": MAX_DELAY,
            "bias": False,
            "lambda": RIDGE_LAMBDA,
            "complex_coefficient_count": NUM_COEFFICIENTS,
            "structure_changed": False,
        },
        "ABC": {
            "definition": "original unequal split",
            "A": [0, 12288],
            "B": [12288, 17203],
            "C": [17203, 24576],
            "ownership_lengths": {"A": 12288, "B": 4915, "C": 7373},
            "model_valid_lengths_5B": EXPECTED_SEGMENT_LENGTHS_5B,
            "model_valid_lengths_observation": spec.expected_segment_lengths,
            "model_valid_lengths_1B": EXPECTED_SEGMENT_LENGTHS_1B,
        },
        "LUT_train_segment": "A",
        "LUT_ILC_stage": "per-state A_end",
        "Query_train_segment": "C",
        "Query_ILC_stage": C2_STAGE,
        "fingerprint_probe_segment": "B",
        "self_candidate_allowed": True,
        "shareable_threshold_dB": SHAREABLE_THRESHOLD_DB,
        "shareable_rule": "retrieved 5B Real-B CNMSE < -40 dB",
        "one_B_used_for_retrieval": tag == "1B",
        "one_B_used_for_final_DPD_shareability_validation": False,
        "half_B_used_for_retrieval": tag == "0p5B",
        "half_B_used_for_final_DPD_shareability_validation": False,
        "five_B_real_B_used_as_ground_truth": True,
        "retrieval_selection_uses_real_B": False,
        "real_B_is_evaluation_only": True,
        "real_B_observation_operator_applied": False,
        "basis_generation_order": (
            f"construct basis at 5B first, apply model-valid support, then apply {bandwidth_label} "
            "observation operator"
        ),
        "same_observation_operator_for_basis_and_output": True,
        "preprocessing_order": (
            "full-record alignment -> fixed ABC split -> segment-wise complex gain adjustment"
        ),
        "raw_data_modified": False,
        "old_5B_result_overwritten": False,
        "low_bandwidth_operator_modified": False,
        "low_bandwidth_operator": observation.describe(),
        "aend_stage_distribution": {
            str(k): int(v) for k, v in stage_map["ilc_A_end"].value_counts().sort_index().items()
        },
        "c2_available_count": int(stage_map["C2_available"].sum()),
        "Aend_model_count": int(phase_observation.aend_coefficients.shape[0]),
        "C2_model_count": int(phase_observation.c2_coefficients.shape[0]),
        "Aend_fingerprint_shape": list(phase_observation.aend_fingerprints.shape),
        "C2_query_fingerprint_shape": list(phase_observation.c2_fingerprints.shape),
        "fingerprint_cnmse_matrix_shape": list(tables["distance"].shape),
        "numerical_stability": {
            "all_fingerprint_finite": bool(
                np.all(np.isfinite(phase_observation.aend_fingerprints))
                and np.all(np.isfinite(phase_observation.c2_fingerprints))
            ),
            "all_fingerprint_norms_positive": bool(
                np.all(np.linalg.norm(phase_observation.aend_fingerprints, axis=1) > 0)
                and np.all(np.linalg.norm(phase_observation.c2_fingerprints, axis=1) > 0)
            ),
            "all_coefficients_finite": bool(
                np.all(np.isfinite(phase_observation.aend_coefficients))
                and np.all(np.isfinite(phase_observation.c2_coefficients))
            ),
            "distance_nan_count": int(np.isnan(tables["distance"]).sum()),
            "distance_posinf_count": int(np.isposinf(tables["distance"]).sum()),
            "distance_neginf_count": int(np.isneginf(tables["distance"]).sum()),
            "rank_collapse": bool(
                (phase_observation.model_metrics[["Y_Aend_rank", "Y_C2_rank"]] < NUM_COEFFICIENTS)
                .to_numpy()
                .any()
            ),
        },
        "top1_tie_rule": (
            "candidate state_id ascending because candidates are stored 0...424 and "
            "numpy argmin returns the first minimum"
        ),
        "five_B_regression_guard": guard,
        "observation_length_checks": phase_observation.length_checks,
        "one_B_length_checks": (phase_observation.length_checks if tag == "1B" else None),
        "half_B_length_checks": (phase_observation.length_checks if tag == "0p5B" else None),
        "random_consistency_checks": random_checks,
        "summary": tables["summary_payload"],
        "figure_contract": _figure_contract(
            bandwidth_label=bandwidth_label,
            source=f"retrieval_results_{tag}.csv",
        ),
        "figures": {"main": main_figure, "real_B": real_b_figure},
        "protection_before": protection_before,
        "protection_after": protection_after,
        "protection_verification": protection,
        "output_files": {
            "a_end_stage_map": str(root / "a_end_stage_map.csv"),
            f"model_metrics_{tag}": str(root / f"model_metrics_{tag}.csv"),
            f"retrieval_results_{tag}": str(root / f"retrieval_results_{tag}.csv"),
            f"top1_retrieval_{tag}": str(root / f"top1_retrieval_{tag}.csv"),
            f"retrieval_diagnostics_{tag}": str(root / f"retrieval_diagnostics_{tag}.csv"),
            f"retrieval_failures_{tag}": str(root / f"retrieval_failures_{tag}.csv"),
            f"retrieval_summary_{tag}": str(root / f"retrieval_summary_{tag}.csv"),
            f"fingerprint_cnmse_matrix_{tag}_npy": str(
                root / f"fingerprint_cnmse_matrix_{tag}.npy"
            ),
            f"fingerprint_cnmse_matrix_{tag}_csv": str(
                root / f"fingerprint_cnmse_matrix_{tag}.csv"
            ),
            f"query_to_lut_cnmse_{tag}": str(root / f"query_to_lut_cnmse_{tag}.npy"),
            f"lut_fingerprints_{tag}": str(root / f"lut_fingerprints_{tag}.npz"),
            f"query_fingerprints_{tag}": str(root / f"query_fingerprints_{tag}.npz"),
            f"Aend_model_coefficients_{tag}": str(root / f"Aend_model_coefficients_{tag}.npy"),
            f"C2_model_coefficients_{tag}": str(root / f"C2_model_coefficients_{tag}.npy"),
            "main_figure_png": main_figure["path_png"],
            "real_B_figure_png": real_b_figure["path_png"],
        },
        "lut_retrieval_executed": True,
        "full_bandwidth_scan_executed": False,
    }
    _write_json(root / f"retrieval_summary_{tag}.json", tables["summary_payload"])
    _write_json(root / "validation.json", validation)
    return {
        "baseline": baseline,
        "phase_5b": phase_5b,
        "phase_observation": phase_observation,
        "phase_1b": phase_observation,
        "stage_map": stage_map,
        "tables": tables,
        "validation": validation,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress-interval", type=int, default=25)
    args = parser.parse_args()
    result = run_experiment(progress_interval=args.progress_interval)
    summary = result["tables"]["summary_payload"]
    print(
        json.dumps(
            {
                "result_root": str(RESULT_ROOT),
                "state_count": STATE_COUNT,
                "five_B_regression_guard": result["validation"]["five_B_regression_guard"]["pass"],
                "exact_hit_count": summary["exact_hit_count"],
                "shareable_count": summary["shareable_count"],
                "failure_count": summary["failure_count"],
                "failure_state_ids": summary["failure_state_ids"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
