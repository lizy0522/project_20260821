"""Frozen Round 0--4 statewise-best 5B C2 -> Aend retrieval."""

from __future__ import annotations

import json
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

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

from behavior_modeling.shared.basis import build_mp_basis  # noqa: E402
from behavior_modeling.shared.evaluation import calculate_nmse  # noqa: E402
from behavior_modeling.shared.select_statewise_best_round0_4_models import (  # noqa: E402
    RESULT_ROOT as MODEL_SELECTION_ROOT,
)
from behavior_modeling.shared.statewise_adaptive_model_search import (  # noqa: E402
    STATE_COUNT,
    PreparedStatePairs,
    prepare_state_pairs,
)
from data_management.shared import build_state_table  # noqa: E402

BASELINE_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_retrieval"
    / "scenario_2" /"scenario_2_C2_to_Aend_retrieval"
    / "scenario_2_C2_to_Aend"
)
RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "retrieval_oriented_model_selection"
    / "scenario_2_statewise_best_round0_4_5b_retrieval"
)
DPD_SHAREABLE_THRESHOLD_DB = -40.0
FORMAL_A_LENGTH = 12288
FORMAL_B_LENGTH = 4915
FORMAL_C_LENGTH = 7373
WAVEFORM_LENGTH = 24576
STATE_IDS = np.arange(STATE_COUNT, dtype=np.int64)


def _parse_json(value: Any) -> Any:
    if not isinstance(value, str):
        raise ValueError("selected model descriptor must be a JSON string")
    return json.loads(value)


def _candidate_descriptor(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "orders": tuple(int(value) for value in _parse_json(row["orders"])),
        "memory": {
            int(key): int(value) for key, value in _parse_json(row["memory_definition"]).items()
        },
        "max_delay": int(row["max_delay"]),
        "coefficient_count": int(row["coefficient_count"]),
        "lambda": float(row["lambda"]),
        "candidate_id": int(row["candidate_id"]),
        "search_round": int(row["search_round"]),
    }


def _load_selected_inputs() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    dict[str, np.ndarray],
    dict[str, np.ndarray],
    np.ndarray,
    dict[str, Any],
]:
    selection_validation_path = MODEL_SELECTION_ROOT / "selection_validation.json"
    if not selection_validation_path.is_file():
        raise FileNotFoundError("selected model validation is missing")
    selection_validation = json.loads(selection_validation_path.read_text(encoding="utf-8"))
    if (
        selection_validation.get("persisted_search_round_max") != 4
        or selection_validation.get("persisted_candidate_count") != 1064
        or selection_validation.get("round_5_used") is not False
        or selection_validation.get("additional_model_search_performed") is not False
    ):
        raise RuntimeError("selected model source is not the frozen Round 0--4 pool")
    aend = pd.read_csv(MODEL_SELECTION_ROOT / "selected_Aend_models.csv")
    c2 = pd.read_csv(MODEL_SELECTION_ROOT / "selected_C2_models.csv")
    provenance = {
        "persisted_search_round_max": int(selection_validation["persisted_search_round_max"]),
        "persisted_candidate_count": int(selection_validation["persisted_candidate_count"]),
        "round_5_used": False,
        "selection_validation": selection_validation,
    }
    if aend.shape[0] != STATE_COUNT or c2.shape[0] != STATE_COUNT:
        raise RuntimeError("selected Aend/C2 counts must both be 425")
    for frame in (aend, c2):
        if frame["search_round"].astype(int).max() > 4:
            raise RuntimeError("selected model uses a Round 5 candidate")
        if not np.all(np.isfinite(frame["native_train_NMSE_dB"].to_numpy(dtype=float))):
            raise RuntimeError("selected model train metrics are not finite")
        if not np.all(np.isfinite(frame["native_B_NMSE_dB"].to_numpy(dtype=float))):
            raise RuntimeError("selected model B metrics are not finite")
    theta_a_path = MODEL_SELECTION_ROOT / "selected_Aend_coefficients.npz"
    theta_c_path = MODEL_SELECTION_ROOT / "selected_C2_coefficients.npz"
    with np.load(theta_a_path, allow_pickle=False) as data:
        theta_a = {key: np.asarray(data[key], dtype=np.complex128) for key in data.files}
    with np.load(theta_c_path, allow_pickle=False) as data:
        theta_c = {key: np.asarray(data[key], dtype=np.complex128) for key in data.files}
    expected_keys = {f"state_{state_id:03d}" for state_id in STATE_IDS}
    if set(theta_a) != expected_keys or set(theta_c) != expected_keys:
        raise RuntimeError("selected coefficient NPZ keys do not cover all 425 states")
    with np.load(BASELINE_ROOT / "lut_fingerprints_Aend.npz", allow_pickle=False) as data:
        common_b = np.asarray(data["common_B_input"], dtype=np.complex128)
    if common_b.shape != (FORMAL_B_LENGTH,) or not np.all(np.isfinite(common_b)):
        raise RuntimeError("formal common B probe is invalid")
    return aend, c2, theta_a, theta_c, common_b, provenance


def _common_metrics(
    prepared: PreparedStatePairs,
    side: str,
    descriptor: Mapping[str, Any],
    theta: np.ndarray,
    common_delay: int,
) -> tuple[float, float]:
    canonical = prepared.a_end if side == "Aend" else prepared.c2
    train_name = "A" if side == "Aend" else "C"
    train = canonical[train_name]
    b_segment = canonical["B"]
    phi_train = build_mp_basis(train.input, descriptor["orders"], descriptor["memory"])
    phi_b = build_mp_basis(b_segment.input, descriptor["orders"], descriptor["memory"])
    native_train = phi_train @ theta
    native_b = phi_b @ theta
    offset = common_delay - int(descriptor["max_delay"])
    if offset < 0:
        raise RuntimeError("common support precedes a selected model native support")
    train_nmse = calculate_nmse(
        np.asarray(train.output[common_delay:], dtype=np.complex128), native_train[offset:]
    )
    b_nmse = calculate_nmse(
        np.asarray(b_segment.output[common_delay:], dtype=np.complex128), native_b[offset:]
    )
    if not np.isfinite(train_nmse) or not np.isfinite(b_nmse):
        raise RuntimeError("common-support metric is non-finite")
    return float(train_nmse), float(b_nmse)


def build_common_support_diagnostics(
    aend: pd.DataFrame,
    c2: pd.DataFrame,
    theta_a: Mapping[str, np.ndarray],
    theta_c: Mapping[str, np.ndarray],
    common_delay: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    a_by_state = {int(row["state_id"]): row for row in aend.to_dict("records")}
    c_by_state = {int(row["state_id"]): row for row in c2.to_dict("records")}
    for state_id in STATE_IDS:
        state_id_int = int(state_id)
        prepared = prepare_state_pairs(state_id_int)
        a_desc = _candidate_descriptor(a_by_state[state_id_int])
        c_desc = _candidate_descriptor(c_by_state[state_id_int])
        a_train, a_b = _common_metrics(
            prepared, "Aend", a_desc, theta_a[f"state_{state_id_int:03d}"], common_delay
        )
        c_train, c_b = _common_metrics(
            prepared, "C2", c_desc, theta_c[f"state_{state_id_int:03d}"], common_delay
        )
        rows.append(
            {
                "state_id": state_id_int,
                "Aend_common_train_NMSE_dB": a_train,
                "Aend_common_B_NMSE_dB": a_b,
                "C2_common_train_NMSE_dB": c_train,
                "C2_common_B_NMSE_dB": c_b,
                "M_common": common_delay,
            }
        )
        if (state_id_int + 1) % 25 == 0 or state_id_int == STATE_COUNT - 1:
            print(f"common support: processed {state_id_int + 1}/{STATE_COUNT} states", flush=True)
    return pd.DataFrame(rows).sort_values("state_id").reset_index(drop=True)


def build_fingerprints(
    selected: pd.DataFrame,
    theta: Mapping[str, np.ndarray],
    common_b_input: np.ndarray,
    common_delay: int,
) -> np.ndarray:
    fingerprints = np.empty((STATE_COUNT, FORMAL_B_LENGTH - common_delay), dtype=np.complex128)
    for row in selected.sort_values("state_id").to_dict("records"):
        state_id = int(row["state_id"])
        descriptor = _candidate_descriptor(row)
        phi = build_mp_basis(common_b_input, descriptor["orders"], descriptor["memory"])
        prediction = phi @ theta[f"state_{state_id:03d}"]
        offset = common_delay - descriptor["max_delay"]
        if offset < 0:
            raise RuntimeError("fingerprint common support offset is negative")
        value = np.asarray(prediction[offset:], dtype=np.complex128)
        if value.shape != (FORMAL_B_LENGTH - common_delay,) or not np.all(np.isfinite(value)):
            raise RuntimeError(f"state={state_id} fingerprint shape/finite check failed")
        fingerprints[state_id] = value
    if not np.all(np.linalg.norm(fingerprints, axis=1) > 0):
        raise RuntimeError("fingerprint norm is not positive for all states")
    return fingerprints


def compute_generic_cnmse_distance_matrix(query: np.ndarray, candidate: np.ndarray) -> np.ndarray:
    query = np.asarray(query, dtype=np.complex128)
    candidate = np.asarray(candidate, dtype=np.complex128)
    if query.ndim != 2 or candidate.ndim != 2 or query.shape != candidate.shape:
        raise ValueError("query and candidate fingerprints must have the same 2D shape")
    if (
        query.shape[0] != STATE_COUNT
        or not np.all(np.isfinite(query))
        or not np.all(np.isfinite(candidate))
    ):
        raise ValueError("fingerprint matrix shape/finite check failed")
    q_energy = np.sum(np.abs(query) ** 2, axis=1, dtype=np.float64)
    c_energy = np.sum(np.abs(candidate) ** 2, axis=1, dtype=np.float64)
    if np.any(q_energy <= 0) or np.any(c_energy <= 0):
        raise ValueError("fingerprint energy must be positive")
    with np.errstate(over="ignore", invalid="ignore"):
        squared = q_energy[:, None] + c_energy[None, :] - 2.0 * np.real(query @ candidate.conj().T)
    scale = q_energy[:, None] + c_energy[None, :] + 1.0
    squared[np.abs(squared) <= 32.0 * np.finfo(np.float64).eps * scale] = 0.0
    squared = np.maximum(squared, 0.0)
    if np.array_equal(query, candidate):
        np.fill_diagonal(squared, 0.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        distance = 10.0 * np.log10(squared / q_energy[:, None])
    distance[squared == 0.0] = -np.inf
    if np.isnan(distance).any() or np.isposinf(distance).any():
        raise RuntimeError("fingerprint CNMSE matrix contains NaN/+Inf")
    return distance.astype(np.float64, copy=False)


def stable_top1(distance: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    distance = np.asarray(distance, dtype=np.float64)
    if (
        distance.shape != (STATE_COUNT, STATE_COUNT)
        or np.isnan(distance).any()
        or np.isposinf(distance).any()
    ):
        raise ValueError("distance matrix shape/finite check failed")
    selected = np.empty(STATE_COUNT, dtype=np.int64)
    selected_distance = np.empty(STATE_COUNT, dtype=np.float64)
    tie_count = np.empty(STATE_COUNT, dtype=np.int64)
    ranking = np.empty((STATE_COUNT, STATE_COUNT), dtype=np.int64)
    for state_id in STATE_IDS:
        order = np.lexsort((STATE_IDS, distance[int(state_id)]))
        ranking[int(state_id)] = order
        selected[int(state_id)] = int(order[0])
        selected_distance[int(state_id)] = float(distance[int(state_id), order[0]])
        tie_count[int(state_id)] = int(
            np.count_nonzero(distance[int(state_id)] == distance[int(state_id), order[0]])
        )
    return selected, selected_distance, tie_count, ranking


def load_canonical_real_b() -> np.ndarray:
    with np.load(BASELINE_ROOT / "real_B_distance_matrix.npz", allow_pickle=False) as data:
        distance = np.asarray(data["D_B"], dtype=np.float64)
        ranking = np.asarray(data["R_B"], dtype=np.float64)
    if distance.shape != (STATE_COUNT, STATE_COUNT) or ranking.shape != distance.shape:
        raise RuntimeError("canonical Real-B shape mismatch")
    if np.isnan(distance).any() or np.isposinf(distance).any() or not np.all(np.isfinite(ranking)):
        raise RuntimeError("canonical Real-B contains illegal values")
    if not np.all(np.isneginf(np.diag(distance))):
        raise RuntimeError("canonical Real-B diagonal must be -Inf")
    return distance


def format_load_config(row: Any) -> str:
    def mismatch(value: Any) -> str:
        number = int(round(float(value)))
        return "0" if number == 0 else f"{number / 100:.2f}".rstrip("0").rstrip(".")

    return (
        f"funMng={mismatch(row.funMng)}, funAng={int(row.funAng)}°, "
        f"secMng={mismatch(row.secMng)}, secAng={int(row.secAng)}°"
    )


def build_retrieval_tables(
    aend: pd.DataFrame,
    c2: pd.DataFrame,
    common_diag: pd.DataFrame,
    state_q: np.ndarray,
    top1_distance: np.ndarray,
    real_b: np.ndarray,
    raw_scalars: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    state_frame = pd.DataFrame(build_state_table()).sort_values("state_id").reset_index(drop=True)
    a = aend.sort_values("state_id").reset_index(drop=True)
    c = c2.sort_values("state_id").reset_index(drop=True)
    common = common_diag.sort_values("state_id").reset_index(drop=True)
    if common.shape[0] != STATE_COUNT or not np.array_equal(
        common["state_id"].to_numpy(dtype=np.int64), STATE_IDS
    ):
        raise RuntimeError("common-support diagnostics must cover state_id 0...424")
    raw = raw_scalars.sort_values("state_id").reset_index(drop=True)
    load_config = [format_load_config(row) for row in state_frame.itertuples(index=False)]
    exact = state_q == STATE_IDS
    retrieved = np.asarray(real_b, dtype=float)
    if not np.array_equal(np.isneginf(retrieved), exact):
        raise RuntimeError("canonical Real-B -Inf pattern does not match self Top1")
    result = pd.DataFrame(
        {
            "state_id_R": STATE_IDS,
            "load_config": load_config,
            "nmse_withoutdpd_dB": raw["nmse_withoutdpd"].to_numpy(dtype=float),
            "acpr_withoutdpd_avg_dBc": raw["acpr_withoutdpd_avg_dBc"].to_numpy(dtype=float),
            "Y_Aend_train_NMSE_dB": a["native_train_NMSE_dB"].to_numpy(dtype=float),
            "Y_Aend_B_NMSE_dB": a["native_B_NMSE_dB"].to_numpy(dtype=float),
            "Y_C2_train_NMSE_dB": c["native_train_NMSE_dB"].to_numpy(dtype=float),
            "Y_C2_B_NMSE_dB": c["native_B_NMSE_dB"].to_numpy(dtype=float),
            "state_id_Q": state_q.astype(np.int64),
            "retrieved_real_B_CNMSE_dB": retrieved,
        }
    )
    if result.shape != (STATE_COUNT, 10):
        raise RuntimeError("retrieval results must be 425x10")
    diagnostics = pd.DataFrame(
        {
            "state_id_R": STATE_IDS,
            "state_id_Q": state_q.astype(np.int64),
            "state_id_delta_signed": state_q.astype(np.int64) - STATE_IDS,
            "state_id_delta_abs": np.abs(state_q.astype(np.int64) - STATE_IDS),
            "top1_fingerprint_CNMSE_dB": top1_distance,
            "retrieved_real_B_CNMSE_dB": retrieved,
            "exact_hit": exact,
            "dpd_shareable": np.isneginf(retrieved) | (retrieved < DPD_SHAREABLE_THRESHOLD_DB),
        }
    )
    return result, diagnostics


__all__ = [
    "BASELINE_ROOT",
    "DPD_SHAREABLE_THRESHOLD_DB",
    "FORMAL_B_LENGTH",
    "RESULT_ROOT",
    "STATE_COUNT",
    "build_common_support_diagnostics",
    "build_fingerprints",
    "build_retrieval_tables",
    "compute_generic_cnmse_distance_matrix",
    "load_canonical_real_b",
    "stable_top1",
]
