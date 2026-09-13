"""Compact odd-order, order-dependent-memory MP scan for the failed-14 study.

This module owns only the new 14-state behavior-model experiment.  It keeps the
existing full-record canonical preprocessing and uses ordinary complex
``numpy.linalg.lstsq``.  Candidate-level multiprocessing is implemented by the
runner; a worker loads the read-only preprocessed state cache once and then
evaluates one candidate across all requested states.
"""

# Imports intentionally follow the BLAS-thread environment setup below.
# ruff: noqa: E402,I001

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

for _thread_name in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_name] = "1"

import numpy as np
import pandas as pd

from behavior_model.evaluation import calculate_nmse
from data_manager import load_by_id
from signal_segmentation import (
    build_partition_from_xin,
    get_common_probe,
    get_ilc_pair,
    preprocess_full_pair,
)

STATE_COUNT = 14
ALL_STATE_COUNT = 425
FAILED_STATE_IDS = (
    187,
    189,
    195,
    196,
    199,
    206,
    323,
    327,
    330,
    335,
    340,
    344,
    346,
    354,
)
WAVEFORM_LENGTH = 24576
FORMAL_A_LENGTH = 12288
FORMAL_B_LENGTH = 4915
FORMAL_C_LENGTH = 7373
C2_STAGE = 2
THRESHOLD_DB = -40.0
ALLOWED_ORDERS = (1, 3, 5, 7, 9)
MEMORY_MAX = 4
REGRESSION_STATE_IDS = (187, 195, 340, 354)
REGRESSION_PROFILES = (
    (3, (2, 1)),
    (5, (3, 2, 1)),
    (7, (4, 3, 2, 1)),
    (9, (4, 3, 2, 1, 1)),
)


@dataclass(frozen=True)
class VariableMemoryCandidate:
    """One deterministic odd-order candidate (IDs are zero-based)."""

    candidate_id: int
    P: int
    orders: tuple[int, ...]
    memory_profile: tuple[int, ...]
    max_delay: int
    coefficient_count: int
    memory_profile_string: str

    @property
    def memory_definition(self) -> dict[int, int]:
        return {int(order): int(depth) for order, depth in zip(self.orders, self.memory_profile)}

    @property
    def basis_terms(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (int(order), int(delay))
            for order, depth in zip(self.orders, self.memory_profile)
            for delay in range(int(depth))
        )


@dataclass(frozen=True)
class PreparedState:
    """Canonical arrays needed by the failed-14 study."""

    state_id: int
    ilc_A_end: int
    common_b_input: np.ndarray
    aend_A_input: np.ndarray
    aend_A_output: np.ndarray
    aend_B_output: np.ndarray
    c2_C_input: np.ndarray
    c2_C_output: np.ndarray
    c2_B_output: np.ndarray
    rough_delay_Aend: int
    fraction_delay_Aend: float
    rough_delay_C2: int
    fraction_delay_C2: float


@dataclass(frozen=True)
class StateBases:
    """Maximum odd-order/M=4 bases with original-time row support."""

    state_id: int
    ilc_A_end: int
    common_b_phi: np.ndarray
    aend_A_phi: np.ndarray
    aend_A_y: np.ndarray
    aend_B_y: np.ndarray
    c2_C_phi: np.ndarray
    c2_C_y: np.ndarray
    c2_B_y: np.ndarray


FULL_TERMS = tuple(
    (int(order), int(delay)) for order in ALLOWED_ORDERS for delay in range(MEMORY_MAX)
)
FULL_TERM_TO_COLUMN = {term: index for index, term in enumerate(FULL_TERMS)}


def _json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _memory_profiles(order_count: int) -> Iterable[tuple[int, ...]]:
    """Yield all 1..4 profiles in lexicographic order with non-increase."""

    if not 1 <= int(order_count) <= len(ALLOWED_ORDERS):
        raise ValueError("order_count must be between 1 and 5")
    profile = [1] * int(order_count)

    def visit(position: int) -> Iterable[tuple[int, ...]]:
        if position == order_count:
            yield tuple(profile)
            return
        upper = profile[position - 1] if position > 0 else MEMORY_MAX
        for depth in range(1, upper + 1):
            profile[position] = depth
            yield from visit(position + 1)

    yield from visit(0)


def generate_candidates() -> tuple[VariableMemoryCandidate, ...]:
    """Generate the complete 4+10+20+35+56 deterministic candidate grid."""

    candidates: list[VariableMemoryCandidate] = []
    candidate_id = 0
    expected_by_p = {1: 4, 3: 10, 5: 20, 7: 35, 9: 56}
    for order_count, P in enumerate(ALLOWED_ORDERS, start=1):
        profiles = tuple(_memory_profiles(order_count))
        if len(profiles) != expected_by_p[P]:
            raise RuntimeError(f"P={P} profile count mismatch: {len(profiles)}")
        for memory_profile in profiles:
            candidate = VariableMemoryCandidate(
                candidate_id=candidate_id,
                P=int(P),
                orders=tuple(ALLOWED_ORDERS[:order_count]),
                memory_profile=tuple(int(value) for value in memory_profile),
                max_delay=int(max(memory_profile) - 1),
                coefficient_count=int(sum(memory_profile)),
                memory_profile_string=(
                    f"P{P}_M[{','.join(str(value) for value in memory_profile)}]"
                ),
            )
            candidates.append(candidate)
            candidate_id += 1
    if len(candidates) != 125:
        raise RuntimeError(f"candidate_count must be 125, got {len(candidates)}")
    validate_candidate_grid(candidates)
    return tuple(candidates)


def validate_candidate_grid(candidates: Sequence[VariableMemoryCandidate]) -> None:
    """Validate counts, odd orders, monotonic profiles, K and max delay."""

    expected_counts = {1: 4, 3: 10, 5: 20, 7: 35, 9: 56}
    if len(candidates) != 125:
        raise ValueError(f"candidate_count must be 125, got {len(candidates)}")
    ids = [candidate.candidate_id for candidate in candidates]
    if ids != list(range(125)):
        raise ValueError("candidate_id must be deterministic 0...124")
    for P, expected in expected_counts.items():
        if sum(candidate.P == P for candidate in candidates) != expected:
            raise ValueError(f"P={P} candidate count mismatch")
    for candidate in candidates:
        if candidate.orders != tuple(range(1, candidate.P + 1, 2)):
            raise ValueError(f"candidate={candidate.candidate_id} orders are not contiguous odd")
        if any(order % 2 == 0 for order in candidate.orders):
            raise ValueError(f"candidate={candidate.candidate_id} contains an even order")
        if any(not 1 <= depth <= MEMORY_MAX for depth in candidate.memory_profile):
            raise ValueError(f"candidate={candidate.candidate_id} memory is outside 1...4")
        if any(
            candidate.memory_profile[index] < candidate.memory_profile[index + 1]
            for index in range(len(candidate.memory_profile) - 1)
        ):
            raise ValueError(f"candidate={candidate.candidate_id} memory is not non-increasing")
        if candidate.coefficient_count != sum(candidate.memory_profile):
            raise ValueError(f"candidate={candidate.candidate_id} K mismatch")
        if candidate.max_delay != max(candidate.memory_profile) - 1:
            raise ValueError(f"candidate={candidate.candidate_id} max_delay mismatch")
        if len(candidate.basis_terms) != candidate.coefficient_count:
            raise ValueError(f"candidate={candidate.candidate_id} basis term mismatch")
        if any(order % 2 == 0 for order, _ in candidate.basis_terms):
            raise ValueError(f"candidate={candidate.candidate_id} basis has an even order")


def candidate_grid_frame(candidates: Sequence[VariableMemoryCandidate]) -> pd.DataFrame:
    validate_candidate_grid(candidates)
    rows = [
        {
            "candidate_id": candidate.candidate_id,
            "P": candidate.P,
            "orders": _json_compact(list(candidate.orders)),
            "memory_profile": _json_compact(list(candidate.memory_profile)),
            "memory_profile_string": candidate.memory_profile_string,
            "memory_definition": _json_compact(candidate.memory_definition),
            "max_delay": candidate.max_delay,
            "coefficient_count": candidate.coefficient_count,
            "even_orders_used": False,
            "ridge_used": False,
        }
        for candidate in candidates
    ]
    return pd.DataFrame(rows).sort_values("candidate_id").reset_index(drop=True)


def _as_vector(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 2 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 1 or not np.iscomplexobj(array):
        raise ValueError(f"{name} must be a one-dimensional complex vector")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} contains NaN/Inf")
    return array.astype(np.complex128, copy=False)


def prepare_failed_state(state_id: int) -> PreparedState:
    """Apply the frozen full-record alignment/ABC/segment-gain pipeline once."""

    state_id = int(state_id)
    if state_id not in FAILED_STATE_IDS:
        raise ValueError(f"state_id {state_id} is not in the frozen failed-14 list")
    state_data = load_by_id(state_id)
    xin = _as_vector(state_data["xin"], "xin")
    if xin.size != WAVEFORM_LENGTH:
        raise ValueError(f"state={state_id} xin length must be {WAVEFORM_LENGTH}")
    partition = build_partition_from_xin(xin)
    if (partition.n_a, partition.n_b, partition.n_c) != (
        FORMAL_A_LENGTH,
        FORMAL_B_LENGTH,
        FORMAL_C_LENGTH,
    ):
        raise RuntimeError("frozen ABC ownership changed")
    input_history = np.asarray(state_data["xin_pd_ori_ilc"])
    output_history = np.asarray(state_data["yout_withdpd_ori_ilc"])
    if input_history.ndim != 2 or output_history.shape != input_history.shape:
        raise ValueError(f"state={state_id} ILC history shape mismatch")
    if input_history.shape[0] != WAVEFORM_LENGTH or input_history.shape[1] < C2_STAGE:
        raise RuntimeError(f"state={state_id} does not have a real C2 column")
    ilc_A_end = int(input_history.shape[1])
    a_pair = get_ilc_pair(state_data, ilc_A_end - 1)
    a_end = preprocess_full_pair(
        a_pair.input_full,
        a_pair.output_raw_full,
        partition,
        pair_type="ILC",
        iteration_index=ilc_A_end - 1,
        input_peak_normalization_factor=a_pair.input_peak_normalization_factor,
    )
    c_pair = get_ilc_pair(state_data, C2_STAGE - 1)
    c2 = (
        a_end
        if ilc_A_end == C2_STAGE
        else preprocess_full_pair(
            c_pair.input_full,
            c_pair.output_raw_full,
            partition,
            pair_type="ILC",
            iteration_index=C2_STAGE - 1,
            input_peak_normalization_factor=c_pair.input_peak_normalization_factor,
        )
    )
    common_b_input = np.asarray(get_common_probe(state_data, partition), dtype=np.complex128)
    if common_b_input.shape != (FORMAL_B_LENGTH,) or not np.all(np.isfinite(common_b_input)):
        raise RuntimeError(f"state={state_id} common B probe is invalid")
    return PreparedState(
        state_id=state_id,
        ilc_A_end=ilc_A_end,
        common_b_input=common_b_input,
        aend_A_input=np.asarray(a_end["A"].input, dtype=np.complex128),
        aend_A_output=np.asarray(a_end["A"].output, dtype=np.complex128),
        aend_B_output=np.asarray(a_end["B"].output, dtype=np.complex128),
        c2_C_input=np.asarray(c2["C"].input, dtype=np.complex128),
        c2_C_output=np.asarray(c2["C"].output, dtype=np.complex128),
        c2_B_output=np.asarray(c2["B"].output, dtype=np.complex128),
        rough_delay_Aend=int(a_end.rough_delay),
        fraction_delay_Aend=float(a_end.fraction_delay),
        rough_delay_C2=int(c2.rough_delay),
        fraction_delay_C2=float(c2.fraction_delay),
    )


def save_prepared_state(prepared: PreparedState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        state_id=np.asarray(prepared.state_id, dtype=np.int64),
        ilc_A_end=np.asarray(prepared.ilc_A_end, dtype=np.int64),
        common_b_input=prepared.common_b_input,
        aend_A_input=prepared.aend_A_input,
        aend_A_output=prepared.aend_A_output,
        aend_B_output=prepared.aend_B_output,
        c2_C_input=prepared.c2_C_input,
        c2_C_output=prepared.c2_C_output,
        c2_B_output=prepared.c2_B_output,
        rough_delay_Aend=np.asarray(prepared.rough_delay_Aend, dtype=np.int64),
        fraction_delay_Aend=np.asarray(prepared.fraction_delay_Aend, dtype=np.float64),
        rough_delay_C2=np.asarray(prepared.rough_delay_C2, dtype=np.int64),
        fraction_delay_C2=np.asarray(prepared.fraction_delay_C2, dtype=np.float64),
    )


def load_prepared_state(path: Path) -> PreparedState:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "state_id",
            "ilc_A_end",
            "common_b_input",
            "aend_A_input",
            "aend_A_output",
            "aend_B_output",
            "c2_C_input",
            "c2_C_output",
            "c2_B_output",
            "rough_delay_Aend",
            "fraction_delay_Aend",
            "rough_delay_C2",
            "fraction_delay_C2",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"prepared cache missing fields: {missing}")
        arrays = {name: np.asarray(data[name]) for name in required}
    state_id = int(np.asarray(arrays["state_id"]).reshape(-1)[0])
    vector_fields = (
        "common_b_input",
        "aend_A_input",
        "aend_A_output",
        "aend_B_output",
        "c2_C_input",
        "c2_C_output",
        "c2_B_output",
    )
    for name in vector_fields:
        value = arrays[name]
        if value.ndim != 1 or not np.iscomplexobj(value) or not np.all(np.isfinite(value)):
            raise ValueError(f"prepared cache field {name} is invalid for state {state_id}")
    expected = {
        "common_b_input": FORMAL_B_LENGTH,
        "aend_A_input": FORMAL_A_LENGTH,
        "aend_A_output": FORMAL_A_LENGTH,
        "aend_B_output": FORMAL_B_LENGTH,
        "c2_C_input": FORMAL_C_LENGTH,
        "c2_C_output": FORMAL_C_LENGTH,
        "c2_B_output": FORMAL_B_LENGTH,
    }
    for name, length in expected.items():
        if arrays[name].shape != (length,):
            raise ValueError(f"prepared cache {name} shape mismatch: {arrays[name].shape}")
    return PreparedState(
        state_id=state_id,
        ilc_A_end=int(np.asarray(arrays["ilc_A_end"]).reshape(-1)[0]),
        common_b_input=arrays["common_b_input"].astype(np.complex128, copy=False),
        aend_A_input=arrays["aend_A_input"].astype(np.complex128, copy=False),
        aend_A_output=arrays["aend_A_output"].astype(np.complex128, copy=False),
        aend_B_output=arrays["aend_B_output"].astype(np.complex128, copy=False),
        c2_C_input=arrays["c2_C_input"].astype(np.complex128, copy=False),
        c2_C_output=arrays["c2_C_output"].astype(np.complex128, copy=False),
        c2_B_output=arrays["c2_B_output"].astype(np.complex128, copy=False),
        rough_delay_Aend=int(np.asarray(arrays["rough_delay_Aend"]).reshape(-1)[0]),
        fraction_delay_Aend=float(np.asarray(arrays["fraction_delay_Aend"]).reshape(-1)[0]),
        rough_delay_C2=int(np.asarray(arrays["rough_delay_C2"]).reshape(-1)[0]),
        fraction_delay_C2=float(np.asarray(arrays["fraction_delay_C2"]).reshape(-1)[0]),
    )


def _full_basis(x: np.ndarray) -> np.ndarray:
    """Build the P9/M=4 basis with original-time rows retained."""

    x = np.asarray(x, dtype=np.complex128)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("basis input must be finite complex vector")
    phi = np.full((x.size, len(FULL_TERMS)), np.nan + 0j, dtype=np.complex128)
    for column, (order, delay) in enumerate(FULL_TERMS):
        delayed = x[: x.size - delay]
        phi[delay:, column] = delayed * np.abs(delayed) ** (order - 1)
    return phi


def build_state_bases(prepared: PreparedState) -> StateBases:
    """Build maximum bases once for one cached state."""

    common_phi = _full_basis(prepared.common_b_input)
    a_phi = _full_basis(prepared.aend_A_input)
    c_phi = _full_basis(prepared.c2_C_input)
    for name, value in (("common", common_phi), ("Aend", a_phi), ("C2", c_phi)):
        if not np.all(np.isfinite(value[MEMORY_MAX - 1 :])):
            raise RuntimeError(f"{name} maximum basis contains nonfinite valid rows")
    return StateBases(
        state_id=prepared.state_id,
        ilc_A_end=prepared.ilc_A_end,
        common_b_phi=common_phi,
        aend_A_phi=a_phi,
        aend_A_y=prepared.aend_A_output,
        aend_B_y=prepared.aend_B_output,
        c2_C_phi=c_phi,
        c2_C_y=prepared.c2_C_output,
        c2_B_y=prepared.c2_B_output,
    )


def load_state_bases(cache_dir: Path, state_ids: Iterable[int]) -> dict[int, StateBases]:
    result: dict[int, StateBases] = {}
    for state_id in state_ids:
        path = Path(cache_dir) / f"state_{int(state_id):03d}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"missing prepared state cache: {path}")
        prepared = load_prepared_state(path)
        if prepared.state_id != int(state_id):
            raise RuntimeError(f"prepared cache state mismatch: {path}")
        result[prepared.state_id] = build_state_bases(prepared)
    return result


def candidate_columns(candidate: VariableMemoryCandidate) -> tuple[int, ...]:
    columns = tuple(FULL_TERM_TO_COLUMN[term] for term in candidate.basis_terms)
    if len(columns) != candidate.coefficient_count:
        raise RuntimeError("candidate basis column count mismatch")
    return columns


def _fit_side(
    phi_train_full: np.ndarray,
    y_train_full: np.ndarray,
    phi_b_full: np.ndarray,
    y_b: np.ndarray,
    candidate: VariableMemoryCandidate,
    side: str,
) -> tuple[dict[str, Any], np.ndarray | None]:
    delay = int(candidate.max_delay)
    columns = candidate_columns(candidate)
    phi_train = np.asarray(phi_train_full[delay:, :][:, columns], dtype=np.complex128)
    y_train = np.asarray(y_train_full[delay:], dtype=np.complex128)
    phi_b = np.asarray(phi_b_full[delay:, :][:, columns], dtype=np.complex128)
    y_b = np.asarray(y_b[delay:], dtype=np.complex128)
    prefix = f"{side}_"
    result: dict[str, Any] = {
        f"{prefix}train_NMSE_dB": np.nan,
        f"{prefix}B_NMSE_dB": np.nan,
        f"{prefix}generalization_gap_dB": np.nan,
        f"{prefix}matrix_rank": 0,
        f"{prefix}column_count": candidate.coefficient_count,
        f"{prefix}rank_ratio": np.nan,
        f"{prefix}singular_value_max": np.nan,
        f"{prefix}singular_value_min": np.nan,
        f"{prefix}condition_number": np.nan,
        f"{prefix}theta_l2_norm": np.nan,
        f"{prefix}theta_max_abs": np.nan,
        f"{prefix}n_train_samples": int(phi_train.shape[0]),
        f"{prefix}n_B_samples": int(phi_b.shape[0]),
        f"{prefix}finite": False,
        f"{prefix}valid": False,
        f"{prefix}failure_reason": "",
    }
    try:
        if phi_train.shape[0] <= candidate.coefficient_count:
            raise ValueError("valid training rows are not greater than coefficient count")
        if phi_b.shape[0] != y_b.size or y_train.size != phi_train.shape[0]:
            raise ValueError("basis/target support length mismatch")
        if not np.all(np.isfinite(phi_train)) or not np.all(np.isfinite(phi_b)):
            raise ValueError("basis contains NaN/Inf")
        theta, _, rank, singular = np.linalg.lstsq(phi_train, y_train, rcond=None)
        singular = np.asarray(singular, dtype=np.float64)
        if int(rank) != candidate.coefficient_count:
            raise ValueError(f"rank deficient {rank}/{candidate.coefficient_count}")
        if singular.size != candidate.coefficient_count or singular[-1] <= 0:
            raise ValueError("singular values are invalid")
        if not np.all(np.isfinite(theta)):
            raise ValueError("theta contains NaN/Inf")
        train_value = float(calculate_nmse(y_train, phi_train @ theta))
        b_value = float(calculate_nmse(y_b, phi_b @ theta))
        if not np.isfinite(train_value) or not np.isfinite(b_value):
            raise ValueError("NMSE is not finite")
        condition = float(singular[0] / singular[-1])
        result.update(
            {
                f"{prefix}train_NMSE_dB": train_value,
                f"{prefix}B_NMSE_dB": b_value,
                f"{prefix}generalization_gap_dB": b_value - train_value,
                f"{prefix}matrix_rank": int(rank),
                f"{prefix}rank_ratio": float(rank / candidate.coefficient_count),
                f"{prefix}singular_value_max": float(singular[0]),
                f"{prefix}singular_value_min": float(singular[-1]),
                f"{prefix}condition_number": condition,
                f"{prefix}theta_l2_norm": float(np.linalg.norm(theta)),
                f"{prefix}theta_max_abs": float(np.max(np.abs(theta))),
                f"{prefix}finite": True,
                f"{prefix}valid": bool(np.isfinite(condition)),
                f"{prefix}failure_reason": "" if np.isfinite(condition) else "condition_nonfinite",
            }
        )
        return result, np.asarray(theta, dtype=np.complex128)
    except Exception as exc:
        result[f"{prefix}failure_reason"] = f"{type(exc).__name__}: {exc}"
        return result, None


def evaluate_candidate_on_bases(
    candidate: VariableMemoryCandidate,
    bases_by_state: Mapping[int, StateBases],
    *,
    return_theta: bool = True,
) -> dict[str, Any]:
    """Evaluate one candidate over all supplied states (serial reference too)."""

    rows: list[dict[str, Any]] = []
    theta_a: dict[int, np.ndarray] = {}
    theta_c: dict[int, np.ndarray] = {}
    for state_id in sorted(bases_by_state):
        bases = bases_by_state[state_id]
        a_result, a_theta = _fit_side(
            bases.aend_A_phi,
            bases.aend_A_y,
            bases.common_b_phi,
            bases.aend_B_y,
            candidate,
            "Aend",
        )
        c_result, c_theta = _fit_side(
            bases.c2_C_phi,
            bases.c2_C_y,
            bases.common_b_phi,
            bases.c2_B_y,
            candidate,
            "C2",
        )
        a_train = float(a_result["Aend_train_NMSE_dB"])
        a_b = float(a_result["Aend_B_NMSE_dB"])
        c_train = float(c_result["C2_train_NMSE_dB"])
        c_b = float(c_result["C2_B_NMSE_dB"])
        a_joint = bool(a_result["Aend_valid"] and a_train < THRESHOLD_DB and a_b < THRESHOLD_DB)
        c_joint = bool(c_result["C2_valid"] and c_train < THRESHOLD_DB and c_b < THRESHOLD_DB)
        row: dict[str, Any] = {
            "candidate_id": candidate.candidate_id,
            "P": candidate.P,
            "orders": _json_compact(list(candidate.orders)),
            "memory_profile": _json_compact(list(candidate.memory_profile)),
            "memory_profile_string": candidate.memory_profile_string,
            "max_delay": candidate.max_delay,
            "coefficient_count": candidate.coefficient_count,
            "state_id": int(state_id),
            "ilc_A_end": bases.ilc_A_end,
            "B_probe_source": "common_xin_B_probe",
            "Aend_joint_pass": a_joint,
            "C2_joint_pass": c_joint,
            "all_four_pass": bool(a_joint and c_joint),
            "valid": bool(a_result["Aend_valid"] and c_result["C2_valid"]),
            "failure_reason": "; ".join(
                value
                for value in (a_result["Aend_failure_reason"], c_result["C2_failure_reason"])
                if value
            ),
            **a_result,
            **c_result,
        }
        rows.append(row)
        if return_theta:
            if a_theta is not None:
                theta_a[int(state_id)] = a_theta
            if c_theta is not None:
                theta_c[int(state_id)] = c_theta
    return {
        "candidate_id": candidate.candidate_id,
        "rows": rows,
        "theta_a": theta_a,
        "theta_c": theta_c,
    }


def target_worker_count(logical_cpu_count: int | None = None) -> int:
    logical = int(logical_cpu_count or os.cpu_count() or 1)
    return max(1, int(math.floor(0.90 * logical)))


__all__ = [
    "ALL_STATE_COUNT",
    "ALLOWED_ORDERS",
    "C2_STAGE",
    "FAILED_STATE_IDS",
    "FORMAL_A_LENGTH",
    "FORMAL_B_LENGTH",
    "FORMAL_C_LENGTH",
    "MEMORY_MAX",
    "PreparedState",
    "REGRESSION_PROFILES",
    "REGRESSION_STATE_IDS",
    "STATE_COUNT",
    "StateBases",
    "THRESHOLD_DB",
    "VariableMemoryCandidate",
    "WAVEFORM_LENGTH",
    "build_state_bases",
    "candidate_grid_frame",
    "candidate_columns",
    "evaluate_candidate_on_bases",
    "generate_candidates",
    "load_prepared_state",
    "load_state_bases",
    "prepare_failed_state",
    "save_prepared_state",
    "target_worker_count",
    "validate_candidate_grid",
]
