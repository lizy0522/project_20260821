"""P=5 [1,2,3,5] variable-memory OLS scan for the failed-14 ablation.

The canonical state preparation and read-only cache format are reused from the
previous failed-14 odd-only scan.  This module changes only the basis family:
the even second-order term is added, all 35 monotone memory profiles are
enumerated, and no Ridge/regularization path is present.
"""

# Imports intentionally follow the BLAS-thread environment setup below.
# ruff: noqa: E402,I001,E501

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
from behavior_model.odd_order_variable_memory_scan import (
    FAILED_STATE_IDS,
    FORMAL_A_LENGTH,
    FORMAL_B_LENGTH,
    FORMAL_C_LENGTH,
    PreparedState,
    load_prepared_state,
)

STATE_COUNT = 14
P = 5
ORDERS = (1, 2, 3, 5)
MEMORY_MAX = 4
MEMORY_PROFILES = tuple(
    tuple(values)
    for values in (
        (1, 1, 1, 1),
        (2, 1, 1, 1),
        (2, 2, 1, 1),
        (2, 2, 2, 1),
        (2, 2, 2, 2),
        (3, 1, 1, 1),
        (3, 2, 1, 1),
        (3, 2, 2, 1),
        (3, 2, 2, 2),
        (3, 3, 1, 1),
        (3, 3, 2, 1),
        (3, 3, 2, 2),
        (3, 3, 3, 1),
        (3, 3, 3, 2),
        (3, 3, 3, 3),
        (4, 1, 1, 1),
        (4, 2, 1, 1),
        (4, 2, 2, 1),
        (4, 2, 2, 2),
        (4, 3, 1, 1),
        (4, 3, 2, 1),
        (4, 3, 2, 2),
        (4, 3, 3, 1),
        (4, 3, 3, 2),
        (4, 3, 3, 3),
        (4, 4, 1, 1),
        (4, 4, 2, 1),
        (4, 4, 2, 2),
        (4, 4, 3, 1),
        (4, 4, 3, 2),
        (4, 4, 3, 3),
        (4, 4, 4, 1),
        (4, 4, 4, 2),
        (4, 4, 4, 3),
        (4, 4, 4, 4),
    )
)
THRESHOLD_DB = -40.0
REGRESSION_STATE_IDS = (187, 195, 340, 354)
REGRESSION_MEMORY_PROFILES = (
    (1, 1, 1, 1),
    (2, 2, 1, 1),
    (3, 2, 2, 1),
    (4, 3, 2, 1),
    (4, 4, 4, 4),
)


@dataclass(frozen=True)
class P5Order2Candidate:
    """One deterministic P5 candidate with an order-2 term (zero-based ID)."""

    candidate_id: int
    P: int
    orders: tuple[int, int, int, int]
    memory_profile: tuple[int, int, int, int]
    M1: int
    M2: int
    M3: int
    M5: int
    max_delay: int
    coefficient_count: int
    ridge_used: bool = False
    lambda_value: None = None

    @property
    def memory_definition(self) -> dict[int, int]:
        return {1: self.M1, 2: self.M2, 3: self.M3, 5: self.M5}

    @property
    def memory_profile_string(self) -> str:
        return f"P5_M[{','.join(str(v) for v in self.memory_profile)}]"

    @property
    def basis_terms(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (order, delay)
            for order, depth in zip(self.orders, self.memory_profile, strict=True)
            for delay in range(depth)
        )


@dataclass(frozen=True)
class StateBases:
    """P5+[2] maximum basis for one preprocessed failed state."""

    state_id: int
    ilc_A_end: int
    common_b_phi: np.ndarray
    aend_A_phi: np.ndarray
    aend_A_y: np.ndarray
    aend_B_y: np.ndarray
    c2_C_phi: np.ndarray
    c2_C_y: np.ndarray
    c2_B_y: np.ndarray


FULL_TERMS = tuple((order, delay) for order in ORDERS for delay in range(MEMORY_MAX))
FULL_TERM_TO_COLUMN = {term: index for index, term in enumerate(FULL_TERMS)}


def _json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def generate_candidates() -> tuple[P5Order2Candidate, ...]:
    candidates: list[P5Order2Candidate] = []
    for candidate_id, profile in enumerate(MEMORY_PROFILES):
        candidates.append(
            P5Order2Candidate(
                candidate_id=candidate_id,
                P=P,
                orders=ORDERS,
                memory_profile=profile,
                M1=profile[0],
                M2=profile[1],
                M3=profile[2],
                M5=profile[3],
                max_delay=max(profile) - 1,
                coefficient_count=sum(profile),
            )
        )
    validate_candidate_grid(candidates)
    return tuple(candidates)


def validate_candidate_grid(candidates: Sequence[P5Order2Candidate]) -> None:
    if len(MEMORY_PROFILES) != 35 or len(candidates) != 35:
        raise ValueError("memory_profile_count and candidate_count must both be 35")
    if [candidate.candidate_id for candidate in candidates] != list(range(35)):
        raise ValueError("candidate IDs must be deterministic 0...34")
    for candidate in candidates:
        if candidate.P != 5 or candidate.orders != ORDERS:
            raise ValueError("all candidates must use P=5 and orders=[1,2,3,5]")
        if any(order % 2 == 0 for order in candidate.orders) and 2 not in candidate.orders:
            raise ValueError("order-2 term is missing")
        if not all(1 <= depth <= MEMORY_MAX for depth in candidate.memory_profile):
            raise ValueError("memory depth outside 1...4")
        if not (candidate.M1 >= candidate.M2 >= candidate.M3 >= candidate.M5):
            raise ValueError("memory profile is not non-increasing")
        if candidate.coefficient_count != sum(candidate.memory_profile):
            raise ValueError("coefficient count mismatch")
        if candidate.max_delay != candidate.M1 - 1:
            raise ValueError("max_delay mismatch")
        if candidate.lambda_value is not None or candidate.ridge_used:
            raise ValueError("this experiment must be OLS-only")
        if len(candidate.basis_terms) != candidate.coefficient_count:
            raise ValueError("basis term count mismatch")


def candidate_grid_frame(candidates: Sequence[P5Order2Candidate]) -> pd.DataFrame:
    validate_candidate_grid(candidates)
    return pd.DataFrame(
        [
            {
                "candidate_id": candidate.candidate_id,
                "P": candidate.P,
                "orders": _json_compact(list(candidate.orders)),
                "M1": candidate.M1,
                "M2": candidate.M2,
                "M3": candidate.M3,
                "M5": candidate.M5,
                "memory_profile": _json_compact(list(candidate.memory_profile)),
                "memory_profile_string": candidate.memory_profile_string,
                "memory_definition": _json_compact(candidate.memory_definition),
                "max_delay": candidate.max_delay,
                "coefficient_count": candidate.coefficient_count,
                "ridge_used": False,
                "lambda": None,
            }
            for candidate in candidates
        ]
    )


def memory_profiles_frame() -> pd.DataFrame:
    return (
        candidate_grid_frame(
            tuple(
                P5Order2Candidate(
                    candidate_id=index,
                    P=P,
                    orders=ORDERS,
                    memory_profile=profile,
                    M1=profile[0],
                    M2=profile[1],
                    M3=profile[2],
                    M5=profile[3],
                    max_delay=max(profile) - 1,
                    coefficient_count=sum(profile),
                )
                for index, profile in enumerate(MEMORY_PROFILES)
            )
        )
        .loc[
            :,
            [
                "candidate_id",
                "M1",
                "M2",
                "M3",
                "M5",
                "memory_profile",
                "memory_profile_string",
                "max_delay",
                "coefficient_count",
            ],
        ]
        .rename(columns={"candidate_id": "profile_index"})
    )


def _full_basis(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex128)
    if x.ndim != 1 or not np.all(np.isfinite(x)):
        raise ValueError("basis input must be finite complex vector")
    phi = np.full((x.size, len(FULL_TERMS)), np.nan + 0j, dtype=np.complex128)
    for column, (order, delay) in enumerate(FULL_TERMS):
        delayed = x[: x.size - delay]
        phi[delay:, column] = delayed * np.abs(delayed) ** (order - 1)
    return phi


def build_state_bases(prepared: PreparedState) -> StateBases:
    common_phi = _full_basis(prepared.common_b_input)
    a_phi = _full_basis(prepared.aend_A_input)
    c_phi = _full_basis(prepared.c2_C_input)
    for value in (common_phi, a_phi, c_phi):
        if not np.all(np.isfinite(value[MEMORY_MAX - 1 :])):
            raise RuntimeError("maximum P5+[2] basis contains nonfinite valid rows")
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
        prepared = load_prepared_state(path)
        result[int(state_id)] = build_state_bases(prepared)
    return result


def candidate_columns(candidate: P5Order2Candidate) -> tuple[int, ...]:
    columns = tuple(FULL_TERM_TO_COLUMN[term] for term in candidate.basis_terms)
    if len(columns) != candidate.coefficient_count:
        raise RuntimeError("basis column count mismatch")
    return columns


def _fit_side(
    bases: StateBases,
    candidate: P5Order2Candidate,
    side: str,
) -> tuple[dict[str, Any], np.ndarray | None]:
    delay = candidate.max_delay
    columns = candidate_columns(candidate)
    if side == "Aend":
        phi_train_full, y_train_full, y_b_full = bases.aend_A_phi, bases.aend_A_y, bases.aend_B_y
    else:
        phi_train_full, y_train_full, y_b_full = bases.c2_C_phi, bases.c2_C_y, bases.c2_B_y
    phi_train = np.asarray(phi_train_full[delay:, :][:, columns], dtype=np.complex128)
    y_train = np.asarray(y_train_full[delay:], dtype=np.complex128)
    phi_b = np.asarray(bases.common_b_phi[delay:, :][:, columns], dtype=np.complex128)
    y_b = np.asarray(y_b_full[delay:], dtype=np.complex128)
    prefix = f"{side}_"
    row: dict[str, Any] = {
        f"{prefix}train_NMSE_dB": np.nan,
        f"{prefix}B_NMSE_dB": np.nan,
        f"{prefix}generalization_gap_dB": np.nan,
        f"{prefix}matrix_rank": 0,
        f"{prefix}column_count": candidate.coefficient_count,
        f"{prefix}rank_ratio": np.nan,
        f"{prefix}sigma_max": np.nan,
        f"{prefix}sigma_min": np.nan,
        f"{prefix}condition_number": np.nan,
        f"{prefix}theta_l2_norm": np.nan,
        f"{prefix}theta_max_abs": np.nan,
        f"{prefix}residual_norm": np.nan,
        f"{prefix}n_train_samples": int(phi_train.shape[0]),
        f"{prefix}n_B_samples": int(phi_b.shape[0]),
        f"{prefix}finite": False,
        f"{prefix}valid": False,
        f"{prefix}failure_reason": "",
    }
    try:
        if phi_train.shape[0] <= candidate.coefficient_count:
            raise ValueError("valid training rows are not greater than coefficient count")
        if phi_train.shape[0] != y_train.size or phi_b.shape[0] != y_b.size:
            raise ValueError("basis/target support length mismatch")
        if not np.all(np.isfinite(phi_train)) or not np.all(np.isfinite(phi_b)):
            raise ValueError("basis contains NaN/Inf")
        theta, _, rank, singular = np.linalg.lstsq(phi_train, y_train, rcond=None)
        singular = np.asarray(singular, dtype=np.float64)
        if int(rank) != candidate.coefficient_count or singular[-1] <= 0:
            raise ValueError(f"rank deficient {rank}/{candidate.coefficient_count}")
        train_value = float(calculate_nmse(y_train, phi_train @ theta))
        b_value = float(calculate_nmse(y_b, phi_b @ theta))
        if not np.isfinite(train_value) or not np.isfinite(b_value):
            raise ValueError("NMSE is not finite")
        residual = float(np.linalg.norm(y_train - phi_train @ theta))
        row.update(
            {
                f"{prefix}train_NMSE_dB": train_value,
                f"{prefix}B_NMSE_dB": b_value,
                f"{prefix}generalization_gap_dB": b_value - train_value,
                f"{prefix}matrix_rank": int(rank),
                f"{prefix}rank_ratio": float(rank / candidate.coefficient_count),
                f"{prefix}sigma_max": float(singular[0]),
                f"{prefix}sigma_min": float(singular[-1]),
                f"{prefix}condition_number": float(singular[0] / singular[-1]),
                f"{prefix}theta_l2_norm": float(np.linalg.norm(theta)),
                f"{prefix}theta_max_abs": float(np.max(np.abs(theta))),
                f"{prefix}residual_norm": residual,
                f"{prefix}finite": True,
                f"{prefix}valid": True,
            }
        )
        return row, np.asarray(theta, dtype=np.complex128)
    except Exception as exc:
        row[f"{prefix}failure_reason"] = f"{type(exc).__name__}: {exc}"
        return row, None


def evaluate_candidate_on_bases(
    candidate: P5Order2Candidate,
    bases_by_state: Mapping[int, StateBases],
    *,
    return_theta: bool = True,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    theta_a: dict[int, np.ndarray] = {}
    theta_c: dict[int, np.ndarray] = {}
    for state_id in sorted(bases_by_state):
        bases = bases_by_state[state_id]
        a_row, a_theta = _fit_side(bases, candidate, "Aend")
        c_row, c_theta = _fit_side(bases, candidate, "C2")
        a_train, a_b = float(a_row["Aend_train_NMSE_dB"]), float(a_row["Aend_B_NMSE_dB"])
        c_train, c_b = float(c_row["C2_train_NMSE_dB"]), float(c_row["C2_B_NMSE_dB"])
        a_joint = bool(a_row["Aend_valid"] and a_train < THRESHOLD_DB and a_b < THRESHOLD_DB)
        c_joint = bool(c_row["C2_valid"] and c_train < THRESHOLD_DB and c_b < THRESHOLD_DB)
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "P": candidate.P,
                "orders": _json_compact(list(candidate.orders)),
                "M1": candidate.M1,
                "M2": candidate.M2,
                "M3": candidate.M3,
                "M5": candidate.M5,
                "memory_profile": _json_compact(list(candidate.memory_profile)),
                "memory_profile_string": candidate.memory_profile_string,
                "max_delay": candidate.max_delay,
                "coefficient_count": candidate.coefficient_count,
                "ridge_used": False,
                "lambda": None,
                "state_id": int(state_id),
                "ilc_A_end": bases.ilc_A_end,
                "B_probe_source": "common_xin_B_probe",
                "Aend_joint_pass": a_joint,
                "C2_joint_pass": c_joint,
                "all_four_pass": bool(a_joint and c_joint),
                "valid": bool(a_row["Aend_valid"] and c_row["C2_valid"]),
                "failure_reason": "; ".join(
                    value
                    for value in (a_row["Aend_failure_reason"], c_row["C2_failure_reason"])
                    if value
                ),
                **a_row,
                **c_row,
            }
        )
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
    return max(1, int(math.floor(0.80 * logical)))


__all__ = [
    "FAILED_STATE_IDS",
    "FORMAL_A_LENGTH",
    "FORMAL_B_LENGTH",
    "FORMAL_C_LENGTH",
    "MEMORY_PROFILES",
    "ORDERS",
    "P5Order2Candidate",
    "REGRESSION_MEMORY_PROFILES",
    "REGRESSION_STATE_IDS",
    "STATE_COUNT",
    "THRESHOLD_DB",
    "build_state_bases",
    "candidate_columns",
    "candidate_grid_frame",
    "evaluate_candidate_on_bases",
    "generate_candidates",
    "load_state_bases",
    "memory_profiles_frame",
    "target_worker_count",
    "validate_candidate_grid",
]
