"""Frozen-model-neighborhood memory/Ridge scan for the failed-14 states.

All candidates keep the frozen orders ``[1, 2, 3, 5, 7, 9]``.  Only the
low-order memory allocation and one shared Ridge lambda vary.  The canonical
pair preparation is reused from the existing behavior-model helper and is
materialised into a task-local read-only cache so Candidate-level workers can
reuse it without touching raw data.
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
from behavior_model.odd_order_variable_memory_scan import FAILED_STATE_IDS
from behavior_model.statewise_adaptive_model_search import (
    FORMAL_A_LENGTH,
    FORMAL_B_LENGTH,
    FORMAL_C_LENGTH,
    prepare_state_pairs,
)

STATE_COUNT = 14
ALL_STATE_COUNT = 425
ORDERS = (1, 2, 3, 5, 7, 9)
MEMORY_MAX = 4
THRESHOLD_DB = -40.0
STRUCTURES = (
    ("S0", (3, 2, 2, 1, 1, 1), "frozen structure"),
    ("S1", (4, 2, 2, 1, 1, 1), "deeper first-order memory"),
    ("S2", (4, 2, 1, 1, 1, 1), "deeper first-order, shorter third-order"),
    ("S3", (4, 1, 1, 1, 1, 1), "low-order long-memory compact form"),
    ("S4", (3, 2, 1, 1, 1, 1), "reduced third-order memory"),
    ("S5", (3, 1, 1, 1, 1, 1), "more compact neighborhood"),
    ("S6", (4, 3, 2, 1, 1, 1), "increased second-order memory"),
)
# Compatibility aliases used by the other behavior-model scan modules.  The
# present experiment owns the seven explicit structures, but exposing these
# names keeps imports and manifests straightforward.
MEMORY_PROFILES = {structure_id: profile for structure_id, profile, _ in STRUCTURES}
LAMBDA_GRID = (0.0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3)
FROZEN_STRUCTURE_ID = "S0"
FROZEN_MEMORY = (3, 2, 2, 1, 1, 1)
FROZEN_LAMBDA = 1e-8


@dataclass(frozen=True)
class NeighborhoodCandidate:
    """One candidate in the seven-structure, nine-lambda grid."""

    candidate_id: int
    structure_id: str
    structure_description: str
    orders: tuple[int, ...]
    memory_profile: tuple[int, ...]
    M1: int
    M2: int
    M3: int
    M5: int
    M7: int
    M9: int
    max_delay: int
    coefficient_count: int
    lambda_value: float
    ridge_used: bool
    is_frozen_baseline: bool
    lambda_index: int

    @property
    def memory_definition(self) -> dict[int, int]:
        return {order: depth for order, depth in zip(self.orders, self.memory_profile, strict=True)}

    @property
    def memory_profile_string(self) -> str:
        return f"{self.structure_id}_M[{','.join(str(v) for v in self.memory_profile)}]"

    @property
    def basis_terms(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (order, delay)
            for order, depth in zip(self.orders, self.memory_profile, strict=True)
            for delay in range(depth)
        )


@dataclass(frozen=True)
class PreparedState:
    """Candidate-independent arrays from the existing canonical pipeline."""

    state_id: int
    ilc_A_end: int
    aend_A_input: np.ndarray
    aend_A_output: np.ndarray
    aend_B_input: np.ndarray
    aend_B_output: np.ndarray
    c2_C_input: np.ndarray
    c2_C_output: np.ndarray
    c2_B_input: np.ndarray
    c2_B_output: np.ndarray
    rough_delay_Aend: int
    fraction_delay_Aend: float
    rough_delay_C2: int
    fraction_delay_C2: float


@dataclass(frozen=True)
class StateBases:
    state_id: int
    ilc_A_end: int
    aend_A_phi: np.ndarray
    aend_A_y: np.ndarray
    aend_B_phi: np.ndarray
    aend_B_y: np.ndarray
    c2_C_phi: np.ndarray
    c2_C_y: np.ndarray
    c2_B_phi: np.ndarray
    c2_B_y: np.ndarray


FULL_TERMS = tuple((order, delay) for order in ORDERS for delay in range(MEMORY_MAX))
FULL_TERM_TO_COLUMN = {term: index for index, term in enumerate(FULL_TERMS)}


def _json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def generate_candidates() -> tuple[NeighborhoodCandidate, ...]:
    candidates: list[NeighborhoodCandidate] = []
    candidate_id = 0
    for structure_id, profile, description in STRUCTURES:
        for lambda_index, lambda_value in enumerate(LAMBDA_GRID):
            candidates.append(
                NeighborhoodCandidate(
                    candidate_id=candidate_id,
                    structure_id=structure_id,
                    structure_description=description,
                    orders=ORDERS,
                    memory_profile=profile,
                    M1=profile[0],
                    M2=profile[1],
                    M3=profile[2],
                    M5=profile[3],
                    M7=profile[4],
                    M9=profile[5],
                    max_delay=profile[0] - 1,
                    coefficient_count=sum(profile),
                    lambda_value=float(lambda_value),
                    ridge_used=bool(lambda_value > 0),
                    is_frozen_baseline=structure_id == FROZEN_STRUCTURE_ID and lambda_value == FROZEN_LAMBDA,
                    lambda_index=lambda_index,
                )
            )
            candidate_id += 1
    validate_candidate_grid(candidates)
    return tuple(candidates)


def validate_candidate_grid(candidates: Sequence[NeighborhoodCandidate]) -> None:
    if len(STRUCTURES) != 7 or len(LAMBDA_GRID) != 9 or len(candidates) != 63:
        raise ValueError("structure/lambda/candidate count must be 7/9/63")
    if [candidate.candidate_id for candidate in candidates] != list(range(63)):
        raise ValueError("candidate IDs must be deterministic 0...62")
    if sum(candidate.is_frozen_baseline for candidate in candidates) != 1:
        raise ValueError("frozen baseline must occur exactly once")
    for candidate in candidates:
        if candidate.orders != ORDERS:
            raise ValueError("orders are not frozen to [1,2,3,5,7,9]")
        if not all(1 <= depth <= MEMORY_MAX for depth in candidate.memory_profile):
            raise ValueError("memory depth outside 1...4")
        if not all(left >= right for left, right in zip(candidate.memory_profile, candidate.memory_profile[1:])):
            raise ValueError("memory profile is not non-increasing")
        if candidate.memory_profile[3:] != (1, 1, 1):
            raise ValueError("M5/M7/M9 must remain one tap")
        if candidate.coefficient_count != sum(candidate.memory_profile):
            raise ValueError("coefficient count mismatch")
        if candidate.max_delay != candidate.M1 - 1:
            raise ValueError("max_delay mismatch")
    frozen = next(candidate for candidate in candidates if candidate.is_frozen_baseline)
    if frozen.memory_profile != FROZEN_MEMORY or frozen.lambda_value != FROZEN_LAMBDA:
        raise ValueError("frozen baseline definition mismatch")


def candidate_grid_frame(candidates: Sequence[NeighborhoodCandidate]) -> pd.DataFrame:
    validate_candidate_grid(candidates)
    return pd.DataFrame(
        [
            {
                "candidate_id": candidate.candidate_id,
                "structure_id": candidate.structure_id,
                "structure_description": candidate.structure_description,
                "P_max": 9,
                "orders": _json_compact(list(candidate.orders)),
                "M1": candidate.M1,
                "M2": candidate.M2,
                "M3": candidate.M3,
                "M5": candidate.M5,
                "M7": candidate.M7,
                "M9": candidate.M9,
                "memory_profile": _json_compact(list(candidate.memory_profile)),
                "memory_profile_string": candidate.memory_profile_string,
                "max_delay": candidate.max_delay,
                "coefficient_count": candidate.coefficient_count,
                "lambda": candidate.lambda_value,
                "ridge_used": candidate.ridge_used,
                "is_frozen_baseline": candidate.is_frozen_baseline,
            }
            for candidate in candidates
        ]
    )


def structures_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "structure_id": structure_id,
                "memory_profile": _json_compact(list(profile)),
                "memory_profile_string": f"{structure_id}_M[{','.join(str(v) for v in profile)}]",
                "M1": profile[0],
                "M2": profile[1],
                "M3": profile[2],
                "M5": profile[3],
                "M7": profile[4],
                "M9": profile[5],
                "max_delay": profile[0] - 1,
                "coefficient_count": sum(profile),
                "description": description,
            }
            for structure_id, profile, description in STRUCTURES
        ]
    )


def memory_profiles_frame() -> pd.DataFrame:
    """Return the seven local memory structures as a table."""

    return structures_frame()


def _prepared_from_pairs(state_id: int, prepared: Any) -> PreparedState:
    """Convert canonical pair objects to the task cache representation."""

    a = prepared.a_end
    c = prepared.c2
    return PreparedState(
        state_id=int(state_id),
        ilc_A_end=int(prepared.ilc_A_end),
        aend_A_input=np.asarray(a["A"].input, dtype=np.complex128),
        aend_A_output=np.asarray(a["A"].output, dtype=np.complex128),
        aend_B_input=np.asarray(a["B"].input, dtype=np.complex128),
        aend_B_output=np.asarray(a["B"].output, dtype=np.complex128),
        c2_C_input=np.asarray(c["C"].input, dtype=np.complex128),
        c2_C_output=np.asarray(c["C"].output, dtype=np.complex128),
        c2_B_input=np.asarray(c["B"].input, dtype=np.complex128),
        c2_B_output=np.asarray(c["B"].output, dtype=np.complex128),
        rough_delay_Aend=int(a.rough_delay),
        fraction_delay_Aend=float(a.fraction_delay),
        rough_delay_C2=int(c.rough_delay),
        fraction_delay_C2=float(c.fraction_delay),
    )


def prepare_state(state_id: int) -> PreparedState:
    state_id = int(state_id)
    if state_id not in FAILED_STATE_IDS:
        raise ValueError(f"state {state_id} is not in failed14 selection scope")
    return _prepared_from_pairs(state_id, prepare_state_pairs(state_id))


def prepare_state_any(state_id: int) -> PreparedState:
    """Prepare any of the 425 Scenario-2 states through the canonical path."""

    state_id = int(state_id)
    if not 0 <= state_id < ALL_STATE_COUNT:
        raise ValueError(f"state_id must be in 0...{ALL_STATE_COUNT - 1}: {state_id!r}")
    return _prepared_from_pairs(state_id, prepare_state_pairs(state_id))


def save_prepared_state(prepared: PreparedState, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        state_id=np.asarray(prepared.state_id, dtype=np.int64),
        ilc_A_end=np.asarray(prepared.ilc_A_end, dtype=np.int64),
        aend_A_input=prepared.aend_A_input,
        aend_A_output=prepared.aend_A_output,
        aend_B_input=prepared.aend_B_input,
        aend_B_output=prepared.aend_B_output,
        c2_C_input=prepared.c2_C_input,
        c2_C_output=prepared.c2_C_output,
        c2_B_input=prepared.c2_B_input,
        c2_B_output=prepared.c2_B_output,
        rough_delay_Aend=np.asarray(prepared.rough_delay_Aend, dtype=np.int64),
        fraction_delay_Aend=np.asarray(prepared.fraction_delay_Aend, dtype=np.float64),
        rough_delay_C2=np.asarray(prepared.rough_delay_C2, dtype=np.int64),
        fraction_delay_C2=np.asarray(prepared.fraction_delay_C2, dtype=np.float64),
    )


def load_prepared_state(path: Path) -> PreparedState:
    with np.load(path, allow_pickle=False) as data:
        state_id = int(np.asarray(data["state_id"]).reshape(-1)[0])
        return PreparedState(
            state_id=state_id,
            ilc_A_end=int(np.asarray(data["ilc_A_end"]).reshape(-1)[0]),
            aend_A_input=np.asarray(data["aend_A_input"], dtype=np.complex128),
            aend_A_output=np.asarray(data["aend_A_output"], dtype=np.complex128),
            aend_B_input=np.asarray(data["aend_B_input"], dtype=np.complex128),
            aend_B_output=np.asarray(data["aend_B_output"], dtype=np.complex128),
            c2_C_input=np.asarray(data["c2_C_input"], dtype=np.complex128),
            c2_C_output=np.asarray(data["c2_C_output"], dtype=np.complex128),
            c2_B_input=np.asarray(data["c2_B_input"], dtype=np.complex128),
            c2_B_output=np.asarray(data["c2_B_output"], dtype=np.complex128),
            rough_delay_Aend=int(np.asarray(data["rough_delay_Aend"]).reshape(-1)[0]),
            fraction_delay_Aend=float(np.asarray(data["fraction_delay_Aend"]).reshape(-1)[0]),
            rough_delay_C2=int(np.asarray(data["rough_delay_C2"]).reshape(-1)[0]),
            fraction_delay_C2=float(np.asarray(data["fraction_delay_C2"]).reshape(-1)[0]),
        )


def _full_basis(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.complex128)
    phi = np.full((x.size, len(FULL_TERMS)), np.nan + 0j, dtype=np.complex128)
    for column, (order, delay) in enumerate(FULL_TERMS):
        delayed = x[: x.size - delay]
        phi[delay:, column] = delayed * np.abs(delayed) ** (order - 1)
    return phi


def build_state_bases(prepared: PreparedState) -> StateBases:
    return StateBases(
        state_id=prepared.state_id,
        ilc_A_end=prepared.ilc_A_end,
        aend_A_phi=_full_basis(prepared.aend_A_input),
        aend_A_y=prepared.aend_A_output,
        aend_B_phi=_full_basis(prepared.aend_B_input),
        aend_B_y=prepared.aend_B_output,
        c2_C_phi=_full_basis(prepared.c2_C_input),
        c2_C_y=prepared.c2_C_output,
        c2_B_phi=_full_basis(prepared.c2_B_input),
        c2_B_y=prepared.c2_B_output,
    )


def load_state_bases(cache_dir: Path, state_ids: Iterable[int]) -> dict[int, StateBases]:
    result: dict[int, StateBases] = {}
    for state_id in state_ids:
        prepared = load_prepared_state(Path(cache_dir) / f"state_{int(state_id):03d}.npz")
        result[prepared.state_id] = build_state_bases(prepared)
    return result


def candidate_columns(candidate: NeighborhoodCandidate) -> tuple[int, ...]:
    columns = tuple(FULL_TERM_TO_COLUMN[term] for term in candidate.basis_terms)
    if len(columns) != candidate.coefficient_count:
        raise RuntimeError("basis column count mismatch")
    return columns


def _fit_ols(phi: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, float]:
    theta, _, rank, singular = np.linalg.lstsq(phi, y, rcond=None)
    singular = np.asarray(singular, dtype=np.float64)
    if int(rank) != phi.shape[1] or singular[-1] <= 0:
        raise RuntimeError(f"OLS rank deficient {rank}/{phi.shape[1]}")
    return np.asarray(theta, dtype=np.complex128), singular, int(rank), float(np.linalg.norm(y - phi @ theta))


def build_ols_reference(bases_by_state: Mapping[int, StateBases]) -> dict[str, np.ndarray]:
    """Fit OLS references for every structure and supplied state.

    The state dimension is intentionally derived from ``bases_by_state`` so
    the same routine can serve the failed-14 selection scan and the later
    all-425 post-selection validation without changing the fitting semantics.
    """

    state_ids = tuple(sorted(int(state_id) for state_id in bases_by_state))
    if not state_ids:
        raise ValueError("at least one state is required for OLS references")
    max_k = max(sum(profile) for _, profile, _ in STRUCTURES)
    n_structures = len(STRUCTURES)
    n_states = len(state_ids)
    theta_a = np.zeros((n_structures, n_states, max_k), dtype=np.complex128)
    theta_c = np.zeros_like(theta_a)
    sigma_a = np.full((n_structures, STATE_COUNT, max_k), np.nan, dtype=np.float64)
    sigma_c = np.full_like(sigma_a, np.nan)
    rank_a = np.zeros((n_structures, STATE_COUNT), dtype=np.int64)
    rank_c = np.zeros_like(rank_a)
    residual_a = np.full((n_structures, STATE_COUNT), np.nan, dtype=np.float64)
    residual_c = np.full_like(residual_a, np.nan)
    for structure_index, (_, profile, _) in enumerate(STRUCTURES):
        candidate = next(c for c in generate_candidates() if c.structure_id == STRUCTURES[structure_index][0] and c.lambda_value == 0.0)
        columns = candidate_columns(candidate)
        for state_index, state_id in enumerate(state_ids):
            bases = bases_by_state[int(state_id)]
            delay = candidate.max_delay
            a_theta, a_sigma, a_rank, a_residual = _fit_ols(bases.aend_A_phi[delay:, :][:, columns], bases.aend_A_y[delay:])
            c_theta, c_sigma, c_rank, c_residual = _fit_ols(bases.c2_C_phi[delay:, :][:, columns], bases.c2_C_y[delay:])
            K = candidate.coefficient_count
            theta_a[structure_index, state_index, :K] = a_theta
            theta_c[structure_index, state_index, :K] = c_theta
            sigma_a[structure_index, state_index, :K] = a_sigma
            sigma_c[structure_index, state_index, :K] = c_sigma
            rank_a[structure_index, state_index] = a_rank
            rank_c[structure_index, state_index] = c_rank
            residual_a[structure_index, state_index] = a_residual
            residual_c[structure_index, state_index] = c_residual
    return {
        "state_ids": np.asarray(state_ids, dtype=np.int64),
        "theta_Aend": theta_a,
        "theta_C2": theta_c,
        "singular_Aend": sigma_a,
        "singular_C2": sigma_c,
        "rank_Aend": rank_a,
        "rank_C2": rank_c,
        "residual_Aend": residual_a,
        "residual_C2": residual_c,
        "theta_norm_Aend": np.linalg.norm(theta_a, axis=2),
        "theta_norm_C2": np.linalg.norm(theta_c, axis=2),
    }


def save_ols_reference(path: Path, reference: Mapping[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **reference)


def load_ols_reference(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]) for name in data.files}


def _fit_side(
    bases: StateBases,
    candidate: NeighborhoodCandidate,
    side: str,
    reference: Mapping[str, np.ndarray],
    state_index: int,
    support_delay: int | None = None,
) -> tuple[dict[str, Any], np.ndarray | None]:
    delay = candidate.max_delay if support_delay is None else int(support_delay)
    if delay < candidate.max_delay:
        raise ValueError("support delay cannot precede candidate max_delay")
    columns = candidate_columns(candidate)
    structure_index = next(index for index, item in enumerate(STRUCTURES) if item[0] == candidate.structure_id)
    if side == "Aend":
        train_phi_full, train_y_full, b_phi_full, b_y_full = bases.aend_A_phi, bases.aend_A_y, bases.aend_B_phi, bases.aend_B_y
        theta_ref = reference["theta_Aend"][structure_index, state_index, : candidate.coefficient_count]
        sigma_ref = reference["singular_Aend"][structure_index, state_index, : candidate.coefficient_count]
        rank_ref = int(reference["rank_Aend"][structure_index, state_index])
        residual_ref = float(reference["residual_Aend"][structure_index, state_index])
        norm_ref = float(reference["theta_norm_Aend"][structure_index, state_index])
    else:
        train_phi_full, train_y_full, b_phi_full, b_y_full = bases.c2_C_phi, bases.c2_C_y, bases.c2_B_phi, bases.c2_B_y
        theta_ref = reference["theta_C2"][structure_index, state_index, : candidate.coefficient_count]
        sigma_ref = reference["singular_C2"][structure_index, state_index, : candidate.coefficient_count]
        rank_ref = int(reference["rank_C2"][structure_index, state_index])
        residual_ref = float(reference["residual_C2"][structure_index, state_index])
        norm_ref = float(reference["theta_norm_C2"][structure_index, state_index])
    phi = np.asarray(train_phi_full[delay:, :][:, columns], dtype=np.complex128)
    y = np.asarray(train_y_full[delay:], dtype=np.complex128)
    phi_b = np.asarray(b_phi_full[delay:, :][:, columns], dtype=np.complex128)
    y_b = np.asarray(b_y_full[delay:], dtype=np.complex128)
    prefix = f"{side}_"
    row: dict[str, Any] = {
        f"{prefix}train_NMSE_dB": np.nan,
        f"{prefix}B_NMSE_dB": np.nan,
        f"{prefix}generalization_gap_dB": np.nan,
        f"{prefix}rank": rank_ref,
        f"{prefix}column_count": candidate.coefficient_count,
        f"{prefix}rank_ratio": float(rank_ref / candidate.coefficient_count),
        f"{prefix}sigma_max": float(sigma_ref[0]),
        f"{prefix}sigma_min": float(sigma_ref[-1]),
        f"{prefix}condition_number_Phi": float(sigma_ref[0] / sigma_ref[-1]),
        f"{prefix}condition_number_augmented": np.nan,
        f"{prefix}theta_l2_norm": np.nan,
        f"{prefix}theta_max_abs": np.nan,
        f"{prefix}residual_norm": np.nan,
        f"{prefix}ridge_penalty": 0.0,
        f"{prefix}theta_norm_ratio_vs_same_structure_OLS": np.nan,
        f"{prefix}n_train_samples": int(phi.shape[0]),
        f"{prefix}n_B_samples": int(phi_b.shape[0]),
        f"{prefix}finite": False,
        f"{prefix}valid": False,
        f"{prefix}failure_reason": "",
    }
    try:
        if phi.shape[0] <= candidate.coefficient_count or phi.shape[0] != y.size or phi_b.shape[0] != y_b.size:
            raise ValueError("basis/target support mismatch")
        if not np.all(np.isfinite(phi)) or not np.all(np.isfinite(phi_b)):
            raise ValueError("basis contains NaN/Inf")
        if candidate.lambda_value == 0.0 and delay == candidate.max_delay:
            # Reuse the cached native-support OLS solution for the formal
            # scan.  A common-support sensitivity run may request a larger
            # prefix; in that case it must refit on the requested rows.
            theta = theta_ref
            sigma_aug = sigma_ref
            residual = residual_ref
        elif candidate.lambda_value == 0.0:
            theta, sigma_aug, rank_native, residual = _fit_ols(phi, y)
            if int(rank_native) != candidate.coefficient_count:
                raise ValueError(f"OLS rank deficient {rank_native}/{candidate.coefficient_count}")
            row.update(
                {
                    f"{prefix}rank": int(rank_native),
                    f"{prefix}rank_ratio": float(rank_native / candidate.coefficient_count),
                    f"{prefix}sigma_max": float(sigma_aug[0]),
                    f"{prefix}sigma_min": float(sigma_aug[-1]),
                    f"{prefix}condition_number_Phi": float(sigma_aug[0] / sigma_aug[-1]),
                }
            )
        else:
            identity = np.eye(candidate.coefficient_count, dtype=np.complex128)
            augmented_phi = np.vstack((phi, np.sqrt(phi.shape[0] * candidate.lambda_value) * identity))
            augmented_y = np.concatenate((y, np.zeros(candidate.coefficient_count, dtype=np.complex128)))
            theta, _, rank_aug, sigma_aug = np.linalg.lstsq(augmented_phi, augmented_y, rcond=None)
            if int(rank_aug) != candidate.coefficient_count:
                raise ValueError(f"augmented rank deficient {rank_aug}/{candidate.coefficient_count}")
            residual = float(np.linalg.norm(y - phi @ theta))
        theta = np.asarray(theta, dtype=np.complex128)
        sigma_aug = np.asarray(sigma_aug, dtype=np.float64)
        train_value = float(calculate_nmse(y, phi @ theta))
        b_value = float(calculate_nmse(y_b, phi_b @ theta))
        if not np.all(np.isfinite(theta)) or not np.isfinite(train_value) or not np.isfinite(b_value):
            raise ValueError("nonfinite fit or NMSE")
        norm_theta = float(np.linalg.norm(theta))
        row.update(
            {
                f"{prefix}train_NMSE_dB": train_value,
                f"{prefix}B_NMSE_dB": b_value,
                f"{prefix}generalization_gap_dB": b_value - train_value,
                f"{prefix}condition_number_augmented": float(sigma_aug[0] / sigma_aug[-1]),
                f"{prefix}theta_l2_norm": norm_theta,
                f"{prefix}theta_max_abs": float(np.max(np.abs(theta))),
                f"{prefix}residual_norm": residual,
                f"{prefix}ridge_penalty": float(candidate.lambda_value * norm_theta**2),
                f"{prefix}theta_norm_ratio_vs_same_structure_OLS": float(norm_theta / norm_ref),
                f"{prefix}finite": True,
                f"{prefix}valid": True,
            }
        )
        return row, theta
    except Exception as exc:
        row[f"{prefix}failure_reason"] = f"{type(exc).__name__}: {exc}"
        return row, None


def evaluate_candidate_on_bases(
    candidate: NeighborhoodCandidate,
    bases_by_state: Mapping[int, StateBases],
    reference: Mapping[str, np.ndarray],
    *,
    return_theta: bool = True,
    support_delay: int | None = None,
) -> dict[str, Any]:
    state_index = {int(value): index for index, value in enumerate(reference["state_ids"])}
    rows: list[dict[str, Any]] = []
    theta_a: dict[int, np.ndarray] = {}
    theta_c: dict[int, np.ndarray] = {}
    for state_id in sorted(bases_by_state):
        a_row, a_theta = _fit_side(bases_by_state[state_id], candidate, "Aend", reference, state_index[state_id], support_delay)
        c_row, c_theta = _fit_side(bases_by_state[state_id], candidate, "C2", reference, state_index[state_id], support_delay)
        a_train, a_b = float(a_row["Aend_train_NMSE_dB"]), float(a_row["Aend_B_NMSE_dB"])
        c_train, c_b = float(c_row["C2_train_NMSE_dB"]), float(c_row["C2_B_NMSE_dB"])
        a_joint = bool(a_row["Aend_valid"] and a_train < THRESHOLD_DB and a_b < THRESHOLD_DB)
        c_joint = bool(c_row["C2_valid"] and c_train < THRESHOLD_DB and c_b < THRESHOLD_DB)
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "structure_id": candidate.structure_id,
                "orders": _json_compact(list(candidate.orders)),
                "M1": candidate.M1,
                "M2": candidate.M2,
                "M3": candidate.M3,
                "M5": candidate.M5,
                "M7": candidate.M7,
                "M9": candidate.M9,
                "memory_profile": _json_compact(list(candidate.memory_profile)),
                "memory_profile_string": candidate.memory_profile_string,
                "max_delay": candidate.max_delay,
                "coefficient_count": candidate.coefficient_count,
                "lambda": candidate.lambda_value,
                "ridge_used": candidate.ridge_used,
                "is_frozen_baseline": candidate.is_frozen_baseline,
                "state_id": int(state_id),
                "ilc_A_end": bases_by_state[state_id].ilc_A_end,
                "B_probe_source": "canonical_pair_B",
                "support_delay_used": candidate.max_delay if support_delay is None else int(support_delay),
                "Aend_joint_pass": a_joint,
                "C2_joint_pass": c_joint,
                "all_four_pass": bool(a_joint and c_joint),
                "valid": bool(a_row["Aend_valid"] and c_row["C2_valid"]),
                "failure_reason": "; ".join(value for value in (a_row["Aend_failure_reason"], c_row["C2_failure_reason"]) if value),
                **a_row,
                **c_row,
            }
        )
        if return_theta:
            if a_theta is not None:
                theta_a[int(state_id)] = a_theta
            if c_theta is not None:
                theta_c[int(state_id)] = c_theta
    return {"candidate_id": candidate.candidate_id, "rows": rows, "theta_a": theta_a, "theta_c": theta_c}


def target_worker_count(logical_cpu_count: int | None = None) -> int:
    logical = int(logical_cpu_count or os.cpu_count() or 1)
    return max(1, int(math.floor(0.80 * logical)))


__all__ = [
    "ALL_STATE_COUNT",
    "FAILED_STATE_IDS",
    "FORMAL_A_LENGTH",
    "FORMAL_B_LENGTH",
    "FORMAL_C_LENGTH",
    "FROZEN_LAMBDA",
    "FROZEN_MEMORY",
    "FROZEN_STRUCTURE_ID",
    "LAMBDA_GRID",
    "MEMORY_PROFILES",
    "NeighborhoodCandidate",
    "ORDERS",
    "PreparedState",
    "STATE_COUNT",
    "StateBases",
    "STRUCTURES",
    "THRESHOLD_DB",
    "build_ols_reference",
    "build_state_bases",
    "candidate_columns",
    "candidate_grid_frame",
    "evaluate_candidate_on_bases",
    "generate_candidates",
    "load_ols_reference",
    "load_prepared_state",
    "load_state_bases",
    "memory_profiles_frame",
    "prepare_state",
    "prepare_state_any",
    "save_ols_reference",
    "save_prepared_state",
    "structures_frame",
    "target_worker_count",
    "validate_candidate_grid",
]
