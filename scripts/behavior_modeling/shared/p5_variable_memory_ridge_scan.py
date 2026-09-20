"""P=5 odd-order variable-memory MP plus Ridge candidate scan.

The module is intentionally separate from the previous finite OLS failed-14
experiment.  It reuses only the canonical-state cache/basis construction and
adds a mathematically explicit augmented least-squares Ridge path with the
frozen objective ``1/N*||y-Phi theta||^2 + lambda*||theta||^2``.
"""

# Imports intentionally follow the BLAS-thread environment setup below.
# ruff: noqa: E402,I001,E501

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
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

from behavior_modeling.shared.evaluation import calculate_nmse
from behavior_modeling.shared.odd_order_variable_memory_scan import (
    FAILED_STATE_IDS,
    FORMAL_A_LENGTH,
    FORMAL_B_LENGTH,
    FORMAL_C_LENGTH,
    MEMORY_MAX,
    STATE_COUNT,
    StateBases,
    candidate_columns,
)

P = 5
ORDERS = (1, 3, 5)
MEMORY_PROFILES = (
    (1, 1, 1),
    (2, 1, 1),
    (2, 2, 1),
    (2, 2, 2),
    (3, 1, 1),
    (3, 2, 1),
    (3, 2, 2),
    (3, 3, 1),
    (3, 3, 2),
    (3, 3, 3),
    (4, 1, 1),
    (4, 2, 1),
    (4, 2, 2),
    (4, 3, 1),
    (4, 3, 2),
    (4, 3, 3),
    (4, 4, 1),
    (4, 4, 2),
    (4, 4, 3),
    (4, 4, 4),
)
RIDGE_LAMBDAS = (0.0, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4)
THRESHOLD_DB = -40.0
REGRESSION_STATE_IDS = (187, 195, 340, 354)
REGRESSION_MEMORY_PROFILES = ((1, 1, 1), (3, 2, 1), (4, 4, 1), (4, 4, 4))
REGRESSION_LAMBDAS = (0.0, 1e-10, 1e-8, 1e-5)


@dataclass(frozen=True)
class P5RidgeCandidate:
    """One candidate defined by a memory profile and one Ridge strength."""

    candidate_id: int
    P: int
    orders: tuple[int, int, int]
    memory_profile: tuple[int, int, int]
    M1: int
    M3: int
    M5: int
    max_delay: int
    coefficient_count: int
    ridge_lambda: float
    ridge_used: bool
    memory_profile_string: str
    profile_index: int
    lambda_index: int

    @property
    def memory_definition(self) -> dict[int, int]:
        return {1: self.M1, 3: self.M3, 5: self.M5}

    @property
    def basis_terms(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (order, delay)
            for order, depth in zip(self.orders, self.memory_profile, strict=True)
            for delay in range(depth)
        )


def _json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def generate_candidates() -> tuple[P5RidgeCandidate, ...]:
    if len(MEMORY_PROFILES) != 20 or len(RIDGE_LAMBDAS) != 10:
        raise RuntimeError("frozen memory profile or lambda count changed")
    candidates: list[P5RidgeCandidate] = []
    candidate_id = 0
    for profile_index, profile in enumerate(MEMORY_PROFILES):
        for lambda_index, ridge_lambda in enumerate(RIDGE_LAMBDAS):
            candidates.append(
                P5RidgeCandidate(
                    candidate_id=candidate_id,
                    P=P,
                    orders=ORDERS,
                    memory_profile=profile,
                    M1=profile[0],
                    M3=profile[1],
                    M5=profile[2],
                    max_delay=max(profile) - 1,
                    coefficient_count=sum(profile),
                    ridge_lambda=float(ridge_lambda),
                    ridge_used=bool(ridge_lambda > 0.0),
                    memory_profile_string=f"P5_M[{','.join(str(v) for v in profile)}]",
                    profile_index=profile_index,
                    lambda_index=lambda_index,
                )
            )
            candidate_id += 1
    validate_candidate_grid(candidates)
    return tuple(candidates)


def validate_candidate_grid(candidates: Sequence[P5RidgeCandidate]) -> None:
    if len(candidates) != 200:
        raise ValueError(f"candidate_count must be 200, got {len(candidates)}")
    if [candidate.candidate_id for candidate in candidates] != list(range(200)):
        raise ValueError("candidate IDs must be 0...199")
    if len(set(candidate.memory_profile for candidate in candidates)) != 20:
        raise ValueError("memory profile count must be 20")
    if len(set(candidate.ridge_lambda for candidate in candidates)) != 10:
        raise ValueError("lambda count must be 10")
    for candidate in candidates:
        if candidate.P != 5 or candidate.orders != ORDERS:
            raise ValueError("all candidates must use P=5 and orders=[1,3,5]")
        if candidate.memory_profile not in MEMORY_PROFILES:
            raise ValueError("candidate has an unknown memory profile")
        if not (candidate.M1 >= candidate.M3 >= candidate.M5):
            raise ValueError("memory profile is not monotonic non-increasing")
        if min(candidate.memory_profile) < 1 or max(candidate.memory_profile) > MEMORY_MAX:
            raise ValueError("memory depth outside 1...4")
        if candidate.coefficient_count != candidate.M1 + candidate.M3 + candidate.M5:
            raise ValueError("K mismatch")
        if candidate.max_delay != candidate.M1 - 1:
            raise ValueError("max_delay mismatch")
        if candidate.coefficient_count != len(candidate.basis_terms):
            raise ValueError("basis term count mismatch")
        if any(order % 2 == 0 for order, _ in candidate.basis_terms):
            raise ValueError("even-order basis term detected")


def candidate_grid_frame(candidates: Sequence[P5RidgeCandidate]) -> pd.DataFrame:
    validate_candidate_grid(candidates)
    rows = [
        {
            "candidate_id": candidate.candidate_id,
            "P": candidate.P,
            "orders": _json_compact(list(candidate.orders)),
            "M1": candidate.M1,
            "M3": candidate.M3,
            "M5": candidate.M5,
            "memory_profile": _json_compact(list(candidate.memory_profile)),
            "memory_profile_string": candidate.memory_profile_string,
            "memory_definition": _json_compact(candidate.memory_definition),
            "max_delay": candidate.max_delay,
            "coefficient_count": candidate.coefficient_count,
            "lambda": candidate.ridge_lambda,
            "ridge_used": candidate.ridge_used,
        }
        for candidate in candidates
    ]
    return pd.DataFrame(rows)


def memory_profiles_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "profile_index": index,
                "memory_profile": _json_compact(list(profile)),
                "memory_profile_string": f"P5_M[{','.join(str(v) for v in profile)}]",
                "M1": profile[0],
                "M3": profile[1],
                "M5": profile[2],
                "max_delay": max(profile) - 1,
                "coefficient_count": sum(profile),
                "is_uniform_memory": len(set(profile)) == 1,
            }
            for index, profile in enumerate(MEMORY_PROFILES)
        ]
    )


def ridge_grid_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "lambda_index": index,
                "lambda": value,
                "ridge_used": bool(value > 0),
                "label": "OLS" if value == 0 else f"{value:.0e}",
            }
            for index, value in enumerate(RIDGE_LAMBDAS)
        ]
    )


def _fit_ols_solution(phi: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, float]:
    theta, _, rank, singular = np.linalg.lstsq(phi, y, rcond=None)
    singular = np.asarray(singular, dtype=np.float64)
    if int(rank) != phi.shape[1] or singular.size != phi.shape[1] or singular[-1] <= 0:
        raise RuntimeError(f"OLS reference rank deficiency: {rank}/{phi.shape[1]}")
    residual_norm = float(np.linalg.norm(y - phi @ theta))
    return np.asarray(theta, dtype=np.complex128), singular, int(rank), residual_norm


def build_ols_reference(bases_by_state: Mapping[int, StateBases]) -> dict[str, np.ndarray]:
    """Build one OLS solution per profile/state/side for lambda=0 and ratios."""

    max_k = max(sum(profile) for profile in MEMORY_PROFILES)
    n_profiles = len(MEMORY_PROFILES)
    n_states = len(FAILED_STATE_IDS)
    theta_a = np.zeros((n_profiles, n_states, max_k), dtype=np.complex128)
    theta_c = np.zeros_like(theta_a)
    sigma_a = np.full((n_profiles, n_states, max_k), np.nan, dtype=np.float64)
    sigma_c = np.full_like(sigma_a, np.nan)
    rank_a = np.zeros((n_profiles, n_states), dtype=np.int64)
    rank_c = np.zeros_like(rank_a)
    residual_a = np.full((n_profiles, n_states), np.nan, dtype=np.float64)
    residual_c = np.full_like(residual_a, np.nan)
    for profile_index, profile in enumerate(MEMORY_PROFILES):
        candidate = P5RidgeCandidate(
            candidate_id=profile_index * len(RIDGE_LAMBDAS),
            P=P,
            orders=ORDERS,
            memory_profile=profile,
            M1=profile[0],
            M3=profile[1],
            M5=profile[2],
            max_delay=max(profile) - 1,
            coefficient_count=sum(profile),
            ridge_lambda=0.0,
            ridge_used=False,
            memory_profile_string=f"P5_M[{','.join(str(v) for v in profile)}]",
            profile_index=profile_index,
            lambda_index=0,
        )
        columns = candidate_columns(candidate)
        for state_index, state_id in enumerate(FAILED_STATE_IDS):
            bases = bases_by_state[int(state_id)]
            delay = candidate.max_delay
            phi_a = bases.aend_A_phi[delay:, :][:, columns]
            phi_c = bases.c2_C_phi[delay:, :][:, columns]
            y_a = bases.aend_A_y[delay:]
            y_c = bases.c2_C_y[delay:]
            a_theta, a_sigma, a_rank, a_residual = _fit_ols_solution(phi_a, y_a)
            c_theta, c_sigma, c_rank, c_residual = _fit_ols_solution(phi_c, y_c)
            K = candidate.coefficient_count
            theta_a[profile_index, state_index, :K] = a_theta
            theta_c[profile_index, state_index, :K] = c_theta
            sigma_a[profile_index, state_index, :K] = a_sigma
            sigma_c[profile_index, state_index, :K] = c_sigma
            rank_a[profile_index, state_index] = a_rank
            rank_c[profile_index, state_index] = c_rank
            residual_a[profile_index, state_index] = a_residual
            residual_c[profile_index, state_index] = c_residual
    return {
        "state_ids": np.asarray(FAILED_STATE_IDS, dtype=np.int64),
        "profile_indices": np.arange(n_profiles, dtype=np.int64),
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
        required = {
            "state_ids",
            "profile_indices",
            "theta_Aend",
            "theta_C2",
            "singular_Aend",
            "singular_C2",
            "rank_Aend",
            "rank_C2",
            "residual_Aend",
            "residual_C2",
            "theta_norm_Aend",
            "theta_norm_C2",
        }
        if not required.issubset(data.files):
            raise ValueError(f"OLS reference missing fields: {sorted(required - set(data.files))}")
        reference = {name: np.asarray(data[name]) for name in required}
    if not np.array_equal(reference["state_ids"], np.asarray(FAILED_STATE_IDS, dtype=np.int64)):
        raise ValueError("OLS reference state IDs mismatch")
    if reference["theta_Aend"].shape[:2] != (20, STATE_COUNT):
        raise ValueError("OLS reference shape mismatch")
    return reference


def _fit_side(
    bases: StateBases,
    candidate: P5RidgeCandidate,
    side: str,
    reference: Mapping[str, np.ndarray],
    state_index: int,
) -> tuple[dict[str, Any], np.ndarray | None]:
    delay = candidate.max_delay
    columns = candidate_columns(candidate)
    if side == "Aend":
        phi_train_full, y_train_full = bases.aend_A_phi, bases.aend_A_y
        y_b_full = bases.aend_B_y
        theta_ref = reference["theta_Aend"][
            candidate.profile_index, state_index, : candidate.coefficient_count
        ]
        singular_ref = reference["singular_Aend"][
            candidate.profile_index, state_index, : candidate.coefficient_count
        ]
        rank_ref = int(reference["rank_Aend"][candidate.profile_index, state_index])
        residual_ref = float(reference["residual_Aend"][candidate.profile_index, state_index])
        norm_ref = float(reference["theta_norm_Aend"][candidate.profile_index, state_index])
    else:
        phi_train_full, y_train_full = bases.c2_C_phi, bases.c2_C_y
        y_b_full = bases.c2_B_y
        theta_ref = reference["theta_C2"][
            candidate.profile_index, state_index, : candidate.coefficient_count
        ]
        singular_ref = reference["singular_C2"][
            candidate.profile_index, state_index, : candidate.coefficient_count
        ]
        rank_ref = int(reference["rank_C2"][candidate.profile_index, state_index])
        residual_ref = float(reference["residual_C2"][candidate.profile_index, state_index])
        norm_ref = float(reference["theta_norm_C2"][candidate.profile_index, state_index])
    phi_train = np.asarray(phi_train_full[delay:, :][:, columns], dtype=np.complex128)
    y_train = np.asarray(y_train_full[delay:], dtype=np.complex128)
    phi_b = np.asarray(bases.common_b_phi[delay:, :][:, columns], dtype=np.complex128)
    y_b = np.asarray(y_b_full[delay:], dtype=np.complex128)
    prefix = f"{side}_"
    row: dict[str, Any] = {
        f"{prefix}train_NMSE_dB": np.nan,
        f"{prefix}B_NMSE_dB": np.nan,
        f"{prefix}generalization_gap_dB": np.nan,
        f"{prefix}matrix_rank": rank_ref,
        f"{prefix}column_count": candidate.coefficient_count,
        f"{prefix}rank_ratio": float(rank_ref / candidate.coefficient_count),
        f"{prefix}sigma_max": float(singular_ref[0]),
        f"{prefix}sigma_min": float(singular_ref[-1]),
        f"{prefix}condition_number": float(singular_ref[0] / singular_ref[-1]),
        f"{prefix}condition_number_augmented": np.nan,
        f"{prefix}theta_l2_norm": np.nan,
        f"{prefix}theta_max_abs": np.nan,
        f"{prefix}theta_norm_ratio_vs_OLS": np.nan,
        f"{prefix}residual_norm": np.nan,
        f"{prefix}regularization_penalty": np.nan,
        f"{prefix}n_train_samples": int(phi_train.shape[0]),
        f"{prefix}n_B_samples": int(phi_b.shape[0]),
        f"{prefix}finite": False,
        f"{prefix}valid": False,
        f"{prefix}failure_reason": "",
    }
    try:
        if phi_train.shape[0] <= candidate.coefficient_count:
            raise ValueError("valid training rows are not greater than coefficient count")
        if phi_b.shape[0] != y_b.shape[0] or phi_train.shape[0] != y_train.shape[0]:
            raise ValueError("basis/target support length mismatch")
        if not np.all(np.isfinite(phi_train)) or not np.all(np.isfinite(phi_b)):
            raise ValueError("basis contains NaN/Inf")
        if candidate.ridge_lambda == 0.0:
            theta = theta_ref
            singular_aug = singular_ref
            residual_norm = residual_ref
        else:
            N = phi_train.shape[0]
            identity = np.eye(candidate.coefficient_count, dtype=np.complex128)
            augmented_phi = np.vstack((phi_train, np.sqrt(N * candidate.ridge_lambda) * identity))
            augmented_y = np.concatenate(
                (y_train, np.zeros(candidate.coefficient_count, dtype=np.complex128))
            )
            theta, _, rank_aug, singular_aug = np.linalg.lstsq(
                augmented_phi, augmented_y, rcond=None
            )
            if int(rank_aug) != candidate.coefficient_count:
                raise ValueError(
                    f"augmented rank deficient {rank_aug}/{candidate.coefficient_count}"
                )
            residual_norm = float(np.linalg.norm(y_train - phi_train @ theta))
        theta = np.asarray(theta, dtype=np.complex128)
        singular_aug = np.asarray(singular_aug, dtype=np.float64)
        if not np.all(np.isfinite(theta)) or singular_aug[-1] <= 0:
            raise ValueError("theta or augmented singular values are invalid")
        train_value = float(calculate_nmse(y_train, phi_train @ theta))
        b_value = float(calculate_nmse(y_b, phi_b @ theta))
        if not np.isfinite(train_value) or not np.isfinite(b_value):
            raise ValueError("NMSE is not finite")
        norm_theta = float(np.linalg.norm(theta))
        penalty = float(candidate.ridge_lambda * norm_theta**2)
        row.update(
            {
                f"{prefix}train_NMSE_dB": train_value,
                f"{prefix}B_NMSE_dB": b_value,
                f"{prefix}generalization_gap_dB": b_value - train_value,
                f"{prefix}condition_number_augmented": float(singular_aug[0] / singular_aug[-1]),
                f"{prefix}theta_l2_norm": norm_theta,
                f"{prefix}theta_max_abs": float(np.max(np.abs(theta))),
                f"{prefix}theta_norm_ratio_vs_OLS": float(norm_theta / norm_ref),
                f"{prefix}residual_norm": residual_norm,
                f"{prefix}regularization_penalty": penalty,
                f"{prefix}finite": True,
                f"{prefix}valid": True,
            }
        )
        return row, theta
    except Exception as exc:
        row[f"{prefix}failure_reason"] = f"{type(exc).__name__}: {exc}"
        return row, None


def evaluate_candidate_on_bases(
    candidate: P5RidgeCandidate,
    bases_by_state: Mapping[int, StateBases],
    reference: Mapping[str, np.ndarray],
    *,
    return_theta: bool = True,
) -> dict[str, Any]:
    state_index = {int(value): index for index, value in enumerate(reference["state_ids"])}
    rows: list[dict[str, Any]] = []
    theta_a: dict[int, np.ndarray] = {}
    theta_c: dict[int, np.ndarray] = {}
    for state_id in sorted(bases_by_state):
        index = state_index[int(state_id)]
        bases = bases_by_state[int(state_id)]
        a_row, a_theta = _fit_side(bases, candidate, "Aend", reference, index)
        c_row, c_theta = _fit_side(bases, candidate, "C2", reference, index)
        a_train = float(a_row["Aend_train_NMSE_dB"])
        a_b = float(a_row["Aend_B_NMSE_dB"])
        c_train = float(c_row["C2_train_NMSE_dB"])
        c_b = float(c_row["C2_B_NMSE_dB"])
        a_joint = bool(a_row["Aend_valid"] and a_train < THRESHOLD_DB and a_b < THRESHOLD_DB)
        c_joint = bool(c_row["C2_valid"] and c_train < THRESHOLD_DB and c_b < THRESHOLD_DB)
        rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "P": candidate.P,
                "orders": _json_compact(list(candidate.orders)),
                "M1": candidate.M1,
                "M3": candidate.M3,
                "M5": candidate.M5,
                "memory_profile": _json_compact(list(candidate.memory_profile)),
                "memory_profile_string": candidate.memory_profile_string,
                "max_delay": candidate.max_delay,
                "coefficient_count": candidate.coefficient_count,
                "lambda": candidate.ridge_lambda,
                "ridge_used": candidate.ridge_used,
                "profile_index": candidate.profile_index,
                "lambda_index": candidate.lambda_index,
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
    return max(1, int(math.floor(0.90 * logical)))


__all__ = [
    "ORDERS",
    "FAILED_STATE_IDS",
    "FORMAL_A_LENGTH",
    "FORMAL_B_LENGTH",
    "FORMAL_C_LENGTH",
    "MEMORY_PROFILES",
    "P5RidgeCandidate",
    "REGRESSION_LAMBDAS",
    "REGRESSION_MEMORY_PROFILES",
    "REGRESSION_STATE_IDS",
    "RIDGE_LAMBDAS",
    "STATE_COUNT",
    "THRESHOLD_DB",
    "build_ols_reference",
    "candidate_grid_frame",
    "candidate_columns",
    "evaluate_candidate_on_bases",
    "generate_candidates",
    "load_ols_reference",
    "memory_profiles_frame",
    "ridge_grid_frame",
    "save_ols_reference",
    "target_worker_count",
    "validate_candidate_grid",
]
