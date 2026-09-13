"""Frozen MP plus sparse GMP cross-memory basis utilities.

This module owns only deterministic basis construction and complex Ridge
fitting.  It does not load data, choose terms, or inspect B-segment targets.
The runner controls the A/C-only discovery boundary and opens B evaluation
only after a selected structure has been persisted.
"""

# ruff: noqa: E402,E501

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
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

from behavior_model.basis import build_mp_basis

MP_ORDERS = (1, 2, 3, 5, 7, 9)
MP_MEMORY = {1: 3, 2: 2, 3: 2, 5: 1, 7: 1, 9: 1}
MP_MAX_DELAY = 2
GMP_ORDERS = (2, 3, 5)
GMP_DELAYS = (0, 1, 2)
GMP_MAX_DELAY = 2
COMMON_MAX_DELAY = 2
RIDGE_LAMBDA = 1e-8
FROZEN_MP_K = 10


@dataclass(frozen=True, order=True)
class GMPTerm:
    """One cross-memory generalized-memory-polynomial term."""

    term_index: int
    term_id: str
    p: int
    m: int
    q: int

    @property
    def term_tuple(self) -> tuple[int, int, int]:
        return self.p, self.m, self.q

    @property
    def expression(self) -> str:
        return f"x[n-{self.m}]|x[n-{self.q}]|^{self.p - 1}"


@dataclass(frozen=True)
class RidgeDiagnostics:
    ridge_lambda: float
    n_samples: int
    column_count: int
    rank_phi: int
    rank_augmented: int
    sigma_max_phi: float
    sigma_min_phi: float
    condition_number_phi: float
    sigma_max_augmented: float
    sigma_min_augmented: float
    condition_number_augmented: float
    theta_l2_norm: float
    residual_norm: float
    ridge_penalty: float


def _json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def generate_gmp_dictionary() -> tuple[GMPTerm, ...]:
    terms: list[GMPTerm] = []
    index = 1
    for p in GMP_ORDERS:
        for m in GMP_DELAYS:
            for q in GMP_DELAYS:
                if m == q:
                    continue
                terms.append(GMPTerm(index, f"GMP{index:03d}", p, m, q))
                index += 1
    validate_gmp_dictionary(terms)
    return tuple(terms)


def validate_gmp_dictionary(terms: Sequence[GMPTerm], *, require_full: bool = True) -> None:
    if require_full and len(terms) != 18:
        raise ValueError(f"GMP dictionary must contain 18 terms, got {len(terms)}")
    ids = [term.term_id for term in terms]
    if require_full and ids != [f"GMP{index:03d}" for index in range(1, 19)]:
        raise ValueError("GMP term IDs are not deterministic GMP001...GMP018")
    tuples = [term.term_tuple for term in terms]
    if len(set(tuples)) != len(terms):
        raise ValueError("GMP dictionary contains duplicate terms")
    if any(term.p not in GMP_ORDERS for term in terms):
        raise ValueError("GMP dictionary contains a forbidden order")
    if any(term.m not in GMP_DELAYS or term.q not in GMP_DELAYS or term.m == term.q for term in terms):
        raise ValueError("GMP dictionary contains an invalid delay pair")
    mp_terms = {(order, delay) for order, depth in MP_MEMORY.items() for delay in range(depth)}
    if any((term.p, term.m) in mp_terms and term.m == term.q for term in terms):
        raise ValueError("aligned MP term leaked into the GMP dictionary")


def dictionary_frame(terms: Sequence[GMPTerm] | None = None):
    import pandas as pd

    terms = generate_gmp_dictionary() if terms is None else tuple(terms)
    validate_gmp_dictionary(terms, require_full=False)
    return pd.DataFrame(
        [
            {
                "term_index": term.term_index,
                "term_id": term.term_id,
                "p": term.p,
                "m": term.m,
                "q": term.q,
                "expression": term.expression,
                "cross_memory": True,
                "aligned_MP_equivalent": False,
            }
            for term in terms
        ]
    )


def build_frozen_mp_basis(x: np.ndarray) -> np.ndarray:
    """Build the ten frozen MP columns on the common dmax=2 support."""

    return build_mp_basis(np.asarray(x, dtype=np.complex128), MP_ORDERS, MP_MEMORY)


def build_gmp_basis(x: np.ndarray, terms: Sequence[GMPTerm] | None = None) -> np.ndarray:
    """Build GMP columns ``x[n-m] * abs(x[n-q])**(p-1)`` on ``x[2:]``."""

    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    if x.size <= COMMON_MAX_DELAY:
        raise ValueError("x is shorter than the common GMP support")
    terms = generate_gmp_dictionary() if terms is None else tuple(terms)
    validate_gmp_dictionary(terms, require_full=False)
    valid_start = COMMON_MAX_DELAY
    columns: list[np.ndarray] = []
    for term in terms:
        carrier = x[valid_start - term.m : x.size - term.m]
        envelope = np.abs(x[valid_start - term.q : x.size - term.q]) ** (term.p - 1)
        columns.append(carrier * envelope)
    return np.column_stack(columns).astype(np.complex128, copy=False)


def build_combined_basis(x: np.ndarray, selected_terms: Sequence[GMPTerm] | None = None) -> np.ndarray:
    """Concatenate frozen MP columns and the requested GMP columns."""

    mp = build_frozen_mp_basis(x)
    selected_terms = tuple(selected_terms or ())
    if not selected_terms:
        return mp
    dictionary = {term.term_id: term for term in generate_gmp_dictionary()}
    selected = tuple(dictionary[term.term_id] if isinstance(term, GMPTerm) else dictionary[str(term)] for term in selected_terms)
    return np.column_stack((mp, build_gmp_basis(x, selected)))


def fit_ridge(phi: np.ndarray, y: np.ndarray, ridge_lambda: float = RIDGE_LAMBDA) -> tuple[np.ndarray, RidgeDiagnostics]:
    """Fit ``||e||²/N + lambda||theta||²`` with augmented complex LS."""

    phi = np.asarray(phi, dtype=np.complex128)
    y = np.asarray(y, dtype=np.complex128).reshape(-1)
    if phi.ndim != 2 or y.ndim != 1 or phi.shape[0] != y.size or phi.shape[0] <= phi.shape[1]:
        raise ValueError("invalid Ridge system dimensions")
    if not np.all(np.isfinite(phi)) or not np.all(np.isfinite(y)):
        raise ValueError("Ridge system contains non-finite values")
    ridge_lambda = float(ridge_lambda)
    if not np.isfinite(ridge_lambda) or ridge_lambda < 0:
        raise ValueError("ridge_lambda must be finite and non-negative")
    theta_ols, _, rank_phi, singular_phi = np.linalg.lstsq(phi, y, rcond=None)
    singular_phi = np.asarray(singular_phi, dtype=float)
    if int(rank_phi) != phi.shape[1] or singular_phi[-1] <= 0:
        raise ValueError(f"rank deficient design matrix {rank_phi}/{phi.shape[1]}")
    if ridge_lambda == 0.0:
        theta = theta_ols
        singular_aug = singular_phi
        rank_aug = int(rank_phi)
    else:
        augmented_phi = np.vstack((phi, np.sqrt(phi.shape[0] * ridge_lambda) * np.eye(phi.shape[1], dtype=np.complex128)))
        augmented_y = np.concatenate((y, np.zeros(phi.shape[1], dtype=np.complex128)))
        theta, _, rank_aug, singular_aug = np.linalg.lstsq(augmented_phi, augmented_y, rcond=None)
        singular_aug = np.asarray(singular_aug, dtype=float)
    theta = np.asarray(theta, dtype=np.complex128)
    residual = float(np.linalg.norm(y - phi @ theta))
    diagnostics = RidgeDiagnostics(
        ridge_lambda=ridge_lambda,
        n_samples=phi.shape[0],
        column_count=phi.shape[1],
        rank_phi=int(rank_phi),
        rank_augmented=int(rank_aug),
        sigma_max_phi=float(singular_phi[0]),
        sigma_min_phi=float(singular_phi[-1]),
        condition_number_phi=float(singular_phi[0] / singular_phi[-1]),
        sigma_max_augmented=float(singular_aug[0]),
        sigma_min_augmented=float(singular_aug[-1]),
        condition_number_augmented=float(singular_aug[0] / singular_aug[-1]),
        theta_l2_norm=float(np.linalg.norm(theta)),
        residual_norm=residual,
        ridge_penalty=float(ridge_lambda * np.linalg.norm(theta) ** 2),
    )
    return theta, diagnostics


def residual_correlation(phi_column: np.ndarray, residual: np.ndarray) -> float:
    phi_column = np.asarray(phi_column, dtype=np.complex128).reshape(-1)
    residual = np.asarray(residual, dtype=np.complex128).reshape(-1)
    denominator = float(np.linalg.norm(phi_column) * np.linalg.norm(residual))
    if denominator <= 0:
        return 0.0
    return float(abs(np.vdot(phi_column, residual)) / denominator)


def residualized_column(phi_column: np.ndarray, current_phi: np.ndarray) -> np.ndarray:
    """Return the candidate column after projection away from current basis."""

    phi_column = np.asarray(phi_column, dtype=np.complex128).reshape(-1)
    current_phi = np.asarray(current_phi, dtype=np.complex128)
    coefficients, _, _, _ = np.linalg.lstsq(current_phi, phi_column, rcond=None)
    return phi_column - current_phi @ coefficients


def term_by_id(term_id: str) -> GMPTerm:
    terms = generate_gmp_dictionary()
    return next(term for term in terms if term.term_id == str(term_id))


def terms_to_json(terms: Iterable[GMPTerm]) -> list[dict[str, Any]]:
    return [
        {"term_index": term.term_index, "term_id": term.term_id, "p": term.p, "m": term.m, "q": term.q, "expression": term.expression}
        for term in terms
    ]


__all__ = [
    "COMMON_MAX_DELAY",
    "FROZEN_MP_K",
    "GMP_DELAYS",
    "GMP_MAX_DELAY",
    "GMP_ORDERS",
    "GMPTerm",
    "MP_MEMORY",
    "MP_MAX_DELAY",
    "MP_ORDERS",
    "RIDGE_LAMBDA",
    "RidgeDiagnostics",
    "build_combined_basis",
    "build_frozen_mp_basis",
    "build_gmp_basis",
    "dictionary_frame",
    "fit_ridge",
    "generate_gmp_dictionary",
    "residual_correlation",
    "residualized_column",
    "term_by_id",
    "terms_to_json",
    "validate_gmp_dictionary",
]
