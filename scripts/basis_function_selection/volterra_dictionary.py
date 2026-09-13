"""Deterministic strict complex-baseband Volterra dictionary construction."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import combinations_with_replacement

import numpy as np

from .config import CANDIDATE_COUNT, DMAX, MANDATORY_BASIS_ID, ORDERS


@dataclass(frozen=True)
class VolterraBasis:
    """One canonical complex-baseband Volterra monomial."""

    index: int
    basis_id: str
    order: int
    nonconjugate_delays: tuple[int, ...]
    conjugate_delays: tuple[int, ...]
    formula: str
    mandatory: bool

    @property
    def signature(self) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
        return (self.order, self.nonconjugate_delays, self.conjugate_delays)


def _delay_token(prefix: str, delays: Sequence[int]) -> str:
    return "_".join(f"{prefix}{delay:02d}" for delay in delays)


def _factor(delay: int, *, conjugate: bool) -> str:
    suffix = "*" if conjugate else ""
    return f"x{suffix}[n-{delay}]" if delay else f"x{suffix}[n]"


def _make_basis(
    index: int,
    order: int,
    nonconjugate: tuple[int, ...],
    conjugate: tuple[int, ...],
) -> VolterraBasis:
    if order == 1:
        basis_id = f"V1_u{nonconjugate[0]}"
    else:
        basis_id = f"V{order}_{_delay_token('u', nonconjugate)}_{_delay_token('c', conjugate)}"
    factors = [_factor(delay, conjugate=False) for delay in nonconjugate]
    factors.extend(_factor(delay, conjugate=True) for delay in conjugate)
    return VolterraBasis(
        index=index,
        basis_id=basis_id,
        order=order,
        nonconjugate_delays=nonconjugate,
        conjugate_delays=conjugate,
        formula=" ".join(factors),
        mandatory=basis_id == MANDATORY_BASIS_ID,
    )


def build_dictionary() -> tuple[VolterraBasis, ...]:
    """Generate exactly 3 + 18 + 60 canonical basis functions."""

    terms: list[VolterraBasis] = []
    for delay in range(DMAX + 1):
        terms.append(_make_basis(len(terms), 1, (delay,), ()))
    for nonconjugate in combinations_with_replacement(range(DMAX + 1), 2):
        for conjugate_delay in range(DMAX + 1):
            terms.append(_make_basis(len(terms), 3, nonconjugate, (conjugate_delay,)))
    for nonconjugate in combinations_with_replacement(range(DMAX + 1), 3):
        for conjugate in combinations_with_replacement(range(DMAX + 1), 2):
            terms.append(_make_basis(len(terms), 5, nonconjugate, conjugate))
    validate_dictionary(terms)
    return tuple(terms)


def validate_dictionary(terms: Sequence[VolterraBasis]) -> None:
    """Validate count, orders, canonical signatures, indices, and mandatory term."""

    if len(terms) != CANDIDATE_COUNT:
        raise RuntimeError(f"Volterra candidate count must be {CANDIDATE_COUNT}, got {len(terms)}")
    if [term.index for term in terms] != list(range(CANDIDATE_COUNT)):
        raise RuntimeError("Volterra dictionary indices are not contiguous")
    if len({term.basis_id for term in terms}) != CANDIDATE_COUNT:
        raise RuntimeError("Volterra basis IDs are not unique")
    if len({term.signature for term in terms}) != CANDIDATE_COUNT:
        raise RuntimeError("Volterra algebraic signatures are not unique")
    if {term.order for term in terms} != set(ORDERS):
        raise RuntimeError("Volterra dictionary order set is incorrect")
    mandatory = [term for term in terms if term.mandatory]
    if len(mandatory) != 1 or mandatory[0].basis_id != MANDATORY_BASIS_ID:
        raise RuntimeError("Exactly x[n] must be mandatory")
    for term in terms:
        delays = term.nonconjugate_delays + term.conjugate_delays
        if not delays or min(delays) < 0 or max(delays) > DMAX:
            raise RuntimeError(f"Invalid delay in {term.basis_id}")
        if tuple(sorted(term.nonconjugate_delays)) != term.nonconjugate_delays:
            raise RuntimeError(f"Nonconjugate delays are not canonical in {term.basis_id}")
        if tuple(sorted(term.conjugate_delays)) != term.conjugate_delays:
            raise RuntimeError(f"Conjugate delays are not canonical in {term.basis_id}")


def build_basis_bank(x: np.ndarray, terms: Sequence[VolterraBasis]) -> np.ndarray:
    """Build a complex128 basis bank using only history inside ``x``."""

    signal = np.asarray(x, dtype=np.complex128).reshape(-1)
    if signal.size <= DMAX or not np.all(np.isfinite(signal)):
        raise ValueError("x must be a finite complex vector longer than dmax")
    delayed = np.column_stack(
        [signal[DMAX - delay : signal.size - delay] for delay in range(DMAX + 1)]
    )
    bank = np.empty((signal.size - DMAX, len(terms)), dtype=np.complex128)
    for term in terms:
        value = np.ones(signal.size - DMAX, dtype=np.complex128)
        for delay in term.nonconjugate_delays:
            value *= delayed[:, delay]
        for delay in term.conjugate_delays:
            value *= np.conj(delayed[:, delay])
        bank[:, term.index] = value
    if not np.all(np.isfinite(bank)):
        raise RuntimeError("Volterra basis bank contains NaN/Inf")
    return bank


def dictionary_gate() -> dict[str, object]:
    """Run deterministic algebraic and numerical identity gates."""

    terms = build_dictionary()
    rng = np.random.default_rng(20260912)
    x = rng.normal(size=512) + 1j * rng.normal(size=512)
    bank = build_basis_bank(x, terms)
    by_signature = {term.signature: term for term in terms}
    diagonal_v3 = by_signature[(3, (0, 0), (0,))]
    diagonal_v5 = by_signature[(5, (0, 0, 0), (0, 0))]
    x_valid = x[DMAX:]
    error_v3 = float(np.max(np.abs(bank[:, diagonal_v3.index] - x_valid * np.abs(x_valid) ** 2)))
    error_v5 = float(np.max(np.abs(bank[:, diagonal_v5.index] - x_valid * np.abs(x_valid) ** 4)))
    tolerance = 1e-12
    if error_v3 > tolerance or error_v5 > tolerance:
        raise RuntimeError(f"Volterra identity gate failed: V3={error_v3}, V5={error_v5}")
    return {
        "candidate_count": len(terms),
        "unique_ids": len({term.basis_id for term in terms}),
        "unique_signatures": len({term.signature for term in terms}),
        "v3_identity_max_abs_error": error_v3,
        "v5_identity_max_abs_error": error_v5,
        "pass": True,
    }


def support_ids(terms: Sequence[VolterraBasis], indices: Iterable[int]) -> tuple[str, ...]:
    """Map ordered integer indices to basis IDs."""

    return tuple(terms[int(index)].basis_id for index in indices)


__all__ = [
    "VolterraBasis",
    "build_basis_bank",
    "build_dictionary",
    "dictionary_gate",
    "support_ids",
    "validate_dictionary",
]
