"""Canonical Envelope75 dictionary for the Hard-20 forward-behavior task.

The dictionary in this module is deliberately task-local.  It adds the even
orders ``p=4, 6, 8`` to the already validated Gate-B48 envelope family while
keeping the old frozen-centered dictionary untouched.  Every column is
constructed from history local to the input vector passed to
``build_envelope_bank``; no samples are borrowed across a segment boundary.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .frozen_centered_dictionary import (
    FrozenCenteredBasis,
    build_frozen_centered_bank,
    build_frozen_centered_dictionary,
    gate_indices,
    validate_frozen_centered_dictionary,
)

DMAX = 2
LINEAR_MEMORY = 2
ENVELOPE_ORDERS = (2, 3, 4, 5, 6, 7, 8, 9)
GATE_B_ORDERS = (2, 3, 5, 7, 9)


@dataclass(frozen=True)
class EnvelopeBasis:
    """One canonical basis term in the Envelope75 family."""

    index: int
    basis_id: str
    formula: str
    family: str
    order: int
    signal_delay: int | None
    envelope_delay: int | None
    mandatory: bool

    @property
    def signature(self) -> tuple[Any, ...]:
        if self.order == 1:
            return ("LINEAR", int(self.signal_delay))
        return ("ENVELOPE", self.order, int(self.signal_delay), int(self.envelope_delay))


def _delay_text(delay: int) -> str:
    return "x[n]" if delay == 0 else f"x[n-{delay}]"


def _formula(order: int, signal_delay: int, envelope_delay: int) -> str:
    return f"{_delay_text(signal_delay)}|{_delay_text(envelope_delay)}|^({order - 1})"


def build_envelope_dictionary(
    orders: Sequence[int] = ENVELOPE_ORDERS,
    dmax: int = DMAX,
    linear_memory: int = LINEAR_MEMORY,
) -> tuple[EnvelopeBasis, ...]:
    """Build the canonical linear-plus-envelope dictionary.

    The default construction is exactly three linear terms plus
    ``8 * 3 * 3 = 72`` nonlinear terms, i.e. Envelope75.  The arguments are
    exposed for focused synthetic tests, but the formal task always uses the
    default values.
    """

    normalized_orders = tuple(int(order) for order in orders)
    if not normalized_orders or len(set(normalized_orders)) != len(normalized_orders):
        raise ValueError("orders must be a nonempty sequence of unique integers")
    if normalized_orders != tuple(sorted(normalized_orders)):
        raise ValueError("orders must be sorted in ascending order")
    if any(order <= 1 for order in normalized_orders):
        raise ValueError("envelope orders must be greater than one")
    if not isinstance(dmax, int) or dmax < 0:
        raise ValueError("dmax must be a nonnegative integer")
    if not isinstance(linear_memory, int) or linear_memory != dmax:
        raise ValueError("linear_memory must equal dmax for the formal dictionary")

    terms: list[EnvelopeBasis] = []
    for delay in range(dmax + 1):
        terms.append(
            EnvelopeBasis(
                index=len(terms),
                basis_id=f"LIN_d{delay}",
                formula=_delay_text(delay),
                family="LINEAR",
                order=1,
                signal_delay=delay,
                envelope_delay=None,
                mandatory=delay <= linear_memory,
            )
        )
    for order in normalized_orders:
        for signal_delay in range(dmax + 1):
            for envelope_delay in range(dmax + 1):
                family = "ALIGNED_ENVELOPE" if signal_delay == envelope_delay else "CROSS_ENVELOPE"
                terms.append(
                    EnvelopeBasis(
                        index=len(terms),
                        basis_id=f"ENV_p{order:02d}_m{signal_delay}_q{envelope_delay}",
                        formula=_formula(order, signal_delay, envelope_delay),
                        family=family,
                        order=order,
                        signal_delay=signal_delay,
                        envelope_delay=envelope_delay,
                        mandatory=False,
                    )
                )
    validate_envelope_dictionary(terms, orders=normalized_orders, dmax=dmax)
    return tuple(terms)


def validate_envelope_dictionary(
    terms: Sequence[EnvelopeBasis],
    *,
    orders: Sequence[int] = ENVELOPE_ORDERS,
    dmax: int = DMAX,
) -> None:
    """Validate canonical ordering, IDs, signatures, and mandatory terms."""

    expected_count = dmax + 1 + len(tuple(orders)) * (dmax + 1) ** 2
    if len(terms) != expected_count:
        raise RuntimeError(f"Envelope dictionary count must be {expected_count}, got {len(terms)}")
    if [term.index for term in terms] != list(range(expected_count)):
        raise RuntimeError("Envelope dictionary indices are not contiguous")
    if len({term.basis_id for term in terms}) != expected_count:
        raise RuntimeError("Envelope basis IDs are not unique")
    if len({term.signature for term in terms}) != expected_count:
        raise RuntimeError("Envelope algebraic signatures are not unique")
    if tuple(term.basis_id for term in terms[: dmax + 1]) != tuple(
        f"LIN_d{delay}" for delay in range(dmax + 1)
    ):
        raise RuntimeError("Envelope linear-term order is not canonical")
    mandatory = [term for term in terms if term.mandatory]
    if len(mandatory) != dmax + 1:
        raise RuntimeError("All formal linear memory terms must be mandatory")
    if any(term.order == 1 and term.envelope_delay is not None for term in terms):
        raise RuntimeError("Linear terms cannot carry an envelope delay")
    for term in terms:
        if term.order == 1:
            if term.signal_delay is None or not 0 <= term.signal_delay <= dmax:
                raise RuntimeError(f"Invalid linear delay in {term.basis_id}")
        else:
            if term.signal_delay is None or term.envelope_delay is None:
                raise RuntimeError(f"Nonlinear term lacks m/q in {term.basis_id}")
            if not (0 <= term.signal_delay <= dmax and 0 <= term.envelope_delay <= dmax):
                raise RuntimeError(f"Invalid nonlinear delay in {term.basis_id}")


def build_envelope_bank(
    x: np.ndarray,
    terms: Sequence[EnvelopeBasis],
    *,
    dmax: int = DMAX,
) -> np.ndarray:
    """Build an Envelope basis bank using only history inside ``x``."""

    signal = np.asarray(x, dtype=np.complex128).reshape(-1)
    if signal.size <= dmax or not np.all(np.isfinite(signal)):
        raise ValueError("x must be a finite complex vector longer than dmax")
    delayed = np.column_stack(
        [signal[dmax - delay : signal.size - delay] for delay in range(dmax + 1)]
    )
    bank = np.empty((signal.size - dmax, len(terms)), dtype=np.complex128)
    for term in terms:
        if term.order == 1:
            assert term.signal_delay is not None
            bank[:, term.index] = delayed[:, term.signal_delay]
        else:
            assert term.signal_delay is not None and term.envelope_delay is not None
            bank[:, term.index] = delayed[:, term.signal_delay] * np.abs(
                delayed[:, term.envelope_delay]
            ) ** (term.order - 1)
    if not np.all(np.isfinite(bank)):
        raise RuntimeError("Envelope basis bank contains NaN/Inf")
    return bank


def _frozen_gate_summary(
    frozen_terms: Sequence[FrozenCenteredBasis],
) -> dict[str, int | bool]:
    counts = {
        gate: len(gate_indices(frozen_terms, gate))
        for gate in ("FROZEN10", "GATE_A", "GATE_B", "GATE_C")
    }
    if counts != {"FROZEN10": 10, "GATE_A": 18, "GATE_B": 48, "GATE_C": 108}:
        raise RuntimeError(f"Frozen gate regression failed: {counts}")
    return counts


def dictionary_gate() -> dict[str, object]:
    """Validate Envelope75 and its exact relationship with historical Gate-B48."""

    envelope_terms = build_envelope_dictionary()
    frozen_terms = build_frozen_centered_dictionary()
    frozen_summary = validate_frozen_centered_dictionary(frozen_terms)
    counts = _frozen_gate_summary(frozen_terms)
    gate_b_indices = gate_indices(frozen_terms, "GATE_B")
    gate_b_ids = tuple(frozen_terms[index].basis_id for index in gate_b_indices)
    envelope_by_id = {term.basis_id: term for term in envelope_terms}
    missing = sorted(set(gate_b_ids) - set(envelope_by_id))
    if missing:
        raise RuntimeError(f"Gate-B basis IDs missing from Envelope75: {missing}")

    rng = np.random.default_rng(20260915)
    x = rng.normal(size=513) + 1j * rng.normal(size=513)
    envelope_indices = tuple(envelope_by_id[basis_id].index for basis_id in gate_b_ids)
    envelope_bank = build_envelope_bank(x, envelope_terms)[:, envelope_indices]
    frozen_bank = build_frozen_centered_bank(x, frozen_terms, gate_b_indices)
    max_error = float(np.max(np.abs(envelope_bank - frozen_bank)))
    if max_error >= 1e-8:
        raise RuntimeError(f"Gate-B numerical subset identity failed: {max_error:.3e}")

    even_ids = tuple(term.basis_id for term in envelope_terms if term.order in {4, 6, 8})
    expected_even = 27
    if len(even_ids) != expected_even:
        raise RuntimeError(f"Envelope75 even-order difference must contain {expected_even} terms")
    return {
        "envelope_count": len(envelope_terms),
        "envelope_unique_ids": len({term.basis_id for term in envelope_terms}),
        "envelope_unique_signatures": len({term.signature for term in envelope_terms}),
        "frozen_gate_counts": counts,
        "frozen_gate_validation": frozen_summary,
        "gate_b_count": len(gate_b_ids),
        "gate_b_subset_of_envelope75": True,
        "envelope_minus_gate_b_count": len(envelope_terms) - len(gate_b_ids),
        "envelope_minus_gate_b_orders": sorted(
            {term.order for term in envelope_terms if term.basis_id not in set(gate_b_ids)}
        ),
        "gate_b_max_identity_error": max_error,
        "pass": True,
    }


__all__ = [
    "DMAX",
    "ENVELOPE_ORDERS",
    "EnvelopeBasis",
    "GATE_B_ORDERS",
    "LINEAR_MEMORY",
    "build_envelope_bank",
    "build_envelope_dictionary",
    "dictionary_gate",
    "validate_envelope_dictionary",
]
