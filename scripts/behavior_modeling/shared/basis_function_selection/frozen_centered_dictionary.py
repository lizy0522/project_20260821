"""Nested Frozen-centered Gate A/B/C basis dictionaries."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from .config import DMAX
from .reference_gate import FROZEN_BASES
from .volterra_dictionary import build_dictionary as build_strict81


@dataclass(frozen=True)
class FrozenCenteredBasis:
    """One canonical basis with nested-gate membership metadata."""

    index: int
    basis_id: str
    formula: str
    family: str
    order: int
    signal_delay: int | None
    envelope_delay: int | None
    nonconjugate_delays: tuple[int, ...]
    conjugate_delays: tuple[int, ...]
    algebraic_signature: tuple[Any, ...]
    in_frozen10: bool
    in_gate_a: bool
    in_gate_b: bool
    in_gate_c: bool


def _poly_signature(
    nonconjugate: Sequence[int],
    conjugate: Sequence[int],
) -> tuple[Any, ...]:
    return ("POLY", tuple(sorted(nonconjugate)), tuple(sorted(conjugate)))


def _envelope_signature(order: int, signal_delay: int, envelope_delay: int) -> tuple[Any, ...]:
    if order == 2:
        return ("ABS_ENVELOPE", order, signal_delay, envelope_delay)
    repeats = (order - 1) // 2
    return _poly_signature(
        (signal_delay, *([envelope_delay] * repeats)),
        [envelope_delay] * repeats,
    )


def _delay_text(delay: int) -> str:
    return "x[n]" if delay == 0 else f"x[n-{delay}]"


def _envelope_formula(order: int, signal_delay: int, envelope_delay: int) -> str:
    carrier = _delay_text(signal_delay)
    envelope = _delay_text(envelope_delay)
    return f"{carrier}|{envelope}|^{order - 1}"


def _frozen_signatures() -> set[tuple[Any, ...]]:
    return {
        _envelope_signature(basis.order, basis.memory_delay, basis.memory_delay)
        if basis.order > 1
        else _poly_signature((basis.memory_delay,), ())
        for basis in FROZEN_BASES
    }


def build_frozen_centered_dictionary() -> tuple[FrozenCenteredBasis, ...]:
    """Generate the canonical 108-basis Gate-C union with nested memberships."""

    records: dict[tuple[Any, ...], dict[str, Any]] = {}
    frozen_signatures = _frozen_signatures()

    def add_record(
        *,
        basis_id: str,
        formula: str,
        family: str,
        order: int,
        signal_delay: int | None,
        envelope_delay: int | None,
        nonconjugate: tuple[int, ...],
        conjugate: tuple[int, ...],
        signature: tuple[Any, ...],
        gate_a: bool,
        gate_b: bool,
        gate_c: bool,
    ) -> None:
        if signature in records:
            existing = records[signature]
            existing["in_gate_a"] = bool(existing["in_gate_a"] or gate_a)
            existing["in_gate_b"] = bool(existing["in_gate_b"] or gate_b)
            existing["in_gate_c"] = bool(existing["in_gate_c"] or gate_c)
            return
        records[signature] = {
            "basis_id": basis_id,
            "formula": formula,
            "family": family,
            "order": order,
            "signal_delay": signal_delay,
            "envelope_delay": envelope_delay,
            "nonconjugate_delays": nonconjugate,
            "conjugate_delays": conjugate,
            "algebraic_signature": signature,
            "in_frozen10": signature in frozen_signatures,
            "in_gate_a": gate_a,
            "in_gate_b": gate_b,
            "in_gate_c": gate_c,
        }

    for delay in range(DMAX + 1):
        signature = _poly_signature((delay,), ())
        add_record(
            basis_id=f"LIN_d{delay}",
            formula=_delay_text(delay),
            family="LINEAR",
            order=1,
            signal_delay=delay,
            envelope_delay=None,
            nonconjugate=(delay,),
            conjugate=(),
            signature=signature,
            gate_a=True,
            gate_b=True,
            gate_c=True,
        )

    for order in (2, 3, 5, 7, 9):
        for signal_delay in range(DMAX + 1):
            for envelope_delay in range(DMAX + 1):
                signature = _envelope_signature(order, signal_delay, envelope_delay)
                aligned = signal_delay == envelope_delay
                add_record(
                    basis_id=f"ENV_p{order:02d}_m{signal_delay}_q{envelope_delay}",
                    formula=_envelope_formula(order, signal_delay, envelope_delay),
                    family="ALIGNED_ENVELOPE" if aligned else "CROSS_ENVELOPE",
                    order=order,
                    signal_delay=signal_delay,
                    envelope_delay=envelope_delay,
                    nonconjugate=signature[1] if signature[0] == "POLY" else (),
                    conjugate=signature[2] if signature[0] == "POLY" else (),
                    signature=signature,
                    gate_a=aligned,
                    gate_b=True,
                    gate_c=True,
                )

    for strict in build_strict81():
        signature = _poly_signature(strict.nonconjugate_delays, strict.conjugate_delays)
        add_record(
            basis_id=strict.basis_id,
            formula=strict.formula,
            family="LINEAR" if strict.order == 1 else f"STRICT_VOLTERRA_{strict.order}",
            order=strict.order,
            signal_delay=strict.nonconjugate_delays[0] if strict.order == 1 else None,
            envelope_delay=None,
            nonconjugate=strict.nonconjugate_delays,
            conjugate=strict.conjugate_delays,
            signature=signature,
            gate_a=False,
            gate_b=False,
            gate_c=True,
        )

    terms = tuple(
        FrozenCenteredBasis(index=index, **record) for index, record in enumerate(records.values())
    )
    validate_frozen_centered_dictionary(terms)
    return terms


def gate_indices(
    terms: Sequence[FrozenCenteredBasis],
    gate_id: str,
) -> tuple[int, ...]:
    """Return canonical indices for Frozen10, Gate A, Gate B, or Gate C."""

    attribute = {
        "FROZEN10": "in_frozen10",
        "GATE_A": "in_gate_a",
        "GATE_B": "in_gate_b",
        "GATE_C": "in_gate_c",
    }.get(gate_id)
    if attribute is None:
        raise KeyError(f"Unknown gate_id={gate_id!r}")
    return tuple(term.index for term in terms if bool(getattr(term, attribute)))


def validate_frozen_centered_dictionary(
    terms: Sequence[FrozenCenteredBasis],
) -> dict[str, object]:
    """Validate counts, IDs, signatures, and exact nested memberships."""

    if len(terms) != 108:
        raise RuntimeError(f"Gate C must contain 108 unique bases, got {len(terms)}")
    if len({term.basis_id for term in terms}) != 108:
        raise RuntimeError("Frozen-centered basis IDs are not unique")
    if len({term.algebraic_signature for term in terms}) != 108:
        raise RuntimeError("Frozen-centered algebraic signatures are not unique")
    counts = {
        gate: len(gate_indices(terms, gate)) for gate in ("FROZEN10", "GATE_A", "GATE_B", "GATE_C")
    }
    if counts != {"FROZEN10": 10, "GATE_A": 18, "GATE_B": 48, "GATE_C": 108}:
        raise RuntimeError(f"Nested gate counts are incorrect: {counts}")
    sets = {gate: set(gate_indices(terms, gate)) for gate in counts}
    if not (sets["FROZEN10"] < sets["GATE_A"] < sets["GATE_B"] < sets["GATE_C"]):
        raise RuntimeError("Frozen10/Gate A/Gate B/Gate C are not strictly nested")
    if (
        max(
            delay
            for term in terms
            for delay in (term.nonconjugate_delays + term.conjugate_delays)
            if term.algebraic_signature[0] == "POLY"
        )
        != DMAX
    ):
        raise RuntimeError("Frozen-centered dictionary max polynomial delay is not 2")
    return {"counts": counts, "strictly_nested": True, "pass": True}


def build_frozen_centered_bank(
    x: np.ndarray,
    terms: Sequence[FrozenCenteredBasis],
    indices: Sequence[int],
) -> np.ndarray:
    """Build selected canonical columns using history local to ``x``."""

    signal = np.asarray(x, dtype=np.complex128).reshape(-1)
    if signal.size <= DMAX or not np.all(np.isfinite(signal)):
        raise ValueError("x must be a finite complex vector longer than dmax")
    delayed = np.column_stack(
        [signal[DMAX - delay : signal.size - delay] for delay in range(DMAX + 1)]
    )
    selected = tuple(int(index) for index in indices)
    bank = np.empty((signal.size - DMAX, len(selected)), dtype=np.complex128)
    for column, index in enumerate(selected):
        term = terms[index]
        if term.algebraic_signature[0] == "ABS_ENVELOPE":
            assert term.signal_delay is not None and term.envelope_delay is not None
            bank[:, column] = delayed[:, term.signal_delay] * np.abs(
                delayed[:, term.envelope_delay]
            ) ** (term.order - 1)
        else:
            value = np.ones(signal.size - DMAX, dtype=np.complex128)
            for delay in term.nonconjugate_delays:
                value *= delayed[:, delay]
            for delay in term.conjugate_delays:
                value *= np.conj(delayed[:, delay])
            bank[:, column] = value
    if not np.all(np.isfinite(bank)):
        raise RuntimeError("Frozen-centered basis bank contains NaN/Inf")
    return bank


def dictionary_numerical_gate() -> dict[str, object]:
    """Run deterministic membership and representative identity checks."""

    terms = build_frozen_centered_dictionary()
    rng = np.random.default_rng(20260912)
    x = rng.normal(size=512) + 1j * rng.normal(size=512)
    all_indices = gate_indices(terms, "GATE_C")
    bank = build_frozen_centered_bank(x, terms, all_indices)
    signatures = [term.algebraic_signature for term in terms]
    if len(signatures) != len(set(signatures)):
        raise RuntimeError("Numerical dictionary gate found algebraic duplicates")
    by_signature = {term.algebraic_signature: term.index for term in terms}
    checks = []
    for order in (3, 5):
        repeats = (order - 1) // 2
        signature = _poly_signature((0, *([1] * repeats)), [1] * repeats)
        index = by_signature[signature]
        expected = x[DMAX:] * np.abs(x[DMAX - 1 : -1]) ** (order - 1)
        error = float(np.max(np.abs(bank[:, index] - expected)))
        checks.append({"order": order, "max_abs_error": error})
        if error > 1e-12:
            raise RuntimeError(f"Gate dictionary identity failed for order {order}")
    return {
        **validate_frozen_centered_dictionary(terms),
        "identity_checks": checks,
        "pass": True,
    }


__all__ = [
    "FrozenCenteredBasis",
    "build_frozen_centered_bank",
    "build_frozen_centered_dictionary",
    "dictionary_numerical_gate",
    "gate_indices",
    "validate_frozen_centered_dictionary",
]
