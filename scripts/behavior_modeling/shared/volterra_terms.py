"""Lazy, canonical extension of the project's strict complex-baseband Volterra terms.

The existing frozen Hard-20 dictionary remains unchanged.  This extension uses
the same convention: for odd order p, (p+1)/2 nonconjugate x factors and
(p-1)/2 conjugate x factors, with permutations within each group equivalent.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from itertools import combinations_with_replacement

import numpy as np

P_MAX = 11
M_MAX = 4  # Inclusive delay index, matching the existing dictionary's DMAX.


@dataclass(frozen=True, slots=True)
class VolterraTermSpec:
    basis_id: str
    nonlinear_order: int
    nonconjugate_delays: tuple[int, ...]
    conjugate_delays: tuple[int, ...]

    def __post_init__(self) -> None:
        u, c = self.nonconjugate_delays, self.conjugate_delays
        order = len(u) + len(c)
        if order not in (1, 3, 5, 7, 9, 11) or len(u) != (order + 1) // 2:
            raise ValueError("invalid complex-baseband Volterra order")
        if tuple(sorted(u)) != u or tuple(sorted(c)) != c:
            raise ValueError("noncanonical delay permutation")
        if not all(isinstance(delay, int) and not isinstance(delay, bool) for delay in u + c):
            raise TypeError("delays must be integers")
        if any(delay < 0 or delay > M_MAX for delay in u + c):
            raise ValueError("delay outside 0..4")
        expected_id = (
            f"V1_u{u[0]}" if order == 1 else
            f"V{order}_" + "_".join(f"u{delay:02d}" for delay in u)
            + "_" + "_".join(f"c{delay:02d}" for delay in c)
        )
        if order != self.nonlinear_order or self.basis_id != expected_id:
            raise ValueError("noncanonical Volterra descriptor or basis_id")

    @classmethod
    def canonical(
        cls, nonconjugate_delays: Sequence[int], conjugate_delays: Sequence[int] = ()
    ) -> VolterraTermSpec:
        u = tuple(sorted(nonconjugate_delays))
        c = tuple(sorted(conjugate_delays))
        order = len(u) + len(c)
        if order not in (1, 3, 5, 7, 9, 11) or len(u) != (order + 1) // 2:
            raise ValueError("use the existing odd-order complex-baseband Volterra convention")
        if not all(isinstance(delay, int) and not isinstance(delay, bool) for delay in u + c):
            raise TypeError("delays must be integers")
        if any(delay < 0 or delay > M_MAX for delay in u + c):
            raise ValueError("delay outside inclusive range 0..4")
        basis_id = (
            f"V1_u{u[0]}" if order == 1 else
            f"V{order}_" + "_".join(f"u{delay:02d}" for delay in u)
            + "_" + "_".join(f"c{delay:02d}" for delay in c)
        )
        return cls(basis_id, order, u, c)

    @property
    def delay_tuple(self) -> tuple[int, ...]:
        return self.nonconjugate_delays + self.conjugate_delays

    @property
    def conjugate_pattern(self) -> tuple[bool, ...]:
        return (False,) * len(self.nonconjugate_delays) + (True,) * len(self.conjugate_delays)

    @property
    def is_diagonal(self) -> bool:
        return len(set(self.delay_tuple)) == 1

    @property
    def is_cross_memory(self) -> bool:
        return not self.is_diagonal

    @property
    def max_delay(self) -> int:
        return max(self.delay_tuple)


def build_candidate_dictionary(
    *, orders: Sequence[int] = (1, 3, 5, 7, 9, 11),
    max_delay: int = M_MAX,
    memory_by_order: Mapping[int, int] | None = None,
) -> dict[str, VolterraTermSpec]:
    """Generate descriptors only; never materialize a state-wide design bank."""

    if not orders or len(set(orders)) != len(orders) or any(
        order not in (1, 3, 5, 7, 9, 11) for order in orders
    ):
        raise ValueError("orders must be unique supported odd orders through 11")
    if not isinstance(max_delay, int) or not 0 <= max_delay <= M_MAX:
        raise ValueError("max_delay must be 0..4")
    depths = dict(memory_by_order or {})
    if set(depths) - set(orders):
        raise ValueError("memory specified for an absent order")
    dictionary: dict[str, VolterraTermSpec] = {}
    for order in sorted(orders):
        depth = depths.get(order, max_delay)
        if not isinstance(depth, int) or not 0 <= depth <= max_delay:
            raise ValueError("per-order memory must be within 0..max_delay")
        u = combinations_with_replacement(range(depth + 1), (order + 1) // 2)
        # Iterators from itertools are one-shot; re-create the conjugate iterator.
        for nonconjugate in u:
            c = combinations_with_replacement(range(depth + 1), (order - 1) // 2)
            for conjugate in c:
                term = VolterraTermSpec.canonical(nonconjugate, conjugate)
                dictionary[term.basis_id] = term
    return dictionary


def canonical_terms(terms: Iterable[VolterraTermSpec]) -> dict[str, VolterraTermSpec]:
    """Deduplicate permutations by mathematical signature, not input ordering."""

    result: dict[str, VolterraTermSpec] = {}
    for term in terms:
        canonical = VolterraTermSpec.canonical(
            term.nonconjugate_delays, term.conjugate_delays
        )
        result[canonical.basis_id] = canonical
    return dict(sorted(result.items()))


def build_basis_columns(
    x_5b: np.ndarray,
    basis_ids: Sequence[str],
    dictionary: Mapping[str, VolterraTermSpec],
    *,
    dmax: int = M_MAX,
) -> np.ndarray:
    """Materialize only requested columns, using history within the 5B segment."""

    signal = np.asarray(x_5b, dtype=np.complex128)
    if signal.ndim != 1 or signal.size <= dmax or not np.all(np.isfinite(signal)):
        raise ValueError("finite 5B segment must exceed dmax")
    if not basis_ids or len(set(basis_ids)) != len(basis_ids):
        raise ValueError("nonempty, unique support required")
    if not 0 <= dmax <= M_MAX:
        raise ValueError("dmax outside 0..4")
    columns = []
    for basis_id in basis_ids:
        term = dictionary[basis_id]
        if term.max_delay > dmax:
            raise ValueError("term needs history beyond common dmax")
        value = np.ones(signal.size - dmax, dtype=np.complex128)
        for delay in term.nonconjugate_delays:
            value *= signal[dmax - delay : signal.size - delay]
        for delay in term.conjugate_delays:
            value *= np.conj(signal[dmax - delay : signal.size - delay])
        columns.append(value)
    return np.column_stack(columns)


__all__ = [
    "P_MAX", "M_MAX", "VolterraTermSpec", "build_candidate_dictionary",
    "canonical_terms", "build_basis_columns",
]
