"""Deterministic SOMP proposal screening for exact shareability search.

This module never computes Common-B fingerprints, a 425x425 retrieval matrix,
or ``N_shareable``.  It ranks *proposal* bases only; the exact evaluator
remains the sole authority for Beam selection.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
from behavior_modeling.shared.volterra_terms import VolterraTermSpec, build_basis_columns

from .shareability_sufficient_stats import ParentFactor, ParentSupportStats, canonical_support
from .shareability_types import CandidateScreeningSpec


def candidate_pool_fingerprint(basis_ids: tuple[str, ...] | list[str]) -> str:
    unique = tuple(sorted(set(basis_ids)))
    if len(unique) != len(basis_ids):
        raise ValueError("candidate pool contains duplicate basis IDs")
    return hashlib.sha256(json.dumps(unique, separators=(",", ":")).encode()).hexdigest()


def screening_policy_fingerprint(
    exact_evaluation_fingerprint: str,
    spec: CandidateScreeningSpec,
    *,
    search_mode: str = "screened",
    implementation_fingerprint: str | None = None,
) -> str:
    """Fingerprint checkpoint/screening artefacts, not exact CandidateScores."""
    if len(exact_evaluation_fingerprint) != 64 or search_mode not in {"screened", "exhaustive"}:
        raise ValueError("full exact fingerprint and known search mode are required")
    if implementation_fingerprint is not None and len(implementation_fingerprint) != 64:
        raise ValueError("screening implementation fingerprint must be SHA256 when supplied")
    payload = {
        "exact_evaluation_fingerprint": exact_evaluation_fingerprint,
        "search_mode": search_mode,
        "screening": spec.to_dict() if search_mode == "screened" else None,
        "screening_implementation_fingerprint": implementation_fingerprint,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def descriptor_stratum(term: VolterraTermSpec) -> str:
    interaction = "diagonal" if term.is_diagonal else "cross_memory"
    return f"p{term.nonlinear_order}|{interaction}|d{term.max_delay}"


@dataclass(frozen=True, slots=True)
class ScreeningShortlistEntry:
    basis_id: str
    score: float
    rank: int
    stratum: str
    proposal_source: str


@dataclass(slots=True)
class ParentScreeningScores:
    parent_support: tuple[str, ...]
    candidate_ids: tuple[str, ...]
    scores: np.ndarray
    aend_scores: np.ndarray
    c2_scores: np.ndarray
    strata: tuple[str, ...]
    diagnostics: dict[str, int | float]

    def shortlist(
        self,
        spec: CandidateScreeningSpec,
        dictionary: dict[str, VolterraTermSpec],
    ) -> tuple[ScreeningShortlistEntry, ...]:
        return build_shortlist(self, spec, dictionary)


def residual_energy_from_statistics(
    stats: ParentSupportStats,
    coefficients: np.ndarray,
    target_energy: float,
) -> tuple[float, float]:
    """Return residual energy and a scale-aware zero-residual tolerance."""
    if coefficients.shape != stats.rhs.shape or not np.isfinite(target_energy):
        raise ValueError("invalid parent coefficients or target energy")
    quadratic = np.vdot(coefficients, stats.gram @ coefficients)
    linear = np.vdot(coefficients, stats.rhs)
    energy = float(target_energy - 2.0 * np.real(linear) + np.real(quadratic))
    scale = abs(target_energy) + abs(linear) + abs(quadratic) + 1.0
    tolerance = 64.0 * np.finfo(np.float64).eps * scale
    if not np.isfinite(energy):
        raise FloatingPointError("non-finite residual energy")
    return energy, tolerance


def normalized_residual_correlation_from_statistics(
    factor: ParentFactor,
    cross: np.ndarray,
    basis_energy: np.ndarray,
    basis_rhs: np.ndarray,
    target_energy: float,
) -> tuple[np.ndarray, dict[str, int]]:
    """Compute |phi^H r|^2 / (phi^H phi * r^H r) without materializing r.

    ``cross`` is Phi^H phi and ``basis_rhs`` is phi^H y, so the complex
    residual correlation is exactly ``basis_rhs - cross^H w``.
    """
    k, width = cross.shape
    if (
        k != len(factor.stats.basis_ids)
        or basis_energy.shape != (width,)
        or basis_rhs.shape != (width,)
    ):
        raise ValueError("screening statistics shape mismatch")
    residual_energy, residual_tolerance = residual_energy_from_statistics(
        factor.stats, factor.coefficients, target_energy
    )
    diagnostics = {
        "zero_residual_count": 0,
        "near_zero_basis_energy_count": 0,
        "nan_count": 0,
        "inf_count": 0,
    }
    score = np.zeros(width, dtype=np.float64)
    if residual_energy <= residual_tolerance:
        diagnostics["zero_residual_count"] = 1
        return score, diagnostics
    if residual_energy < 0:
        raise FloatingPointError("residual energy is negative beyond numerical tolerance")
    energy_tolerance = (
        64.0 * np.finfo(np.float64).eps * (np.abs(basis_energy) + abs(target_energy) + 1.0)
    )
    valid = basis_energy > energy_tolerance
    diagnostics["near_zero_basis_energy_count"] = int(np.count_nonzero(~valid))
    residual_correlation = basis_rhs - cross.conj().T @ factor.coefficients
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        score[valid] = (
            np.abs(residual_correlation[valid]) ** 2 / (basis_energy[valid] * residual_energy)
        ).real
    diagnostics["nan_count"] = int(np.count_nonzero(np.isnan(score)))
    diagnostics["inf_count"] = int(np.count_nonzero(np.isinf(score)))
    if diagnostics["nan_count"] or diagnostics["inf_count"]:
        raise FloatingPointError("non-finite SOMP residual correlation")
    return score, diagnostics


def screen_candidate_pool(
    parents: tuple[tuple[str, ...], ...],
    candidate_basis_ids: tuple[str, ...],
    *,
    x_a: np.ndarray,
    y_a: np.ndarray,
    x_c: np.ndarray,
    y_c: np.ndarray,
    dictionary: dict[str, VolterraTermSpec],
    ridge_lambda: float,
    spec: CandidateScreeningSpec,
    dmax: int = 4,
    block_size: int = 64,
) -> dict[tuple[str, ...], ParentScreeningScores]:
    """Screen a whole candidate partition with shared waveform generation.

    Parent scores are separate, while each state/segment/candidate block is
    generated once and shared across all active parents.  Contributions are
    averaged over all 425 states; degenerate state/segment contributions are
    explicitly zero, not silently epsilon-adjusted.
    """
    if not spec.enabled or spec.method != "somp_residual_correlation":
        raise ValueError("screen_candidate_pool requires enabled SOMP screening")
    if ridge_lambda != 1e-8 or dmax != 4 or block_size < 1:
        raise ValueError("screening requires frozen structure lambda=1e-8 and dmax=4")
    if not 1 <= len(parents) <= 3:
        raise ValueError("one to three active parents are required")
    parent_ids = tuple(canonical_support(parent) for parent in parents)
    if len(set(parent_ids)) != len(parent_ids):
        raise ValueError("duplicate active parent support")
    candidates = tuple(dict.fromkeys(candidate_basis_ids))
    if len(candidates) != len(candidate_basis_ids) or not candidates:
        raise ValueError("candidate IDs must be nonempty and unique")
    if any(candidate not in dictionary for candidate in candidates):
        raise ValueError("candidate absent from Volterra dictionary")
    if any(basis not in dictionary for parent in parent_ids for basis in parent):
        raise ValueError("parent basis absent from Volterra dictionary")
    n_states = x_a.shape[0]
    if (
        x_a.shape != y_a.shape
        or x_c.shape != y_c.shape
        or x_a.shape != (n_states, 12288)
        or x_c.shape != (n_states, 7373)
    ):
        raise ValueError("screening requires untrimmed 5B Aend/C2 arrays")
    union_ids = tuple(sorted(set().union(*map(set, parent_ids))))
    parent_indices = [tuple(union_ids.index(basis) for basis in parent) for parent in parent_ids]
    aggregate: dict[tuple[str, ...], dict[str, Any]] = {}
    for parent in parent_ids:
        aggregate[parent] = {
            "aend": np.zeros(len(candidates), dtype=np.float64),
            "c2": np.zeros(len(candidates), dtype=np.float64),
            "diagnostics": {
                "candidate_columns": 0,
                "parent_factorizations": 0,
                "zero_residual_count": 0,
                "near_zero_basis_energy_count": 0,
                "nan_count": 0,
                "inf_count": 0,
                "parent_basis_seconds": 0.0,
                "parent_factor_seconds": 0.0,
                "candidate_basis_seconds": 0.0,
                "unary_cross_seconds": 0.0,
                "somp_seconds": 0.0,
            },
        }
    for state_id in range(n_states):
        for segment_name, x, y, raw_length in (
            ("aend", x_a[state_id], y_a[state_id], 12288),
            ("c2", x_c[state_id], y_c[state_id], 7373),
        ):
            if x.shape != (raw_length,) or y.shape != (raw_length,):
                raise ValueError("screening detected a double-trimmed segment")
            target = y[dmax:]
            target_energy = float(np.vdot(target, target).real)
            tick = perf_counter()
            union_design = build_basis_columns(x, union_ids, dictionary, dmax=dmax)
            elapsed = perf_counter() - tick
            factors: list[ParentFactor] = []
            for parent, indices in zip(parent_ids, parent_indices, strict=True):
                aggregate[parent]["diagnostics"]["parent_basis_seconds"] += elapsed
                tick = perf_counter()
                parent_design = union_design[:, indices]
                factor = ParentFactor.build(
                    ParentSupportStats.from_design(parent, parent_design, target), ridge_lambda
                )
                aggregate[parent]["diagnostics"]["parent_factor_seconds"] += perf_counter() - tick
                aggregate[parent]["diagnostics"]["parent_factorizations"] += 1
                factors.append(factor)
            for start in range(0, len(candidates), block_size):
                ids = candidates[start : start + block_size]
                tick = perf_counter()
                columns = build_basis_columns(x, ids, dictionary, dmax=dmax)
                elapsed = perf_counter() - tick
                tick = perf_counter()
                energy = np.einsum("nb,nb->b", columns.conj(), columns).real
                rhs = columns.conj().T @ target
                union_cross = union_design.conj().T @ columns
                unary_cross_elapsed = perf_counter() - tick
                for parent_index, (parent, indices) in enumerate(
                    zip(parent_ids, parent_indices, strict=True)
                ):
                    diagnostics = aggregate[parent]["diagnostics"]
                    diagnostics["candidate_basis_seconds"] += elapsed
                    diagnostics["unary_cross_seconds"] += unary_cross_elapsed
                    diagnostics["candidate_columns"] += len(ids)
                    valid = np.array([basis not in parent for basis in ids], dtype=bool)
                    if not np.any(valid):
                        continue
                    tick = perf_counter()
                    rho, local = normalized_residual_correlation_from_statistics(
                        factors[parent_index],
                        union_cross[np.ix_(indices, np.flatnonzero(valid))],
                        energy[valid],
                        rhs[valid],
                        target_energy,
                    )
                    diagnostics["somp_seconds"] += perf_counter() - tick
                    for key, value in local.items():
                        diagnostics[key] += value
                    candidate_indices = start + np.flatnonzero(valid)
                    aggregate[parent][segment_name][candidate_indices] += rho
    result: dict[tuple[str, ...], ParentScreeningScores] = {}
    strata = tuple(descriptor_stratum(dictionary[basis]) for basis in candidates)
    for parent in parent_ids:
        aend = aggregate[parent]["aend"] / n_states
        c2 = aggregate[parent]["c2"] / n_states
        combined = spec.aend_weight * aend + spec.c2_weight * c2
        selected = np.array([basis in parent for basis in candidates], dtype=bool)
        combined[selected] = np.nan
        aend[selected] = np.nan
        c2[selected] = np.nan
        result[parent] = ParentScreeningScores(
            parent, candidates, combined, aend, c2, strata, aggregate[parent]["diagnostics"]
        )
    return result


def build_shortlist(
    scores: ParentScreeningScores,
    spec: CandidateScreeningSpec,
    dictionary: dict[str, VolterraTermSpec],
) -> tuple[ScreeningShortlistEntry, ...]:
    """Top exploit plus deterministic stratum round-robin exploration."""
    if tuple(scores.strata) != tuple(
        descriptor_stratum(dictionary[b]) for b in scores.candidate_ids
    ):
        raise ValueError("screening strata do not match current canonical dictionary")
    candidates = [
        (basis, float(score), stratum)
        for basis, score, stratum in zip(
            scores.candidate_ids, scores.scores, scores.strata, strict=True
        )
        if basis not in scores.parent_support and np.isfinite(score)
    ]
    ranked = sorted(candidates, key=lambda item: (-item[1], item[0]))
    target = min(spec.shortlist_size, len(ranked))
    exploit_count = min(spec.exploit_size, target)
    selected = ranked[:exploit_count]
    selected_ids = {basis for basis, _, _ in selected}
    remaining = [item for item in ranked if item[0] not in selected_ids]
    groups: dict[str, list[tuple[str, float, str]]] = {}
    for item in remaining:
        groups.setdefault(item[2], []).append(item)
    exploration: list[tuple[str, float, str]] = []
    while len(selected) + len(exploration) < target:
        advanced = False
        for stratum in sorted(groups):
            group = groups[stratum]
            if group and len(selected) + len(exploration) < target:
                exploration.append(group.pop(0))
                advanced = True
        if not advanced:
            break
    if len(selected) + len(exploration) != target:
        raise RuntimeError("deterministic exploration did not fill shortlist")
    rank = {basis: index + 1 for index, (basis, _, _) in enumerate(ranked)}
    entries = [
        ScreeningShortlistEntry(basis, score, rank[basis], stratum, "exploit")
        for basis, score, stratum in selected
    ]
    entries.extend(
        ScreeningShortlistEntry(basis, score, rank[basis], stratum, "exploration")
        for basis, score, stratum in exploration
    )
    if len({entry.basis_id for entry in entries}) != len(entries):
        raise RuntimeError("shortlist contains duplicate basis")
    return tuple(entries)


def merge_screening_partitions(
    partitions: tuple[dict[tuple[str, ...], ParentScreeningScores], ...],
    candidate_ids: tuple[str, ...],
) -> dict[tuple[str, ...], ParentScreeningScores]:
    """Merge non-overlapping deterministic worker partitions in canonical order."""
    if not partitions or not candidate_ids or len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("nonempty unique candidate IDs and partitions are required")
    parent_sets = [set(partition) for partition in partitions]
    if any(parents != parent_sets[0] for parents in parent_sets[1:]):
        raise ValueError("screening partitions have different active parents")
    merged: dict[tuple[str, ...], ParentScreeningScores] = {}
    for parent in sorted(parent_sets[0]):
        by_id: dict[str, tuple[float, float, float, str]] = {}
        diagnostics: dict[str, int | float] = {}
        for partition in partitions:
            row = partition[parent]
            if row.parent_support != parent:
                raise ValueError("screening partition parent mismatch")
            for basis, score, aend, c2, stratum in zip(
                row.candidate_ids,
                row.scores,
                row.aend_scores,
                row.c2_scores,
                row.strata,
                strict=True,
            ):
                if basis in by_id:
                    raise ValueError("overlapping screening candidate partition")
                by_id[basis] = (float(score), float(aend), float(c2), stratum)
            for key, value in row.diagnostics.items():
                diagnostics[key] = diagnostics.get(key, 0) + value
        if set(by_id) != set(candidate_ids):
            raise ValueError("incomplete screening candidate partition")
        values = [by_id[basis] for basis in candidate_ids]
        merged[parent] = ParentScreeningScores(
            parent,
            candidate_ids,
            np.asarray([value[0] for value in values]),
            np.asarray([value[1] for value in values]),
            np.asarray([value[2] for value in values]),
            tuple(value[3] for value in values),
            diagnostics,
        )
    return merged


__all__ = [
    "CandidateScreeningSpec",
    "ParentScreeningScores",
    "ScreeningShortlistEntry",
    "build_shortlist",
    "candidate_pool_fingerprint",
    "descriptor_stratum",
    "normalized_residual_correlation_from_statistics",
    "residual_energy_from_statistics",
    "merge_screening_partitions",
    "screen_candidate_pool",
    "screening_policy_fingerprint",
]
