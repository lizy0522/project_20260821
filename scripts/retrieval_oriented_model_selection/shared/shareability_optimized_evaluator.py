"""State-streamed candidate partitions with one union of active parent columns.

Only the current deterministic partition is kept in memory. Parent bases and
Cholesky factors are reused for every block in that partition and candidate
columns are constructed once per state/segment across all beam parents.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from time import perf_counter

import numpy as np
from behavior_fingerprint_ranking_consistency.shared.distance_matrix import (
    compute_cnmse_distance_matrix,
)
from behavior_modeling.shared.basis_function_selection.model_solver import fit_ridge
from behavior_modeling.shared.volterra_terms import VolterraTermSpec, build_basis_columns

from .shareability_sufficient_stats import ParentFactor, ParentSupportStats, canonical_support


def evaluate_fixed_seed(
    seed_basis_id: str,
    *,
    x_a: np.ndarray,
    y_a: np.ndarray,
    x_c: np.ndarray,
    y_c: np.ndarray,
    common_b: np.ndarray,
    real_b: np.ndarray,
    dictionary: Mapping[str, VolterraTermSpec],
    ridge_lambda: float = 1e-8,
    include_details: bool = False,
) -> dict[str, object]:
    """Exactly one canonical x[n] seed, using full segment before dmax row selection."""
    term = dictionary[seed_basis_id]
    if (
        term.nonlinear_order != 1
        or term.nonconjugate_delays != (0,)
        or term.conjugate_delays
        or ridge_lambda != 1e-8
    ):
        raise ValueError("fixed structure seed must be canonical x[n] at lambda=1e-8")
    n = x_a.shape[0]
    if (
        x_a.shape != y_a.shape
        or x_c.shape != y_c.shape
        or x_a.shape != (n, 12288)
        or x_c.shape != (n, 7373)
        or common_b.shape != (4915,)
        or real_b.shape != (n, n)
    ):
        raise ValueError("seed evaluator requires full segment arrays")
    theta_a = np.empty((n, 1), dtype=np.complex128)
    theta_c = np.empty_like(theta_a)
    for state in range(n):
        for x, y, destination in (
            (x_a[state], y_a[state], theta_a),
            (x_c[state], y_c[state], theta_c),
        ):
            design = build_basis_columns(x, (seed_basis_id,), dictionary, dmax=4)
            # The single seed is fitted only once. Preserve the exact
            # augmented-LS reference path for near-zero CNMSE pairs.
            destination[state] = fit_ridge(design, y[4:], ridge_lambda).theta
    probe = build_basis_columns(common_b, (seed_basis_id,), dictionary, dmax=4)
    return rank_and_score(theta_a, theta_c, probe, real_b, include_details=include_details)


def rank_and_score(
    theta_a: np.ndarray,
    theta_c: np.ndarray,
    probe: np.ndarray,
    real_b: np.ndarray,
    *,
    threshold_db: float = -40.0,
    include_details: bool = False,
) -> dict[str, object]:
    """Exact waveform reference metric; coefficient shortcut is gated separately."""
    query = (probe @ theta_c.T).T
    lut = (probe @ theta_a.T).T
    distance = compute_cnmse_distance_matrix(query, lut)
    n = theta_a.shape[0]
    identity = np.arange(n)
    ranking = np.vstack([np.lexsort((identity, row)) for row in distance])
    top1 = ranking[:, 0]
    real = real_b[identity, top1]

    def count(k: int) -> int:
        return int(np.count_nonzero(np.any(ranking[:, :k] == identity[:, None], axis=1)))

    summary: dict[str, object] = {
        "N_shareable": int(np.count_nonzero(real < threshold_db)),
        "N_self": int(np.count_nonzero(top1 == identity)),
        "N_top3": count(3),
        "N_top5": count(5),
        "N_top10": count(10),
    }
    if include_details:
        summary.update(
            Q1=top1.tolist(),
            top10=ranking[:, :10].tolist(),
            true_state_rank=np.argmax(ranking == identity[:, None], axis=1).tolist(),
            distance=distance,
        )
    return summary


def evaluate_partition(
    parents: Sequence[tuple[str, ...]],
    candidate_basis_ids: Sequence[str],
    *,
    x_a: np.ndarray,
    y_a: np.ndarray,
    x_c: np.ndarray,
    y_c: np.ndarray,
    common_b: np.ndarray,
    real_b: np.ndarray,
    dictionary: Mapping[str, VolterraTermSpec],
    ridge_lambda: float,
    dmax: int = 4,
    block_size: int = 32,
    include_details: bool = False,
) -> tuple[dict[tuple[str, ...], dict[str, object]], dict[str, int | float]]:
    """Score valid unique child supports using all states, without pruning.

    State outer loop reuses parent union over all candidate blocks of this
    partition. Output is deterministic regardless of partition assignment.
    No worker pool is created here: the caller may dispatch non-overlapping
    partitions through the existing spawn orchestration.
    """
    if dmax != 4 or ridge_lambda != 1e-8 or block_size < 1:
        raise ValueError("structure stage requires global dmax=4 and lambda=1e-8")
    if len(parents) not in (1, 2, 3) or any(not p for p in parents):
        raise ValueError("one to three active parents required")
    n = x_a.shape[0]
    if (
        x_a.shape != y_a.shape
        or x_c.shape != y_c.shape
        or x_c.shape[0] != n
        or x_a.shape[1] != 12288
        or x_c.shape[1] != 7373
        or common_b.shape != (4915,)
        or real_b.shape != (n, n)
    ):
        raise ValueError("expected untrimmed Aend/C2/Common-B segment arrays")
    parent_ids = tuple(canonical_support(p) for p in parents)
    if len(set(parent_ids)) != len(parent_ids):
        raise ValueError("duplicate beam parents")
    candidates = tuple(dict.fromkeys(candidate_basis_ids))
    if not candidates or any(b not in dictionary for b in candidates):
        raise ValueError("unknown/empty candidate partition")
    union_ids = tuple(sorted(set().union(*map(set, parent_ids))))
    parent_indices = [tuple(union_ids.index(b) for b in parent) for parent in parent_ids]
    child_rows: dict[tuple[str, ...], tuple[np.ndarray, np.ndarray]] = {}
    owner: dict[tuple[str, ...], int] = {}
    associations: dict[tuple[int, int], tuple[tuple[str, ...], np.ndarray]] = {}
    for p_index, parent in enumerate(parent_ids):
        for b_index, basis_id in enumerate(candidates):
            if basis_id in parent:
                continue
            child = canonical_support((*parent, basis_id))
            if child not in child_rows:
                child_rows[child] = (
                    np.empty((n, len(child)), complex),
                    np.empty((n, len(child)), complex),
                )
                owner[child] = p_index
            if owner[child] != p_index:
                continue
            # Map parent-order-plus-new-column to canonical child ordering.
            associations[(p_index, b_index)] = (
                child,
                np.array([(*parent, basis_id).index(b) for b in child]),
            )
    if not child_rows:
        return {}, {"generated_columns": 0, "fallbacks": 0}
    diagnostics: dict[str, int | float] = {
        "generated_columns": 0,
        "fallbacks": 0,
        "parent_factorizations": 0,
        "parent_basis_seconds": 0.0,
        "parent_factor_seconds": 0.0,
        "candidate_basis_seconds": 0.0,
        "unary_cross_seconds": 0.0,
        "schur_solve_seconds": 0.0,
        "common_b_seconds": 0.0,
        "retrieval_and_oracle_seconds": 0.0,
    }
    for state in range(n):
        for segment, x, y, length in (
            ("a", x_a[state], y_a[state], 12288),
            ("c", x_c[state], y_c[state], 7373),
        ):
            if x.shape != (length,) or y.shape != (length,):
                raise ValueError("double-trimmed runtime segment")
            target = y[dmax:]
            tick = perf_counter()
            phi_union = build_basis_columns(x, union_ids, dictionary, dmax=dmax)
            diagnostics["parent_basis_seconds"] += perf_counter() - tick
            tick = perf_counter()
            factors = []
            for parent, indices in zip(parent_ids, parent_indices, strict=True):
                phi = phi_union[:, indices]
                factors.append(
                    ParentFactor.build(
                        ParentSupportStats.from_design(parent, phi, target), ridge_lambda
                    )
                )
                diagnostics["parent_factorizations"] += 1
            diagnostics["parent_factor_seconds"] += perf_counter() - tick
            for start in range(0, len(candidates), block_size):
                ids = candidates[start : start + block_size]
                tick = perf_counter()
                columns = build_basis_columns(x, ids, dictionary, dmax=dmax)
                diagnostics["candidate_basis_seconds"] += perf_counter() - tick
                diagnostics["generated_columns"] += len(ids)
                tick = perf_counter()
                energy = np.einsum("nb,nb->b", columns.conj(), columns).real
                rhs = columns.conj().T @ target
                cross_union = phi_union.conj().T @ columns
                diagnostics["unary_cross_seconds"] += perf_counter() - tick
                tick = perf_counter()
                for p_index, (parent, indices) in enumerate(
                    zip(parent_ids, parent_indices, strict=True)
                ):
                    valid = [i for i, b in enumerate(ids) if (p_index, start + i) in associations]
                    if not valid:
                        continue
                    coefficients, _, fallbacks = factors[p_index].expand_from_statistics(
                        tuple(ids[i] for i in valid),
                        cross_union[np.ix_(indices, valid)],
                        energy[valid],
                        rhs[valid],
                    )
                    diagnostics["fallbacks"] += len(fallbacks)
                    for j, i in enumerate(valid):
                        child, reorder = associations[(p_index, start + i)]
                        buffer = child_rows[child][0 if segment == "a" else 1]
                        buffer[state] = coefficients[j, reorder]
                diagnostics["schur_solve_seconds"] += perf_counter() - tick
    # Probe columns are generated once per partition, then reused by every
    # candidate child. This uses read-only Common-B, not a 10-worker copy.
    probe_ids = tuple(sorted(set(union_ids) | set(candidates)))
    tick = perf_counter()
    probe = build_basis_columns(common_b, probe_ids, dictionary, dmax=dmax)
    diagnostics["common_b_seconds"] += perf_counter() - tick
    probe_index = {basis: i for i, basis in enumerate(probe_ids)}
    scores = {}
    for child, (theta_a, theta_c) in child_rows.items():
        e = probe[:, [probe_index[basis] for basis in child]]
        tick = perf_counter()
        scores[child] = rank_and_score(theta_a, theta_c, e, real_b, include_details=include_details)
        diagnostics["retrieval_and_oracle_seconds"] += perf_counter() - tick
    return scores, diagnostics
