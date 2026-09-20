# ruff: noqa: E501

"""
Statewise-adaptive Memory Polynomial model search for Scenario 2.

This module adds a model-selection layer around the existing MATLAB-compatible
Memory Polynomial basis and Ridge solver.  The frozen 5B preprocessing path is
used for every pair: full-record Rough/Fine alignment, fixed unequal ABC
ownership and segment-wise complex-gain adjustment.  Aend and C2 are fitted
independently for each state.

The first search round is the existing 200-row candidate bank from the
retrieval-oriented scan.  Later rounds add deterministic higher-order,
longer-memory and outward log-lambda shells only for unresolved state/side
pairs.  Native-support metrics are followed by a common-support validation so
that a long memory cannot pass only by discarding an excessive prefix.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from data_management.shared import load_by_id
from retrieval_oriented_model_selection.shared.scenario2_retrieval_oriented_model_scan import (
    CANDIDATES as EXISTING_CANDIDATES,
)
from retrieval_oriented_model_selection.shared.scenario2_retrieval_oriented_model_scan import (
    LAMBDA_GRID as EXISTING_LAMBDA_GRID,
)
from retrieval_oriented_model_selection.shared.scenario2_retrieval_oriented_model_scan import (
    MEMORY_PROFILES as EXISTING_MEMORY_PROFILES,
)
from signal_segmentation.shared import build_partition_from_xin, get_ilc_pair, preprocess_full_pair

from behavior_modeling.shared.basis import build_mp_basis
from behavior_modeling.shared.evaluation import calculate_nmse
from behavior_modeling.shared.ridge import fit_coefficients_ridge

STATE_COUNT = 425
WAVEFORM_LENGTH = 24576
FORMAL_A_LENGTH = 12288
FORMAL_B_LENGTH = 4915
FORMAL_C_LENGTH = 7373
DPD_SHAREABLE_THRESHOLD_DB = -40.0
C2_STAGE = 2
BASELINE_LAMBDA = 1e-8
BASELINE_ORDERS = (1, 2, 3, 5, 7, 9)
BASELINE_MEMORY = {1: 3, 2: 2, 3: 2, 5: 1, 7: 1, 9: 1}

# Existing candidate-bank facts are imported rather than redefined.  The
# search module only owns the expansion policy and statewise selection.
EXISTING_CANDIDATE_COUNT = len(EXISTING_CANDIDATES)
EXISTING_LAMBDA_GRID = tuple(float(value) for value in EXISTING_LAMBDA_GRID)
SUPPORTED_ORDERS = (1, 2, 3) + tuple(range(5, 101, 2))
BASE_MEMORY_BY_ORDER = {
    int(order): int(depth) for order, depth in {**BASELINE_MEMORY, **{11: 1, 13: 1, 15: 1}}.items()
}


@dataclass(frozen=True)
class AdaptiveCandidate:
    """One statewise-search candidate and its deterministic provenance."""

    candidate_id: int
    search_round: int
    source: str
    order_profile: str
    orders: tuple[int, ...]
    memory_profile: str
    memory_definition: dict[int, int]
    ridge_lambda: float
    max_delay: int
    coefficient_count: int
    structure_id: str
    is_baseline: bool = False

    @property
    def key(self) -> tuple[Any, ...]:
        return (
            self.order_profile,
            self.orders,
            self.memory_profile,
            tuple(sorted(self.memory_definition.items())),
            float(self.ridge_lambda),
        )

    @property
    def basis_terms(self) -> tuple[tuple[int, int], ...]:
        return tuple(
            (int(order), int(memory))
            for order in self.orders
            for memory in range(int(self.memory_definition[int(order)]))
        )


@dataclass(frozen=True)
class PreparedStatePairs:
    """Canonical Aend/C2 pairs for one state and the formal B probe."""

    state_id: int
    ilc_A_end: int
    a_end: Any
    c2: Any
    common_b_input: np.ndarray


@dataclass(frozen=True)
class CandidateFit:
    """One candidate fit, including a finite theta when valid."""

    row: dict[str, Any]
    theta: np.ndarray | None


def _json_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _normalise_existing_candidate(candidate: Any) -> AdaptiveCandidate:
    return AdaptiveCandidate(
        candidate_id=int(candidate.candidate_id),
        search_round=0,
        source="existing_retrieval_oriented_candidate_bank",
        order_profile=str(candidate.order_profile),
        orders=tuple(int(value) for value in candidate.orders),
        memory_profile=str(candidate.memory_profile),
        memory_definition={
            int(key): int(value) for key, value in candidate.memory_definition.items()
        },
        ridge_lambda=float(candidate.ridge_lambda),
        max_delay=int(candidate.max_delay),
        coefficient_count=int(candidate.n_complex_coefficients),
        structure_id=str(candidate.structure_id),
        is_baseline=bool(candidate.is_baseline),
    )


def initial_candidate_bank() -> tuple[AdaptiveCandidate, ...]:
    """Return the existing 200-row candidate bank without redefining it."""

    candidates = tuple(_normalise_existing_candidate(item) for item in EXISTING_CANDIDATES)
    ids = [candidate.candidate_id for candidate in candidates]
    if ids != list(range(1, EXISTING_CANDIDATE_COUNT + 1)):
        raise RuntimeError("existing candidate bank IDs are not consecutive")
    baseline = [candidate for candidate in candidates if candidate.is_baseline]
    if len(baseline) != 1:
        raise RuntimeError("existing candidate bank must contain one baseline")
    if (
        baseline[0].orders != BASELINE_ORDERS
        or baseline[0].memory_definition != BASELINE_MEMORY
        or baseline[0].ridge_lambda != BASELINE_LAMBDA
    ):
        raise RuntimeError("existing candidate bank baseline is not P9/M0/lambda=1e-8")
    return candidates


def _order_prefix(max_order: int) -> tuple[int, ...]:
    values = tuple(order for order in SUPPORTED_ORDERS if order <= max_order)
    if 7 not in values:
        raise ValueError("order prefix must include at least P7")
    return values


def _expanded_memory(level: int, orders: Iterable[int]) -> dict[int, int]:
    """Build a progressive memory shell using the frozen M0 per-order anchors."""

    if level < 4:
        raise ValueError("expanded memory shells start at M4")
    return {int(order): int(BASE_MEMORY_BY_ORDER.get(int(order), 1) + level) for order in orders}


def expanded_lambda_grid(search_round: int) -> tuple[float, ...]:
    """Expand the existing positive lambda decades one shell at a time."""

    if search_round < 1:
        return EXISTING_LAMBDA_GRID
    values = {float(value) for value in EXISTING_LAMBDA_GRID}
    # Existing grid spans 1e-12 ... 1e-4.  Round r adds one decade on each side.
    values.add(10.0 ** (-12 - search_round))
    values.add(10.0 ** (-4 + search_round))
    return (0.0, *sorted(value for value in values if value > 0.0))


def expand_candidate_shell(
    existing: Sequence[AdaptiveCandidate], search_round: int
) -> tuple[AdaptiveCandidate, ...]:
    """Return a deterministic new shell for a progressive search round.

    The shell adds two new order prefixes and one longer-memory profile for all
    prefixes through the new maximum order.  Existing candidates are removed by
    their full structural/lambda key.  No candidate is silently overwritten.
    """

    if search_round < 1:
        raise ValueError("search_round must be >= 1")
    max_order = 15 + 4 * search_round
    new_order_maxima = (max_order - 2, max_order)
    existing_keys = {candidate.key for candidate in existing}
    next_id = max(candidate.candidate_id for candidate in existing) + 1
    lambdas = expanded_lambda_grid(search_round)
    new_candidates: list[AdaptiveCandidate] = []

    # Higher-order shell with all previously available memory profiles.  For
    # orders above 15, M0-M3 use one tap, which is the extension of the
    # existing candidate-bank convention.
    for order_max in new_order_maxima:
        orders = _order_prefix(order_max)
        for memory_profile, template in EXISTING_MEMORY_PROFILES.items():
            memory = {int(order): int(template.get(int(order), 1)) for order in orders}
            for ridge_lambda in lambdas:
                candidate = AdaptiveCandidate(
                    candidate_id=next_id,
                    search_round=search_round,
                    source="progressive_order_shell",
                    order_profile=f"P{order_max}",
                    orders=orders,
                    memory_profile=str(memory_profile),
                    memory_definition=memory,
                    ridge_lambda=float(ridge_lambda),
                    max_delay=max(memory.values()) - 1,
                    coefficient_count=sum(memory.values()),
                    structure_id=f"P{order_max}_{memory_profile}",
                )
                next_id += 1
                if candidate.key not in existing_keys:
                    existing_keys.add(candidate.key)
                    new_candidates.append(candidate)

    # Longer-memory shell, including all order prefixes available this round.
    memory_profile = f"M{3 + search_round}"
    for order_max in range(7, max_order + 1, 2):
        orders = _order_prefix(order_max)
        memory = _expanded_memory(3 + search_round, orders)
        for ridge_lambda in lambdas:
            candidate = AdaptiveCandidate(
                candidate_id=next_id,
                search_round=search_round,
                source="progressive_memory_shell",
                order_profile=f"P{order_max}",
                orders=orders,
                memory_profile=memory_profile,
                memory_definition=memory,
                ridge_lambda=float(ridge_lambda),
                max_delay=max(memory.values()) - 1,
                coefficient_count=sum(memory.values()),
                structure_id=f"P{order_max}_{memory_profile}",
            )
            next_id += 1
            if candidate.key not in existing_keys:
                existing_keys.add(candidate.key)
                new_candidates.append(candidate)

    # Candidate generation itself is finite for the current shell.  The
    # caller may keep opening later shells until every new shell is unusable
    # because coefficient_count >= B rows, which is a real basis blocker.
    return tuple(new_candidates)


def candidate_grid(candidates: Sequence[AdaptiveCandidate]) -> pd.DataFrame:
    """Return a machine-readable candidate bank with explicit definitions."""

    rows = [
        {
            "candidate_id": candidate.candidate_id,
            "search_round": candidate.search_round,
            "source": candidate.source,
            "order_profile": candidate.order_profile,
            "orders": _json_compact(list(candidate.orders)),
            "memory_profile": candidate.memory_profile,
            "memory_definition": _json_compact(candidate.memory_definition),
            "lambda": candidate.ridge_lambda,
            "max_delay": candidate.max_delay,
            "coefficient_count": candidate.coefficient_count,
            "structure_id": candidate.structure_id,
            "is_baseline": candidate.is_baseline,
        }
        for candidate in candidates
    ]
    frame = pd.DataFrame(rows).sort_values("candidate_id").reset_index(drop=True)
    expected_ids = np.arange(1, frame.shape[0] + 1, dtype=np.int64)
    if not np.array_equal(frame["candidate_id"].to_numpy(dtype=np.int64), expected_ids):
        raise RuntimeError("candidate IDs must be consecutive after shell expansion")
    if frame["coefficient_count"].le(0).any():
        raise RuntimeError("candidate coefficient count must be positive")
    return frame


def prepare_state_pairs(state_id: int) -> PreparedStatePairs:
    """Load one state through data_management and the canonical preprocessing API."""

    if not isinstance(state_id, (int, np.integer)) or not 0 <= int(state_id) < STATE_COUNT:
        raise ValueError(f"state_id must be in 0...424: {state_id!r}")
    state_id = int(state_id)
    state_data = load_by_id(state_id)
    xin = np.asarray(state_data["xin"])
    if xin.ndim == 2 and xin.shape[1] == 1:
        xin = xin[:, 0]
    if xin.ndim != 1 or not np.iscomplexobj(xin) or xin.size != WAVEFORM_LENGTH:
        raise ValueError(f"state={state_id} xin must be complex length {WAVEFORM_LENGTH}")
    partition = build_partition_from_xin(xin)
    if (partition.n_a, partition.n_b, partition.n_c) != (
        FORMAL_A_LENGTH,
        FORMAL_B_LENGTH,
        FORMAL_C_LENGTH,
    ):
        raise RuntimeError("formal ABC ownership changed")

    input_history = np.asarray(state_data["xin_pd_ori_ilc"])
    output_history = np.asarray(state_data["yout_withdpd_ori_ilc"])
    if input_history.ndim != 2 or output_history.shape != input_history.shape:
        raise ValueError(f"state={state_id} ILC input/output history shape mismatch")
    if input_history.shape[0] != WAVEFORM_LENGTH or input_history.shape[1] < C2_STAGE:
        raise RuntimeError(f"state={state_id} does not have a real C2 pair")
    ilc_count = int(input_history.shape[1])
    a_pair = get_ilc_pair(state_data, ilc_count - 1)
    a_end = preprocess_full_pair(
        a_pair.input_full,
        a_pair.output_raw_full,
        partition,
        pair_type="ILC",
        iteration_index=ilc_count - 1,
        input_peak_normalization_factor=a_pair.input_peak_normalization_factor,
    )
    c_pair = get_ilc_pair(state_data, C2_STAGE - 1)
    c2 = (
        a_end
        if ilc_count == C2_STAGE
        else preprocess_full_pair(
            c_pair.input_full,
            c_pair.output_raw_full,
            partition,
            pair_type="ILC",
            iteration_index=C2_STAGE - 1,
            input_peak_normalization_factor=c_pair.input_peak_normalization_factor,
        )
    )
    return PreparedStatePairs(
        state_id=state_id,
        ilc_A_end=ilc_count,
        a_end=a_end,
        c2=c2,
        common_b_input=np.asarray(xin[partition.b_slice], dtype=np.complex128),
    )


def _full_memory_for_group(
    candidates: Sequence[AdaptiveCandidate],
) -> tuple[tuple[int, ...], dict[int, int]]:
    """Return a full order-major basis specification for one memory profile."""

    if not candidates:
        raise ValueError("candidate group cannot be empty")
    all_orders = sorted(
        {order for candidate in candidates for order in candidate.orders},
        key=lambda value: (value != 1, value != 2, value),
    )
    # The sort key above keeps 1 and 2 before the odd-order sequence.  A more
    # explicit order is used below to retain the existing order-major contract.
    all_orders = [order for order in SUPPORTED_ORDERS if order in all_orders]
    full_memory: dict[int, int] = {}
    for order in all_orders:
        full_memory[order] = max(
            int(candidate.memory_definition.get(order, 1)) for candidate in candidates
        )
    return tuple(all_orders), full_memory


def _basis_columns(
    full_orders: Sequence[int], full_memory: Mapping[int, int], candidate: AdaptiveCandidate
) -> tuple[int, ...]:
    full_terms = tuple(
        (int(order), memory)
        for order in full_orders
        for memory in range(int(full_memory[int(order)]))
    )
    columns = {term: index for index, term in enumerate(full_terms)}
    return tuple(columns[term] for term in candidate.basis_terms)


def _fit_one(
    *,
    state_id: int,
    side: str,
    canonical: Any,
    train_segment_name: str,
    candidate: AdaptiveCandidate,
    full_orders: Sequence[int],
    full_memory: Mapping[int, int],
    phi_train_full: np.ndarray,
    y_train_full: np.ndarray,
    phi_b_full: np.ndarray,
    y_b_full: np.ndarray,
    ols_cache: dict[tuple[int, ...], tuple[np.ndarray, int, np.ndarray]],
) -> CandidateFit:
    del canonical, train_segment_name
    row: dict[str, Any] = {
        "state_id": state_id,
        "side": side,
        "candidate_id": candidate.candidate_id,
        "search_round": candidate.search_round,
        "source": candidate.source,
        "order_profile": candidate.order_profile,
        "orders": _json_compact(list(candidate.orders)),
        "memory_profile": candidate.memory_profile,
        "memory_definition": _json_compact(candidate.memory_definition),
        "lambda": candidate.ridge_lambda,
        "max_delay": candidate.max_delay,
        "coefficient_count": candidate.coefficient_count,
        "structure_id": candidate.structure_id,
        "native_train_NMSE_dB": np.nan,
        "native_B_NMSE_dB": np.nan,
        "rank": 0,
        "condition_number_phi": np.nan,
        "condition_number_augmented": np.nan,
        "n_train_samples": 0,
        "theta_l2_norm": np.nan,
        "finite": False,
        "valid": False,
        "failure_reason": "",
    }
    try:
        columns = _basis_columns(full_orders, full_memory, candidate)
        phi_train = np.asarray(phi_train_full[:, columns], dtype=np.complex128)
        phi_b = np.asarray(phi_b_full[:, columns], dtype=np.complex128)
        if phi_train.shape[1] != candidate.coefficient_count:
            raise RuntimeError("candidate coefficient count does not match selected basis columns")
        if candidate.max_delay != max(full_memory.values()) - 1:
            raise RuntimeError("candidate max_delay does not match its native basis support")
        if phi_train.shape[0] <= phi_train.shape[1]:
            raise ValueError("valid row count is not greater than coefficient count")
        if not np.all(np.isfinite(phi_train)) or not np.all(np.isfinite(phi_b)):
            raise ValueError("basis contains NaN/Inf")
        key = columns
        if key not in ols_cache:
            ols_cache[key] = tuple(np.linalg.lstsq(phi_train, y_train_full, rcond=None))  # type: ignore[assignment]
        theta_ols, _, rank, singular = ols_cache[key]
        if int(rank) != candidate.coefficient_count:
            raise ValueError(f"rank deficient {rank}/{candidate.coefficient_count}")
        theta, diagnostics = fit_coefficients_ridge(
            phi_train,
            y_train_full,
            candidate.ridge_lambda,
            ols_solution=(theta_ols, int(rank), np.asarray(singular)),
        )
        train_prediction = phi_train @ theta
        b_prediction = phi_b @ theta
        train_nmse = calculate_nmse(y_train_full, train_prediction)
        b_nmse = calculate_nmse(y_b_full, b_prediction)
        if not np.isfinite(train_nmse) or not np.isfinite(b_nmse):
            raise ValueError("NMSE is not finite")
        row.update(
            {
                "native_train_NMSE_dB": float(train_nmse),
                "native_B_NMSE_dB": float(b_nmse),
                "rank": int(diagnostics.rank_phi),
                "condition_number_phi": float(diagnostics.condition_number_phi),
                "condition_number_augmented": float(diagnostics.condition_number_augmented),
                "n_train_samples": int(diagnostics.n_train_samples),
                "theta_l2_norm": float(diagnostics.theta_l2_norm),
                "finite": True,
                "valid": True,
            }
        )
        return CandidateFit(row=row, theta=np.asarray(theta, dtype=np.complex128))
    except Exception as exc:  # candidate-level rejection is intentional
        row["failure_reason"] = f"{type(exc).__name__}: {exc}"
        return CandidateFit(row=row, theta=None)


def fit_candidates_for_state(
    prepared: PreparedStatePairs,
    candidates: Sequence[AdaptiveCandidate],
) -> tuple[list[CandidateFit], dict[tuple[str, int], np.ndarray]]:
    """Fit all supplied candidates for one state, grouped by memory shell."""

    fits: list[CandidateFit] = []
    theta_store: dict[tuple[str, int], np.ndarray] = {}
    for memory_profile in sorted({candidate.memory_profile for candidate in candidates}):
        group = [
            candidate for candidate in candidates if candidate.memory_profile == memory_profile
        ]
        full_orders, full_memory = _full_memory_for_group(group)
        max_delay = max(full_memory.values()) - 1
        side_specs = (
            ("Aend", prepared.a_end, "A"),
            ("C2", prepared.c2, "C"),
        )
        for side, canonical, train_segment_name in side_specs:
            train_segment = canonical[train_segment_name]
            b_segment = canonical["B"]
            phi_train_full = build_mp_basis(train_segment.input, full_orders, full_memory)
            phi_b_full = build_mp_basis(b_segment.input, full_orders, full_memory)
            y_train_full = np.asarray(train_segment.output[max_delay:], dtype=np.complex128)
            y_b_full = np.asarray(b_segment.output[max_delay:], dtype=np.complex128)
            expected_train = train_segment.ownership_length - max_delay
            expected_b = b_segment.ownership_length - max_delay
            if phi_train_full.shape[0] != expected_train or phi_b_full.shape[0] != expected_b:
                raise RuntimeError("native support length mismatch")
            if y_train_full.shape != (expected_train,) or y_b_full.shape != (expected_b,):
                raise RuntimeError("native target length mismatch")
            ols_cache: dict[tuple[int, ...], tuple[np.ndarray, int, np.ndarray]] = {}
            for candidate in group:
                fit = _fit_one(
                    state_id=prepared.state_id,
                    side=side,
                    canonical=canonical,
                    train_segment_name=train_segment_name,
                    candidate=candidate,
                    full_orders=full_orders,
                    full_memory=full_memory,
                    phi_train_full=phi_train_full,
                    y_train_full=y_train_full,
                    phi_b_full=phi_b_full,
                    y_b_full=y_b_full,
                    ols_cache=ols_cache,
                )
                fits.append(fit)
                if fit.theta is not None:
                    theta_store[(side, candidate.candidate_id)] = fit.theta
    return fits, theta_store


def native_feasible(row: Mapping[str, Any]) -> bool:
    """Strict native-support feasibility gate."""

    return bool(
        row.get("valid", False)
        and float(row.get("native_train_NMSE_dB", np.inf)) < DPD_SHAREABLE_THRESHOLD_DB
        and float(row.get("native_B_NMSE_dB", np.inf)) < DPD_SHAREABLE_THRESHOLD_DB
    )


def _selection_key(row: Mapping[str, Any], *, train_field: str, b_field: str) -> tuple[Any, ...]:
    return (
        int(row["coefficient_count"]),
        int(row["max_delay"]),
        -float(row["lambda"]),
        max(float(row[train_field]), float(row[b_field])),
        int(row["candidate_id"]),
    )


def select_native_candidates(
    records: Mapping[tuple[str, int, int], Mapping[str, Any]],
    state_ids: Iterable[int] = range(STATE_COUNT),
) -> tuple[dict[tuple[str, int], int], dict[str, list[int]]]:
    """Select the least-complexity native-feasible candidate per state/side."""

    selected: dict[tuple[str, int], int] = {}
    unresolved = {"Aend": [], "C2": []}
    for side in ("Aend", "C2"):
        for state_id in state_ids:
            options = [
                row
                for (row_side, row_state, _), row in records.items()
                if row_side == side and row_state == int(state_id) and native_feasible(row)
            ]
            if not options:
                unresolved[side].append(int(state_id))
                continue
            best = min(
                options,
                key=lambda row: _selection_key(
                    row,
                    train_field="native_train_NMSE_dB",
                    b_field="native_B_NMSE_dB",
                ),
            )
            selected[(side, int(state_id))] = int(best["candidate_id"])
    return selected, unresolved


def _common_metrics(
    *,
    prepared: PreparedStatePairs,
    side: str,
    candidate: AdaptiveCandidate,
    theta: np.ndarray,
    common_delay: int,
) -> tuple[float, float]:
    canonical = prepared.a_end if side == "Aend" else prepared.c2
    train_name = "A" if side == "Aend" else "C"
    train = canonical[train_name]
    b = canonical["B"]
    native_train_pred = (
        build_mp_basis(train.input, candidate.orders, candidate.memory_definition) @ theta
    )
    native_b_pred = build_mp_basis(b.input, candidate.orders, candidate.memory_definition) @ theta
    offset = int(common_delay) - int(candidate.max_delay)
    if offset < 0:
        raise ValueError("common support cannot precede candidate native support")
    train_pred = native_train_pred[offset:]
    b_pred = native_b_pred[offset:]
    train_ref = np.asarray(train.output[common_delay:], dtype=np.complex128)
    b_ref = np.asarray(b.output[common_delay:], dtype=np.complex128)
    if train_pred.shape != train_ref.shape or b_pred.shape != b_ref.shape:
        raise RuntimeError("common support prediction/reference length mismatch")
    train_nmse = calculate_nmse(train_ref, train_pred)
    b_nmse = calculate_nmse(b_ref, b_pred)
    if not np.isfinite(train_nmse) or not np.isfinite(b_nmse):
        raise ValueError("common-support NMSE is not finite")
    return float(train_nmse), float(b_nmse)


def select_common_candidates(
    *,
    records: Mapping[tuple[str, int, int], Mapping[str, Any]],
    candidates_by_id: Mapping[int, AdaptiveCandidate],
    theta_getter: Any,
    prepared: PreparedStatePairs,
    state_ids: Iterable[int],
    common_delay: int,
) -> tuple[
    dict[tuple[str, int], int],
    dict[str, list[int]],
    dict[tuple[str, int, int], tuple[float, float]],
]:
    """Select among native-feasible candidates using common-support metrics."""

    selected: dict[tuple[str, int], int] = {}
    unresolved = {"Aend": [], "C2": []}
    metrics_cache: dict[tuple[str, int, int], tuple[float, float]] = {}
    for side in ("Aend", "C2"):
        train_field = "common_train_NMSE_dB"
        b_field = "common_B_NMSE_dB"
        for state_id in state_ids:
            options = [
                row
                for (row_side, row_state, _), row in records.items()
                if row_side == side and row_state == int(state_id) and native_feasible(row)
            ]
            if not options:
                unresolved[side].append(int(state_id))
                continue
            prepared_for_state = prepared if prepared.state_id == int(state_id) else None
            if prepared_for_state is None:
                raise RuntimeError("prepared state does not match common-support selection")
            common_rows: list[dict[str, Any]] = []
            for row in options:
                candidate = candidates_by_id[int(row["candidate_id"])]
                theta = theta_getter(side, int(state_id), candidate.candidate_id)
                if theta is None:
                    continue
                cache_key = (side, int(state_id), candidate.candidate_id)
                try:
                    train_nmse, b_nmse = _common_metrics(
                        prepared=prepared_for_state,
                        side=side,
                        candidate=candidate,
                        theta=theta,
                        common_delay=common_delay,
                    )
                except Exception:
                    continue
                metrics_cache[cache_key] = (train_nmse, b_nmse)
                if train_nmse < DPD_SHAREABLE_THRESHOLD_DB and b_nmse < DPD_SHAREABLE_THRESHOLD_DB:
                    common_row = dict(row)
                    common_row[train_field] = train_nmse
                    common_row[b_field] = b_nmse
                    common_rows.append(common_row)
            if not common_rows:
                unresolved[side].append(int(state_id))
                continue
            best = min(
                common_rows,
                key=lambda row: _selection_key(row, train_field=train_field, b_field=b_field),
            )
            selected[(side, int(state_id))] = int(best["candidate_id"])
    return selected, unresolved, metrics_cache


def selected_model_statistics(frame: pd.DataFrame) -> dict[str, dict[str, float | int | None]]:
    """Return transparent complexity statistics for the final selected bank."""

    result: dict[str, dict[str, float | int | None]] = {}
    for side, group in frame.groupby("side", sort=False):
        coeff = group["coefficient_count"].to_numpy(dtype=float)
        delay = group["max_delay"].to_numpy(dtype=float)
        lambdas = group["lambda"].to_numpy(dtype=float)
        result[str(side)] = {
            "coefficient_count_min": int(np.min(coeff)),
            "coefficient_count_median": float(np.median(coeff)),
            "coefficient_count_q95": float(np.quantile(coeff, 0.95)),
            "coefficient_count_max": int(np.max(coeff)),
            "max_delay_min": int(np.min(delay)),
            "max_delay_median": float(np.median(delay)),
            "max_delay_q95": float(np.quantile(delay, 0.95)),
            "max_delay_max": int(np.max(delay)),
            "lambda_min": float(np.min(lambdas)),
            "lambda_median": float(np.median(lambdas)),
            "lambda_q95": float(np.quantile(lambdas, 0.95)),
            "lambda_max": float(np.max(lambdas)),
        }
    return result


__all__ = [
    "AdaptiveCandidate",
    "BASELINE_LAMBDA",
    "BASELINE_MEMORY",
    "BASELINE_ORDERS",
    "C2_STAGE",
    "DPD_SHAREABLE_THRESHOLD_DB",
    "EXISTING_CANDIDATE_COUNT",
    "EXISTING_LAMBDA_GRID",
    "FORMAL_A_LENGTH",
    "FORMAL_B_LENGTH",
    "FORMAL_C_LENGTH",
    "PreparedStatePairs",
    "CandidateFit",
    "STATE_COUNT",
    "WAVEFORM_LENGTH",
    "candidate_grid",
    "expand_candidate_shell",
    "expanded_lambda_grid",
    "fit_candidates_for_state",
    "initial_candidate_bank",
    "native_feasible",
    "prepare_state_pairs",
    "select_common_candidates",
    "select_native_candidates",
    "selected_model_statistics",
]
