"""Unified odd-order MP capacity scan with state-level multiprocessing.

The module deliberately does not import the historical Ridge path.  It uses
the generic MATLAB-compatible ``build_mp_basis`` helper and ordinary complex
``numpy.linalg.lstsq`` with a uniform memory depth for every odd nonlinear
order.  A worker prepares one state once per shell and evaluates every new
candidate for both Aend and C2 before returning a Python result to the parent.
"""

# Imports intentionally follow the BLAS-thread environment setup below.
# ruff: noqa: E402,I001

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Prevent BLAS x process oversubscription before NumPy is imported by workers.
for _thread_var in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ[_thread_var] = "1"

import numpy as np
import pandas as pd

from behavior_model.basis import build_mp_basis
from behavior_model.evaluation import calculate_nmse
from data_manager import load_by_id
from signal_segmentation import build_partition_from_xin, get_ilc_pair, preprocess_full_pair

STATE_COUNT = 425
WAVEFORM_LENGTH = 24576
FORMAL_A_LENGTH = 12288
FORMAL_B_LENGTH = 4915
FORMAL_C_LENGTH = 7373
DEVELOPMENT_COUNT = 340
VALIDATION_COUNT = 85
THRESHOLD_DB = -40.0
DPD_SHAREABLE_THRESHOLD_DB = THRESHOLD_DB
C2_STAGE = 2

ORDER_MAXIMA_ROUND0 = (1, 3, 5, 7, 9, 11, 13, 15)
SUPPORTED_ODD_ORDER_MAXIMA = tuple(range(1, 102, 2))


@dataclass(frozen=True)
class OddOrderCandidate:
    """One unified structure candidate."""

    candidate_id: int
    P: int
    orders: tuple[int, ...]
    M: int
    max_delay: int
    coefficient_count: int
    search_round: int
    source: str

    @property
    def memory_definition(self) -> dict[int, int]:
        return {int(order): int(self.M) for order in self.orders}

    @property
    def basis_terms(self) -> tuple[tuple[int, int], ...]:
        return tuple((int(order), int(memory)) for order in self.orders for memory in range(self.M))

    @property
    def key(self) -> tuple[int, int]:
        return self.P, self.M


@dataclass(frozen=True)
class PreparedCanonicalState:
    """One state after the frozen full-record preprocessing path."""

    state_id: int
    ilc_A_end: int
    a_end: Any
    c2: Any


@dataclass(frozen=True)
class ShellResult:
    """All candidate/state rows from one parallel shell."""

    search_round: int
    candidates: tuple[OddOrderCandidate, ...]
    rows: tuple[dict[str, Any], ...]
    theta_a: Mapping[tuple[int, int], np.ndarray]
    theta_c: Mapping[tuple[int, int], np.ndarray]
    elapsed_seconds: float
    completed_states: int
    worker_count: int


def odd_orders(P: int) -> tuple[int, ...]:
    """Return the contiguous odd-order sequence [1, 3, ..., P]."""

    if not isinstance(P, (int, np.integer)) or isinstance(P, bool) or int(P) < 1 or int(P) % 2 == 0:
        raise ValueError("P must be a positive odd integer")
    return tuple(range(1, int(P) + 1, 2))


def coefficient_count(P: int, M: int) -> int:
    """Return K(P,M)=((P+1)/2)*M for uniform memory."""

    if not isinstance(M, (int, np.integer)) or isinstance(M, bool) or int(M) < 1:
        raise ValueError("M must be a positive integer")
    return len(odd_orders(P)) * int(M)


def candidate_grid_rows(candidates: Sequence[OddOrderCandidate]) -> pd.DataFrame:
    rows = [
        {
            "candidate_id": candidate.candidate_id,
            "P": candidate.P,
            "orders": json.dumps(list(candidate.orders), separators=(",", ":")),
            "M": candidate.M,
            "max_delay": candidate.max_delay,
            "coefficient_count": candidate.coefficient_count,
            "search_round": candidate.search_round,
            "source": candidate.source,
            "ridge_used": False,
            "lambda": None,
            "even_orders_used": False,
        }
        for candidate in candidates
    ]
    return pd.DataFrame(rows).sort_values("candidate_id").reset_index(drop=True)


def _candidate_key(candidate: OddOrderCandidate) -> tuple[Any, ...]:
    return candidate.P, candidate.M


def generate_shell(
    existing: Sequence[OddOrderCandidate], search_round: int
) -> tuple[OddOrderCandidate, ...]:
    """Generate a new P/M shell without repeating earlier candidates.

    Round 0 is defined separately as P<=15 and M<=5.  Each later shell adds
    two new order maxima and a five-deep memory band.  The shell can continue
    until data support or the Development saturation rule provides a stop.
    """

    if search_round < 0:
        raise ValueError("search_round must be non-negative")
    if search_round == 0:
        p_values = ORDER_MAXIMA_ROUND0
        m_values = range(1, 6)
        source = "round_0_initial_P1_to_P15_M1_to_M5"
    else:
        previous_p_max = 15 + 4 * (search_round - 1)
        p_values = tuple(
            p for p in SUPPORTED_ODD_ORDER_MAXIMA if previous_p_max < p <= previous_p_max + 4
        )
        m_low = 1 + 5 * search_round
        m_high = m_low + 4
        # Add the new memory band for all P values available by this shell.
        m_values = range(m_low, m_high + 1)
        source = f"progressive_shell_{search_round}"
    existing_keys = {_candidate_key(candidate) for candidate in existing}
    next_id = max((candidate.candidate_id for candidate in existing), default=0) + 1
    result: list[OddOrderCandidate] = []
    if search_round == 0:
        order_prefixes = p_values
        for P in order_prefixes:
            for M in m_values:
                candidate = OddOrderCandidate(
                    candidate_id=next_id,
                    P=int(P),
                    orders=odd_orders(int(P)),
                    M=int(M),
                    max_delay=int(M) - 1,
                    coefficient_count=coefficient_count(int(P), int(M)),
                    search_round=search_round,
                    source=source,
                )
                next_id += 1
                result.append(candidate)
        return tuple(result)

    # New high-order prefixes use all memory values already tested up to the
    # current band; the memory band also extends lower-order prefixes.
    available_p = tuple(p for p in SUPPORTED_ODD_ORDER_MAXIMA if p <= 15 + 4 * search_round)
    tested_m_max = 5 * search_round + 5
    for P in p_values:
        for M in range(1, tested_m_max + 1):
            candidate = OddOrderCandidate(
                candidate_id=next_id,
                P=int(P),
                orders=odd_orders(int(P)),
                M=int(M),
                max_delay=int(M) - 1,
                coefficient_count=coefficient_count(int(P), int(M)),
                search_round=search_round,
                source=source,
            )
            next_id += 1
            if _candidate_key(candidate) not in existing_keys:
                existing_keys.add(_candidate_key(candidate))
                result.append(candidate)
    for P in available_p:
        for M in m_values:
            candidate = OddOrderCandidate(
                candidate_id=next_id,
                P=int(P),
                orders=odd_orders(int(P)),
                M=int(M),
                max_delay=int(M) - 1,
                coefficient_count=coefficient_count(int(P), int(M)),
                search_round=search_round,
                source=source,
            )
            next_id += 1
            if _candidate_key(candidate) not in existing_keys:
                existing_keys.add(_candidate_key(candidate))
                result.append(candidate)
    return tuple(result)


def load_frozen_split(path: Path) -> pd.DataFrame:
    """Load the existing 340/85 split without re-randomising it."""

    frame = pd.read_csv(path)
    required = {"state_id", "split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"split definition missing fields: {missing}")
    frame = frame.sort_values("state_id").reset_index(drop=True)
    if frame.shape[0] != STATE_COUNT or not np.array_equal(
        frame["state_id"].to_numpy(dtype=np.int64), np.arange(STATE_COUNT, dtype=np.int64)
    ):
        raise ValueError("frozen split must cover state_id 0...424")
    normalized = (
        frame["split"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({"development": "Development", "validation": "Validation"})
    )
    if set(normalized) != {"Development", "Validation"}:
        raise ValueError("split must contain Development and Validation only")
    frame["split"] = normalized
    if int((frame["split"] == "Development").sum()) != DEVELOPMENT_COUNT:
        raise ValueError("Development must contain 340 states")
    if int((frame["split"] == "Validation").sum()) != VALIDATION_COUNT:
        raise ValueError("Validation must contain 85 states")
    if set(frame.loc[frame["split"] == "Development", "state_id"]).intersection(
        set(frame.loc[frame["split"] == "Validation", "state_id"])
    ):
        raise ValueError("Development and Validation overlap")
    return frame


def prepare_canonical_state(state_id: int) -> PreparedCanonicalState:
    """Read one state and execute the frozen canonical preprocessing."""

    state_id = int(state_id)
    data = load_by_id(state_id)
    xin = np.asarray(data["xin"])
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
        raise RuntimeError("frozen ABC ownership changed")
    input_history = np.asarray(data["xin_pd_ori_ilc"])
    output_history = np.asarray(data["yout_withdpd_ori_ilc"])
    if input_history.ndim != 2 or output_history.shape != input_history.shape:
        raise ValueError(f"state={state_id} ILC histories have different shapes")
    if input_history.shape[0] != WAVEFORM_LENGTH or input_history.shape[1] < C2_STAGE:
        raise RuntimeError(f"state={state_id} lacks ILC column 2")
    ilc_count = int(input_history.shape[1])
    a_pair = get_ilc_pair(data, ilc_count - 1)
    a_end = preprocess_full_pair(
        a_pair.input_full,
        a_pair.output_raw_full,
        partition,
        pair_type="ILC",
        iteration_index=ilc_count - 1,
        input_peak_normalization_factor=a_pair.input_peak_normalization_factor,
    )
    c_pair = get_ilc_pair(data, C2_STAGE - 1)
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
    return PreparedCanonicalState(state_id=state_id, ilc_A_end=ilc_count, a_end=a_end, c2=c2)


def _full_basis(
    segment: Any,
    full_orders: Sequence[int],
    max_memory: int,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(segment.input, dtype=np.complex128)
    y = np.asarray(segment.output, dtype=np.complex128)
    terms = tuple(
        (int(order), int(memory)) for order in full_orders for memory in range(int(max_memory))
    )
    # Keep original-time row mapping.  The first max_memory-1 rows are
    # intentionally NaN only for columns whose delay is not yet available;
    # candidate-specific slicing starts at its own M-1 rows.
    phi = np.full((x.size, len(terms)), np.nan + 0j, dtype=np.complex128)
    for column, (order, memory) in enumerate(terms):
        delayed = x[: x.size - memory]
        phi[memory:, column] = delayed * np.abs(delayed) ** (order - 1)
    if phi.shape[0] != y.size or not np.all(np.isfinite(y)):
        raise RuntimeError("full basis/target contains invalid values")
    return phi, y


def direct_basis_for_candidate(
    segment: Any, candidate: OddOrderCandidate
) -> tuple[np.ndarray, np.ndarray]:
    """Build a candidate basis directly, mainly for regression tests."""

    phi = build_mp_basis(segment.input, candidate.orders, candidate.memory_definition)
    y = np.asarray(segment.output[candidate.max_delay :], dtype=np.complex128)
    return phi, y


def slice_basis_from_max(
    phi_full: np.ndarray,
    y_full: np.ndarray,
    full_orders: Sequence[int],
    max_memory: int,
    candidate: OddOrderCandidate,
) -> tuple[np.ndarray, np.ndarray]:
    """Slice a max-(P,M) basis while restoring candidate-specific row support."""

    terms = tuple(
        (int(order), int(memory)) for order in full_orders for memory in range(int(max_memory))
    )
    term_to_column = {term: index for index, term in enumerate(terms)}
    columns = tuple(term_to_column[term] for term in candidate.basis_terms)
    phi = np.asarray(phi_full[candidate.max_delay :, :][:, columns], dtype=np.complex128)
    y = np.asarray(y_full[candidate.max_delay :], dtype=np.complex128)
    if phi.shape[0] != y.size:
        raise RuntimeError("sliced basis/target row mismatch")
    return phi, y


def _fit_candidate_from_basis(
    phi: np.ndarray,
    y: np.ndarray,
    candidate: OddOrderCandidate,
    side: str,
    state_id: int,
    segment_name: str,
) -> tuple[dict[str, Any], np.ndarray | None]:
    row: dict[str, Any] = {
        "candidate_id": candidate.candidate_id,
        "P": candidate.P,
        "orders": json.dumps(list(candidate.orders), separators=(",", ":")),
        "M": candidate.M,
        "max_delay": candidate.max_delay,
        "coefficient_count": candidate.coefficient_count,
        "search_round": candidate.search_round,
        "state_id": state_id,
        "split": "",
        "side": side,
        "segment": segment_name,
        "train_NMSE_dB": np.nan,
        "B_NMSE_dB": np.nan,
        "rank": 0,
        "rank_ratio": np.nan,
        "condition_number": np.nan,
        "n_train_samples": int(phi.shape[0]),
        "finite": False,
        "valid": False,
        "failure_reason": "",
    }
    try:
        if phi.shape[0] <= phi.shape[1]:
            raise ValueError("valid rows are not greater than coefficient count")
        theta, _, rank, singular = np.linalg.lstsq(phi, y, rcond=None)
        if int(rank) != candidate.coefficient_count:
            raise ValueError(f"rank deficient {rank}/{candidate.coefficient_count}")
        if not np.all(np.isfinite(theta)):
            raise ValueError("theta contains NaN/Inf")
        prediction = phi @ theta
        train_nmse = calculate_nmse(y, prediction)
        if not np.isfinite(train_nmse):
            raise ValueError("train NMSE is not finite")
        singular = np.asarray(singular, dtype=float)
        condition = float(singular[0] / singular[-1]) if singular[-1] > 0 else math.inf
        row.update(
            {
                "train_NMSE_dB": float(train_nmse),
                "rank": int(rank),
                "rank_ratio": float(rank / candidate.coefficient_count),
                "condition_number": condition,
                "finite": True,
                "valid": bool(np.isfinite(condition)),
                "failure_reason": "" if np.isfinite(condition) else "condition_nonfinite",
            }
        )
        if not row["valid"]:
            return row, None
        return row, np.asarray(theta, dtype=np.complex128)
    except Exception as exc:
        row["failure_reason"] = f"{type(exc).__name__}: {exc}"
        return row, None


def _evaluate_state_shell_impl(
    state_id: int,
    candidates: Sequence[OddOrderCandidate],
    split_label: str = "",
) -> dict[str, Any]:
    prepared = prepare_canonical_state(int(state_id))
    if not candidates:
        raise ValueError("candidate shell is empty")
    max_P = max(candidate.P for candidate in candidates)
    max_M = max(candidate.M for candidate in candidates)
    full_orders = odd_orders(max_P)
    phi_a, y_a = _full_basis(prepared.a_end["A"], full_orders, max_M)
    phi_ab, y_ab = _full_basis(prepared.a_end["B"], full_orders, max_M)
    phi_c, y_c = _full_basis(prepared.c2["C"], full_orders, max_M)
    phi_cb, y_cb = _full_basis(prepared.c2["B"], full_orders, max_M)
    rows: list[dict[str, Any]] = []
    theta_a: dict[tuple[int, int], np.ndarray] = {}
    theta_c: dict[tuple[int, int], np.ndarray] = {}
    for candidate in candidates:
        a_phi, a_y = slice_basis_from_max(phi_a, y_a, full_orders, max_M, candidate)
        ab_phi, ab_y = slice_basis_from_max(phi_ab, y_ab, full_orders, max_M, candidate)
        c_phi, c_y = slice_basis_from_max(phi_c, y_c, full_orders, max_M, candidate)
        cb_phi, cb_y = slice_basis_from_max(phi_cb, y_cb, full_orders, max_M, candidate)
        a_row, a_theta = _fit_candidate_from_basis(
            a_phi, a_y, candidate, "Aend", prepared.state_id, "A"
        )
        c_row, c_theta = _fit_candidate_from_basis(
            c_phi, c_y, candidate, "C2", prepared.state_id, "C"
        )
        a_row["split"] = split_label
        c_row["split"] = split_label
        if a_theta is not None:
            a_b_prediction = ab_phi @ a_theta
            a_b_nmse = calculate_nmse(ab_y, a_b_prediction)
            a_row["B_NMSE_dB"] = float(a_b_nmse)
            if not np.isfinite(a_b_nmse):
                a_row["valid"] = False
                a_row["failure_reason"] = "B prediction NMSE nonfinite"
            else:
                theta_a[(prepared.state_id, candidate.candidate_id)] = a_theta
        if c_theta is not None:
            c_b_prediction = cb_phi @ c_theta
            c_b_nmse = calculate_nmse(cb_y, c_b_prediction)
            c_row["B_NMSE_dB"] = float(c_b_nmse)
            if not np.isfinite(c_b_nmse):
                c_row["valid"] = False
                c_row["failure_reason"] = "B prediction NMSE nonfinite"
            else:
                theta_c[(prepared.state_id, candidate.candidate_id)] = c_theta
        rows.extend((a_row, c_row))
    return {
        "state_id": prepared.state_id,
        "ilc_A_end": prepared.ilc_A_end,
        "rows": rows,
        "theta_a": theta_a,
        "theta_c": theta_c,
    }


def evaluate_state_shell_serial(
    state_id: int,
    candidates: Sequence[OddOrderCandidate],
    split_label: str = "",
) -> dict[str, Any]:
    """Serial reference path used by the parallel regression test."""

    return _evaluate_state_shell_impl(state_id, candidates, split_label)


def _worker_initializer() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"


def _worker_entry(payload: tuple[int, tuple[OddOrderCandidate, ...], str]) -> dict[str, Any]:
    state_id, candidates, split_label = payload
    return _evaluate_state_shell_impl(state_id, candidates, split_label)


def target_worker_count(logical_cpu_count: int | None = None) -> int:
    logical = int(logical_cpu_count or os.cpu_count() or 1)
    return max(1, int(math.floor(0.90 * logical)))


def _rows_to_frame(rows: Iterable[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(list(rows))
    if frame.empty:
        return frame
    return frame.sort_values(["candidate_id", "state_id", "side"]).reset_index(drop=True)


def evaluate_shell_parallel(
    candidates: Sequence[OddOrderCandidate],
    split_frame: pd.DataFrame,
    *,
    worker_count: int | None = None,
    completed_state_ids: Iterable[int] = (),
    progress_callback: Any | None = None,
    state_result_callback: Any | None = None,
) -> ShellResult:
    """Evaluate a shell with State-level ProcessPool parallelism."""

    if not candidates:
        raise ValueError("candidate shell is empty")
    logical = int(os.cpu_count() or 1)
    workers = int(worker_count or target_worker_count(logical))
    workers = max(1, min(workers, STATE_COUNT))
    completed = {int(value) for value in completed_state_ids}
    tasks = [
        (
            int(row.state_id),
            tuple(candidates),
            str(row.split),
        )
        for row in split_frame.sort_values("state_id").itertuples(index=False)
        if int(row.state_id) not in completed
    ]
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    theta_a: dict[tuple[int, int], np.ndarray] = {}
    theta_c: dict[tuple[int, int], np.ndarray] = {}
    done = len(completed)
    if not tasks:
        return ShellResult(
            search_round=candidates[0].search_round,
            candidates=tuple(candidates),
            rows=tuple(),
            theta_a=theta_a,
            theta_c=theta_c,
            elapsed_seconds=0.0,
            completed_states=done,
            worker_count=workers,
        )
    with ProcessPoolExecutor(max_workers=workers, initializer=_worker_initializer) as executor:
        futures = [executor.submit(_worker_entry, task) for task in tasks]
        for future in futures:
            result = future.result()
            rows.extend(result["rows"])
            theta_a.update(result["theta_a"])
            theta_c.update(result["theta_c"])
            done += 1
            if state_result_callback is not None:
                state_result_callback(result, done, time.perf_counter() - started)
            if progress_callback is not None:
                progress_callback(done, STATE_COUNT, time.perf_counter() - started)
    elapsed = time.perf_counter() - started
    return ShellResult(
        search_round=candidates[0].search_round,
        candidates=tuple(candidates),
        rows=tuple(rows),
        theta_a=theta_a,
        theta_c=theta_c,
        elapsed_seconds=float(elapsed),
        completed_states=done,
        worker_count=workers,
    )


def add_pass_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["Aend_joint_lt_minus40"] = (
        (result["side"] == "Aend")
        & (result["train_NMSE_dB"].astype(float) < THRESHOLD_DB)
        & (result["B_NMSE_dB"].astype(float) < THRESHOLD_DB)
    )
    result["C2_joint_lt_minus40"] = (
        (result["side"] == "C2")
        & (result["train_NMSE_dB"].astype(float) < THRESHOLD_DB)
        & (result["B_NMSE_dB"].astype(float) < THRESHOLD_DB)
    )
    result["all_four_lt_minus40"] = False
    for candidate_id, group in result.groupby("candidate_id", sort=False):
        by_state = group.set_index(["state_id", "side"])
        for state_id in sorted(group["state_id"].unique()):
            try:
                a = by_state.loc[(state_id, "Aend")]
                c = by_state.loc[(state_id, "C2")]
                passed = bool(
                    float(a["train_NMSE_dB"]) < THRESHOLD_DB
                    and float(a["B_NMSE_dB"]) < THRESHOLD_DB
                    and float(c["train_NMSE_dB"]) < THRESHOLD_DB
                    and float(c["B_NMSE_dB"]) < THRESHOLD_DB
                )
                result.loc[
                    (result["candidate_id"] == candidate_id) & (result["state_id"] == state_id),
                    "all_four_lt_minus40",
                ] = passed
            except KeyError:
                continue
    return result


def aggregate_candidate_summary(
    state_metrics: pd.DataFrame,
    candidates: Sequence[OddOrderCandidate],
    split_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate Development and Validation metrics per candidate."""

    frame = add_pass_columns(state_metrics)
    rows: list[dict[str, Any]] = []
    for candidate in candidates:
        candidate_rows = frame.loc[frame["candidate_id"] == candidate.candidate_id].copy()
        row: dict[str, Any] = {
            "candidate_id": candidate.candidate_id,
            "P": candidate.P,
            "orders": json.dumps(list(candidate.orders), separators=(",", ":")),
            "M": candidate.M,
            "max_delay": candidate.max_delay,
            "coefficient_count": candidate.coefficient_count,
            "search_round": candidate.search_round,
            "ridge_used": False,
            "lambda": np.nan,
            "valid": bool(candidate_rows["valid"].all()) if not candidate_rows.empty else False,
        }
        for split_name in ("Development", "Validation"):
            split_rows = candidate_rows.loc[candidate_rows["split"] == split_name]
            expected = DEVELOPMENT_COUNT if split_name == "Development" else VALIDATION_COUNT
            for side in ("Aend", "C2"):
                side_rows = split_rows.loc[split_rows["side"] == side]
                for metric, short in (("train_NMSE_dB", "train"), ("B_NMSE_dB", "B")):
                    values = side_rows[metric].to_numpy(dtype=float)
                    prefix = "dev" if split_name == "Development" else "val"
                    label = "Aend" if side == "Aend" else "C2"
                    if values.size != expected:
                        row[f"{prefix}_{label}_{short}_count"] = int(values.size)
                        for stat in (
                            "mean",
                            "median",
                            "std",
                            "min",
                            "max",
                            "q05",
                            "q25",
                            "q75",
                            "q95",
                            "worst",
                        ):
                            row[f"{prefix}_{label}_{short}_{stat}"] = np.nan
                    else:
                        row[f"{prefix}_{label}_{short}_count"] = int(values.size)
                        row[f"{prefix}_{label}_{short}_mean"] = float(np.mean(values))
                        row[f"{prefix}_{label}_{short}_median"] = float(np.median(values))
                        row[f"{prefix}_{label}_{short}_std"] = float(np.std(values))
                        row[f"{prefix}_{label}_{short}_min"] = float(np.min(values))
                        row[f"{prefix}_{label}_{short}_max"] = float(np.max(values))
                        row[f"{prefix}_{label}_{short}_q05"] = float(np.quantile(values, 0.05))
                        row[f"{prefix}_{label}_{short}_q25"] = float(np.quantile(values, 0.25))
                        row[f"{prefix}_{label}_{short}_q75"] = float(np.quantile(values, 0.75))
                        row[f"{prefix}_{label}_{short}_q95"] = float(np.quantile(values, 0.95))
                        row[f"{prefix}_{label}_{short}_worst"] = float(np.max(values))
            prefix = "dev" if split_name == "Development" else "val"
            a = split_rows.loc[split_rows["side"] == "Aend"]
            c = split_rows.loc[split_rows["side"] == "C2"]
            a_joint = (a["train_NMSE_dB"] < THRESHOLD_DB) & (a["B_NMSE_dB"] < THRESHOLD_DB)
            c_joint = (c["train_NMSE_dB"] < THRESHOLD_DB) & (c["B_NMSE_dB"] < THRESHOLD_DB)
            row[f"{prefix}_Aend_joint_pass"] = int(a_joint.sum())
            row[f"{prefix}_C2_joint_pass"] = int(c_joint.sum())
            state_worst: list[float] = []
            for state_id in split_rows["state_id"].unique():
                values = split_rows.loc[
                    split_rows["state_id"] == state_id, ["train_NMSE_dB", "B_NMSE_dB"]
                ].to_numpy(dtype=float)
                if values.shape == (2, 2):
                    state_worst.append(float(np.max(values)))
            row[f"{prefix}_all_four_pass"] = int(
                np.count_nonzero(
                    split_rows.groupby("state_id")["all_four_lt_minus40"].max().to_numpy(dtype=bool)
                )
            )
            row[f"{prefix}_worst4_mean"] = float(np.mean(state_worst)) if state_worst else np.nan
            row[f"{prefix}_worst4_median"] = (
                float(np.median(state_worst)) if state_worst else np.nan
            )
            row[f"{prefix}_worst4_q95"] = (
                float(np.quantile(state_worst, 0.95)) if state_worst else np.nan
            )
            row[f"{prefix}_worst4_max"] = float(np.max(state_worst)) if state_worst else np.nan
        rows.append(row)
    return pd.DataFrame(rows).sort_values("candidate_id").reset_index(drop=True)


def choose_unified_candidate(summary: pd.DataFrame) -> dict[str, Any]:
    """Apply the frozen Development-only unified-model selection rule."""

    if summary.empty:
        raise ValueError("candidate summary is empty")
    valid = summary.loc[summary["valid"].astype(bool)].copy()
    if valid.empty:
        raise RuntimeError("no valid odd-order candidate")
    if int(valid["dev_all_four_pass"].max()) == DEVELOPMENT_COUNT:
        pool = valid.loc[valid["dev_all_four_pass"] == DEVELOPMENT_COUNT].copy()
        pool = pool.sort_values(
            ["coefficient_count", "M", "P", "dev_worst4_median", "dev_worst4_q95", "candidate_id"],
            ascending=[True, True, True, True, True, True],
            kind="mergesort",
        )
        rule = "minimum_complexity_among_dev_340_all_four_pass"
    else:
        pool = valid.sort_values(
            [
                "dev_all_four_pass",
                "dev_C2_joint_pass",
                "dev_Aend_joint_pass",
                "dev_worst4_median",
                "dev_C2_B_median",
                "coefficient_count",
                "M",
                "P",
                "candidate_id",
            ],
            ascending=[False, False, False, True, True, True, True, True, True],
            kind="mergesort",
        )
        rule = "max_dev_all_four_then_C2_Aend_then_worst4_then_C2_B_then_complexity"
    chosen = pool.iloc[0].to_dict()
    chosen["selection_rule"] = rule
    chosen["selection_based_on"] = "Development_only"
    return chosen


def selected_metrics_for_candidate(
    state_metrics: pd.DataFrame,
    candidate_id: int,
    split_frame: pd.DataFrame,
) -> pd.DataFrame:
    result = state_metrics.loc[state_metrics["candidate_id"] == int(candidate_id)].copy()
    split_lookup = split_frame.set_index("state_id")["split"]
    result["split"] = result["state_id"].map(split_lookup)
    return result.sort_values(["state_id", "side"]).reset_index(drop=True)


__all__ = [
    "C2_STAGE",
    "DEVELOPMENT_COUNT",
    "DPD_SHAREABLE_THRESHOLD_DB",
    "FORMAL_A_LENGTH",
    "FORMAL_B_LENGTH",
    "FORMAL_C_LENGTH",
    "OddOrderCandidate",
    "PreparedCanonicalState",
    "SUPPORTED_ODD_ORDER_MAXIMA",
    "STATE_COUNT",
    "THRESHOLD_DB",
    "VALIDATION_COUNT",
    "WAVEFORM_LENGTH",
    "add_pass_columns",
    "aggregate_candidate_summary",
    "candidate_grid_rows",
    "choose_unified_candidate",
    "coefficient_count",
    "direct_basis_for_candidate",
    "evaluate_shell_parallel",
    "evaluate_state_shell_serial",
    "generate_shell",
    "load_frozen_split",
    "odd_orders",
    "prepare_canonical_state",
    "selected_metrics_for_candidate",
    "slice_basis_from_max",
    "target_worker_count",
]
