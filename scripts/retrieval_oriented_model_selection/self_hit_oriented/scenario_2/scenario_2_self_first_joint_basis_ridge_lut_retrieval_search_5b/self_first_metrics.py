# ruff: noqa: E501

"""Metrics and deterministic ranking for the Self-First retrieval objective.

The module deliberately keeps fingerprint distance (the quantity that selects
``State_Q``) separate from the Real-B matrix (the quantity that labels a
non-self result as a shareable fallback).  It is shared by the long runner and
its independent validator.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

STATE_COUNT = 425
TIE_TOLERANCE_DB = 1e-12
REAL_B_THRESHOLD_DB = -40.0
CLASS_NAMES = ("SELF", "SHAREABLE_FALLBACK", "FAIL")


def canonical_support_hash(basis_ids: Sequence[str]) -> str:
    """Hash a canonical ordered tuple of basis IDs."""

    payload = json.dumps([str(item) for item in basis_ids], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a stable SHA256 for a JSON-serializable value."""

    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _finite_quantile(values: np.ndarray, quantile: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.quantile(finite, quantile)) if finite.size else float("nan")


def _finite_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _safe_difference(left: float, right: float) -> float:
    """Subtract d_other-d_self while preserving meaningful -Inf ties."""

    if np.isneginf(left) and np.isneginf(right):
        return 0.0
    if np.isneginf(right) and np.isfinite(left):
        return float("inf")
    if np.isfinite(left) and np.isfinite(right):
        return float(left - right)
    return float("nan")


def top1_from_distance(distance: np.ndarray) -> dict[str, np.ndarray]:
    """Apply the project tie rule: minimum distance, then smallest State ID."""

    matrix = np.asarray(distance, dtype=np.float64)
    if matrix.shape != (STATE_COUNT, STATE_COUNT):
        raise ValueError(f"distance must have shape {(STATE_COUNT, STATE_COUNT)}, got {matrix.shape}")
    if np.isnan(matrix).any() or np.isposinf(matrix).any():
        raise ValueError("distance contains NaN or +Inf")

    state_ids = np.arange(STATE_COUNT, dtype=np.int64)
    selected = np.empty(STATE_COUNT, dtype=np.int64)
    selected_distance = np.empty(STATE_COUNT, dtype=np.float64)
    true_rank = np.empty(STATE_COUNT, dtype=np.int64)
    tie_count = np.empty(STATE_COUNT, dtype=np.int64)
    for state_id in range(STATE_COUNT):
        row = matrix[state_id]
        minimum = float(np.min(row))
        tied = np.flatnonzero(row <= minimum + TIE_TOLERANCE_DB)
        selected[state_id] = int(np.min(tied))
        selected_distance[state_id] = float(row[selected[state_id]])
        tie_count[state_id] = int(tied.size)
        order = np.lexsort((state_ids, row))
        true_rank[state_id] = int(np.flatnonzero(order == state_id)[0] + 1)
    return {
        "selected": selected,
        "selected_distance": selected_distance,
        "true_rank": true_rank,
        "tie_count": tie_count,
    }


def _class_index(name: str) -> int:
    try:
        return CLASS_NAMES.index(name)
    except ValueError as exc:
        raise ValueError(f"unknown retrieval class: {name}") from exc


def evaluate_distance(
    distance: np.ndarray,
    real_b_distance: np.ndarray,
    shareability: np.ndarray,
    *,
    candidate: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], np.ndarray]:
    """Evaluate one 425x425 distance matrix under Self-First semantics.

    ``real_b_distance`` is only used after the fingerprint Top-1 has been
    frozen.  The returned state frame contains all three classes and the
    self/fallback margins and ranks used by the lexicographic ranking.
    """

    matrix = np.asarray(distance, dtype=np.float64)
    real_b = np.asarray(real_b_distance, dtype=np.float64)
    share = np.asarray(shareability, dtype=bool)
    if matrix.shape != (STATE_COUNT, STATE_COUNT):
        raise ValueError("fingerprint distance matrix shape is not 425x425")
    if real_b.shape != matrix.shape or share.shape != matrix.shape:
        raise ValueError("Real-B matrix/shareability shape mismatch")
    if np.isnan(matrix).any() or np.isposinf(matrix).any():
        raise ValueError("fingerprint distance contains NaN or +Inf")
    if np.isnan(real_b).any() or np.isposinf(real_b).any():
        raise ValueError("Real-B distance contains NaN or +Inf")
    if not np.all(np.isneginf(np.diag(real_b))):
        raise ValueError("Real-B diagonal must be -Inf")

    top1 = top1_from_distance(matrix)
    state_ids = np.arange(STATE_COUNT, dtype=np.int64)
    selected = top1["selected"]
    exact = selected == state_ids
    fallback_mask = share.copy()
    np.fill_diagonal(fallback_mask, False)
    selected_fallback = fallback_mask[state_ids, selected]
    classes = np.full(STATE_COUNT, "FAIL", dtype=object)
    classes[exact] = "SELF"
    classes[~exact & selected_fallback] = "SHAREABLE_FALLBACK"

    self_distance = matrix[state_ids, state_ids]
    self_margin = np.empty(STATE_COUNT, dtype=np.float64)
    fallback_margin = np.full(STATE_COUNT, np.nan, dtype=np.float64)
    fallback_rank = np.zeros(STATE_COUNT, dtype=np.int64)
    fallback_candidate_count = np.zeros(STATE_COUNT, dtype=np.int64)
    fallback_candidate_ids: list[str] = []
    nearest_fallback_distance = np.full(STATE_COUNT, np.nan, dtype=np.float64)
    nearest_nonshareable_distance = np.full(STATE_COUNT, np.nan, dtype=np.float64)
    retrieved_real_b = real_b[state_ids, selected]
    for state_id in range(STATE_COUNT):
        row = matrix[state_id]
        nonself = row.copy()
        nonself[state_id] = np.inf
        nearest_other = float(np.min(nonself))
        self_margin[state_id] = _safe_difference(nearest_other, float(self_distance[state_id]))

        fallback_ids = np.flatnonzero(fallback_mask[state_id])
        fallback_candidate_count[state_id] = int(fallback_ids.size)
        fallback_candidate_ids.append(json.dumps(fallback_ids.astype(int).tolist()))
        order_nonself = np.lexsort((state_ids, nonself))
        order_nonself = order_nonself[order_nonself != state_id]
        shareable_in_order = [int(item) for item in order_nonself if fallback_mask[state_id, item]]
        if shareable_in_order:
            first_id = shareable_in_order[0]
            fallback_rank[state_id] = int(np.flatnonzero(order_nonself == first_id)[0] + 1)
            nearest_fallback_distance[state_id] = float(row[first_id])
            nonshareable = [int(item) for item in order_nonself if not fallback_mask[state_id, item]]
            if nonshareable:
                nearest_nonshareable_distance[state_id] = float(row[nonshareable[0]])
                fallback_margin[state_id] = _safe_difference(
                    nearest_nonshareable_distance[state_id], nearest_fallback_distance[state_id]
                )

    self_mrr = float(np.mean(1.0 / top1["true_rank"]))
    fallback_mrr = float(np.mean(np.divide(1.0, fallback_rank, out=np.zeros(STATE_COUNT, dtype=float), where=fallback_rank > 0)))
    self_only = fallback_candidate_count == 0
    summary: dict[str, Any] = {
        "candidate_id": int(candidate.get("candidate_id", -1)) if candidate else -1,
        "candidate_name": str(candidate.get("candidate_name", "")) if candidate else "",
        "support_hash": str(candidate.get("support_hash", "")) if candidate else "",
        "K": int(candidate.get("K", -1)) if candidate else -1,
        "lambda": float(candidate.get("lambda", np.nan)) if candidate else float("nan"),
        "N_self": int(np.count_nonzero(exact)),
        "N_fallback": int(np.count_nonzero(~exact & selected_fallback)),
        "N_valid": int(np.count_nonzero(exact) + np.count_nonzero(~exact & selected_fallback)),
        "N_fail": int(np.count_nonzero(classes == "FAIL")),
        "self_rate": float(np.mean(exact)),
        "valid_rate": float(np.mean(classes != "FAIL")),
        "fallback_success_given_self_miss": float(
            np.count_nonzero(~exact & selected_fallback) / max(1, np.count_nonzero(~exact))
        ),
        "self_Q01": _finite_quantile(self_margin, 0.01),
        "self_Q05": _finite_quantile(self_margin, 0.05),
        "self_Q10": _finite_quantile(self_margin, 0.10),
        "self_margin_median": _finite_quantile(self_margin, 0.50),
        "self_margin_mean": _finite_mean(self_margin),
        "self_margin_worst": float(np.nanmin(self_margin)) if np.any(np.isfinite(self_margin)) else float("nan"),
        "self_MRR": self_mrr,
        "fallback_Q05": _finite_quantile(fallback_margin, 0.05),
        "fallback_Q10": _finite_quantile(fallback_margin, 0.10),
        "fallback_margin_median": _finite_quantile(fallback_margin, 0.50),
        "fallback_margin_mean": _finite_mean(fallback_margin),
        "fallback_margin_worst": float(np.nanmin(fallback_margin)) if np.any(np.isfinite(fallback_margin)) else float("nan"),
        "fallback_MRR": fallback_mrr,
        "Self_Top1": int(np.count_nonzero(top1["true_rank"] <= 1)),
        "Self_Top2": int(np.count_nonzero(top1["true_rank"] <= 2)),
        "Self_Top3": int(np.count_nonzero(top1["true_rank"] <= 3)),
        "Self_Top5": int(np.count_nonzero(top1["true_rank"] <= 5)),
        "Self_Top10": int(np.count_nonzero(top1["true_rank"] <= 10)),
        "fallback_Top1": int(np.count_nonzero((fallback_rank > 0) & (fallback_rank <= 1))),
        "fallback_Top2": int(np.count_nonzero((fallback_rank > 0) & (fallback_rank <= 2))),
        "fallback_Top3": int(np.count_nonzero((fallback_rank > 0) & (fallback_rank <= 3))),
        "fallback_Top5": int(np.count_nonzero((fallback_rank > 0) & (fallback_rank <= 5))),
        "fallback_Top10": int(np.count_nonzero((fallback_rank > 0) & (fallback_rank <= 10))),
        "self_miss_state_ids": json.dumps(state_ids[~exact].astype(int).tolist()),
        "fail_state_ids": json.dumps(state_ids[classes == "FAIL"].astype(int).tolist()),
        "self_only_state_ids": json.dumps(state_ids[self_only].astype(int).tolist()),
        "self_only_state_count": int(np.count_nonzero(self_only)),
        "distance_minimum_tie_count_max": int(np.max(top1["tie_count"])),
        "retrieved_real_B_finite_count": int(np.count_nonzero(np.isfinite(retrieved_real_b))),
    }
    if summary["N_self"] + summary["N_fallback"] + summary["N_fail"] != STATE_COUNT:
        raise RuntimeError(f"Self-First class count identity failed: {summary}")
    summary["N_valid"] = int(summary["N_self"] + summary["N_fallback"])

    state_frame: dict[str, Any] = {
        "State_R": state_ids,
        "State_Q": selected,
        "retrieval_fingerprint_CNMSE_dB": top1["selected_distance"],
        "exact_hit": exact,
        "true_state_rank": top1["true_rank"],
        "minimum_tie_count": top1["tie_count"],
        "retrieved_real_B_CNMSE_dB": retrieved_real_b,
        "retrieval_class": classes,
        "self_distance_dB": self_distance,
        "self_margin_dB": self_margin,
        "fallback_available": fallback_candidate_count > 0,
        "fallback_candidate_count": fallback_candidate_count,
        "fallback_candidate_ids": fallback_candidate_ids,
        "fallback_rank": fallback_rank,
        "fallback_margin_dB": fallback_margin,
        "nearest_shareable_distance_dB": nearest_fallback_distance,
        "nearest_nonshareable_distance_dB": nearest_nonshareable_distance,
        "retrieved_real_B_pass": (~np.isnan(retrieved_real_b))
        & (~np.isposinf(retrieved_real_b))
        & (retrieved_real_b < REAL_B_THRESHOLD_DB),
        "self_only": self_only,
    }
    state_frame["realB_pass"] = state_frame["retrieved_real_B_pass"]
    return summary, state_frame, np.asarray(classes, dtype=object)


def transition_matrix(parent_classes: Sequence[str], child_classes: Sequence[str]) -> dict[str, Any]:
    """Return complete 3x3 Parent→Child counts and State IDs."""

    parent = np.asarray(parent_classes, dtype=object)
    child = np.asarray(child_classes, dtype=object)
    if parent.shape != (STATE_COUNT,) or child.shape != (STATE_COUNT,):
        raise ValueError("transition classes must both have 425 entries")
    counts = np.zeros((3, 3), dtype=np.int64)
    state_ids: dict[str, list[int]] = {}
    for source in CLASS_NAMES:
        for target in CLASS_NAMES:
            mask = (parent == source) & (child == target)
            source_index = _class_index(source)
            target_index = _class_index(target)
            counts[source_index, target_index] = int(np.count_nonzero(mask))
            state_ids[f"{source}_TO_{target}"] = np.flatnonzero(mask).astype(int).tolist()
    return {"class_names": list(CLASS_NAMES), "counts": counts.astype(int).tolist(), "state_ids": state_ids}


def lexicographic_key(row: Mapping[str, Any]) -> tuple[Any, ...]:
    """Key for the single Self-First ranking used everywhere."""

    def descending(name: str) -> float:
        value = row.get(name, float("nan"))
        try:
            number = float(value)
        except (TypeError, ValueError):
            return float("inf")
        return -number if np.isfinite(number) or np.isinf(number) else float("inf")

    return (
        -int(row.get("N_self", 0)),
        -int(row.get("N_valid", 0)),
        descending("self_Q05"),
        descending("self_MRR"),
        descending("fallback_Q05"),
        descending("fallback_MRR"),
        -int(row.get("Self_Top3", 0)),
        int(row.get("K", 10**9)),
        str(row.get("support_hash", "")),
        float(row.get("lambda", float("inf"))),
    )


def rank_frame(frame: Any) -> Any:
    """Sort a pandas-like frame without importing pandas into this module."""

    ranked = frame.copy()
    if "rank" in ranked.columns:
        ranked = ranked.drop(columns="rank")
    ranked["_lex_key"] = [lexicographic_key(row) for row in ranked.to_dict("records")]
    ranked = ranked.sort_values("_lex_key", kind="stable").drop(columns="_lex_key").reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, ranked.shape[0] + 1, dtype=np.int64))
    return ranked


__all__ = [
    "CLASS_NAMES",
    "REAL_B_THRESHOLD_DB",
    "STATE_COUNT",
    "TIE_TOLERANCE_DB",
    "canonical_support_hash",
    "evaluate_distance",
    "lexicographic_key",
    "rank_frame",
    "sha256_json",
    "top1_from_distance",
    "transition_matrix",
]
