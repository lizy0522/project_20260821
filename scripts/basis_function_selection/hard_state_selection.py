"""Deterministic selection of the 20 worst no-DPD Train-NMSE states."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .config import HARD_STATE_COUNT, STATE_COUNT


def select_hard_states(
    rows: Sequence[Mapping[str, object]],
) -> tuple[list[dict[str, object]], tuple[int, ...]]:
    """Rank 425 rows by descending no-DPD Train NMSE, then ascending state ID."""

    if len(rows) != STATE_COUNT:
        raise RuntimeError(f"Expected {STATE_COUNT} ranking rows, got {len(rows)}")
    normalized = [dict(row) for row in rows]
    state_ids = [int(row["state_id"]) for row in normalized]
    if sorted(state_ids) != list(range(STATE_COUNT)):
        raise RuntimeError("Ranking rows do not cover state IDs 0..424 exactly once")
    normalized.sort(key=lambda row: (-float(row["train_noDPD_NMSE_dB"]), int(row["state_id"])))
    for rank, row in enumerate(normalized, start=1):
        row["rank"] = rank
        row["is_hard20"] = rank <= HARD_STATE_COUNT
    hard_ids = tuple(int(row["state_id"]) for row in normalized[:HARD_STATE_COUNT])
    if len(set(hard_ids)) != HARD_STATE_COUNT:
        raise RuntimeError("Hard-20 selection contains duplicate state IDs")
    return normalized, hard_ids


__all__ = ["select_hard_states"]
