"""Real-B oracle and post-ranking DPD shareability score.

The oracle is never passed into fingerprint construction or retrieval.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from core.shared.metrics import cnmse

from .shareability_pipeline import RetrievalRanking
from .shareability_types import REAL_B_THRESHOLD_DB


@dataclass(frozen=True)
class ShareabilityScore:
    n_states: int
    n_shareable: int
    shareable_rate: float
    n_self: int
    self_rate: float
    n_top3_contains_true: int
    top3_contains_true_rate: float
    n_top5_contains_true: int
    top5_contains_true_rate: float
    n_top10_contains_true: int
    top10_contains_true_rate: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


class RealBShareabilityOracle:
    """Truth matrix indexed [real R, retrieved Q], only queried after ranking."""

    def __init__(self, cnmse_db: np.ndarray, threshold_db: float = REAL_B_THRESHOLD_DB):
        matrix = np.asarray(cnmse_db, dtype=np.float64)
        if (
            matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]
            or np.isnan(matrix).any() or np.isposinf(matrix).any()
        ):
            raise ValueError("Real-B oracle must be square, nonempty, and contain no NaN/+Inf")
        if not np.all(np.isneginf(np.diag(matrix))):
            raise ValueError("same-state Real-B CNMSE diagonal must be -Inf")
        if not np.isfinite(threshold_db):
            raise ValueError("threshold must be finite")
        self.matrix = matrix.copy()
        self.threshold_db = float(threshold_db)

    @classmethod
    def from_outputs(
        cls, outputs: Sequence[np.ndarray], threshold_db: float = REAL_B_THRESHOLD_DB
    ) -> RealBShareabilityOracle:
        """Compute a small oracle on demand; callers control data size."""

        vectors = [np.asarray(value, dtype=np.complex128) for value in outputs]
        if not vectors or any(
            value.ndim != 1 or value.shape != vectors[0].shape or not np.all(np.isfinite(value))
            for value in vectors
        ):
            raise ValueError("finite equal-length Real-B outputs required")
        matrix = np.array(
            [[cnmse(reference, candidate) for candidate in vectors] for reference in vectors],
            dtype=np.float64,
        )
        np.fill_diagonal(matrix, -np.inf)
        return cls(matrix, threshold_db)

    @property
    def shareable_mask(self) -> np.ndarray:
        return self.matrix < self.threshold_db

    def selected_mask(self, top1_indices: np.ndarray) -> np.ndarray:
        indices = np.asarray(top1_indices, dtype=np.int64)
        n = self.matrix.shape[0]
        if indices.shape != (n,) or np.any(indices < 0) or np.any(indices >= n):
            raise ValueError("Top-1 indices must cover oracle states")
        return self.shareable_mask[np.arange(n), indices]

    def save(self, path: Path) -> None:
        """Persist numeric matrix and explicit threshold without Python pickle."""

        if path.exists():
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, cnmse_db=self.matrix, threshold_db=self.threshold_db)

    @classmethod
    def load(cls, path: Path) -> RealBShareabilityOracle:
        with np.load(path, allow_pickle=False) as data:
            return cls(data["cnmse_db"], float(data["threshold_db"]))


def score_ranking(
    ranking: RetrievalRanking, oracle: RealBShareabilityOracle
) -> ShareabilityScore:
    """Primary objective is Top-1 Real-B shareability; Top-k tests true ID only."""

    rows = np.asarray(ranking.ranking_matrix)
    n = oracle.matrix.shape[0]
    if rows.shape != (n, n) or ranking.top1_indices.shape != (n,):
        raise ValueError("ranking and oracle state counts differ")
    if not np.array_equal(rows[:, 0], ranking.top1_indices) or any(
        set(row.tolist()) != set(range(n)) for row in rows
    ):
        raise ValueError("ranking is not a complete permutation")
    true_ids = np.arange(n)
    n_shareable = int(np.count_nonzero(oracle.selected_mask(ranking.top1_indices)))
    n_self = int(np.count_nonzero(ranking.top1_indices == true_ids))

    def count(k: int) -> int:
        return int(np.count_nonzero(np.any(rows[:, :min(k, n)] == true_ids[:, None], axis=1)))

    top3, top5, top10 = (count(k) for k in (3, 5, 10))
    return ShareabilityScore(
        n, n_shareable, n_shareable / n, n_self, n_self / n,
        top3, top3 / n, top5, top5 / n, top10, top10 / n,
    )


__all__ = ["RealBShareabilityOracle", "ShareabilityScore", "score_ranking"]
