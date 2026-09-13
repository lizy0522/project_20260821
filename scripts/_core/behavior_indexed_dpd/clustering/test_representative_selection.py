"""Unit tests for generic minimax representative selection."""

# ruff: noqa: E402,I001

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parents[3]
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _core.behavior_indexed_dpd.clustering import cluster_behavior_states
from _core.behavior_indexed_dpd.clustering.representative_selection import (
    select_cluster_representative,
)


class RepresentativeSelectionTests(unittest.TestCase):
    def test_minimax_primary_score(self) -> None:
        matrix = np.asarray(
            [
                [-np.inf, -50.0, -48.0, -40.5],
                [-50.0, -np.inf, -45.0, -43.0],
                [-48.0, -45.0, -np.inf, -44.0],
                [-40.5, -43.0, -44.0, -np.inf],
            ]
        )
        result = select_cluster_representative((0, 1, 2, 3), matrix)
        self.assertEqual(result.representative_state_id, 2)

    def test_median_tie_break(self) -> None:
        matrix = np.full((4, 4), -np.inf)
        matrix[0, 1:] = [-40.0, -42.0, -43.0]
        matrix[1, 0] = -40.0
        matrix[1, 2:] = [-45.0, -44.0]
        matrix[2, 0] = -42.0
        matrix[2, 1] = -45.0
        matrix[2, 3] = -39.0
        matrix[3, :3] = [-43.0, -44.0, -39.0]
        result = select_cluster_representative((0, 1, 2, 3), matrix)
        self.assertEqual(result.representative_state_id, 1)

    def test_mean_tie_break(self) -> None:
        matrix = np.full((5, 5), -np.inf)
        matrix[0, 1:] = [-40.0, -42.0, -43.0, -45.0]
        matrix[1, 0] = -40.0
        matrix[1, 2:] = [-42.0, -43.0, -47.0]
        matrix[2, 0] = -42.0
        matrix[2, 1] = -42.0
        matrix[2, 3:] = [-39.0, -39.0]
        matrix[3, 0] = -43.0
        matrix[3, 1] = -43.0
        matrix[3, 2] = -39.0
        matrix[3, 4] = -39.0
        matrix[4, :4] = [-45.0, -47.0, -39.0, -39.0]
        result = select_cluster_representative((0, 1, 2, 3, 4), matrix)
        self.assertEqual(result.representative_state_id, 1)

    def test_state_id_tie_break(self) -> None:
        matrix = np.asarray([[-np.inf, -43.0], [-43.0, -np.inf]])
        result = select_cluster_representative((0, 1), matrix)
        self.assertEqual(result.representative_state_id, 0)

    def test_singleton(self) -> None:
        matrix = np.full((38, 38), -50.0)
        np.fill_diagonal(matrix, -np.inf)
        result = select_cluster_representative((37,), matrix)
        self.assertEqual(result.representative_state_id, 37)
        self.assertTrue(np.isneginf(result.worst_cnmse_db))

    def test_high_level_membership_and_representative_map(self) -> None:
        matrix = np.asarray(
            [
                [-np.inf, -45.0, -42.0, -38.0],
                [-45.0, -np.inf, -43.0, -39.0],
                [-42.0, -43.0, -np.inf, -41.0],
                [-38.0, -39.0, -41.0, -np.inf],
            ]
        )
        result = cluster_behavior_states(matrix, -40.0)
        self.assertEqual(result.clusters, ((0, 1, 2), (3,)))
        self.assertEqual(result.representative_state_ids.tolist(), [1, 3])
        self.assertEqual(result.representative_for_state(0), 1)
        self.assertEqual(result.representative_for_state(3), 3)
        for cluster_id, members in enumerate(result.clusters):
            self.assertIn(int(result.representative_state_ids[cluster_id]), members)


if __name__ == "__main__":
    unittest.main()
