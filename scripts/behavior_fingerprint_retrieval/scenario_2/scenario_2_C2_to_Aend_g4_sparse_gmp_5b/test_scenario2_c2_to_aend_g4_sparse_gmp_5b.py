"""Post-run regression checks for the G4 retrieval artifact."""

# ruff: noqa: E501

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
RESULT_ROOT = PROJECT_ROOT / "results" / "behavior_fingerprint_retrieval" / "scenario_2" /"scenario_2_C2_to_Aend_G4_sparse_gmp_5B"


class G4RetrievalArtifactTests(unittest.TestCase):
    def test_g4_definition_and_shapes(self) -> None:
        structure_path = PROJECT_ROOT / "results" / "behavior_modeling" / "scenario_2" /"scenario_2_failed14_frozen_mp_residual_sparse_gmp_5B" / "models" / "selected_sparse_gmp_structure.json"
        structure = json.loads(structure_path.read_text())
        self.assertEqual(structure["selected_sparse_term_ids"], ["GMP005", "GMP011", "GMP001", "GMP017"])
        self.assertEqual(structure["K_total"], 14)
        with np.load(RESULT_ROOT / "cache" / "Aend_fingerprints.npz", allow_pickle=False) as data:
            self.assertEqual(data["Aend_LUT_fingerprints"].shape, (425, 4913))
        with np.load(RESULT_ROOT / "cache" / "C2_Query_fingerprints.npz", allow_pickle=False) as data:
            self.assertEqual(data["C2_Query_fingerprints"].shape, (425, 4913))
        with np.load(RESULT_ROOT / "cache" / "fingerprint_distance_matrix.npz", allow_pickle=False) as data:
            self.assertEqual(data["D_C2_Aend"].shape, (425, 425))

    def test_summary_counts(self) -> None:
        frame = pd.read_csv(RESULT_ROOT / "tables" / "retrieval_state_summary.csv")
        self.assertEqual(frame.shape[0], 425)
        self.assertEqual(int(frame["exact_hit"].sum()), 235)
        self.assertEqual(int(frame["shareable"].sum()), 419)
        self.assertEqual(int(frame["failure"].sum()), 6)


if __name__ == "__main__":
    unittest.main()
