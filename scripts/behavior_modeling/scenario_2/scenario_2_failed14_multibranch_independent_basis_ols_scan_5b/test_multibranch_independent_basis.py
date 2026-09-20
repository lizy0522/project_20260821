"""Unit tests for the independent multi-branch basis dictionary."""

# ruff: noqa: E402,E501

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT.parent) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT.parent))

from behavior_modeling.shared import multibranch_independent_basis as mb


class MultibranchBasisTests(unittest.TestCase):
    def test_reference_profiles_and_counts(self) -> None:
        source = mb.validate_config(mb.SOURCE_PROFILE)
        compact = mb.validate_config(mb.COMPACT_PROFILE)
        self.assertTrue(source["valid"])
        self.assertTrue(compact["valid"])
        self.assertEqual(source["basis_count"], 224)
        self.assertEqual(compact["basis_count"], 63)
        self.assertEqual(source["max_effective_delay"], 26)
        self.assertEqual(compact["max_effective_delay"], 8)

    def test_family_formulas_and_support(self) -> None:
        config = dict(mb.COMPACT_PROFILE)
        terms = mb.build_basis_terms(config)
        ids = {term.basis_id: term for term in terms}
        self.assertEqual(ids["B1_DYN_0"].expression, "z[n-0]")
        self.assertEqual(ids["B1_STATIC_3"].expression, "z[n]|z[n]|^2")
        self.assertEqual(ids["B1_MP_3_1"].expression, "z[n-1]|z[n-1]|^2")
        self.assertEqual(ids["B1_EMEM_3_1"].expression, "z[n]|z[n-1]|^2")
        self.assertEqual(ids["B1_LAG_3_1_1"].expression, "z[n-1]|z[n-2]|^2")
        self.assertEqual(ids["B1_LEAD_3_1_1"].expression, "z[n-1]|z[n-0]|^2")
        self.assertEqual(ids["B1_V3_0_0_1"].expression, "z[n-0]z[n-0]z*[n-1]")
        self.assertEqual(ids["B1_V5_0_0_0_0_1"].expression, "z[n-0]z[n-0]z[n-0]z*[n-0]z*[n-1]")
        self.assertEqual(ids["B3_ENV_DYN_0"].expression, "|x[n-0]|")
        x = np.arange(80, dtype=float) + 1j * np.arange(80, dtype=float)[::-1]
        z = 0.7 * x + 0.2 * np.roll(x, 1)
        z[0] = 0.0
        z_terms = [
            ids[key]
            for key in (
                "B1_DYN_0", "B1_STATIC_3", "B1_MP_3_1", "B1_EMEM_3_1",
                "B1_LAG_3_1_1", "B1_LEAD_3_1_1", "B1_V3_0_0_1",
                "B1_V5_0_0_0_0_1",
            )
        ]
        matrix = mb.build_family_matrix(z, z_terms, trim=mb.COMMON_SUPPORT_TRIM)
        value = z[mb.COMMON_SUPPORT_TRIM]
        self.assertTrue(np.allclose(matrix[0, 0], z[mb.COMMON_SUPPORT_TRIM]))
        self.assertTrue(np.allclose(matrix[0, 1], value * abs(value) ** 2))
        self.assertEqual(matrix.shape[0], x.size - mb.COMMON_SUPPORT_TRIM)
        self.assertTrue(np.isfinite(matrix).all())

    def test_final_matrix_excludes_prelinear_columns_and_is_unique(self) -> None:
        config = dict(mb.COMPACT_PROFILE)
        x = np.linspace(0.1, 1.0, 96) + 1j * np.linspace(1.0, 0.1, 96)
        z = np.zeros_like(x)
        z[mb.COMMON_SUPPORT_TRIM:] = x[mb.COMMON_SUPPORT_TRIM:]
        matrix, terms = mb.build_total_matrix(config, x, z, trim=mb.COMMON_SUPPORT_TRIM)
        self.assertEqual(matrix.shape, (96 - mb.COMMON_SUPPORT_TRIM, 63))
        self.assertEqual(len(terms), 63)
        self.assertTrue(all(term.family != "Prelinear" for term in terms))
        mb.validate_basis_uniqueness(terms)
        self.assertEqual(len({term.basis_id for term in terms}), len(terms))


if __name__ == "__main__":
    unittest.main()
