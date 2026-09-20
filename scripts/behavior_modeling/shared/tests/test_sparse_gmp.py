"""Small deterministic tests for the Sparse-GMP basis contract."""

from __future__ import annotations

import unittest

import numpy as np

from behavior_modeling.shared.sparse_gmp import (
    COMMON_MAX_DELAY,
    FROZEN_MP_K,
    build_combined_basis,
    build_gmp_basis,
    generate_gmp_dictionary,
)


class SparseGMPTests(unittest.TestCase):
    def test_dictionary_is_complete_and_cross_memory_only(self) -> None:
        terms = generate_gmp_dictionary()
        self.assertEqual(len(terms), 18)
        self.assertEqual(
            [term.term_id for term in terms],
            [f"GMP{index:03d}" for index in range(1, 19)],
        )
        self.assertTrue(all(term.m != term.q for term in terms))

    def test_explicit_cross_term_orientation(self) -> None:
        x = np.arange(12, dtype=float) + 1j * np.arange(12, dtype=float)
        terms = generate_gmp_dictionary()
        observed = build_gmp_basis(x)
        term = next(item for item in terms if item.p == 3 and item.m == 1 and item.q == 0)
        expected = x[COMMON_MAX_DELAY - 1 : x.size - 1] * np.abs(x[COMMON_MAX_DELAY :]) ** 2
        np.testing.assert_allclose(observed[:, term.term_index - 1], expected)

        term = next(item for item in terms if item.p == 5 and item.m == 0 and item.q == 2)
        expected = x[COMMON_MAX_DELAY:] * np.abs(x[COMMON_MAX_DELAY - 2 : x.size - 2]) ** 4
        np.testing.assert_allclose(observed[:, term.term_index - 1], expected)

    def test_combined_basis_column_count(self) -> None:
        x = np.ones(20, dtype=np.complex128)
        terms = generate_gmp_dictionary()
        self.assertEqual(build_combined_basis(x).shape[1], FROZEN_MP_K)
        self.assertEqual(build_combined_basis(x, terms[:8]).shape[1], FROZEN_MP_K + 8)
        self.assertEqual(build_combined_basis(x, terms).shape[1], FROZEN_MP_K + 18)


if __name__ == "__main__":
    unittest.main()
