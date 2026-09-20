"""Direct gates for the P5+[2] variable-memory OLS ablation."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from behavior_modeling.shared import p5_with_order2_variable_memory_ols_scan as scan  # noqa: E402


def test_candidate_grid() -> None:
    candidates = scan.generate_candidates()
    assert len(candidates) == 35
    assert len(scan.MEMORY_PROFILES) == 35
    assert [candidate.candidate_id for candidate in candidates] == list(range(35))
    assert all(candidate.orders == (1, 2, 3, 5) for candidate in candidates)
    assert all(
        candidate.M1 >= candidate.M2 >= candidate.M3 >= candidate.M5 for candidate in candidates
    )
    assert all(
        candidate.ridge_used is False and candidate.lambda_value is None for candidate in candidates
    )


def test_basis_terms_and_support() -> None:
    candidate = next(
        value for value in scan.generate_candidates() if value.memory_profile == (3, 2, 2, 1)
    )
    assert candidate.candidate_id == 7
    assert candidate.coefficient_count == 8
    assert candidate.max_delay == 2
    assert candidate.basis_terms == (
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 0),
        (2, 1),
        (3, 0),
        (3, 1),
        (5, 0),
    )
    rng = np.random.default_rng(20260909)
    x = rng.normal(size=31) + 1j * rng.normal(size=31)
    full = scan._full_basis(x)
    selected = full[candidate.max_delay :, :][:, scan.candidate_columns(candidate)]
    assert selected.shape == (29, 8)
    assert np.all(np.isfinite(selected))


def test_ols_solution_is_finite() -> None:
    rng = np.random.default_rng(23)
    phi = rng.normal(size=(25, 4)) + 1j * rng.normal(size=(25, 4))
    y = rng.normal(size=25) + 1j * rng.normal(size=25)
    theta, _, rank, singular = np.linalg.lstsq(phi, y, rcond=None)
    assert rank == 4
    assert np.all(np.isfinite(theta))
    assert np.all(np.isfinite(singular))


def main() -> None:
    for label, test in (
        ("35-candidate OLS grid gate", test_candidate_grid),
        ("order-2 basis/support gate", test_basis_terms_and_support),
        ("OLS finite-solution gate", test_ols_solution_is_finite),
    ):
        test()
        print(f"{label}: PASS")


if __name__ == "__main__":
    main()
