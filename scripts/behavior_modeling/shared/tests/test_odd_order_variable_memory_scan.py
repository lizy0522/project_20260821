"""Direct tests for the compact failed-14 variable-memory MP definition."""

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

from behavior_modeling.shared import odd_order_variable_memory_scan as scan  # noqa: E402


def test_candidate_grid_gate() -> None:
    candidates = scan.generate_candidates()
    assert len(candidates) == 125
    assert [sum(candidate.P == P for candidate in candidates) for P in scan.ALLOWED_ORDERS] == [
        4,
        10,
        20,
        35,
        56,
    ]
    assert all(all(order % 2 == 1 for order in candidate.orders) for candidate in candidates)
    assert all(
        all(
            left >= right
            for left, right in zip(
                candidate.memory_profile, candidate.memory_profile[1:], strict=False
            )
        )
        for candidate in candidates
    )


def test_basis_order_and_coefficient_count() -> None:
    candidates = scan.generate_candidates()
    candidate = next(
        value for value in candidates if value.P == 9 and value.memory_profile == (4, 3, 2, 1, 1)
    )
    assert candidate.coefficient_count == 11
    assert candidate.max_delay == 3
    assert candidate.basis_terms == (
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (3, 0),
        (3, 1),
        (3, 2),
        (5, 0),
        (5, 1),
        (7, 0),
        (9, 0),
    )
    columns = scan.candidate_columns(candidate)
    assert columns == (0, 1, 2, 3, 4, 5, 6, 8, 9, 12, 16)


def test_original_time_basis_support() -> None:
    rng = np.random.default_rng(20260908)
    x = rng.normal(size=20) + 1j * rng.normal(size=20)
    phi = scan._full_basis(x)
    assert phi.shape == (20, 20)
    assert np.isnan(phi[:3, 19]).all()
    candidate = scan.generate_candidates()[-1]
    selected = phi[candidate.max_delay :, :][:, scan.candidate_columns(candidate)]
    assert selected.shape == (17, candidate.coefficient_count)
    assert np.all(np.isfinite(selected))


def main() -> None:
    for label, test in (
        ("candidate grid gate", test_candidate_grid_gate),
        ("basis order and K gate", test_basis_order_and_coefficient_count),
        ("candidate-specific support gate", test_original_time_basis_support),
    ):
        test()
        print(f"{label}: PASS")


if __name__ == "__main__":
    main()
