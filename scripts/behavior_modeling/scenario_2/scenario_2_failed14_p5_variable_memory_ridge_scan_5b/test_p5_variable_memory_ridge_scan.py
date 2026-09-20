"""Direct gates for the finite P=5 variable-memory Ridge candidate family."""

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

from behavior_modeling.shared import p5_variable_memory_ridge_scan as scan  # noqa: E402
from behavior_modeling.shared.odd_order_variable_memory_scan import (  # noqa: E402
    _full_basis,
    candidate_columns,
)


def test_candidate_grid_gate() -> None:
    candidates = scan.generate_candidates()
    assert len(candidates) == 200
    assert len(scan.MEMORY_PROFILES) == 20
    assert len(scan.RIDGE_LAMBDAS) == 10
    assert candidates[0].candidate_id == 0
    assert candidates[-1].candidate_id == 199
    assert sum(candidate.ridge_lambda == 0.0 for candidate in candidates) == 20
    assert all(candidate.P == 5 and candidate.orders == (1, 3, 5) for candidate in candidates)
    assert all(candidate.M1 >= candidate.M3 >= candidate.M5 for candidate in candidates)


def test_basis_support_gate() -> None:
    candidate = next(
        value
        for value in scan.generate_candidates()
        if value.memory_profile == (4, 4, 1) and value.ridge_lambda == 0.0
    )
    rng = np.random.default_rng(20260909)
    x = rng.normal(size=20) + 1j * rng.normal(size=20)
    phi = _full_basis(x)
    selected = phi[candidate.max_delay :, :][:, candidate_columns(candidate)]
    assert selected.shape == (17, 9)
    assert np.all(np.isfinite(selected))
    assert candidate.max_delay == 3


def test_ridge_augmented_objective_gate() -> None:
    rng = np.random.default_rng(11)
    n, k = 13, 3
    phi = rng.normal(size=(n, k)) + 1j * rng.normal(size=(n, k))
    y = rng.normal(size=n) + 1j * rng.normal(size=n)
    ridge_lambda = 1e-8
    identity = np.eye(k, dtype=np.complex128)
    augmented = np.vstack((phi, np.sqrt(n * ridge_lambda) * identity))
    target = np.concatenate((y, np.zeros(k, dtype=np.complex128)))
    theta, _, rank, _ = np.linalg.lstsq(augmented, target, rcond=None)
    objective = np.linalg.norm(phi @ theta - y) ** 2 / n + ridge_lambda * np.linalg.norm(theta) ** 2
    augmented_objective = np.linalg.norm(augmented @ theta - target) ** 2 / n
    assert rank == k
    assert abs(float(objective - augmented_objective)) <= 1e-12


def main() -> None:
    for label, test in (
        ("candidate grid gate", test_candidate_grid_gate),
        ("candidate-specific B support gate", test_basis_support_gate),
        ("Ridge augmented objective gate", test_ridge_augmented_objective_gate),
    ):
        test()
        print(f"{label}: PASS")


if __name__ == "__main__":
    main()
