"""Direct-run tests for unified odd-order MP capacity scan primitives."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
ROOT = PROJECT_ROOT
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from behavior_modeling.shared.basis import build_mp_basis  # noqa: E402
from behavior_modeling.shared.odd_order_mp_capacity_scan import (  # noqa: E402
    OddOrderCandidate,
    _full_basis,
    coefficient_count,
    evaluate_shell_parallel,
    evaluate_state_shell_serial,
    odd_orders,
    slice_basis_from_max,
)


def test_odd_order_and_uniform_memory_contract() -> None:
    assert odd_orders(1) == (1,)
    assert odd_orders(9) == (1, 3, 5, 7, 9)
    assert 2 not in odd_orders(15)
    assert coefficient_count(9, 4) == 20
    candidate = OddOrderCandidate(1, 9, odd_orders(9), 4, 3, 20, 0, "test")
    assert candidate.memory_definition == {1: 4, 3: 4, 5: 4, 7: 4, 9: 4}
    assert candidate.max_delay == 3
    assert candidate.coefficient_count == 20


def test_slice_from_max_basis_matches_direct_basis() -> None:
    rng = np.random.default_rng(20260908)
    x = rng.normal(size=512) + 1j * rng.normal(size=512)
    y = rng.normal(size=512) + 1j * rng.normal(size=512)
    full_orders = odd_orders(15)
    max_memory = 5
    phi_full, y_full = _full_basis(SimpleNamespace(input=x, output=y), full_orders, max_memory)
    candidate = OddOrderCandidate(1, 9, odd_orders(9), 3, 2, coefficient_count(9, 3), 0, "test")
    phi_slice, y_slice = slice_basis_from_max(phi_full, y_full, full_orders, max_memory, candidate)
    phi_direct = build_mp_basis(x, candidate.orders, candidate.memory_definition)
    np.testing.assert_allclose(phi_slice, phi_direct, rtol=0, atol=0)
    np.testing.assert_allclose(y_slice, y[candidate.max_delay :], rtol=0, atol=0)


def test_serial_parallel_real_state_regression() -> None:
    candidates = tuple(
        OddOrderCandidate(index, P, odd_orders(P), M, M - 1, coefficient_count(P, M), 0, "test")
        for index, (P, M) in enumerate(((3, 1), (5, 2)), start=1)
    )
    states = (0, 1)
    serial_rows = []
    serial_a = {}
    serial_c = {}
    for state_id in states:
        result = evaluate_state_shell_serial(state_id, candidates, "Development")
        serial_rows.extend(result["rows"])
        serial_a.update(result["theta_a"])
        serial_c.update(result["theta_c"])
    split = pd.DataFrame({"state_id": states, "split": ["Development", "Development"]})
    parallel = evaluate_shell_parallel(candidates, split, worker_count=2)
    serial = (
        pd.DataFrame(serial_rows)
        .sort_values(["state_id", "side", "candidate_id"])
        .reset_index(drop=True)
    )
    parallel_frame = (
        pd.DataFrame(parallel.rows)
        .sort_values(["state_id", "side", "candidate_id"])
        .reset_index(drop=True)
    )
    np.testing.assert_allclose(
        serial[["train_NMSE_dB", "B_NMSE_dB"]],
        parallel_frame[["train_NMSE_dB", "B_NMSE_dB"]],
        rtol=0,
        atol=1e-10,
    )
    for key, value in serial_a.items():
        np.testing.assert_allclose(value, parallel.theta_a[key], rtol=0, atol=1e-10)
    for key, value in serial_c.items():
        np.testing.assert_allclose(value, parallel.theta_c[key], rtol=0, atol=1e-10)


def main() -> None:
    tests = [
        ("odd-order/uniform-memory contract", test_odd_order_and_uniform_memory_contract),
        ("max-basis slicing regression", test_slice_from_max_basis_matches_direct_basis),
        ("serial/parallel real-state regression", test_serial_parallel_real_state_regression),
    ]
    for label, test in tests:
        test()
        print(f"{label}: PASS")


if __name__ == "__main__":
    main()
