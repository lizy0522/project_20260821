"""Direct executable regression checks for the Envelope75 task contract."""

# ruff: noqa: E402

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

from behavior_modeling.shared.basis_function_selection.envelope_dictionary import (  # noqa: E402
    build_envelope_bank,
    build_envelope_dictionary,
    dictionary_gate,
)
from behavior_modeling.shared.basis_function_selection.model_solver import (
    ols_scaling_equivalence_gate,  # noqa: E402
)


def test_envelope75_dictionary_contract() -> None:
    terms = build_envelope_dictionary()
    assert len(terms) == 75
    assert len({term.basis_id for term in terms}) == 75
    assert len({term.signature for term in terms}) == 75
    gate = dictionary_gate()
    assert gate["pass"] is True
    assert gate["frozen_gate_counts"] == {
        "FROZEN10": 10,
        "GATE_A": 18,
        "GATE_B": 48,
        "GATE_C": 108,
    }
    assert gate["envelope_minus_gate_b_count"] == 27
    assert gate["envelope_minus_gate_b_orders"] == [4, 6, 8]


def test_envelope75_local_history_and_ols_gate() -> None:
    terms = build_envelope_dictionary()
    rng = np.random.default_rng(20260915)
    x = rng.normal(size=257) + 1j * rng.normal(size=257)
    bank = build_envelope_bank(x, terms)
    assert bank.shape == (255, 75)
    assert np.all(np.isfinite(bank))
    # The first valid row uses x[2], x[1], x[0] only; no pre-segment sample exists.
    np.testing.assert_allclose(bank[0, 0], x[2])
    phi = rng.normal(size=(512, 5)) + 1j * rng.normal(size=(512, 5))
    target = phi @ (np.arange(5, dtype=np.float64) + 1j)
    target = target + 0.05 * (rng.normal(size=512) + 1j * rng.normal(size=512))
    gate = ols_scaling_equivalence_gate(phi, target)
    assert gate["pass"] is True


def main() -> None:
    test_envelope75_dictionary_contract()
    test_envelope75_local_history_and_ols_gate()
    print("Envelope75 direct regression: PASS")


if __name__ == "__main__":
    main()
