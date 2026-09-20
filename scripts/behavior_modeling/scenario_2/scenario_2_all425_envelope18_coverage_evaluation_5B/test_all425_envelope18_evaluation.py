"""Direct executable contract checks for the frozen Envelope18 coverage task."""

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

from behavior_modeling.shared.basis_function_selection.all425_envelope18_evaluation import (  # noqa: E402
    DMAX,
    FINAL_SUPPORT_IDS,
    TEST_LENGTH,
    TRAIN_LENGTH,
    load_type_from_state,
    support_hash,
    verify_frozen_support,
)
from behavior_modeling.shared.basis_function_selection.envelope_dictionary import (
    build_envelope_bank,  # noqa: E402
)


def test_frozen_envelope18_contract() -> None:
    terms, indices, digest = verify_frozen_support()
    assert len(FINAL_SUPPORT_IDS) == 18
    assert len(indices) == 18
    assert digest == support_hash()
    assert tuple(terms[index].basis_id for index in indices) == FINAL_SUPPORT_IDS


def test_frozen_bank_shapes_and_load_groups() -> None:
    terms, indices, _ = verify_frozen_support()
    rng = np.random.default_rng(20260915)
    train = rng.normal(size=TRAIN_LENGTH) + 1j * rng.normal(size=TRAIN_LENGTH)
    test = rng.normal(size=TEST_LENGTH) + 1j * rng.normal(size=TEST_LENGTH)
    train_bank = build_envelope_bank(train, terms)[:, indices]
    test_bank = build_envelope_bank(test, terms)[:, indices]
    assert train_bank.shape == (TRAIN_LENGTH - DMAX, 18)
    assert test_bank.shape == (TEST_LENGTH - DMAX, 18)
    assert np.all(np.isfinite(train_bank))
    assert np.all(np.isfinite(test_bank))
    assert (
        load_type_from_state(
            {"funMng": 0, "secMng": 0, "funAng": 0, "secAng": 0, "Vm": 2.3, "Pin": -21.0}
        )
        == "matched"
    )
    assert (
        load_type_from_state(
            {"funMng": 10, "secMng": 0, "funAng": 0, "secAng": 0, "Vm": 2.3, "Pin": -21.0}
        )
        == "fundamental_only"
    )
    assert (
        load_type_from_state(
            {"funMng": 0, "secMng": 15, "funAng": 0, "secAng": 0, "Vm": 2.3, "Pin": -21.0}
        )
        == "second_harmonic_only"
    )
    assert (
        load_type_from_state(
            {"funMng": 10, "secMng": 15, "funAng": 0, "secAng": 0, "Vm": 2.3, "Pin": -21.0}
        )
        == "joint_mismatch"
    )


def main() -> None:
    test_frozen_envelope18_contract()
    test_frozen_bank_shapes_and_load_groups()
    print("All-425 Envelope18 direct regression: PASS")


if __name__ == "__main__":
    main()
