"""Direct contracts for the ilc_end Aend-free behavior dataset definition."""

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
    FINAL_SUPPORT_IDS,
    verify_frozen_support,
)
from behavior_modeling.shared.basis_function_selection.envelope_dictionary import (  # noqa: E402
    build_envelope_bank,
    build_envelope_dictionary,
)
from behavior_modeling.shared.basis_function_selection.ilcend_behavior_dataset import (  # noqa: E402
    prepare_ilcend_state,
)


def test_ilcend_state_contract() -> None:
    state = prepare_ilcend_state(325)
    assert state.valid_ilc_count >= 2
    assert state.ilc_end_index == state.valid_ilc_count - 1
    assert state.x_train.shape == (16_384,)
    assert state.y_train_adjusted.shape == (16_384,)
    assert state.x_test.shape == (8_192,)
    assert state.y_test_adjusted.shape == (8_192,)
    assert np.isfinite(state.y_train_adjusted).all()
    assert np.isfinite(state.y_test_adjusted).all()


def test_ilcend_dictionary_and_frozen_baseline_contract() -> None:
    terms = build_envelope_dictionary()
    _, support, digest = verify_frozen_support()
    assert len(terms) == 75
    assert len(support) == 18
    assert tuple(terms[index].basis_id for index in support) == FINAL_SUPPORT_IDS
    assert digest
    state = prepare_ilcend_state(325)
    train_bank = build_envelope_bank(state.x_train, terms)
    test_bank = build_envelope_bank(state.x_test, terms)
    assert train_bank[:, support].shape == (16_382, 18)
    assert test_bank[:, support].shape == (8_190, 18)


def main() -> None:
    test_ilcend_state_contract()
    test_ilcend_dictionary_and_frozen_baseline_contract()
    print("ilc_end behavior dataset direct regression: PASS")


if __name__ == "__main__":
    main()
