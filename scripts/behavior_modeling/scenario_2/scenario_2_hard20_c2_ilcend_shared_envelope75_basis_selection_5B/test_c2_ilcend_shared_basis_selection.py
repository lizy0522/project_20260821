"""Direct contracts for the shared C2 plus ILC_END support-selection task."""

# ruff: noqa: E501

from __future__ import annotations

from pathlib import Path

import numpy as np

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
from behavior_modeling.scenario_2.scenario_2_hard20_c2_ilcend_shared_envelope75_basis_selection_5B.hard20_c2_ilcend_shared_envelope75_selection import (  # noqa: E402
    EXPECTED_HARD20_IDS,
    TARGET_NMSE_DB,
    assert_test_access_allowed,
    load_supports,
)
from behavior_modeling.shared.basis_function_selection.dual_ilc_behavior_dataset import (  # noqa: E402
    BEHAVIORS,
    ILC_COL2,
    ILC_END,
    prepare_dual_state,
)
from behavior_modeling.shared.basis_function_selection.envelope_dictionary import (  # noqa: E402
    DMAX,
    build_envelope_dictionary,
)


def test_dictionary_and_baseline_union_contract() -> None:
    terms, supports, hashes = load_supports()
    assert len(terms) == 75
    assert len(build_envelope_dictionary()) == 75
    assert len(supports["Envelope18"]) == 18
    assert len(supports["Envelope23-ilcEnd"]) == 23
    assert len(supports["Union24"]) == 24
    assert hashes["Envelope18"]
    assert hashes["Envelope23-ilcEnd"]
    assert hashes["Union24"]
    assert all(
        terms[index].basis_id in {term.basis_id for term in terms}
        for index in supports["Union24"]
    )


def test_hard20_contract_and_dual_train_shapes() -> None:
    assert EXPECTED_HARD20_IDS == (
        325,
        332,
        333,
        334,
        336,
        353,
        324,
        338,
        350,
        326,
        349,
        339,
        351,
        342,
        328,
        323,
        355,
        331,
        327,
        356,
    )
    terms = tuple(build_envelope_dictionary())
    prepared = prepare_dual_state(325, terms, include_test=False)
    assert BEHAVIORS == (ILC_END, ILC_COL2)
    assert prepared.end.ilc_column_index == prepared.end.valid_ilc_count - 1
    assert prepared.col2.ilc_column_index == 1
    for behavior in BEHAVIORS:
        item = prepared.behavior(behavior)
        assert item.full_bank.shape == (16382, 75)
        assert item.full_target.shape == (16382,)
        assert tuple(bank.shape[0] for bank in item.block_banks) == (5459, 5460, 5459)
        assert item.x_test is None
        assert item.y_test_adjusted is None
        assert item.test_gain is None
        assert item.ilc_column_index >= 1
        assert np.all(np.isfinite(item.full_bank))
        assert np.all(np.isfinite(item.full_target))
    assert DMAX == 2
    assert TARGET_NMSE_DB == -40.0


def test_test_lock_contract() -> None:
    try:
        assert_test_access_allowed(model_frozen=False, test_unlocked=False)
    except RuntimeError as exc:
        assert "before shared support is frozen" in str(exc)
    else:
        raise AssertionError("Test access was not blocked before support freeze")
    try:
        assert_test_access_allowed(model_frozen=True, test_unlocked=False)
    except RuntimeError as exc:
        assert "still locked" in str(exc)
    else:
        raise AssertionError("Test access was not blocked while still locked")
    assert_test_access_allowed(model_frozen=True, test_unlocked=True)


def test_forbidden_paths_are_not_imported_by_task_runner() -> None:
    source = (
        Path(__file__).resolve().parent
        / "hard20_c2_ilcend_shared_envelope75_selection.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "fit_ridge",
        "top1_retrieval",
        "fingerprint_cnmse_matrix",
        "common_B",
        "real_B_retrieval",
    ):
        assert forbidden not in source
    assert "prepare_dual_state" in source
    assert "support freeze" in source or "support_frozen_test_unlocked" in source


def main() -> None:
    test_dictionary_and_baseline_union_contract()
    test_hard20_contract_and_dual_train_shapes()
    test_test_lock_contract()
    test_forbidden_paths_are_not_imported_by_task_runner()
    print("C2 + ILC_END shared Envelope75 direct regression: PASS")


if __name__ == "__main__":
    main()
