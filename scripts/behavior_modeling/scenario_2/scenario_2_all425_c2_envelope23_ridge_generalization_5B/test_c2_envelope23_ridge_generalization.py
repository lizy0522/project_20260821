"""Direct contracts for the C2-only frozen Envelope23 Ridge task."""

from __future__ import annotations

import inspect
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

from behavior_modeling.scenario_2.scenario_2_all425_c2_envelope23_ridge_generalization_5B.c2_envelope23_ridge_generalization import (  # noqa: E402,E501
    C2_COLUMN_INDEX,
    C2_STAGE_NUMBER,
    C_RAW_LENGTH,
    C_VALID_LENGTH,
    CV_BLOCK_RAW_LENGTHS,
    CV_BLOCK_VALID_LENGTHS,
    DMAX,
    EXPECTED_SUPPORT_HASH,
    EXPECTED_SUPPORT_IDS,
    RIDGE_LAMBDA_GRID,
    assert_b_evaluation_allowed,
    build_c_block_slices,
    evaluate_final_state,
    fit_ols,
    fit_ridge,
    verify_frozen_support,
)


def test_frozen_support_contract() -> None:
    terms, support, digest = verify_frozen_support()
    assert len(terms) == 75
    assert len(support) == 23
    assert tuple(terms[index].basis_id for index in support) == EXPECTED_SUPPORT_IDS
    assert digest == EXPECTED_SUPPORT_HASH
    assert C2_COLUMN_INDEX == 1
    assert C2_STAGE_NUMBER == 2


def test_c_block_contract() -> None:
    blocks = build_c_block_slices()
    assert tuple((block.start, block.stop) for block in blocks) == (
        (0, 2458),
        (2458, 4916),
        (4916, 7373),
    )
    assert C_RAW_LENGTH == 7373
    assert C_VALID_LENGTH == 7371
    assert CV_BLOCK_RAW_LENGTHS == (2458, 2458, 2457)
    assert CV_BLOCK_VALID_LENGTHS == (2456, 2456, 2455)
    assert DMAX == 2


def test_ridge_solver_and_b_lock_contract() -> None:
    rng = np.random.default_rng(20260915)
    phi = rng.normal(size=(64, 5)) + 1j * rng.normal(size=(64, 5))
    target = rng.normal(size=64) + 1j * rng.normal(size=64)
    ols = fit_ols(phi, target, scale_columns=True)
    ridge_zero = fit_ridge(phi, target, 0.0)
    np.testing.assert_allclose(ridge_zero.theta, ols.theta, rtol=0.0, atol=1e-12)
    positive = fit_ridge(phi, target, 1e-8)
    assert positive.theta.shape == (5,)
    assert np.all(np.isfinite(positive.theta))
    assert RIDGE_LAMBDA_GRID[0] == 0.0
    try:
        assert_b_evaluation_allowed(lambda_frozen=False, b_evaluation_unlocked=False)
    except RuntimeError as exc:
        assert "before Ridge lambda is frozen" in str(exc)
    else:
        raise AssertionError("B lock did not reject pre-freeze access")


def test_no_lut_path_in_task_runner() -> None:
    runner = (
        Path(__file__).resolve().parent
        / "run_scenario2_all425_c2_envelope23_ridge_generalization_5b.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "top1_retrieval",
        "compute_cnmse_matrix",
        "get_common_probe",
        "fingerprint_cnmse_matrix",
        "common_B_fingerprint",
    ):
        assert forbidden not in runner
    assert "C-only contiguous CV" in runner
    assert "B_used_for_lambda_selection" in runner


def test_final_b_guard_is_explicit() -> None:
    signature = inspect.signature(evaluate_final_state)
    assert "lambda_frozen" in signature.parameters
    assert "b_evaluation_unlocked" in signature.parameters
    try:
        evaluate_final_state(
            0,
            0.0,
            lambda_frozen=False,
            b_evaluation_unlocked=False,
        )
    except RuntimeError as exc:
        assert "before Ridge lambda is frozen" in str(exc)
    else:
        raise AssertionError("final B evaluator did not enforce lambda freeze")


def main() -> None:
    test_frozen_support_contract()
    test_c_block_contract()
    test_ridge_solver_and_b_lock_contract()
    test_no_lut_path_in_task_runner()
    test_final_b_guard_is_explicit()
    print("C2 Envelope23 Ridge generalization direct regression: PASS")


if __name__ == "__main__":
    main()
