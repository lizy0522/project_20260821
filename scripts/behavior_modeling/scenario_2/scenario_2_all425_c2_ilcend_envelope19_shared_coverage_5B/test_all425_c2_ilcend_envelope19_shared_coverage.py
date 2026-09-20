"""Direct contracts for Frozen Envelope19 All425 dual-behavior coverage."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from behavior_modeling.shared.basis_function_selection.all425_c2_ilcend_envelope19_shared_coverage import (  # noqa: E402,E501
    EXPECTED_MODEL_NAME,
    EXPECTED_SUPPORT_HASH,
    FULL_LENGTH,
    TEST_LENGTH,
    TRAIN_LENGTH,
    load_frozen_model,
)


def test_frozen_model_contract() -> None:
    terms, support, ids, digest, frame = load_frozen_model()
    assert len(terms) == 75
    assert len(support) == 19
    assert len(ids) == 19
    assert digest == EXPECTED_SUPPORT_HASH
    assert set(frame["K"].astype(int)) == {19}
    assert set(frame["dmax"].astype(int)) == {2}
    assert set(frame["lambda"].astype(float)) == {0.0}
    assert set(frame["solver"].astype(str)) == {"OLS"}
    assert set(frame["model_name"].astype(str)) == {EXPECTED_MODEL_NAME}


def test_shape_contract() -> None:
    assert FULL_LENGTH == 24_576
    assert TRAIN_LENGTH == 16_384
    assert TEST_LENGTH == 8_192


def test_evaluation_runner_has_no_selection_or_retrieval_path() -> None:
    runner = (
        Path(__file__).resolve().parent
        / "run_scenario2_all425_c2_ilcend_envelope19_shared_coverage_5b.py"
    ).read_text(encoding="utf-8")
    for forbidden in (
        "fit_ridge",
        "build_partition_from_xin",
        "top1_retrieval",
        "fingerprint_cnmse_matrix",
        "common_B",
        "basis_search",
    ):
        assert forbidden not in runner
    assert "evaluation_only" in runner
    assert "Test" in runner


def main() -> None:
    test_frozen_model_contract()
    test_shape_contract()
    test_evaluation_runner_has_no_selection_or_retrieval_path()
    print("All425 Envelope19 dual-behavior coverage direct regression: PASS")


if __name__ == "__main__":
    main()
