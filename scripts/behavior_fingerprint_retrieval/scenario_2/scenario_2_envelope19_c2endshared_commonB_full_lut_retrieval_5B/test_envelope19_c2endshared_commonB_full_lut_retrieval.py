"""Direct contracts for Frozen Envelope19 Full-LUT retrieval."""

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

from behavior_fingerprint_retrieval.shared.envelope19_c2endshared_commonB_full_lut import (  # noqa: E402
    EXPECTED_COMMON_B_SHA,
    EXPECTED_SUPPORT_HASH,
    MODEL_NAME,
    load_frozen_support,
    verify_common_b_contract,
)


def test_frozen_support_contract() -> None:
    terms, support, ids, digest, frame = load_frozen_support()
    assert len(terms) == 75
    assert len(support) == 19
    assert len(ids) == 19
    assert digest == EXPECTED_SUPPORT_HASH
    assert set(frame["K"].astype(int)) == {19}
    assert set(frame["dmax"].astype(int)) == {2}
    assert set(frame["lambda"].astype(float)) == {0.0}
    assert set(frame["solver"].astype(str)) == {"OLS"}
    assert set(frame["model_name"].astype(str)) == {MODEL_NAME}


def test_common_b_contract() -> None:
    common_b, metadata = verify_common_b_contract()
    assert common_b.shape == (4915,)
    assert metadata["sha256"] == EXPECTED_COMMON_B_SHA
    assert np.all(np.isfinite(common_b))


def test_exact_protocol_runner_scope() -> None:
    source = (
        Path(__file__).resolve().parents[2]
        / "shared"
        / "envelope19_retrieval_runner.py"
    ).read_text(encoding="utf-8")
    assert "fit_ridge" not in source
    assert "build_envelope_bank" in source
    assert "top1_retrieval" in source
    assert "compute_cnmse_matrix" in source
    assert "common-B" not in source or "common_B" in source


def main() -> None:
    test_frozen_support_contract()
    test_common_b_contract()
    test_exact_protocol_runner_scope()
    print("Envelope19 Full-LUT retrieval direct regression: PASS")


if __name__ == "__main__":
    main()
