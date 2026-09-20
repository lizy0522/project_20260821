# ruff: noqa: E402,I001

"""Direct contracts for the historical MP10/E19 Full-LUT integration."""

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

from behavior_fingerprint_retrieval.scenario_2.scenario_2_mp10_full_lut_retrieval_5b import mp10_c2endshared_commonB_full_lut as mp10  # noqa: E402,E501


def test_historical_contract() -> None:
    contract = mp10.historical_contract()
    assert contract["orders"] == [1, 2, 3, 5, 7, 9]
    assert contract["memory"] == [3, 2, 2, 1, 1, 1]
    assert contract["K"] == 10
    assert contract["dmax"] == 2
    assert contract["ridge_lambda"] == 1e-8
    assert contract["separate_hnorm_function_found"] is False


def test_common_b_backend_shape() -> None:
    common_b, metadata = mp10.verify_common_b_contract()
    assert common_b.shape == (4915,)
    assert metadata["sha256"] == mp10.EXPECTED_COMMON_B_SHA
    backend = mp10.verify_backend_contract(common_b)
    assert backend["common_B_phi_shape"] == [4913, 10]


def test_backend_basis_shape_and_finiteness() -> None:
    x = np.arange(32, dtype=float) + 1j * np.arange(32, dtype=float)
    phi = mp10.historical_mp10.build_frozen_mp_basis(x)
    assert phi.shape == (30, 10)
    assert np.iscomplexobj(phi)
    assert np.all(np.isfinite(phi))


def main() -> None:
    test_historical_contract()
    test_common_b_backend_shape()
    test_backend_basis_shape_and_finiteness()
    print("Historical MP10/E19 Full-LUT direct regression: PASS")


if __name__ == "__main__":
    main()
