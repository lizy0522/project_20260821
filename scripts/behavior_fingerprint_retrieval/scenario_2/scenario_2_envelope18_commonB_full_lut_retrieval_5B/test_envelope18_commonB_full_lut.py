"""Direct executable contracts for the frozen Envelope18 retrieval task."""

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

from core.shared.metrics import cnmse  # noqa: E402

from behavior_fingerprint_retrieval.shared.envelope18_commonB_full_lut import (  # noqa: E402
    build_common_b_probe,
    compute_cnmse_matrix,
    top1_retrieval,
    verify_runtime_contract,
)


def test_frozen_support_and_common_b_contract() -> None:
    terms, support, digest = verify_runtime_contract()
    common_b, common_phi, metadata = build_common_b_probe()
    assert len(terms) == 75
    assert len(support) == 18
    assert digest
    assert common_b.shape == (4915,)
    assert common_phi.shape == (4913, 18)
    assert np.all(np.isfinite(common_b))
    assert np.all(np.isfinite(common_phi))
    assert metadata["raw_length"] == 4915
    assert metadata["valid_length"] == 4913


def test_cnmse_direction_and_stable_top1_tie_break() -> None:
    rng = np.random.default_rng(20260915)
    query = rng.normal(size=(2, 4913)) + 1j * rng.normal(size=(2, 4913))
    candidates = query.copy()
    candidates[1] = candidates[1] * (1.0 + 0.01j)
    distance = compute_cnmse_matrix(query, candidates)
    assert distance.shape == (2, 2)
    assert np.isneginf(distance[0, 0])
    np.testing.assert_allclose(distance[0, 1], cnmse(query[0], candidates[1]), rtol=0.0, atol=1e-12)

    full_distance = np.full((425, 425), 1.0, dtype=np.float64)
    full_distance[:, 0] = 0.0
    full_distance[7, 2] = -1.0
    full_distance[7, 3] = -1.0
    selected, selected_distance, true_rank, tie_count = top1_retrieval(full_distance)
    assert selected[7] == 2
    assert selected_distance[7] == -1.0
    assert tie_count[7] == 2
    assert true_rank[7] >= 1


def main() -> None:
    test_frozen_support_and_common_b_contract()
    test_cnmse_direction_and_stable_top1_tie_break()
    print("Envelope18 common-B Full-LUT direct regression: PASS")


if __name__ == "__main__":
    main()
