"""Contract, synthetic correctness and frozen-input regression tests."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = next(
    parent
    for parent in Path(__file__).resolve().parents
    if (parent / "environment.yml").is_file() and (parent / "scripts").is_dir()
)
SCRIPTS_ROOT = PROJECT_ROOT / "scripts"
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from behavior_fingerprint_ranking_consistency.shared.distance_matrix import (  # noqa: E402
    compute_cnmse_distance_matrix,
)
from core.shared.metrics import cnmse  # noqa: E402

from behavior_fingerprint_retrieval.shared.scenario2_c3_to_a2_retrieval import (  # noqa: E402
    FINGERPRINT_LENGTH,
    STATE_COUNT,
    circular_difference_deg,
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_retrieval"
    / "scenario_2" /"scenario_2_C3_to_A2_retrieval"
    / "scenario_2_C3_to_A2"
)
REFERENCE_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2" /"scenario2_ranking_consistency"
    / "scenario_2"
)
A123_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2" /"scenario2_y_lut_fusion_a123_analysis"
    / "scenario_2"
)


def _tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    for file in sorted(
        (
            item
            for item in path.rglob("*")
            if item.is_file() and "__pycache__" not in item.parts and item.suffix.lower() != ".pyc"
        ),
        key=lambda item: str(item).lower(),
    ):
        content = file.read_bytes()
        digest.update(file.relative_to(path).as_posix().encode() + b"\0")
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def _raw_manifest_digest() -> str:
    raw_root = PROJECT_ROOT / "data" / "raw"
    digest = hashlib.sha256()
    for file in sorted(
        (item for item in raw_root.rglob("*") if item.is_file()),
        key=lambda item: str(item).lower(),
    ):
        digest.update(str(file.relative_to(raw_root)).replace("\\", "/").encode() + b"\0")
        digest.update(file.stat().st_size.to_bytes(8, "little"))
    return digest.hexdigest()


def _load_validation() -> dict[str, object]:
    with (RESULT_ROOT / "validation.json").open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def test_synthetic_cnmse_direction_and_argmin() -> None:
    query = np.zeros((1, FINGERPRINT_LENGTH), dtype=np.complex128)
    query[0, 0] = 1.0
    close = query.copy()
    far = np.zeros_like(query)
    far[0, 1] = 1.0
    distance = compute_cnmse_distance_matrix(query, np.vstack([far, close]))
    assert distance[0, 1] == -np.inf
    assert distance[0, 1] < distance[0, 0]
    np.testing.assert_equal(distance[0, 1], cnmse(query[0], close[0]))
    assert int(np.argmin(distance[0])) == 1


def test_synthetic_tie_uses_lowest_state_id() -> None:
    from behavior_fingerprint_retrieval.shared.scenario2_c3_to_a2_retrieval import _topk_retrieval

    distance = np.full((STATE_COUNT, STATE_COUNT), 2.0, dtype=np.float64)
    distance[:, 0] = -3.0
    distance[:, 1] = -3.0
    state_ids = np.arange(STATE_COUNT, dtype=np.int64)
    selected, selected_distance, _, ties, _ = _topk_retrieval(distance, state_ids)
    assert np.all(selected == 0)
    assert np.all(selected_distance == -3.0)
    assert np.all(ties == 2)


def test_exact_distance_is_negative_infinity() -> None:
    sample = np.zeros((1, FINGERPRINT_LENGTH), dtype=np.complex128)
    sample[0, 0] = 2.0 + 1.0j
    distance = compute_cnmse_distance_matrix(sample, sample)
    assert distance[0, 0] == -np.inf


def test_circular_difference_contract() -> None:
    assert circular_difference_deg(0, 315) == 45.0
    assert circular_difference_deg(315, 0) == 45.0
    assert circular_difference_deg(45, 315) == 90.0


def test_frozen_shapes_and_dtypes() -> None:
    with np.load(A123_ROOT / "lut_fingerprints.npz", allow_pickle=False) as lut:
        with np.load(A123_ROOT / "query_fingerprints.npz", allow_pickle=False) as query:
            assert lut["Y_A2_fingerprints"].shape == (425, 4913)
            assert query["Q_C3"].shape == (425, 4913)
            assert lut["Y_A2_fingerprints"].dtype == np.complex128
            assert query["Q_C3"].dtype == np.complex128
            assert np.all(np.isfinite(lut["Y_A2_fingerprints"]))
            assert np.all(np.isfinite(query["Q_C3"]))


def test_candidate_and_matrix_contract() -> None:
    with np.load(RESULT_ROOT / "retrieval_distance_matrix.npz", allow_pickle=False) as data:
        assert data["D_QA2"].shape == (425, 425)
        assert data["R_QA2"].shape == (425, 425)
        assert not np.isnan(data["D_QA2"]).any()
        assert not np.isposinf(data["D_QA2"]).any()
    with np.load(RESULT_ROOT / "real_B_distance_matrix.npz", allow_pickle=False) as data:
        assert data["D_B"].shape == (425, 425)
        assert data["R_B"].shape == (425, 425)
        assert np.all(np.isneginf(np.diag(data["D_B"])))


def test_retrieval_has_425_rows_and_topk_fields() -> None:
    table = pd.read_csv(RESULT_ROOT / "retrieval_results.csv")
    assert table.shape[0] == 425
    assert np.array_equal(table["State_n_R"].to_numpy(), np.arange(425))
    for column in ("exact_hit", "top1_hit", "top3_hit", "top5_hit"):
        assert column in table.columns
    assert table["State_n_Q"].between(0, 424).all()
    assert table["true_state_rank_in_A2_LUT"].between(1, 425).all()


def test_real_b_diagnostics_preserve_exact_negative_infinity() -> None:
    retrieval = pd.read_csv(RESULT_ROOT / "retrieval_results.csv")
    diagnostics = pd.read_csv(RESULT_ROOT / "retrieved_real_B_cnmse.csv")
    exact_ids = retrieval.loc[retrieval.exact_hit, "State_n_R"].to_numpy()
    if exact_ids.size:
        exact_values = diagnostics.set_index("State_n_R").loc[
            exact_ids, "retrieved_real_B_CNMSE_dB"
        ]
        assert np.all(np.isneginf(exact_values.to_numpy(dtype=float)))


def test_oracle_excludes_self() -> None:
    diagnostics = pd.read_csv(RESULT_ROOT / "retrieved_real_B_cnmse.csv")
    assert np.all(
        diagnostics["nearest_nonself_B_state"].to_numpy(dtype=int)
        != diagnostics["State_n_R"].to_numpy(dtype=int)
    )


def test_nonexact_regret_is_nonnegative_when_finite() -> None:
    retrieval = pd.read_csv(RESULT_ROOT / "retrieval_results.csv")
    diagnostics = pd.read_csv(RESULT_ROOT / "retrieved_real_B_cnmse.csv")
    finite = diagnostics.loc[
        ~retrieval["exact_hit"].to_numpy(dtype=bool), "nonself_behavioral_regret_dB"
    ].to_numpy(dtype=float)
    finite = finite[np.isfinite(finite)]
    assert np.all(finite >= -1e-10)


def test_threshold_sweep_is_descriptive() -> None:
    validation = _load_validation()
    table = pd.read_csv(RESULT_ROOT / "threshold_sweep.csv")
    assert table.shape[0] == 5
    assert validation["behavioral_success_threshold_frozen"] is False
    assert validation["threshold_sweep_values_dB"] == [-20.0, -25.0, -30.0, -35.0, -40.0]


def test_summary_counts_and_rank_range() -> None:
    summary = pd.read_csv(RESULT_ROOT / "retrieval_summary.csv").iloc[0]
    assert int(summary["state_count"]) == 425
    assert int(summary["candidate_count_per_query"]) == 425
    assert int(summary["non_exact_count"]) + int(summary["exact_hit_count"]) == 425
    assert 1 <= float(summary["true_state_rank_median"]) <= 425
    assert 1 <= int(summary["true_state_rank_max"]) <= 425


def test_c3_effective_stage_rule() -> None:
    validation = _load_validation()
    assert validation["query_effective_rule"] == "effective_C_n = min(3, Ns)"
    assert validation["query_effective_rule_valid"] is True


def test_c3_to_a2_and_real_b_rankings_regress() -> None:
    validation = _load_validation()
    assert validation["c3_to_a2_ranking_regression"]["ranking_identical"] is True
    assert validation["real_B_ranking_regression"]["ranking_identical"] is True
    assert validation["c3_to_a2_ranking_regression"]["pass"] is True
    assert validation["real_B_ranking_regression"]["pass"] is True


def test_figures_are_readable() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name in (
        "figure1_state_retrieval_mapping.png",
        "figure2_retrieved_real_B_cnmse_by_state.png",
        "figure3_retrieved_real_B_cnmse_distribution.png",
    ):
        image = plt.imread(RESULT_ROOT / name)
        assert image.size > 0
        assert image.ndim in (2, 3)
    plt.close("all")


def test_protected_sources_and_raw_manifest_unchanged() -> None:
    expected = {
        PROJECT_ROOT / "results" / "behavior_fingerprint_ranking_consistency" /"scenario_2" / "scenario_2":  # noqa: E501
            "7a7e30b8fe5901c254d4b4815c07fabcfe612e9aafa2f7df45a38ca3e4fe9abe",
        PROJECT_ROOT
        / "results"
        / "behavior_fingerprint_ranking_consistency"
        / "scenario_2" /"scenario_2_all_ilc":
            "513fc7a5f0982ca29b7a00ae90accfc2b041396821e46d3fdfa312ae25b667a1",
        PROJECT_ROOT
        / "results"
        / "behavior_fingerprint_ranking_consistency"
        / "scenario_2" /"scenario_2_y_lut_fusion":
            "146369951909106e315fcc0272079b89603bb4f67d61c1442e750a17333d412b",
        A123_ROOT:
            "68f405fe4ea6e27bdb3524d9556e76922635370570b98d949889c3978feb742a",
        PROJECT_ROOT
        / "results"
        / "behavior_fingerprint_ranking_consistency"
        / "scenario_2" /"scenario_2_y_lut_weighted_fusion":
            "4203fccd4751f486ebcc9091f3c89b0928c01e28f2f1763b4961b5491e902c93",
        PROJECT_ROOT / "scripts" / "behavior_fingerprint_ranking_consistency":
            "9fad9fcd47e398928bf0937ead01467076a060e4a40f61ab7859e3776d95c510",
        PROJECT_ROOT / "scripts" / "behavior_modeling":
            "f503577b541220588069434d364b1171a51487a01e9a8231e86fd37d6cbd5127",
        PROJECT_ROOT / "scripts" / "signal_segmentation":
            "bf804cb139a68fdb336865fff514831b14b0f56b178f5840e5dedef92e1311ef",
    }
    for path, digest in expected.items():
        assert _tree_digest(path) == digest
    assert _raw_manifest_digest() == (
        "910b06e4be6b60b1f4dc63409a3083eb328765347fbec3b89a97a41d6633a1e0"
    )


def test_validation_records_no_raw_or_model_changes() -> None:
    validation = _load_validation()
    assert validation["raw_data_modified"] is False
    assert validation["raw_data_read"] is False
    assert validation["models_retrained"] is False
    assert validation["ridge_rescanned"] is False
    assert validation["canonical_changed"] is False
    assert validation["real_B_used_as_LUT"] is False
    assert validation["real_B_used_as_reference"] is True


def main() -> None:
    tests = [
        test_synthetic_cnmse_direction_and_argmin,
        test_synthetic_tie_uses_lowest_state_id,
        test_exact_distance_is_negative_infinity,
        test_circular_difference_contract,
        test_frozen_shapes_and_dtypes,
        test_candidate_and_matrix_contract,
        test_retrieval_has_425_rows_and_topk_fields,
        test_real_b_diagnostics_preserve_exact_negative_infinity,
        test_oracle_excludes_self,
        test_nonexact_regret_is_nonnegative_when_finite,
        test_threshold_sweep_is_descriptive,
        test_summary_counts_and_rank_range,
        test_c3_effective_stage_rule,
        test_c3_to_a2_and_real_b_rankings_regress,
        test_figures_are_readable,
        test_protected_sources_and_raw_manifest_unchanged,
        test_validation_records_no_raw_or_model_changes,
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: PASS")


if __name__ == "__main__":
    main()
