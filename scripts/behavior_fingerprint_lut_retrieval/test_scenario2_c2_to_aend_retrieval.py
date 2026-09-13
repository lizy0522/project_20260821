"""Synthetic contracts, frozen regressions and output-protection tests."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from _core.metrics import cnmse  # noqa: E402
from behavior_fingerprint_ranking_consistency.distance_matrix import (  # noqa: E402
    compute_cnmse_distance_matrix,
)

from behavior_fingerprint_lut_retrieval.scenario2_c2_to_aend_retrieval import (  # noqa: E402
    FINGERPRINT_LENGTH,
    dpd_shareable_mask,
    select_a_end_stage_indices,
    validate_c2_availability,
)

RESULT_ROOT = (
    PROJECT_ROOT / "results" / "behavior_fingerprint_lut_retrieval" / "scenario_2_C2_to_Aend"
)
OLD_ROOT = PROJECT_ROOT / "results" / "behavior_fingerprint_lut_retrieval" / "scenario_2_C3_to_A2"


def _load_validation() -> dict[str, object]:
    with (RESULT_ROOT / "validation.json").open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def _tree_digest(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        (
            item
            for item in path.rglob("*")
            if item.is_file() and "__pycache__" not in item.parts and item.suffix.lower() != ".pyc"
        ),
        key=lambda item: str(item).lower(),
    )
    for file in files:
        content = file.read_bytes()
        digest.update(file.relative_to(path).as_posix().encode() + b"\0")
        digest.update(len(content).to_bytes(8, "little"))
        digest.update(hashlib.sha256(content).digest())
    return digest.hexdigest()


def _raw_manifest_digest() -> str:
    raw_root = Path(r"\\?\D:\Project_Files\python_project\project_20260821\data\raw")
    digest = hashlib.sha256()
    files = sorted(
        (item for item in raw_root.rglob("*") if item.is_file()),
        key=lambda item: str(item).lower(),
    )
    for file in files:
        digest.update(str(file.relative_to(raw_root)).replace("\\", "/").encode() + b"\0")
        digest.update(file.stat().st_size.to_bytes(8, "little"))
    return digest.hexdigest()


def test_a_end_selects_last_available_stage() -> None:
    counts = np.asarray([2, 3, 5], dtype=np.int64)
    np.testing.assert_array_equal(select_a_end_stage_indices(counts), [1, 2, 4])


def test_c2_requires_real_second_column() -> None:
    try:
        validate_c2_availability(np.asarray([1, 2], dtype=np.int64))
    except RuntimeError:
        pass
    else:
        raise AssertionError("Ns=1必须阻止正式C2分析")
    assert validate_c2_availability(np.asarray([2, 3, 5], dtype=np.int64)) is True


def test_strict_shareable_rule_boundaries() -> None:
    values = np.asarray([-40.000001, -40.0, -39.999999, -np.inf])
    np.testing.assert_array_equal(dpd_shareable_mask(values), [True, False, False, True])


def test_cnmse_direction_and_exact_negative_infinity() -> None:
    query = np.zeros((1, FINGERPRINT_LENGTH), dtype=np.complex128)
    query[0, 0] = 1.0
    different = np.zeros_like(query)
    different[0, 1] = 1.0
    matrix = compute_cnmse_distance_matrix(query, np.vstack([different, query]))
    assert matrix[0, 1] == -np.inf
    assert matrix[0, 1] < matrix[0, 0]
    np.testing.assert_equal(matrix[0, 1], cnmse(query[0], query[0]))


def test_stage_map_and_distribution() -> None:
    table = pd.read_csv(RESULT_ROOT / "a_end_stage_map.csv")
    assert table.shape[0] == 425
    assert table["C2_available"].all()
    assert table["N_ilc_available"].value_counts().sort_index().to_dict() == {
        2: 1,
        3: 416,
        4: 7,
        5: 1,
    }
    assert table["ilc_A_end"].value_counts().sort_index().to_dict() == {
        2: 1,
        3: 416,
        4: 7,
        5: 1,
    }


def test_fingerprint_shapes_and_finiteness() -> None:
    with np.load(RESULT_ROOT / "lut_fingerprints_Aend.npz", allow_pickle=False) as lut:
        with np.load(RESULT_ROOT / "query_fingerprints_C2.npz", allow_pickle=False) as query:
            assert lut["Y_Aend_fingerprints"].shape == (425, 4913)
            assert query["Q_C2"].shape == (425, 4913)
            assert lut["Y_Aend_fingerprints"].dtype == np.complex128
            assert query["Q_C2"].dtype == np.complex128
            assert np.all(np.isfinite(lut["Y_Aend_fingerprints"]))
            assert np.all(np.isfinite(query["Q_C2"]))


def test_a_end_and_c2_frozen_regressions() -> None:
    validation = _load_validation()
    assert validation["a_end_fingerprint_regression"]["pass"] is True
    assert validation["a_end_fingerprint_regression"]["max_abs_error"] < 1e-12
    assert validation["c2_fingerprint_regression"]["pass"] is True
    assert validation["c2_fingerprint_regression"]["max_abs_error"] == 0.0


def test_distance_and_ranking_matrix_contract() -> None:
    with np.load(RESULT_ROOT / "retrieval_distance_matrix.npz", allow_pickle=False) as data:
        assert data["D_C2_Aend"].shape == (425, 425)
        assert not np.isnan(data["D_C2_Aend"]).any()
        assert not np.isposinf(data["D_C2_Aend"]).any()
    with np.load(RESULT_ROOT / "retrieval_ranking_matrix.npz", allow_pickle=False) as data:
        assert data["R_C2_Aend"].shape == (425, 425)
        assert np.all(np.isfinite(data["R_C2_Aend"]))


def test_retrieval_results_have_425_rows_and_fields() -> None:
    table = pd.read_csv(RESULT_ROOT / "retrieval_results.csv")
    assert table.shape[0] == 425
    assert np.array_equal(table["State_n_R"].to_numpy(), np.arange(425))
    assert table["State_n_Q"].between(0, 424).all()
    assert table["true_state_rank"].between(1, 425).all()
    for column in (
        "exact_hit",
        "top3_true_state_hit",
        "top5_true_state_hit",
        "Aend_stage_R",
        "Aend_stage_Q",
        "funAng_diff_deg",
        "secAng_diff_deg",
    ):
        assert column in table.columns


def test_real_b_matrix_reuses_r_rr_and_keeps_negative_infinity() -> None:
    validation = _load_validation()
    with np.load(RESULT_ROOT / "real_B_distance_matrix.npz", allow_pickle=False) as data:
        assert data["D_B"].shape == (425, 425)
        assert data["R_B"].shape == (425, 425)
        assert np.all(np.isneginf(np.diag(data["D_B"])))
    assert validation["real_B_regression"]["pass"] is True
    assert validation["real_B_regression"]["ranking_identical_vs_R_RR"] is True
    assert validation["real_B_regression"]["diagonal_all_negative_infinity"] is True


def test_shareable_and_failure_partition() -> None:
    retrieval = pd.read_csv(RESULT_ROOT / "retrieval_results.csv")
    diagnostics = pd.read_csv(RESULT_ROOT / "retrieved_real_B_cnmse.csv")
    values = diagnostics["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float)
    shareable = diagnostics["dpd_shareable"].to_numpy(dtype=bool)
    assert np.array_equal(shareable, values < -40.0)
    assert np.all(
        ~retrieval.loc[shareable, "exact_hit"].to_numpy(dtype=bool) | shareable[shareable]
    )
    failed = pd.read_csv(RESULT_ROOT / "failed_retrieval_states.csv")
    if not failed.empty:
        assert np.all(failed["retrieved_real_B_CNMSE_dB"].to_numpy(dtype=float) >= -40.0)


def test_summary_matches_formal_threshold() -> None:
    validation = _load_validation()
    summary = pd.read_csv(RESULT_ROOT / "retrieval_summary.csv").iloc[0]
    assert summary["DPD_shareable_threshold_dB"] == -40.0
    assert validation["dpd_shareable_threshold_frozen"] is True
    assert validation["threshold_is_strict"] is True
    assert int(summary["DPD_shareable_count"]) == 411
    assert int(summary["failure_count"]) == 14
    assert int(summary["DPD_shareable_count"]) + int(summary["failure_count"]) == 425


def test_oracle_shareable_rank_and_topk_coverage() -> None:
    oracle = pd.read_csv(RESULT_ROOT / "oracle_shareable_rank.csv")
    assert oracle.shape[0] == 425
    assert oracle["shareable_in_top3"].mean() == 0.9952941176470588
    assert oracle["shareable_in_top5"].mean() == 0.9976470588235294
    assert oracle["shareable_in_top10"].all()


def test_condition_region_summary_is_complete() -> None:
    table = pd.read_csv(RESULT_ROOT / "condition_region_summary.csv")
    assert table.shape[0] == 425
    assert table["state_count"].sum() == 425
    assert table["failure_count"].sum() == 14
    assert table["dpd_shareable_count"].sum() == 411


def test_comparison_vs_previous_c3_a2() -> None:
    table = pd.read_csv(RESULT_ROOT / "comparison_vs_C3_to_A2.csv")
    assert set(table["metric"]) >= {
        "exact_hit_rate",
        "DPD_shareable_rate",
        "failure_count",
        "non_exact_B_CNMSE_median",
    }
    indexed = table.set_index("metric")
    assert indexed.loc["DPD_shareable_count", "C2_to_Aend"] == 411
    assert indexed.loc["DPD_shareable_count", "C3_to_A2"] == 407
    assert indexed.loc["failure_count", "C2_to_Aend"] == 14
    assert indexed.loc["failure_count", "C3_to_A2"] == 18


def test_circular_difference_examples() -> None:
    from behavior_fingerprint_lut_retrieval.scenario2_c2_to_aend_retrieval import (
        circular_difference_deg,
    )

    assert circular_difference_deg(0, 315) == 45.0
    assert circular_difference_deg(315, 0) == 45.0
    assert circular_difference_deg(45, 315) == 90.0


def test_four_figures_are_readable() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name in (
        "figure1_state_retrieval_mapping.png",
        "figure2_real_B_cnmse_by_state.png",
        "figure3_real_B_cnmse_distribution.png",
        "figure4_dpd_shareable_success_by_state.png",
    ):
        image = plt.imread(RESULT_ROOT / name)
        assert image.size > 0
        assert image.ndim in (2, 3)
    plt.close("all")


def test_validation_scope_flags() -> None:
    validation = _load_validation()
    assert validation["query_fingerprint_type"] == "Y-C2"
    assert validation["lut_fingerprint_type"] == "Y-Aend"
    assert validation["query_saturation_allowed"] is False
    assert validation["real_B_used_as_LUT"] is False
    assert validation["real_B_used_as_reference"] is True
    assert validation["models_retrained"] is False
    assert validation["ridge_rescanned"] is False
    assert validation["ABC_changed"] is False
    assert validation["canonical_changed"] is False
    assert validation["raw_data_modified"] is False


def test_previous_c3_a2_and_other_protected_sources_unchanged() -> None:
    expected = {
        PROJECT_ROOT / "results" / "behavior_fingerprint_ranking_consistency" / "scenario_2":
            "7a7e30b8fe5901c254d4b4815c07fabcfe612e9aafa2f7df45a38ca3e4fe9abe",
        PROJECT_ROOT
        / "results"
        / "behavior_fingerprint_ranking_consistency"
        / "scenario_2_all_ilc":
            "513fc7a5f0982ca29b7a00ae90accfc2b041396821e46d3fdfa312ae25b667a1",
        PROJECT_ROOT
        / "results"
        / "behavior_fingerprint_ranking_consistency"
        / "scenario_2_y_lut_fusion":
            "146369951909106e315fcc0272079b89603bb4f67d61c1442e750a17333d412b",
        PROJECT_ROOT
        / "results"
        / "behavior_fingerprint_ranking_consistency"
        / "scenario_2_y_lut_fusion_A123":
            "68f405fe4ea6e27bdb3524d9556e76922635370570b98d949889c3978feb742a",
        PROJECT_ROOT
        / "results"
        / "behavior_fingerprint_ranking_consistency"
        / "scenario_2_y_lut_weighted_fusion":
            "4203fccd4751f486ebcc9091f3c89b0928c01e28f2f1763b4961b5491e902c93",
        OLD_ROOT:
            "a6c9e289c5771003f3d9681ecae9b1278a50b4b81c5f2b4acb11d30a296b41f1",
        PROJECT_ROOT / "scripts" / "behavior_fingerprint_ranking_consistency":
            "9fad9fcd47e398928bf0937ead01467076a060e4a40f61ab7859e3776d95c510",
        PROJECT_ROOT / "scripts" / "behavior_model":
            "f503577b541220588069434d364b1171a51487a01e9a8231e86fd37d6cbd5127",
        PROJECT_ROOT / "scripts" / "signal_segmentation":
            "bf804cb139a68fdb336865fff514831b14b0f56b178f5840e5dedef92e1311ef",
    }
    for path, digest in expected.items():
        assert _tree_digest(path) == digest
    assert _raw_manifest_digest() == (
        "910b06e4be6b60b1f4dc63409a3083eb328765347fbec3b89a97a41d6633a1e0"
    )


def main() -> None:
    tests = [
        test_a_end_selects_last_available_stage,
        test_c2_requires_real_second_column,
        test_strict_shareable_rule_boundaries,
        test_cnmse_direction_and_exact_negative_infinity,
        test_stage_map_and_distribution,
        test_fingerprint_shapes_and_finiteness,
        test_a_end_and_c2_frozen_regressions,
        test_distance_and_ranking_matrix_contract,
        test_retrieval_results_have_425_rows_and_fields,
        test_real_b_matrix_reuses_r_rr_and_keeps_negative_infinity,
        test_shareable_and_failure_partition,
        test_summary_matches_formal_threshold,
        test_oracle_shareable_rank_and_topk_coverage,
        test_condition_region_summary_is_complete,
        test_comparison_vs_previous_c3_a2,
        test_circular_difference_examples,
        test_four_figures_are_readable,
        test_validation_scope_flags,
        test_previous_c3_a2_and_other_protected_sources_unchanged,
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: PASS")


if __name__ == "__main__":
    main()
