"""行为指纹排序一致性模块的契约、方向、输出结构和保护测试。"""

from __future__ import annotations

import json
import sys
from functools import lru_cache
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

from core.shared.metrics import cnmse  # noqa: E402

from behavior_fingerprint_ranking_consistency.shared.distance_matrix import (  # noqa: E402
    compute_cnmse_distance_matrix,
)
from behavior_fingerprint_ranking_consistency.shared.plotting import _spearman_ylim  # noqa: E402
from behavior_fingerprint_ranking_consistency.shared.ranking import (  # noqa: E402
    distance_to_ranks,
    spearman_statistic,
)

RESULT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "behavior_fingerprint_ranking_consistency"
    / "scenario_2" /"scenario2_ranking_consistency"
    / "scenario_2"
)


@lru_cache(maxsize=1)
def _load_npz(name: str) -> dict[str, np.ndarray]:
    with np.load(RESULT_ROOT / name, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _load_validation() -> dict[str, object]:
    with (RESULT_ROOT / "validation.json").open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


def test_frozen_ridge_source_and_model_contract() -> None:
    validation = _load_validation()
    assert validation["ridge_lambda"] == 1e-8
    assert validation["model_ids"] == ["X-A", "Y-A-1", "Y-C-1"]
    assert validation["used_frozen_ridge_models"] is True
    assert validation["retrained_behavior_models"] is False
    assert validation["only_ilc_iteration_1_used"] is True


def test_fingerprint_shapes_and_finiteness() -> None:
    data = _load_npz("fingerprints.npz")
    assert data["state_ids"].shape == (425,)
    assert np.array_equal(data["state_ids"], np.arange(425))
    assert data["common_B_input"].shape == (4915,)
    fingerprint_names = (
        "X_A_fingerprints",
        "Y_A_fingerprints",
        "Y_C_query_fingerprints",
        "real_B_fingerprints",
    )
    for name in fingerprint_names:
        assert data[name].shape == (425, 4913)
        assert data[name].dtype == np.complex128
        assert np.all(np.isfinite(data[name]))


def test_common_b_probe_validation() -> None:
    validation = _load_validation()
    assert validation["common_B_input_consistent"] is True
    assert validation["common_B_input_shape"] == [4915]
    assert validation["fingerprint_length"] == 4913
    assert validation["common_B_max_abs_difference"] <= 1e-12


def test_cnmse_query_candidate_direction() -> None:
    query = np.zeros((1, 4913), dtype=np.complex128)
    query[0, 0] = 1.0 + 0.0j
    same = query.copy()
    different = np.zeros((1, 4913), dtype=np.complex128)
    different[0, 1] = 1.0 + 0.0j
    matrix = compute_cnmse_distance_matrix(query, np.vstack([same, different]))
    assert matrix[0, 0] == -np.inf
    assert matrix[0, 0] < matrix[0, 1]
    np.testing.assert_equal(matrix[0, 0], cnmse(query[0], same[0]))


def test_self_match_and_matrix_shapes() -> None:
    distances = _load_npz("distance_matrices.npz")
    rankings = _load_npz("ranking_matrices.npz")
    for name in ("D_RR", "D_CX", "D_CY", "D_CR"):
        assert distances[name].shape == (425, 425)
        assert not np.isnan(distances[name]).any()
        assert not np.isposinf(distances[name]).any()
    for name in ("R_RR", "R_CX", "R_CY", "R_CR"):
        assert rankings[name].shape == (425, 425)
        assert np.all(np.isfinite(rankings[name]))
    d_rr = distances["D_RR"]
    assert np.all(d_rr[np.arange(425), np.arange(425)] <= np.min(d_rr, axis=1))


def test_rankdata_average_ties_and_negative_infinity() -> None:
    row = np.concatenate(([-np.inf, -2.0, -2.0, 1.0], np.arange(2.0, 423.0)))
    padded = np.vstack([row] * 425)
    ranks = distance_to_ranks(padded)
    np.testing.assert_allclose(ranks[0, :4], [1.0, 2.5, 2.5, 4.0])


def test_spearman_same_and_reverse_order() -> None:
    reference = np.asarray([1.0, 2.0, 3.0, 4.0])
    same = np.asarray([1.0, 2.0, 3.0, 4.0])
    reverse = np.asarray([4.0, 3.0, 2.0, 1.0])
    np.testing.assert_allclose(spearman_statistic(reference, same), 1.0)
    np.testing.assert_allclose(spearman_statistic(reference, reverse), -1.0)


def test_reference_direction_is_real_real() -> None:
    validation = _load_validation()
    assert validation["reference_ranking"] == "R_RR = Real-B to Real-B"
    assert validation["distance_direction"].startswith("CNMSE(Query, Candidate)")


def test_spearman_counts_and_range() -> None:
    table = pd.read_csv(RESULT_ROOT / "spearman_by_state.csv")
    assert table.shape[0] == 425
    assert np.array_equal(table["state_id"].to_numpy(), np.arange(425))
    for column in ("spearman_X_A_LUT", "spearman_Y_A_LUT", "spearman_Real_B_LUT"):
        values = table[column].to_numpy(dtype=float)
        assert np.all(np.isfinite(values))
        assert np.all((values >= -1.0) & (values <= 1.0))
    summary = pd.read_csv(RESULT_ROOT / "spearman_summary.csv")
    assert summary.shape == (3, 9)
    assert summary["count"].tolist() == [425, 425, 425]


def test_two_formal_figures_exist_and_are_readable() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for name in ("spearman_boxplot_three_luts.png", "spearman_by_state_three_luts.png"):
        path = RESULT_ROOT / name
        assert path.is_file()
        assert path.stat().st_size > 0
        image = plt.imread(path)
        assert image.size > 0
        assert image.ndim in (2, 3)
    plt.close("all")


def test_plot_y_limits_follow_actual_spearman_extrema() -> None:
    table = pd.read_csv(RESULT_ROOT / "spearman_by_state.csv")
    values = [
        table[column].to_numpy(dtype=float)
        for column in (
            "spearman_X_A_LUT",
            "spearman_Y_A_LUT",
            "spearman_Real_B_LUT",
        )
    ]
    y_min, y_max = _spearman_ylim(values)
    combined = np.concatenate(values)
    assert y_min == float(np.min(combined))
    assert y_max == float(np.max(combined))


def test_validation_matrix_sanity_counts() -> None:
    validation = _load_validation()
    assert validation["state_count"] == 425
    assert validation["candidate_count_per_query"] == 425
    assert validation["self_match_included"] is True
    assert validation["all_spearman_finite"] is True
    matrix_validation = validation["matrix_validation"]
    assert matrix_validation["D_RR_self_row_min_count"] == 425
    assert matrix_validation["R_RR_self_min_tied_rank_count"] == 425


def main() -> None:
    tests = [
        test_frozen_ridge_source_and_model_contract,
        test_fingerprint_shapes_and_finiteness,
        test_common_b_probe_validation,
        test_cnmse_query_candidate_direction,
        test_self_match_and_matrix_shapes,
        test_rankdata_average_ties_and_negative_infinity,
        test_spearman_same_and_reverse_order,
        test_reference_direction_is_real_real,
        test_spearman_counts_and_range,
        test_two_formal_figures_exist_and_are_readable,
        test_plot_y_limits_follow_actual_spearman_extrema,
        test_validation_matrix_sanity_counts,
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: PASS")


if __name__ == "__main__":
    main()
