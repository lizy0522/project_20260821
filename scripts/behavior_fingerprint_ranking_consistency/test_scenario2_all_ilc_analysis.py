"""Scenario 2 全 ILC 扫描的有效迭代、模型、矩阵、图和保护契约测试。"""

from __future__ import annotations

import hashlib
import json
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from data_manager import load_by_id  # noqa: E402

from behavior_fingerprint_ranking_consistency.plotting import (  # noqa: E402
    _all_ilc_series,
    _spearman_ylim,
)
from behavior_fingerprint_ranking_consistency.scenario2_all_ilc_analysis import (  # noqa: E402
    effective_iteration,
)

RESULT_ROOT = (
    PROJECT_ROOT / "results" / "behavior_fingerprint_ranking_consistency" / "scenario_2_all_ilc"
)
SOURCE_ROOT = PROJECT_ROOT / "results" / "behavior_fingerprint_ranking_consistency" / "scenario_2"


@lru_cache(maxsize=1)
def _validation() -> dict[str, object]:
    with (RESULT_ROOT / "validation.json").open("r", encoding="utf-8-sig") as handle:
        return json.load(handle)


@lru_cache(maxsize=1)
def _availability() -> pd.DataFrame:
    return pd.read_csv(RESULT_ROOT / "ilc_availability.csv")


def test_effective_iteration_capping_contract() -> None:
    assert effective_iteration(1, 2) == (1, 0, False)
    assert effective_iteration(2, 2) == (2, 1, False)
    assert effective_iteration(3, 2) == (2, 1, True)
    assert effective_iteration(5, 2) == (2, 1, True)
    assert effective_iteration(3, 5) == (3, 2, False)


def test_dynamic_ilc_availability_and_column_pairing() -> None:
    availability = _availability()
    validation = _validation()
    assert availability.shape[0] == 425
    assert availability["ilc_column_count"].min() >= 1
    assert int(availability["ilc_column_count"].max()) == validation["global_ilc_max"]
    assert validation["ilc_input_output_columns_equal"] is True
    assert sum(validation["ilc_availability_summary"].values()) == 425


def test_effective_iteration_map_has_statewise_saturation() -> None:
    availability = _availability().set_index("state_id")
    mapping = pd.read_csv(RESULT_ROOT / "effective_iteration_map.csv")
    nmax = int(_validation()["global_ilc_max"])
    assert mapping.shape[0] == 425 * nmax
    for row in mapping.itertuples(index=False):
        expected = min(
            int(row.requested_n),
            int(availability.loc[row.state_id, "ilc_column_count"]),
        )
        assert int(row.effective_n) == expected
        assert int(row.python_index) == expected - 1
        assert bool(row.is_saturated) == (int(row.requested_n) > int(row.actual_max_n))


def test_actual_theta_slots_and_model_counts() -> None:
    validation = _validation()
    with np.load(RESULT_ROOT / "all_ilc_theta.npz", allow_pickle=False) as data:
        nmax = int(validation["global_ilc_max"])
        mask = np.asarray(data["available_mask"])
        theta_a = np.asarray(data["theta_Y_A_actual"])
        theta_c = np.asarray(data["theta_Y_C_actual"])
        assert mask.shape == (425, nmax)
        assert theta_a.shape == (425, nmax, 10)
        assert theta_c.shape == (425, nmax, 10)
        assert mask.dtype == np.bool_
        assert np.all(np.isfinite(theta_a[mask]))
        assert np.all(np.isfinite(theta_c[mask]))
        assert np.all(np.isnan(theta_a[~mask]))
        assert np.all(np.isnan(theta_c[~mask]))
    metrics = pd.read_csv(RESULT_ROOT / "all_ilc_model_metrics.csv")
    expected = int(_availability()["ilc_column_count"].sum())
    assert metrics.shape[0] == 2 * expected
    assert metrics["model_role"].value_counts().to_dict() == {
        "Y-A": expected,
        "Y-C": expected,
    }
    assert np.all(np.isfinite(metrics["train_NMSE_dB"]))
    assert np.all(np.isfinite(metrics["B_generalization_NMSE_dB"]))
    assert np.all(metrics["rank"] == 10)


def test_each_ilc_column_has_its_own_peak_factor() -> None:
    data = load_by_id(0)
    input_history = np.asarray(data["xin_pd_ori_ilc"])
    metrics = pd.read_csv(RESULT_ROOT / "all_ilc_model_metrics.csv")
    state0 = metrics[metrics["state_id"] == 0].sort_values(["actual_ilc_n", "model_role"])
    for actual_n in range(1, input_history.shape[1] + 1):
        expected_peak = float(np.max(np.abs(input_history[:, actual_n - 1])))
        observed = state0.loc[
            state0["actual_ilc_n"] == actual_n,
            "input_peak_normalization_factor",
        ].to_numpy(dtype=float)
        assert observed.size == 2
        np.testing.assert_allclose(observed, expected_peak, rtol=0, atol=1e-12)


def test_ilc1_regression_and_frozen_structure() -> None:
    validation = _validation()
    assert validation["ILC1_regression_pass"] is True
    assert validation["ILC1_Y_A_max_abs_error"] <= 1e-10
    assert validation["ILC1_Y_C_max_abs_error"] <= 1e-10
    assert validation["ridge_lambda"] == 1e-8
    assert validation["ridge_lambda_rescanned"] is False
    assert validation["mp_structure_changed"] is False
    assert validation["canonical_changed"] is False


def test_all_distance_and_ranking_matrices_have_dynamic_shapes() -> None:
    validation = _validation()
    nmax = int(validation["global_ilc_max"])
    with np.load(RESULT_ROOT / "distance_matrices_all_ilc.npz", allow_pickle=False) as data:
        distance_names = tuple(data.files)
        assert "state_ids" in distance_names
        assert "D_RR" in distance_names
        distance_count = len([name for name in distance_names if name != "state_ids"])
        assert distance_count == nmax * (nmax + 2) + 1
        for name in distance_names:
            if name != "state_ids":
                assert data[name].shape == (425, 425)
                assert not np.isnan(data[name]).any()
                assert not np.isposinf(data[name]).any()
    with np.load(RESULT_ROOT / "ranking_matrices_all_ilc.npz", allow_pickle=False) as data:
        ranking_names = tuple(data.files)
        assert "R_RR" in ranking_names
        assert len([name for name in ranking_names if name != "state_ids"]) == nmax * (nmax + 2) + 1
        for name in ranking_names:
            if name != "state_ids":
                assert data[name].shape == (425, 425)
                assert np.all(np.isfinite(data[name]))


def test_reference_self_match_and_candidate_count() -> None:
    validation = _validation()
    assert validation["candidate_count_per_query"] == 425
    assert validation["self_match_included"] is True
    assert validation["D_RR_self_row_min_count"] == 425
    assert validation["R_RR_self_rank1_count"] == 425
    assert validation["D_RR_negative_infinity_count"] == 425


def test_spearman_long_summary_counts() -> None:
    validation = _validation()
    nmax = int(validation["global_ilc_max"])
    long_table = pd.read_csv(RESULT_ROOT / "spearman_all_ilc_long.csv")
    expected = 425 * nmax * (nmax + 2)
    assert long_table.shape[0] == expected
    assert long_table["spearman"].notna().all()
    assert long_table["spearman"].between(-1.0, 1.0).all()
    assert long_table["lut_type"].value_counts().to_dict() == {
        "Y-A": 425 * nmax * nmax,
        "X-A": 425 * nmax,
        "Real-B": 425 * nmax,
    }
    summary = pd.read_csv(RESULT_ROOT / "spearman_summary_all_ilc.csv")
    assert summary.shape[0] == nmax * (nmax + 2)
    assert (summary["count"] == 425).all()


def test_per_n1_wide_tables_and_png_count() -> None:
    validation = _validation()
    nmax = int(validation["global_ilc_max"])
    columns, labels = _all_ilc_series(nmax)
    for n1 in range(1, nmax + 1):
        table = pd.read_csv(RESULT_ROOT / f"spearman_n1_{n1:02d}_by_state.csv")
        assert table.shape[0] == 425
        assert np.array_equal(table["state_id"].to_numpy(dtype=int), np.arange(425))
        assert table["requested_n1"].eq(n1).all()
        assert set(columns).issubset(table.columns)
        assert np.all(np.isfinite(table[list(columns)].to_numpy(dtype=float)))
        assert len(labels) == nmax + 2
    pngs = sorted(RESULT_ROOT.glob("n1_*_spearman_*.png"))
    assert len(pngs) == 2 * nmax
    assert all(path.stat().st_size > 0 for path in pngs)


def test_each_figure_uses_actual_minimum_and_maximum_range() -> None:
    validation = _validation()
    nmax = int(validation["global_ilc_max"])
    columns, _ = _all_ilc_series(nmax)
    for n1 in range(1, nmax + 1):
        table = pd.read_csv(RESULT_ROOT / f"spearman_n1_{n1:02d}_by_state.csv")
        values = [table[column].to_numpy(dtype=float) for column in columns]
        lower, upper = _spearman_ylim(values)
        assert lower == float(np.min(np.concatenate(values)))
        assert upper == float(np.max(np.concatenate(values)))
        assert lower < upper


def test_combined_2x5_figure_and_common_y_range() -> None:
    validation = _validation()
    combined_path = Path(validation["combined_2x5_path"])
    assert validation["combined_2x5_subplot_count"] == 10
    assert validation["combined_2x5_y_axis_mode"] == (
        "common actual minimum-to-maximum across all 10 subplots"
    )
    assert combined_path.is_file()
    assert combined_path.stat().st_size > 0
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = plt.imread(combined_path)
    assert image.size > 0
    plt.close("all")
    nmax = int(validation["global_ilc_max"])
    columns, _ = _all_ilc_series(nmax)
    all_values = []
    for n1 in range(1, nmax + 1):
        table = pd.read_csv(RESULT_ROOT / f"spearman_n1_{n1:02d}_by_state.csv")
        all_values.extend(table[column].to_numpy(dtype=float) for column in columns)
    assert validation["combined_2x5_y_min"] == float(np.min(np.concatenate(all_values)))
    assert validation["combined_2x5_y_max"] == float(np.max(np.concatenate(all_values)))


def test_y_a_only_combined_2x5_figure_and_common_y_range() -> None:
    validation = _validation()
    combined_path = Path(validation["combined_2x5_YA_only_path"])
    assert validation["combined_2x5_YA_only_series_count"] == 5
    assert validation["combined_2x5_YA_only_subplots"] == 10
    assert validation["combined_2x5_YA_only_y_axis_mode"] == (
        "common actual minimum-to-maximum across all 10 Y-A-only subplots"
    )
    assert combined_path.is_file()
    assert combined_path.stat().st_size > 0
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = plt.imread(combined_path)
    assert image.size > 0
    plt.close("all")
    nmax = int(validation["global_ilc_max"])
    columns = tuple(f"spearman_Y_A_n2_{n:02d}" for n in range(1, nmax + 1))
    all_values = []
    for n1 in range(1, nmax + 1):
        table = pd.read_csv(RESULT_ROOT / f"spearman_n1_{n1:02d}_by_state.csv")
        all_values.extend(table[column].to_numpy(dtype=float) for column in columns)
    assert validation["combined_2x5_YA_only_y_min"] == float(
        np.min(np.concatenate(all_values))
    )
    assert validation["combined_2x5_YA_only_y_max"] == float(
        np.max(np.concatenate(all_values))
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
    raw_root = Path(r"\\?\D:\Project_Files\python_project\project_20260821\data\raw")
    digest = hashlib.sha256()
    for file in sorted(
        (item for item in raw_root.rglob("*") if item.is_file()),
        key=lambda item: str(item).lower(),
    ):
        digest.update(str(file.relative_to(raw_root)).replace("\\", "/").encode() + b"\0")
        digest.update(file.stat().st_size.to_bytes(8, "little"))
    return digest.hexdigest()


def test_old_results_and_raw_are_unchanged() -> None:
    expected = {
        PROJECT_ROOT / "results" / "behavior_fingerprint_ranking_consistency" / "scenario_2":
            "7a7e30b8fe5901c254d4b4815c07fabcfe612e9aafa2f7df45a38ca3e4fe9abe",
        PROJECT_ROOT / "results" / "behavior_model" / "scenario_2_ridge_analysis":
            "67d89dc336e439022093b126ab94c4984a2d0241db0033945d4b0d6fa5141188",
        PROJECT_ROOT / "results" / "behavior_model" / "scenario_2_xy_analysis":
            "715e1436707e13d23c0e6634dec5ab58b348f8d6485dcd81b60c26f83f2cd48e",
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
        test_effective_iteration_capping_contract,
        test_dynamic_ilc_availability_and_column_pairing,
        test_effective_iteration_map_has_statewise_saturation,
        test_actual_theta_slots_and_model_counts,
        test_each_ilc_column_has_its_own_peak_factor,
        test_ilc1_regression_and_frozen_structure,
        test_all_distance_and_ranking_matrices_have_dynamic_shapes,
        test_reference_self_match_and_candidate_count,
        test_spearman_long_summary_counts,
        test_per_n1_wide_tables_and_png_count,
        test_each_figure_uses_actual_minimum_and_maximum_range,
        test_combined_2x5_figure_and_common_y_range,
        test_y_a_only_combined_2x5_figure_and_common_y_range,
        test_old_results_and_raw_are_unchanged,
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: PASS")


if __name__ == "__main__":
    main()
