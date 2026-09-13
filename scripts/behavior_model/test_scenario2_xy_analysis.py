"""
功能说明：验证正式ILC1三模型定义、State0回归、canonical/B泛化/probe来源、数量和输出结构。
输入：State0与少量真实状态、现有State0 OLS基线和data_manager状态字典。
输出：正式论文Y定义保护测试PASS；不写正式results目录。
用途：防止ILC2以后、processed输出、STALE、Ridge或错误CNMSE方向进入全状态基线。
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SCRIPTS_ROOT.parent
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from data_manager import load_by_id  # noqa: E402
from signal_segmentation import (  # noqa: E402
    build_partition_from_xin,
    get_common_probe,
    get_ilc_pair,
    preprocess_full_pair,
)

from behavior_model import calculate_cnmse  # noqa: E402
from behavior_model.scenario2_xy_analysis import (  # noqa: E402
    MODEL_IDS,
    analyze_state_xy,
    validate_state0_regression,
)


@lru_cache(maxsize=8)
def _state_result(state_id: int):
    return analyze_state_xy(state_id)


def test_state0_regression_metrics_and_theta() -> None:
    baseline_root = PROJECT_ROOT / "results" / "behavior_model" / "state_000" / "xy_equivalence"
    validation = validate_state0_regression(_state_result(0), baseline_root, atol=1e-10)
    assert validation["state0_regression_passed"] is True
    assert validation["state0_metric_max_abs_error"] <= 1e-10
    assert validation["state0_theta_max_abs_error"] <= 1e-10


def test_only_three_models_and_only_ilc1() -> None:
    result = _state_result(0)
    assert tuple(model.model_id for model in result.models) == MODEL_IDS
    assert len(result.models) == 3
    assert result.only_ilc_iteration_1_used is True
    assert [model.ilc_iteration_used for model in result.models] == [None, 1, 1]


def test_all_states_have_ilc_iteration_1() -> None:
    for state_id in range(425):
        data = load_by_id(state_id)
        assert np.asarray(data["xin_pd_ori_ilc"]).shape[1] >= 1
        assert np.asarray(data["yout_withdpd_ori_ilc"]).shape[1] >= 1


def test_canonical_and_b_generalization_sources() -> None:
    result = _state_result(0)
    sources = {
        model.model_id: (model.train_source, model.B_generalization_source)
        for model in result.models
    }
    assert sources == {
        "X-A": ("OFF.A", "OFF.B"),
        "Y-A-1": ("ILC1.A", "ILC1.B"),
        "Y-C-1": ("ILC1.C", "ILC1.B"),
    }
    assert all(model.common_probe_source == "OFF.B.input" for model in result.models)


def test_common_probe_and_cnmse_directions() -> None:
    result = _state_result(0)
    data = load_by_id(0)
    partition = build_partition_from_xin(np.asarray(data["xin"]))
    expected_probe = get_common_probe(data, partition)
    np.testing.assert_array_equal(result.common_probe, expected_probe)
    x_model, y_a_model, y_c_model = result.models
    assert np.isnan(x_model.commonB_vs_X_cnmse_db)
    np.testing.assert_allclose(
        y_a_model.commonB_vs_X_cnmse_db,
        calculate_cnmse(x_model.common_B_response, y_a_model.common_B_response),
        rtol=0,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        y_c_model.commonB_vs_X_cnmse_db,
        calculate_cnmse(x_model.common_B_response, y_c_model.common_B_response),
        rtol=0,
        atol=1e-12,
    )
    for model in result.models:
        np.testing.assert_allclose(
            model.commonB_vs_realB_cnmse_db,
            calculate_cnmse(result.real_B_valid, model.common_B_response),
            rtol=0,
            atol=1e-12,
        )


def test_lengths_theta_rank_and_finiteness() -> None:
    for state_id in (0, 100, 424):
        result = _state_result(state_id)
        assert result.common_probe.shape == (4915,)
        assert result.real_B_valid.shape == (4913,)
        assert result.theta.shape == (3, 10)
        assert result.theta.dtype == np.complex128
        assert np.all(np.isfinite(result.theta))
        for model in result.models:
            assert model.common_B_response.shape == (4913,)
            assert model.coefficient_count == 10
            assert model.n_train in (12286, 7371)


def test_ilc1_pair_is_explicitly_preprocessed_without_other_iterations() -> None:
    data = load_by_id(0)
    partition = build_partition_from_xin(np.asarray(data["xin"]))
    pair = get_ilc_pair(data, 0)
    canonical = preprocess_full_pair(
        pair.input_full,
        pair.output_raw_full,
        partition,
        pair_type="ILC",
        iteration_index=0,
        input_peak_normalization_factor=pair.input_peak_normalization_factor,
    )
    raw_first_column = np.asarray(data["xin_pd_ori_ilc"])[:, 0]
    expected_actual_input = raw_first_column / np.max(np.abs(raw_first_column))
    np.testing.assert_array_equal(pair.input_full, expected_actual_input)
    np.testing.assert_array_equal(
        canonical["A"].input,
        expected_actual_input[partition.a_slice],
    )
    assert canonical.iteration_index == 0
    assert np.isclose(
        np.max(np.abs(canonical["A"].input)),
        np.max(np.abs(pair.input_full[partition.a_slice])),
    )


def test_small_batch_result_shapes() -> None:
    subset_results = tuple(_state_result(state_id) for state_id in (0, 1, 2))
    assert len(subset_results) == 3
    assert sum(len(result.models) for result in subset_results) == 9
    assert all(result.only_ilc_iteration_1_used for result in subset_results)


def main() -> None:
    tests = [
        ("State0 metric theta regression test", test_state0_regression_metrics_and_theta),
        ("three-model ILC1-only test", test_only_three_models_and_only_ilc1),
        ("all-state ILC1 availability test", test_all_states_have_ilc_iteration_1),
        ("canonical and B-source test", test_canonical_and_b_generalization_sources),
        ("common-probe CNMSE-direction test", test_common_probe_and_cnmse_directions),
        ("length theta rank finite test", test_lengths_theta_rank_and_finiteness),
        (
            "explicit ILC1 preprocessing test",
            test_ilc1_pair_is_explicitly_preprocessed_without_other_iterations,
        ),
        ("small-batch shape test", test_small_batch_result_shapes),
    ]
    for label, test in tests:
        test()
        print(f"{label}: PASS")


if __name__ == "__main__":
    main()
