"""
功能说明：验证State0 11模型、真实B泛化、公共xin_B响应和21个CNMSE的冻结统计逻辑。
输入：data_manager和signal_segmentation生成的State0 canonical OFF/ILC数据。
输出：模型数量、theta/rank、维度、probe来源、泛化来源和CNMSE方向测试PASS信息。
用途：防止将Y真实B泛化与公共B-Probe等效验证混淆，且不写任务结果。
"""

from __future__ import annotations

import sys
from functools import lru_cache
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

from data_management.shared import load_by_id  # noqa: E402
from signal_segmentation.shared import (  # noqa: E402
    build_ilc_segments,
    build_partition_from_xin,
    get_common_probe,
)

from behavior_modeling.shared import (  # noqa: E402
    MAX_DELAY,
    calculate_cnmse,
    calculate_nmse,
    predict_behavior,
)
from behavior_modeling.shared.xy_equivalence import (  # noqa: E402
    State0XYEquivalenceResult,
    analyze_state0_xy_equivalence,
)


@lru_cache(maxsize=1)
def _result() -> State0XYEquivalenceResult:
    return analyze_state0_xy_equivalence()


def test_model_and_metric_counts() -> None:
    result = _result()
    items = result.evaluations
    assert len(items) == 11
    assert sum(item.model_id == "X-A" for item in items) == 1
    assert sum(item.model_id.startswith("Y-A-") for item in items) == 5
    assert sum(item.model_id.startswith("Y-C-") for item in items) == 5
    assert len([item.train_nmse_db for item in items]) == 11
    assert len([item.b_generalization_nmse_db for item in items]) == 11


def test_theta_rank_and_finiteness() -> None:
    for item in _result().evaluations:
        assert item.theta.shape == (10,)
        assert np.iscomplexobj(item.theta)
        assert np.all(np.isfinite(item.theta))
        assert item.rank == 10


def test_local_model_valid_lengths() -> None:
    result = _result()
    assert result.common_probe.shape == (4915,)
    assert result.common_probe_valid.shape == (4913,)
    assert result.real_b_valid.shape == (4913,)
    assert all(response.shape == (4913,) for response in result.common_responses.values())
    expected_train_samples = {"A": 12286, "C": 7371}
    for item in result.evaluations:
        assert item.n_train_samples == expected_train_samples[item.train_segment]


def test_every_model_uses_identical_common_probe() -> None:
    result = _result()
    for item in result.evaluations:
        recomputed = predict_behavior(result.common_probe, item.theta)
        np.testing.assert_array_equal(recomputed, item.common_b_response)


def test_y_b_generalization_uses_own_ilc_b() -> None:
    result = _result()
    state = load_by_id(0)
    partition = build_partition_from_xin(np.asarray(state["xin"]))
    ilc = build_ilc_segments(state, partition)
    for item in result.evaluations:
        if item.behavior_class != "Y":
            continue
        iteration_index = int(item.ilc_iteration) - 1
        canonical_b = ilc[iteration_index]["B"]
        prediction = predict_behavior(canonical_b.input, item.theta)
        expected_nmse = calculate_nmse(canonical_b.output[MAX_DELAY:], prediction)
        np.testing.assert_allclose(
            expected_nmse,
            item.b_generalization_nmse_db,
            rtol=0,
            atol=1e-12,
        )
        assert item.b_generalization_source == f"ILC[{item.ilc_iteration}].B"


def test_cnmse_counts_and_reference_directions() -> None:
    result = _result()
    items = result.evaluations
    x_item = items[0]
    assert np.isnan(x_item.common_b_vs_x_cnmse_db)
    np.testing.assert_allclose(
        x_item.common_b_vs_real_b_cnmse_db,
        calculate_cnmse(result.real_b_valid, x_item.common_b_response),
        rtol=0,
        atol=1e-12,
    )
    y_items = [item for item in items if item.behavior_class == "Y"]
    for item in y_items:
        np.testing.assert_allclose(
            item.common_b_vs_x_cnmse_db,
            calculate_cnmse(x_item.common_b_response, item.common_b_response),
            rtol=0,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            item.common_b_vs_real_b_cnmse_db,
            calculate_cnmse(result.real_b_valid, item.common_b_response),
            rtol=0,
            atol=1e-12,
        )
    assert len(y_items) == 10
    assert sum(not np.isnan(item.common_b_vs_x_cnmse_db) for item in y_items) == 10
    assert sum(not np.isnan(item.common_b_vs_real_b_cnmse_db) for item in y_items) == 10
    total = sum(
        not np.isnan(value)
        for item in items
        for value in (
            item.common_b_vs_x_cnmse_db,
            item.common_b_vs_real_b_cnmse_db,
        )
    )
    assert total == 21


def test_common_probe_is_off_xin_b() -> None:
    result = _result()
    state = load_by_id(0)
    partition = build_partition_from_xin(np.asarray(state["xin"]))
    expected = get_common_probe(state, partition)
    np.testing.assert_array_equal(result.common_probe, expected)
    np.testing.assert_array_equal(result.common_probe_valid, expected[MAX_DELAY:])


def main() -> None:
    tests = [
        ("model and metric count test", test_model_and_metric_counts),
        ("theta rank finite test", test_theta_rank_and_finiteness),
        ("local valid length test", test_local_model_valid_lengths),
        ("identical common probe test", test_every_model_uses_identical_common_probe),
        ("Y own-B generalization test", test_y_b_generalization_uses_own_ilc_b),
        ("CNMSE direction and count test", test_cnmse_counts_and_reference_directions),
        ("common probe source test", test_common_probe_is_off_xin_b),
    ]
    for label, test in tests:
        test()
        print(f"{label}: PASS")


if __name__ == "__main__":
    main()
