"""
功能说明：直接执行或由pytest收集的canonical A/B/C模块单元和state0集成测试。
输入：确定性合成信号以及data_manager只读加载的state0完整原始数据。
输出：Partition、切分、三类数据对、归一化、完整同步和逐段adjust测试PASS信息。
用途：验证ABC ownership冻结且不被对齐、MP、奇数长度或ILC列处理改变。
"""

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

from data_management.shared import load_by_id  # noqa: E402

from signal_segmentation.shared import (  # noqa: E402
    build_ilc_segments,
    build_off_segments,
    build_partition,
    build_partition_from_xin,
    build_stale_segments,
    get_common_probe,
    get_ilc_pair,
    get_off_pair,
    get_stale_pair,
    preprocess_full_pair,
    split_signal,
    validate_current_experiment_partition,
    validate_partition,
)


def _complex_signal(length: int, seed: int = 20260826) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=length) + 1j * rng.normal(size=length)


def _synthetic_state(length: int = 128, iterations: int = 3) -> dict[str, np.ndarray]:
    xin = _complex_signal(length)
    input_history = np.column_stack([(column + 1.5) * xin for column in range(iterations)])
    output_history = np.column_stack([(0.7 + 0.1j * column) * xin for column in range(iterations)])
    return {
        "xin": xin[:, None],
        "yout_withoutdpd_ori": (1.2 - 0.3j) * xin[:, None],
        "xin_pd_ori_ilc": input_history,
        "yout_withdpd_ori_ilc": output_history,
        "xin_pd_nominal": xin[:, None],
        "yout_withdpd_stale_ori": (0.8 + 0.2j) * xin[:, None],
        "nth_inter": np.array([[iterations]]),
    }


def test_current_partition_boundaries() -> None:
    partition = build_partition(24576)
    validate_current_experiment_partition(partition)
    assert partition.a_slice == slice(0, 12288)
    assert partition.b_slice == slice(12288, 17203)
    assert partition.c_slice == slice(17203, 24576)
    assert (partition.n_a, partition.n_b, partition.n_c) == (12288, 4915, 7373)


def test_arbitrary_odd_length_partition() -> None:
    partition = build_partition(101)
    assert (partition.n_a, partition.n_b, partition.n_c) == (50, 20, 31)
    validate_partition(partition)


def test_one_dimensional_split_reconstructs_original() -> None:
    signal = np.arange(101)
    segments = split_signal(signal, build_partition(signal.size))
    np.testing.assert_array_equal(
        np.concatenate([segments["A"], segments["B"], segments["C"]]),
        signal,
    )


def test_two_dimensional_split_preserves_columns() -> None:
    signal = np.arange(24576 * 5).reshape(24576, 5)
    segments = split_signal(signal, build_partition(24576))
    assert segments["A"].shape == (12288, 5)
    assert segments["B"].shape == (4915, 5)
    assert segments["C"].shape == (7373, 5)


def test_partition_has_no_overlap_or_omission() -> None:
    partition = build_partition(24576)
    ownership = np.concatenate(
        [np.arange(partition.total_length)[partition.slice_for(name)] for name in ("A", "B", "C")]
    )
    np.testing.assert_array_equal(ownership, np.arange(partition.total_length))


def test_ilc_peak_normalization_is_column_local() -> None:
    state = _synthetic_state()
    for iteration in range(3):
        pair = get_ilc_pair(state, iteration)
        np.testing.assert_allclose(np.max(np.abs(pair.input_full)), 1.0, atol=1e-14)
        assert pair.input_peak_normalization_factor is not None


def test_zero_peak_ilc_column_raises() -> None:
    state = _synthetic_state()
    state["xin_pd_ori_ilc"][:, 1] = 0
    try:
        get_ilc_pair(state, 1)
    except ValueError as exc:
        assert "峰值必须大于0" in str(exc)
    else:
        raise AssertionError("零峰值ILC列必须抛出ValueError")


def test_full_pair_length_mismatch_raises() -> None:
    x = _complex_signal(128)
    y = _complex_signal(127, seed=9)
    try:
        preprocess_full_pair(x, y, build_partition(128))
    except ValueError as exc:
        assert "长度不一致" in str(exc)
    else:
        raise AssertionError("完整输入输出长度不一致必须报错")


def test_ilc_column_mismatch_raises() -> None:
    state = _synthetic_state()
    state["yout_withdpd_ori_ilc"] = state["yout_withdpd_ori_ilc"][:, :2]
    try:
        get_ilc_pair(state, 0)
    except ValueError as exc:
        assert "shape不一致" in str(exc)
    else:
        raise AssertionError("ILC输入输出列数不一致必须报错")


def test_segment_adjust_is_independent() -> None:
    x = _complex_signal(2048)
    partition = build_partition(x.size)
    y = np.empty_like(x)
    gains = {"A": 1.2 + 0.0j, "B": 0.7 + 0.0j, "C": 1.8 + 0.0j}
    for name, gain in gains.items():
        y[partition.slice_for(name)] = gain * x[partition.slice_for(name)]
    canonical = preprocess_full_pair(x, y, partition, pair_type="TEST", subtime=64)
    for name in ("A", "B", "C"):
        np.testing.assert_allclose(
            canonical[name].output,
            canonical[name].input,
            rtol=1e-12,
            atol=1e-12,
        )


def test_raw_pair_field_contracts() -> None:
    state = _synthetic_state()
    off = get_off_pair(state)
    stale = get_stale_pair(state)
    assert off.pair_type == "OFF"
    assert stale.pair_type == "STALE"
    np.testing.assert_array_equal(off.input_full, state["xin"][:, 0])
    np.testing.assert_array_equal(off.output_raw_full, state["yout_withoutdpd_ori"][:, 0])
    np.testing.assert_array_equal(stale.input_full, state["xin_pd_nominal"][:, 0])
    np.testing.assert_array_equal(
        stale.output_raw_full,
        state["yout_withdpd_stale_ori"][:, 0],
    )


def test_real_state0_off_ilc_stale_integration() -> None:
    state = load_by_id(0)
    partition = build_partition_from_xin(np.asarray(state["xin"]))
    validate_current_experiment_partition(partition)
    off = build_off_segments(state, partition)
    ilc = build_ilc_segments(state, partition)
    stale = build_stale_segments(state, partition)
    assert off.pair_type == "OFF"
    assert stale.pair_type == "STALE"
    assert len(ilc) == 5
    for canonical in (off, *ilc, stale):
        assert tuple(canonical[name].ownership_length for name in ("A", "B", "C")) == (
            12288,
            4915,
            7373,
        )
    for canonical in ilc:
        full_input = np.concatenate([canonical[name].input for name in ("A", "B", "C")])
        np.testing.assert_allclose(np.max(np.abs(full_input)), 1.0, atol=1e-12)
    common_probe = get_common_probe(state, partition)
    np.testing.assert_array_equal(common_probe, np.asarray(state["xin"])[12288:17203, 0])


def main() -> None:
    tests = [
        ("24576 partition test", test_current_partition_boundaries),
        ("odd length partition test", test_arbitrary_odd_length_partition),
        ("1D split reconstruction test", test_one_dimensional_split_reconstructs_original),
        ("2D ILC split test", test_two_dimensional_split_preserves_columns),
        ("ownership integrity test", test_partition_has_no_overlap_or_omission),
        ("ILC peak normalization test", test_ilc_peak_normalization_is_column_local),
        ("zero peak guard test", test_zero_peak_ilc_column_raises),
        ("full pair length guard test", test_full_pair_length_mismatch_raises),
        ("ILC column guard test", test_ilc_column_mismatch_raises),
        ("segment adjust independence test", test_segment_adjust_is_independent),
        ("raw pair field contract test", test_raw_pair_field_contracts),
        ("real state0 integration test", test_real_state0_off_ilc_stale_integration),
    ]
    for label, test in tests:
        test()
        print(f"{label}: PASS")


if __name__ == "__main__":
    main()
