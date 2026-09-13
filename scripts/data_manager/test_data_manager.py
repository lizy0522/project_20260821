"""
功能说明：直接执行或由pytest收集的数据管理模块回归测试。
输入：当前 experiment_2026_0816 场景2的425个只读状态MAT文件。
输出：状态表、文件索引、两种读取方式和Dataset测试的PASS信息。
用途：验证state_id、六参数、文件路径和MAT变量之间的一致性。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPTS_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SCRIPTS_ROOT))

from data_manager import (  # noqa: E402
    FILE_STATE_MAP,
    PARAM_FILE_MAP,
    STATE_FILE_MAP,
    PAStateDataset,
    build_state_table,
    get_state_id,
    get_state_id_by_file,
    get_state_info,
    load_by_condition,
    load_by_id,
    load_variable_by_id,
    parse_state_filename,
)


def test_state_table() -> None:
    states = build_state_table()
    assert len(states) == 425
    assert states[0] == {
        "state_id": 0,
        "funMng": 0,
        "funAng": 0,
        "secMng": 0,
        "secAng": 0,
        "Vm": 2.30,
        "Pin": -21,
    }
    assert states[-1] == {
        "state_id": 424,
        "funMng": 30,
        "funAng": 315,
        "secMng": 30,
        "secAng": 315,
        "Vm": 2.30,
        "Pin": -21,
    }
    for state_id in (0, 100, 424):
        info = get_state_info(state_id)
        assert get_state_id(**info) == state_id


def test_file_index() -> None:
    assert len(STATE_FILE_MAP) == 425
    assert len(PARAM_FILE_MAP) == 425
    assert len(FILE_STATE_MAP) == 425
    assert len(set(STATE_FILE_MAP.values())) == 425
    for state_id in (0, 100, 424):
        path = STATE_FILE_MAP[state_id]
        condition = parse_state_filename(path)
        assert get_state_id(**condition) == state_id
        assert get_state_id_by_file(path) == state_id


def test_load_by_id() -> None:
    data = load_by_id(0)
    required = {"xin", "yout_withoutdpd", "xin_pd", "yout_withdpd"}
    assert required.issubset(data)
    assert data["xin"].shape == (24576, 1)
    assert np.iscomplexobj(data["xin"])
    variable = load_variable_by_id(0, "yout_withoutdpd")
    np.testing.assert_array_equal(variable, data["yout_withoutdpd"])


def test_load_by_condition() -> None:
    data_by_id = load_by_id(0)
    data_by_condition = load_by_condition(
        funMng=0,
        funAng=0,
        secMng=0,
        secAng=0,
        Vm=2.30,
        Pin=-21,
    )
    np.testing.assert_array_equal(data_by_id["xin"], data_by_condition["xin"])
    np.testing.assert_array_equal(
        data_by_id["yout_withoutdpd"],
        data_by_condition["yout_withoutdpd"],
    )


def test_dataset() -> None:
    dataset = PAStateDataset()
    assert len(dataset) == 425
    state_zero = dataset[0]
    iterated_zero = next(iter(dataset))
    np.testing.assert_array_equal(state_zero["xin"], iterated_zero["xin"])
    state_last = dataset[424]
    assert state_last["xin"].shape == (24576, 1)


def main() -> None:
    tests = [
        ("state table test", test_state_table),
        ("file index test", test_file_index),
        ("load by id test", test_load_by_id),
        ("load by condition test", test_load_by_condition),
        ("dataset test", test_dataset),
    ]
    for label, test in tests:
        test()
        print(f"{label}: PASS")


if __name__ == "__main__":
    main()
