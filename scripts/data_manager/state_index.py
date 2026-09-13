"""
功能说明：建立 state_id 与六个 PA 状态参数之间的稳定双向映射。
输入：state_id，或 funMng、funAng、secMng、secAng、Vm、Pin 六个状态参数。
输出：425行状态表、单个状态信息或对应 state_id。
用途：为文件索引、数据加载和后续算法提供唯一且可复现的 d0 状态顺序。
"""

from __future__ import annotations

from numbers import Integral, Real

FUN_MNG_LIST = (0, 10, 20, 30)
FUN_ANG_LIST = (0, 45, 90, 135, 180, 225, 270, 315)
SEC_MNG_LIST = (0, 15, 30)
SEC_ANG_LIST = (0, 45, 90, 135, 180, 225, 270, 315)
VM_LIST = (2.30,)
PIN_LIST = (-21,)

STATE_PARAMETER_NAMES = ("funMng", "funAng", "secMng", "secAng", "Vm", "Pin")


def _valid_fundamental_conditions() -> tuple[tuple[int, int], ...]:
    conditions = [(0, 0)]
    conditions.extend(
        (funMng, funAng) for funMng in FUN_MNG_LIST if funMng != 0 for funAng in FUN_ANG_LIST
    )
    return tuple(conditions)


def _valid_second_harmonic_conditions() -> tuple[tuple[int, int], ...]:
    conditions = [(0, 0)]
    conditions.extend(
        (secMng, secAng) for secMng in SEC_MNG_LIST if secMng != 0 for secAng in SEC_ANG_LIST
    )
    return tuple(conditions)


def _make_state_table() -> tuple[dict[str, int | float], ...]:
    rows: list[dict[str, int | float]] = []
    for funMng, funAng in _valid_fundamental_conditions():
        for secMng, secAng in _valid_second_harmonic_conditions():
            for Vm in VM_LIST:
                for Pin in PIN_LIST:
                    rows.append(
                        {
                            "state_id": len(rows),
                            "funMng": funMng,
                            "funAng": funAng,
                            "secMng": secMng,
                            "secAng": secAng,
                            "Vm": Vm,
                            "Pin": Pin,
                        }
                    )
    if len(rows) != 425:
        raise RuntimeError(f"内部状态空间应为425，实际为{len(rows)}")
    return tuple(rows)


_STATE_TABLE = _make_state_table()
_PARAM_TO_ID = {
    tuple(row[name] for name in STATE_PARAMETER_NAMES): int(row["state_id"]) for row in _STATE_TABLE
}


def build_state_table() -> list[dict[str, int | float]]:
    """返回按 funMng→funAng→secMng→secAng→Vm→Pin 排序的425行状态表副本。"""
    return [dict(row) for row in _STATE_TABLE]


def get_state_info(state_id: int) -> dict[str, int | float]:
    """根据 state_id 返回可直接传给 get_state_id 的六参数字典。"""
    if not isinstance(state_id, Integral) or isinstance(state_id, bool):
        raise TypeError("state_id 必须是整数")
    normalized_id = int(state_id)
    if not 0 <= normalized_id < len(_STATE_TABLE):
        raise IndexError(f"state_id 必须位于[0, {len(_STATE_TABLE) - 1}]")
    row = _STATE_TABLE[normalized_id]
    return {name: row[name] for name in STATE_PARAMETER_NAMES}


def _normalize_integer_parameter(value: Real, name: str) -> int:
    if not isinstance(value, Real) or isinstance(value, bool):
        raise TypeError(f"{name} 必须是实数")
    normalized = int(value)
    if value != normalized:
        raise ValueError(f"{name} 必须是整数编码，实际为{value}")
    return normalized


def get_state_id(
    funMng: Real,
    funAng: Real,
    secMng: Real,
    secAng: Real,
    Vm: Real,
    Pin: Real,
) -> int:
    """根据六个状态参数返回唯一 state_id；无效幅相组合会显式报错。"""
    if not isinstance(Vm, Real) or isinstance(Vm, bool):
        raise TypeError("Vm 必须是实数")
    if not isinstance(Pin, Real) or isinstance(Pin, bool):
        raise TypeError("Pin 必须是实数")
    condition = (
        _normalize_integer_parameter(funMng, "funMng"),
        _normalize_integer_parameter(funAng, "funAng"),
        _normalize_integer_parameter(secMng, "secMng"),
        _normalize_integer_parameter(secAng, "secAng"),
        float(Vm),
        float(Pin),
    )
    try:
        return _PARAM_TO_ID[condition]
    except KeyError as exc:
        raise KeyError(f"不存在状态条件{condition}") from exc


__all__ = [
    "FUN_ANG_LIST",
    "FUN_MNG_LIST",
    "PIN_LIST",
    "SEC_ANG_LIST",
    "SEC_MNG_LIST",
    "STATE_PARAMETER_NAMES",
    "VM_LIST",
    "build_state_table",
    "get_state_id",
    "get_state_info",
]
