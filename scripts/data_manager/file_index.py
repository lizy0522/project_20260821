"""
功能说明：扫描425个状态MAT文件，建立 state_id、六参数和文件路径的双向索引。
输入：config.STATE_DATA_DIR 中的实际MAT文件及其文件名。
输出：STATE_FILE_MAP、PARAM_FILE_MAP、FILE_STATE_MAP及路径查询函数。
用途：在读取数据前验证文件完整性、唯一性和状态参数覆盖。
"""

from __future__ import annotations

import os
import re
from numbers import Real
from pathlib import Path

from .config import STATE_DATA_DIR
from .state_index import STATE_PARAMETER_NAMES, build_state_table, get_state_id, get_state_info

MAT_FILENAME_PATTERN = re.compile(
    r"^funMng_(?P<funMng>\d{3})_funAng_(?P<funAng>\d{3})_"
    r"secMng_(?P<secMng>\d{3})_secAng_(?P<secAng>\d{3})_"
    r"vCarrier_(?P<Vm>-?\d+(?:\.\d+)?)_"
    r"inputPower_(?P<Pin>-?\d+(?:\.\d+)?)\.mat$"
)


class DataIndexError(RuntimeError):
    """表示状态文件缺失、重复、命名无效或覆盖不完整。"""


def _as_io_path(path: Path) -> str:
    resolved = str(path.resolve())
    if os.name == "nt" and not resolved.startswith("\\\\?\\"):
        return f"\\\\?\\{resolved}"
    return resolved


def _strip_extended_prefix(path: str) -> str:
    if path.startswith("\\\\?\\"):
        return path[4:]
    return path


def _scan_mat_files() -> list[Path]:
    root = _as_io_path(STATE_DATA_DIR)
    if not os.path.isdir(root):
        raise DataIndexError(f"状态数据目录不存在：{STATE_DATA_DIR}")
    files: list[Path] = []
    for current, _, names in os.walk(root):
        for name in names:
            if name.lower().endswith(".mat"):
                full_path = _strip_extended_prefix(os.path.join(current, name))
                files.append(Path(full_path))
    return sorted(files, key=lambda path: str(path).lower())


def parse_state_filename(path: str | Path) -> dict[str, int | float]:
    """从完整实验文件名解析六个论文状态参数。"""
    filename = Path(path).name
    match = MAT_FILENAME_PATTERN.fullmatch(filename)
    if match is None:
        raise DataIndexError(f"无法解析状态MAT文件名：{filename}")
    values = match.groupdict()
    return {
        "funMng": int(values["funMng"]),
        "funAng": int(values["funAng"]),
        "secMng": int(values["secMng"]),
        "secAng": int(values["secAng"]),
        "Vm": float(values["Vm"]),
        "Pin": float(values["Pin"]),
    }


def _build_file_maps() -> tuple[dict[int, Path], dict[tuple[int | float, ...], Path]]:
    state_file_map: dict[int, Path] = {}
    param_file_map: dict[tuple[int | float, ...], Path] = {}
    files = _scan_mat_files()
    for path in files:
        condition = parse_state_filename(path)
        try:
            state_id = get_state_id(**condition)
        except (KeyError, TypeError, ValueError) as exc:
            raise DataIndexError(f"文件包含无效状态条件：{path}") from exc
        key = tuple(condition[name] for name in STATE_PARAMETER_NAMES)
        if state_id in state_file_map:
            raise DataIndexError(
                f"state_id={state_id}对应多个文件：{state_file_map[state_id]} 和 {path}"
            )
        if key in param_file_map:
            raise DataIndexError(f"状态参数{key}对应多个文件")
        state_file_map[state_id] = path
        param_file_map[key] = path

    expected_ids = {int(row["state_id"]) for row in build_state_table()}
    actual_ids = set(state_file_map)
    missing_ids = sorted(expected_ids - actual_ids)
    unexpected_ids = sorted(actual_ids - expected_ids)
    if len(files) != 425 or missing_ids or unexpected_ids:
        raise DataIndexError(
            "状态文件覆盖不完整："
            f"files={len(files)}, missing_ids={missing_ids}, unexpected_ids={unexpected_ids}"
        )
    return dict(sorted(state_file_map.items())), param_file_map


STATE_FILE_MAP, PARAM_FILE_MAP = _build_file_maps()
FILE_STATE_MAP = {path: state_id for state_id, path in STATE_FILE_MAP.items()}


def get_file_by_id(state_id: int) -> Path:
    """返回 state_id 对应的MAT文件路径。"""
    get_state_info(state_id)
    return STATE_FILE_MAP[int(state_id)]


def get_file_by_condition(
    funMng: Real,
    funAng: Real,
    secMng: Real,
    secAng: Real,
    Vm: Real,
    Pin: Real,
) -> Path:
    """返回六参数状态对应的MAT文件路径。"""
    return get_file_by_id(get_state_id(funMng, funAng, secMng, secAng, Vm, Pin))


def get_state_id_by_file(path: str | Path) -> int:
    """根据已索引MAT文件路径返回 state_id。"""
    normalized = Path(os.path.abspath(path))
    try:
        return FILE_STATE_MAP[normalized]
    except KeyError as exc:
        raise KeyError(f"文件不在当前425状态索引中：{normalized}") from exc


__all__ = [
    "DataIndexError",
    "FILE_STATE_MAP",
    "MAT_FILENAME_PATTERN",
    "PARAM_FILE_MAP",
    "STATE_FILE_MAP",
    "get_file_by_condition",
    "get_file_by_id",
    "get_state_id_by_file",
    "parse_state_filename",
]
