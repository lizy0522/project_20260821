"""
功能说明：统一导出PA状态表、文件索引、MAT加载和Dataset公开接口。
输入：具体输入由各公开函数定义。
输出：模块级路径、映射、函数和 PAStateDataset 类。
用途：允许后续任务通过 data_manager 单一入口访问当前实验数据。
"""

from .config import (
    DATA_ROOT,
    EXPERIMENT_NAME,
    EXPERIMENT_ROOT,
    NOMINAL_DIR,
    PROJECT_ROOT,
    SCENARIO_ROOT,
    STATE_DATA_DIR,
)
from .dataset import PAStateDataset
from .file_index import (
    FILE_STATE_MAP,
    PARAM_FILE_MAP,
    STATE_FILE_MAP,
    DataIndexError,
    get_file_by_condition,
    get_file_by_id,
    get_state_id_by_file,
    parse_state_filename,
)
from .loader import load_by_condition, load_by_id, load_variable_by_id
from .state_index import build_state_table, get_state_id, get_state_info

__all__ = [
    "DATA_ROOT",
    "EXPERIMENT_NAME",
    "EXPERIMENT_ROOT",
    "FILE_STATE_MAP",
    "NOMINAL_DIR",
    "PARAM_FILE_MAP",
    "PROJECT_ROOT",
    "PAStateDataset",
    "SCENARIO_ROOT",
    "STATE_DATA_DIR",
    "STATE_FILE_MAP",
    "DataIndexError",
    "build_state_table",
    "get_file_by_condition",
    "get_file_by_id",
    "get_state_id",
    "get_state_id_by_file",
    "get_state_info",
    "load_by_condition",
    "load_by_id",
    "load_variable_by_id",
    "parse_state_filename",
]
