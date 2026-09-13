"""
功能说明：统一定义当前 PA 实验的数据根路径、实验目录和场景目录。
输入：无。
输出：供 data_manager 其他模块复用的 pathlib.Path 常量。
用途：避免在文件索引和加载代码中重复硬编码绝对路径。
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data" / "raw"
EXPERIMENT_NAME = "experiment_2026_0816"
EXPERIMENT_ROOT = DATA_ROOT / EXPERIMENT_NAME
SCENARIO_ROOT = EXPERIMENT_ROOT / "data_record" / "scenario_2"
STATE_DATA_DIR = SCENARIO_ROOT / "withoutdpd_withdpd_own_stale_fixPa_diffState"
NOMINAL_DIR = SCENARIO_ROOT / "nominal_reference"

__all__ = [
    "DATA_ROOT",
    "EXPERIMENT_NAME",
    "EXPERIMENT_ROOT",
    "NOMINAL_DIR",
    "PROJECT_ROOT",
    "SCENARIO_ROOT",
    "STATE_DATA_DIR",
]
