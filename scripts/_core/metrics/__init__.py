"""
功能说明：导出共享 NMSE、CNMSE 和 ACPR 指标接口。
输入：一维数值信号及指标所需参数。
输出：以 dB 表示的误差或邻道功率比。
数学定义：具体定义见 nmse、cnmse 和 acpr 模块。
"""

from .acpr import acpr
from .cnmse import cnmse
from .nmse import nmse

__all__ = ["acpr", "cnmse", "nmse"]
