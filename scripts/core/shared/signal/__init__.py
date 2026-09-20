"""
功能说明：导出共享时间对齐、复增益校准和预处理接口。
输入：一维复数参考信号和待处理信号。
输出：对齐或校准后的信号以及相应估计参数。
数学定义：具体定义见 alignment、calibration 和 preprocess 模块。
"""

from .alignment import fine_align, rough_align
from .calibration import adjust_complex_gain
from .preprocess import preprocess_pair

__all__ = ["adjust_complex_gain", "fine_align", "preprocess_pair", "rough_align"]
