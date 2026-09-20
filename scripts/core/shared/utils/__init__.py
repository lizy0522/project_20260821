"""
功能说明：导出共享输入验证工具。
输入：NumPy 数组、参数名称及验证选项。
输出：通过验证的数组或数组对。
数学定义：本初始化文件不执行数学计算。
"""

from .validation import as_numeric_vector, validate_pair

__all__ = ["as_numeric_vector", "validate_pair"]
