"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

# 导入训练器基类
from ._solver import BaseSolver
# 导入分类任务专用训练器
from .clas_solver import ClasSolver
# 导入检测任务专用训练器
from .det_solver import DetSolver


# 导入类型提示，用于声明字典类型
from typing import Dict

# 任务类型映射字典
# 作用：根据任务名称（字符串）自动找到对应的训练器类
TASKS :Dict[str, BaseSolver] = {
    # 分类任务 → 使用 ClasSolver
    'classification': ClasSolver,
    # 检测任务 → 使用 DetSolver
    'detection': DetSolver,
}