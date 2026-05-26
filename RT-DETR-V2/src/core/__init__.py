"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

# 从 workspace 模块导入 核心注册器、全局配置、对象创建函数
# GLOBAL_CONFIG：全局注册字典，存放所有注册的类/函数
# register：装饰器，用于注册模型、优化器、数据集等组件
# create：根据配置自动创建对象实例（框架核心函数）
from .workspace import GLOBAL_CONFIG, register, create

# 从 yaml_utils 模块导入所有配置相关工具函数
# 包含：load_config 加载配置、merge_config 合并配置、parse_cli 解析命令行参数
from .yaml_utils import *

# 从 _config 模块导入 基础配置基类
# BaseConfig：定义训练所需的所有基础属性（模型、优化器、数据集、训练参数等）
from ._config import BaseConfig

# 从 yaml_config 模块导入 YAML 配置类
# YAMLConfig：继承 BaseConfig，自动从 YAML 文件构建所有训练组件
from .yaml_config import YAMLConfig