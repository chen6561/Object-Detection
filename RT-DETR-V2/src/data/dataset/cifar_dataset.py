"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

# 导入 torchvision 内置的 CIFAR10 数据集实现
import torchvision
# 导入类型提示工具：可选类型、可调用对象类型
from typing import Optional, Callable

# 从框架核心模块导入注册器
# 作用：把这个数据集类注册到全局配置，让 create() 函数可以自动创建
from ...core import register


# 使用 @register() 装饰器
# 将 CIFAR10 类注册到框架的全局数据集列表中
@register()
class CIFAR10(torchvision.datasets.CIFAR10):
    """
    自定义 CIFAR10 数据集类
    功能：继承 torchvision 官方 CIFAR10，同时适配框架的配置化创建系统
    属于框架的数据集封装层
    """

    # 声明需要**自动注入**的参数（框架核心功能）
    # transform、target_transform 会由配置系统自动创建并传入
    __inject__ = ['transform', 'target_transform']

    # 构造函数：初始化数据集
    # 参数和官方 torchvision.datasets.CIFAR10 保持一致
    def __init__(
        self,
        root: str,                # 数据集存放根路径
        train: bool = True,       # True=训练集，False=测试集
        transform: Optional[Callable] = None,    # 图像预处理/增强函数
        target_transform: Optional[Callable] = None,  # 标签预处理函数
        download: bool = False    # 是否自动下载数据集
    ) -> None:
        # 调用父类（官方 CIFAR10）的构造函数
        super().__init__(root, train, transform, target_transform, download)