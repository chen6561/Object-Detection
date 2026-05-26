"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
COCO evaluator that works in distributed mode.
大多数代码复制自：https://github.com/pytorch/vision/blob/edfd5a7/references/detection/coco_eval.py
区别在于：文件末尾减少了从 pycocotools 中复制的代码，
因为 python3 可以使用 contextlib 抑制打印信息

# MiXaiLL76 用 faster-coco-eval 替换了 pycocotools，以获得更好性能与兼容性
"""

# 导入框架的注册器，用于把评估器注册到全局系统
from ...core import register

# 导入更快的 COCO 评估器基类（替代原版 pycocotools，速度更快）
from faster_coco_eval.utils.pytorch import FasterCocoEvaluator

# 使用 @register() 装饰器，将 CocoEvaluator 注册到框架
# 注册后可以通过 create() 函数自动创建评估器实例
@register()
# 定义 CocoEvaluator 类，完全继承 FasterCocoEvaluator 所有功能
# 这里没有新增任何逻辑，只是为了适配框架的注册系统
class CocoEvaluator(FasterCocoEvaluator):
    # 空实现，完全复用父类 FasterCocoEvaluator 的功能
    pass