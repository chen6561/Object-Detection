"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

# 【注：原代码注释掉的基类导入】
# from ._dataset import DetDataset

# 导入 CIFAR10 数据集类（分类任务，已注册到框架）
from .cifar_dataset import CIFAR10

# 导入 COCO 检测数据集类（已注册到框架）
from .coco_dataset import CocoDetection

# 从 COCO 数据集模块导入：
# CocoDetection: COCO 数据集类
# mscoco_category2name: 类别 ID → 类别名称
# mscoco_category2label: 类别 ID → 训练用连续标签 (0~79)
# mscoco_label2category: 训练标签 → 原始类别 ID
from .coco_dataset import (
    CocoDetection,
    mscoco_category2name,
    mscoco_category2label,
    mscoco_label2category,
)

# 导入 COCO 评估器（用于计算 AP/mAP，已注册）
from .coco_eval import CocoEvaluator

# 导入 COCO 工具函数：从数据集获取标准 COCO API 结构（用于评估）
from .coco_utils import get_coco_api_from_dataset

# 导入 VOC 数据集类（已注册到框架）
from .voc_detection import VOCDetection

# 导入 VOC 评估器（用于 VOC 数据集精度评估）
from .voc_eval import VOCEvaluator