"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

from sympy import im
import torch
import torchvision
import torchvision.transforms.functional as TVF 

import os
from PIL import Image
from typing import Optional, Callable

# 安全解析 XML（防止 XML 注入攻击）
try:
    from defusedxml.ElementTree import parse as ET_parse
except ImportError:
    from xml.etree.ElementTree import ET_parse

# 导入检测数据集基类（提供统一接口）
from ._dataset import DetDataset
# 导入工具：把普通张量转为 torchvision TVTensor（框/掩码）
from .._misc import convert_to_tv_tensor
# 导入注册器，让数据集可以被框架自动创建
from ...core import register


# 注册数据集到框架
@register()
# 继承 torchvision 官方 VOCDetection + 框架基类 DetDataset
class VOCDetection(torchvision.datasets.VOCDetection, DetDataset):
    # 声明需要自动注入的参数（数据增强）
    __inject__ = ['transforms', ]

    def __init__(self, root: str, ann_file: str = "trainval.txt", label_file: str = "label_list.txt", transforms: Optional[Callable] = None):
        """
        初始化 VOC 检测数据集
        Args:
            root: 数据集根目录
            ann_file: 存储图片路径与标注路径的文件（如 trainval.txt）
            label_file: 类别名称列表文件
            transforms: 数据增强/预处理函数
        """
        # 读取图片与标注对应关系文件
        with open(os.path.join(root, ann_file), 'r') as f:
            lines = [x.strip() for x in f.readlines()]
            lines = [x.split(' ') for x in lines]

        # 构建图片路径列表
        self.images = [os.path.join(root, lin[0]) for lin in lines]
        # 构建标注 XML 路径列表
        self.targets = [os.path.join(root, lin[1]) for lin in lines]
        # 确保图片数量 = 标注数量
        assert len(self.images) == len(self.targets)

        # 读取类别标签文件
        with open(os.path.join(root + label_file), 'r') as f:
            labels = f.readlines()
            labels = [lab.strip() for lab in labels]

        # 保存数据增强
        self.transforms = transforms
        # 构建 类别名称 → 数字标签 的映射
        self.labels_map = {lab: i for i, lab in enumerate(labels)}

    def __getitem__(self, index: int):
        """
        PyTorch Dataset 标准接口：根据索引获取数据
        """
        # 1. 加载原始图片 + 标注
        image, target = self.load_item(index)

        # 2. 执行数据增强
        if self.transforms is not None:
            image, target, _ = self.transforms(image, target, self)

        return image, target

    def load_item(self, index: int):
        """
        实现 DetDataset 规定的方法：加载并解析原始数据
        """
        # 打开图片并转为 RGB
        image = Image.open(self.images[index]).convert("RGB")

        # 解析 XML 标注文件
        target = self.parse_voc_xml(ET_parse(self.targets[index]).getroot())

        # 初始化输出字典
        output = {}
        output["image_id"] = torch.tensor([index])
        # 初始化框、标签、面积、iscrowd 列表
        for k in ['area', 'boxes', 'labels', 'iscrowd']:
            output[k] = []

        # 遍历每个目标（物体）
        for blob in target['annotation']['object']:
            # 读取 XML 中的 xmin, ymin, xmax, ymax
            box = [float(v) for v in blob['bndbox'].values()]
            output["boxes"].append(box)
            # 读取类别名称
            output["labels"].append(blob['name'])
            # 计算框面积
            output["area"].append((box[2] - box[0]) * (box[3] - box[1]))
            # VOC 默认 iscrowd = 0
            output["iscrowd"].append(0)

        # 图片宽高
        w, h = image.size

        # 处理框：空数据时返回 0 行 4 列张量
        boxes = torch.tensor(output["boxes"]) if len(output["boxes"]) > 0 else torch.zeros(0, 4)
        # 转为 torchvision TVTensor（方便数据增强）
        output['boxes'] = convert_to_tv_tensor(boxes, 'boxes', box_format='xyxy', spatial_size=[h, w])

        # 类别名称 → 数字标签
        output['labels'] = torch.tensor([self.labels_map[lab] for lab in output["labels"]])
        # 面积张量
        output['area'] = torch.tensor(output['area'])
        # iscrowd 张量
        output["iscrowd"] = torch.tensor(output["iscrowd"])
        # 原始图像尺寸（用于评估）
        output["orig_size"] = torch.tensor([w, h])

        return image, output