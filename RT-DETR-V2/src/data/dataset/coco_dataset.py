"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
Mostly copy-paste from https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
from faster_coco_eval.utils.pytorch import FasterCocoDetection
import torchvision

from PIL import Image 
from faster_coco_eval.core import mask as coco_mask

# 继承自定义的检测数据集基类 DetDataset
from ._dataset import DetDataset
# 工具函数：转换为 torchvision 规范张量
from .._misc import convert_to_tv_tensor
# 注册器：将数据集类注册到框架
from ...core import register

# 暴露给外部调用的类名
__all__ = ['CocoDetection']

# 关闭 torchvision 关于 beta 版 transforms 的警告
torchvision.disable_beta_transforms_warning()

# 将 CocoDetection 注册到框架的数据集列表中
@register()
class CocoDetection(FasterCocoDetection, DetDataset):
    """
    自定义 COCO 检测数据集类
    继承关系：
    1. FasterCocoDetection: 快速 COCO 数据读取基类
    2. DetDataset: 框架自定义的检测数据集基类
    功能：读取 COCO 数据集图片和标注，并完成格式转换
    """
    # 声明需要框架自动注入的参数（数据增强流水线）
    __inject__ = ['transforms', ]
    # 声明需要共享的配置参数（类别映射开关）
    __share__ = ['remap_mscoco_category']

    def __init__(self, img_folder, ann_file, transforms, return_masks=False, remap_mscoco_category=False):
        """
        初始化 COCO 数据集
        Args:
            img_folder: 图片文件夹路径
            ann_file: 标注 json 文件路径
            transforms: 数据预处理/增强函数
            return_masks: 是否返回分割掩码
            remap_mscoco_category: 是否将 COCO 类别 ID 重映射为连续标签
        """
        # 调用父类初始化（图片路径 + 标注文件）
        super(FasterCocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        # 初始化 COCO 多边形标注转掩码工具
        self.prepare = ConvertCocoPolysToMask(return_masks)
        self.img_folder = img_folder
        self.ann_file = ann_file
        self.return_masks = return_masks
        self.remap_mscoco_category = remap_mscoco_category

    def __getitem__(self, idx):
        """
        PyTorch Dataset 标准接口：根据索引获取数据
        """
        # 1. 加载原始图片和标注
        img, target = self.load_item(idx)
        # 2. 执行数据增强
        if self._transforms is not None:
            img, target, _ = self._transforms(img, target, self)
        return img, target

    def load_item(self, idx):
        """
        实现 DetDataset 要求的抽象方法：加载并预处理原始数据
        """
        # 调用父类读取图片和原始标注
        image, target = super(FasterCocoDetection, self).__getitem__(idx)
        # 获取图片 ID
        image_id = self.ids[idx]
        target = {'image_id': image_id, 'annotations': target}

        # 如果需要重映射类别（把不连续的 COCO ID 转为 0~79）
        if self.remap_mscoco_category:
            # 使用预设的 MSCOCO 映射表转换标签
            image, target = self.prepare(image, target, category2label=mscoco_category2label)
        else:
            # 直接使用原始类别 ID
            image, target = self.prepare(image, target)

        # 把样本索引存入 target
        target['idx'] = torch.tensor([idx])

        # 将 boxes 转换为 torchvision 规范的 Boxes 类型
        if 'boxes' in target:
            target['boxes'] = convert_to_tv_tensor(target['boxes'], key='boxes', spatial_size=image.size[::-1])

        # 将 masks 转换为 torchvision 规范的 Mask 类型
        if 'masks' in target:
            target['masks'] = convert_to_tv_tensor(target['masks'], key='masks')

        return image, target

    def extra_repr(self) -> str:
        """
        打印数据集信息时的额外描述
        """
        s = f' img_folder: {self.img_folder}\n ann_file: {self.ann_file}\n'
        s += f' return_masks: {self.return_masks}\n'
        if hasattr(self, '_transforms') and self._transforms is not None:
            s += f' transforms:\n   {repr(self._transforms)}'
        return s

    @property
    def categories(self, ):
        """获取 COCO 类别列表"""
        return self.coco.dataset['categories']

    @property
    def category2name(self, ):
        """类别 ID 映射到类别名称"""
        return {cat['id']: cat['name'] for cat in self.categories}

    @property
    def category2label(self, ):
        """类别 ID 映射到连续训练标签（0,1,2...）"""
        return {cat['id']: i for i, cat in enumerate(self.categories)}

    @property
    def label2category(self, ):
        """训练标签映射回原始类别 ID"""
        return {i: cat['id'] for i, cat in enumerate(self.categories)}


def convert_coco_poly_to_mask(segmentations, height, width):
    """
    将 COCO 的多边形标注（polygon）转换为二值掩码图
    Args:
        segmentations: 多边形标注列表
        height: 图片高度
        width: 图片宽度
    Returns:
        掩码张量 [N, H, W]
    """
    masks = []
    for polygons in segmentations:
        # 将多边形编码为 COCO 格式的 RLE
        rles = coco_mask.frPyObjects(polygons, height, width)
        # 解码为掩码
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        # 合并通道，得到单通道掩码
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    """
    COCO 标注格式转换器
    功能：把 JSON 里的原始标注 → 模型训练用的标准格式
    """
    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image: Image.Image, target, **kwargs):
        """
        处理单张图片的标注
        """
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        # 过滤掉 crowd 标注（不参与正常检测训练）
        anno = [obj for obj in anno if 'iscrowd' not in obj or obj['iscrowd'] == 0]

        # 提取 bounding boxes
        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # COCO bbox 格式: x, y, w, h → 转为 xmin, ymin, xmax, ymax
        boxes[:, 2:] += boxes[:, :2]
        # 防止越界
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # 处理类别标签
        category2label = kwargs.get('category2label', None)
        if category2label is not None:
            labels = [category2label[obj["category_id"]] for obj in anno]
        else:
            labels = [obj["category_id"] for obj in anno]

        labels = torch.tensor(labels, dtype=torch.int64)

        # 处理分割掩码
        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        # 处理关键点（如果有）
        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        # 过滤掉无效框（宽/高 ≤ 0）
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        labels = labels[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        # 组装成模型需要的 target 字典
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # 保留用于 COCO 评估的字段
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno])
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        # 原始图像尺寸
        target["orig_size"] = torch.as_tensor([int(w), int(h)])

        return image, target


# MSCOCO 80 类：类别 ID → 类别名称 映射表
mscoco_category2name = {
    1: 'person',
    2: 'bicycle',
    3: 'car',
    4: 'motorcycle',
    5: 'airplane',
    6: 'bus',
    7: 'train',
    8: 'truck',
    9: 'boat',
    10: 'traffic light',
    11: 'fire hydrant',
    13: 'stop sign',
    14: 'parking meter',
    15: 'bench',
    16: 'bird',
    17: 'cat',
    18: 'dog',
    19: 'horse',
    20: 'sheep',
    21: 'cow',
    22: 'elephant',
    23: 'bear',
    24: 'zebra',
    25: 'giraffe',
    27: 'backpack',
    28: 'umbrella',
    31: 'handbag',
    32: 'tie',
    33: 'suitcase',
    34: 'frisbee',
    35: 'skis',
    36: 'snowboard',
    37: 'sports ball',
    38: 'kite',
    39: 'baseball bat',
    40: 'baseball glove',
    41: 'skateboard',
    42: 'surfboard',
    43: 'tennis racket',
    44: 'bottle',
    46: 'wine glass',
    47: 'cup',
    48: 'fork',
    49: 'knife',
    50: 'spoon',
    51: 'bowl',
    52: 'banana',
    53: 'apple',
    54: 'sandwich',
    55: 'orange',
    56: 'broccoli',
    57: 'carrot',
    58: 'hot dog',
    59: 'pizza',
    60: 'donut',
    61: 'cake',
    62: 'chair',
    63: 'couch',
    64: 'potted plant',
    65: 'bed',
    67: 'dining table',
    70: 'toilet',
    72: 'tv',
    73: 'laptop',
    74: 'mouse',
    75: 'remote',
    76: 'keyboard',
    77: 'cell phone',
    78: 'microwave',
    79: 'oven',
    80: 'toaster',
    81: 'sink',
    82: 'refrigerator',
    84: 'book',
    85: 'clock',
    86: 'vase',
    87: 'scissors',
    88: 'teddy bear',
    89: 'hair drier',
    90: 'toothbrush'
}

# 类别 ID → 连续训练标签（0~79）
mscoco_category2label = {k: i for i, k in enumerate(mscoco_category2name.keys())}
# 训练标签 → 类别 ID
mscoco_label2category = {v: k for k, v in mscoco_category2label.items()}