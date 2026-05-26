"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

import random
from PIL import Image

# 工具函数：把普通张量转成 TVTensor（框、掩码）
from .._misc import convert_to_tv_tensor
# 注册器：把 Mosaic 注册到框架，可在 yaml 里直接使用
from ...core import register


# 注册 Mosaic 增强到全局配置
@register()
class Mosaic(T.Transform):
    """
    Mosaic 数据增强
    功能：把 4 张图片拼成 1 张，丰富背景、增加小目标，提升检测效果
    """
    def __init__(self, size, max_size=None) -> None:
        super().__init__()

        # 缩放：把图片统一缩放到指定尺寸
        self.resize = T.Resize(size=size, max_size=max_size)

        # 随机裁剪：把拼接后的大图裁剪回目标尺寸
        self.crop = T.RandomCrop(size=max_size if max_size else size)

        # 随机仿射变换（旋转、平移、缩放），填充值 114
        self.random_affine = T.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.5, 1.5),
            fill=114
        )

    def forward(self, *inputs):
        """
        Mosaic 核心逻辑
        """
        # 解析输入：图像、标注、数据集对象
        inputs = inputs if len(inputs) > 1 else inputs[0]
        image, target, dataset = inputs

        # 用于存放另外 3 张图片和标注
        images = []
        targets = []

        # 随机从数据集中再选 3 张图片
        indices = random.choices(range(len(dataset)), k=3)

        # 遍历 3 张图片，加载并缩放
        for i in indices:
            image, target = dataset.load_item(i)
            image, target = self.resize(image, target)
            images.append(image)
            targets.append(target)

        # 获取缩放后的单张图片尺寸
        h, w = F.get_spatial_size(images[0])

        # 4 张图在大图中的左上角坐标
        # 第 1 张：左上 (0,0)
        # 第 2 张：右上 (w,0)
        # 第 3 张：左下 (0,h)
        # 第 4 张：右下 (w,h)
        offset = [[0, 0], [w, 0], [0, h], [w, h]]

        # 创建 2w × 2h 的黑色大图
        image = Image.new(mode=images[0].mode, size=(w * 2, h * 2), color=0)

        # 把 4 张图贴到大图上
        for i, im in enumerate(images):
            image.paste(im, offset[i])

        # 偏移量扩展为 [x1,y1,x2,y2] 格式，用于框坐标偏移
        offset = torch.tensor([[0, 0], [w, 0], [0, h], [w, h]]).repeat(1, 2)

        # 合并 4 张图的标注
        target = {}
        for k in targets[0]:
            # 如果是框，需要加上对应图片的偏移量
            if k == 'boxes':
                v = [t[k] + offset[i] for i, t in enumerate(targets)]
            else:
                # 其他标注（标签、面积等）直接拼接
                v = [t[k] for t in targets]

            # 如果是张量，在第 0 维拼接
            if isinstance(v[0], torch.Tensor):
                v = torch.cat(v, dim=0)

            target[k] = v

        # 把框转为 TVTensor 类型
        if 'boxes' in target:
            w, h = image.size
            target['boxes'] = convert_to_tv_tensor(
                target['boxes'],
                'boxes',
                box_format='xyxy',
                spatial_size=[h, w]
            )

        # 把掩码转为 TVTensor 类型
        if 'masks' in target:
            target['masks'] = convert_to_tv_tensor(target['masks'], 'masks')

        # 随机仿射增强
        image, target = self.random_affine(image, target)

        # 随机裁剪回目标尺寸
        image, target = self.crop(image, target)

        # 返回最终图像、标注、数据集
        return image, target, dataset