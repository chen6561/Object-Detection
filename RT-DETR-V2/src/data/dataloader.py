"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.utils.data as data
import torch.nn.functional as F
from torch.utils.data import default_collate

import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as VT
from torchvision.transforms.v2 import functional as VF, InterpolationMode

import random
from functools import partial

# 注册器，用于将 DataLoader 和 Collate 类注册到框架
from ..core import register


# 暴露给外部调用的接口
__all__ = [
    'DataLoader',
    'BaseCollateFunction',
    'BatchImageCollateFunction',
    'batch_image_collate_fn'
]


# 注册自定义 DataLoader
@register()
class DataLoader(data.DataLoader):
    # 声明需要框架自动注入的参数：dataset 和 collate_fn
    __inject__ = ['dataset', 'collate_fn']

    def __repr__(self) -> str:
        """
        自定义打印格式，方便调试时查看 DataLoader 参数
        """
        format_string = self.__class__.__name__ + "("
        for n in ['dataset', 'batch_size', 'num_workers', 'drop_last', 'collate_fn']:
            format_string += "\n"
            format_string += "    {0}: {1}".format(n, getattr(self, n))
        format_string += "\n)"
        return format_string

    def set_epoch(self, epoch):
        """
        设置当前 epoch，并同步给 dataset 和 collate_fn
        用于控制动态数据增强策略
        """
        self._epoch = epoch
        self.dataset.set_epoch(epoch)
        self.collate_fn.set_epoch(epoch)

    @property
    def epoch(self):
        """
        获取当前 epoch
        """
        return self._epoch if hasattr(self, '_epoch') else -1

    @property
    def shuffle(self):
        """
        获取 shuffle 属性
        """
        return self._shuffle

    @shuffle.setter
    def shuffle(self, shuffle):
        """
        设置 shuffle 属性（允许运行时修改）
        """
        assert isinstance(shuffle, bool), 'shuffle must be a boolean'
        self._shuffle = shuffle


# 注册简单的批处理函数
@register()
def batch_image_collate_fn(items):
    """仅批处理图像
    将列表中的图像堆叠成 batch，标签保持列表形式
    """
    return torch.cat([x[0][None] for x in items], dim=0), [x[1] for x in items]


class BaseCollateFunction(object):
    """批处理基类，提供 epoch 管理功能
    """
    def set_epoch(self, epoch):
        self._epoch = epoch

    @property
    def epoch(self):
        return self._epoch if hasattr(self, '_epoch') else -1

    def __call__(self, items):
        raise NotImplementedError('子类必须实现 __call__ 方法')


# 注册带多尺度训练的批处理类
@register()
class BatchImageCollateFunction(BaseCollateFunction):
    def __init__(
        self,
        scales=None,    # 多尺度列表，例如 [640, 672, 704]
        stop_epoch=None, # 到第几轮停止多尺度训练
    ) -> None:
        super().__init__()
        self.scales = scales
        # 默认永不停止
        self.stop_epoch = stop_epoch if stop_epoch is not None else 100000000

    def __call__(self, items):
        """
        批处理核心逻辑
        """
        # 将图像堆叠成 batch: [batch, 3, H, W]
        images = torch.cat([x[0][None] for x in items], dim=0)
        # 目标保持列表形式
        targets = [x[1] for x in items]

        # 如果开启多尺度，且未到停止轮次，则随机 resize
        if self.scales is not None and self.epoch < self.stop_epoch:
            # 随机选择一个尺寸
            sz = random.choice(self.scales)
            # 对整个 batch 图像进行插值缩放
            images = F.interpolate(images, size=sz)

            # 如果有掩码，同步缩放（当前未实现完整逻辑）
            if 'masks' in targets[0]:
                for tg in targets:
                    tg['masks'] = F.interpolate(tg['masks'], size=sz, mode='nearest')
                raise NotImplementedError('掩码多尺度暂未实现')

        return images, targets