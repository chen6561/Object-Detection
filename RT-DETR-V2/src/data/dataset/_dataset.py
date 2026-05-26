"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.utils.data as data


class DetDataset(data.Dataset):
    """
    目标检测数据集基类（抽象基类）
    所有自定义的检测数据集（如 COCO、VOC 等）都需要继承这个类
    继承自 torch.utils.data.Dataset
    """

    def __getitem__(self, index):
        """
        PyTorch Dataset 必须实现的核心方法
        根据索引 index 获取一条数据：图像 + 标签
        Args:
            index: 数据索引
        Returns:
            img: 经过预处理的图像
            target: 图像对应的标注（框、类别等）
        """
        # 1. 调用 load_item 加载原始图像和标注（子类必须实现）
        img, target = self.load_item(index)

        # 2. 如果定义了数据增强/预处理 transforms，则对图像和标签进行处理
        if self.transforms is not None:
            img, target, _ = self.transforms(img, target, self)

        # 3. 返回最终处理好的图像和标注
        return img, target

    def load_item(self, index):
        """
        子类必须实现的抽象方法
        作用：在数据增强之前，加载原始的图像和标注
        Args:
            index: 数据索引
        Returns:
            img: 原始图像
            target: 原始标注
        """
        raise NotImplementedError("Please implement this function to return item before `transforms`.")

    def set_epoch(self, epoch) -> None:
        """
        设置当前训练的 epoch
        可用于实现按 epoch 切换数据增强策略等功能
        Args:
            epoch: 当前训练轮数
        """
        self._epoch = epoch

    @property
    def epoch(self):
        """
        获取当前 epoch（通过 @property 变成属性调用）
        Returns:
            当前 epoch，未设置时返回 -1
        """
        return self._epoch if hasattr(self, '_epoch') else -1