"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn

import torchvision
torchvision.disable_beta_transforms_warning()
import torchvision.transforms.v2 as T

from typing import Any, Dict, List, Optional

# 导入空变换（占位用）
from ._transforms import EmptyTransform
# 导入注册器和全局配置字典
from ...core import register, GLOBAL_CONFIG


# 注册 Compose 增强组合类
@register()
class Compose(T.Compose):
    def __init__(self, ops, policy=None) -> None:
        transforms = []

        # 解析 YAML 中的增强操作列表
        if ops is not None:
            for op in ops:
                # 如果 op 是字典（从配置创建）
                if isinstance(op, dict):
                    # 取出类型名，如 RandomHorizontalFlip
                    name = op.pop('type')
                    # 从全局注册器中找到对应类并实例化
                    transfom = getattr(GLOBAL_CONFIG[name]['_pymodule'], GLOBAL_CONFIG[name]['_name'])(**op)
                    transforms.append(transfom)
                    # 恢复 type 字段，防止影响原字典
                    op['type'] = name

                # 如果已经是模块，直接加入列表
                elif isinstance(op, nn.Module):
                    transforms.append(op)

                else:
                    raise ValueError('Unsupported transform type')

        # 如果没有任何增强，添加空变换占位
        else:
            transforms = [EmptyTransform(), ]

        # 调用父类初始化
        super().__init__(transforms=transforms)

        # 增强策略：默认 default
        if policy is None:
            policy = {'name': 'default'}

        self.policy = policy
        # 全局样本计数（用于 stop_sample 策略）
        self.global_samples = 0

    def forward(self, *inputs: Any) -> Any:
        """
        根据策略名称，自动选择前向传播方式
        """
        return self.get_forward(self.policy['name'])(*inputs)

    def get_forward(self, name):
        """
        策略映射表
        """
        forwards = {
            'default': self.default_forward,           # 正常执行
            'stop_epoch': self.stop_epoch_forward,     # 到指定 epoch 后停止某些增强
            'stop_sample': self.stop_sample_forward,   # 到指定样本数后停止某些增强
        }
        return forwards[name]

    def default_forward(self, *inputs: Any) -> Any:
        """
        默认前向：顺序执行所有增强
        """
        sample = inputs if len(inputs) > 1 else inputs[0]
        for transform in self.transforms:
            sample = transform(sample)
        return sample

    def stop_epoch_forward(self, *inputs):
        """
        按 epoch 停止增强：
        达到指定 epoch 后，跳过 policy_ops 中的增强
        """
        sample = inputs if len(inputs) > 1 else inputs[0]
        dataset = sample[-1]

        # 获取当前 epoch
        cur_epoch = dataset.epoch
        policy_ops = self.policy['ops']       # 要停止的增强列表
        policy_epoch = self.policy['epoch']   # 从第几轮开始停止

        # 遍历增强
        for transform in self.transforms:
            # 如果当前增强在停止列表 && 已达到指定 epoch → 跳过
            if type(transform).__name__ in policy_ops and cur_epoch >= policy_epoch:
                pass
            else:
                sample = transform(sample)

        return sample

    def stop_sample_forward(self, *inputs):
        """
        按样本数停止增强：
        训练总样本达到指定数量后，停止某些增强
        """
        sample = inputs if len(inputs) > 1 else inputs[0]
        dataset = sample[-1]

        policy_ops = self.policy['ops']
        policy_sample = self.policy['sample']

        for transform in self.transforms:
            # 总样本数足够 → 跳过指定增强
            if type(transform).__name__ in policy_ops and self.global_samples >= policy_sample:
                pass
            else:
                sample = transform(sample)

        # 全局样本计数 +1
        self.global_samples += 1

        return sample