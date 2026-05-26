"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 
from typing import List 

# 导入注册器，用于把模型注册到框架中
from ...core import register


# 对外暴露的类名
__all__ = ['RTDETR', ]


# 注册模型，让框架可以通过名称找到这个类
@register()
class RTDETR(nn.Module):
    """
    RT-DETR 目标检测模型总结构
    由 backbone + encoder + decoder 三部分组成
    这是整个模型的最上层容器，负责串联各模块
    """
    # 注入模块列表（框架自动注入这些子模块）
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(
        self,
        backbone: nn.Module,  # 主干网络（特征提取）
        encoder: nn.Module,   # Transformer 编码器
        decoder: nn.Module,   # Transformer 解码器（预测头）
    ):
        super().__init__()
        # 主干：提取图像多尺度特征
        self.backbone = backbone
        # 解码器：输出最终检测框、类别
        self.decoder = decoder
        # 编码器：特征增强、注意力交互
        self.encoder = encoder

    def forward(self, x, targets=None):
        """
        模型前向传播
        Args:
            x: 输入图像 [B, C, H, W]
            targets: 训练时传入标注框/类别
        Returns:
            训练：返回损失
            推理：返回预测框、类别、分数
        """
        # 1. 主干网络提取特征
        x = self.backbone(x)
        # 2. 编码器特征增强
        x = self.encoder(x)
        # 3. 解码器预测（训练时需要 targets 计算损失）
        x = self.decoder(x, targets)

        return x

    def deploy(self, ):
        """
        模型部署模式转换
        用于推理优化：
        - 切换到 eval 模式
        - 把训练组件转为推理组件（如重参数化、融合 BN）
        """
        self.eval()
        for m in self.modules():
            # 如果子模块有部署转换方法，自动调用
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self