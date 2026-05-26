"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn

# 导入 torchvision 库
import torchvision
# 关闭 torchvision v2 transforms 的 beta 版本警告
torchvision.disable_beta_transforms_warning()

# 导入 v2 版数据增强（支持图像、框、掩码同步变换）
import torchvision.transforms.v2 as T
import torchvision.transforms.v2.functional as F

# PIL 图像库
import PIL
import PIL.Image

# 类型提示
from typing import Any, Dict, List, Optional

# 导入工具函数：转换为 TVTensor (BoundingBox/Mask)
from .._misc import convert_to_tv_tensor, _boxes_keys
# 导入 TVTensor 类型（图像、视频、掩码、框）
from .._misc import Image, Video, Mask, BoundingBoxes
# 导入清理无效框的工具
from .._misc import SanitizeBoundingBoxes

# 导入注册器，将所有 Transform 注册到框架，可在 yaml 中配置使用
from ...core import register


# ----------------------
# 注册 torchvision 官方自带的增强算子
# ----------------------
# 随机光度扭曲（对比度、亮度、饱和度、色调）
RandomPhotometricDistort = register()(T.RandomPhotometricDistort)
# 随机向外缩放（给图像加边框扩大视野）
RandomZoomOut = register()(T.RandomZoomOut)
# 随机水平翻转
RandomHorizontalFlip = register()(T.RandomHorizontalFlip)
# 调整尺寸
Resize = register()(T.Resize)
# 清理无效框（去除面积为0的框）
SanitizeBoundingBoxes = register(name='SanitizeBoundingBoxes')(SanitizeBoundingBoxes)
# 随机裁剪
RandomCrop = register()(T.RandomCrop)
# 归一化
Normalize = register()(T.Normalize)


# ----------------------
# 自定义 Transform 1: 空变换（什么都不做，用于占位）
# ----------------------
@register()
class EmptyTransform(T.Transform):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, *inputs):
        # 直接返回输入，不做任何处理
        inputs = inputs if len(inputs) > 1 else inputs[0]
        return inputs


# ----------------------
# 自定义 Transform 2: 填充到指定尺寸
# 功能：将图像右下角填充到 target_size，不足的部分补 0
# ----------------------
@register()
class PadToSize(T.Pad):
    # 定义支持的数据类型：PIL图、图像张量、视频、掩码、框
    _transformed_types = (
        PIL.Image.Image,
        Image,
        Video,
        Mask,
        BoundingBoxes,
    )

    def _get_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        # 获取当前图像尺寸
        sp = F.get_spatial_size(flat_inputs[0])
        # 计算需要填充的宽度和高度 (右、下)
        h, w = self.size[1] - sp[0], self.size[0] - sp[1]
        # padding = [左, 上, 右, 下]
        self.padding = [0, 0, w, h]
        return dict(padding=self.padding)

    def make_params(self, flat_inputs: List[Any]) -> Dict[str, Any]:
        return self._get_params(flat_inputs)

    def __init__(self, size, fill=0, padding_mode='constant') -> None:
        # 支持传入单个数字 size=640
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        # 父类初始化，暂时 padding=0
        super().__init__(0, fill, padding_mode)

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        fill = self._fill[type(inpt)]
        padding = params['padding']
        # 执行填充操作
        return F.pad(inpt, padding=padding, fill=fill, padding_mode=self.padding_mode)

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)

    def __call__(self, *inputs: Any) -> Any:
        outputs = super().forward(*inputs)
        # 将 padding 信息记录到 target 中
        if len(outputs) > 1 and isinstance(outputs[1], dict):
            outputs[1]['padding'] = torch.tensor(self.padding)
        return outputs


# ----------------------
# 自定义 Transform 3: 随机 IoU 裁剪
# 功能：按 IoU 策略裁剪，常用于检测数据增强
# ----------------------
@register()
class RandomIoUCrop(T.RandomIoUCrop):
    def __init__(self, min_scale: float = 0.3, max_scale: float = 1, min_aspect_ratio: float = 0.5, max_aspect_ratio: float = 2, sampler_options: Optional[List[float]] = None, trials: int = 40, p: float = 1.0):
        super().__init__(min_scale, max_scale, min_aspect_ratio, max_aspect_ratio, sampler_options, trials)
        # 新增概率参数 p：控制是否执行裁剪
        self.p = p

    def __call__(self, *inputs: Any) -> Any:
        # 按概率 p 决定是否执行
        if torch.rand(1) >= self.p:
            return inputs if len(inputs) > 1 else inputs[0]

        return super().forward(*inputs)


# ----------------------
# 自定义 Transform 4: 框格式转换
# 功能：xyxy <-> xywh / 归一化
# ----------------------
@register()
class ConvertBoxes(T.Transform):
    _transformed_types = (BoundingBoxes,)

    def __init__(self, fmt='', normalize=False) -> None:
        super().__init__()
        self.fmt = fmt          # 目标格式 'xyxy' / 'xywh'
        self.normalize = normalize  # 是否归一化到 0~1

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        spatial_size = getattr(inpt, _boxes_keys[1])
        # 格式转换
        if self.fmt:
            in_fmt = inpt.format.value.lower()
            inpt = torchvision.ops.box_convert(inpt, in_fmt=in_fmt, out_fmt=self.fmt.lower())
            inpt = convert_to_tv_tensor(inpt, key='boxes', box_format=self.fmt.upper(), spatial_size=spatial_size)

        # 归一化：除以图像宽高
        if self.normalize:
            inpt = inpt / torch.tensor(spatial_size[::-1]).tile(2)[None]

        return inpt

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)


# ----------------------
# 自定义 Transform 5: PIL 图像转张量
# 功能：PIL Image → 张量 + 归一化到 0~1
# ----------------------
@register()
class ConvertPILImage(T.Transform):
    _transformed_types = (PIL.Image.Image,)

    def __init__(self, dtype='float32', scale=True) -> None:
        super().__init__()
        self.dtype = dtype
        self.scale = scale

    def _transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        # PIL → Tensor
        inpt = F.pil_to_tensor(inpt)
        # 转为 float32
        if self.dtype == 'float32':
            inpt = inpt.float()
        # 缩放到 0~1
        if self.scale:
            inpt = inpt / 255.
        # 包装为 Image TVTensor
        inpt = Image(inpt)
        return inpt

    def transform(self, inpt: Any, params: Dict[str, Any]) -> Any:
        return self._transform(inpt, params)