"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import importlib.metadata
from torch import Tensor 

# --------------------------
# 版本兼容：根据安装的 torchvision 版本，导入对应的类
# 因为 0.15 / 0.16 / 0.17+ 的 TVTensor 路径不一样
# --------------------------

# 如果是 torchvision 0.15.2 版本
if importlib.metadata.version('torchvision') == '0.15.2':
    import torchvision
    torchvision.disable_beta_transforms_warning()  # 关闭 beta 警告

    # 0.15 版本从 datapoints 导入
    from torchvision.datapoints import BoundingBox as BoundingBoxes
    from torchvision.datapoints import BoundingBoxFormat, Mask, Image, Video
    from torchvision.transforms.v2 import SanitizeBoundingBox as SanitizeBoundingBoxes
    _boxes_keys = ['format', 'spatial_size']  # 框参数 key：格式 + 图像尺寸

# 如果是 0.16 <= 版本 < 0.17
elif '0.17' > importlib.metadata.version('torchvision') >= '0.16':
    import torchvision
    torchvision.disable_beta_transforms_warning()

    # 0.16 开始迁移到 tv_tensors
    from torchvision.transforms.v2 import SanitizeBoundingBoxes
    from torchvision.tv_tensors import (
        BoundingBoxes, BoundingBoxFormat, Mask, Image, Video)
    _boxes_keys = ['format', 'canvas_size']  # 参数名改为 canvas_size

# 如果是 0.17 及以上版本
elif importlib.metadata.version('torchvision') >= '0.17':
    import torchvision
    from torchvision.transforms.v2 import SanitizeBoundingBoxes
    from torchvision.tv_tensors import (
        BoundingBoxes, BoundingBoxFormat, Mask, Image, Video)
    _boxes_keys = ['format', 'canvas_size']

# 低于 0.15.2 直接报错
else:
    raise RuntimeError('Please make sure torchvision version >= 0.15.2')


# --------------------------
# 核心函数：把普通张量 → torchvision TVTensor
# 让框、掩码能被 v2 自动处理，不会在数据增强时被破坏
# --------------------------
def convert_to_tv_tensor(tensor: Tensor, key: str, box_format='xyxy', spatial_size=None) -> Tensor:
    """
    将普通张量转换为 TorchVision TVTensor（自动适配版本）
    Args:
        tensor: 输入张量 (boxes 或 masks)
        key: 类型，只支持 'boxes' 或 'masks'
        box_format: 框格式，默认 'xyxy'
        spatial_size: 图像尺寸 (H, W)
    Return:
        包装好的 TVTensor (BoundingBoxes / Mask)
    """
    # 只支持框和掩码转换
    assert key in ('boxes', 'masks', ), "Only support 'boxes' and 'masks'"

    # 转换为 BoundingBoxes 类型（带格式、尺寸信息）
    if key == 'boxes':
        box_format = getattr(BoundingBoxFormat, box_format.upper())  # 转成枚举类型
        _kwargs = dict(zip(_boxes_keys, [box_format, spatial_size])) # 适配不同版本参数名
        return BoundingBoxes(tensor, **_kwargs)

    # 转换为 Mask 类型
    if key == 'masks':
       return Mask(tensor)