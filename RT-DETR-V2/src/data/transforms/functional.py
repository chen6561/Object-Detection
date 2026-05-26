import torch
import torchvision.transforms.functional as F

from packaging import version
from typing import Optional, List
from torch import Tensor

# 处理低版本 torchvision 空张量的 bug
import torchvision

if version.parse(torchvision.__version__) < version.parse('0.7'):
    from torchvision.ops import _new_empty_tensor
    from torchvision.ops.misc import _output_size


def interpolate(input, size=None, scale_factor=None, mode="nearest", align_corners=None):
    # type: (Tensor, Optional[List[int]], Optional[float], str, Optional[bool]) -> Tensor
    """
    兼容版上采样/下采样函数
    等价于 nn.functional.interpolate，但兼容空张量（解决低版本 torchvision 报错）
    未来 PyTorch 原生支持后可删除
    """
    # 低版本 torchvision 单独处理空张量
    if version.parse(torchvision.__version__) < version.parse('0.7'):
        if input.numel() > 0:
            return torch.nn.functional.interpolate(
                input, size, scale_factor, mode, align_corners
            )

        output_shape = _output_size(2, input, size, scale_factor)
        output_shape = list(input.shape[:-2]) + list(output_shape)
        return _new_empty_tensor(input, output_shape)
    else:
        return torchvision.ops.misc.interpolate(input, size, scale_factor, mode, align_corners)


def crop(image, target, region):
    """
    对图像和标注（框、掩码、标签）同步执行裁剪
    Args:
        image: 输入图像
        target: 标注字典（boxes、labels、masks等）
        region: 裁剪区域 (i, j, h, w) → 起始行、起始列、高度、宽度
    Return:
        裁剪后的图像 + 更新后的标注
    """
    # 裁剪图像
    cropped_image = F.crop(image, *region)

    target = target.copy()
    i, j, h, w = region

    # 更新目标尺寸
    target["size"] = torch.tensor([h, w])

    # 需要保留的标注字段
    fields = ["labels", "area", "iscrowd"]

    # 如果有框，更新框坐标
    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        # 减去裁剪偏移量
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        # 限制在裁剪区域内
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        # 重新计算面积
        area = (cropped_boxes[:, 1, :] - cropped_boxes[:, 0, :]).prod(dim=1)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        target["area"] = area
        fields.append("boxes")

    # 如果有掩码，裁剪掩码
    if "masks" in target:
        target['masks'] = target['masks'][:, i:i + h, j:j + w]
        fields.append("masks")

    # 移除面积为0的无效目标
    if "boxes" in target or "masks" in target:
        # 优先用框判断有效性
        if "boxes" in target:
            cropped_boxes = target['boxes'].reshape(-1, 2, 2)
            keep = torch.all(cropped_boxes[:, 1, :] > cropped_boxes[:, 0, :], dim=1)
        else:
            keep = target['masks'].flatten(1).any(1)

        # 过滤所有标注字段
        for field in fields:
            target[field] = target[field][keep]

    return cropped_image, target


def hflip(image, target):
    """
    图像 + 标注 同步水平翻转
    自动修正框坐标、翻转掩码
    """
    flipped_image = F.hflip(image)

    w, h = image.size

    target = target.copy()
    # 翻转框坐标
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes

    # 水平翻转掩码
    if "masks" in target:
        target['masks'] = target['masks'].flip(-1)

    return flipped_image, target


def resize(image, target, size, max_size=None):
    """
    图像 + 标注 同步缩放
    自动保持宽高比、缩放框、缩放掩码
    Args:
        size: 目标最短边 / (w, h)
        max_size: 最长边上限
    """

    # 根据宽高比计算新尺寸
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))

        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)

        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)

        return (oh, ow)

    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)

    # 计算最终输出尺寸
    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)

    if target is None:
        return rescaled_image, None

    # 计算宽高缩放比例
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios

    target = target.copy()
    # 缩放框
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes

    # 缩放面积
    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area

    h, w = size
    target["size"] = torch.tensor([h, w])

    # 缩放掩码
    if "masks" in target:
        target['masks'] = interpolate(
            target['masks'][:, None].float(), size, mode="nearest")[:, 0] > 0.5

    return rescaled_image, target


def pad(image, target, padding):
    """
    图像 + 标注 向右、向下填充
    只在右侧和下侧 padding，常用于统一到网络输入尺寸
    """
    # 只在右下角填充 (左, 上, 右, 下)
    padded_image = F.pad(image, (0, 0, padding[0], padding[1]))
    if target is None:
        return padded_image, None

    target = target.copy()
    target["size"] = torch.tensor(padded_image.size[::-1])

    # 掩码同步填充
    if "masks" in target:
        target['masks'] = torch.nn.functional.pad(target['masks'], (0, padding[0], 0, padding[1]))

    return padded_image, target