"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torchvision
from torch import Tensor 
from typing import List, Tuple


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    计算 批量 广义交并比 (Generalized IoU)
    直接调用 torchvision 官方实现
    """
    # 检查框格式是否合法（x2 >= x1，y2 >= y1）
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    # 返回 GIoU 矩阵 [N, M]
    return torchvision.ops.generalized_box_iou(boxes1, boxes2)


# elementwise
def elementwise_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    逐元素计算 IoU（一对一，不是矩阵）
    Args:
        boxes1: [N, 4]
        boxes2: [N, 4]
    Returns:
        iou: [N, ]  每个框对的 iou
        union: [N, ] 每个框对的并集面积
    """
    # 计算两个框的面积
    area1 = torchvision.ops.box_area(boxes1)  # [N, ]
    area2 = torchvision.ops.box_area(boxes2)  # [N, ]

    # 相交区域左上角 = 取最大
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N, 2]
    # 相交区域右下角 = 取最小
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N, 2]

    # 相交宽高，最小为0避免负数
    wh = (rb - lt).clamp(min=0)  # [N, 2]
    inter = wh[:, 0] * wh[:, 1]  # [N, ] 相交面积

    # 并集面积
    union = area1 + area2 - inter
    # IoU = 交集 / 并集
    iou = inter / union

    return iou, union


def elementwise_generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    """
    逐元素计算 GIoU（一对一）
    解决 IoU 不相交时无法优化的问题
    Args:
        boxes1: [N, 4] (x1,y1,x2,y2)
        boxes2: [N, 4] (x1,y1,x2,y2)
    Returns:
        giou: [N, ]
    """
    # 检查框格式
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()

    # 先算 IoU
    iou, union = elementwise_box_iou(boxes1, boxes2)

    # 最小外接矩形（能包住两个框的最小框）
    lt = torch.min(boxes1[:, :2], boxes2[:, :2])  # [N, 2]
    rb = torch.max(boxes1[:, 2:], boxes2[:, 2:])  # [N, 2]
    wh = (rb - lt).clamp(min=0)
    area = wh[:, 0] * wh[:, 1]  # 外接矩形面积

    # GIoU 公式
    return iou - (area - union) / area


def check_point_inside_box(points: Tensor, boxes: Tensor, eps=1e-9) -> Tensor:
    """
    判断 点 是否在 框 内部
    Args:
        points: [K, 2]  (x,y)
        boxes: [N, 4]   (x1,y1,x2,y2)
    Returns:
        mask: [K, N]   每个点对应每个框是否在内部
    """
    # 拆分点坐标
    x, y = [p.unsqueeze(-1) for p in points.unbind(-1)]
    # 拆分框坐标
    x1, y1, x2, y2 = [x.unsqueeze(0) for x in boxes.unbind(-1)]

    # 计算点到四条边的距离
    l = x - x1   # 点到左边
    t = y - y1   # 点到上边
    r = x2 - x   # 点到右边
    b = y2 - y   # 点到下边

    # 全部 > 0 才在框内
    ltrb = torch.stack([l, t, r, b], dim=-1)
    mask = ltrb.min(dim=-1).values > eps

    return mask


def point_box_distance(points: Tensor, boxes: Tensor) -> Tensor:
    """
    计算 点 到 框四条边 的距离
    许多检测模型（如 YOLO、DETR）使用这种回归方式
    Args:
        boxes: [N, 4]  (x1,y1,x2,y2)
        points: [N, 2] (x,y)
    Returns:
        [N, 4]  (left, top, right, bottom)
    """
    # 拆分框为左上角、右下角
    x1y1, x2y2 = torch.split(boxes, 2, dim=-1)

    # 点到左边、上边距离
    lt = points - x1y1
    # 点到右边、下边距离
    rb = x2y2 - points

    # 拼接成 [l, t, r, b]
    return torch.concat([lt, rb], dim=-1)


def point_distance_box(points: Tensor, distances: Tensor) -> Tensor:
    """
    反向操作：根据 点 + 到四边距离 恢复 框坐标
    Args:
        points: [N, 2]   (x,y)
        distances: [N, 4] (l,t,r,b)
    Returns:
        boxes: [N, 4] (x1,y1,x2,y2)
    """
    # 拆分距离为 lt 和 rb
    lt, rb = torch.split(distances, 2, dim=-1)

    # 左上角 = 点 - 左上边距离
    x1y1 = -lt + points
    # 右下角 = 点 + 右下边距离
    x2y2 = rb + points

    # 拼接成框
    boxes = torch.concat([x1y1, x2y2], dim=-1)

    return boxes