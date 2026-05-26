"""
copy and modified https://github.com/pytorch/vision/blob/main/references/detection/coco_utils.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.utils.data
import torchvision
import torchvision.transforms.functional as TVF
# 导入更快的 COCO 评估库
from faster_coco_eval import COCO
import faster_coco_eval.core.mask as mask_util


def convert_coco_poly_to_mask(segmentations, height, width):
    """
    将 COCO 的多边形标注 (polygon) 转换为二进制掩码张量
    Args:
        segmentations: 标注的多边形坐标列表
        height: 图像高度
        width: 图像宽度
    Returns:
        掩码张量 [N, H, W]
    """
    masks = []
    # 遍历每个目标的分割标注
    for polygons in segmentations:
        # 将多边形转为 RLE 压缩格式
        rles = mask_util.frPyObjects(polygons, height, width)
        # 解码为掩码数组
        mask = mask_util.decode(rles)
        # 确保是 3 维数组 (H, W, 1)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        # 转为 PyTorch 张量
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        # 合并通道，得到单通道二值掩码
        mask = mask.any(dim=2)
        masks.append(mask)

    # 堆叠所有目标的掩码
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        # 无目标时返回空张量
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask:
    """
    COCO 标注转换器
    功能：把 JSON 原始标注 → 模型训练用的字典格式
    包含：框、标签、掩码、关键点
    """
    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        anno = target["annotations"]

        # 过滤掉 crowd 标注（不参与训练）
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        # 1. 处理边界框
        boxes = [obj["bbox"] for obj in anno]
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        # COCO 格式 (x,y,w,h) → 标准格式 (x1,y1,x2,y2)
        boxes[:, 2:] += boxes[:, :2]
        # 防止坐标越界
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # 2. 处理类别
        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        # 3. 处理分割掩码
        segmentations = [obj["segmentation"] for obj in anno]
        masks = convert_coco_poly_to_mask(segmentations, h, w)

        # 4. 处理关键点（如果有）
        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        # 过滤无效框（宽/高为 0）
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        # 组装最终训练用 target
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # 评估用字段
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target


def _coco_remove_images_without_annotations(dataset, cat_list=None):
    """
    过滤掉没有有效标注的图像（空图、无效框、关键点不足）
    """
    def _has_only_empty_bbox(anno):
        # 判断是否所有框面积都接近 0
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _count_visible_keypoints(anno):
        # 统计可见关键点数量
        return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

    min_keypoints_per_image = 10

    def _has_valid_annotation(anno):
        # 无标注 → 无效
        if len(anno) == 0:
            return False
        # 全是空框 → 无效
        if _has_only_empty_bbox(anno):
            return False
        # 普通检测任务 → 有效
        if "keypoints" not in anno[0]:
            return True
        # 关键点任务 → 点数足够 → 有效
        if _count_visible_keypoints(anno) >= min_keypoints_per_image:
            return True
        return False

    # 遍历数据集，保留有效图片索引
    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)
        # 按类别过滤
        if cat_list:
            anno = [obj for obj in anno if obj["category_id"] in cat_list]
        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    # 构建只包含有效图片的子集
    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset


def convert_to_coco_api(ds):
    """
    将自定义 Dataset 转换为 COCO API 格式（用于评估）
    把模型输出的预测结果转换成 COCO 标准格式
    """
    coco_ds = COCO()
    ann_id = 1  # 标注 ID 必须从 1 开始
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()

    for img_idx in range(len(ds)):
        # 加载原始数据（不经过 transform）
        img, targets = ds.load_item(img_idx)
        width, height = img.size

        image_id = targets["image_id"].item()

        # 构建图像信息
        img_dict = {}
        img_dict["id"] = image_id
        img_dict["width"] = width
        img_dict["height"] = height
        dataset["images"].append(img_dict)

        # 转换框格式：xyxy → xywh
        bboxes = targets["boxes"].clone()
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()

        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()

        # 处理掩码
        if "masks" in targets:
            masks = targets["masks"]
            # 调整内存格式以适配 COCO 工具
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)

        # 处理关键点
        if "keypoints" in targets:
            keypoints = targets["keypoints"]
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()

        # 逐个目标构建标注
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i]
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id

            if "masks" in targets:
                ann["segmentation"] = mask_util.encode(masks[i].numpy())
            if "keypoints" in targets:
                ann["keypoints"] = keypoints[i]
                ann["num_keypoints"] = sum(k != 0 for k in keypoints[i][2::3])

            dataset["annotations"].append(ann)
            ann_id += 1

    # 填充类别信息
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()  # 建立索引
    return coco_ds


def get_coco_api_from_dataset(dataset):
    """
    从 Dataset 中获取 COCO API 对象
    自动处理嵌套的 Subset 类型
    """
    # 最多解包 10 层 Subset，找到原始数据集
    for _ in range(10):
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset

    # 如果是官方 COCO 数据集，直接返回其 coco 属性
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco

    # 否则手动转换为 COCO API
    return convert_to_coco_api(dataset)