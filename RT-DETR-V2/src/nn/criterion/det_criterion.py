"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn.functional as F 
import torch.distributed
import torchvision

from ...misc import box_ops
from ...misc import dist_utils
from ...core import register


@register()
class DetCriterion(torch.nn.Module):
    """
    目标检测 损失函数 总入口（RT-DETR 专用）
    支持：
        1. boxes 损失（L1 + GIoU）
        2. giou 单独损失
        3. vfl 损失（Variable Focal Loss）
        4. focal 损失（Focal Loss）
    作用：
        接收模型输出 + 标签 → 计算全部训练损失
    """
    __share__ = ['num_classes']   # 共享参数：类别数
    __inject__ = ['matcher']      # 自动注入：标签匹配器（预测 ↔ 标签）

    def __init__(self,
                losses,
                weight_dict,
                num_classes=80,
                alpha=0.75,
                gamma=2.0,
                box_fmt='cxcywh',
                matcher=None):
        """
        初始化损失函数
        Args:
            losses: 要计算哪些损失，如 ['boxes', 'vfl', 'focal']
            weight_dict: 各项损失的权重
            num_classes: 类别数（默认 80 类 COCO）
            alpha: Focal Loss 超参
            gamma: Focal Loss 超参
            box_fmt: 框格式（cxcywh / xyxy）
            matcher: 预测框 ↔ 标签框 的匹配器
        """
        super().__init__()
        self.losses = losses
        self.weight_dict = weight_dict
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes
        self.box_fmt = box_fmt
        assert matcher is not None, 'Matcher 不能为空，必须注入'
        self.matcher = matcher

    def forward(self, outputs, targets, **kwargs):
        """
        前向计算所有损失
        Args:
            outputs: 模型输出 → pred_boxes, pred_logits
            targets: 数据标签 → boxes, labels
            kwargs: 可传入 epoch 等信息
        Return:
            所有损失组成的字典
        """
        # 1. 使用 matcher 计算：预测 ↔ 标签 的匹配结果
        matched = self.matcher(outputs, targets)
        values = matched['values']
        indices = matched['indices']

        # 2. 计算正样本数量（用于归一化损失）
        num_boxes = self._get_positive_nums(indices)

        # 3. 遍历并计算所有需要的损失
        losses = {}
        for loss in self.losses:
            # 获取单个损失
            l_dict = self.get_loss(loss, outputs, targets, indices, num_boxes)
            # 乘以损失权重
            l_dict = {k: l_dict[k] * self.weight_dict[k] for k in l_dict if k in self.weight_dict}
            losses.update(l_dict)

        return losses

    def _get_src_permutation_idx(self, indices):
        """
        获取 预测框 被匹配到的索引
        返回 (batch_idx, src_idx)
        """
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        """
        获取 标签框 被匹配到的索引
        """
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def _get_positive_nums(self, indices):
        """
        计算全局正样本数量（多卡同步）
        用于平均损失，保证多卡训练数值一致
        """
        num_pos = sum(len(i) for (i, _) in indices)
        num_pos = torch.as_tensor([num_pos], dtype=torch.float32, device=indices[0][0].device)

        # 多卡同步求和
        if dist_utils.is_dist_available_and_initialized():
            torch.distributed.all_reduce(num_pos)

        # 全局平均，至少为 1，避免除 0
        num_pos = torch.clamp(num_pos / dist_utils.get_world_size(), min=1).item()
        return num_pos

    def loss_labels_focal(self, outputs, targets, indices, num_boxes):
        """
        Focal Loss 分类损失
        用于抑制负样本，解决类别不平衡
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        # 取出正样本索引
        idx = self._get_src_permutation_idx(indices)

        # 构建目标类别（背景 = num_classes）
        target_classes_o = torch.cat([t["labels"][j] for t, (_, j) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        # one-hot 标签
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1].to(src_logits.dtype)

        # 计算 focal loss
        loss = torchvision.ops.sigmoid_focal_loss(src_logits, target, self.alpha, self.gamma, reduction='none')
        loss = loss.sum() / num_boxes

        return {'loss_focal': loss}

    def loss_labels_vfl(self, outputs, targets, indices, num_boxes):
        """
        VFL = Variable Focal Loss
        用 IoU 作为分类监督信号，让分类分数与定位质量对齐
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        # 取出正样本框
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][j] for t, (_, j) in zip(targets, indices)], dim=0)

        # 转换框格式 → 计算 IoU
        src_boxes = torchvision.ops.box_convert(src_boxes, in_fmt=self.box_fmt, out_fmt='xyxy')
        target_boxes = torchvision.ops.box_convert(target_boxes, in_fmt=self.box_fmt, out_fmt='xyxy')
        iou, _ = box_ops.elementwise_box_iou(src_boxes.detach(), target_boxes)

        # 取出分类 logits
        src_logits: torch.Tensor = outputs['pred_logits']

        # 构建目标类别
        target_classes_o = torch.cat([t["labels"][j] for t, (_, j) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        # one-hot 标签
        target = F.one_hot(target_classes, num_classes=self.num_classes + 1)[..., :-1]

        # VFL 核心：用 IoU 作为监督权重
        target_score_o = torch.zeros_like(target_classes, dtype=src_logits.dtype)
        target_score_o[idx] = iou.to(src_logits.dtype)
        target_score = target_score_o.unsqueeze(-1) * target

        # 带权重的交叉熵
        src_score = F.sigmoid(src_logits.detach())
        weight = self.alpha * src_score.pow(self.gamma) * (1 - target) + target_score

        loss = F.binary_cross_entropy_with_logits(src_logits, target_score, weight=weight, reduction='none')
        loss = loss.sum() / num_boxes

        return {'loss_vfl': loss}

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """
        框回归损失 = L1 Loss + GIoU Loss
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)

        # 正样本框
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}

        # L1 损失
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes

        # GIoU 损失
        src_boxes = torchvision.ops.box_convert(src_boxes, in_fmt=self.box_fmt, out_fmt='xyxy')
        target_boxes = torchvision.ops.box_convert(target_boxes, in_fmt=self.box_fmt, out_fmt='xyxy')
        loss_giou = 1 - box_ops.elementwise_generalized_box_iou(src_boxes, target_boxes)
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def loss_boxes_giou(self, outputs, targets, indices, num_boxes):
        """
        仅计算 GIoU 损失（单独使用）
        """
        assert 'pred_boxes' in outputs
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)

        losses = {}
        src_boxes = torchvision.ops.box_convert(src_boxes, in_fmt=self.box_fmt, out_fmt='xyxy')
        target_boxes = torchvision.ops.box_convert(target_boxes, in_fmt=self.box_fmt, out_fmt='xyxy')
        loss_giou = 1 - box_ops.elementwise_generalized_box_iou(src_boxes, target_boxes)
        losses['loss_giou'] = loss_giou.sum() / num_boxes

        return losses

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        """
        损失映射表
        根据字符串名称调用对应损失函数
        """
        loss_map = {
            'boxes': self.loss_boxes,
            'giou': self.loss_boxes_giou,
            'vfl': self.loss_labels_vfl,
            'focal': self.loss_labels_focal,
        }
        assert loss in loss_map, f'损失 {loss} 未定义，请检查配置'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)