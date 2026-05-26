"""
Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/engine.py

Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import sys
import math
from typing import Iterable

import torch
import torch.amp 
from torch.utils.tensorboard import SummaryWriter
from torch.cuda.amp.grad_scaler import GradScaler

from ..optim import ModelEMA, Warmup
from ..data import CocoEvaluator
from ..misc import MetricLogger, SmoothedValue, dist_utils


def train_one_epoch(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    data_loader: Iterable,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    max_norm: float = 0,
    **kwargs
):
    """
    训练一个 epoch 的核心逻辑
    支持：混合精度、EMA、梯度裁剪、学习率预热、多卡分布式
    """
    # 开启训练模式
    model.train()
    criterion.train()

    # 日志管理器：自动平滑 loss、打印、计时、分布式同步
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    # 打印频率
    print_freq = kwargs.get('print_freq', 10)
    # TensorBoard 日志
    writer :SummaryWriter = kwargs.get('writer', None)

    # EMA 模型
    ema :ModelEMA = kwargs.get('ema', None)
    # 混合精度 scaler
    scaler :GradScaler = kwargs.get('scaler', None)
    # 学习率预热
    lr_warmup_scheduler :Warmup = kwargs.get('lr_warmup_scheduler', None)

    # ===================== 遍历一个 epoch 的所有批次 =====================
    for i, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # 数据搬到设备
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 全局步数
        global_step = epoch * len(data_loader) + i
        metas = dict(epoch=epoch, step=i, global_step=global_step)

        # ===================== 混合精度训练 =====================
        if scaler is not None:
            # 自动混合精度前向
            with torch.autocast(device_type=str(device), cache_enabled=True):
                outputs = model(samples, targets=targets)

            # 关闭 amp 计算损失
            with torch.autocast(device_type=str(device), enabled=False):
                loss_dict = criterion(outputs, targets, **metas)

            # 总损失
            loss = sum(loss_dict.values())

            # 反向传播（缩放梯度）
            scaler.scale(loss).backward()

            # 梯度裁剪
            if max_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            # 更新参数 + 清空梯度
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # ===================== 普通精度训练 =====================
        else:
            outputs = model(samples, targets=targets)
            loss_dict = criterion(outputs, targets, **metas)

            loss : torch.Tensor = sum(loss_dict.values())
            optimizer.zero_grad()
            loss.backward()

            if max_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)

            optimizer.step()

        # ===================== EMA 更新 =====================
        if ema is not None:
            ema.update(model)

        # ===================== 学习率预热 step =====================
        if lr_warmup_scheduler is not None:
            lr_warmup_scheduler.step()

        # ===================== 多卡损失聚合 =====================
        loss_dict_reduced = dist_utils.reduce_dict(loss_dict)
        loss_value = sum(loss_dict_reduced.values())

        # 防止 loss 爆炸
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # ===================== 日志更新 =====================
        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        # ===================== TensorBoard 写入 =====================
        if writer and dist_utils.is_main_process():
            writer.add_scalar('Loss/total', loss_value.item(), global_step)
            for j, pg in enumerate(optimizer.param_groups):
                writer.add_scalar(f'Lr/pg_{j}', pg['lr'], global_step)
            for k, v in loss_dict_reduced.items():
                writer.add_scalar(f'Loss/{k}', v.item(), global_step)

    # ===================== 多卡指标同步 =====================
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # 返回全局平均指标
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    criterion: torch.nn.Module,
    postprocessor,
    data_loader,
    coco_evaluator: CocoEvaluator,
    device
):
    """
    验证/测试逻辑
    输出：COCO 指标（AP、AP50、AP75...）
    """
    model.eval()
    criterion.eval()
    coco_evaluator.cleanup()
    iou_types = coco_evaluator.iou_types

    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    # ===================== 遍历验证集 =====================
    for samples, targets in metric_logger.log_every(data_loader, 10, header):
        samples = samples.to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # 模型前向
        outputs = model(samples)

        # 原始图像尺寸（用于还原框）
        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)

        # 后处理：输出框 → 原始图像坐标
        results = postprocessor(outputs, orig_target_sizes)

        # 按 image_id 组织结果
        res = {target['image_id'].item(): output for target, output in zip(targets, results)}

        # 更新 COCO 评估器
        if coco_evaluator is not None:
            coco_evaluator.update(res)

    # ===================== 多卡同步 =====================
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    if coco_evaluator is not None:
        coco_evaluator.synchronize_between_processes()

    # ===================== 计算 COCO 指标 =====================
    if coco_evaluator is not None:
        coco_evaluator.accumulate()
        coco_evaluator.summarize()

    # 收集指标
    stats = {}
    if coco_evaluator is not None:
        if 'bbox' in iou_types:
            stats['coco_eval_bbox'] = coco_evaluator.coco_eval['bbox'].stats.tolist()
        if 'segm' in iou_types:
            stats['coco_eval_masks'] = coco_evaluator.coco_eval['segm'].stats.tolist()

    return stats, coco_evaluator