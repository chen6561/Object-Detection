"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn 

from ..misc import (MetricLogger, SmoothedValue, reduce_dict)


def train_one_epoch(model: nn.Module, criterion: nn.Module, dataloader, optimizer, ema, epoch, device):
    """
    图像分类任务：训练一个 epoch 的核心函数
    """
    # 把模型设置为训练模式（启用 Dropout/BatchNorm 等）
    model.train()

    # 日志记录器，用于平滑打印 loss、lr 等指标
    metric_logger = MetricLogger(delimiter="  ")
    # 单独添加学习率指标监控
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # 日志打印频率
    print_freq = 100
    # 打印标题
    header = 'Epoch: [{}]'.format(epoch)

    # 遍历数据集，自动打印进度、耗时、ETA
    for imgs, labels in metric_logger.log_every(dataloader, print_freq, header):
        # 把数据搬到指定设备（GPU/CPU）
        imgs = imgs.to(device)
        labels = labels.to(device)

        # 模型前向传播，得到预测值
        preds = model(imgs)
        # 计算分类损失
        loss: torch.Tensor = criterion(preds, labels)

        # 反向传播三步曲：清空梯度 → 反向 → 优化器更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 如果使用 EMA，更新滑动平均模型
        if ema is not None:
            ema.update(model)

        # 多卡分布式环境下，对 loss 进行聚合（求和/平均）
        loss_reduced_values = {k: v.item() for k, v in reduce_dict({'loss': loss}).items()}
        # 更新日志指标
        metric_logger.update(**loss_reduced_values)
        # 记录当前学习率
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    # 多卡之间同步指标
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # 返回全局平均指标（用于日志保存）
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats


@torch.no_grad()
def evaluate(model, criterion, dataloader, device):
    """
    图像分类模型：在验证集上评估准确率和损失
    """
    # 设置为评估模式，关闭 Dropout/BatchNorm
    model.eval()

    # 日志记录器
    metric_logger = MetricLogger(delimiter="  ")
    # 监控：准确率 & 损失
    metric_logger.add_meter('acc', SmoothedValue(window_size=1))
    metric_logger.add_meter('loss', SmoothedValue(window_size=1))

    header = 'Test:'
    # 遍历验证集
    for imgs, labels in metric_logger.log_every(dataloader, 10, header):
        imgs, labels = imgs.to(device), labels.to(device)
        # 模型前向
        preds = model(imgs)

        # 计算当前 batch 准确率
        acc = (preds.argmax(dim=-1) == labels).sum() / preds.shape[0]
        # 计算损失
        loss = criterion(preds, labels)

        # 多卡聚合指标
        dict_reduced = reduce_dict({'acc': acc, 'loss': loss})
        reduced_values = {k: v.item() for k, v in dict_reduced.items()}
        metric_logger.update(**reduced_values)

    # 多卡同步
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    # 返回全局平均的 acc 和 loss
    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats