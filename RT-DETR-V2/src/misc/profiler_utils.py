"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import re
import torch
import torch.nn as nn
from torch import Tensor

from typing import List


def stats(
    model: nn.Module,
    data: Tensor = None,
    input_shape: List = [1, 3, 640, 640],
    device: str = 'cpu',
    verbose=False
) -> str:
    """
    模型性能统计工具
    计算：可训练参数量、FLOPs（计算量）、前向耗时
    Args:
        model: 待测试模型
        data: 输入数据（不指定则自动生成）
        input_shape: 自动生成输入的形状 [batch, channel, H, W]
        device: 使用 cpu / cuda
        verbose: 是否打印详细信息
    Returns:
        参数量、FLOPs、原始 profiler 信息
    """
    # 记录模型原来的训练/评估状态
    is_training = model.training

    # --------------------------
    # 1. 计算可训练参数量
    # --------------------------
    model.train()
    # 只计算需要梯度的参数（可训练参数）
    num_params = sum([p.numel() for p in model.parameters() if p.requires_grad])

    # 切换到评估模式（避免 BN/Dropout 影响测速）
    model.eval()
    model = model.to(device)

    # 如果没传入数据，自动生成随机输入
    if data is None:
        data = torch.rand(*input_shape, device=device)

    # --------------------------
    # 2. 启用 PyTorch Profiler 测速、算 FLOPs
    # --------------------------
    def trace_handler(prof):
        # 可选：自定义打印 prof 结果
        print(prof.key_averages().table(
            sort_by="self_cuda_time_total", row_limit=-1))

    # 连续测 2 轮 active 阶段
    num_active = 2

    # 启动性能分析器
    with torch.profiler.profile(
        # 监控 CPU + CUDA
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        # 调度：等待1轮 → 热身1轮 → 正式测试2轮
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=1,
            active=num_active,
            repeat=1
        ),
        # 开启 FLOPs 统计（核心）
        with_flops=True,
    ) as p:

        # 跑 5 次前向，让 profiler 完整采集数据
        for _ in range(5):
            _ = model(data)
            p.step()  # 告诉 profiler 进入下一步

    # 恢复模型原来的训练状态
    if is_training:
        model.train()

    # --------------------------
    # 3. 从 profiler 结果中提取 FLOPs
    # --------------------------
    # 获取完整性能表格
    info = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)

    # 用正则提取 FLOPs 数值
    num_flops = sum([float(v.strip()) for v in re.findall('(\d+.?\d+ *\n)', info)]) / num_active

    # --------------------------
    # 4. 打印结果
    # --------------------------
    if verbose:
        print(f'Total number of trainable parameters: {num_params}')
        print(f'Total number of flops: {int(num_flops)}M with {input_shape}')

    # 返回参数量、FLOPs、完整日志
    return {'n_parameters': num_params, 'n_flops': num_flops, 'info': info}