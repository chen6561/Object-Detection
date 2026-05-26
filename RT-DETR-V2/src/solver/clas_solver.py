"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import time 
import json
import datetime
from pathlib import Path

import torch 
import torch.nn as nn 

from ..misc import dist_utils
from ._solver import BaseSolver
# 导入分类任务专用的训练/评估引擎
from .clas_engine import train_one_epoch, evaluate


class ClasSolver(BaseSolver):
    """
    图像分类任务专用训练器
    继承自 BaseSolver，实现 fit() 训练主循环
    """

    def fit(self, ):
        """
        分类模型完整训练入口
        包含：训练 → 验证 → 保存模型 → 记录日志
        """
        print("Start training")
        # 初始化训练环境（模型、优化器、数据加载器等）
        self.train()
        args = self.cfg

        # 统计并打印模型可训练参数量
        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('Number of params:', n_parameters)

        # 确保输出目录存在
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)

        # 记录训练开始时间
        start_time = time.time()
        # 从上次保存的 epoch 继续训练
        start_epoch = self.last_epoch + 1

        # ========== 主训练循环 ==========
        for epoch in range(start_epoch, args.epoches):

            # 分布式训练：设置 sampler 的 epoch，保证数据打乱正确
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            # ========== 训练一个 epoch ==========
            train_stats = train_one_epoch(
                self.model,             # 模型
                self.criterion,         # 损失函数（交叉熵）
                self.train_dataloader,  # 训练数据
                self.optimizer,         # 优化器
                self.ema,               # EMA 模型
                epoch=epoch,            # 当前轮数
                device=self.device      # 设备
            )

            # 更新学习率调度器
            self.lr_scheduler.step()
            # 更新当前已训练的 epoch 数
            self.last_epoch += 1

            # ========== 保存模型 checkpoint ==========
            if output_dir:
                # 总是保存最新模型 checkpoint.pth
                checkpoint_paths = [output_dir / 'checkpoint.pth']
                # 按照配置频率定期保存额外的 checkpoint
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(output_dir / f'checkpoint{epoch:04}.pth')
                # 仅在主进程保存，避免多卡重复写
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(epoch), checkpoint_path)

            # ========== 执行验证 ==========
            # 如果有 EMA，使用 EMA 模型验证，效果更好
            module = self.ema.module if self.ema else self.model
            test_stats = evaluate(module, self.criterion, self.val_dataloader, self.device)

            # 整理日志信息
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            # 主进程将日志写入文件
            if output_dir and dist_utils.is_main_process():
                with (output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        # 训练结束，统计总耗时
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))