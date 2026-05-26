"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import time
import json
import datetime

import torch

from ..misc import dist_utils, profiler_utils

from ._solver import BaseSolver
from .det_engine import train_one_epoch, evaluate


class DetSolver(BaseSolver):
    """
    目标检测专用训练器
    继承 BaseSolver，实现完整的 fit() 训练 + val() 验证逻辑
    """

    def fit(self, ):
        """
        完整训练主循环入口
        包含：训练 → 验证 → 保存模型 → 记录日志
        """
        print("Start training")
        # 初始化训练环境（模型、优化器、dataloader等）
        self.train()
        args = self.cfg

        ####################################################
        # 方法一：用 torchviz 画出网络结构图（最直观）
        ####################################################
        # 画出网络结构图
        # from torchviz import make_dot
        # import torch
        #
        # x = torch.randn(1, 3, 640, 640).cuda()  # 造一个假输入
        #
        # # 前向传播（不需要标签，只看结构）
        # self.model.eval()
        # with torch.no_grad():  # 绘图不需要梯度
        #     y = self.model(x)
        #
        # # 拿出一个 Tensor ！！！关键在这里
        # if isinstance(y, dict):
        #     y = y["pred_logits"]  # 随便拿一个输出 tensor
        #
        # # 画图
        # make_dot(y, params=dict(self.model.named_parameters())).render("model_struct", format="png")
        ####################################################
        # 方法一：用 torchviz 画出网络结构图（最直观）
        ####################################################

        ####################################################
        # 方法二：用torchsummary 可视化
        ####################################################
        from torchinfo import summary  # 用于打印复杂的网络结构

        self.model.eval()

        # 正确、稳定、支持 RT-DETR
        summary(
            self.model,
            input_size=(1, 3, 640, 640),  # batch, channel, H, W
            device="cuda" if next(self.model.parameters()).is_cuda else "cpu",
            mode="eval"
        )
        ####################################################
        # 方法二：用torchsummary 可视化
        ####################################################

        # 统计可训练参数量
        n_parameters = sum([p.numel() for p in self.model.parameters() if p.requires_grad])
        print(f'number of trainable parameters: {n_parameters}')

        # 记录最佳指标（初始值）
        best_stat = {'epoch': -1, }

        # 训练开始时间
        start_time = time.time()
        start_epcoch = self.last_epoch + 1

        # ============== 主训练循环 ==============
        for epoch in range(start_epcoch, args.epoches):

            # 设置当前 epoch（用于分布式 sampler 打乱数据）
            self.train_dataloader.set_epoch(epoch)
            if dist_utils.is_dist_available_and_initialized():
                self.train_dataloader.sampler.set_epoch(epoch)

            # ============== 训练一个 epoch ==============
            train_stats = train_one_epoch(
                self.model,
                self.criterion,
                self.train_dataloader,
                self.optimizer,
                self.device,
                epoch,
                max_norm=args.clip_max_norm,       # 梯度裁剪
                print_freq=args.print_freq,       # 打印频率
                ema=self.ema,                     # EMA 模型
                scaler=self.scaler,               # 混合精度
                lr_warmup_scheduler=self.lr_warmup_scheduler,
                writer=self.writer                # TensorBoard 日志
            )

            # 学习率调度（warmup 结束后才更新）
            if self.lr_warmup_scheduler is None or self.lr_warmup_scheduler.finished():
                self.lr_scheduler.step()

            # 更新当前 epoch
            self.last_epoch += 1

            # ============== 保存 checkpoint ==============
            if self.output_dir:
                # 总是保存 last.pth
                checkpoint_paths = [self.output_dir / 'last.pth']

                # 每隔 N 轮额外保存一个 checkpoint
                if (epoch + 1) % args.checkpoint_freq == 0:
                    checkpoint_paths.append(self.output_dir / f'checkpoint{epoch:04}.pth')

                # 主进程保存
                for checkpoint_path in checkpoint_paths:
                    dist_utils.save_on_master(self.state_dict(), checkpoint_path)

            # ============== 执行验证 ==============
            # 使用 EMA 模型（如果有）
            module = self.ema.module if self.ema else self.model
            test_stats, coco_evaluator = evaluate(
                module,
                self.criterion,
                self.postprocessor,
                self.val_dataloader,
                self.evaluator,
                self.device
            )

            # ============== 记录最佳模型 ==============
            for k in test_stats:
                # 写入 TensorBoard
                if self.writer and dist_utils.is_main_process():
                    for i, v in enumerate(test_stats[k]):
                        self.writer.add_scalar(f'Test/{k}_{i}', v, epoch)

                # 更新最佳指标
                if k in best_stat:
                    best_stat['epoch'] = epoch if test_stats[k][0] > best_stat[k] else best_stat['epoch']
                    best_stat[k] = max(best_stat[k], test_stats[k][0])
                else:
                    best_stat['epoch'] = epoch
                    best_stat[k] = test_stats[k][0]

                # 如果当前轮是最佳，保存 best.pth
                if best_stat['epoch'] == epoch and self.output_dir:
                    dist_utils.save_on_master(self.state_dict(), self.output_dir / 'best.pth')

            print(f'best_stat: {best_stat}')

            # ============== 保存日志 ==============
            log_stats = {
                **{f'train_{k}': v for k, v in train_stats.items()},
                **{f'test_{k}': v for k, v in test_stats.items()},
                'epoch': epoch,
                'n_parameters': n_parameters
            }

            # 主进程写入 log.txt
            if self.output_dir and dist_utils.is_main_process():
                with (self.output_dir / "log.txt").open("a") as f:
                    f.write(json.dumps(log_stats) + "\n")

                # 保存 COCO 评估结果
                if coco_evaluator is not None:
                    (self.output_dir / 'eval').mkdir(exist_ok=True)
                    if "bbox" in coco_evaluator.coco_eval:
                        filenames = ['latest.pth']
                        if epoch % 50 == 0:
                            filenames.append(f'{epoch:03}.pth')
                        for name in filenames:
                            torch.save(coco_evaluator.coco_eval["bbox"].eval,
                                    self.output_dir / "eval" / name)

        # ============== 训练结束，统计总时间 ==============
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Training time {}'.format(total_time_str))

    def val(self, ):
        """
        单独验证/测试入口（只跑验证集，不训练）
        """
        self.eval()

        # 使用 EMA 模型
        module = self.ema.module if self.ema else self.model
        test_stats, coco_evaluator = evaluate(module, self.criterion, self.postprocessor,
                self.val_dataloader, self.evaluator, self.device)

        # 保存验证结果
        if self.output_dir:
            dist_utils.save_on_master(coco_evaluator.coco_eval["bbox"].eval, self.output_dir / "eval.pth")

        return