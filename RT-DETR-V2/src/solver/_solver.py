"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn

from datetime import datetime
from pathlib import Path
from typing import Dict
import atexit

from ..misc import dist_utils
from ..core import BaseConfig


def to(m: nn.Module, device: str):
    """把模块移动到指定设备，兼容 None 输入"""
    if m is None:
        return None
    return m.to(device)


class BaseSolver(object):
    """
    训练/验证 基类（Solver = 求解器）
    所有训练器都继承这个类，统一管理：
    模型、优化器、EMA、分布式、保存、恢复、初始化、验证
    """
    def __init__(self, cfg: BaseConfig) -> None:
        # 保存配置
        self.cfg = cfg

    def _setup(self, ):
        """
        核心初始化函数
        避免提前创建不必要的对象，延迟加载
        """
        cfg = self.cfg

        # 设置设备：cuda / cpu
        if cfg.device:
            device = torch.device(cfg.device)
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 取出模型
        self.model = cfg.model
        # --------------------------
        # Tuning 模式：加载预训练权重（不加载优化器等）
        # --------------------------
        if self.cfg.tuning:
            print(f'tuning checkpoint from {self.cfg.tuning}')
            self.load_tuning_state(self.cfg.tuning)

        # --------------------------
        # 包装模型：分布式 DDP / DP + SyncBN
        # --------------------------
        self.model = dist_utils.warp_model(
            self.model.to(device),
            sync_bn=cfg.sync_bn,
            find_unused_parameters=cfg.find_unused_parameters
        )

        # 损失函数、后处理、EMA、混合精度 scaler
        self.criterion = to(cfg.criterion, device)
        self.postprocessor = to(cfg.postprocessor, device)
        self.ema = to(cfg.ema, device)
        self.scaler = cfg.scaler

        # 设备 & 最后一轮 epoch
        self.device = device
        self.last_epoch = self.cfg.last_epoch

        # 输出目录
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 日志 writer（TensorBoard）
        self.writer = cfg.writer
        if self.writer:
            atexit.register(self.writer.close)
            if dist_utils.is_main_process():
                self.writer.add_text(f'config', '{:s}'.format(cfg.__repr__()), 0)

    def cleanup(self, ):
        """关闭 writer"""
        if self.writer:
            atexit.register(self.writer.close)

    def train(self, ):
        """
        训练模式初始化
        构建：优化器、lr调度、dataloader、评估器
        恢复 resume 权重
        """
        self._setup()

        # 优化器 & 学习率调度
        self.optimizer = self.cfg.optimizer
        self.lr_scheduler = self.cfg.lr_scheduler
        self.lr_warmup_scheduler = self.cfg.lr_warmup_scheduler

        # 分布式包装 dataloader
        self.train_dataloader = dist_utils.warp_loader(
            self.cfg.train_dataloader,
            shuffle=self.cfg.train_dataloader.shuffle
        )
        self.val_dataloader = dist_utils.warp_loader(
            self.cfg.val_dataloader,
            shuffle=self.cfg.val_dataloader.shuffle
        )

        # 评估器
        self.evaluator = self.cfg.evaluator

        # 恢复训练
        if self.cfg.resume:
            print(f'Resume checkpoint from {self.cfg.resume}')
            self.load_resume_state(self.cfg.resume)

    def eval(self, ):
        """
        验证/测试模式初始化
        """
        self._setup()

        # 分布式 dataloader
        self.val_dataloader = dist_utils.warp_loader(
            self.cfg.val_dataloader,
            shuffle=self.cfg.val_dataloader.shuffle
        )

        self.evaluator = self.cfg.evaluator

        # 加载权重
        if self.cfg.resume:
            print(f'Resume checkpoint from {self.cfg.resume}')
            self.load_resume_state(self.cfg.resume)

    def to(self, device):
        """把所有带 .to() 方法的成员移动到设备"""
        for k, v in self.__dict__.items():
            if hasattr(v, 'to'):
                v.to(device)

    def state_dict(self):
        """
        获取所有可保存的状态
        用于保存 checkpoint：模型、EMA、优化器、lr_scheduler...
        """
        state = {}
        state['date'] = datetime.now().isoformat()
        state['last_epoch'] = self.last_epoch

        # 保存所有带 state_dict 的对象
        for k, v in self.__dict__.items():
            if hasattr(v, 'state_dict'):
                v = dist_utils.de_parallel(v)
                state[k] = v.state_dict()

        return state

    def load_state_dict(self, state):
        """
        加载状态字典
        恢复模型、EMA、优化器、lr_scheduler...
        """
        if 'last_epoch' in state:
            self.last_epoch = state['last_epoch']
            print('Load last_epoch')

        for k, v in self.__dict__.items():
            if hasattr(v, 'load_state_dict') and k in state:
                v = dist_utils.de_parallel(v)
                v.load_state_dict(state[k])
                print(f'Load {k}.state_dict')

            if hasattr(v, 'load_state_dict') and k not in state:
                print(f'Not load {k}.state_dict')

    def load_resume_state(self, path: str):
        """
        完整恢复训练（模型 + EMA + 优化器 + 调度器）
        """
        if path.startswith('http'):
            state = torch.hub.load_state_dict_from_url(path, map_location='cpu')
        else:
            state = torch.load(path, map_location='cpu')

        self.load_state_dict(state)

    def load_tuning_state(self, path: str,):
        """
        只加载模型权重（用于微调 tuning）
        自动跳过 shape 不匹配的层
        """
        if path.startswith('http'):
            state = torch.hub.load_state_dict_from_url(path, map_location='cpu')
        else:
            state = torch.load(path, map_location='cpu')

        # 去掉 DDP 包装
        module = dist_utils.de_parallel(self.model)

        # 加载模型权重（优先 EMA）
        if 'ema' in state:
            stat, infos = self._matched_state(module.state_dict(), state['ema']['module'])
        else:
            stat, infos = self._matched_state(module.state_dict(), state['model'])

        module.load_state_dict(stat, strict=False)
        print(f'Load model.state_dict, {infos}')

    @staticmethod
    def _matched_state(state: Dict[str, torch.Tensor], params: Dict[str, torch.Tensor]):
        """
        权重匹配工具：只加载 shape 相同的权重
        返回：匹配成功的权重 + 缺失/不匹配信息
        """
        missed_list = []
        unmatched_list = []
        matched_state = {}

        for k, v in state.items():
            if k in params:
                if v.shape == params[k].shape:
                    matched_state[k] = params[k]
                else:
                    unmatched_list.append(k)
            else:
                missed_list.append(k)

        return matched_state, {'missed': missed_list, 'unmatched': unmatched_list}

    def fit(self, ):
        """训练入口，子类实现"""
        raise NotImplementedError('')

    def val(self, ):
        """验证入口，子类实现"""
        raise NotImplementedError('')