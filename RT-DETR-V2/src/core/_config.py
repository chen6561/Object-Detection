"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.tensorboard import SummaryWriter

from pathlib import Path
from typing import Callable, List, Dict

# 暴露给外部导入的类名
__all__ = ['BaseConfig', ]


class BaseConfig(object):
    """
    深度学习训练配置基类
    统一管理：模型、数据集、优化器、训练参数、运行时配置
    采用 Python property 装饰器实现安全的属性访问与类型校验
    """
    # TODO 待完善：可扩展更多属性校验与自动初始化逻辑

    def __init__(self) -> None:
        """
        初始化所有配置项，设置默认值
        分为三大类：模型组件、数据集参数、训练运行时参数
        """
        super().__init__()

        # ==================== 1. 任务基本设置 ====================
        # 任务名称，如 detection, classification, segmentation
        self.task: str = None

        # ==================== 2. 模型核心组件（私有变量，通过 property 访问） ====================
        # 模型网络（私有，防止外部直接修改）
        self._model: nn.Module = None
        # 后处理模块（如 NMS、解码、结果格式化）
        self._postprocessor: nn.Module = None
        # 损失函数
        self._criterion: nn.Module = None
        # 优化器
        self._optimizer: Optimizer = None
        # 学习率调度器
        self._lr_scheduler: LRScheduler = None
        # 学习率预热调度器
        self._lr_warmup_scheduler: LRScheduler = None
        # 训练数据加载器
        self._train_dataloader: DataLoader = None
        # 验证数据加载器
        self._val_dataloader: DataLoader = None
        # 指数移动平均模型（EMA）
        self._ema: nn.Module = None
        # 自动混合精度梯度缩放器
        self._scaler: GradScaler = None
        # 训练数据集
        self._train_dataset: Dataset = None
        # 验证数据集
        self._val_dataset: Dataset = None
        # 数据拼接函数（处理 batch 数据）
        self._collate_fn: Callable = None
        # 模型评估函数
        self._evaluator: Callable[[nn.Module, DataLoader, str], ] = None
        # TensorBoard 日志写入器
        self._writer: SummaryWriter = None

        # ==================== 3. 数据集加载参数 ====================
        # DataLoader 加载线程数
        self.num_workers: int = 0
        # 通用 batch size（训练/验证可分别覆盖）
        self.batch_size: int = None
        # 训练 batch size
        self._train_batch_size: int = None
        # 验证 batch size
        self._val_batch_size: int = None
        # 训练集是否打乱
        self._train_shuffle: bool = None
        # 验证集是否打乱
        self._val_shuffle: bool = None

        # ==================== 4. 训练运行时参数 ====================
        # 恢复训练的权重路径
        self.resume: str = None
        # 微调模型的路径
        self.tuning: str = None

        # 总训练轮数
        self.epoches: int = None
        # 最后一轮的轮数（用于恢复训练）
        self.last_epoch: int = -1

        # 是否使用自动混合精度训练
        self.use_amp: bool = False
        # 是否使用 EMA
        self.use_ema: bool = False
        # EMA 衰减系数
        self.ema_decay: float = 0.9999
        # EMA 预热步数
        self.ema_warmups: int = 2000
        # 是否使用同步 BN（多卡训练）
        self.sync_bn: bool = False
        # 梯度裁剪最大范数
        self.clip_max_norm: float = 0.

        # DDP 模式下是否查找未使用参数
        self.find_unused_parameters: bool = None

        # 随机种子
        self.seed: int = None
        # 日志打印频率（步）
        self.print_freq: int = None
        # 权重保存频率（轮）
        self.checkpoint_freq: int = 1
        # 输出目录（保存权重、日志）
        self.output_dir: str = None
        # TensorBoard 日志目录
        self.summary_dir: str = None
        # 运行设备（cuda / cpu）
        self.device: str = ''

    # ==================== Property 装饰器：安全获取/设置属性 + 类型校验 ====================
    # ==================== Property 装饰器：把一个 方法 → 变成 可以像属性一样访问的东西 ========
    @property
    def model(self, ) -> nn.Module:
        """获取模型"""
        return self._model

    @model.setter
    def model(self, m):
        """设置模型，强制校验类型"""
        assert isinstance(m, nn.Module), f'{type(m)} != nn.Module, please check your model class'
        self._model = m

    @property
    def postprocessor(self, ) -> nn.Module:
        """获取后处理模块"""
        return self._postprocessor

    @postprocessor.setter
    def postprocessor(self, m):
        assert isinstance(m, nn.Module), f'{type(m)} != nn.Module, please check your postprocessor'
        self._postprocessor = m

    @property
    def criterion(self, ) -> nn.Module:
        """获取损失函数"""
        return self._criterion

    @criterion.setter
    def criterion(self, m):
        assert isinstance(m, nn.Module), f'{type(m)} != nn.Module, please check your criterion'
        self._criterion = m

    @property
    def optimizer(self, ) -> Optimizer:
        """获取优化器"""
        return self._optimizer

    @optimizer.setter
    def optimizer(self, m):
        assert isinstance(m, Optimizer), f'{type(m)} != optim.Optimizer, check your optimizer'
        self._optimizer = m

    @property
    def lr_scheduler(self, ) -> LRScheduler:
        """获取学习率调度器"""
        return self._lr_scheduler

    @lr_scheduler.setter
    def lr_scheduler(self, m):
        assert isinstance(m, LRScheduler), f'{type(m)} != LRScheduler, check your lr_scheduler'
        self._lr_scheduler = m

    @property
    def lr_warmup_scheduler(self, ) -> LRScheduler:
        """获取学习率预热调度器"""
        return self._lr_warmup_scheduler

    @lr_warmup_scheduler.setter
    def lr_warmup_scheduler(self, m):
        self._lr_warmup_scheduler = m

    # ==================== 训练集 DataLoader 自动构建 ====================
    @property
    def train_dataloader(self) -> DataLoader:
        """
        自动构建训练 DataLoader
        如果未手动设置，则根据 dataset、batch_size 自动创建
        """
        if self._train_dataloader is None and self.train_dataset is not None:
            loader = DataLoader(
                self.train_dataset,
                batch_size=self.train_batch_size,
                num_workers=self.num_workers,
                collate_fn=self.collate_fn,
                shuffle=self.train_shuffle,
            )
            loader.shuffle = self.train_shuffle
            self._train_dataloader = loader

        return self._train_dataloader

    @train_dataloader.setter
    def train_dataloader(self, loader):
        """手动设置训练 DataLoader"""
        self._train_dataloader = loader

    # ==================== 验证集 DataLoader 自动构建 ====================
    @property
    def val_dataloader(self) -> DataLoader:
        """
        自动构建验证 DataLoader
        验证集默认不丢弃最后一个不完整 batch
        """
        if self._val_dataloader is None and self.val_dataset is not None:
            loader = DataLoader(
                self.val_dataset,
                batch_size=self.val_batch_size,
                num_workers=self.num_workers,
                drop_last=False,
                collate_fn=self.collate_fn,
                shuffle=self.val_shuffle
            )
            loader.shuffle = self.val_shuffle
            self._val_dataloader = loader

        return self._val_dataloader

    @val_dataloader.setter
    def val_dataloader(self, loader):
        """手动设置验证 DataLoader"""
        self._val_dataloader = loader

    # ==================== EMA 模型自动构建 ====================
    @property
    def ema(self, ) -> nn.Module:
        """
        自动初始化 EMA
        只有开启 use_ema 且模型存在时才创建
        """
        if self._ema is None and self.use_ema and self.model is not None:
            from ..optim import ModelEMA
            self._ema = ModelEMA(self.model, self.ema_decay, self.ema_warmups)
        return self._ema

    @ema.setter
    def ema(self, obj):
        self._ema = obj

    # ==================== AMP 梯度缩放器自动构建 ====================
    @property
    def scaler(self) -> GradScaler:
        """
        自动构建混合精度梯度缩放器
        只有开启 use_amp 且 CUDA 可用时才创建
        """
        if self._scaler is None and self.use_amp and torch.cuda.is_available():
            self._scaler = GradScaler()
        return self._scaler

    @scaler.setter
    def scaler(self, obj: GradScaler):
        self._scaler = obj

    # ==================== 验证集 shuffle 控制 ====================
    @property
    def val_shuffle(self) -> bool:
        """验证集默认不打乱数据"""
        if self._val_shuffle is None:
            print('warning: set default val_shuffle=False')
            return False
        return self._val_shuffle

    @val_shuffle.setter
    def val_shuffle(self, shuffle):
        assert isinstance(shuffle, bool), 'shuffle must be bool'
        self._val_shuffle = shuffle

    # ==================== 训练集 shuffle 控制 ====================
    @property
    def train_shuffle(self) -> bool:
        """训练集默认打乱数据"""
        if self._train_shuffle is None:
            print('warning: set default train_shuffle=True')
            return True
        return self._train_shuffle

    @train_shuffle.setter
    def train_shuffle(self, shuffle):
        assert isinstance(shuffle, bool), 'shuffle must be bool'
        self._train_shuffle = shuffle

    # ==================== 训练 batch_size 自动设置 ====================
    @property
    def train_batch_size(self) -> int:
        """
        训练 batch_size 优先级：
        手动设置 train_batch_size > 通用 batch_size
        """
        if self._train_batch_size is None and isinstance(self.batch_size, int):
            print(f'warning: set train_batch_size=batch_size={self.batch_size}')
            return self.batch_size
        return self._train_batch_size

    @train_batch_size.setter
    def train_batch_size(self, batch_size):
        assert isinstance(batch_size, int), 'batch_size must be int'
        self._train_batch_size = batch_size

    # ==================== 验证 batch_size 自动设置 ====================
    @property
    def val_batch_size(self) -> int:
        """
        验证 batch_size 优先级：
        手动设置 val_batch_size > 通用 batch_size
        """
        if self._val_batch_size is None:
            print(f'warning: set val_batch_size=batch_size={self.batch_size}')
            return self.batch_size
        return self._val_batch_size

    @val_batch_size.setter
    def val_batch_size(self, batch_size):
        assert isinstance(batch_size, int), 'batch_size must be int'
        self._val_batch_size = batch_size

    # ==================== 训练数据集 ====================
    @property
    def train_dataset(self) -> Dataset:
        return self._train_dataset

    @train_dataset.setter
    def train_dataset(self, dataset):
        assert isinstance(dataset, Dataset), f'{type(dataset)} must be Dataset'
        self._train_dataset = dataset

    # ==================== 验证数据集 ====================
    @property
    def val_dataset(self) -> Dataset:
        return self._val_dataset

    @val_dataset.setter
    def val_dataset(self, dataset):
        assert isinstance(dataset, Dataset), f'{type(dataset)} must be Dataset'
        self._val_dataset = dataset

    # ==================== 数据拼接函数 ====================
    @property
    def collate_fn(self) -> Callable:
        return self._collate_fn

    @collate_fn.setter
    def collate_fn(self, fn):
        assert isinstance(fn, Callable), f'{type(fn)} must be Callable'
        self._collate_fn = fn

    # ==================== 评估器 ====================
    @property
    def evaluator(self) -> Callable:
        return self._evaluator

    @evaluator.setter
    def evaluator(self, fn):
        assert isinstance(fn, Callable), f'{type(fn)} must be Callable'
        self._evaluator = fn

    # ==================== TensorBoard 写入器自动构建 ====================
    @property
    def writer(self) -> SummaryWriter:
        """
        自动构建 SummaryWriter
        优先使用 summary_dir，否则使用 output_dir/summary
        """
        if self._writer is None:
            if self.summary_dir:
                self._writer = SummaryWriter(self.summary_dir)
            elif self.output_dir:
                self._writer = SummaryWriter(Path(self.output_dir) / 'summary')
        return self._writer

    @writer.setter
    def writer(self, m):
        assert isinstance(m, SummaryWriter), f'{type(m)} must be SummaryWriter'
        self._writer = m

    # ==================== 打印配置信息 ====================
    def __repr__(self, ):
        """
        打印当前配置类的所有公开参数
        方便调试、查看配置是否正确
        """
        s = ''
        for k, v in self.__dict__.items():
            if not k.startswith('_'):
                s += f'{k}: {v}\n'
        return s