"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import re
import copy

# 继承自基础配置类
from ._config import BaseConfig

# 核心创建函数：根据配置自动实例化对象（模型、优化器、数据加载器等）
from .workspace import create

# YAML 配置文件工具：加载、合并配置
from .yaml_utils import load_config, merge_config, merge_dict


class YAMLConfig(BaseConfig):
    """
    YAML 配置类，继承自 BaseConfig
    功能：从 YAML 文件中自动加载、解析、创建所有训练需要的组件
    包括：model、postprocessor、criterion、optimizer、lr_scheduler、dataloader 等
    """

    def __init__(self, cfg_path: str, **kwargs) -> None:
        """
        初始化 YAML 配置
        :param cfg_path: yaml 配置文件路径
        :param kwargs: 额外的配置参数，用于覆盖 yaml 中的配置
        """
        # 调用父类 BaseConfig 的初始化
        super().__init__()

        # 1. 从 yaml 文件加载原始配置字典
        cfg = load_config(cfg_path)

        # 2. 将传入的 kwargs 合并到配置中（允许命令行/代码动态覆盖配置）
        cfg = merge_dict(cfg, kwargs)

        # 3. 深拷贝一份完整的 yaml 配置，用于后续创建组件
        self.yaml_cfg = copy.deepcopy(cfg)

        # 4. 将 yaml 中的公开配置（非下划线开头）赋值给当前实例属性
        # 例如：epoches, num_workers, use_amp, output_dir 等
        for k in super().__dict__:
            if not k.startswith('_') and k in cfg:
                self.__dict__[k] = cfg[k]

    @property
    def global_cfg(self, ):
        """
        获取全局完整配置（合并后的最终配置）
        用于创建各种组件时传入，保证配置一致性
        """
        return merge_config(self.yaml_cfg, inplace=False, overwrite=False)

    # ==================== 自动创建模型 ====================
    @property
    def model(self, ) -> torch.nn.Module:
        """
        重写父类 model 属性
        功能：如果模型未创建，且 yaml 配置中有 model 字段，则自动创建模型
        """
        if self._model is None and 'model' in self.yaml_cfg:
            self._model = create(self.yaml_cfg['model'], self.global_cfg)
        return super().model

    # ==================== 自动创建后处理 ====================
    @property
    def postprocessor(self, ) -> torch.nn.Module:
        """
        自动创建后处理模块（如 NMS、解码、结果转换）
        """
        if self._postprocessor is None and 'postprocessor' in self.yaml_cfg:
            self._postprocessor = create(self.yaml_cfg['postprocessor'], self.global_cfg)
        return super().postprocessor

    # ==================== 自动创建损失函数 ====================
    @property
    def criterion(self, ) -> torch.nn.Module:
        """
        自动创建损失函数
        """
        if self._criterion is None and 'criterion' in self.yaml_cfg:
            self._criterion = create(self.yaml_cfg['criterion'], self.global_cfg)
        return super().criterion

    # ==================== 自动创建优化器 ====================
    @property
    def optimizer(self, ) -> optim.Optimizer:
        """
        自动创建优化器，并自动处理参数分组（weight decay / 层学习率）
        """
        if self._optimizer is None and 'optimizer' in self.yaml_cfg:
            # 获取优化参数（支持分层学习率、偏置不衰减等高级配置）
            params = self.get_optim_params(self.yaml_cfg['optimizer'], self.model)
            # 创建优化器
            self._optimizer = create('optimizer', self.global_cfg, params=params)
        return super().optimizer

    # ==================== 自动创建学习率调度器 ====================
    @property
    def lr_scheduler(self, ) -> optim.lr_scheduler.LRScheduler:
        """
        自动创建学习率调度器（Step/Cosine/Poly 等）
        """
        if self._lr_scheduler is None and 'lr_scheduler' in self.yaml_cfg:
            self._lr_scheduler = create('lr_scheduler', self.global_cfg, optimizer=self.optimizer)
            print(f'Initial lr: {self._lr_scheduler.get_last_lr()}')
        return super().lr_scheduler

    # ==================== 自动创建学习率预热 ====================
    @property
    def lr_warmup_scheduler(self, ) -> optim.lr_scheduler.LRScheduler:
        """
        自动创建学习率预热调度器
        """
        if self._lr_warmup_scheduler is None and 'lr_warmup_scheduler' in self.yaml_cfg:
            self._lr_warmup_scheduler = create('lr_warmup_scheduler', self.global_cfg, lr_scheduler=self.lr_scheduler)
        return super().lr_warmup_scheduler

    # ==================== 自动创建训练集 DataLoader ====================
    @property
    def train_dataloader(self, ) -> DataLoader:
        if self._train_dataloader is None and 'train_dataloader' in self.yaml_cfg:
            self._train_dataloader = self.build_dataloader('train_dataloader')
        return super().train_dataloader

    # ==================== 自动创建验证集 DataLoader ====================
    @property
    def val_dataloader(self, ) -> DataLoader:
        if self._val_dataloader is None and 'val_dataloader' in self.yaml_cfg:
            self._val_dataloader = self.build_dataloader('val_dataloader')
        return super().val_dataloader

    # ==================== 自动创建 EMA ====================
    @property
    def ema(self, ) -> torch.nn.Module:
        if self._ema is None and self.yaml_cfg.get('use_ema', False):
            self._ema = create('ema', self.global_cfg, model=self.model)
        return super().ema

    # ==================== 自动创建 AMP 混合精度 Scaler ====================
    @property
    def scaler(self, ):
        if self._scaler is None and self.yaml_cfg.get('use_amp', False):
            self._scaler = create('scaler', self.global_cfg)
        return super().scaler

    # ==================== 自动创建评估器 ====================
    @property
    def evaluator(self, ):
        """
        自动创建评估器（目前支持 CocoEvaluator）
        用于验证集指标计算（AP、AR 等）
        """
        if self._evaluator is None and 'evaluator' in self.yaml_cfg:
            # 如果是 COCO 评估器，需要从数据集获取 coco api 实例
            if self.yaml_cfg['evaluator']['type'] == 'CocoEvaluator':
                from ..data import get_coco_api_from_dataset
                base_ds = get_coco_api_from_dataset(self.val_dataloader.dataset)
                self._evaluator = create('evaluator', self.global_cfg, coco_gt=base_ds)
            else:
                raise NotImplementedError(f"{self.yaml_cfg['evaluator']['type']}")
        return super().evaluator

    @staticmethod
    def get_optim_params(cfg: dict, model: nn.Module):
        """
        静态方法：根据正则表达式规则，自动分组模型参数
        用于实现：不同层用不同学习率、weight decay、偏置不衰减等高级优化策略

        支持正则规则示例：
            ^(?=.*a)(?=.*b).*$    包含 a 且 包含 b
            ^(?=.*(?:a|b)).*$     包含 a 或 包含 b
            ^(?=.*a)(?!.*b).*$    包含 a 但 不包含 b
        """
        assert 'type' in cfg, '优化器配置必须包含 type 字段'
        cfg = copy.deepcopy(cfg)

        # 如果没有配置 params，直接返回全部模型参数
        if 'params' not in cfg:
            return model.parameters()

        assert isinstance(cfg['params'], list), 'params 必须是列表格式'

        param_groups = []  # 参数分组
        visited = []       # 记录已经分配过的参数名

        # 遍历每一组参数配置
        for pg in cfg['params']:
            pattern = pg['params']
            # 根据正则匹配参数名
            params = {k: v for k, v in model.named_parameters()
                      if v.requires_grad and len(re.findall(pattern, k)) > 0}

            # 将参数加入当前组
            pg['params'] = params.values()
            param_groups.append(pg)
            visited.extend(list(params.keys()))

        # 收集没有被匹配到的参数，加入默认组
        names = [k for k, v in model.named_parameters() if v.requires_grad]
        if len(visited) < len(names):
            unseen = set(names) - set(visited)
            params = {k: v for k, v in model.named_parameters()
                      if v.requires_grad and k in unseen}
            param_groups.append({'params': params.values()})
            visited.extend(list(params.keys()))

        # 确保所有参数都被分配，没有遗漏
        assert len(visited) == len(names), '存在未分配的参数，请检查正则表达式配置'

        return param_groups

    @staticmethod
    def get_rank_batch_size(cfg):
        """
        静态方法：分布式训练时，自动计算单张显卡的 batch size
        如果配置了 total_batch_size，会自动除以 GPU 数量
        """
        # 二选一：要么配置 batch_size，要么配置 total_batch_size
        assert ('total_batch_size' in cfg or 'batch_size' in cfg) \
               and not ('total_batch_size' in cfg and 'batch_size' in cfg), \
               '请选择配置 batch_size 或 total_batch_size 其中一种'

        total_batch_size = cfg.get('total_batch_size', None)
        if total_batch_size is None:
            bs = cfg.get('batch_size')
        else:
            # 分布式环境：总 batch size / 显卡数 = 单卡 batch size
            from ..misc import dist_utils
            assert total_batch_size % dist_utils.get_world_size() == 0, \
                'total_batch_size 必须能被显卡数量整除'
            bs = total_batch_size // dist_utils.get_world_size()

        return bs

    def build_dataloader(self, name: str):
        """
        根据配置自动构建 DataLoader
        :param name: 'train_dataloader' 或 'val_dataloader'
        """
        # 获取单卡 batch size
        bs = self.get_rank_batch_size(self.yaml_cfg[name])
        global_cfg = self.global_cfg

        # 如果配置了 total_batch_size，需要删除该 key，避免传入 DataLoader 报错
        if 'total_batch_size' in global_cfg[name]:
            _ = global_cfg[name].pop('total_batch_size')

        print(f'building {name} with batch_size={bs}...')

        # 自动创建 DataLoader
        loader = create(name, global_cfg, batch_size=bs)

        # 设置 shuffle
        loader.shuffle = self.yaml_cfg[name].get('shuffle', False)

        return loader