"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import copy
import yaml
from typing import Any, Dict, Optional, List

# 导入全局注册配置（来自之前的 register/create 系统）
from .workspace import GLOBAL_CONFIG

# 对外暴露的接口函数
__all__ = [
    'load_config',
    'merge_config',
    'merge_dict',
    'parse_cli',
]

# YAML 配置里用于导入其他配置的关键字
INCLUDE_KEY = '__include__'


def load_config(file_path, cfg=dict()):
    """
    递归加载 YAML 配置文件（支持 __include__ 导入其他 yaml）
    Args:
        file_path: 主配置文件路径
        cfg: 用于递归合并的配置字典
    Return:
        合并完成后的完整配置字典
    """
    # 检查文件后缀是否合法
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files"

    # 打开并读取当前 yaml 文件
    with open(file_path) as f:
        file_cfg = yaml.load(f, Loader=yaml.Loader)
        if file_cfg is None:
            return {}

    # 如果当前配置里有 __include__ 字段，递归加载基础配置
    if INCLUDE_KEY in file_cfg:
        base_yamls = list(file_cfg[INCLUDE_KEY])
        for base_yaml in base_yamls:
            # 处理 ~ 路径
            if base_yaml.startswith('~'):
                base_yaml = os.path.expanduser(base_yaml)

            # 如果是相对路径，拼接成绝对路径
            if not base_yaml.startswith('/'):
                base_yaml = os.path.join(os.path.dirname(file_path), base_yaml)

            # 递归加载被 include 的基础配置
            with open(base_yaml) as f:
                base_cfg = load_config(base_yaml, cfg)
                merge_dict(cfg, base_cfg)

    # 把当前文件配置合并到总 cfg 并返回
    return merge_dict(cfg, file_cfg)


def merge_dict(dct, another_dct, inplace=True) -> Dict:
    """
    将 another_dct 合并到 dct 中（字典嵌套递归合并）
    Args:
        dct: 原始字典
        another_dct: 要合并进来的字典
        inplace: 是否直接修改原字典
    Return:
        合并后的字典
    """
    def _merge(dct, another) -> Dict:
        for k in another:
            # 如果两个 key 对应的 value 都是字典，递归合并
            if (k in dct and isinstance(dct[k], dict) and isinstance(another[k], dict)):
                _merge(dct[k], another[k])
            else:
                # 否则直接覆盖
                dct[k] = another[k]

        return dct

    # 非原地合并则先深拷贝
    if not inplace:
        dct = copy.deepcopy(dct)

    return _merge(dct, another_dct)


def dictify(s: str, v: Any) -> Dict:
    """
    把 a.b.c=3 这种字符串转成嵌套字典 {a:{b:{c:3}}}
    """
    if '.' not in s:
        return {s: v}

    # 按第一个 . 分割
    key, rest = s.split('.', 1)
    return {key: dictify(rest, v)}


def parse_cli(nargs: List[str]) -> Dict:
    """
    解析命令行参数
    例如：a.c=3 b=10 → {'a': {'c': 3}, 'b': 10}
    """
    cfg = {}
    if nargs is None or len(nargs) == 0:
        return cfg

    for s in nargs:
        s = s.strip()
        # 按第一个 = 分割 key 和 value
        k, v = s.split('=', 1)
        # 把字符串转成字典
        d = dictify(k, yaml.load(v, Loader=yaml.Loader))
        # 合并到总配置
        cfg = merge_dict(cfg, d)

    return cfg


def merge_config(cfg, another_cfg=GLOBAL_CONFIG, inplace: bool=False, overwrite: bool=False):
    """
    把 another_cfg（默认是全局注册配置）合并到用户 cfg
    用于把注册器信息注入到配置里，供 create 函数使用

    典型用法：
        cfg = load_config('xxx.yaml')
        cfg = merge_config(cfg, inplace=True)
        model = create(cfg['model'], cfg)
    """
    def _merge(dct, another):
        for k in another:
            # 如果 key 不存在，直接加入
            if k not in dct:
                dct[k] = another[k]

            # 如果都是字典，递归合并
            elif isinstance(dct[k], dict) and isinstance(another[k], dict):
                _merge(dct[k], another[k])

            # 如果开启 overwrite，则覆盖已有值
            elif overwrite:
                dct[k] = another[k]

        return cfg

    # 非原地合并 → 深拷贝
    if not inplace:
        cfg = copy.deepcopy(cfg)

    return _merge(cfg, another_cfg)