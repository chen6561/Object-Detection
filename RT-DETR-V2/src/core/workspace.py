"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import inspect
import importlib
import functools
import inspect
from collections import defaultdict
from typing import Any, Dict, Optional, List


# 全局配置字典，存储所有注册的类/函数及其参数信息
GLOBAL_CONFIG = defaultdict(dict)


def register(dct: Any = GLOBAL_CONFIG, name=None, force=False):
    """
    注册装饰器：用于把 类/函数 注册到全局字典中，方便后续自动创建
    Args:
        dct: 注册目标容器
            - 如果是 dict：将对象以 key-value 形式存入字典
            - 如果是 class：将对象作为类的属性
        name: 注册时使用的名称，默认使用对象本身的 __name__
        force: 是否强制覆盖已注册的对象
    Returns:
        装饰器函数
    """
    def decorator(foo):
        # 确定注册名称：未指定则使用函数/类名
        register_name = foo.__name__ if name is None else name

        # 非强制模式下，检查是否重复注册
        if not force:
            if inspect.isclass(dct):
                # 注册到类：检查类是否已有同名属性
                assert not hasattr(dct, foo.__name__), \
                    f'module {dct.__name__} has {foo.__name__}'
            else:
                # 注册到字典：检查字典是否已有同名key
                assert foo.__name__ not in dct, \
                    f'{foo.__name__} has been already registered'

        # 如果是【函数】，包装后注册
        if inspect.isfunction(foo):
            @functools.wraps(foo)  # 保留原函数元信息（名称、文档等）
            def wrap_func(*args, **kwargs):
                return foo(*args, **kwargs)

            # 存入字典 或 设置为类属性
            if isinstance(dct, dict):
                dct[foo.__name__] = wrap_func
            elif inspect.isclass(dct):
                setattr(dct, foo.__name__, wrap_func)
            else:
                raise AttributeError('不支持的注册容器类型')

            return wrap_func

        # 如果是【类】，提取参数结构后注册
        elif inspect.isclass(foo):
            # 提取类的 __init__ 参数信息，存入全局配置
            dct[register_name] = extract_schema(foo)

        else:
            raise ValueError(f'不支持注册 {type(foo)} 类型对象')

        return foo

    return decorator


def extract_schema(module: type):
    """
    提取类的构造函数参数结构（核心工具函数）
    自动解析 __init__ 方法的：必选参数、默认参数、共享参数、注入参数
    Args:
        module: 要解析的类
    Returns:
        包含参数信息的字典 schema
    """
    # 获取 __init__ 方法的参数信息
    argspec = inspect.getfullargspec(module.__init__)
    # 去掉 self，获取所有参数名
    arg_names = [arg for arg in argspec.args if arg != 'self']
    # 默认参数的数量
    num_defaults = len(argspec.defaults) if argspec.defaults is not None else 0
    # 必选参数数量 = 总参数 - 默认参数
    num_requires = len(arg_names) - num_defaults

    # 构建参数字典
    schema = dict()
    schema['_name'] = module.__name__                # 类名
    schema['_pymodule'] = importlib.import_module(module.__module__)  # 所属模块
    schema['_inject'] = getattr(module, '__inject__', [])  # 需要自动注入的参数
    schema['_share'] = getattr(module, '__share__', [])    # 需要共享的全局参数
    schema['_kwargs'] = {}                               # 所有参数默认值

    # 遍历所有参数，赋值默认值
    for i, name in enumerate(arg_names):
        # 处理共享配置（必须有默认值）
        if name in schema['_share']:
            assert i >= num_requires, 'share config must have default value.'
            value = argspec.defaults[i - num_requires]

        # 处理带默认值的参数
        elif i >= num_requires:
            value = argspec.defaults[i - num_requires]

        # 必选参数，默认值为 None
        else:
            value = None

        schema[name] = value
        schema['_kwargs'][name] = value

    return schema


def create(type_or_name, global_cfg=GLOBAL_CONFIG, **kwargs):
    """
    根据名称/类型 + 配置，**自动创建对象实例**（框架核心函数）
    支持：
        1. 直接传入类名创建
        2. 传入带 type 的配置字典创建
        3. 自动处理参数注入、参数共享
    Args:
        type_or_name: 类对象 或 注册名称字符串
        global_cfg: 全局配置字典
        kwargs: 额外覆盖的参数
    Returns:
        创建好的对象实例
    """
    # 只支持 类 或 字符串 两种输入
    assert type(type_or_name) in (type, str), 'create should be modules or name.'

    # 统一转为名称字符串
    name = type_or_name if isinstance(type_or_name, str) else type_or_name.__name__

    # 检查是否已注册
    if name in global_cfg:
        if hasattr(global_cfg[name], '__dict__'):
            return global_cfg[name]
    else:
        raise ValueError(f'The module {name} is not registered')

    cfg = global_cfg[name]

    # 如果配置是字典且包含 type，说明是嵌套创建（先创建type对应的对象）
    if isinstance(cfg, dict) and 'type' in cfg:
        _cfg: dict = global_cfg[cfg['type']]
        # 清空旧参数，恢复默认值
        _keys = [k for k in _cfg.keys() if not k.startswith('_')]
        for _arg in _keys:
            del _cfg[_arg]

        _cfg.update(_cfg['_kwargs'])    # 恢复默认参数
        _cfg.update(cfg)               # 覆盖用户配置参数
        _cfg.update(kwargs)            # 覆盖传入的额外参数
        name = _cfg.pop('type')        # 取出 type 字段作为真正要创建的类名

        return create(name, global_cfg)

    # 获取模块和类
    module = getattr(cfg['_pymodule'], name)
    module_kwargs = {}
    module_kwargs.update(cfg)

    # ==================== 处理共享参数（从全局配置读取） ====================
    for k in cfg['_share']:
        if k in global_cfg:
            module_kwargs[k] = global_cfg[k]
        else:
            module_kwargs[k] = cfg[k]

    # ==================== 处理参数注入（自动创建依赖对象） ====================
    for k in cfg['_inject']:
        _k = cfg[k]

        if _k is None:
            continue

        # 注入方式1：字符串（对应已注册的模块名）
        if isinstance(_k, str):
            if _k not in global_cfg:
                raise ValueError(f'Missing inject config of {_k}.')

            _cfg = global_cfg[_k]

            if isinstance(_cfg, dict):
                module_kwargs[k] = create(_cfg['_name'], global_cfg)
            else:
                module_kwargs[k] = _cfg

        # 注入方式2：字典（包含 type，动态创建）
        elif isinstance(_k, dict):
            if 'type' not in _k.keys():
                raise ValueError(f'Missing inject for `type` style.')

            _type = str(_k['type'])
            if _type not in global_cfg:
                raise ValueError(f'Missing {_type} in inspect stage.')

            # 清理并重建参数
            _cfg: dict = global_cfg[_type]
            _keys = [k for k in _cfg.keys() if not k.startswith('_')]
            for _arg in _keys:
                del _cfg[_arg]

            _cfg.update(_cfg['_kwargs'])
            _cfg.update(_k)
            name = _cfg.pop('type')
            module_kwargs[k] = create(name, global_cfg)

        else:
            raise ValueError(f'Inject does not support {_k}')

    # 过滤掉内部以下划线开头的参数
    module_kwargs = {k: v for k, v in module_kwargs.items() if not k.startswith('_')}

    # 最终创建实例
    return module(**module_kwargs)