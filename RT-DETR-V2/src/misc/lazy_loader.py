"""
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/util/lazy_loader.py
"""

import types
import importlib

class LazyLoader(types.ModuleType):
  """
  延迟加载（懒加载）模块
  作用：避免一开始就导入巨大的依赖库，只有在真正使用时才加载
  例如：paddle、ffmpeg 这种很大、不一定每次都用到的库
  """

  def __init__(self, local_name, parent_module_globals, name, warning=None):
    """
    初始化 LazyLoader
    Args:
      local_name: 本地要使用的变量名（比如 nn）
      parent_module_globals: 父模块的全局变量字典（globals()）
      name: 真正要导入的模块名（比如 "paddle.nn"）
      warning: 加载时是否打印提示
    """
    self._local_name = local_name                # 本地变量名
    self._parent_module_globals = parent_module_globals  # 全局作用域
    self._warning = warning                     # 提示信息

    # 以下两个属性是为了让 doctest 能正常识别，不会提前触发加载
    self.__module__ = name.rsplit(".", 1)[0]    # 模块名
    self.__wrapped__ = None                     # 包装标记

    super(LazyLoader, self).__init__(name)

  def _load(self):
    """
    真正加载模块，并把模块插入到全局变量中
    """
    # 真正导入模块
    module = importlib.import_module(self.__name__)

    # 把真实模块替换到全局作用域
    self._parent_module_globals[self._local_name] = module

    # 如果设置了警告，只打印一次
    if self._warning:
      self._warning = None

    # 把真实模块的所有属性复制到当前对象
    # 这样后续访问属性就不会再触发 __getattr__
    self.__dict__.update(module.__dict__)

    return module

  def __getattr__(self, item):
    """
    当访问属性时才触发加载
    """
    module = self._load()
    return getattr(module, item)

  def __repr__(self):
    """
    打印对象时不触发加载，避免意外加载
    """
    return f"<LazyLoader {self.__name__} as {self._local_name}>"

  def __dir__(self):
    """
    使用 dir() 时才触发加载
    """
    module = self._load()
    return dir(module)


# 使用示例（注释掉的代码）
# import paddle.nn as nn
# nn = LazyLoader("nn", globals(), "paddle.nn")

# class M(nn.Layer):
#     def __init__(self) -> None:
#       super().__init__()