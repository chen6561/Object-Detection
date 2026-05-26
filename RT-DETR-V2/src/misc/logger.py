"""
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
https://github.com/facebookresearch/detr/blob/main/util/misc.py
Mostly copy-paste from torchvision references.
"""

import time
import pickle
import datetime
from collections import defaultdict, deque
from typing import Dict

import torch
import torch.distributed as tdist

from .dist_utils import is_dist_available_and_initialized, get_world_size


class SmoothedValue(object):
    """
    跟踪一系列数值，提供滑动窗口平滑值 / 全局平均值
    用于训练中平滑显示 loss、lr、accuracy 等指标
    """

    def __init__(self, window_size=20, fmt=None):
        # 输出格式
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"

        # 双端队列，只保存最近 window_size 个值（滑动窗口）
        self.deque = deque(maxlen=window_size)
        self.total = 0.0    # 全局总值
        self.count = 0      # 全局计数
        self.fmt = fmt

    def update(self, value, n=1):
        """更新数值"""
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        分布式多卡同步 count 和 total（不同步 deque 窗口）
        """
        if not is_dist_available_and_initialized():
            return

        # 把 count 和 total 转成 tensor 用于多卡聚合
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        tdist.barrier()
        tdist.all_reduce(t)  # 多卡求和
        t = t.tolist()

        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        """窗口中位数"""
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        """窗口平均值"""
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        """全局平均值（整个训练过程）"""
        return self.total / self.count

    @property
    def max(self):
        """窗口最大值"""
        return max(self.deque)

    @property
    def value(self):
        """最新值"""
        return self.deque[-1]

    def __str__(self):
        """格式化输出"""
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


def all_gather(data):
    """
    任意可序列化数据的多卡全收集（不限于 tensor）
    常用于收集各卡预测结果、指标等
    """
    world_size = get_world_size()
    if world_size == 1:
        return [data]

    # 1. 把数据序列化 → 字节 → byte tensor
    buffer = pickle.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to("cuda")

    # 2. 收集各卡 tensor 长度
    local_size = torch.tensor([tensor.numel()], device="cuda")
    size_list = [torch.tensor([0], device="cuda") for _ in range(world_size)]
    tdist.all_gather(size_list, local_size)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    # 3. 统一补齐到最大长度，执行 all_gather
    tensor_list = []
    for _ in size_list:
        tensor_list.append(torch.empty((max_size,), dtype=torch.uint8, device="cuda"))

    if local_size != max_size:
        padding = torch.empty(size=(max_size - local_size,), dtype=torch.uint8, device="cuda")
        tensor = torch.cat((tensor, padding), dim=0)

    tdist.all_gather(tensor_list, tensor)

    # 4. 反序列化恢复数据
    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))

    return data_list


def reduce_dict(input_dict, average=True) -> Dict[str, torch.Tensor]:
    """
    分布式多卡聚合字典中的所有值（loss 字典最常用）
    默认求平均，也可以求和
    """
    world_size = get_world_size()
    if world_size < 2:
        return input_dict

    with torch.no_grad():
        names = []
        values = []

        # 按 key 排序，保证所有卡顺序一致
        for k in sorted(input_dict.keys()):
            names.append(k)
            values.append(input_dict[k])

        values = torch.stack(values, dim=0)
        tdist.all_reduce(values)

        if average:
            values /= world_size

        reduced_dict = {k: v for k, v in zip(names, values)}

    return reduced_dict


class MetricLogger(object):
    """
    训练指标日志管理器（最核心类）
    管理多个 SmoothedValue，自动打印、计时、分布式同步
    """
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)  # 自动创建指标
        self.delimiter = delimiter               # 打印分隔符

    def update(self, **kwargs):
        """更新指标，支持 loss=..., lr=..., acc=..."""
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        """允许直接访问 logger.loss / logger.lr"""
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{attr}'")

    def __str__(self):
        """把所有指标拼成一行字符串"""
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(f"{name}: {str(meter)}")
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        """多卡同步所有指标"""
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        """手动添加自定义 meter"""
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        """
        迭代器 + 自动定时打印日志（训练最常用）
        自动计算：迭代速度、ETA、显存、指标
        """
        i = 0
        if not header:
            header = ''

        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')    # 迭代时间
        data_time = SmoothedValue(fmt='{avg:.4f}')    # 数据加载时间
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'

        # 日志格式
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}',
                'max mem: {memory:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                'eta: {eta}',
                '{meters}',
                'time: {time}',
                'data: {data}'
            ])

        MB = 1024.0 * 1024.0

        for obj in iterable:
            # 记录数据加载时间
            data_time.update(time.time() - end)

            # 抛出数据给训练
            yield obj

            # 记录迭代时间
            iter_time.update(time.time() - end)

            # 定时打印
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))

                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))

            i += 1
            end = time.time()

        # 总耗时统计
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f'{header} Total time: {total_time_str} ({total_time / len(iterable):.4f} s / it)')