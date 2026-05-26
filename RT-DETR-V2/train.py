"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

# 导入系统内置模块
import os
import sys

# 导入命令行参数解析工具
import argparse

# 导入项目自定义模块
from src.misc import dist_utils       # 分布式训练工具包
from src.core import YAMLConfig, yaml_utils  # 配置文件加载与解析工具
from src.solver import TASKS          # 任务调度器（训练/验证/测试）


def main(args, ) -> None:
    """主函数：程序核心执行逻辑
    Args:
        args: 命令行解析后的参数对象
    """
    # 1. 初始化分布式训练环境（设置打印等级、随机种子，保证多卡训练一致性）
    dist_utils.setup_distributed(args.print_rank, args.print_method, seed=args.seed)

    # 2. 参数合法性校验：resume(断点续训) 和 tuning(微调) 不能同时启用，二选一
    assert not all([args.tuning, args.resume]), \
        'Only support from_scrach or resume or tuning at one time'

    # 3. 解析命令行传入的配置覆盖参数（-u/--update），转为字典格式
    update_dict = yaml_utils.parse_cli(args.update)
    # 4. 合并命令行参数：将非空的命令行参数更新到配置字典中，覆盖yaml配置
    update_dict.update({k: v for k, v in args.__dict__.items() \
        if k not in ['update', ] and v is not None})

    # 5. 加载YAML配置文件，并合并命令行覆盖的参数，生成最终训练配置
    cfg = YAMLConfig(args.config, **update_dict)
    # 打印最终配置信息（调试用）
    print('cfg: ', cfg.__dict__)

    # 6. 根据配置中的task类型（train/val），创建对应的求解器（训练/验证核心类）
    solver = TASKS[cfg.yaml_cfg['task']](cfg)

    # 7. 判断执行模式：仅测试/验证  OR  完整训练+验证流程
    if args.test_only:
        # 仅执行验证/测试逻辑
        solver.val()
    else:
        # 执行完整训练流程（训练+周期性验证）
        solver.fit()

    # 8. 清理分布式训练环境，释放资源
    dist_utils.cleanup()


if __name__ == '__main__':
    # 命令行参数解析器初始化
    parser = argparse.ArgumentParser(description="RT-DETR 训练/验证主程序")

    # ===================== 优先级0：核心基础参数 =====================
    # 必选参数：指定YAML模型配置文件路径
    parser.add_argument('-c', '--config', type=str, required=True, help='必选参数，模型YAML配置文件路径')
    # 可选参数：断点续训，从指定检查点恢复训练
    parser.add_argument('-r', '--resume', type=str, help='断点续训：从指定的模型检查点恢复训练')
    # 可选参数：模型微调，从指定预训练权重开始微调
    parser.add_argument('-t', '--tuning', type=str, help='模型微调：从指定预训练权重开始微调')    # 可选参数：指定运行设备（cpu/cuda/cuda:0等）
    parser.add_argument('-d', '--device', type=str, help='指定运行设备，如cuda、cpu、cuda:0')
    # 可选参数：随机种子，保证实验可复现
    parser.add_argument('--seed', type=int, help='随机种子，保证实验结果可复现')
    # 可选参数：启用自动混合精度训练（节省显存，加速训练）
    parser.add_argument('--use-amp', action='store_true', help='启用自动混合精度训练(AMP)')
    # 可选参数：指定模型权重、日志输出目录
    parser.add_argument('--output-dir', type=str, help='模型权重、日志输出目录')
    # 可选参数：指定TensorBoard日志保存目录
    parser.add_argument('--summary-dir', type=str, help='TensorBoard可视化日志保存目录')
    # 可选参数：仅执行验证/测试，不训练
    parser.add_argument('--test-only', action='store_true', default=False, help='仅执行验证/测试，不进行训练')

    # ===================== 优先级1：配置覆盖参数 =====================
    # 可选参数：命令行直接覆盖YAML配置（格式：key=value key2=value2）
    parser.add_argument('-u', '--update', nargs='+', help='命令行覆盖YAML配置，格式：key=value key2=value2')

    # ===================== 分布式环境参数 =====================
    # 打印方式配置
    parser.add_argument('--print-method', type=str, default='builtin', help='分布式训练打印方式')
    # 指定哪个GPU卡号打印日志（默认0号卡）
    parser.add_argument('--print-rank', type=int, default=0, help='指定打印日志的GPU卡号')
    # 分布式训练本地卡号（PyTorch分布式自动传入）
    parser.add_argument('--local-rank', type=int, help='分布式训练本地GPU卡号（自动传入，无需手动设置）')

    # 解析所有命令行参数
    args = parser.parse_args()

    # 调用主函数，执行程序
    main(args)