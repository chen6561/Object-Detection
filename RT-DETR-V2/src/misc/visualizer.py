"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import torch
import torch.utils.data

import torchvision
# 关闭 torchvision v2 transforms 的 beta 版本警告
torchvision.disable_beta_transforms_warning()

import PIL

# 对外暴露的接口
__all__ = ['show_sample']


def show_sample(sample):
    """
    可视化 COCO 数据集中的一张图片和它的标注框
    适用于 COCO 数据集 / DataLoader 输出的样本
    """
    # 导入绘图工具
    import matplotlib.pyplot as plt
    # 导入 torchvision 图像转换函数
    from torchvision.transforms.v2 import functional as F
    # 导入画框工具函数
    from torchvision.utils import draw_bounding_boxes

    # 把输入样本拆分为 图像 和 标注
    image, target = sample

    # 如果图像是 PIL 格式，先转为张量格式
    if isinstance(image, PIL.Image.Image):
        image = F.to_image_tensor(image)

    # 把图像数据类型转为 uint8（0~255，画图必需）
    image = F.convert_dtype(image, torch.uint8)

    # 在图像上绘制黄色标注框，宽度 3 像素
    annotated_image = draw_bounding_boxes(image, target["boxes"], colors="yellow", width=3)

    # 创建画布
    fig, ax = plt.subplots()

    # 显示带框的图像，把通道从 [C, H, W] 转为 [H, W, C]
    ax.imshow(annotated_image.permute(1, 2, 0).numpy())

    # 隐藏坐标轴刻度
    ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

    # 紧凑布局
    fig.tight_layout()

    # 显示图像窗口
    fig.show()
    plt.show()