"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
RT-DETRv2 推理代码 | 完全对齐官方逻辑 | 修复坐标错位+检测不准
"""

# 导入系统模块，用于路径处理和环境配置
import os
import sys
# 将项目根目录添加到Python环境变量，确保能导入项目内的模块
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

# 导入深度学习框架PyTorch
import torch
# 导入PyTorch神经网络模块
import torch.nn as nn
# 导入OpenCV，用于图像读取、预处理、可视化
import cv2
# 导入numpy，用于数值计算
import numpy as np

# -------------------------- 【直接修改这里的参数】 --------------------------
# 模型配置文件路径（yaml格式，定义模型结构、超参数）
CONFIG_PATH = "../configs/rtdetrv2/rtdetrv2_r50vd_6x_coco.yml"
# 训练好的模型权重文件路径（pth格式）
RESUME_PATH = "../output/rtdetrv2_r50vd_6x_coco/best.pth"
# 模型输入图像尺寸（正方形，宽高均为640）
INPUT_SIZE = 640
# 待检测的图片路径
IMAGE_PATH = "../test.bmp"
# 检测置信度阈值（低于该值的目标会被过滤）
SCORE_THRESHOLD = 0.2
# 是否使用离散采样（部署ONNX时打开，普通推理关闭，解决部署算子不兼容问题）
USE_DISCRETE_SAMPLE = False
# -------------------------------------------------------------------------

# 导入项目中的配置解析器，用于加载yaml模型配置
from src.core import YAMLConfig

# -------------------------- 离散采样算子（部署用） --------------------------
def discrete_sample(input: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """
    RT-DETRv2官方离散采样算子，替代torch的grid_sample
    仅在导出ONNX部署时使用，常规推理保持关闭即可
    Args:
        input: 输入特征图张量 [B, C, H, W]
        grid: 采样网格坐标张量
    Return:
        采样后的特征图张量
    """
    # 获取输入特征图的维度：批次、通道、高度、宽度
    B, C, H, W = input.shape
    # 获取网格坐标的形状
    *grid_shape, _ = grid.shape

    # 坐标转换：将网格坐标从[-1,1]归一化范围 转换为 图像像素坐标[0, W-1] / [0, H-1]
    grid_x = (grid[..., 0] + 1) * (W - 1) / 2
    grid_y = (grid[..., 1] + 1) * (H - 1) / 2

    # 坐标四舍五入取整，得到整数像素坐标
    grid_x = torch.round(grid_x).long()
    grid_y = torch.round(grid_y).long()

    # 坐标越界保护：防止坐标超出图像范围
    grid_x = torch.clamp(grid_x, 0, W-1)
    grid_y = torch.clamp(grid_y, 0, H-1)

    # 生成批次索引，匹配网格形状
    batch_idx = torch.arange(B, device=input.device)[:, None, None].expand(*grid_shape)
    # 按照坐标索引采样特征图，并调整维度顺序
    return input[batch_idx, :, grid_y, grid_x].permute(0, 3, 1, 2)


def load_model():
    """
    加载RT-DETRv2模型，完全对齐官方export.py的逻辑
    Return:
        加载好权重的推理模型（eval模式）
    """
    # 1. 加载yaml模型配置文件
    cfg = YAMLConfig(CONFIG_PATH)

    # 2. 加载pth权重文件，映射到CPU（避免GPU显存不足问题）
    checkpoint = torch.load(RESUME_PATH, map_location='cpu')
    # 优先加载EMA权重（训练中EMA权重精度更高），没有则加载普通model权重
    state = checkpoint['ema']['module'] if 'ema' in checkpoint else checkpoint['model']
    # 将权重加载到模型中
    cfg.model.load_state_dict(state)

    # 3. 构建部署专用模型（简化前向逻辑，去除训练分支）
    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # 初始化模型主干（deploy模式）
            self.model = cfg.model.deploy()
            # 初始化后处理模块（解码检测框、置信度、类别）
            self.postprocessor = cfg.postprocessor.deploy()

            # 如果开启离散采样，替换模型中的grid_sample算子（部署专用）
            if USE_DISCRETE_SAMPLE:
                self._replace_grid_sample()

        def _replace_grid_sample(self):
            """
            递归遍历模型，替换所有可变形卷积中的grid_sample为自定义discrete_sample
            解决ONNX部署不支持grid_sample的问题
            """
            for name, module in self.named_modules():
                # 筛选可变形卷积模块
                if hasattr(module, 'forward') and 'deformable' in name.lower():
                    if hasattr(module, 'sampling_op'):
                        # 替换采样算子
                        module.sampling_op = discrete_sample

        def forward(self, images, orig_target_sizes):
            """模型前向推理流程"""
            # 主干网络特征提取+预测
            outputs = self.model(images)
            # 后处理：解码出最终的类别、检测框、置信度
            return self.postprocessor(outputs, orig_target_sizes)

    # 实例化模型
    model = Model()
    # 设置为评估模式（关闭dropout、batchnorm等训练层）
    model.eval()
    return model


def preprocess():
    """
    【完全对齐官方】RT-DETRv2图像预处理
    核心：等比例缩放+灰色填充 → 不拉伸物体 → 保证检测精度
    Return:
        tensor: 模型输入张量
        img_origin: 原始读取图像（用于最终可视化）
        scale: 缩放比例
        pad_left: 左侧填充像素数
        pad_top: 顶部填充像素数
    """
    # 1. 读取原始图像（OpenCV默认读取为BGR格式）
    img_origin = cv2.imread(IMAGE_PATH)
    # 获取原始图像的高度、宽度
    orig_h, orig_w = img_origin.shape[:2]

    # 2. 计算等比例缩放比例：保证图像缩放到640内，不拉伸
    scale = min(INPUT_SIZE / orig_w, INPUT_SIZE / orig_h)
    # 计算缩放后的图像宽高
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))

    # 3. 等比例缩放图像（双线性插值，保证图像质量）
    img_resize = cv2.resize(img_origin, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    # 4. 计算需要填充的像素数（将缩放后的图像填充为640x640正方形）
    pad_w = INPUT_SIZE - new_w
    pad_h = INPUT_SIZE - new_h
    # 左右均分填充
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    # 上下均分填充
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    # 5. 灰色填充(114,114,114)到INPUT_SIZE×INPUT_SIZE
    img_pad = cv2.copyMakeBorder(
        img_resize, pad_top, pad_bottom, pad_left, pad_right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )

    # 6. BGR转RGB（模型训练用RGB格式，OpenCV读取是BGR）
    img_rgb = cv2.cvtColor(img_pad, cv2.COLOR_BGR2RGB)

    # 7. 图像归一化：转为浮点型 → 除以255 → 减均值除以方差（对齐训练预处理）
    img_rgb = img_rgb.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)  # 图像均值
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)   # 图像方差
    img_rgb = (img_rgb - mean) / std

    # 8. 维度转换：HWC(高宽通道) → CHW(通道高宽) → 增加batch维度 → 转为张量
    tensor = torch.from_numpy(img_rgb.transpose(2, 0, 1)).unsqueeze(0)

    # 返回预处理结果+原始图像+缩放比例+填充偏移量（用于后续坐标还原）
    return tensor, img_origin, scale, pad_left, pad_top


def visualize(img_origin, labels, boxes, scores):
    """
    检测结果可视化，坐标完全对齐原始图像
    Args:
        img_origin: 原始未处理的图像
        labels: 检测类别张量
        boxes: 检测框坐标张量
        scores: 置信度张量
    """
    # 遍历所有检测目标
    for label, box, score in zip(labels, boxes, scores):
        # 过滤低置信度目标
        if score.item() < SCORE_THRESHOLD:
            continue

        # 将坐标转为整数（像素坐标必须为整数）
        x1, y1, x2, y2 = map(int, box)
        # 绘制绿色检测框（线宽2）
        cv2.rectangle(img_origin, (x1, y1), (x2, y2), (0, 255, 0), 2)
        # 绘制红色标签+置信度文本
        text = f"cls:{label.item()} {score.item():.2f}"
        cv2.putText(img_origin, text, (x1, y1-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # 保存可视化结果
    cv2.imwrite("detect_result.jpg", img_origin)
    print("✅ 推理结果已保存为: detect_result.jpg")


def main():
    """主函数：模型加载→图像预处理→推理→坐标还原→结果打印→可视化"""
    # 1. 加载模型
    model = load_model()
    # 自动选择设备：有GPU用cuda，无GPU用cpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"✅ 模型加载完成，运行设备: {device}")
    print(f"✅ 使用离散采样: {USE_DISCRETE_SAMPLE}")

    # 2. 图像预处理（对齐官方逻辑）
    tensor, img_origin, scale, pad_left, pad_top = preprocess()
    # 将输入张量放到指定设备
    tensor = tensor.to(device)

    # 3. 模型推理（无梯度计算，提升推理速度）
    with torch.no_grad():
        # 传入输入尺寸[640,640]，让后处理输出640尺度的像素坐标
        labels, boxes, scores = model(tensor, torch.tensor([[INPUT_SIZE, INPUT_SIZE]], device=device))

    # 4. 压缩batch维度（去掉batch=1的冗余维度）
    labels = labels.squeeze(0)
    boxes = boxes.squeeze(0)
    scores = scores.squeeze(0)

    # ==================== 【核心：唯一正确的坐标还原逻辑】 ====================
    # 第一步：减去预处理时的填充偏移量（还原到缩放后的图像坐标）
    boxes[:, 0] -= pad_left  # 左上角x坐标 - 左填充
    boxes[:, 1] -= pad_top   # 左上角y坐标 - 上填充
    boxes[:, 2] -= pad_left  # 右下角x坐标 - 左填充
    boxes[:, 3] -= pad_top   # 右下角y坐标 - 上填充

    # 第二步：除以缩放比例（还原到原始图像的真实坐标）
    boxes /= scale
    # ====================================================================

    # 5. 打印检测结果
    print("\n" + "="*50)
    # 统计有效目标数量
    print(f"检测到目标数: {len([s for s in scores if s >= SCORE_THRESHOLD])}")
    for i, (label, box, score) in enumerate(zip(labels, boxes, scores)):
        if score.item() >= SCORE_THRESHOLD:
            # 保留两位小数，格式化输出坐标
            x1, y1, x2, y2 = map(lambda x: round(x.item(), 2), box)
            print(f"目标{i+1}: 类别={label.item()}, 置信度={score.item():.4f}, 坐标=[{x1}, {y1}, {x2}, {y2}]")
    print("="*50 + "\n")

    # 6. 结果可视化并保存
    visualize(img_origin, labels, boxes, scores)


# 程序入口
if __name__ == '__main__':
    main()