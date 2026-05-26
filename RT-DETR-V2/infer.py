"""Copyright(c) 2023 lyuwenyu. All Rights Reserved.
"""

import os
import sys
import time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import cv2
import numpy as np
import torch
import onnxruntime as ort

from src.core import YAMLConfig, yaml_utils
from PIL import Image


def main(args):
    # 1. 加载配置（和 export 完全一致）
    update_dict = yaml_utils.parse_cli(args.update) if args.update else {}
    update_dict.update({k: v for k, v in args.__dict__.items()
                        if k not in ['update', ] and v is not None})
    cfg = YAMLConfig(args.config, **update_dict)

    # 2. 创建输出文件夹
    os.makedirs(args.output_dir, exist_ok=True)

    # 3. 加载 ONNX 模型（只加载一次）
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    session = ort.InferenceSession(args.onnx_path, providers=providers)

    # 4. 获取图片列表
    img_dir = args.image_dir
    img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                 if f.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))]

    print(f"✅ 找到 {len(img_paths)} 张图片，开始推理...\n")

    # 遍历推理
    for idx, img_path in enumerate(img_paths):
        # 读取图片
        img = cv2.imread(img_path)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        h, w = img.shape[:2]

        # 预处理
        img_pil = img_pil.resize((args.input_size, args.input_size), Image.BILINEAR)
        img_tensor = torch.from_numpy(np.array(img_pil)).permute(2, 0, 1).float() / 255.0
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std
        img_tensor = img_tensor.unsqueeze(0)
        orig_size = np.array([[h, w]]).astype(np.int64)

        # 推理 + 计时
        start = time.time()

        inputs = {
            'images': img_tensor.numpy(),
            'orig_target_sizes': orig_size
        }
        labels, boxes, scores = session.run(None, inputs)

        end = time.time()
        infer_time = end - start

        # 绘制结果
        for label, box, score in zip(labels[0], boxes[0], scores[0]):
            if score < args.conf_thres:
                continue
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, f"{int(label)}:{score:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 保存
        name = os.path.basename(img_path)
        save_path = os.path.join(args.output_dir, name)
        cv2.imwrite(save_path, img)

        # 打印信息
        print(f"[{idx+1}/{len(img_paths)}] {name} | 推理时间: {infer_time*1000:.2f} ms")

    print(f"\n✅ 全部推理完成！结果保存在: {args.output_dir}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', '-c', type=str, required=True)
    parser.add_argument('--onnx-path', '-m', type=str, required=True)
    parser.add_argument('--image-dir', '-i', type=str, default='D:/datasets/object_detection/overhang/new/images')
    parser.add_argument('--input_size', '-s', type=int, default=640)
    parser.add_argument('--output-dir', '-o', type=str, default='results', help='output/result')
    parser.add_argument('--conf-thres', '-t', type=float, default=0.5)
    parser.add_argument('--update', '-u', nargs='+', help='update yaml config')

    args = parser.parse_args()

    main(args)