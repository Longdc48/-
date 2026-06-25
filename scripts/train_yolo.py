#!/usr/bin/env python
"""
YOLOv8 口罩检测训练脚本

用法:
    # 使用 YOLOv8n 训练（推荐入门）
    python scripts/train_yolo.py --model yolov8n.pt --epochs 100

    # 使用更大的模型
    python scripts/train_yolo.py --model yolov8s.pt --epochs 100 --batch 32

    # 对比多个模型
    python scripts/train_yolo.py --model yolov8n.pt yolov8s.pt --epochs 100
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from ultralytics import YOLO


# YOLOv8 模型变体说明
MODEL_INFO = {
    "yolov8n.pt": "YOLOv8 Nano — 最轻量，适合快速验证",
    "yolov8s.pt": "YOLOv8 Small — 轻量与精度平衡",
    "yolov8m.pt": "YOLOv8 Medium — 中等精度",
    "yolov8l.pt": "YOLOv8 Large — 高精度",
    "yolov8x.pt": "YOLOv8 XLarge — 最高精度",
}


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 口罩检测训练")
    parser.add_argument("--model", type=str, nargs="+",
                        default=["yolov8n.pt"],
                        help="YOLO 模型，支持多个用空格分隔")
    parser.add_argument("--data", type=str, default="dataset_yolo/data.yaml",
                        help="data.yaml 路径")
    parser.add_argument("--epochs", type=int, default=100,
                        help="训练轮数")
    parser.add_argument("--batch", type=int, default=16,
                        help="批次大小")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="输入图像尺寸")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="初始学习率")
    parser.add_argument("--patience", type=int, default=15,
                        help="早停耐心值")
    parser.add_argument("--device", type=str, default="0",
                        help="设备 (0=GPU 0, cpu=CPU)")
    parser.add_argument("--project", type=str, default="runs",
                        help="训练输出目录")
    parser.add_argument("--resume", action="store_true",
                        help="从上次中断处恢复")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"\n{'='*60}")
    print(f"  YOLOv8 口罩检测训练")
    print(f"  数据集: {args.data}")
    print(f"  模型: {args.model}")
    print(f"  设备: {args.device}")
    print(f"{'='*60}\n")

    results_summary = {}

    for model_name in args.model:
        model_display = model_name.replace(".pt", "")
        print(f"\n{'─'*60}")
        print(f"  训练: {model_name}")
        if model_name in MODEL_INFO:
            print(f"  说明: {MODEL_INFO[model_name]}")
        print(f"{'─'*60}\n")

        # ── 加载模型 ────────────────────────────────────────────
        # 如果 .pt 文件已存在则加载，否则自动下载预训练权重
        model = YOLO(model_name)

        # ── 训练 ────────────────────────────────────────────────
        results = model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            lr0=args.lr,
            patience=args.patience,
            device=args.device,
            project=args.project,
            name=f"mask_detect_{model_display}",
            exist_ok=True,
            resume=args.resume,
            # 数据增强
            hsv_h=0.015,        # HSV-Hue 增强
            hsv_s=0.7,          # HSV-Saturation 增强
            hsv_v=0.4,          # HSV-Value 增强
            degrees=10.0,       # 旋转
            translate=0.1,      # 平移
            scale=0.5,          # 缩放
            shear=0.0,          # 剪切
            flipud=0.0,         # 上下翻转概率
            fliplr=0.5,         # 左右翻转概率
            mosaic=1.0,         # Mosaic 增强
            mixup=0.1,          # MixUp 增强
            # 优化器
            optimizer="auto",   # 自动选择
            cos_lr=True,        # Cosine 学习率调度
            close_mosaic=10,    # 最后 10 轮关闭 mosaic
        )

        # ── 记录结果 ────────────────────────────────────────────
        results_summary[model_display] = {
            "mAP50": float(results.results_dict.get("metrics/mAP50(B)", 0)),
            "mAP50-95": float(results.results_dict.get("metrics/mAP50-95(B)", 0)),
            "precision": float(results.results_dict.get("metrics/precision(B)", 0)),
            "recall": float(results.results_dict.get("metrics/recall(B)", 0)),
        }

    # ── 汇总 ────────────────────────────────────────────────────
    if len(args.model) > 1:
        print(f"\n{'='*60}")
        print(f"  模型对比汇总")
        print(f"{'='*60}")
        header = f"  {'模型':<20} {'mAP@50':>10} {'mAP@50-95':>12} {'Precision':>12} {'Recall':>12}"
        print(header)
        print(f"  {'-'*68}")
        for name, metrics in results_summary.items():
            print(
                f"  {name:<20} "
                f"{metrics['mAP50']:>10.4f} "
                f"{metrics['mAP50-95']:>12.4f} "
                f"{metrics['precision']:>12.4f} "
                f"{metrics['recall']:>12.4f}"
            )

    print(f"\n{'='*60}")
    print(f"  训练完成！结果保存在: {args.project}/mask_detect_*/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
