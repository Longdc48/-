#!/usr/bin/env python
"""
YOLOv8 口罩检测评估脚本

用法:
    # 评估单个模型
    python scripts/evaluate_yolo.py --weights runs/mask_detect_yolov8n/weights/best.pt

    # 对比多个模型
    python scripts/evaluate_yolo.py --weights runs/mask_detect_yolov8n/weights/best.pt runs/mask_detect_yolov8s/weights/best.pt --compare
"""

import os
import sys
import argparse
import random
import shutil
import cv2
import yaml
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from ultralytics import YOLO

# ── 中文字体 ────────────────────────────────────────────────────
try:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass
plt.rcParams["figure.dpi"] = 150
sns.set_style("whitegrid")

CLASS_NAMES = ["without_mask", "with_mask"]
CLASS_NAMES_CN = ["没戴口罩", "戴口罩"]


def parse_args():
    parser = argparse.ArgumentParser(description="YOLOv8 口罩检测评估")
    parser.add_argument("--weights", type=str, nargs="+", required=True,
                        help="模型权重路径（支持多个）")
    parser.add_argument("--data", type=str, default="dataset_yolo/data.yaml",
                        help="data.yaml 路径")
    parser.add_argument("--output_dir", type=str, default="results",
                        help="结果输出目录")
    parser.add_argument("--device", type=str, default="0",
                        help="设备")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="输入图像尺寸")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.45,
                        help="IoU 阈值")
    parser.add_argument("--compare", action="store_true",
                        help="生成多模型对比图表")
    parser.add_argument("--save_samples", type=int, default=20,
                        help="保存检测结果样本数")
    return parser.parse_args()


def plot_confusion_matrix(cm_path: str, output_dir: str, model_name: str):
    """读取 YOLO 生成的混淆矩阵并美化"""
    # YOLO 会自己生成混淆矩阵，这里做二次美化
    pass


def plot_training_curves(results_dir: str, output_dir: str, model_name: str):
    """从训练结果 CSV 重新绘制更美观的损失曲线"""
    csv_path = Path(results_dir) / "results.csv"
    if not csv_path.exists():
        print(f"  [警告] 找不到训练结果: {csv_path}")
        return

    df = pd.read_csv(csv_path)
    # 清理列名（去空格）
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ── Box Loss ────────────────────────────────────────────────
    if "train/box_loss" in df.columns:
        ax = axes[0, 0]
        ax.plot(df.index, df["train/box_loss"], "b-", alpha=0.6, label="Train Box Loss", linewidth=1)
        ax.plot(df.index, df["val/box_loss"], "r-", alpha=0.8, label="Val Box Loss", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Box Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # ── Class Loss ──────────────────────────────────────────────
    if "train/cls_loss" in df.columns:
        ax = axes[0, 1]
        ax.plot(df.index, df["train/cls_loss"], "b-", alpha=0.6, label="Train Cls Loss", linewidth=1)
        ax.plot(df.index, df["val/cls_loss"], "r-", alpha=0.8, label="Val Cls Loss", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("Classification Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # ── DFL Loss ────────────────────────────────────────────────
    if "train/dfl_loss" in df.columns:
        ax = axes[1, 0]
        ax.plot(df.index, df["train/dfl_loss"], "b-", alpha=0.6, label="Train DFL Loss", linewidth=1)
        ax.plot(df.index, df["val/dfl_loss"], "r-", alpha=0.8, label="Val DFL Loss", linewidth=1.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.set_title("DFL Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # ── mAP 曲线 ────────────────────────────────────────────────
    ax = axes[1, 1]
    if "metrics/mAP50(B)" in df.columns:
        ax.plot(df.index, df["metrics/mAP50(B)"], "g-", label="mAP@50", linewidth=2)
    if "metrics/mAP50-95(B)" in df.columns:
        ax.plot(df.index, df["metrics/mAP50-95(B)"], "orange", label="mAP@50-95", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mAP")
    ax.set_title("Mean Average Precision")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Training Curves — {model_name}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{model_name}_training_curves.png")
    fig.savefig(save_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] 训练曲线: {save_path}")


def save_detection_samples(model, data_yaml: str, output_dir: str, model_name: str, n: int = 20):
    """保存测试集检测结果可视化样本"""
    with open(data_yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    test_img_dir = Path(config["path"]) / "images" / "test"
    if not test_img_dir.exists():
        test_img_dir = Path(config["path"]) / config.get("test", "images/test")

    all_images = list(Path(test_img_dir).glob("*.jpg"))
    if not all_images:
        print(f"  [警告] 找不到测试图片: {test_img_dir}")
        return

    samples = random.sample(all_images, min(n, len(all_images)))
    sample_dir = Path(output_dir) / f"{model_name}_detection_samples"
    sample_dir.mkdir(parents=True, exist_ok=True)

    for img_path in samples:
        results = model(str(img_path), conf=0.25)
        annotated = results[0].plot(show=False, line_width=2, font_size=10)
        save_path = sample_dir / f"{img_path.stem}_detected.jpg"
        cv2.imwrite(str(save_path), annotated)

    print(f"  [OK] 检测样本 ({len(samples)} 张): {sample_dir}")


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    all_metrics = {}

    for weight_path in args.weights:
        if not os.path.exists(weight_path):
            print(f"  [警告] 权重文件不存在: {weight_path}")
            continue

        # ── 提取模型名称 ────────────────────────────────────────
        # 路径格式: runs/detect/mask_detect_yolov8n/weights/best.pt
        parts = Path(weight_path).parts
        model_name = "model"
        for p in parts:
            if "mask_detect" in p:
                model_name = p
                break
        if model_name == "model":
            model_name = Path(weight_path).stem

        print(f"\n{'='*60}")
        print(f"  评估: {model_name}")
        print(f"  权重: {weight_path}")
        print(f"{'='*60}\n")

        # ── 加载模型 ────────────────────────────────────────────
        model = YOLO(weight_path)

        # ── 验证（在测试集上） ──────────────────────────────────
        # YOLO val 会自动使用 data.yaml 中定义的 test 集
        metrics = model.val(
            data=args.data,
            device=args.device,
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.iou,
            split="test",
            project=args.output_dir,
            name=f"{model_name}_val",
            exist_ok=True,
        )

        # ── 收集指标 ────────────────────────────────────────────
        m = {
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        }

        # 各类别指标
        print(f"\n  整体指标:")
        print(f"    mAP@50:     {m['mAP50']:.4f}")
        print(f"    mAP@50-95:  {m['mAP50-95']:.4f}")
        print(f"    Precision:  {m['precision']:.4f}")
        print(f"    Recall:     {m['recall']:.4f}")

        # 各类别 mAP
        if hasattr(metrics.box, 'ap50'):
            ap50_per_class = metrics.box.ap50
            print(f"\n  各类别 AP@50:")
            for i, ap in enumerate(ap50_per_class):
                name = CLASS_NAMES_CN[i] if i < len(CLASS_NAMES_CN) else f"class_{i}"
                print(f"    {name}: {ap:.4f}")

        all_metrics[model_name] = m

        # ── 绘制训练曲线 ────────────────────────────────────────
        results_dir = Path(weight_path).parent.parent  # .../mask_detect_xxx/
        plot_training_curves(str(results_dir), args.output_dir, model_name)

        # ── 保存检测样本 ────────────────────────────────────────
        save_detection_samples(model, args.data, args.output_dir, model_name, args.save_samples)

        # ── YOLO 自动生成的图表也复制过来 ───────────────────────
        val_dir = Path(args.output_dir) / f"{model_name}_val"
        for generated_file in val_dir.glob("*.png"):
            dest = Path(args.output_dir) / f"{model_name}_{generated_file.name}"
            shutil.copy2(str(generated_file), str(dest))
        for generated_file in val_dir.glob("*.jpg"):
            dest = Path(args.output_dir) / f"{model_name}_{generated_file.name}"
            shutil.copy2(str(generated_file), str(dest))

    # ── 多模型对比 ──────────────────────────────────────────────
    if args.compare and len(all_metrics) > 1:
        print(f"\n{'='*60}")
        print(f"  多模型对比")
        print(f"{'='*60}")

        # 表格
        header = f"  {'模型':<30} {'mAP@50':>10} {'mAP@50-95':>12} {'Precision':>12} {'Recall':>12}"
        print(header)
        print(f"  {'-'*78}")
        for name, m in all_metrics.items():
            print(f"  {name:<30} {m['mAP50']:>10.4f} {m['mAP50-95']:>12.4f} {m['precision']:>12.4f} {m['recall']:>12.4f}")

        # 柱状图对比
        fig, ax = plt.subplots(figsize=(10, 6))
        models_list = list(all_metrics.keys())
        x = np.arange(len(models_list))
        width = 0.2

        ax.bar(x - 1.5*width, [all_metrics[n]["mAP50"] for n in models_list], width, label="mAP@50")
        ax.bar(x - 0.5*width, [all_metrics[n]["mAP50-95"] for n in models_list], width, label="mAP@50-95")
        ax.bar(x + 0.5*width, [all_metrics[n]["precision"] for n in models_list], width, label="Precision")
        ax.bar(x + 1.5*width, [all_metrics[n]["recall"] for n in models_list], width, label="Recall")

        ax.set_xticks(x)
        ax.set_xticklabels(models_list, rotation=15, ha="right", fontsize=9)
        ax.set_ylabel("Score")
        ax.set_title("Model Comparison — Mask Detection", fontweight="bold")
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis="y")

        compare_path = os.path.join(args.output_dir, "model_comparison.png")
        plt.tight_layout()
        fig.savefig(compare_path, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  [OK] 模型对比图: {compare_path}")

        # 对比报告
        report_path = os.path.join(args.output_dir, "comparison_report.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("Mask Detection — Model Comparison Report\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"{'Model':<30} {'mAP@50':>10} {'mAP@50-95':>12} {'Precision':>12} {'Recall':>12}\n")
            f.write("-" * 78 + "\n")
            for name, m in all_metrics.items():
                f.write(f"{name:<30} {m['mAP50']:>10.4f} {m['mAP50-95']:>12.4f} {m['precision']:>12.4f} {m['recall']:>12.4f}\n")
        print(f"  [OK] 对比报告: {report_path}")

    print(f"\n{'='*60}")
    print(f"  评估完成！结果保存在: {args.output_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
