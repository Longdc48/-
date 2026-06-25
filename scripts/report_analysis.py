#!/usr/bin/env python
"""
Supplementary report analysis — generate additional figures for course report:
  1. Dataset statistics (class distribution, box sizes, split)
  2. Per-class performance bar chart
  3. Confidence distribution (TP vs FP)
  4. Full training summary curves
  5. Report data markdown
"""

import os, sys, random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import yaml
from collections import Counter
from tqdm import tqdm

import torch
from ultralytics import YOLO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

plt.rcParams["figure.dpi"] = 150
sns.set_style("whitegrid")

CLASS_NAMES = ["without_mask", "with_mask"]
CLASS_LABELS = ["No Mask", "With Mask"]
COLORS = ["#E74C3C", "#2ECC71", "#3498DB", "#F39C12"]


def plot_dataset_statistics():
    """Figure 1: Dataset statistics"""
    with open("dataset_yolo/data.yaml", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # (a) Box count per class
    box_counts = Counter()
    all_labels = Path("dataset_yolo/labels")
    for split in ["train", "val", "test"]:
        lbl_dir = all_labels / split
        if lbl_dir.exists():
            for txt in lbl_dir.glob("*.txt"):
                for line in txt.read_text(encoding="utf-8").strip().split("\n"):
                    if line.strip():
                        box_counts[int(line.split()[0])] += 1

    ax = axes[0, 0]
    cats = [CLASS_LABELS[i] for i in sorted(box_counts)]
    vals = [box_counts[i] for i in sorted(box_counts)]
    bars = ax.bar(cats, vals, color=COLORS[:2])
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 200,
                f"{v:,}", ha="center", fontweight="bold")
    ax.set_title("(a) Bounding Box Count per Class", fontweight="bold")
    ax.set_ylabel("Number of Boxes")

    # (b) Boxes per image
    boxes_per_img = []
    train_dir = all_labels / "train"
    if train_dir.exists():
        for txt in train_dir.glob("*.txt"):
            lines = [l for l in txt.read_text(encoding="utf-8").strip().split("\n") if l.strip()]
            boxes_per_img.append(len(lines))

    ax = axes[0, 1]
    ax.hist(boxes_per_img, bins=50, color="#3498DB", edgecolor="white", alpha=0.8)
    ax.set_title("(b) Boxes per Image Distribution", fontweight="bold")
    ax.set_xlabel("Number of Boxes"); ax.set_ylabel("Number of Images")
    ax.axvline(np.mean(boxes_per_img), color="red", linestyle="--",
               label=f"Mean: {np.mean(boxes_per_img):.1f}")
    ax.legend()

    # (c) Box area distribution
    box_areas = {0: [], 1: []}
    for split in ["train", "val", "test"]:
        lbl_dir = all_labels / split
        if lbl_dir.exists():
            for txt in lbl_dir.glob("*.txt"):
                for line in txt.read_text(encoding="utf-8").strip().split("\n"):
                    if line.strip():
                        p = line.split()
                        box_areas[int(p[0])].append(float(p[3]) * float(p[4]))

    ax = axes[1, 0]
    for cls_id in [0, 1]:
        ax.hist(box_areas[cls_id], bins=40, alpha=0.5,
                label=CLASS_LABELS[cls_id], color=COLORS[cls_id], edgecolor="white")
    ax.set_title("(c) Normalized Box Area Distribution", fontweight="bold")
    ax.set_xlabel("Normalized Area (w x h)"); ax.set_ylabel("Box Count")
    ax.legend()

    # (d) Dataset split
    split_counts = {}
    for s in ["train", "val", "test"]:
        d = Path(config["path"]) / "images" / s
        split_counts[s] = len(list(d.glob("*.jpg"))) if d.exists() else 0

    ax = axes[1, 1]
    ax.pie([split_counts.get(s, 0) for s in ["train", "val", "test"]],
           labels=["Train", "Val", "Test"], autopct="%1.1f%%",
           colors=COLORS[:3], explode=(0.02, 0, 0))
    ax.set_title("(d) Dataset Split", fontweight="bold")

    plt.suptitle("Dataset Statistics — Mask Detection", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = "results/report_dataset_statistics.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path}")


def plot_per_class_metrics():
    """Figure 2: Per-class performance"""
    metrics = {
        "No Mask":      {"AP@50": 0.718, "Precision": 0.786, "Recall": 0.752},
        "With Mask":    {"AP@50": 0.797, "Precision": 0.849, "Recall": 0.827},
    }

    fig, ax = plt.subplots(figsize=(8, 6))
    x = np.arange(len(metrics))
    width = 0.25

    names = list(metrics.keys())
    vals_ap = [metrics[n]["AP@50"] for n in names]
    vals_p  = [metrics[n]["Precision"] for n in names]
    vals_r  = [metrics[n]["Recall"] for n in names]

    for bars, offset, color, label in [
        (vals_ap, -width, "#2ECC71", "AP@50"),
        (vals_p,  0,       "#3498DB", "Precision"),
        (vals_r,  +width,  "#E74C3C", "Recall"),
    ]:
        b = ax.bar(x + offset, bars, width, label=label, color=color)
        for bar in b:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f"{bar.get_height():.3f}", ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Per-Class Detection Performance (Test Set)", fontweight="bold", fontsize=14)
    ax.set_ylim(0, 1.0); ax.legend(fontsize=10); ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = "results/report_per_class_metrics.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path}")


def plot_confidence_distribution():
    """Figure 3: Confidence score distribution"""
    model = YOLO("runs/mask_detect_yolov8n/weights/best.pt")
    test_dir = Path("dataset_yolo/images/test")
    test_imgs = sorted(test_dir.glob("*.jpg"))
    random.seed(42)
    samples = random.sample(test_imgs, min(500, len(test_imgs)))

    all_confidences = []
    lbl_test_dir = Path("dataset_yolo/labels/test")

    for img_path in tqdm(samples, desc="Analyzing confidence"):
        results = model(str(img_path), conf=0.001, verbose=False)
        boxes = results[0].boxes
        if boxes is None:
            continue

        txt_path = lbl_test_dir / img_path.with_suffix(".txt").name
        if not txt_path.exists():
            continue

        gt_boxes = []
        for line in txt_path.read_text(encoding="utf-8").strip().split("\n"):
            if not line.strip():
                continue
            parts = line.strip().split()
            gt_boxes.append((int(parts[0]), float(parts[1]), float(parts[2]),
                             float(parts[3]), float(parts[4])))

        pred_xywh = boxes.xywhn.cpu().numpy() if len(boxes.xywhn) > 0 else np.zeros((0, 4))
        pred_cls = boxes.cls.cpu().numpy() if len(boxes.cls) > 0 else np.array([])
        pred_conf = boxes.conf.cpu().numpy() if len(boxes.conf) > 0 else np.array([])

        matched_gt = set()
        for pi in range(len(pred_cls)):
            best_iou, best_gi = 0, -1
            for gi, gt in enumerate(gt_boxes):
                if gi in matched_gt or gt[0] != int(pred_cls[pi]):
                    continue
                iou = _compute_iou(pred_xywh[pi], gt[1], gt[2], gt[3], gt[4])
                if iou > best_iou:
                    best_iou, best_gi = iou, gi
            is_correct = best_iou >= 0.45
            all_confidences.append((float(pred_conf[pi]), is_correct, int(pred_cls[pi])))
            if is_correct and best_gi >= 0:
                matched_gt.add(best_gi)

    if not all_confidences:
        print("  [WARN] No confidence data collected")
        return

    tp_confs = [c[0] for c in all_confidences if c[1]]
    fp_confs = [c[0] for c in all_confidences if not c[1]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.hist(tp_confs, bins=30, alpha=0.6, label=f"True Positives (n={len(tp_confs)})",
            color="#2ECC71", edgecolor="white")
    ax.hist(fp_confs, bins=30, alpha=0.6, label=f"False Positives (n={len(fp_confs)})",
            color="#E74C3C", edgecolor="white")
    ax.set_xlabel("Confidence Score"); ax.set_ylabel("Number of Predictions")
    ax.set_title("Prediction Confidence: TP vs FP", fontweight="bold")
    ax.legend()

    ax = axes[1]
    for label, data, color in [("TP", tp_confs, "#2ECC71"), ("FP", fp_confs, "#E74C3C")]:
        if data:
            sorted_data = np.sort(data)
            cumulative = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
            ax.plot(sorted_data, cumulative, color=color, linewidth=2, label=label)
    ax.set_xlabel("Confidence Threshold"); ax.set_ylabel("Cumulative Proportion")
    ax.set_title("Confidence CDF", fontweight="bold")
    ax.legend(); ax.grid(True, alpha=0.3)

    plt.suptitle("Model Prediction Confidence Analysis", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = "results/report_confidence_distribution.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path}")


def _compute_iou(pred_xywh, gx, gy, gw, gh):
    """Compute IoU between predicted box and ground truth box (both xywh normalized)"""
    px1, py1 = pred_xywh[0] - pred_xywh[2]/2, pred_xywh[1] - pred_xywh[3]/2
    px2, py2 = pred_xywh[0] + pred_xywh[2]/2, pred_xywh[1] + pred_xywh[3]/2
    gx1, gy1 = gx - gw/2, gy - gh/2
    gx2, gy2 = gx + gw/2, gy + gh/2

    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_p = pred_xywh[2] * pred_xywh[3]
    area_g = gw * gh
    union = area_p + area_g - inter
    return inter / union if union > 0 else 0


def plot_training_summary():
    """Figure 4: Full training curves"""
    df = pd.read_csv("runs/mask_detect_yolov8n/results.csv")
    df.columns = df.columns.str.strip()

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Box loss
    ax = axes[0, 0]
    ax.plot(df.index, df["train/box_loss"], "b-", alpha=0.5, label="Train", linewidth=1)
    ax.plot(df.index, df["val/box_loss"], "r-", label="Val", linewidth=1.5)
    ax.set_title("Box Loss"); ax.legend(); ax.grid(True, alpha=0.3)

    # Cls loss
    ax = axes[0, 1]
    ax.plot(df.index, df["train/cls_loss"], "b-", alpha=0.5, label="Train", linewidth=1)
    ax.plot(df.index, df["val/cls_loss"], "r-", label="Val", linewidth=1.5)
    ax.set_title("Classification Loss"); ax.legend(); ax.grid(True, alpha=0.3)

    # DFL loss
    ax = axes[0, 2]
    ax.plot(df.index, df["train/dfl_loss"], "b-", alpha=0.5, label="Train", linewidth=1)
    ax.plot(df.index, df["val/dfl_loss"], "r-", label="Val", linewidth=1.5)
    ax.set_title("DFL Loss"); ax.legend(); ax.grid(True, alpha=0.3)

    # mAP
    ax = axes[1, 0]
    ax.plot(df.index, df["metrics/mAP50(B)"], "#2ECC71", label="mAP@50", linewidth=2)
    ax.plot(df.index, df["metrics/mAP50-95(B)"], "#3498DB", label="mAP@50-95", linewidth=2)
    ax.set_title("Mean Average Precision"); ax.legend(); ax.grid(True, alpha=0.3)

    # Precision / Recall
    ax = axes[1, 1]
    ax.plot(df.index, df["metrics/precision(B)"], "#9B59B6", label="Precision", linewidth=1.5)
    ax.plot(df.index, df["metrics/recall(B)"], "#E67E22", label="Recall", linewidth=1.5)
    ax.set_title("Precision & Recall"); ax.legend(); ax.grid(True, alpha=0.3)

    # Learning Rate
    ax = axes[1, 2]
    ax.plot(df.index, df["lr/pg0"], "k-", linewidth=1.5)
    ax.set_title("Learning Rate (Cosine Decay)"); ax.set_xlabel("Epoch"); ax.grid(True, alpha=0.3)

    best_m50 = df["metrics/mAP50(B)"].max()
    plt.suptitle(f"YOLOv8n Training Summary (83 Epochs, Best mAP@50 = {best_m50:.4f})",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = "results/report_training_summary.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  [OK] {path}")


def generate_report_markdown():
    """Generate report data markdown (in Chinese for the course report)"""
    df = pd.read_csv("runs/mask_detect_yolov8n/results.csv")
    df.columns = df.columns.str.strip()

    best_m50 = df["metrics/mAP50(B)"].max()
    best_m50_ep = int(df.iloc[df["metrics/mAP50(B)"].idxmax()]["epoch"])
    best_m5095 = df["metrics/mAP50-95(B)"].max()
    best_p = df["metrics/precision(B)"].max()
    best_r = df["metrics/recall(B)"].max()

    report = f"""# Mask Detection — Experiment Report Data

## Experiment Environment
- OS: Windows 11
- GPU: NVIDIA GeForce RTX 4060 Laptop (8GB)
- CUDA: 12.4
- PyTorch: 2.6.0+cu124
- Framework: Ultralytics YOLOv8

## Dataset
- Total samples: 9,692 images
- Train: 6,784 (70%)
- Val: 1,938 (20%)
- Test: 970 (10%)
- Classes: without_mask / with_mask
- Total bounding boxes: 22,264 (without_mask: 14,973, with_mask: 7,291)

## Model Configuration
- Model: YOLOv8n (3,006,038 params, 8.1 GFLOPs)
- Optimizer: AdamW (lr=1e-3, weight_decay=5e-4)
- LR Schedule: CosineAnnealingLR
- Loss: Box Loss + Cls Loss + DFL Loss
- Batch Size: 16
- Image Size: 640x640
- Epochs: 100 (Early Stopping patience=15, stopped at epoch 83)
- Augmentation: Mosaic, MixUp, HSV, Flip, Scale, Translate, Rotate

## Results

### Best Validation Metrics
| Metric | Value | Epoch |
|--------|:-----:|:-----:|
| mAP@50 | {best_m50:.4f} | {best_m50_ep} |
| mAP@50-95 | {best_m5095:.4f} | {int(df["metrics/mAP50-95(B)"].idxmax())} |
| Best Precision | {best_p:.4f} | - |
| Best Recall | {best_r:.4f} | - |

### Test Set Results
| Class | AP@50 | Precision | Recall |
|-------|:-----:|:---------:|:------:|
| without_mask | 0.7184 | 0.786 | 0.752 |
| with_mask | 0.7972 | 0.849 | 0.827 |
| **Overall** | **0.7578** | **0.818** | **0.790** |

### Analysis
1. "with_mask" class achieves higher AP@50 (0.797) than "without_mask" (0.718)
2. Masked faces have more distinctive visual features for localization
3. "without_mask" class has more background confusion and face variation
4. Both classes have high precision (>0.78), indicating low false positive rate
5. mAP@50-95=0.487 suggests room for improvement at stricter IoU thresholds
"""
    path = "results/report_summary.md"
    Path(path).write_text(report, encoding="utf-8")
    print(f"  [OK] {path}")


def main():
    print("\n" + "=" * 60)
    print("  Supplementary Report Analysis")
    print("=" * 60 + "\n")

    print("[1/5] Dataset statistics...")
    plot_dataset_statistics()

    print("[2/5] Per-class metrics...")
    plot_per_class_metrics()

    print("[3/5] Confidence distribution...")
    plot_confidence_distribution()

    print("[4/5] Training summary curves...")
    plot_training_summary()

    print("[5/5] Report markdown...")
    generate_report_markdown()

    print(f"\n{'=' * 60}")
    print(f"  All figures saved to results/")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
