# Mask Detection System — YOLOv8 Object Detection

> 基于 YOLOv8 的口罩佩戴目标检测系统课程设计

## Overview

本系统使用 YOLOv8n 深度学习模型，实现图片中人脸的口罩佩戴状态检测。支持两类目标：

| Class | Label | Description |
|:-----:|-------|-------------|
| 0 | `without_mask` | 未佩戴口罩 |
| 1 | `with_mask` | 佩戴口罩 |

## Project Structure

```text
├── scripts/                              # 核心脚本
│   ├── prepare_data.py                   # 数据预处理 & YOLO 格式划分
│   ├── train_yolo.py                     # YOLOv8 训练入口
│   ├── evaluate_yolo.py                  # 评估 & 可视化
│   └── report_analysis.py                # 报告补充分析
├── results/                              # 实验结果与图表
│   ├── report_summary.md                 # 课程报告（含 Mermaid 架构图）
│   ├── report_dataset_statistics.png     # 数据集统计分析
│   ├── report_per_class_metrics.png      # 各类别性能对比
│   ├── report_confidence_distribution.png # 置信度分布分析
│   ├── report_training_summary.png       # 训练全貌 6 合 1
│   ├── mask_detect_yolov8n_confusion_matrix.png     # 混淆矩阵
│   ├── mask_detect_yolov8n_BoxPR_curve.png       # PR 曲线
│   ├── mask_detect_yolov8n_BoxF1_curve.png       # F1 曲线
│   ├── mask_detect_yolov8n_training_curves.png   # 损失曲线
│   └── mask_detect_yolov8n_detection_samples/    # 检测结果样本
├── runs/mask_detect_yolov8n/              # 训练产物
│   ├── weights/best.pt                   # 最佳模型权重
│   ├── results.csv                       # 完整训练日志
│   ├── results.png                       # 训练结果总览
│   ├── train_batch0.jpg                  # 训练 batch 样本
│   └── val_batch0_pred.jpg               # 验证集预测对比
├── requirements.txt                      # Python 依赖
├── ENVIRONMENT.md                        # 硬件 & 环境说明
└── README.md
```

## Quick Start

### 1. Environment

```bash
conda activate torch_env
pip install -r requirements.txt
```

### 2. Prepare Data

```bash
# 自动合并 all_mask + new_mask_data，按 70/20/10 划分 train/val/test
python scripts/prepare_data.py
```

### 3. Train

```bash
# YOLOv8n (baseline)
python scripts/train_yolo.py --model yolov8n.pt --epochs 100 --batch 16

# YOLOv8s (higher accuracy)
python scripts/train_yolo.py --model yolov8s.pt --epochs 100 --batch 8
```

### 4. Evaluate

```bash
python scripts/evaluate_yolo.py --weights runs/mask_detect_yolov8n/weights/best.pt
```

### 5. Generate Report Figures

```bash
python scripts/report_analysis.py
```

## Model Performance

### Test Set Results

| Class | AP@50 | Precision | Recall |
|-------|:-----:|:---------:|:------:|
| without_mask | 0.718 | 0.786 | 0.752 |
| with_mask | 0.797 | 0.849 | 0.827 |
| **Overall** | **0.758** | **0.818** | **0.790** |

### Training Summary

| Metric | Best Value | Epoch |
|--------|:----------:|:-----:|
| mAP@50 (val) | 0.803 | 72 |
| mAP@50-95 (val) | 0.511 | 68 |
| Total Epochs | 83 (early stop) | — |

## Dataset

| Split | Images | Ratio |
|-------|:------:|:-----:|
| Train | 6,784 | 70% |
| Val | 1,938 | 20% |
| Test | 970 | 10% |
| **Total** | **9,692** | — |

Source: `all_mask/` (9,240) + `new_mask_data/` (577), after filtering empty annotations.

## Technical Details

- **Framework**: PyTorch 2.6 + Ultralytics YOLOv8
- **Model**: YOLOv8n (3,006,038 params, 8.1 GFLOPs)
- **GPU**: NVIDIA GeForce RTX 4060 Laptop (8 GB)
- **CUDA**: 12.4
- **Optimizer**: AdamW (lr=1e-3) + CosineAnnealingLR
- **Augmentation**: Mosaic, MixUp, HSV, Flip, Scale, Translate, Rotate

## Report

See [results/report_summary.md](results/report_summary.md) for the complete course design report with system architecture diagrams, experimental data, and analysis.
