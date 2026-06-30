# 口罩检测系统 — YOLOv8 目标检测

> 基于 YOLOv8 的口罩佩戴目标检测系统 —— 深度学习课程设计

## 项目概述

使用 YOLOv8n 深度学习模型，实现图片中人脸的口罩佩戴状态检测。支持两类目标：

| Class | Label | 说明 |
|:-----:|-------|------|
| 0 | without_mask | 未佩戴口罩 |
| 1 | with_mask | 佩戴口罩 |

## 项目结构

```text
├── scripts/
│   ├── prepare_data.py          # 数据预处理 & YOLO 格式划分
│   ├── train_yolo.py            # YOLOv8 训练入口
│   ├── evaluate_yolo.py         # 评估 & 报告生成（含图表分析）
│   ├── error_analysis.py        # 错误分析（准确率 + 误分类样本）
│   └── inference_demo.py        # 批量推理演示
├── results/                     # 实验结果与图表
│   ├── report_summary.md        # 课程设计实验报告
│   ├── misclassified_samples.png # 误分类样本可视化
│   └── *.png                    # 各类分析图表
├── runs/mask_detect_yolov8n/    # 训练产物（权重、日志）
├── requirements.txt
└── README.md
```

## 环境配置

| 组件 | 详情 |
|------|------|
| OS | Windows 11 Home |
| GPU | NVIDIA GeForce RTX 4060 Laptop (8 GB) |
| CUDA | 12.4 |
| Python | 3.10.20 (conda: torch_env) |
| PyTorch | 2.6.0+cu124 |

```bash
conda activate torch_env
pip install -r requirements.txt
```

验证 GPU 可用：
```bash
python -c "import torch; print(torch.cuda.is_available())"  # 应输出 True
python -c "import torch; print(torch.cuda.get_device_name(0))"  # 应显示 RTX 4060
```

## 快速开始

### 1. 数据准备

```bash
python scripts/prepare_data.py
```

自动合并 `dataset/all_mask/` + `dataset/new_mask_data/`，按 70/20/10 划分 train/val/test。

### 2. 训练

```bash
# YOLOv8n（推荐）
python scripts/train_yolo.py --model yolov8n.pt --epochs 100 --batch 16

# 更大模型
python scripts/train_yolo.py --model yolov8s.pt --epochs 100 --batch 8
```

### 3. 评估 & 报告

```bash
# 完整评估 + 报告图表
python scripts/evaluate_yolo.py --weights runs/mask_detect_yolov8n/weights/best.pt --report
```

### 4. 错误分析

```bash
# 计算准确率 + 生成误分类样本图
python scripts/error_analysis.py
```

### 5. 推理演示

```bash
python scripts/inference_demo.py --dir your_photos/
```

## 模型性能

### 测试集结果

| 类别 | AP@50 | Precision | Recall |
|------|:-----:|:---------:|:------:|
| without_mask | 0.718 | 0.786 | 0.752 |
| with_mask | 0.797 | 0.849 | 0.827 |
| **Overall** | **0.758** | **0.818** | **0.790** |

### 训练最佳（验证集）

| Metric | Best | Epoch |
|--------|:----:|:-----:|
| mAP@50 | 0.803 | 72 |
| mAP@50-95 | 0.511 | 68 |

## 数据集

| Split | Images | Ratio |
|-------|:------:|:-----:|
| Train | 6,784 | 70% |
| Val | 1,938 | 20% |
| Test | 970 | 10% |
| **Total** | **9,692** | — |

## 技术要点

- **框架**: PyTorch 2.6 + Ultralytics YOLOv8
- **模型**: YOLOv8n (3,006,038 参数, 8.1 GFLOPs)
- **优化器**: AdamW + CosineAnnealingLR
- **损失**: Box Loss + Cls Loss + DFL Loss
- **增强**: Mosaic, MixUp, HSV, Flip, Scale, Translate, Rotate
- **早停**: patience=15 (实际 83/100 epoch 停止)

## 课程报告

完整报告见 [results/report_summary.md](results/report_summary.md)，包含系统架构、数据集分析、实验结果、混淆矩阵、准确率/精确率/召回率、损失函数分析、错误分类样本分析等。
