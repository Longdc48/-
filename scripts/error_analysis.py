#!/usr/bin/env python
"""
错误分析脚本 — 计算准确率、识别并可视化误分类样本

用法:
    python scripts/error_analysis.py
    python scripts/error_analysis.py --weights runs/mask_detect_yolov8n/weights/best.pt --max_samples 30
"""

import os
import sys
import argparse
import random
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml
from tqdm import tqdm
from ultralytics import YOLO

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# ── 中文字体 ────────────────────────────────────────────────────
try:
    plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
except Exception:
    pass
plt.rcParams["figure.dpi"] = 100

CLASS_NAMES = ["without_mask", "with_mask"]
CLASS_LABELS_CN = ["没戴口罩", "戴口罩"]
COLORS = {
    "FN": (0, 200, 255),    # 黄色 — 漏检
    "FP": (0, 0, 255),      # 红色 — 误检
    "confusion": (0, 140, 255),  # 橙色 — 类别混淆
}


def parse_args():
    parser = argparse.ArgumentParser(description="错误分析 — 准确率 + 误分类样本")
    parser.add_argument("--data", type=str, default="dataset_yolo/data.yaml")
    parser.add_argument("--weights", type=str,
                        default="runs/mask_detect_yolov8n/weights/best.pt")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="置信度阈值")
    parser.add_argument("--iou_thresh", type=float, default=0.45,
                        help="匹配用的 IoU 阈值")
    parser.add_argument("--max_samples", type=int, default=24,
                        help="可视化的最大误分类样本数")
    parser.add_argument("--device", type=str, default="0")
    return parser.parse_args()


def load_ground_truths(lbl_dir: Path) -> list[dict]:
    """加载某目录下所有 YOLO 格式标注，返回 ground truth 列表"""
    gts = []
    for txt_path in sorted(lbl_dir.glob("*.txt")):
        lines = [l.strip() for l in txt_path.read_text(encoding="utf-8").split("\n") if l.strip()]
        boxes = []
        for line in lines:
            parts = line.split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:5])
                boxes.append({"cls": cls_id, "bbox": (cx, cy, w, h)})
        if boxes:
            img_name = txt_path.with_suffix(".jpg").name
            gts.append({"image": img_name, "boxes": boxes, "txt_path": txt_path})
    return gts


def compute_iou(pred_xywh, gt_cx, gt_cy, gt_w, gt_h):
    """计算两个归一化 xywh 框的 IoU"""
    px1 = pred_xywh[0] - pred_xywh[2] / 2
    py1 = pred_xywh[1] - pred_xywh[3] / 2
    px2 = pred_xywh[0] + pred_xywh[2] / 2
    py2 = pred_xywh[1] + pred_xywh[3] / 2

    gx1 = gt_cx - gt_w / 2
    gy1 = gt_cy - gt_h / 2
    gx2 = gt_cx + gt_w / 2
    gy2 = gt_cy + gt_h / 2

    ix1, iy1 = max(px1, gx1), max(py1, gy1)
    ix2, iy2 = min(px2, gx2), min(py2, gy2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)

    area_p = pred_xywh[2] * pred_xywh[3]
    area_g = gt_w * gt_h
    union = area_p + area_g - inter
    return inter / union if union > 0 else 0


def analyze_errors(model, data_yaml: str, conf: float, iou_thresh: float):
    """在测试集上运行推理，匹配预测与真实框，统计各类错误"""
    with open(data_yaml, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    img_dir = Path(config["path"]) / "images" / "test"
    lbl_dir = Path(config["path"]) / "labels" / "test"

    gts = load_ground_truths(lbl_dir)
    if not gts:
        print("  [错误] 未找到测试集标注")
        return None

    # 按图片名建立索引
    gt_by_name = {g["image"]: g for g in gts}

    # ── 统计变量 ────────────────────────────────────────────────
    per_class = {0: {"tp": 0, "fp": 0, "fn": 0},
                 1: {"tp": 0, "fp": 0, "fn": 0}}
    confusion_count = 0  # 类别混淆

    # 收集误分类样本（用于可视化）
    error_samples = []  # 每项: {image_path, annotated_img, error_type, desc}

    test_images = sorted(img_dir.glob("*.jpg"))
    print(f"  测试集图片: {len(test_images)} 张\n")

    for img_path in tqdm(test_images, desc="分析错误"):
        img_name = img_path.name
        gt_data = gt_by_name.get(img_name)
        gt_boxes = gt_data["boxes"] if gt_data else []

        # 模型推理
        results = model(str(img_path), conf=conf, verbose=False)
        boxes = results[0].boxes

        pred_boxes = []
        if boxes is not None and len(boxes) > 0:
            xywhn = boxes.xywhn.cpu().numpy()
            cls_arr = boxes.cls.cpu().numpy()
            conf_arr = boxes.conf.cpu().numpy()
            for i in range(len(cls_arr)):
                pred_boxes.append({
                    "cls": int(cls_arr[i]),
                    "bbox": tuple(xywhn[i]),
                    "conf": float(conf_arr[i]),
                })

        # ── 匹配逻辑：贪心匹配（按置信度降序） ─────────────────
        matched_gt = set()
        matched_pred = set()

        # 按置信度降序排列预测框
        pred_sorted = sorted(enumerate(pred_boxes), key=lambda x: x[1]["conf"], reverse=True)

        for pi, pred in pred_sorted:
            best_iou = 0
            best_gi = -1
            for gi, gt in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                if gt["cls"] != pred["cls"]:
                    continue
                iou = compute_iou(pred["bbox"], *gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_gi = gi
            if best_iou >= iou_thresh:
                matched_pred.add(pi)
                matched_gt.add(best_gi)
                per_class[pred["cls"]]["tp"] += 1

        # 未匹配的预测 → FP
        img_has_error = False
        for pi, pred in enumerate(pred_boxes):
            if pi not in matched_pred:
                # 检查是否和某个不同类的 GT 有高 IoU（类别混淆）
                confused_with = -1
                best_confuse_iou = 0
                for gi, gt in enumerate(gt_boxes):
                    if gi in matched_gt:
                        continue
                    if gt["cls"] == pred["cls"]:
                        continue
                    iou = compute_iou(pred["bbox"], *gt["bbox"])
                    if iou >= iou_thresh and iou > best_confuse_iou:
                        best_confuse_iou = iou
                        confused_with = gi

                if confused_with >= 0:
                    # 类别混淆
                    confusion_count += 1
                    matched_gt.add(confused_with)
                    gt_cls = gt_boxes[confused_with]["cls"]
                    per_class[gt_cls]["tp"] += 1  # GT 还是被检测到了，只是类别错了
                    error_type = "confusion"
                    desc = (f"类别混淆: 预测={CLASS_LABELS_CN[pred['cls']]} "
                            f"({pred['conf']:.2f}), "
                            f"真实={CLASS_LABELS_CN[gt_cls]}")
                else:
                    # 真正的 FP
                    per_class[pred["cls"]]["fp"] += 1
                    error_type = "FP"
                    desc = (f"假正例(FP): 预测={CLASS_LABELS_CN[pred['cls']]} "
                            f"({pred['conf']:.2f}), 此处无对应目标")

                if len(error_samples) < args_max * 3:
                    img_has_error = True
                    error_samples.append({
                        "image_path": img_path,
                        "error_type": error_type,
                        "desc": desc,
                        "pred": pred,
                        "gt": None if confused_with < 0 else gt_boxes[confused_with],
                    })

        # 未匹配的真实框 → FN
        for gi, gt in enumerate(gt_boxes):
            if gi not in matched_gt:
                per_class[gt["cls"]]["fn"] += 1
                if len(error_samples) < args_max * 3:
                    img_has_error = True
                    error_samples.append({
                        "image_path": img_path,
                        "error_type": "FN",
                        "desc": f"假负例(FN): 漏检了 {CLASS_LABELS_CN[gt['cls']]}",
                        "pred": None,
                        "gt": gt,
                    })

    result = {
        "per_class": per_class,
        "confusion_count": confusion_count,
        "error_samples": error_samples,
        "total_images": len(test_images),
    }
    return result


def draw_error_visualization(image: np.ndarray, sample: dict) -> np.ndarray:
    """在图片上绘制误分类标注"""
    annotated = image.copy()
    h, w = annotated.shape[:2]

    error_type = sample["error_type"]
    color = COLORS.get(error_type, (128, 128, 128))

    if sample["pred"] is not None:
        pred = sample["pred"]
        cx, cy, bw, bh = pred["bbox"]
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        thickness = 3
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)
        label = f"Pred: {CLASS_LABELS_CN[pred['cls']]} ({pred['conf']:.2f})"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    if sample["gt"] is not None:
        gt = sample["gt"]
        cx, cy, bw, bh = gt["bbox"]
        x1 = int((cx - bw / 2) * w)
        y1 = int((cy - bh / 2) * h)
        x2 = int((cx + bw / 2) * w)
        y2 = int((cy + bh / 2) * h)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        cv2.rectangle(annotated, (x1, y1), (x2, y2), COLORS["FN"], 2)
        label = f"GT: {CLASS_LABELS_CN[gt['cls']]}"
        cv2.putText(annotated, label, (x1 + 3, y2 - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS["FN"], 1, cv2.LINE_AA)

    return annotated


def build_error_mosaic(error_samples: list[dict], output_dir: str, max_show: int = 24):
    """将误分类样本拼接成可视化大图"""
    if not error_samples:
        print("  [信息] 没有误分类样本可显示")
        return

    # 限制数量，各类别均衡采样
    fp_samples = [s for s in error_samples if s["error_type"] == "FP"]
    fn_samples = [s for s in error_samples if s["error_type"] == "FN"]
    cf_samples = [s for s in error_samples if s["error_type"] == "confusion"]

    # 每种类型最多取 max_show/3 个
    each = max(1, max_show // 3)
    sampled = []
    sampled.extend(fp_samples[:each])
    sampled.extend(fn_samples[:each])
    sampled.extend(cf_samples[:each])
    random.shuffle(sampled)
    sampled = sampled[:max_show]

    n = len(sampled)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten() if rows > 1 else ([axes] if cols == 1 else axes)

    for i, sample in enumerate(sampled):
        img = cv2.imread(str(sample["image_path"]))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        annotated = draw_error_visualization(img, sample)

        ax = axes[i]
        ax.imshow(annotated)
        ax.set_title(sample["desc"], fontsize=8, color={
            "FP": "red", "FN": "darkorange", "confusion": "darkviolet"
        }.get(sample["error_type"], "black"))
        ax.axis("off")

    # 隐藏多余子图
    for j in range(n, len(axes)):
        axes[j].axis("off")

    error_type_counts = {
        "FP": len(fp_samples),
        "FN": len(fn_samples),
        "类别混淆": len(cf_samples),
    }
    title = (f"误分类样本分析 — 共 {len(error_samples)} 个错误 "
             f"(FP: {error_type_counts['FP']}, "
             f"FN: {error_type_counts['FN']}, "
             f"类别混淆: {error_type_counts['类别混淆']})")
    fig.suptitle(title, fontsize=12, fontweight="bold", y=0.98)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "misclassified_samples.png")
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"\n  [OK] 误分类样本图: {save_path} (展示 {n}/{len(error_samples)} 个)")
    return error_type_counts


def print_accuracy_report(analysis: dict):
    """打印准确率报告"""
    pc = analysis["per_class"]

    print(f"\n{'='*60}")
    print(f"  错误分析 & 准确率报告")
    print(f"{'='*60}")

    total_tp = 0
    total_fp = 0
    total_fn = 0
    per_class_accuracy = {}

    for cls_id in sorted(pc.keys()):
        tp = pc[cls_id]["tp"]
        fp = pc[cls_id]["fp"]
        fn = pc[cls_id]["fn"]
        total_tp += tp
        total_fp += fp
        total_fn += fn

        acc = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        per_class_accuracy[cls_id] = acc

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0

        print(f"\n  类别 {cls_id} — {CLASS_LABELS_CN[cls_id]}:")
        print(f"    TP: {tp:>6}   FP: {fp:>6}   FN: {fn:>6}")
        print(f"    准确率 (Accuracy):  {acc:.4f}  ({acc*100:.1f}%)")
        print(f"    精确率 (Precision): {prec:.4f}")
        print(f"    召回率 (Recall):    {rec:.4f}")

    overall_acc = total_tp / (total_tp + total_fp + total_fn) if (total_tp + total_fp + total_fn) > 0 else 0
    print(f"\n  {'─'*50}")
    print(f"  整体统计:")
    print(f"    总 TP: {total_tp}   总 FP: {total_fp}   总 FN: {total_fn}")
    print(f"    类别混淆: {analysis['confusion_count']}")
    print(f"    整体准确率: {overall_acc:.4f} ({overall_acc*100:.1f}%)")

    return per_class_accuracy, overall_acc


def main():
    global args_max
    args = parse_args()
    args_max = args.max_samples

    print(f"\n{'='*60}")
    print(f"  口罩检测 — 错误分析")
    print(f"{'='*60}")
    print(f"  模型: {args.weights}")
    print(f"  置信度阈值: {args.conf}")
    print(f"  IoU 阈值: {args.iou_thresh}")

    # ── 加载模型 ────────────────────────────────────────────────
    model = YOLO(args.weights)

    # ── 错误分析 ────────────────────────────────────────────────
    analysis = analyze_errors(model, args.data, args.conf, args.iou_thresh)
    if analysis is None:
        return

    # ── 打印准确率报告 ──────────────────────────────────────────
    per_class_acc, overall_acc = print_accuracy_report(analysis)

    # ── 生成误分类可视化 ────────────────────────────────────────
    error_counts = build_error_mosaic(
        analysis["error_samples"], args.output_dir, args.max_samples
    )

    # ── 保存统计分析 CSV ────────────────────────────────────────
    csv_path = os.path.join(args.output_dir, "error_analysis.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("error_type,count\n")
        if error_counts:
            for etype, count in error_counts.items():
                f.write(f"{etype},{count}\n")
        f.write(f"total_images,{analysis['total_images']}\n")
        f.write(f"overall_accuracy,{overall_acc:.4f}\n")
        for cls_id, acc in per_class_acc.items():
            f.write(f"accuracy_{CLASS_NAMES[cls_id]},{acc:.4f}\n")
    print(f"  [OK] 统计数据: {csv_path}")

    print(f"\n{'='*60}")
    print(f"  错误分析完成！")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
