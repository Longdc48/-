#!/usr/bin/env python
"""
口罩检测批量推理演示 — 对指定目录下的图片进行口罩检测并汇总结果

用法:
    python scripts/inference_demo.py                           # 默认检测 results/detection_samples/ 下的图片
    python scripts/inference_demo.py --dir test_photos         # 检测指定目录
    python scripts/inference_demo.py --dir photo1.jpg photo2.jpg  # 检测指定图片
"""

import os
import sys
import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# ── 项目根目录 ────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── 常量 ──────────────────────────────────────────────────────────
CLASS_LABELS_CN = ["没戴口罩", "戴口罩"]
CLASS_COLORS = {0: (0, 0, 255), 1: (0, 255, 0)}    # 红 / 绿


def find_model_weights():
    """自动搜索项目中的 best.pt"""
    for pattern in ["runs/mask_detect_*/weights/best.pt", "runs/**/best.pt"]:
        candidates = list(PROJECT_ROOT.glob(pattern))
        if candidates:
            return str(candidates[0])
    if (PROJECT_ROOT / "best.pt").exists():
        return str(PROJECT_ROOT / "best.pt")
    return None


def parse_args():
    auto_weights = find_model_weights()
    parser = argparse.ArgumentParser(description="口罩检测批量推理演示")
    parser.add_argument("--dir", type=str, nargs="+",
                        default=["results/mask_detect_yolov8n_detection_samples"],
                        help="图片目录或图片路径列表")
    parser.add_argument("--weights", type=str, default=auto_weights,
                        help="模型权重路径")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="置信度阈值")
    parser.add_argument("--output", type=str, default="inference_results",
                        help="结果保存目录")
    return parser.parse_args()


def collect_images(sources: list[str]) -> list[Path]:
    """从目录或文件列表中收集所有图片"""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    images = []
    for src in sources:
        p = Path(src)
        if p.is_dir():
            for ext in exts:
                images.extend(p.glob(f"*{ext}"))
                images.extend(p.glob(f"*{ext.upper()}"))
        elif p.is_file() and p.suffix.lower() in exts:
            images.append(p)
    return sorted(set(images))


def draw_boxes(image: np.ndarray, results) -> np.ndarray:
    """在图片上绘制检测框"""
    annotated = image.copy()
    boxes = results.boxes
    if boxes is None or len(boxes) == 0:
        return annotated

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        color = CLASS_COLORS.get(cls_id, (255, 255, 255))
        label = f"{CLASS_LABELS_CN[cls_id]} {conf:.2f}"

        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated, label, (x1 + 3, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return annotated


def main():
    args = parse_args()

    # ── 加载模型 ──────────────────────────────────────────────────
    if not args.weights or not os.path.exists(args.weights):
        print(f"[错误] 模型权重未找到，请先训练或指定 --weights")
        sys.exit(1)

    print(f"模型: {args.weights}")
    model = YOLO(args.weights)
    model.to("0")

    # ── 收集图片 ──────────────────────────────────────────────────
    images = collect_images(args.dir)
    if not images:
        print(f"[错误] 未找到图片: {args.dir}")
        sys.exit(1)

    print(f"找到 {len(images)} 张图片\n")

    # ── 批量推理 ──────────────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_faces = 0
    total_no_mask = 0
    total_with_mask = 0

    for i, img_path in enumerate(images, 1):
        results = model(str(img_path), conf=args.conf, verbose=False)
        boxes = results[0].boxes

        n_total = len(boxes) if boxes else 0
        n_no = sum(1 for b in boxes if int(b.cls[0]) == 0) if boxes else 0
        n_yes = n_total - n_no

        total_faces += n_total
        total_no_mask += n_no
        total_with_mask += n_yes

        # 打印单张结果
        status = "✅" if n_no == 0 and n_total > 0 else ("⚠️" if n_no > 0 else "⚪")
        print(f"  [{i:>2}/{len(images)}] {status} {img_path.name:<50s} "
              f"人脸:{n_total}  没戴:{n_no}  戴:{n_yes}")

        # 保存标注图片
        annotated = draw_boxes(cv2.imread(str(img_path)), results[0])
        cv2.imwrite(str(output_dir / f"{img_path.stem}_detected.jpg"), annotated)

    # ── 汇总 ──────────────────────────────────────────────────────
    print(f"\n{'='*55}")
    print(f"  检测汇总")
    print(f"{'='*55}")
    print(f"  处理图片:     {len(images)} 张")
    print(f"  检测到人脸:   {total_faces} 个")
    print(f"    ❌ 没戴口罩: {total_no_mask} 人")
    print(f"    ✅ 戴口罩:   {total_with_mask} 人")
    if total_faces > 0:
        rate = total_with_mask / total_faces * 100
        print(f"  📊 口罩佩戴率: {rate:.1f}%")
    print(f"{'='*55}")
    print(f"  标注结果已保存至: {output_dir}/\n")


if __name__ == "__main__":
    main()
