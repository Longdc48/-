#!/usr/bin/env python
"""
数据集预处理脚本 — 合并 all_mask + new_mask_data，按 YOLO 格式划分 train/val/test

输出结构:
    dataset_yolo/
    ├── images/
    │   ├── train/          # 70%
    │   ├── val/            # 20%
    │   └── test/           # 10%
    ├── labels/
    │   ├── train/          # YOLO .txt 标注
    │   ├── val/
    │   └── test/
    └── data.yaml           # YOLOv8 配置
"""

import os
import sys
import shutil
import random
import argparse
from pathlib import Path

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def parse_args():
    parser = argparse.ArgumentParser(description="准备 YOLO 目标检测数据集")
    parser.add_argument("--source_dirs", type=str, nargs="+",
                        default=["dataset/all_mask", "dataset/new_mask_data"],
                        help="原始数据集目录")
    parser.add_argument("--output_dir", type=str, default="dataset_yolo",
                        help="输出目录")
    parser.add_argument("--train_ratio", type=float, default=0.70,
                        help="训练集比例")
    parser.add_argument("--val_ratio", type=float, default=0.20,
                        help="验证集比例")
    parser.add_argument("--seed", type=int, default=42,
                        help="随机种子")
    parser.add_argument("--skip_empty", action="store_true", default=True,
                        help="跳过空标注文件")
    parser.add_argument("--min_boxes", type=int, default=0,
                        help="最少标注框数量（0=不过滤）")
    return parser.parse_args()


def collect_samples(source_dirs: list[str], skip_empty: bool = True) -> list[dict]:
    """收集所有 (图片, 标注) 对"""
    samples = []
    for src_dir in source_dirs:
        src = Path(src_dir)
        if not src.exists():
            print(f"  [警告] 目录不存在: {src}")
            continue
        jpgs = sorted(src.glob("*.jpg"))
        for jpg in jpgs:
            txt = jpg.with_suffix(".txt")
            if not txt.exists():
                continue
            content = txt.read_text(encoding="utf-8").strip()
            lines = [l for l in content.split("\n") if l.strip()]
            if skip_empty and not lines:
                continue
            # 使用前缀避免重名
            prefix = src.name + "_"
            samples.append({
                "image": jpg,
                "label": txt,
                "lines": lines,
                "prefix": prefix,
                "source": src.name,
            })
    return samples


def main():
    args = parse_args()
    random.seed(args.seed)

    # ── 收集样本 ────────────────────────────────────────────────
    print(f"\n  收集数据...")
    samples = collect_samples(args.source_dirs, skip_empty=args.skip_empty)
    print(f"  有效样本数: {len(samples)}")

    # ── 按标注框数过滤 ──────────────────────────────────────────
    if args.min_boxes > 0:
        samples = [s for s in samples if len(s["lines"]) >= args.min_boxes]
        print(f"  过滤后样本数 (≥{args.min_boxes} boxes): {len(samples)}")

    # ── 统计类别分布 ────────────────────────────────────────────
    from collections import Counter
    class_counter = Counter()
    box_counter = Counter()
    for s in samples:
        classes_in_img = set()
        for line in s["lines"]:
            parts = line.split()
            if parts:
                cls_id = int(parts[0])
                box_counter[cls_id] += 1
                classes_in_img.add(cls_id)
        for c in classes_in_img:
            class_counter[c] += 1

    print(f"\n  类别分布 (按图片):")
    for cls_id in sorted(class_counter):
        print(f"    class_{cls_id}: {class_counter[cls_id]} 张图片")
    print(f"  标注框分布:")
    for cls_id in sorted(box_counter):
        print(f"    class_{cls_id}: {box_counter[cls_id]} 个框")

    # ── 打乱 ────────────────────────────────────────────────────
    random.shuffle(samples)
    n = len(samples)
    train_end = int(n * args.train_ratio)
    val_end = train_end + int(n * args.val_ratio)

    splits = {
        "train": samples[:train_end],
        "val":   samples[train_end:val_end],
        "test":  samples[val_end:],
    }

    print(f"\n  划分结果: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    # ── 复制文件到 YOLO 目录结构 ────────────────────────────────
    output = Path(args.output_dir)
    for split_name, split_samples in splits.items():
        img_dir = output / "images" / split_name
        lbl_dir = output / "labels" / split_name
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for s in split_samples:
            new_name = s["prefix"] + s["image"].name
            # 复制图片
            dst_img = img_dir / new_name
            if not dst_img.exists():
                shutil.copy2(s["image"], dst_img)
            # 复制标注（保持原始 class_id，不做修改）
            dst_lbl = lbl_dir / (new_name.rsplit(".", 1)[0] + ".txt")
            dst_lbl.write_text("\n".join(s["lines"]), encoding="utf-8")

        print(f"  [{split_name}] {len(split_samples)} 样本 → {img_dir}")

    # ── 生成 data.yaml ──────────────────────────────────────────
    # 推断类别名称
    all_classes = sorted(box_counter.keys())
    names = {}
    for cls_id in all_classes:
        if cls_id == 0:
            names[cls_id] = "without_mask"
        elif cls_id == 1:
            names[cls_id] = "with_mask"
        else:
            names[cls_id] = f"class_{cls_id}"

    data_yaml = {
        "path": str(output.absolute()),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "nc": len(all_classes),
        "names": [names[i] for i in sorted(all_classes)],
    }

    yaml_path = output / "data.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, allow_unicode=True, sort_keys=False)

    print(f"\n  [✓] data.yaml 已生成: {yaml_path}")
    print(f"  类别: {data_yaml['names']}")
    print(f"  类别数: {data_yaml['nc']}")

    # ── 打印目录树 ──────────────────────────────────────────────
    print(f"\n  目录结构:")
    for root, dirs, files in os.walk(output):
        level = root.replace(str(output), "").count(os.sep)
        indent = "    " * level
        print(f"  {indent}{os.path.basename(root)}/")
        if level < 3:
            subindent = "    " * (level + 1)
            img_count = len([f for f in files if f.endswith(".jpg")])
            lbl_count = len([f for f in files if f.endswith(".txt")])
            print(f"  {subindent}({img_count} jpg, {lbl_count} txt)")

    print(f"\n{'='*60}")
    print(f"  数据准备完成！")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
