import os
import json
import cv2

from utils.train_val_split import collect_detect_pairs
from utils.yolo_format import read_yolo_seg_label


def yolo_dataset_to_coco(dataset_dir, output_json_path=None):
    """Reads a YOLO-format dataset (images/, labels/, classes.txt) produced
    by the detection labeling tab and converts it into a single COCO-style
    JSON annotation file. Category ids match the YOLO class ids (0-indexed).

    รองรับทั้ง dataset แบบ flat (images/) และแบบที่แยก train/val แล้ว
    (train/images, val/images) - กรณีหลัง file_name จะเก็บเป็น
    train/images/xxx.jpg เพื่อไม่ให้ชื่อไฟล์ซ้ำกันข้ามฝั่ง

    Returns: (output_json_path, num_images, num_annotations)
    """
    classes_path = os.path.join(dataset_dir, "classes.txt")

    # ใช้ตัวรวบรวมเดียวกับตอนแยก train/val จึงเห็นภาพครบทั้งแบบ flat และแบบแยกแล้ว
    pairs = collect_detect_pairs(dataset_dir)
    if not pairs:
        raise FileNotFoundError(
            "ไม่พบภาพใน dataset ที่เลือก (ควรมี images/ หรือ train/images + val/images)"
        )

    class_names = []
    if os.path.isfile(classes_path):
        with open(classes_path, "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f if line.strip()]

    categories = [
        {"id": i, "name": name, "supercategory": "none"}
        for i, name in enumerate(class_names)
    ]

    images = []
    annotations = []
    ann_id = 1

    for img_id, stem in enumerate(sorted(pairs), start=1):
        img_path, label_path = pairs[stem]
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        # ชื่อไฟล์อ้างอิงจากรากของ dataset เพื่อให้รู้ว่าอยู่ฝั่ง train หรือ val
        fname = os.path.relpath(img_path, dataset_dir).replace(os.sep, "/")
        images.append({"id": img_id, "file_name": fname, "width": w, "height": h})

        if not label_path or not os.path.isfile(label_path):
            continue

        with open(label_path, "r", encoding="utf-8") as lf:
            for line in lf:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 5:
                    continue
                class_id = int(parts[0])
                xc, yc, bw, bh = (float(v) for v in parts[1:5])
                box_w = bw * w
                box_h = bh * h
                x = xc * w - box_w / 2
                y = yc * h - box_h / 2
                annotations.append({
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": class_id,
                    "bbox": [round(x, 2), round(y, 2), round(box_w, 2), round(box_h, 2)],
                    "area": round(box_w * box_h, 2),
                    "iscrowd": 0,
                })
                ann_id += 1

    coco = {"images": images, "annotations": annotations, "categories": categories}

    if output_json_path is None:
        output_json_path = os.path.join(dataset_dir, "annotations_coco.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    return output_json_path, len(images), len(annotations)


def _polygon_area(points):
    """พื้นที่ polygon ด้วยสูตร shoelace (คืนค่าสัมบูรณ์ ไม่สนทิศทางการวน)"""
    area = 0.0
    n = len(points)
    for i in range(n):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def yolo_seg_dataset_to_coco(dataset_dir, output_json_path=None):
    """แปลง dataset แบบ YOLO-seg (images/ + labels/ ที่ label เป็น polygon)
    ให้เป็นไฟล์ COCO JSON เดียว

    ต่างจาก yolo_dataset_to_coco ตรงที่ annotation แต่ละอันจะมีคีย์
    "segmentation" เป็น [[x1, y1, x2, y2, ...]] หน่วยพิกเซล ส่วน "bbox"
    คำนวณจากกรอบนอกสุดของ polygon และ "area" ใช้สูตร shoelace ของ polygon
    จริง (ไม่ใช่พื้นที่กรอบสี่เหลี่ยม)

    รองรับทั้ง dataset แบบ flat และแบบที่แยก train/val แล้ว

    Returns: (output_json_path, num_images, num_annotations)
    """
    classes_path = os.path.join(dataset_dir, "classes.txt")

    pairs = collect_detect_pairs(dataset_dir)
    if not pairs:
        raise FileNotFoundError(
            "ไม่พบภาพใน dataset ที่เลือก (ควรมี images/ หรือ train/images + val/images)"
        )

    class_names = []
    if os.path.isfile(classes_path):
        with open(classes_path, "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f if line.strip()]

    categories = [
        {"id": i, "name": name, "supercategory": "none"}
        for i, name in enumerate(class_names)
    ]

    images = []
    annotations = []
    ann_id = 1

    for img_id, stem in enumerate(sorted(pairs), start=1):
        img_path, label_path = pairs[stem]
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        fname = os.path.relpath(img_path, dataset_dir).replace(os.sep, "/")
        images.append({"id": img_id, "file_name": fname, "width": w, "height": h})

        if not label_path or not os.path.isfile(label_path):
            continue

        for class_id, points in read_yolo_seg_label(label_path, w, h):
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            x, y = min(xs), min(ys)
            box_w, box_h = max(xs) - x, max(ys) - y
            flat = []
            for px, py in points:
                flat.append(round(px, 2))
                flat.append(round(py, 2))
            annotations.append({
                "id": ann_id,
                "image_id": img_id,
                "category_id": class_id,
                "segmentation": [flat],
                "bbox": [round(x, 2), round(y, 2), round(box_w, 2), round(box_h, 2)],
                "area": round(_polygon_area(points), 2),
                "iscrowd": 0,
            })
            ann_id += 1

    coco = {"images": images, "annotations": annotations, "categories": categories}

    if output_json_path is None:
        output_json_path = os.path.join(dataset_dir, "annotations_coco.json")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(coco, f, ensure_ascii=False, indent=2)

    return output_json_path, len(images), len(annotations)
