import os
import json
import cv2


def yolo_dataset_to_coco(dataset_dir, output_json_path=None):
    """Reads a YOLO-format dataset (images/, labels/, classes.txt) produced
    by the detection labeling tab and converts it into a single COCO-style
    JSON annotation file. Category ids match the YOLO class ids (0-indexed).

    Returns: (output_json_path, num_images, num_annotations)
    """
    images_dir = os.path.join(dataset_dir, "images")
    labels_dir = os.path.join(dataset_dir, "labels")
    classes_path = os.path.join(dataset_dir, "classes.txt")

    if not os.path.isdir(images_dir) or not os.path.isdir(labels_dir):
        raise FileNotFoundError("ไม่พบโฟลเดอร์ images/ หรือ labels/ ใน dataset ที่เลือก")

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

    img_files = sorted(
        f for f in os.listdir(images_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    )

    for img_id, fname in enumerate(img_files, start=1):
        img_path = os.path.join(images_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        images.append({"id": img_id, "file_name": fname, "width": w, "height": h})

        label_path = os.path.join(labels_dir, os.path.splitext(fname)[0] + ".txt")
        if not os.path.isfile(label_path):
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
