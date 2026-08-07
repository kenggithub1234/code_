import os


def save_yolo_label(label_path, boxes, img_w, img_h):
    """boxes: list of (class_id, x1, y1, x2, y2) in pixel coordinates.
    Writes a YOLO-format .txt label file:
        class_id x_center y_center width height   (all normalized 0-1)
    """
    lines = []
    for class_id, x1, y1, x2, y2 in boxes:
        xc = (x1 + x2) / 2 / img_w
        yc = (y1 + y2) / 2 / img_h
        w = abs(x2 - x1) / img_w
        h = abs(y2 - y1) / img_h
        lines.append(f"{class_id} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_yolo_seg_label(label_path, polygons, img_w, img_h):
    """polygons: list of (class_id, [(x, y), ...]) in pixel coordinates.
    Writes a YOLO-seg format .txt label file:
        class_id x1 y1 x2 y2 ... xn yn   (all normalized 0-1)
    """
    lines = []
    for class_id, points in polygons:
        coords = []
        for x, y in points:
            coords.append(f"{x / img_w:.6f}")
            coords.append(f"{y / img_h:.6f}")
        lines.append(f"{class_id} " + " ".join(coords))
    with open(label_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def read_yolo_seg_label(label_path, img_w, img_h):
    """อ่านไฟล์ label แบบ YOLO-seg กลับมาเป็นพิกัดพิกเซล

    คืน list ของ (class_id, [(x, y), ...]) โดย x, y เป็นพิกเซลจริงในภาพ
    บรรทัดที่จำนวนตัวเลขไม่ครบคู่ หรือมีจุดน้อยกว่า 3 จุด จะถูกข้าม
    """
    polygons = []
    if not os.path.isfile(label_path):
        return polygons
    with open(label_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 7:  # class + อย่างน้อย 3 จุด (6 ตัวเลข)
                continue
            try:
                class_id = int(float(parts[0]))
                coords = [float(v) for v in parts[1:]]
            except ValueError:
                continue
            if len(coords) % 2:
                coords = coords[:-1]
            points = [
                (coords[i] * img_w, coords[i + 1] * img_h)
                for i in range(0, len(coords), 2)
            ]
            if len(points) >= 3:
                polygons.append((class_id, points))
    return polygons


def save_classes_file(classes_path, class_names):
    with open(classes_path, "w", encoding="utf-8") as f:
        f.write("\n".join(class_names))


def save_data_yaml(yaml_path, dataset_root, class_names, split=False):
    """split=False: train/val ชี้ที่ images/ เดียวกัน (ยังไม่ได้แยกข้อมูล)
    split=True:  ชี้ที่ train/images และ val/images ที่แยกไว้แล้ว
    """
    names_list = "[" + ", ".join(f"'{c}'" for c in class_names) + "]"
    train_dir = "train/images" if split else "images"
    val_dir = "val/images" if split else "images"
    content = (
        f"path: {dataset_root}\n"
        f"train: {train_dir}\n"
        f"val: {val_dir}\n\n"
        f"nc: {len(class_names)}\n"
        f"names: {names_list}\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)
