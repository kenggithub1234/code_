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


def save_classes_file(classes_path, class_names):
    with open(classes_path, "w", encoding="utf-8") as f:
        f.write("\n".join(class_names))


def save_data_yaml(yaml_path, dataset_root, class_names):
    names_list = "[" + ", ".join(f"'{c}'" for c in class_names) + "]"
    content = (
        f"path: {dataset_root}\n"
        f"train: images\n"
        f"val: images\n\n"
        f"nc: {len(class_names)}\n"
        f"names: {names_list}\n"
    )
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(content)
