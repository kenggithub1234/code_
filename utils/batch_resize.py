import os
import shutil

from utils.train_val_split import collect_detect_pairs
import cv2
import numpy as np


def _resize_stretch(img, target_w, target_h):
    """Resize width/height independently to exactly (target_w, target_h).
    Normalized YOLO bbox coordinates are unaffected by this because each
    axis is already normalized relative to its own dimension, so label
    files don't need any adjustment."""
    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)


def _resize_letterbox(img, target_w, target_h, color=(114, 114, 114)):
    """Resize preserving aspect ratio and pad the rest with a neutral gray
    (same approach YOLO itself uses at inference time). Returns the new
    image plus the scale factor and padding offsets, needed to remap
    bounding boxes onto the new canvas."""
    h, w = img.shape[:2]
    scale = min(target_w / w, target_h / h)
    new_w, new_h = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(img, (new_w, new_h), interpolation=interp)
    canvas = np.full((target_h, target_w, 3), color, dtype=np.uint8)
    pad_x = (target_w - new_w) // 2
    pad_y = (target_h - new_h) // 2
    canvas[pad_y:pad_y + new_h, pad_x:pad_x + new_w] = resized
    return canvas, scale, pad_x, pad_y


def resize_detection_dataset(dataset_dir, target_w, target_h, mode="letterbox",
                              output_dir=None, overwrite=False, progress_callback=None):
    """Resize every image in a YOLO-format detection dataset (images/ +
    labels/), rewriting the label files so bounding boxes still line up
    with the resized images.

    mode: "stretch" or "letterbox" (see helper functions above).
    output_dir: where to write results. If None, defaults to a new
      "resized_<W>x<H>" folder inside dataset_dir (or dataset_dir itself
      if overwrite=True).

    Returns: (num_images_processed, out_images_dir, out_labels_dir)
    """
    # ใช้ตัวรวบรวมเดียวกับตอนแยก train/val จึงเห็นภาพครบทั้งแบบ flat และแบบแยกแล้ว
    pairs = collect_detect_pairs(dataset_dir)
    if not pairs:
        raise FileNotFoundError(
            "ไม่พบภาพใน dataset ที่เลือก (ควรมี images/ หรือ train/images + val/images)"
        )

    if output_dir is None:
        output_dir = dataset_dir if overwrite else os.path.join(
            dataset_dir, f"resized_{target_w}x{target_h}"
        )
    out_images_dir = os.path.join(output_dir, "images")
    out_labels_dir = os.path.join(output_dir, "labels")
    # โฟลเดอร์ปลายทางจริงถูกสร้างตอนวนแต่ละภาพ (ขึ้นกับว่าต้นทาง flat หรือแยก train/val แล้ว)
    os.makedirs(output_dir, exist_ok=True)

    for fname in ("classes.txt", "data.yaml"):
        src = os.path.join(dataset_dir, fname)
        if os.path.isfile(src) and os.path.abspath(output_dir) != os.path.abspath(dataset_dir):
            shutil.copy2(src, os.path.join(output_dir, fname))

    stems = sorted(pairs)
    total = len(stems)
    count = 0

    for i, stem in enumerate(stems):
        img_path, label_path = pairs[stem]
        img = cv2.imread(img_path)
        if img is None:
            if progress_callback:
                progress_callback(i + 1, total)
            continue
        h, w = img.shape[:2]
        fname = os.path.basename(img_path)
        base = os.path.splitext(fname)[0]
        # คงโครงสร้างของต้นทางไว้ในผลลัพธ์: images/ -> images/ และ
        # train/images -> train/images (พร้อม labels ของฝั่งนั้น)
        rel_img_dir = os.path.relpath(os.path.dirname(img_path), dataset_dir)
        sub = os.path.dirname(rel_img_dir)  # "" สำหรับ flat, "train"/"val" สำหรับที่แยกแล้ว
        dst_images_dir = os.path.join(output_dir, sub, "images")
        dst_labels_dir = os.path.join(output_dir, sub, "labels")
        os.makedirs(dst_images_dir, exist_ok=True)
        os.makedirs(dst_labels_dir, exist_ok=True)

        if mode == "stretch":
            new_img = _resize_stretch(img, target_w, target_h)
            new_lines = None
            if label_path and os.path.isfile(label_path):
                with open(label_path, "r", encoding="utf-8") as f:
                    new_lines = f.read()
        else:  # letterbox
            new_img, scale, pad_x, pad_y = _resize_letterbox(img, target_w, target_h)
            new_lines_list = []
            if label_path and os.path.isfile(label_path):
                with open(label_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        if len(parts) < 5:
                            continue
                        class_id = parts[0]
                        xc, yc, bw, bh = (float(v) for v in parts[1:5])
                        xc_px = xc * w
                        yc_px = yc * h
                        bw_px = bw * w
                        bh_px = bh * h
                        new_xc = (xc_px * scale + pad_x) / target_w
                        new_yc = (yc_px * scale + pad_y) / target_h
                        new_bw = (bw_px * scale) / target_w
                        new_bh = (bh_px * scale) / target_h
                        new_lines_list.append(
                            f"{class_id} {new_xc:.6f} {new_yc:.6f} {new_bw:.6f} {new_bh:.6f}"
                        )
            new_lines = "\n".join(new_lines_list)

        cv2.imwrite(os.path.join(dst_images_dir, fname), new_img)
        if new_lines is not None:
            with open(os.path.join(dst_labels_dir, base + ".txt"), "w", encoding="utf-8") as f:
                f.write(new_lines)

        count += 1
        if progress_callback:
            progress_callback(i + 1, total)

    return count, out_images_dir, out_labels_dir


def resize_classification_dataset(dataset_dir, target_w, target_h, mode="stretch",
                                   output_dir=None, overwrite=False, progress_callback=None):
    """Resize every image inside a classification dataset laid out as
    dataset_dir/<class_name>/*.jpg, preserving the per-class folder
    structure (no label files to worry about here).

    Returns: (num_images_processed, output_dir)
    """
    if output_dir is None:
        output_dir = dataset_dir if overwrite else os.path.join(
            dataset_dir, f"resized_{target_w}x{target_h}"
        )
    os.makedirs(output_dir, exist_ok=True)

    class_dirs = sorted(
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)) and not d.startswith("resized_")
    )

    all_files = []
    for class_name in class_dirs:
        class_dir = os.path.join(dataset_dir, class_name)
        for fname in sorted(os.listdir(class_dir)):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                all_files.append((class_name, fname))

    total = len(all_files)
    count = 0
    for i, (class_name, fname) in enumerate(all_files):
        img_path = os.path.join(dataset_dir, class_name, fname)
        img = cv2.imread(img_path)
        if img is None:
            if progress_callback:
                progress_callback(i + 1, total)
            continue

        if mode == "stretch":
            new_img = _resize_stretch(img, target_w, target_h)
        else:
            new_img, _, _, _ = _resize_letterbox(img, target_w, target_h)

        out_class_dir = os.path.join(output_dir, class_name)
        os.makedirs(out_class_dir, exist_ok=True)
        cv2.imwrite(os.path.join(out_class_dir, fname), new_img)
        count += 1
        if progress_callback:
            progress_callback(i + 1, total)

    return count, output_dir
