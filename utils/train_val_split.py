import os
import random
import shutil

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def detect_split_dirs(dataset_dir, sub):
    """โฟลเดอร์ปลายทางของฝั่ง train หรือ val: (images_dir, labels_dir)"""
    return (
        os.path.join(dataset_dir, sub, "images"),
        os.path.join(dataset_dir, sub, "labels"),
    )


def is_detect_split(dataset_dir):
    """dataset แบบ detection ถูกแยก train/val แล้วหรือยัง"""
    return os.path.isdir(os.path.join(dataset_dir, "train", "images"))


def has_detect_data(dataset_dir):
    """มีข้อมูลให้ทำงานด้วยไหม (รองรับทั้งแบบ flat และแบบที่แยก train/val แล้ว)"""
    return (
        os.path.isdir(os.path.join(dataset_dir, "images"))
        or is_detect_split(dataset_dir)
        or os.path.isdir(os.path.join(dataset_dir, "val", "images"))
    )


def count_detect_dataset(dataset_dir):
    """นับจำนวนภาพในแต่ละส่วน คืน (n_train, n_val, n_unsplit)
    n_unsplit = ภาพที่ยังอยู่ใน images/ ชั้นนอก (ยังไม่ถูกแยก)"""

    def count_in(folder):
        if not os.path.isdir(folder):
            return 0
        return sum(
            1 for name in os.listdir(folder)
            if os.path.isfile(os.path.join(folder, name))
            and name.lower().endswith(IMAGE_EXTS)
        )

    return (
        count_in(detect_split_dirs(dataset_dir, "train")[0]),
        count_in(detect_split_dirs(dataset_dir, "val")[0]),
        count_in(os.path.join(dataset_dir, "images")),
    )


def collect_detect_pairs(dataset_dir):
    """รวบรวมคู่ (ภาพ, label) ของ dataset แบบ detection จาก

      * โครงสร้าง flat            : images/ + labels/
      * โครงสร้างที่แยกแล้ว        : train/images + train/labels, val/images + val/labels
      * โครงสร้างแบบเก่า (legacy) : images/train + labels/train, images/val + labels/val

    ทำให้สั่งแยกซ้ำได้เรื่อยๆ เมื่อ label ภาพเพิ่ม และย้ายจากโครงสร้างเก่ามาโครงสร้าง
    ใหม่ให้อัตโนมัติ

    คืน dict: stem -> (image_path, label_path หรือ None)
    """
    images_root = os.path.join(dataset_dir, "images")
    labels_root = os.path.join(dataset_dir, "labels")

    pairs = {}
    # ไล่จาก flat และ legacy ก่อน แล้วค่อยโครงสร้างใหม่ เพื่อให้ของใหม่มาทับ (กันซ้ำ)
    locations = [
        (images_root, labels_root),
        (os.path.join(images_root, "train"), os.path.join(labels_root, "train")),
        (os.path.join(images_root, "val"), os.path.join(labels_root, "val")),
        detect_split_dirs(dataset_dir, "train"),
        detect_split_dirs(dataset_dir, "val"),
    ]
    for img_dir, lbl_dir in locations:
        if not os.path.isdir(img_dir):
            continue
        for name in sorted(os.listdir(img_dir)):
            img_path = os.path.join(img_dir, name)
            if not os.path.isfile(img_path) or not name.lower().endswith(IMAGE_EXTS):
                continue
            stem = os.path.splitext(name)[0]
            lbl_path = os.path.join(lbl_dir, stem + ".txt")
            pairs[stem] = (img_path, lbl_path if os.path.isfile(lbl_path) else None)
    return pairs


def _move_over(src, dst):
    """ย้ายไฟล์ทับได้ (shutil.move จะ error ถ้าปลายทางมีไฟล์อยู่แล้วบน Windows)"""
    if os.path.abspath(src) == os.path.abspath(dst):
        return
    if os.path.exists(dst):
        os.remove(dst)
    shutil.move(src, dst)


def split_detect_train_val(dataset_dir, train_ratio=0.8, seed=42):
    """แยก dataset แบบ detection ออกเป็น train/val โดยแต่ละฝั่งมี images และ labels
    ของตัวเอง:

        train/images/xxx.jpg   train/labels/xxx.txt
        val/images/xxx.jpg     val/labels/xxx.txt

    ภาพกับไฟล์ label ของภาพเดียวกันจะถูกย้ายไปฝั่งเดียวกันเสมอ
    สั่งซ้ำได้ - จะรวมภาพทั้งหมด (ทั้งที่แยกแล้วและที่เพิ่งบันทึกเพิ่ม) มาสุ่มแบ่งใหม่

    คืน (n_train, n_val, n_missing_label)
    """
    pairs = collect_detect_pairs(dataset_dir)
    if not pairs:
        raise FileNotFoundError(
            "ไม่พบภาพใน dataset นี้ กรุณาบันทึกภาพ + label อย่างน้อย 1 ภาพก่อน"
        )

    for sub in ("train", "val"):
        for folder in detect_split_dirs(dataset_dir, sub):
            os.makedirs(folder, exist_ok=True)

    stems = sorted(pairs)
    rng = random.Random(seed)
    rng.shuffle(stems)

    n_train = int(round(len(stems) * train_ratio))
    # กันกรณีสุดขั้ว: ถ้ามีมากกว่า 1 ภาพ ต้องมีอย่างน้อยฝั่งละ 1 ภาพ
    if len(stems) > 1:
        n_train = max(1, min(len(stems) - 1, n_train))

    n_missing_label = 0
    for i, stem in enumerate(stems):
        sub = "train" if i < n_train else "val"
        dst_images, dst_labels = detect_split_dirs(dataset_dir, sub)
        img_path, lbl_path = pairs[stem]

        _move_over(img_path, os.path.join(dst_images, os.path.basename(img_path)))

        if lbl_path:
            _move_over(lbl_path, os.path.join(dst_labels, stem + ".txt"))
        else:
            # ไม่มี label = ภาพพื้นหลัง ให้สร้างไฟล์เปล่าไว้เพื่อไม่ให้ ultralytics เตือน
            n_missing_label += 1
            open(os.path.join(dst_labels, stem + ".txt"), "w", encoding="utf-8").close()

    _cleanup_empty_dirs(dataset_dir)
    return n_train, len(stems) - n_train, n_missing_label


def _cleanup_empty_dirs(dataset_dir):
    """ลบโฟลเดอร์ที่ว่างเปล่าหลังย้ายไฟล์ออกไปแล้ว รวมถึงโครงสร้างแบบเก่า
    (images/train, labels/val, ...) เพื่อไม่ให้เหลือโฟลเดอร์กำพร้าค้างไว้"""
    images_root = os.path.join(dataset_dir, "images")
    labels_root = os.path.join(dataset_dir, "labels")
    candidates = [
        os.path.join(images_root, "train"), os.path.join(images_root, "val"),
        os.path.join(labels_root, "train"), os.path.join(labels_root, "val"),
        images_root, labels_root,
    ]
    for folder in candidates:
        try:
            if os.path.isdir(folder) and not os.listdir(folder):
                os.rmdir(folder)
        except OSError:
            pass


def split_train_val(dataset_dir, train_ratio=0.8, seed=42, move=False):
    """Reorganizes dataset_dir/<class_name>/*.jpg into
    dataset_dir/train/<class_name>/*.jpg and dataset_dir/val/<class_name>/*.jpg
    using a random per-class split (so each class keeps roughly the same
    train/val ratio, even with class imbalance).

    move=False (default): copies files, original class folders stay intact.
    move=True: moves files instead (saves disk space, but the original
    per-class folders end up empty and are removed).

    Returns: (train_count, val_count, class_counts) where class_counts is
    {class_name: (n_train, n_val)}.
    """
    rng = random.Random(seed)
    class_dirs = sorted(
        d for d in os.listdir(dataset_dir)
        if os.path.isdir(os.path.join(dataset_dir, d)) and d not in ("train", "val")
        and not d.startswith("resized_")
    )
    if not class_dirs:
        raise FileNotFoundError(
            "ไม่พบโฟลเดอร์คลาสใน dataset ที่เลือก (ควรมีโฟลเดอร์ย่อยชื่อคลาสอยู่ข้างใน เช่น cat/, dog/)"
        )

    train_root = os.path.join(dataset_dir, "train")
    val_root = os.path.join(dataset_dir, "val")

    train_count = 0
    val_count = 0
    class_counts = {}

    op = shutil.move if move else shutil.copy2

    for class_name in class_dirs:
        class_dir = os.path.join(dataset_dir, class_name)
        files = sorted(
            f for f in os.listdir(class_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        if not files:
            continue
        rng.shuffle(files)
        n_train = int(round(len(files) * train_ratio))
        # guarantee at least 1 file in val if there's more than 1 file total
        if n_train == len(files) and len(files) > 1:
            n_train = len(files) - 1
        train_files = files[:n_train]
        val_files = files[n_train:]

        out_train_dir = os.path.join(train_root, class_name)
        out_val_dir = os.path.join(val_root, class_name)
        os.makedirs(out_train_dir, exist_ok=True)
        os.makedirs(out_val_dir, exist_ok=True)

        for f in train_files:
            op(os.path.join(class_dir, f), os.path.join(out_train_dir, f))
        for f in val_files:
            op(os.path.join(class_dir, f), os.path.join(out_val_dir, f))

        train_count += len(train_files)
        val_count += len(val_files)
        class_counts[class_name] = (len(train_files), len(val_files))

        if move:
            try:
                if not os.listdir(class_dir):
                    os.rmdir(class_dir)
            except OSError:
                pass

    return train_count, val_count, class_counts
