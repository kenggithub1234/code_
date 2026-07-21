import os
import random
import shutil


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
