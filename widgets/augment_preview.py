"""สร้างภาพตัวอย่างผล Data Augmentation สำหรับแสดงใน tooltip ตอนเอาเมาส์ไปชี้

แต่ละพารามิเตอร์ของ ultralytics จะถูกจำลองผลด้วย OpenCV ให้เห็นภาพคร่าวๆ ว่า
ค่าที่ตั้งไว้ทำอะไรกับภาพ (ไม่ได้เรียกโค้ด augment จริงของ ultralytics แต่ใช้
สูตรเดียวกัน) ภาพถูกฝังเป็น base64 PNG ใน HTML ของ tooltip
"""

import base64
import os

import cv2
import numpy as np

PREVIEW_W = 150
PREVIEW_H = 150

# คำอธิบายของแต่ละกลุ่ม ใช้เป็น tooltip สำรองตอนกลุ่มถูกปิด (Qt จะ disable ลูกทั้งหมด
# ทำให้ลูกไม่รับ event tooltip เอง เมาส์จะไปโดน tooltip ของ QGroupBox แทน)
GROUP_HELP = {
    "color": "ปรับสีของภาพ (เฉดสี/ความอิ่มตัว/ความสว่าง) โดยไม่ขยับตำแหน่งวัตถุ "
             "เหมาะกับงานที่แสงหรือสีของฉากเปลี่ยนไปมา",
    "geometric": "ปรับรูปทรงและมุมมองของภาพ (หมุน เลื่อน ย่อขยาย บิด พลิก) "
                 "ทำให้โมเดลทนต่อการวางวัตถุคนละตำแหน่ง/มุม",
    "mix": "สร้างภาพใหม่จากหลายภาพรวมกัน (ต่อ 4 ภาพ ผสมโปร่งแสง คัดลอกวัตถุไปแปะ) "
           "ช่วยเพิ่มความหลากหลายของข้อมูลอย่างมาก",
}

# คำอธิบายสั้นๆ ของแต่ละพารามิเตอร์
PARAM_HELP = {
    "hsv_h": "สุ่มเปลี่ยนเฉดสี (hue) ของภาพ ±ค่าที่ตั้ง ช่วยให้โมเดลไม่ยึดติดกับสีวัตถุ",
    "hsv_s": "สุ่มเพิ่ม/ลดความอิ่มตัวของสี รับมือกับภาพสีจัดหรือสีซีด",
    "hsv_v": "สุ่มเพิ่ม/ลดความสว่าง รับมือกับสภาพแสงที่ต่างกัน",
    "degrees": "สุ่มหมุนภาพ ±องศาที่ตั้ง ใช้เมื่อวัตถุอาจวางเอียงได้",
    "translate": "สุ่มเลื่อนภาพตามสัดส่วนที่ตั้ง ทำให้โมเดลไม่ยึดติดตำแหน่งวัตถุ",
    "scale": "สุ่มย่อ/ขยายภาพ ช่วยให้ตรวจจับวัตถุได้หลายขนาด",
    "shear": "บิดเฉือนภาพเป็นรูปสี่เหลี่ยมด้านขนาน จำลองมุมกล้องที่เอียง",
    "perspective": "บิดมุมมองแบบ 3 มิติ จำลองการถ่ายจากมุมเฉียง (ค่าน้อยมากก็เห็นผลชัด)",
    "flipud": "สุ่มพลิกภาพบน-ล่าง ตามความน่าจะเป็นที่ตั้ง (ระวัง: ไม่เหมาะกับวัตถุที่มีบน-ล่างชัดเจน)",
    "fliplr": "สุ่มพลิกภาพซ้าย-ขวา ตามความน่าจะเป็นที่ตั้ง ใช้ได้ดีกับงานทั่วไป",
    "mosaic": "ต่อ 4 ภาพเป็นภาพเดียว ทำให้โมเดลเห็นวัตถุหลายขนาด/หลายบริบทพร้อมกัน",
    "mixup": "ผสม 2 ภาพซ้อนกันแบบโปร่งแสง ช่วยลด overfitting",
    "copy_paste": "คัดลอกวัตถุจากภาพหนึ่งไปแปะอีกภาพ (ได้ผลดีกับงาน segmentation)",
}


def make_sample_image(w=PREVIEW_W, h=PREVIEW_H):
    """ภาพสังเคราะห์สำรอง ใช้เมื่อยังไม่ได้เลือก dataset - มีพื้นหลังไล่เฉด
    และวัตถุสีสดที่มีบน-ล่าง/ซ้าย-ขวาชัดเจน เพื่อให้เห็นผลของ augment ง่าย"""
    img = np.zeros((h, w, 3), dtype=np.uint8)
    for y in range(h):
        img[y, :] = (int(120 * y / h) + 40, int(90 * y / h) + 30, int(60 * y / h) + 25)

    cv2.rectangle(img, (int(w * 0.18), int(h * 0.30)), (int(w * 0.52), int(h * 0.78)),
                  (60, 180, 250), -1)
    cv2.circle(img, (int(w * 0.70), int(h * 0.38)), int(w * 0.13), (80, 220, 120), -1)
    pts = np.array([[int(w * 0.62), int(h * 0.82)],
                    [int(w * 0.80), int(h * 0.55)],
                    [int(w * 0.92), int(h * 0.82)]], dtype=np.int32)
    cv2.fillPoly(img, [pts], (200, 120, 240))
    cv2.putText(img, "UP", (int(w * 0.20), int(h * 0.22)),
                cv2.FONT_HERSHEY_SIMPLEX, w / 320.0, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(img, (0, 0), (w - 1, h - 1), (90, 90, 90), 1)
    return img


def _read_first_image_in(folder, w, h, recursive=False):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    if not os.path.isdir(folder):
        return None
    if recursive:
        for dirpath, _dirnames, filenames in os.walk(folder):
            for name in sorted(filenames)[:200]:
                if name.lower().endswith(exts):
                    img = _imread(os.path.join(dirpath, name))
                    if img is not None:
                        return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
        return None
    for name in sorted(os.listdir(folder))[:200]:
        if name.lower().endswith(exts):
            img = _imread(os.path.join(folder, name))
            if img is not None:
                return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
    return None


def _imread(path):
    """cv2.imread ที่รองรับพาธภาษาไทย/ยูนิโค้ดบน Windows"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def load_sample_from_folder(folder, w=PREVIEW_W, h=PREVIEW_H):
    """หาภาพแรกในโฟลเดอร์ dataset แบบ classification (root/train/<class>/*.jpg)"""
    try:
        return _read_first_image_in(folder, w, h, recursive=True)
    except Exception:
        return None


def load_sample_from_dataset(data_yaml_path, w=PREVIEW_W, h=PREVIEW_H):
    """หาภาพจริงภาพแรกจาก dataset ที่ระบุใน data.yaml มาใช้เป็นตัวอย่าง
    คืน None ถ้าหาไม่เจอ (แล้วผู้เรียกจะ fallback ไปใช้ภาพสังเคราะห์)"""
    if not data_yaml_path or not os.path.isfile(data_yaml_path):
        return None
    try:
        root, train_sub = "", "images"
        with open(data_yaml_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("path:"):
                    root = line.split(":", 1)[1].strip()
                elif line.startswith("train:"):
                    train_sub = line.split(":", 1)[1].strip()

        candidates = []
        if root:
            candidates.append(os.path.join(root, train_sub))
            candidates.append(root)
        candidates.append(os.path.join(os.path.dirname(data_yaml_path), train_sub))
        candidates.append(os.path.dirname(data_yaml_path))

        for folder in candidates:
            img = _read_first_image_in(folder, w, h)
            if img is not None:
                return img
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------
# การจำลองผลของแต่ละพารามิเตอร์
# ---------------------------------------------------------------------------

def _apply_hsv(img, key, value):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    if key == "hsv_h":
        hsv[..., 0] = (hsv[..., 0] + value * 180.0) % 180.0
    elif key == "hsv_s":
        hsv[..., 1] = np.clip(hsv[..., 1] * (1.0 + value), 0, 255)
    else:  # hsv_v
        hsv[..., 2] = np.clip(hsv[..., 2] * (1.0 + value), 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def _warp(img, matrix, perspective=False):
    h, w = img.shape[:2]
    border = (114, 114, 114)
    if perspective:
        return cv2.warpPerspective(img, matrix, (w, h), borderValue=border)
    return cv2.warpAffine(img, matrix, (w, h), borderValue=border)


def _apply_geometric(img, key, value):
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0

    if key == "degrees":
        return _warp(img, cv2.getRotationMatrix2D((cx, cy), value, 1.0))

    if key == "translate":
        m = np.float32([[1, 0, value * w], [0, 1, value * h]])
        return _warp(img, m)

    if key == "scale":
        # ultralytics สุ่มในช่วง 1±scale ตัวอย่างนี้แสดงฝั่งย่อลง
        factor = max(0.05, 1.0 - value)
        return _warp(img, cv2.getRotationMatrix2D((cx, cy), 0, factor))

    if key == "shear":
        rad = np.tan(np.radians(value))
        m = np.float32([[1, rad, -rad * cy], [0, 1, 0]])
        return _warp(img, m)

    if key == "perspective":
        m = np.eye(3, dtype=np.float32)
        m[2, 0] = value
        m[2, 1] = value
        return _warp(img, m, perspective=True)

    if key == "flipud":
        return cv2.flip(img, 0)

    if key == "fliplr":
        return cv2.flip(img, 1)

    return img


def _apply_mix(img, key, value):
    h, w = img.shape[:2]

    if key == "mosaic":
        tiles = [
            img,
            cv2.flip(img, 1),
            _apply_hsv(img, "hsv_h", 0.35),
            _warp(img, cv2.getRotationMatrix2D((w / 2, h / 2), 12, 0.9)),
        ]
        small = [cv2.resize(t, (w // 2, h // 2), interpolation=cv2.INTER_AREA) for t in tiles]
        top = np.hstack([small[0], small[1]])
        bottom = np.hstack([small[2], small[3]])
        out = np.vstack([top, bottom])
        cv2.line(out, (w // 2, 0), (w // 2, h), (255, 255, 255), 1)
        cv2.line(out, (0, h // 2), (w, h // 2), (255, 255, 255), 1)
        return out

    if key == "mixup":
        other = cv2.flip(_apply_hsv(img, "hsv_h", 0.4), 1)
        alpha = 0.5 if value <= 0 else min(0.5, 0.25 + value * 0.25)
        return cv2.addWeighted(img, 1 - alpha, other, alpha, 0)

    if key == "copy_paste":
        out = img.copy()
        src = cv2.flip(img, 1)
        y1, y2 = int(h * 0.30), int(h * 0.78)
        x1, x2 = int(w * 0.18), int(w * 0.52)
        patch = src[y1:y2, x1:x2]
        py, px = int(h * 0.15), int(w * 0.44)
        ph, pw = patch.shape[:2]
        ph = min(ph, h - py)
        pw = min(pw, w - px)
        out[py:py + ph, px:px + pw] = patch[:ph, :pw]
        cv2.rectangle(out, (px, py), (px + pw, py + ph), (0, 255, 255), 1)
        return out

    return img


def apply_preview(key, value, img):
    """คืนภาพที่ผ่าน augment ตามค่าที่ระบุ"""
    try:
        if key.startswith("hsv_"):
            return _apply_hsv(img, key, value)
        if key in ("degrees", "translate", "scale", "shear", "perspective", "flipud", "fliplr"):
            return _apply_geometric(img, key, value)
        return _apply_mix(img, key, value)
    except Exception:
        return img


# ---------------------------------------------------------------------------
# แปลงเป็น HTML สำหรับ tooltip
# ---------------------------------------------------------------------------

def _img_to_base64(img):
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")


def render_tooltip_html(key, value, enabled, sample_img):
    """สร้าง HTML ของ tooltip: คำอธิบาย + ภาพ ก่อน/หลัง"""
    help_text = PARAM_HELP.get(key, "")

    if not enabled or value == 0.0:
        note = ("ยังไม่ได้ติ๊กใช้งาน - จะส่งค่า 0 (ไม่ใช้)"
                if not enabled else "ค่าเป็น 0 จึงยังไม่มีผลกับภาพ")
        before_b64 = _img_to_base64(sample_img)
        return (
            f"<div style='max-width:340px'>"
            f"<b>{key}</b><br>{help_text}<br>"
            f"<img src='data:image/png;base64,{before_b64}'><br>"
            f"<i style='color:#ffcc66'>{note}</i></div>"
        )

    after = apply_preview(key, value, sample_img)
    before_b64 = _img_to_base64(sample_img)
    after_b64 = _img_to_base64(after)

    if key in ("flipud", "fliplr"):
        value_note = f"ความน่าจะเป็น {value:g} (ภาพขวาคือกรณีที่ถูกพลิก)"
    elif key == "mosaic":
        value_note = f"ความน่าจะเป็น {value:g} ที่จะต่อ 4 ภาพแบบนี้"
    elif key in ("mixup", "copy_paste"):
        value_note = f"ความน่าจะเป็น {value:g} ที่จะเกิดแบบนี้"
    else:
        value_note = f"ค่าปัจจุบัน {value:g} (ของจริงจะสุ่มในช่วง ±{value:g} ทุกภาพ)"

    return (
        f"<div style='max-width:360px'>"
        f"<b>{key}</b><br>{help_text}<br>"
        f"<table cellspacing='6'><tr>"
        f"<td align='center'>ก่อน<br><img src='data:image/png;base64,{before_b64}'></td>"
        f"<td align='center'>หลัง<br><img src='data:image/png;base64,{after_b64}'></td>"
        f"</tr></table>"
        f"<i style='color:#9ecbff'>{value_note}</i></div>"
    )
