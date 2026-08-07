import os

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QGridLayout, QHBoxLayout, QLabel,
    QProgressBar, QPushButton, QScrollArea, QVBoxLayout, QWidget
)

THUMB_W = 200
THUMB_H = 150
COLUMNS = 4


class ThumbnailLoader(QThread):
    """อ่านเฟรมและวาดกรอบลงไป แล้วส่งภาพย่อกลับให้ GUI ทีละรูป

    ส่งกลับเป็น numpy array (RGB) ไม่ใช่ QPixmap เพราะ QPixmap สร้างนอก
    GUI thread ไม่ได้ ฝั่ง GUI จะแปลงเป็น QPixmap เอง
    """

    thumb_ready = pyqtSignal(int, object)   # (frame_idx, rgb_ndarray)
    progress_signal = pyqtSignal(int, int)
    finished_signal = pyqtSignal()

    def __init__(self, indices, frame_boxes, classes, colors,
                 source_type="video", video_path=None, image_paths=None,
                 shape_type="box"):
        super().__init__()
        self.indices = list(indices)
        self.shape_type = shape_type  # "box" หรือ "polygon"
        self.frame_boxes = {i: list(frame_boxes.get(i, [])) for i in self.indices}
        self.classes = list(classes)
        self.colors = list(colors)
        self.source_type = source_type
        self.video_path = video_path
        self.image_paths = image_paths or []
        self._stop = False

    def stop(self):
        self._stop = True

    def _read_frame(self, cap, idx):
        if self.source_type == "video":
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            return frame if ok else None
        if idx < 0 or idx >= len(self.image_paths):
            return None
        return cv2.imread(self.image_paths[idx])

    def _shape_color(self, class_id):
        color = self.colors[class_id % len(self.colors)]
        # ค่าสีในโปรเจกต์เก็บเป็น RGB ส่วน cv2 ใช้ BGR จึงต้องสลับ
        return (color[2], color[1], color[0])

    def _class_name(self, class_id):
        return self.classes[class_id] if class_id < len(self.classes) else str(class_id)

    def _draw_boxes(self, frame, idx):
        out = frame.copy()
        h, w = out.shape[:2]
        thickness = max(1, int(round(min(w, h) / 200)))
        font_scale = min(w, h) / 500.0

        for shape in self.frame_boxes.get(idx, []):
            if self.shape_type == "polygon":
                class_id = shape["class_id"]
                points = shape["points"]
                if len(points) < 3:
                    continue
                bgr = self._shape_color(class_id)
                pts = np.array([[int(x), int(y)] for x, y in points], dtype=np.int32)
                # ระบายโปร่งแสงทับ เพื่อให้เห็นพื้นที่ที่ polygon ครอบจริงๆ
                overlay = out.copy()
                cv2.fillPoly(overlay, [pts], bgr)
                cv2.addWeighted(overlay, 0.35, out, 0.65, 0, out)
                cv2.polylines(out, [pts], True, bgr, thickness, cv2.LINE_AA)
                anchor = (int(min(x for x, _ in points)),
                          max(12, int(min(y for _, y in points)) - 4))
            else:
                class_id, x1, y1, x2, y2 = shape
                bgr = self._shape_color(class_id)
                cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), bgr, thickness)
                anchor = (int(x1), max(12, int(y1) - 4))

            cv2.putText(out, self._class_name(class_id), anchor,
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, bgr,
                        thickness, cv2.LINE_AA)
        return out

    def run(self):
        cap = None
        if self.source_type == "video":
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.finished_signal.emit()
                return

        total = len(self.indices)
        for i, idx in enumerate(self.indices):
            if self._stop:
                break
            frame = self._read_frame(cap, idx)
            if frame is not None:
                drawn = self._draw_boxes(frame, idx)
                scale = min(THUMB_W / drawn.shape[1], THUMB_H / drawn.shape[0])
                nw = max(1, int(drawn.shape[1] * scale))
                nh = max(1, int(drawn.shape[0] * scale))
                thumb = cv2.resize(drawn, (nw, nh), interpolation=cv2.INTER_AREA)
                self.thumb_ready.emit(idx, cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))
            self.progress_signal.emit(i + 1, total)

        if cap is not None:
            cap.release()
        self.finished_signal.emit()


class PreviewItem(QWidget):
    """ภาพย่อ 1 รูป พร้อม checkbox ว่าจะบันทึกหรือไม่"""

    def __init__(self, idx, caption, n_boxes, shape_word="กรอบ", parent=None):
        super().__init__(parent)
        self.idx = idx

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(3)

        self.check = QCheckBox(caption)
        # ค่าเริ่มต้น: ติ๊กเฉพาะเฟรมที่มีกรอบ ส่วนเฟรมว่างปล่อยไว้ให้เลือกเอง
        self.check.setChecked(n_boxes > 0)
        layout.addWidget(self.check)

        self.image_label = QLabel("กำลังโหลด...")
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setFixedSize(THUMB_W, THUMB_H)
        self.image_label.setStyleSheet("background-color:#1a1a1a;border:1px solid #444;")
        layout.addWidget(self.image_label)

        info = f"{n_boxes} {shape_word}" if n_boxes else f"ไม่มี{shape_word}"
        self.info_label = QLabel(info)
        self.info_label.setStyleSheet(
            "color:#9ecbff;" if n_boxes else "color:#e0a83a;"
        )
        layout.addWidget(self.info_label)

        # คลิกที่รูปเพื่อติ๊ก/ไม่ติ๊กได้เลย ไม่ต้องเล็งที่ช่องเล็กๆ
        self.image_label.mousePressEvent = self._toggle

    def _toggle(self, _event):
        self.check.setChecked(not self.check.isChecked())

    def set_thumbnail(self, rgb):
        h, w, _ = rgb.shape
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self.image_label.setPixmap(QPixmap.fromImage(qimg))

    def is_checked(self):
        return self.check.isChecked()


class SavePreviewDialog(QDialog):
    """แสดงภาพตัวอย่างทุกเฟรมที่ label ไว้ พร้อม checkbox ให้เลือกว่าจะบันทึกรูปไหน"""

    def __init__(self, parent, frame_boxes, classes, colors,
                 source_type="video", video_path=None, image_paths=None,
                 shape_type="box"):
        super().__init__(parent)
        self.setWindowTitle("เลือกภาพที่ต้องการบันทึก")
        self.resize(950, 700)

        self.frame_boxes = frame_boxes
        self.shape_type = shape_type
        self.shape_word = "polygon" if shape_type == "polygon" else "กรอบ"
        self.source_type = source_type
        self.image_paths = image_paths or []
        self.items = {}

        root = QVBoxLayout(self)

        hint = QLabel(
            "ติ๊กเลือกเฉพาะรูปที่ต้องการบันทึก (คลิกที่รูปก็ติ๊ก/ยกเลิกได้) "
            f"ค่าเริ่มต้นจะติ๊กให้เฉพาะเฟรมที่มี{self.shape_word}"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#9ecbff;")
        root.addWidget(hint)

        # ---- ปุ่มเลือกแบบรวดเดียว ----
        tools = QHBoxLayout()
        for text, slot in (
            ("เลือกทั้งหมด", lambda: self._set_all(True)),
            ("ไม่เลือกเลย", lambda: self._set_all(False)),
            (f"เฉพาะที่มี{self.shape_word}", self._select_with_boxes),
            ("สลับการเลือก", self._invert),
        ):
            btn = QPushButton(text)
            btn.clicked.connect(slot)
            tools.addWidget(btn)
        tools.addStretch()
        self.count_label = QLabel("")
        self.count_label.setStyleSheet("font-weight:bold;")
        tools.addWidget(self.count_label)
        root.addLayout(tools)

        # ---- ตารางภาพย่อ ----
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        container = QWidget()
        self.grid = QGridLayout(container)
        self.grid.setAlignment(Qt.AlignTop)
        scroll.setWidget(container)
        root.addWidget(scroll, stretch=1)

        self.progress = QProgressBar()
        root.addWidget(self.progress)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.ok_button = buttons.button(QDialogButtonBox.Ok)
        self.ok_button.setText("บันทึกรูปที่เลือก")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        root.addWidget(buttons)

        self._build_items(classes)
        self._start_loader(classes, colors, video_path)

    # ------------------------------------------------------------------
    def _caption(self, idx):
        if self.source_type == "images" and idx < len(self.image_paths):
            name = os.path.basename(self.image_paths[idx])
            return name if len(name) <= 22 else name[:19] + "..."
        return f"เฟรม {idx}"

    def _build_items(self, classes):
        for pos, idx in enumerate(sorted(self.frame_boxes)):
            n_boxes = len(self.frame_boxes[idx])
            item = PreviewItem(idx, self._caption(idx), n_boxes, self.shape_word)
            item.check.toggled.connect(self._update_count)
            self.grid.addWidget(item, pos // COLUMNS, pos % COLUMNS)
            self.items[idx] = item
        self._update_count()

    def _start_loader(self, classes, colors, video_path):
        self.loader = ThumbnailLoader(
            indices=sorted(self.frame_boxes),
            frame_boxes=self.frame_boxes,
            classes=classes,
            colors=colors,
            source_type=self.source_type,
            video_path=video_path,
            image_paths=self.image_paths,
            shape_type=self.shape_type,
        )
        self.loader.thumb_ready.connect(self._on_thumb)
        self.loader.progress_signal.connect(self._on_progress)
        self.loader.finished_signal.connect(lambda: self.progress.setVisible(False))
        self.loader.start()

    def _on_thumb(self, idx, rgb):
        item = self.items.get(idx)
        if item is not None:
            item.set_thumbnail(rgb)

    def _on_progress(self, done, total):
        self.progress.setMaximum(max(1, total))
        self.progress.setValue(done)
        self.progress.setFormat(f"กำลังสร้างภาพตัวอย่าง {done}/{total}")

    # ------------------------------------------------------------------
    def _set_all(self, checked):
        for item in self.items.values():
            item.check.setChecked(checked)

    def _select_with_boxes(self):
        for idx, item in self.items.items():
            item.check.setChecked(bool(self.frame_boxes.get(idx)))

    def _invert(self):
        for item in self.items.values():
            item.check.setChecked(not item.is_checked())

    def _update_count(self):
        n = len(self.selected_indices())
        self.count_label.setText(f"เลือกไว้ {n} / {len(self.items)} รูป")
        if hasattr(self, "ok_button"):
            self.ok_button.setEnabled(n > 0)
            self.ok_button.setText(f"บันทึก {n} รูป" if n else "บันทึกรูปที่เลือก")

    def selected_indices(self):
        return [idx for idx, item in self.items.items() if item.is_checked()]

    # ------------------------------------------------------------------
    def closeEvent(self, event):
        if getattr(self, "loader", None) and self.loader.isRunning():
            self.loader.stop()
            self.loader.wait(2000)
        super().closeEvent(event)

    def done(self, result):
        if getattr(self, "loader", None) and self.loader.isRunning():
            self.loader.stop()
            self.loader.wait(2000)
        super().done(result)
