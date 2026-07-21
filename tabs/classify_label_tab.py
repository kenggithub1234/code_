import os
import cv2
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider, QSpinBox,
    QDoubleSpinBox, QListWidget, QFileDialog, QMessageBox, QInputDialog, QGroupBox,
    QLineEdit, QCheckBox, QProgressBar, QShortcut, QComboBox, QApplication
)
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import Qt

from widgets.canvas import ImageCanvas
from widgets.resize_dialog import BatchResizeDialog
from utils.keras_classify import get_model_input_size, preprocess_image
from utils.keras_classify_worker import KerasClassifyWorker
from utils.train_val_split import split_train_val
from tabs.detect_label_tab import COLORS  # shared color palette for overlays


class ClassifyLabelTab(QWidget):
    def __init__(self):
        super().__init__()
        self.cap = None
        self.video_path = None
        self.source_mode = "video"  # "video" or "images"
        self.image_paths = []       # used when source_mode == "images"
        self.total_frames = 0
        self.current_idx = 0
        self.current_frame = None
        self.classes = []

        # whole-frame assignments: idx -> class_name
        self.frame_class = {}
        # sub-region ("crop") assignments: idx -> [{"box": (x1,y1,x2,y2), "class_name": str|None}, ...]
        self.frame_crops = {}

        self.action_stack = []  # undo history

        self.output_dir = os.path.join(os.getcwd(), "dataset_classify")

        self._cached_keras_model = None
        self._cached_keras_model_path = None
        self.auto_worker = None

        self._build_ui()

    def _build_ui(self):
        main_layout = QHBoxLayout(self)
        left = QVBoxLayout()

        top_bar = QHBoxLayout()
        load_btn = QPushButton("โหลดวิดีโอ")
        load_btn.clicked.connect(self.load_video)
        top_bar.addWidget(load_btn)

        load_images_btn = QPushButton("โหลดรูปภาพ (หลายไฟล์)")
        load_images_btn.clicked.connect(self.load_images_files)
        top_bar.addWidget(load_images_btn)

        load_folder_btn = QPushButton("โหลดโฟลเดอร์รูปภาพ")
        load_folder_btn.clicked.connect(self.load_images_folder)
        top_bar.addWidget(load_folder_btn)

        self.video_label = QLabel("ยังไม่ได้โหลดวิดีโอ/รูปภาพ")
        top_bar.addWidget(self.video_label)
        top_bar.addStretch()
        left.addLayout(top_bar)

        # Drawing is enabled here so the user can optionally mark
        # sub-regions ("crops") within the frame to classify individually,
        # instead of always classifying the whole frame.
        self.canvas = ImageCanvas(draw_enabled=True)
        self.canvas.box_drawn.connect(self.on_crop_drawn)
        left.addWidget(self.canvas, stretch=1)

        nav_bar = QHBoxLayout()
        self.prev_btn = QPushButton("◀ ก่อนหน้า")
        self.prev_btn.clicked.connect(self.prev_frame)
        nav_bar.addWidget(self.prev_btn)

        self.slider = QSlider(Qt.Horizontal)
        self.slider.valueChanged.connect(self.on_slider_changed)
        nav_bar.addWidget(self.slider, stretch=1)

        self.next_btn = QPushButton("ถัดไป ▶")
        self.next_btn.clicked.connect(self.next_frame)
        nav_bar.addWidget(self.next_btn)

        nav_bar.addWidget(QLabel("เฟรม/ภาพ:"))
        self.frame_spin = QSpinBox()
        self.frame_spin.valueChanged.connect(self.on_spin_changed)
        nav_bar.addWidget(self.frame_spin)

        nav_bar.addWidget(QLabel("ก้าวกระโดด:"))
        self.step_spin = QSpinBox()
        self.step_spin.setRange(1, 500)
        self.step_spin.setValue(1)
        nav_bar.addWidget(self.step_spin)
        left.addLayout(nav_bar)

        self.status_label = QLabel("เฟรมนี้ยังไม่ถูกกำหนดคลาส")
        left.addWidget(self.status_label)

        hint = QLabel(
            "เคล็ดลับ: ลากเมาส์บนภาพเพื่อวาดกรอบย่อย (crop) แล้วกดปุ่มเลข 1-9 เพื่อกำหนดคลาสให้ crop "
            "ที่เลือกไว้ในลิสต์อย่างรวดเร็ว หรือถ้าไม่ได้เลือก crop ไว้ ปุ่มเลขจะกำหนดคลาสให้ทั้งเฟรมแทน"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#9ecbff;")
        left.addWidget(hint)

        main_layout.addLayout(left, stretch=3)

        # ---------------- right panel ----------------
        right = QVBoxLayout()

        class_group = QGroupBox("คลาส (Classes) - กด 1-9 เพื่อกำหนดคลาสอย่างเร็ว")
        class_layout = QVBoxLayout()
        self.class_list = QListWidget()
        class_layout.addWidget(self.class_list)
        class_btns = QHBoxLayout()
        add_class_btn = QPushButton("เพิ่มคลาส")
        add_class_btn.clicked.connect(self.add_class)
        remove_class_btn = QPushButton("ลบคลาส")
        remove_class_btn.clicked.connect(self.remove_class)
        class_btns.addWidget(add_class_btn)
        class_btns.addWidget(remove_class_btn)
        class_layout.addLayout(class_btns)
        class_group.setLayout(class_layout)
        right.addWidget(class_group)

        crop_group = QGroupBox("Crop ย่อยในเฟรมนี้ (วาดบนภาพเพื่อเพิ่ม)")
        crop_layout = QVBoxLayout()
        self.crop_list = QListWidget()
        crop_layout.addWidget(self.crop_list)
        crop_btns = QHBoxLayout()
        assign_crop_btn = QPushButton("กำหนดคลาสให้ crop ที่เลือก")
        assign_crop_btn.clicked.connect(self.assign_class_to_selected_crop)
        crop_btns.addWidget(assign_crop_btn)
        del_crop_btn = QPushButton("ลบ crop ที่เลือก")
        del_crop_btn.clicked.connect(self.delete_selected_crop)
        crop_btns.addWidget(del_crop_btn)
        crop_layout.addLayout(crop_btns)
        undo_btn = QPushButton("↩ Undo (Ctrl+Z)")
        undo_btn.clicked.connect(self.undo)
        crop_layout.addWidget(undo_btn)
        crop_group.setLayout(crop_layout)
        right.addWidget(crop_group)

        # ---- Keras auto-label group ----
        auto_group = QGroupBox("Auto-label ด้วยโมเดล Keras Classification ที่เทรนแล้ว")
        auto_layout = QVBoxLayout()

        model_bar = QHBoxLayout()
        self.keras_model_edit = QLineEdit()
        self.keras_model_edit.setPlaceholderText("เลือกไฟล์โมเดล .h5 หรือ .keras")
        model_bar.addWidget(self.keras_model_edit)
        browse_model_btn = QPushButton("เลือกโมเดล")
        browse_model_btn.clicked.connect(self.browse_keras_model)
        model_bar.addWidget(browse_model_btn)
        auto_layout.addLayout(model_bar)

        labels_bar = QHBoxLayout()
        self.labels_file_edit = QLineEdit()
        self.labels_file_edit.setPlaceholderText(
            "(ไม่บังคับ) ไฟล์ .txt รายชื่อคลาส บรรทัดละ 1 ชื่อ ตามลำดับ output ของโมเดล"
        )
        labels_bar.addWidget(self.labels_file_edit)
        browse_labels_btn = QPushButton("เลือกไฟล์ labels")
        browse_labels_btn.clicked.connect(self.browse_labels_file)
        labels_bar.addWidget(browse_labels_btn)
        auto_layout.addLayout(labels_bar)

        opts_bar = QHBoxLayout()
        opts_bar.addWidget(QLabel("Preprocessing:"))
        self.preprocess_combo = QComboBox()
        self.preprocess_combo.addItems([
            "หาร 255 (0-1)",
            "หาร 127.5 ลบ 1 (-1 ถึง 1, แบบ MobileNet)",
            "ไม่ทำ (โมเดลมี Rescaling อยู่แล้ว)",
        ])
        opts_bar.addWidget(self.preprocess_combo)
        opts_bar.addWidget(QLabel("Confidence ขั้นต่ำ:"))
        self.keras_conf_spin = QDoubleSpinBox()
        self.keras_conf_spin.setRange(0.0, 1.0)
        self.keras_conf_spin.setSingleStep(0.05)
        self.keras_conf_spin.setValue(0.5)
        opts_bar.addWidget(self.keras_conf_spin)
        auto_layout.addLayout(opts_bar)

        self.use_crops_check = QCheckBox("ใช้ crop ที่วาดไว้แทนการจำแนกทั้งเฟรม (ถ้าเฟรมนั้นมี crop)")
        self.use_crops_check.setChecked(True)
        auto_layout.addWidget(self.use_crops_check)

        auto_btn_bar = QHBoxLayout()
        auto_current_btn = QPushButton("Auto-label เฟรมนี้")
        auto_current_btn.clicked.connect(self.auto_label_current_frame_keras)
        auto_btn_bar.addWidget(auto_current_btn)

        self.auto_remaining_btn = QPushButton("Auto-label เฟรมที่เหลือทั้งหมด")
        self.auto_remaining_btn.clicked.connect(self.start_auto_label_remaining_keras)
        auto_btn_bar.addWidget(self.auto_remaining_btn)

        self.auto_stop_btn = QPushButton("หยุด")
        self.auto_stop_btn.setEnabled(False)
        self.auto_stop_btn.clicked.connect(self.stop_auto_label_keras)
        auto_btn_bar.addWidget(self.auto_stop_btn)
        auto_layout.addLayout(auto_btn_bar)

        self.auto_progress = QProgressBar()
        auto_layout.addWidget(self.auto_progress)
        self.auto_status_label = QLabel("")
        auto_layout.addWidget(self.auto_status_label)

        auto_group.setLayout(auto_layout)
        right.addWidget(auto_group)

        # ---- save / output group ----
        out_group = QGroupBox("บันทึกผลลัพธ์ (โฟลเดอร์ต่อคลาส)")
        out_layout = QVBoxLayout()
        out_dir_bar = QHBoxLayout()
        self.out_dir_edit = QLineEdit(self.output_dir)
        out_dir_bar.addWidget(self.out_dir_edit)
        browse_out_btn = QPushButton("เลือกโฟลเดอร์")
        browse_out_btn.clicked.connect(self.browse_output)
        out_dir_bar.addWidget(browse_out_btn)
        out_layout.addLayout(out_dir_bar)

        assign_btn = QPushButton("💾 กำหนดคลาสให้ทั้งเฟรมและบันทึก")
        assign_btn.clicked.connect(self.assign_and_save)
        out_layout.addWidget(assign_btn)

        save_crops_btn = QPushButton("💾 บันทึก crop ที่กำหนดคลาสแล้วทั้งหมด (เฟรมนี้)")
        save_crops_btn.clicked.connect(self.save_all_assigned_crops)
        out_layout.addWidget(save_crops_btn)

        batch_resize_btn = QPushButton("🔧 Batch Resize รูปทั้ง Dataset")
        batch_resize_btn.clicked.connect(self.open_batch_resize_dialog)
        out_layout.addWidget(batch_resize_btn)

        out_group.setLayout(out_layout)
        right.addWidget(out_group)

        range_group = QGroupBox("กำหนดคลาสแบบช่วงเฟรม (Batch, ทั้งเฟรม)")
        range_layout = QVBoxLayout()
        range_bar = QHBoxLayout()
        range_bar.addWidget(QLabel("จากเฟรม:"))
        self.range_from = QSpinBox()
        self.range_from.setRange(0, 999999)
        range_bar.addWidget(self.range_from)
        range_bar.addWidget(QLabel("ถึงเฟรม:"))
        self.range_to = QSpinBox()
        self.range_to.setRange(0, 999999)
        range_bar.addWidget(self.range_to)
        range_layout.addLayout(range_bar)
        batch_btn = QPushButton("บันทึกทั้งช่วงด้วยคลาสที่เลือก")
        batch_btn.clicked.connect(self.assign_range_and_save)
        range_layout.addWidget(batch_btn)
        range_group.setLayout(range_layout)
        right.addWidget(range_group)

        # ---- train/val split group ----
        split_group = QGroupBox("แบ่ง Train / Val อัตโนมัติ")
        split_layout = QVBoxLayout()
        split_bar = QHBoxLayout()
        split_bar.addWidget(QLabel("สัดส่วน Train:"))
        self.train_ratio_spin = QDoubleSpinBox()
        self.train_ratio_spin.setRange(0.5, 0.95)
        self.train_ratio_spin.setSingleStep(0.05)
        self.train_ratio_spin.setValue(0.8)
        split_bar.addWidget(self.train_ratio_spin)
        split_bar.addStretch()
        split_layout.addLayout(split_bar)

        self.move_files_check = QCheckBox("ย้ายไฟล์แทนการคัดลอก (ประหยัดพื้นที่ แต่ไฟล์ต้นฉบับในโฟลเดอร์คลาสเดิมจะหายไป)")
        split_layout.addWidget(self.move_files_check)

        split_btn = QPushButton("📊 แบ่ง Train/Val อัตโนมัติ")
        split_btn.clicked.connect(self.run_train_val_split)
        split_layout.addWidget(split_btn)

        self.split_status_label = QLabel("")
        self.split_status_label.setWordWrap(True)
        split_layout.addWidget(self.split_status_label)

        split_group.setLayout(split_layout)
        right.addWidget(split_group)

        right.addStretch()
        main_layout.addLayout(right, stretch=1)

        # ---- keyboard shortcuts ----
        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.activated.connect(self.undo)

        self._number_shortcuts = []
        for n in range(1, 10):
            sc = QShortcut(QKeySequence(str(n)), self)
            sc.activated.connect(lambda num=n: self.assign_class_by_number(num))
            self._number_shortcuts.append(sc)

    # ---------------- video / image handling ----------------
    def load_video(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "เลือกวิดีโอ", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if not path:
            return
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            QMessageBox.warning(self, "ผิดพลาด", "ไม่สามารถเปิดวิดีโอนี้ได้")
            return
        if self.cap:
            self.cap.release()
        self.cap = cap
        self.source_mode = "video"
        self.image_paths = []
        self.video_path = path
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_label.setText(os.path.basename(path))
        self.slider.setRange(0, max(0, self.total_frames - 1))
        self.frame_spin.setRange(0, max(0, self.total_frames - 1))
        self.range_from.setRange(0, max(0, self.total_frames - 1))
        self.range_to.setRange(0, max(0, self.total_frames - 1))
        self.frame_class = {}
        self.frame_crops = {}
        self.action_stack = []
        self.current_idx = 0
        self.slider.setValue(0)
        self.show_frame(0)

    def load_images_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "เลือกรูปภาพ (เลือกได้หลายไฟล์)", "",
            "Image Files (*.jpg *.jpeg *.png *.bmp)"
        )
        if not paths:
            return
        self._load_images_common(sorted(paths))

    def load_images_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "เลือกโฟลเดอร์รูปภาพ")
        if not folder:
            return
        exts = (".jpg", ".jpeg", ".png", ".bmp")
        paths = sorted(
            os.path.join(folder, f) for f in os.listdir(folder)
            if f.lower().endswith(exts)
        )
        if not paths:
            QMessageBox.warning(self, "แจ้งเตือน", "ไม่พบไฟล์รูปภาพในโฟลเดอร์นี้")
            return
        self._load_images_common(paths)

    def _load_images_common(self, paths):
        if self.cap:
            self.cap.release()
            self.cap = None
        self.source_mode = "images"
        self.image_paths = paths
        self.video_path = None
        self.total_frames = len(paths)
        self.video_label.setText(f"โหมดรูปภาพ: {len(paths)} ไฟล์")
        self.slider.setRange(0, max(0, self.total_frames - 1))
        self.frame_spin.setRange(0, max(0, self.total_frames - 1))
        self.range_from.setRange(0, max(0, self.total_frames - 1))
        self.range_to.setRange(0, max(0, self.total_frames - 1))
        self.frame_class = {}
        self.frame_crops = {}
        self.action_stack = []
        self.current_idx = 0
        self.slider.setValue(0)
        self.show_frame(0)

    def show_frame(self, idx):
        if self.source_mode == "video":
            if not self.cap or self.total_frames == 0:
                return
            idx = max(0, min(self.total_frames - 1, idx))
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = self.cap.read()
            if not ret:
                return
        elif self.source_mode == "images":
            if not self.image_paths:
                return
            idx = max(0, min(self.total_frames - 1, idx))
            frame = cv2.imread(self.image_paths[idx])
            if frame is None:
                return
        else:
            return

        self.current_idx = idx
        self.current_frame = frame
        self.canvas.set_frame(frame)
        self._refresh_crop_overlay()
        self._refresh_crop_list()

        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)
        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(idx)
        self.frame_spin.blockSignals(False)

        assigned = self.frame_class.get(idx)
        self.status_label.setText(f"เฟรมนี้: {'ยังไม่กำหนด' if not assigned else assigned}")

    def on_slider_changed(self, val):
        self.show_frame(val)

    def on_spin_changed(self, val):
        self.show_frame(val)

    def prev_frame(self):
        self.show_frame(self.current_idx - self.step_spin.value())

    def next_frame(self):
        self.show_frame(self.current_idx + self.step_spin.value())

    # ---------------- classes ----------------
    def add_class(self):
        name, ok = QInputDialog.getText(self, "เพิ่มคลาส", "ชื่อคลาส:")
        if ok and name.strip():
            self.classes.append(name.strip())
            self.class_list.addItem(name.strip())
            if self.class_list.count() == 1:
                self.class_list.setCurrentRow(0)

    def remove_class(self):
        row = self.class_list.currentRow()
        if row < 0:
            return
        self.classes.pop(row)
        self.class_list.takeItem(row)

    def current_class_name(self):
        row = self.class_list.currentRow()
        if row < 0:
            if not self.classes:
                QMessageBox.warning(self, "แจ้งเตือน", "กรุณาเพิ่มคลาสก่อน")
                return None
            row = 0
            self.class_list.setCurrentRow(0)
        return self.classes[row]

    def get_or_create_class(self, name):
        if name not in self.classes:
            self.classes.append(name)
            self.class_list.addItem(name)
        return name

    # ---------------- keyboard shortcut assignment ----------------
    def assign_class_by_number(self, n):
        idx = n - 1
        if idx >= len(self.classes):
            return
        class_name = self.classes[idx]
        self.class_list.setCurrentRow(idx)

        row = self.crop_list.currentRow()
        crops = self.frame_crops.get(self.current_idx, [])
        if 0 <= row < len(crops):
            self._assign_crop_class(row, class_name, auto_save=True)
        else:
            self.assign_and_save()

    # ---------------- crop (sub-region) handling ----------------
    def on_crop_drawn(self, x1, y1, x2, y2):
        crop = {"box": (x1, y1, x2, y2), "class_name": None}
        crops = self.frame_crops.setdefault(self.current_idx, [])
        crops.append(crop)
        self.action_stack.append(("add_crop", self.current_idx, len(crops) - 1, None))
        self._refresh_crop_overlay()
        self._refresh_crop_list()

    def _refresh_crop_overlay(self):
        crops = self.frame_crops.get(self.current_idx, [])
        display_boxes = []
        for i, crop in enumerate(crops):
            x1, y1, x2, y2 = crop["box"]
            name = crop["class_name"]
            if name and name in self.classes:
                color = COLORS[self.classes.index(name) % len(COLORS)]
                label = name
            else:
                color = (150, 150, 150)
                label = "?"
            display_boxes.append((i, x1, y1, x2, y2, label, color))
        self.canvas.set_boxes(display_boxes)

    def _refresh_crop_list(self):
        self.crop_list.clear()
        crops = self.frame_crops.get(self.current_idx, [])
        for i, crop in enumerate(crops):
            x1, y1, x2, y2 = crop["box"]
            name = crop["class_name"] or "ยังไม่กำหนด"
            self.crop_list.addItem(f"{i}: ({x1},{y1})-({x2},{y2}) -> {name}")

    def delete_selected_crop(self):
        row = self.crop_list.currentRow()
        crops = self.frame_crops.get(self.current_idx, [])
        if not (0 <= row < len(crops)):
            return
        crop = crops.pop(row)
        self.action_stack.append(("delete_crop", self.current_idx, row, crop))
        self._refresh_crop_overlay()
        self._refresh_crop_list()

    def assign_class_to_selected_crop(self):
        row = self.crop_list.currentRow()
        if row < 0:
            QMessageBox.warning(self, "แจ้งเตือน", "กรุณาเลือก crop ในลิสต์ก่อน")
            return
        class_name = self.current_class_name()
        if class_name is None:
            return
        self._assign_crop_class(row, class_name, auto_save=True)

    def _assign_crop_class(self, row, class_name, auto_save=False):
        crops = self.frame_crops.get(self.current_idx, [])
        if not (0 <= row < len(crops)):
            return
        old_name = crops[row]["class_name"]
        crops[row]["class_name"] = class_name
        self.action_stack.append(("crop_assign", self.current_idx, row, old_name))
        self._refresh_crop_overlay()
        self._refresh_crop_list()
        if auto_save:
            self._save_single_crop(self.current_idx, row)

    def _base_name(self, idx):
        if self.source_mode == "images" and 0 <= idx < len(self.image_paths):
            return os.path.splitext(os.path.basename(self.image_paths[idx]))[0]
        return f"frame_{idx:06d}"

    def _save_single_crop(self, frame_idx, row):
        if frame_idx != self.current_idx or self.current_frame is None:
            return
        crops = self.frame_crops.get(frame_idx, [])
        if not (0 <= row < len(crops)):
            return
        crop = crops[row]
        if not crop["class_name"]:
            return
        x1, y1, x2, y2 = crop["box"]
        patch = self.current_frame[max(0, y1):y2, max(0, x1):x2]
        if patch.size == 0:
            return
        class_dir = os.path.join(self.out_dir_edit.text(), crop["class_name"])
        os.makedirs(class_dir, exist_ok=True)
        fname = f"{self._base_name(frame_idx)}_crop_{row}.jpg"
        cv2.imwrite(os.path.join(class_dir, fname), patch)

    def save_all_assigned_crops(self):
        if self.current_frame is None:
            QMessageBox.warning(self, "แจ้งเตือน", "กรุณาโหลดวิดีโอหรือรูปภาพก่อน")
            return
        crops = self.frame_crops.get(self.current_idx, [])
        saved = 0
        for row, crop in enumerate(crops):
            if crop["class_name"]:
                self._save_single_crop(self.current_idx, row)
                saved += 1
        if saved == 0:
            QMessageBox.information(self, "แจ้งเตือน", "ยังไม่มี crop ที่กำหนดคลาสแล้วในเฟรมนี้")
        else:
            self.status_label.setText(f"บันทึก crop แล้ว {saved} รายการในเฟรมนี้")

    # ---------------- undo ----------------
    def undo(self):
        if not self.action_stack:
            return
        action = self.action_stack.pop()
        kind, frame_idx = action[0], action[1]

        if kind == "add_crop":
            _, _, insert_index, _ = action
            crops = self.frame_crops.get(frame_idx, [])
            if 0 <= insert_index < len(crops):
                crops.pop(insert_index)
        elif kind == "delete_crop":
            _, _, index, crop = action
            crops = self.frame_crops.setdefault(frame_idx, [])
            insert_at = index if index <= len(crops) else len(crops)
            crops.insert(insert_at, crop)
        elif kind == "crop_assign":
            _, _, row, old_name = action
            crops = self.frame_crops.get(frame_idx, [])
            if 0 <= row < len(crops):
                crops[row]["class_name"] = old_name
        elif kind == "frame_assign":
            _, _, old_name = action
            if old_name is None:
                self.frame_class.pop(frame_idx, None)
            else:
                self.frame_class[frame_idx] = old_name

        if frame_idx != self.current_idx:
            self.show_frame(frame_idx)
        else:
            self._refresh_crop_overlay()
            self._refresh_crop_list()
            assigned = self.frame_class.get(self.current_idx)
            self.status_label.setText(f"เฟรมนี้: {'ยังไม่กำหนด' if not assigned else assigned}")

    # ---------------- whole-frame assignment ----------------
    def browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "เลือกโฟลเดอร์ผลลัพธ์", self.output_dir)
        if folder:
            self.output_dir = folder
            self.out_dir_edit.setText(folder)

    def _save_frame_to_class(self, frame, idx, class_name):
        output_dir = self.out_dir_edit.text()
        class_dir = os.path.join(output_dir, class_name)
        os.makedirs(class_dir, exist_ok=True)
        img_path = os.path.join(class_dir, f"{self._base_name(idx)}.jpg")
        cv2.imwrite(img_path, frame)

    def assign_and_save(self):
        if self.current_frame is None:
            QMessageBox.warning(self, "แจ้งเตือน", "กรุณาโหลดวิดีโอหรือรูปภาพก่อน")
            return
        class_name = self.current_class_name()
        if class_name is None:
            return
        old_name = self.frame_class.get(self.current_idx)
        self.frame_class[self.current_idx] = class_name
        self.action_stack.append(("frame_assign", self.current_idx, old_name))
        self._save_frame_to_class(self.current_frame, self.current_idx, class_name)
        self.status_label.setText(f"เฟรมนี้: {class_name} (บันทึกแล้ว)")
        self.next_frame()

    def assign_range_and_save(self):
        if self.source_mode == "video" and not self.cap:
            QMessageBox.warning(self, "แจ้งเตือน", "กรุณาโหลดวิดีโอก่อน")
            return
        if self.source_mode == "images" and not self.image_paths:
            QMessageBox.warning(self, "แจ้งเตือน", "กรุณาโหลดรูปภาพก่อน")
            return
        class_name = self.current_class_name()
        if class_name is None:
            return
        f_from = self.range_from.value()
        f_to = self.range_to.value()
        if f_from > f_to:
            f_from, f_to = f_to, f_from
        count = 0
        for idx in range(f_from, f_to + 1):
            if self.source_mode == "video":
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = self.cap.read()
                if not ret:
                    continue
            else:
                if idx < 0 or idx >= len(self.image_paths):
                    continue
                frame = cv2.imread(self.image_paths[idx])
                if frame is None:
                    continue
            self.frame_class[idx] = class_name
            self._save_frame_to_class(frame, idx, class_name)
            count += 1
        QMessageBox.information(self, "เสร็จสิ้น", f"บันทึก {count} เฟรม ในคลาส '{class_name}'")
        self.show_frame(self.current_idx)

    # ---------------- Keras auto-label ----------------
    def browse_keras_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "เลือกไฟล์โมเดล Keras", "", "Keras Model (*.h5 *.keras)"
        )
        if path:
            self.keras_model_edit.setText(path)

    def browse_labels_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "เลือกไฟล์ labels", "", "Text Files (*.txt)")
        if path:
            self.labels_file_edit.setText(path)

    def _preprocessing_mode_key(self):
        idx = self.preprocess_combo.currentIndex()
        return ["rescale_0_1", "rescale_-1_1", "raw"][idx]

    def _get_keras_model(self, model_path):
        if self._cached_keras_model is not None and self._cached_keras_model_path == model_path:
            return self._cached_keras_model
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
        try:
            import tensorflow as tf
        except ImportError:
            QMessageBox.warning(
                self, "แจ้งเตือน",
                "ไม่พบไลบรารี tensorflow กรุณาติดตั้งก่อนด้วยคำสั่ง: pip install tensorflow"
            )
            return None
        try:
            model = tf.keras.models.load_model(model_path)
        except Exception as e:
            QMessageBox.warning(self, "ผิดพลาด", f"โหลดโมเดล Keras ไม่สำเร็จ: {e}")
            return None
        self._cached_keras_model = model
        self._cached_keras_model_path = model_path
        return model

    def _load_optional_labels(self):
        labels_path = self.labels_file_edit.text().strip()
        if labels_path and os.path.isfile(labels_path):
            with open(labels_path, "r", encoding="utf-8") as f:
                return [line.strip() for line in f if line.strip()]
        return None

    def auto_label_current_frame_keras(self):
        model_path = self.keras_model_edit.text().strip()
        if not model_path or not os.path.isfile(model_path):
            QMessageBox.warning(self, "แจ้งเตือน", "กรุณาเลือกไฟล์โมเดล Keras (.h5/.keras) ก่อน")
            return
        if self.current_frame is None:
            QMessageBox.warning(self, "แจ้งเตือน", "กรุณาโหลดวิดีโอหรือรูปภาพก่อน")
            return

        model = self._get_keras_model(model_path)
        if model is None:
            return

        input_size = get_model_input_size(model)
        preprocessing = self._preprocessing_mode_key()
        class_names = self._load_optional_labels()

        crops = self.frame_crops.get(self.current_idx, [])
        regions = []
        if crops and self.use_crops_check.isChecked():
            for i, crop in enumerate(crops):
                x1, y1, x2, y2 = crop["box"]
                patch = self.current_frame[max(0, y1):y2, max(0, x1):x2]
                if patch.size == 0:
                    continue
                regions.append(("crop", i, patch))
        else:
            regions.append(("frame", None, self.current_frame))

        applied = 0
        for kind, idx_or_none, patch in regions:
            try:
                batch = preprocess_image(patch, input_size, preprocessing)
                preds = model.predict(batch, verbose=0)[0]
            except Exception as e:
                QMessageBox.warning(self, "ผิดพลาด", f"ทำนายไม่สำเร็จ: {e}")
                continue

            best_idx = int(preds.argmax())
            confidence = float(preds[best_idx])
            if confidence < self.keras_conf_spin.value():
                continue

            if class_names and best_idx < len(class_names):
                name = class_names[best_idx]
            else:
                name = f"class_{best_idx}"
            self.get_or_create_class(name)

            if kind == "frame":
                old_name = self.frame_class.get(self.current_idx)
                self.frame_class[self.current_idx] = name
                self.action_stack.append(("frame_assign", self.current_idx, old_name))
                self._save_frame_to_class(self.current_frame, self.current_idx, name)
            else:
                old_name = crops[idx_or_none]["class_name"]
                crops[idx_or_none]["class_name"] = name
                self.action_stack.append(("crop_assign", self.current_idx, idx_or_none, old_name))
                self._save_single_crop(self.current_idx, idx_or_none)
            applied += 1

        self._refresh_crop_overlay()
        self._refresh_crop_list()
        assigned = self.frame_class.get(self.current_idx)
        self.status_label.setText(f"เฟรมนี้: {'ยังไม่กำหนด' if not assigned else assigned}")
        self.auto_status_label.setText(f"Auto-label แล้ว {applied} รายการในเฟรมนี้")

    def start_auto_label_remaining_keras(self):
        model_path = self.keras_model_edit.text().strip()
        if not model_path or not os.path.isfile(model_path):
            QMessageBox.warning(self, "แจ้งเตือน", "กรุณาเลือกไฟล์โมเดล Keras (.h5/.keras) ก่อน")
            return
        if self.source_mode == "video" and not self.video_path:
            QMessageBox.warning(self, "แจ้งเตือน", "กรุณาโหลดวิดีโอก่อน")
            return
        if self.source_mode == "images" and not self.image_paths:
            QMessageBox.warning(self, "แจ้งเตือน", "กรุณาโหลดรูปภาพก่อน")
            return
        if self.auto_worker and self.auto_worker.isRunning():
            QMessageBox.information(self, "แจ้งเตือน", "กำลัง auto-label อยู่ กรุณารอให้เสร็จ หรือกดหยุดก่อน")
            return

        step = self.step_spin.value()
        frame_indices = list(range(self.current_idx, self.total_frames, step))
        if not frame_indices:
            return

        frame_crops_plain = {
            idx: [c["box"] for c in crops]
            for idx, crops in self.frame_crops.items()
        }
        labels_path = self.labels_file_edit.text().strip() or None

        self.auto_progress.setMaximum(len(frame_indices))
        self.auto_progress.setValue(0)
        self.auto_status_label.setText(f"กำลัง auto-label {len(frame_indices)} รายการ...")
        self.auto_remaining_btn.setEnabled(False)
        self.auto_stop_btn.setEnabled(True)

        self.auto_worker = KerasClassifyWorker(
            source_type=self.source_mode,
            video_path=self.video_path,
            image_paths=self.image_paths,
            model_path=model_path,
            labels_path=labels_path,
            preprocessing=self._preprocessing_mode_key(),
            frame_indices=frame_indices,
            frame_crops=frame_crops_plain,
            use_crops=self.use_crops_check.isChecked(),
            conf_threshold=self.keras_conf_spin.value(),
            output_dir=self.out_dir_edit.text(),
        )
        self.auto_worker.progress_signal.connect(self.on_auto_progress)
        self.auto_worker.frame_result_signal.connect(self.on_auto_frame_result_keras)
        self.auto_worker.finished_signal.connect(self.on_auto_finished_keras)
        self.auto_worker.log_signal.connect(lambda msg: self.auto_status_label.setText(msg))
        self.auto_worker.start()

    def stop_auto_label_keras(self):
        if self.auto_worker:
            self.auto_worker.stop()

    def on_auto_progress(self, done, total):
        self.auto_progress.setMaximum(max(1, total))
        self.auto_progress.setValue(done)

    def on_auto_frame_result_keras(self, frame_idx, results):
        for kind, crop_idx, name, confidence, saved_path in results:
            self.get_or_create_class(name)
            if kind == "frame":
                self.frame_class[frame_idx] = name
            else:
                crops = self.frame_crops.get(frame_idx)
                if crops and 0 <= crop_idx < len(crops):
                    crops[crop_idx]["class_name"] = name

        if frame_idx == self.current_idx:
            self._refresh_crop_overlay()
            self._refresh_crop_list()
            assigned = self.frame_class.get(self.current_idx)
            self.status_label.setText(f"เฟรมนี้: {'ยังไม่กำหนด' if not assigned else assigned}")

    def on_auto_finished_keras(self, success, message):
        self.auto_remaining_btn.setEnabled(True)
        self.auto_stop_btn.setEnabled(False)
        if success:
            self.auto_status_label.setText(f"Auto-label {message}")
        else:
            self.auto_status_label.setText("เกิดข้อผิดพลาด: " + message)
            QMessageBox.warning(self, "ผิดพลาด", message)

    # ---------------- batch resize ----------------
    def open_batch_resize_dialog(self):
        dialog = BatchResizeDialog(self, "classification", self.out_dir_edit.text())
        dialog.exec_()

    # ---------------- train/val split ----------------
    def run_train_val_split(self):
        dataset_dir = self.out_dir_edit.text().strip()
        if not dataset_dir or not os.path.isdir(dataset_dir):
            QMessageBox.warning(self, "แจ้งเตือน", "กรุณาระบุโฟลเดอร์ dataset ที่ถูกต้องก่อน")
            return

        move = self.move_files_check.isChecked()
        msg = (
            "การย้ายไฟล์ไม่สามารถย้อนกลับได้ ควรสำรองข้อมูลก่อน ต้องการดำเนินการต่อหรือไม่?"
            if move else
            "จะคัดลอกไฟล์เข้าโฟลเดอร์ train/ และ val/ (ไฟล์ต้นฉบับยังอยู่ครบ) ดำเนินการต่อหรือไม่?"
        )
        confirm = QMessageBox.question(self, "ยืนยัน", msg)
        if confirm != QMessageBox.Yes:
            return

        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            train_count, val_count, class_counts = split_train_val(
                dataset_dir, train_ratio=self.train_ratio_spin.value(), move=move
            )
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.warning(self, "ผิดพลาด", f"แบ่ง train/val ไม่สำเร็จ: {e}")
            return
        QApplication.restoreOverrideCursor()

        summary_lines = [f"{name}: train {t}, val {v}" for name, (t, v) in class_counts.items()]
        summary = "\n".join(summary_lines)
        self.split_status_label.setText(
            f"เสร็จสิ้น! train ทั้งหมด {train_count} ภาพ, val ทั้งหมด {val_count} ภาพ"
        )
        QMessageBox.information(
            self, "สำเร็จ",
            f"แบ่ง Train/Val เสร็จแล้ว\n\nTrain: {train_count} ภาพ\nVal: {val_count} ภาพ\n\n{summary}"
        )
