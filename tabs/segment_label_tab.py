import os
import cv2
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSlider, QSpinBox,
    QDoubleSpinBox, QListWidget, QFileDialog, QMessageBox, QInputDialog, QLineEdit,
    QGroupBox, QCheckBox, QProgressBar, QShortcut, QDialog
)
from PyQt5.QtGui import QKeySequence
from PyQt5.QtCore import Qt

from widgets.canvas import ImageCanvas
from utils.yolo_format import save_yolo_seg_label, save_classes_file, save_data_yaml
from utils.train_val_split import (
    count_detect_dataset, has_detect_data, is_detect_split, split_detect_train_val
)
from utils.coco_format import yolo_seg_dataset_to_coco
from utils.save_all_worker import SaveAllWorker
from widgets.resize_dialog import BatchResizeDialog
from widgets.save_preview_dialog import SavePreviewDialog
from utils.auto_label_seg import AutoLabelSegWorker
from tabs.detect_label_tab import COLORS  # shared color palette for overlays


class SegmentLabelTab(QWidget):
    def __init__(self):
        super().__init__()
        self.cap = None
        self.video_path = None
        self.source_mode = "video"  # "video" or "images"
        self.image_paths = []       # used when source_mode == "images"
        self.total_frames = 0
        self.current_idx = 0
        self.current_frame = None
        # idx -> list of {"points": [(x,y), ...], "class_id": int}
        self.frame_polygons = {}
        self.classes = []
        self.output_dir = os.path.join(os.getcwd(), "dataset_segment")

        self.action_stack = []  # undo history: (action, frame_idx, polygon, extra_index)

        self.auto_model_path = ""
        self._cached_model = None
        self._cached_model_path = None
        self.auto_worker = None
        self.save_all_worker = None

        self._build_ui()

    def _build_ui(self):
        main_layout = QHBoxLayout(self)

        # ---------------- left: video + canvas ----------------
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

        self.canvas = ImageCanvas(draw_enabled=True, draw_mode="polygon")
        self.canvas.polygon_drawn.connect(self.on_polygon_drawn)
        left.addWidget(self.canvas, stretch=1)

        hint = QLabel(
            "วาด polygon: คลิกซ้ายทีละจุดรอบวัตถุ (อย่างน้อย 3 จุด) แล้วดับเบิลคลิก หรือคลิกขวา "
            "เพื่อปิดรูป กด Esc เพื่อยกเลิก polygon ที่วาดค้างอยู่"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#9ecbff;")
        left.addWidget(hint)

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

        main_layout.addLayout(left, stretch=2)

        # ---------------- right: classes / polygons / save / auto-label ----------------
        right = QVBoxLayout()

        class_group = QGroupBox("คลาส (Classes)")
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

        poly_group = QGroupBox("Polygon ในเฟรมนี้ - เลือกคลาสก่อนวาดบนภาพ")
        poly_layout = QVBoxLayout()
        self.poly_list = QListWidget()
        poly_layout.addWidget(self.poly_list)
        poly_btns = QHBoxLayout()
        del_poly_btn = QPushButton("ลบ polygon ที่เลือก")
        del_poly_btn.clicked.connect(self.delete_selected_polygon)
        poly_btns.addWidget(del_poly_btn)
        undo_btn = QPushButton("↩ Undo (Ctrl+Z)")
        undo_btn.clicked.connect(self.undo)
        poly_btns.addWidget(undo_btn)
        poly_layout.addLayout(poly_btns)
        poly_group.setLayout(poly_layout)
        right.addWidget(poly_group)

        # ---- auto-label group ----
        auto_group = QGroupBox("Auto-label ด้วยโมเดล YOLO11-seg ที่เทรนแล้ว")
        auto_layout = QVBoxLayout()

        model_bar = QHBoxLayout()
        self.auto_model_edit = QLineEdit()
        self.auto_model_edit.setPlaceholderText("เลือกไฟล์ .pt เช่น best.pt จากแท็บเทรน Segmentation")
        model_bar.addWidget(self.auto_model_edit)
        browse_model_btn = QPushButton("เลือกโมเดล")
        browse_model_btn.clicked.connect(self.browse_auto_model)
        model_bar.addWidget(browse_model_btn)
        auto_layout.addLayout(model_bar)

        conf_bar = QHBoxLayout()
        conf_bar.addWidget(QLabel("Confidence:"))
        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setMinimumWidth(90)
        self.conf_spin.setRange(0.05, 0.95)
        self.conf_spin.setSingleStep(0.05)
        self.conf_spin.setValue(0.25)
        conf_bar.addWidget(self.conf_spin)
        conf_bar.addStretch()
        auto_layout.addLayout(conf_bar)

        self.only_empty_check = QCheckBox("เฉพาะเฟรมที่ยังไม่มี polygon")
        self.only_empty_check.setChecked(True)
        auto_layout.addWidget(self.only_empty_check)

        auto_current_btn = QPushButton("Auto-label เฟรมนี้")
        auto_current_btn.clicked.connect(self.auto_label_current_frame)
        auto_layout.addWidget(auto_current_btn)

        remaining_bar = QHBoxLayout()
        self.auto_remaining_btn = QPushButton("Auto-label เฟรมที่เหลือทั้งหมด")
        self.auto_remaining_btn.clicked.connect(self.start_auto_label_remaining)
        remaining_bar.addWidget(self.auto_remaining_btn, stretch=1)

        self.auto_stop_btn = QPushButton("หยุด")
        self.auto_stop_btn.setEnabled(False)
        self.auto_stop_btn.clicked.connect(self.stop_auto_label)
        remaining_bar.addWidget(self.auto_stop_btn)
        auto_layout.addLayout(remaining_bar)

        self.auto_progress = QProgressBar()
        self.auto_progress.setValue(0)
        auto_layout.addWidget(self.auto_progress)
        self.auto_status_label = QLabel("")
        auto_layout.addWidget(self.auto_status_label)

        auto_group.setLayout(auto_layout)
        right.addWidget(auto_group)

        # ---- save / export group ----
        out_group = QGroupBox("บันทึกผลลัพธ์ (YOLO-seg format)")
        out_layout = QVBoxLayout()
        out_dir_bar = QHBoxLayout()
        self.out_dir_edit = QLineEdit(self.output_dir)
        out_dir_bar.addWidget(self.out_dir_edit)
        browse_out_btn = QPushButton("เลือกโฟลเดอร์")
        browse_out_btn.clicked.connect(self.browse_output)
        out_dir_bar.addWidget(browse_out_btn)
        out_layout.addLayout(out_dir_bar)

        self.auto_next_check = QCheckBox("ไปเฟรมถัดไปอัตโนมัติหลังบันทึก")
        self.auto_next_check.setChecked(True)
        out_layout.addWidget(self.auto_next_check)

        save_btn = QPushButton("💾 บันทึกภาพ + label (เฟรมนี้)")
        save_btn.clicked.connect(self.save_current_label)
        out_layout.addWidget(save_btn)

        save_all_bar = QHBoxLayout()
        self.save_all_btn = QPushButton("🖼 เลือกรูปที่จะบันทึก (preview)")
        self.save_all_btn.clicked.connect(self.save_all_labels)
        save_all_bar.addWidget(self.save_all_btn, stretch=1)
        self.save_all_stop_btn = QPushButton("หยุด")
        self.save_all_stop_btn.setEnabled(False)
        self.save_all_stop_btn.clicked.connect(self.stop_save_all)
        save_all_bar.addWidget(self.save_all_stop_btn)
        out_layout.addLayout(save_all_bar)

        self.save_all_progress = QProgressBar()
        self.save_all_progress.setValue(0)
        out_layout.addWidget(self.save_all_progress)

        export_coco_btn = QPushButton("📦 Export เป็น COCO format")
        export_coco_btn.clicked.connect(self.export_coco)
        out_layout.addWidget(export_coco_btn)

        batch_resize_btn = QPushButton("🔧 Batch Resize รูปทั้ง Dataset")
        batch_resize_btn.clicked.connect(self.open_batch_resize_dialog)
        out_layout.addWidget(batch_resize_btn)

        split_btn = QPushButton("✂ แยกข้อมูลเป็น train / val")
        split_btn.clicked.connect(self.split_train_val)
        out_layout.addWidget(split_btn)

        self.split_status_label = QLabel("")
        self.split_status_label.setWordWrap(True)
        self.split_status_label.setStyleSheet("color:#9ecbff;")
        out_layout.addWidget(self.split_status_label)

        out_group.setLayout(out_layout)
        right.addWidget(out_group)

        right.addStretch()
        main_layout.addLayout(right, stretch=1)

        # keyboard shortcut for undo
        self.undo_shortcut = QShortcut(QKeySequence("Ctrl+Z"), self)
        self.undo_shortcut.activated.connect(self.undo)

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
        self.frame_polygons = {}
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
        self.frame_polygons = {}
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
        self._refresh_polygon_overlay()
        self._refresh_polygon_list()

        self.slider.blockSignals(True)
        self.slider.setValue(idx)
        self.slider.blockSignals(False)
        self.frame_spin.blockSignals(True)
        self.frame_spin.setValue(idx)
        self.frame_spin.blockSignals(False)

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
        confirm = QMessageBox.question(
            self, "ยืนยัน",
            "ลบคลาสนี้? polygon เดิมที่อ้างอิง id นี้จะยังคงเก็บ id เดิมไว้"
        )
        if confirm == QMessageBox.Yes:
            self.classes.pop(row)
            self.class_list.takeItem(row)
            self._refresh_polygon_overlay()
            self._refresh_polygon_list()

    def current_class_id(self):
        row = self.class_list.currentRow()
        if row < 0:
            if not self.classes:
                QMessageBox.warning(self, "แจ้งเตือน", "กรุณาเพิ่มคลาสก่อนวาด polygon")
                return None
            row = 0
            self.class_list.setCurrentRow(0)
        return row

    def get_or_create_class_id(self, name):
        """Map a class name (e.g. from model predictions) to an id, adding
        it to the class list if it doesn't exist yet. Only ever called from
        the GUI thread."""
        if name in self.classes:
            return self.classes.index(name)
        self.classes.append(name)
        self.class_list.addItem(name)
        return len(self.classes) - 1

    # ---------------- polygons ----------------
    def on_polygon_drawn(self, points):
        class_id = self.current_class_id()
        if class_id is None:
            return
        polygon = {"points": points, "class_id": class_id}
        polys = self.frame_polygons.setdefault(self.current_idx, [])
        polys.append(polygon)
        self.action_stack.append(("add", self.current_idx, polygon, None))
        self._refresh_polygon_overlay()
        self._refresh_polygon_list()

    def _refresh_polygon_overlay(self):
        polys = self.frame_polygons.get(self.current_idx, [])
        display_polys = []
        for i, poly in enumerate(polys):
            class_id = poly["class_id"]
            name = self.classes[class_id] if class_id < len(self.classes) else str(class_id)
            color = COLORS[class_id % len(COLORS)]
            display_polys.append((i, poly["points"], name, color))
        self.canvas.set_polygons(display_polys)

    def _refresh_polygon_list(self):
        self.poly_list.clear()
        polys = self.frame_polygons.get(self.current_idx, [])
        for i, poly in enumerate(polys):
            class_id = poly["class_id"]
            name = self.classes[class_id] if class_id < len(self.classes) else str(class_id)
            self.poly_list.addItem(f"{i}: {name}  ({len(poly['points'])} จุด)")

    def delete_selected_polygon(self):
        row = self.poly_list.currentRow()
        if row < 0:
            return
        polys = self.frame_polygons.get(self.current_idx, [])
        if 0 <= row < len(polys):
            polygon = polys.pop(row)
            self.action_stack.append(("delete", self.current_idx, polygon, row))
        self._refresh_polygon_overlay()
        self._refresh_polygon_list()

    def undo(self):
        if not self.action_stack:
            return
        action, frame_idx, polygon, index = self.action_stack.pop()
        polys = self.frame_polygons.setdefault(frame_idx, [])
        if action == "add":
            if polygon in polys:
                polys.remove(polygon)
        elif action == "delete":
            insert_at = index if index is not None and index <= len(polys) else len(polys)
            polys.insert(insert_at, polygon)

        if frame_idx != self.current_idx:
            self.show_frame(frame_idx)
        else:
            self._refresh_polygon_overlay()
            self._refresh_polygon_list()

    # ---------------- auto-label ----------------
    def browse_auto_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "เลือกไฟล์โมเดล", "", "PyTorch Model (*.pt)")
        if path:
            self.auto_model_edit.setText(path)

    def _get_model(self, model_path):
        """Lazily load & cache the ultralytics model so repeated single-frame
        auto-label clicks don't reload weights every time."""
        if self._cached_model is not None and self._cached_model_path == model_path:
            return self._cached_model
        try:
            from ultralytics import YOLO
        except ImportError:
            QMessageBox.warning(
                self, "แจ้งเตือน",
                "ไม่พบไลบรารี ultralytics กรุณาติดตั้งก่อนด้วยคำสั่ง: pip install ultralytics"
            )
            return None
        try:
            model = YOLO(model_path)
        except Exception as e:
            QMessageBox.warning(self, "ผิดพลาด", f"โหลดโมเดลไม่สำเร็จ: {e}")
            return None
        self._cached_model = model
        self._cached_model_path = model_path
        return model

    def auto_label_current_frame(self):
        model_path = self.auto_model_edit.text().strip()
        if not model_path or not os.path.isfile(model_path):
            QMessageBox.warning(self, "แจ้งเตือน", "กรุณาเลือกไฟล์โมเดล .pt ก่อน")
            return
        if self.current_frame is None:
            QMessageBox.warning(self, "แจ้งเตือน", "กรุณาโหลดวิดีโอก่อน")
            return

        model = self._get_model(model_path)
        if model is None:
            return

        try:
            results = model.predict(source=self.current_frame, conf=self.conf_spin.value(), verbose=False)
        except Exception as e:
            QMessageBox.warning(self, "ผิดพลาด", f"เกิดข้อผิดพลาดตอนทำนาย: {e}")
            return

        detections = []
        if results:
            r = results[0]
            names = r.names
            boxes = r.boxes
            masks = r.masks
            if boxes is not None and masks is not None:
                for cls_tensor, poly in zip(boxes.cls, masks.xy):
                    cls_id = int(cls_tensor)
                    name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else str(names[cls_id])
                    points = [(int(x), int(y)) for x, y in poly.tolist()]
                    if len(points) >= 3:
                        detections.append((name, points))

        if not detections:
            self.auto_status_label.setText("ไม่พบวัตถุในเฟรมนี้")
            return

        polys_list = self.frame_polygons.setdefault(self.current_idx, [])
        for name, points in detections:
            class_id = self.get_or_create_class_id(name)
            polygon = {"points": points, "class_id": class_id}
            polys_list.append(polygon)
            self.action_stack.append(("add", self.current_idx, polygon, None))

        self._refresh_polygon_overlay()
        self._refresh_polygon_list()
        self.auto_status_label.setText(f"เพิ่ม polygon อัตโนมัติ {len(detections)} รายการในเฟรมนี้")

    def start_auto_label_remaining(self):
        model_path = self.auto_model_edit.text().strip()
        if not model_path or not os.path.isfile(model_path):
            QMessageBox.warning(self, "แจ้งเตือน", "กรุณาเลือกไฟล์โมเดล .pt ก่อน")
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

        self.auto_progress.setMaximum(len(frame_indices))
        self.auto_progress.setValue(0)
        self.auto_status_label.setText(f"กำลัง auto-label {len(frame_indices)} รายการ...")
        self.auto_remaining_btn.setEnabled(False)
        self.auto_stop_btn.setEnabled(True)

        self.auto_worker = AutoLabelSegWorker(
            source_type=self.source_mode,
            video_path=self.video_path,
            image_paths=self.image_paths,
            model_path=model_path,
            frame_indices=frame_indices,
            conf=self.conf_spin.value(),
        )
        self.auto_worker.progress_signal.connect(self.on_auto_progress)
        self.auto_worker.frame_result_signal.connect(self.on_auto_frame_result)
        self.auto_worker.finished_signal.connect(self.on_auto_finished)
        self.auto_worker.log_signal.connect(lambda msg: self.auto_status_label.setText(msg))
        self.auto_worker.start()

    def stop_auto_label(self):
        if self.auto_worker:
            self.auto_worker.stop()

    def on_auto_progress(self, done, total):
        self.auto_progress.setMaximum(max(1, total))
        self.auto_progress.setValue(done)

    def on_auto_frame_result(self, frame_idx, detections):
        # detections: list of (name, [(x,y), ...]) - runs in GUI thread
        existing = self.frame_polygons.get(frame_idx, [])
        if self.only_empty_check.isChecked() and existing:
            return  # don't overwrite frames the user already labeled

        new_polys = []
        for name, points in detections:
            class_id = self.get_or_create_class_id(name)
            new_polys.append({"points": points, "class_id": class_id})
        self.frame_polygons[frame_idx] = new_polys

        if frame_idx == self.current_idx:
            self._refresh_polygon_overlay()
            self._refresh_polygon_list()

    def on_auto_finished(self, success, message):
        self.auto_remaining_btn.setEnabled(True)
        self.auto_stop_btn.setEnabled(False)
        if success:
            self.auto_status_label.setText(f"Auto-label {message}")
        else:
            self.auto_status_label.setText("เกิดข้อผิดพลาด: " + message)
            QMessageBox.warning(self, "ผิดพลาด", message)

    # ---------------- saving / export ----------------
    def browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "เลือกโฟลเดอร์ผลลัพธ์", self.output_dir)
        if folder:
            self.output_dir = folder
            self.out_dir_edit.setText(folder)

    def save_current_label(self):
        if self.current_frame is None:
            QMessageBox.warning(self, "แจ้งเตือน", "กรุณาโหลดวิดีโอหรือรูปภาพก่อน")
            return
        if not self.classes:
            QMessageBox.warning(self, "แจ้งเตือน", "กรุณาเพิ่มคลาสก่อน")
            return

        output_dir = self.out_dir_edit.text()
        images_dir = os.path.join(output_dir, "images")
        labels_dir = os.path.join(output_dir, "labels")
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)

        h, w = self.current_frame.shape[:2]
        if self.source_mode == "images" and self.current_idx < len(self.image_paths):
            base = os.path.splitext(os.path.basename(self.image_paths[self.current_idx]))[0]
        else:
            base = f"frame_{self.current_idx:06d}"
        img_path = os.path.join(images_dir, base + ".jpg")
        label_path = os.path.join(labels_dir, base + ".txt")
        cv2.imwrite(img_path, self.current_frame)

        polys = self.frame_polygons.get(self.current_idx, [])
        polygons_for_save = [(p["class_id"], p["points"]) for p in polys]
        save_yolo_seg_label(label_path, polygons_for_save, w, h)
        save_classes_file(os.path.join(output_dir, "classes.txt"), self.classes)
        # ถ้าเคยแยก train/val ไว้แล้ว ต้องคง path ใน data.yaml ไว้แบบเดิม
        # ไม่อย่างนั้นการบันทึกเฟรมใหม่จะเขียนทับกลับไปเป็นแบบ flat
        save_data_yaml(
            os.path.join(output_dir, "data.yaml"), output_dir, self.classes,
            split=is_detect_split(output_dir),
        )
        self.refresh_split_status()

        if self.auto_next_check.isChecked():
            self.next_frame()

    # ---------------- preview / save all ----------------
    def save_all_labels(self):
        """เปิด preview ให้ติ๊กเลือกว่าจะบันทึก polygon ของเฟรมไหนบ้าง"""
        if not self.classes:
            QMessageBox.warning(self, "แจ้งเตือน", "กรุณาเพิ่มคลาสก่อน")
            return
        if self.source_mode == "video" and not self.video_path:
            QMessageBox.warning(self, "แจ้งเตือน", "กรุณาโหลดวิดีโอก่อน")
            return
        if self.source_mode == "images" and not self.image_paths:
            QMessageBox.warning(self, "แจ้งเตือน", "กรุณาโหลดรูปภาพก่อน")
            return
        if self.save_all_worker and self.save_all_worker.isRunning():
            QMessageBox.information(self, "แจ้งเตือน", "กำลังบันทึกอยู่ กรุณารอให้เสร็จ หรือกดหยุดก่อน")
            return
        if self.auto_worker and self.auto_worker.isRunning():
            QMessageBox.information(
                self, "แจ้งเตือน", "กำลัง auto-label อยู่ กรุณารอให้เสร็จก่อนจึงบันทึก"
            )
            return
        if not self.frame_polygons:
            QMessageBox.warning(self, "แจ้งเตือน", "ยังไม่มีเฟรมที่ label ไว้")
            return

        dialog = SavePreviewDialog(
            self,
            frame_boxes=self.frame_polygons,
            classes=self.classes,
            colors=COLORS,
            source_type=self.source_mode,
            video_path=self.video_path,
            image_paths=self.image_paths,
            shape_type="polygon",
        )
        if dialog.exec_() != QDialog.Accepted:
            return

        selected = dialog.selected_indices()
        if not selected:
            return
        to_save = {idx: self.frame_polygons.get(idx, []) for idx in selected}

        output_dir = self.out_dir_edit.text()
        self.save_all_progress.setMaximum(len(to_save))
        self.save_all_progress.setValue(0)
        self.save_all_btn.setEnabled(False)
        self.save_all_stop_btn.setEnabled(True)

        self.save_all_worker = SaveAllWorker(
            frame_boxes=to_save,
            images_dir=os.path.join(output_dir, "images"),
            labels_dir=os.path.join(output_dir, "labels"),
            source_type=self.source_mode,
            video_path=self.video_path,
            image_paths=self.image_paths,
            label_kind="segment",
        )
        self.save_all_worker.progress_signal.connect(self.on_save_all_progress)
        self.save_all_worker.log_signal.connect(lambda m: self.split_status_label.setText(m))
        self.save_all_worker.finished_signal.connect(self.on_save_all_finished)
        self.save_all_worker.start()

    def stop_save_all(self):
        if self.save_all_worker:
            self.save_all_worker.stop()

    def on_save_all_progress(self, done, total):
        self.save_all_progress.setMaximum(max(1, total))
        self.save_all_progress.setValue(done)

    def on_save_all_finished(self, success, message, saved_count):
        self.save_all_btn.setEnabled(True)
        self.save_all_stop_btn.setEnabled(False)

        if success and saved_count:
            output_dir = self.out_dir_edit.text()
            save_classes_file(os.path.join(output_dir, "classes.txt"), self.classes)
            save_data_yaml(
                os.path.join(output_dir, "data.yaml"), output_dir, self.classes,
                split=is_detect_split(output_dir),
            )

        self.refresh_split_status()
        if success:
            QMessageBox.information(self, "สำเร็จ", message)
        else:
            QMessageBox.warning(self, "ผิดพลาด", message)

    # ---------------- train/val split ----------------
    def refresh_split_status(self):
        """อัปเดตข้อความสรุปสถานะการแยก train/val ของ dataset ปัจจุบัน"""
        output_dir = self.out_dir_edit.text()
        if not has_detect_data(output_dir):
            self.split_status_label.setText("")
            return

        n_train, n_val, n_unsplit = count_detect_dataset(output_dir)
        if not is_detect_split(output_dir):
            self.split_status_label.setStyleSheet("color:#9ecbff;")
            self.split_status_label.setText(
                f"ยังไม่ได้แยก train/val ({n_unsplit} ภาพอยู่ใน images/) — กดปุ่มด้านบนเพื่อแยก"
            )
            return

        text = f"แยกแล้ว: train {n_train} ภาพ / val {n_val} ภาพ"
        if n_unsplit:
            text += (
                f"\n⚠ มีอีก {n_unsplit} ภาพที่บันทึกเพิ่มภายหลังและยังไม่ถูกแยก "
                f"(ยังไม่ถูกใช้เทรน) กดปุ่มแยกอีกครั้งเพื่อรวมเข้าไป"
            )
            self.split_status_label.setStyleSheet("color:#ffcc66;")
        else:
            self.split_status_label.setStyleSheet("color:#9ecbff;")
        self.split_status_label.setText(text)

    def split_train_val(self):
        output_dir = self.out_dir_edit.text()
        if not has_detect_data(output_dir):
            QMessageBox.warning(
                self, "แจ้งเตือน",
                "ยังไม่พบข้อมูลใน dataset นี้ กรุณาบันทึกภาพ + label อย่างน้อย 1 ภาพก่อน"
            )
            return

        percent, ok = QInputDialog.getInt(
            self, "แยก train / val",
            "สัดส่วนข้อมูลสำหรับ train (%)\n(ที่เหลือจะเป็น val)",
            80, 5, 95, 5
        )
        if not ok:
            return

        try:
            n_train, n_val, n_missing = split_detect_train_val(
                output_dir, train_ratio=percent / 100.0
            )
        except Exception as e:
            QMessageBox.warning(self, "ผิดพลาด", f"ไม่สามารถแยก train/val ได้: {e}")
            return

        save_data_yaml(
            os.path.join(output_dir, "data.yaml"), output_dir, self.classes, split=True
        )
        self.refresh_split_status()

        msg = (
            f"แยกข้อมูลเรียบร้อย\n\n"
            f"train: {n_train} ภาพ\n"
            f"val:   {n_val} ภาพ\n\n"
            f"โครงสร้างใหม่:\n"
            f"  train/images, train/labels\n"
            f"  val/images,   val/labels\n\n"
            f"data.yaml ถูกอัปเดตให้ชี้ไปที่โฟลเดอร์ใหม่แล้ว"
        )
        if n_missing:
            msg += f"\n\nหมายเหตุ: มี {n_missing} ภาพที่ไม่มีไฟล์ label จึงสร้างไฟล์เปล่าให้"
        QMessageBox.information(self, "สำเร็จ", msg)

    # ---------------- export / resize ----------------
    def export_coco(self):
        output_dir = self.out_dir_edit.text()
        if not has_detect_data(output_dir):
            QMessageBox.warning(
                self, "แจ้งเตือน",
                "ยังไม่พบข้อมูลใน dataset นี้ กรุณาบันทึกภาพ + label อย่างน้อย 1 เฟรมก่อน"
            )
            return
        try:
            json_path, n_images, n_anns = yolo_seg_dataset_to_coco(output_dir)
        except Exception as e:
            QMessageBox.warning(self, "ผิดพลาด", f"ไม่สามารถแปลงเป็น COCO ได้: {e}")
            return
        QMessageBox.information(
            self, "สำเร็จ",
            f"แปลงเป็น COCO format (segmentation) แล้ว\nไฟล์: {json_path}\n"
            f"จำนวนภาพ: {n_images}\nจำนวน polygon: {n_anns}"
        )

    def open_batch_resize_dialog(self):
        dialog = BatchResizeDialog(self, "segmentation", self.out_dir_edit.text())
        dialog.exec_()

