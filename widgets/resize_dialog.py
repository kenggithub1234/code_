import os
from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSpinBox,
    QComboBox, QLineEdit, QFileDialog, QProgressBar, QMessageBox, QCheckBox
)

from utils.resize_worker import BatchResizeWorker


class BatchResizeDialog(QDialog):
    """kind: 'detection' (images/ + labels/, boxes get remapped),
    'segmentation' (same layout, polygon points get remapped) or
    'classification' (per-class subfolders, no labels to remap)."""

    def __init__(self, parent, kind, dataset_dir):
        super().__init__(parent)
        self.kind = kind
        self.worker = None
        self.setWindowTitle("Batch Resize รูปทั้ง Dataset")
        self.setMinimumWidth(520)
        self._build_ui(dataset_dir)

    def _build_ui(self, dataset_dir):
        layout = QVBoxLayout(self)

        dir_bar = QHBoxLayout()
        dir_bar.addWidget(QLabel("Dataset:"))
        self.dir_edit = QLineEdit(dataset_dir)
        dir_bar.addWidget(self.dir_edit)
        browse_btn = QPushButton("เลือกโฟลเดอร์")
        browse_btn.clicked.connect(self.browse_dataset)
        dir_bar.addWidget(browse_btn)
        layout.addLayout(dir_bar)

        size_bar = QHBoxLayout()
        size_bar.addWidget(QLabel("กว้าง:"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(32, 4096)
        self.width_spin.setSingleStep(32)
        self.width_spin.setValue(640)
        size_bar.addWidget(self.width_spin)
        size_bar.addWidget(QLabel("สูง:"))
        self.height_spin = QSpinBox()
        self.height_spin.setRange(32, 4096)
        self.height_spin.setSingleStep(32)
        self.height_spin.setValue(640)
        size_bar.addWidget(self.height_spin)
        size_bar.addStretch()
        layout.addLayout(size_bar)

        mode_bar = QHBoxLayout()
        mode_bar.addWidget(QLabel("วิธี resize:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems([
            "Letterbox (รักษาสัดส่วน + เติมขอบเทา) - แนะนำ",
            "Stretch (ยืด/บีบตรงๆ ให้พอดีขนาด)",
        ])
        mode_bar.addWidget(self.mode_combo)
        layout.addLayout(mode_bar)

        if self.kind == "detection":
            note_text = (
                "หมายเหตุ: ไฟล์ label (.txt) จะถูกคำนวณพิกัดกรอบใหม่ให้ตรงกับภาพที่ resize แล้วโดยอัตโนมัติ "
                "(ทั้ง 2 โหมด) — ไม่ต้องแก้ label เอง"
            )
        elif self.kind == "segmentation":
            note_text = (
                "หมายเหตุ: จุดของ polygon ทุกจุดจะถูกคำนวณใหม่ให้ตรงกับภาพที่ resize แล้วโดยอัตโนมัติ "
                "— ไม่ต้องแก้ label เอง"
            )
        else:
            note_text = "โครงสร้างโฟลเดอร์ย่อยตามชื่อคลาสจะถูกคงไว้เหมือนเดิม"
        note = QLabel(note_text)
        note.setWordWrap(True)
        note.setStyleSheet("color:#9ecbff;")
        layout.addWidget(note)

        self.overwrite_check = QCheckBox("เขียนทับไฟล์เดิม (ไม่แนะนำ - ควรสำรองข้อมูลก่อนกดเริ่ม)")
        layout.addWidget(self.overwrite_check)

        out_bar = QHBoxLayout()
        out_bar.addWidget(QLabel("โฟลเดอร์ผลลัพธ์:"))
        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("ว่าง = สร้างโฟลเดอร์ resized_<W>x<H> ให้อัตโนมัติ")
        out_bar.addWidget(self.output_edit)
        browse_out_btn = QPushButton("เลือกโฟลเดอร์")
        browse_out_btn.clicked.connect(self.browse_output)
        out_bar.addWidget(browse_out_btn)
        layout.addLayout(out_bar)

        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        self.status_label = QLabel("")
        layout.addWidget(self.status_label)

        btn_bar = QHBoxLayout()
        self.run_btn = QPushButton("▶ เริ่ม Resize")
        self.run_btn.setStyleSheet(
            "background-color:#27ae60;color:white;font-weight:bold;padding:8px 16px;"
        )
        self.run_btn.clicked.connect(self.start_resize)
        btn_bar.addWidget(self.run_btn)
        close_btn = QPushButton("ปิด")
        close_btn.clicked.connect(self.close)
        btn_bar.addWidget(close_btn)
        layout.addLayout(btn_bar)

    def browse_dataset(self):
        folder = QFileDialog.getExistingDirectory(self, "เลือกโฟลเดอร์ dataset", self.dir_edit.text())
        if folder:
            self.dir_edit.setText(folder)

    def browse_output(self):
        folder = QFileDialog.getExistingDirectory(self, "เลือกโฟลเดอร์ผลลัพธ์", self.dir_edit.text())
        if folder:
            self.output_edit.setText(folder)

    def start_resize(self):
        dataset_dir = self.dir_edit.text().strip()
        if not dataset_dir or not os.path.isdir(dataset_dir):
            QMessageBox.warning(self, "แจ้งเตือน", "กรุณาเลือกโฟลเดอร์ dataset ที่ถูกต้อง")
            return
        if self.worker and self.worker.isRunning():
            return

        mode = "letterbox" if self.mode_combo.currentIndex() == 0 else "stretch"
        overwrite = self.overwrite_check.isChecked()
        output_dir = self.output_edit.text().strip() or None

        if overwrite:
            confirm = QMessageBox.question(
                self, "ยืนยัน",
                "การเขียนทับไฟล์เดิมไม่สามารถย้อนกลับได้ แนะนำให้สำรองข้อมูลก่อน\n"
                "ต้องการดำเนินการต่อหรือไม่?"
            )
            if confirm != QMessageBox.Yes:
                return
            output_dir = dataset_dir

        self.run_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        self.status_label.setText("กำลังประมวลผล...")

        self.worker = BatchResizeWorker(
            kind=self.kind,
            dataset_dir=dataset_dir,
            target_w=self.width_spin.value(),
            target_h=self.height_spin.value(),
            mode=mode,
            output_dir=output_dir,
            overwrite=overwrite,
        )
        self.worker.progress_signal.connect(self.on_progress)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.start()

    def on_progress(self, done, total):
        self.progress_bar.setMaximum(max(1, total))
        self.progress_bar.setValue(done)

    def on_finished(self, success, message, count):
        self.run_btn.setEnabled(True)
        if success:
            self.status_label.setText(f"เสร็จสิ้น! ปรับขนาดภาพ {count} ไฟล์ -> {message}")
            QMessageBox.information(
                self, "สำเร็จ", f"Resize เสร็จแล้ว {count} ไฟล์\nผลลัพธ์อยู่ที่:\n{message}"
            )
        else:
            self.status_label.setText("เกิดข้อผิดพลาด: " + message)
            QMessageBox.warning(self, "ผิดพลาด", message)
