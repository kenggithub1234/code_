import os

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QFileDialog, QComboBox, QSpinBox, QTextEdit, QGroupBox, QMessageBox,
    QApplication
)

from widgets.augment_group import AugmentGroupBox
from utils.yolo_train_worker import YoloTrainWorker


class ClassifyTrainTab(QWidget):
    """Trains a YOLO11 classification model straight from the folder-per-class
    dataset produced by the 'Label สำหรับ Classification' tab (after running
    its Train/Val split), e.g. dataset_classify/train/<class>, .../val/<class>."""

    def __init__(self):
        super().__init__()
        self.worker = None
        self.last_best_path = ""
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        cfg_group = QGroupBox("ตั้งค่าการเทรน YOLO11 Classification")
        cfg_layout = QVBoxLayout()

        data_bar = QHBoxLayout()
        data_bar.addWidget(QLabel("โฟลเดอร์ dataset (มี train/ และ val/):"))
        self.dataset_edit = QLineEdit(os.path.join(os.getcwd(), "dataset_classify"))
        self.dataset_edit.setPlaceholderText(
            "เลือกโฟลเดอร์ dataset จากแท็บ 'Label สำหรับ Classification' หลังกด 'แบ่ง Train/Val อัตโนมัติ'"
        )
        data_bar.addWidget(self.dataset_edit)
        browse_data_btn = QPushButton("เลือกโฟลเดอร์")
        browse_data_btn.clicked.connect(self.browse_dataset_dir)
        data_bar.addWidget(browse_data_btn)
        cfg_layout.addLayout(data_bar)

        model_bar = QHBoxLayout()
        model_bar.addWidget(QLabel("โมเดลฐาน:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(
            ["yolo11n-cls.pt", "yolo11s-cls.pt", "yolo11m-cls.pt", "yolo11l-cls.pt", "yolo11x-cls.pt"]
        )
        model_bar.addWidget(self.model_combo)
        model_bar.addWidget(QLabel("(ดาวน์โหลดอัตโนมัติจาก Ultralytics ครั้งแรกที่ใช้)"))
        model_bar.addStretch()
        cfg_layout.addLayout(model_bar)

        params_bar = QHBoxLayout()
        params_bar.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 100000)
        self.epochs_spin.setValue(100)
        params_bar.addWidget(self.epochs_spin)

        params_bar.addWidget(QLabel("Image size:"))
        self.imgsz_spin = QSpinBox()
        self.imgsz_spin.setRange(32, 4096)
        self.imgsz_spin.setSingleStep(32)
        self.imgsz_spin.setValue(224)
        params_bar.addWidget(self.imgsz_spin)

        params_bar.addWidget(QLabel("Batch:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(-1, 1024)
        self.batch_spin.setValue(16)
        params_bar.addWidget(self.batch_spin)

        params_bar.addWidget(QLabel("Device:"))
        self.device_edit = QLineEdit()
        self.device_edit.setPlaceholderText("ว่าง=auto, หรือใส่ 0 / cpu / 0,1")
        self.device_edit.setMaximumWidth(140)
        params_bar.addWidget(self.device_edit)
        params_bar.addStretch()
        cfg_layout.addLayout(params_bar)

        out_bar = QHBoxLayout()
        out_bar.addWidget(QLabel("โฟลเดอร์ผลลัพธ์ (project):"))
        self.project_edit = QLineEdit(os.path.join(os.getcwd(), "runs"))
        out_bar.addWidget(self.project_edit)
        browse_out_btn = QPushButton("เลือกโฟลเดอร์")
        browse_out_btn.clicked.connect(self.browse_project_dir)
        out_bar.addWidget(browse_out_btn)
        out_bar.addWidget(QLabel("ชื่อ run:"))
        self.run_name_edit = QLineEdit("train1")
        self.run_name_edit.setMaximumWidth(120)
        out_bar.addWidget(self.run_name_edit)
        cfg_layout.addLayout(out_bar)

        cfg_group.setLayout(cfg_layout)
        layout.addWidget(cfg_group)

        self.aug_group = AugmentGroupBox()
        layout.addWidget(self.aug_group)

        btn_bar = QHBoxLayout()
        self.start_btn = QPushButton("▶ เริ่มเทรน")
        self.start_btn.setStyleSheet(
            "background-color:#27ae60;color:white;font-weight:bold;padding:8px 18px;"
        )
        self.start_btn.clicked.connect(self.start_training)
        btn_bar.addWidget(self.start_btn)
        btn_bar.addStretch()
        layout.addLayout(btn_bar)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(
            "background-color:#1a1a1a;color:#c8f0c8;font-family:Consolas,'Courier New',monospace;"
        )
        layout.addWidget(self.log_text, stretch=1)

        result_bar = QHBoxLayout()
        self.result_label = QLabel("ยังไม่มีผลลัพธ์การเทรน")
        result_bar.addWidget(self.result_label, stretch=1)
        self.copy_path_btn = QPushButton("คัดลอกพาธ best.pt")
        self.copy_path_btn.setEnabled(False)
        self.copy_path_btn.clicked.connect(self.copy_best_path)
        result_bar.addWidget(self.copy_path_btn)
        layout.addLayout(result_bar)

        hint = QLabel(
            "ใช้ dataset เดียวกับแท็บ 'Label สำหรับ Classification' ได้เลย เพียงกด 'แบ่ง Train/Val อัตโนมัติ' "
            "ในแท็บนั้นก่อน เพื่อให้ได้โครงสร้างโฟลเดอร์ train/<class> และ val/<class> ตามที่ YOLO11 classify ต้องการ"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#9ecbff;")
        layout.addWidget(hint)

    def browse_dataset_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "เลือกโฟลเดอร์ dataset", self.dataset_edit.text())
        if folder:
            self.dataset_edit.setText(folder)

    def browse_project_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "เลือกโฟลเดอร์ผลลัพธ์", self.project_edit.text())
        if folder:
            self.project_edit.setText(folder)

    def append_log(self, text):
        self.log_text.append(text)
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def start_training(self):
        dataset_dir = self.dataset_edit.text().strip()
        train_dir = os.path.join(dataset_dir, "train")
        val_dir = os.path.join(dataset_dir, "val")
        if not dataset_dir or not os.path.isdir(train_dir) or not os.path.isdir(val_dir):
            QMessageBox.warning(
                self, "แจ้งเตือน",
                "กรุณาเลือกโฟลเดอร์ dataset ที่มี train/ และ val/ อยู่ข้างใน "
                "(กด 'แบ่ง Train/Val อัตโนมัติ' ในแท็บ 'Label สำหรับ Classification' ก่อน)"
            )
            return
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "แจ้งเตือน", "กำลังเทรนอยู่ กรุณารอให้เสร็จก่อน")
            return

        self.log_text.clear()
        self.result_label.setText("กำลังเทรน... (ดู progress ในกล่อง log ด้านล่าง)")
        self.copy_path_btn.setEnabled(False)
        self.start_btn.setEnabled(False)

        project_dir = self.project_edit.text().strip() or os.path.join(os.getcwd(), "runs")
        os.makedirs(project_dir, exist_ok=True)

        augment_params = self.aug_group.values()

        self.worker = YoloTrainWorker(
            model_name=self.model_combo.currentText(),
            data=dataset_dir,
            epochs=self.epochs_spin.value(),
            imgsz=self.imgsz_spin.value(),
            batch=self.batch_spin.value(),
            device=self.device_edit.text().strip(),
            project_dir=project_dir,
            run_name=self.run_name_edit.text().strip() or "train1",
            augment_params=augment_params,
        )
        self.worker.log_signal.connect(self.append_log)
        self.worker.finished_signal.connect(self.on_training_finished)
        self.worker.start()

    def on_training_finished(self, success, message):
        self.start_btn.setEnabled(True)
        if success:
            self.last_best_path = message
            self.result_label.setText(f"เทรนเสร็จสิ้น! best weights: {message}")
            self.copy_path_btn.setEnabled(True)
            QMessageBox.information(self, "สำเร็จ", f"เทรนเสร็จแล้ว\nweights ที่ดีที่สุดอยู่ที่:\n{message}")
        else:
            self.result_label.setText("การเทรนล้มเหลว: " + message)
            QMessageBox.warning(self, "ผิดพลาด", "การเทรนล้มเหลว:\n" + message)

    def copy_best_path(self):
        QApplication.clipboard().setText(self.last_best_path)
