import os
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QFileDialog, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QGroupBox,
    QMessageBox, QCheckBox, QProgressBar, QApplication
)

from utils.keras_train_worker import KerasTrainWorker


class TrainKerasTab(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.last_model_path = ""
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        # ---------------- dataset / model config ----------------
        cfg_group = QGroupBox("ตั้งค่าการเทรน Keras Classification")
        cfg_layout = QVBoxLayout()

        data_bar = QHBoxLayout()
        data_bar.addWidget(QLabel("โฟลเดอร์ Dataset:"))
        self.dataset_edit = QLineEdit()
        self.dataset_edit.setPlaceholderText(
            "โฟลเดอร์ dataset_classify (ควรมี train/ และ val/ ข้างใน จากปุ่ม 'แบ่ง Train/Val อัตโนมัติ' ในแท็บ Classify)"
        )
        data_bar.addWidget(self.dataset_edit)
        browse_data_btn = QPushButton("เลือกโฟลเดอร์")
        browse_data_btn.clicked.connect(self.browse_dataset)
        data_bar.addWidget(browse_data_btn)
        cfg_layout.addLayout(data_bar)

        arch_bar = QHBoxLayout()
        arch_bar.addWidget(QLabel("สถาปัตยกรรมโมเดล:"))
        self.arch_combo = QComboBox()
        self.arch_combo.addItems([
            "MobileNetV2 (transfer learning) - แนะนำ, เร็วและแม่นยำแม้ dataset เล็ก",
            "EfficientNetB0 (transfer learning) - แม่นยำกว่าแต่ช้ากว่า",
            "CNN ธรรมดา (train ตั้งแต่ต้น) - ไม่โหลด internet แต่ต้องการข้อมูลเยอะกว่า",
        ])
        arch_bar.addWidget(self.arch_combo)
        cfg_layout.addLayout(arch_bar)

        params_bar = QHBoxLayout()
        params_bar.addWidget(QLabel("ขนาดภาพ:"))
        self.img_size_spin = QSpinBox()
        self.img_size_spin.setRange(32, 512)
        self.img_size_spin.setSingleStep(16)
        self.img_size_spin.setValue(224)
        params_bar.addWidget(self.img_size_spin)

        params_bar.addWidget(QLabel("Batch:"))
        self.batch_spin = QSpinBox()
        self.batch_spin.setRange(1, 512)
        self.batch_spin.setValue(16)
        params_bar.addWidget(self.batch_spin)

        params_bar.addWidget(QLabel("Epochs:"))
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(1, 10000)
        self.epochs_spin.setValue(20)
        params_bar.addWidget(self.epochs_spin)

        params_bar.addWidget(QLabel("Learning rate:"))
        self.lr_spin = QDoubleSpinBox()
        self.lr_spin.setRange(0.00001, 1.0)
        self.lr_spin.setDecimals(5)
        self.lr_spin.setSingleStep(0.0001)
        self.lr_spin.setValue(0.0005)
        params_bar.addWidget(self.lr_spin)
        params_bar.addStretch()
        cfg_layout.addLayout(params_bar)

        out_bar = QHBoxLayout()
        out_bar.addWidget(QLabel("โฟลเดอร์ผลลัพธ์ (project):"))
        self.project_edit = QLineEdit(os.path.join(os.getcwd(), "runs_keras"))
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

        # ---------------- data augmentation (toggleable) ----------------
        aug_group = QGroupBox("ปรับแต่งภาพก่อนเทรน (Data Augmentation) - ติ๊กเพื่อเปิดใช้แต่ละอย่าง")
        aug_layout = QVBoxLayout()

        rot_bar = QHBoxLayout()
        self.use_rotation_check = QCheckBox("หมุนภาพ (Rotation)")
        self.use_rotation_check.setChecked(True)
        rot_bar.addWidget(self.use_rotation_check)
        rot_bar.addWidget(QLabel("มุมสูงสุด (องศา):"))
        self.rotation_spin = QSpinBox()
        self.rotation_spin.setRange(0, 180)
        self.rotation_spin.setValue(20)
        rot_bar.addWidget(self.rotation_spin)
        rot_bar.addStretch()
        aug_layout.addLayout(rot_bar)

        wshift_bar = QHBoxLayout()
        self.use_width_shift_check = QCheckBox("เลื่อนซ้าย-ขวา (Width Shift)")
        self.use_width_shift_check.setChecked(True)
        wshift_bar.addWidget(self.use_width_shift_check)
        wshift_bar.addWidget(QLabel("สัดส่วนสูงสุด:"))
        self.width_shift_spin = QDoubleSpinBox()
        self.width_shift_spin.setRange(0.0, 0.5)
        self.width_shift_spin.setSingleStep(0.05)
        self.width_shift_spin.setValue(0.2)
        wshift_bar.addWidget(self.width_shift_spin)
        wshift_bar.addStretch()
        aug_layout.addLayout(wshift_bar)

        hshift_bar = QHBoxLayout()
        self.use_height_shift_check = QCheckBox("เลื่อนบน-ล่าง (Height Shift)")
        self.use_height_shift_check.setChecked(True)
        hshift_bar.addWidget(self.use_height_shift_check)
        hshift_bar.addWidget(QLabel("สัดส่วนสูงสุด:"))
        self.height_shift_spin = QDoubleSpinBox()
        self.height_shift_spin.setRange(0.0, 0.5)
        self.height_shift_spin.setSingleStep(0.05)
        self.height_shift_spin.setValue(0.2)
        hshift_bar.addWidget(self.height_shift_spin)
        hshift_bar.addStretch()
        aug_layout.addLayout(hshift_bar)

        bright_bar = QHBoxLayout()
        self.use_brightness_check = QCheckBox("ปรับความสว่าง (Brightness)")
        self.use_brightness_check.setChecked(True)
        bright_bar.addWidget(self.use_brightness_check)
        bright_bar.addWidget(QLabel("ต่ำสุด:"))
        self.brightness_min_spin = QDoubleSpinBox()
        self.brightness_min_spin.setRange(0.1, 2.0)
        self.brightness_min_spin.setSingleStep(0.05)
        self.brightness_min_spin.setValue(0.7)
        bright_bar.addWidget(self.brightness_min_spin)
        bright_bar.addWidget(QLabel("สูงสุด:"))
        self.brightness_max_spin = QDoubleSpinBox()
        self.brightness_max_spin.setRange(0.1, 3.0)
        self.brightness_max_spin.setSingleStep(0.05)
        self.brightness_max_spin.setValue(1.3)
        bright_bar.addWidget(self.brightness_max_spin)
        bright_bar.addStretch()
        aug_layout.addLayout(bright_bar)

        aug_note = QLabel(
            "หมายเหตุ: ยิ่งเปิดหลาย option พร้อมกัน ภาพจะยิ่งถูกบิดเบือนมากขึ้นต่อรอบ ช่วยลด overfitting "
            "แต่ถ้าตั้งค่ามากเกินไปอาจทำให้โมเดลเรียนรู้ได้ยากขึ้น ควรเริ่มจากค่าเริ่มต้นแล้วปรับดูผลลัพธ์"
        )
        aug_note.setWordWrap(True)
        aug_note.setStyleSheet("color:#9ecbff;")
        aug_layout.addWidget(aug_note)

        aug_group.setLayout(aug_layout)
        layout.addWidget(aug_group)

        # ---------------- run controls ----------------
        btn_bar = QHBoxLayout()
        self.start_btn = QPushButton("▶ เริ่มเทรน")
        self.start_btn.setStyleSheet(
            "background-color:#27ae60;color:white;font-weight:bold;padding:8px 18px;"
        )
        self.start_btn.clicked.connect(self.start_training)
        btn_bar.addWidget(self.start_btn)
        btn_bar.addStretch()
        layout.addLayout(btn_bar)

        self.epoch_progress = QProgressBar()
        layout.addWidget(self.epoch_progress)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setStyleSheet(
            "background-color:#1a1a1a;color:#c8e0ff;font-family:Consolas,'Courier New',monospace;"
        )
        layout.addWidget(self.log_text, stretch=1)

        result_bar = QHBoxLayout()
        self.result_label = QLabel("ยังไม่มีผลลัพธ์การเทรน")
        result_bar.addWidget(self.result_label, stretch=1)
        self.copy_path_btn = QPushButton("คัดลอกพาธโมเดล")
        self.copy_path_btn.setEnabled(False)
        self.copy_path_btn.clicked.connect(self.copy_model_path)
        result_bar.addWidget(self.copy_path_btn)
        layout.addLayout(result_bar)

        hint = QLabel(
            "หลังเทรนเสร็จ จะได้ไฟล์ model.keras, classes.txt, recommended_preprocessing.txt และ "
            "training_history.png ในโฟลเดอร์ run นำไฟล์ model.keras + classes.txt ไปใส่ในช่อง "
            "'Auto-label ด้วยโมเดล Keras' ที่แท็บ 3 ได้เลย (อย่าลืมเลือก Preprocessing ให้ตรงกับที่แนะนำไว้ในไฟล์ "
            "recommended_preprocessing.txt ด้วย)"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#9ecbff;")
        layout.addWidget(hint)

    def browse_dataset(self):
        folder = QFileDialog.getExistingDirectory(self, "เลือกโฟลเดอร์ dataset", self.dataset_edit.text())
        if folder:
            self.dataset_edit.setText(folder)

    def browse_project_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "เลือกโฟลเดอร์ผลลัพธ์", self.project_edit.text())
        if folder:
            self.project_edit.setText(folder)

    def _architecture_key(self):
        idx = self.arch_combo.currentIndex()
        return ["mobilenetv2", "efficientnetb0", "simple_cnn"][idx]

    def append_log(self, text):
        self.log_text.append(text)
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def start_training(self):
        dataset_dir = self.dataset_edit.text().strip()
        if not dataset_dir or not os.path.isdir(dataset_dir):
            QMessageBox.warning(self, "แจ้งเตือน", "กรุณาเลือกโฟลเดอร์ dataset ที่ถูกต้อง")
            return
        if self.worker and self.worker.isRunning():
            QMessageBox.information(self, "แจ้งเตือน", "กำลังเทรนอยู่ กรุณารอให้เสร็จก่อน")
            return

        augmentation = {
            "use_rotation": self.use_rotation_check.isChecked(),
            "rotation_range": self.rotation_spin.value(),
            "use_width_shift": self.use_width_shift_check.isChecked(),
            "width_shift_range": self.width_shift_spin.value(),
            "use_height_shift": self.use_height_shift_check.isChecked(),
            "height_shift_range": self.height_shift_spin.value(),
            "use_brightness": self.use_brightness_check.isChecked(),
            "brightness_min": self.brightness_min_spin.value(),
            "brightness_max": self.brightness_max_spin.value(),
        }

        project_dir = self.project_edit.text().strip() or os.path.join(os.getcwd(), "runs_keras")
        os.makedirs(project_dir, exist_ok=True)

        self.log_text.clear()
        self.epoch_progress.setValue(0)
        self.epoch_progress.setMaximum(max(1, self.epochs_spin.value()))
        self.result_label.setText("กำลังเทรน... (ดู progress ในกล่อง log ด้านล่าง)")
        self.copy_path_btn.setEnabled(False)
        self.start_btn.setEnabled(False)

        self.worker = KerasTrainWorker(
            dataset_dir=dataset_dir,
            architecture=self._architecture_key(),
            img_size=self.img_size_spin.value(),
            batch_size=self.batch_spin.value(),
            epochs=self.epochs_spin.value(),
            learning_rate=self.lr_spin.value(),
            augmentation=augmentation,
            output_dir=project_dir,
            run_name=self.run_name_edit.text().strip() or "train1",
        )
        self.worker.log_signal.connect(self.append_log)
        self.worker.epoch_progress_signal.connect(self.on_epoch_progress)
        self.worker.finished_signal.connect(self.on_training_finished)
        self.worker.start()

    def on_epoch_progress(self, current_epoch, total_epochs):
        self.epoch_progress.setMaximum(max(1, total_epochs))
        self.epoch_progress.setValue(current_epoch)

    def on_training_finished(self, success, message):
        self.start_btn.setEnabled(True)
        if success:
            self.last_model_path = message
            self.result_label.setText(f"เทรนเสร็จสิ้น! โมเดลอยู่ที่: {message}")
            self.copy_path_btn.setEnabled(True)
            QMessageBox.information(self, "สำเร็จ", f"เทรนเสร็จแล้ว\nโมเดลอยู่ที่:\n{message}")
        else:
            self.result_label.setText("การเทรนล้มเหลว: " + message)
            QMessageBox.warning(self, "ผิดพลาด", "การเทรนล้มเหลว:\n" + message)

    def copy_model_path(self):
        QApplication.clipboard().setText(self.last_model_path)
