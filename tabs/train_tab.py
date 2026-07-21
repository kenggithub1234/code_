import os
import sys
import io
import traceback

from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QLineEdit,
    QFileDialog, QComboBox, QSpinBox, QTextEdit, QGroupBox, QMessageBox,
    QApplication
)
from PyQt5.QtCore import QThread, pyqtSignal


class _QtLogStream(io.TextIOBase):
    """Writable stream that forwards each printed line to a Qt signal.
    Used to capture ultralytics' console output (print / tqdm progress
    bars) into the on-screen log widget."""

    def __init__(self, emit_fn):
        super().__init__()
        self._emit_fn = emit_fn
        self._buffer = ""

    def write(self, s):
        self._buffer += s
        while "\n" in self._buffer or "\r" in self._buffer:
            idx_n = self._buffer.find("\n")
            idx_r = self._buffer.find("\r")
            candidates = [i for i in (idx_n, idx_r) if i != -1]
            idx = min(candidates)
            line = self._buffer[:idx]
            self._buffer = self._buffer[idx + 1:]
            if line.strip():
                self._emit_fn(line)
        return len(s)

    def flush(self):
        pass


class TrainWorker(QThread):
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, model_name, data_yaml, epochs, imgsz, batch, device, project_dir, run_name):
        super().__init__()
        self.model_name = model_name
        self.data_yaml = data_yaml
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        self.device = device
        self.project_dir = project_dir
        self.run_name = run_name

    def run(self):
        old_stdout, old_stderr = sys.stdout, sys.stderr
        stream = _QtLogStream(self.log_signal.emit)
        sys.stdout = stream
        sys.stderr = stream
        try:
            try:
                from ultralytics import YOLO
            except ImportError:
                self.finished_signal.emit(
                    False,
                    "ไม่พบไลบรารี ultralytics กรุณาติดตั้งก่อนด้วยคำสั่ง: pip install ultralytics",
                )
                return

            self.log_signal.emit(f"กำลังโหลดโมเดลฐาน: {self.model_name}")
            model = YOLO(self.model_name)
            self.log_signal.emit("เริ่มการเทรน...")
            results = model.train(
                data=self.data_yaml,
                epochs=self.epochs,
                imgsz=self.imgsz,
                batch=self.batch,
                device=self.device if self.device else None,
                project=self.project_dir,
                name=self.run_name,
            )
            save_dir = str(getattr(results, "save_dir", os.path.join(self.project_dir, self.run_name)))
            best_path = os.path.join(save_dir, "weights", "best.pt")
            self.finished_signal.emit(True, best_path)
        except Exception as e:
            self.log_signal.emit("เกิดข้อผิดพลาด: " + str(e))
            self.log_signal.emit(traceback.format_exc())
            self.finished_signal.emit(False, str(e))
        finally:
            sys.stdout, sys.stderr = old_stdout, old_stderr


class TrainTab(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.last_best_path = ""
        self._build_ui()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        cfg_group = QGroupBox("ตั้งค่าการเทรน YOLO11")
        cfg_layout = QVBoxLayout()

        data_bar = QHBoxLayout()
        data_bar.addWidget(QLabel("ไฟล์ data.yaml:"))
        self.data_yaml_edit = QLineEdit()
        self.data_yaml_edit.setPlaceholderText(
            "เลือกไฟล์ data.yaml ที่ได้จากแท็บ 'Label สำหรับ Object Detection'"
        )
        data_bar.addWidget(self.data_yaml_edit)
        browse_data_btn = QPushButton("เลือกไฟล์")
        browse_data_btn.clicked.connect(self.browse_data_yaml)
        data_bar.addWidget(browse_data_btn)
        cfg_layout.addLayout(data_bar)

        model_bar = QHBoxLayout()
        model_bar.addWidget(QLabel("โมเดลฐาน:"))
        self.model_combo = QComboBox()
        self.model_combo.addItems(
            ["yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt"]
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
        self.imgsz_spin.setValue(640)
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
            "หลังเทรนเสร็จ นำพาธ best.pt ไปใช้ในช่อง 'โมเดลสำหรับ Auto-label' ที่แท็บ 2 ได้เลย "
            "เพื่อช่วย pre-label เฟรมถัดไปโดยอัตโนมัติ"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#9ecbff;")
        layout.addWidget(hint)

    def browse_data_yaml(self):
        path, _ = QFileDialog.getOpenFileName(self, "เลือก data.yaml", "", "YAML Files (*.yaml *.yml)")
        if path:
            self.data_yaml_edit.setText(path)

    def browse_project_dir(self):
        folder = QFileDialog.getExistingDirectory(self, "เลือกโฟลเดอร์ผลลัพธ์", self.project_edit.text())
        if folder:
            self.project_edit.setText(folder)

    def append_log(self, text):
        self.log_text.append(text)
        sb = self.log_text.verticalScrollBar()
        sb.setValue(sb.maximum())

    def start_training(self):
        data_yaml = self.data_yaml_edit.text().strip()
        if not data_yaml or not os.path.isfile(data_yaml):
            QMessageBox.warning(
                self, "แจ้งเตือน",
                "กรุณาเลือกไฟล์ data.yaml ที่ถูกต้อง (สร้างอัตโนมัติจากแท็บ 'Label สำหรับ Object Detection' "
                "ทุกครั้งที่กดบันทึกภาพ+label)"
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

        self.worker = TrainWorker(
            model_name=self.model_combo.currentText(),
            data_yaml=data_yaml,
            epochs=self.epochs_spin.value(),
            imgsz=self.imgsz_spin.value(),
            batch=self.batch_spin.value(),
            device=self.device_edit.text().strip(),
            project_dir=project_dir,
            run_name=self.run_name_edit.text().strip() or "train1",
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
