import os
import sys
import time
import cv2
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QLineEdit, QFileDialog, QMessageBox, QSpinBox
)
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap

from utils.camera_open_worker import CameraOpenWorker, is_ip_source, mask_credentials

# Silence OpenCV's internal logging (e.g. the harmless
# "obsensor_uvc_stream_channel ... Camera index out of range" message that
# is printed while probing camera indices that don't exist).
try:
    cv2.setLogLevel(0)  # 0 = SILENT
except Exception:
    pass

# Backends to try, in order, per OS. Some machines only work reliably
# with one specific backend, so we fall back through the list instead of
# giving up after the first failure.
if sys.platform.startswith("win"):
    BACKEND_CANDIDATES = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
elif sys.platform.startswith("linux"):
    BACKEND_CANDIDATES = [cv2.CAP_V4L2, cv2.CAP_ANY]
elif sys.platform == "darwin":
    BACKEND_CANDIDATES = [cv2.CAP_AVFOUNDATION, cv2.CAP_ANY]
else:
    BACKEND_CANDIDATES = [cv2.CAP_ANY]


class _SuppressStderr:
    """Temporarily redirects the OS-level stderr file descriptor to
    /dev/null (or NUL on Windows). Needed because some OpenCV videoio
    backends (e.g. the bundled Orbbec 'obsensor' UVC backend) print
    directly via C++ fprintf to stderr when probing a camera index that
    doesn't exist - that bypasses cv2.setLogLevel() entirely, so it can
    only be silenced at the file-descriptor level."""

    def __enter__(self):
        self._null_fd = os.open(os.devnull, os.O_RDWR)
        self._saved_stderr_fd = os.dup(2)
        os.dup2(self._null_fd, 2)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.dup2(self._saved_stderr_fd, 2)
        os.close(self._saved_stderr_fd)
        os.close(self._null_fd)
        return False


def _open_capture(index, backend):
    """Open a cv2.VideoCapture while suppressing noisy backend stderr spam."""
    with _SuppressStderr():
        cap = cv2.VideoCapture(index, backend)
        is_opened = cap.isOpened()
    return cap, is_opened


def _open_and_verify_camera(index, backends=BACKEND_CANDIDATES, read_attempts=10):
    """Try each backend in turn. A backend can report isOpened()=True even
    when the camera can't actually deliver frames (busy / no permission /
    wrong driver), so we also require at least one successful read() before
    trusting it. Returns (cap, backend_used) or (None, None)."""
    for backend in backends:
        cap, ok = _open_capture(index, backend)
        if not ok:
            cap.release()
            continue
        for _ in range(read_attempts):
            with _SuppressStderr():
                ret, frame = cap.read()
            if ret and frame is not None:
                return cap, backend
            time.sleep(0.05)
        cap.release()
    return None, None


class RecordTab(QWidget):
    def __init__(self):
        super().__init__()
        self.cap = None
        self.writer = None
        self.recording = False
        self.paused = False
        self.start_time = None
        self.elapsed_paused = 0.0
        self.current_filename = ""
        self.last_frame = None
        self.save_dir = os.path.join(os.getcwd(), "recordings")
        os.makedirs(self.save_dir, exist_ok=True)

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.connect_worker = None

        self._build_ui()
        self._refresh_cameras()

    def _build_ui(self):
        layout = QVBoxLayout(self)

        top_bar = QHBoxLayout()
        top_bar.addWidget(QLabel("ประเภทกล้อง:"))
        self.source_combo = QComboBox()
        self.source_combo.addItems(["กล้อง USB", "IP camera (RTSP/HTTP)"])
        self.source_combo.currentIndexChanged.connect(self.on_source_kind_changed)
        top_bar.addWidget(self.source_combo)

        self.camera_label = QLabel("กล้อง:")
        top_bar.addWidget(self.camera_label)
        self.camera_combo = QComboBox()
        top_bar.addWidget(self.camera_combo)

        self.refresh_btn = QPushButton("รีเฟรชรายการกล้อง")
        self.refresh_btn.clicked.connect(self._refresh_cameras)
        top_bar.addWidget(self.refresh_btn)

        self.open_btn = QPushButton("เปิดกล้อง / พรีวิว")
        self.open_btn.clicked.connect(self.open_camera)
        top_bar.addWidget(self.open_btn)
        top_bar.addStretch()
        layout.addLayout(top_bar)

        # ---- แถว URL สำหรับ IP camera (ซ่อนไว้เมื่อเลือกกล้อง USB) ----
        self.ip_bar_widget = QWidget()
        ip_bar = QHBoxLayout(self.ip_bar_widget)
        ip_bar.setContentsMargins(0, 0, 0, 0)
        ip_bar.addWidget(QLabel("URL:"))
        self.ip_url_edit = QLineEdit()
        self.ip_url_edit.setPlaceholderText(
            "rtsp://user:pass@192.168.1.10:554/stream1   |   "
            "rtsp://user:pass@192.168.1.10:554/user=admin&password=xxxx&channel=1&stream=0.sdp"
        )
        ip_bar.addWidget(self.ip_url_edit)
        layout.addWidget(self.ip_bar_widget)

        self.ip_hint = QLabel(
            "ตัวอย่าง path ตามยี่ห้อ — Hikvision: /Streaming/Channels/101 | "
            "Dahua: /cam/realmonitor?channel=1&subtype=0 | TP-Link: /stream1 | "
            "ONVIF ทั่วไป: /onvif1 | "
            "XM/Xiongmai: /user=admin&password=xxxx&channel=1&stream=0.sdp\n"
            "(บังคับใช้ RTSP over TCP ให้อัตโนมัติแล้ว รหัสผ่านจะถูกซ่อนในข้อความแจ้งเตือน)"
        )
        self.ip_hint.setWordWrap(True)
        self.ip_hint.setStyleSheet("color:#9ecbff;")
        layout.addWidget(self.ip_hint)

        res_bar = QHBoxLayout()
        res_bar.addWidget(QLabel("ความละเอียด:"))
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItems([
            "ค่าเริ่มต้นกล้อง",
            "1920x1080 (Full HD)",
            "1280x720 (HD)",
            "640x480 (VGA)",
            "320x240",
            "กำหนดเอง...",
        ])
        self.resolution_combo.currentIndexChanged.connect(self.on_resolution_preset_changed)
        res_bar.addWidget(self.resolution_combo)

        res_bar.addWidget(QLabel("กว้าง:"))
        self.width_spin = QSpinBox()
        self.width_spin.setRange(160, 7680)
        self.width_spin.setSingleStep(16)
        self.width_spin.setValue(1280)
        res_bar.addWidget(self.width_spin)

        res_bar.addWidget(QLabel("สูง:"))
        self.height_spin = QSpinBox()
        self.height_spin.setRange(120, 4320)
        self.height_spin.setSingleStep(16)
        self.height_spin.setValue(720)
        res_bar.addWidget(self.height_spin)

        apply_res_btn = QPushButton("ใช้ความละเอียดนี้")
        apply_res_btn.clicked.connect(self.apply_resolution)
        res_bar.addWidget(apply_res_btn)

        res_bar.addWidget(QLabel("FPS ที่ใช้บันทึก:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(0, 120)
        self.fps_spin.setValue(0)
        self.fps_spin.setSpecialValueText("อัตโนมัติ")
        self.fps_spin.setToolTip(
            "IP camera มักรายงานค่า FPS ไม่ตรงความจริง (คืน 0 หรือ 90000) "
            "ทำให้วิดีโอที่อัดได้เร็วหรือช้าผิดปกติ\n"
            "ถ้าเจออาการนั้น ให้ใส่ค่า FPS จริงของกล้องตรงนี้"
        )
        res_bar.addWidget(self.fps_spin)
        res_bar.addStretch()
        layout.addLayout(res_bar)

        self.preview = QLabel("ไม่มีสัญญาณภาพ - กด 'เปิดกล้อง / พรีวิว'")
        self.preview.setAlignment(Qt.AlignCenter)
        self.preview.setMinimumSize(800, 450)
        self.preview.setStyleSheet("background-color: #202020; color: white;")
        layout.addWidget(self.preview, stretch=1)

        path_bar = QHBoxLayout()
        path_bar.addWidget(QLabel("โฟลเดอร์บันทึก:"))
        self.path_edit = QLineEdit(self.save_dir)
        path_bar.addWidget(self.path_edit)
        browse_btn = QPushButton("เลือกโฟลเดอร์")
        browse_btn.clicked.connect(self.browse_folder)
        path_bar.addWidget(browse_btn)
        layout.addLayout(path_bar)

        ctrl_bar = QHBoxLayout()
        self.record_btn = QPushButton("● บันทึก")
        self.record_btn.setStyleSheet(
            "background-color:#c0392b;color:white;font-weight:bold;padding:8px;"
        )
        self.record_btn.clicked.connect(self.start_recording)
        ctrl_bar.addWidget(self.record_btn)

        self.pause_btn = QPushButton("⏸ พัก")
        self.pause_btn.setEnabled(False)
        self.pause_btn.clicked.connect(self.toggle_pause)
        ctrl_bar.addWidget(self.pause_btn)

        self.stop_btn = QPushButton("■ หยุด")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_recording)
        ctrl_bar.addWidget(self.stop_btn)

        self.capture_btn = QPushButton("📷 ถ่ายภาพ")
        self.capture_btn.setStyleSheet(
            "background-color:#2980b9;color:white;font-weight:bold;padding:8px;"
        )
        self.capture_btn.clicked.connect(self.capture_photo)
        ctrl_bar.addWidget(self.capture_btn)

        self.status_label = QLabel("สถานะ: ยังไม่ได้บันทึก | เวลา: 00:00")
        ctrl_bar.addWidget(self.status_label)
        ctrl_bar.addStretch()
        layout.addLayout(ctrl_bar)

        # ค่าเริ่มต้นเป็นกล้อง USB จึงต้องซ่อนแถว IP ไว้ก่อน
        # (currentIndexChanged ไม่ยิงเองตอนสร้าง UI)
        self.on_source_kind_changed(0)

    def _refresh_cameras(self):
        self.camera_combo.clear()
        found = []
        misses = 0
        for i in range(6):
            cap, backend = _open_and_verify_camera(i, read_attempts=2)
            if cap is not None:
                found.append(i)
                cap.release()
                misses = 0
            else:
                misses += 1
            # Stop early once we've seen a couple of consecutive empty
            # indices - avoids probing indices that will never exist.
            if misses >= 2 and found:
                break
        if not found:
            found = [0]
        for i in found:
            self.camera_combo.addItem(f"กล้อง {i}", i)

    def browse_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "เลือกโฟลเดอร์บันทึก", self.save_dir)
        if folder:
            self.save_dir = folder
            self.path_edit.setText(folder)

    def current_source(self):
        """คืนค่าแหล่งภาพปัจจุบัน: int (USB index) หรือ str (URL ของ IP camera)"""
        if self.source_combo.currentIndex() == 1:
            return self.ip_url_edit.text().strip()
        index = self.camera_combo.currentData()
        return 0 if index is None else index

    def on_source_kind_changed(self, _index):
        is_ip = self.source_combo.currentIndex() == 1
        self.ip_bar_widget.setVisible(is_ip)
        self.ip_hint.setVisible(is_ip)
        self.camera_label.setVisible(not is_ip)
        self.camera_combo.setVisible(not is_ip)
        self.refresh_btn.setVisible(not is_ip)

    def open_camera(self):
        source = self.current_source()
        if self.source_combo.currentIndex() == 1:
            if not source:
                QMessageBox.warning(self, "แจ้งเตือน", "กรุณากรอก URL ของ IP camera ก่อน")
                return
            if not is_ip_source(source):
                QMessageBox.warning(
                    self, "แจ้งเตือน",
                    "URL ต้องขึ้นต้นด้วย rtsp:// หรือ http:// เช่น\n"
                    "rtsp://user:pass@192.168.1.10:554/stream1"
                )
                return

        # ปิดตัวเดิมก่อน แล้วยกเลิกการเชื่อมต่อที่ค้างอยู่ (ถ้ามี)
        if self.timer.isActive():
            self.timer.stop()
        if self.cap:
            self.cap.release()
            self.cap = None
        if self.connect_worker and self.connect_worker.isRunning():
            self.connect_worker.cancel()

        self.open_btn.setEnabled(False)
        shown = mask_credentials(source) if isinstance(source, str) else f"กล้อง {source}"
        self.preview.setText(f"กำลังเชื่อมต่อ...\n{shown}")

        self.connect_worker = CameraOpenWorker(source)
        self.connect_worker.opened.connect(self.on_camera_opened)
        self.connect_worker.start()

    def on_camera_opened(self, cap, err):
        self.open_btn.setEnabled(True)
        if cap is None:
            self._read_fail_count = 0
            if err == "usb-no-signal":
                QMessageBox.warning(
                    self,
                    "ไม่มีสัญญาณภาพ",
                    "เปิดกล้องได้ (หรือเปิดไม่ได้เลย) แต่ไม่ได้รับภาพจากกล้อง สาเหตุที่เป็นไปได้:\n\n"
                    "1. กล้องกำลังถูกใช้งานโดยโปรแกรมอื่นอยู่ (Zoom, Teams, เบราว์เซอร์, กล้องในตัวเครื่อง) "
                    "ให้ปิดโปรแกรมเหล่านั้นก่อนแล้วลองใหม่\n"
                    "2. ยังไม่ได้อนุญาตสิทธิ์กล้องให้ Python/Terminal\n"
                    "   - macOS: System Settings > Privacy & Security > Camera > เปิดสิทธิ์ให้ Terminal/python\n"
                    "   - Windows: Settings > Privacy & security > Camera > อนุญาตแอปเดสก์ท็อปเข้าถึงกล้อง\n"
                    "3. เลือกกล้องผิดตัวในช่อง 'กล้อง' ลองกด 'รีเฟรชรายการกล้อง' แล้วเลือกตัวอื่น\n"
                    "4. ไดรเวอร์กล้อง USB มีปัญหา ลองถอดปลั๊กแล้วเสียบใหม่",
                )
            else:
                QMessageBox.warning(self, "เชื่อมต่อไม่สำเร็จ", err)
            self.preview.setText("ไม่มีสัญญาณภาพ - กด 'เปิดกล้อง / พรีวิว'")
            return

        self.cap = cap
        self._read_fail_count = 0
        if not self.timer.isActive():
            self.timer.start(33)

        # Re-apply any resolution the user had already selected (other than
        # the "camera default" option) so it survives re-opening the camera.
        # IP camera ส่วนใหญ่ไม่รับคำสั่งนี้ ต้องไปตั้งความละเอียดที่หน้าเว็บของกล้องแทน
        if (self.source_combo.currentIndex() == 0
                and self.resolution_combo.currentText() != "ค่าเริ่มต้นกล้อง"):
            self._set_capture_resolution(self.width_spin.value(), self.height_spin.value())

    def on_resolution_preset_changed(self, _index):
        text = self.resolution_combo.currentText()
        if "1920x1080" in text:
            self.width_spin.setValue(1920)
            self.height_spin.setValue(1080)
        elif "1280x720" in text:
            self.width_spin.setValue(1280)
            self.height_spin.setValue(720)
        elif "640x480" in text:
            self.width_spin.setValue(640)
            self.height_spin.setValue(480)
        elif "320x240" in text:
            self.width_spin.setValue(320)
            self.height_spin.setValue(240)
        # "ค่าเริ่มต้นกล้อง" and "กำหนดเอง..." leave the spin boxes as-is

    def _set_capture_resolution(self, w, h):
        if not self.cap:
            return None, None
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return actual_w, actual_h

    def apply_resolution(self):
        if not self.cap:
            QMessageBox.information(self, "แจ้งเตือน", "กรุณาเปิดกล้องก่อน จึงจะตั้งค่าความละเอียดได้")
            return

        if self.resolution_combo.currentText() == "ค่าเริ่มต้นกล้อง":
            # There's no reliable "reset to default" call in OpenCV, so the
            # simplest way is to close and reopen the capture device fresh.
            self.open_camera()
            return

        w = self.width_spin.value()
        h = self.height_spin.value()
        actual_w, actual_h = self._set_capture_resolution(w, h)
        if actual_w is None:
            return

        if (actual_w, actual_h) != (w, h):
            QMessageBox.information(
                self, "แจ้งเตือน",
                f"กล้องนี้ไม่รองรับความละเอียด {w}x{h} พอดี "
                f"ระบบจึงใช้ค่าที่ใกล้เคียงที่สุดที่กล้องรองรับแทนคือ {actual_w}x{actual_h}"
            )
        self.width_spin.blockSignals(True)
        self.height_spin.blockSignals(True)
        self.width_spin.setValue(actual_w)
        self.height_spin.setValue(actual_h)
        self.width_spin.blockSignals(False)
        self.height_spin.blockSignals(False)
        self.status_label.setText(f"ความละเอียดปัจจุบัน: {actual_w}x{actual_h}")

    def capture_photo(self):
        if self.last_frame is None:
            QMessageBox.warning(self, "แจ้งเตือน", "ยังไม่มีภาพจากกล้อง กรุณาเปิดกล้องก่อน")
            return
        save_dir = self.path_edit.text()
        capture_dir = os.path.join(save_dir, "captures")
        os.makedirs(capture_dir, exist_ok=True)
        filename = os.path.join(
            capture_dir, f"capture_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
        )
        # Avoid overwriting if two captures happen within the same second.
        counter = 1
        base_filename = filename
        while os.path.exists(filename):
            filename = base_filename.replace(".jpg", f"_{counter}.jpg")
            counter += 1
        cv2.imwrite(filename, self.last_frame)
        self.status_label.setText(f"ถ่ายภาพแล้ว -> {filename}")

    def update_frame(self):
        if not self.cap:
            return
        with _SuppressStderr():
            ret, frame = self.cap.read()
        if not ret or frame is None:
            self._read_fail_count = getattr(self, "_read_fail_count", 0) + 1
            # ~2 seconds of consecutive failed reads at 33ms/tick
            if self._read_fail_count > 60:
                self.preview.setText(
                    "สัญญาณภาพขาดหาย - กล้องอาจถูกถอด หรือถูกโปรแกรมอื่นแย่งใช้งาน\n"
                    "ลองกด 'เปิดกล้อง / พรีวิว' ใหม่อีกครั้ง"
                )
            return
        self._read_fail_count = 0

        if self.recording and not self.paused and self.writer is not None:
            self.writer.write(frame)

        self.last_frame = frame

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pix = QPixmap.fromImage(qimg).scaled(
            self.preview.width(), self.preview.height(), Qt.KeepAspectRatio
        )
        self.preview.setPixmap(pix)

        if self.recording:
            if self.paused:
                elapsed = self.elapsed_paused
            else:
                elapsed = time.time() - self.start_time + self.elapsed_paused
            mm = int(elapsed // 60)
            ss = int(elapsed % 60)
            state = "พัก" if self.paused else "กำลังบันทึก"
            self.status_label.setText(f"สถานะ: {state} | เวลา: {mm:02d}:{ss:02d}")

    def start_recording(self):
        if not self.cap:
            self.open_camera()
            if not self.cap:
                return
        w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
        h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
        fps = float(self.fps_spin.value())
        if fps <= 0:
            fps = self.cap.get(cv2.CAP_PROP_FPS)
            # IP camera มักคืนค่าเพี้ยน (0 หรือ 90000 ซึ่งเป็น time base ของ RTSP)
            # ถ้าใช้ค่านั้นตรงๆ ไฟล์ที่อัดจะเร็ว/ช้าผิดความจริงอย่างแรง
            if not fps or fps <= 1 or fps > 120:
                fps = 20.0

        save_dir = self.path_edit.text()
        os.makedirs(save_dir, exist_ok=True)
        filename = os.path.join(save_dir, f"record_{time.strftime('%Y%m%d_%H%M%S')}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.writer = cv2.VideoWriter(filename, fourcc, fps, (w, h))
        if not self.writer.isOpened():
            QMessageBox.warning(self, "ผิดพลาด", "ไม่สามารถสร้างไฟล์วิดีโอได้")
            self.writer = None
            return

        self.recording = True
        self.paused = False
        self.start_time = time.time()
        self.elapsed_paused = 0.0
        self.current_filename = filename

        self.record_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.pause_btn.setText("⏸ พัก")
        self.stop_btn.setEnabled(True)

    def toggle_pause(self):
        if not self.recording:
            return
        if not self.paused:
            self.paused = True
            self.elapsed_paused = time.time() - self.start_time + self.elapsed_paused
            self.pause_btn.setText("▶ บันทึกต่อ")
        else:
            self.paused = False
            self.start_time = time.time()
            self.pause_btn.setText("⏸ พัก")

    def stop_recording(self):
        if self.writer:
            self.writer.release()
            self.writer = None
        self.recording = False
        self.paused = False
        self.record_btn.setEnabled(True)
        self.pause_btn.setEnabled(False)
        self.pause_btn.setText("⏸ พัก")
        self.stop_btn.setEnabled(False)
        self.status_label.setText(f"สถานะ: บันทึกเสร็จสิ้น -> {self.current_filename}")

    def closeEvent(self, event):
        if self.writer:
            self.writer.release()
        if self.cap:
            self.cap.release()
        event.accept()
