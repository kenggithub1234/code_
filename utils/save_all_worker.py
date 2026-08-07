import os

import cv2
from PyQt5.QtCore import QThread, pyqtSignal

from utils.yolo_format import save_yolo_label


class SaveAllWorker(QThread):
    """บันทึกภาพ + label ของทุกเฟรมที่ label ไว้แล้วในครั้งเดียว

    อ่านเฟรมเองจากไฟล์วิดีโอ/ไฟล์ภาพ (เปิด VideoCapture ของตัวเอง ไม่ไปแตะ
    ตัวที่ GUI ใช้อยู่ ซึ่งไม่ปลอดภัยถ้าเรียกข้ามเธรด) และเขียนไฟล์ลงดิสก์
    เท่านั้น ไม่ยุ่งกับ widget ใดๆ ส่วนการเขียน classes.txt / data.yaml
    ปล่อยให้ฝั่ง GUI ทำหลังจบงาน
    """

    progress_signal = pyqtSignal(int, int)      # (done, total)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str, int)  # (success, message, saved_count)

    def __init__(self, frame_boxes, images_dir, labels_dir,
                 source_type="video", video_path=None, image_paths=None):
        super().__init__()
        # copy กันไว้ ป้องกันกรณีผู้ใช้แก้ label ต่อระหว่างที่กำลังบันทึก
        self.frame_boxes = {idx: list(boxes) for idx, boxes in frame_boxes.items()}
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.source_type = source_type
        self.video_path = video_path
        self.image_paths = image_paths or []
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def _read_frame(self, cap, idx):
        if self.source_type == "video":
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            return frame if ret else None
        if idx < 0 or idx >= len(self.image_paths):
            return None
        return cv2.imread(self.image_paths[idx])

    def _base_name(self, idx):
        # ใช้ชื่อไฟล์เดิมเมื่อโหลดมาจากรูปภาพ ให้ตรงกับตอนบันทึกทีละเฟรม
        if self.source_type == "images" and idx < len(self.image_paths):
            return os.path.splitext(os.path.basename(self.image_paths[idx]))[0]
        return f"frame_{idx:06d}"

    def run(self):
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.labels_dir, exist_ok=True)

        cap = None
        if self.source_type == "video":
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.finished_signal.emit(False, "ไม่สามารถเปิดไฟล์วิดีโอเพื่อบันทึกได้", 0)
                return

        indices = sorted(self.frame_boxes)
        total = len(indices)
        saved = 0
        failed = 0

        for i, idx in enumerate(indices):
            if self._stop_requested:
                self.log_signal.emit("หยุดการบันทึกตามคำขอของผู้ใช้")
                break

            frame = self._read_frame(cap, idx)
            if frame is None:
                failed += 1
                self.progress_signal.emit(i + 1, total)
                continue

            h, w = frame.shape[:2]
            base = self._base_name(idx)
            try:
                cv2.imwrite(os.path.join(self.images_dir, base + ".jpg"), frame)
                save_yolo_label(
                    os.path.join(self.labels_dir, base + ".txt"),
                    self.frame_boxes[idx], w, h,
                )
                saved += 1
            except Exception as e:  # noqa: BLE001
                failed += 1
                self.log_signal.emit(f"บันทึกรายการที่ {idx} ไม่สำเร็จ: {e}")

            self.progress_signal.emit(i + 1, total)

        if cap is not None:
            cap.release()

        msg = f"บันทึกแล้ว {saved} รายการ"
        if failed:
            msg += f" (อ่าน/เขียนไม่สำเร็จ {failed} รายการ)"
        if self._stop_requested:
            msg += " - หยุดกลางคัน"
        self.finished_signal.emit(True, msg, saved)
