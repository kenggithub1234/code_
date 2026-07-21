import cv2
from PyQt5.QtCore import QThread, pyqtSignal


class AutoLabelWorker(QThread):
    """Runs a trained ultralytics YOLO model over a list of frame indices,
    reading frames either from a video file or from a list of standalone
    image files, and emits detections back to the GUI thread. Class
    matching/creation is intentionally NOT done here - detections are
    emitted as (class_name, x1, y1, x2, y2) and the receiving slot (running
    in the GUI thread) is responsible for mapping names to class ids and
    touching any Qt widgets, since it's unsafe to do that from a QThread.
    """

    progress_signal = pyqtSignal(int, int)          # (done, total)
    frame_result_signal = pyqtSignal(int, list)      # (frame_idx, [(name,x1,y1,x2,y2), ...])
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, model_path, frame_indices, conf=0.25,
                 source_type="video", video_path=None, image_paths=None):
        super().__init__()
        self.source_type = source_type  # "video" or "images"
        self.video_path = video_path
        self.image_paths = image_paths or []
        self.model_path = model_path
        self.frame_indices = list(frame_indices)
        self.conf = conf
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def _read_frame(self, cap, idx):
        if self.source_type == "video":
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            return frame if ret else None
        else:  # "images"
            if idx < 0 or idx >= len(self.image_paths):
                return None
            return cv2.imread(self.image_paths[idx])

    def run(self):
        try:
            from ultralytics import YOLO
        except ImportError:
            self.finished_signal.emit(
                False,
                "ไม่พบไลบรารี ultralytics กรุณาติดตั้งก่อนด้วยคำสั่ง: pip install ultralytics",
            )
            return

        try:
            model = YOLO(self.model_path)
        except Exception as e:
            self.finished_signal.emit(False, f"โหลดโมเดลไม่สำเร็จ: {e}")
            return

        cap = None
        if self.source_type == "video":
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.finished_signal.emit(False, "ไม่สามารถเปิดไฟล์วิดีโอสำหรับ auto-label ได้")
                return

        total = len(self.frame_indices)
        for i, idx in enumerate(self.frame_indices):
            if self._stop_requested:
                self.log_signal.emit("หยุดการทำงานตามคำขอของผู้ใช้")
                break

            frame = self._read_frame(cap, idx)
            if frame is None:
                self.progress_signal.emit(i + 1, total)
                continue

            try:
                results = model.predict(source=frame, conf=self.conf, verbose=False)
            except Exception as e:
                self.log_signal.emit(f"รายการที่ {idx}: เกิดข้อผิดพลาดตอนทำนาย - {e}")
                self.progress_signal.emit(i + 1, total)
                continue

            detections = []
            if results:
                r = results[0]
                names = r.names
                boxes = r.boxes
                if boxes is not None:
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
                        if isinstance(names, dict):
                            name = names.get(cls_id, str(cls_id))
                        else:
                            name = str(names[cls_id])
                        detections.append((name, int(x1), int(y1), int(x2), int(y2)))

            self.frame_result_signal.emit(idx, detections)
            self.progress_signal.emit(i + 1, total)

        if cap is not None:
            cap.release()

        if not self._stop_requested:
            self.finished_signal.emit(True, "เสร็จสิ้น")
        else:
            self.finished_signal.emit(True, "หยุดกลางคัน")
