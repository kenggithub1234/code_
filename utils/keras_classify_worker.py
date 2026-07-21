import os
import cv2
from PyQt5.QtCore import QThread, pyqtSignal

from utils.keras_classify import get_model_input_size, preprocess_image


class KerasClassifyWorker(QThread):
    """Runs a trained Keras classification model over frames/images,
    reading them either from a video file or from a list of standalone
    image files, optionally classifying user-drawn sub-regions (crops)
    instead of the whole frame, and saves the resulting patch straight
    into output_dir/<class_name>/. Only lightweight bookkeeping is emitted
    back to the GUI thread - no raw image data crosses threads and no Qt
    widgets are touched here."""

    progress_signal = pyqtSignal(int, int)  # done, total
    # frame_idx, list of (kind, crop_index_or_None, class_name, confidence, saved_path)
    frame_result_signal = pyqtSignal(int, list)
    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, model_path, labels_path, preprocessing,
                 frame_indices, frame_crops, use_crops, conf_threshold, output_dir,
                 source_type="video", video_path=None, image_paths=None):
        super().__init__()
        self.source_type = source_type  # "video" or "images"
        self.video_path = video_path
        self.image_paths = image_paths or []
        self.model_path = model_path
        self.labels_path = labels_path
        self.preprocessing = preprocessing
        self.frame_indices = list(frame_indices)
        # frame_crops: dict idx -> list of plain (x1,y1,x2,y2) tuples
        self.frame_crops = frame_crops
        self.use_crops = use_crops
        self.conf_threshold = conf_threshold
        self.output_dir = output_dir
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

    def _base_name(self, idx):
        if self.source_type == "images" and 0 <= idx < len(self.image_paths):
            return os.path.splitext(os.path.basename(self.image_paths[idx]))[0]
        return f"frame_{idx:06d}"

    def run(self):
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

        try:
            import tensorflow as tf
        except ImportError:
            self.finished_signal.emit(
                False,
                "ไม่พบไลบรารี tensorflow กรุณาติดตั้งก่อนด้วยคำสั่ง: pip install tensorflow",
            )
            return

        try:
            model = tf.keras.models.load_model(self.model_path)
        except Exception as e:
            self.finished_signal.emit(False, f"โหลดโมเดล Keras ไม่สำเร็จ: {e}")
            return

        class_names = None
        if self.labels_path and os.path.isfile(self.labels_path):
            with open(self.labels_path, "r", encoding="utf-8") as f:
                class_names = [line.strip() for line in f if line.strip()]

        input_size = get_model_input_size(model)

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

            crops = self.frame_crops.get(idx, []) if self.use_crops else []
            regions = []  # (kind, crop_index_or_None, image_patch)
            if crops:
                for c_idx, (x1, y1, x2, y2) in enumerate(crops):
                    patch = frame[max(0, y1):y2, max(0, x1):x2]
                    if patch.size == 0:
                        continue
                    regions.append(("crop", c_idx, patch))
            else:
                regions.append(("frame", None, frame))

            base = self._base_name(idx)
            results = []
            for kind, crop_idx, patch in regions:
                try:
                    batch = preprocess_image(patch, input_size, self.preprocessing)
                    preds = model.predict(batch, verbose=0)[0]
                except Exception as e:
                    self.log_signal.emit(f"รายการที่ {idx}: ทำนายไม่สำเร็จ - {e}")
                    continue

                best_idx = int(preds.argmax())
                confidence = float(preds[best_idx])
                if confidence < self.conf_threshold:
                    continue

                if class_names and best_idx < len(class_names):
                    name = class_names[best_idx]
                else:
                    name = f"class_{best_idx}"

                class_dir = os.path.join(self.output_dir, name)
                os.makedirs(class_dir, exist_ok=True)
                if kind == "frame":
                    fname = f"{base}.jpg"
                else:
                    fname = f"{base}_crop_{crop_idx}.jpg"
                save_path = os.path.join(class_dir, fname)
                cv2.imwrite(save_path, patch)

                results.append((kind, crop_idx, name, confidence, save_path))

            self.frame_result_signal.emit(idx, results)
            self.progress_signal.emit(i + 1, total)

        if cap is not None:
            cap.release()

        if not self._stop_requested:
            self.finished_signal.emit(True, "เสร็จสิ้น")
        else:
            self.finished_signal.emit(True, "หยุดกลางคัน")
