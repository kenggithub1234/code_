import os
import sys
import io
import traceback

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


class YoloTrainWorker(QThread):
    """Runs ultralytics model.train() on a background thread. `data` is
    either a data.yaml path (detect/segment) or a dataset root folder
    containing train/ and val/ subfolders (classify) - ultralytics accepts
    both depending on the loaded model's task."""

    log_signal = pyqtSignal(str)
    finished_signal = pyqtSignal(bool, str)

    def __init__(self, model_name, data, epochs, imgsz, batch, device, project_dir, run_name,
                 augment_params=None):
        super().__init__()
        self.model_name = model_name
        self.data = data
        self.epochs = epochs
        self.imgsz = imgsz
        self.batch = batch
        self.device = device
        self.project_dir = project_dir
        self.run_name = run_name
        self.augment_params = augment_params or {}

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
                data=self.data,
                epochs=self.epochs,
                imgsz=self.imgsz,
                batch=self.batch,
                device=self.device if self.device else None,
                project=self.project_dir,
                name=self.run_name,
                **self.augment_params,
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
