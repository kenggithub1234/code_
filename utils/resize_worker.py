import os
from PyQt5.QtCore import QThread, pyqtSignal

from utils.batch_resize import (
    resize_detection_dataset, resize_segmentation_dataset, resize_classification_dataset
)


class BatchResizeWorker(QThread):
    progress_signal = pyqtSignal(int, int)          # (done, total)
    finished_signal = pyqtSignal(bool, str, int)     # (success, output_dir_or_error, count)

    def __init__(self, kind, dataset_dir, target_w, target_h, mode, output_dir, overwrite):
        super().__init__()
        self.kind = kind  # "detection", "segmentation" or "classification"
        self.dataset_dir = dataset_dir
        self.target_w = target_w
        self.target_h = target_h
        self.mode = mode
        self.output_dir = output_dir
        self.overwrite = overwrite

    def _progress(self, done, total):
        self.progress_signal.emit(done, total)

    def run(self):
        try:
            if self.kind in ("detection", "segmentation"):
                resize_fn = (
                    resize_detection_dataset if self.kind == "detection"
                    else resize_segmentation_dataset
                )
                count, out_images_dir, _ = resize_fn(
                    self.dataset_dir, self.target_w, self.target_h,
                    mode=self.mode, output_dir=self.output_dir, overwrite=self.overwrite,
                    progress_callback=self._progress,
                )
                out_dir = os.path.dirname(out_images_dir)
            else:
                count, out_dir = resize_classification_dataset(
                    self.dataset_dir, self.target_w, self.target_h,
                    mode=self.mode, output_dir=self.output_dir, overwrite=self.overwrite,
                    progress_callback=self._progress,
                )
            self.finished_signal.emit(True, out_dir, count)
        except Exception as e:
            self.finished_signal.emit(False, str(e), 0)
