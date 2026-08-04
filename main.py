import os
import sys

# Must happen before PyQt5 is imported: on Windows, once PyQt5's QApplication
# has loaded its native Qt runtime, TensorFlow's own native DLL init
# (_pywrap_tensorflow_internal) reliably fails with a DLL initialization
# error. Importing TensorFlow first lets it claim its native dependencies
# before PyQt5 does, so later lazy `import tensorflow` calls in worker
# threads (see utils/keras_train_worker.py) just hit the module cache.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
try:
    import tensorflow  # noqa: F401
except ImportError:
    pass

from PyQt5.QtWidgets import QApplication, QMainWindow, QTabWidget

from tabs.record_tab import RecordTab
from tabs.detect_label_tab import DetectLabelTab
from tabs.classify_label_tab import ClassifyLabelTab
from tabs.train_tab import TrainTab
from tabs.train_keras_tab import TrainKerasTab
from tabs.segment_label_tab import SegmentLabelTab
from tabs.segment_train_tab import SegmentTrainTab
from tabs.classify_train_tab import ClassifyTrainTab

APP_STYLESHEET = """
QWidget {
    background-color: #262626;
    color: #e6e6e6;
    font-size: 10.5pt;
}
QMainWindow {
    background-color: #262626;
}
QTabWidget::pane {
    border: 1px solid #3a3a3a;
    background: #1f1f1f;
    top: -1px;
}
QTabBar::tab {
    background: #333333;
    color: #cfcfcf;
    padding: 10px 18px;
    border-top-left-radius: 6px;
    border-top-right-radius: 6px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background: #3f8efc;
    color: white;
    font-weight: 600;
}
QTabBar::tab:hover:!selected {
    background: #3d3d3d;
}
QGroupBox {
    border: 1px solid #454545;
    border-radius: 8px;
    margin-top: 14px;
    padding-top: 12px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: #8ec0ff;
}
QPushButton {
    background-color: #3a3a3a;
    border: 1px solid #4c4c4c;
    border-radius: 6px;
    padding: 6px 14px;
}
QPushButton:hover {
    background-color: #454545;
    border-color: #3f8efc;
}
QPushButton:pressed {
    background-color: #2d2d2d;
}
QPushButton:disabled {
    color: #767676;
    background-color: #2a2a2a;
    border-color: #3a3a3a;
}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QListWidget {
    background-color: #1b1b1b;
    border: 1px solid #454545;
    border-radius: 4px;
    padding: 4px;
    selection-background-color: #3f8efc;
}
QComboBox {
    padding-right: 22px;
}
QComboBox::drop-down {
    subcontrol-origin: padding;
    subcontrol-position: top right;
    width: 20px;
    border: none;
}
QSpinBox, QDoubleSpinBox {
    padding-right: 20px;
}
QSpinBox::up-button, QDoubleSpinBox::up-button {
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 16px;
}
QSpinBox::down-button, QDoubleSpinBox::down-button {
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 16px;
}
QSlider::groove:horizontal {
    height: 6px;
    background: #454545;
    border-radius: 3px;
}
QSlider::handle:horizontal {
    background: #3f8efc;
    width: 16px;
    margin: -6px 0;
    border-radius: 8px;
}
QCheckBox {
    spacing: 6px;
}
QProgressBar {
    border: 1px solid #454545;
    border-radius: 4px;
    text-align: center;
    background-color: #1b1b1b;
}
QProgressBar::chunk {
    background-color: #3f8efc;
    border-radius: 4px;
}
QScrollBar:vertical {
    background: #262626;
    width: 12px;
}
QScrollBar::handle:vertical {
    background: #4c4c4c;
    border-radius: 6px;
    min-height: 24px;
}
QScrollBar::handle:vertical:hover {
    background: #5c5c5c;
}
"""


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO11 Dataset Tool")
        self.resize(1500, 950)

        tabs = QTabWidget()
        self.record_tab = RecordTab()
        self.detect_tab = DetectLabelTab()
        self.classify_tab = ClassifyLabelTab()
        self.train_tab = TrainTab()
        self.train_keras_tab = TrainKerasTab()
        self.segment_label_tab = SegmentLabelTab()
        self.segment_train_tab = SegmentTrainTab()
        self.classify_train_tab = ClassifyTrainTab()

        tabs.addTab(self.record_tab, "1. บันทึกวิดีโอจากกล้อง USB")
        tabs.addTab(self.detect_tab, "2. Label สำหรับ Object Detection")
        tabs.addTab(self.classify_tab, "3. Label สำหรับ Keras Classification")
        tabs.addTab(self.segment_label_tab, "4. Label สำหรับ Segmentation")
        tabs.addTab(self.train_tab, "5. เทรนโมเดล YOLO11 Object Detection")
        tabs.addTab(self.train_keras_tab, "6. เทรนโมเดล Keras Classification")
        # tabs.addTab(self.segment_label_tab, "6. Label สำหรับ Segmentation")
        tabs.addTab(self.segment_train_tab, "7. เทรนโมเดล YOLO11 Segmentation")
        tabs.addTab(self.classify_train_tab, "8. เทรนโมเดล YOLO11 Classification")

        self.setCentralWidget(tabs)

    def closeEvent(self, event):
        self.record_tab.closeEvent(event)
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLESHEET)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
