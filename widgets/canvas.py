import cv2
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage
from PyQt5.QtCore import Qt, QRect, QPoint, pyqtSignal


class ImageCanvas(QWidget):
    """Widget that shows a video frame and lets the user drag out
    bounding boxes on top of it. Coordinates emitted by box_drawn are
    in ORIGINAL IMAGE pixel space (not widget space)."""

    box_drawn = pyqtSignal(int, int, int, int)  # x1, y1, x2, y2

    def __init__(self, parent=None, draw_enabled=True):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.pixmap = None
        self.orig_w = 0
        self.orig_h = 0
        self.scale = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0

        self.drawing = False
        self.start_point = None
        self.end_point = None
        self.draw_enabled = draw_enabled

        # list of (class_id, x1, y1, x2, y2, class_name, (r,g,b))
        self.boxes = []
        self.setMinimumSize(640, 400)

    def set_frame(self, frame_bgr):
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        self.orig_w, self.orig_h = w, h
        bytes_per_line = ch * w
        qimg = QImage(rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        self.pixmap = QPixmap.fromImage(qimg.copy())
        self.update()

    def set_boxes(self, boxes):
        self.boxes = boxes
        self.update()

    def _compute_transform(self):
        if not self.pixmap:
            return
        widget_w, widget_h = max(1, self.width()), max(1, self.height())
        pw, ph = self.pixmap.width(), self.pixmap.height()
        if pw == 0 or ph == 0:
            return
        scale_w = widget_w / pw
        scale_h = widget_h / ph
        self.scale = min(scale_w, scale_h)
        disp_w = pw * self.scale
        disp_h = ph * self.scale
        self.offset_x = (widget_w - disp_w) / 2
        self.offset_y = (widget_h - disp_h) / 2

    def to_image_coords(self, pos):
        if self.scale == 0:
            return 0, 0
        x = (pos.x() - self.offset_x) / self.scale
        y = (pos.y() - self.offset_y) / self.scale
        x = max(0, min(self.orig_w - 1 if self.orig_w else 0, x))
        y = max(0, min(self.orig_h - 1 if self.orig_h else 0, y))
        return int(x), int(y)

    def to_widget_rect(self, x1, y1, x2, y2):
        wx1 = x1 * self.scale + self.offset_x
        wy1 = y1 * self.scale + self.offset_y
        wx2 = x2 * self.scale + self.offset_x
        wy2 = y2 * self.scale + self.offset_y
        return QRect(QPoint(int(wx1), int(wy1)), QPoint(int(wx2), int(wy2)))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(35, 35, 35))
        if self.pixmap:
            self._compute_transform()
            disp_w = int(self.pixmap.width() * self.scale)
            disp_h = int(self.pixmap.height() * self.scale)
            painter.drawPixmap(int(self.offset_x), int(self.offset_y), disp_w, disp_h, self.pixmap)

            for box in self.boxes:
                class_id, x1, y1, x2, y2, class_name, color = box
                rect = self.to_widget_rect(x1, y1, x2, y2)
                pen = QPen(QColor(*color), 2)
                painter.setPen(pen)
                painter.drawRect(rect)
                painter.drawText(rect.topLeft().x() + 2, max(12, rect.topLeft().y() - 4), str(class_name))

            if self.drawing and self.start_point and self.end_point:
                pen = QPen(QColor(0, 255, 0), 2, Qt.DashLine)
                painter.setPen(pen)
                rect = QRect(self.start_point, self.end_point).normalized()
                painter.drawRect(rect)
        else:
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(self.rect(), Qt.AlignCenter, "ไม่มีภาพ")

    def mousePressEvent(self, event):
        if not self.pixmap or not self.draw_enabled:
            return
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_point = event.pos()
            self.end_point = event.pos()
            self.update()

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.drawing and event.button() == Qt.LeftButton:
            self.drawing = False
            self.end_point = event.pos()
            x1, y1 = self.to_image_coords(self.start_point)
            x2, y2 = self.to_image_coords(self.end_point)
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            if abs(x2 - x1) > 3 and abs(y2 - y1) > 3:
                self.box_drawn.emit(x1, y1, x2, y2)
            self.start_point = None
            self.end_point = None
            self.update()
