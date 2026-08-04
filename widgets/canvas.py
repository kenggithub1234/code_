import cv2
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage, QPolygon
from PyQt5.QtCore import Qt, QRect, QPoint, pyqtSignal


class ImageCanvas(QWidget):
    """Widget that shows a video frame and lets the user draw annotations
    on top of it. Coordinates emitted by box_drawn / polygon_drawn are in
    ORIGINAL IMAGE pixel space (not widget space).

    draw_mode="box" (default): drag out bounding boxes, emits box_drawn.
    draw_mode="polygon": click point-by-point, double-click or right-click
    to close the shape (needs >= 3 points), Esc cancels the current
    in-progress polygon; emits polygon_drawn.
    """

    box_drawn = pyqtSignal(int, int, int, int)  # x1, y1, x2, y2
    polygon_drawn = pyqtSignal(list)            # [(x, y), ...]

    def __init__(self, parent=None, draw_enabled=True, draw_mode="box"):
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)
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
        self.draw_mode = draw_mode

        # list of (class_id, x1, y1, x2, y2, class_name, (r,g,b))
        self.boxes = []
        # list of (poly_id, [(x,y), ...], class_name, (r,g,b))
        self.polygons = []
        # points of the polygon currently being drawn (image coords)
        self.current_polygon = []
        self._last_mouse_pos = None
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

    def set_polygons(self, polygons):
        self.polygons = polygons
        self.update()

    def set_draw_mode(self, mode):
        self.draw_mode = mode
        self.current_polygon = []
        self.drawing = False
        self.start_point = None
        self.end_point = None
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

    def to_widget_point(self, x, y):
        return QPoint(int(x * self.scale + self.offset_x), int(y * self.scale + self.offset_y))

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

            for poly_id, points, class_name, color in self.polygons:
                if len(points) < 2:
                    continue
                widget_points = [self.to_widget_point(x, y) for x, y in points]
                pen = QPen(QColor(*color), 2)
                painter.setPen(pen)
                painter.drawPolygon(QPolygon(widget_points))
                painter.drawText(
                    widget_points[0].x() + 2, max(12, widget_points[0].y() - 4), str(class_name)
                )

            if self.draw_mode == "polygon":
                if self.current_polygon:
                    pen = QPen(QColor(0, 255, 0), 2)
                    painter.setPen(pen)
                    widget_points = [self.to_widget_point(x, y) for x, y in self.current_polygon]
                    for i in range(len(widget_points) - 1):
                        painter.drawLine(widget_points[i], widget_points[i + 1])
                    for wp in widget_points:
                        painter.drawEllipse(wp, 3, 3)
                    if self._last_mouse_pos is not None:
                        dash_pen = QPen(QColor(0, 255, 0), 1, Qt.DashLine)
                        painter.setPen(dash_pen)
                        painter.drawLine(widget_points[-1], self._last_mouse_pos)
            elif self.drawing and self.start_point and self.end_point:
                pen = QPen(QColor(0, 255, 0), 2, Qt.DashLine)
                painter.setPen(pen)
                rect = QRect(self.start_point, self.end_point).normalized()
                painter.drawRect(rect)
        else:
            painter.setPen(QColor(200, 200, 200))
            painter.drawText(self.rect(), Qt.AlignCenter, "ไม่มีภาพ")

    def _finish_polygon(self):
        if len(self.current_polygon) >= 3:
            self.polygon_drawn.emit(list(self.current_polygon))
        self.current_polygon = []
        self.update()

    def mousePressEvent(self, event):
        if not self.pixmap or not self.draw_enabled:
            return

        if self.draw_mode == "polygon":
            if event.button() == Qt.LeftButton:
                x, y = self.to_image_coords(event.pos())
                self.current_polygon.append((x, y))
                self.update()
            elif event.button() == Qt.RightButton:
                self._finish_polygon()
            return

        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.start_point = event.pos()
            self.end_point = event.pos()
            self.update()

    def mouseDoubleClickEvent(self, event):
        if self.draw_mode == "polygon" and event.button() == Qt.LeftButton:
            # the 2nd click of the double-click already added a (near-)duplicate
            # point via mousePressEvent, drop it before closing the shape
            if len(self.current_polygon) >= 2:
                self.current_polygon.pop()
            self._finish_polygon()

    def mouseMoveEvent(self, event):
        if self.draw_mode == "polygon":
            self._last_mouse_pos = event.pos()
            if self.current_polygon:
                self.update()
            return

        if self.drawing:
            self.end_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if self.draw_mode == "polygon":
            return
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

    def keyPressEvent(self, event):
        if self.draw_mode == "polygon" and event.key() == Qt.Key_Escape:
            self.current_polygon = []
            self.update()
        else:
            super().keyPressEvent(event)
