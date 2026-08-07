import os
import re
import time

# ต้องตั้งก่อนสร้าง VideoCapture ตัวแรก: บังคับให้ RTSP วิ่งบน TCP
# ถ้าปล่อยเป็น UDP ตามค่าเริ่มต้น ภาพจะแตกเป็นบล็อกๆ บ่อยมากเมื่อเน็ตไม่นิ่ง
os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp")

import cv2
from PyQt5.QtCore import QThread, pyqtSignal

IP_URL_PREFIXES = ("rtsp://", "rtsps://", "http://", "https://")


def mask_credentials(url):
    """ซ่อนรหัสผ่านใน URL ก่อนเอาไปแสดง/บันทึก log

    กล้องบางยี่ห้อ (เช่นตระกูล XM/Xiongmai) ใส่รหัสผ่านซ้ำอีกรอบใน query
    string ด้วย ต้องปิดทั้งสองที่ ไม่งั้นรหัสผ่านจะโผล่ในกล่อง error

    rtsp://admin:secret@10.0.0.5/user=admin&password=secret&channel=1
      -> rtsp://admin:****@10.0.0.5/user=admin&password=****&channel=1
    """
    if not url:
        return ""
    # 1) รูปแบบมาตรฐาน //user:pass@host
    masked = re.sub(r"(//[^:/@]+:)([^@/]+)(@)", r"\1****\3", url)
    # 2) รหัสผ่านที่ซ้ำอยู่ใน query string (password=, pwd=, pass=)
    masked = re.sub(
        r"((?:password|passwd|pwd|pass)=)([^&\s]*)",
        r"\1****", masked, flags=re.IGNORECASE,
    )
    return masked


def is_ip_source(source):
    return isinstance(source, str) and source.strip().lower().startswith(IP_URL_PREFIXES)


class CameraOpenWorker(QThread):
    """เปิดกล้องในเธรดแยก เพื่อไม่ให้ GUI ค้าง

    กล้อง USB เปิดเร็วจนไม่รู้สึก แต่ IP camera ที่ URL ผิดหรือเน็ตไปไม่ถึง
    อาจค้างได้หลายสิบวินาที ถ้าเปิดบนเธรด GUI หน้าต่างจะขาวค้างเหมือนโปรแกรมแฮงค์

    ส่ง cap กลับให้ฝั่ง GUI ไปใช้ต่อ (VideoCapture ใช้ข้ามเธรดได้ ไม่เหมือน
    widget ของ Qt) โดยเธรดนี้จะเลิกยุ่งกับมันทันทีหลัง emit
    """

    opened = pyqtSignal(object, str)   # (cap หรือ None, ข้อความ error)

    def __init__(self, source, backends=None, read_attempts=10, parent=None):
        super().__init__(parent)
        self.source = source           # int (USB index) หรือ str (URL)
        self.backends = backends
        self.read_attempts = read_attempts
        self._cancelled = False

    def cancel(self):
        """ผู้ใช้เปลี่ยนใจ/เปิดกล้องตัวใหม่ระหว่างที่ยังเชื่อมต่อไม่เสร็จ
        ผลลัพธ์ที่ได้ทีหลังจะถูกทิ้งและปิด cap ให้เรียบร้อย"""
        self._cancelled = True

    # ------------------------------------------------------------------
    def _open_ip(self):
        cap = cv2.VideoCapture(self.source, cv2.CAP_FFMPEG)
        if not cap.isOpened():
            cap.release()
            return None, (
                "เชื่อมต่อ IP camera ไม่สำเร็จ\n\n"
                f"URL: {mask_credentials(self.source)}\n\n"
                "สาเหตุที่พบบ่อย:\n"
                "1. URL ไม่ถูกต้อง (path ของแต่ละยี่ห้อไม่เหมือนกัน)\n"
                "2. ชื่อผู้ใช้/รหัสผ่านผิด\n"
                "3. เครื่องนี้เข้าถึงกล้องไม่ได้ (คนละวง LAN / ติด firewall)\n"
                "4. กล้องรับการเชื่อมต่อพร้อมกันได้จำกัด และเต็มอยู่"
            )

        # ลดดีเลย์สะสม: เก็บ buffer ให้น้อยที่สุดเท่าที่ backend ยอม
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass

        for _ in range(self.read_attempts):
            if self._cancelled:
                break
            ret, frame = cap.read()
            if ret and frame is not None:
                return cap, ""
            time.sleep(0.1)

        cap.release()
        return None, (
            "เชื่อมต่อกล้องได้ แต่ไม่ได้รับภาพ\n\n"
            f"URL: {mask_credentials(self.source)}\n\n"
            "ลองตรวจสอบว่า path ของสตรีมถูกต้องหรือไม่ "
            "(กล้องหลายรุ่นมีทั้งสตรีมความละเอียดสูง/ต่ำคนละ path กัน)"
        )

    def _open_usb(self):
        from tabs.record_tab import _open_and_verify_camera  # หลีกเลี่ยง import วน
        cap, _backend = _open_and_verify_camera(
            self.source, read_attempts=self.read_attempts
        )
        if cap is None:
            return None, "usb-no-signal"
        return cap, ""

    def run(self):
        try:
            if is_ip_source(self.source):
                cap, err = self._open_ip()
            else:
                cap, err = self._open_usb()
        except Exception as e:  # noqa: BLE001
            cap, err = None, f"เกิดข้อผิดพลาดตอนเปิดกล้อง: {e}"

        if self._cancelled:
            if cap is not None:
                cap.release()
            return
        self.opened.emit(cap, err)
