from PyQt5.QtCore import QEvent
from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton, QDoubleSpinBox, QFormLayout,
    QLabel, QCheckBox, QToolTip
)

from widgets.augment_preview import (
    GROUP_HELP, PARAM_HELP, load_sample_from_dataset, load_sample_from_folder,
    make_sample_image, render_tooltip_html
)

# ค่าเริ่มต้นตามที่ ultralytics ใช้ (default.yaml) แยกเป็น 3 กลุ่มตามลักษณะการ augment
AUGMENT_DEFAULTS = {
    "hsv_h": 0.015,
    "hsv_s": 0.7,
    "hsv_v": 0.4,
    "degrees": 0.0,
    "translate": 0.1,
    "scale": 0.5,
    "shear": 0.0,
    "perspective": 0.0,
    "flipud": 0.0,
    "fliplr": 0.5,
    "mosaic": 1.0,
    "mixup": 0.0,
    "copy_paste": 0.0,
}

# พารามิเตอร์ที่อยู่ในแต่ละกลุ่ม ใช้ตอนสั่งปิดกลุ่ม (ส่งค่า 0.0 = ไม่ใช้ augment นั้น)
GROUP_KEYS = {
    "color": ["hsv_h", "hsv_s", "hsv_v"],
    "geometric": ["degrees", "translate", "scale", "shear", "perspective", "flipud", "fliplr"],
    "mix": ["mosaic", "mixup", "copy_paste"],
}

# ค่าแนะนำเมื่อผู้ใช้ติ๊กเปิดพารามิเตอร์ที่ค่าเริ่มต้นเป็น 0 (ติ๊กแล้วค่ายัง 0 จะไม่มีผลใดๆ
# จึงเติมค่าตั้งต้นที่ใช้งานได้จริงให้ แล้วผู้ใช้ปรับต่อเองได้)
SUGGESTED_ON_ENABLE = {
    "degrees": 10.0,
    "shear": 5.0,
    "perspective": 0.0005,
    "flipud": 0.5,
    "mixup": 0.1,
    "copy_paste": 0.1,
}


class AugmentGroupBox(QGroupBox):
    """Reusable 'Data Augmentation' settings panel shared by every ultralytics
    training tab (detect / segment / classify). Values map directly onto
    model.train(**kwargs) keyword arguments.

    Two levels of on/off control:
      * each of the 3 sub-groups has a checkbox in its title
      * each individual parameter has its own checkbox

    Anything unchecked reports 0.0 - how ultralytics expresses 'this
    augmentation is off' - while the number the user tuned stays on screen
    and comes back as soon as it is re-checked."""

    def __init__(self, parent=None):
        super().__init__("การเพิ่มข้อมูลภาพระหว่างเทรน (Data Augmentation)", parent)
        self.spins = {}
        self.checks = {}
        self.group_boxes = {}
        self._sample_image = make_sample_image()
        self._widget_key = {}   # widget -> ชื่อพารามิเตอร์ ใช้ตอนสร้าง tooltip
        self._build_ui()

    def set_sample_image_from_yaml(self, data_yaml_path):
        """ใช้ภาพจริงจาก dataset เป็นตัวอย่างใน tooltip ถ้าหาเจอ
        (ถ้าไม่เจอจะใช้ภาพสังเคราะห์เดิม)"""
        img = load_sample_from_dataset(data_yaml_path)
        self._sample_image = img if img is not None else make_sample_image()
        return img is not None

    def set_sample_image_from_folder(self, folder):
        """เหมือน set_sample_image_from_yaml แต่รับเป็นโฟลเดอร์ dataset (งาน classify)"""
        img = load_sample_from_folder(folder)
        self._sample_image = img if img is not None else make_sample_image()
        return img is not None

    def _build_ui(self):
        outer = QVBoxLayout()

        hint = QLabel(
            "ติ๊กที่หัวข้อกลุ่มเพื่อเปิด/ปิดทั้งกลุ่ม และติ๊กหน้าแต่ละรายการเพื่อเปิด/ปิดทีละตัว "
            "รายการที่ไม่ติ๊กจะถูกส่งเป็น 0 (ไม่ใช้) แต่ตัวเลขที่ตั้งไว้จะยังคงอยู่"
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("color:#9ecbff;font-weight:normal;")
        outer.addWidget(hint)

        row = QHBoxLayout()

        color_box, color_form = self._make_group("color", "1. ปรับสี (Color Space Augmentation)")
        self._add_row(color_form, "hsv_h", "Hue (hsv_h)", 0.0, 1.0, 0.005, 3)
        self._add_row(color_form, "hsv_s", "Saturation (hsv_s)", 0.0, 1.0, 0.05, 2)
        self._add_row(color_form, "hsv_v", "Value/Brightness (hsv_v)", 0.0, 1.0, 0.05, 2)
        row.addWidget(color_box)

        geo_box, geo_form = self._make_group("geometric", "2. ปรับรูปทรง/มุมมอง (Geometric Transformation)")
        self._add_row(geo_form, "degrees", "หมุน (degrees)", 0.0, 180.0, 1.0, 1)
        self._add_row(geo_form, "translate", "เลื่อนภาพ (translate)", 0.0, 1.0, 0.05, 2)
        self._add_row(geo_form, "scale", "ย่อ/ขยาย (scale)", 0.0, 2.0, 0.05, 2)
        self._add_row(geo_form, "shear", "บิดเฉือน (shear)", 0.0, 45.0, 1.0, 1)
        self._add_row(geo_form, "perspective", "มุมมอง (perspective)", 0.0, 0.001, 0.0001, 5)
        self._add_row(geo_form, "flipud", "พลิกบน-ล่าง (flipud)", 0.0, 1.0, 0.1, 2)
        self._add_row(geo_form, "fliplr", "พลิกซ้าย-ขวา (fliplr)", 0.0, 1.0, 0.1, 2)
        row.addWidget(geo_box)

        mix_box, mix_form = self._make_group("mix", "3. ผสมหลายภาพ (Mix-based Augmentation)")
        self._add_row(mix_form, "mosaic", "Mosaic (ต่อ 4 ภาพ)", 0.0, 1.0, 0.1, 2)
        self._add_row(mix_form, "mixup", "MixUp (ผสม 2 ภาพ)", 0.0, 1.0, 0.1, 2)
        self._add_row(mix_form, "copy_paste", "Copy-Paste", 0.0, 1.0, 0.1, 2)
        row.addWidget(mix_box)

        outer.addLayout(row)

        btn_bar = QHBoxLayout()
        btn_bar.addStretch()
        check_all_btn = QPushButton("ติ๊กทั้งหมด")
        check_all_btn.clicked.connect(lambda: self.set_all_checked(True))
        btn_bar.addWidget(check_all_btn)
        uncheck_all_btn = QPushButton("ไม่ใช้ Augmentation เลย")
        uncheck_all_btn.clicked.connect(lambda: self.set_all_checked(False))
        btn_bar.addWidget(uncheck_all_btn)
        reset_btn = QPushButton("รีเซ็ตค่า Augmentation เป็นค่าเริ่มต้น")
        reset_btn.clicked.connect(self.reset_defaults)
        btn_bar.addWidget(reset_btn)
        outer.addLayout(btn_bar)

        self.setLayout(outer)

    def _make_group(self, name, title):
        box = QGroupBox(title)
        box.setCheckable(True)
        box.setChecked(True)
        form = QFormLayout()
        box.setLayout(form)
        box.toggled.connect(lambda _checked, n=name: self._sync_group(n))
        box.setToolTip(
            f"<div style='max-width:320px'><b>{title}</b><br>{GROUP_HELP.get(name, '')}<br><br>"
            f"<i style='color:#9ecbff'>ติ๊กเปิดกลุ่มนี้ แล้วเอาเมาส์ชี้ที่แต่ละรายการ "
            f"เพื่อดูภาพตัวอย่างก่อน/หลัง</i></div>"
        )
        self.group_boxes[name] = box
        return box, form

    def _add_row(self, form, key, label, minimum, maximum, step, decimals):
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setValue(AUGMENT_DEFAULTS[key])
        self.spins[key] = spin

        # ค่าเริ่มต้นของ ultralytics ที่เป็น 0 คือ "ไม่ใช้" อยู่แล้ว จึงเริ่มต้นแบบไม่ติ๊ก
        check = QCheckBox(label)
        check.setChecked(AUGMENT_DEFAULTS[key] > 0.0)
        check.toggled.connect(lambda checked, k=key: self._on_param_toggled(k, checked))
        self.checks[key] = check

        spin.setEnabled(check.isChecked())

        # เอาเมาส์ชี้ที่ชื่อหรือช่องกรอก แล้วสร้างภาพตัวอย่างตามค่าปัจจุบัน
        for widget in (check, spin):
            self._widget_key[widget] = key
            widget.installEventFilter(self)
            widget.setToolTip(PARAM_HELP.get(key, ""))

        form.addRow(check, spin)

    def eventFilter(self, obj, event):
        if event.type() == QEvent.ToolTip:
            key = self._widget_key.get(obj)
            if key is not None:
                group = next(n for n, keys in GROUP_KEYS.items() if key in keys)
                enabled = self.group_boxes[group].isChecked() and self.checks[key].isChecked()
                QToolTip.showText(
                    event.globalPos(),
                    render_tooltip_html(key, self.spins[key].value(), enabled, self._sample_image),
                    obj,
                )
                return True
        return super().eventFilter(obj, event)

    def _on_param_toggled(self, key, checked):
        self.spins[key].setEnabled(checked)
        # ติ๊กเปิดรายการที่ค่ายังเป็น 0 จะไม่เกิดผลใดๆ จึงเติมค่าแนะนำให้เริ่มต้น
        if checked and self.spins[key].value() == 0.0 and key in SUGGESTED_ON_ENABLE:
            self.spins[key].setValue(SUGGESTED_ON_ENABLE[key])

    def _sync_group(self, name):
        """QGroupBox ที่ checkable จะสั่ง enable/disable ลูกทั้งหมดเวลาติ๊ก/ไม่ติ๊ก
        ต้องกำหนดสถานะของ spin กลับตาม checkbox ของแต่ละรายการอีกครั้ง"""
        group_on = self.group_boxes[name].isChecked()
        for key in GROUP_KEYS[name]:
            self.spins[key].setEnabled(group_on and self.checks[key].isChecked())

    def set_all_checked(self, checked):
        for box in self.group_boxes.values():
            box.setChecked(checked)
        for check in self.checks.values():
            check.setChecked(checked)
        for name in self.group_boxes:
            self._sync_group(name)

    def reset_defaults(self):
        for box in self.group_boxes.values():
            box.setChecked(True)
        for key, spin in self.spins.items():
            # ตั้งค่าตัวเลขก่อนติ๊ก เพื่อไม่ให้ _on_param_toggled เติมค่าแนะนำทับ
            spin.setValue(AUGMENT_DEFAULTS[key])
            self.checks[key].setChecked(AUGMENT_DEFAULTS[key] > 0.0)
        for name in self.group_boxes:
            self._sync_group(name)

    def enabled_groups(self):
        """ชื่อกลุ่มที่ถูกติ๊กไว้ (ใช้แสดงใน log ก่อนเริ่มเทรน)"""
        return [name for name, box in self.group_boxes.items() if box.isChecked()]

    def enabled_params(self):
        """ชื่อพารามิเตอร์ที่เปิดใช้จริง (ทั้งกลุ่มและตัวมันเองถูกติ๊ก)"""
        return [
            key
            for name, keys in GROUP_KEYS.items()
            for key in keys
            if self.group_boxes[name].isChecked() and self.checks[key].isChecked()
        ]

    def values(self):
        """คืนค่า kwargs สำหรับ model.train() - รายการที่ไม่ได้ติ๊ก (หรืออยู่ในกลุ่มที่ไม่ติ๊ก)
        จะถูกส่งเป็น 0.0 คือไม่ใช้ augment นั้น"""
        result = {}
        for name, keys in GROUP_KEYS.items():
            group_on = self.group_boxes[name].isChecked()
            for key in keys:
                on = group_on and self.checks[key].isChecked()
                result[key] = self.spins[key].value() if on else 0.0
        return result
