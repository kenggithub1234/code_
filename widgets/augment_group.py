from PyQt5.QtWidgets import (
    QGroupBox, QVBoxLayout, QHBoxLayout, QPushButton, QDoubleSpinBox, QFormLayout
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


class AugmentGroupBox(QGroupBox):
    """Reusable 'Data Augmentation' settings panel shared by every ultralytics
    training tab (detect / segment / classify). Values map directly onto
    model.train(**kwargs) keyword arguments."""

    def __init__(self, parent=None):
        super().__init__("การเพิ่มข้อมูลภาพระหว่างเทรน (Data Augmentation)", parent)
        self.spins = {}
        self._build_ui()

    def _build_ui(self):
        outer = QVBoxLayout()
        row = QHBoxLayout()

        color_box = QGroupBox("1. ปรับสี (Color Space Augmentation)")
        color_form = QFormLayout()
        color_form.addRow("Hue (hsv_h):", self._make_spin("hsv_h", 0.0, 1.0, 0.005, 3))
        color_form.addRow("Saturation (hsv_s):", self._make_spin("hsv_s", 0.0, 1.0, 0.05, 2))
        color_form.addRow("Value/Brightness (hsv_v):", self._make_spin("hsv_v", 0.0, 1.0, 0.05, 2))
        color_box.setLayout(color_form)
        row.addWidget(color_box)

        geo_box = QGroupBox("2. ปรับรูปทรง/มุมมอง (Geometric Transformation)")
        geo_form = QFormLayout()
        geo_form.addRow("หมุน (degrees):", self._make_spin("degrees", 0.0, 180.0, 1.0, 1))
        geo_form.addRow("เลื่อนภาพ (translate):", self._make_spin("translate", 0.0, 1.0, 0.05, 2))
        geo_form.addRow("ย่อ/ขยาย (scale):", self._make_spin("scale", 0.0, 2.0, 0.05, 2))
        geo_form.addRow("บิดเฉือน (shear):", self._make_spin("shear", 0.0, 45.0, 1.0, 1))
        geo_form.addRow("มุมมอง (perspective):", self._make_spin("perspective", 0.0, 0.001, 0.0001, 5))
        geo_form.addRow("พลิกบน-ล่าง (flipud):", self._make_spin("flipud", 0.0, 1.0, 0.1, 2))
        geo_form.addRow("พลิกซ้าย-ขวา (fliplr):", self._make_spin("fliplr", 0.0, 1.0, 0.1, 2))
        geo_box.setLayout(geo_form)
        row.addWidget(geo_box)

        mix_box = QGroupBox("3. ผสมหลายภาพ (Mix-based Augmentation)")
        mix_form = QFormLayout()
        mix_form.addRow("Mosaic (ต่อ 4 ภาพ):", self._make_spin("mosaic", 0.0, 1.0, 0.1, 2))
        mix_form.addRow("MixUp (ผสม 2 ภาพ):", self._make_spin("mixup", 0.0, 1.0, 0.1, 2))
        mix_form.addRow("Copy-Paste:", self._make_spin("copy_paste", 0.0, 1.0, 0.1, 2))
        mix_box.setLayout(mix_form)
        row.addWidget(mix_box)

        outer.addLayout(row)

        reset_bar = QHBoxLayout()
        reset_bar.addStretch()
        reset_btn = QPushButton("รีเซ็ตค่า Augmentation เป็นค่าเริ่มต้น")
        reset_btn.clicked.connect(self.reset_defaults)
        reset_bar.addWidget(reset_btn)
        outer.addLayout(reset_bar)

        self.setLayout(outer)

    def _make_spin(self, key, minimum, maximum, step, decimals):
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setValue(AUGMENT_DEFAULTS[key])
        self.spins[key] = spin
        return spin

    def reset_defaults(self):
        for key, spin in self.spins.items():
            spin.setValue(AUGMENT_DEFAULTS[key])

    def values(self):
        return {key: spin.value() for key, spin in self.spins.items()}
