# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec สำหรับ YOLO11 Dataset Tool (โหมดโฟลเดอร์ / onedir)

ทำไมต้องมี spec แทนการสั่ง pyinstaller ตรงๆ:
  * ultralytics โหลดไฟล์ .yaml (cfg/ และ datasets/) ตอนรันไทม์ ต้อง collect
    ให้ครบ ไม่งั้นจะฟ้อง FileNotFoundError ตอนเริ่มเทรน
  * tab แต่ละตัวถูก import แบบตรงๆ ก็จริง แต่ torch/tensorflow มี submodule
    ที่โหลดแบบ dynamic ต้องประกาศเพิ่ม
"""
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules

datas = []
binaries = []
hiddenimports = []

# ultralytics: ต้องได้ไฟล์ config ทั้งหมดติดไปด้วย
for pkg in ("ultralytics",):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

# โมดูลของโปรเจกต์เอง (ถูกอ้างผ่าน import ปกติ แต่ประกาศไว้กันพลาด)
hiddenimports += collect_submodules("tabs")
hiddenimports += collect_submodules("utils")
hiddenimports += collect_submodules("widgets")

# ตัวที่ PyInstaller มักตรวจไม่เจอเอง
hiddenimports += [
    "PyQt5.QtCore", "PyQt5.QtGui", "PyQt5.QtWidgets",
    "cv2",
    "scipy.special._cdflib",
]

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    # ตัดของที่ไม่ได้ใช้ออก ลดขนาดและเวลา build
    excludes=[
        "tkinter",
        "PyQt6", "PySide2", "PySide6",
        "notebook", "jupyter", "IPython",
        "pytest", "sphinx",
    ],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="YOLO11DatasetTool",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,          # ไม่ต้องมีหน้าต่าง console ดำๆ โผล่มา
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="YOLO11DatasetTool",
)
