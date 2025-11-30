# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['src\\app_optimized.py'],
    pathex=[],
    binaries=[],
    datas=[('gesture_config.json', '.'), ('src', 'src'), ('C:\\Users\\Hi\\AppData\\Local\\Packages\\PythonSoftwareFoundation.Python.3.9_qbz5n2kfra8p0\\LocalCache\\local-packages\\Python39\\site-packages\\mediapipe\\modules', 'mediapipe/modules')],
    hiddenimports=['mediapipe', 'cv2', 'numpy', 'PyQt6', 'pynput', 'psutil', 'matplotlib', 'PIL'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=['PyQt5', 'tkinter'],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='HandGestureControl',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
