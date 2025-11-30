# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['src\\app_optimized.py'],
    pathex=[],
    binaries=[],
    datas=[('gesture_config.json', '.'), ('src', 'src')],
    hiddenimports=['mediapipe', 'mediapipe.python.solutions', 'cv2', 'numpy', 'PyQt6', 'pynput', 'psutil', 'matplotlib', 'PIL'],
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
    name='HandGestureControl_Debug',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
