# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec file for Hand Gesture Control

block_cipher = None

a = Analysis(
    ['src/app_optimized.py'],
    pathex=['d:/hand-gesture-control'],
    binaries=[],
    datas=[
        ('gesture_config.json', '.'),
    ],
    hiddenimports=[
        'mediapipe',
        'cv2',
        'numpy',
        'PyQt6',
        'PyQt6.QtCore',
        'PyQt6.QtGui', 
        'PyQt6.QtWidgets',
        'pynput',
        'pynput.keyboard',
        'pynput.mouse',
        'psutil',
        'src.hand_detector',
        'src.command_mapper',
        'src.optimized_recognizer',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'tkinter',
        'PIL',
        'scipy',
        'pandas',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='HandGestureControl',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # No console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here if you have one
)
