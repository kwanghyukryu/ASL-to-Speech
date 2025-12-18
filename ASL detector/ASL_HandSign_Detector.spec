# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['f3.py'],
    pathex=[],
    binaries=[],
    datas=[('trained_models', 'trained_models')],
    hiddenimports=['cv2', 'mediapipe', 'numpy', 'pandas', 'sklearn', 'joblib', 'matplotlib', 'seaborn', 'collections'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
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
    name='ASL_HandSign_Detector',
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
