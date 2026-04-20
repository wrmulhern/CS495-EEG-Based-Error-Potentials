# -*- mode: python ; coding: utf-8 -*-
#
# ErrP Visualizer — PyInstaller spec file
#
# Usage:
#   uv run pyinstaller ErrPVisualizer.spec
#

import os
import sys
import glob

# ── Locate Brainflow native libraries ────────────────────────────────────────

import brainflow
brainflow_dir = os.path.dirname(brainflow.__file__)
brainflow_lib_dir = os.path.join(brainflow_dir, 'lib')

# Collect all native libs for the current platform
if sys.platform == 'win32':
    lib_pattern = os.path.join(brainflow_lib_dir, '*.dll')
    lib_dest    = 'brainflow\\lib'
else:
    # macOS (.dylib) and Linux (.so)
    lib_pattern = os.path.join(brainflow_lib_dir, '*')
    lib_dest    = 'brainflow/lib'

brainflow_binaries = [
    (lib, lib_dest)
    for lib in glob.glob(lib_pattern)
    if os.path.isfile(lib)
]

print(f"[spec] Bundling {len(brainflow_binaries)} Brainflow library files from {brainflow_lib_dir}")

# ── Analysis ──────────────────────────────────────────────────────────────────
a = Analysis(
    ['src/main.py'],
    pathex=['.'],
    binaries=brainflow_binaries,
    datas=[],
    hiddenimports=[
        # Brainflow internals PyInstaller misses
        'brainflow',
        'brainflow.board_shim',
        'brainflow.data_filter',
        'brainflow.exit_codes',
        'brainflow.ml_model',
        # scipy signal processing used in converter
        'scipy.signal',
        'scipy.io',
        'scipy.io.matlab',
        # other runtime imports
        'pandas',
        'numpy',
        'matplotlib',
        'matplotlib.backends.backend_qt5agg',
        'PyQt5',
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

# ── Bundle ────────────────────────────────────────────────────────────────────
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='ErrPVisualizer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,       # no terminal window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='assets/app_icon.ico',
)

# macOS: wrap the executable in a .app bundle
if sys.platform == 'darwin':
    app = BUNDLE(
        exe,
        name='ErrPVisualizer.app',
        icon=None,   # replace with 'src/assets/icon.icns' when ready
        bundle_identifier='edu.ua.htil.errpvisualizer',
        info_plist={
            'NSBluetoothAlwaysUsageDescription':
                'ErrP Visualizer uses Bluetooth to connect to the OpenBCI Ganglion EEG headset.',
            'NSBluetoothPeripheralUsageDescription':
                'ErrP Visualizer uses Bluetooth to connect to the OpenBCI Ganglion EEG headset.',
        },
        icon='assets/app_icon.icns',
    )
