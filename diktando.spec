# -*- mode: python ; coding: utf-8 -*-
import os
from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs

block_cipher = None

# Get site-packages directory
site_packages = os.path.join(os.__file__, '..', '..', 'site-packages')

# Collect PyQt5 data files and binaries
qt_data = collect_data_files('PyQt5', include_py_files=True)
qt_binaries = collect_dynamic_libs('PyQt5')

# Add icon to data files
icon_data = [('icon.ico', '.')]

a = Analysis(
    ['main.py', 'clipboard_manager.py', 'updater.py'],
    pathex=[],
    binaries=qt_binaries,
    datas=qt_data + icon_data,
    hiddenimports=[
        'numpy', 
        'sounddevice', 
        'soundfile', 
        'requests', 
        'tqdm', 
        'keyboard', 
        'pynput', 
        'pyperclip', 
        'PyQt5.QtCore',
        'PyQt5.QtGui',
        'PyQt5.QtWidgets',
        'clipboard_manager',
        'packaging.version',
        'packaging.specifiers',
        'packaging.requirements',
        'packaging.markers'
    ],
    hookspath=[],
    hooksconfig={
        'PyQt5': {
            'gui': True
        }
    },
    runtime_hooks=[],
    excludes=[],
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
    name='DiktandoApp_v1.1',
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
    icon='icon.ico'
)
