# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all
from stdlib_list import stdlib_list
import os
import sys

with open('PyHook\\_version.py') as ver_file:
    exec(ver_file.read())

is_64_bit = os.getenv('CONDA_FORCE_32BIT', '0') == '0'
name = f'PyHook-{__version__}-{"win_amd64" if is_64_bit else "win32"}'

datas = []
binaries = []
hiddenimports = stdlib_list("3.9") # Update to 3.10 when possible
tmp_ret = collect_all('pyinjector')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# Pack DLLs from conda environment
conda_env_path = os.path.dirname(sys.executable)
binaries += [(f"{conda_env_path}\\python3.dll", ".")]
if is_64_bit:
    binaries += [(f"{conda_env_path}\\vcruntime140_1.dll", ".")]

block_cipher = None

a = Analysis(
    ['PyHook\\gui.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Avoid duplicates
for b in a.binaries.copy():
    for d in a.datas:
        if b[1].endswith(d[0]):
            a.binaries.remove(b)
            break

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name=name,
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
    icon='pyhook_icon.ico',
    version='VERSION.txt',
)
