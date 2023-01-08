# -*- mode: python ; coding: utf-8 -*-
import os
import re
import sys
from subprocess import check_call
from zipfile import ZipFile

from PyInstaller.utils.hooks import collect_all
from stdlib_list import stdlib_list

with open('PyHook\\_version.py') as ver_file:
    exec(ver_file.read())

is_64_bit = os.getenv('CONDA_FORCE_32BIT', '0') == '0'
name = f'PyHook-{__version__}-{"win_amd64" if is_64_bit else "win32"}'

datas = []
binaries = []
hiddenimports = stdlib_list("3.9") # Update to 3.10 when possible
tmp_ret = collect_all('pyinjector')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# Add pipeline utils and PIL.ImageEnhance to frozen bundle
hiddenimports += ['pipeline_utils', 'PIL.ImageEnhance']

# Pack DLLs from conda environment
conda_env_path = os.path.dirname(sys.executable)
binaries += [(f"{conda_env_path}\\python3.dll", ".")]
if is_64_bit:
    binaries += [(f"{conda_env_path}\\vcruntime140_1.dll", ".")]

# Pack PyHook addon DLLs
datas += [("Addon\\Release\\PyHook.addon", "lib32")]
if is_64_bit:
    datas += [("Addon\\x64\\Release\\PyHook.addon", ".")]

# Pack 32-bit pyinjector code into 64-bit version
if is_64_bit:
    pyinjector_version = "1.1.0"
    with open("requirements.txt") as requirements:
        pyinjector_version = re.findall("^pyinjector==(.*?)$", requirements.read(), re.MULTILINE)[0]
    try:
        check_call(
            (
                f"{sys.executable} -m pip download "
                "--only-binary :all: "
                "--no-cache "
                "--exists-action i "
                "--platform win32 "
                "--python-version 3.10 "
                f"pyinjector=={pyinjector_version}"
            )
        )
        whl_path = f"pyinjector-{pyinjector_version}-cp310-cp310-win32.whl"
        whl_files = [
            "pyinjector/__init__.py",
            "pyinjector/libinjector.cp310-win32.pyd",
            "pyinjector/pyinjector.py",
        ]
        with ZipFile(whl_path, "r") as whl_archive:
            for whl_file in whl_files:
                whl_archive.extract(whl_file, "lib32")
                datas += [(f"lib32\\{whl_file}", "lib32\\pyinjector")]
    finally:
        if os.path.exists(whl_path):
            os.remove(whl_path)

block_cipher = None

a = Analysis(
    ['PyHook\\main.py'],
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
