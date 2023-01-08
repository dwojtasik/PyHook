"""
utils.external for PyHook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
External utils for PyHook
:copyright: (c) 2023 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""

import os
import sys
from os.path import abspath, exists
from subprocess import check_call
from zipfile import ZipFile

from utils.common import get_frozen_path, is_frozen_bundle
from win.api import CREATE_NO_WINDOW

# Directory for 32-bit data.
_LIB32_DIRECTORY = "lib32"
# Version of pyinjector module.
_PYINJECTOR_VERSION = "1.1.0"


def unpack_32bit_injector() -> None:
    """Prepares files for 32-bit injector if needed.

    Downloads Python 3.10 wheel for 32-bit pyinjector and
    unpacks files required for injector into ./lib32/pyinjector.
    """
    whl_files = [
        "pyinjector/__init__.py",
        "pyinjector/libinjector.cp310-win32.pyd",
        "pyinjector/pyinjector.py",
    ]
    if any(not exists(f"{_LIB32_DIRECTORY}\\{whl_file}") for whl_file in whl_files):
        try:
            check_call(
                (
                    f"{sys.executable} -m pip download "
                    "--only-binary :all: "
                    "--no-cache "
                    "--exists-action i "
                    "--platform win32 "
                    "--python-version 3.10 "
                    f"pyinjector=={_PYINJECTOR_VERSION}"
                ),
                shell=False,
                creationflags=CREATE_NO_WINDOW,
            )
            whl_path = f"pyinjector-{_PYINJECTOR_VERSION}-cp310-cp310-win32.whl"
            with ZipFile(whl_path, "r") as whl_archive:
                for whl_file in whl_files:
                    whl_archive.extract(whl_file, _LIB32_DIRECTORY)
        finally:
            if exists(whl_path):
                os.remove(whl_path)


def inject_external(exe_path: str, pid: int, dll_path: str) -> None:
    """Injects DLL from given path into process with given PID.

    Used only to inject 32-bit DLL in 64-bit PyHook.

    Args:
        exe_path (str): Python executable path.
        pid (int): Process ID.
        dll_path (str): DLL path.
    """
    lib_init = f"{_LIB32_DIRECTORY}\\pyinjector\\__init__.py"
    if is_frozen_bundle():
        lib_init = get_frozen_path(lib_init)
    else:
        lib_init = abspath(lib_init)
    check_call(
        [
            exe_path,
            "-c",
            (
                """import importlib.util, sys;"""
                f"""spec = importlib.util.spec_from_file_location("pyinjector_ext", "{lib_init}");"""
                """module = importlib.util.module_from_spec(spec);"""
                """sys.modules["pyinjector_ext"] = module;"""
                """spec.loader.exec_module(module);"""
                f"""module.inject({pid}, "{dll_path}")"""
            ).replace("\\", "\\\\"),
        ],
        shell=False,
        creationflags=CREATE_NO_WINDOW,
    )
