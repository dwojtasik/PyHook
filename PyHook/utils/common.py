"""
utils.common for PyHook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Common utils
:copyright: (c) 2023 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""

import os
import sys
import uuid
from os.path import basename

# Name of temporary directory that PyInstaller will create with bundled Python.
_MEIPASS = "_MEIPASS"


def is_frozen_bundle() -> bool:
    """Checks if app is running in PyInstaller frozen bundle.

    Returns:
        bool: True if app is running in PyInstaller frozen bundle.
    """
    return getattr(sys, "frozen", False) and hasattr(sys, _MEIPASS)


def get_frozen_path(path: str) -> str:
    """Return frozen path to given file.

    Args:
        path (str): Path to file.

    Returns:
        str: Path inside frozen bundle.
    """
    if path.startswith(".\\") or path.startswith("./"):
        path = path[2:]
    return f"{getattr(sys, _MEIPASS)}\\{path}"


def delete_self_exe(updated_exe: str = None) -> None:
    """Deletes own executable file and starts new updated one if provided.

    Args:
        updated_exe (str, optional): Updated executable path. Defaults to None.
    """
    if is_frozen_bundle():
        script_name = f"{uuid.uuid4()}.bat"
        script_content = [
            "@echo off\n",
            "timeout 1 >nul\n",
            f'DEL /F "{basename(sys.executable)}"\n',
            f'DEL /F "{script_name}"\n',
        ]
        if updated_exe is not None:
            script_content.insert(3, f'start /d "{os.getcwd()}" {updated_exe}\n')
        with open(script_name, "w+", encoding="utf-8") as bat_file:
            bat_file.writelines(script_content)
        os.startfile(f"{os.getcwd()}\\{script_name}")
