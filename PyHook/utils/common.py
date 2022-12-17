"""
utils.common for PyHook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Common utils
:copyright: (c) 2022 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""

import sys

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
