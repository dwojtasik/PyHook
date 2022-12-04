"""
utils.common for PyHook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Common utils
:copyright: (c) 2022 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""

import sys

# Name of temporary directory that Pyinstaller will create with bundled Python.
_MEIPASS = "_MEIPASS"


def is_frozen_bundle() -> bool:
    """Checks if app is running in PyInstaller frozen bundle.

    Returns:
        bool: True if app is running in PyInstaller frozen bundle.
    """
    return getattr(sys, "frozen", False) and hasattr(sys, _MEIPASS)
