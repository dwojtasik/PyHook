"""
dll_utils for PyHook
~~~~~~~~~~~~~~~~~~~~~~~
Utils for DLL management
:copyright: (c) 2022 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""

from ctypes import *
from os.path import abspath, basename, exists
from typing import TypeVar

import psutil
from pyinjector import inject

from win_utils import is_process_64_bit, to_arch_string

T = TypeVar('T')

_ADDON_PATHS_32BIT = [
    './Addons/PyHook32.addon',
    './Addon/Release/PyHook.addon',
    './Addon/Debug/PyHook.addon',
    './PyHook32.addon',
    './PyHook.addon'
]
_ADDON_PATHS_64BIT = [
    './Addons/PyHook64.addon',
    './Addon/x64/Release/PyHook.addon',
    './Addon/x64/Debug/PyHook.addon',
    './PyHook64.addon',
    './PyHook.addon'
]

_RESHADE_VERSION_EXTERN = 'ReShadeVersion'
_RESHADE_MIN_VERSION = '5.0.0'
_RESHADE_VALID_DLL_NAMES = [
    'd3d9.dll', 'd3d10.dll', 'd3d11.dll', 'd3d12.dll', 'dxgi.dll', 'opengl32.dll'
]


class AddonNotFoundException(Exception):
    """Raised when addon DLL file is not found."""
    pass


class ReShadeNotFoundException(Exception):
    """Raised when required version of ReShade is not loaded in any active process."""
    pass


class NotAReShadeProcessException(Exception):
    """Raised when required version of ReShade is not loaded in given process."""
    pass


class AddonHandler:
    """Handler for PyHook addon management.

    Responsibilities:
    - reads information about given process,
    - checks if required ReShade version is loaded,
    - injects PyHook addon DLL.

    process (psutil.Process): The process to validate and addon injection.
    process_name (str): The process name.
    pid (int): The process ID.
    is_64_bit (bool): Flag if given process is 64 bit.
    addon_path (str): The absolute path to addon DLL file.

    Raises:
        NotAReShadeProcessException: If process does not have required ReShade version loaded.
        AddonNotFoundException: If process does not have required ReShade version loaded.
    """

    def __init__(self, process: psutil.Process):
        self._matching_dlls = list(filter(
            lambda path: basename(path) in _RESHADE_VALID_DLL_NAMES,
            [dll_info.path for dll_info in process.memory_maps()]
        ))
        if self._matching_dlls:
            self.process_name = process.name()
            self.pid = process.pid
            self.is_64_bit = is_process_64_bit(self.pid)
            if self._has_loaded_reshade():
                self.addon_path = self._find_addon_path()
                return
        raise NotAReShadeProcessException()

    def get_info(self) -> str:
        """Returns handler informations.

        Returns:
            str: The textual informations about handler.
        """
        return f'{self.process_name} [PID={self.pid}, {to_arch_string(self.is_64_bit)}] with ReShade v{self.reshade_version} @ {self.reshade_path}'

    def inject_addon(self) -> None:
        """Injects addon DLL into process."""
        inject(self.pid, self.addon_path)

    def _find_addon_path(self) -> str:
        """Returns addon DLL absolute path.

        Returns:
            str: The absolute path to addon DLL file.

        Raises:
            AddonNotFoundException: If addon DLL file cannot be found.
        """
        PATHS = _ADDON_PATHS_64BIT if self.is_64_bit else _ADDON_PATHS_32BIT
        for path in PATHS:
            if exists(path):
                return abspath(path)
        raise AddonNotFoundException()

    def _has_loaded_reshade(self) -> bool:
        """Validates if given process has loaded required ReShade version.

        ReShade version is verified via DLL extern variable.

        Returns:
            bool: True if process has valid ReShade version loaded.
        """
        for test_dll_path in self._matching_dlls:
            test_dll_handle = None
            try:
                test_dll_handle = cdll[test_dll_path]
                if hasattr(test_dll_handle, _RESHADE_VERSION_EXTERN):
                    self.reshade_version = _get_dll_extern_variable(
                        test_dll_handle,
                        _RESHADE_VERSION_EXTERN,
                        c_char_p
                    ).decode('utf-8')
                    if self.reshade_version >= _RESHADE_MIN_VERSION:
                        self.reshade_path = test_dll_path
                        return True
                    return False
            except:
                pass
            finally:
                if test_dll_handle is not None:
                    _unload_dll(test_dll_handle, self.is_64_bit)
                    del test_dll_handle
        return False


def _unload_dll(dll_handle: CDLL, is_64_bit: bool) -> None:
    """Unloads given dll from PyHook process.

    NOTE: After this 'del variable' should be called to remove it from memory.

    Args:
        dll_handle (ctypes.CDLL): Loaded dll handle.
        is_64_bit (bool): Flag if true owner process is 64 bit.
    """
    if is_64_bit:
        windll.kernel32.FreeLibrary(c_void_p(dll_handle._handle))
    else:
        windll.kernel32.FreeLibrary(dll_handle._handle)


def _get_dll_extern_variable(dll_handle: CDLL, variable_name: str, out_type: T) -> T:
    """Returns extern value of given output type from DLL.

    Args:
        dll_handle (ctypes.CDLL): Loaded dll handle.
        variable_name (str): The name of extern C variable to get.
        out_type (T): The type of C variable from ctypes.

    Returns:
        T: The DLL's extern C variable casted to valid Python type using ctypes.

    Raises:
        ValueError: If variable cannot be read.
    """
    try:
        return out_type.in_dll(dll_handle, variable_name).value
    except:
        raise ValueError(
            f'Cannot read variable "{variable_name}" of type "{out_type}" from DLL@{dll_handle}')


def get_reshade_addon_handler() -> AddonHandler:
    """Returns addon handler with required process information.

    Returns:
        AddonHandler: The handler for PyHook addon management.

    Raises:
        ReShadeNotFindException: If required ReShade version is not loaded in any active process
    """
    for process in psutil.process_iter():
        try:
            return AddonHandler(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, NotAReShadeProcessException):
            pass
    raise ReShadeNotFoundException()
