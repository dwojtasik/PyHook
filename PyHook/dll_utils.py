"""
dll_utils for PyHook
~~~~~~~~~~~~~~~~~~~~~~~
Utils for DLL management
:copyright: (c) 2022 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""

import glob
import os
from ctypes import CDLL, c_char_p, c_void_p, cdll, windll
from os.path import abspath, basename, dirname, exists
from typing import TypeVar

import psutil
from pyinjector import inject

from win_utils import is_process_64_bit, to_arch_string

T = TypeVar("T")

# Search paths (in priority order) for 32-bit addon file.
_ADDON_PATHS_32BIT = [
    "./Addons/PyHook32.addon",
    "./Addon/Release/PyHook.addon",
    "./Addon/Debug/PyHook.addon",
    "./PyHook32.addon",
    "./PyHook.addon",
]

# Search paths (in priority order) for 64-bit addon file.
_ADDON_PATHS_64BIT = [
    "./Addons/PyHook64.addon",
    "./Addon/x64/Release/PyHook.addon",
    "./Addon/x64/Debug/PyHook.addon",
    "./PyHook64.addon",
    "./PyHook.addon",
]

# ReShade extern variable to read version.
_RESHADE_VERSION_EXTERN = "ReShadeVersion"
# ReShade minimum version supported.
_RESHADE_MIN_VERSION = "5.0.0"
# ReShade valid DLL names to load and check.
_RESHADE_VALID_DLL_NAMES = ["d3d9.dll", "d3d10.dll", "d3d11.dll", "d3d12.dll", "dxgi.dll", "opengl32.dll"]


class AddonNotFoundException(Exception):
    """Raised when addon DLL file cannot be found."""


class ProcessNotFoundException(Exception):
    """Raised when process with given PID does not exists."""


class ReShadeNotFoundException(Exception):
    """Raised when required version of ReShade is not loaded in any active process."""


class NotAReShadeProcessException(Exception):
    """Raised when required version of ReShade is not loaded in given process."""


class AddonHandler:
    """Handler for PyHook addon management.

    Responsibilities:
    - reads information about given process,
    - checks if required ReShade version is loaded,
    - injects PyHook addon DLL.

    process (psutil.Process): The process to validate and addon injection.
    verify (bool, optional): Flag if PyHook should verify if required version of ReShade is loaded.
        Defaults to True.
    process_name (str): The process name.
    pid (int): The process ID.
    is_64_bit (bool): Flag if given process is 64 bit.
    exe (str): The process executable as an absolute path.
    dir_path (str): The process directory as an absolute path.
    addon_path (str): The absolute path to addon DLL file.

    Raises:
        NotAReShadeProcessException: When process does not have required ReShade version loaded.
        AddonNotFoundException: When addon DLL file cannot be found.
    """

    def __init__(self, process: psutil.Process, verify: bool = True):
        self.verify = verify
        self._matching_dlls = (
            []
            if not verify
            else list(
                filter(
                    lambda path: basename(path) in _RESHADE_VALID_DLL_NAMES,
                    [dll_info.path for dll_info in process.memory_maps()],
                )
            )
        )
        if not verify or self._matching_dlls:
            self.process_name = process.name()
            self.pid = process.pid
            self.is_64_bit = is_process_64_bit(self.pid)
            self.exe = process.exe()
            self.dir_path = dirname(self.exe)
            if not verify or self._has_loaded_reshade():
                self.addon_path = self._find_addon_path()
                return
        raise NotAReShadeProcessException()

    def get_info(self) -> str:
        """Returns handler informations.

        Returns:
            str: The textual informations about handler.
        """
        reshade_version_string = ""
        if self.verify:
            reshade_version_string = f" with ReShade v{self.reshade_version} @ {self.reshade_path}"
        return f"{self.process_name} [PID={self.pid}, {to_arch_string(self.is_64_bit)}]{reshade_version_string}"

    def inject_addon(self) -> None:
        """Injects addon DLL into process."""
        inject(self.pid, self.addon_path)

    def _find_addon_path(self) -> str:
        """Returns addon DLL absolute path.

        Returns:
            str: The absolute path to addon DLL file.

        Raises:
            AddonNotFoundException: When addon DLL file cannot be found.
        """
        paths = _ADDON_PATHS_64BIT if self.is_64_bit else _ADDON_PATHS_32BIT
        for path in paths:
            if exists(path):
                return abspath(path)
        raise AddonNotFoundException()

    def _has_loaded_reshade(self) -> bool:
        """Validates if given process has loaded required ReShade version.

        ReShade version is verified via DLL extern variable.
        All logs created during DLL loading will be removed.

        Returns:
            bool: True if process has valid ReShade version loaded.
        """
        logs_before = glob.glob(f"{self.dir_path}/*.log")
        try:
            for test_dll_path in self._matching_dlls:
                test_dll_handle = None
                try:
                    test_dll_handle = cdll[test_dll_path]
                    if hasattr(test_dll_handle, _RESHADE_VERSION_EXTERN):
                        self.reshade_version = _get_dll_extern_variable(
                            test_dll_handle, _RESHADE_VERSION_EXTERN, c_char_p
                        ).decode("utf-8")
                        if self.reshade_version >= _RESHADE_MIN_VERSION:
                            self.reshade_path = test_dll_path
                            return True
                        return False
                except Exception:
                    pass
                finally:
                    if test_dll_handle is not None:
                        _unload_dll(test_dll_handle, self.is_64_bit)
                        del test_dll_handle
            return False
        finally:
            logs_after = glob.glob(f"{self.dir_path}/*.log")
            new_logs = [log_f for log_f in logs_after if log_f not in logs_before]
            for new_log in new_logs:
                os.remove(new_log)


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
        ValueError: When variable cannot be read.
    """
    try:
        return out_type.in_dll(dll_handle, variable_name).value
    except Exception as ex:
        raise ValueError(f'Cannot read variable "{variable_name}" of type "{out_type}" from DLL@{dll_handle}') from ex


def get_reshade_addon_handler(pid: int = None) -> AddonHandler:
    """Returns addon handler with required process information.

    Args:
        pid (int, optional): PID of ReShade owner process.
            If supplied PyHook will skip tests for ReShade!

    Returns:
        AddonHandler: The handler for PyHook addon management.

    Raises:
        ProcessNotFoundException: When process with PID does not exists.
        ReShadeNotFindException: When required ReShade version is not loaded in any active process.
    """
    if pid is not None:
        if not psutil.pid_exists(pid):
            raise ProcessNotFoundException()
        return AddonHandler(psutil.Process(pid), False)
    for process in psutil.process_iter():
        try:
            return AddonHandler(process)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, NotAReShadeProcessException):
            pass
    raise ReShadeNotFoundException()
