"""
dll_utils for PyHook
~~~~~~~~~~~~~~~~~~~~~~~
Utils for DLL management
:copyright: (c) 2023 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""

import glob
import os
import sys
from ctypes import c_char_p
from os.path import abspath, basename, dirname, exists
from typing import List

import psutil
from pyinjector import inject

from keys import SettingsKeys
from utils.common import get_frozen_path, is_frozen_bundle
from utils.external import unpack_32bit_injector, inject_external
from win.api import DONT_RESOLVE_DLL_REFERENCES, FreeLibrary, GetProcAddress, LoadLibrary
from win.utils import is_process_64_bit, to_arch_string

# Search paths (in priority order) for 32-bit addon file.
_ADDON_PATHS_32BIT = [
    "./Addon/Release/PyHook.addon",
    "./PyHook32.addon",
    "./PyHook.addon",
    "./Addon/Debug/PyHook.addon",
]

# Search paths (in priority order) for 64-bit addon file.
_ADDON_PATHS_64BIT = [
    "./Addon/x64/Release/PyHook.addon",
    "./PyHook64.addon",
    "./PyHook.addon",
    "./Addon/x64/Debug/PyHook.addon",
]

# ReShade extern variable to read version.
_RESHADE_VERSION_EXTERN = "ReShadeVersion"
# ReShade minimum version supported.
_RESHADE_MIN_VERSION = "5.0.0"
# ReShade valid DLL names to load and check.
_RESHADE_VALID_DLL_NAMES = [
    "d3d9.dll",
    "d3d10.dll",
    "d3d11.dll",
    "d3d12.dll",
    "dxgi.dll",
    "opengl32.dll",
    "reshade64.dll",
    "reshade32.dll",
]


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
    dlls_to_skip (List[str], optional): Optional list of DLLs to skip during DLL verification process.
        Defaults to None.
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

    def __init__(self, process: psutil.Process, verify: bool = True, dlls_to_skip: List[str] = None):
        self.verify = verify
        self._matching_dlls = (
            []
            if not verify
            else list(
                filter(
                    lambda path: path not in dlls_to_skip and basename(path).lower() in _RESHADE_VALID_DLL_NAMES,
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
            if not verify or self._has_loaded_reshade(dlls_to_skip):
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
        """Injects addon DLL into process.

        Raises:
            ValueError: When trying to inject into 32-bit process on 64-bit OS
                without setting path to 32-bit Python executable.
        """
        if not self.is_64_bit and sys.maxsize > 2**32:
            if not is_frozen_bundle():
                unpack_32bit_injector()
            local_path = os.getenv(SettingsKeys.KEY_LOCAL_PYTHON_32.upper(), "")
            if len(local_path) == 0:
                raise ValueError("Path to 32-bit Python executable is not set.")
            inject_external(local_path, self.pid, self.addon_path)
        else:
            inject(self.pid, self.addon_path)

    def has_addon_loaded(self) -> bool:
        """Checks if addon is still loaded into process.

        Returns:
            bool: True if addon is still loaded into process with given pid.
        """
        process = psutil.Process(self.pid)
        return self.addon_path in [dll_info.path for dll_info in process.memory_maps()]

    def _find_addon_path(self) -> str:
        """Returns addon DLL absolute path.

        Returns:
            str: The absolute path to addon DLL file.

        Raises:
            AddonNotFoundException: When addon DLL file cannot be found.
        """
        if is_frozen_bundle():
            frozen_path = get_frozen_path(f"{'.' if self.is_64_bit else 'lib32'}\\PyHook.addon")
            if exists(frozen_path):
                return frozen_path
        paths = _ADDON_PATHS_64BIT if self.is_64_bit else _ADDON_PATHS_32BIT
        for path in paths:
            if exists(path):
                return abspath(path)
        raise AddonNotFoundException()

    def _has_loaded_reshade(self, dlls_to_skip: List[str] = None) -> bool:
        """Validates if given process has loaded required ReShade version.

        ReShade version is verified via DLL extern variable.
        All logs created during DLL loading will be removed.

        Args:
            dlls_to_skip (List[str], optional): Optional list of DLLs to extend if new DLLs dont pass verification.
                Defaults to None.

        Returns:
            bool: True if process has valid ReShade version loaded.
        """
        logs_before = glob.glob(f"{self.dir_path}/*.log")
        try:
            reshade_version_p = _RESHADE_VERSION_EXTERN.encode("utf-8")
            for test_dll_path in self._matching_dlls:
                test_dll_hmod = None
                try:
                    test_dll_hmod = LoadLibrary(test_dll_path, 0, DONT_RESOLVE_DLL_REFERENCES)
                    if test_dll_hmod is not None:
                        version_str_p = GetProcAddress(test_dll_hmod, reshade_version_p)
                        if version_str_p is not None:
                            self.reshade_version = c_char_p.from_address(version_str_p).value.decode("utf-8")
                            if self.reshade_version >= _RESHADE_MIN_VERSION:
                                self.reshade_path = test_dll_path
                                return True
                            dlls_to_skip.append(test_dll_path)
                            return False
                except Exception:
                    pass
                finally:
                    if test_dll_hmod is not None:
                        FreeLibrary(test_dll_hmod)
                        del test_dll_hmod
                dlls_to_skip.append(test_dll_path)
            return False
        finally:
            logs_after = glob.glob(f"{self.dir_path}/*.log")
            new_logs = [log_f for log_f in logs_after if log_f not in logs_before]
            for new_log in new_logs:
                os.remove(new_log)


def get_reshade_addon_handler(pid: int = None, pids_to_skip: List[int] = None) -> AddonHandler:
    """Returns addon handler with required process information.

    Args:
        pid (int, optional): PID of ReShade owner process.
            If supplied PyHook will skip tests for ReShade!
        pids_to_skip (List[int], optional): List of PIDs to skip in process iteration.

    Returns:
        AddonHandler: The handler for PyHook addon management.

    Raises:
        ProcessNotFoundException: When process with PID does not exists.
        ReShadeNotFindException: When required ReShade version is not loaded in any active process.
    """
    if pid is not None:
        if not psutil.pid_exists(pid):
            raise ProcessNotFoundException()
        return AddonHandler(psutil.Process(pid), verify=False)
    checked_dlls = []
    for process in psutil.process_iter():
        try:
            if pids_to_skip is not None and process.pid in pids_to_skip:
                continue
            return AddonHandler(process, dlls_to_skip=checked_dlls)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, NotAReShadeProcessException):
            pass
    raise ReShadeNotFoundException()
