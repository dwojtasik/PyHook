"""
win_utils for PyHook
~~~~~~~~~~~~~~~~~~~~~~~
Utils for Windows OS
:copyright: (c) 2022 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""

import ctypes
from platform import architecture

from psutil import pid_exists

# String for 32-bit architecture.
_ARCH_32_BIT = "32bit"
# String for 64-bit architecture.
_ARCH_64_BIT = "64bit"
# Predefined process-specific access rights.
_PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

# Handle for kernel32 DLL.
_KERNEL32 = ctypes.windll.kernel32


def to_arch_string(is_64_bit: bool) -> str:
    """Returns architecture string from 64 bit flag.

    Args:
        is_64_bit (bool): The flag if process is 64 bit.

    Returns:
        str: The architecture string.
    """
    return _ARCH_64_BIT if is_64_bit else _ARCH_32_BIT


def is_32_bit_os() -> bool:
    """Cheks if OS is 32 bit.

    Returns:
        bool: True if OS is 32 bit
    """
    return architecture()[0] == _ARCH_32_BIT


def is_process_64_bit(pid) -> bool:
    """Checks if process of given PID is 64 bit.
    For 32 bit system it will simply return False.
    For 64 bit it will check IsWow64Process output from kernel32.

    Args:
        pid (int): The process ID.

    Returns:
        bool: True if process is 64 bit.

    Raises:
        ValueError: When given PID does not exists.
    """
    if not pid_exists(pid):
        raise ValueError(f"Process with PID={pid} does not exists")
    if is_32_bit_os():
        return False
    try:
        is_wow64_process = _KERNEL32.IsWow64Process
    except Exception:
        return False
    handle = _KERNEL32.OpenProcess(_PROCESS_QUERY_LIMITED_INFORMATION, 0, pid)
    if handle:
        try:
            is_32_bit = ctypes.c_int32()
            if is_wow64_process(handle, ctypes.byref(is_32_bit)):
                return not is_32_bit.value
        finally:
            _KERNEL32.CloseHandle(handle)


def is_started_as_admin() -> bool:
    """Cheks if program was started with administrator rights.

    Returns:
        bool: True if program was stared as admin.
    """
    return ctypes.windll.shell32.IsUserAnAdmin() != 0
