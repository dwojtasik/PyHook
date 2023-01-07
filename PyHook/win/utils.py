"""
win.utils for PyHook
~~~~~~~~~~~~~~~~~~~~~~~
Utils for Windows OS
:copyright: (c) 2023 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""

from platform import architecture

from psutil import pid_exists

from win.api import is_wow_process_64_bit

# String for 32-bit architecture.
_ARCH_32_BIT = "32bit"
# String for 64-bit architecture.
_ARCH_64_BIT = "64bit"


def to_arch_string(is_64_bit: bool) -> str:
    """Returns architecture string from 64 bit flag.

    Args:
        is_64_bit (bool): The flag if process is 64 bit.

    Returns:
        str: The architecture string.
    """
    return _ARCH_64_BIT if is_64_bit else _ARCH_32_BIT


def is_32_bit_os() -> bool:
    """Checks if OS is 32 bit.

    Returns:
        bool: True if OS is 32 bit
    """
    return architecture()[0] == _ARCH_32_BIT


def is_process_64_bit(pid: int) -> bool:
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
    return is_wow_process_64_bit(pid)
