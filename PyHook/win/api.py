"""
win.api for PyHook
~~~~~~~~~~~~~~~~~~~~~~~
Windows API in ctypes
:copyright: (c) 2022 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""

# pylint: disable=invalid-name

import ctypes
from ctypes import c_int32, c_void_p
from ctypes.wintypes import BOOL, DWORD, HANDLE, LPCWSTR, LPVOID, PBOOL
from typing import TypeVar

T = TypeVar("T")

# Predefined process-specific access rights.
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
# The state of the specified object is signaled.
WAIT_OBJECT_0 = 0x00000000

# Handle for kernel32 DLL.
KERNEL32 = ctypes.windll.kernel32

# Tests whether the current user is a member of the Administrator's group.
# BOOL IsUserAnAdmin();
IsUserAnAdmin = ctypes.windll.shell32.IsUserAnAdmin
IsUserAnAdmin.restype = BOOL

# Opens an existing local process object.
# HANDLE OpenProcess(
#  [in] DWORD dwDesiredAccess,
#  [in] BOOL  bInheritHandle,
#  [in] DWORD dwProcessId
# );
OpenProcess = KERNEL32.OpenProcess
OpenProcess.argtypes = [DWORD, BOOL, DWORD]
OpenProcess.restype = HANDLE

# Closes an open object handle.
# BOOL CloseHandle(
#   [in] HANDLE hObject
# );
CloseHandle = KERNEL32.CloseHandle
CloseHandle.argtypes = [HANDLE]
CloseHandle.restype = BOOL

# Determines whether the specified process is running under WOW64 or an Intel64 of x64 processor.
# BOOL IsWow64Process(
#  [in]  HANDLE hProcess,
#  [out] PBOOL  Wow64Process
# );
IsWow64Process = None
if hasattr(KERNEL32, "IsWow64Process"):
    IsWow64Process = KERNEL32.IsWow64Process
    IsWow64Process.argtypes = [HANDLE, PBOOL]
    IsWow64Process.restype = BOOL

# Creates or opens a named or unnamed event object.
# HANDLE CreateEventW(
#   [in, optional] LPSECURITY_ATTRIBUTES lpEventAttributes,
#   [in]           BOOL                  bManualReset,
#   [in]           BOOL                  bInitialState,
#   [in, optional] LPCWSTR               lpName
# );
CreateEvent = KERNEL32.CreateEventW
CreateEvent.argtypes = [LPVOID, BOOL, BOOL, LPCWSTR]
CreateEvent.restype = HANDLE

# Waits until the specified object is in the signaled state or the time-out interval elapses.
# DWORD WaitForSingleObject(
#   [in] HANDLE hHandle,
#   [in] DWORD  dwMilliseconds
# );
WaitForSingleObject = KERNEL32.WaitForSingleObject
WaitForSingleObject.argtypes = [HANDLE, DWORD]
WaitForSingleObject.restype = DWORD

# Sets the specified event object to the nonsignaled state.
# BOOL SetEvent(
#   [in] HANDLE hEvent
# );
SetEvent = KERNEL32.SetEvent
SetEvent.argtypes = [HANDLE]
SetEvent.restype = BOOL


def is_started_as_admin() -> bool:
    """Cheks if program was started with administrator rights.

    Returns:
        bool: True if program was stared as admin.
    """
    return IsUserAnAdmin() != 0


def is_wow_process_64_bit(pid: int) -> bool:
    """Checks if process with given PID is 64-bit.

    IsWow64Process exists only in 64-bit Windows.

    Args:
        pid (int): Process ID.

    Returns:
        bool: Flag if process with given PID is 64-bit.

    Raises:
        ValueError: When cannot determine process architecture.
    """
    if IsWow64Process is None:
        return False
    handle = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, 0, pid)
    if handle:
        try:
            is_32_bit = c_int32()
            if IsWow64Process(handle, ctypes.byref(is_32_bit)):
                return not is_32_bit.value
        finally:
            CloseHandle(handle)
    raise ValueError(f"Cannot determine architecture for given PID={pid}")


def unload_dll(dll_handle: ctypes.CDLL, is_64_bit: bool) -> None:
    """Unloads given dll from PyHook process.

    NOTE: After this 'del variable' should be called to remove it from memory.

    Args:
        dll_handle (ctypes.CDLL): Loaded dll handle.
        is_64_bit (bool): Flag if true owner process is 64 bit.
    """
    if is_64_bit:
        KERNEL32.FreeLibrary(c_void_p(dll_handle._handle))
    else:
        KERNEL32.FreeLibrary(dll_handle._handle)


def get_dll_extern_variable(dll_handle: ctypes.CDLL, variable_name: str, out_type: T) -> T:
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
