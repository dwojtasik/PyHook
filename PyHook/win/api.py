"""
win.api for PyHook
~~~~~~~~~~~~~~~~~~~~~~~
Windows API in ctypes
:copyright: (c) 2022 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""

# pylint: disable=invalid-name

import ctypes
from ctypes import c_int, c_int32, c_ubyte
from ctypes.wintypes import BOOL, DWORD, HANDLE, HGLOBAL, HMODULE, HRSRC, LPCSTR, LPCWSTR, LPVOID, PBOOL
from typing import List, TypeVar

T = TypeVar("T")

# Predefined process-specific access rights.
PROCESS_QUERY_LIMITED_INFORMATION = 0x1000

# The state of the specified object is signaled.
WAIT_OBJECT_0 = 0x00000000

# The action to be taken when loading the module.
# If this value is used, and the executable module is a DLL, the system does not call DllMain
# for process and thread initialization and termination. Also, the system does not load
# additional executable modules that are referenced by the specified module.
DONT_RESOLVE_DLL_REFERENCES = 1

# The action to be taken when loading the module.
# If this value is used, the system maps the file into the calling process's virtual address
# space as if it were a data file.
LOAD_LIBRARY_AS_DATAFILE = 2

# The type of the resource for which the name is being enumerated.
# Hardware-dependent icon resource.
RT_ICON = 3

# Handle for kernel32 DLL.
KERNEL32 = ctypes.WinDLL("kernel32", use_last_error=True)

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

# Loads the specified module into the address space of the calling process.
# HMODULE LoadLibraryExW(
#   [in] LPCWSTR lpLibFileName,
#        HANDLE  hFile,
#   [in] DWORD   dwFlags
# );
LoadLibrary = KERNEL32.LoadLibraryExW
LoadLibrary.argtypes = [LPCWSTR, HANDLE, DWORD]
LoadLibrary.restype = HMODULE

# Frees the loaded dynamic-link library (DLL) module and, if necessary, decrements its reference count.
# BOOL FreeLibrary(
#   [in] HMODULE hLibModule
# );
FreeLibrary = KERNEL32.FreeLibrary
FreeLibrary.argtypes = [HMODULE]
FreeLibrary.restype = BOOL

# An application-defined callback function used with the EnumResourceNames and EnumResourceNamesEx functions.
# It receives the type and name of a resource.
# BOOL Enumresnameprocw(
#   [in, optional] HMODULE hModule,
#                  LPCWSTR lpType,
#                  LPWSTR lpName,
#   [in]           LONG_PTR lParam
# )
EnumResNameProc = ctypes.CFUNCTYPE(BOOL, HMODULE, c_int, c_int, c_int)

# Enumerates resources of a specified type within a binary module.
# BOOL EnumResourceNamesW(
#   [in, optional] HMODULE          hModule,
#   [in]           LPCWSTR          lpType,
#   [in]           ENUMRESNAMEPROCW lpEnumFunc,
#   [in]           LONG_PTR         lParam
# );
EnumResourceNames = KERNEL32.EnumResourceNamesW
EnumResourceNames.argtypes = [HMODULE, c_int, EnumResNameProc, c_int]
EnumResourceNames.restype = BOOL

# Determines the location of a resource with the specified type and name in the specified module.
# HRSRC FindResourceW(
#   [in, optional] HMODULE hModule,
#   [in]           LPCWSTR lpName,
#   [in]           LPCWSTR lpType
# );
FindResource = KERNEL32.FindResourceW
FindResource.argtypes = [HMODULE, c_int, c_int]
FindResource.restype = HRSRC

# Retrieves a handle that can be used to obtain a pointer to the first byte of the specified resource in memory.
# HGLOBAL LoadResource(
#   [in, optional] HMODULE hModule,
#   [in]           HRSRC   hResInfo
# );
LoadResource = KERNEL32.LoadResource
LoadResource.argtypes = [HMODULE, HRSRC]
LoadResource.restype = HGLOBAL

# Retrieves the size, in bytes, of the specified resource.
# DWORD SizeofResource(
#   [in, optional] HMODULE hModule,
#   [in]           HRSRC   hResInfo
# );
SizeofResource = KERNEL32.SizeofResource
SizeofResource.argtypes = [HMODULE, HRSRC]
SizeofResource.restype = DWORD

# Retrieves a pointer to the specified resource in memory.
# LPVOID LockResource(
#   [in] HGLOBAL hResData
# );
LockResource = KERNEL32.LockResource
LockResource.argtypes = [HGLOBAL]
LockResource.restype = LPVOID

# Retrieves the address of an exported function (also known as a procedure)
# or variable from the specified dynamic-link library (DLL).
# FARPROC GetProcAddress(
#   [in] HMODULE hModule,
#   [in] LPCSTR  lpProcName
# );
GetProcAddress = KERNEL32.GetProcAddress
GetProcAddress.argtypes = [HMODULE, LPCSTR]
GetProcAddress.restype = LPVOID


def is_started_as_admin() -> bool:
    """Checks if program was started with administrator rights.

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


def get_icon_resources(hmodule: HMODULE) -> List[c_int]:
    """Returns list of icon resources identifiers.

    Args:
        hmodule (HMODULE): A handle to the module whose executable file
            contains the resources that are being enumerated.

    Returns:
        List[c_int]: List of resource identifiers.
    """
    icons = []

    @EnumResNameProc
    def add_icon_idx(_hModule, _lpType, lpName, _lParam):
        icons.append(lpName)
        return True

    EnumResourceNames(hmodule, RT_ICON, add_icon_idx, 0)
    return icons


def get_hq_icon_raw(path: str) -> bytes | None:
    """Returns high quality icon from given executable path if possible.

    Image will be returned as raw BGRA pixel data.

    Args:
        path (str): Path to executable.

    Returns:
        bytes | None: Icon resource bytes.
    """
    header_shift = 40
    hlib = None
    try:
        hlib = LoadLibrary(path, 0, LOAD_LIBRARY_AS_DATAFILE)
        if hlib:
            icons = get_icon_resources(hlib)
            if len(icons) > 0:
                hres = FindResource(hlib, icons[-1], RT_ICON)
                size = SizeofResource(hlib, hres)
                res = LoadResource(hlib, hres)
                mem_pointer = LockResource(res)
                px_res = int((size // 4) ** 0.5) // 8 * 8
                return bytes((c_ubyte * (px_res * px_res * 4)).from_address(mem_pointer + header_shift))
        return None
    except Exception:
        return None
    finally:
        if hlib:
            FreeLibrary(hlib)
        del hlib
