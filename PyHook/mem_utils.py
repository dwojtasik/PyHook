"""
mem_utils for PyHook
~~~~~~~~~~~~~~~~~~~~~~~
Utils for shared memory management
:copyright: (c) 2022 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""

import mmap
from ctypes import *

_MAX_WIDTH = 3840
_MAX_HEIGHT = 2160
_KERNEL32 = windll.kernel32

SIZE_ARRAY = _MAX_WIDTH * _MAX_HEIGHT * 3
FRAME_ARRAY = c_uint8 * SIZE_ARRAY


class SharedData(Structure):
    """Sturcture for shared memory data.

    frame_count (c_longlong): Actual frame count calculated since addon initialization.
    width (c_uint32): Frame width in pixels.
    height (c_uint32): Frame height in pixels.
    multisampled (c_bool): Flag if buffer is using multisampling.
    frame (FRAME_ARRAY): Frame data as array in format [R,G,B,R,G,B,R,G,B...].
        Each pixel RGB component is stored in c_uint8 format.
        Pixels are read row by row from top left frame corner.
        FRAME_ARRAY has declared size of pixel component count in max supported resolution (4K).
        Additional data (above actual frame resolution) is unused and filled with zeros.
    """
    _fields_ = [
        ('frame_count', c_longlong),
        ('width', c_uint32),
        ('height', c_uint32),
        ('multisampled', c_bool),
        ('frame', FRAME_ARRAY),
    ]


class MemoryManager:
    """Manages shared memory between Python and C++ DLL.

    pid (int): Process ID that has PyHook addon DLL loaded.

    _lock_event (HANDLE): Lock event handle. When in signaled state it allows
        to process data in PyHook Python part.
    _unlock_event (HANDLE): Unlock event handle. When in signaled state it allows
        to process data in ReShade addon part.
    _shmem (mmap.mmap): Shared memory implementation using memory-mapped file object.
    """

    _EVENT_LOCK_NAME = "PyHookEvLOCK_"
    _EVENT_UNLOCK_NAME = "PyHookEvUNLOCK_"
    _SHMEM_NAME = "PyHookSHMEM_"

    def __init__(self, pid: int):
        spid = str(pid)
        self._lock_event = _KERNEL32.CreateEventW(
            0, 0, 0, self._EVENT_LOCK_NAME + spid)
        self._unlock_event = _KERNEL32.CreateEventW(
            0, 0, 0, self._EVENT_UNLOCK_NAME + spid)
        self._shmem = mmap.mmap(-1, sizeof(SharedData),
                                self._SHMEM_NAME + spid)

    def read_shared_data(self) -> SharedData:
        """Reads data buffer from shared memory and wraps it to Python object.

        Returns:
            SharedData: The shared memory data.
        """
        return SharedData.from_buffer(self._shmem)

    def wait(self) -> None:
        """Waits for lock event handle in signaled state.
        After finish it allows Python to process frame from ReShade.
        """
        _KERNEL32.WaitForSingleObject(self._lock_event, 0xFFFFFFFF)

    def unlock(self) -> None:
        """Sends signal to unlock event handle.
        After finish it allows ReShade to process frame from Python.
        """
        _KERNEL32.SetEvent(self._unlock_event)
