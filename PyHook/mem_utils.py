"""
mem_utils for PyHook
~~~~~~~~~~~~~~~~~~~~~~~
Utils for shared memory management
:copyright: (c) 2022 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""

import mmap
from ctypes import *
from typing import List, Tuple

from pipeline import Pipeline

_MAX_WIDTH = 3840
_MAX_HEIGHT = 2160
_KERNEL32 = windll.kernel32

SIZE_ARRAY = _MAX_WIDTH * _MAX_HEIGHT * 3
FRAME_ARRAY = c_uint8 * SIZE_ARRAY

PIPELINE_LIMIT = 100
PIPELINE_SHORT_STRING = c_char * 12
PIPELINE_STRING = c_char * 64
PIPELINE_TEXT = c_char * 512


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


class ActivePipeline(Structure):
    """Sturcture for active pipeline data.

    enabled (c_bool): Flag if given pipeline is enabled.
    file (PIPELINE_STRING): Pipeline filename.
    """
    _fields_ = [
        ('enabled', c_bool),
        ('file', PIPELINE_STRING),
    ]


class PipelineData(ActivePipeline):
    """Sturcture for shared pipeline data.

    name (PIPELINE_STRING): Pipeline name.
    version (PIPELINE_SHORT_STRING): Pipeline version.
    desc (PIPELINE_TEXT): Pipeline description.
    """
    _fields_ = [
        ('name', PIPELINE_STRING),
        ('version', PIPELINE_SHORT_STRING),
        ('desc', PIPELINE_TEXT),
    ]


PIPELINE_ARRAY = PIPELINE_LIMIT * PipelineData
PIPELINE_ORDER = PIPELINE_LIMIT * PIPELINE_STRING


class ActiveConfigData(Structure):
    """Pipeline configuration modification structure.

    modified (c_bool): Flag is configuration was modified in ReShade.
    """
    _fields_ = [
        ('modified', c_bool)
    ]


class SharedConfigData(ActiveConfigData):
    """Pipeline configuration structure.

    count (c_int): Count of pipelines.
    order (PIPELINE_ORDER): Pipeline order.
    pipelines (PIPELINE_ARRAY): Pipeline data array.
    """
    _fields_ = [
        ('count', c_int),
        ('order', PIPELINE_ORDER),
        ('pipelines', PIPELINE_ARRAY),
    ]


class MemoryManager:
    """Manages shared memory between Python and C++ DLL.

    pid (int): Process ID that has PyHook addon DLL loaded.

    _lock_event (HANDLE): Lock event handle. When in signaled state it allows
        to process data in PyHook Python part.
    _unlock_event (HANDLE): Unlock event handle. When in signaled state it allows
        to process data in ReShade addon part.
    _shmem (mmap.mmap): Shared memory using memory-mapped file object for frame processing.
    _shcfg (mmap.mmap): Shared memory using memory-mapped file object for configuration.
    _active_pipelines (List[str]): List of active pipelines order.
    """

    _EVENT_LOCK_NAME = "PyHookEvLOCK_"
    _EVENT_UNLOCK_NAME = "PyHookEvUNLOCK_"
    _SHMEM_NAME = "PyHookSHMEM_"
    _SHCFG_NAME = "PyHookSHCFG_"

    def __init__(self, pid: int):
        spid = str(pid)
        self._lock_event = _KERNEL32.CreateEventW(
            0, 0, 0, self._EVENT_LOCK_NAME + spid)
        self._unlock_event = _KERNEL32.CreateEventW(
            0, 0, 0, self._EVENT_UNLOCK_NAME + spid)
        self._shmem = mmap.mmap(-1, sizeof(SharedData),
                                self._SHMEM_NAME + spid)
        self._shcfg = mmap.mmap(-1, sizeof(SharedConfigData),
                                self._SHCFG_NAME + spid)
        self._active_pipelines = []

    def read_shared_data(self) -> SharedData:
        """Reads data buffer from shared memory and wraps it to Python object.

        Returns:
            SharedData: The shared memory data.
        """
        return SharedData.from_buffer(self._shmem)

    def read_pipelines(self) -> Tuple[List[str], List[str], List[str]]:
        """Reads pipeline lists from shared configuration.

        Contains lists of pipelines: active, to unload and to load.
        If configuration was modified new active pipelines order will be decoded.

        Returns:
            Tuple[List[str], List[str], List[str]]: Returns multiple list with following data:
                ([The active pipeline list][The pipelines to unload][The pipelines to load])
        """
        active_data = ActiveConfigData.from_buffer(self._shcfg)
        to_unload = []
        to_load = []
        if active_data.modified:
            pipeline_data = SharedConfigData.from_buffer(self._shcfg)
            pipeline_array = [
                ActivePipeline.from_buffer(buf)
                for buf in pipeline_data.pipelines[:pipeline_data.count]
            ]
            active_pipelines = [
                pipeline.file.decode('utf8')
                for pipeline in pipeline_array
                if pipeline.enabled
            ]
            pipeline_order = [
                file.value.decode('utf8')
                for file in pipeline_data.order[:pipeline_data.count]
            ]
            old_pipelines = self._active_pipelines
            self._active_pipelines = [
                file
                for file in pipeline_order
                if file in active_pipelines
            ]
            to_unload = [
                file
                for file in old_pipelines
                if file not in self._active_pipelines
            ]
            to_load = [
                file
                for file in self._active_pipelines
                if file not in old_pipelines
            ]
            active_data.modified = False
        return (self._active_pipelines, to_unload, to_load)

    def write_shared_pipelines(self, pipelines: List[Pipeline]) -> None:
        """Writes pipelines data into shared configuration.

        Args:
            pipelines (List[Pipeline]): Pipeline list to write into shared configuration.
        """
        pipeline_data = SharedConfigData.from_buffer(self._shcfg)
        pipeline_data.modified = False
        pipeline_data.count = len(pipelines)
        pipeline_order = (PIPELINE_ORDER)()
        pipeline_array = (PIPELINE_ARRAY)()
        for i in range(len(pipelines)):
            encoded_file = pipelines[i].file[:PIPELINE_STRING._length_].encode(
                'utf8')
            pipeline_order[i] = (PIPELINE_STRING)(*encoded_file)
            pipeline_array[i] = PipelineData(
                enabled=False,
                file=encoded_file,
                name=pipelines[i].name[:PIPELINE_STRING._length_].encode(
                    'utf8'),
                version=pipelines[i].version[:PIPELINE_SHORT_STRING._length_].encode(
                    'utf8'),
                desc=pipelines[i].desc[:PIPELINE_TEXT._length_].encode('utf8')
            )
        pipeline_data.order = pipeline_order
        pipeline_data.pipelines = pipeline_array

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
