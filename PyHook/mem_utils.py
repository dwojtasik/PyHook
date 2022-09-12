"""
mem_utils for PyHook
~~~~~~~~~~~~~~~~~~~~~~~
Utils for shared memory management
:copyright: (c) 2022 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""

import mmap

# pylint: disable=wildcard-import,unused-wildcard-import
from ctypes import *
from os import getpid
from typing import List, Tuple

from psutil import pid_exists

from pipeline import Pipeline, PipelineRuntimeData
from dll_utils import AddonHandler

# Handle for kernel32 DLL.
_KERNEL32 = windll.kernel32

# Timeout in millis for event singnaling.
# If timeout occurs PyHook will check if connected process does still exists.
_WAIT_TIME_MS = 2000

# Const values for shared memory allocation.
# Max resolution width.
_MAX_WIDTH = 3840
# Max resolution height.
_MAX_HEIGHT = 2160
# Frame array size (width * height * 3 color channel (RGB)).
SIZE_ARRAY = _MAX_WIDTH * _MAX_HEIGHT * 3
# Frame array type, where each RGB component is stored as uint8.
FRAME_ARRAY = c_uint8 * SIZE_ARRAY
# Max pipeline definitions.
PIPELINE_LIMIT = 100
# Max variable count per pipeline.
PIPELINE_VAR_LIMIT = 10

# Pipeline C types for strings with multiple lengths.
PIPELINE_SHORT_STRING = c_char * 12  # Used in version string.
PIPELINE_KEY_STRING = c_char * 32  # Used in variable names.
PIPELINE_STRING = c_char * 64  # Used in pipeline names.
PIPELINE_SHORT_TEXT = c_char * 256  # Used in variable tooltip.
PIPELINE_TEXT = c_char * 512  # Used in pipeline description.


class WaitProcessNotFoundException(Exception):
    """Raised when process with given PID does not exists anymore and cannot set signaled state."""


class WaitAddonNotFoundException(Exception):
    """Raised when process with given PID does not have addon loadad anymore and cannot set signaled state."""


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
        ("frame_count", c_longlong),
        ("width", c_uint32),
        ("height", c_uint32),
        ("multisampled", c_bool),
        ("frame", FRAME_ARRAY),
    ]


class PipelineVar(Structure):
    """Sturcture for active pipeline variables.

    modified (c_bool): Flag if given variable was modified.
    key (PIPELINE_KEY_STRING): Name of the variable.
    value (c_float): Value of the variable.
    type (c_short): Type of the variable.
        0 = bool, 1 = int, 2 = float.
    min (c_float): Minimum value of the variable.
    max (c_float): Maximum value of the variable.
    step (c_float): Change step between min and max values.
    tooltip (PIPELINE_SHORT_TEXT): Tooltip to be displayed.
    """

    _fields_ = [
        ("modified", c_bool),
        ("key", PIPELINE_KEY_STRING),
        ("value", c_float),
        ("type", c_short),
        ("min", c_float),
        ("max", c_float),
        ("step", c_float),
        ("tooltip", PIPELINE_SHORT_TEXT),
    ]


# Pipeline variables C array.
PIPELINE_SETTINGS = PIPELINE_VAR_LIMIT * PipelineVar


class ActivePipeline(Structure):
    """Sturcture for active pipeline data.

    enabled (c_bool): Flag if given pipeline is enabled.
    modified (c_bool): Flag if given pipeline had it settings modified.
    file (PIPELINE_STRING): Pipeline filename.
    var_count (c_int): Count of settings in settings list.
    settings (PIPELINE_SETTINGS): Array of pipeline variables.
    """

    _fields_ = [
        ("enabled", c_bool),
        ("modified", c_bool),
        ("file", PIPELINE_STRING),
        ("var_count", c_int),
        ("settings", PIPELINE_SETTINGS),
    ]


class PipelineData(ActivePipeline):
    """Sturcture for shared pipeline data.

    name (PIPELINE_STRING): Pipeline name.
    version (PIPELINE_SHORT_STRING): Pipeline version.
    desc (PIPELINE_TEXT): Pipeline description.
    """

    _fields_ = [
        ("name", PIPELINE_STRING),
        ("version", PIPELINE_SHORT_STRING),
        ("desc", PIPELINE_TEXT),
    ]


# Pipelines C array.
PIPELINE_ARRAY = PIPELINE_LIMIT * PipelineData
# C string array of pipelines IDs (filenames) as order of processing.
PIPELINE_ORDER = 3 * PIPELINE_LIMIT * PIPELINE_STRING


class ActiveConfigData(Structure):
    """Pipeline configuration modification structure.

    modified (c_bool): Flag is configuration was modified in ReShade.
    """

    _fields_ = [("modified", c_bool)]


class SharedConfigData(ActiveConfigData):
    """Pipeline configuration structure.

    pyhook_pid (c_int): Process ID of PyHook application.
    count (c_int): Count of pipelines.
    order_count (c_int): Count of pipeline passes in order.
    order (PIPELINE_ORDER): Pipeline order.
    pipelines (PIPELINE_ARRAY): Pipeline data array.
    """

    _fields_ = [
        ("pyhook_pid", c_int),
        ("count", c_int),
        ("order_count", c_int),
        ("order", PIPELINE_ORDER),
        ("pipelines", PIPELINE_ARRAY),
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
    _pipeline_order (List[str]): Order of the pipelines.
    _active_pipelines (List[str]): List of active pipelines order.
    """

    _EVENT_LOCK_NAME = "PyHookEvLOCK_"
    _EVENT_UNLOCK_NAME = "PyHookEvUNLOCK_"
    _SHMEM_NAME = "PyHookSHMEM_"
    _SHCFG_NAME = "PyHookSHCFG_"

    def __init__(self, pid: int):
        self.pid = pid
        spid = str(pid)
        self._lock_event = _KERNEL32.CreateEventW(0, 0, 0, self._EVENT_LOCK_NAME + spid)
        self._unlock_event = _KERNEL32.CreateEventW(0, 0, 0, self._EVENT_UNLOCK_NAME + spid)
        self._shmem = mmap.mmap(-1, sizeof(SharedData), self._SHMEM_NAME + spid)
        self._shcfg = mmap.mmap(-1, sizeof(SharedConfigData), self._SHCFG_NAME + spid)
        self._pipeline_order = []
        self._active_pipelines = []

    def read_shared_data(self) -> SharedData:
        """Reads data buffer from shared memory and wraps it to Python object.

        Returns:
            SharedData: The shared memory data.
        """
        return SharedData.from_buffer(self._shmem)

    def read_pipelines(self) -> Tuple[PipelineRuntimeData, bool]:
        """Reads pipeline runtime data from shared configuration.

        If configuration was modified new active pipelines order will be decoded.

        Returns:
            Tuple[PipelineRuntimeData, bool]: Pipelines runtime data and flag if needs save.
                Settings for update are stored in dictionary, where key is pipeline file and value is
                dictionary of modified key-value pairs.
        """
        needs_save = False
        active_data = ActiveConfigData.from_buffer(self._shcfg)
        to_unload = []
        to_load = []
        changes = {}
        if active_data.modified:
            needs_save = True
            pipeline_data = SharedConfigData.from_buffer(self._shcfg)
            pipeline_array = [
                ActivePipeline.from_buffer(buf) for buf in pipeline_data.pipelines[: pipeline_data.count]
            ]
            for pipeline in pipeline_array:
                if pipeline.modified:
                    p_key = pipeline.file.decode("utf8")
                    changes[p_key] = {}
                    for i in range(pipeline.var_count):
                        variable = pipeline.settings[i]
                        if variable.modified:
                            changes[p_key][variable.key.decode("utf8")] = variable.value
                            variable.modified = False
                    pipeline.modified = False
            active_pipelines = [pipeline.file.decode("utf8") for pipeline in pipeline_array if pipeline.enabled]
            self._pipeline_order = [
                file.value.decode("utf8") for file in pipeline_data.order[: pipeline_data.order_count]
            ]
            old_pipelines = self._active_pipelines
            self._active_pipelines = []
            for file in self._pipeline_order:
                if file in active_pipelines and file not in self._active_pipelines:
                    self._active_pipelines.append(file)
            to_unload = [file for file in old_pipelines if file not in self._active_pipelines]
            to_load = [file for file in self._active_pipelines if file not in old_pipelines]
            active_data.modified = False
        return (
            PipelineRuntimeData(self._pipeline_order, self._active_pipelines, to_unload, to_load, changes),
            needs_save,
        )

    def write_shared_pipelines(self, pipelines: List[Pipeline], runtime_data: PipelineRuntimeData) -> None:
        """Writes pipelines data into shared configuration.

        Args:
            pipelines (List[Pipeline]): Pipeline list to write into shared configuration.
            runtime_data (PipelineRuntimeData): Pipeline runtime data.
        """
        self._pipeline_order = runtime_data.pipeline_order
        self._active_pipelines = []
        pipeline_data = SharedConfigData.from_buffer(self._shcfg)
        pipeline_data.modified = False
        pipeline_data.pyhook_pid = getpid()
        pipeline_data.count = min(PIPELINE_LIMIT, len(pipelines))
        pipeline_data.order_count = min(3 * PIPELINE_LIMIT, len(runtime_data.pipeline_order))
        pipeline_order = (PIPELINE_ORDER)()
        for i in range(min(PIPELINE_LIMIT, len(runtime_data.pipeline_order))):
            encoded_file = runtime_data.pipeline_order[i][: PIPELINE_STRING._length_].encode("utf8")
            pipeline_order[i] = (PIPELINE_STRING)(*encoded_file)
        pipeline_array = (PIPELINE_ARRAY)()
        for i in range(min(PIPELINE_LIMIT, len(pipelines))):
            settings = (PIPELINE_SETTINGS)()
            if pipelines[i].settings is not None:
                var_idx = 0
                for key, data_list in list(pipelines[i].settings.items())[:PIPELINE_VAR_LIMIT]:
                    settings[var_idx] = PipelineVar(
                        modified=False,
                        key=key[: PIPELINE_KEY_STRING._length_].encode("utf8"),
                        value=float(data_list[0]),
                        type=pipelines[i].mappings[key],
                        min=float(0 if data_list[1] is None else data_list[1]),
                        max=float(0 if data_list[2] is None else data_list[2]),
                        step=float(0 if data_list[3] is None else data_list[3]),
                        tooltip=("" if data_list[4] is None else data_list[4])[: PIPELINE_SHORT_TEXT._length_].encode(
                            "utf8"
                        ),
                    )
                    var_idx += 1

            encoded_file = pipelines[i].file[: PIPELINE_STRING._length_].encode("utf8")
            pipeline_array[i] = PipelineData(
                enabled=pipelines[i].file in runtime_data.active_pipelines,
                modified=False,
                file=encoded_file,
                var_count=min(PIPELINE_VAR_LIMIT, 0 if pipelines[i].settings is None else len(pipelines[i].settings)),
                settings=settings,
                name=pipelines[i].name[: PIPELINE_STRING._length_].encode("utf8"),
                version=pipelines[i].version[: PIPELINE_SHORT_STRING._length_].encode("utf8"),
                desc=pipelines[i].desc[: PIPELINE_TEXT._length_].encode("utf8"),
            )
        pipeline_data.order = pipeline_order
        pipeline_data.pipelines = pipeline_array

    def wait(self, addon_handler: AddonHandler) -> None:
        """Waits for lock event handle in signaled state.
        After finish it allows Python to process frame from ReShade.
        If signaling process will exit in the meantime exception will be thrown.

        Args:
            addon_handler (AddonHandler): Handler for PyHook addon management.

        Raises:
            WaitProcessNotFoundException: When process with given PID does not exists anymore.
            WaitAddonNotFoundException: When process with given PID does not have addon loaded anymore.
        """
        while True:
            wait_result = _KERNEL32.WaitForSingleObject(self._lock_event, c_ulong(_WAIT_TIME_MS))
            if wait_result == 0x00000000:
                return
            if not pid_exists(self.pid):
                raise WaitProcessNotFoundException()
            if not addon_handler.has_addon_loaded():
                raise WaitAddonNotFoundException()

    def unlock(self) -> None:
        """Sends signal to unlock event handle.
        After finish it allows ReShade to process frame from Python.
        """
        _KERNEL32.SetEvent(self._unlock_event)
