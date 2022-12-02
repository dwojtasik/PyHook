"""
PyHook
~~~~~~~~~~~~~
Python hook for ReShade processing
:copyright: (c) 2022 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""
import json
import logging
import sys
from logging.handlers import QueueHandler
from multiprocessing import Queue, Value
from os.path import exists
from threading import Timer

import numpy as np

# To be available in PyInstaller frozen bundle.
import utils  # pylint: disable=unused-import
from _version import __version__
from dll_utils import (
    AddonNotFoundException,
    ProcessNotFoundException,
    ReShadeNotFoundException,
    get_reshade_addon_handler,
)
from keys import SettingsKeys
from mem_utils import FRAME_ARRAY, SIZE_ARRAY, MemoryManager, WaitAddonNotFoundException, WaitProcessNotFoundException
from pipeline import (
    FrameProcessingError,
    FrameSizeModificationError,
    PipelinesDirNotFoundError,
    load_pipelines,
    load_settings,
    save_settings,
)
from win_utils import is_started_as_admin

# Settings file path
_SETTINGS_PATH = "settings.json"

# PyHook settings
_SETTINGS = {
    SettingsKeys.KEY_AUTOSAVE: 5,
    SettingsKeys.KEY_AUTODOWNLOAD: True,
    SettingsKeys.KEY_DOWNLOADED: [],
}

# PyHook main logger
_LOGGER: logging.Logger = None


class LogWriter:
    """Allows to redirect stdout to logger.

    Args:
        logger (logging.Logger): Logger to write.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger

    def write(self, text: str) -> None:
        """Writes message to logger with info level.

        Args:
            text (str): Text to write.
        """
        for line in text.splitlines():
            if line:
                self.logger.info(line.rstrip())

    def flush(self) -> None:
        """Empty flush method."""
        pass  # pylint: disable=unnecessary-pass


def _init_logger(log_queue: Queue) -> None:
    """Initializes logger for PyHook.

    Args:
        log_queue (Queue): Log queue from parent process.
    """
    global _LOGGER  # pylint: disable=global-statement
    if _LOGGER is None:
        _LOGGER = logging.getLogger("PyHook")
        _LOGGER.setLevel(logging.INFO)
        _LOGGER.addHandler(QueueHandler(log_queue))


def _save_settings() -> None:
    """Saves PyHook settings to file."""
    try:
        with open(_SETTINGS_PATH, "w", encoding="utf-8") as settings_file:
            json.dump(_SETTINGS, settings_file, indent=4)
    except PermissionError:
        if _LOGGER is not None:
            _LOGGER.info(
                "-- Error: Cannot save settings to file %s. Permission denied. Try to run PyHook as admin.",
                _SETTINGS_PATH,
            )
    except Exception as ex:
        if _LOGGER is not None:
            _LOGGER.error(
                "-- Error: Cannot save settings to file %s. Unhandled exception occurres.", _SETTINGS_PATH, exc_info=ex
            )


def _load_settings() -> None:
    """Loads PyHook settings from file."""
    _LOGGER.info("- Loading settings...")
    if exists(_SETTINGS_PATH):
        with open(_SETTINGS_PATH, encoding="utf-8") as settings_file:
            settings = json.load(settings_file)
            for key, value in settings.items():
                if key == SettingsKeys.KEY_AUTOSAVE:
                    if not isinstance(value, int):
                        _LOGGER.info("-- Warning: Invalid value for key: %s=%s. Skipping...", key, str(value))
                        continue
                    if int(value) < 1:
                        _LOGGER.info(
                            "-- Warning: Value for %s has to be bigger than 0. Actual value: %s. Restored value %s.",
                            key,
                            str(value),
                            str(_SETTINGS[key]),
                        )
                        continue
                    _SETTINGS[key] = int(value)
                elif key == SettingsKeys.KEY_AUTODOWNLOAD:
                    _SETTINGS[key] = bool(value)
                elif key == SettingsKeys.KEY_DOWNLOADED:
                    if not isinstance(value, list):
                        _LOGGER.info("-- Warning: Invalid value for key: %s=%s. Skipping...", key, str(value))
                        continue
                    _SETTINGS[key] = list(value)
                else:
                    _LOGGER.info("-- Warning: Unrecognized settings key: %s. Skipping...", key)
        _LOGGER.info("-- Settings have been loaded.")
    else:
        _LOGGER.info("-- Settings not found. Creating default settings...")
        _save_settings()


def _exit(running: Value, code: int) -> None:
    """Changes flag for session state and exits.

    Args:
        running (Value[bool]): Shared flag if process is running.
        code (int): The exit code.
    """
    running.value = False
    sys.exit(code)


def _decode_frame(data) -> np.array:
    """Decodes frame from shared data as image in numpy array format.

    Args:
        data (SharedData): The shared data to read frame C array.

    Returns:
        numpy.array: The frame image as numpy array.
            Array has to be 3-D with height, width, channels as dimensions.
            Array has to contains uint8 values.
    """
    arr = np.ctypeslib.as_array(data.frame)[: 3 * data.width * data.height]
    return arr.reshape((data.height, data.width, 3))


def _encode_frame(data, frame) -> None:
    """Encodes numpy image array as C array and stores it in shared data.

    Args:
        data (SharedData): The shared data to store frame C array.
        frame (numpy.array): The frame image as numpy array.
            Array has to be 3-D with height, width, channels as dimensions.
            Array has to contains uint8 values.
    """
    arr = np.zeros(SIZE_ARRAY, dtype=np.uint8)
    arr[: 3 * data.width * data.height] = frame.ravel()
    data.frame = FRAME_ARRAY.from_buffer(arr)


def pyhook_main(running: Value, pid: Value, name: Value, log_queue: Queue) -> None:
    """PyHook entrypoint.

    Args:
        running (Value[bool]): Shared flag if process is running.
        pid (Value[int]): Shared integer process id.
        name (Value[str]): Shared string process name.
        log_queue (Queue): Log queue from parent process.
    """
    try:
        displayed_ms_error = False
        _init_logger(log_queue)
        sys.stdout = LogWriter(_LOGGER)
        _LOGGER.info("PyHook v%s (c) 2022 by Dominik Wojtasik", __version__)
        _load_settings()
        _LOGGER.info("- Loading pipelines...")
        pipelines, has_changes = load_pipelines(_SETTINGS, _LOGGER)
        if has_changes:
            _save_settings()
        if len(pipelines) == 0:
            _LOGGER.error("-- Cannot find any pipeline to process.")
            _exit(running, 1)
        if pid.value < 0:
            try:
                _LOGGER.info("- Searching for process with ReShade...")
                addon_handler = get_reshade_addon_handler()
                pid.value = addon_handler.pid
                name.value = str.encode(addon_handler.process_name)
            except ReShadeNotFoundException:
                _LOGGER.error("-- Cannot find any active process with ReShade loaded.")
                if not is_started_as_admin():
                    _LOGGER.info("-- NOTE: Try to run this program as administrator.")
                _exit(running, 1)
        else:
            _LOGGER.info("- Creating handler for PID: %d...", pid.value)
            addon_handler = get_reshade_addon_handler(pid.value)
        memory_manager = MemoryManager(addon_handler.pid)
        _LOGGER.info("-- Selected process: %s", addon_handler.get_info())
        _LOGGER.info("- Started addon injection for %s...", addon_handler.addon_path)
        try:
            addon_handler.inject_addon()
            _LOGGER.info("-- Addon injected!")
        except Exception as ex:
            _LOGGER.error("-- Cannot inject addon into given process.", exc_info=ex)
            _exit(running, 1)
        _LOGGER.info("- Loading PyHook configuration if exists...")
        save_later = None
        runtime_data, data_exists = load_settings(pipelines, addon_handler.dir_path)
        if not data_exists:
            save_settings(
                pipelines, runtime_data.pipeline_order, runtime_data.active_pipelines, addon_handler.dir_path, _LOGGER
            )
        _LOGGER.info("- Writing configuration to addon...")
        memory_manager.write_shared_pipelines(list(pipelines.values()), runtime_data)
        _LOGGER.info("- Started processing...")
        while running.value:
            memory_manager.wait(addon_handler)
            data = memory_manager.read_shared_data()
            # Multisampled buffer cannot be processed.
            if data.multisampled:
                if not displayed_ms_error:
                    _LOGGER.error("-- Disable multisampling (MSAA) in game to process frames!")
                    displayed_ms_error = True
                memory_manager.unlock()
                continue
            # Process pipelines changes.
            runtime_data, needs_save = memory_manager.read_pipelines()
            for unload_pipeline in runtime_data.to_unload:
                try:
                    pipelines[unload_pipeline].unload()
                except Exception as ex:
                    _LOGGER.error(
                        '-- ERROR: Unexpected error during unload of pipeline="%s"', unload_pipeline, exc_info=ex
                    )
            has_to_disable = []
            for load_pipeline in runtime_data.to_load:
                try:
                    pipelines[load_pipeline].load()
                except Exception as ex:
                    _LOGGER.error(
                        '-- ERROR: Unexpected error during load of pipeline="%s"', load_pipeline, exc_info=ex
                    )
                    runtime_data.active_pipelines.remove(load_pipeline)
                    has_to_disable.append(load_pipeline)
            if len(has_to_disable) > 0:
                memory_manager.force_disable_pipelines(has_to_disable)
            for update_pipeline, settings in runtime_data.changes.items():
                is_active = update_pipeline in runtime_data.active_pipelines
                for key, value in settings.items():
                    old_value = pipelines[update_pipeline].settings[key][0]
                    try:
                        pipelines[update_pipeline].change_settings(is_active, key, value)
                    except Exception as ex:
                        pipelines[update_pipeline].settings[key][0] = old_value
                        _LOGGER.error(
                            '-- ERROR: Unexpected error during setting change of pipeline="%s" for "%s"=%s',
                            update_pipeline,
                            key,
                            pipelines[update_pipeline]._to_value(key, value),
                            exc_info=ex,
                        )
                        _LOGGER.info("--- Restored old value=%s", old_value)
            # Autosave settings.
            if needs_save:
                if save_later is not None and not save_later.finished.is_set():
                    save_later.cancel()
                save_later = Timer(
                    _SETTINGS[SettingsKeys.KEY_AUTOSAVE],
                    save_settings,
                    [pipelines, runtime_data.pipeline_order, runtime_data.active_pipelines, addon_handler.dir_path],
                )
                save_later.start()
            # Skip frame processing if user didn't select any pipeline.
            if len(runtime_data.active_pipelines) == 0:
                memory_manager.unlock()
                continue
            # Process all selected pipelines in order.
            frame = _decode_frame(data)
            try:
                passes = {}
                f_width = data.width
                f_height = data.height
                f_channels = frame.shape[2]
                f_count = data.frame_count
                runtime_order = [p for p in runtime_data.pipeline_order if p in runtime_data.active_pipelines]
                for active_pipeline in runtime_order:
                    pipeline = pipelines[active_pipeline]
                    stage = None
                    if pipeline.multistage > 1:
                        if active_pipeline not in passes:
                            passes[active_pipeline] = 0
                        passes[active_pipeline] += 1
                        stage = passes[active_pipeline]
                    frame = pipeline.process_frame(frame, f_width, f_height, f_count, stage)
                    if stage is not None:
                        f_width = frame.shape[1]
                        f_height = frame.shape[0]
                if f_width != data.width or f_height != data.height or f_channels != frame.shape[2]:
                    raise FrameSizeModificationError(
                        f"multistage processing in frame={f_count}",
                    )
                _encode_frame(data, frame)
            except FrameSizeModificationError as ex:
                _LOGGER.info("-- ERROR: Frame modification detected for %s! Frame skipped...", ex)
            except FrameProcessingError as ex:
                _LOGGER.error("-- ERROR: %s", ex.message, exc_info=ex.exception)
            memory_manager.unlock()
        sys.exit(0)
    except PipelinesDirNotFoundError:
        _LOGGER.error("-- Cannot find pipelines directory.")
        _LOGGER.info("-- Make sure pipelines directory exists in PyHook directory.")
        _exit(running, 1)
    except AddonNotFoundException:
        _LOGGER.error("-- Cannot find addon file.")
        _LOGGER.info("-- Make sure that *.addon file is built and exists in PyHook directory.")
        _exit(running, 1)
    except ProcessNotFoundException:
        _LOGGER.error("--- Process with given PID does not exists.")
        _exit(running, 1)
    except WaitProcessNotFoundException:
        _LOGGER.error("-- Connected process does not exists anymore. Exiting...")
        _exit(running, 1)
    except WaitAddonNotFoundException:
        _LOGGER.error("-- Connected process does not have addon loaded anymore. Exiting...")
        _LOGGER.error("-- Check ReShade logs for more informations.")
        _exit(running, 1)
    except Exception as ex:
        _LOGGER.error("Unhandled exception occurres.", exc_info=ex)
        _exit(running, 1)
