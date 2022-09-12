"""
PyHook
~~~~~~~~~~~~~
Python hook for ReShade processing
:copyright: (c) 2022 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""
import logging
import os
import sys
from threading import Timer

import numpy as np

# pylint: disable=unused-import
import utils  # To be available in PyInstaller frozen bundle.
from _version import __version__
from dll_utils import (
    AddonHandler,
    AddonNotFoundException,
    ProcessNotFoundException,
    ReShadeNotFoundException,
    get_reshade_addon_handler,
)
from mem_utils import FRAME_ARRAY, SIZE_ARRAY, MemoryManager, WaitAddonNotFoundException, WaitProcessNotFoundException
from pipeline import (
    FrameSizeModificationError,
    PipelinesDirNotFoundError,
    load_pipelines,
    load_settings,
    save_settings,
)
from win_utils import is_started_as_admin

# Time in seconds after last settings change to wait until autosave.
_AUTOSAVE_SETTINGS_SECONDS = 5


def _get_logger() -> logging.Logger:
    """Returns logger for PyHook.

    Returns:
        logging.Logger: The configured logger.
    """
    logger = logging.getLogger("PyHook")
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    logger.addHandler(handler)
    return logger


def _wait_on_exit(code: int) -> None:
    """Waits for dummy input to display errors before exiting.

    Args:
        code (int): The exit code.
    """
    input("Press any key to exit...")
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
        array (numpy.array): The frame image as numpy array.
            Array has to be 3-D with height, width, channels as dimensions.
            Array has to contains uint8 values.
    """
    arr = np.zeros(SIZE_ARRAY, dtype=np.uint8)
    arr[: 3 * data.width * data.height] = frame.ravel()
    data.frame = FRAME_ARRAY.from_buffer(arr)


def _pid_input_fallback(logger: logging.Logger) -> AddonHandler:
    """Fallback to manual PID supply.

    Args:
        logger (logging.Logger): Logger for error display.

    Returns:
        AddonHandler: The handler for PyHook addon management.
    """
    logger.info("- Fallback to manual PID input...")
    pid = None
    try:
        pid = int(input("-- PID: "))
    except ValueError:
        logger.error("--- Invalid PID number.")
        _wait_on_exit(1)
    return get_reshade_addon_handler(pid)


def _main():
    """Script entrypoint"""
    try:
        os.system("cls")  # To clear PyInstaller warnings.
        displayed_ms_error = False
        logger = _get_logger()
        logger.info("PyHook v%s (c) 2022 by Dominik Wojtasik", __version__)
        logger.info("- Loading pipelines...")
        pipelines = load_pipelines(logger)
        if len(pipelines) == 0:
            logger.error("-- Cannot find any pipeline to process.")
            _wait_on_exit(1)
        try:
            logger.info("- Searching for process with ReShade...")
            addon_handler = get_reshade_addon_handler()
        except ReShadeNotFoundException:
            logger.error("-- Cannot find any active process with ReShade loaded.")
            if not is_started_as_admin():
                logger.info("-- NOTE: Try to run this program as administrator.")
            addon_handler = _pid_input_fallback(logger)
        memory_manager = MemoryManager(addon_handler.pid)
        logger.info("-- Selected process: %s", addon_handler.get_info())
        logger.info("- Started addon injection for %s...", addon_handler.addon_path)
        try:
            addon_handler.inject_addon()
            logger.info("-- Addon injected!")
        except Exception as ex:
            logger.error("-- Cannot inject addon into given process.", exc_info=ex)
            _wait_on_exit(1)
        logger.info("- Loading PyHook configuration if exists...")
        save_later = None
        runtime_data, data_exists = load_settings(pipelines, addon_handler.dir_path)
        if not data_exists:
            save_settings(
                pipelines, runtime_data.pipeline_order, runtime_data.active_pipelines, addon_handler.dir_path, logger
            )
        logger.info("- Writing configuration to addon...")
        memory_manager.write_shared_pipelines(list(pipelines.values()), runtime_data)
        logger.info("- Started processing...")
        while True:
            memory_manager.wait(addon_handler)
            data = memory_manager.read_shared_data()
            # Multisampled buffer cannot be processed.
            if data.multisampled:
                if not displayed_ms_error:
                    logger.error("-- Disable multisampling (MSAA) in game to process frames!")
                    displayed_ms_error = True
                memory_manager.unlock()
                continue
            # Process pipelines changes.
            runtime_data, needs_save = memory_manager.read_pipelines()
            for unload_pipeline in runtime_data.to_unload:
                pipelines[unload_pipeline].unload()
            for load_pipeline in runtime_data.to_load:
                pipelines[load_pipeline].load()
            for update_pipeline, settings in runtime_data.changes.items():
                is_active = update_pipeline in runtime_data.active_pipelines
                for key, value in settings.items():
                    pipelines[update_pipeline].change_settings(is_active, key, value)
            # Autosave settings.
            if needs_save:
                if save_later is not None and not save_later.finished.is_set():
                    save_later.cancel()
                save_later = Timer(
                    _AUTOSAVE_SETTINGS_SECONDS,
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
                    raise FrameSizeModificationError()
                _encode_frame(data, frame)
            except FrameSizeModificationError:
                logger.info("-- ERROR: Frame modification detected! Frame skipped...")
            memory_manager.unlock()
    except PipelinesDirNotFoundError:
        logger.error("-- Cannot find pipelines directory.")
        logger.info("-- Make sure pipelines directory exists in PyHook directory.")
        _wait_on_exit(1)
    except AddonNotFoundException:
        logger.error("-- Cannot find addon file.")
        logger.info("-- Make sure that *.addon file is built and exists in PyHook directory.")
        _wait_on_exit(1)
    except ProcessNotFoundException:
        logger.error("--- Process with given PID does not exists.")
        _wait_on_exit(1)
    except WaitProcessNotFoundException:
        logger.error("-- Connected process does not exists anymore. Exiting...")
        _wait_on_exit(1)
    except WaitAddonNotFoundException:
        logger.error("-- Connected process does not have addon loaded anymore. Exiting...")
        logger.error("-- Check ReShade logs for more informations.")
        _wait_on_exit(1)
    except Exception as ex:
        logger.error("Unhandled exception occurres.", exc_info=ex)
        _wait_on_exit(1)


if __name__ == "__main__":
    _main()
