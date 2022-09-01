#!/usr/bin/python3

"""
PyHook
~~~~~~~~~~~~~
Python hook for ReShade processing
:copyright: (c) 2022 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""
__version__ = "0.0.1"

import logging
import sys

import numpy as np
import psutil

from dll_utils import *
from mem_utils import FRAME_ARRAY, SIZE_ARRAY, MemoryManager
from pipeline import PipelinesDirNotFoundError, load_pipelines
from win_utils import is_started_as_admin


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
    input('Press any key to exit...')
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
    arr = np.ctypeslib.as_array(data.frame)[:3 * data.width * data.height]
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
    arr[:3 * data.width * data.height] = frame.ravel()
    data.frame = FRAME_ARRAY.from_buffer(arr)


def _pid_input_fallback(logger: logging.Logger) -> AddonHandler:
    logger.info('Fallback to manual PID input...')
    pid = None
    try:
        pid = int(input('PID: '))
    except ValueError:
        logger.error('Invalid PID number.')
        _wait_on_exit()
    return get_reshade_addon_handler(pid)

def _main():
    """Script entrypoint"""
    try:
        displayed_ms_error = False
        logger = _get_logger()
        pipelines = load_pipelines(logger)
        if len(pipelines) == 0:
            logger.error('Cannot find any pipeline to process.')
            _wait_on_exit(1)
        try:
            addon_handler = get_reshade_addon_handler()
        except ReShadeNotFoundException:
            logger.error('Cannot find any active process with ReShade loaded.')
            if not is_started_as_admin():
                logger.info('NOTE: Try to run this program as administrator.')
            addon_handler = _pid_input_fallback(logger)
        memory_manager = MemoryManager(addon_handler.pid)
        logger.info(f'Selected process: {addon_handler.get_info()}')
        logger.info(
            f'- Started addon injection for {addon_handler.addon_path}...')
        try:
            addon_handler.inject_addon()
            logger.info(f'-- Addon injected!')
        except Exception as ex:
            logger.error(f'-- Cannot inject addon into given process. {ex}')
            exit(1)
        logger.info(f'-- Writing configuration to addon...')
        memory_manager.write_shared_pipelines(list(pipelines.values()))
        logger.info(f'-- Started processing...')
        while True:
            memory_manager.wait()
            data = memory_manager.read_shared_data()
            # Multisampled buffer cannot be processed
            if data.multisampled:
                if not displayed_ms_error:
                    logger.error(
                        'Disable multisampling (MSAA) in game to process frames!')
                    displayed_ms_error = True
                memory_manager.unlock()
                continue
            active_pipelines, to_unload, to_load, changes = memory_manager.read_pipelines()
            for unload_pipeline in to_unload:
                pipelines[unload_pipeline].unload()
            for load_pipeline in to_load:
                pipelines[load_pipeline].load()
            for update_pipeline, settings in changes.items():
                is_active = update_pipeline in active_pipelines
                for k, v in settings.items():
                    pipelines[update_pipeline].change_settings(
                        is_active, k, v
                    )
            # Skip if user didn't select any pipeline
            if len(active_pipelines) == 0:
                memory_manager.unlock()
                continue
            # Process all selected pipelines in order
            frame = _decode_frame(data)
            for active_pipeline in active_pipelines:
                frame = pipelines[active_pipeline].process_frame(
                    frame, data.width, data.height, data.frame_count)
            _encode_frame(data, frame)
            memory_manager.unlock()
    except PipelinesDirNotFoundError:
        logger.error('Cannot find pipelines directory.')
        logger.info(
            'Make sure pipelines directory exists in PyHook directory.')
        _wait_on_exit(1)
    except AddonNotFoundException:
        logger.error('Cannot find addon file.')
        logger.info(
            'Make sure that *.addon file is built and exists in PyHook directory.')
        _wait_on_exit(1)
    except Exception as ex:
        logger.error('Unhandled exception occurres.', ex)
        _wait_on_exit(1)


if __name__ == '__main__':
    _main()
