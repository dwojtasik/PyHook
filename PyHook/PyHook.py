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

from dll_utils import (AddonNotFoundException, ReShadeNotFoundException,
                       get_reshade_addon_handler)
from mem_utils import FRAME_ARRAY, SIZE_ARRAY, MemoryManager
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
    exit(code)


def _decode_frame(data) -> np.array:
    """Decodes frame from shared data as image in numpy array format.

    Args:
        data (SharedData): The shared data to read frame C array.

    Returns:
        numpy.array: The frame image as numpy array.
    """
    arr = np.ctypeslib.as_array(data.frame)[:3 * data.width * data.height]
    return arr.reshape((data.height, data.width, 3))


def _encode_frame(data, frame) -> None:
    """Encodes numpy image array as C array and stores it in shared data.

    Args:
        data (SharedData): The shared data to store frame C array.
        array (numpy.array): The frame image as numpy array.
    """
    arr = np.zeros(SIZE_ARRAY, dtype=np.uint8)
    arr[:3 * data.width * data.height] = frame.ravel()
    data.frame = FRAME_ARRAY.from_buffer(arr)


def _process_frame(frame: np.array, width: int, height: int, frame_num: int) -> np.array:
    """Frame processing function.

    NOTE: Dummy processing for now, to test if everything works as expected.
          Simply set red channel to 255 for each pixel.

    Args:
        frame (numpy.array): The frame image as numpy array.
        width (int): The frame width in pixels.
        height (int): The frame height in pixels.
        frame_num (int): The frame number.

    Returns:
        numpy.array: The processed frame image as numpy array.
    """
    frame[:, :, 0] = 255
    return frame


def _main():
    """Scipt entrypoint"""
    try:
        logger = _get_logger()
        addon_handler = get_reshade_addon_handler()
        memory_manager = MemoryManager(addon_handler.pid)
        logger.info(f'Detected process: {addon_handler.get_info()}')
        logger.info(
            f'- Started addon injection for {addon_handler.addon_path}...')
        try:
            addon_handler.inject_addon()
            logger.info(f'-- Addon injected!')
        except Exception as ex:
            logger.error(f'-- Cannot inject addon into given process. {ex}')
            exit(1)
        logger.info(f'-- Started processing...')
        while True:
            memory_manager.wait()
            data = memory_manager.read_shared_data()
            frame = _decode_frame(data)
            frame = _process_frame(
                frame, data.width, data.height, data.frame_count)
            _encode_frame(data, frame)
            memory_manager.unlock()
    except AddonNotFoundException:
        logger.error('Cannot find addon file.')
        logger.info(
            'Make sure that *.addon file is built and exists in PyHook directory.')
        _wait_on_exit(1)
    except ReShadeNotFoundException:
        logger.error('Cannot find any active process with ReShade loaded.')
        if not is_started_as_admin():
            logger.info('Try to run this program as administrator.')
        _wait_on_exit(1)
    except Exception as ex:
        logger.error('Unhandled exception occurres.', ex)
        _wait_on_exit(1)

if __name__ == '__main__':
    _main()
