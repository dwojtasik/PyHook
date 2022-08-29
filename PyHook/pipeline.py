"""
pipeline for PyHook
~~~~~~~~~~~~~~~~~~~~~~
PyHook pipeline definition
:copyright: (c) 2022 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""

import glob
import importlib.util
import logging
import sys
from os.path import abspath, basename, isdir
from typing import Callable, Dict

import numpy as np

_PIPELINE_DIRS = [
    "./pipelines",
    "./PyHook/pipelines"
]


class PipelinesDirNotFoundError(Exception):
    """Raised when pipelines directory does not exists."""
    pass


class FrameSizeModificationError(Exception):
    """Raised when frame shape changes during processing."""
    pass


class Pipeline:
    """Pipeline definition for frame processing.

    path (str): Path to pipeline file.
    name (str): Unique pipeline name.
    on_frame_process (Callable[[numpy.array, int, int, int], numpy.array]): Callback for frame
        processing function. Array shape must remain unchanged after processing.
    on_load (Callable[[], None], optional): Callback for pipeline loading. Should create all
        necessary objects that will be later used in on_frame_process callback.
    on_unload (Callable[[], None], optional): Callback for pipeline unloading. Should clear and
        remove all objects that are no longer used.
    """

    def __init__(
        self,
        path: str,
        name: str,
        on_frame_process: Callable[[np.array, int, int, int], np.array],
        on_load: Callable[[], None] = None,
        on_unload: Callable[[], None] = None
    ):
        self.path = path
        self.name = name
        self.on_load = on_load
        self.on_frame_process = on_frame_process
        self.on_unload = on_unload

    def load(self) -> None:
        """Calls on_load callback to initialize pipeline"""
        if self.on_load is not None:
            self.on_load()

    def process_frame(self, frame: np.array, width: int, height: int, frame_num: int) -> np.array:
        """Frame processing function.

        Calls on_frame_process(np.array, int, int, int) -> np.array callback from external file.

        Args:
            frame (numpy.array): The frame image as numpy array.
            width (int): The frame width in pixels.
            height (int): The frame height in pixels.
            frame_num (int): The frame number.

        Returns:
            numpy.array: The processed frame image as numpy array.

        Raises:
            FrameSizeModificationError: When frame shape changes during processing.
        """
        input_shape = frame.shape
        frame = self.on_frame_process(frame, width, height, frame_num)
        output_shape = frame.shape
        if input_shape != output_shape:
            raise FrameSizeModificationError()
        return frame

    def unload(self) -> None:
        """Calls on_unload callback to destroy pipeline"""
        if self.on_unload:
            self.on_unload()


def _build_pipeline(module: 'sys.ModuleType', name: str, path: str) -> Pipeline:
    """Builds pipeline object.

    Args:
        module (sys.ModuleType): The loaded module file.
        name (str): Fallback name for the pipeline, used when name is not defined inside file.
        path (str): Absolute path to the pipeline file on disk.

    Returns:
        Pipeline: The pipeline object.

    Raises:
        ValueError: When given module file was invalid pipeline.
    """
    if not hasattr(module, "on_frame_process"):
        raise ValueError(
            "Invalid pipeline file. Missing on_frame_process(numpy.array,int,int,int)->numpy.array callback.")
    if hasattr(module, "name"):
        name = module.name
    return Pipeline(
        path, name, module.on_frame_process,
        on_load=None if not hasattr(module, "on_load") else module.on_load,
        on_unload=None if not hasattr(
            module, "on_unload") else module.on_unload
    )


def load_pipelines(logger: logging.Logger = None) -> Dict[str, Pipeline]:
    """Loads pipelines for frame processing.

    Args:
        logger (logging.Logger, optional): Logger to display errors while loading pipeline files.

    Returns:
        Dict[str, Pipeline]: Name to pipeline map.

    Raises:
        PipelinesDirNotFoundError: When pipelines directory does not exists.
    """
    pipeline_dir = None
    for path in _PIPELINE_DIRS:
        if isdir(path):
            pipeline_dir = abspath(path)
    if pipeline_dir is None:
        raise PipelinesDirNotFoundError()

    pipelines = {}
    pipeline_files = glob.glob(f'{pipeline_dir}/*.py')

    for path in pipeline_files:
        module_name = basename(path)[:-3]
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            pipeline = _build_pipeline(module, module_name, path)
            pipelines[pipeline.name] = pipeline
        except Exception as ex:
            if logger is not None:
                logger.error(f'Cannot load pipeline file "{path}"', ex)
    return pipelines
