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
from typing import Any, Callable, Dict, List

import numpy as np

_PIPELINE_DIRS = ["./pipelines", "./PyHook/pipelines"]


class PipelinesDirNotFoundError(Exception):
    """Raised when pipelines directory does not exists."""


class FrameSizeModificationError(Exception):
    """Raised when frame shape changes during processing."""


class PipelineCallbacks:
    """Contains pipeline callbacks.

    on_frame_process (Callable[[numpy.array, int, int, int], numpy.array]): Callback for frame
        processing function. Array shape must remain unchanged after processing.
    on_load (Callable[[], None], optional): Callback for pipeline loading. Should create all
        necessary objects that will be later used in on_frame_process callback. Defaults to None.
    on_unload (Callable[[], None], optional): Callback for pipeline unloading. Should clear and
        remove all objects that are no longer used. Defaults to None.
    before_change_settings (Callable[[str, float], None], optional): Callback for settings change.
        Called right before settings modification for given key-value pair. Defaults to None.
    after_change_settings (Callable[[str, float], None], optional): Callback for settings change.
        Called right after settings modification for given key-value pair. Defaults to None.
    """

    def __init__(
        self,
        on_frame_process: Callable[[np.array, int, int, int], np.array],
        on_load: Callable[[], None] = None,
        on_unload: Callable[[], None] = None,
        before_change_settings: Callable[[str, float], None] = None,
        after_change_settings: Callable[[str, float], None] = None,
    ):
        self.on_load = on_load
        self.on_frame_process = on_frame_process
        self.on_unload = on_unload
        self.before_change_settings = before_change_settings
        self.after_change_settings = after_change_settings


class Pipeline:
    """Pipeline definition for frame processing.

    path (str): Path to pipeline file.
    name (str): Pipeline name.
    callbacks (PipelineCallbacks): The pipeline callbacks.
    version (str, optional): Pipeline version. Defaults to None.
    desc (str, optional): Pipeline description. Defaults to None.
    settings (Dict[str, List[Any]], optional): Pipeline settings variables. Defaults to None.
    mappings (Dict[str, int]): Internal mappings for settings variables.
    """

    def __init__(
        self,
        path: str,
        name: str,
        callbacks: PipelineCallbacks,
        version: str = None,
        desc: str = None,
        settings: Dict[str, List[Any]] = None,
    ):
        self.path = path
        self.file = basename(path)
        self.name = name
        self.callbacks = callbacks
        self.version = version
        self.desc = desc
        self.settings = settings
        self.mappings = {} if settings is None else {k: self._to_internal_type(v[0]) for k, v in settings.items()}

    def _to_internal_type(self, value: Any) -> int:
        """Converts given value to its internal type.

        Args:
            value (Any): Value to convert.

        Returns:
            int: Code for internal type.
        """
        if str(value) in ["True", "False"]:
            return 0
        if isinstance(value, int):
            return 1
        return 2

    def _to_value(self, key: str, value: float) -> Any:
        """Maps value for given key to its original type.

        Args:
            key (str): Variable name.
            value (float): Variable value as float.

        Returns:
            Any: Variable value as its original type.
        """
        if self.mappings[key] == 0:
            return bool(value)
        if self.mappings[key] == 1:
            return int(value)
        return value

    def change_settings(self, enabled: bool, key: str, new_value: float) -> None:
        """Changes given key-value pair and calls before_change_settings and after_change_settings callbacks.

        Args:
            enabled (bool): Flag if this pipeline is enabled.
            key (str): Variable name.
            new_value (float): New value to be set.
        """
        if enabled and self.callbacks.before_change_settings is not None:
            self.callbacks.before_change_settings(key, new_value)
        self.settings[key][0] = self._to_value(key, new_value)
        if enabled and self.callbacks.after_change_settings is not None:
            self.callbacks.after_change_settings(key, new_value)

    def load(self) -> None:
        """Calls on_load callback to initialize pipeline."""
        if self.callbacks.on_load is not None:
            self.callbacks.on_load()

    def process_frame(self, frame: np.array, width: int, height: int, frame_num: int) -> np.array:
        """Frame processing function.

        Calls on_frame_process(np.array, int, int, int) -> np.array callback from external file.

        Args:
            frame (numpy.array): The frame image as numpy array.
                Array has to be 3-D with height, width, channels as dimensions.
                Array has to contains uint8 values.
            width (int): The frame width in pixels.
            height (int): The frame height in pixels.
            frame_num (int): The frame number.

        Returns:
            numpy.array: The processed frame image as numpy array.

        Raises:
            FrameSizeModificationError: When frame shape changes during processing.
        """
        input_shape = frame.shape
        frame = self.callbacks.on_frame_process(frame, width, height, frame_num)
        output_shape = frame.shape
        if input_shape != output_shape:
            raise FrameSizeModificationError()
        return frame

    def unload(self) -> None:
        """Calls on_unload callback to destroy pipeline."""
        if self.callbacks.on_unload:
            self.callbacks.on_unload()


def _build_pipeline(module: "sys.ModuleType", name: str, path: str) -> Pipeline:
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
            "Invalid pipeline file. Missing on_frame_process(numpy.array,int,int,int)->numpy.array callback."
        )
    callbacks = PipelineCallbacks(
        on_frame_process=module.on_frame_process,
        on_load=None if not hasattr(module, "on_load") else module.on_load,
        on_unload=None if not hasattr(module, "on_unload") else module.on_unload,
        before_change_settings=None
        if not hasattr(module, "before_change_settings")
        else module.before_change_settings,
        after_change_settings=None if not hasattr(module, "after_change_settings") else module.after_change_settings,
    )
    return Pipeline(
        path=path,
        name=name if not hasattr(module, "name") else module.name,
        callbacks=callbacks,
        version="" if not hasattr(module, "version") else module.version,
        desc="" if not hasattr(module, "desc") else module.desc,
        settings=None if not hasattr(module, "settings") else module.settings,
    )


def load_pipelines(logger: logging.Logger = None) -> Dict[str, Pipeline]:
    """Loads pipelines for frame processing.

    Args:
        logger (logging.Logger, optional): Logger to display errors while loading pipeline files.
            Defaults to None.

    Returns:
        Dict[str, Pipeline]: File to pipeline map.

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
    pipeline_files = glob.glob(f"{pipeline_dir}/*.py")

    for path in pipeline_files:
        module_name = basename(path)[:-3]
        try:
            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            pipeline = _build_pipeline(module, module_name, path)
            pipelines[pipeline.file] = pipeline
            if logger is not None:
                logger.info('-- Loaded pipeline: "%s".', path)
        except Exception as ex:
            if logger is not None:
                logger.error('-- Cannot load pipeline file "%s".', path)
                logger.error("--- Error: %s", ex)
    return pipelines
