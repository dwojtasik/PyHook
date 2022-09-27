"""
pipeline for PyHook
~~~~~~~~~~~~~~~~~~~~~~
PyHook pipeline definition
:copyright: (c) 2022 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""

import glob
import importlib.util
import json
import logging
import re
import sys
from dataclasses import dataclass
from os.path import abspath, basename, exists, isdir
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

from downloader import download_file
from keys import SettingsKeys

# Name of settings file.
_SETTINGS_FILE = "pyhook.json"
# Search paths (in priority order) for pipelines directory.
_PIPELINE_DIRS = ["./pipelines", "./PyHook/pipelines"]
# Regex for combo variable detection.
_COMBO_TAG_REGEX = re.compile(r"^%COMBO\[(.*?,)*.*?\].*$", re.MULTILINE)


class PipelinesDirNotFoundError(Exception):
    """Raised when pipelines directory does not exists."""


class FrameProcessingError(Exception):
    """Raised when any error was raised during frame processing.

    message (str): Error message to display.
    exception (Exception): The cause of exception in pipeline.
    """

    def __init__(self, message: str, exception: Exception):
        self.message = message
        self.exception = exception


class FrameSizeModificationError(Exception):
    """Raised when frame shape changes during processing."""


class PipelineCallbacks:
    """Contains pipeline callbacks.

    Pipeline has to implement on_frame_process or on_frame_process_stage callbacks
    based on multistage value.

    If multistage == 1: on_frame_process callback is required.
    If multistage > 1: on_frame_process_stage callback is required.

    on_frame_process (Callable[[numpy.array, int, int, int], numpy.array]): Callback for frame
        processing function. Array shape must remain unchanged after processing.
    on_frame_process_stage (Callable[[numpy.array, int, int, int, int], numpy.array]): Callback for frame
        processing function. Array shape can be changed during processing. Must implement multiple stages.
        The last stage must restore array shape.
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
        on_frame_process: Callable[[np.array, int, int, int], np.array] = None,
        on_frame_process_stage: Callable[[np.array, int, int, int, int], np.array] = None,
        on_load: Callable[[], None] = None,
        on_unload: Callable[[], None] = None,
        before_change_settings: Callable[[str, float], None] = None,
        after_change_settings: Callable[[str, float], None] = None,
    ):
        self.on_load = on_load
        self.on_frame_process = on_frame_process
        self.on_frame_process_stage = on_frame_process_stage
        self.on_unload = on_unload
        self.before_change_settings = before_change_settings
        self.after_change_settings = after_change_settings


class Pipeline:
    """Pipeline definition for frame processing.

    path (str): Path to pipeline file.
    name (str): Pipeline name.
    callbacks (PipelineCallbacks): The pipeline callbacks.
    multistage (int): Number of pipeline passes per frame processing.
    version (str, optional): Pipeline version. Defaults to None.
    desc (str, optional): Pipeline description. Defaults to None.
    settings (Dict[str, List[Any]], optional): Pipeline settings variables. Defaults to None.
    mappings (Dict[str, int]): Internal mappings for settings variables.
    """

    def __init__(
        self,
        path: str,
        name: str,
        multistage: int,
        callbacks: PipelineCallbacks,
        version: str = None,
        desc: str = None,
        settings: Dict[str, List[Any]] = None,
    ):
        self.path = path
        self.file = basename(path)
        self.name = name
        self.multistage = multistage
        self.callbacks = callbacks
        self.version = version
        self.desc = desc
        self.settings = settings
        self.mappings = (
            {} if settings is None else {k: self._to_internal_type(v[0], v[4]) for k, v in settings.items()}
        )

    def _to_internal_type(self, value: Any, tooltip: str) -> int:
        """Converts given value to its internal type.

        Args:
            value (Any): Value to convert.
            tooltip (str): Tooltip to combo box detection.

        Returns:
            int: Code for internal type.
                0 - bool
                1 - int
                2 - float
                3 - int, displayed as combo box selection
        """
        if str(value) in ["True", "False"]:
            return 0
        if isinstance(value, int):
            if _COMBO_TAG_REGEX.match(tooltip):
                return 3
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
        if self.mappings[key] == 1 or self.mappings[key] == 3:
            return int(value)
        step = str(self.settings[key][3])
        if "." in step:
            value = round(value, len(step.rsplit(".", maxsplit=1)[-1]))
        return value

    def set_initial_value(self, key: str, value: Any) -> None:
        """Sets initial value for given key.

        Args:
            key (str): Variable name.
            value (Any): Variable value.
        """
        if self.mappings[key] == 0:
            self.settings[key][0] = bool(value)
            return
        min_val = self.settings[key][1]
        max_val = self.settings[key][2]
        step = self.settings[key][3]
        if self.mappings[key] == 1 or self.mappings[key] == 3:
            self.settings[key][0] = int(max(min_val, min(max_val, round(int(value) / step) * step)))
            return
        precision = None
        str_step = str(step)
        if "." in str_step:
            precision = len(str_step.rsplit(".", maxsplit=1)[-1])
        self.settings[key][0] = max(min_val, min(max_val, round(round(float(value) / step) * step, precision)))

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

    def process_frame(self, frame: np.array, width: int, height: int, frame_num: int, stage: int = None) -> np.array:
        """Frame processing function.

        Calls on_frame_process(np.array, int, int, int) -> np.array callback from external file.

        Args:
            frame (numpy.array): The frame image as numpy array.
                Array has to be 3-D with height, width, channels as dimensions.
                Array has to contains uint8 values.
            width (int): The frame width in pixels.
            height (int): The frame height in pixels.
            frame_num (int): The frame number.
            stage (int, optional): The pipeline stage (pass) number.

        Returns:
            numpy.array: The processed frame image as numpy array.

        Raises:
            FrameProcessingError: When any error was raised during frame processing.
            FrameSizeModificationError: When frame shape changes during processing for not multistage pipeline.
        """
        try:
            if stage is None:
                input_shape = frame.shape
                frame = self.callbacks.on_frame_process(frame, width, height, frame_num)
                output_shape = frame.shape
                if input_shape != output_shape:
                    raise FrameSizeModificationError(f'pipeline="{self.file}", frame={frame_num}')
            else:
                frame = self.callbacks.on_frame_process_stage(frame, width, height, frame_num, stage)
            return frame
        except Exception as ex:
            if isinstance(ex, FrameSizeModificationError):
                raise ex
            raise FrameProcessingError(
                "Unexpected error during frame processing: "
                f'Pipeline="{self.file}", stage={stage}, frame={frame_num}:',
                ex,
            ) from ex

    def unload(self) -> None:
        """Calls on_unload callback to destroy pipeline."""
        if self.callbacks.on_unload:
            self.callbacks.on_unload()


@dataclass
class PipelineRuntimeData:
    """Holds pipeline runtime informations.

    pipeline_order (List[str]): Order of the pipeline to process.
    active_pipelines (List[str]): List of active pipelines.
    to_unload (List[str]): List of pipelines to unload.
    to_load (List[str]): List of pipelines to load.
    changes (Dict[str, Dict[str, float]]): Pipelines settings changes.
        Settings for update are stored in dictionary, where key is pipeline file and value is
        dictionary of modified key-value pairs.
    """

    pipeline_order: List[str]
    active_pipelines: List[str]
    to_unload: List[str]
    to_load: List[str]
    changes: Dict[str, Dict[str, float]]


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
    multistage = 1 if not hasattr(module, "multistage") else module.multistage
    if multistage > 1 and not hasattr(module, "on_frame_process_stage"):
        raise ValueError(
            "Invalid pipeline file. Missing required callback for multistage pipeline: \
                on_frame_process_stage(numpy.array,int,int,int,int)->numpy.array."
        )
    if multistage == 1 and not hasattr(module, "on_frame_process"):
        raise ValueError(
            "Invalid pipeline file. Missing required callback: on_frame_process(numpy.array,int,int,int)->numpy.array."
        )
    callbacks = PipelineCallbacks(
        on_frame_process=None if not hasattr(module, "on_frame_process") else module.on_frame_process,
        on_frame_process_stage=None
        if not hasattr(module, "on_frame_process_stage")
        else module.on_frame_process_stage,
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
        multistage=multistage,
        callbacks=callbacks,
        version="" if not hasattr(module, "version") else module.version,
        desc="" if not hasattr(module, "desc") else module.desc,
        settings=None if not hasattr(module, "settings") else module.settings,
    )


def _download_files(pipeline_dir: str, pipeline_file: str, logger: logging.Logger = None) -> None:
    """Downloads files if download.txt file exists in pipeline data directory.

    Args:
        pipeline_dir (str): The pipelines directory.
        pipeline_file (str): The pipeline filename.
        logger (logging.Logger, optional): Logger to display informations. Defaults to None.
    """
    pipeline_data_dir = f"{pipeline_dir}\\{pipeline_file[:-3]}"
    download_list_file = f"{pipeline_data_dir}\\download.txt"
    if exists(download_list_file):
        try:
            with open(download_list_file, encoding="utf-8") as d_file:
                urls = [url for url in d_file.read().split("\n") if len(url) > 1 and not url.startswith("#")]
                for url in urls:
                    download_file(url, pipeline_data_dir)
        except Exception as ex:
            if logger is not None:
                logger.info('--- Cannot read / download pipeline files: "%s"', ex)


def load_pipelines(settings: Dict[str, Any], logger: logging.Logger = None) -> Tuple[Dict[str, Pipeline], bool]:
    """Loads pipelines for frame processing.

    Args:
        settings (Dict[str, Any]): PyHook settings json.
        logger (logging.Logger, optional): Logger to display errors while loading pipeline files.
            Defaults to None.

    Returns:
        Tuple[Dict[str, Pipeline], bool]: File to pipeline map and flag if settings were changed.

    Raises:
        PipelinesDirNotFoundError: When pipelines directory does not exists.
    """
    has_settings_change = False
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
            if logger is not None:
                logger.info('-- Loading pipeline: "%s".', path)
            spec = importlib.util.spec_from_file_location(module_name, path)
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            pipeline = _build_pipeline(module, module_name, path)
            pipelines[pipeline.file] = pipeline
            if settings[SettingsKeys.KEY_AUTODOWNLOAD] and pipeline.file not in settings[SettingsKeys.KEY_DOWNLOADED]:
                _download_files(pipeline_dir, pipeline.file, logger)
                settings[SettingsKeys.KEY_DOWNLOADED].append(pipeline.file)
                has_settings_change = True
        except Exception as ex:
            if logger is not None:
                logger.error('-- Cannot load pipeline file "%s".', path)
                logger.error("--- Error: %s", ex)
    return pipelines, has_settings_change


def save_settings(
    pipelines: Dict[str, Pipeline], order: List[str], active: List[str], dir_path: str, logger: logging.Logger = None
) -> None:
    """Saves pipelines settings to file.

    Args:
        pipelines (Dict[str, Pipeline]): Loaded pipelines map.
        order (List[str]): Order of the pipeline to process.
        active (List[str]): List of active pipelines.
        dir_path (str): The directory path to save settings JSON file.
        logger (logging.Logger, optional): Logger for error display. Defaults to None.
    """
    settings = {}
    settings["order"] = order
    settings["active"] = active
    for p_file, pipeline in pipelines.items():
        if pipeline.settings is not None:
            settings[p_file] = {}
            for key, var_list in pipeline.settings.items():
                settings[p_file][key] = var_list[0]
    try:
        with open(f"{dir_path}\\{_SETTINGS_FILE}", "w", encoding="utf-8") as settings_file:
            json.dump(settings, settings_file, indent=4)
    except PermissionError:
        if logger is not None:
            logger.info("-- Error: Cannot save %s. Permission denied. Try to run PyHook as admin.", _SETTINGS_FILE)
    except Exception as ex:
        if logger is not None:
            logger.error("-- Error: Cannot save %s. Unhandled exception occurres.", _SETTINGS_FILE, exc_info=ex)


def load_settings(pipelines: Dict[str, Pipeline], dir_path: str) -> Tuple[PipelineRuntimeData, bool]:
    """Loads pipelines settings from file.

    Args:
        pipelines (Dict[str, Pipeline]): Loaded pipelines map.
        dir_path (str): The directory path to load settings JSON file.

    Returns:
        Tuple[PipelineRuntimeData, bool]: The pipeline runtime data and flag is data was read from file.
    """
    settings_path = f"{dir_path}\\{_SETTINGS_FILE}"
    if exists(settings_path):
        with open(settings_path, encoding="utf-8") as settings_file:
            settings = json.load(settings_file)
            for p_file, p_settings in settings.items():
                if p_file in ["order", "active"]:
                    continue
                if p_file in pipelines and pipelines[p_file].settings is not None:
                    for key, value in p_settings.items():
                        if key in pipelines[p_file].settings:
                            pipelines[p_file].set_initial_value(key, value)
            total_order = []
            for _, pipeline in pipelines.items():
                total_order.extend([pipeline.file] * pipeline.multistage)
            order = []
            for p_file in settings["order"]:
                if p_file in total_order:
                    order.append(total_order.pop(total_order.index(p_file)))
            order.extend(total_order)
            active = [p for p in settings["active"] if p in pipelines]
            return PipelineRuntimeData(order, active, [], active, {}), True
    return PipelineRuntimeData(list(pipelines.keys()), [], [], [], {}), False
