"""
gui.settings for PyHook
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Settings window for PyHook
:copyright: (c) 2022 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""

import copy
import json
import os
import sys
import uuid
from os.path import exists, dirname
from subprocess import PIPE, Popen, call
from typing import Any, Dict, List

import PySimpleGUI as sg

from gui.style import FONT_SMALL_DEFAULT
from gui.ui_keys import SGKeys
from gui.utils import center_in_parent, show_popup_text
from keys import SettingsKeys
from win.api import CREATE_NO_WINDOW

# Settings file path.
_SETTINGS_PATH = "settings.json"

# PyHook settings.
_SETTINGS = {
    SettingsKeys.KEY_AUTOSAVE: 5,
    SettingsKeys.KEY_AUTOUPDATE: False,
    SettingsKeys.KEY_AUTODOWNLOAD: True,
    SettingsKeys.KEY_DOWNLOADED: [],
    SettingsKeys.KEY_LOCAL_PYTHON_32: "",
    SettingsKeys.KEY_LOCAL_PYTHON_64: "",
}

# Minimum value in seconds for autosave interval.
_AUTOSAVE_SEC_MIN = 3
# Maximum value in seconds for autosave interval.
_AUTOSAVE_SEC_MAX = 60
# Text format for slider label.
_SLIDER_TEXT_FORMAT = "Save PyHook config every < %d > seconds"
# Timeout in seconds for process to communicate.
_PROCESS_TIMEOUT_SEC = 2


def get_settings() -> Dict[str, Any]:
    """Returns actual PyHook settings clone.

    Returns:
        Dict[str, Any]: Actual PyHook settings.
    """
    return copy.deepcopy(_SETTINGS)


def save_settings(new_settings: Dict[str, Any] = None, parent: sg.Window = None) -> None:
    """Saves PyHook settings to file.

    Args:
        new_settings (Dict[str, Any], optional): New settings to save. Defaults to None.
        parent (sg.Window, optional): Parent window for centering. Defaults to None.
    """
    if new_settings is not None:
        for key in new_settings:
            _SETTINGS[key] = new_settings[key]
    try:
        with open(_SETTINGS_PATH, "w", encoding="utf-8") as settings_file:
            json.dump(_SETTINGS, settings_file, indent=4)
    except PermissionError:
        show_popup_text(
            "Error",
            f"Cannot save settings to file {_SETTINGS_PATH}.\nPermission denied. Try to run PyHook as admin.",
            parent=parent,
        )
    except Exception:
        show_popup_text(
            "Error", f"Cannot save settings to file {_SETTINGS_PATH}.\nUnhandled exception occurred.", parent=parent
        )


def load_settings(parent: sg.Window = None) -> None:
    """Loads PyHook settings from file.

    Args:
        parent (sg.Window, optional): Parent window for centering. Defaults to None.
    """
    if exists(_SETTINGS_PATH):
        with open(_SETTINGS_PATH, encoding="utf-8") as settings_file:
            settings = json.load(settings_file)
            for key, value in settings.items():
                if key == SettingsKeys.KEY_AUTOSAVE:
                    if not isinstance(value, int):
                        continue
                    if int(value) < 1:
                        continue
                    _SETTINGS[key] = int(value)
                elif key in (SettingsKeys.KEY_LOCAL_PYTHON_32, SettingsKeys.KEY_LOCAL_PYTHON_64):
                    _SETTINGS[key] = str(value)
                elif key in (SettingsKeys.KEY_AUTOUPDATE, SettingsKeys.KEY_AUTODOWNLOAD):
                    _SETTINGS[key] = bool(value)
                elif key == SettingsKeys.KEY_DOWNLOADED:
                    if not isinstance(value, list):
                        continue
                    _SETTINGS[key] = list(value)
    else:
        save_settings(parent=parent)


def _get_python_settings_layout(settings: Dict[str, Any]) -> List[List[sg.Column]]:
    """Returns settings fragment layout with Python executable paths.

    Args:
        settings (Dict[str, Any]): Actual PyHook settings.

    Returns:
        List[List[sg.Column]]: Layout for settings fragment with Python executable paths.
    """
    layout = []
    is_os_64_bit = sys.maxsize > 2**32
    for is_64_bit in [False, True] if is_os_64_bit else [False]:
        path = settings[SettingsKeys.KEY_LOCAL_PYTHON_64 if is_64_bit else SettingsKeys.KEY_LOCAL_PYTHON_32]
        initial_folder = os.getcwd() if len(path) == 0 or not exists(path) else dirname(path)
        tooltip = (
            "Path to Python executable that will be used in pipelines as local Python.\n"
            f"Value set in here will override LOCAL_PYTHON_{64 if is_64_bit else 32} environment variables."
        )
        layout.extend(
            [
                [
                    sg.Text(
                        f"Python {64 if is_64_bit else 32}-bit executable path:",
                        tooltip=tooltip,
                    ),
                    sg.FileBrowse(
                        size=(10, 1),
                        initial_folder=initial_folder,
                        tooltip=tooltip,
                        file_types=[("EXE Files", "*.exe")],
                        key=SGKeys.SETTINGS_PYTHON_64_BROWSE if is_64_bit else SGKeys.SETTINGS_PYTHON_32_BROWSE,
                        target=SGKeys.SETTINGS_PYTHON_64_INPUT if is_64_bit else SGKeys.SETTINGS_PYTHON_32_INPUT,
                    ),
                ],
                [
                    sg.Input(
                        default_text=path,
                        tooltip=tooltip,
                        size=(45, 1),
                        enable_events=True,
                        key=SGKeys.SETTINGS_PYTHON_64_INPUT if is_64_bit else SGKeys.SETTINGS_PYTHON_32_INPUT,
                    )
                ],
            ]
        )
    return layout


def _validate_python_paths(settings: Dict[str, Any], parent: sg.Window = None) -> bool:
    """Validates paths to local Python executables in settings.

    Path can be either empty or has to point to valid Python executable.
    If any error occurs it will be displayed as popup message.

    Args:
        settings (Dict[str, Any]): Actual PyHook settings.
        parent (sg.Window, optional): Parent window for centering. Defaults to None.

    Returns:
        bool: Flag if paths points to valid Python executables.
    """
    error_messages = []
    for bit, key in {32: SettingsKeys.KEY_LOCAL_PYTHON_32, 64: SettingsKeys.KEY_LOCAL_PYTHON_64}.items():
        test_path = settings[key]
        if len(test_path) > 0:
            if not exists(test_path):
                error_messages.append(f"Path to {bit}-bit Python executable does not exists.")
            else:
                try:
                    test_uuid = str(uuid.uuid4())
                    with Popen(
                        f"{test_path} -c \"print('{test_uuid}',end='')\"",
                        stdout=PIPE,
                        shell=False,
                        creationflags=CREATE_NO_WINDOW,
                        start_new_session=True,
                    ) as process:
                        try:
                            out, _ = process.communicate(timeout=_PROCESS_TIMEOUT_SEC)
                            resp_uuid = out.decode("utf-8")
                            if test_uuid != resp_uuid:
                                raise ValueError("Invalid response token")
                        except Exception as ex:
                            call(
                                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                                shell=False,
                                creationflags=CREATE_NO_WINDOW,
                            )
                            raise ex
                except Exception:
                    error_messages.append(
                        f"Path to {bit}-bit Python executable points to invalid Python installation."
                    )
    if len(error_messages) > 0:
        show_popup_text(
            "Error", "Cannot save settings due to following errors:\n" + "\n".join(error_messages), parent=parent
        )
        return False
    return True


def display_settings_window(parent: sg.Window = None) -> None:
    """Displays settings window.

    Args:
        parent (sg.Window, optional): Parent window for centering. Defaults to None.
    """
    temp_settings = get_settings()
    window = sg.Window(
        "Settings",
        [
            [
                sg.Checkbox(
                    "Try to update app on start",
                    default=temp_settings[SettingsKeys.KEY_AUTOUPDATE],
                    tooltip="On application start checks if new PyHook version is available and asks for update.",
                    enable_events=True,
                    key=SGKeys.SETTINGS_AUTOUPDATE_CHECKBOX,
                )
            ],
            [
                sg.Checkbox(
                    "Download pipeline files on start",
                    default=temp_settings[SettingsKeys.KEY_AUTODOWNLOAD],
                    tooltip=(
                        "On application start verifies pipelines files to download.\n"
                        'Files to download are read from "download.txt" files in pipeline directories.\n'
                        "If any file is missing or incomplete it will be re-downloaded."
                    ),
                    enable_events=True,
                    key=SGKeys.SETTINGS_AUTODOWNLOAD_CHECKBOX,
                )
            ],
            [
                sg.Text(
                    _SLIDER_TEXT_FORMAT % temp_settings[SettingsKeys.KEY_AUTOSAVE],
                    tooltip=(
                        "Describes how ofter pipeline's config should be saved.\n"
                        'Files to download are read from "download.txt" files in pipeline directories.\n'
                        "This is configuration used by PyHook addon and displayed in ImGui."
                    ),
                    key=SGKeys.SETTINGS_AUTOSAVE_TEXT,
                )
            ],
            [
                sg.Slider(
                    orientation="horizontal",
                    range=(_AUTOSAVE_SEC_MIN, _AUTOSAVE_SEC_MAX),
                    default_value=temp_settings[SettingsKeys.KEY_AUTOSAVE],
                    tooltip=(
                        "Describes how ofter pipeline's config should be saved.\n"
                        "This is configuration used by PyHook addon and displayed in ImGui."
                    ),
                    disable_number_display=True,
                    enable_events=True,
                    key=SGKeys.SETTINGS_AUTOSAVE_SLIDER,
                )
            ],
            *_get_python_settings_layout(temp_settings),
            [
                sg.Button("Save", size=(10, 1), pad=((5, 5), (10, 5)), key=SGKeys.SETTINGS_SAVE_BUTTON),
                sg.Button("Cancel", size=(10, 1), pad=((5, 5), (10, 5)), key=SGKeys.SETTINGS_CANCEL_BUTTON),
            ],
            [sg.Image(size=(350, 0), pad=(0, 0))],
        ],
        font=FONT_SMALL_DEFAULT,
        element_justification="center",
        disable_minimize=True,
        modal=True,
        keep_on_top=True,
    )
    center_in_parent(window, parent)
    while True:
        event, values = window.read()
        if event in (sg.WIN_CLOSED, SGKeys.EXIT, SGKeys.SETTINGS_CANCEL_BUTTON):
            break
        if event == SGKeys.SETTINGS_AUTOUPDATE_CHECKBOX:
            temp_settings[SettingsKeys.KEY_AUTOUPDATE] = values[event]
        elif event == SGKeys.SETTINGS_AUTODOWNLOAD_CHECKBOX:
            temp_settings[SettingsKeys.KEY_AUTODOWNLOAD] = values[event]
        elif event == SGKeys.SETTINGS_AUTOSAVE_SLIDER:
            new_value = int(values[event])
            window[SGKeys.SETTINGS_AUTOSAVE_TEXT].update(value=_SLIDER_TEXT_FORMAT % new_value)
            temp_settings[SettingsKeys.KEY_AUTOSAVE] = new_value
        elif event == SGKeys.SETTINGS_PYTHON_32_INPUT:
            temp_settings[SettingsKeys.KEY_LOCAL_PYTHON_32] = values[event]
        elif event == SGKeys.SETTINGS_PYTHON_64_INPUT:
            temp_settings[SettingsKeys.KEY_LOCAL_PYTHON_64] = values[event]
        elif event == SGKeys.SETTINGS_SAVE_BUTTON:
            if not _validate_python_paths(temp_settings, window):
                continue
            save_settings(temp_settings, window)
            break
    window.close()
