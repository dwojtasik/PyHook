"""
gui.settings for PyHook
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Settings window for PyHook
:copyright: (c) 2022 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""

import copy
import json
from os.path import exists
from typing import Any, Dict

import PySimpleGUI as sg

from gui.keys import SGKeys
from gui.style import FONT_SMALL_DEFAULT
from gui.utils import show_popup_text
from keys import SettingsKeys

# Settings file path.
_SETTINGS_PATH = "settings.json"

# PyHook settings.
_SETTINGS = {
    SettingsKeys.KEY_AUTOSAVE: 5,
    SettingsKeys.KEY_AUTOUPDATE: False,
    SettingsKeys.KEY_AUTODOWNLOAD: True,
    SettingsKeys.KEY_DOWNLOADED: [],
}

# Minimum value in seconds for autosave interval.
_AUTOSAVE_SEC_MIN = 3
# Maximum value in seconds for autosave interval.
_AUTOSAVE_SEC_MAX = 60
# Text format for slider label.
_SLIDER_TEXT_FORMAT = "Save PyHook config every < %d > seconds"


def get_settings() -> Dict[str, Any]:
    """Returns actual PyHook settings clone.

    Returns:
        Dict[str, Any]: Actual PyHook settings.
    """
    return copy.deepcopy(_SETTINGS)


def save_settings(new_settings: Dict[str, Any] = None) -> None:
    """Saves PyHook settings to file.

    Args:
        new_settings (Dict[str, Any], optional): New settings to save. Defaults to None.
    """
    if new_settings is not None:
        for key in new_settings:
            _SETTINGS[key] = new_settings[key]
    try:
        with open(_SETTINGS_PATH, "w", encoding="utf-8") as settings_file:
            json.dump(_SETTINGS, settings_file, indent=4)
    except PermissionError:
        show_popup_text(
            "Error", f"Cannot save settings to file {_SETTINGS_PATH}.\nPermission denied. Try to run PyHook as admin."
        )
    except Exception:
        show_popup_text("Error", f"Cannot save settings to file {_SETTINGS_PATH}.\nUnhandled exception occurred.")


def load_settings() -> None:
    """Loads PyHook settings from file."""
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
                elif key in (SettingsKeys.KEY_AUTOUPDATE, SettingsKeys.KEY_AUTODOWNLOAD):
                    _SETTINGS[key] = bool(value)
                elif key == SettingsKeys.KEY_DOWNLOADED:
                    if not isinstance(value, list):
                        continue
                    _SETTINGS[key] = list(value)
    else:
        save_settings()


def display_settings_window():
    """Displays settings window."""
    temp_settings = get_settings()
    window = sg.Window(
        "Settings",
        [
            [
                sg.Checkbox(
                    "Try to update app on start",
                    default=temp_settings[SettingsKeys.KEY_AUTOUPDATE],
                    enable_events=True,
                    key=SGKeys.SETTINGS_AUTOUPDATE_CHECKBOX,
                )
            ],
            [
                sg.Checkbox(
                    "Download pipeline files on start",
                    default=temp_settings[SettingsKeys.KEY_AUTODOWNLOAD],
                    enable_events=True,
                    key=SGKeys.SETTINGS_AUTODOWNLOAD_CHECKBOX,
                )
            ],
            [
                sg.Text(
                    _SLIDER_TEXT_FORMAT % temp_settings[SettingsKeys.KEY_AUTOSAVE], key=SGKeys.SETTINGS_AUTOSAVE_TEXT
                )
            ],
            [
                sg.Slider(
                    orientation="horizontal",
                    range=(_AUTOSAVE_SEC_MIN, _AUTOSAVE_SEC_MAX),
                    default_value=temp_settings[SettingsKeys.KEY_AUTOSAVE],
                    disable_number_display=True,
                    enable_events=True,
                    key=SGKeys.SETTINGS_AUTOSAVE_SLIDER,
                )
            ],
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
        finalize=True,
        location=(None, None),
    )
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
        elif event == SGKeys.SETTINGS_SAVE_BUTTON:
            save_settings(temp_settings)
            break
    window.close()
