"""
keys for PyHook
~~~~~~~~~~~~~~~~~~~~~~~
Contains keys enums
:copyright: (c) 2022 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""

import re


class SettingsKeys:
    """Settings keys.

    KEY_AUTOSAVE (str): Key for: Time in seconds after last settings change to wait until autosave.
    KEY_AUTODOWNLOAD (str): Key for: Flag if pipeline files should be automatically downloaded.
    KEY_DOWNLOADED (str): Key for: List of pipelines that have all files already downloaded.
    """

    KEY_AUTOSAVE = "autosave_settings_seconds"
    KEY_AUTODOWNLOAD = "automatic_download"
    KEY_DOWNLOADED = "downloaded"


class SGKeys:
    """Strings used by PySimpleGUI as keys.

    PROCESS_LIST (str): Key for: Process list select box.
    PROCESS_RELOAD (str): Key for: Reload process list button.
    INJECT (str): Key for: Injection button.
    INJECT_CLEAR (str): Key for: Clear injection select box button.
    INJECT_AUTO (str): Key for: Automatic injection button.
    POPUP_KEY_OK_BUTTON (str): Key for: Popup OK button.
    SESSION_LIST (str): Key for: Session list column.
    SESSION_PREFIX (str): Key for: Session button key prefix.
    SESSION_TITLE (str): Key for: Session title text.
    SESSION_KILL_BUTTON (str): Key for: Session kill button.
    SESSION_RESTART_BUTTON (str): Key for: Session restart button.
    SESSION_CLOSE_OVERVIEW_BUTTON (str): Key for: Session overview close button.
    SESSION_LOGS (str): Key for: Session logs multiline box.
    SESSION_LOGS_SCROLL_TOP_BUTTON (str): Key for: Session logs scroll top button.
    SESSION_LOGS_CLEAR_BUTTON (str): Key for: Session logs clear button.
    SESSION_LOGS_SCROLL_BOT_BUTTON (str): Key for: Session logs scroll bottom button.
    BORDER_SUFFIX (str): Key for: Border suffix.
    """

    PROCESS_LIST = "-PROCESS_LIST-"
    PROCESS_RELOAD = "-PROCESS_RELOAD-"
    INJECT = "-INJECT-"
    INJECT_CLEAR = "-INJECT_CLEAR-"
    INJECT_AUTO = "-INJECT_AUTO-"
    POPUP_KEY_OK_BUTTON = "-OK-"
    SESSION_LIST = "-SESSION_LIST-"
    SESSION_PREFIX = "-SESSION_IDX_"
    SESSION_TITLE = "-SESSION_TITLE-"
    SESSION_KILL_BUTTON = "-SESSION_KILL_BUTTON-"
    SESSION_RESTART_BUTTON = "-SESSION_RESTART_BUTTON-"
    SESSION_CLOSE_OVERVIEW_BUTTON = "-SESSION_CLOSE_OVERVIEW_BUTTON-"
    SESSION_LOGS = "-SESSION_LOGS-"
    SESSION_LOGS_SCROLL_TOP_BUTTON = "-SESSION_LOGS_SCROLL_TOP_BUTTON-"
    SESSION_LOGS_CLEAR_BUTTON = "-SESSION_LOGS_CLEAR_BUTTON-"
    SESSION_LOGS_SCROLL_BOT_BUTTON = "-SESSION_LOGS_SCROLL_BOT_BUTTON-"
    BORDER_SUFFIX = "-BORDER-"

    @staticmethod
    def get_session_key(session_idx: int) -> str:
        """Returns key for given session index.

        Args:
            session_idx (int): Session index in UI.

        Returns:
            str: String key for given session index.
        """
        return f"{SGKeys.SESSION_PREFIX}{session_idx}-"

    @staticmethod
    def get_session_idx(session_key: str) -> int:
        """Returns session index for given session key.

        Args:
            session_key (str): String key for given session.

        Returns:
            int: Session index in UI.
        """
        return int(re.findall(r"\d+", session_key)[0])
