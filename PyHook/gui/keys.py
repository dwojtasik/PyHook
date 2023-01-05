"""
gui.keys for PyHook
~~~~~~~~~~~~~~~~~~~~~~~
Contains UI keys enums
:copyright: (c) 2022 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""

import re


class SGKeys:
    """Strings used by PySimpleGUI as keys.

    PROCESS_LIST (str): Key for: Process list select box.
    PROCESS_RELOAD (str): Key for: Reload process list button.
    INJECT (str): Key for: Injection button.
    INJECT_CLEAR (str): Key for: Clear injection select box button.
    INJECT_AUTO (str): Key for: Automatic injection button.
    POPUP_KEY_OK_BUTTON (str): Key for: Popup OK button.
    POPUP_KEY_CANCEL_BUTTON (str): Key for: Popup cancel button.
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
    MENU_SETTINGS_OPTION (str): Key for: Menu settings option.
    MENU_PIPELINE_FORCE_DOWNLOAD_OPTION (str): Key for: Menu pipeline force download option.
    MENU_UPDATE_OPTION (str): Key for: Menu update option.
    MENU_ABOUT_OPTION (str): Key for: Menu about option.
    SETTINGS_AUTOUPDATE_CHECKBOX (str): Key for: Settings autoupdate checkbox.
    SETTINGS_AUTODOWNLOAD_CHECKBOX (str): Key for: Settings autodownload checkbox.
    SETTINGS_AUTOSAVE_SLIDER (str): Key for: Settings autosave slider.
    SETTINGS_AUTOSAVE_TEXT (str): Key for: Settings autosave text.
    SETTINGS_PYTHON_32_INPUT (str): Key for: Settings Python 32-bit path input.
    SETTINGS_PYTHON_32_BROWSE (str): Key for: Settings Python 32-bit path browse.
    SETTINGS_PYTHON_64_INPUT (str): Key for: Settings Python 64-bit path input.
    SETTINGS_PYTHON_64_BROWSE (str): Key for: Settings Python 64-bit path browse.
    SETTINGS_SAVE_BUTTON (str): Key for: Settings save button.
    SETTINGS_CANCEL_BUTTON (str): Key for: Settings cancel button.
    UPDATE_RELEASE_LINK (str): Key for: Update release link.
    ABOUT_GITHUB_BUTTON (str): Key for: About GitHub button.
    DOWNLOAD_TEXT (str): Key for: Download text.
    DOWNLOAD_VERIFY_TEXT (str): Key for: Download verify text.
    DOWNLOAD_FILE_VERIFY_TEXT (str): Key for: Download file verify text.
    DOWNLOAD_PROGRESS_BAR (str): Key for: Download progress bar.
    DOWNLOAD_PROGRESS_PLACEHOLDER_TEXT (str): Key for: Download progress placeholder text.
    DOWNLOAD_CANCEL_EVENT (str): Key for: Download cancel event.
    EXIT (str): Key for: Application exit.
    """

    PROCESS_LIST = "-PROCESS_LIST-"
    PROCESS_RELOAD = "-PROCESS_RELOAD-"
    INJECT = "-INJECT-"
    INJECT_CLEAR = "-INJECT_CLEAR-"
    INJECT_AUTO = "-INJECT_AUTO-"
    POPUP_KEY_OK_BUTTON = "-OK-"
    POPUP_KEY_CANCEL_BUTTON = "-CANCEL-"
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
    MENU_SETTINGS_OPTION = "Settings"
    MENU_PIPELINE_FORCE_DOWNLOAD_OPTION = "Force re-download"
    MENU_UPDATE_OPTION = "Update"
    MENU_ABOUT_OPTION = "About"
    SETTINGS_AUTOUPDATE_CHECKBOX = "-SETTINGS_AUTOUPDATE_CHECKBOX-"
    SETTINGS_AUTODOWNLOAD_CHECKBOX = "-SETTINGS_AUTODOWNLOAD_CHECKBOX-"
    SETTINGS_AUTOSAVE_SLIDER = "-SETTINGS_AUTOSAVE_SLIDER-"
    SETTINGS_AUTOSAVE_TEXT = "-SETTINGS_AUTOSAVE_TEXT-"
    SETTINGS_PYTHON_32_INPUT = "-SETTINGS_PYTHON_32_INPUT-"
    SETTINGS_PYTHON_32_BROWSE = "-SETTINGS_PYTHON_32_BROWSE-"
    SETTINGS_PYTHON_64_INPUT = "-SETTINGS_PYTHON_64_INPUT-"
    SETTINGS_PYTHON_64_BROWSE = "-SETTINGS_PYTHON_64_BROWSE-"
    SETTINGS_SAVE_BUTTON = "-SETTINGS_SAVE_BUTTON-"
    SETTINGS_CANCEL_BUTTON = "-SETTINGS_CANCEL_BUTTON-"
    UPDATE_RELEASE_LINK = "-UPDATE_RELEASE_LINK-"
    ABOUT_GITHUB_BUTTON = "-GITHUB-"
    DOWNLOAD_TEXT = "-DOWNLOAD_TEXT-"
    DOWNLOAD_VERIFY_TEXT = "-DOWNLOAD_VERIFY_TEXT-"
    DOWNLOAD_FILE_VERIFY_TEXT = "-DOWNLOAD_FILE_VERIFY_TEXT-"
    DOWNLOAD_PROGRESS_BAR = "-DOWNLOAD_PROGRESS_BAR-"
    DOWNLOAD_PROGRESS_PLACEHOLDER_TEXT = "-DOWNLOAD_PROGRESS_PLACEHOLDER_TEXT-"
    DOWNLOAD_CANCEL_EVENT = "-DOWNLOAD_CANCEL_EVENT-"
    EXIT = "Exit"

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


class TKKeys:
    """Strings used by Tkinter as keys.

    CMD_BIND (str): Key for: Tkinter bind command.
    EVENT_KEY_PRESS (str): Key for: Tkinter KeyPress event.
    EVENT_KEY_RELEASE (str): Key for: Tkinter KeyRelease event.
    EVENT_MOUSE_BUTTON_CLICK (str): Key for: Tkinter Button-1 event.
    BUTTON_BACKSPACE (str): Key for: Tkinter backspace button code.
    BUTTON_DELETE (str): Key for: Tkinter delete button code.
    INSERT (str): Key for: Tkinter writing cursor code.
    POPDOWN_WINDOW (str): Key for: Tkinter ComboBox PopdownWindow element key.
    POPDOWN_WINDOW_SUFFIX (str): Key for: Tkinter ComboBox PopdownWindow element suffix key.
    """

    CMD_BIND = "bind"
    EVENT_KEY_PRESS = "<KeyPress>"
    EVENT_KEY_RELEASE = "<KeyRelease>"
    EVENT_MOUSE_BUTTON_CLICK = "<Button-1>"
    BUTTON_BACKSPACE = "BackSpace"
    BUTTON_DELETE = "Delete"
    INSERT = "insert"
    POPDOWN_WINDOW = "ttk::combobox::PopdownWindow"
    POPDOWN_WINDOW_SUFFIX = ".f.l"
