"""
keys for PyHook
~~~~~~~~~~~~~~~~~~~~~~~
Contains keys enums
:copyright: (c) 2022 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""


class SettingsKeys:
    """Settings keys.

    KEY_AUTOSAVE (str): Key for: Time in seconds after last settings change to wait until autosave.
    KEY_AUTOUPDATE (str): Key for: Flag if PyHook should be automatically updated.
    KEY_AUTODOWNLOAD (str): Key for: Flag if pipeline files should be automatically downloaded.
    KEY_DOWNLOADED (str): Key for: List of pipelines that have all files already downloaded.
    KEY_LOCAL_PYTHON_32 (str): Key for: Path to local 32-bit Python executable.
    KEY_LOCAL_PYTHON_64 (str): Key for: Path to local 64-bit Python executable.
    """

    KEY_AUTOSAVE = "autosave_settings_seconds"
    KEY_AUTOUPDATE = "automatic_update"
    KEY_AUTODOWNLOAD = "automatic_download"
    KEY_DOWNLOADED = "downloaded"
    KEY_LOCAL_PYTHON_32 = "local_python_32"
    KEY_LOCAL_PYTHON_64 = "local_python_64"
