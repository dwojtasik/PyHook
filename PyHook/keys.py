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
    KEY_AUTODOWNLOAD (str): Key for: Flag if pipeline files should be automatically downloaded.
    KEY_DOWNLOADED (str): Key for: List of pipelines that have all files already downloaded.
    """

    KEY_AUTOSAVE = "autosave_settings_seconds"
    KEY_AUTODOWNLOAD = "automatic_download"
    KEY_DOWNLOADED = "downloaded"
