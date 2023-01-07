"""
keys for PyHook
~~~~~~~~~~~~~~~~~~~~~~~
Contains keys enums
:copyright: (c) 2023 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""

from typing import Tuple


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


class TimingsKeys:
    """Timings keys.

    TIMINGS_TIMESTAMP (str): Key for: Last timings timestamp.
    RESHADE_PROCESSING (str): Key for: Reshade processing process.
    DATA_SYNC (str): Key for: Data synchronization process.
    FRAME_DECODING (str): Key for: Frame decoding process.
    FRAME_ENCODING (str): Key for: Frame encoding process.
    """

    TIMINGS_TIMESTAMP = "Timings timestamp"
    RESHADE_PROCESSING = "Reshade processing"
    DATA_SYNC = "Data synchronization"
    FRAME_DECODING = "Frame decoding"
    FRAME_ENCODING = "Frame encoding"

    @staticmethod
    def with_idx(key: str, idx: int) -> str:
        """Returns key with index to keep ordering.

        Args:
            key (str): Timings key.
            idx (int): Timings index.

        Returns:
            str: Key with index.
        """
        return f"{idx};{key}"

    @staticmethod
    def to_idx_and_key(key_with_idx: str) -> Tuple[int, str]:
        """Returns index and key from connected key.

        Args:
            key_with_idx (str): Key with index.

        Returns:
            Tuple[int, str]: Index and key.
        """
        idx, key = key_with_idx.split(";", maxsplit=1)
        return int(idx), key

    @staticmethod
    def to_timings_key(name: str, stage: int | None, stages: int) -> str:
        """Builds timings key for pipeline data.

        Args:
            name (str): Pipeline name.
            stage (int | None): Actual stage of processing.
            stages (int): Count of stages for given pipeline.

        Returns:
            str: Timings key for pipeline data.
        """
        if stage is None:
            return name
        return f"{name} [pass {stage}/{stages}]"
