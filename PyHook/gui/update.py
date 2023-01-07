"""
gui.update for PyHook
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Update window for PyHook
:copyright: (c) 2023 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""

import os
import sys
import webbrowser
import zipfile
from typing import Tuple

import PySimpleGUI as sg
import requests

from _version import __version__
from gui.keys import SGKeys
from gui.settings import get_settings
from gui.style import COLOR_TEXT_URL, FONT_SMALL_DEFAULT
from gui.utils import EventCallback, center_in_parent, show_popup, show_popup_text
from keys import SettingsKeys
from pipeline import get_pipeline_directory
from utils.common import is_frozen_bundle
from utils.downloader import download_file

_GITHUB_URL = "https://github.com/dwojtasik/PyHook"
_TAG_URL = f"{_GITHUB_URL}/releases/tag/"
_LATEST_TAG_URL = f"{_GITHUB_URL}/releases/latest"
_DOWNLOAD_URL = f"{_GITHUB_URL}/releases/download"


def _get_release_url(tag: str) -> str:
    """Returns release URL.

    Args:
        tag (str): Version tag.

    Returns:
        str: The release URL.
    """
    arch = "win_amd64" if sys.maxsize > 2**32 else "win32"
    return f"{_DOWNLOAD_URL}/{tag}/PyHook-{tag}-{arch}.zip"


def _get_latest_version_tag() -> str | None:
    """Returns latest release tag from PyHook GitHub if possible, None otherwise.

    Returns:
        str | None: Latest release tag or None.
    """
    try:
        response = requests.head(_LATEST_TAG_URL, allow_redirects=True, timeout=5)
        response.raise_for_status()
        return response.url[len(_TAG_URL) :]
    except Exception:
        return None


def _ask_for_restart(parent: sg.Window = None) -> bool:
    """Asks for application restart and starts updated version.

    Args:
        parent (sg.Window, optional): Parent window for centering. Defaults to None.

    Returns:
        bool: Flag if app should be restarted for update.
    """
    return show_popup_text(
        "Restart to update", "Update is ready. Do you want to restart PyHook now?", "Yes", True, "No", parent=parent
    )


def _process_update(tag: str, parent: sg.Window = None) -> str | None:
    """Processes update for given release tag.

    Downloads zip release archive, extracts its content and clears old files.

    Args:
        tag (str): PyHook release tag.
        parent (sg.Window, optional): Parent window for centering. Defaults to None.

    Returns:
        str | None: Name of updated executable or None.
    """
    release_url = _get_release_url(tag)
    last_progress = 0
    cancel_popup: sg.Window = None

    window = sg.Window(
        "PyHook update...",
        [
            [
                sg.Text(
                    "Downloading update...",
                    key=SGKeys.DOWNLOAD_TEXT,
                )
            ],
            [
                sg.ProgressBar(
                    100,
                    size_px=(300, 14),
                    visible=True,
                    key=SGKeys.DOWNLOAD_PROGRESS_BAR,
                ),
            ],
            [sg.Image(size=(300, 0), pad=(0, 0))],
        ],
        font=FONT_SMALL_DEFAULT,
        element_justification="center",
        enable_close_attempted_event=True,
        disable_minimize=True,
        modal=True,
        keep_on_top=True,
    )
    center_in_parent(window, parent)
    window.refresh()

    def _download_callback(progress: int) -> bool:
        """Download callback to manage download process and display progress bar.

        Args:
            progress (int): Downloading progress percentage.

        Returns:
            bool: Flag if downloading should be continued.
        """
        nonlocal last_progress, cancel_popup
        if progress > last_progress or progress == 0:
            window[SGKeys.DOWNLOAD_PROGRESS_BAR].update(current_count=max(progress, 0))
            last_progress = progress
        event, _ = window.read(0)
        if event == sg.WIN_CLOSE_ATTEMPTED_EVENT:
            if cancel_popup is None:
                cancel_popup = show_popup_text(
                    "Confirm cancel",
                    "Are you sure to cancel update?",
                    ok_label="Yes",
                    cancel_button=True,
                    cancel_label="No",
                    return_window=True,
                    parent=window,
                )
        if cancel_popup is not None:
            popup_event, _ = cancel_popup.read(0)
            if popup_event in (
                sg.WIN_CLOSED,
                SGKeys.EXIT,
                SGKeys.POPUP_KEY_OK_BUTTON,
                SGKeys.POPUP_KEY_CANCEL_BUTTON,
            ):
                cancel_popup.close()
                cancel_popup = None
                return popup_event != SGKeys.POPUP_KEY_OK_BUTTON
        return True

    # Download update
    was_cancelled = not _download_callback(0) or download_file(release_url, os.getcwd(), _download_callback)
    if cancel_popup is not None:
        cancel_popup.close()
        cancel_popup = None
    if was_cancelled:
        window.close()
        return None

    # Extract update. At this point cancel is NOT POSSIBLE.
    window[SGKeys.DOWNLOAD_PROGRESS_BAR].update(current_count=0)
    window[SGKeys.DOWNLOAD_TEXT].update(value="Extracting update...")
    basename = release_url.split("/")[-1][:-4]
    executable_name = f"{basename}.exe"
    pipelines_in_zip = f"{basename}/pipelines"
    pipelines_dir = get_pipeline_directory()
    file_count = 0
    with zipfile.ZipFile(f"{basename}.zip", "r") as zip_archive:
        file_list = [file for file in zip_archive.namelist() if not file.endswith("/")]
        for file in file_list:
            if file.startswith(pipelines_in_zip):
                zip_info = zip_archive.getinfo(file)
                zip_info.filename = zip_info.filename.replace(pipelines_in_zip, "")
                zip_archive.extract(zip_info, pipelines_dir)
            elif file.endswith(executable_name):
                zip_info = zip_archive.getinfo(file)
                zip_info.filename = executable_name
                zip_archive.extract(zip_info)
            file_count += 1
            window[SGKeys.DOWNLOAD_PROGRESS_BAR].update(current_count=int(file_count / len(file_list) * 100))
    os.remove(f"{basename}.zip")

    window.close()
    return executable_name


def try_update(
    forced: bool = False, updated_exe: str | None = None, parent: sg.Window = None
) -> Tuple[str | None, bool]:
    """Tries to update app version to the latest GitHub release tag.

    Only frozen bundle can be updated.

    Args:
        forced (bool, optional): Flag if update should be forced. Defaults to False.
        updated_exe (str | None, optional): Updated executable path. Defaults to None.
        parent (sg.Window, optional): Parent window for centering. Defaults to None.

    Returns:
        Tuple[str | None, bool]: Name of updated executable or None and flag if app should be restarted.
    """
    if not is_frozen_bundle():
        if forced:
            show_popup_text("Update info", "Only built app can be updated.", parent=parent)
        return None, False
    if updated_exe is not None:
        return updated_exe, _ask_for_restart(parent)
    settings = get_settings()
    if settings[SettingsKeys.KEY_AUTOUPDATE] or forced:
        latest_tag = _get_latest_version_tag()
        if latest_tag is None:
            if forced:
                show_popup_text("Update info", "Cannot find newer version.", parent=parent)
            return None, False
        if latest_tag > __version__:

            def open_url_for_tag() -> None:
                """Opens PyHook GitHub release page in default OS web browser."""
                webbrowser.open(_TAG_URL + latest_tag)

            if show_popup(
                "Update found",
                [
                    [
                        sg.Text(
                            f"New version found! Do you want to update to version {latest_tag}?",
                            justification="center",
                        )
                    ],
                    [
                        sg.Text(
                            "> Preview release <",
                            text_color=COLOR_TEXT_URL,
                            justification="center",
                            enable_events=True,
                            key=SGKeys.UPDATE_RELEASE_LINK,
                        )
                    ],
                ],
                ok_label="Yes",
                cancel_button=True,
                cancel_label="No",
                events={SGKeys.UPDATE_RELEASE_LINK: EventCallback(open_url_for_tag, False)},
                parent=parent,
            ):
                updated_executable = _process_update(latest_tag, parent)
                return updated_executable, False if updated_executable is None else _ask_for_restart(parent)
            return None, False
    if forced:
        show_popup_text("Update info", "Cannot find newer version.", parent=parent)
    return None, False
