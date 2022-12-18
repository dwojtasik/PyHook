"""
gui.download for PyHook
~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pipeline download window for PyHook
:copyright: (c) 2022 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""

from os.path import basename, exists

import PySimpleGUI as sg

from keys import SettingsKeys
from pipeline import PipelinesDirNotFoundError, get_pipeline_directory, get_pipeline_file_list
from gui.keys import SGKeys
from gui.settings import get_settings, save_settings
from gui.style import FONT_SMALL_DEFAULT
from gui.utils import show_popup_exception, show_popup_text
from utils.downloader import download_file

# Text format for pipeline verification.
_VERIFY_TEXT_FORMAT = "Pipeline (%d/%d): %s"
# Text format for pipeline file verification.
_FILE_VERIFY_TEXT_FORMAT = "File (%d/%d): %s"
# Maximum number of characters to display for URLs.
_MAX_URL_LENGTH = 60


def _verify_files(window: sg.Window, pipeline_dir: str, pipeline_file: str) -> bool:
    """Verify and download files for given pipeline.

    Args:
        window (sg.Window): UI window with progress.
        pipeline_dir (str): Pipeline directory.
        pipeline_file (str): Pipeline file.

    Returns:
        bool: Flag if all files were successfully verified.
    """

    last_progress = 0
    cancel_popup: sg.Window = None

    def _download_callback(progress: int) -> bool:
        """Download callback to manage download process and display progress bar.

        Args:
            progress (int): Downloading progress percentage.

        Returns:
            bool: Flag if downloading should be continued.
        """
        nonlocal last_progress, cancel_popup
        if progress > last_progress:
            window[SGKeys.DOWNLOAD_PROGRESS_BAR].update(current_count=max(progress, 0), visible=progress > -1)
            window[SGKeys.DOWNLOAD_PROGRESS_PLACEHOLDER_TEXT].update(visible=progress < 0)
            last_progress = progress
        event, _ = window.read(0)
        if event == sg.WIN_CLOSE_ATTEMPTED_EVENT:
            if cancel_popup is None:
                cancel_popup = show_popup_text(
                    "Confirm cancel",
                    "Are you sure to cancel download?",
                    ok_label="Yes",
                    cancel_button=True,
                    cancel_label="No",
                    return_window=True,
                )
        if cancel_popup is not None:
            popup_event, _ = cancel_popup.read(0)
            if popup_event in (
                sg.WIN_CLOSED,
                SGKeys.EXIT,
                SGKeys.POPUP_KEY_OK_BUTTON,
                SGKeys.POPUP_KEY_CANCEL_BUTTON,
            ):
                should_continue = popup_event != SGKeys.POPUP_KEY_OK_BUTTON
                cancel_popup.close()
                cancel_popup = None
                return should_continue
        return True

    pipeline_data_dir = f"{pipeline_dir}\\{pipeline_file[:-3]}"
    download_list_file = f"{pipeline_data_dir}\\download.txt"
    if exists(download_list_file):
        try:
            with open(download_list_file, encoding="utf-8") as d_file:
                urls = [url for url in d_file.read().split("\n") if len(url) > 1 and not url.startswith("#")]
                count = len(urls)
                for i, url in enumerate(urls):
                    window[SGKeys.DOWNLOAD_FILE_VERIFY_TEXT].update(
                        value=_FILE_VERIFY_TEXT_FORMAT
                        % (
                            i + 1,
                            count,
                            url
                            if len(url) <= _MAX_URL_LENGTH + 3
                            else f"{url[: _MAX_URL_LENGTH // 2]}...{url[-_MAX_URL_LENGTH // 2 :]}",
                        )
                    )
                    _download_callback(-1)
                    was_cancelled = download_file(url, pipeline_data_dir, _download_callback)
                    if was_cancelled:
                        window.write_event_value(SGKeys.DOWNLOAD_CANCEL_EVENT, ())
                        return False
            return True
        except Exception as ex:
            show_popup_exception("Error", "Cannot read / download pipeline files!", ex)
            return False


def verify_download(forced: bool = False) -> None:
    """Verifies if all pipeline files were downloaded.
    If any file is missing or it's size differ with URL ones it will be downloaded.

    Args:
        forced (bool, optional): Forces all files download. Defaults to False.
    """
    has_change = forced
    settings = get_settings()
    if forced:
        settings[SettingsKeys.KEY_DOWNLOADED] = []
    if settings[SettingsKeys.KEY_AUTODOWNLOAD]:
        pipeline_dir = None
        try:
            pipeline_dir = get_pipeline_directory()
        except PipelinesDirNotFoundError:
            show_popup_text(
                "Error", "Cannot find pipelines directory!\nMake sure pipelines directory exists in PyHook directory."
            )
            return
        pipeline_files = get_pipeline_file_list(pipeline_dir)
        pipelines = [basename(path) for path in pipeline_files]
        count = len(pipelines)
        display_ui = forced or any(path not in settings[SettingsKeys.KEY_DOWNLOADED] for path in pipelines)
        if display_ui:
            window = sg.Window(
                "Verifying pipeline files...",
                [
                    [
                        sg.Text(
                            _VERIFY_TEXT_FORMAT % (1, count, ""),
                            key=SGKeys.DOWNLOAD_VERIFY_TEXT,
                        )
                    ],
                    [
                        sg.Text(
                            "",
                            key=SGKeys.DOWNLOAD_FILE_VERIFY_TEXT,
                        )
                    ],
                    [
                        sg.ProgressBar(
                            100,
                            size_px=(450, 14),
                            visible=True,
                            key=SGKeys.DOWNLOAD_PROGRESS_BAR,
                        ),
                        sg.Text(
                            "Verifying for download...", visible=False, key=SGKeys.DOWNLOAD_PROGRESS_PLACEHOLDER_TEXT
                        ),
                    ],
                    [sg.Image(size=(600, 0), pad=(0, 0))],
                ],
                font=FONT_SMALL_DEFAULT,
                element_justification="center",
                enable_close_attempted_event=True,
                disable_minimize=True,
                modal=True,
                keep_on_top=True,
                finalize=True,
                location=(None, None),
            )
            window.refresh()
            for i, pipeline in enumerate(pipelines):
                window[SGKeys.DOWNLOAD_VERIFY_TEXT].update(value=_VERIFY_TEXT_FORMAT % (i + 1, count, pipeline))
                window[SGKeys.DOWNLOAD_FILE_VERIFY_TEXT].update(value="")
                window.refresh()
                if pipeline not in settings[SettingsKeys.KEY_DOWNLOADED]:
                    if _verify_files(window, pipeline_dir, pipeline):
                        settings[SettingsKeys.KEY_DOWNLOADED].append(pipeline)
                        has_change = True
                    else:
                        event, _ = window.read(0)
                        if event == SGKeys.DOWNLOAD_CANCEL_EVENT:
                            break
            window.close()
    if has_change:
        save_settings(settings)
