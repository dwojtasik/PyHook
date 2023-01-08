"""
gui.pipeline_actions for PyHook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Pipeline actions windows for PyHook
:copyright: (c) 2023 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""

from os.path import basename, exists
from subprocess import PIPE, Popen
from typing import Tuple

import PySimpleGUI as sg

from keys import SettingsKeys
from pipeline import PipelinesDirNotFoundError, get_pipeline_directory, get_pipeline_file_list, supports_platform
from gui.settings import get_settings, save_settings
from gui.style import FONT_SMALL_DEFAULT
from gui.ui_keys import SGKeys
from gui.utils import center_in_parent, show_popup_exception, show_popup_text
from utils.downloader import download_file
from utils.threading import BackgroundTask
from win.api import CREATE_NO_WINDOW

# Text format for pipeline.
_PIPELINE_TEXT_FORMAT = "Pipeline (%d/%d): %s"
# Text format for pipeline file.
_FILE_TEXT_FORMAT = "File (%d/%d): %s"
# Maximum number of characters to display for URLs.
_MAX_URL_LENGTH = 60


def _verify_files(
    window: sg.Window, cancel_popup: sg.Window | None, pipeline_dir: str, pipeline_file: str
) -> Tuple[sg.Window, bool]:
    """Verify and download files for given pipeline.

    Args:
        window (sg.Window): UI window with progress.
        cancel_popup: (sg.Window | None): Cancel popup window.
        pipeline_dir (str): Pipeline directory.
        pipeline_file (str): Pipeline file.

    Returns:
        Tuple[sg.Window, bool]: Cancel popup window and flag if all files were successfully verified.
    """

    is_cancelled = False
    last_progress = 0
    act_progress: int | None = None

    def _download_callback(progress: int | None) -> bool:
        """Download callback to manage download process and display progress bar in UI thread.

        Args:
            progress (int | None): Downloading progress percentage. Could be None.

        Returns:
            bool: Flag if downloading should be continued.
        """
        nonlocal act_progress
        act_progress = progress
        return not is_cancelled

    pipeline_data_dir = f"{pipeline_dir}\\{pipeline_file[:-3]}"
    download_list_file = f"{pipeline_data_dir}\\download.txt"
    if exists(download_list_file):
        try:
            with open(download_list_file, encoding="utf-8") as d_file:
                urls = [url for url in d_file.read().split("\n") if len(url) > 1 and not url.startswith("#")]
                count = len(urls)
                for i, url in enumerate(urls):
                    window[SGKeys.DOWNLOAD_FILE_VERIFY_TEXT].update(
                        value=_FILE_TEXT_FORMAT
                        % (
                            i + 1,
                            count,
                            url
                            if len(url) <= _MAX_URL_LENGTH + 3
                            else f"{url[: _MAX_URL_LENGTH // 2]}...{url[-_MAX_URL_LENGTH // 2 :]}",
                        )
                    )
                    bg_task = BackgroundTask(download_file, [url, pipeline_data_dir, _download_callback])
                    bg_task.start()
                    while bg_task.is_running():
                        event, _ = window.read(0)
                        if act_progress is None or act_progress > last_progress:
                            window[SGKeys.DOWNLOAD_PROGRESS_BAR].update(
                                current_count=max(0 if act_progress is None else act_progress, 0),
                                visible=act_progress is not None,
                            )
                            window[SGKeys.DOWNLOAD_PROGRESS_PLACEHOLDER_TEXT].update(visible=act_progress is None)
                            last_progress = 0 if act_progress is None else act_progress
                        if event == sg.WIN_CLOSE_ATTEMPTED_EVENT:
                            if cancel_popup is None:
                                cancel_popup = show_popup_text(
                                    "Confirm cancel",
                                    "Are you sure to cancel files verifying?",
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
                                is_cancelled = popup_event == SGKeys.POPUP_KEY_OK_BUTTON
                    bg_task.join()
                    if bg_task.get_output():
                        window.write_event_value(SGKeys.DOWNLOAD_CANCEL_EVENT, ())
                        return cancel_popup, False
            return cancel_popup, True
        except Exception as ex:
            show_popup_exception(
                "Error",
                "Cannot read / download pipeline files with following error:",
                ex,
                ex_width=75,
                parent=window,
            )
            return cancel_popup, False
    return cancel_popup, True


def verify_download(forced: bool = False, parent: sg.Window = None) -> None:
    """Verifies if all pipeline files were downloaded.
    If any file is missing or it's size differ with URL ones it will be downloaded.

    Args:
        forced (bool, optional): Forces all files download. Defaults to False.
        parent (sg.Window, optional): Parent window for centering. Defaults to None.
    """
    has_change = forced
    settings = get_settings()
    if forced:
        settings[SettingsKeys.KEY_DOWNLOADED] = []
    if settings[SettingsKeys.KEY_AUTODOWNLOAD] or forced:
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
            cancel_popup: sg.Window = None
            window = sg.Window(
                "Verifying pipeline files...",
                [
                    [
                        sg.Text(
                            _PIPELINE_TEXT_FORMAT % (1, count, ""),
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
            )
            center_in_parent(window, parent)
            window.refresh()
            for i, pipeline in enumerate(pipelines):
                window[SGKeys.DOWNLOAD_VERIFY_TEXT].update(value=_PIPELINE_TEXT_FORMAT % (i + 1, count, pipeline))
                window[SGKeys.DOWNLOAD_FILE_VERIFY_TEXT].update(value="")
                window.refresh()
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
                        if popup_event == SGKeys.POPUP_KEY_OK_BUTTON:
                            break
                if pipeline not in settings[SettingsKeys.KEY_DOWNLOADED]:
                    cancel_popup, success = _verify_files(window, cancel_popup, pipeline_dir, pipeline)
                    if success:
                        settings[SettingsKeys.KEY_DOWNLOADED].append(pipeline)
                        has_change = True
                    else:
                        event, _ = window.read(0)
                        if event == SGKeys.DOWNLOAD_CANCEL_EVENT:
                            break
            if cancel_popup is not None:
                cancel_popup.close()
            window.close()
    if has_change:
        save_settings(settings)


def _install_with_pip(local_python_path: str, file: str) -> None:
    """Installs requirements from given file.

    Args:
        local_python_path (str): Path to local Python executable.
        file (str): Requirements file path.

    Raises:
        Exception: When installation did not succeed.
    """
    with Popen(
        f"{local_python_path} -m pip install -r {file} --quiet --disable-pip-version-check",
        stderr=PIPE,
        shell=False,
        creationflags=CREATE_NO_WINDOW,
        start_new_session=True,
    ) as process:
        _, err = process.communicate()
        if process.returncode != 0:
            raise RuntimeError(err.decode("utf-8"))


def install_requirements(local_python_path: str, platform: int, parent: sg.Window = None) -> None:
    """Installs pipelines requirements for local Python.

    Args:
        local_python_path (str): Path to local Python executable.
        platform (int): Platform bit, either 32 or 64.
        parent (sg.Window, optional): Parent window for centering. Defaults to None.
    """
    pipeline_dir = None
    try:
        pipeline_dir = get_pipeline_directory()
    except PipelinesDirNotFoundError:
        show_popup_text(
            "Error", "Cannot find pipelines directory!\nMake sure pipelines directory exists in PyHook directory."
        )
        return
    pipeline_files = [path for path in get_pipeline_file_list(pipeline_dir) if supports_platform(path, platform)]
    pipelines = [basename(path) for path in pipeline_files]
    count = len(pipelines)
    cancel_popup: sg.Window = None
    cancelled = False
    window = sg.Window(
        "Installing requirements...",
        [
            [
                sg.Text(
                    _PIPELINE_TEXT_FORMAT % (1, count, ""),
                    key=SGKeys.REQUIREMENTS_INSTALL_TEXT,
                )
            ],
            [
                sg.ProgressBar(
                    100,
                    size_px=(450, 14),
                    visible=True,
                    key=SGKeys.REQUIREMENTS_INSTALL_PROGRESS_BAR,
                )
            ],
            [sg.Image(size=(450, 0), pad=(0, 0))],
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

    def is_cancelled() -> bool:
        """Checks if user confirmed cancellation.

        Returns:
            bool: Flag if user confirmed cancellation.
        """
        if cancelled:
            return True
        nonlocal cancel_popup
        event, _ = window.read(0)
        if event == sg.WIN_CLOSE_ATTEMPTED_EVENT:
            if cancel_popup is None:
                cancel_popup = show_popup_text(
                    "Confirm cancel",
                    "Are you sure to cancel requirements installation?",
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
                return popup_event == SGKeys.POPUP_KEY_OK_BUTTON
        return False

    for i, pipeline in enumerate(pipelines):
        window[SGKeys.REQUIREMENTS_INSTALL_TEXT].update(value=_PIPELINE_TEXT_FORMAT % (i + 1, count, pipeline))
        window.refresh()
        if is_cancelled():
            break
        pipeline_requirements = f"{pipeline_dir}\\{pipeline[:-3]}.requirements.txt"
        if exists(pipeline_requirements):
            bg_task = BackgroundTask(_install_with_pip, [local_python_path, pipeline_requirements])
            bg_task.start()
            while bg_task.is_running():
                if is_cancelled():
                    window[SGKeys.REQUIREMENTS_INSTALL_TEXT].update(value="Cancelling, please wait...")
                    window.refresh()
                    cancelled = True
                    break
            bg_task.join()
            if cancelled:
                break
            try:
                bg_task.get_output()
            except Exception as ex:
                if not show_popup_exception(
                    title="Error",
                    text=f'Error occurred while installing requirements for pipeline "{pipeline}".',
                    ex=ex,
                    ok_label="Yes",
                    cancel_button=True,
                    cancel_label="No",
                    ex_width=100,
                    text_after="Do you want to continue installation for other pipelines?",
                    parent=window,
                ):
                    break
        window[SGKeys.REQUIREMENTS_INSTALL_PROGRESS_BAR].update(current_count=int((i + 1) / count * 100))
        window.refresh()
    if cancel_popup is not None:
        cancel_popup.close()
    window.close()
