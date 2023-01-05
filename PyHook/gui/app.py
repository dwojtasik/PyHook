"""
gui.app for PyHook
~~~~~~~~~~~~~~~~~~~~~~~
GUI for PyHook
:copyright: (c) 2022 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""

import atexit
import os
import string
import sys
import webbrowser
from threading import Thread
from time import sleep
from typing import Dict, List

import PySimpleGUI as sg

from _version import __version__
from session import ProcessInfo, Session, get_process_list
from gui.image import format_raw_data, get_as_buffer, get_button_image_template, get_img
from gui.keys import SGKeys, TKKeys
from gui.pipeline_actions import install_requirements, verify_download
from gui.settings import display_settings_window, get_settings, load_settings
from gui.style import *  # pylint: disable=wildcard-import, unused-wildcard-import
from gui.update import try_update
from gui.utils import EventCallback, show_popup, show_popup_text, with_border
from keys import SettingsKeys
from utils.common import delete_self_exe, is_frozen_bundle
from win.api import get_hq_icon_raw

# Flag if PyHook is started as 64-bit app.
_IS_64_BIT = sys.maxsize > 2**32

# Maximum amount of sessions.
_MAX_SESSIONS = 15

# Default icon to display.
if is_frozen_bundle():
    _APP_ICON = get_as_buffer(format_raw_data(get_hq_icon_raw(sys.executable), thumb_size=(128, 128)))
else:
    _APP_ICON = get_img(f"{os.getcwd()}\\pyhook_icon.ico", thumb_size=(128, 128))

# Default clear button image.
_BUTTON_IMAGE_NONE = get_as_buffer(get_button_image_template())

# Set default theme.
sg.theme("DarkBlue")


def _get_sessions_layout() -> List[List[sg.Column]]:
    """Returns session list layout.

    Each session is displayed as button. Maximum of 3 sessions displayed per row.
    First 6 sessions has to be initialized as visible to calculate valid scroll view size.

    Returns:
        List[List[sg.Column]]: Session list layout.
    """
    rows: List[List[sg.Column]] = []
    per_row_sessions = 3
    for i in range(_MAX_SESSIONS):
        if len(rows) < i // per_row_sessions + 1:
            rows.append([])
        rows[i // per_row_sessions].append(
            with_border(
                sg.Button(
                    image_data=_BUTTON_IMAGE_NONE,
                    size=SESSION_BUTTON_ICON_SIZE,
                    pad=(2, 3),
                    button_color=SESSION_BUTTON_COLORS,
                    tooltip="",
                    key=SGKeys.get_session_key(i),
                ),
                color="green",
                visible=i < 6,
            )
        )
    return [
        [
            sg.Column(
                rows,
                size=(390, 340),
                scrollable=True,
                expand_y=True,
                vertical_scroll_only=True,
                key=SGKeys.SESSION_LIST,
            )
        ]
    ]


def _update_process_list(
    window: sg.Window, process_list: List[ProcessInfo], filter_string: str, open_dropdown: bool = False
) -> None:
    """Updates process list combo box.

    Args:
        window (sg.Window): Parent window.
        process_list (List[ProcessInfo]): List of new process info.
        filter_string (str): Filter to be applied. For empty string filtering will be omitted.
        open_dropdown (bool, optional): Flag if combo box should be opened after updating it's state.
            Defaults to False.
    """
    combo: sg.Combo = window[SGKeys.PROCESS_LIST]
    combo.update(
        value=filter_string,
        values=_to_combo_list(process_list, filter_string if filter_string else None),
    )
    if open_dropdown:
        combo.TKCombo.event_generate(TKKeys.EVENT_MOUSE_BUTTON_CLICK)


def _update_sessions_active_view(window: sg.Window, sessions: List[Session], selected_session: Session | None) -> None:
    """Updates sessions list view after dynamic change from subprocess.

    For running session border color is set to green. Otherwise red.
    Session button image is updated with app built-in icon, pid and executable name.
    Tooltip for session button is updated with it's name and status.
    Restart button in session overview is enabled only if session is exited.

    Args:
        window (sg.Window): Parent window.
        sessions (List[Session]): List of sessions.
        selected_session (Session | None): Session selected by user.
            None value means that user does not have any session selected.
    """
    for i, session in enumerate(sessions):
        session_key = SGKeys.get_session_key(i)
        running = session.is_running()
        window[session_key].ParentRowFrame.config(background="green" if running else "red")
        window[session_key].update(image_data=sessions[i].button_image)
        window[session_key].set_tooltip(
            f"Process: {sessions[i].get_name()}\nStatus: {'Running' if running else 'Exited'}"
        )
        if selected_session is not None and selected_session.pid.value == session.pid.value:
            window[SGKeys.SESSION_TITLE].update(value=selected_session.get_name())
            window[SGKeys.SESSION_RESTART_BUTTON].update(disabled=running)
        if selected_session is None:
            window[SGKeys.SESSION_TITLE].update(value="Select session...")


def _update_sessions_view(window: sg.Window, sessions: List[Session]) -> None:
    """Updates sessions list view.

    Displays new sessions state on view. Called on session create and delete.

    Args:
        window (sg.Window): Parent window.
        sessions (List[Session]): List of sessions.
    """
    sessions_count = len(sessions)
    for i in range(_MAX_SESSIONS):
        session_key = SGKeys.get_session_key(i)
        if i < sessions_count:
            running = sessions[i].is_running()
            window[session_key + SGKeys.BORDER_SUFFIX].update(visible=True)
            window[session_key].ParentRowFrame.config(background="green" if running else "red")
            if sessions[i].button_image is None:
                window[session_key].update(image_data=_BUTTON_IMAGE_NONE)
            else:
                window[session_key].update(image_data=sessions[i].button_image)
            window[session_key].set_tooltip(
                f"Process: {sessions[i].get_name()}\nStatus: {'Running' if running else 'Exited'}"
            )
        else:
            window[session_key].update(image_data=_BUTTON_IMAGE_NONE)
            window[session_key].set_tooltip("")
            window[session_key + SGKeys.BORDER_SUFFIX].update(visible=False)
    window[SGKeys.SESSION_LIST].contents_changed()


def _update_session_overview(window: sg.Window, selected_session: Session | None) -> None:
    """Updates session overview view.

    Displays all elements in session overview frame, e.g. buttons, log view, texts.

    Args:
        window (sg.Window): Parent window.
        selected_session (Session | None): Session selected by user.
            None value means that user does not have any session selected.
    """
    visible = selected_session is not None
    window[SGKeys.SESSION_TITLE].update(value="Select session..." if not visible else selected_session.get_name())
    window[SGKeys.SESSION_KILL_BUTTON].update(visible=visible)
    window[SGKeys.SESSION_RESTART_BUTTON].update(visible=visible, disabled=visible and selected_session.is_running())
    window[SGKeys.SESSION_CLOSE_OVERVIEW_BUTTON].update(visible=visible)
    window[SGKeys.SESSION_LOGS].update(
        value="" if not visible else selected_session.get_logs(), visible=visible, autoscroll=True
    )
    window[SGKeys.SESSION_LOGS_SCROLL_TOP_BUTTON].update(visible=visible)
    window[SGKeys.SESSION_LOGS_CLEAR_BUTTON].update(visible=visible)
    window[SGKeys.SESSION_LOGS_SCROLL_BOT_BUTTON].update(visible=visible)


def _to_combo_list(process_list: List[ProcessInfo], filter_string: str = None) -> List[str]:
    """Filters list of processes to combo list.

    Args:
        process_list (List[ProcessInfo]): List of processes.
        filter_string (str, optional): Filter to be applied. Defaults to None.

    Returns:
        List[str]: List of combo box options.
    """
    if filter_string is None:
        return [process.get_combo_string() for process in process_list]
    if filter_string.isnumeric():
        filter_pid = filter_string
        filter_name = ""
    else:
        filter_pid = ""
        filter_name = filter_string.lower()
    return [
        process.get_combo_string()
        for process in process_list
        if (filter_pid and filter_pid in str(process.pid)) or (filter_name and filter_name in process.name.lower())
    ]


def _bind_combo_popdown_event(window: sg.Window) -> None:
    """Binds KeyPress and KeyRelease events to TTK combobox popdown window.

    Args:
        window (sg.Window): Parent window.
    """
    combo: sg.Combo = window[SGKeys.PROCESS_LIST]
    popdown_ref = combo.TKCombo.tk.call(TKKeys.POPDOWN_WINDOW, combo.TKCombo) + TKKeys.POPDOWN_WINDOW_SUFFIX

    pressed = []
    allowed_chars = string.ascii_lowercase + string.ascii_uppercase + string.digits + " "

    def _on_key_press(event) -> None:
        """Callback for KeyPress event.

        Args:
            event (tkinter.Event): KeyPress event.
        """
        if event.char in pressed or event.keysym in pressed:
            return
        if event.char in allowed_chars:
            pressed.append(event.char)
            idx = combo.TKCombo.index(TKKeys.INSERT)
            filter_string = combo.TKCombo.get()
            combo.TKCombo.event_generate(TKKeys.EVENT_MOUSE_BUTTON_CLICK)
            combo.TKCombo.set(filter_string[:idx] + event.char + filter_string[idx:])
            combo.TKCombo.icursor(idx + 1)
        elif event.keysym == TKKeys.BUTTON_BACKSPACE:
            pressed.append(event.keysym)
            idx = combo.TKCombo.index(TKKeys.INSERT)
            if idx > 0:
                filter_string = combo.TKCombo.get()
                combo.TKCombo.event_generate(TKKeys.EVENT_MOUSE_BUTTON_CLICK)
                combo.TKCombo.set(filter_string[: idx - 1] + filter_string[idx:])
                combo.TKCombo.icursor(idx - 1)
        elif event.keysym == TKKeys.BUTTON_DELETE:
            pressed.append(event.keysym)
            idx = combo.TKCombo.index(TKKeys.INSERT)
            filter_string = combo.TKCombo.get()
            if idx < len(filter_string):
                combo.TKCombo.event_generate(TKKeys.EVENT_MOUSE_BUTTON_CLICK)
                combo.TKCombo.set(filter_string[:idx] + filter_string[idx + 1 :])

    def _on_key_release(event) -> None:
        """Callback for KeyRelease event.

        Args:
            event (tkinter.Event): KeyRelease event.
        """
        if event.char in pressed:
            pressed.remove(event.char)
        if event.keysym in pressed:
            pressed.remove(event.keysym)

    combo.TKCombo._bind((TKKeys.CMD_BIND, popdown_ref), TKKeys.EVENT_KEY_PRESS, _on_key_press, None)
    combo.TKCombo._bind((TKKeys.CMD_BIND, popdown_ref), TKKeys.EVENT_KEY_RELEASE, _on_key_release, None)


# Application menu layout.
_MENU_LAYOUT = [
    ["App", [SGKeys.MENU_SETTINGS_OPTION, SGKeys.EXIT]],
    ["Pipeline", [SGKeys.MENU_PIPELINE_FORCE_DOWNLOAD_OPTION, SGKeys.MENU_PIPELINE_INSTALL_REQUIREMENTS_OPTION]],
    ["Help", [SGKeys.MENU_UPDATE_OPTION, SGKeys.MENU_ABOUT_OPTION]],
]

# Application UI layout.
_APP_LAYOUT = [
    [sg.Menu(_MENU_LAYOUT, font=FONT_SMALL_DEFAULT, text_color="black", background_color="white")],
    [
        sg.Text("Process"),
        sg.Combo(
            values=[],
            key=SGKeys.PROCESS_LIST,
            enable_events=True,
            font=FONT_MONO_DEFAULT,
            size=(50, 1),
            tooltip="Process to inject PyHook",
        ),
        sg.Button("\u274C", key=SGKeys.INJECT_CLEAR, font=FONT_MONO_DEFAULT, size=(2, 1), tooltip="Clear input"),
        sg.Button(
            "\u21BB",
            key=SGKeys.PROCESS_RELOAD,
            font=FONT_MONO_DEFAULT,
            size=(2, 1),
            tooltip="Reload process list",
        ),
        sg.Button("Inject", key=SGKeys.INJECT, size=(4, 1), tooltip="Inject PyHook into selected process"),
        sg.Button(
            "Auto",
            key=SGKeys.INJECT_AUTO,
            size=(4, 1),
            tooltip="Try to automatically find process with ReShade and PyHook loaded",
        ),
    ],
    [
        sg.Frame(
            "Sessions",
            _get_sessions_layout(),
            border_width=3,
            expand_x=True,
            expand_y=True,
        ),
        sg.Frame(
            "Session overview",
            [
                [
                    sg.Text(
                        "Select session...",
                        font=FONT_MID_DEFAULT,
                        pad=(10, 10),
                        justification="left",
                        key=SGKeys.SESSION_TITLE,
                    ),
                    sg.Push(),
                    sg.Button(
                        "Kill",
                        size=(6, 1),
                        key=SGKeys.SESSION_KILL_BUTTON,
                        tooltip="Kill this session and remove from sessions list",
                        visible=True,
                    ),
                    sg.Button(
                        "\u21BB",
                        key=SGKeys.SESSION_RESTART_BUTTON,
                        font=FONT_MONO_DEFAULT,
                        size=(2, 1),
                        tooltip="Restart exited session",
                        disabled=True,
                        visible=True,
                    ),
                    sg.Button(
                        "\u274C",
                        key=SGKeys.SESSION_CLOSE_OVERVIEW_BUTTON,
                        font=FONT_MONO_DEFAULT,
                        size=(2, 1),
                        tooltip="Close overview",
                        visible=True,
                    ),
                ],
                [
                    sg.Multiline(
                        "",
                        font=FONT_CONSOLE,
                        size=(80, 16),
                        key=SGKeys.SESSION_LOGS,
                        enable_events=True,
                        autoscroll=True,
                        disabled=True,
                        expand_x=True,
                        expand_y=True,
                        visible=True,
                    )
                ],
                [
                    sg.Column(
                        [
                            [
                                sg.Button(
                                    "\u2191",
                                    key=SGKeys.SESSION_LOGS_SCROLL_TOP_BUTTON,
                                    font=FONT_MONO_DEFAULT,
                                    size=(2, 1),
                                    tooltip="Scroll to top",
                                    visible=True,
                                ),
                                sg.Button(
                                    "Clear logs",
                                    size=(10, 1),
                                    key=SGKeys.SESSION_LOGS_CLEAR_BUTTON,
                                    tooltip="Clear session logs",
                                    visible=True,
                                ),
                                sg.Button(
                                    "\u2193",
                                    key=SGKeys.SESSION_LOGS_SCROLL_BOT_BUTTON,
                                    font=FONT_MONO_DEFAULT,
                                    size=(2, 1),
                                    tooltip="Scroll to bottom",
                                    visible=True,
                                ),
                            ]
                        ],
                        justification="center",
                    )
                ],
            ],
            border_width=3,
            expand_x=True,
            expand_y=True,
        ),
    ],
]


def open_github() -> None:
    """Opens PyHook GitHub page in default OS web browser."""
    webbrowser.open("https://github.com/dwojtasik/PyHook")


def gui_main() -> None:
    """App GUI entrypoint."""

    # Flag if GUI is running.
    running = True
    # UI worker thread.
    ui_worker: Thread = None
    # Last read process list.
    process_list = get_process_list()
    # Last process filter string.
    last_process_filter = ""
    # Last selected PID.
    last_pid: int = None
    # List of active sessions.
    sessions: List[Session] = []
    # Dictionary of session uuid to it's killing thread.
    killed_sessions: Dict[str, Thread] = {}
    # Selected session to display overview.
    selected_session: Session = None
    # Name of updated executable.
    updated_executable: str = None
    # Flag if app should restart for update.
    update_restart = False

    # Application window.
    window = sg.Window(
        f"PyHook v{__version__} (c) 2022 by Dominik Wojtasik",
        _APP_LAYOUT,
        font=FONT_DEFAULT,
        enable_close_attempted_event=True,
        icon=_APP_ICON,
        finalize=True,
    )

    _update_sessions_view(window, sessions)
    _update_session_overview(window, selected_session)
    _update_process_list(window, process_list, "")
    _bind_combo_popdown_event(window)

    load_settings(window)
    verify_download(parent=window)
    updated_executable, update_restart = try_update(parent=window)

    if update_restart:
        running = False

    def _kill_session(session: Session) -> None:
        """Kills given sessions.

        Args:
            session (Session): PyHook session.
        """

        def _kill_self() -> None:
            """Kills session."""
            session.close()
            del killed_sessions[session.uuid]

        killed_sessions[session.uuid] = Thread(target=_kill_self)
        killed_sessions[session.uuid].start()

    def _close_app() -> None:
        """Closes all PyHook sessions and threads on app exit."""
        for session in sessions:
            session.close()
        for killing_thread in list(killed_sessions.values()):
            if killing_thread.is_alive():
                killing_thread.join()
        if updated_executable is not None:
            delete_self_exe()

    atexit.register(_close_app)

    def _update_ui() -> None:
        """Updates UI window."""
        nonlocal last_process_filter, last_pid
        while running:
            try:
                process_filter_value = window[SGKeys.PROCESS_LIST].TKCombo.get()  # pylint: disable=no-member
                if last_process_filter != process_filter_value:
                    last_process_filter = process_filter_value
                    _update_process_list(window, process_list, last_process_filter, True)
                    last_pid = None

                if selected_session is not None:
                    if selected_session.should_update_logs():
                        scroll_state = window[SGKeys.SESSION_LOGS].Widget.yview()
                        window[SGKeys.SESSION_LOGS].update(
                            value=selected_session.get_logs(), autoscroll=scroll_state[1] == 1
                        )

                if any([session.should_update_ui() for session in sessions[:]]):
                    _update_sessions_active_view(window, sessions, selected_session)

                sleep(1 / 60)
            except Exception:
                pass

    if running:
        ui_worker = Thread(target=_update_ui)
        ui_worker.start()

    while running:
        event, values = window.read()
        if event == sg.WIN_CLOSED:
            break
        if event in (sg.WINDOW_CLOSE_ATTEMPTED_EVENT, SGKeys.EXIT):
            if any(session.is_running() for session in sessions):
                if show_popup_text(
                    "Confirm exit",
                    "Are you sure to exit? There are some running PyHook sessions.",
                    ok_label="Yes",
                    cancel_button=True,
                    cancel_label="No",
                    parent=window,
                ):
                    break
            else:
                break
        if event == SGKeys.PROCESS_LIST:
            last_process_filter = values[event]
            pid_string = str(last_process_filter).split("|", maxsplit=1)[0].strip()
            if pid_string.isnumeric():
                last_pid = int(pid_string)
            else:
                last_pid = None
        elif event in (SGKeys.INJECT_CLEAR, SGKeys.PROCESS_RELOAD):
            if event == SGKeys.PROCESS_RELOAD:
                process_list = get_process_list()
            last_process_filter = ""
            _update_process_list(window, process_list, last_process_filter)
            last_pid = None
        elif event == SGKeys.INJECT:
            if last_pid is None:
                show_popup_text("Error", "First select process to inject PyHook.", parent=window)
                continue
            if any(session.pid.value == last_pid for session in sessions):
                show_popup_text("Error", "Session with given PID already exists.", parent=window)
                continue
            if len(sessions) == _MAX_SESSIONS:
                show_popup_text(
                    "Error", "Maximum amount of sessions reached.\nKill old session to start new one.", parent=window
                )
                continue
            last_process_filter = ""
            process_info = ProcessInfo.from_pid(last_pid)
            if process_info is None:
                process_list = get_process_list()
                _update_process_list(window, process_list, last_process_filter)
                show_popup_text("Error", "Process does not exists anymore.", parent=window)
                continue
            _update_process_list(window, process_list, last_process_filter)
            selected_session = Session(process_info)
            sessions.append(selected_session)
            _update_sessions_view(window, sessions)
            _update_session_overview(window, selected_session)
        elif event == SGKeys.INJECT_AUTO:
            auto_sessions: List[Session] = list(filter(lambda session: session.pid.value == -1, sessions))
            if len(auto_sessions) > 0 and auto_sessions[0].is_running():
                show_popup_text("Error", "Automatic session is already running.", parent=window)
                continue
            last_process_filter = ""
            _update_process_list(window, process_list, last_process_filter)
            if len(auto_sessions) > 0:
                selected_session = auto_sessions[0]
                window[SGKeys.SESSION_LOGS].update(value="", autoscroll=True)
                selected_session.restart()
            else:
                if len(sessions) == _MAX_SESSIONS:
                    show_popup_text(
                        "Error",
                        "Maximum amount of sessions reached.\nKill old session to start new one.",
                        parent=window,
                    )
                    continue
                selected_session = Session(None, [int(session.pid.value) for session in sessions])
                sessions.append(selected_session)
                _update_sessions_view(window, sessions)
            _update_session_overview(window, selected_session)
        elif event.startswith(SGKeys.SESSION_PREFIX):
            selected_session = sessions[SGKeys.get_session_idx(event)]
            _update_session_overview(window, selected_session)
        elif event == SGKeys.SESSION_KILL_BUTTON:
            if show_popup_text(
                "Confirm session kill",
                f"Are you sure to kill session: {selected_session.get_name()}?",
                ok_label="Yes",
                cancel_button=True,
                cancel_label="No",
                parent=window,
            ):
                sessions = [session for session in sessions if session.pid.value != selected_session.pid.value]
                _kill_session(selected_session)
                _update_sessions_view(window, sessions)
                for session in sessions:
                    if session._is_auto:
                        session.pids_to_skip.remove(int(selected_session.pid.value))
                selected_session = None
                _update_session_overview(window, selected_session)
        elif event == SGKeys.SESSION_RESTART_BUTTON:
            window[SGKeys.SESSION_LOGS].update(value="", autoscroll=True)
            selected_session.restart()
        elif event == SGKeys.SESSION_CLOSE_OVERVIEW_BUTTON:
            selected_session = None
            _update_session_overview(window, selected_session)
        elif event == SGKeys.SESSION_LOGS_SCROLL_TOP_BUTTON:
            window[SGKeys.SESSION_LOGS].Widget.yview_moveto(0)
        elif event == SGKeys.SESSION_LOGS_SCROLL_BOT_BUTTON:
            window[SGKeys.SESSION_LOGS].Widget.yview_moveto(1)
        elif event == SGKeys.SESSION_LOGS_CLEAR_BUTTON:
            selected_session.clear_logs()
            window[SGKeys.SESSION_LOGS].update(value="", autoscroll=True)
        elif event == SGKeys.MENU_SETTINGS_OPTION:
            display_settings_window(window)
        elif event == SGKeys.MENU_PIPELINE_FORCE_DOWNLOAD_OPTION:
            verify_download(True, parent=window)
        elif event == SGKeys.MENU_PIPELINE_INSTALL_REQUIREMENTS_OPTION:
            settings = get_settings()
            local_path = settings[SettingsKeys.KEY_LOCAL_PYTHON_64 if _IS_64_BIT else SettingsKeys.KEY_LOCAL_PYTHON_32]
            if len(local_path) == 0:
                if show_popup_text(
                    "Error",
                    (
                        "Cannot install requirements due to empty local Python executable path.\n"
                        "Do you want to open settings and configure it now?"
                    ),
                    ok_label="Yes",
                    cancel_button=True,
                    cancel_label="No",
                    parent=window,
                ):
                    display_settings_window(window)
                continue
            install_requirements(local_path, window)
        elif event == SGKeys.MENU_UPDATE_OPTION:
            updated_executable, update_restart = try_update(True, updated_executable, parent=window)
            if update_restart:
                break
        elif event == SGKeys.MENU_ABOUT_OPTION:
            show_popup(
                "About",
                [
                    [sg.Image(data=_APP_ICON, size=(128, 128))],
                    [sg.Text(f"PyHook v{__version__}", justification="center")],
                    [sg.Text("(c) 2022 by Dominik Wojtasik", justification="center")],
                    [sg.Button("GitHub", size=(10, 1), pad=((0, 0), (10, 0)), key=SGKeys.ABOUT_GITHUB_BUTTON)],
                ],
                events={SGKeys.ABOUT_GITHUB_BUTTON: EventCallback(open_github, False)},
                min_width=275,
                parent=window,
            )

    running = False
    if ui_worker is not None:
        ui_worker.join()
    window.close()
    _close_app()
    if update_restart:
        delete_self_exe(updated_executable)
    sys.exit(0)
