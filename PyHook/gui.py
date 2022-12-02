"""
gui for PyHook
~~~~~~~~~~~~~~~~~~~~~~~
GUI for PyHook
:copyright: (c) 2022 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""

import atexit
import sys
from multiprocessing import freeze_support
from typing import List

import PySimpleGUI as sg

from _version import __version__
from gui_utils import show_popup, to_combo_list, with_border
from keys import SGKeys
from session import ProcessInfo, Session, get_process_list

# Default font
_FONT_DEFAULT = ("Arial", 16)
# Smaller default font
_FONT_MID_DEFAULT = ("Arial", 14)
# Default monospace font
_FONT_MONO_DEFAULT = ("Consolas", 14)
# Default console font
_FONT_CONSOLE = ("Consolas", 10)
# Maximum amount of sessions
_MAX_SESSIONS = 15

# Set default theme
sg.theme("DarkBlue")


def _get_sessions_layout() -> List[List[sg.Column]]:
    """Returns session list layout.

    Each session is displayed as button. Maximum of 3 sessions displayed per row.

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
                sg.Button("", size=(9, 6), pad=(3, 3), tooltip="", key=SGKeys.get_session_key(i)),
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


def _update_process_list(window: sg.Window, process_list: List[ProcessInfo], filter_string: str) -> None:
    """Updates process list combo box.

    Args:
        window (sg.Window): Parent window.
        process_list (List[ProcessInfo]): List of new process info.
        filter_string (str): Filter to be applied. For empty string filtering will be ommited.
    """
    window[SGKeys.PROCESS_LIST].update(
        value=filter_string,
        values=to_combo_list(process_list, filter_string if filter_string else None),
    )


def _update_sessions_active_view(window: sg.Window, sessions: List[Session], selected_session: Session | None) -> None:
    """Updates sessions list view after dynamic change from subprocess.

    For running session border color is set to green. Otherwise red.
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
        window[session_key].update(text=f"{sessions[i].get_name()}")
        window[session_key].set_tooltip(
            f"Process: {sessions[i].get_name()}\nStatus: {'Running' if running else 'Exited'}"
        )
        if selected_session is not None and selected_session.pid.value == session.pid.value:
            window[SGKeys.SESSION_RESTART_BUTTON].update(disabled=running)


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
            window[session_key].update(text=f"{sessions[i].get_name()}")
            window[session_key].set_tooltip(
                f"Process: {sessions[i].get_name()}\nStatus: {'Running' if running else 'Exited'}"
            )
        else:
            window[session_key].update(text="")
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


# Application UI layout
_APP_LAYOUT = [
    [
        sg.Text("Process"),
        sg.Combo(
            [],
            key=SGKeys.PROCESS_LIST,
            enable_events=True,
            font=_FONT_MONO_DEFAULT,
            size=(50, 1),
            tooltip="Process to inject PyHook",
        ),
        sg.Button("\u274C", key=SGKeys.INJECT_CLEAR, font=_FONT_MONO_DEFAULT, size=(2, 1), tooltip="Clear input"),
        sg.Button(
            "\u21BB",
            key=SGKeys.PROCESS_RELOAD,
            font=_FONT_MONO_DEFAULT,
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
                        font=_FONT_MID_DEFAULT,
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
                        font=_FONT_MONO_DEFAULT,
                        size=(2, 1),
                        tooltip="Restart exited session",
                        disabled=True,
                        visible=True,
                    ),
                    sg.Button(
                        "\u274C",
                        key=SGKeys.SESSION_CLOSE_OVERVIEW_BUTTON,
                        font=_FONT_MONO_DEFAULT,
                        size=(2, 1),
                        tooltip="Close overview",
                        visible=True,
                    ),
                ],
                [
                    sg.Multiline(
                        "",
                        font=_FONT_CONSOLE,
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
                                    font=_FONT_MONO_DEFAULT,
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
                                    font=_FONT_MONO_DEFAULT,
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


def _main():
    """App entrypoint"""

    # Last read process list
    process_list = get_process_list()
    # Last process filter string
    last_process_filter = ""
    # Last selected PID
    last_pid: int = None
    # List of active sessions
    sessions: List[Session] = []
    # Selected session to display overview
    selected_session: Session = None

    def close_all_sessions():
        """Closes all PyHook sessions on app exit."""
        for session in sessions:
            session.close()

    atexit.register(close_all_sessions)

    # Application window
    window = sg.Window(
        f"PyHook v{__version__} (c) 2022 by Dominik Wojtasik",
        _APP_LAYOUT,
        font=_FONT_DEFAULT,
        finalize=True,
    )

    _update_sessions_view(window, sessions)
    _update_session_overview(window, selected_session)
    _update_process_list(window, process_list, "")

    while True:
        event, values = window.read(timeout=1000 / 60)
        if event in (sg.WIN_CLOSED, "Exit"):
            break
        if event == sg.TIMEOUT_EVENT:
            if last_process_filter != values[SGKeys.PROCESS_LIST]:
                last_process_filter = values[SGKeys.PROCESS_LIST]
                _update_process_list(window, process_list, last_process_filter)
                last_pid = None
        elif event == SGKeys.PROCESS_LIST:
            last_process_filter = values[SGKeys.PROCESS_LIST]
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
                show_popup("Error", "First select process to inject PyHook.")
                continue
            if any(session.pid.value == last_pid for session in sessions):
                show_popup("Error", "Session with given PID already exists.")
                continue
            if len(sessions) == _MAX_SESSIONS:
                show_popup("Error", "Maximum amount of sessions reached.\nKill old session to start new one.")
                continue
            last_process_filter = ""
            process_info = ProcessInfo.from_pid(last_pid)
            if process_info is None:
                process_list = get_process_list()
                _update_process_list(window, process_list, last_process_filter)
                show_popup("Error", "Process does not exists anymore.")
                continue
            _update_process_list(window, process_list, last_process_filter)
            selected_session = Session(process_info)
            sessions.append(selected_session)
            _update_sessions_view(window, sessions)
            _update_session_overview(window, selected_session)
        elif event == SGKeys.INJECT_AUTO:
            auto_sessions: List[Session] = list(filter(lambda session: session.pid.value == -1, sessions))
            if len(auto_sessions) > 0 and auto_sessions[0].is_running():
                show_popup("Error", "Automatic session is already running.")
                continue
            last_process_filter = ""
            _update_process_list(window, process_list, last_process_filter)
            if len(auto_sessions) > 0:
                selected_session = auto_sessions[0]
                window[SGKeys.SESSION_LOGS].update(value="", autoscroll=True)
                selected_session.restart()
            else:
                if len(sessions) == _MAX_SESSIONS:
                    show_popup("Error", "Maximum amount of sessions reached.\nKill old session to start new one.")
                    continue
                selected_session = Session()
                sessions.append(selected_session)
                _update_sessions_view(window, sessions)
            _update_session_overview(window, selected_session)
        elif event.startswith(SGKeys.SESSION_PREFIX):
            selected_session = sessions[SGKeys.get_session_idx(event)]
            _update_session_overview(window, selected_session)
        elif event == SGKeys.SESSION_KILL_BUTTON:
            selected_session.close()
            sessions = [session for session in sessions if session.pid.value != selected_session.pid.value]
            _update_sessions_view(window, sessions)
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

        if selected_session is not None:
            if selected_session.should_update_logs():
                scroll_state = window[SGKeys.SESSION_LOGS].Widget.yview()
                window[SGKeys.SESSION_LOGS].update(value=selected_session.get_logs(), autoscroll=scroll_state[1] == 1)

        if any([session.should_update_ui() for session in sessions]):
            _update_sessions_active_view(window, sessions, selected_session)

    window.close()
    close_all_sessions()
    sys.exit(0)


if __name__ == "__main__":
    freeze_support()
    _main()
