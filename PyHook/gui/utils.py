"""
gui.utils for PyHook
~~~~~~~~~~~~~~~~~~~~~~~
GUI utilities for PyHook
:copyright: (c) 2022 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""

from typing import Callable, Dict, List

import PySimpleGUI as sg

from session import ProcessInfo
from gui.keys import SGKeys
from gui.style import FONT_SMALL


class EventCallback:
    """Callback for UI event.

    callback (Callable): Callback to execute.
    close_window (bool, optional): Flag if windows should be closed after event.
        Defaults to False.
    """

    def __init__(self, callback: Callable, close_window: bool = False):
        self.callback = callback
        self.close_window = close_window


def to_combo_list(process_list: List[ProcessInfo], filter_string: str = None) -> List[str]:
    """Filters list of processes to combo list.

    Args:
        process_list (List[ProcessInfo]): List of processes.
        filter_string (str, optional): Filter to be applied. Defaults to None.

    Returns:
        List[str]: List of combo strings.
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


def with_border(elem: sg.Element, color: str, visible: bool = True) -> sg.Column:
    """Decorates given UI element with border in given color.

    Args:
        elem (sg.Element): GUI element to decorate.
        color (str): Border color.
        visible (bool): Flag if element with border should be visible.

    Returns:
        sg.Column: Decorated UI element.
    """
    return sg.Column(
        [[elem]], pad=(3, 3), background_color=color, key=elem.key + SGKeys.BORDER_SUFFIX, visible=visible
    )


def show_popup(
    title: str,
    layout: List[List[sg.Column]],
    ok_label: str = "OK",
    cancel_button: bool = False,
    cancel_label: str = "Cancel",
    events: Dict[str, EventCallback] = None,
) -> bool:
    """Displays customized popup window.

    Args:
        title (str): Popup title.
        layout (List[List[sg.Column]]): Popup layout without buttons section.
        ok_label (str, optional): Label for OK button. Defaults to "OK".
        cancel_button (bool, optional): Flag if cancel button should be displayed. Defaults to False.
        cancel_label (str, optional): Label for cancel button. Defaults to "Cancel".
        events (Dict[str, EventCallback], optional): Map of event keys with callbacks. Defaults to None.

    Returns:
        bool: Flag if OK button was pressed.
    """
    buttons = [sg.Button(ok_label, size=(8, 1), font=FONT_SMALL, key=SGKeys.POPUP_KEY_OK_BUTTON)]
    if cancel_button:
        buttons.append(sg.Button(cancel_label, size=(8, 1), font=FONT_SMALL))
    layout.append(buttons)
    popup = sg.Window(
        title,
        layout,
        element_justification="c",
        disable_minimize=True,
        modal=True,
        keep_on_top=True,
        finalize=True,
        location=(None, None),
    )
    if events is None:
        event, _ = popup.read(close=True)
        return event == SGKeys.POPUP_KEY_OK_BUTTON
    result = False
    while True:
        event, _ = popup.read()
        if event in (sg.WIN_CLOSED, SGKeys.MENU_EXIT_OPTION, SGKeys.POPUP_KEY_OK_BUTTON):
            result = event == SGKeys.POPUP_KEY_OK_BUTTON
            break
        if event in events:
            events[event].callback()
            if events[event].close_window:
                break
    popup.close()
    return result


def show_popup_text(
    title: str,
    text: str,
    ok_label: str = "OK",
    cancel_button: bool = False,
    cancel_label: str = "Cancel",
) -> bool:
    """Displays customized popup window.

    Args:
        title (str): Popup title.
        text (str): Popup text.
        ok_label (str, optional): Label for OK button. Defaults to "OK".
        cancel_button (bool, optional): Flag if cancel button should be displayed. Defaults to False.
        cancel_label (str, optional): Label for cancel button. Defaults to "Cancel".

    Returns:
        bool: Flag if OK button was pressed.
    """
    layout = [[sg.Text(text, justification="center")]]
    return show_popup(title, layout, ok_label, cancel_button, cancel_label)
