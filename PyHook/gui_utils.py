"""
gui_utils for PyHook
~~~~~~~~~~~~~~~~~~~~~~~
GUI utilities for PyHook
:copyright: (c) 2022 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""

from typing import List

import PySimpleGUI as sg

from keys import SGKeys
from session import ProcessInfo


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
    buttons = [sg.Button(ok_label, key=SGKeys.POPUP_KEY_OK_BUTTON, size=(4, 1))]
    if cancel_button:
        buttons.append(sg.Button(cancel_label, size=(4, 1)))
    popup = sg.Window(
        title,
        [
            [sg.Text(text, justification="center")],
            buttons,
        ],
        element_justification="c",
        disable_minimize=True,
        modal=True,
        keep_on_top=True,
        finalize=True,
        location=(None, None),
    )
    event, _ = popup.read(close=True)
    return event == SGKeys.POPUP_KEY_OK_BUTTON
