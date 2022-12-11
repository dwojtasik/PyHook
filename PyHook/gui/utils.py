"""
gui.utils for PyHook
~~~~~~~~~~~~~~~~~~~~~~~
GUI utilities for PyHook
:copyright: (c) 2022 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""
import textwrap
from typing import Callable, Dict, List

import PySimpleGUI as sg

from gui.keys import SGKeys
from gui.style import FONT_CONSOLE, FONT_SMALL_DEFAULT


class EventCallback:
    """Callback for UI event.

    callback (Callable[[], None]): Callback to execute.
    close_window (bool, optional): Flag if windows should be closed after event.
        Defaults to False.
    """

    def __init__(self, callback: Callable[[], None], close_window: bool = False):
        self.callback = callback
        self.close_window = close_window


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
    min_width: int = None,
) -> bool:
    """Displays customized popup window.

    Args:
        title (str): Popup title.
        layout (List[List[sg.Column]]): Popup layout without buttons section.
        ok_label (str, optional): Label for OK button. Defaults to "OK".
        cancel_button (bool, optional): Flag if cancel button should be displayed. Defaults to False.
        cancel_label (str, optional): Label for cancel button. Defaults to "Cancel".
        events (Dict[str, EventCallback], optional): Map of event keys with callbacks. Defaults to None.
        min_width (int, optional): Minimum width of popup window in pixels. Defaults to None.

    Returns:
        bool: Flag if OK button was pressed.
    """
    buttons = [sg.Button(ok_label, size=(8, 1), pad=((5, 5), (10, 5)), key=SGKeys.POPUP_KEY_OK_BUTTON)]
    if cancel_button:
        buttons.append(sg.Button(cancel_label, size=(8, 1), pad=((5, 5), (10, 5)), key=SGKeys.POPUP_KEY_CANCEL_BUTTON))
    layout.append(buttons)
    if min_width is not None:
        layout.append([sg.Image(size=(min_width, 0), pad=(0, 0))])
    popup = sg.Window(
        title,
        layout,
        font=FONT_SMALL_DEFAULT,
        element_justification="center",
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
        if event in (
            sg.WIN_CLOSED,
            SGKeys.EXIT,
            SGKeys.POPUP_KEY_OK_BUTTON,
            SGKeys.POPUP_KEY_CANCEL_BUTTON,
        ):
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


def show_popup_exception(
    title: str,
    text: str,
    ex: Exception,
    ok_label: str = "OK",
    cancel_button: bool = False,
    cancel_label: str = "Cancel",
) -> bool:
    """Displays customized popup window.

    Args:
        title (str): Popup title.
        text (str): Popup text.
        ex (Exception): Exception to be displayed.
        ok_label (str, optional): Label for OK button. Defaults to "OK".
        cancel_button (bool, optional): Flag if cancel button should be displayed. Defaults to False.
        cancel_label (str, optional): Label for cancel button. Defaults to "Cancel".

    Returns:
        bool: Flag if OK button was pressed.
    """
    ex_lines = "\n".join(textwrap.wrap(f"Error: {ex}", width=50, break_on_hyphens=False))
    layout = [
        [sg.Text(text, justification="center")],
        [sg.Text(ex_lines, justification="left", font=FONT_CONSOLE)],
    ]
    return show_popup(title, layout, ok_label, cancel_button, cancel_label)
