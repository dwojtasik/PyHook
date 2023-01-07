"""
gui.utils for PyHook
~~~~~~~~~~~~~~~~~~~~~~~
GUI utilities for PyHook
:copyright: (c) 2023 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""
import textwrap
from typing import Callable, Dict, List

import PySimpleGUI as sg

from gui.style import FONT_CONSOLE, FONT_SMALL_DEFAULT
from gui.ui_keys import SGKeys


class EventCallback:
    """Callback for UI event.

    callback (Callable[[], None]): Callback to execute.
    close_window (bool, optional): Flag if window should be closed after event.
        Defaults to False.
    """

    def __init__(self, callback: Callable[[], None], close_window: bool = False):
        self.callback = callback
        self.close_window = close_window


def with_border(elem: sg.Element, color: str, visible: bool = True) -> sg.Column:
    """Decorates given UI element with border in given color.

    Args:
        elem (sg.Element): GUI element to decorate.
        color (str): Border color. Can be color name e.g. 'white' or hex value e.g. '#FFFFFF'.
        visible (bool): Flag if element with border should be visible.
            Defaults to True.

    Returns:
        sg.Column: Decorated UI element.
    """
    return sg.Column(
        [[elem]], pad=(3, 3), background_color=color, key=elem.key + SGKeys.BORDER_SUFFIX, visible=visible
    )


def center_in_parent(child: sg.Window, parent: sg.Window) -> None:
    """Centers child window in parent.

    If parent is not provided it will finalize child if needed.
    Child window will be hidden (by alpha transparency) while centering if not yet finalized.

    Args:
        child (sg.Window): Child window to center.
        parent (sg.Window): Parent window.
    """
    child_was_finalized = child.finalize_in_progress
    if parent is None:
        if not child_was_finalized:
            child.finalize()
        return
    parent_x, parent_y = parent.current_location(True)
    parent_w, parent_h = parent.current_size_accurate()
    child_x, child_y = parent_x + parent_w // 2, parent_y + parent_h // 2
    if child_was_finalized:
        child.move(child_x, child_y)
    else:
        child.Location = (child_x, child_y)
        child._AlphaChannel = 0
        child.finalize()
    child_w, child_h = child.current_size_accurate()
    child.move(child_x - child_w // 2, child_y - child_h // 2)
    child.refresh()
    if not child_was_finalized:
        child.reappear()


def show_popup(
    title: str,
    layout: List[List[sg.Column]],
    ok_label: str = "OK",
    cancel_button: bool = False,
    cancel_label: str = "Cancel",
    events: Dict[str, EventCallback] = None,
    min_width: int = None,
    parent: sg.Window = None,
    return_window: bool = False,
) -> sg.Window | bool:
    """Displays customized popup window.

    Args:
        title (str): Popup title.
        layout (List[List[sg.Column]]): Popup layout without buttons section.
        ok_label (str, optional): Label for OK button. Defaults to "OK".
        cancel_button (bool, optional): Flag if cancel button should be displayed. Defaults to False.
        cancel_label (str, optional): Label for cancel button. Defaults to "Cancel".
        events (Dict[str, EventCallback], optional): Map of event keys with callbacks. Defaults to None.
        min_width (int, optional): Minimum width of popup window in pixels. Defaults to None.
        parent (sg.Window, optional): Parent window for centering. Defaults to None.
        return_window (bool, optional): Flag if popup window should be returned for custom event loop.
            Defaults to False.

    Returns:
        sg.Window | bool: Popup window if return_window==True, otherwise flag if OK button was pressed.
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
    )
    center_in_parent(popup, parent)
    if return_window:
        return popup
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
    parent: sg.Window = None,
    return_window: bool = False,
) -> sg.Window | bool:
    """Displays customized popup window.

    Args:
        title (str): Popup title.
        text (str): Popup text.
        ok_label (str, optional): Label for OK button. Defaults to "OK".
        cancel_button (bool, optional): Flag if cancel button should be displayed. Defaults to False.
        cancel_label (str, optional): Label for cancel button. Defaults to "Cancel".
        parent (sg.Window, optional): Parent window for centering. Defaults to None.
        return_window (bool, optional): Flag if popup window should be returned for custom event loop.
            Defaults to False.

    Returns:
        sg.Window | bool: Popup window if return_window==True, otherwise flag if OK button was pressed.
    """
    layout = [[sg.Text(text, justification="center")]]
    return show_popup(title, layout, ok_label, cancel_button, cancel_label, parent=parent, return_window=return_window)


def show_popup_exception(
    title: str,
    text: str,
    ex: Exception,
    ok_label: str = "OK",
    cancel_button: bool = False,
    cancel_label: str = "Cancel",
    ex_width: int = 50,
    text_after: str = None,
    parent: sg.Window = None,
    return_window: bool = False,
) -> sg.Window | bool:
    """Displays customized popup window for exception.

    Args:
        title (str): Popup title.
        text (str): Popup text.
        ex (Exception): Exception to be displayed.
        ok_label (str, optional): Label for OK button. Defaults to "OK".
        cancel_button (bool, optional): Flag if cancel button should be displayed. Defaults to False.
        cancel_label (str, optional): Label for cancel button. Defaults to "Cancel".
        ex_width (int, optional): Length for wrapping exception stack. Defaults to 50.
        text_after (str, optional): Text to display after exception stack. Defaults to None.
        parent (sg.Window, optional): Parent window for centering. Defaults to None.
        return_window (bool, optional): Flag if popup window should be returned for custom event loop.
            Defaults to False.

    Returns:
        sg.Window | bool: Popup window if return_window==True, otherwise flag if OK button was pressed.
    """
    ex_lines = "\n".join(
        ["\n".join(textwrap.wrap(line, width=ex_width, break_on_hyphens=False)) for line in f"{ex}".split("\n")]
    ).rstrip()
    layout = [
        [sg.Text(text, justification="center")],
        [sg.Text(ex_lines, justification="left", font=FONT_CONSOLE)],
    ]
    if text_after is not None:
        layout.append([sg.Text(text_after, justification="center")])
    return show_popup(title, layout, ok_label, cancel_button, cancel_label, parent=parent, return_window=return_window)
