"""
gui.image for PyHook
~~~~~~~~~~~~~~~~~~~~~~~
GUI Image utilities for PyHook
:copyright: (c) 2023 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""

import textwrap
from io import BytesIO
from typing import Tuple

from PIL import Image, ImageDraw, ImageFont, ImageOps

from gui.style import *  # pylint: disable=wildcard-import, unused-wildcard-import

# Default image resampling type.
_RESAMPLING = Image.Resampling.LANCZOS
# Number of bytes before raw RGBA ICO data
_ICO_HEADER_SHIFT = 40


def get_as_buffer(img: Image.Image) -> bytes:
    """Saves image to buffer and returns it's byte content.

    Args:
        img (Image.Image): Image object.

    Returns:
        bytes: Byte content.
    """
    bio = BytesIO()
    img.save(bio, format="PNG")
    return bio.getvalue()


def get_img(path: str, thumb_size: Tuple[int, int] = None) -> bytes:
    """Returns image data from given image path.

    Args:
        path (str): Path to image file.
        thumb_size (Tuple[int, int], optional): Size of additional thumbnail in format (width, height).
            Defaults to None.

    Returns:
        bytes: Image data.
    """
    img = Image.open(path)
    if thumb_size is not None:
        img.thumbnail(thumb_size, _RESAMPLING)
    return get_as_buffer(img)


def format_raw_data(raw_data: bytes, thumb_size: Tuple[int, int] = None) -> Image.Image:
    """Formats raw image data into RGBA PNG format.

    Args:
        raw_data (bytes): Raw bytes read from memory in PNG or ICO format.
        thumb_size (Tuple[int, int], optional): Size of additional thumbnail in format (width, height).
            Defaults to None.

    Returns:
        Image.Image: Formatted image.
    """
    if raw_data[1:4] == b"PNG":
        img = Image.open(BytesIO(raw_data))
    else:
        px_res = int((len(raw_data) // 4) ** 0.5) // 8 * 8
        size = px_res * px_res * 4
        img = Image.frombuffer(
            "RGBA", (px_res, px_res), raw_data[_ICO_HEADER_SHIFT : _ICO_HEADER_SHIFT + size], "raw", "BGRA", 0, 1
        )
        img = ImageOps.flip(img)
    if thumb_size is not None:
        img.thumbnail(thumb_size, _RESAMPLING)
    return img


def get_button_image_template() -> Image.Image:
    """Returns button clear image.

    Returns:
        Image.Image: Button clear image.
    """
    return Image.new("RGBA", SESSION_BUTTON_SIZE, color="#00000000")


def get_button_image(icon: Image.Image | bytes | None, text: str, pid: int | None) -> Image.Image:
    """Returns button image with given icon, text and pid.

    Args:
        icon (Image.Image | bytes | None): Icon image or raw byte data to be displayed.
            Could be omitted to display only given text.
        text (str): Text to be displayed.
        pid (int | None): Process identifier to be displayed.

    Returns:
        Image.Image: Prepared button image.
    """
    if icon is not None:
        if isinstance(icon, bytes):
            icon = format_raw_data(icon)
        icon = icon.resize(SESSION_BUTTON_ICON_SIZE, _RESAMPLING)

    button_w, button_h = SESSION_BUTTON_SIZE
    button_img = get_button_image_template()
    if icon is not None:
        icon_w, icon_h = SESSION_BUTTON_ICON_SIZE
        button_img.paste(
            icon, ((button_w - icon_w) // 2, (button_h - icon_h) // 2 + SESSION_BUTTON_ICON_CENTER_OFFSET_Y)
        )

    font_size = MAX_BUTTON_TEXT_SIZE
    font = ImageFont.truetype(FONT_IMG_NAME_DEFAULT, font_size)
    text_lines = textwrap.wrap(text, width=15)
    longest_line = sorted(text_lines, key=lambda line: font.getbbox(line)[2], reverse=True)[0]
    while font.getbbox(longest_line)[2] > 0.95 * button_w:
        font_size -= 1
        font = ImageFont.truetype(FONT_IMG_NAME_DEFAULT, font_size)

    canvas = ImageDraw.Draw(button_img)
    offset_y = SESSION_BUTTON_TEXT_CENTER_OFFSET_Y
    for line in text_lines:
        _, _, line_w, line_h = font.getbbox(line)
        canvas.text(
            ((button_w - line_w) / 2, (button_h - line_h * len(text_lines)) / 2 + offset_y),
            line,
            font=font,
            fill=SESSION_BUTTON_TEXT_COLOR,
        )
        offset_y += line_h
    if pid:
        pid_text = f"[{pid}]"
        font = ImageFont.truetype("arial.ttf", 12)
        canvas.text((button_w - font.getbbox(pid_text)[2] - 1, 1), pid_text, font=font, fill="#ffffff")
    return button_img
