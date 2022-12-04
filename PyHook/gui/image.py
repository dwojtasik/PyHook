"""
gui.image for PyHook
~~~~~~~~~~~~~~~~~~~~~~~
GUI Image utilities for PyHook
:copyright: (c) 2022 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""

from io import BytesIO
from typing import Tuple

from PIL import Image, ImageOps

# Default image resampling
_RESAMPLING = Image.Resampling.LANCZOS


def get_as_buffer(img: Image) -> bytes:
    """Saves image to buffer and returns it's byte content.

    Args:
        img (Image): Image object.

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


def format_raw_data(raw_data: bytes, thumb_size: Tuple[int, int] = None) -> bytes:
    """Formats image data into RGBA PNG format.

    Args:
        raw_data (bytes): Raw bytes read from memory.
        thumb_size (Tuple[int, int], optional): Size of additional thumbnail in format (width, height).
            Defaults to None.

    Returns:
        bytes: Formatted image data.
    """
    px_res = int((len(raw_data) // 4) ** 0.5) // 8 * 8
    img = Image.frombuffer("RGBA", (px_res, px_res), raw_data, "raw", "BGRA", 0, 1)
    img = ImageOps.flip(img)
    if thumb_size is not None:
        img.thumbnail(thumb_size, _RESAMPLING)
    return get_as_buffer(img)
