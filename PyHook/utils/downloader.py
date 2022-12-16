"""
utils.downloader for PyHook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Simple file downloader
:copyright: (c) 2022 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""

import re
from os.path import basename, exists, getsize
from typing import Callable
from urllib.parse import unquote, urlparse

import requests

# Default chunk size in bytes for stream downloading.
_CHUNK_SIZE = 4096
# Regex filename for Google Drive URLs.
_FILENAME_REGEX = re.compile(r"^.*?filename=\"(.*?)\";.*$")


def download_file(url: str, directory: str, callback: Callable[[int], None] = None) -> None:
    """Download file from given url and save it into directory.

    Args:
        url (str): The url to given file.
        directory (str): The directory to save downloaded file.
        callback (Callable[[int], None], optional): Optional callback for progress.
            Defaults to None.
    """
    response_stream = requests.get(url, stream=True, timeout=10)
    response_stream.raise_for_status()
    if url.startswith("https://drive.google.com"):
        filename = _FILENAME_REGEX.match(response_stream.headers["Content-Disposition"]).group(1)
    else:
        filename = unquote(basename(urlparse(url).path))
    filepath = f"{directory}\\{filename}"
    filesize = int(response_stream.headers["Content-Length"])
    byte_count = 0
    last_percent = 0
    if not exists(filepath) or getsize(filepath) != filesize:
        with open(filepath, "wb") as d_file:
            for chunk in response_stream.iter_content(_CHUNK_SIZE):
                d_file.write(chunk)
                byte_count += _CHUNK_SIZE
                if byte_count > filesize:
                    byte_count = filesize
                percent = byte_count / filesize * 100
                if callback is not None:
                    if percent > last_percent:
                        callback(percent)
                        last_percent = percent
