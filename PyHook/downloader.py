"""
downloader for PyHook
~~~~~~~~~~~~~~~~~~~~~~~
Simple file downloader
:copyright: (c) 2022 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""

import logging
import re
from os.path import basename, exists, getsize
from urllib.parse import unquote, urlparse

import requests

# Default chunk size in bytes for stream downloading.
_CHUNK_SIZE = 4096

# Regex filename for Google Drive URLs.
_FILENAME_REGEX = re.compile(r"^.*?filename=\"(.*?)\";.*$")

# Steps displayed in progress bar for downloading.
_DOWNLOAD_PROGRESS_STEPS = 20


def download_file(url: str, directory: str, logger: logging.Logger = None) -> None:
    """Download file from given url and save it into directory.

    Args:
        url (str): The url to given file.
        directory (str): The directory to save downloaded file.
        logger (logging.Logger, optional): Optional logger to log progress.
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
    step = -1
    if not exists(filepath) or getsize(filepath) != filesize:
        with open(filepath, "wb") as d_file:
            if logger is not None:
                logger.info(f"--- Downloading file: {filename}")
            for chunk in response_stream.iter_content(_CHUNK_SIZE):
                d_file.write(chunk)
                byte_count += _CHUNK_SIZE
                if byte_count > filesize:
                    byte_count = filesize
                percent = byte_count / filesize * 100
                if logger is not None:
                    actual_step = int(percent // (100 // _DOWNLOAD_PROGRESS_STEPS))
                    if actual_step > step:
                        step = actual_step
                        logger.info(f"---- Progress: [{'|' * step}{' ' * (_DOWNLOAD_PROGRESS_STEPS - step)}]")
