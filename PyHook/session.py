"""
session for PyHook
~~~~~~~~~~~~~~~~~~~~~~~
PyHook subprocess sessions for PyHook
:copyright: (c) 2022 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""

import logging
import queue
from ctypes import c_char_p
from multiprocessing import Process, Queue, Value
from threading import Thread
from time import sleep
from typing import List

import psutil

from pyhook import pyhook_main

# Default log formatter.
_DEFAULT_FORMATTER = logging.Formatter()

# Name for unknown process.
_UNKNOWN_PROCESS = "Unknown"

# Name for automatic detection process.
_AUTO_NAME = "AUTO"


class ProcessInfo:
    """Contains basic info about process to be displayed.

    process (psutil.Process): Process object.
    pid (str): Process ID.
    name (str): Process name if available.
    """

    def __init__(self, process: psutil.Process):
        self.pid = process.pid
        _proc_name = process.name()
        self.name = _proc_name if len(_proc_name) > 0 else _UNKNOWN_PROCESS

    def get_combo_string(self) -> str:
        """Returns formatted string to be displayed in process combo box.

        Returns:
            str: Formatted string.
        """
        return f"{self.pid:<5} | {self.name}"

    @staticmethod
    def from_pid(pid: int) -> "ProcessInfo | None":
        """Returns process info from given process id.

        Args:
            pid (int): Process id.

        Returns:
            ProcessInfo | None: Process basic info or none if not exists.
        """
        if not psutil.pid_exists(pid):
            return None
        return ProcessInfo(psutil.Process(pid))


class Session:
    """PyHook Session object. Allows to communicate with PyHook instance.

    If process_info is not supplied automatic injection will be used.

    process_info (ProcessInfo, optional): Basic process info.
    pid (Value[int]): Shared integer process id.
    name (Value[bytes]): Shared bytes string process name.
    _is_auto (bool): Flag if PyHook session is running in automatic detection mode.
    _has_new_logs (bool): Flag if new logs are ready to be displayed.
    _has_ui_change (bool): Flag if PyHook session was stopped by any reason.
    _is_closed (bool): Flag if PyHook session was closed.
    _running (Value[bool]): Shared flag if process is running.
    _log_queue (Queue): Log queue from PyHook session.
    _log (str): Log from PyHook session.
    _process (Process): PyHook session subprocess.
    _worker (Thread): Thread worker for local tasks: updating logs, watching value changes.
    """

    def __init__(self, process_info: ProcessInfo | None = None):
        self.pid = Value("i", -1 if process_info is None else process_info.pid)
        self.name = Value(c_char_p, b"" if process_info is None else str.encode(process_info.name))
        self._is_auto = process_info is None
        self._has_new_logs = False
        self._has_ui_change = False
        self._is_closed = False
        self._running = Value("b", True)
        self._log_queue = Queue(-1)
        self._log = ""
        self._process = Process(
            target=pyhook_main,
            args=(self._running, self.pid, self.name, self._log_queue),
        )
        self._worker = Thread(target=self._update_self)
        self._process.start()
        self._worker.start()

    def get_name(self) -> str:
        """Returns process name as string.

        Returns:
            str: Process name.
        """
        if self._is_auto:
            return _AUTO_NAME
        return self.name.value.decode("utf-8") + f" [{self.pid.value}]"

    def is_running(self) -> bool:
        """Checks if PyHook session is still running.

        Returns:
            bool: PyHook session running flag.
        """
        return bool(self._running.value)

    def close(self) -> None:
        """Closes PyHook session subprocess and local worker."""
        if not self._is_closed:
            self._is_closed = True
            self._running.value = False
            self._process.join()
            self._worker.join()
            self.clear_logs()
            self._has_ui_change = True

    def restart(self) -> None:
        """Restarts PyHook session and local worker."""
        if not self._is_closed:
            self.close()
        self._has_new_logs = False
        self._has_ui_change = True
        self._is_closed = False
        self._running.value = True
        self._process = Process(
            target=pyhook_main,
            args=(self._running, self.pid, self.name, self._log_queue),
        )
        self._worker = Thread(target=self._update_self)
        self._process.start()
        self._worker.start()

    def get_logs(self) -> str:
        """Returns PyHook session logs.

        Returns:
            str: PyHook session logs.
        """
        return self._log

    def clear_logs(self) -> None:
        """Clears log."""
        self._log = ""

    def should_update_ui(self) -> bool:
        """Checks if UI should be updated.

        Returns:
            bool: Flag if UI should be updated.
        """
        output = self._has_ui_change
        self._has_ui_change = False
        return output

    def should_update_logs(self) -> bool:
        """Checks if log view should be updated.

        Returns:
            bool: Flag if log view should be updated.
        """
        output = self._has_new_logs
        self._has_new_logs = False
        return output

    def _update_self(self) -> None:
        """Updated self state based on subprocess data.
        Formats and appends new logs from queue to log string.
        Detects changes that require UI refresh.
        """
        while self._running.value:
            if self._is_auto:
                if self.pid.value != -1:
                    self._is_auto = False
                    self._has_ui_change = True
            try:
                self._log += _DEFAULT_FORMATTER.format(self._log_queue.get(block=False)) + "\n"
                self._has_new_logs = True
            except queue.Empty:
                pass
            sleep(1 / 60)
        self._has_ui_change = True
        while not self._log_queue.empty():
            self._log += _DEFAULT_FORMATTER.format(self._log_queue.get()) + "\n"
            self._has_new_logs = True


def get_process_list() -> List[ProcessInfo]:
    """Returns list of active processes.

    Returns:
        List[ProcessInfo]: List of active processes.
    """
    return [ProcessInfo(process) for process in psutil.process_iter()]
