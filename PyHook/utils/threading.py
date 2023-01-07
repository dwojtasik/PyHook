"""
utils.threading for PyHook
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Threading utils for PyHook
:copyright: (c) 2023 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""

from threading import Thread
from typing import Any, Callable, List


class BackgroundTask(Thread):
    """Background task for PyHook.

    _running (bool): Flag if task is still running.
    _output (Any): Output of task.
    _exception (Exception): Exception that occurred during task execution.
    task (Callable[[List[Any]], Any]): Task to be called.
    args (List[Any]): Argument list for task.
    """

    def __init__(
        self,
        task: Callable[[List[Any]], Any],
        args: List[Any],
    ):
        self._running = False
        self._output: Any = None
        self._exception: Exception = None
        self.task = task
        self.args = args
        Thread.__init__(self)

    def is_running(self) -> bool:
        """Returns flag if background task is still running.

        Returns:
            bool: Flag if background task is still running.
        """
        return self._running

    def get_output(self) -> Any:
        """Returns background task output or raises it's exception.

        Returns:
            Any: Background task output.

        Raises:
            Exception: When any exception occurred during task execution.
        """
        if self._exception is not None:
            raise self._exception
        return self._output

    def run(self) -> None:
        """Runs background task and stores it's output and exception in variables."""
        try:
            self._output = self.task(*self.args)
        except Exception as ex:
            self._exception = ex
        self._running = False

    def start(self) -> None:
        """Starts the background task."""
        self._running = True
        Thread.start(self)
