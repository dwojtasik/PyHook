"""
utils for PyHook
~~~~~~~~~~~~~~~~~~~
Utils for pipeline creation
:copyright: (c) 2022 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""

import ctypes
import os
import sys
from subprocess import check_output
from typing import Any, Dict, List, Union
from os.path import abspath, dirname

# Runtime info
_LOCAL_PYTHON_EXE = None
_LOCAL_PATHS = []
_RUNTIME_HANDLE = None

_IS_64_BIT = sys.maxsize > 2**32
_LOCAL_PYTHON_ENV = "LOCAL_PYTHON"
_LOCAL_PYTHON_ENV_32 = "LOCAL_PYTHON_32"
_LOCAL_PYTHON_ENV_64 = "LOCAL_PYTHON_64"
_RUNTIME_DLL = "vcruntime140_1.dll"
_MEIPASS = "_MEIPASS"


def _is_frozen_bundle() -> bool:
    """Checks if app is running in PyInstaller frozen bundle.

    Returns:
        bool: True if app is running in PyInstaller frozen bundle.
    """
    return getattr(sys, "frozen", False) and hasattr(sys, _MEIPASS)


_DIR = os.getcwd() if _is_frozen_bundle() else dirname(abspath(__file__))


class _LocalPython:
    """Allows to use local Python setup.

    Go back to bundled only env by calling close() or using it in a with statement.

    _sys_path (List[str]): Frozen sys.path list from bundled Python.
    _added_paths (List[os._AddedDllDirectory]): List of added paths to sys.path and DLL search path.
        All DLL directory handles will be closed by calling close() on _LocalPython object.
    """

    def __init__(self):
        self._sys_path = [p for p in sys.path]
        self._added_paths = [self._add_path(p) for p in _LOCAL_PATHS if self._is_valid_path(p)]

    def _is_valid_path(self, path: str) -> bool:
        """Checks if path is valid to be used in bundled sys.path.

        Args:
            path (str): The path to be checked.

        Returns:
            bool: True if path is valid to use.
        """
        return len(path) > 0 and not path.endswith(".zip")

    def _add_path(self, path: str) -> os._AddedDllDirectory:
        """Adds path to bundled sys.path and DLL search path.

        Args:
            path (str): The path to be added.

        Returns:
            os._AddedDllDirectory: DLL directory handle to be closed after usage.
        """
        sys.path.append(path)
        return os.add_dll_directory(path)

    def close(self):
        """Restores bundled sys.path and closes all DLL directory handles."""
        for path in self._added_paths:
            path.close()
        sys.path.clear()
        sys.path.extend(self._sys_path)
        self._sys_path = None
        self._added_paths = None

    def __enter__(self) -> "_LocalPython":
        """Called at the start of with block.

        Returns:
            _LocalPython: Local Python handle.
        """
        return self

    def __exit__(self, *args) -> None:
        """Called at the end of with block.

        Closes local Python handle.
        """
        self.close()


def _set_local_python() -> None:
    """Reads and stores local Python executable path and local Python sys.path.

    Firstly checks user defined envs "LOCAL_PYTHON_64", "LOCAL_PYTHON_32" and "LOCAL_PYTHON".
    If not set it will try to read executable path from python3 binary that is set in path.
    When executable is found it will try to read local Python sys.path.
    """
    # pylint: disable=global-statement
    global _LOCAL_PYTHON_EXE, _LOCAL_PATHS
    path_from_env = os.getenv(
        _LOCAL_PYTHON_ENV_64 if _IS_64_BIT else _LOCAL_PYTHON_ENV_32, os.getenv(_LOCAL_PYTHON_ENV, None)
    )
    if path_from_env is None:
        try:
            _LOCAL_PYTHON_EXE = check_output("python3 -c \"import sys;print(sys.executable,end='')\"").decode("utf-8")
        except FileNotFoundError as ex:
            raise ValueError(
                "Local Python3 executable not found. Please update system path or set LOCAL_PYTHON env."
            ) from ex
    else:
        try:
            check_output(f'{path_from_env} -c "1"')
            _LOCAL_PYTHON_EXE = path_from_env
        except FileNotFoundError as ex:
            raise ValueError("LOCAL_PYTHON is pointing to invalid Python3 executable.") from ex
    _LOCAL_PATHS = (
        check_output(f"{_LOCAL_PYTHON_EXE} -c \"import sys;print(';'.join(sys.path),end='')\"")
        .decode("utf-8")
        .split(";")
    )


def use_local_python() -> _LocalPython:
    """Allows to use local Python setup in pipelines.

    Use it with when statement for imports, e.g.
    with use_local_python():
        import moduleA
        from moduleB import X
        ...

    Returns:
        _LocalPython: Local Python handle. When not closed it allows to load modules from local setup.
    """
    # pylint: disable=global-statement
    global _RUNTIME_HANDLE
    # For 64-bit vcruntime needs additional library to be loaded
    if _IS_64_BIT and _RUNTIME_HANDLE is None and _is_frozen_bundle():
        _RUNTIME_HANDLE = ctypes.cdll[f"{getattr(sys, _MEIPASS)}\\{_RUNTIME_DLL}"]
    if _LOCAL_PYTHON_EXE is None:
        _set_local_python()
    return _LocalPython()


def resolve_path(file_path: str) -> str:
    """Returns absolute path to pipeline resource file.

    Args:
        file_path (str): Realtive path to resource file.

    Returns:
        str: Absolute path to resource file.
    """
    return f"{_DIR}/pipelines/{file_path}"


def build_variable(
    value: Any, min_value: Any = None, max_value: Any = None, step: Any = None, tooltip: str = None
) -> List[Union[Any, float, float, float, str]]:
    """Builds variable data list for pipeline settings.

    Args:
        value (Any): The initial value.
        min_value (Any, optional): Minimum value for variable. Defaults to None.
        max_value (Any, optional): Maximum value for variable. Defaults to None.
        step (Any, optional): Step between min->max values. Defaults to None.
        tooltip (str, optional): Tooltip describing variable. Defaults to None.

    Raises:
        ValueError: When value has not allowed type.

    Returns:
        List[Any, float, float, float, str]: The variable data list.
    """
    if isinstance(value, (bool, int, float)):
        return [
            value,
            None if min_value is None else float(min_value),
            None if max_value is None else float(max_value),
            None if step is None else float(step),
            tooltip,
        ]
    raise ValueError(f"Invalid type for value: {type(value)}. Allowed types: [bool, int, float].")


def read_value(settings: Dict[str, List[Union[Any, float, float, float, str]]], key: str) -> Any:
    """Reads value from settings.

    Args:
        settings (Dict[str, List[Union[Any, float, float, float, str]]]): Settings dictionary.
        key (str): Variable name to be read.

    Returns:
        Any: Value of the variable
    """
    return settings[key][0]
