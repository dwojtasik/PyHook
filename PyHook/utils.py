"""
utils for PyHook
~~~~~~~~~~~~~~~~~~~
Utils for pipeline creation
:copyright: (c) 2022 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""

from os.path import abspath, dirname
from typing import Any, Dict, List, Union

_DIR = dirname(abspath(__file__))

def resolve_path(file_path: str) -> str:
    """Returns absolute path to pipeline resource file.

    Args:
        file_path (str): Realtive path to resource file.

    Returns:
        str: Absolute path to resource file.
    """
    return f'{_DIR}/pipelines/{file_path}'

def build_variable(value: Any, min: Any = None, max: Any = None, step: Any = None, tooltip: str = None) -> List[Union[Any, float, float, float, str]]:
    """Builds variable data list for pipeline settings.

    Args:
        value (Any): The initial value.
        min (Any, optional): Minimum value for variable. Defaults to None.
        max (Any, optional): Maximum value for variable. Defaults to None.
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
            None if min is None else float(min),
            None if max is None else float(max),
            None if step is None else float(step),
            tooltip
        ]
    raise ValueError(f'Invalid type for value: {type(value)}. Allowed types: [bool, int, float].')

def read_value(settings: Dict[str, List[Union[Any, float, float, float, str]]], key: str) -> Any:
    """Reads value from settings.

    Args:
        settings (Dict[str, List[Union[Any, float, float, float, str]]]): Settings dictionary.
        key (str): Variable name to be read.

    Returns:
        Any: Value of the variable
    """
    return settings[key][0]
