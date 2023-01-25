"""
main for PyHook
~~~~~~~~~~~~~~~~~~~~~~~
PyHook app entrypoint
:copyright: (c) 2022 by Dominik Wojtasik.
:license: MIT, see LICENSE for more details.
"""

from multiprocessing import freeze_support

from gui.app import gui_main

if __name__ == "__main__":
    freeze_support()
    gui_main()
