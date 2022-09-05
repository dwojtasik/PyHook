===========================================
PyHook - Python hook for ReShade processing
===========================================

**PyHook** is an MIT licensed software, written in Python and C++, for access and
modification of ReShade back buffer.

PyHook consists of two elements:

- Python code that finds program with ReShade loaded, inject addon into it and allows to process frames in code.
- C++ addon DLL written using ReShade API that exposes back buffer via shared memory.

.. contents:: **Table of Contents**

Features
========

- Automatic ReShade detection and DLL validation
- Automatic addon DLL injection
- Shared memory as storage
- Dynamic pipelines ordering and settings in ReShade UI via addon
- Pipeline lifecycle support (load/process/unload)
- Pipeline settings callbacks (before and after change)
- Frame processing in Python via ``numpy`` array
- Local Python environment usage in pipeline code

Requirements
============

Runtime
-------
- `ReShade <https://reshade.me/>`_ >= 5.0.0
- `Python <https://www.python.org/>`_ == ``3.10.6 for 64-bit`` | ``3.10.4 for 32-bit`` (for pipelines only)
- Only for specific pipelines: Any libraries that are required by pipeline code

Build
-----
- Same as runtime, but for ReShade addon only included headers are needed
- `Boost <https://www.boost.org/>`_ == 1.80.0 (used for Boost.Interprocess shared memory)
- `Dear ImGui <https://github.com/ocornut/imgui>`_ == 1.86
- `NumPy <https://pypi.org/project/numpy/>`_ == 1.23.2
- `psutil <https://pypi.org/project/psutil/>`_ == 5.9.2
- `Pyinjector <https://pypi.org/project/pyinjector/>`_ == 1.1.0

EXE Build
---------
- Same as build
- `PyInstaller <https://pypi.org/project/pyinstaller/>`_ == 5.3
- `Python Standard Library List <https://pypi.org/project/stdlib-list/>`_ == 0.8.0

Installation
============

You can download selected binary files from `Releases <https://github.com/dwojtasik/pyhook/releases/latest>`_.

1. Download executable and \*.addon files. Place both in same directory.
2. Start game with `ReShade <https://reshade.me/>`_ installed.
3. Start PyHook.exe.

If antyvirus detects PyHook as dangerous software add exception for it because it is due to DLL injection capabilities.

Build
=====

To build PyHook simply run ``build.bat`` in `Anaconda <https://www.anaconda.com/>`_ Prompt.

If any Python package is missing try to update your conda environment and add conda-forge channel:

.. code-block:: powershell

    $ conda config --add channels conda-forge

To build PyHook addon download `Boost <https://www.boost.org/>`_ and place header files in Addon/include.
Then open \*.sln project and build given release.

History
=======
DEV / NEXT
----------
- Added dynamic modules load from local Python environment.
- Added fallback to manual PID supply.
- Updated pipeline template.
- Added new callbacks for settings changes (before and after change).
- Added ReShade UI for pipeline settings in ImGui.
- Added pipeline utils to faster pipeline creation.
- Added dynamic pipeline variables parsing.
- Added shared memory segment for pipeline settings.
- Added AI pipeline example using https://github.com/mmalotin/pytorch-fast-neural-style-mobilenetV2
- Added pipeline lifecycle support (load/process/unload).
- Added pipeline ordering and selection GUI in ReShade addon UI.
- Added shared memory for configuration.
- Added multisampling error in PyHook.
- Added pipeline processing for dynamic effects loading.
- Added shared data refresh on in-game settings changes.
- Disabled multisampling on swapchain creation.
- Fixed error display on app exit.

0.0.1 (2022-08-27)
------------------
- Initial version.
