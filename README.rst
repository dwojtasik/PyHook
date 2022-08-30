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

- automatic ReShade detection and DLL validation
- automatic addon DLL injection
- shared memory as storage
- dynamic pipelines ordering in ReShade UI via addon
- pipeline lifecycle support (load/process/unload)
- frame processing in Python via ``numpy`` array

Requirements
============

Runtime
-------
- `ReShade <https://reshade.me/>`_ >= 5.0.0
- `Python <https://www.python.org/>`_ >= 3.7 (for pipelines)
- `numpy <https://pypi.org/project/numpy/>`_ (for pipelines)

Build
-----
- Same as runtime
- `Dear ImGui <https://github.com/ocornut/imgui>`_
- `Boost <https://www.boost.org/>`_ (used for Interprocessed shared memory)
- `psutil <https://pypi.org/project/psutil/>`_
- `pyinjector <https://pypi.org/project/pyinjector/>`_

Installation
============

You can download selected binary files from `Releases <https://github.com/dwojtasik/pyhook/releases/latest>`_.

1. Download executable and \*.addon files. Place both in same directory.
2. Start game with `ReShade <https://reshade.me/>`_ installed.
3. Start PyHook.exe.

If antyvirus detects PyHook as dangerous software add exception for it.
It is due to DLL injection capabilities.

Build
=====

To build PyHook simply run ``build.bat`` in `Anaconda <https://www.anaconda.com/>`_ Prompt.

To build PyHook addon download `Boost <https://www.boost.org/>`_ and place header files in Addon/include.
Then open \*.sln project and build given release.

History
=======
DEV / NEXT
----------
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
