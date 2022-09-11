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
- JSON file with pipelines settings
- Local Python environment usage in pipeline code

Graphics API support
====================

+----------------+--------+-----------+------------+------------+------------+--------+
| PyHook version | OpenGL | DirectX 9 | DirectX 10 | DirectX 11 | DirectX 12 | Vulkan |
+================+========+===========+============+============+============+========+
| 32-bit         | ✔      | ✔         | ✔          | ✔          | ❌          | ❌      |
+----------------+--------+-----------+------------+------------+------------+--------+
| 64-bit         | ✔      | ✔         | ✔          | ✔          | ❌          | ❌      |
+----------------+--------+-----------+------------+------------+------------+--------+

Do note that multisampling is not supported by PyHook at all with any API.

Requirements
============

Runtime
-------
- `ReShade <https://reshade.me/>`_ >= 5.0.0
- `Python <https://www.python.org/>`_ == ``3.10.6 for 64-bit`` | ``3.10.4 for 32-bit`` (for pipelines only)
- `CUDA <https://developer.nvidia.com/cuda-11.3.0-download-archive>`_ == 11.3 (optional for AI pipelines only)
- | Only for specific pipelines: Any libraries that are required by pipeline code.
  | Do note that AI pipelines requires PyTorch which doesn't work on 32-bit system.

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

For custom pipelines (e.g. AI ones) install requirements and setup ENV variables that points to Python3 binary in required version.

Available ENV variables:

- LOCAL_PYTHON_32 (path to 32-bit Python)
- LOCAL_PYTHON_64 (path to 64-bit Python)
- LOCAL_PYTHON (fallback path if none of above is set)

Models for pipelines can be downloaded by links from ``download.txt`` that are supplied in their respective directory.

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
- Improved synchronization between PyHook and addon.
- Added OpenGL support.
- Added multiple texture formats support.
- Added logs removal from DLL loading.
- Added JSON settings for pipelines.
- Added combo box selection in settings UI.
- Added AI colorization pipeline example using https://github.com/richzhang/colorization
- Added AI Cartoon-GAN pipeline example using https://github.com/FilipAndersson245/cartoon-gan
- Added dynamic modules load from local Python environment.
- Added fallback to manual PID supply.
- Updated pipeline template.
- Added new callbacks for settings changes (before and after change).
- Added ReShade UI for pipeline settings in ImGui.
- Added pipeline utils to faster pipeline creation.
- Added dynamic pipeline variables parsing.
- Added shared memory segment for pipeline settings.
- Added AI style transfer pipeline example using https://github.com/mmalotin/pytorch-fast-neural-style-mobilenetV2
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
