===========================================
PyHook - Python hook for ReShade processing
===========================================

**PyHook** is an MIT licensed software, written in Python and C++, for access and
modification of ReShade back buffer.

PyHook consists of two elements:

- | Python code that finds process with ReShade loaded, injects addon into it
  | and allows to process frames in code via dynamic pipelines.
- | C++ addon DLL written using ReShade API that exposes back buffer via shared memory
  | and allows to configure dynamic pipelines in settings UI.

.. contents:: **Table of Contents**

Features
========

- Automatic ReShade detection and DLL validation
- Automatic addon DLL injection
- Shared memory as storage
- Dynamic pipelines ordering and settings in ReShade UI via addon
- Pipeline lifecycle support (load/process/unload)
- Pipeline settings callbacks (before and after change)
- Pipeline multipass - process frame multiple times in single pipeline
- Frame processing in Python via ``numpy`` array
- JSON file with pipelines settings
- Local Python environment usage in pipeline code
- Automatic pipeline files download

Graphics API support
====================

+----------------+--------+-----------+------------+------------+------------+--------+
| PyHook version | OpenGL | DirectX 9 | DirectX 10 | DirectX 11 | DirectX 12 | Vulkan |
+================+========+===========+============+============+============+========+
| 32-bit         | ✔      | ✔         | ✔          | ✔          | ✔*         | ✔*     |
+----------------+--------+-----------+------------+------------+------------+--------+
| 64-bit         | ✔      | ✔         | ✔          | ✔          | ✔*         | ✔*     |
+----------------+--------+-----------+------------+------------+------------+--------+

| \*ReShade version up to 5.4.2 has ImGui affected by pipelines for these API, due to bug:
| https://github.com/crosire/reshade/commit/d2d9ae4f6704208c74f7b8971c3d66bf01deec28
|

**Do note that multisampling is not supported by PyHook at all with any API.**

Pipeline results
================

.. list-table::
   :widths: 10 30 30 30
   :header-rows: 1

   * - Pipeline
     - GTA V
     - Crysis 3
     - Trek to Yomi
   * - None
     - .. image:: https://raw.githubusercontent.com/dwojtasik/PyHook/main/PyHook/pipelines/test_static_img/gta5.jpg
          :alt: PyHook/pipelines/test_static_img/gta5.jpg
     - .. image:: https://raw.githubusercontent.com/dwojtasik/PyHook/main/PyHook/pipelines/test_static_img/crysis3.jpg
          :alt: PyHook/pipelines/test_static_img/crysis3.jpg
     - .. image:: https://raw.githubusercontent.com/dwojtasik/PyHook/main/PyHook/pipelines/test_static_img/trek_to_yomi.jpg
          :alt: PyHook/pipelines/test_static_img/trek_to_yomi.jpg
   * - | `DNN Super Resolution <https://github.com/dwojtasik/PyHook/blob/main/PyHook/pipelines/ai_dnn_super_resolution.py>`_
       | Scale: 2
       | Model: FSRCNN
     - .. image:: https://raw.githubusercontent.com/dwojtasik/PyHook/main/docs/images/gta5_superres.jpg
          :target: https://dwojtasik.github.io/PyHook/?imgl=https://raw.githubusercontent.com/dwojtasik/PyHook/main/PyHook/pipelines/test_static_img/gta5.jpg&imgr=https://raw.githubusercontent.com/dwojtasik/PyHook/main/docs/images/gta5_superres.jpg&labl=Base&labr=DNN%20Super%20Resolution
          :alt: docs/images/gta5_superres.jpg
     - .. image:: https://raw.githubusercontent.com/dwojtasik/PyHook/main/docs/images/crysis3_superres.jpg
          :target: https://dwojtasik.github.io/PyHook/?imgl=https://raw.githubusercontent.com/dwojtasik/PyHook/main/PyHook/pipelines/test_static_img/crysis3.jpg&imgr=https://raw.githubusercontent.com/dwojtasik/PyHook/main/docs/images/crysis3_superres.jpg&labl=Base&labr=DNN%20Super%20Resolution
          :alt: docs/images/crysis3_superres.jpg
     - .. image:: https://raw.githubusercontent.com/dwojtasik/PyHook/main/docs/images/trek_to_yomi_superres.jpg
          :target: https://dwojtasik.github.io/PyHook/?imgl=https://raw.githubusercontent.com/dwojtasik/PyHook/main/PyHook/pipelines/test_static_img/trek_to_yomi.jpg&imgr=https://raw.githubusercontent.com/dwojtasik/PyHook/main/docs/images/trek_to_yomi_superres.jpg&labl=Base&labr=DNN%20Super%20Resolution
          :alt: docs/images/trek_to_yomi_superres.jpg
   * - | `Cartoon-GAN <https://github.com/dwojtasik/PyHook/blob/main/PyHook/pipelines/ai_cartoon_gan.py>`_
       | Scale: 1.0
     - .. image:: https://raw.githubusercontent.com/dwojtasik/PyHook/main/docs/images/gta5_cartoon.jpg
          :target: https://dwojtasik.github.io/PyHook/?imgl=https://raw.githubusercontent.com/dwojtasik/PyHook/main/PyHook/pipelines/test_static_img/gta5.jpg&imgr=https://raw.githubusercontent.com/dwojtasik/PyHook/main/docs/images/gta5_cartoon.jpg&labl=Base&labr=Cartoon-GAN
          :alt: docs/images/gta5_cartoon.jpg
     - .. image:: https://raw.githubusercontent.com/dwojtasik/PyHook/main/docs/images/crysis3_cartoon.jpg
          :target: https://dwojtasik.github.io/PyHook/?imgl=https://raw.githubusercontent.com/dwojtasik/PyHook/main/PyHook/pipelines/test_static_img/crysis3.jpg&imgr=https://raw.githubusercontent.com/dwojtasik/PyHook/main/docs/images/crysis3_cartoon.jpg&labl=Base&labr=Cartoon-GAN
          :alt: docs/images/crysis3_cartoon.jpg
     - .. image:: https://raw.githubusercontent.com/dwojtasik/PyHook/main/docs/images/trek_to_yomi_cartoon.jpg
          :target: https://dwojtasik.github.io/PyHook/?imgl=https://raw.githubusercontent.com/dwojtasik/PyHook/main/PyHook/pipelines/test_static_img/trek_to_yomi.jpg&imgr=https://raw.githubusercontent.com/dwojtasik/PyHook/main/docs/images/trek_to_yomi_cartoon.jpg&labl=Base&labr=Cartoon-GAN
          :alt: docs/images/trek_to_yomi_cartoon.jpg
   * - | `Colorization <https://github.com/dwojtasik/PyHook/blob/main/PyHook/pipelines/ai_colorization.py>`_
       | Scale: 1.0
     - .. image:: https://raw.githubusercontent.com/dwojtasik/PyHook/main/docs/images/gta5_colorization.jpg
          :target: https://dwojtasik.github.io/PyHook/?imgl=https://raw.githubusercontent.com/dwojtasik/PyHook/main/PyHook/pipelines/test_static_img/gta5.jpg&imgr=https://raw.githubusercontent.com/dwojtasik/PyHook/main/docs/images/gta5_colorization.jpg&labl=Base&labr=Colorization
          :alt: docs/images/gta5_colorization.jpg
     - .. image:: https://raw.githubusercontent.com/dwojtasik/PyHook/main/docs/images/crysis3_colorization.jpg
          :target: https://dwojtasik.github.io/PyHook/?imgl=https://raw.githubusercontent.com/dwojtasik/PyHook/main/PyHook/pipelines/test_static_img/crysis3.jpg&imgr=https://raw.githubusercontent.com/dwojtasik/PyHook/main/docs/images/crysis3_colorization.jpg&labl=Base&labr=Colorization
          :alt: docs/images/crysis3_colorization.jpg
     - .. image:: https://raw.githubusercontent.com/dwojtasik/PyHook/main/docs/images/trek_to_yomi_colorization.jpg
          :target: https://dwojtasik.github.io/PyHook/?imgl=https://raw.githubusercontent.com/dwojtasik/PyHook/main/PyHook/pipelines/test_static_img/trek_to_yomi.jpg&imgr=https://raw.githubusercontent.com/dwojtasik/PyHook/main/docs/images/trek_to_yomi_colorization.jpg&labl=Base&labr=Colorization
          :alt: docs/images/trek_to_yomi_colorization.jpg
   * - | `Cartoon-GAN <https://github.com/dwojtasik/PyHook/blob/main/PyHook/pipelines/ai_cartoon_gan.py>`_
       | and
       | `Colorization <https://github.com/dwojtasik/PyHook/blob/main/PyHook/pipelines/ai_colorization.py>`_
       | Scale: 1.0
     - .. image:: https://raw.githubusercontent.com/dwojtasik/PyHook/main/docs/images/gta5_cartoon+colorization.jpg
          :target: https://dwojtasik.github.io/PyHook/?imgl=https://raw.githubusercontent.com/dwojtasik/PyHook/main/PyHook/pipelines/test_static_img/gta5.jpg&imgr=https://raw.githubusercontent.com/dwojtasik/PyHook/main/docs/images/gta5_cartoon+colorization.jpg&labl=Base&labr=Cartoon-GAN%20and%20Colorization
          :alt: docs/images/gta5_cartoon+colorization.jpg
     - .. image:: https://raw.githubusercontent.com/dwojtasik/PyHook/main/docs/images/crysis3_cartoon+colorization.jpg
          :target: https://dwojtasik.github.io/PyHook/?imgl=https://raw.githubusercontent.com/dwojtasik/PyHook/main/PyHook/pipelines/test_static_img/crysis3.jpg&imgr=https://raw.githubusercontent.com/dwojtasik/PyHook/main/docs/images/crysis3_cartoon+colorization.jpg&labl=Base&labr=Cartoon-GAN%20and%20Colorization
          :alt: docs/images/crysis3_cartoon+colorization.jpg
     - .. image:: https://raw.githubusercontent.com/dwojtasik/PyHook/main/docs/images/trek_to_yomi_cartoon+colorization.jpg
          :target: https://dwojtasik.github.io/PyHook/?imgl=https://raw.githubusercontent.com/dwojtasik/PyHook/main/PyHook/pipelines/test_static_img/trek_to_yomi.jpg&imgr=https://raw.githubusercontent.com/dwojtasik/PyHook/main/docs/images/trek_to_yomi_cartoon+colorization.jpg&labl=Base&labr=Cartoon-GAN%20and%20Colorization
          :alt: docs/images/trek_to_yomi_cartoon+colorization.jpg
   * - | `Style Transfer <https://github.com/dwojtasik/PyHook/blob/main/PyHook/pipelines/ai_style_transfer.py>`_
       | Scale: 1.0
       | Model: Mosaic
     - .. image:: https://raw.githubusercontent.com/dwojtasik/PyHook/main/docs/images/gta5_style_mosaic.jpg
          :target: https://dwojtasik.github.io/PyHook/?imgl=https://raw.githubusercontent.com/dwojtasik/PyHook/main/PyHook/pipelines/test_static_img/gta5.jpg&imgr=https://raw.githubusercontent.com/dwojtasik/PyHook/main/docs/images/gta5_style_mosaic.jpg&labl=Base&labr=Style%20Transfer
          :alt: docs/images/gta5_style_mosaic.jpg
     - .. image:: https://raw.githubusercontent.com/dwojtasik/PyHook/main/docs/images/crysis3_style_mosaic.jpg
          :target: https://dwojtasik.github.io/PyHook/?imgl=https://raw.githubusercontent.com/dwojtasik/PyHook/main/PyHook/pipelines/test_static_img/crysis3.jpg&imgr=https://raw.githubusercontent.com/dwojtasik/PyHook/main/docs/images/crysis3_style_mosaic.jpg&labl=Base&labr=Style%20Transfer
          :alt: docs/images/crysis3_style_mosaic.jpg
     - .. image:: https://raw.githubusercontent.com/dwojtasik/PyHook/main/docs/images/trek_to_yomi_style_mosaic.jpg
          :target: https://dwojtasik.github.io/PyHook/?imgl=https://raw.githubusercontent.com/dwojtasik/PyHook/main/PyHook/pipelines/test_static_img/trek_to_yomi.jpg&imgr=https://raw.githubusercontent.com/dwojtasik/PyHook/main/docs/images/trek_to_yomi_style_mosaic.jpg&labl=Base&labr=Style%20Transfer
          :alt: docs/images/trek_to_yomi_style_mosaic.jpg

Benchmark
---------

Benchmark setup:

- `UNIGINE Superposition 64-bit DX11 <https://benchmark.unigine.com/superposition>`_
- 1280x720, windowed, lowest preset
- Intel Core i9 9900KS
- RTX 2080 Super 8GB
- 32GB DDR4 RAM

Benchmark command:

.. code-block:: powershell

    $ .\superposition.exe -preset 0 -video_app direct3d11 -shaders_quality 0 -textures_quality 0 ^
    -dof 0 -motion_blur 0 -video_vsync 0 -video_mode -1 ^
    -console_command "world_load superposition/superposition && render_manager_create_textures 1" ^
    -project_name Superposition -video_fullscreen 0 -video_width 1280 -video_height 720 ^
    -extern_plugin GPUMonitor -mode 0 -sound 0 -tooltips 1

Results:

.. list-table::
   :widths: 38 14 14 14 20
   :header-rows: 1

   * - PyHook settings
     - FPS min
     - FPS avg
     - FPS max
     - Score
   * - PyHook disabled
     - 128
     - 227
     - 331
     - 30357
   * - PyHook enabled
     - 76
     - 101
     - 120
     - 13449
   * - | `DNN Super Resolution <https://github.com/dwojtasik/PyHook/blob/main/PyHook/pipelines/ai_dnn_super_resolution.py>`_
       | Scale: 2
       | Model: FSRCNN
     - 30
     - 33
     - 35
     - 4472
   * - | `Style Transfer <https://github.com/dwojtasik/PyHook/blob/main/PyHook/pipelines/ai_style_transfer.py>`_
       | Scale: 1.0
       | Model: Mosaic
     - 9
     - 10
     - 10
     - 1305
   * - | `Cartoon-GAN <https://github.com/dwojtasik/PyHook/blob/main/PyHook/pipelines/ai_cartoon_gan.py>`_
       | Scale: 1.0
     - 4
     - 4
     - 4
     - 579
   * - | `Colorization <https://github.com/dwojtasik/PyHook/blob/main/PyHook/pipelines/ai_colorization.py>`_
       | Scale: 1.0
     - 14
     - 15
     - 15
     - 1956
   * - | `Cartoon-GAN <https://github.com/dwojtasik/PyHook/blob/main/PyHook/pipelines/ai_cartoon_gan.py>`_
       | `Colorization <https://github.com/dwojtasik/PyHook/blob/main/PyHook/pipelines/ai_colorization.py>`_
       | Scale: 1.0
     - 3
     - 3
     - 4
     - 464
   * - | `DNN Super Resolution <https://github.com/dwojtasik/PyHook/blob/main/PyHook/pipelines/ai_dnn_super_resolution.py>`_
       | Scale: 2
       | Model: FSRCNN
       | `Cartoon-GAN <https://github.com/dwojtasik/PyHook/blob/main/PyHook/pipelines/ai_cartoon_gan.py>`_
       | `Colorization <https://github.com/dwojtasik/PyHook/blob/main/PyHook/pipelines/ai_colorization.py>`_
       | Scale: 1.0
     - 8
     - 8
     - 8
     - 1074

Super-resolution
----------------

DNN super-resolution is crucial for fast AI pipeline processing. It allows to process multiple AI effects much faster due to smaller input frame.

.. image:: https://raw.githubusercontent.com/dwojtasik/PyHook/main/docs/images/sr_flow.jpg
   :alt: Go to /docs/images/sr_flow.jpg

As shown in the flowchart super-resolution consists of following steps:

- Scale base image down by some factor.
- Process small frame through AI pipelines to achieve much better performance.
- Scale processed frame back using DNN super-resolution.

| Possible FPS gains can be checked in `Benchmark <#benchmark>`_ section.
| Difference between CPU and GPU super-resolution processing can be checked in `OpenCV CPU vs GPU <#opencv-cpu-vs-gpu>`_ section.

User interface
==============

``PyHook`` uses ``ReShade ImGui UI`` to display list of available pipelines and their respective settings.

To display pipeline list, open ``ReShade`` UI and go to ``Add-ons`` tab:

.. image:: https://raw.githubusercontent.com/dwojtasik/PyHook/main/docs/images/ui_pipeline_list.jpg
   :alt: Go to /docs/images/ui_pipeline_list.jpg

Settings for enabled pipelines are displayed below mentioned list:

.. image:: https://raw.githubusercontent.com/dwojtasik/PyHook/main/docs/images/ui_settings.jpg
   :alt: Go to /docs/images/ui_settings.jpg

Supported UI widgets (read more in `pipeline template <https://github.com/dwojtasik/PyHook/blob/main/PyHook/pipelines/pipeline_template>`_):

- Checkbox
- Slider (integer value)
- Slider (float value)
- Combo box (single value select)

Requirements
============

Runtime
-------
- `ReShade <https://reshade.me/>`_ >= 5.0.0
- `Python <https://www.python.org/>`_ == ``3.10.6 for 64-bit`` | ``3.10.4 for 32-bit`` (for pipelines only)
- `CUDA <https://developer.nvidia.com/cuda-zone>`_ == 11.3 (optional for AI pipelines only)
- `cuDNN <https://developer.nvidia.com/cudnn>`_ == 8.4.1 (optional for AI pipelines only)
- | Only for specific pipelines: Any libraries that are required by pipeline code.
  | Do note that AI pipelines that requires PyTorch or TensorFlow will not work on 32-bit system because libraries does not support it.

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

1. Download and unpack zip catalog with PyHook executable, addon and pipelines.
2. | Prepare Python local environment (read more in `Virtual environment <#virtual-environment>`_) and download pipelines files if needed.
   | Pipelines has own directories with ``download.txt`` file that has list of files to download.
3. Start game with `ReShade <https://reshade.me/>`_ installed.
4. Start PyHook.exe.

For custom pipelines (e.g. AI ones) install requirements and setup ENV variables that points to Python3 binary in required version.

Available ENV variables:

- ``LOCAL_PYTHON_32`` (path to 32-bit Python)
- ``LOCAL_PYTHON_64`` (path to 64-bit Python)
- ``LOCAL_PYTHON`` (fallback path if none of above is set)

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

Virtual environment
===================

Creation
--------

PyHook allows to freely use virtual environment from `Anaconda <https://www.anaconda.com/>`_.

To create virtual env (64-bit) u can use following commands in Anaconda Prompt:

.. code-block:: powershell

    $ conda create -n pyhook64env python=3.10.6 -y
    $ conda activate pyhook64env
    $ conda install pip -y
    $ pip install -r any_pipeline.requirements.txt
    $ conda deactivate

For 32-bit different Python version have to be used (no new version at the time of writing):

.. code-block:: powershell

    $ set CONDA_FORCE_32BIT=1                         // Only for 64-bit system
    $ conda create -n pyhook32env python=3.10.4 -y
    $ conda activate pyhook32env
    $ conda install pip -y
    $ pip install -r any_pipeline.requirements.txt
    $ conda deactivate
    $ set CONDA_FORCE_32BIT=                          // Only for 64-bit system

When virtual environment is ready to be used, copy it's Python executable path and set system environment variables
described in `Installation <#installation>`_.

OpenCV with CUDA support
------------------------

| OpenCV Python module is not shipped with CUDA support by default so you have to build it from the source.
| To do this install all requirements listed below:

- `Anaconda <https://www.anaconda.com/>`_ for virual environment
- `CUDA <https://developer.nvidia.com/cuda-zone>`_ == 11.3 (or last supported by your GPU and pipeline modules)
- `cuDNN <https://developer.nvidia.com/cudnn>`_ == 8.4.1 (or last supported by your CUDA version)
- `Visual Studio <https://visualstudio.microsoft.com/pl/vs/community/>`_ >= 16 with C++ support
- `git <https://git-scm.com/>`_ for version control
- `CMake <https://cmake.org/>`_ for source build

After installation make sure that following environment variables are set:

- ``CUDA_PATH`` (e.g. "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.3")
- | ``PATH`` with paths to CUDA + cuDNN and CMake, e.g.:
  | "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.3\\bin"
  | "C:\\Program Files\\CMake\\bin"

When requirements are set, run Anaconda Prompt and follow code from file:
`build_opencv_cuda.bat <https://github.com/dwojtasik/PyHook/blob/main/docs/build_opencv_cuda.bat>`_

After build new environment variables have to be set:

- ``OpenCV_DIR`` (e.g. "C:\\OpenCV\\OpenCV-4.6.0")
- ``PATH``, add path to OpenCV built binaries (e.g. "C:\\OpenCV\\OpenCV-4.6.0\\x64\\vc16\\bin")
- ``OPENCV_LOG_LEVEL`` "ERROR", to suppress warning messages

| To verify that OpenCV was built with CUDA support, restart Anaconda Prompt, enable OpenCV virtual env and use following code in it's Python:
| NOTE: Env from ``build_opencv_cuda.bat`` has name ``opencv_build``.

.. code-block:: python

    >>> import cv2
    >>> print(cv2.cuda.getCudaEnabledDeviceCount())
    >>> print(cv2.getBuildInformation())


| For first print output should be greater than 0.
| In second print output find following fragment with 2x YES:

.. image:: https://raw.githubusercontent.com/dwojtasik/PyHook/main/docs/images/cv2_cuda.jpg
   :alt: Go to /docs/images/cv2_cuda.jpg

| The last step is to connect ``OpenCV`` to ``PyHook``. To do this setup ``LOCAL_PYTHON_64`` to executable file from OpenCV virual environment.
| Executable path can be read from python itself:

.. code-block:: python

    >>> import sys
    >>> print(sys.executable)
    'C:\\Users\\xyz\\anaconda3\\envs\\opencv_build\\python.exe'

OpenCV CPU vs GPU
*****************

`DNN Super Resolution pipeline <https://github.com/dwojtasik/PyHook/blob/main/PyHook/pipelines/ai_dnn_super_resolution.py>`_
supports both CPU and GPU OpenCV versions and will be used as benchmark.

Benchmark setup:

- Game @ 1280x720 resolution, 60 FPS
- DNN Super Resolution pipeline with `FSRCNN <https://github.com/Saafke/FSRCNN_Tensorflow>`_ model
- Intel Core i9 9900KS
- RTX 2080 Super 8GB
- 32GB DDR4 RAM

Results:

+-------------+--------+-----------+------------+-------------+--------+
| DNN version | FPS    | GPU Usage | GPU Mem MB | CPU Usage % | RAM MB |
+=============+========+===========+============+=============+========+
| CPU 2x      | 8      | 2%        | 0          | 75          | 368    |
+-------------+--------+-----------+------------+-------------+--------+
| CPU 3x      | 16     | 4%        | 0          | 67          | 257    |
+-------------+--------+-----------+------------+-------------+--------+
| CPU 4x      | 24     | 5%        | 0          | 60          | 216    |
+-------------+--------+-----------+------------+-------------+--------+
| GPU CUDA 2x | 35     | 27%       | 697        | 12          | 1440   |
+-------------+--------+-----------+------------+-------------+--------+
| GPU CUDA 3x | 37     | 21%       | 617        | 12          | 1354   |
+-------------+--------+-----------+------------+-------------+--------+
| GPU CUDA 4x | 41     | 17%       | 601        | 12          | 1289   |
+-------------+--------+-----------+------------+-------------+--------+

NOTE: Values in ``GPU Mem MB`` and ``RAM MB`` contains memory loaded by pipeline only (game not included).

Conclusion:

GPU support allows to achieve over ``4x better performance`` for best quality (2x) DNN super resolution and almost 2x for worst (4x).

History
=======

NEXT / DEV
----------
- Added automatic pipeline files download on startup.

0.8.1 (2022-09-17)
------------------
- Added support for DirectX 12 and Vulkan with fallback for older ReShade version.
- Added support for Vulkan DLL names.
- Added AI super resolution example using OpenCV DNN super resolution.
- Added multistage (multiple passes per frame) pipelines support.
- Improved error handling in ReShade addon.
- Added error notification on settings save.
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
