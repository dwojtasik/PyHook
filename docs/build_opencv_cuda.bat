::=======================================
:: Run this in anaconda3
::---------------------------------------
:: Make sure all requirements are set:
:: Visual Studio >= 16 with C++ support
:: CMake
:: CUDA with cuDNN
:: Setup CUDA_PATH and PATH for CUDA
::=======================================

:: Prepare virtual env
set cwd=%cd%
set envName=opencv_build
set opencv-version=4.6.0
conda create -y --name %envName% python=3.10.6 numpy

:: Clone OpenCV repository
cd %cwd%
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout tags/%opencv-version%

:: Clone OpenCV contrib repository
cd %cwd%
git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
git checkout tags/%opencv-version%

:: Build OpenCV
conda activate %envName%
set CONDA_PREFIX=%CONDA_PREFIX:\=/%

cd %cwd%
mkdir OpenCV-%opencv-version%

cd opencv
mkdir build
cd build

cmake ^
-G "Visual Studio 16 2019" ^
-T host=x64 ^
-DCMAKE_BUILD_TYPE=RELEASE ^
-DCMAKE_INSTALL_PREFIX=%cwd%/OpenCV-%opencv-version% ^
-DOPENCV_EXTRA_MODULES_PATH=%cwd%/opencv_contrib/modules ^
-DINSTALL_PYTHON_EXAMPLES=OFF ^
-DINSTALL_C_EXAMPLES=OFF ^
-DPYTHON_EXECUTABLE=%CONDA_PREFIX%/python3 ^
-DPYTHON3_LIBRARY=%CONDA_PREFIX%/libs/python3 ^
-DWITH_CUDA=ON ^
-DWITH_CUDNN=ON ^
-DOPENCV_DNN_CUDA=ON ^
-DWITH_CUBLAS=ON ^
..

cmake --build . --config Release --target INSTALL

:: Remove not needed files
cd %cwd%
rmdir /s /q opencv opencv_contrib
