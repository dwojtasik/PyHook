::=========================
:: Run this in anaconda3
::=========================

set VERSION=0.0.1

::32bit
set CONDA_FORCE_32BIT=1
call conda create -n pyhook32 python=3.10.6 -y
call conda activate pyhook32
call conda install pip -y
pip install -r requirements.txt
pip install pyinstaller==5.3
pyinstaller --clean --onefile ^
    --collect-all pyinjector ^
    --version-file VERSION.txt ^
    --name=PyHook-%VERSION%-win32 ^
    .\PyHook\pyhook.py
call conda deactivate
set CONDA_FORCE_32BIT=

::64bit
call conda create -n pyhook64 python=3.10.6 -y
call conda activate pyhook64
call conda install pip -y
pip install -r requirements.txt
pip install pyinstaller==5.3
pyinstaller --clean --onefile ^
    --collect-all pyinjector ^
    --version-file VERSION.txt ^
    --name=PyHook-%VERSION%-win_amd64 ^
    .\PyHook\pyhook.py
call conda deactivate
