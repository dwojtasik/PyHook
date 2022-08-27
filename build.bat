::=========================
:: Run this in anaconda3
::=========================

set VERSION=0.0.1

::32bit
set CONDA_FORCE_32BIT=1
call conda create -n pyhook32 -y
call conda activate pyhook32
call conda install pip -y
pip install -r requirements.txt
pip install pyinstaller
pyinstaller --collect-all pyinjector --onefile --name=PyHook-%VERSION%-win32 .\PyHook\PyHook.py
call conda deactivate
set CONDA_FORCE_32BIT=

::64bit
call conda create -n pyhook64 -y
call conda activate pyhook64
call conda install pip -y
pip install -r requirements.txt
pip install pyinstaller
pyinstaller --collect-all pyinjector --onefile --name=PyHook-%VERSION%-win_amd64 .\PyHook\PyHook.py
call conda deactivate
