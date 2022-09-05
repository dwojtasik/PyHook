::=========================
:: Run this in anaconda3
::=========================

::32bit
set CONDA_FORCE_32BIT=1
call conda create -n pyhook32 python=3.10.4 -y
call conda activate pyhook32
call conda install pip -y
pip install -r requirements.txt
pip install -r build_requirements.txt
pyinstaller --clean PyHook.spec
call conda deactivate
set CONDA_FORCE_32BIT=

::64bit
call conda create -n pyhook64 python=3.10.6 -y
call conda activate pyhook64
call conda install pip -y
pip install -r requirements.txt
pip install -r build_requirements.txt
pyinstaller --clean PyHook.spec
call conda deactivate
