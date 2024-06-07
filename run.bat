@echo off
setlocal
REM Activate conda environment
set ENV_NAME=arn

REM Initialize conda
REM call conda init

REM Activate the conda environment
call conda activate %ENV_NAME%

REM Define the parameters to use
set params=600 700 800 900 1000 2000

REM Loop through the parameters and run the Python script
for %%i in (%params%) do (
    call python lstm_train.py %%i
)

REM Deactivate the conda environment
call conda deactivate

endlocal
pause