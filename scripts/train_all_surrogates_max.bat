@echo off
setlocal

REM Max-throughput wrapper for train_all_parallel.py.
REM Launches PARALLEL training jobs concurrently (default 4) using
REM torch.compile + on-GPU dataset to push GPU utilization toward 100%.
REM Override defaults via environment variables:
REM   set PARALLEL=6          (number of concurrent processes)
REM   set EPOCHS=100
REM   set DEVICE=cuda         (force CUDA or set 'cpu')

if "%PARALLEL%"=="" set "PARALLEL=4"
if "%EPOCHS%"=="" set "EPOCHS=100"
if "%DEVICE%"=="" set "DEVICE=auto"

set "PY=%~dp0..\.venv\Scripts\python.exe"
set "SCRIPT=%~dp0train_all_parallel.py"

echo [%date% %time%] launching max-throughput pipeline
echo   PARALLEL=%PARALLEL%  EPOCHS=%EPOCHS%  DEVICE=%DEVICE%
echo.

"%PY%" "%SCRIPT%" --parallel %PARALLEL% --epochs %EPOCHS% --device %DEVICE% %*
exit /b %errorlevel%
