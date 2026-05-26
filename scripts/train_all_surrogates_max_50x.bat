@echo off
setlocal

REM Max-throughput wrapper for train_all_parallel_50x.py.
REM Launches PARALLEL training jobs concurrently (default 2 because
REM Heston-25M with preload uses ~1.7 GB VRAM per process; 4060 has 8 GB
REM total ~6.3 GB free).
REM Override defaults via environment variables:
REM   set PARALLEL=2          (number of concurrent processes; max 2 with Heston-25M preload)
REM   set EPOCHS=100
REM   set DEVICE=cuda         (force CUDA or set 'cpu')

if "%PARALLEL%"=="" set "PARALLEL=2"
if "%EPOCHS%"=="" set "EPOCHS=100"
if "%DEVICE%"=="" set "DEVICE=auto"

set "PY=%~dp0..\.venv\Scripts\python.exe"
set "SCRIPT=%~dp0train_all_parallel_50x.py"

echo [%date% %time%] launching max-throughput 50x pipeline
echo   PARALLEL=%PARALLEL%  EPOCHS=%EPOCHS%  DEVICE=%DEVICE%
echo.

"%PY%" "%SCRIPT%" --parallel %PARALLEL% --epochs %EPOCHS% --device %DEVICE% %*
exit /b %errorlevel%
