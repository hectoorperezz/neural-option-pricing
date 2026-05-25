@echo off
setlocal

REM Launcher for the 50x dataset pipeline with overall progress bar.
REM Default WORKERS=24 (matches the i9 14900K's 24 physical cores; HT at
REM 32 saturated the system on the prior attempt).

if "%WORKERS%"=="" set "WORKERS=24"

set "PY=%~dp0..\.venv\Scripts\python.exe"
set "SCRIPT=%~dp0generate_all_50x_progress.py"

"%PY%" "%SCRIPT%"
exit /b %errorlevel%
