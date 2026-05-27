@echo off
setlocal

REM 50x-scale dataset generation. Same seeds, samplers and order as the
REM baseline pipeline (generate_all_datasets.bat) but every n_samples is
REM multiplied by 50. Output filenames carry the absolute size to avoid
REM confusion with the baseline datasets.
REM Override defaults via environment variables:
REM   set WORKERS=16     (default 24)
REM   set DATA_DIR=...

if "%WORKERS%"=="" set "WORKERS=24"
if "%DATA_DIR%"=="" set "DATA_DIR=%~dp0..\data"

set "PY=%~dp0..\.venv\Scripts\python.exe"
set "SCRIPT=%~dp0generate_dataset.py"

if not exist "%DATA_DIR%" mkdir "%DATA_DIR%"

echo [%date% %time%] START 50x datasets, workers=%WORKERS%, data_dir=%DATA_DIR%
echo (estimated total runtime ~10 hours)

echo.
echo === [1/9] bs_train_10M_uniform_delta ===
"%PY%" "%SCRIPT%" --family black_scholes --sampler uniform --n-samples 10000000 --batch-size 100000 --seed 45 --include-delta --workers %WORKERS% --overwrite --output "%DATA_DIR%\bs_train_10M_uniform_delta.npz" || goto :error

echo.
echo === [2/9] bs_validation_2500k_uniform_delta ===
"%PY%" "%SCRIPT%" --family black_scholes --sampler uniform --n-samples 2500000 --batch-size 100000 --seed 49 --include-delta --workers %WORKERS% --overwrite --output "%DATA_DIR%\bs_validation_2500k_uniform_delta.npz" || goto :error

echo.
echo === [3/9] bs_test_6250k_balanced_delta ===
"%PY%" "%SCRIPT%" --family black_scholes --sampler balanced --samples-per-bin 250000 --batch-size 100000 --seed 47 --include-delta --workers %WORKERS% --overwrite --output "%DATA_DIR%\bs_test_6250k_balanced_delta.npz" || goto :error

echo.
echo === [4/9] heston_train_25M_uniform ===
"%PY%" "%SCRIPT%" --family heston --sampler uniform --n-samples 25000000 --batch-size 5000 --seed 42 --workers %WORKERS% --overwrite --output "%DATA_DIR%\heston_train_25M_uniform.npz" || goto :error

echo.
echo === [5/9] heston_train_25M_focused ===
"%PY%" "%SCRIPT%" --family heston --sampler focused --n-samples 25000000 --batch-size 5000 --seed 44 --workers %WORKERS% --overwrite --output "%DATA_DIR%\heston_train_25M_focused.npz" || goto :error

echo.
echo === [6/9] heston_train_5M_uniform ===
"%PY%" "%SCRIPT%" --family heston --sampler uniform --n-samples 5000000 --batch-size 5000 --seed 48 --workers %WORKERS% --overwrite --output "%DATA_DIR%\heston_train_5M_uniform.npz" || goto :error

echo.
echo === [7/9] heston_train_5M_uniform_delta ===
"%PY%" "%SCRIPT%" --family heston --sampler uniform --n-samples 5000000 --batch-size 5000 --seed 43 --include-delta --workers %WORKERS% --overwrite --output "%DATA_DIR%\heston_train_5M_uniform_delta.npz" || goto :error

echo.
echo === [8/9] heston_validation_2500k_uniform ===
"%PY%" "%SCRIPT%" --family heston --sampler uniform --n-samples 2500000 --batch-size 5000 --seed 50 --workers %WORKERS% --overwrite --output "%DATA_DIR%\heston_validation_2500k_uniform.npz" || goto :error

echo.
echo === [9/9] heston_test_6250k_balanced_delta ===
"%PY%" "%SCRIPT%" --family heston --sampler balanced --samples-per-bin 250000 --batch-size 1000 --seed 46 --include-delta --workers %WORKERS% --overwrite --output "%DATA_DIR%\heston_test_6250k_balanced_delta.npz" || goto :error

echo.
echo [%date% %time%] ALL 50x DATASETS GENERATED
exit /b 0

:error
echo.
echo ============================================================
echo ERROR: dataset generation failed (errorlevel %errorlevel%)
echo ============================================================
exit /b 1
