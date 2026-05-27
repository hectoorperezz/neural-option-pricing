@echo off
setlocal

REM Windows equivalent of train_all_surrogates.sh.
REM Trains all 11 surrogates sequentially using train_surrogate.py.
REM Override defaults via environment variables, e.g.:
REM   set DEVICE=cuda
REM   set BS_EPOCHS=50
REM   set HESTON_BATCH_SIZE=4096

if "%DATA_DIR%"=="" set "DATA_DIR=%~dp0..\data"
if "%RESULTS_DIR%"=="" set "RESULTS_DIR=%~dp0..\results\checkpoints"
if "%LOG_DIR%"=="" set "LOG_DIR=%~dp0..\results\logs"
if "%DEVICE%"=="" set "DEVICE=auto"

if "%BS_EPOCHS%"=="" set "BS_EPOCHS=100"
if "%HESTON_EPOCHS%"=="" set "HESTON_EPOCHS=100"
if "%SMALL_EPOCHS%"=="" set "SMALL_EPOCHS=100"
if "%BS_BATCH_SIZE%"=="" set "BS_BATCH_SIZE=8192"
if "%HESTON_BATCH_SIZE%"=="" set "HESTON_BATCH_SIZE=8192"
if "%SMALL_BATCH_SIZE%"=="" set "SMALL_BATCH_SIZE=4096"
if "%LEARNING_RATE%"=="" set "LEARNING_RATE=0.001"
if "%HIDDEN_WIDTH%"=="" set "HIDDEN_WIDTH=128"
if "%HIDDEN_LAYERS%"=="" set "HIDDEN_LAYERS=4"
if "%NUM_WORKERS%"=="" set "NUM_WORKERS=0"

set "PY=%~dp0..\.venv\Scripts\python.exe"
set "SCRIPT=%~dp0train_surrogate.py"

if not exist "%RESULTS_DIR%" mkdir "%RESULTS_DIR%"
if not exist "%LOG_DIR%" mkdir "%LOG_DIR%"

echo [%date% %time%] START all surrogates, device=%DEVICE%, results=%RESULTS_DIR%

echo.
echo === [1/11] BS-1 (relu, price, 200k) ===
"%PY%" "%SCRIPT%" --experiment-id BS-1 --output-dir "%RESULTS_DIR%\BS-1" --train "%DATA_DIR%\bs_train_200k_uniform_delta.npz" --validation "%DATA_DIR%\bs_validation_50k_uniform_delta.npz" --loss price --activation relu --epochs %BS_EPOCHS% --batch-size %BS_BATCH_SIZE% --learning-rate %LEARNING_RATE% --hidden-width %HIDDEN_WIDTH% --hidden-layers %HIDDEN_LAYERS% --num-workers %NUM_WORKERS% --device %DEVICE% --seed 101 || goto :error

echo.
echo === [2/11] BS-2 (softplus, price, 200k) ===
"%PY%" "%SCRIPT%" --experiment-id BS-2 --output-dir "%RESULTS_DIR%\BS-2" --train "%DATA_DIR%\bs_train_200k_uniform_delta.npz" --validation "%DATA_DIR%\bs_validation_50k_uniform_delta.npz" --loss price --activation softplus --epochs %BS_EPOCHS% --batch-size %BS_BATCH_SIZE% --learning-rate %LEARNING_RATE% --hidden-width %HIDDEN_WIDTH% --hidden-layers %HIDDEN_LAYERS% --num-workers %NUM_WORKERS% --device %DEVICE% --seed 102 || goto :error

echo.
echo === [3/11] BS-3 (swish, price, 200k) ===
"%PY%" "%SCRIPT%" --experiment-id BS-3 --output-dir "%RESULTS_DIR%\BS-3" --train "%DATA_DIR%\bs_train_200k_uniform_delta.npz" --validation "%DATA_DIR%\bs_validation_50k_uniform_delta.npz" --loss price --activation swish --epochs %BS_EPOCHS% --batch-size %BS_BATCH_SIZE% --learning-rate %LEARNING_RATE% --hidden-width %HIDDEN_WIDTH% --hidden-layers %HIDDEN_LAYERS% --num-workers %NUM_WORKERS% --device %DEVICE% --seed 103 || goto :error

echo.
echo === [4/11] BS-4 (tanh, price, 200k) ===
"%PY%" "%SCRIPT%" --experiment-id BS-4 --output-dir "%RESULTS_DIR%\BS-4" --train "%DATA_DIR%\bs_train_200k_uniform_delta.npz" --validation "%DATA_DIR%\bs_validation_50k_uniform_delta.npz" --loss price --activation tanh --epochs %BS_EPOCHS% --batch-size %BS_BATCH_SIZE% --learning-rate %LEARNING_RATE% --hidden-width %HIDDEN_WIDTH% --hidden-layers %HIDDEN_LAYERS% --num-workers %NUM_WORKERS% --device %DEVICE% --seed 104 || goto :error

echo.
echo === [5/11] H-1 (relu, price, Heston 500k uniform) ===
"%PY%" "%SCRIPT%" --experiment-id H-1 --output-dir "%RESULTS_DIR%\H-1" --train "%DATA_DIR%\heston_train_500k_uniform.npz" --validation "%DATA_DIR%\heston_validation_50k_uniform.npz" --loss price --activation relu --epochs %HESTON_EPOCHS% --batch-size %HESTON_BATCH_SIZE% --learning-rate %LEARNING_RATE% --hidden-width %HIDDEN_WIDTH% --hidden-layers %HIDDEN_LAYERS% --num-workers %NUM_WORKERS% --device %DEVICE% --seed 201 || goto :error

echo.
echo === [6/11] H-2 (softplus, price, Heston 500k uniform) ===
"%PY%" "%SCRIPT%" --experiment-id H-2 --output-dir "%RESULTS_DIR%\H-2" --train "%DATA_DIR%\heston_train_500k_uniform.npz" --validation "%DATA_DIR%\heston_validation_50k_uniform.npz" --loss price --activation softplus --epochs %HESTON_EPOCHS% --batch-size %HESTON_BATCH_SIZE% --learning-rate %LEARNING_RATE% --hidden-width %HIDDEN_WIDTH% --hidden-layers %HIDDEN_LAYERS% --num-workers %NUM_WORKERS% --device %DEVICE% --seed 202 || goto :error

echo.
echo === [7/11] H-3 (swish, price, Heston 500k uniform) ===
"%PY%" "%SCRIPT%" --experiment-id H-3 --output-dir "%RESULTS_DIR%\H-3" --train "%DATA_DIR%\heston_train_500k_uniform.npz" --validation "%DATA_DIR%\heston_validation_50k_uniform.npz" --loss price --activation swish --epochs %HESTON_EPOCHS% --batch-size %HESTON_BATCH_SIZE% --learning-rate %LEARNING_RATE% --hidden-width %HIDDEN_WIDTH% --hidden-layers %HIDDEN_LAYERS% --num-workers %NUM_WORKERS% --device %DEVICE% --seed 203 || goto :error

echo.
echo === [8/11] H-4 (tanh, price, Heston 500k uniform) ===
"%PY%" "%SCRIPT%" --experiment-id H-4 --output-dir "%RESULTS_DIR%\H-4" --train "%DATA_DIR%\heston_train_500k_uniform.npz" --validation "%DATA_DIR%\heston_validation_50k_uniform.npz" --loss price --activation tanh --epochs %HESTON_EPOCHS% --batch-size %HESTON_BATCH_SIZE% --learning-rate %LEARNING_RATE% --hidden-width %HIDDEN_WIDTH% --hidden-layers %HIDDEN_LAYERS% --num-workers %NUM_WORKERS% --device %DEVICE% --seed 204 || goto :error

echo.
echo === [9/11] H-5 (swish, price, Heston 500k focused) ===
"%PY%" "%SCRIPT%" --experiment-id H-5 --output-dir "%RESULTS_DIR%\H-5" --train "%DATA_DIR%\heston_train_500k_focused.npz" --validation "%DATA_DIR%\heston_validation_50k_uniform.npz" --loss price --activation swish --epochs %HESTON_EPOCHS% --batch-size %HESTON_BATCH_SIZE% --learning-rate %LEARNING_RATE% --hidden-width %HIDDEN_WIDTH% --hidden-layers %HIDDEN_LAYERS% --num-workers %NUM_WORKERS% --device %DEVICE% --seed 205 || goto :error

echo.
echo === [10/11] H-3-small (swish, price, Heston 100k uniform) ===
"%PY%" "%SCRIPT%" --experiment-id H-3-small --output-dir "%RESULTS_DIR%\H-3-small" --train "%DATA_DIR%\heston_train_100k_uniform.npz" --validation "%DATA_DIR%\heston_validation_50k_uniform.npz" --loss price --activation swish --epochs %SMALL_EPOCHS% --batch-size %SMALL_BATCH_SIZE% --learning-rate %LEARNING_RATE% --hidden-width %HIDDEN_WIDTH% --hidden-layers %HIDDEN_LAYERS% --num-workers %NUM_WORKERS% --device %DEVICE% --seed 206 || goto :error

echo.
echo === [11/11] H-6-small (swish, differential, Heston 100k uniform+delta) ===
"%PY%" "%SCRIPT%" --experiment-id H-6-small --output-dir "%RESULTS_DIR%\H-6-small" --train "%DATA_DIR%\heston_train_100k_uniform_delta.npz" --validation "%DATA_DIR%\heston_validation_50k_uniform.npz" --loss differential --activation swish --epochs %SMALL_EPOCHS% --batch-size %SMALL_BATCH_SIZE% --learning-rate %LEARNING_RATE% --hidden-width %HIDDEN_WIDTH% --hidden-layers %HIDDEN_LAYERS% --num-workers %NUM_WORKERS% --device %DEVICE% --seed 207 || goto :error

echo.
echo [%date% %time%] ALL SURROGATES TRAINED
exit /b 0

:error
echo.
echo ============================================================
echo ERROR: training failed (errorlevel %errorlevel%)
echo ============================================================
exit /b 1
