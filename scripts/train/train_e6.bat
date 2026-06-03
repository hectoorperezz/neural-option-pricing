@echo off
setlocal

REM Entrena las tres variantes nuevas de E6 (H-7-shallow, H-8-deep y
REM H-9-lr-schedule) sobre el mismo regimen de datos que H-3 baseline
REM (heston_train_25M_uniform.npz, heston_validation_2500k_uniform.npz).
REM Misma semilla del entrenador (203), batch size (32768), learning
REM rate inicial (1e-3) y numero de epocas (100) que H-3; la unica
REM variable que cambia entre las cuatro redes es la profundidad
REM (H-7, H-3, H-8) o la presencia del scheduler (H-3, H-9).
REM
REM Uso: ejecutar desde la raiz del repo con el .venv activado.

echo =========================================
echo E6 - Training H-7-shallow
echo =========================================
python scripts/train/train_surrogate.py ^
  --train data/heston_train_25M_uniform.npz ^
  --validation data/heston_validation_2500k_uniform.npz ^
  --output-dir results/checkpoints/H-7-shallow ^
  --experiment-id H-7-shallow ^
  --loss price ^
  --activation swish ^
  --hidden-width 128 ^
  --hidden-layers 2 ^
  --epochs 100 ^
  --batch-size 32768 ^
  --learning-rate 1e-3 ^
  --seed 203 ^
  --device auto ^
  --preload-to-device

if errorlevel 1 goto :fail

echo =========================================
echo E6 - Training H-8-deep
echo =========================================
python scripts/train/train_surrogate.py ^
  --train data/heston_train_25M_uniform.npz ^
  --validation data/heston_validation_2500k_uniform.npz ^
  --output-dir results/checkpoints/H-8-deep ^
  --experiment-id H-8-deep ^
  --loss price ^
  --activation swish ^
  --hidden-width 128 ^
  --hidden-layers 6 ^
  --epochs 100 ^
  --batch-size 32768 ^
  --learning-rate 1e-3 ^
  --seed 203 ^
  --device auto ^
  --preload-to-device

if errorlevel 1 goto :fail

echo =========================================
echo E6 - Training H-9-lr-schedule
echo =========================================
python scripts/train/train_surrogate.py ^
  --train data/heston_train_25M_uniform.npz ^
  --validation data/heston_validation_2500k_uniform.npz ^
  --output-dir results/checkpoints/H-9-lr-schedule ^
  --experiment-id H-9-lr-schedule ^
  --loss price ^
  --activation swish ^
  --hidden-width 128 ^
  --hidden-layers 4 ^
  --epochs 100 ^
  --batch-size 32768 ^
  --learning-rate 1e-3 ^
  --scheduler plateau ^
  --scheduler-factor 0.5 ^
  --scheduler-patience 3 ^
  --scheduler-min-lr 1e-5 ^
  --seed 203 ^
  --device auto ^
  --preload-to-device

if errorlevel 1 goto :fail

echo.
echo E6 training finished successfully.
goto :end

:fail
echo.
echo E6 training failed.
exit /b 1

:end
endlocal
