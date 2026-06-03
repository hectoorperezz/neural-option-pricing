@echo off
setlocal

echo =========================================
echo E6 - Training H-7-shallow
echo =========================================
python scripts/train/train_surrogate.py ^
  --train data/heston_train_500k_uniform.npz ^
  --validation data/heston_validation_50k_uniform.npz ^
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
  --device cpu

if errorlevel 1 goto :fail

echo =========================================
echo E6 - Training H-8-deep
echo =========================================
python scripts/train/train_surrogate.py ^
  --train data/heston_train_500k_uniform.npz ^
  --validation data/heston_validation_50k_uniform.npz ^
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
  --device cpu

if errorlevel 1 goto :fail

echo =========================================
echo E6 - Training H-9-lr-schedule
echo =========================================
python scripts/train/train_surrogate.py ^
  --train data/heston_train_500k_uniform.npz ^
  --validation data/heston_validation_50k_uniform.npz ^
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
  --device cpu

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