#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data}"
LOG_DIR="${LOG_DIR:-results/logs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
OVERWRITE="${OVERWRITE:-0}"

mkdir -p "$DATA_DIR" "$LOG_DIR"

timestamp() {
  date +"%Y-%m-%dT%H:%M:%S%z"
}

run_dataset() {
  local name="$1"
  shift
  local log_file="$LOG_DIR/${name}.log"

  echo "[$(timestamp)] START $name" | tee "$log_file"
  if [[ "$OVERWRITE" == "1" ]]; then
    "$PYTHON_BIN" scripts/generate_dataset.py "$@" --overwrite 2>&1 | tee -a "$log_file"
  else
    "$PYTHON_BIN" scripts/generate_dataset.py "$@" 2>&1 | tee -a "$log_file"
  fi
  echo "[$(timestamp)] DONE $name" | tee -a "$log_file"
}

run_dataset "heston_benchmark_5k_delta" \
  --family heston \
  --sampler uniform \
  --n-samples 5000 \
  --batch-size 1000 \
  --seed 42 \
  --include-delta \
  --output "$DATA_DIR/heston_benchmark_5k_delta.npz"

run_dataset "bs_train_200k_uniform_delta" \
  --family black_scholes \
  --sampler uniform \
  --n-samples 200000 \
  --batch-size 20000 \
  --seed 45 \
  --include-delta \
  --output "$DATA_DIR/bs_train_200k_uniform_delta.npz"

run_dataset "bs_validation_50k_uniform_delta" \
  --family black_scholes \
  --sampler uniform \
  --n-samples 50000 \
  --batch-size 20000 \
  --seed 49 \
  --include-delta \
  --output "$DATA_DIR/bs_validation_50k_uniform_delta.npz"

run_dataset "bs_test_125k_balanced_delta" \
  --family black_scholes \
  --sampler balanced \
  --samples-per-bin 5000 \
  --batch-size 20000 \
  --seed 47 \
  --include-delta \
  --output "$DATA_DIR/bs_test_125k_balanced_delta.npz"

run_dataset "heston_train_500k_uniform" \
  --family heston \
  --sampler uniform \
  --n-samples 500000 \
  --batch-size 5000 \
  --seed 42 \
  --output "$DATA_DIR/heston_train_500k_uniform.npz"

run_dataset "heston_train_500k_focused" \
  --family heston \
  --sampler focused \
  --n-samples 500000 \
  --batch-size 5000 \
  --seed 44 \
  --output "$DATA_DIR/heston_train_500k_focused.npz"

run_dataset "heston_train_100k_uniform" \
  --family heston \
  --sampler uniform \
  --n-samples 100000 \
  --batch-size 5000 \
  --seed 48 \
  --output "$DATA_DIR/heston_train_100k_uniform.npz"

run_dataset "heston_train_100k_uniform_delta" \
  --family heston \
  --sampler uniform \
  --n-samples 100000 \
  --batch-size 5000 \
  --seed 43 \
  --include-delta \
  --output "$DATA_DIR/heston_train_100k_uniform_delta.npz"

run_dataset "heston_validation_50k_uniform" \
  --family heston \
  --sampler uniform \
  --n-samples 50000 \
  --batch-size 5000 \
  --seed 50 \
  --output "$DATA_DIR/heston_validation_50k_uniform.npz"

run_dataset "heston_test_125k_balanced_delta" \
  --family heston \
  --sampler balanced \
  --samples-per-bin 5000 \
  --batch-size 1000 \
  --seed 46 \
  --include-delta \
  --output "$DATA_DIR/heston_test_125k_balanced_delta.npz"

echo "[$(timestamp)] ALL DATASETS GENERATED"
