#!/usr/bin/env bash
set -euo pipefail

DATA_DIR="${DATA_DIR:-data}"
RESULTS_DIR="${RESULTS_DIR:-results/checkpoints}"
LOG_DIR="${LOG_DIR:-results/logs}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-auto}"

BS_EPOCHS="${BS_EPOCHS:-100}"
HESTON_EPOCHS="${HESTON_EPOCHS:-100}"
SMALL_EPOCHS="${SMALL_EPOCHS:-100}"
BS_BATCH_SIZE="${BS_BATCH_SIZE:-8192}"
HESTON_BATCH_SIZE="${HESTON_BATCH_SIZE:-8192}"
SMALL_BATCH_SIZE="${SMALL_BATCH_SIZE:-4096}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
HIDDEN_WIDTH="${HIDDEN_WIDTH:-128}"
HIDDEN_LAYERS="${HIDDEN_LAYERS:-4}"
NUM_WORKERS="${NUM_WORKERS:-0}"

mkdir -p "$RESULTS_DIR" "$LOG_DIR"

timestamp() {
  date +"%Y-%m-%dT%H:%M:%S%z"
}

run_training() {
  local experiment_id="$1"
  shift
  local log_file="$LOG_DIR/train_${experiment_id}.log"

  echo "[$(timestamp)] START $experiment_id" | tee "$log_file"
  "$PYTHON_BIN" scripts/train_surrogate.py \
    --experiment-id "$experiment_id" \
    --output-dir "$RESULTS_DIR/$experiment_id" \
    --learning-rate "$LEARNING_RATE" \
    --hidden-width "$HIDDEN_WIDTH" \
    --hidden-layers "$HIDDEN_LAYERS" \
    --num-workers "$NUM_WORKERS" \
    --device "$DEVICE" \
    "$@" 2>&1 | tee -a "$log_file"
  echo "[$(timestamp)] DONE $experiment_id" | tee -a "$log_file"
}

run_training "BS-1" \
  --train "$DATA_DIR/bs_train_200k_uniform_delta.npz" \
  --validation "$DATA_DIR/bs_validation_50k_uniform_delta.npz" \
  --loss price \
  --activation relu \
  --epochs "$BS_EPOCHS" \
  --batch-size "$BS_BATCH_SIZE" \
  --seed 101

run_training "BS-2" \
  --train "$DATA_DIR/bs_train_200k_uniform_delta.npz" \
  --validation "$DATA_DIR/bs_validation_50k_uniform_delta.npz" \
  --loss price \
  --activation softplus \
  --epochs "$BS_EPOCHS" \
  --batch-size "$BS_BATCH_SIZE" \
  --seed 102

run_training "BS-3" \
  --train "$DATA_DIR/bs_train_200k_uniform_delta.npz" \
  --validation "$DATA_DIR/bs_validation_50k_uniform_delta.npz" \
  --loss price \
  --activation swish \
  --epochs "$BS_EPOCHS" \
  --batch-size "$BS_BATCH_SIZE" \
  --seed 103

run_training "BS-4" \
  --train "$DATA_DIR/bs_train_200k_uniform_delta.npz" \
  --validation "$DATA_DIR/bs_validation_50k_uniform_delta.npz" \
  --loss price \
  --activation tanh \
  --epochs "$BS_EPOCHS" \
  --batch-size "$BS_BATCH_SIZE" \
  --seed 104

run_training "H-1" \
  --train "$DATA_DIR/heston_train_500k_uniform.npz" \
  --validation "$DATA_DIR/heston_validation_50k_uniform.npz" \
  --loss price \
  --activation relu \
  --epochs "$HESTON_EPOCHS" \
  --batch-size "$HESTON_BATCH_SIZE" \
  --seed 201

run_training "H-2" \
  --train "$DATA_DIR/heston_train_500k_uniform.npz" \
  --validation "$DATA_DIR/heston_validation_50k_uniform.npz" \
  --loss price \
  --activation softplus \
  --epochs "$HESTON_EPOCHS" \
  --batch-size "$HESTON_BATCH_SIZE" \
  --seed 202

run_training "H-3" \
  --train "$DATA_DIR/heston_train_500k_uniform.npz" \
  --validation "$DATA_DIR/heston_validation_50k_uniform.npz" \
  --loss price \
  --activation swish \
  --epochs "$HESTON_EPOCHS" \
  --batch-size "$HESTON_BATCH_SIZE" \
  --seed 203

run_training "H-4" \
  --train "$DATA_DIR/heston_train_500k_uniform.npz" \
  --validation "$DATA_DIR/heston_validation_50k_uniform.npz" \
  --loss price \
  --activation tanh \
  --epochs "$HESTON_EPOCHS" \
  --batch-size "$HESTON_BATCH_SIZE" \
  --seed 204

run_training "H-5" \
  --train "$DATA_DIR/heston_train_500k_focused.npz" \
  --validation "$DATA_DIR/heston_validation_50k_uniform.npz" \
  --loss price \
  --activation swish \
  --epochs "$HESTON_EPOCHS" \
  --batch-size "$HESTON_BATCH_SIZE" \
  --seed 205

run_training "H-3-small" \
  --train "$DATA_DIR/heston_train_100k_uniform.npz" \
  --validation "$DATA_DIR/heston_validation_50k_uniform.npz" \
  --loss price \
  --activation swish \
  --epochs "$SMALL_EPOCHS" \
  --batch-size "$SMALL_BATCH_SIZE" \
  --seed 206

run_training "H-6-small" \
  --train "$DATA_DIR/heston_train_100k_uniform_delta.npz" \
  --validation "$DATA_DIR/heston_validation_50k_uniform.npz" \
  --loss differential \
  --activation swish \
  --epochs "$SMALL_EPOCHS" \
  --batch-size "$SMALL_BATCH_SIZE" \
  --seed 207

echo "[$(timestamp)] ALL SURROGATES TRAINED"
