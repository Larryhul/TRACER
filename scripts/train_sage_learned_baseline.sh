#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 4 ]]; then
  echo "Usage: bash scripts/train_sage_learned_baseline.sh <mid|trajectron_plus_plus> <dataset> <epochs> <device> [config] [extra args...]"
  exit 1
fi

BASELINE="$1"
DATASET="$2"
EPOCHS="$3"
DEVICE="$4"
CONFIG="${5:-configs/sage.yaml}"
shift 4
if [[ $# -gt 0 ]]; then
  shift 1
fi

DATA_DIR="models/processed_data_noise"
SAVE_DIR="experiments/baselines/${BASELINE}/eth_ucy"
EXTRA_DATA_ARGS=()
if [[ "${DATASET}" == "sdd" ]]; then
  DATA_DIR="raw_data/stanford"
  SAVE_DIR="experiments/baselines/${BASELINE}/sdd"
  EXTRA_DATA_ARGS=(--source official --sdd-root data/external/sdd/annotations)
fi

python -m sage.experiments.train_learned_baseline \
  --baseline "${BASELINE}" \
  --config "${CONFIG}" \
  --dataset "${DATASET}" \
  --data-dir "${DATA_DIR}" \
  "${EXTRA_DATA_ARGS[@]}" \
  --epochs "${EPOCHS}" \
  --batch-size "${BATCH_SIZE:-256}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --device "${DEVICE}" \
  --save-dir "${SAVE_DIR}" \
  --save-every "${SAVE_EVERY:-20}" \
  "$@"
