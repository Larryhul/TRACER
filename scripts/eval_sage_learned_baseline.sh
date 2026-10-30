#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 5 ]]; then
  echo "Usage: bash scripts/eval_sage_learned_baseline.sh <mid|trajectron_plus_plus> <dataset> <checkpoint> <device> <config> [extra args...]"
  exit 1
fi

BASELINE="$1"
DATASET="$2"
CHECKPOINT="$3"
DEVICE="$4"
CONFIG="$5"
shift 5

DATA_DIR="models/processed_data_noise"
EXTRA_DATA_ARGS=()
EXTRA_METRIC_ARGS=()
if [[ "${DATASET}" == "sdd" ]]; then
  DATA_DIR="raw_data/stanford"
  EXTRA_DATA_ARGS=(--source official --sdd-root data/external/sdd/annotations)
  EXTRA_METRIC_ARGS=(--use-geometry-margin --role-metrics)
fi

python -m tracer.experiments.eval_learned_baseline \
  --baseline "${BASELINE}" \
  --config "${CONFIG}" \
  --dataset "${DATASET}" \
  --data-dir "${DATA_DIR}" \
  "${EXTRA_DATA_ARGS[@]}" \
  --checkpoint "${CHECKPOINT}" \
  --samples "${SAMPLES:-20}" \
  --stride "${STRIDE:-10}" \
  --batch-size "${BATCH_SIZE:-256}" \
  --num-workers "${NUM_WORKERS:-8}" \
  --device "${DEVICE}" \
  --save-csv "${SAVE_CSV:-results/sage_baseline_comparison.csv}" \
  "${EXTRA_METRIC_ARGS[@]}" \
  "$@"
