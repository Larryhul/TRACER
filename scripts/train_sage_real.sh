#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-eth}"
EPOCHS="${2:-100}"
DEVICE="${3:-cuda}"
CONFIG="${4:-configs/sage.yaml}"
shift $(( $# < 4 ? $# : 4 ))

python -m sage.experiments.train_real_sage \
  --config "${CONFIG}" \
  --data-dir models/processed_data_noise \
  --dataset "${DATASET}" \
  --epochs "${EPOCHS}" \
  --device "${DEVICE}" \
  --save-every 10 \
  "$@"
