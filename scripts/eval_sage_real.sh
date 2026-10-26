#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-eth}"
CHECKPOINT="${2:-experiments/sage/eth_ucy/goal/${DATASET}_sage_epoch100.pt}"
DEVICE="${3:-cuda}"
CONFIG="${4:-configs/sage.yaml}"

python -m sage.experiments.eval_real_sage \
  --config "${CONFIG}" \
  --data-dir models/processed_data_noise \
  --dataset "${DATASET}" \
  --split test \
  --checkpoint "${CHECKPOINT}" \
  --device "${DEVICE}"
