#!/usr/bin/env bash
set -euo pipefail

DATASET="${1:-eth}"
SPLIT="${2:-train}"
CONFIG="${3:-configs/tracer.yaml}"

python -m tracer.experiments.check_real_data \
  --config "${CONFIG}" \
  --dataset "${DATASET}" \
  --split "${SPLIT}" \
  --batch-size 8 \
  --limit-samples 32
