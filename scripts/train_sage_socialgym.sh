#!/usr/bin/env bash
set -euo pipefail

EPOCHS=${1:-60}
DEVICE=${2:-cuda}
CONFIG=${3:-configs/sage_socialgym.yaml}
SAVE_DIR=${4:-experiments/tracer/socialgym/default}
NUM_ROBOTS=${5:-}
NUM_ENTITIES=${6:-}

CMD=(python -m tracer.experiments.train_socialgym_sage \
  --config "${CONFIG}" \
  --epochs "${EPOCHS}" \
  --batch-size 128 \
  --num-workers 8 \
  --device "${DEVICE}" \
  --save-every 10 \
  --save-dir "${SAVE_DIR}")

if [[ -n "${NUM_ROBOTS}" ]]; then
  CMD+=(--num-robots "${NUM_ROBOTS}")
fi
if [[ -n "${NUM_ENTITIES}" ]]; then
  CMD+=(--num-entities "${NUM_ENTITIES}")
fi

"${CMD[@]}"
