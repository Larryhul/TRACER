#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=${1:?checkpoint path required}
CSV=${2:-results/sage_socialgym.csv}
DEVICE=${3:-cuda}
CONFIG=${4:-configs/sage_socialgym.yaml}
NUM_ROBOTS=${5:-}
NUM_ENTITIES=${6:-}

CMD=(python -m tracer.experiments.eval_socialgym_sage \
  --config "${CONFIG}" \
  --split test \
  --checkpoint "${CHECKPOINT}" \
  --samples 20 \
  --batch-size 128 \
  --num-workers 8 \
  --guidance-scale 0.05 \
  --role-metrics \
  --device "${DEVICE}" \
  --save-csv "${CSV}")

if [[ -n "${NUM_ROBOTS}" ]]; then
  CMD+=(--num-robots "${NUM_ROBOTS}")
fi
if [[ -n "${NUM_ENTITIES}" ]]; then
  CMD+=(--num-entities "${NUM_ENTITIES}")
fi

"${CMD[@]}"
