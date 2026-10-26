#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT="${1:-experiments/sage/sdd/goal/sdd_sage_epoch80.pt}"
DEVICE="${2:-cuda}"
CONFIG="${3:-configs/sage_sdd.yaml}"
SOURCE="${4:-}"
SDD_ROOT="${5:-}"

CMD=(python -m sage.experiments.eval_sdd_sage
  --config "${CONFIG}"
  --data-dir raw_data/stanford
  --split test
  --checkpoint "${CHECKPOINT}"
  --samples 20
  --stride 10
  --role-metrics
  --device "${DEVICE}"
  --save-csv results/sage_sdd.csv)

if [[ -n "${SOURCE}" ]]; then
  CMD+=(--source "${SOURCE}")
fi

if [[ "${SOURCE}" == "official" ]]; then
  if [[ -z "${SDD_ROOT}" ]]; then
    echo "SDD_ROOT is required when SOURCE=official" >&2
    exit 2
  fi
  CMD+=(--sdd-root "${SDD_ROOT}")
fi

"${CMD[@]}"
