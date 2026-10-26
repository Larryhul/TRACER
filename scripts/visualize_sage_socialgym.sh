#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=${1:?checkpoint path required}
OUT=${2:-results/figures/socialgym_qualitative_overview.png}
DEVICE=${3:-cuda}
CONFIG=${4:-configs/sage_socialgym.yaml}
INDICES=${5:-0,1,2,3}

python -m sage.experiments.visualize_socialgym_qualitative \
  --config "${CONFIG}" \
  --split test \
  --checkpoint "${CHECKPOINT}" \
  --indices "${INDICES}" \
  --samples 8 \
  --stride 10 \
  --guidance-scale 0.05 \
  --device "${DEVICE}" \
  --out "${OUT}"
