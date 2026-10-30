#!/usr/bin/env bash
set -euo pipefail

SPLIT=${1:-train}
CONFIG=${2:-configs/sage_socialgym.yaml}
SOURCE=${3:-style}

python -m tracer.experiments.check_socialgym_data \
  --config "${CONFIG}" \
  --split "${SPLIT}" \
  --source "${SOURCE}"
