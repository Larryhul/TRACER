#!/usr/bin/env bash
set -euo pipefail

EPOCHS="${1:-100}"
DEVICE="${2:-cuda}"
CONFIG="${3:-configs/sage_sdd.yaml}"
SOURCE="${4:-}"
SDD_ROOT="${5:-}"

CMD=(python -m sage.experiments.train_sdd_sage
  --config "${CONFIG}"
  --data-dir raw_data/stanford
  --epochs "${EPOCHS}"
  --device "${DEVICE}"
  --save-every 10)

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
