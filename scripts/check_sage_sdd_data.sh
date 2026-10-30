#!/usr/bin/env bash
set -euo pipefail

SPLIT="${1:-train}"
CONFIG="${2:-configs/sage_sdd.yaml}"
SOURCE="${3:-}"
SDD_ROOT="${4:-}"

CMD=(python -m tracer.experiments.check_sdd_data
  --config "${CONFIG}"
  --data-dir raw_data/stanford
  --split "${SPLIT}"
  --batch-size 8
  --limit-samples 32)

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
