#!/usr/bin/env bash
set -euo pipefail

BENCHMARK=${1:-synthetic}
CSV=${2:-results/social_force_baseline.csv}
DEVICE=${3:-cuda}
CONFIG=${4:-}
NUM_ROBOTS=${5:-}
NUM_ENTITIES=${6:-}
BATCH_SIZE=${7:-128}
NUM_WORKERS=${8:-8}

if [[ -z "${CONFIG}" ]]; then
  if [[ "${BENCHMARK}" == "socialgym" ]]; then
    CONFIG=configs/sage_socialgym.yaml
  else
    CONFIG=configs/sage_synthetic.yaml
  fi
fi

CMD=(python -m sage.experiments.eval_social_force_baseline \
  --benchmark "${BENCHMARK}" \
  --config "${CONFIG}" \
  --split test \
  --batch-size "${BATCH_SIZE}" \
  --num-workers "${NUM_WORKERS}" \
  --role-metrics \
  --device "${DEVICE}" \
  --save-csv "${CSV}")

if [[ "${BENCHMARK}" == "socialgym" ]]; then
  CMD+=(--source style)
fi
if [[ -n "${NUM_ROBOTS}" ]]; then
  CMD+=(--num-robots "${NUM_ROBOTS}")
fi
if [[ -n "${NUM_ENTITIES}" ]]; then
  CMD+=(--num-entities "${NUM_ENTITIES}")
fi

echo "[Social Force] benchmark=${BENCHMARK} csv=${CSV} device=${DEVICE} config=${CONFIG} N_r=${NUM_ROBOTS:-default} N_e=${NUM_ENTITIES:-default}"
"${CMD[@]}"
