#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-${ROOT}/.conda/envs/hulan/bin/python}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/external_emission_sweep_val}"
SEEDS="${SEEDS:-2027 2031 2039}"
WEIGHTS="${WEIGHTS:-0.5 1.0 2.0 5.0}"

for weight in ${WEIGHTS}; do
  for seed in ${SEEDS}; do
    PYTHONPATH=. "${PYTHON}" -u -m tracer.experiments.run_external_response_benchmark \
      --dataset thor_magni \
      --data-dir experiments/external_response_data \
      --output-dir "${OUTPUT_ROOT}/weight${weight}" \
      --seed "${seed}" \
      --device "${DEVICE}" \
      --response-epochs 20 \
      --recurrent-epochs 0 \
      --batch-size 64 \
      --mode-discrimination-weight "${weight}" \
      --evaluation-splits val \
      --skip-history-baseline
  done
done
