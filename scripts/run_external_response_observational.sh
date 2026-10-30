#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PYTHON="${PYTHON:-${ROOT}/.conda/envs/hulan/bin/python}"
DEVICE="${DEVICE:-cuda}"
DATA_DIR="${DATA_DIR:-experiments/external_response_data}"
OUTPUT_ROOT="${OUTPUT_ROOT:-experiments/external_response_benchmark_formal_v2}"
SEEDS="${SEEDS:-2027 2031 2039 2053 2063}"

for dataset in thor_magni huron_sacson; do
  max_entities=4
  if [[ "${dataset}" == "huron_sacson" ]]; then
    max_entities=1
  fi
  for seed in ${SEEDS}; do
    PYTHONPATH=. "${PYTHON}" -u -m tracer.experiments.run_external_response_benchmark \
      --dataset "${dataset}" \
      --data-dir "${DATA_DIR}" \
      --output-dir "${OUTPUT_ROOT}/${dataset}" \
      --seed "${seed}" \
      --device "${DEVICE}" \
      --max-entities "${max_entities}" \
      --response-epochs 20 \
      --recurrent-epochs 20 \
      --batch-size 64
  done
done
