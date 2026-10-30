#!/usr/bin/env bash
set -euo pipefail

DATASET_ROOT=${1:-data/external/bi3}
OUTPUT_DIR=${2:-experiments/real_h1_bi3_20260906}
BENCHMARK_PREFIX=${3:-experiments/real_h1_bi3_benchmark_20260906}
PYTHON=${PYTHON:-"$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/.conda/envs/hulan/bin/python"}
DEVICE=${DEVICE:-cuda}
SEEDS=${SEEDS:-"2027 2031 2039"}
REPORT_PREFIX=${REPORT_PREFIX:-reports/real_h1_bi3_multiseed_20260906}
read -r -a SEED_LIST <<< "$SEEDS"

REPO_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
cd "$REPO_ROOT"

"$PYTHON" -m tracer.data.real_response \
  --dataset bi3 \
  --root "$DATASET_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --history 9 --horizon 12 --windows 8 --execute-steps 3 \
  --max-entities 2 --min-entities 1 --dt 0.4 --episode-stride 1

BENCHMARK_ROOTS=()
for seed in "${SEED_LIST[@]}"; do
  BENCHMARK_ROOTS+=("${BENCHMARK_PREFIX}_seed${seed}")
  "$PYTHON" -m tracer.experiments.run_external_response_benchmark \
    --dataset bi3 \
    --data-dir "$OUTPUT_DIR" \
    --output-dir "${BENCHMARK_PREFIX}_seed${seed}" \
    --device "$DEVICE" \
    --seed "$seed" \
    --data-seed 20260906 \
    --history 9 --horizon 12 --windows 8 --execute-steps 3 --dt 0.4 \
    --max-entities 2 \
    --response-epochs 10 --recurrent-epochs 10 \
    --velocity-residual --geometry-features --dynamic-interaction-features \
    --temporal-pooling attention \
    --evaluation-splits both
done

"$PYTHON" -m tracer.experiments.summarize_real_h1 \
  --dataset Bi3 --output "$REPORT_PREFIX" \
  --benchmark-root "${BENCHMARK_ROOTS[@]}" --seed "${SEED_LIST[@]}"
