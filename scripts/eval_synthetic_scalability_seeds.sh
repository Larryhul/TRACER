#!/usr/bin/env bash
set -euo pipefail

CSV=${1:-results/sage_synthetic_scalability_seeds_k20.csv}
SUMMARY_CSV=${2:-results/sage_synthetic_scalability_seed_summary_k20.csv}
DEVICE=${3:-cuda}
CONFIG=${4:-configs/sage_synthetic.yaml}
SAMPLES=${5:-20}
BATCH_SIZE=${6:-256}
NUM_WORKERS=${7:-8}
NUM_ENTITIES=${NUM_ENTITIES:-50}
EPOCH=${EPOCH:-50}
SEEDS=${SEEDS:-"2027 2028 2029"}
ROBOT_COUNTS=${ROBOT_COUNTS:-"3 6 9 12 16 20"}
LIMIT_SAMPLES=${LIMIT_SAMPLES:-0}

mkdir -p "$(dirname "${CSV}")"
mkdir -p "$(dirname "${SUMMARY_CSV}")"

run_eval() {
  local nr=$1
  local seed=$2
  local mode=$3
  local checkpoint="experiments/sage/synthetic/nr${nr}_ne${NUM_ENTITIES}/synthetic_sage_epoch${EPOCH}.pt"

  if [[ ! -f "${checkpoint}" ]]; then
    echo "[Synthetic seeds] missing checkpoint=${checkpoint}" >&2
    return 1
  fi

  local cmd=(python -m sage.experiments.eval_synthetic_sage
    --config "${CONFIG}"
    --split test
    --checkpoint "${checkpoint}"
    --num-robots "${nr}"
    --num-entities "${NUM_ENTITIES}"
    --samples "${SAMPLES}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --seed "${seed}"
    --guidance-scale 0.05
    --role-metrics
    --device "${DEVICE}"
    --save-csv "${CSV}")

  if [[ "${LIMIT_SAMPLES}" != "0" ]]; then
    cmd+=(--limit-samples "${LIMIT_SAMPLES}")
  fi
  if [[ "${mode}" == "unguided" ]]; then
    cmd+=(--unguided)
  fi

  echo "[Synthetic seeds] mode=${mode} seed=${seed} N_r=${nr} N_e=${NUM_ENTITIES} checkpoint=${checkpoint} csv=${CSV}"
  "${cmd[@]}"
}

for seed in ${SEEDS}; do
  for nr in ${ROBOT_COUNTS}; do
    run_eval "${nr}" "${seed}" guided
    run_eval "${nr}" "${seed}" unguided
  done
done

python -m sage.experiments.summarize_synthetic_seed_results   --input "${CSV}"   --output "${SUMMARY_CSV}"

echo "[Synthetic seeds] summary_csv=${SUMMARY_CSV}"
