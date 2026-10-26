#!/usr/bin/env bash
set -euo pipefail

CSV=${CSV:-results/social_force_seed_sweep.csv}
SUMMARY=${SUMMARY:-results/social_force_seed_summary.csv}
SYNTHETIC_CONFIG=${SYNTHETIC_CONFIG:-configs/sage_synthetic.yaml}
SOCIALGYM_CONFIG=${SOCIALGYM_CONFIG:-configs/sage_socialgym.yaml}
DEVICE=${DEVICE:-cuda}
SYNTHETIC_SEEDS=${SYNTHETIC_SEEDS:-"2027 2028 2029"}
SOCIALGYM_SEEDS=${SOCIALGYM_SEEDS:-"31415 31416 31417"}
ROBOT_COUNTS=${ROBOT_COUNTS:-"3 6 9 12 16 20"}
SYNTHETIC_NUM_ENTITIES=${SYNTHETIC_NUM_ENTITIES:-50}
SOCIALGYM_NUM_ROBOTS=${SOCIALGYM_NUM_ROBOTS:-6}
SOCIALGYM_NUM_ENTITIES=${SOCIALGYM_NUM_ENTITIES:-40}
BATCH_SIZE=${BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-8}
LIMIT_SAMPLES=${LIMIT_SAMPLES:-0}

mkdir -p "$(dirname "${CSV}")"

run_social_force() {
  local benchmark=$1
  local seed=$2
  local nr=$3
  local ne=$4
  local config=$5
  local cmd=(python -m sage.experiments.eval_social_force_baseline
    --benchmark "${benchmark}"
    --config "${config}"
    --split test
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --num-robots "${nr}"
    --num-entities "${ne}"
    --limit-samples "${LIMIT_SAMPLES}"
    --seed "${seed}"
    --role-metrics
    --device "${DEVICE}"
    --save-csv "${CSV}")

  if [[ "${benchmark}" == "socialgym" ]]; then
    cmd+=(--source style)
  fi

  echo "[Social Force seed sweep] benchmark=${benchmark} seed=${seed} N_r=${nr} N_e=${ne} device=${DEVICE} csv=${CSV}"
  "${cmd[@]}"
}

for seed in ${SYNTHETIC_SEEDS}; do
  for nr in ${ROBOT_COUNTS}; do
    run_social_force synthetic "${seed}" "${nr}" "${SYNTHETIC_NUM_ENTITIES}" "${SYNTHETIC_CONFIG}"
  done
done

for seed in ${SOCIALGYM_SEEDS}; do
  run_social_force socialgym "${seed}" "${SOCIALGYM_NUM_ROBOTS}" "${SOCIALGYM_NUM_ENTITIES}" "${SOCIALGYM_CONFIG}"
done

python -m sage.experiments.summarize_synthetic_seed_results   --input "${CSV}"   --output "${SUMMARY}"
