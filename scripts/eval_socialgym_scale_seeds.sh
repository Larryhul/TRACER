#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=${CHECKPOINT:-experiments/tracer/socialgym/nr6_ne40_bs64/socialgym_sage_epoch60.pt}
CSV=${CSV:-results/sage_socialgym_scale_seeds_k20.csv}
SUMMARY=${SUMMARY:-results/sage_socialgym_scale_seed_summary_k20.csv}
CONFIG=${CONFIG:-configs/sage_socialgym.yaml}
DEVICE=${DEVICE:-cuda}
SEEDS=${SEEDS:-"31415 31416 31417"}
SAMPLES=${SAMPLES:-20}
STRIDE=${STRIDE:-10}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-8}
LIMIT_SAMPLES=${LIMIT_SAMPLES:-0}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-0.05}

mkdir -p "$(dirname "${CSV}")"

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "[SocialGym scale seeds] missing checkpoint: ${CHECKPOINT}" >&2
  exit 1
fi

run_eval() {
  local seed=$1
  local nr=$2
  local ne=$3
  local mode=$4
  local cmd=(python -m tracer.experiments.eval_socialgym_sage
    --config "${CONFIG}"
    --split test
    --source style
    --checkpoint "${CHECKPOINT}"
    --samples "${SAMPLES}"
    --stride "${STRIDE}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --num-robots "${nr}"
    --num-entities "${ne}"
    --limit-samples "${LIMIT_SAMPLES}"
    --seed "${seed}"
    --guidance-scale "${GUIDANCE_SCALE}"
    --role-metrics
    --device "${DEVICE}"
    --save-csv "${CSV}")

  if [[ "${mode}" == "unguided" ]]; then
    cmd+=(--unguided)
  fi

  echo "[SocialGym scale seeds] seed=${seed} mode=${mode} N_r=${nr} N_e=${ne} csv=${CSV}"
  "${cmd[@]}"
}

for seed in ${SEEDS}; do
  for nr in 3 6 9; do
    run_eval "${seed}" "${nr}" 40 guided
    run_eval "${seed}" "${nr}" 40 unguided
  done
  for ne in 20 80; do
    run_eval "${seed}" 6 "${ne}" guided
    run_eval "${seed}" 6 "${ne}" unguided
  done
done

python -m tracer.experiments.summarize_synthetic_seed_results   --input "${CSV}"   --output "${SUMMARY}"
