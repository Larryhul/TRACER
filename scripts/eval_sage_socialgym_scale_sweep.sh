#!/usr/bin/env bash
set -euo pipefail

CHECKPOINT=${1:?checkpoint path required}
CSV=${2:-results/sage_socialgym_scale_sweep.csv}
DEVICE=${3:-cuda}
CONFIG=${4:-configs/sage_socialgym.yaml}
SAMPLES=${5:-20}
BATCH_SIZE=${6:-128}
NUM_WORKERS=${7:-8}

run_eval() {
  local nr=$1
  local ne=$2
  local mode=$3

  local cmd=(python -m sage.experiments.eval_socialgym_sage
    --config "${CONFIG}"
    --split test
    --checkpoint "${CHECKPOINT}"
    --samples "${SAMPLES}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --num-robots "${nr}"
    --num-entities "${ne}"
    --guidance-scale 0.05
    --role-metrics
    --device "${DEVICE}"
    --save-csv "${CSV}")

  if [[ "${mode}" == "unguided" ]]; then
    cmd+=(--unguided)
  fi

  echo "[SocialGym sweep] mode=${mode} N_r=${nr} N_e=${ne} csv=${CSV}"
  "${cmd[@]}"
}

mkdir -p "$(dirname "${CSV}")"

# Robot-team scalability at the training-density entity count.
for nr in 3 6 9; do
  run_eval "${nr}" 40 guided
  run_eval "${nr}" 40 unguided
done

# Entity-density OOD generalization at the training robot-team size.
# N_e=40 is already covered above by N_r=6, N_e=40.
for ne in 20 80; do
  run_eval 6 "${ne}" guided
  run_eval 6 "${ne}" unguided
done

