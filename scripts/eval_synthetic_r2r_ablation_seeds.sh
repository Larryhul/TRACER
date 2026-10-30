#!/usr/bin/env bash
set -euo pipefail

CSV=${CSV:-results/sage_synthetic_r2r_ablation_seeds_k20.csv}
SUMMARY=${SUMMARY:-results/sage_synthetic_r2r_ablation_seed_summary_k20.csv}
CONFIG=${CONFIG:-configs/sage_synthetic.yaml}
DEVICE=${DEVICE:-cuda}
SEEDS=${SEEDS:-"2027 2028 2029"}
ROBOT_COUNTS=${ROBOT_COUNTS:-"12 20"}
NUM_ENTITIES=${NUM_ENTITIES:-50}
SAMPLES=${SAMPLES:-20}
STRIDE=${STRIDE:-10}
BATCH_SIZE=${BATCH_SIZE:-256}
NUM_WORKERS=${NUM_WORKERS:-8}
LIMIT_SAMPLES=${LIMIT_SAMPLES:-0}
GUIDANCE_SCALE=${GUIDANCE_SCALE:-0.05}
CHECKPOINT_TEMPLATE=${CHECKPOINT_TEMPLATE:-"experiments/tracer/synthetic/nrNR_neNE/synthetic_sage_epoch50.pt"}

mkdir -p "$(dirname "${CSV}")"

checkpoint_for() {
  local nr=$1
  local ne=$2
  local path=${CHECKPOINT_TEMPLATE//NR/${nr}}
  path=${path//NE/${ne}}
  printf '%s' "${path}"
}

run_variant() {
  local seed=$1
  local nr=$2
  local label=$3
  local use_edges=$4
  local w_r2r=$5
  local ckpt
  ckpt=$(checkpoint_for "${nr}" "${NUM_ENTITIES}")
  if [[ ! -f "${ckpt}" ]]; then
    echo "[R2R seed sweep] missing checkpoint: ${ckpt}" >&2
    return 1
  fi

  local cmd=(python -m tracer.experiments.eval_synthetic_sage
    --config "${CONFIG}"
    --split test
    --checkpoint "${ckpt}"
    --samples "${SAMPLES}"
    --stride "${STRIDE}"
    --batch-size "${BATCH_SIZE}"
    --num-workers "${NUM_WORKERS}"
    --num-robots "${nr}"
    --num-entities "${NUM_ENTITIES}"
    --limit-samples "${LIMIT_SAMPLES}"
    --seed "${seed}"
    --guidance-scale "${GUIDANCE_SCALE}"
    --w-robot-robot-collision "${w_r2r}"
    --role-metrics
    --device "${DEVICE}"
    --save-csv "${CSV}")

  if [[ "${use_edges}" == "false" ]]; then
    cmd+=(--disable-robot-robot-edges)
  fi

  echo "[R2R seed sweep] seed=${seed} N_r=${nr} variant=${label} edges=${use_edges} w_r2r=${w_r2r} csv=${CSV}"
  "${cmd[@]}"
}

for seed in ${SEEDS}; do
  for nr in ${ROBOT_COUNTS}; do
    run_variant "${seed}" "${nr}" full true 1.5
    run_variant "${seed}" "${nr}" edges_only true 0.0
    run_variant "${seed}" "${nr}" energy_only false 1.5
    run_variant "${seed}" "${nr}" no_r2r_coord false 0.0
  done
done

python -m tracer.experiments.summarize_synthetic_seed_results   --input "${CSV}"   --output "${SUMMARY}"
