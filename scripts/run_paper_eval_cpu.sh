#!/bin/bash
# SAGE Paper Table A4/A5/A7 - CPU evaluation
# ETH/UCY only (SDD needs official annotations on server)
# 运行: bash scripts/run_paper_eval_cpu.sh 2>&1 | tee results/paper_eval_cpu.log

RESULT="results/paper_eval_cpu.csv"
CFG="configs/sage.yaml"
DEV="cpu"
SAMPLES=20
STRIDE=10
BATCH=32
LIMIT=500
DS="eth hotel univ zara1 zara2"

rm -f $RESULT

run_homo() {
  local guided=$1
  for ds in $DS; do
    echo "=== [Homogeneous] $ds guided=$guided ==="
    local extra=""
    [ "$guided" = "false" ] && extra="--unguided"
    python -m sage.experiments.eval_real_sage \
      --config configs/sage_homogeneous.yaml \
      --dataset $ds --split test \
      --checkpoint experiments/sage/eth_ucy/homogeneous/v2/${ds}_sage_epoch100.pt \
      --samples $SAMPLES --stride $STRIDE --batch-size $BATCH \
      --limit-samples $LIMIT --device $DEV \
      $extra --save-csv $RESULT
  done
}

run_decoupled() {
  local guided=$1
  for ds in $DS; do
    echo "=== [Decoupled] $ds guided=$guided ==="
    local extra=""
    [ "$guided" = "false" ] && extra="--unguided"
    python -m sage.experiments.eval_real_sage \
      --config configs/sage_decoupled.yaml \
      --dataset $ds --split test \
      --checkpoint experiments/sage/eth_ucy/decoupled/v2/${ds}_sage_epoch100.pt \
      --samples $SAMPLES --stride $STRIDE --batch-size $BATCH \
      --limit-samples $LIMIT --device $DEV \
      $extra --save-csv $RESULT
  done
}

echo "==========================================="
echo "Paper Evaluation - CPU (ETH/UCY only)"
echo "limit-samples=$LIMIT  samples=$SAMPLES  stride=$STRIDE"
echo "Log: results/paper_eval_cpu.log"
echo "Output: $RESULT"
echo "==========================================="

echo ""
echo "=== Phase 1/5: MID Baseline ==="
for ds in $DS; do
  echo "--- [MID] $ds ---"
  python -m sage.experiments.eval_learned_baseline \
    --config $CFG --dataset $ds --split test \
    --checkpoint experiments/baselines/mid/eth_ucy/${ds}_mid_epoch60.pt \
    --baseline mid \
    --samples $SAMPLES --stride $STRIDE --batch-size $BATCH \
    --limit-samples $LIMIT --device $DEV \
    --save-csv $RESULT
done

echo ""
echo "=== Phase 2/5: Homogeneous - Guided ==="
run_homo true

echo ""
echo "=== Phase 3/5: Homogeneous - Unguided ==="
run_homo false

echo ""
echo "=== Phase 4/5: Decoupled - Guided ==="
run_decoupled true

echo ""
echo "=== Phase 5/5: Decoupled - Unguided ==="
run_decoupled false

echo ""
echo "=== ALL DONE ==="
echo "Results saved to: $RESULT"
