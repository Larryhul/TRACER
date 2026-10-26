#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PYTHON="${PYTHON:-.conda/envs/hulan/bin/python}"
exec "$PYTHON" -m sage.experiments.run_sicnav_tracer_loop \
  --response-backend crowdsimplus-paired \
  --candidate-profile sage-recovery \
  --recovery-activation always --recovery-candidates 16 --recovery-directions 48 \
  --kinematic-safety-margin 0.0 --swept-human-guard \
  --crowd-risk-aggregation union-bound --temporal-risk-sampling independent \
  --predictive-clearance-weight 0.15 --predictive-clearance-steps 4 \
  --sage-diversity-weight 0.4 \
  --safety-weight 2.0 --human-cost-weight 0.6 --risk-weight 0.5 \
  --execute-steps 1 --max-steps 60 --windows 60 \
  --seeds 2027 2031 2039 --case-start 48 --cases 16 \
  --policies tracer-persistent --ranking-audit-windows 1 \
  --output-dir experiments/repair_20260905/reproduced "$@"
