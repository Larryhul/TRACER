#!/usr/bin/env bash
set -euo pipefail

ROOT=experiments/planner_aligned_response_20260906
REPO=.
PYTHON="$REPO/.conda/envs/hulan/bin/python"
SOURCE="$ROOT/reliability_source_v9"
SPLITS="$ROOT/behavior_case_splits_policy_extension_v1.json"
METADATA="$REPO/experiments/h3_external_static_response_calibrated_20260906/metadata.json"
CHECKPOINT="$REPO/experiments/tracer_live_belief_static_regret_20260906/seed2027/response.pt"
CALIBRATION="$REPO/experiments/external_static_response_calibration_20260906/seed2027/calibration.json"
EFFECT_MODEL="$ROOT/pairwise_effect_behavior_extension_v1/candidate_policy.json"
CLASSIFIER_MODEL="$ROOT/pairwise_reliable_behavior_extension_width16_v1/candidate_policy.json"
MODEL=""
if [[ -f "$EFFECT_MODEL" ]]; then
    MODEL="$EFFECT_MODEL"
elif [[ -f "$CLASSIFIER_MODEL" ]]; then
    MODEL="$CLASSIFIER_MODEL"
fi
DEVELOPMENT="$ROOT/behavior_policy_development_width16_v1/reliable_assessment.json"
OUTPUT="$ROOT/pairwise_effect_behavior_confirmation_v1"

export PYTHONPATH="$REPO:$SOURCE"
cd "$REPO"

if [[ -z "$MODEL" || ! -f "$DEVELOPMENT" ]]; then
    echo '{"status":"MISSING_DEVELOPMENT_OR_MODEL"}' > "$ROOT/pairwise_effect_behavior_confirmation_v1.json"
    exit 0
fi

if ! "$PYTHON" - "$DEVELOPMENT" <<'PY'
import json
import sys
if not json.loads(open(sys.argv[1]).read()).get("development_screen_passed", False):
    raise SystemExit(1)
PY
then
    echo '{"status":"DEVELOPMENT_SCREEN_FAILED"}' > "$ROOT/pairwise_effect_behavior_confirmation_v1.json"
    exit 0
fi

if [[ ! -f "$OUTPUT/summary.json" ]]; then
    "$PYTHON" -u "$REPO/sage/experiments/confirm_executable_value.py" \
        --checkpoint "$CHECKPOINT" --calibration "$CALIBRATION" --source-metadata "$METADATA" \
        --value-model "$MODEL" --output-dir "$OUTPUT" --case-start 1400 --cases 64 \
        --excluded-cases $(seq 628 755) --device cuda
fi

echo "Behavior-policy independent confirmation completed at $(date -u --iso-8601=seconds)"
