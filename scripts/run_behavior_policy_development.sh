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
BEHAVIOR_MODEL="$ROOT/executable_value_full_prefix_nonlinear_v1/value_model.json"
EFFECT_MODEL="$ROOT/pairwise_effect_behavior_extension_v1/candidate_policy.json"
CLASSIFIER_MODEL="$ROOT/pairwise_reliable_behavior_extension_width16_v1/candidate_policy.json"
OUTPUT="$ROOT/behavior_policy_development_width16_v1"
MODEL=""
if [[ -f "$EFFECT_MODEL" ]]; then
    MODEL="$EFFECT_MODEL"
elif [[ -f "$CLASSIFIER_MODEL" ]]; then
    MODEL="$CLASSIFIER_MODEL"
fi
IDENTITY_MODEL="$ROOT/pairwise_reliable_behavior_extension_width16_v1/zero_refinement_identity.json"
IDENTITY_OUTPUT="$ROOT/behavior_policy_reference_identity_width16_v1"

export PYTHONPATH="$REPO:$SOURCE"
export OPENBLAS_NUM_THREADS=2
cd "$REPO"

if [[ -z "$MODEL" ]]; then
    echo '{"status":"NO_ELIGIBLE_MODEL"}' > "$ROOT/behavior_policy_development_width16_v1.json"
    exit 0
fi

if [[ ! -f "$OUTPUT/parent/summary.json" ]]; then
    "$PYTHON" -u -m tracer.experiments.eval_external_information_value \
        --checkpoint "$CHECKPOINT" --calibration "$CALIBRATION" --source-metadata "$METADATA" \
        --output-dir "$OUTPUT/parent" \
        --case-subset $(seq 1340 1371) --audit-stride 4 --navigation-only \
        --executable-value-model "$BEHAVIOR_MODEL" --device cuda
fi

if [[ ! -f "$IDENTITY_MODEL" ]]; then
    "$PYTHON" - "$MODEL" "$IDENTITY_MODEL" <<'PY'
import json
import sys

model = json.loads(open(sys.argv[1]).read())
for member in model["ensemble"]:
    member["output_weight"] = [[0.0 for _ in row] for row in member["output_weight"]]
    member["output_bias"] = [-100.0, -100.0]
model["deployment_status"] = "IDENTITY_AUDIT_ONLY"
with open(sys.argv[2], "x") as stream:
    stream.write(json.dumps(model, indent=2, allow_nan=False) + "\n")
PY
fi

if [[ ! -f "$IDENTITY_OUTPUT/candidate/summary.json" ]]; then
    "$PYTHON" -u -m tracer.experiments.eval_external_information_value \
        --checkpoint "$CHECKPOINT" --calibration "$CALIBRATION" --source-metadata "$METADATA" \
        --output-dir "$IDENTITY_OUTPUT/candidate" --case-splits "$SPLITS" \
        --case-subset 1340 --audit-stride 4 --navigation-only \
        --executable-value-model "$IDENTITY_MODEL" --device cuda
fi

if [[ ! -f "$IDENTITY_OUTPUT/summary.json" ]]; then
    "$PYTHON" -m tracer.experiments.verify_executable_reference_identity \
        --parent-csv "$OUTPUT/parent/episodes.csv" \
        --candidate-csv "$IDENTITY_OUTPUT/candidate/episodes.csv" \
        --output "$IDENTITY_OUTPUT/summary.json"
fi

if [[ ! -f "$OUTPUT/candidate/summary.json" ]]; then
    "$PYTHON" -u -m tracer.experiments.eval_external_information_value \
        --checkpoint "$CHECKPOINT" --calibration "$CALIBRATION" --source-metadata "$METADATA" \
        --output-dir "$OUTPUT/candidate" --case-splits "$SPLITS" \
        --case-subset $(seq 1340 1371) --audit-stride 4 --navigation-only \
        --executable-value-model "$MODEL" --device cuda
fi

if [[ ! -f "$OUTPUT/comparison.json" ]]; then
    "$PYTHON" "$REPO/tracer/experiments/summarize_executable_value_navigation.py" \
        --parent-csv "$OUTPUT/parent/episodes.csv" --candidate-dir "$OUTPUT/candidate" \
        --output "$OUTPUT/comparison.json"
fi

if [[ ! -f "$OUTPUT/reliable_assessment.json" ]]; then
    DEVELOPMENT_ROOT="$OUTPUT" "$PYTHON" - <<'PY'
import csv
import json
import os
from pathlib import Path

from tracer.experiments.assess_reliable_navigation import assess_reliable_navigation

root = Path(os.environ["DEVELOPMENT_ROOT"])
parent = list(csv.DictReader((root / "parent/episodes.csv").open()))
candidate = list(csv.DictReader((root / "candidate/episodes.csv").open()))
report = assess_reliable_navigation(parent, candidate)
(root / "reliable_assessment.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
print(json.dumps({"development_screen_passed": report["development_screen_passed"],
                  "reliability_status": report["reliability_status"]}))
PY
fi

echo "Behavior-policy development navigation completed at $(date -u --iso-8601=seconds)"
