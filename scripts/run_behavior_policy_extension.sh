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

export PYTHONPATH="$SOURCE"
export OPENBLAS_NUM_THREADS=2
cd "$REPO"

collect() {
    local split="$1"
    local name="$2"
    shift 2
    if [[ -f "$ROOT/$name/manifest.json" ]] && grep -q '"status": "COMPLETED"' "$ROOT/$name/manifest.json"; then
        echo "SKIP completed $name"
        return
    fi
    "$PYTHON" -u "$SOURCE/tracer/experiments/collect_external_planner_supervision.py" \
        --source-metadata "$METADATA" --checkpoint "$CHECKPOINT" --calibration "$CALIBRATION" \
        --output-dir "$ROOT/$name" --split "$split" --case-splits "$SPLITS" \
        --case-subset "$@" --audit-stride 4 --continuation-windows 60 --continuation-branches 6 \
        --behavior-value-model "$BEHAVIOR_MODEL" --device cuda \
        > "$ROOT/$name.log" 2>&1
}

collect train behavior_policy_train_1308_1315_v1 $(seq 1308 1315)
collect train behavior_policy_train_1316_1323_v1 $(seq 1316 1323)
collect train behavior_policy_train_1324_1331_v1 $(seq 1324 1331)
collect train behavior_policy_train_1332_1339_v1 $(seq 1332 1339)
collect val behavior_policy_val_1340_1347_v1 $(seq 1340 1347)
collect val behavior_policy_val_1348_1355_v1 $(seq 1348 1355)
collect val behavior_policy_val_1356_1363_v1 $(seq 1356 1363)
collect val behavior_policy_val_1364_1371_v1 $(seq 1364 1371)

echo "Behavior-policy extension collection completed at $(date -u --iso-8601=seconds)"
bash "$REPO/scripts/finish_behavior_policy_extension.sh"
