#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-$(pwd)}
PYTHON=${PYTHON:-python}
export QT_QPA_PLATFORM=${QT_QPA_PLATFORM:-offscreen}
export ROS_HOSTNAME=${ROS_HOSTNAME:-127.0.0.1}
export ROS_IP=${ROS_IP:-127.0.0.1}
SOURCE=${SOURCE:-raw_data/external/source_repos/socialgym2_official}
RESPONSE_CHECKPOINT=${RESPONSE_CHECKPOINT:-experiments/formal_v2/h2/persistent_trajectory_suite_tracer_effect/seed2027/response.pt}
SAGE_CHECKPOINT=${SAGE_CHECKPOINT:-experiments/pilot/h3_sage_generator_p0_seed2027/synthetic_sage_epoch20.pt}
OUTPUT_ROOT=${OUTPUT_ROOT:-experiments/socialgym2_joint_h3_multiseed_20260906_v5}
EPISODES=${EPISODES:-16}
MAX_STEPS=${MAX_STEPS:-60}
SIM_DT=${SIM_DT:-0.025}
SIM_STEPS_PER_WINDOW=${SIM_STEPS_PER_WINDOW:-16}
DEVICE=${DEVICE:-cuda}
SEEDS=(2027 2031 2039 2053 2063 2069 2081 2089 2099 2111)
SCENARIOS=(envs/scenario/intersection envs/scenario/open)
POLICIES=(tracer-persistent tracer-reset tracer-uniform tracer-no-pair official-goalone official-halt)
BELIEF_COST_LEVERAGE=${BELIEF_COST_LEVERAGE:-1.0}
ROS_RESTART_DELAY=${ROS_RESTART_DELAY:-5}
CALIBRATION_ARTIFACT=${CALIBRATION_ARTIFACT:-experiments/socialgym2_joint_h3_calibration_20260906_v5.json}
RUNTIME_MANIFEST=${RUNTIME_MANIFEST:-experiments/socialgym2_joint_runtime_manifest_20260906_v7/manifest.json}
PHYSICS_AUDIT_ROOT=${PHYSICS_AUDIT_ROOT:-experiments/socialgym2_joint_physics_full_20260906_v1}

cd "$ROOT"
LIKELIHOOD_SCALE=${LIKELIHOOD_SCALE:-$("$PYTHON" -c 'import json,sys; print(json.load(open(sys.argv[1]))["selected"]["likelihood_scale"])' "$CALIBRATION_ARTIFACT")}
mkdir -p "$OUTPUT_ROOT"

inputs=()

is_complete_artifact() {
    local output_dir=$1
    local metrics_file="$output_dir/episode_metrics.csv"
    local metadata_file="$output_dir/metadata.json"
    local manifest_file="$output_dir/manifest.json"
    local episode_count
    local effective_parameters

    [[ -f "$metrics_file" && -f "$metadata_file" && -f "$manifest_file" ]] || return 1
    grep -q '"status": "COMPLETED"' "$manifest_file" || return 1
    grep -q 'tracer-socialgym2-joint-belief-synchronous-v5' "$metadata_file" || return 1
    effective_parameters=$("$PYTHON" - "$metadata_file" <<'PY'
import json
import sys

metadata = json.load(open(sys.argv[1], encoding="utf-8"))
parameters = metadata.get("parameters", {})
print(json.dumps({
    "episodes": parameters.get("episodes"),
    "max_steps": parameters.get("max_steps"),
        "likelihood_scale": parameters.get("likelihood_scale"),
        "sim_dt": parameters.get("sim_dt"),
        "sim_steps_per_window": parameters.get("sim_steps_per_window"),
        "belief_cost_leverage": parameters.get("belief_cost_leverage", 1.0),
}))
PY
    )
    [[ "$effective_parameters" == *"\"episodes\": $EPISODES"* ]] || return 1
    [[ "$effective_parameters" == *"\"max_steps\": $MAX_STEPS"* ]] || return 1
    [[ "$effective_parameters" == *"\"likelihood_scale\": $LIKELIHOOD_SCALE"* ]] || return 1
    [[ "$effective_parameters" == *"\"sim_dt\": $SIM_DT"* ]] || return 1
    [[ "$effective_parameters" == *"\"sim_steps_per_window\": $SIM_STEPS_PER_WINDOW"* ]] || return 1
    [[ "$effective_parameters" == *"\"belief_cost_leverage\": $BELIEF_COST_LEVERAGE"* ]] || return 1
    grep -q '"artifact": "'"$CALIBRATION_ARTIFACT"'"' "$metadata_file" || return 1
    episode_count=$(awk 'NR > 1 && NF { count += 1 } END { print count + 0 }' "$metrics_file")
    [[ "$episode_count" -eq "$EPISODES" ]]
}

for scenario in "${SCENARIOS[@]}"; do
    scenario_name=${scenario##*/}
    for seed in "${SEEDS[@]}"; do
        for policy in "${POLICIES[@]}"; do
            # RosSocialEnv is a process-wide singleton and the official launch
            # file contains fixed node names.  One policy per process keeps all
            # policies paired by reset seed while preventing stale ROS nodes
            # from contaminating the next policy.
            output_dir="$OUTPUT_ROOT/${scenario_name}_seed${seed}/${policy}"
            if is_complete_artifact "$output_dir"; then
                echo "Reusing completed artifact: $output_dir"
                inputs+=("$output_dir")
                continue
            fi
            "$PYTHON" -m tracer.experiments.run_socialgym2_tracer_loop \
                --source "$SOURCE" \
                --output-dir "$output_dir" \
                --response-checkpoint "$RESPONSE_CHECKPOINT" \
                --sage-checkpoint "$SAGE_CHECKPOINT" \
                --device "$DEVICE" \
                --seed "$seed" \
                --episodes "$EPISODES" \
                --max-steps "$MAX_STEPS" \
                --num-agents 3 \
                --num-humans 2 \
                --partial-observation \
                --scenario "$scenario" \
                --likelihood-scale "$LIKELIHOOD_SCALE" \
                --belief-cost-leverage "$BELIEF_COST_LEVERAGE" \
                --sim-dt "$SIM_DT" \
                --sim-steps-per-window "$SIM_STEPS_PER_WINDOW" \
                --calibration-artifact "$CALIBRATION_ARTIFACT" \
                --runtime-manifest "$RUNTIME_MANIFEST" \
                --policies "$policy" \
                --belief-selection-audit
            inputs+=("$output_dir")
            sleep "$ROS_RESTART_DELAY"
        done
    done
done

"$PYTHON" -m tracer.experiments.aggregate_socialgym2_tracer \
    --output-dir "$OUTPUT_ROOT/aggregate" \
    --expected-policies "${POLICIES[@]}" \
    --expected-seeds "${SEEDS[@]}" \
    --expected-scenarios "${SCENARIOS[@]}" \
    --expected-episodes "$EPISODES" \
    --physics-audits "$PHYSICS_AUDIT_ROOT/intersection.json" "$PHYSICS_AUDIT_ROOT/open.json" \
    -- \
    "${inputs[@]}"
