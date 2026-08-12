#!/bin/bash
# Submit dependent 200-run evals for B3 across planning value weights.

set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/valhalla/projects/bg-eng-01/GeoJEPA}"
TRAIN_JOB="${TRAIN_JOB:?Set TRAIN_JOB to the successful/active B3 training job id}"
POLICY="${POLICY:-lewm_reacher_b3_goal_value/weights_epoch_3.pt}"
WEIGHTS="${WEIGHTS:-0 0.03 0.1 0.3 1.0}"
PLANNING_VALUE_GAMMA="${PLANNING_VALUE_GAMMA:-0.99}"
LOG_COST_STATS="${LOG_COST_STATS:-1}"
COST_LOG_LIMIT="${COST_LOG_LIMIT:-6}"

cd "${PROJECT_DIR}"

for weight in ${WEIGHTS}; do
    tag_weight="${weight//./p}"
    tag_weight="${tag_weight//-/m}"
    policy_tag="b3_goal_value_w${tag_weight}"
    job_id=$(sbatch --parsable \
        --dependency=afterok:${TRAIN_JOB} \
        --job-name="b3gv_w${tag_weight}" \
        --export=ALL,POLICY="${POLICY}",POLICY_TAG="${policy_tag}",PLANNING_VALUE_WEIGHT="${weight}",PLANNING_VALUE_GAMMA="${PLANNING_VALUE_GAMMA}",LOG_COST_STATS="${LOG_COST_STATS}",COST_LOG_LIMIT="${COST_LOG_LIMIT}",FORCE=1 \
        slurm/discoverer_reacher_eval200_loop.sh)
    echo "${weight} ${job_id}"
done
