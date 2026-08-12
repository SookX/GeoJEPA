#!/bin/bash
# Evaluate a trained LeWM Reacher policy and write JSON metrics.

set -euo pipefail

DEFAULT_PROJECT_DIR="/valhalla/projects/bg-eng-01/GeoJEPA"
if [ -d "${DEFAULT_PROJECT_DIR}" ]; then
    ROOT_DIR="${ROOT_DIR:-${DEFAULT_PROJECT_DIR}}"
else
    ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
fi

LEWM_DIR="${LEWM_DIR:-${ROOT_DIR}/le-wm}"
STABLEWM_HOME="${STABLEWM_HOME:-${ROOT_DIR}/stablewm_home}"
LOCAL_DATASET_DIR="${LOCAL_DATASET_DIR:-${STABLEWM_HOME}}"
PYTHON="${PYTHON:-python}"

POLICY="${POLICY:-lewm_reacher_baseline/weights_epoch_3.pt}"
DATASET_NAME="${DATASET_NAME:-dmc/reacher_random}"
NUM_EVAL="${NUM_EVAL:-50}"
SAMPLE_COUNT="${SAMPLE_COUNT:-}"
SAMPLE_OFFSET="${SAMPLE_OFFSET:-0}"
GOAL_OFFSET_STEPS="${GOAL_OFFSET_STEPS:-25}"
EVAL_BUDGET="${EVAL_BUDGET:-50}"
IMG_SIZE="${IMG_SIZE:-224}"
SEED="${SEED:-42}"
SOLVER_SAMPLES="${SOLVER_SAMPLES:-300}"
SOLVER_STEPS="${SOLVER_STEPS:-30}"
SOLVER_TOPK="${SOLVER_TOPK:-30}"
PLANNING_VALUE_WEIGHT="${PLANNING_VALUE_WEIGHT:-}"
PLANNING_VALUE_GAMMA="${PLANNING_VALUE_GAMMA:-}"
LOG_COST_STATS="${LOG_COST_STATS:-0}"
COST_LOG_LIMIT="${COST_LOG_LIMIT:-8}"
DEVICE="${DEVICE:-cuda}"
OUTPUT_JSON="${OUTPUT_JSON:-${STABLEWM_HOME}/checkpoints/reacher_workshop/eval_${SEED}_${NUM_EVAL}.json}"
PATHS_JSON="${PATHS_JSON:-}"
PATH_KEYS="${PATH_KEYS:-qpos,qvel,goal_qpos}"
RANDOM_POLICY="${RANDOM_POLICY:-0}"

export STABLEWM_HOME
export LOCAL_DATASET_DIR
export LEWM_DIR
export PYTHONPATH="${LEWM_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTHONIOENCODING=utf-8
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

[ -d "${LEWM_DIR}" ] || { echo "Missing LEWM_DIR=${LEWM_DIR}"; exit 1; }
[ -f "${STABLEWM_HOME}/datasets/dmc/reacher_random.h5" ] || { echo "Missing Reacher dataset"; exit 1; }

args=(
    "${ROOT_DIR}/scripts/evaluate_reacher_policy.py"
    --lewm-dir "${LEWM_DIR}"
    --cache-dir "${STABLEWM_HOME}"
    --policy "${POLICY}"
    --dataset-name "${DATASET_NAME}"
    --num-eval "${NUM_EVAL}"
    --sample-offset "${SAMPLE_OFFSET}"
    --goal-offset-steps "${GOAL_OFFSET_STEPS}"
    --eval-budget "${EVAL_BUDGET}"
    --img-size "${IMG_SIZE}"
    --seed "${SEED}"
    --solver-samples "${SOLVER_SAMPLES}"
    --solver-steps "${SOLVER_STEPS}"
    --solver-topk "${SOLVER_TOPK}"
    --device "${DEVICE}"
    --output-json "${OUTPUT_JSON}"
    --path-keys "${PATH_KEYS}"
)

if [ -n "${SAMPLE_COUNT}" ]; then
    args+=(--sample-count "${SAMPLE_COUNT}")
fi

if [ -n "${PATHS_JSON}" ]; then
    args+=(--paths-json "${PATHS_JSON}")
fi

if [ -n "${PLANNING_VALUE_WEIGHT}" ]; then
    args+=(--planning-value-weight "${PLANNING_VALUE_WEIGHT}")
fi
if [ -n "${PLANNING_VALUE_GAMMA}" ]; then
    args+=(--planning-value-gamma "${PLANNING_VALUE_GAMMA}")
fi
if [ "${LOG_COST_STATS}" = "1" ] || [ "${LOG_COST_STATS}" = "true" ]; then
    args+=(--log-cost-stats --cost-log-limit "${COST_LOG_LIMIT}")
fi

if [ "${RANDOM_POLICY}" = "1" ] || [ "${RANDOM_POLICY}" = "true" ]; then
    args+=(--random-policy)
fi

"${PYTHON}" "${args[@]}"
