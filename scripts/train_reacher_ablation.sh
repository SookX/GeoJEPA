#!/bin/bash
# Bounded LeWM Reacher training launcher for workshop-scale ablations.

set -euo pipefail

DEFAULT_PROJECT_DIR="/valhalla/projects/bg-eng-01/GeoJEPA"
if [ -d "${DEFAULT_PROJECT_DIR}" ]; then
    ROOT_DIR="${ROOT_DIR:-${DEFAULT_PROJECT_DIR}}"
else
    ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
fi
LEWM_DIR="${LEWM_DIR:-${ROOT_DIR}/le-wm}"
PYTHON="${PYTHON:-python}"
STABLEWM_HOME="${STABLEWM_HOME:-${ROOT_DIR}/stablewm_home}"
LOCAL_DATASET_DIR="${LOCAL_DATASET_DIR:-${STABLEWM_HOME}}"

RUN_GROUP="${RUN_GROUP:-reacher_workshop}"
ABLATION_NAME="${ABLATION_NAME:-baseline}"
SEED="${SEED:-3072}"
EPOCHS="${EPOCHS:-3}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-2000}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-128}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
IMG_SIZE="${IMG_SIZE:-224}"
LR="${LR:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-3}"
SIGREG_WEIGHT="${SIGREG_WEIGHT:-0.09}"
HISTORY_SIZE="${HISTORY_SIZE:-3}"
NUM_PREDS="${NUM_PREDS:-1}"
PRECISION="${PRECISION:-bf16}"
MODEL_CONFIG="${MODEL_CONFIG:-lewm}"
DATASET_NAME="${DATASET_NAME:-dmc/reacher_random.h5}"
DATASET_KEYS_TO_LOAD="${DATASET_KEYS_TO_LOAD:-}"
DATASET_KEYS_TO_CACHE="${DATASET_KEYS_TO_CACHE:-}"
VALUE_WEIGHT="${VALUE_WEIGHT:-}"
VALUE_GAMMA="${VALUE_GAMMA:-0.99}"
VALUE_HORIZON="${VALUE_HORIZON:-}"
PLANNING_VALUE_WEIGHT="${PLANNING_VALUE_WEIGHT:-}"
PLANNING_VALUE_GAMMA="${PLANNING_VALUE_GAMMA:-}"
GEO_ANISO_WEIGHT="${GEO_ANISO_WEIGHT:-}"
GEO_SCALE_WEIGHT="${GEO_SCALE_WEIGHT:-}"
GEO_RESOLUTION="${GEO_RESOLUTION:-}"
GEO_WEIGHT="${GEO_WEIGHT:-}"
GEO_ALPHA_WEIGHT="${GEO_ALPHA_WEIGHT:-}"
GEO_ALPHA_TAU="${GEO_ALPHA_TAU:-}"
GEO_ALPHA_MIN="${GEO_ALPHA_MIN:-}"
GEO_ALPHA0="${GEO_ALPHA0:-1.0}"
GEO_TEACHER_WEIGHT="${GEO_TEACHER_WEIGHT:-}"
GEO_TEACHER_MODEL_PATH="${GEO_TEACHER_MODEL_PATH:-}"
GEO_K="${GEO_K:-4}"
GEO_MAX_POINTS="${GEO_MAX_POINTS:-}"
GEO_TARGET="${GEO_TARGET:-}"
GEO_ACTION_BASIS="${GEO_ACTION_BASIS:-}"
GEO_FRAMESKIP="${GEO_FRAMESKIP:-5}"
DATASET_NUM_STEPS="${DATASET_NUM_STEPS:-}"
INIT_MODEL_PATH="${INIT_MODEL_PATH:-}"
OUTPUT_MODEL_NAME="${OUTPUT_MODEL_NAME:-lewm_reacher_${ABLATION_NAME}}"
RUN_ID="${RUN_ID:-${RUN_GROUP}/${ABLATION_NAME}_seed${SEED}}"
RUN_SMOKE="${RUN_SMOKE:-1}"
SMOKE_BATCH_SIZE="${SMOKE_BATCH_SIZE:-2}"
SMOKE_STEPS="${SMOKE_STEPS:-2}"

export STABLEWM_HOME
export LOCAL_DATASET_DIR
export LEWM_DIR
export PYTHONPATH="${LEWM_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTHONIOENCODING=utf-8
export SPT_CACHE_DIR="${SPT_CACHE_DIR:-${ROOT_DIR}/.cache/stable-pretraining}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

[ -d "${LEWM_DIR}" ] || { echo "Missing LEWM_DIR=${LEWM_DIR}"; exit 1; }
mkdir -p "${STABLEWM_HOME}/checkpoints/${RUN_GROUP}" "${ROOT_DIR}/logs"

cat <<EOF
====================================================
LeWM Reacher ablation
  run_id=${RUN_ID}
  lewm_dir=${LEWM_DIR}
  stablewm_home=${STABLEWM_HOME}
  dataset=${DATASET_NAME}
  epochs=${EPOCHS}
  max_train_batches=${MAX_TRAIN_BATCHES}
  batch_size=${BATCH_SIZE}
  img_size=${IMG_SIZE}
  model_config=${MODEL_CONFIG}
  sigreg_weight=${SIGREG_WEIGHT}
  value_weight=${VALUE_WEIGHT:-0}
  value_horizon=${VALUE_HORIZON:-0}
  planning_value_weight=${PLANNING_VALUE_WEIGHT:-0}
  geo_aniso_weight=${GEO_ANISO_WEIGHT:-0}
  geo_scale_weight=${GEO_SCALE_WEIGHT:-0}
  geo_resolution=${GEO_RESOLUTION:-b}
  geo_weight=${GEO_WEIGHT:-0}
  geo_alpha_weight=${GEO_ALPHA_WEIGHT:-0}
  geo_alpha_tau=${GEO_ALPHA_TAU:-default}
  geo_alpha_min=${GEO_ALPHA_MIN:-default}
  geo_alpha0=${GEO_ALPHA0}
  geo_teacher_weight=${GEO_TEACHER_WEIGHT:-0}
  geo_teacher_model_path=${GEO_TEACHER_MODEL_PATH:-none}
  geo_k=${GEO_K}
  geo_max_points=${GEO_MAX_POINTS:-all}
  geo_target=${GEO_TARGET:-effect}
  geo_action_basis=${GEO_ACTION_BASIS:-full}
  init_model_path=${INIT_MODEL_PATH:-none}
  dataset_num_steps=${DATASET_NUM_STEPS:-default}
====================================================
EOF

if [ "${RUN_SMOKE}" = "1" ] || [ "${RUN_SMOKE}" = "true" ]; then
    "${PYTHON}" "${ROOT_DIR}/scripts/gradient_smoke.py" \
        --lewm-dir "${LEWM_DIR}" \
        --cache-dir "${STABLEWM_HOME}" \
        --dataset-name "${DATASET_NAME}" \
        --batch-size "${SMOKE_BATCH_SIZE}" \
        --steps "${SMOKE_STEPS}" \
        --img-size "${IMG_SIZE}" \
        --history-size "${HISTORY_SIZE}" \
        --num-preds "${NUM_PREDS}" \
        --model-config "${MODEL_CONFIG}" \
        --precision "${PRECISION}" \
        --output-json "${STABLEWM_HOME}/checkpoints/${RUN_ID}/gradient_smoke.json" \
        --value-weight "${VALUE_WEIGHT:-0}" \
        --value-gamma "${VALUE_GAMMA}" \
        --value-horizon "${VALUE_HORIZON}" \
        --planning-value-weight "${PLANNING_VALUE_WEIGHT}" \
        --planning-value-gamma "${PLANNING_VALUE_GAMMA}" \
        --geo-aniso-weight "${GEO_ANISO_WEIGHT}" \
        --geo-scale-weight "${GEO_SCALE_WEIGHT}" \
        --geo-resolution "${GEO_RESOLUTION}" \
        --geo-weight "${GEO_WEIGHT}" \
        --geo-alpha-weight "${GEO_ALPHA_WEIGHT}" \
        --geo-alpha-tau "${GEO_ALPHA_TAU}" \
        --geo-alpha-min "${GEO_ALPHA_MIN}" \
        --geo-alpha0 "${GEO_ALPHA0}" \
        --geo-teacher-weight "${GEO_TEACHER_WEIGHT}" \
        --geo-teacher-model-path "${GEO_TEACHER_MODEL_PATH}" \
        --geo-k "${GEO_K}" \
        --geo-max-points "${GEO_MAX_POINTS}" \
        --geo-target "${GEO_TARGET}" \
        --geo-action-basis "${GEO_ACTION_BASIS}" \
        --geo-frameskip "${GEO_FRAMESKIP}" \
        --init-model-path "${INIT_MODEL_PATH}" \
        --dataset-num-steps "${DATASET_NUM_STEPS}" \
        --dataset-keys-to-load "${DATASET_KEYS_TO_LOAD}" \
        --dataset-keys-to-cache "${DATASET_KEYS_TO_CACHE}"
fi

T0=$(date +%s)
cd "${LEWM_DIR}"

train_args=(
    train.py
    data=dmc \
    model="${MODEL_CONFIG}" \
    data.dataset.name="${DATASET_NAME}" \
    output_model_name="${OUTPUT_MODEL_NAME}" \
    subdir="${RUN_ID}" \
    seed="${SEED}" \
    img_size="${IMG_SIZE}" \
    history_size="${HISTORY_SIZE}" \
    num_preds="${NUM_PREDS}" \
    loader.batch_size="${BATCH_SIZE}" \
    loader.num_workers="${NUM_WORKERS}" \
    loader.persistent_workers=false \
    loader.prefetch_factor=null \
    trainer.max_epochs="${EPOCHS}" \
    +trainer.limit_train_batches="${MAX_TRAIN_BATCHES}" \
    +trainer.limit_val_batches="${MAX_VAL_BATCHES}" \
    trainer.precision="${PRECISION}" \
    optimizer.lr="${LR}" \
    optimizer.weight_decay="${WEIGHT_DECAY}" \
    loss.sigreg.weight="${SIGREG_WEIGHT}" \
    wandb.enabled=false
)

if [ -n "${DATASET_KEYS_TO_LOAD}" ]; then
    train_args+=("data.dataset.keys_to_load=[${DATASET_KEYS_TO_LOAD}]")
fi
if [ -n "${DATASET_KEYS_TO_CACHE}" ]; then
    train_args+=("data.dataset.keys_to_cache=[${DATASET_KEYS_TO_CACHE}]")
fi
if [ -n "${VALUE_WEIGHT}" ]; then
    train_args+=("+loss.value.weight=${VALUE_WEIGHT}")
    train_args+=("+loss.value.gamma=${VALUE_GAMMA}")
    if [ -n "${VALUE_HORIZON}" ]; then
        train_args+=("+loss.value.horizon=${VALUE_HORIZON}")
    fi
fi
if [ -n "${PLANNING_VALUE_WEIGHT}" ]; then
    train_args+=("model.planning_value_weight=${PLANNING_VALUE_WEIGHT}")
fi
if [ -n "${PLANNING_VALUE_GAMMA}" ]; then
    train_args+=("model.planning_value_gamma=${PLANNING_VALUE_GAMMA}")
fi
if [ -n "${DATASET_NUM_STEPS}" ]; then
    train_args+=("data.dataset.num_steps=${DATASET_NUM_STEPS}")
fi
if [ -n "${GEO_ANISO_WEIGHT}" ]; then
    train_args+=("+loss.geo.aniso_weight=${GEO_ANISO_WEIGHT}")
fi
if [ -n "${GEO_SCALE_WEIGHT}" ]; then
    train_args+=("+loss.geo.scale_weight=${GEO_SCALE_WEIGHT}")
fi
if [ -n "${GEO_RESOLUTION}" ]; then
    train_args+=("+loss.geo.resolution=${GEO_RESOLUTION}")
fi
if [ -n "${GEO_WEIGHT}" ]; then
    train_args+=("+loss.geo.weight=${GEO_WEIGHT}")
fi
if [ -n "${GEO_ALPHA_WEIGHT}" ]; then
    train_args+=("+loss.geo.alpha_weight=${GEO_ALPHA_WEIGHT}")
fi
if [ -n "${GEO_ALPHA_TAU}" ]; then
    train_args+=("+loss.geo.alpha_tau=${GEO_ALPHA_TAU}")
fi
if [ -n "${GEO_TEACHER_WEIGHT}" ]; then
    train_args+=("+loss.geo.teacher_weight=${GEO_TEACHER_WEIGHT}")
fi
if [ -n "${GEO_TEACHER_MODEL_PATH}" ]; then
    train_args+=("+loss.geo.teacher_model_path=${GEO_TEACHER_MODEL_PATH}")
fi
if [ -n "${GEO_ANISO_WEIGHT}" ] || [ -n "${GEO_SCALE_WEIGHT}" ] || [ -n "${GEO_WEIGHT}" ] || [ -n "${GEO_ALPHA_WEIGHT}" ] || [ -n "${GEO_TEACHER_WEIGHT}" ]; then
    train_args+=("+loss.geo.alpha0=${GEO_ALPHA0}")
    train_args+=("+loss.geo.k=${GEO_K}")
    if [ -n "${GEO_MAX_POINTS}" ]; then
        train_args+=("+loss.geo.max_points=${GEO_MAX_POINTS}")
    fi
    if [ -n "${GEO_TARGET}" ]; then
        train_args+=("+loss.geo.target=${GEO_TARGET}")
    fi
    if [ -n "${GEO_ACTION_BASIS}" ]; then
        train_args+=("+loss.geo.action_basis=${GEO_ACTION_BASIS}")
    fi
    if [ -n "${GEO_FRAMESKIP}" ]; then
        train_args+=("+loss.geo.frameskip=${GEO_FRAMESKIP}")
    fi
fi
if [ -n "${GEO_ALPHA_MIN}" ]; then
    train_args+=("model.alpha_head.alpha_min=${GEO_ALPHA_MIN}")
fi
if [ -n "${INIT_MODEL_PATH}" ]; then
    train_args+=("+init_model_path=${INIT_MODEL_PATH}")
fi

"${PYTHON}" "${train_args[@]}"

T1=$(date +%s)
echo "DONE: elapsed=$((T1 - T0))s, checkpoint_root=${STABLEWM_HOME}/checkpoints/${RUN_ID}"
