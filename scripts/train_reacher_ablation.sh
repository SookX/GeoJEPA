#!/bin/bash
# Bounded LeWM Reacher training launcher for workshop-scale ablations.

set -euo pipefail

DEFAULT_PROJECT_DIR="/valhalla/projects/bg-eng-1/GeoJEPA"
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
DATASET_NAME="${DATASET_NAME:-dmc/reacher_random.h5}"
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
  sigreg_weight=${SIGREG_WEIGHT}
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
        --precision "${PRECISION}" \
        --output-json "${STABLEWM_HOME}/checkpoints/${RUN_ID}/gradient_smoke.json"
fi

T0=$(date +%s)
cd "${LEWM_DIR}"

"${PYTHON}" train.py \
    data=dmc \
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

T1=$(date +%s)
echo "DONE: elapsed=$((T1 - T0))s, checkpoint_root=${STABLEWM_HOME}/checkpoints/${RUN_ID}"

