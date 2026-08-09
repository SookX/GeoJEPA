#!/bin/bash
# Discoverer/Slurm launcher for bounded LeWM Reacher ablations.
#
# Submit from the GeoJEPA repo root, for example:
#   sbatch slurm/discoverer_reacher_ablation.sh
#
# Useful ablation knobs:
#   ABLATION_NAME=sigreg003 SIGREG_WEIGHT=0.03 sbatch ...
#   ABLATION_NAME=short_h2 HISTORY_SIZE=2 sbatch ...
#   EPOCHS=5 MAX_TRAIN_BATCHES=4000 BATCH_SIZE=64 sbatch ...

#SBATCH --partition=common
#SBATCH --qos=bg-eng-1
#SBATCH --account=bg-eng-1
#SBATCH --job-name=lewm_reacher
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH -o logs/lewm_reacher.%j.out
#SBATCH -e logs/lewm_reacher.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="${PROJECT_DIR:-/valhalla/projects/bg-eng-1/GeoJEPA}"
LEWM_DIR="${LEWM_DIR:-${PROJECT_DIR}/le-wm}"
VIRTUAL_ENV="${VIRTUAL_ENV:-${PROJECT_DIR}/.venv}"
STABLEWM_HOME="${STABLEWM_HOME:-${PROJECT_DIR}/stablewm_home}"
PYTHON="${PYTHON:-${VIRTUAL_ENV}/bin/python}"

ABLATION_NAME="${ABLATION_NAME:-baseline}"
RUN_GROUP="${RUN_GROUP:-reacher_workshop}"
SEED="${SEED:-3072}"
EPOCHS="${EPOCHS:-3}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-2000}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-128}"
BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
IMG_SIZE="${IMG_SIZE:-224}"
LR="${LR:-5e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-3}"
SIGREG_WEIGHT="${SIGREG_WEIGHT:-0.09}"
HISTORY_SIZE="${HISTORY_SIZE:-3}"
NUM_PREDS="${NUM_PREDS:-1}"
PRECISION="${PRECISION:-bf16}"
DATASET_NAME="${DATASET_NAME:-dmc/reacher_random.h5}"
RUN_SMOKE="${RUN_SMOKE:-1}"

[ -d "${PROJECT_DIR}" ] || { echo "Missing PROJECT_DIR=${PROJECT_DIR}"; exit 1; }
[ -d "${LEWM_DIR}" ] || { echo "Missing LEWM_DIR=${LEWM_DIR}"; exit 1; }
[ -x "${PYTHON}" ] || { echo "Missing executable PYTHON=${PYTHON}"; exit 1; }

export VIRTUAL_ENV
export PATH="${VIRTUAL_ENV}/bin:${PATH}"
export PYTHONPATH="${LEWM_DIR}:${PYTHONPATH:-}"
export STABLEWM_HOME
export LOCAL_DATASET_DIR="${LOCAL_DATASET_DIR:-${STABLEWM_HOME}}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTHONIOENCODING=utf-8
export SPT_CACHE_DIR="${SPT_CACHE_DIR:-${PROJECT_DIR}/.cache/stable-pretraining}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "${PROJECT_DIR}"
mkdir -p logs "${STABLEWM_HOME}/checkpoints/${RUN_GROUP}"

"${PYTHON}" - <<'PY'
import torch
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for Discoverer LeWM training.")
for i in range(torch.cuda.device_count()):
    props = torch.cuda.get_device_properties(i)
    print(f"  cuda:{i} = {props.name} | mem={props.total_memory / 1e9:.1f} GB")
PY

RUN_ID="${RUN_GROUP}/${ABLATION_NAME}_seed${SEED}"
OUTPUT_MODEL_NAME="lewm_reacher_${ABLATION_NAME}"

cat <<EOF
====================================================
[$(date '+%Y-%m-%d %H:%M:%S')] Discoverer LeWM Reacher
  project=${PROJECT_DIR}
  lewm_dir=${LEWM_DIR}
  stablewm_home=${STABLEWM_HOME}
  dataset=${DATASET_NAME}
  run_id=${RUN_ID}
  epochs=${EPOCHS}
  max_train_batches=${MAX_TRAIN_BATCHES}
====================================================
EOF

RUN_GROUP="${RUN_GROUP}" \
ABLATION_NAME="${ABLATION_NAME}" \
SEED="${SEED}" \
EPOCHS="${EPOCHS}" \
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES}" \
MAX_VAL_BATCHES="${MAX_VAL_BATCHES}" \
BATCH_SIZE="${BATCH_SIZE}" \
NUM_WORKERS="${NUM_WORKERS}" \
IMG_SIZE="${IMG_SIZE}" \
LR="${LR}" \
WEIGHT_DECAY="${WEIGHT_DECAY}" \
SIGREG_WEIGHT="${SIGREG_WEIGHT}" \
HISTORY_SIZE="${HISTORY_SIZE}" \
NUM_PREDS="${NUM_PREDS}" \
PRECISION="${PRECISION}" \
DATASET_NAME="${DATASET_NAME}" \
OUTPUT_MODEL_NAME="${OUTPUT_MODEL_NAME}" \
RUN_ID="${RUN_ID}" \
RUN_SMOKE="${RUN_SMOKE}" \
PYTHON="${PYTHON}" \
ROOT_DIR="${PROJECT_DIR}" \
LEWM_DIR="${LEWM_DIR}" \
STABLEWM_HOME="${STABLEWM_HOME}" \
LOCAL_DATASET_DIR="${LOCAL_DATASET_DIR}" \
bash scripts/train_reacher_ablation.sh

