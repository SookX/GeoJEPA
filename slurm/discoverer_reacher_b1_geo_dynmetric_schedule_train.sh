#!/bin/bash
# Discoverer/Slurm launcher for scheduled from-scratch B1 + dynamics-metric matching.
#
# Stages:
#   1. train B1 normally from scratch;
#   2. continue with trace-normalized dynamics-metric loss at 1e-4;
#   3. continue with trace-normalized dynamics-metric loss at 1e-3.

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=lewm_b1dyns
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH -o logs/lewm_b1dyns.%j.out
#SBATCH -e logs/lewm_b1dyns.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="${PROJECT_DIR:-/valhalla/projects/bg-eng-01/GeoJEPA}"
TORCH_ENV="${TORCH_ENV:-/valhalla/projects/bg-eng-01/conda_envs/torch}"
LEWM_DIR="${LEWM_DIR:-${PROJECT_DIR}/le-wm}"
STABLEWM_HOME="${STABLEWM_HOME:-${PROJECT_DIR}/stablewm_home}"
PYTHON="${PYTHON:-${TORCH_ENV}/bin/python}"

[ -d "${PROJECT_DIR}" ] || { echo "Missing PROJECT_DIR=${PROJECT_DIR}"; exit 1; }
[ -d "${LEWM_DIR}" ] || { echo "Missing LEWM_DIR=${LEWM_DIR}"; exit 1; }
[ -x "${PYTHON}" ] || { echo "Missing executable PYTHON=${PYTHON}"; exit 1; }
[ -f "${STABLEWM_HOME}/datasets/dmc/reacher_random.h5" ] || { echo "Missing Reacher dataset"; exit 1; }

export PROJECT_DIR ROOT_DIR="${PROJECT_DIR}"
export VIRTUAL_ENV="${TORCH_ENV}"
export PATH="${TORCH_ENV}/bin:${PATH}"
export PYTHON PYTHONPATH="${LEWM_DIR}:${PYTHONPATH:-}"
export LEWM_DIR STABLEWM_HOME LOCAL_DATASET_DIR="${STABLEWM_HOME}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTHONIOENCODING=utf-8
export SPT_CACHE_DIR="${PROJECT_DIR}/.cache/stable-pretraining"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "${PROJECT_DIR}"
mkdir -p logs "${STABLEWM_HOME}/checkpoints/reacher_workshop"

"${PYTHON}" scripts/patch_lewm_compat.py --lewm-dir "${LEWM_DIR}"

"${PYTHON}" - <<'PY'
import hdf5plugin  # noqa: F401
import stable_worldmodel as swm
import torch
from stable_worldmodel.data.format import FORMATS
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())
print("StableWM formats:", sorted(FORMATS.keys()))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for Discoverer LeWM training.")
if "hdf5" not in FORMATS:
    raise SystemExit("Missing StableWM hdf5 format; install hdf5plugin.")
PY

COMMON_RUN_GROUP="${RUN_GROUP:-reacher_workshop}"
COMMON_SEED="${SEED:-3072}"
COMMON_IMG_SIZE="${IMG_SIZE:-224}"
COMMON_WEIGHT_DECAY="${WEIGHT_DECAY:-1e-3}"
COMMON_SIGREG_WEIGHT="${SIGREG_WEIGHT:-0.09}"
COMMON_HISTORY_SIZE="${HISTORY_SIZE:-3}"
COMMON_NUM_PREDS="${NUM_PREDS:-1}"
COMMON_PRECISION="${PRECISION:-bf16}"
COMMON_DATASET_NAME="${DATASET_NAME:-dmc/reacher_random.h5}"

STAGE1_OUTPUT="${STAGE1_OUTPUT:-lewm_reacher_b1_dynmetric_sched_stage1_b1}"
STAGE2_OUTPUT="${STAGE2_OUTPUT:-lewm_reacher_b1_dynmetric_sched_stage2_w1e4}"
FINAL_OUTPUT="${FINAL_OUTPUT:-lewm_reacher_b1_dynmetric_sched_w1e3}"
FINAL_TAG="${FINAL_TAG:-b1_dynmetric_sched_w1e3}"

echo "===================================================="
echo "Scheduled dynamics-metric experiment"
echo "  stage1_output=${STAGE1_OUTPUT}"
echo "  stage2_output=${STAGE2_OUTPUT}"
echo "  final_output=${FINAL_OUTPUT}"
echo "  final_tag=${FINAL_TAG}"
echo "===================================================="

# Stage 1: learn a normal B1 model from scratch so the latent dynamics target is not random.
ABLATION_NAME="${STAGE1_ABLATION_NAME:-b1_dynmetric_sched_stage1_b1}" \
MODEL_CONFIG="lewm_b1_effect" \
OUTPUT_MODEL_NAME="${STAGE1_OUTPUT}" \
RUN_ID="${COMMON_RUN_GROUP}/${STAGE1_ABLATION_NAME:-b1_dynmetric_sched_stage1_b1}_seed${COMMON_SEED}" \
RUN_GROUP="${COMMON_RUN_GROUP}" \
SEED="${COMMON_SEED}" \
EPOCHS="${STAGE1_EPOCHS:-1}" \
MAX_TRAIN_BATCHES="${STAGE1_MAX_TRAIN_BATCHES:-2000}" \
MAX_VAL_BATCHES="${STAGE1_MAX_VAL_BATCHES:-128}" \
BATCH_SIZE="${STAGE1_BATCH_SIZE:-64}" \
NUM_WORKERS="${NUM_WORKERS:-8}" \
IMG_SIZE="${COMMON_IMG_SIZE}" \
LR="${STAGE1_LR:-5e-5}" \
WEIGHT_DECAY="${COMMON_WEIGHT_DECAY}" \
SIGREG_WEIGHT="${COMMON_SIGREG_WEIGHT}" \
HISTORY_SIZE="${COMMON_HISTORY_SIZE}" \
NUM_PREDS="${COMMON_NUM_PREDS}" \
PRECISION="${COMMON_PRECISION}" \
DATASET_NAME="${COMMON_DATASET_NAME}" \
RUN_SMOKE="${RUN_SMOKE:-0}" \
PYTHON="${PYTHON}" \
ROOT_DIR="${PROJECT_DIR}" \
LEWM_DIR="${LEWM_DIR}" \
STABLEWM_HOME="${STABLEWM_HOME}" \
LOCAL_DATASET_DIR="${LOCAL_DATASET_DIR}" \
bash scripts/train_reacher_ablation.sh

# Stage 2: introduce a small dynamics-metric loss.
ABLATION_NAME="${STAGE2_ABLATION_NAME:-b1_dynmetric_sched_stage2_w1e4}" \
MODEL_CONFIG="lewm_b1_effect" \
OUTPUT_MODEL_NAME="${STAGE2_OUTPUT}" \
RUN_ID="${COMMON_RUN_GROUP}/${STAGE2_ABLATION_NAME:-b1_dynmetric_sched_stage2_w1e4}_seed${COMMON_SEED}" \
RUN_GROUP="${COMMON_RUN_GROUP}" \
SEED="${COMMON_SEED}" \
EPOCHS="${STAGE2_EPOCHS:-1}" \
MAX_TRAIN_BATCHES="${STAGE2_MAX_TRAIN_BATCHES:-1000}" \
MAX_VAL_BATCHES="${STAGE2_MAX_VAL_BATCHES:-64}" \
BATCH_SIZE="${STAGE2_BATCH_SIZE:-16}" \
NUM_WORKERS="${NUM_WORKERS:-8}" \
IMG_SIZE="${COMMON_IMG_SIZE}" \
LR="${STAGE2_LR:-2e-5}" \
WEIGHT_DECAY="${COMMON_WEIGHT_DECAY}" \
SIGREG_WEIGHT="${COMMON_SIGREG_WEIGHT}" \
GEO_RESOLUTION="dynmetric" \
GEO_WEIGHT="${STAGE2_GEO_WEIGHT:-1e-4}" \
GEO_K="${GEO_K:-4}" \
GEO_MAX_POINTS="${GEO_MAX_POINTS:-8}" \
GEO_TARGET="${GEO_TARGET:-effect}" \
GEO_ACTION_BASIS="${GEO_ACTION_BASIS:-full}" \
GEO_FRAMESKIP="${GEO_FRAMESKIP:-5}" \
INIT_MODEL_PATH="${STAGE1_OUTPUT}/weights_epoch_${STAGE1_EPOCHS:-1}.pt" \
HISTORY_SIZE="${COMMON_HISTORY_SIZE}" \
NUM_PREDS="${COMMON_NUM_PREDS}" \
PRECISION="${COMMON_PRECISION}" \
DATASET_NAME="${COMMON_DATASET_NAME}" \
RUN_SMOKE="0" \
PYTHON="${PYTHON}" \
ROOT_DIR="${PROJECT_DIR}" \
LEWM_DIR="${LEWM_DIR}" \
STABLEWM_HOME="${STABLEWM_HOME}" \
LOCAL_DATASET_DIR="${LOCAL_DATASET_DIR}" \
bash scripts/train_reacher_ablation.sh

# Stage 3: raise the dynamics-metric loss to the target value.
ABLATION_NAME="${FINAL_TAG}" \
MODEL_CONFIG="lewm_b1_effect" \
OUTPUT_MODEL_NAME="${FINAL_OUTPUT}" \
RUN_ID="${COMMON_RUN_GROUP}/${FINAL_TAG}_seed${COMMON_SEED}" \
RUN_GROUP="${COMMON_RUN_GROUP}" \
SEED="${COMMON_SEED}" \
EPOCHS="${STAGE3_EPOCHS:-1}" \
MAX_TRAIN_BATCHES="${STAGE3_MAX_TRAIN_BATCHES:-1000}" \
MAX_VAL_BATCHES="${STAGE3_MAX_VAL_BATCHES:-64}" \
BATCH_SIZE="${STAGE3_BATCH_SIZE:-16}" \
NUM_WORKERS="${NUM_WORKERS:-8}" \
IMG_SIZE="${COMMON_IMG_SIZE}" \
LR="${STAGE3_LR:-2e-5}" \
WEIGHT_DECAY="${COMMON_WEIGHT_DECAY}" \
SIGREG_WEIGHT="${COMMON_SIGREG_WEIGHT}" \
GEO_RESOLUTION="dynmetric" \
GEO_WEIGHT="${STAGE3_GEO_WEIGHT:-1e-3}" \
GEO_K="${GEO_K:-4}" \
GEO_MAX_POINTS="${GEO_MAX_POINTS:-8}" \
GEO_TARGET="${GEO_TARGET:-effect}" \
GEO_ACTION_BASIS="${GEO_ACTION_BASIS:-full}" \
GEO_FRAMESKIP="${GEO_FRAMESKIP:-5}" \
INIT_MODEL_PATH="${STAGE2_OUTPUT}/weights_epoch_${STAGE2_EPOCHS:-1}.pt" \
HISTORY_SIZE="${COMMON_HISTORY_SIZE}" \
NUM_PREDS="${COMMON_NUM_PREDS}" \
PRECISION="${COMMON_PRECISION}" \
DATASET_NAME="${COMMON_DATASET_NAME}" \
RUN_SMOKE="0" \
PYTHON="${PYTHON}" \
ROOT_DIR="${PROJECT_DIR}" \
LEWM_DIR="${LEWM_DIR}" \
STABLEWM_HOME="${STABLEWM_HOME}" \
LOCAL_DATASET_DIR="${LOCAL_DATASET_DIR}" \
bash scripts/train_reacher_ablation.sh
