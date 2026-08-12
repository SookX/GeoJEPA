#!/bin/bash
# Discoverer/Slurm launcher for B2: effect representation + value shaping.

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=lewm_b2
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH -o logs/lewm_b2.%j.out
#SBATCH -e logs/lewm_b2.%j.err

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

ABLATION_NAME="${ABLATION_NAME:-b2_effect_value}" \
MODEL_CONFIG="${MODEL_CONFIG:-lewm_b2_effect_value}" \
OUTPUT_MODEL_NAME="${OUTPUT_MODEL_NAME:-lewm_reacher_b2_effect_value}" \
RUN_ID="${RUN_ID:-reacher_workshop/b2_effect_value_seed3072}" \
RUN_GROUP="${RUN_GROUP:-reacher_workshop}" \
SEED="${SEED:-3072}" \
EPOCHS="${EPOCHS:-3}" \
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-2000}" \
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-128}" \
BATCH_SIZE="${BATCH_SIZE:-64}" \
NUM_WORKERS="${NUM_WORKERS:-8}" \
IMG_SIZE="${IMG_SIZE:-224}" \
LR="${LR:-5e-5}" \
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-3}" \
SIGREG_WEIGHT="${SIGREG_WEIGHT:-0.09}" \
VALUE_WEIGHT="${VALUE_WEIGHT:-0.02}" \
VALUE_GAMMA="${VALUE_GAMMA:-0.99}" \
DATASET_KEYS_TO_LOAD="${DATASET_KEYS_TO_LOAD:-pixels,action,observation,reward}" \
DATASET_KEYS_TO_CACHE="${DATASET_KEYS_TO_CACHE:-action,observation,reward}" \
HISTORY_SIZE="${HISTORY_SIZE:-3}" \
NUM_PREDS="${NUM_PREDS:-1}" \
PRECISION="${PRECISION:-bf16}" \
DATASET_NAME="${DATASET_NAME:-dmc/reacher_random.h5}" \
RUN_SMOKE="${RUN_SMOKE:-1}" \
PYTHON="${PYTHON}" \
ROOT_DIR="${PROJECT_DIR}" \
LEWM_DIR="${LEWM_DIR}" \
STABLEWM_HOME="${STABLEWM_HOME}" \
LOCAL_DATASET_DIR="${LOCAL_DATASET_DIR}" \
bash scripts/train_reacher_ablation.sh
