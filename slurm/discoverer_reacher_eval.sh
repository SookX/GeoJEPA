#!/bin/bash
# Discoverer/Slurm launcher for LeWM Reacher policy evaluation.

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=lewm_eval
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx1
#SBATCH -o logs/lewm_eval.%j.out
#SBATCH -e logs/lewm_eval.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="${PROJECT_DIR:-/valhalla/projects/bg-eng-01/GeoJEPA}"
TORCH_ENV="${TORCH_ENV:-/valhalla/projects/bg-eng-01/conda_envs/torch}"
LEWM_DIR="${LEWM_DIR:-${PROJECT_DIR}/le-wm}"
STABLEWM_HOME="${STABLEWM_HOME:-${PROJECT_DIR}/stablewm_home}"
PYTHON="${PYTHON:-${TORCH_ENV}/bin/python}"

POLICY="${POLICY:-lewm_reacher_baseline/weights_epoch_3.pt}"
NUM_EVAL="${NUM_EVAL:-10}"
SAMPLE_COUNT="${SAMPLE_COUNT:-200}"
SAMPLE_OFFSET="${SAMPLE_OFFSET:-0}"
GOAL_OFFSET_STEPS="${GOAL_OFFSET_STEPS:-25}"
EVAL_BUDGET="${EVAL_BUDGET:-50}"
SEED="${SEED:-42}"
SOLVER_SAMPLES="${SOLVER_SAMPLES:-128}"
SOLVER_STEPS="${SOLVER_STEPS:-12}"
SOLVER_TOPK="${SOLVER_TOPK:-16}"
OUTPUT_JSON="${OUTPUT_JSON:-${STABLEWM_HOME}/checkpoints/reacher_workshop/eval_${SEED}_${SAMPLE_OFFSET}.json}"
PATHS_JSON="${PATHS_JSON:-${STABLEWM_HOME}/checkpoints/reacher_workshop/eval_${SEED}_paths_${SAMPLE_OFFSET}.json}"
PATH_KEYS="${PATH_KEYS:-qpos,qvel,goal_qpos}"

[ -d "${PROJECT_DIR}" ] || { echo "Missing PROJECT_DIR=${PROJECT_DIR}"; exit 1; }
[ -d "${LEWM_DIR}" ] || { echo "Missing LEWM_DIR=${LEWM_DIR}"; exit 1; }
[ -x "${PYTHON}" ] || { echo "Missing executable PYTHON=${PYTHON}"; exit 1; }

export VIRTUAL_ENV="${TORCH_ENV}"
export PATH="${TORCH_ENV}/bin:${PATH}"
export PYTHON
export PROJECT_DIR ROOT_DIR="${PROJECT_DIR}"
export LEWM_DIR STABLEWM_HOME LOCAL_DATASET_DIR="${STABLEWM_HOME}"
export PYTHONPATH="${LEWM_DIR}:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export PYTHONIOENCODING=utf-8
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "${PROJECT_DIR}"
mkdir -p logs "$(dirname "${OUTPUT_JSON}")" "$(dirname "${PATHS_JSON}")"

"${PYTHON}" - <<'PY'
import hdf5plugin  # noqa: F401
import stable_worldmodel as swm
import torch
from stable_worldmodel.data.format import FORMATS
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available(), "| devices:", torch.cuda.device_count())
print("StableWM formats:", sorted(FORMATS.keys()))
if not torch.cuda.is_available():
    raise SystemExit("CUDA is required for LeWM Reacher evaluation.")
if "hdf5" not in FORMATS:
    raise SystemExit("Missing StableWM hdf5 format; install hdf5plugin.")
PY

POLICY="${POLICY}" \
NUM_EVAL="${NUM_EVAL}" \
SAMPLE_COUNT="${SAMPLE_COUNT}" \
SAMPLE_OFFSET="${SAMPLE_OFFSET}" \
GOAL_OFFSET_STEPS="${GOAL_OFFSET_STEPS}" \
EVAL_BUDGET="${EVAL_BUDGET}" \
SEED="${SEED}" \
SOLVER_SAMPLES="${SOLVER_SAMPLES}" \
SOLVER_STEPS="${SOLVER_STEPS}" \
SOLVER_TOPK="${SOLVER_TOPK}" \
OUTPUT_JSON="${OUTPUT_JSON}" \
PATHS_JSON="${PATHS_JSON}" \
PATH_KEYS="${PATH_KEYS}" \
PYTHON="${PYTHON}" \
ROOT_DIR="${PROJECT_DIR}" \
LEWM_DIR="${LEWM_DIR}" \
STABLEWM_HOME="${STABLEWM_HOME}" \
LOCAL_DATASET_DIR="${LOCAL_DATASET_DIR}" \
bash scripts/evaluate_reacher_policy.sh
