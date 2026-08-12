#!/bin/bash
# Discoverer/Slurm launcher for one 200-start Reacher evaluation in safe chunks.

#SBATCH --partition=common
#SBATCH --qos=bg-eng-01
#SBATCH --account=bg-eng-01
#SBATCH --job-name=lewm_eval200
#SBATCH --time=03:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgx1
#SBATCH -o logs/lewm_eval200.%j.out
#SBATCH -e logs/lewm_eval200.%j.err

set -euo pipefail

module purge
module load anaconda3
module load nvidia/cuda/12

PROJECT_DIR="${PROJECT_DIR:-/valhalla/projects/bg-eng-01/GeoJEPA}"
TORCH_ENV="${TORCH_ENV:-/valhalla/projects/bg-eng-01/conda_envs/torch}"
LEWM_DIR="${LEWM_DIR:-${PROJECT_DIR}/le-wm}"
STABLEWM_HOME="${STABLEWM_HOME:-${PROJECT_DIR}/stablewm_home}"
PYTHON="${PYTHON:-${TORCH_ENV}/bin/python}"

POLICY="${POLICY:-${POLICY_PATH:-}}"
[ -n "${POLICY}" ] || { echo "Set POLICY, e.g. lewm_reacher_baseline/weights_epoch_3.pt"; exit 1; }
POLICY_TAG="${POLICY_TAG:?Set POLICY_TAG, e.g. b0_baseline}"
DATASET_NAME="${DATASET_NAME:-dmc/reacher_random}"
SAMPLE_COUNT="${SAMPLE_COUNT:-200}"
CHUNK_SIZE="${CHUNK_SIZE:-10}"
START_OFFSET="${START_OFFSET:-0}"
GOAL_OFFSET_STEPS="${GOAL_OFFSET_STEPS:-25}"
EVAL_BUDGET="${EVAL_BUDGET:-50}"
SEED="${SEED:-42}"
SOLVER_SAMPLES="${SOLVER_SAMPLES:-128}"
SOLVER_STEPS="${SOLVER_STEPS:-12}"
SOLVER_TOPK="${SOLVER_TOPK:-16}"
PLANNING_VALUE_WEIGHT="${PLANNING_VALUE_WEIGHT:-}"
PLANNING_VALUE_GAMMA="${PLANNING_VALUE_GAMMA:-}"
LOG_COST_STATS="${LOG_COST_STATS:-0}"
COST_LOG_LIMIT="${COST_LOG_LIMIT:-8}"
PATH_KEYS="${PATH_KEYS:-qpos,qvel,goal_qpos}"
FORCE="${FORCE:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${STABLEWM_HOME}/checkpoints/reacher_workshop/eval200/${POLICY_TAG}}"
SUMMARY_JSON="${SUMMARY_JSON:-${OUTPUT_DIR}/${POLICY_TAG}_eval200_summary.json}"

[ -d "${PROJECT_DIR}" ] || { echo "Missing PROJECT_DIR=${PROJECT_DIR}"; exit 1; }
[ -d "${LEWM_DIR}" ] || { echo "Missing LEWM_DIR=${LEWM_DIR}"; exit 1; }
[ -x "${PYTHON}" ] || { echo "Missing executable PYTHON=${PYTHON}"; exit 1; }
[ -f "${STABLEWM_HOME}/checkpoints/${POLICY}" ] || { echo "Missing policy checkpoint ${POLICY}"; exit 1; }

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
mkdir -p logs "${OUTPUT_DIR}"

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
    raise SystemExit("CUDA is required for LeWM Reacher evaluation.")
if "hdf5" not in FORMATS:
    raise SystemExit("Missing StableWM hdf5 format; install hdf5plugin.")
PY

echo "===================================================="
echo "LeWM Reacher 200-start eval"
echo "  policy=${POLICY}"
echo "  policy_tag=${POLICY_TAG}"
echo "  output_dir=${OUTPUT_DIR}"
echo "  sample_count=${SAMPLE_COUNT}"
echo "  chunk_size=${CHUNK_SIZE}"
echo "  solver=${SOLVER_SAMPLES}/${SOLVER_STEPS}/topk${SOLVER_TOPK}"
echo "  planning_value_weight=${PLANNING_VALUE_WEIGHT:-0}"
echo "  planning_value_gamma=${PLANNING_VALUE_GAMMA:-default}"
echo "  log_cost_stats=${LOG_COST_STATS}"
echo "===================================================="

end_offset=$((START_OFFSET + SAMPLE_COUNT))
offset="${START_OFFSET}"
while [ "${offset}" -lt "${end_offset}" ]; do
    remaining=$((end_offset - offset))
    num_eval="${CHUNK_SIZE}"
    if [ "${remaining}" -lt "${CHUNK_SIZE}" ]; then
        num_eval="${remaining}"
    fi

    chunk_json="${OUTPUT_DIR}/${POLICY_TAG}_chunk_${offset}.json"
    paths_json="${OUTPUT_DIR}/${POLICY_TAG}_paths_${offset}.json"
    if [ "${FORCE}" != "1" ] && [ -s "${chunk_json}" ]; then
        echo "Skipping existing chunk offset=${offset}: ${chunk_json}"
    else
        echo "Running chunk offset=${offset}, num_eval=${num_eval}"
        POLICY="${POLICY}" \
        DATASET_NAME="${DATASET_NAME}" \
        NUM_EVAL="${num_eval}" \
        SAMPLE_COUNT="${SAMPLE_COUNT}" \
        SAMPLE_OFFSET="${offset}" \
        GOAL_OFFSET_STEPS="${GOAL_OFFSET_STEPS}" \
        EVAL_BUDGET="${EVAL_BUDGET}" \
        SEED="${SEED}" \
        SOLVER_SAMPLES="${SOLVER_SAMPLES}" \
        SOLVER_STEPS="${SOLVER_STEPS}" \
        SOLVER_TOPK="${SOLVER_TOPK}" \
        PLANNING_VALUE_WEIGHT="${PLANNING_VALUE_WEIGHT}" \
        PLANNING_VALUE_GAMMA="${PLANNING_VALUE_GAMMA}" \
        LOG_COST_STATS="${LOG_COST_STATS}" \
        COST_LOG_LIMIT="${COST_LOG_LIMIT}" \
        OUTPUT_JSON="${chunk_json}" \
        PATHS_JSON="${paths_json}" \
        PATH_KEYS="${PATH_KEYS}" \
        PYTHON="${PYTHON}" \
        ROOT_DIR="${PROJECT_DIR}" \
        LEWM_DIR="${LEWM_DIR}" \
        STABLEWM_HOME="${STABLEWM_HOME}" \
        LOCAL_DATASET_DIR="${LOCAL_DATASET_DIR}" \
        bash scripts/evaluate_reacher_policy.sh
    fi
    offset=$((offset + CHUNK_SIZE))
done

POLICY_TAG="${POLICY_TAG}" \
POLICY="${POLICY}" \
OUTPUT_DIR="${OUTPUT_DIR}" \
SUMMARY_JSON="${SUMMARY_JSON}" \
"${PYTHON}" - <<'PY'
import glob
import json
import os
from pathlib import Path
from statistics import median

policy_tag = os.environ["POLICY_TAG"]
policy = os.environ["POLICY"]
output_dir = Path(os.environ["OUTPUT_DIR"])
summary_json = Path(os.environ["SUMMARY_JSON"])

chunks = []
for path in sorted(glob.glob(str(output_dir / f"{policy_tag}_chunk_*.json"))):
    with open(path, encoding="utf-8") as handle:
        chunks.append(json.load(handle))

if not chunks:
    raise SystemExit(f"No chunk JSON files found in {output_dir}")

chunks.sort(key=lambda item: item["sample_offset"])
success_count = sum(item["metrics"]["success_count"] for item in chunks)
num_eval = sum(item["metrics"]["num_eval"] for item in chunks)
first_success_step = [
    step
    for item in chunks
    for step in item["metrics"]["first_success_step"]
]
successful_steps = [step for step in first_success_step if step > 0]
episode_successes = [
    ok
    for item in chunks
    for ok in item["metrics"]["episode_successes"]
]

summary = {
    "ok": True,
    "policy": policy,
    "policy_tag": policy_tag,
    "chunk_count": len(chunks),
    "chunk_offsets": [item["sample_offset"] for item in chunks],
    "sample_count": chunks[0]["sample_count"],
    "seed": chunks[0]["seed"],
    "goal_offset_steps": chunks[0]["goal_offset_steps"],
    "eval_budget": chunks[0]["eval_budget"],
    "solver": chunks[0]["solver"],
    "metrics": {
        "success_count": success_count,
        "num_eval": num_eval,
        "success_rate": 100.0 * success_count / num_eval if num_eval else 0.0,
        "episode_successes": episode_successes,
        "first_success_step": first_success_step,
        "successful_steps_mean": sum(successful_steps) / len(successful_steps) if successful_steps else None,
        "successful_steps_median": median(successful_steps) if successful_steps else None,
        "successful_steps_min": min(successful_steps) if successful_steps else None,
        "successful_steps_max": max(successful_steps) if successful_steps else None,
    },
    "chunks": [
        {
            "path": str(output_dir / f"{policy_tag}_chunk_{item['sample_offset']}.json"),
            "paths_path": str(output_dir / f"{policy_tag}_paths_{item['sample_offset']}.json"),
            "sample_offset": item["sample_offset"],
            "success_count": item["metrics"]["success_count"],
            "num_eval": item["metrics"]["num_eval"],
            "success_rate": item["metrics"]["success_rate"],
        }
        for item in chunks
    ],
}

summary_json.parent.mkdir(parents=True, exist_ok=True)
summary_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps(summary["metrics"], indent=2, sort_keys=True))
print(f"Wrote {summary_json}")
PY
