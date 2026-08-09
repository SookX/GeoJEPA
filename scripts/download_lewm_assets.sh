#!/bin/bash
# Download/install the LeWM assets needed for GeoJEPA workshop ablations.
#
# Intended cluster layout:
#   /valhalla/projects/bg-eng-1/GeoJEPA
#
# What it prepares:
#   - Python package dependencies, optionally
#   - Reacher dataset under $STABLEWM_HOME/datasets/dmc/reacher_random.h5
#   - Reacher checkpoint under $STABLEWM_HOME/checkpoints/quentinll/lewm-reacher
#
# Examples:
#   bash scripts/download_lewm_assets.sh
#   INSTALL_DEPS=0 bash scripts/download_lewm_assets.sh
#   DATASETS="reacher" CHECKPOINTS="reacher" bash scripts/download_lewm_assets.sh

set -euo pipefail

DEFAULT_PROJECT_DIR="/valhalla/projects/bg-eng-1/GeoJEPA"
if [ -d "${DEFAULT_PROJECT_DIR}" ]; then
    PROJECT_DIR="${PROJECT_DIR:-${DEFAULT_PROJECT_DIR}}"
else
    PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
fi

LEWM_DIR="${LEWM_DIR:-${PROJECT_DIR}/le-wm}"
STABLEWM_HOME="${STABLEWM_HOME:-${PROJECT_DIR}/stablewm_home}"
VIRTUAL_ENV="${VIRTUAL_ENV:-${PROJECT_DIR}/.venv}"
PYTHON="${PYTHON:-${VIRTUAL_ENV}/bin/python}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"
INSTALL_LEWM_CODE="${INSTALL_LEWM_CODE:-1}"
LEWM_REPO_URL="${LEWM_REPO_URL:-https://github.com/lucas-maes/le-wm.git}"
DATASETS="${DATASETS:-reacher}"
CHECKPOINTS="${CHECKPOINTS:-reacher}"
HF_HOME="${HF_HOME:-${PROJECT_DIR}/.cache/huggingface}"
TMP_ROOT="${TMP_ROOT:-${PROJECT_DIR}/.cache/downloads}"

export STABLEWM_HOME
export LOCAL_DATASET_DIR="${LOCAL_DATASET_DIR:-${STABLEWM_HOME}}"
export HF_HOME
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false

mkdir -p "${STABLEWM_HOME}/datasets" "${STABLEWM_HOME}/checkpoints" "${TMP_ROOT}" "${PROJECT_DIR}/logs"
cd "${PROJECT_DIR}"

if [ "${INSTALL_LEWM_CODE}" = "1" ] || [ "${INSTALL_LEWM_CODE}" = "true" ]; then
    if [ ! -d "${LEWM_DIR}" ]; then
        command -v git >/dev/null 2>&1 || { echo "git is required to clone ${LEWM_REPO_URL}"; exit 1; }
        echo "Cloning LeWM code: ${LEWM_REPO_URL} -> ${LEWM_DIR}"
        git clone "${LEWM_REPO_URL}" "${LEWM_DIR}"
    else
        echo "LeWM code exists: ${LEWM_DIR}"
    fi
fi

have_python() {
    [ -x "${PYTHON}" ]
}

if ! have_python; then
    if command -v python3 >/dev/null 2>&1; then
        echo "Creating venv at ${VIRTUAL_ENV}"
        python3 -m venv "${VIRTUAL_ENV}"
    elif command -v python >/dev/null 2>&1; then
        echo "Creating venv at ${VIRTUAL_ENV}"
        python -m venv "${VIRTUAL_ENV}"
    else
        echo "No python/python3 found. Load Python or anaconda first."
        exit 1
    fi
fi

if [ "${INSTALL_DEPS}" = "1" ] || [ "${INSTALL_DEPS}" = "true" ]; then
    "${PYTHON}" -m pip install --upgrade pip
    "${PYTHON}" -m pip install 'stable-worldmodel[train,env]' huggingface_hub zstandard
fi

"${PYTHON}" "${PROJECT_DIR}/scripts/patch_lewm_compat.py" --lewm-dir "${LEWM_DIR}"

"${PYTHON}" - <<'PY'
import importlib
missing = []
for name in ("torch", "hydra", "lightning", "stable_worldmodel", "stable_pretraining", "huggingface_hub"):
    try:
        importlib.import_module(name)
    except Exception as exc:
        missing.append(f"{name}: {exc}")
if missing:
    print("Missing dependencies:")
    print("\n".join("  " + item for item in missing))
    raise SystemExit(1)
PY

download_reacher_dataset() {
    local target="${STABLEWM_HOME}/datasets/dmc/reacher_random.h5"
    if [ -f "${target}" ]; then
        echo "Dataset exists: ${target}"
        return 0
    fi

    echo "Downloading dataset quentinll/lewm-reacher -> ${target}"
    mkdir -p "${STABLEWM_HOME}/datasets/dmc"
    DATASET_OUT="${TMP_ROOT}/hf_datasets/lewm-reacher" \
    TARGET_H5="${target}" \
    "${PYTHON}" - <<'PY'
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from huggingface_hub import snapshot_download

out = Path(os.environ["DATASET_OUT"])
target = Path(os.environ["TARGET_H5"])
out.mkdir(parents=True, exist_ok=True)
repo_dir = Path(snapshot_download(
    repo_id="quentinll/lewm-reacher",
    repo_type="dataset",
    local_dir=out,
    local_dir_use_symlinks=False,
))

h5s = sorted(repo_dir.rglob("*.h5"))
if not h5s:
    archives = sorted(repo_dir.rglob("*.tar.zst")) + sorted(repo_dir.rglob("*.tar.gz")) + sorted(repo_dir.rglob("*.tar"))
    if not archives:
        raise SystemExit(f"No .h5 or supported archive found under {repo_dir}")
    extract_dir = repo_dir / "_extracted"
    extract_dir.mkdir(exist_ok=True)
    for archive in archives:
        print(f"Extracting {archive}")
        if archive.suffix == ".zst":
            subprocess.run(["tar", "--zstd", "-xf", str(archive), "-C", str(extract_dir)], check=True)
        else:
            with tarfile.open(archive) as tf:
                tf.extractall(extract_dir)
    h5s = sorted(extract_dir.rglob("*.h5"))

chosen = None
for path in h5s:
    if path.name == "reacher_random.h5":
        chosen = path
        break
chosen = chosen or h5s[0]
target.parent.mkdir(parents=True, exist_ok=True)
if chosen.resolve() != target.resolve():
    shutil.copy2(chosen, target)
print(f"Installed dataset: {target}")
PY
}

download_reacher_checkpoint() {
    local target_dir="${STABLEWM_HOME}/checkpoints/quentinll/lewm-reacher"
    if [ -f "${target_dir}/weights.pt" ] && [ -f "${target_dir}/config.json" ]; then
        echo "Checkpoint exists: ${target_dir}"
        return 0
    fi

    echo "Downloading checkpoint quentinll/lewm-reacher -> ${target_dir}"
    mkdir -p "${target_dir}"
    TARGET_DIR="${target_dir}" \
    "${PYTHON}" - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

target = Path(os.environ["TARGET_DIR"])
snapshot_download(
    repo_id="quentinll/lewm-reacher",
    repo_type="model",
    allow_patterns=["config.json", "weights.pt", "README.md", ".gitattributes"],
    local_dir=target,
    local_dir_use_symlinks=False,
)
print(f"Installed checkpoint: {target}")
PY
}

case " ${DATASETS} " in
    *" reacher "*) download_reacher_dataset ;;
    *" none "*) echo "Skipping datasets" ;;
    *) echo "Unknown DATASETS='${DATASETS}'. Supported: reacher, none"; exit 1 ;;
esac

case " ${CHECKPOINTS} " in
    *" reacher "*) download_reacher_checkpoint ;;
    *" none "*) echo "Skipping checkpoints" ;;
    *) echo "Unknown CHECKPOINTS='${CHECKPOINTS}'. Supported: reacher, none"; exit 1 ;;
esac

cat <<EOF
====================================================
LeWM assets ready
  PROJECT_DIR=${PROJECT_DIR}
  LEWM_DIR=${LEWM_DIR}
  STABLEWM_HOME=${STABLEWM_HOME}
  dataset=${STABLEWM_HOME}/datasets/dmc/reacher_random.h5
  checkpoint=${STABLEWM_HOME}/checkpoints/quentinll/lewm-reacher
====================================================
EOF

