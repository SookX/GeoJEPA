#!/usr/bin/env bash
# Install the LeWM OGBench cube setup under K:\LeWMOfficial.
#
# Default behavior:
#   - clone/update ogbench and le-wm source repos under K:\LeWMOfficial
#   - create/use a Python venv under K:\LeWMOfficial\.venv
#   - install OGBench + StableWorldModel dependencies
#   - download the pretrained cube LeWM checkpoint
#   - smoke-test checkpoint loading and the StableWorldModel cube wrapper
#
# The cube dataset is large (~46GB compressed), so it is opt-in:
#   DOWNLOAD_DATASET=1 bash scripts/install_cube_env_k_lewmofficial.sh
#
# Useful overrides:
#   ROOT_DIR=/k/LeWMOfficial DOWNLOAD_DATASET=1 bash scripts/install_cube_env_k_lewmofficial.sh
#   INSTALL_DEPS=0 bash scripts/install_cube_env_k_lewmofficial.sh

set -euo pipefail

default_root() {
    # Git Bash accepts drive-qualified paths directly. This is more reliable on
    # this machine than cygpath, which may emit "K:LeWMOfficial" for backslashes.
    printf 'K:/LeWMOfficial'
}

ROOT_DIR="${ROOT_DIR:-$(default_root)}"
STABLEWM_HOME="${STABLEWM_HOME:-${ROOT_DIR}/stablewm_home}"
OGBENCH_DIR="${OGBENCH_DIR:-${ROOT_DIR}/ogbench}"
LEWM_DIR="${LEWM_DIR:-${ROOT_DIR}/le-wm}"
VENV_DIR="${VENV_DIR:-${ROOT_DIR}/.venv}"
PYTHON="${PYTHON:-${VENV_DIR}/Scripts/python.exe}"
HF_HOME="${HF_HOME:-${ROOT_DIR}/.cache/huggingface}"
DOWNLOAD_DATASET="${DOWNLOAD_DATASET:-0}"
INSTALL_DEPS="${INSTALL_DEPS:-1}"

OGBENCH_REPO_URL="${OGBENCH_REPO_URL:-https://github.com/seohongpark/ogbench.git}"
LEWM_REPO_URL="${LEWM_REPO_URL:-https://github.com/lucas-maes/le-wm.git}"
CUBE_MODEL_REPO="${CUBE_MODEL_REPO:-quentinll/lewm-cube}"
CUBE_DATASET_REPO="${CUBE_DATASET_REPO:-quentinll/lewm-cube}"

export STABLEWM_HOME
export LOCAL_DATASET_DIR="${LOCAL_DATASET_DIR:-${STABLEWM_HOME}}"
export HF_HOME
export PYTHONUNBUFFERED=1
export PYTHONIOENCODING=utf-8
export TOKENIZERS_PARALLELISM=false

mkdir -p "${ROOT_DIR}" "${STABLEWM_HOME}/checkpoints/quentinll" "${STABLEWM_HOME}/datasets/ogbench" "${HF_HOME}"

clone_or_update() {
    local url="$1"
    local dest="$2"
    if [ -d "${dest}/.git" ]; then
        echo "Repo exists: ${dest}"
        git -C "${dest}" pull --ff-only || echo "Warning: could not fast-forward ${dest}; leaving existing checkout."
    else
        echo "Cloning ${url} -> ${dest}"
        git clone "${url}" "${dest}"
    fi
}

ensure_python() {
    if [ -x "${PYTHON}" ]; then
        return 0
    fi

    local base_python=""
    if command -v python >/dev/null 2>&1; then
        base_python="$(command -v python)"
    elif command -v python3 >/dev/null 2>&1; then
        base_python="$(command -v python3)"
    elif command -v py >/dev/null 2>&1; then
        base_python="$(command -v py)"
    else
        echo "No Python executable found on PATH."
        exit 1
    fi

    echo "Creating venv: ${VENV_DIR}"
    "${base_python}" -m venv "${VENV_DIR}"
}

install_deps() {
    if [ "${INSTALL_DEPS}" != "1" ] && [ "${INSTALL_DEPS}" != "true" ]; then
        echo "Skipping dependency install because INSTALL_DEPS=${INSTALL_DEPS}"
        return 0
    fi

    "${PYTHON}" -m pip install --upgrade pip
    # Avoid the broad stable-worldmodel[env] extra on Windows: it pulls box2d-py,
    # which requires SWIG and is irrelevant for the OGBench cube wrapper.
    "${PYTHON}" -m pip install 'stable-worldmodel[train]' 'transformers==4.57.6' huggingface_hub zstandard hdf5plugin ogbench pygame pymunk shapely
    "${PYTHON}" -m pip install "${OGBENCH_DIR}"
}

download_cube_checkpoint() {
    local target="${STABLEWM_HOME}/checkpoints/quentinll/lewm-cube"
    if [ -f "${target}/weights.pt" ] && [ -f "${target}/config.json" ]; then
        echo "Cube checkpoint exists: ${target}"
        return 0
    fi

    echo "Downloading cube checkpoint ${CUBE_MODEL_REPO} -> ${target}"
    mkdir -p "${target}"
    TARGET_DIR="${target}" CUBE_MODEL_REPO="${CUBE_MODEL_REPO}" "${PYTHON}" - <<'PY'
import os
from pathlib import Path
from huggingface_hub import snapshot_download

target = Path(os.environ["TARGET_DIR"])
snapshot_download(
    repo_id=os.environ["CUBE_MODEL_REPO"],
    repo_type="model",
    allow_patterns=["config.json", "weights.pt", "README.md", ".gitattributes"],
    local_dir=target,
    local_dir_use_symlinks=False,
)
print(f"Installed checkpoint: {target}")
PY
}

download_cube_dataset() {
    if [ "${DOWNLOAD_DATASET}" != "1" ] && [ "${DOWNLOAD_DATASET}" != "true" ]; then
        echo "Skipping cube dataset. Set DOWNLOAD_DATASET=1 to fetch the ~46GB archive."
        return 0
    fi

    local target="${STABLEWM_HOME}/datasets/ogbench/cube_single_expert.h5"
    if [ -f "${target}" ]; then
        echo "Cube dataset exists: ${target}"
        return 0
    fi

    echo "Downloading cube dataset ${CUBE_DATASET_REPO}; this is large (~46GB compressed)."
    TARGET_H5="${target}" CUBE_DATASET_REPO="${CUBE_DATASET_REPO}" "${PYTHON}" - <<'PY'
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from huggingface_hub import snapshot_download

target = Path(os.environ["TARGET_H5"])
cache_root = target.parents[2] / "downloads" / "hf_cube_dataset"
cache_root.mkdir(parents=True, exist_ok=True)

repo_dir = Path(snapshot_download(
    repo_id=os.environ["CUBE_DATASET_REPO"],
    repo_type="dataset",
    allow_patterns=["cube_single_expert.tar.zst", "README.md", ".gitattributes"],
    local_dir=cache_root,
    local_dir_use_symlinks=False,
))

h5s = sorted(repo_dir.rglob("cube_single_expert.h5"))
if not h5s:
    archives = sorted(repo_dir.rglob("cube_single_expert.tar.zst")) + sorted(repo_dir.rglob("cube_single_expert.tar"))
    if not archives:
        raise SystemExit(f"No cube_single_expert archive found under {repo_dir}")
    extract_dir = repo_dir / "_extracted"
    extract_dir.mkdir(exist_ok=True)
    for archive in archives:
        print(f"Extracting {archive}")
        if archive.name.endswith(".tar.zst"):
            subprocess.run(["tar", "--zstd", "-xf", str(archive), "-C", str(extract_dir)], check=True)
        else:
            with tarfile.open(archive) as tf:
                tf.extractall(extract_dir)
    h5s = sorted(extract_dir.rglob("cube_single_expert.h5"))

if not h5s:
    raise SystemExit(f"Could not find cube_single_expert.h5 after extraction under {repo_dir}")

source = h5s[0]
target.parent.mkdir(parents=True, exist_ok=True)
if target.exists() and target.stat().st_size != source.stat().st_size:
    print(f"Removing incomplete target: {target}")
    target.unlink()
if source.resolve() != target.resolve():
    try:
        source.replace(target)
    except OSError:
        shutil.move(str(source), str(target))
print(f"Installed dataset: {target}")
PY
}

smoke_test() {
    "${PYTHON}" - <<'PY'
import gymnasium as gym
import stable_worldmodel as swm
import stable_worldmodel.envs  # noqa: F401

model = swm.wm.utils.load_pretrained("quentinll/lewm-cube")
print("checkpoint_ok", type(model).__name__, sum(p.numel() for p in model.parameters()))

env = gym.make(
    "swm/OGBCube-v0",
    env_type="single",
    ob_type="states",
    multiview=False,
    width=224,
    height=224,
    terminate_at_goal=True,
)
obs, _info = env.reset(seed=0)
print("env_ok", type(env.unwrapped).__name__, env.action_space.shape, getattr(obs, "shape", None))
env.close()
PY
}

clone_or_update "${OGBENCH_REPO_URL}" "${OGBENCH_DIR}"
clone_or_update "${LEWM_REPO_URL}" "${LEWM_DIR}"
ensure_python
install_deps
download_cube_checkpoint
download_cube_dataset
smoke_test

cat <<EOF
====================================================
Cube setup ready
  ROOT_DIR=${ROOT_DIR}
  OGBENCH_DIR=${OGBENCH_DIR}
  LEWM_DIR=${LEWM_DIR}
  STABLEWM_HOME=${STABLEWM_HOME}
  checkpoint=${STABLEWM_HOME}/checkpoints/quentinll/lewm-cube
  dataset=${STABLEWM_HOME}/datasets/ogbench/cube_single_expert.h5

PowerShell env for experiments:
  \$env:STABLEWM_HOME='K:\\LeWMOfficial\\stablewm_home'
  \$env:LOCAL_DATASET_DIR='K:\\LeWMOfficial\\stablewm_home'
====================================================
EOF
