#!/usr/bin/env python
"""Compare OGBench Cube environment and pretrained LeWM local action geometry.

This is a diagnostic-only script:
  * no training,
  * no checkpoint modification,
  * environment Jacobians are estimated from simulator counterfactuals,
  * model Jacobians are computed with respect to the same 5D action-block offset.

The effective transition matches the Cube LeWM data layout:
history_size=3, frameskip=5, num_preds=1.
For a sampled dataset window, the final context state is at raw row
start + (history_size - 1) * frameskip.  The model consumes the actual
5 raw actions in that final block and predicts the embedding 5 simulator
steps later.  The environment finite difference restores qpos/qvel at
that same raw row, perturbs the full 5-action block along one raw action
coordinate, and encodes the resulting image with the frozen LeWM encoder.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_ROOT = Path("K:/LeWMOfficial")
DEFAULT_CACHE = DEFAULT_ROOT / "stablewm_home"
DEFAULT_LEWM_DIR = DEFAULT_ROOT / "le-wm"
DEFAULT_DATASET = DEFAULT_CACHE / "datasets" / "ogbench" / "cube_single_expert.h5"
DEFAULT_OUTDIR = DEFAULT_CACHE / "checkpoints" / "cube_geometry_diagnostic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lewm-dir", type=Path, default=DEFAULT_LEWM_DIR)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--dataset-path", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--dataset-name", default="ogbench/cube_single_expert")
    parser.add_argument("--policy", default="quentinll/lewm-cube")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTDIR)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--eps-list", default="1e-3,3e-3,1e-2")
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--num-preds", type=int, default=1)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--env-id", default="swm/OGBCube-v0")
    parser.add_argument("--env-type", default="single")
    parser.add_argument("--env-mode", default="data_collection")
    parser.add_argument("--reconstruction-checks", type=int, default=20)
    parser.add_argument("--reconstruction-max-median-mae", type=float, default=8.0)
    parser.add_argument("--reconstruction-max-p95-mae", type=float, default=20.0)
    parser.add_argument("--bootstrap-resamples", type=int, default=5000)
    parser.add_argument("--rank-rel-floor", type=float, default=1e-6)
    parser.add_argument("--eig-abs-floor", type=float, default=1e-12)
    parser.add_argument("--save-jacobians", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


@dataclass(frozen=True)
class SampleMeta:
    sample_id: int
    dataset_idx: int
    episode: int
    local_start: int
    raw_row: int
    timestep: int


def as_float(value: Any) -> float:
    return float(np.asarray(value).item())


def q(values: np.ndarray, pct: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), pct))


def finite_array(values: Iterable[float]) -> np.ndarray:
    arr = np.asarray(list(values), dtype=np.float64)
    return arr[np.isfinite(arr)]


def summarize(values: Iterable[float]) -> dict[str, float | None]:
    arr = finite_array(values)
    if arr.size == 0:
        return {
            "mean": None,
            "std": None,
            "median": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "p95": None,
        }
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=0)),
        "median": float(np.median(arr)),
        "p10": q(arr, 10),
        "p25": q(arr, 25),
        "p75": q(arr, 75),
        "p90": q(arr, 90),
        "p95": q(arr, 95),
    }


def bootstrap_ci(
    values: np.ndarray,
    fn,
    resamples: int,
    rng: np.random.Generator,
) -> list[float] | None:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return None
    if resamples <= 0:
        return None
    n = values.size
    stats = np.empty(resamples, dtype=np.float64)
    for i in range(resamples):
        idx = rng.integers(0, n, size=n)
        stats[i] = fn(values[idx])
    return [q(stats, 2.5), q(stats, 97.5)]


def rankdata_simple(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty_like(values, dtype=np.float64)
    ranks[order] = np.arange(values.size, dtype=np.float64)
    return ranks


def corrcoef(x: np.ndarray, y: np.ndarray) -> float | None:
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3:
        return None
    x = x[mask]
    y = y[mask]
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def fraction(values: np.ndarray, predicate) -> float | None:
    values = np.asarray(values)
    mask = np.isfinite(values)
    if mask.sum() == 0:
        return None
    return float(np.mean(predicate(values[mask])))


def disable_fast_attention(torch_module) -> None:
    if hasattr(torch_module.backends, "mha"):
        torch_module.backends.mha.set_fastpath_enabled(False)
    if hasattr(torch_module.backends, "cuda"):
        torch_module.backends.cuda.enable_flash_sdp(False)
        torch_module.backends.cuda.enable_mem_efficient_sdp(False)
        torch_module.backends.cuda.enable_math_sdp(True)


def parse_eps_list(text: str) -> list[float]:
    eps_list = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not eps_list:
        raise ValueError("--eps-list must contain at least one finite-difference scale")
    if any(eps <= 0.0 for eps in eps_list):
        raise ValueError("--eps-list entries must be positive")
    return eps_list


def ensure_paths(args: argparse.Namespace) -> None:
    missing = []
    for label, path in [
        ("LeWM repo", args.lewm_dir),
        ("StableWM cache", args.cache_dir),
        ("Cube dataset", args.dataset_path),
    ]:
        if not Path(path).exists():
            missing.append(f"{label}: {path}")
    checkpoint_dir = Path(args.cache_dir) / "checkpoints" / args.policy
    if not checkpoint_dir.exists():
        missing.append(f"checkpoint dir: {checkpoint_dir}")
    if missing:
        raise FileNotFoundError("Missing required Cube assets:\n" + "\n".join(missing))


def json_safe(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        return value if math.isfinite(value) else None
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    return obj


def import_runtime(args: argparse.Namespace):
    lewm_dir = str(Path(args.lewm_dir).resolve())
    if lewm_dir not in sys.path:
        sys.path.insert(0, lewm_dir)
    os.environ.setdefault("MUJOCO_GL", "glfw")

    import h5py
    import hdf5plugin  # noqa: F401
    import gymnasium as gym
    import matplotlib
    import torch
    from stable_pretraining import data as dt
    import stable_worldmodel as swm
    from utils import get_column_normalizer, get_img_preprocessor

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return {
        "h5py": h5py,
        "gym": gym,
        "torch": torch,
        "swm": swm,
        "Compose": dt.transforms.Compose,
        "get_column_normalizer": get_column_normalizer,
        "get_img_preprocessor": get_img_preprocessor,
        "plt": plt,
    }


def build_env(gym, args: argparse.Namespace):
    env = gym.make(
        args.env_id,
        env_type=args.env_type,
        ob_type="pixels",
        multiview=False,
        width=args.img_size,
        height=args.img_size,
        mode=args.env_mode,
        terminate_at_goal=False,
        visualize_info=False,
    )
    env.reset(seed=args.seed)
    return env


def get_unwrapped(env):
    return env.unwrapped if hasattr(env, "unwrapped") else env


def maybe_set_target(env, h5, raw_row: int) -> None:
    unwrapped = get_unwrapped(env)
    if not hasattr(unwrapped, "set_target_pos"):
        return
    if "privileged_target_block_pos" not in h5:
        return
    pos = np.asarray(h5["privileged_target_block_pos"][raw_row], dtype=np.float64)
    quat = None
    if "privileged_block_0_quat" in h5:
        quat = np.asarray(h5["privileged_block_0_quat"][raw_row], dtype=np.float64)
    try:
        if quat is None:
            unwrapped.set_target_pos(pos)
        else:
            unwrapped.set_target_pos(pos, quat)
    except TypeError:
        try:
            unwrapped.set_target_pos(pos)
        except Exception:
            return
    except Exception:
        return


def restore_state(env, h5, raw_row: int) -> None:
    unwrapped = get_unwrapped(env)
    if not hasattr(unwrapped, "set_state"):
        raise RuntimeError("Cube environment does not expose unwrapped.set_state(qpos, qvel)")
    maybe_set_target(env, h5, raw_row)
    qpos = np.asarray(h5["qpos"][raw_row], dtype=np.float64)
    qvel = np.asarray(h5["qvel"][raw_row], dtype=np.float64)
    unwrapped.set_state(qpos, qvel)


def extract_pixels(obs: Any) -> np.ndarray:
    if isinstance(obs, dict):
        for key in ("pixels", "observation", "image", "rgb"):
            if key in obs:
                return np.asarray(obs[key])
        raise KeyError(f"Could not find pixel key in observation dict keys={list(obs)}")
    return np.asarray(obs)


def current_pixels(env) -> np.ndarray:
    unwrapped = get_unwrapped(env)
    if hasattr(unwrapped, "compute_observation"):
        return extract_pixels(unwrapped.compute_observation())
    obs = env.render()
    return extract_pixels(obs)


def execute_action_block(env, action_block: np.ndarray) -> np.ndarray:
    obs = None
    for action in np.asarray(action_block, dtype=np.float64):
        obs, _reward, _terminated, _truncated, _info = env.step(action.copy())
    if obs is None:
        return current_pixels(env)
    return extract_pixels(obs)


def inspect_hdf5(h5) -> dict[str, Any]:
    keys = {}
    for key in sorted(h5.keys()):
        obj = h5[key]
        shape = tuple(int(dim) for dim in obj.shape)
        keys[key] = {"shape": shape, "dtype": str(obj.dtype)}
    ep_len = np.asarray(h5["ep_len"][:], dtype=np.int64) if "ep_len" in h5 else np.array([])
    action = h5["action"]
    finite_rows = np.isfinite(action[: min(len(action), 10000)]).all(axis=1)
    action_preview = np.asarray(action[: min(len(action), 10000)][finite_rows], dtype=np.float64)
    return {
        "keys": keys,
        "num_transitions": int(action.shape[0]),
        "num_episodes": int(h5["ep_len"].shape[0]) if "ep_len" in h5 else None,
        "episode_length_min": int(ep_len.min()) if ep_len.size else None,
        "episode_length_max": int(ep_len.max()) if ep_len.size else None,
        "action_preview_min": action_preview.min(axis=0).tolist() if action_preview.size else None,
        "action_preview_max": action_preview.max(axis=0).tolist() if action_preview.size else None,
    }


def action_mean_std_torch(dataset, torch_module) -> tuple[np.ndarray, np.ndarray]:
    raw = dataset.get_col_data("action")
    if not isinstance(raw, torch_module.Tensor):
        raw = torch_module.as_tensor(raw)
    raw = raw.float()
    raw = raw[torch_module.isfinite(raw).all(dim=-1)]
    mean = raw.mean(dim=0).detach().cpu().numpy()
    std = raw.std(dim=0).detach().cpu().numpy()
    std = np.maximum(std, 1e-12)
    return mean.astype(np.float64), std.astype(np.float64)


def make_dataset(runtime: dict[str, Any], args: argparse.Namespace):
    swm = runtime["swm"]
    Compose = runtime["Compose"]
    get_img_preprocessor = runtime["get_img_preprocessor"]
    get_column_normalizer = runtime["get_column_normalizer"]

    dataset = swm.data.load_dataset(
        str(args.dataset_path),
        cache_dir=args.cache_dir,
        num_steps=args.history_size + args.num_preds,
        frameskip=args.frameskip,
        format="hdf5",
        keys_to_load=["pixels", "action"],
        keys_to_cache=["action"],
        transform=None,
    )
    image_transform = get_img_preprocessor("pixels", "pixels", args.img_size)
    action_transform = get_column_normalizer(dataset, "action", "action")
    dataset.transform = Compose(image_transform, action_transform)
    return dataset, image_transform, action_transform


def make_preflight(
    args: argparse.Namespace,
    h5,
    dataset,
    action_low: np.ndarray,
    action_high: np.ndarray,
    action_mean: np.ndarray,
    action_std: np.ndarray,
) -> dict[str, Any]:
    dataset_info = inspect_hdf5(h5)
    return {
        "dataset": str(Path(args.dataset_path).resolve()),
        "dataset_name": args.dataset_name,
        "number_of_transitions": dataset_info["num_transitions"],
        "number_of_episodes": dataset_info["num_episodes"],
        "episode_length_min": dataset_info["episode_length_min"],
        "episode_length_max": dataset_info["episode_length_max"],
        "dataset_length_after_windowing": int(len(dataset)),
        "action_dim": int(dataset.get_dim("action")),
        "action_bounds_low": action_low.tolist(),
        "action_bounds_high": action_high.tolist(),
        "action_mean_for_model_normalizer": action_mean.tolist(),
        "action_std_for_model_normalizer": action_std.tolist(),
        "observation_shape": tuple(int(dim) for dim in h5["pixels"].shape[1:]),
        "checkpoint": str(Path(args.cache_dir) / "checkpoints" / args.policy),
        "predictor_input_action_shape": [
            args.history_size,
            args.frameskip * int(dataset.get_dim("action")),
        ],
        "transition_semantics": (
            "model predicts from context rows start,start+frameskip,start+2*frameskip "
            "to row start+3*frameskip using the actual final 5-step action block; "
            "environment finite differences restore qpos/qvel at row start+2*frameskip "
            "and execute the same 5 raw actions with a repeated coordinate perturbation"
        ),
        "simulator_state_reconstruction_method": "restore HDF5 qpos/qvel with env.unwrapped.set_state(qpos, qvel)",
        "hdf5_keys": dataset_info["keys"],
        "action_preview_min_first_10000_rows": dataset_info["action_preview_min"],
        "action_preview_max_first_10000_rows": dataset_info["action_preview_max"],
    }


def print_preflight(preflight: dict[str, Any]) -> None:
    print("\n==================================================")
    print("OGBENCH CUBE PREFLIGHT")
    print("==================================================")
    for key in [
        "dataset",
        "number_of_transitions",
        "number_of_episodes",
        "episode_length_min",
        "episode_length_max",
        "dataset_length_after_windowing",
        "action_dim",
        "action_bounds_low",
        "action_bounds_high",
        "observation_shape",
        "checkpoint",
        "predictor_input_action_shape",
        "simulator_state_reconstruction_method",
        "transition_semantics",
    ]:
        print(f"{key}: {preflight[key]}")
    print("HDF5 keys:")
    for key, value in preflight["hdf5_keys"].items():
        print(f"  {key}: shape={value['shape']} dtype={value['dtype']}")
    print("==================================================\n")


def sample_valid_states(
    h5,
    dataset,
    args: argparse.Namespace,
    action_low: np.ndarray,
    action_high: np.ndarray,
    max_eps: float,
) -> tuple[list[SampleMeta], dict[str, int]]:
    rng = random.Random(args.seed)
    action = h5["action"]
    ep_offsets = np.asarray(h5["ep_offset"][:], dtype=np.int64)
    ep_lens = np.asarray(h5["ep_len"][:], dtype=np.int64)
    clip_lookup: dict[tuple[int, int], int] = {
        (int(ep), int(local_start)): int(idx)
        for idx, (ep, local_start) in enumerate(dataset.clip_indices)
    }
    episodes = list(range(len(ep_lens)))
    rng.shuffle(episodes)
    max_delta = max_eps * (action_high - action_low) / 2.0
    counters = {
        "candidate_episodes": len(episodes),
        "missing_clip": 0,
        "nonfinite_window": 0,
        "bound_failure": 0,
        "terminal_window": 0,
        "accepted": 0,
    }
    samples: list[SampleMeta] = []
    passes = 0
    while len(samples) < args.num_samples and passes < 5:
        passes += 1
        rng.shuffle(episodes)
        for ep in episodes:
            if len(samples) >= args.num_samples:
                break
            ep_len = int(ep_lens[ep])
            span = (args.history_size + args.num_preds) * args.frameskip
            max_local_start = ep_len - span
            if max_local_start < 0:
                continue
            start_choices = list(range(max_local_start + 1))
            rng.shuffle(start_choices)
            for local_start in start_choices:
                dataset_idx = clip_lookup.get((ep, local_start))
                if dataset_idx is None:
                    counters["missing_clip"] += 1
                    continue
                episode_offset = int(ep_offsets[ep])
                raw_row = episode_offset + local_start + (args.history_size - 1) * args.frameskip
                action_start = episode_offset + local_start
                action_end = episode_offset + local_start + span
                action_window = np.asarray(action[action_start:action_end], dtype=np.float64)
                if action_window.shape[0] != span or not np.isfinite(action_window).all():
                    counters["nonfinite_window"] += 1
                    continue
                block = np.asarray(action[raw_row : raw_row + args.frameskip], dtype=np.float64)
                if block.shape != (args.frameskip, int(dataset.get_dim("action"))):
                    counters["nonfinite_window"] += 1
                    continue
                if not np.isfinite(block).all():
                    counters["nonfinite_window"] += 1
                    continue
                if (
                    np.any(block + max_delta[None, :] > action_high[None, :])
                    or np.any(block - max_delta[None, :] < action_low[None, :])
                ):
                    counters["bound_failure"] += 1
                    continue
                if "terminated" in h5:
                    terminated = np.asarray(h5["terminated"][raw_row : raw_row + args.frameskip + 1])
                    if np.any(terminated):
                        counters["terminal_window"] += 1
                        continue
                if "truncated" in h5:
                    truncated = np.asarray(h5["truncated"][raw_row : raw_row + args.frameskip + 1])
                    if np.any(truncated):
                        counters["terminal_window"] += 1
                        continue
                samples.append(
                    SampleMeta(
                        sample_id=len(samples),
                        dataset_idx=int(dataset_idx),
                        episode=int(ep),
                        local_start=int(local_start),
                        raw_row=int(raw_row),
                        timestep=int(local_start + (args.history_size - 1) * args.frameskip),
                    )
                )
                counters["accepted"] += 1
                break
    if len(samples) < args.num_samples:
        raise RuntimeError(
            f"Only found {len(samples)} valid Cube states for max eps={max_eps}; "
            f"needed {args.num_samples}. Counters: {counters}"
        )
    return samples, counters


def image_errors(a: np.ndarray, b: np.ndarray) -> dict[str, float]:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    diff = a - b
    return {
        "mse": float(np.mean(diff * diff)),
        "mae": float(np.mean(np.abs(diff))),
        "max_abs": float(np.max(np.abs(diff))),
    }


def validate_reconstruction(
    env,
    h5,
    samples: list[SampleMeta],
    args: argparse.Namespace,
) -> dict[str, Any]:
    checks = min(args.reconstruction_checks, len(samples))
    rows = []
    for meta in samples[:checks]:
        restore_state(env, h5, meta.raw_row)
        rendered = current_pixels(env)
        stored = np.asarray(h5["pixels"][meta.raw_row])
        restore_err = image_errors(rendered, stored)

        restore_state(env, h5, meta.raw_row)
        block = np.asarray(h5["action"][meta.raw_row : meta.raw_row + args.frameskip], dtype=np.float64)
        rolled = execute_action_block(env, block)
        stored_next = np.asarray(h5["pixels"][meta.raw_row + args.frameskip])
        rollout_err = image_errors(rolled, stored_next)
        rows.append(
            {
                "sample_id": meta.sample_id,
                "dataset_idx": meta.dataset_idx,
                "episode": meta.episode,
                "timestep": meta.timestep,
                "restore_mse": restore_err["mse"],
                "restore_mae": restore_err["mae"],
                "restore_max_abs": restore_err["max_abs"],
                "rollout_mse": rollout_err["mse"],
                "rollout_mae": rollout_err["mae"],
                "rollout_max_abs": rollout_err["max_abs"],
            }
        )
    restore_mae = np.asarray([row["restore_mae"] for row in rows], dtype=np.float64)
    rollout_mae = np.asarray([row["rollout_mae"] for row in rows], dtype=np.float64)
    report = {
        "num_checks": checks,
        "restore_mae": summarize(restore_mae),
        "rollout_mae": summarize(rollout_mae),
        "restore_mse": summarize([row["restore_mse"] for row in rows]),
        "rollout_mse": summarize([row["rollout_mse"] for row in rows]),
        "rows": rows,
    }
    print("\nState reconstruction validation:")
    print(f"  current-frame restore MAE median: {report['restore_mae']['median']:.6g}")
    print(f"  current-frame restore MAE p95:    {report['restore_mae']['p95']:.6g}")
    print(f"  5-step rollout MAE median:        {report['rollout_mae']['median']:.6g}")
    print(f"  5-step rollout MAE p95:           {report['rollout_mae']['p95']:.6g}")
    fail = False
    if report["restore_mae"]["median"] is not None:
        fail = fail or report["restore_mae"]["median"] > args.reconstruction_max_median_mae
        fail = fail or report["restore_mae"]["p95"] > args.reconstruction_max_p95_mae
    if report["rollout_mae"]["median"] is not None:
        fail = fail or report["rollout_mae"]["median"] > args.reconstruction_max_median_mae
        fail = fail or report["rollout_mae"]["p95"] > args.reconstruction_max_p95_mae
    if fail:
        raise RuntimeError(
            "Cube state reconstruction/rollout validation failed. "
            "Refusing to compute G_env because counterfactuals may not start from the stored state. "
            f"Report: {report}"
        )
    return report


def prepare_image_batch(images: list[np.ndarray], torch_module, device) -> dict[str, Any]:
    arr = np.stack([np.asarray(img, dtype=np.uint8) for img in images], axis=0)
    tensor = torch_module.as_tensor(arr).permute(0, 3, 1, 2).unsqueeze(1).to(device)
    return {"pixels": tensor}


def encode_images(model, image_transform, images: list[np.ndarray], torch_module, device) -> np.ndarray:
    batch = prepare_image_batch(images, torch_module, device)
    batch = image_transform(batch)
    with torch_module.no_grad():
        info = model.encode(batch)
    emb = info["emb"]
    if emb.ndim == 3:
        emb = emb[:, 0, :]
    return emb.detach().float().cpu().numpy()


def compute_env_jacobian(
    env,
    h5,
    meta: SampleMeta,
    eps_fraction: float,
    action_low: np.ndarray,
    action_high: np.ndarray,
    action_std: np.ndarray,
    args: argparse.Namespace,
    model,
    image_transform,
    torch_module,
    device,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    action_dim = len(action_low)
    block = np.asarray(h5["action"][meta.raw_row : meta.raw_row + args.frameskip], dtype=np.float64)
    perturb = eps_fraction * (action_high - action_low) / 2.0
    images: list[np.ndarray] = []
    violations = []
    for dim in range(action_dim):
        delta = np.zeros(action_dim, dtype=np.float64)
        delta[dim] = perturb[dim]
        plus = block + delta[None, :]
        minus = block - delta[None, :]
        if np.any(plus > action_high[None, :]) or np.any(minus < action_low[None, :]):
            violations.append(dim)
            continue
        restore_state(env, h5, meta.raw_row)
        images.append(execute_action_block(env, plus))
        restore_state(env, h5, meta.raw_row)
        images.append(execute_action_block(env, minus))
    if violations:
        raise RuntimeError(f"Action-bound finite-difference violation at sample {meta}: {violations}")
    encoded = encode_images(model, image_transform, images, torch_module, device)
    cols_raw = []
    for dim in range(action_dim):
        z_plus = encoded[2 * dim]
        z_minus = encoded[2 * dim + 1]
        cols_raw.append((z_plus - z_minus) / (2.0 * perturb[dim]))
    j_raw = np.stack(cols_raw, axis=1).astype(np.float64)
    j_norm = j_raw * action_std[None, :]
    diagnostics = {
        "eps_fraction": eps_fraction,
        "raw_perturb": perturb.tolist(),
        "base_action_block_mean": block.mean(axis=0).tolist(),
        "base_action_block_min_margin": float(
            np.minimum(block - action_low[None, :], action_high[None, :] - block).min()
        ),
    }
    return j_norm, j_raw, diagnostics


def to_device_batch(sample: dict[str, Any], torch_module, device) -> dict[str, Any]:
    batch = {}
    for key, value in sample.items():
        if isinstance(value, torch_module.Tensor):
            batch[key] = value.unsqueeze(0).to(device)
        else:
            batch[key] = value
    return batch


def compute_model_jacobian(
    dataset,
    meta: SampleMeta,
    args: argparse.Namespace,
    model,
    torch_module,
    device,
) -> np.ndarray:
    sample = dataset[int(meta.dataset_idx)]
    batch = to_device_batch(sample, torch_module, device)
    with torch_module.no_grad():
        info = model.encode(batch)
    emb = info["emb"][:, : args.history_size].detach()
    action = batch["action"][:, : args.history_size].float().detach()
    action_dim = int(dataset.get_dim("action"))
    if action.shape[-1] != args.frameskip * action_dim:
        raise RuntimeError(
            f"Unexpected action shape {tuple(action.shape)}; expected last dim {args.frameskip * action_dim}"
        )
    delta0 = torch_module.zeros((1, action_dim), device=device, dtype=action.dtype, requires_grad=True)

    def fn(delta):
        full_action = action.clone()
        last = full_action[:, -1:, :].reshape(1, 1, args.frameskip, action_dim)
        last = last + delta[:, None, None, :]
        full_action[:, -1:, :] = last.reshape(1, 1, args.frameskip * action_dim)
        effect = (
            model.action_condition(emb, full_action)
            if hasattr(model, "action_condition")
            else model.action_encoder(full_action)
        )
        pred = model.predict(emb, effect)
        return pred[:, -1, :]

    cols = []
    for dim in range(action_dim):
        tangent = torch_module.zeros_like(delta0)
        tangent[:, dim] = 1.0
        _y, j_col = torch_module.autograd.functional.jvp(
            fn,
            (delta0,),
            (tangent,),
            create_graph=False,
            strict=False,
        )
        cols.append(j_col.squeeze(0).detach().float().cpu().numpy())
    return np.stack(cols, axis=1).astype(np.float64)


def sym_gram(jac: np.ndarray) -> np.ndarray:
    gram = jac.T @ jac
    return 0.5 * (gram + gram.T)


def eig_info(gram: np.ndarray, rel_floor: float, abs_floor: float) -> dict[str, Any]:
    gram = 0.5 * (gram + gram.T)
    eigvals_asc, eigvecs_asc = np.linalg.eigh(gram)
    eigvals_asc = np.asarray(eigvals_asc, dtype=np.float64)
    eigvals_desc = eigvals_asc[::-1]
    eigvecs_desc = eigvecs_asc[:, ::-1]
    trace = float(np.trace(gram))
    d = gram.shape[0]
    floor = max(abs_floor, rel_floor * max(trace, 0.0) / max(d, 1))
    eig_clipped_desc = np.maximum(eigvals_desc, 0.0)
    denom = max(float(eigvals_desc[-1]), floor)
    kappa = float(max(float(eigvals_desc[0]), 0.0) / denom) if denom > 0 else math.inf
    rank = int(np.sum(eigvals_asc > floor))
    normalized = eig_clipped_desc / (eig_clipped_desc.sum() + 1e-30)
    entropy = -float(np.sum(normalized * np.log(normalized + 1e-30)))
    erank = float(np.exp(entropy))
    return {
        "eigvals_desc": eigvals_desc,
        "eigvecs_desc": eigvecs_desc,
        "trace": trace,
        "floor": floor,
        "rank_effective": rank,
        "kappa": kappa,
        "normalized_spectrum": normalized,
        "effective_rank": erank,
        "raw_min_eig": float(eigvals_desc[-1]),
        "raw_max_eig": float(eigvals_desc[0]),
    }


def trace_normalize(gram: np.ndarray, tiny: float = 1e-30) -> np.ndarray:
    return gram / (float(np.trace(gram)) + tiny)


def top1_angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    dot = float(abs(np.dot(v1, v2)))
    dot = min(1.0, max(0.0, dot))
    return float(np.degrees(np.arccos(dot)))


def principal_angles_deg(v_env: np.ndarray, v_model: np.ndarray, k: int) -> tuple[float, float]:
    k = min(k, v_env.shape[1], v_model.shape[1])
    if k <= 0:
        return math.nan, math.nan
    singular = np.linalg.svd(v_env[:, :k].T @ v_model[:, :k], compute_uv=False)
    singular = np.clip(singular, 0.0, 1.0)
    angles = np.degrees(np.arccos(singular))
    return float(np.max(angles)), float(np.mean(angles))


def relative_metric_stats(
    g_env: np.ndarray,
    g_model: np.ndarray,
    env_info: dict[str, Any],
) -> dict[str, Any]:
    vals, vecs = np.linalg.eigh(0.5 * (g_env + g_env.T))
    floor = float(env_info["floor"])
    vals_reg = np.maximum(vals, floor)
    invsqrt = vecs @ np.diag(1.0 / np.sqrt(vals_reg)) @ vecs.T
    rel = invsqrt @ g_model @ invsqrt
    rel = 0.5 * (rel + rel.T)
    rel_vals = np.linalg.eigvalsh(rel)
    rel_vals = np.maximum(rel_vals, 0.0)
    rel_min = float(rel_vals[0])
    rel_max = float(rel_vals[-1])
    rel_condition = float(rel_max / max(rel_min, 1e-30))
    gm = float(np.exp(np.mean(np.log(rel_vals + 1e-30))))
    rel_scale_normalized = rel_vals / max(gm, 1e-30)
    return {
        "relative_eigvals": rel_vals,
        "relative_lambda_min": rel_min,
        "relative_lambda_max": rel_max,
        "relative_condition": rel_condition,
        "relative_scale_normalized_eigvals": rel_scale_normalized,
    }


def compute_record(
    meta: SampleMeta,
    eps: float,
    j_env: np.ndarray,
    j_env_raw: np.ndarray,
    j_model: np.ndarray,
    args: argparse.Namespace,
    env_diag: dict[str, Any],
) -> dict[str, Any]:
    g_env = sym_gram(j_env)
    g_model = sym_gram(j_model)
    env_info = eig_info(g_env, args.rank_rel_floor, args.eig_abs_floor)
    model_info = eig_info(g_model, args.rank_rel_floor, args.eig_abs_floor)
    g_env_norm = trace_normalize(g_env)
    g_model_norm = trace_normalize(g_model)
    d_g = float(np.linalg.norm(g_model_norm - g_env_norm, ord="fro"))
    d_kappa = float(math.log(model_info["kappa"] + 1e-30) - math.log(env_info["kappa"] + 1e-30))
    e_kappa = abs(d_kappa)
    r = float(env_info["kappa"] / (model_info["kappa"] + 1e-30))
    d_scale = abs(float(math.log((model_info["trace"] + 1e-30) / (env_info["trace"] + 1e-30))))
    top1 = top1_angle_deg(env_info["eigvecs_desc"][:, 0], model_info["eigvecs_desc"][:, 0])
    sub2_max, sub2_mean = principal_angles_deg(env_info["eigvecs_desc"], model_info["eigvecs_desc"], 2)
    sub3_max, sub3_mean = principal_angles_deg(env_info["eigvecs_desc"], model_info["eigvecs_desc"], 3)
    rel = relative_metric_stats(g_env, g_model, env_info)
    record = {
        "sample_id": meta.sample_id,
        "dataset_idx": meta.dataset_idx,
        "episode": meta.episode,
        "timestep": meta.timestep,
        "raw_row": meta.raw_row,
        "epsilon": eps,
        "D_G": d_g,
        "kappa_env": env_info["kappa"],
        "kappa_model": model_info["kappa"],
        "D_kappa": d_kappa,
        "E_kappa": e_kappa,
        "r_env_over_model": r,
        "D_scale": d_scale,
        "top1_angle_deg": top1,
        "subspace2_angle_max_deg": sub2_max,
        "subspace2_angle_mean_deg": sub2_mean,
        "subspace3_angle_max_deg": sub3_max,
        "subspace3_angle_mean_deg": sub3_mean,
        "effective_rank_env": env_info["effective_rank"],
        "effective_rank_model": model_info["effective_rank"],
        "rank_effective_env": env_info["rank_effective"],
        "rank_effective_model": model_info["rank_effective"],
        "trace_env": env_info["trace"],
        "trace_model": model_info["trace"],
        "raw_min_eig_env": env_info["raw_min_eig"],
        "raw_min_eig_model": model_info["raw_min_eig"],
        "raw_max_eig_env": env_info["raw_max_eig"],
        "raw_max_eig_model": model_info["raw_max_eig"],
        "eig_floor_env": env_info["floor"],
        "eig_floor_model": model_info["floor"],
        "relative_lambda_min": rel["relative_lambda_min"],
        "relative_lambda_max": rel["relative_lambda_max"],
        "relative_condition": rel["relative_condition"],
        "base_action_block_min_margin": env_diag["base_action_block_min_margin"],
    }
    for i, value in enumerate(env_info["eigvals_desc"]):
        record[f"eig_env_{i}"] = float(value)
    for i, value in enumerate(model_info["eigvals_desc"]):
        record[f"eig_model_{i}"] = float(value)
    for i, value in enumerate(env_info["normalized_spectrum"]):
        record[f"spectrum_env_{i}"] = float(value)
    for i, value in enumerate(model_info["normalized_spectrum"]):
        record[f"spectrum_model_{i}"] = float(value)
    return {
        "record": record,
        "G_env": g_env,
        "G_model": g_model,
        "J_env": j_env,
        "J_env_raw": j_env_raw,
        "J_model": j_model,
        "eigvals_env": env_info["eigvals_desc"],
        "eigvals_model": model_info["eigvals_desc"],
        "spectrum_env": env_info["normalized_spectrum"],
        "spectrum_model": model_info["normalized_spectrum"],
        "relative_eigvals": rel["relative_eigvals"],
        "relative_scale_normalized_eigvals": rel["relative_scale_normalized_eigvals"],
    }


def summarize_records(records: list[dict[str, Any]], eps: float, bootstrap_resamples: int) -> dict[str, Any]:
    eps_records = [row for row in records if float(row["epsilon"]) == float(eps)]
    out: dict[str, Any] = {"epsilon": eps, "N": len(eps_records)}
    for key in [
        "D_G",
        "kappa_env",
        "kappa_model",
        "D_kappa",
        "E_kappa",
        "r_env_over_model",
        "D_scale",
        "top1_angle_deg",
        "subspace2_angle_max_deg",
        "subspace2_angle_mean_deg",
        "subspace3_angle_max_deg",
        "subspace3_angle_mean_deg",
        "effective_rank_env",
        "effective_rank_model",
        "relative_condition",
    ]:
        out[key] = summarize([row[key] for row in eps_records])
    k_env = np.asarray([row["kappa_env"] for row in eps_records], dtype=np.float64)
    k_model = np.asarray([row["kappa_model"] for row in eps_records], dtype=np.float64)
    d_k = np.asarray([row["D_kappa"] for row in eps_records], dtype=np.float64)
    d_g = np.asarray([row["D_G"] for row in eps_records], dtype=np.float64)
    e_k = np.asarray([row["E_kappa"] for row in eps_records], dtype=np.float64)
    angle = np.asarray([row["top1_angle_deg"] for row in eps_records], dtype=np.float64)
    out["fraction_kappa_model_gt_env"] = fraction(d_k, lambda x: x > 0.0)
    out["fraction_D_kappa_gt_0"] = out["fraction_kappa_model_gt_env"]
    out["fraction_top1_angle_le_10deg"] = fraction(angle, lambda x: x <= 10.0)
    out["fraction_top1_angle_le_30deg"] = fraction(angle, lambda x: x <= 30.0)
    out["pearson_kappa_env_model"] = corrcoef(k_env, k_model)
    out["spearman_kappa_env_model"] = corrcoef(rankdata_simple(k_env), rankdata_simple(k_model))
    rng = np.random.default_rng(42)
    out["bootstrap_ci_95"] = {
        "median_D_G": bootstrap_ci(d_g, np.median, bootstrap_resamples, rng),
        "median_D_kappa": bootstrap_ci(d_k, np.median, bootstrap_resamples, rng),
        "median_E_kappa": bootstrap_ci(e_k, np.median, bootstrap_resamples, rng),
        "median_top1_angle_deg": bootstrap_ci(angle, np.median, bootstrap_resamples, rng),
        "fraction_D_kappa_gt_0": bootstrap_ci((d_k > 0.0).astype(np.float64), np.mean, bootstrap_resamples, rng),
    }
    action_dim = len([key for key in eps_records[0].keys() if key.startswith("spectrum_env_")]) if eps_records else 0
    if action_dim:
        env_spec = np.asarray([[row[f"spectrum_env_{i}"] for i in range(action_dim)] for row in eps_records])
        model_spec = np.asarray([[row[f"spectrum_model_{i}"] for i in range(action_dim)] for row in eps_records])
        out["median_normalized_spectrum_env"] = np.median(env_spec, axis=0).tolist()
        out["p25_normalized_spectrum_env"] = np.percentile(env_spec, 25, axis=0).tolist()
        out["p75_normalized_spectrum_env"] = np.percentile(env_spec, 75, axis=0).tolist()
        out["median_normalized_spectrum_model"] = np.median(model_spec, axis=0).tolist()
        out["p25_normalized_spectrum_model"] = np.percentile(model_spec, 25, axis=0).tolist()
        out["p75_normalized_spectrum_model"] = np.percentile(model_spec, 75, axis=0).tolist()
    return out


def epsilon_robustness(records: list[dict[str, Any]], eps_list: list[float]) -> dict[str, Any]:
    by_eps = {
        eps: {int(row["sample_id"]): row for row in records if float(row["epsilon"]) == float(eps)}
        for eps in eps_list
    }
    common_ids = set.intersection(*(set(rows.keys()) for rows in by_eps.values())) if by_eps else set()
    result: dict[str, Any] = {"common_states": len(common_ids), "pairwise": {}}
    for i, eps_a in enumerate(eps_list):
        for eps_b in eps_list[i + 1 :]:
            rows_a = by_eps[eps_a]
            rows_b = by_eps[eps_b]
            ids = sorted(set(rows_a).intersection(rows_b))
            key = f"{eps_a:g}_vs_{eps_b:g}"
            result["pairwise"][key] = {
                "N": len(ids),
                "corr_D_G": corrcoef(
                    np.asarray([rows_a[idx]["D_G"] for idx in ids]),
                    np.asarray([rows_b[idx]["D_G"] for idx in ids]),
                ),
                "corr_D_kappa": corrcoef(
                    np.asarray([rows_a[idx]["D_kappa"] for idx in ids]),
                    np.asarray([rows_b[idx]["D_kappa"] for idx in ids]),
                ),
            }
    return result


def write_csv(path: Path, records: list[dict[str, Any]]) -> None:
    if not records:
        return
    fieldnames = list(records[0].keys())
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row.get(key) for key in fieldnames})


def save_npz(path: Path, arrays: dict[str, list[np.ndarray]], records: list[dict[str, Any]], save_jacobians: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "G_env": np.stack(arrays["G_env"], axis=0),
        "G_model": np.stack(arrays["G_model"], axis=0),
        "eigvals_env": np.stack(arrays["eigvals_env"], axis=0),
        "eigvals_model": np.stack(arrays["eigvals_model"], axis=0),
        "spectrum_env": np.stack(arrays["spectrum_env"], axis=0),
        "spectrum_model": np.stack(arrays["spectrum_model"], axis=0),
        "relative_eigvals": np.stack(arrays["relative_eigvals"], axis=0),
        "sample_id": np.asarray([row["sample_id"] for row in records], dtype=np.int64),
        "epsilon": np.asarray([row["epsilon"] for row in records], dtype=np.float64),
        "episode": np.asarray([row["episode"] for row in records], dtype=np.int64),
        "timestep": np.asarray([row["timestep"] for row in records], dtype=np.int64),
        "dataset_idx": np.asarray([row["dataset_idx"] for row in records], dtype=np.int64),
        "raw_row": np.asarray([row["raw_row"] for row in records], dtype=np.int64),
    }
    if save_jacobians:
        payload["J_env"] = np.stack(arrays["J_env"], axis=0)
        payload["J_env_raw"] = np.stack(arrays["J_env_raw"], axis=0)
        payload["J_model"] = np.stack(arrays["J_model"], axis=0)
    np.savez_compressed(path, **payload)


def plot_for_epsilon(records: list[dict[str, Any]], eps: float, outdir: Path, plt) -> list[str]:
    eps_records = [row for row in records if float(row["epsilon"]) == float(eps)]
    if not eps_records:
        return []
    paths: list[str] = []

    def savefig(name: str) -> None:
        png = outdir / f"eps_{eps:g}_{name}.png"
        pdf = outdir / f"eps_{eps:g}_{name}.pdf"
        plt.savefig(png, dpi=220, bbox_inches="tight")
        plt.savefig(pdf, bbox_inches="tight")
        paths.extend([str(png), str(pdf)])
        plt.close()

    k_env = np.asarray([row["kappa_env"] for row in eps_records], dtype=np.float64)
    k_model = np.asarray([row["kappa_model"] for row in eps_records], dtype=np.float64)
    d_k = np.asarray([row["D_kappa"] for row in eps_records], dtype=np.float64)
    d_g = np.asarray([row["D_G"] for row in eps_records], dtype=np.float64)
    angle = np.asarray([row["top1_angle_deg"] for row in eps_records], dtype=np.float64)
    frac_gt = float(np.mean(d_k > 0.0))

    x = np.log(k_env + 1e-30)
    y = np.log(k_model + 1e-30)
    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    plt.figure(figsize=(4.2, 4.0))
    plt.scatter(x, y, s=14, alpha=0.65, linewidths=0.0)
    plt.plot([lo, hi], [lo, hi], color="black", linewidth=1.0)
    plt.xlabel(r"$\log \kappa_{\mathrm{env}}$")
    plt.ylabel(r"$\log \kappa_{\mathrm{model}}$")
    plt.title("Environment vs. model local action anisotropy - OGBench Cube")
    plt.text(
        0.04,
        0.96,
        (
            f"model > env: {100.0 * frac_gt:.1f}%\n"
            f"median env: {np.median(k_env):.3g}\n"
            f"median model: {np.median(k_model):.3g}"
        ),
        transform=plt.gca().transAxes,
        va="top",
        ha="left",
        fontsize=8,
    )
    savefig("kappa_scatter")

    plt.figure(figsize=(4.2, 3.0))
    plt.hist(d_k, bins=32, color="#4C78A8", alpha=0.85)
    plt.axvline(0.0, color="black", linewidth=1.0)
    plt.xlabel(r"$D_\kappa=\log(\kappa_{\mathrm{model}}/\kappa_{\mathrm{env}})$")
    plt.ylabel("states")
    plt.title("Signed eccentricity distortion")
    savefig("d_kappa_hist")

    plt.figure(figsize=(4.2, 3.0))
    plt.hist(d_g, bins=32, color="#59A14F", alpha=0.85)
    plt.xlabel(r"$D_G=\|\tilde G_{\mathrm{model}}-\tilde G_{\mathrm{env}}\|_F$")
    plt.ylabel("states")
    plt.title("Trace-normalized metric discrepancy")
    savefig("d_g_hist")

    plt.figure(figsize=(4.2, 3.0))
    plt.hist(angle, bins=32, color="#F28E2B", alpha=0.85)
    plt.xlabel("top-1 direction angle (deg)")
    plt.ylabel("states")
    plt.title("Dominant action-direction alignment")
    savefig("top1_angle_hist")

    action_dim = len([key for key in eps_records[0].keys() if key.startswith("spectrum_env_")])
    env_spec = np.asarray([[row[f"spectrum_env_{i}"] for i in range(action_dim)] for row in eps_records])
    model_spec = np.asarray([[row[f"spectrum_model_{i}"] for i in range(action_dim)] for row in eps_records])
    xs = np.arange(1, action_dim + 1)
    plt.figure(figsize=(4.4, 3.1))
    env_med = np.median(env_spec, axis=0)
    model_med = np.median(model_spec, axis=0)
    env_p25 = np.percentile(env_spec, 25, axis=0)
    env_p75 = np.percentile(env_spec, 75, axis=0)
    model_p25 = np.percentile(model_spec, 25, axis=0)
    model_p75 = np.percentile(model_spec, 75, axis=0)
    plt.plot(xs, env_med, marker="o", label="Environment", linewidth=1.4)
    plt.fill_between(xs, env_p25, env_p75, alpha=0.18)
    plt.plot(xs, model_med, marker="o", label="LeWM", linewidth=1.4)
    plt.fill_between(xs, model_p25, model_p75, alpha=0.18)
    plt.xlabel("eigenvalue index")
    plt.ylabel("normalized eigenvalue")
    plt.title("Median normalized action-metric spectrum")
    plt.legend(frameon=False)
    savefig("normalized_spectrum")
    return paths


def plot_epsilon_robustness(records: list[dict[str, Any]], eps_list: list[float], outdir: Path, plt) -> list[str]:
    if len(eps_list) < 2:
        return []
    paths: list[str] = []
    med_dg = []
    med_dk = []
    med_angle = []
    for eps in eps_list:
        rows = [row for row in records if float(row["epsilon"]) == float(eps)]
        med_dg.append(float(np.median([row["D_G"] for row in rows])))
        med_dk.append(float(np.median([row["D_kappa"] for row in rows])))
        med_angle.append(float(np.median([row["top1_angle_deg"] for row in rows])))

    def save_line(values: list[float], ylabel: str, name: str) -> None:
        plt.figure(figsize=(4.2, 3.0))
        plt.plot(eps_list, values, marker="o", linewidth=1.4)
        plt.xscale("log")
        plt.xlabel("finite-difference epsilon")
        plt.ylabel(ylabel)
        plt.title("Epsilon robustness")
        png = outdir / f"epsilon_robustness_{name}.png"
        pdf = outdir / f"epsilon_robustness_{name}.pdf"
        plt.savefig(png, dpi=220, bbox_inches="tight")
        plt.savefig(pdf, bbox_inches="tight")
        plt.close()
        paths.extend([str(png), str(pdf)])

    save_line(med_dg, r"median $D_G$", "median_d_g")
    save_line(med_dk, r"median $D_\kappa$", "median_d_kappa")
    save_line(med_angle, "median top-1 angle (deg)", "median_top1_angle")
    return paths


def print_summary(summary: dict[str, Any]) -> None:
    print("\n==================================================")
    print("OGBENCH CUBE LOCAL GEOMETRY DIAGNOSTIC")
    print("==================================================")
    print(f"N valid states: {summary['num_valid_states']}")
    for eps_summary in summary["by_epsilon"]:
        print(f"\nepsilon: {eps_summary['epsilon']}")
        print("D_G")
        print(f"  mean:   {eps_summary['D_G']['mean']:.6g}")
        print(f"  median: {eps_summary['D_G']['median']:.6g}")
        print(f"  p90:    {eps_summary['D_G']['p90']:.6g}")
        ci = eps_summary["bootstrap_ci_95"]["median_D_G"]
        if ci is not None:
            print(f"  95% CI median: [{ci[0]:.6g}, {ci[1]:.6g}]")
        print("Condition numbers")
        print(f"  env median:   {eps_summary['kappa_env']['median']:.6g}")
        print(f"  model median: {eps_summary['kappa_model']['median']:.6g}")
        print("Signed log distortion")
        print(f"  median D_kappa: {eps_summary['D_kappa']['median']:.6g}")
        frac_gt = eps_summary["fraction_kappa_model_gt_env"]
        if frac_gt is not None:
            print(f"  fraction model > env: {100.0 * frac_gt:.2f}%")
        print("Eccentricity error")
        print(f"  median E_kappa: {eps_summary['E_kappa']['median']:.6g}")
        print("Direction alignment")
        print(f"  top-1 median angle: {eps_summary['top1_angle_deg']['median']:.6g} deg")
        print(f"  <= 10 deg: {100.0 * eps_summary['fraction_top1_angle_le_10deg']:.2f}%")
        print(f"  <= 30 deg: {100.0 * eps_summary['fraction_top1_angle_le_30deg']:.2f}%")
        print("Effective rank")
        print(f"  env median:   {eps_summary['effective_rank_env']['median']:.6g}")
        print(f"  model median: {eps_summary['effective_rank_model']['median']:.6g}")
        print("Scale mismatch")
        print(f"  median D_scale: {eps_summary['D_scale']['median']:.6g}")
        print("Correlation")
        print(f"  Pearson kappa:  {eps_summary['pearson_kappa_env_model']}")
        print(f"  Spearman kappa: {eps_summary['spearman_kappa_env_model']}")
    if summary.get("epsilon_robustness"):
        print("\nEpsilon robustness:")
        print(json.dumps(summary["epsilon_robustness"], indent=2))
    print("==================================================\n")


def main() -> None:
    args = parse_args()
    eps_list = parse_eps_list(args.eps_list)
    ensure_paths(args)
    runtime = import_runtime(args)
    torch_module = runtime["torch"]
    disable_fast_attention(torch_module)
    if args.device == "cuda" and not torch_module.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        args.device = "cpu"
    device = torch_module.device(args.device)
    torch_module.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    h5py = runtime["h5py"]
    gym = runtime["gym"]
    swm = runtime["swm"]
    plt = runtime["plt"]

    dataset, image_transform, _action_transform = make_dataset(runtime, args)
    action_mean, action_std = action_mean_std_torch(dataset, torch_module)
    action_dim = int(dataset.get_dim("action"))

    env = build_env(gym, args)
    action_low = np.asarray(env.action_space.low, dtype=np.float64)
    action_high = np.asarray(env.action_space.high, dtype=np.float64)
    if action_low.shape[0] != action_dim:
        raise RuntimeError(f"Env action dim {action_low.shape[0]} != dataset action dim {action_dim}")

    with h5py.File(args.dataset_path, "r", libver="latest", swmr=True) as h5:
        preflight = make_preflight(args, h5, dataset, action_low, action_high, action_mean, action_std)
        print_preflight(preflight)
        if args.preflight_only:
            return

        samples, sample_counters = sample_valid_states(
            h5,
            dataset,
            args,
            action_low,
            action_high,
            max(eps_list),
        )
        print(f"Sampled {len(samples)} valid states. Sampling counters: {sample_counters}")

        reconstruction = validate_reconstruction(env, h5, samples, args)

        print(f"Loading pretrained LeWM policy {args.policy} ...")
        model = swm.wm.utils.load_pretrained(args.policy, cache_dir=args.cache_dir).to(device).eval()
        model.requires_grad_(False)

        output_dir = Path(args.output_dir)
        if args.run_name:
            output_dir = output_dir / args.run_name
        output_dir.mkdir(parents=True, exist_ok=True)

        records: list[dict[str, Any]] = []
        arrays: dict[str, list[np.ndarray]] = {
            "G_env": [],
            "G_model": [],
            "J_env": [],
            "J_env_raw": [],
            "J_model": [],
            "eigvals_env": [],
            "eigvals_model": [],
            "spectrum_env": [],
            "spectrum_model": [],
            "relative_eigvals": [],
            "relative_scale_normalized_eigvals": [],
        }

        model_j_cache: dict[int, np.ndarray] = {}
        for eps in eps_list:
            print(f"\nComputing Cube geometry for epsilon={eps:g}")
            for idx, meta in enumerate(samples):
                if meta.sample_id not in model_j_cache:
                    model_j_cache[meta.sample_id] = compute_model_jacobian(
                        dataset,
                        meta,
                        args,
                        model,
                        torch_module,
                        device,
                    )
                j_model = model_j_cache[meta.sample_id]
                j_env, j_env_raw, env_diag = compute_env_jacobian(
                    env,
                    h5,
                    meta,
                    eps,
                    action_low,
                    action_high,
                    action_std,
                    args,
                    model,
                    image_transform,
                    torch_module,
                    device,
                )
                result = compute_record(meta, eps, j_env, j_env_raw, j_model, args, env_diag)
                records.append(result["record"])
                for key in arrays:
                    if key in result:
                        arrays[key].append(result[key])
                if (idx + 1) % max(1, min(50, len(samples))) == 0 or idx == 0:
                    print(
                        f"  {idx + 1:4d}/{len(samples)} states | "
                        f"D_G={result['record']['D_G']:.4g} "
                        f"D_kappa={result['record']['D_kappa']:.4g} "
                        f"angle={result['record']['top1_angle_deg']:.3g} deg"
                    )

        by_epsilon = [summarize_records(records, eps, args.bootstrap_resamples) for eps in eps_list]
        summary = {
            "preflight": preflight,
            "seed": args.seed,
            "device": args.device,
            "num_valid_states": len(samples),
            "eps_list": eps_list,
            "sample_counters": sample_counters,
            "reconstruction_validation": reconstruction,
            "rank_rel_floor": args.rank_rel_floor,
            "eig_abs_floor": args.eig_abs_floor,
            "action_std_used_to_convert_env_metric_to_model_normalized_coordinates": action_std.tolist(),
            "by_epsilon": by_epsilon,
            "epsilon_robustness": epsilon_robustness(records, eps_list),
        }

        csv_path = output_dir / "cube_env_model_geometry_per_state.csv"
        json_path = output_dir / "cube_env_model_geometry_summary.json"
        npz_path = output_dir / "cube_env_model_geometry_raw.npz"
        write_csv(csv_path, records)
        save_npz(npz_path, arrays, records, args.save_jacobians)
        plot_paths: list[str] = []
        if not args.no_plots:
            for eps in eps_list:
                plot_paths.extend(plot_for_epsilon(records, eps, output_dir, plt))
            plot_paths.extend(plot_epsilon_robustness(records, eps_list, output_dir, plt))
        summary["outputs"] = {
            "summary_json": str(json_path),
            "per_state_csv": str(csv_path),
            "raw_npz": str(npz_path),
            "plots": plot_paths,
        }
        json_path.write_text(json.dumps(json_safe(summary), indent=2), encoding="utf-8")
        print_summary(summary)
        print(f"Saved summary: {json_path}")
        print(f"Saved per-state CSV: {csv_path}")
        print(f"Saved raw NPZ: {npz_path}")


if __name__ == "__main__":
    main()
