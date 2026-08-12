#!/usr/bin/env python
"""Evaluate a LeWM Reacher checkpoint on dataset-goal rollouts."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "egl")

import hydra
import numpy as np
import stable_pretraining as spt
import stable_worldmodel as swm
import torch
from omegaconf import OmegaConf, open_dict
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lewm-dir", default=os.environ.get("LEWM_DIR", "le-wm"))
    parser.add_argument("--cache-dir", default=os.environ.get("STABLEWM_HOME"))
    parser.add_argument("--policy", default="lewm_reacher_baseline/weights_epoch_3.pt")
    parser.add_argument("--dataset-name", default="dmc/reacher_random")
    parser.add_argument("--num-eval", type=int, default=50)
    parser.add_argument("--sample-count", type=int, default=None)
    parser.add_argument("--sample-offset", type=int, default=0)
    parser.add_argument("--goal-offset-steps", type=int, default=25)
    parser.add_argument("--eval-budget", type=int, default=50)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--solver-samples", type=int, default=300)
    parser.add_argument("--solver-steps", type=int, default=30)
    parser.add_argument("--solver-topk", type=int, default=30)
    parser.add_argument("--planning-value-weight", default=os.environ.get("PLANNING_VALUE_WEIGHT", ""))
    parser.add_argument("--planning-value-gamma", default=os.environ.get("PLANNING_VALUE_GAMMA", ""))
    parser.add_argument("--log-cost-stats", action="store_true")
    parser.add_argument("--cost-log-limit", type=int, default=8)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--random-policy", action="store_true")
    parser.add_argument("--video-dir", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--paths-json", default=None)
    parser.add_argument("--path-keys", default="qpos,qvel,goal_qpos")
    return parser.parse_args()


def jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(item) for item in value]
    return value


def img_transform(img_size: int):
    return transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=img_size),
        ]
    )


def get_episodes_length(dataset, episodes: np.ndarray) -> np.ndarray:
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    return np.array([np.max(step_idx[episode_idx == ep_id]) + 1 for ep_id in episodes])


def sample_eval_starts(
    dataset,
    num_eval: int,
    goal_offset_steps: int,
    seed: int,
    sample_count: int | None = None,
    sample_offset: int = 0,
):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices = np.unique(dataset.get_col_data(col_name))
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - goal_offset_steps - 1
    max_start_by_episode = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    max_start_per_row = np.array(
        [max_start_by_episode[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )
    valid_mask = dataset.get_col_data("step_idx") <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]
    sample_count = sample_count or num_eval
    if sample_offset < 0:
        raise ValueError(f"sample_offset must be non-negative, got {sample_offset}")
    if sample_offset + num_eval > sample_count:
        raise ValueError(
            "sample_offset + num_eval must be <= sample_count, got "
            f"{sample_offset} + {num_eval} > {sample_count}"
        )
    if len(valid_indices) < sample_count:
        raise ValueError(f"Need {sample_count} valid starts, found {len(valid_indices)}")

    rng = np.random.default_rng(seed)
    chosen_pool = np.sort(rng.choice(valid_indices, size=sample_count, replace=False))
    chosen = chosen_pool[sample_offset : sample_offset + num_eval]
    rows = dataset.get_row_data(chosen)
    return rows[col_name].tolist(), rows["step_idx"].tolist(), len(valid_indices)


def build_policy(cfg, args: argparse.Namespace, process: dict):
    if args.random_policy:
        return swm.policy.RandomPolicy(), "random"

    model = swm.wm.utils.load_pretrained(args.policy, cache_dir=args.cache_dir)
    model = model.to(args.device).eval()
    model.requires_grad_(False)
    model.interpolate_pos_encoding = True
    if args.planning_value_weight and hasattr(model, "planning_value_weight"):
        model.planning_value_weight = float(args.planning_value_weight)
    if args.planning_value_gamma and hasattr(model, "planning_value_gamma"):
        model.planning_value_gamma = float(args.planning_value_gamma)
    if hasattr(model, "log_cost_stats"):
        model.log_cost_stats = bool(args.log_cost_stats)
    if hasattr(model, "cost_log_limit"):
        model.cost_log_limit = int(args.cost_log_limit)

    with open_dict(cfg):
        cfg.solver.num_samples = args.solver_samples
        cfg.solver.n_steps = args.solver_steps
        cfg.solver.topk = args.solver_topk
        cfg.solver.device = args.device

    plan_config = swm.PlanConfig(**OmegaConf.to_container(cfg.plan_config, resolve=True))
    solver = hydra.utils.instantiate(cfg.solver, model=model)
    transform = {
        "pixels": img_transform(args.img_size),
        "goal": img_transform(args.img_size),
    }
    policy = swm.policy.WorldModelPolicy(
        solver=solver, config=plan_config, process=process, transform=transform
    )
    return policy, args.policy


def evaluate_with_first_success_steps(
    world,
    dataset,
    episodes_idx: list[int],
    start_steps: list[int],
    goal_offset: int,
    eval_budget: int,
    callables: list[dict],
    video_dir: str | None,
    path_keys: list[str] | None = None,
) -> dict:
    from stable_worldmodel.world import world as world_module

    init_state, goal_state, _dataset_videos = world_module._extract_init_goal(
        dataset, episodes_idx, start_steps, goal_offset
    )
    world.reset(seed=init_state.get("seed"))

    if callables:
        merged = {**init_state, **goal_state}
        for i in range(len(episodes_idx)):
            env_init = {key: value[i] for key, value in merged.items()}
            world_module._apply_callables(world.envs.envs[i].unwrapped, callables, env_init)

    shape_prefix = world.infos["pixels"].shape[:2]
    for src in (init_state, goal_state):
        for key, value in src.items():
            if key in world.infos or key in goal_state:
                world.infos[key] = np.broadcast_to(
                    value[:, None, ...], shape_prefix + value.shape[1:]
                ).copy()

    goal_snapshot = {key: world.infos[key].copy() for key in goal_state}
    successes = np.zeros(len(episodes_idx), dtype=bool)
    first_success_step = np.full(len(episodes_idx), -1, dtype=np.int64)
    paths = {key: [] for key in (path_keys or [])}
    step = {"value": 0}

    def on_step(current_world):
        step["value"] += 1
        current_world.infos.update(deepcopy(goal_snapshot))
        for key in paths:
            if key in current_world.infos:
                paths[key].append(np.asarray(current_world.infos[key]).copy())
        newly_successful = np.logical_and(current_world.terminateds, ~successes)
        first_success_step[newly_successful] = step["value"]
        successes[:] = np.logical_or(successes, current_world.terminateds)

    world._run(max_steps=eval_budget, mode="wait", on_step=on_step)

    successful_steps = first_success_step[first_success_step > 0]
    return {
        "success_rate": float(successes.sum()) / len(successes) * 100.0,
        "success_count": int(successes.sum()),
        "num_eval": int(len(successes)),
        "episode_successes": successes,
        "first_success_step": first_success_step,
        "successful_steps_mean": float(successful_steps.mean()) if len(successful_steps) else None,
        "successful_steps_median": float(np.median(successful_steps)) if len(successful_steps) else None,
        "successful_steps_min": int(successful_steps.min()) if len(successful_steps) else None,
        "successful_steps_max": int(successful_steps.max()) if len(successful_steps) else None,
        "paths": {key: np.stack(value, axis=1) for key, value in paths.items() if value},
    }


def main() -> None:
    args = parse_args()
    lewm_dir = Path(args.lewm_dir).resolve()
    if not lewm_dir.exists():
        raise SystemExit(f"Missing LeWM directory: {lewm_dir}")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false.")

    sys.path.insert(0, str(lewm_dir))
    os.environ.setdefault("STABLEWM_HOME", args.cache_dir or "")
    os.environ.setdefault("LOCAL_DATASET_DIR", args.cache_dir or "")

    with hydra.initialize_config_dir(version_base=None, config_dir=str(lewm_dir / "config" / "eval")):
        cfg = hydra.compose(
            config_name="reacher",
            overrides=[
                f"seed={args.seed}",
                f"eval.num_eval={args.num_eval}",
                f"eval.goal_offset_steps={args.goal_offset_steps}",
                f"eval.eval_budget={args.eval_budget}",
                f"eval.img_size={args.img_size}",
                f"eval.dataset_name={args.dataset_name}",
                f"world.max_episode_steps={2 * args.eval_budget}",
            ],
        )

    cache_dir = Path(args.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        args.dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=cache_dir,
    )

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col == "pixels":
            continue
        processor = preprocessing.StandardScaler()
        col_data = dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor
        if col != "action":
            process[f"goal_{col}"] = process[col]

    policy, policy_name = build_policy(cfg, args, process)
    world = swm.World(**cfg.world, image_shape=(224, 224))
    world.set_policy(policy)

    episodes_idx, start_steps, valid_start_count = sample_eval_starts(
        dataset,
        args.num_eval,
        args.goal_offset_steps,
        args.seed,
        sample_count=args.sample_count,
        sample_offset=args.sample_offset,
    )
    start_time = time.time()
    metrics = evaluate_with_first_success_steps(
        world=world,
        dataset=dataset,
        episodes_idx=episodes_idx,
        start_steps=start_steps,
        goal_offset=args.goal_offset_steps,
        eval_budget=args.eval_budget,
        callables=OmegaConf.to_container(cfg.eval.callables, resolve=True),
        video_dir=args.video_dir,
        path_keys=[key.strip() for key in args.path_keys.split(",") if key.strip()],
    )
    elapsed = time.time() - start_time
    paths = metrics.pop("paths", {})

    summary = {
        "ok": True,
        "policy": policy_name,
        "dataset_name": args.dataset_name,
        "valid_start_count": valid_start_count,
        "goal_offset_steps": args.goal_offset_steps,
        "eval_budget": args.eval_budget,
        "seed": args.seed,
        "sample_count": args.sample_count or args.num_eval,
        "sample_offset": args.sample_offset,
        "solver": None
        if args.random_policy
        else {
            "num_samples": args.solver_samples,
            "n_steps": args.solver_steps,
            "topk": args.solver_topk,
        },
        "elapsed_seconds": elapsed,
        "episodes_idx": episodes_idx,
        "start_steps": start_steps,
        "metrics": metrics,
    }

    text = json.dumps(jsonable(summary), indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")
    if args.paths_json:
        paths_payload = {
            "policy": policy_name,
            "dataset_name": args.dataset_name,
            "seed": args.seed,
            "episodes_idx": episodes_idx,
            "start_steps": start_steps,
            "first_success_step": metrics["first_success_step"],
            "episode_successes": metrics["episode_successes"],
            "paths": paths,
        }
        paths_path = Path(args.paths_json)
        paths_path.parent.mkdir(parents=True, exist_ok=True)
        paths_path.write_text(json.dumps(jsonable(paths_payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
