#!/usr/bin/env python
"""Compare true Reacher action geometry with pretrained LeWM geometry."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

os.environ.setdefault("MUJOCO_GL", "glfw" if os.name == "nt" else "egl")
os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parents[1] / ".cache" / "matplotlib"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lewm-dir", default=os.environ.get("LEWM_DIR", "le-wm"))
    parser.add_argument("--cache-dir", default=os.environ.get("STABLEWM_HOME", "stablewm_home"))
    parser.add_argument("--policy", default="quentinll/lewm-reacher")
    parser.add_argument("--dataset-name", default="dmc/reacher_random.h5")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--num-preds", type=int, default=1)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--eval-starts-json", default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fd-step", type=float, default=1e-3)
    parser.add_argument("--env-metric-coordinate", choices=["raw_action", "normalized_action"], default="normalized_action")
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--plot-png", default=None)
    parser.add_argument("--save-per-state", action="store_true")
    return parser.parse_args()


def move_to_device(value: Any, device):
    import torch

    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def percentile(values, q):
    import numpy as np

    return float(np.percentile(values, q)) if len(values) else None


def summarize(values):
    import numpy as np

    arr = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "p10": percentile(arr, 10),
        "p25": percentile(arr, 25),
        "p50": percentile(arr, 50),
        "p75": percentile(arr, 75),
        "p90": percentile(arr, 90),
        "p95": percentile(arr, 95),
        "max": float(arr.max()),
    }


def ratio_and_dom(metric, eps):
    import numpy as np

    eigvals, eigvecs = np.linalg.eigh(metric)
    eigvals = np.clip(eigvals, 0.0, None)
    return (
        float((eigvals[0] + eps) / (eigvals[-1] + eps)),
        eigvals,
        eigvecs[:, -1],
    )


def condition_from_ratio(ratio, eps):
    return 1.0 / max(float(ratio), eps)


def angle_degrees(u, v):
    import numpy as np

    cos = float(abs(np.dot(u, v)))
    cos = min(1.0, max(0.0, cos))
    return float(np.degrees(np.arccos(cos)))


def norm_trace(metric, eps):
    import numpy as np

    return metric / (float(np.trace(metric)) + eps)


def build_env():
    import gymnasium as gym
    import stable_worldmodel.envs  # noqa: F401

    return gym.make("swm/ReacherDMControl-v0", task="qpos_match")


def env_next_state(env, qpos, qvel, action, frameskip):
    import numpy as np

    env.unwrapped.set_state(np.asarray(qpos), np.asarray(qvel))
    action = np.asarray(action, dtype=np.float64)
    for _ in range(frameskip):
        env.step(action.copy())
    info = env.unwrapped.info
    return np.concatenate([info["qpos"], info["qvel"]]).astype(np.float64)


def env_metric(env, qpos, qvel, action, frameskip, fd_step):
    import numpy as np

    cols = []
    for i in range(2):
        delta = np.zeros(2, dtype=np.float64)
        delta[i] = fd_step
        plus = env_next_state(env, qpos, qvel, np.clip(action + delta, -1.0, 1.0), frameskip)
        minus = env_next_state(env, qpos, qvel, np.clip(action - delta, -1.0, 1.0), frameskip)
        cols.append((plus - minus) / (2.0 * fd_step))
    jac = np.stack(cols, axis=1)
    return jac.T @ jac


def action_mean_std(h5):
    import numpy as np

    action = h5["action"]
    mean = np.asarray(action[:], dtype=np.float64).mean(axis=0)
    std = np.asarray(action[:], dtype=np.float64).std(axis=0)
    std = np.where(std > 0, std, 1.0)
    return mean, std


def make_full_action(base_action, template_action, frameskip):
    return base_action.unsqueeze(-2).expand(*base_action.shape[:-1], frameskip, 2).reshape_as(template_action)


def model_metrics_for_batch(model, emb, action, frameskip, eps):
    import torch

    state_window = emb.detach()
    action_window = action.detach()
    base = action_window[:, -1:, :].reshape(action_window.size(0), 1, frameskip, 2).mean(dim=-2)
    base = base.detach().requires_grad_(True)
    cols = []

    def fn(base_action):
        full_action = action_window.clone()
        full_action[:, -1:, :] = make_full_action(base_action, action_window[:, -1:, :], frameskip)
        effect = model.action_condition(state_window, full_action) if hasattr(model, "action_condition") else model.action_encoder(full_action)
        return model.predict(state_window, effect)[:, -1, :]

    for i in range(2):
        tangent = torch.zeros_like(base)
        tangent[..., i] = 1.0
        _, j_col = torch.autograd.functional.jvp(
            fn,
            (base,),
            (tangent,),
            create_graph=False,
            strict=False,
        )
        cols.append(j_col.float())

    j1, j2 = cols
    g00 = (j1 * j1).sum(dim=1)
    g01 = (j1 * j2).sum(dim=1)
    g11 = (j2 * j2).sum(dim=1)
    grams = torch.stack(
        [
            torch.stack([g00, g01], dim=-1),
            torch.stack([g01, g11], dim=-1),
        ],
        dim=-2,
    )
    return grams.detach().cpu().numpy()


def main() -> None:
    args = parse_args()
    output_json = Path(args.output_json).resolve() if args.output_json else None
    plot_png = Path(args.plot_png).resolve() if args.plot_png else None
    eval_starts_json = Path(args.eval_starts_json).resolve() if args.eval_starts_json else None
    lewm_dir = Path(args.lewm_dir).resolve()
    if not lewm_dir.exists():
        raise SystemExit(f"Missing LeWM directory: {lewm_dir}")

    sys.path.insert(0, str(lewm_dir))
    os.environ.setdefault("STABLEWM_HOME", args.cache_dir)
    os.environ.setdefault("LOCAL_DATASET_DIR", args.cache_dir)

    import hydra
    import h5py
    import numpy as np
    import stable_pretraining as spt
    import stable_worldmodel as swm
    import torch
    from omegaconf import OmegaConf, open_dict

    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    os.chdir(lewm_dir)
    from utils import get_column_normalizer, get_img_preprocessor

    device = torch.device(args.device)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false.")

    overrides = [
        "data=dmc",
        f"data.dataset.name={args.dataset_name}",
        f"loader.batch_size={args.batch_size}",
        "loader.num_workers=0",
        "loader.persistent_workers=false",
        f"img_size={args.img_size}",
        f"history_size={args.history_size}",
        f"num_preds={args.num_preds}",
        "model=lewm",
        "data.dataset.keys_to_load=[pixels,action,qpos,qvel]",
        "data.dataset.keys_to_cache=[action,qpos,qvel]",
    ]
    with hydra.initialize_config_dir(version_base=None, config_dir=str(lewm_dir / "config" / "train")):
        cfg = hydra.compose(config_name="lewm", overrides=overrides)

    cache_dir = Path(args.cache_dir).resolve()
    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    dataset = swm.data.load_dataset(dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg)
    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]
    for col in cfg.data.dataset.keys_to_load:
        if not col.startswith("pixels") and col != "reward":
            transforms.append(get_column_normalizer(dataset, col, col))
    dataset.transform = spt.data.transforms.Compose(*transforms)

    with open_dict(cfg):
        cfg.model.action_encoder.input_dim = cfg.data.dataset.frameskip * dataset.get_dim("action")

    eval_start_rows = None
    if eval_starts_json:
        eval_payload = json.loads(eval_starts_json.read_text(encoding="utf-8"))
        eval_episodes = [int(v) for v in eval_payload["episodes_idx"]]
        eval_starts = [int(v) for v in eval_payload["start_steps"]]
        if len(eval_episodes) != len(eval_starts):
            raise ValueError("eval starts JSON has mismatched episodes_idx/start_steps")
        clip_lookup = {(int(ep), int(start)): i for i, (ep, start) in enumerate(dataset.clip_indices)}
        chosen = []
        eval_start_rows = []
        history_end_offset = (args.history_size - 1) * args.frameskip
        for eval_i, (ep_idx, start_step) in enumerate(zip(eval_episodes, eval_starts)):
            local_start = max(0, start_step - history_end_offset)
            key = (ep_idx, local_start)
            if key not in clip_lookup:
                raise KeyError(f"Could not find clip for episode={ep_idx}, local_start={local_start}")
            chosen.append(clip_lookup[key])
            eval_start_rows.append(
                {
                    "eval_index": eval_i,
                    "eval_episode_idx": ep_idx,
                    "eval_start_step": start_step,
                    "geometry_clip_local_start": local_start,
                    "geometry_history_clamped": bool(start_step < history_end_offset),
                }
            )
        chosen_dataset_indices = np.asarray(chosen, dtype=np.int64)
    else:
        rng = np.random.default_rng(args.seed)
        chosen_dataset_indices = rng.choice(len(dataset), size=args.num_samples, replace=False)
    subset = torch.utils.data.Subset(dataset, chosen_dataset_indices.tolist())
    loader = torch.utils.data.DataLoader(subset, batch_size=args.batch_size, shuffle=False, drop_last=False)

    h5_path = cache_dir / "datasets" / "dmc" / "reacher_random.h5"
    h5 = h5py.File(h5_path, "r", libver="latest", swmr=True)
    _action_mean, action_std = action_mean_std(h5)
    action_std_matrix = np.diag(action_std)
    env = build_env()
    env.reset(seed=args.seed)
    model = swm.wm.utils.load_pretrained(args.policy, cache_dir=cache_dir).to(device).eval()
    model.requires_grad_(False)
    if args.device == "cpu":
        torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))

    env_ratios = []
    model_ratios = []
    env_kappas = []
    model_kappas = []
    kappa_ratios = []
    eccentricity_errors = []
    angles = []
    normalized_diffs = []
    env_grams = []
    model_grams = []
    per_state = []
    processed = 0

    for batch_no, batch in enumerate(loader):
        batch = move_to_device(batch, device)
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)
        with torch.no_grad():
            info = model.encode(batch)
        emb = info["emb"][:, : args.history_size]
        action = batch["action"][:, : args.history_size].float()
        model_batch_grams = model_metrics_for_batch(model, emb, action, args.frameskip, args.eps)

        start = batch_no * args.batch_size
        dataset_indices = chosen_dataset_indices[start : start + len(model_batch_grams)]
        for local_i, (dataset_idx, model_g) in enumerate(zip(dataset_indices, model_batch_grams)):
            eval_meta = eval_start_rows[start + local_i] if eval_start_rows is not None else {}
            ep_idx, local_start = dataset.clip_indices[int(dataset_idx)]
            raw_row = int(dataset.offsets[ep_idx] + local_start + (args.history_size - 1) * args.frameskip)
            qpos = h5["qpos"][raw_row]
            qvel = h5["qvel"][raw_row]
            base_action = h5["action"][raw_row]
            env_g_raw = env_metric(env, qpos, qvel, base_action, args.frameskip, args.fd_step)
            env_g = (
                action_std_matrix @ env_g_raw @ action_std_matrix
                if args.env_metric_coordinate == "normalized_action"
                else env_g_raw
            )

            env_ratio, _, env_dom = ratio_and_dom(env_g, args.eps)
            model_ratio, _, model_dom = ratio_and_dom(model_g, args.eps)
            env_n = norm_trace(env_g, args.eps)
            model_n = norm_trace(model_g, args.eps)
            env_trace = float(np.trace(env_g))
            model_trace = float(np.trace(model_g))

            env_ratios.append(env_ratio)
            model_ratios.append(model_ratio)
            env_kappa = condition_from_ratio(env_ratio, args.eps)
            model_kappa = condition_from_ratio(model_ratio, args.eps)
            kappa_ratio = env_kappa / model_kappa
            eccentricity_error = abs(np.log(model_kappa) - np.log(env_kappa))
            env_kappas.append(env_kappa)
            model_kappas.append(model_kappa)
            kappa_ratios.append(kappa_ratio)
            eccentricity_errors.append(eccentricity_error)
            angles.append(angle_degrees(env_dom, model_dom))
            normalized_diffs.append(float(np.linalg.norm(env_n - model_n, ord="fro")))
            env_grams.append(env_g)
            model_grams.append(model_g)
            if args.save_per_state:
                per_state.append(
                    {
                        **eval_meta,
                        "dataset_idx": int(dataset_idx),
                        "raw_row": int(raw_row),
                        "env_ratio_lambda_min_over_max": float(env_ratio),
                        "model_ratio_lambda_min_over_max": float(model_ratio),
                        "kappa_env": float(env_kappa),
                        "kappa_model": float(model_kappa),
                        "kappa_env_over_model": float(kappa_ratio),
                        "eccentricity_error_abs_log_kappa": float(eccentricity_error),
                        "dominant_eigvec_angle_env_model_degrees": float(angles[-1]),
                        "normalized_frobenius_env_minus_model": float(normalized_diffs[-1]),
                        "env_metric_trace": env_trace,
                        "model_metric_trace": model_trace,
                        "scale_error_abs_log_trace_env_over_model": float(abs(np.log((env_trace + args.eps) / (model_trace + args.eps)))),
                        "env_metric_trace_normalized": env_n.tolist(),
                        "model_metric_trace_normalized": model_n.tolist(),
                    }
                )
            processed += 1

    h5.close()
    env.close()

    env_ratios_arr = np.asarray(env_ratios, dtype=np.float64)
    model_ratios_arr = np.asarray(model_ratios, dtype=np.float64)
    kappa_ratios_arr = np.asarray(kappa_ratios, dtype=np.float64)
    env_kappas_arr = np.asarray(env_kappas, dtype=np.float64)
    model_kappas_arr = np.asarray(model_kappas, dtype=np.float64)
    corr = float(np.corrcoef(env_ratios_arr, model_ratios_arr)[0, 1])
    kappa_corr = float(np.corrcoef(env_kappas_arr, model_kappas_arr)[0, 1])
    summary = {
        "ok": True,
        "policy": args.policy,
        "dataset_name": dataset_name,
        "num_samples": processed,
        "eval_starts_json": str(eval_starts_json) if eval_starts_json else None,
        "seed": args.seed,
        "device": args.device,
        "frameskip": args.frameskip,
        "fd_step": args.fd_step,
        "env_metric_coordinate": args.env_metric_coordinate,
        "action_std_used_for_env_metric": action_std.tolist(),
        "env_ratio_lambda_min_over_max": summarize(env_ratios),
        "model_ratio_lambda_min_over_max": summarize(model_ratios),
        "kappa_env_lambda_max_over_min": summarize(env_kappas),
        "kappa_model_lambda_max_over_min": summarize(model_kappas),
        "kappa_env_over_model": {
            **summarize(kappa_ratios),
            "fraction_gt_1": float((kappa_ratios_arr > 1.0).mean()),
            "fraction_gt_2": float((kappa_ratios_arr > 2.0).mean()),
            "fraction_gt_4": float((kappa_ratios_arr > 4.0).mean()),
            "fraction_lt_1": float((kappa_ratios_arr < 1.0).mean()),
        },
        "eccentricity_error_abs_log_kappa": summarize(eccentricity_errors),
        "dominant_eigvec_angle_env_model_degrees": {
            **summarize(angles),
            "fraction_le_10": float((np.asarray(angles) <= 10.0).mean()),
            "fraction_le_20": float((np.asarray(angles) <= 20.0).mean()),
            "fraction_le_30": float((np.asarray(angles) <= 30.0).mean()),
            "fraction_ge_60": float((np.asarray(angles) >= 60.0).mean()),
        },
        "normalized_frobenius_env_minus_model": summarize(normalized_diffs),
        "anisotropy_ratio_correlation": {
            "pearson_env_vs_model_min_over_max": corr,
            "pearson_env_vs_model_kappa": kappa_corr,
        },
    }
    if args.save_per_state:
        summary["per_state"] = per_state
    printed_summary = {key: value for key, value in summary.items() if key != "per_state"}
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(json.dumps(printed_summary, indent=2, sort_keys=True))
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(text + "\n", encoding="utf-8")
    if plot_png:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plot_png.parent.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6.5, 5.5), dpi=160)
        ax.scatter(env_kappas_arr, model_kappas_arr, s=14, alpha=0.5, edgecolors="none")
        lim = max(float(np.percentile(env_kappas_arr, 99)), float(np.percentile(model_kappas_arr, 99)), 1.0)
        lim *= 1.1
        ax.plot([1.0, lim], [1.0, lim], color="black", linewidth=1.0, linestyle="--", label="equal kappa")
        ax.set_xlim(1.0, lim)
        ax.set_ylim(1.0, lim)
        ax.set_xlabel("Environment kappa")
        ax.set_ylabel("Model kappa")
        ax.set_title("Reacher local action-geometry eccentricity")
        ax.legend(frameon=False)
        fig.tight_layout()
        fig.savefig(plot_png)
        plt.close(fig)


if __name__ == "__main__":
    main()
