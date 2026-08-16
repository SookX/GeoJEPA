#!/usr/bin/env python
"""Measure local action-effect / transition geometry for a LeWM Reacher model."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lewm-dir", default=os.environ.get("LEWM_DIR", "le-wm"))
    parser.add_argument("--cache-dir", default=os.environ.get("STABLEWM_HOME"))
    parser.add_argument("--policy", required=True)
    parser.add_argument("--dataset-name", default="dmc/reacher_random.h5")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--num-preds", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--target", choices=["effect", "transition"], default="effect")
    parser.add_argument("--action-basis", choices=["full", "repeat2", "first2"], default="full")
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output-json", default=None)
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


def main() -> None:
    args = parse_args()
    output_json = Path(args.output_json).resolve() if args.output_json else None
    lewm_dir = Path(args.lewm_dir).resolve()
    if not lewm_dir.exists():
        raise SystemExit(f"Missing LeWM directory: {lewm_dir}")

    sys.path.insert(0, str(lewm_dir))
    os.chdir(lewm_dir)

    import hydra
    import numpy as np
    import stable_pretraining as spt
    import stable_worldmodel as swm
    import torch
    from omegaconf import OmegaConf, open_dict

    from utils import get_column_normalizer, get_img_preprocessor

    if args.target == "transition":
        if hasattr(torch.backends, "mha"):
            torch.backends.mha.set_fastpath_enabled(False)
        if hasattr(torch.backends, "cuda"):
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
            torch.backends.cuda.enable_math_sdp(True)

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false.")

    cache_dir = args.cache_dir or os.environ.get("STABLEWM_HOME") or os.environ.get("LOCAL_DATASET_DIR")
    if cache_dir:
        os.environ.setdefault("STABLEWM_HOME", cache_dir)
        os.environ.setdefault("LOCAL_DATASET_DIR", cache_dir)

    overrides = [
        "data=dmc",
        f"data.dataset.name={args.dataset_name}",
        f"loader.batch_size={args.batch_size}",
        f"loader.num_workers={args.num_workers}",
        "loader.persistent_workers=false",
        f"img_size={args.img_size}",
        f"history_size={args.history_size}",
        f"num_preds={args.num_preds}",
        "model=lewm",
    ]
    with hydra.initialize_config_dir(version_base=None, config_dir=str(lewm_dir / "config" / "train")):
        cfg = hydra.compose(config_name="lewm", overrides=overrides)

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

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=args.device == "cuda",
    )

    device = torch.device(args.device)
    model = swm.wm.utils.load_pretrained(args.policy, cache_dir=cache_dir).to(device).eval()
    model.requires_grad_(False)

    lambda_min = []
    lambda_max = []
    ratio_min_over_max = []
    condition = []
    trace_per_dim = []
    aniso = []
    scale_loss_at_one = []
    singular_values = []
    normalized_grams = []
    processed = 0

    def make_full_action(base_action, template_action):
        if args.action_basis == "full":
            return base_action
        full_dim = template_action.size(-1)
        if args.action_basis == "first2":
            full = template_action.clone()
            full[..., :2] = base_action
            return full
        if full_dim % 2 != 0:
            raise ValueError(f"repeat2 basis expects even full action dim, got {full_dim}")
        return base_action.unsqueeze(-2).expand(*base_action.shape[:-1], full_dim // 2, 2).reshape_as(template_action)

    def action_condition(state_window, full_action):
        if hasattr(model, "action_condition"):
            return model.action_condition(state_window, full_action)
        return model.action_encoder(full_action)

    for batch in loader:
        batch = move_to_device(batch, device)
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)
        with torch.no_grad():
            info = model.encode(batch)
        emb = info["emb"][:, : args.history_size]
        action = batch["action"][:, : args.history_size].float()

        for b in range(emb.size(0)):
            if processed >= args.max_samples:
                break
            state_window = emb[b : b + 1].detach()
            action_window = action[b : b + 1].detach()
            if args.action_basis == "full":
                base = action_window[:, -1:, :].clone()
            else:
                reshaped = action_window[:, -1:, :].reshape(1, 1, -1, 2)
                base = reshaped.mean(dim=-2) if args.action_basis == "repeat2" else action_window[:, -1:, :2].clone()
            base = base.detach().requires_grad_(True)
            action_dim = base.size(-1)

            cols = []
            for i in range(action_dim):
                tangent = torch.zeros_like(base)
                tangent[..., i] = 1.0

                def fn(base_action):
                    full_action = action_window.clone()
                    full_action[:, -1:, :] = make_full_action(base_action, action_window[:, -1:, :])
                    effect = action_condition(state_window, full_action)
                    if args.target == "effect":
                        return effect[:, -1, :]
                    return model.predict(state_window, effect)[:, -1, :]

                _, j_col = torch.autograd.functional.jvp(
                    fn,
                    (base,),
                    (tangent,),
                    create_graph=False,
                    strict=False,
                )
                cols.append(j_col.reshape(-1).float())

            jac = torch.stack(cols, dim=1)
            gram = jac.T @ jac
            eig = torch.linalg.eigvalsh(gram).clamp_min(0)
            sv = torch.sqrt(eig)
            lam_min = eig[0]
            lam_max = eig[-1]
            tr = torch.trace(gram)
            mean_scale = tr / action_dim
            residual = gram - mean_scale * torch.eye(action_dim, device=device, dtype=gram.dtype)
            lambda_min.append(float(lam_min.detach().cpu()))
            lambda_max.append(float(lam_max.detach().cpu()))
            ratio_min_over_max.append(float(((lam_min + args.eps) / (lam_max + args.eps)).detach().cpu()))
            condition.append(float(((lam_max + args.eps) / (lam_min + args.eps)).detach().cpu()))
            trace_per_dim.append(float(mean_scale.detach().cpu()))
            aniso.append(float(residual.pow(2).sum().detach().cpu()))
            scale_loss_at_one.append(float((mean_scale - 1.0).pow(2).detach().cpu()))
            singular_values.extend([float(x) for x in sv.detach().cpu()])
            normalized_grams.append((gram / (tr + args.eps)).detach().float().cpu().numpy())
            processed += 1
        if processed >= args.max_samples:
            break

    min_arr = np.asarray(lambda_min, dtype=np.float64)
    max_arr = np.asarray(lambda_max, dtype=np.float64)
    ratio_arr = np.asarray(ratio_min_over_max, dtype=np.float64)
    cond_arr = np.asarray(condition, dtype=np.float64)
    trace_arr = np.asarray(trace_per_dim, dtype=np.float64)
    aniso_arr = np.asarray(aniso, dtype=np.float64)
    scale_arr = np.asarray(scale_loss_at_one, dtype=np.float64)
    sv_arr = np.asarray(singular_values, dtype=np.float64)
    norm_grams_arr = np.stack(normalized_grams, axis=0).astype(np.float64)
    global_metric = norm_grams_arr.mean(axis=0)
    global_eigvals, global_eigvecs = np.linalg.eigh(global_metric)
    global_dom = global_eigvecs[:, -1]
    frob_dist = np.linalg.norm(norm_grams_arr - global_metric[None, :, :], axis=(1, 2))
    angles_deg = []
    alignment_abs_cos = []
    for local_metric in norm_grams_arr:
        _, local_eigvecs = np.linalg.eigh(local_metric)
        local_dom = local_eigvecs[:, -1]
        abs_cos = float(abs(np.dot(local_dom, global_dom)))
        abs_cos = min(1.0, max(0.0, abs_cos))
        alignment_abs_cos.append(abs_cos)
        angles_deg.append(float(np.degrees(np.arccos(abs_cos))))

    summary = {
        "ok": True,
        "policy": args.policy,
        "dataset_name": dataset_name,
        "target": args.target,
        "action_basis": args.action_basis,
        "frameskip": args.frameskip,
        "max_samples": args.max_samples,
        "num_samples": processed,
        "action_dim": int(args.action_basis == "full" and action.size(-1) or 2),
        "lambda_min": {
            "mean": float(min_arr.mean()),
            "p10": percentile(min_arr, 10),
            "p50": percentile(min_arr, 50),
            "p90": percentile(min_arr, 90),
        },
        "lambda_max": {
            "mean": float(max_arr.mean()),
            "p10": percentile(max_arr, 10),
            "p50": percentile(max_arr, 50),
            "p90": percentile(max_arr, 90),
        },
        "ratio_min_over_max": {
            "mean": float(ratio_arr.mean()),
            "p10": percentile(ratio_arr, 10),
            "p50": percentile(ratio_arr, 50),
            "p90": percentile(ratio_arr, 90),
            "fraction_below_0.5": float((ratio_arr < 0.5).mean()),
            "fraction_below_0.25": float((ratio_arr < 0.25).mean()),
            "fraction_below_0.1": float((ratio_arr < 0.1).mean()),
        },
        "condition_lambda_max_over_min": {
            "mean": float(cond_arr.mean()),
            "p50": percentile(cond_arr, 50),
            "p75": percentile(cond_arr, 75),
            "p90": percentile(cond_arr, 90),
            "p95": percentile(cond_arr, 95),
            "max": float(cond_arr.max()),
            "fraction_above_2": float((cond_arr > 2.0).mean()),
            "fraction_above_4": float((cond_arr > 4.0).mean()),
            "fraction_above_10": float((cond_arr > 10.0).mean()),
        },
        "normalized_global_metric": {
            "matrix": global_metric.tolist(),
            "eigvals_ascending": global_eigvals.tolist(),
            "eig_ratio_large_over_small": float((global_eigvals[-1] + args.eps) / (global_eigvals[0] + args.eps)),
            "eig_ratio_small_over_large": float((global_eigvals[0] + args.eps) / (global_eigvals[-1] + args.eps)),
            "dominant_eigvec": global_dom.tolist(),
        },
        "normalized_frobenius_distance_to_global": summarize(frob_dist),
        "dominant_eigvec_angle_to_global_degrees": {
            **summarize(angles_deg),
            "fraction_le_10": float((np.asarray(angles_deg) <= 10.0).mean()),
            "fraction_le_20": float((np.asarray(angles_deg) <= 20.0).mean()),
            "fraction_le_30": float((np.asarray(angles_deg) <= 30.0).mean()),
            "fraction_ge_60": float((np.asarray(angles_deg) >= 60.0).mean()),
        },
        "dominant_eigvec_abs_cos_to_global": summarize(alignment_abs_cos),
        "trace_per_dim": {
            "mean": float(trace_arr.mean()),
            "std": float(trace_arr.std()),
            "min": float(trace_arr.min()),
            "p10": percentile(trace_arr, 10),
            "p50": percentile(trace_arr, 50),
            "p90": percentile(trace_arr, 90),
            "max": float(trace_arr.max()),
        },
        "aniso_loss": {
            "mean": float(aniso_arr.mean()),
            "p50": percentile(aniso_arr, 50),
            "p90": percentile(aniso_arr, 90),
        },
        "scale_loss_at_alpha1": {
            "mean": float(scale_arr.mean()),
            "p50": percentile(scale_arr, 50),
            "p90": percentile(scale_arr, 90),
        },
        "singular_values": {
            "mean": float(sv_arr.mean()),
            "std": float(sv_arr.std()),
            "min": float(sv_arr.min()),
            "p10": percentile(sv_arr, 10),
            "p50": percentile(sv_arr, 50),
            "p90": percentile(sv_arr, 90),
            "max": float(sv_arr.max()),
        },
    }
    text = json.dumps(summary, indent=2, sort_keys=True)
    print(text)
    if args.output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
