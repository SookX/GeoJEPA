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


def main() -> None:
    args = parse_args()
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
        "model=lewm_b1_effect",
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

    trace_per_dim = []
    aniso = []
    scale_loss_at_one = []
    singular_values = []
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
        return base_action.repeat_interleave(full_dim // 2, dim=-1)

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
                    effect = model.action_condition(state_window, full_action)
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
            tr = torch.trace(gram)
            mean_scale = tr / action_dim
            residual = gram - mean_scale * torch.eye(action_dim, device=device, dtype=gram.dtype)
            trace_per_dim.append(float(mean_scale.detach().cpu()))
            aniso.append(float(residual.pow(2).sum().detach().cpu()))
            scale_loss_at_one.append(float((mean_scale - 1.0).pow(2).detach().cpu()))
            singular_values.extend([float(x) for x in sv.detach().cpu()])
            processed += 1
        if processed >= args.max_samples:
            break

    trace_arr = np.asarray(trace_per_dim, dtype=np.float64)
    aniso_arr = np.asarray(aniso, dtype=np.float64)
    scale_arr = np.asarray(scale_loss_at_one, dtype=np.float64)
    sv_arr = np.asarray(singular_values, dtype=np.float64)
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
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
