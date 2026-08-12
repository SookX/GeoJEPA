#!/usr/bin/env python
"""Run a tiny LeWM optimization smoke test and report gradient health."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lewm-dir", default=os.environ.get("LEWM_DIR", "le-wm"))
    parser.add_argument("--data", default="dmc")
    parser.add_argument("--dataset-name", default="dmc/reacher_random.h5")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--img-size", type=int, default=112)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--num-preds", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--model-config", default=os.environ.get("MODEL_CONFIG", "lewm"))
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--precision", default="bf16", choices=["bf16", "fp32"])
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--sigreg-weight", type=float, default=0.09)
    parser.add_argument("--value-weight", type=float, default=float(os.environ.get("VALUE_WEIGHT", "0") or 0))
    parser.add_argument("--value-gamma", type=float, default=float(os.environ.get("VALUE_GAMMA", "0.99") or 0.99))
    parser.add_argument("--value-horizon", default=os.environ.get("VALUE_HORIZON", ""))
    parser.add_argument("--planning-value-weight", default=os.environ.get("PLANNING_VALUE_WEIGHT", ""))
    parser.add_argument("--planning-value-gamma", default=os.environ.get("PLANNING_VALUE_GAMMA", ""))
    parser.add_argument("--geo-aniso-weight", default=os.environ.get("GEO_ANISO_WEIGHT", ""))
    parser.add_argument("--geo-scale-weight", default=os.environ.get("GEO_SCALE_WEIGHT", ""))
    parser.add_argument("--geo-resolution", default=os.environ.get("GEO_RESOLUTION", ""))
    parser.add_argument("--geo-weight", default=os.environ.get("GEO_WEIGHT", ""))
    parser.add_argument("--geo-alpha-weight", default=os.environ.get("GEO_ALPHA_WEIGHT", ""))
    parser.add_argument("--geo-alpha-tau", default=os.environ.get("GEO_ALPHA_TAU", ""))
    parser.add_argument("--geo-alpha-min", default=os.environ.get("GEO_ALPHA_MIN", ""))
    parser.add_argument("--geo-alpha0", default=os.environ.get("GEO_ALPHA0", ""))
    parser.add_argument("--geo-teacher-weight", default=os.environ.get("GEO_TEACHER_WEIGHT", ""))
    parser.add_argument("--geo-teacher-model-path", default=os.environ.get("GEO_TEACHER_MODEL_PATH", ""))
    parser.add_argument("--geo-k", default=os.environ.get("GEO_K", ""))
    parser.add_argument("--geo-max-points", default=os.environ.get("GEO_MAX_POINTS", ""))
    parser.add_argument("--geo-target", default=os.environ.get("GEO_TARGET", ""))
    parser.add_argument("--geo-action-basis", default=os.environ.get("GEO_ACTION_BASIS", ""))
    parser.add_argument("--geo-frameskip", default=os.environ.get("GEO_FRAMESKIP", ""))
    parser.add_argument("--init-model-path", default=os.environ.get("INIT_MODEL_PATH", ""))
    parser.add_argument("--dataset-num-steps", default=os.environ.get("DATASET_NUM_STEPS", ""))
    parser.add_argument("--dataset-keys-to-load", default=os.environ.get("DATASET_KEYS_TO_LOAD", ""))
    parser.add_argument("--dataset-keys-to-cache", default=os.environ.get("DATASET_KEYS_TO_CACHE", ""))
    parser.add_argument("--skip-normalizers", action="store_true")
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def move_to_device(value: Any, device):
    import torch

    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def grad_l2_norm(parameters) -> tuple[float, int, int]:
    import torch

    total_sq = 0.0
    grad_params = 0
    bad_params = 0
    for param in parameters:
        if param.grad is None:
            continue
        grad_params += 1
        grad = param.grad.detach()
        if not torch.isfinite(grad).all():
            bad_params += 1
        total_sq += float(grad.float().pow(2).sum().cpu())
    return math.sqrt(total_sq), grad_params, bad_params


def main() -> None:
    args = parse_args()
    lewm_dir = Path(args.lewm_dir).resolve()
    if not lewm_dir.exists():
        raise SystemExit(f"Missing LeWM directory: {lewm_dir}")

    sys.path.insert(0, str(lewm_dir))
    os.chdir(lewm_dir)

    import hydra
    import stable_pretraining as spt
    import stable_worldmodel as swm
    import torch
    from omegaconf import OmegaConf, open_dict

    from module import SIGReg
    from train import lejepa_forward
    from utils import get_column_normalizer, get_img_preprocessor

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false.")

    cache_dir = args.cache_dir or os.environ.get("STABLEWM_HOME") or os.environ.get("LOCAL_DATASET_DIR")
    if cache_dir:
        os.environ.setdefault("STABLEWM_HOME", cache_dir)
        os.environ.setdefault("LOCAL_DATASET_DIR", cache_dir)

    overrides = [
        f"data={args.data}",
        f"data.dataset.name={args.dataset_name}",
        f"loader.batch_size={args.batch_size}",
        f"loader.num_workers={args.num_workers}",
        "loader.persistent_workers=false",
        f"img_size={args.img_size}",
        f"history_size={args.history_size}",
        f"num_preds={args.num_preds}",
        f"loss.sigreg.weight={args.sigreg_weight}",
        f"model={args.model_config}",
    ]
    if args.value_weight > 0:
        overrides.extend([
            f"+loss.value.weight={args.value_weight}",
            f"+loss.value.gamma={args.value_gamma}",
        ])
        if args.value_horizon:
            overrides.append(f"+loss.value.horizon={args.value_horizon}")
    if args.planning_value_weight:
        overrides.append(f"model.planning_value_weight={args.planning_value_weight}")
    if args.planning_value_gamma:
        overrides.append(f"model.planning_value_gamma={args.planning_value_gamma}")
    if args.geo_aniso_weight:
        overrides.append(f"+loss.geo.aniso_weight={args.geo_aniso_weight}")
    if args.geo_scale_weight:
        overrides.append(f"+loss.geo.scale_weight={args.geo_scale_weight}")
    if args.geo_resolution:
        overrides.append(f"+loss.geo.resolution={args.geo_resolution}")
    if args.geo_weight:
        overrides.append(f"+loss.geo.weight={args.geo_weight}")
    if args.geo_alpha_weight:
        overrides.append(f"+loss.geo.alpha_weight={args.geo_alpha_weight}")
    if args.geo_alpha_tau:
        overrides.append(f"+loss.geo.alpha_tau={args.geo_alpha_tau}")
    if args.geo_alpha_min:
        overrides.append(f"model.alpha_head.alpha_min={args.geo_alpha_min}")
    if args.geo_alpha0:
        overrides.append(f"+loss.geo.alpha0={args.geo_alpha0}")
    if args.geo_teacher_weight:
        overrides.append(f"+loss.geo.teacher_weight={args.geo_teacher_weight}")
    if args.geo_teacher_model_path:
        overrides.append(f"+loss.geo.teacher_model_path={args.geo_teacher_model_path}")
    if args.geo_k:
        overrides.append(f"+loss.geo.k={args.geo_k}")
    if args.geo_max_points:
        overrides.append(f"+loss.geo.max_points={args.geo_max_points}")
    if args.geo_target:
        overrides.append(f"+loss.geo.target={args.geo_target}")
    if args.geo_action_basis:
        overrides.append(f"+loss.geo.action_basis={args.geo_action_basis}")
    if args.geo_frameskip:
        overrides.append(f"+loss.geo.frameskip={args.geo_frameskip}")
    if args.init_model_path:
        overrides.append(f"+init_model_path={args.init_model_path}")
    if args.dataset_num_steps:
        overrides.append(f"data.dataset.num_steps={args.dataset_num_steps}")
    if args.dataset_keys_to_load:
        overrides.append(f"data.dataset.keys_to_load=[{args.dataset_keys_to_load}]")
    if args.dataset_keys_to_cache:
        overrides.append(f"data.dataset.keys_to_cache=[{args.dataset_keys_to_cache}]")

    with hydra.initialize_config_dir(version_base=None, config_dir=str(lewm_dir / "config" / "train")):
        cfg = hydra.compose(config_name="lewm", overrides=overrides)

    dataset_cfg = OmegaConf.to_container(cfg.data.dataset, resolve=True)
    dataset_name = dataset_cfg.pop("name")
    dataset = swm.data.load_dataset(dataset_name, transform=None, cache_dir=cache_dir, **dataset_cfg)

    transforms = [get_img_preprocessor(source="pixels", target="pixels", img_size=cfg.img_size)]
    if not args.skip_normalizers:
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
    model = hydra.utils.instantiate(cfg.model).to(device)
    if args.init_model_path:
        pretrained = swm.wm.utils.load_pretrained(args.init_model_path, cache_dir=cache_dir).to(device)
        incompatible = model.load_state_dict(pretrained.state_dict(), strict=False)
        print(
            "initialized_from="
            f"{args.init_model_path} missing={list(incompatible.missing_keys)} "
            f"unexpected={list(incompatible.unexpected_keys)}",
            flush=True,
        )
    if args.geo_teacher_model_path:
        geo_teacher = swm.wm.utils.load_pretrained(args.geo_teacher_model_path, cache_dir=cache_dir).to(device)
        geo_teacher.eval().requires_grad_(False)
        object.__setattr__(model, "geo_teacher", geo_teacher)
        print(f"geo_teacher={args.geo_teacher_model_path}", flush=True)

    class SmokeModule:
        def __init__(self, model, sigreg):
            self.model = model
            self.sigreg = sigreg

        def log_dict(self, *args, **kwargs):
            return None

    smoke_module = SmokeModule(model, SIGReg(**cfg.loss.sigreg.kwargs).to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=cfg.optimizer.weight_decay)

    use_amp = args.precision == "bf16" and device.type == "cuda"
    iterator = iter(loader)
    history = []
    model.train()

    for step in range(args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            batch = next(iterator)
        batch = move_to_device(batch, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_amp):
            output = lejepa_forward(smoke_module, batch, "smoke", cfg)
        loss = output["loss"]
        if not torch.isfinite(loss):
            raise SystemExit(f"Non-finite loss at step {step}: {float(loss.detach().cpu())}")

        loss.backward()
        grad_norm, grad_params, bad_params = grad_l2_norm(model.parameters())
        if bad_params or grad_params == 0 or not math.isfinite(grad_norm) or grad_norm <= 0:
            raise SystemExit(
                "Bad gradients: "
                f"grad_norm={grad_norm:.6g}, grad_params={grad_params}, bad_params={bad_params}"
            )
        optimizer.step()

        row = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "pred_loss": float(output["pred_loss"].detach().cpu()),
            "sigreg_loss": float(output["sigreg_loss"].detach().cpu()),
            "grad_norm": grad_norm,
            "grad_params": grad_params,
        }
        for key, value in output.items():
            if key.endswith("_loss") and key not in row and torch.is_tensor(value):
                row[key] = float(value.detach().cpu())
            elif key.startswith("geo_") and torch.is_tensor(value):
                row[key] = float(value.detach().cpu())
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    summary = {
        "ok": True,
        "device": str(device),
        "dataset_name": dataset_name,
        "dataset_len": len(dataset),
        "batch_size": args.batch_size,
        "img_size": args.img_size,
        "steps": args.steps,
        "first_loss": history[0]["loss"],
        "last_loss": history[-1]["loss"],
        "last_grad_norm": history[-1]["grad_norm"],
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.output_json:
        output_path = Path(args.output_json)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps({"summary": summary, "history": history}, indent=2) + "\n")


if __name__ == "__main__":
    main()
