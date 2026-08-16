#!/usr/bin/env python
"""Short paired LeWM fine-tuning run with optional oracle geometry loss."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lewm-dir", default=os.environ.get("LEWM_DIR", "le-wm"))
    parser.add_argument("--cache-dir", default=os.environ.get("STABLEWM_HOME", "stablewm_home"))
    parser.add_argument("--init-policy", default="quentinll/lewm-reacher")
    parser.add_argument("--oracle-json", required=True)
    parser.add_argument("--output-run", required=True)
    parser.add_argument("--dataset-name", default="dmc/reacher_random.h5")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--history-size", type=int, default=3)
    parser.add_argument("--num-preds", type=int, default=1)
    parser.add_argument("--frameskip", type=int, default=5)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pred-batch-size", type=int, default=None)
    parser.add_argument("--train-states", type=int, default=None)
    parser.add_argument("--heldout-states", type=int, default=0)
    parser.add_argument("--full-prediction-batches", action="store_true")
    parser.add_argument("--pred-eval-samples", type=int, default=256)
    parser.add_argument("--eval-steps", default="")
    parser.add_argument("--save-eval-checkpoints", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--sigreg-weight", type=float, default=0.09)
    parser.add_argument("--geom-weight", type=float, default=0.0)
    parser.add_argument("--auto-geom-ratio", type=float, default=None)
    parser.add_argument("--anchor-weight", type=float, default=0.0)
    parser.add_argument("--freeze-batchnorm-stats", action="store_true")
    parser.add_argument(
        "--trainable-scope",
        choices=["all", "adaln_action", "adaln_only", "action_only"],
        default="all",
    )
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--rollout-eval-samples", type=int, default=0)
    parser.add_argument("--rollout-horizons", default="5,10,25")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--output-json", default=None)
    return parser.parse_args()


def move_to_device(value: Any, device):
    import torch

    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    return value


def set_batchnorm_eval(model) -> None:
    import torch

    for module in model.modules():
        if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
            module.eval()


def restore_training_mode(model, was_training: bool) -> None:
    if was_training:
        model.train()
        if getattr(model, "_freeze_batchnorm_stats", False):
            set_batchnorm_eval(model)


class OracleGeometryDataset:
    def __init__(self, dataset, per_state):
        import torch

        self.dataset = dataset
        self.dataset_indices = [int(row["dataset_idx"]) for row in per_state]
        self.targets = [
            torch.as_tensor(row["env_metric_trace_normalized"], dtype=torch.float32)
            for row in per_state
        ]
        self.env_traces = [
            float(row["env_metric_trace"]) if "env_metric_trace" in row else None
            for row in per_state
        ]

    def __len__(self):
        return len(self.dataset_indices)

    def __getitem__(self, idx):
        import torch

        sample = dict(self.dataset[self.dataset_indices[idx]])
        sample["oracle_g_env_norm"] = self.targets[idx]
        if self.env_traces[idx] is not None:
            sample["oracle_g_env_trace"] = torch.as_tensor(self.env_traces[idx], dtype=torch.float32)
        sample["dataset_idx"] = self.dataset_indices[idx]
        return sample


def load_oracle(path: str | Path):
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    per_state = payload.get("per_state")
    if not per_state:
        raise ValueError(f"{path} does not contain per_state oracle targets")
    return payload, per_state


def resolve_init_config(cache_dir: Path, init_policy: str) -> Path:
    local = cache_dir / "checkpoints" / init_policy
    if local.is_dir() and (local / "config.json").exists():
        return local / "config.json"
    if (local.parent / "config.json").exists():
        return local.parent / "config.json"
    raise FileNotFoundError(f"Could not find config.json for {init_policy} under {cache_dir / 'checkpoints'}")


def transition_grams(model, emb, action, frameskip):
    import torch

    state_window = emb.detach()
    action_window = action.detach()
    base = action_window[:, -1:, :].reshape(action_window.size(0), 1, frameskip, 2).mean(dim=-2)
    base = base.detach().requires_grad_(True)
    cols = []

    def make_full_action(base_action):
        return base_action.unsqueeze(-2).expand(*base_action.shape[:-1], frameskip, 2).reshape_as(action_window[:, -1:, :])

    def fn(base_action):
        full_action = action_window.clone()
        full_action[:, -1:, :] = make_full_action(base_action)
        effect = model.action_condition(state_window, full_action) if hasattr(model, "action_condition") else model.action_encoder(full_action)
        return model.predict(state_window, effect)[:, -1, :]

    for i in range(2):
        tangent = torch.zeros_like(base)
        tangent[..., i] = 1.0
        _, j_col = torch.autograd.functional.jvp(
            fn,
            (base,),
            (tangent,),
            create_graph=True,
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
    return grams


def normalized_transition_grams(model, emb, action, frameskip, eps=1e-12):
    grams = transition_grams(model, emb, action, frameskip)
    trace = grams.diagonal(dim1=-2, dim2=-1).sum(dim=-1).clamp_min(eps)
    return grams / trace[:, None, None]


def geometry_summaries(model_g, target_g, eps=1e-12):
    import numpy as np
    import torch

    model_np = model_g.detach().float().cpu().numpy()
    target_np = target_g.detach().float().cpu().numpy()
    d_g = np.linalg.norm(model_np - target_np, axis=(1, 2))
    d_k = []
    model_kappa = []
    env_kappa = []
    for mg, eg in zip(model_np, target_np):
        me = np.clip(np.linalg.eigvalsh(mg), 0.0, None)
        ee = np.clip(np.linalg.eigvalsh(eg), 0.0, None)
        mk = float((me[-1] + eps) / (me[0] + eps))
        ek = float((ee[-1] + eps) / (ee[0] + eps))
        model_kappa.append(mk)
        env_kappa.append(ek)
        d_k.append(math.log(mk) - math.log(ek))
    return {
        "d_g_mean": float(d_g.mean()),
        "d_g_median": float(np.median(d_g)),
        "d_kappa_mean": float(np.mean(d_k)),
        "d_kappa_median": float(np.median(d_k)),
        "eccentricity_error_median": float(np.median(np.abs(d_k))),
        "model_kappa_median": float(np.median(model_kappa)),
        "env_kappa_median": float(np.median(env_kappa)),
    }


def prediction_terms(model, sigreg, batch, cfg):
    import torch

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = model.encode(batch)
    emb = output["emb"]
    act_emb = output["act_emb"]
    ctx_len = int(cfg["history_size"])
    n_preds = int(cfg["num_preds"])
    pred_emb = model.predict(emb[:, :ctx_len], act_emb[:, :ctx_len])
    tgt_emb = emb[:, n_preds:]
    pred_loss = (pred_emb - tgt_emb).pow(2).mean()
    sigreg_loss = (
        sigreg(emb.transpose(0, 1))
        if float(cfg["sigreg_weight"]) > 0
        else torch.zeros((), device=pred_loss.device, dtype=pred_loss.dtype)
    )
    return pred_loss, sigreg_loss


def anchor_term(model, anchor_model, batch, cfg):
    import torch

    if anchor_model is None:
        device = next(model.parameters()).device
        return torch.zeros((), device=device)
    was_training = model.training
    model_batch = dict(batch)
    anchor_batch = dict(batch)
    model_batch["action"] = torch.nan_to_num(model_batch["action"], 0.0)
    anchor_batch["action"] = torch.nan_to_num(anchor_batch["action"], 0.0)
    ctx_len = int(cfg["history_size"])
    with torch.no_grad():
        anchor_out = anchor_model.encode(anchor_batch)
        anchor_pred = anchor_model.predict(
            anchor_out["emb"][:, :ctx_len],
            anchor_out["act_emb"][:, :ctx_len],
        )
    try:
        model.eval()
        model_out = model.encode(model_batch)
        model_pred = model.predict(
            model_out["emb"][:, :ctx_len],
            model_out["act_emb"][:, :ctx_len],
        )
        return (model_pred - anchor_pred.detach()).pow(2).mean()
    finally:
        restore_training_mode(model, was_training)


def geometry_term(model, batch, cfg, frameskip):
    import torch

    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = model.encode(batch)
    emb = output["emb"]
    ctx_len = int(cfg["history_size"])
    model_g = normalized_transition_grams(model, emb[:, :ctx_len], batch["action"][:, :ctx_len].float(), frameskip)
    target_g = batch["oracle_g_env_norm"].to(model_g.device, dtype=model_g.dtype)
    return (model_g - target_g.detach()).pow(2).sum(dim=(-2, -1)).mean()


def finite_metrics(model, sigreg, batch, cfg, geom_weight, frameskip):
    import torch

    pred_loss, sigreg_loss = prediction_terms(model, sigreg, batch, cfg)
    geom_loss = torch.zeros((), device=pred_loss.device, dtype=pred_loss.dtype)
    if geom_weight > 0:
        geom_loss = geometry_term(model, batch, cfg, frameskip)
    loss = pred_loss + float(cfg["sigreg_weight"]) * sigreg_loss + geom_weight * geom_loss
    anchor_loss = torch.zeros((), device=pred_loss.device, dtype=pred_loss.dtype)
    return loss, pred_loss, sigreg_loss, geom_loss, anchor_loss


def two_stream_metrics(
    model,
    sigreg,
    pred_batch,
    geom_batch,
    cfg,
    geom_weight,
    frameskip,
    anchor_model=None,
    anchor_weight=0.0,
):
    import torch

    pred_loss, sigreg_loss = prediction_terms(model, sigreg, pred_batch, cfg)
    geom_loss = torch.zeros((), device=pred_loss.device, dtype=pred_loss.dtype)
    if geom_weight > 0:
        geom_loss = geometry_term(model, geom_batch, cfg, frameskip)
    anchor_loss = torch.zeros((), device=pred_loss.device, dtype=pred_loss.dtype)
    if anchor_weight > 0:
        anchor_loss = anchor_term(model, anchor_model, pred_batch, cfg)
    loss = (
        pred_loss
        + float(cfg["sigreg_weight"]) * sigreg_loss
        + geom_weight * geom_loss
        + float(anchor_weight) * anchor_loss
    )
    return loss, pred_loss, sigreg_loss, geom_loss, anchor_loss


def grad_l2_norm(parameters):
    import math
    import torch

    total = 0.0
    for param in parameters:
        if param.grad is not None:
            total += float(param.grad.detach().float().pow(2).sum().cpu())
    return math.sqrt(total)


def choose_geom_weight(model, sigreg, pred_batch, geom_batch, train_cfg, frameskip, target_ratio, params):
    import torch

    model.zero_grad(set_to_none=True)
    pred_loss, _sigreg_loss = prediction_terms(model, sigreg, pred_batch, train_cfg)
    pred_loss.backward()
    pred_norm = grad_l2_norm(params)

    model.zero_grad(set_to_none=True)
    geom_loss = geometry_term(model, geom_batch, train_cfg, frameskip)
    geom_loss.backward()
    geom_norm = grad_l2_norm(params)
    model.zero_grad(set_to_none=True)

    geom_weight = float(target_ratio) * pred_norm / geom_norm if geom_norm > 0 else 0.0
    return geom_weight, {
        "pred_grad_norm": pred_norm,
        "geom_grad_norm_unweighted": geom_norm,
        "target_weighted_geom_over_pred": float(target_ratio),
        "chosen_geom_weight": geom_weight,
        "check_weighted_geom_over_pred": (geom_weight * geom_norm / pred_norm) if pred_norm > 0 else None,
        "probe_pred_loss": float(pred_loss.detach().cpu()),
        "probe_geom_loss": float(geom_loss.detach().cpu()),
    }


def evaluate_oracle_set(model, sigreg, loader, train_cfg, frameskip, device):
    import numpy as np
    import torch

    was_training = model.training
    model.eval()
    pred_losses = []
    d_g_values = []
    d_k_values = []
    model_kappa_values = []
    env_kappa_values = []
    scale_errors = []
    model_traces = []
    env_traces = []
    for batch in loader:
        batch = move_to_device(batch, device)
        batch["action"] = torch.nan_to_num(batch["action"], 0.0)
        with torch.no_grad():
            output = model.encode(batch)
            emb = output["emb"]
            act_emb = output["act_emb"]
            ctx_len = int(train_cfg["history_size"])
            n_preds = int(train_cfg["num_preds"])
            pred_emb = model.predict(emb[:, :ctx_len], act_emb[:, :ctx_len])
            tgt_emb = emb[:, n_preds:]
            pred_losses.append(float((pred_emb - tgt_emb).pow(2).mean().cpu()))
        with torch.enable_grad():
            raw_model_g = transition_grams(model, emb[:, :ctx_len], batch["action"][:, :ctx_len].float(), frameskip)
            model_trace = raw_model_g.diagonal(dim1=-2, dim2=-1).sum(dim=-1).detach().float().cpu().numpy()
            model_g = raw_model_g / raw_model_g.diagonal(dim1=-2, dim2=-1).sum(dim=-1).clamp_min(1e-12)[:, None, None]
            target_g = batch["oracle_g_env_norm"].to(model_g.device, dtype=model_g.dtype)
            model_np = model_g.detach().float().cpu().numpy()
            target_np = target_g.detach().float().cpu().numpy()
            d_g_values.extend(np.linalg.norm(model_np - target_np, axis=(1, 2)).tolist())
            model_traces.extend(model_trace.tolist())
            if "oracle_g_env_trace" in batch:
                env_trace = batch["oracle_g_env_trace"].detach().float().cpu().numpy()
                env_traces.extend(env_trace.tolist())
                scale_errors.extend(np.abs(np.log((env_trace + 1e-12) / (model_trace + 1e-12))).tolist())
            for mg, eg in zip(model_np, target_np):
                me = np.clip(np.linalg.eigvalsh(mg), 0.0, None)
                ee = np.clip(np.linalg.eigvalsh(eg), 0.0, None)
                mk = float((me[-1] + 1e-12) / (me[0] + 1e-12))
                ek = float((ee[-1] + 1e-12) / (ee[0] + 1e-12))
                model_kappa_values.append(mk)
                env_kappa_values.append(ek)
                d_k_values.append(math.log(mk) - math.log(ek))
    restore_training_mode(model, was_training)
    return {
        "pred_loss_mean": float(np.mean(pred_losses)),
        "d_g_mean": float(np.mean(d_g_values)),
        "d_g_median": float(np.median(d_g_values)),
        "signed_log_distortion_mean": float(np.mean(d_k_values)),
        "signed_log_distortion_median": float(np.median(d_k_values)),
        "d_kappa_mean": float(np.mean(d_k_values)),
        "d_kappa_median": float(np.median(d_k_values)),
        "eccentricity_error_median": float(np.median(np.abs(d_k_values))),
        "model_kappa_median": float(np.median(model_kappa_values)),
        "env_kappa_median": float(np.median(env_kappa_values)),
        "model_trace_median": float(np.median(model_traces)) if model_traces else None,
        "env_trace_median": float(np.median(env_traces)) if env_traces else None,
        "scale_error_abs_log_trace_median": float(np.median(scale_errors)) if scale_errors else None,
        "scale_error_abs_log_trace_mean": float(np.mean(scale_errors)) if scale_errors else None,
    }


def evaluate_prediction_set(model, sigreg, loader, train_cfg, device):
    import numpy as np
    import torch

    was_training = model.training
    model.eval()
    pred_losses = []
    sigreg_losses = []
    for batch in loader:
        batch = move_to_device(batch, device)
        with torch.no_grad():
            pred_loss, sigreg_loss = prediction_terms(model, sigreg, batch, train_cfg)
        pred_losses.append(float(pred_loss.cpu()))
        sigreg_losses.append(float(sigreg_loss.cpu()))
    restore_training_mode(model, was_training)
    return {
        "full_pred_loss_mean": float(np.mean(pred_losses)),
        "full_sigreg_loss_mean": float(np.mean(sigreg_losses)),
    }


def parse_horizons(value: str, frameskip: int) -> list[int]:
    horizons = []
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        horizon = int(item)
        if horizon <= 0 or horizon % frameskip != 0:
            raise ValueError(f"rollout horizon must be a positive multiple of frameskip={frameskip}: {horizon}")
        horizons.append(horizon)
    return sorted(set(horizons))


def rollout_errors_for_batch(model, batch, train_cfg, frameskip, horizons):
    import torch

    batch = dict(batch)
    batch["action"] = torch.nan_to_num(batch["action"], 0.0)
    output = model.encode(batch)
    emb = output["emb"]
    action = batch["action"].float()
    ctx_len = int(train_cfg["history_size"])
    max_blocks = max(horizons) // frameskip
    if emb.shape[1] < ctx_len + max_blocks:
        raise ValueError(f"rollout batch has {emb.shape[1]} frames, need {ctx_len + max_blocks}")
    rollout_emb = emb[:, :ctx_len]
    errors = {}
    horizon_set = set(horizons)
    for block in range(1, max_blocks + 1):
        start = block - 1
        emb_window = rollout_emb[:, -ctx_len:]
        action_window = action[:, start : start + ctx_len]
        pred_next = model.predict(emb_window, model.action_encoder(action_window))[:, -1:]
        rollout_emb = torch.cat([rollout_emb, pred_next], dim=1)
        horizon = block * frameskip
        if horizon in horizon_set:
            target = emb[:, ctx_len + block - 1 : ctx_len + block]
            errors[horizon] = (pred_next - target).pow(2).mean()
    return errors


def evaluate_rollout_set(model, anchor_model, loader, train_cfg, frameskip, horizons, device):
    import numpy as np
    import torch

    was_training = model.training
    model.eval()
    if anchor_model is not None:
        anchor_model.eval()
    model_values = {h: [] for h in horizons}
    anchor_values = {h: [] for h in horizons}
    for batch in loader:
        batch = move_to_device(batch, device)
        with torch.no_grad():
            model_errors = rollout_errors_for_batch(model, batch, train_cfg, frameskip, horizons)
            if anchor_model is not None:
                anchor_errors = rollout_errors_for_batch(anchor_model, batch, train_cfg, frameskip, horizons)
            else:
                anchor_errors = {}
        for horizon in horizons:
            model_values[horizon].append(float(model_errors[horizon].cpu()))
            if horizon in anchor_errors:
                anchor_values[horizon].append(float(anchor_errors[horizon].cpu()))
    restore_training_mode(model, was_training)
    result = {}
    for horizon in horizons:
        current = float(np.mean(model_values[horizon]))
        anchor = float(np.mean(anchor_values[horizon])) if anchor_values[horizon] else None
        result[f"h{horizon}_mse"] = current
        result[f"h{horizon}_pretrained_mse"] = anchor
        result[f"h{horizon}_ratio_to_pretrained"] = (current / anchor) if anchor and anchor > 0 else None
    return result


def apply_trainable_scope(model, scope: str, freeze_encoder: bool):
    if scope == "all":
        if freeze_encoder and hasattr(model, "encoder"):
            model.encoder.requires_grad_(False)
    else:
        for param in model.parameters():
            param.requires_grad_(False)
        for name, param in model.named_parameters():
            allow_adaln = scope in {"adaln_action", "adaln_only"} and "adaLN_modulation" in name
            allow_action = scope in {"adaln_action", "action_only"} and name.startswith("action_encoder.")
            if allow_adaln or allow_action:
                param.requires_grad_(True)
    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    return {
        "scope": scope,
        "trainable_param_tensors": len(trainable),
        "trainable_param_count": int(sum(param.numel() for _name, param in trainable)),
        "trainable_names": [name for name, _param in trainable],
    }


def parse_eval_steps(value: str, max_step: int) -> set[int]:
    if not value:
        return set()
    steps = {0, max_step}
    for item in value.split(","):
        item = item.strip()
        if not item:
            continue
        step = int(item)
        if step < 0 or step > max_step:
            raise ValueError(f"eval step {step} outside [0, {max_step}]")
        steps.add(step)
    return steps


def main() -> None:
    args = parse_args()
    lewm_dir = Path(args.lewm_dir).resolve()
    cache_dir = Path(args.cache_dir).resolve()
    oracle_json = Path(args.oracle_json).resolve()
    output_json = Path(args.output_json).resolve() if args.output_json else None
    if not lewm_dir.exists():
        raise SystemExit(f"Missing LeWM directory: {lewm_dir}")

    sys.path.insert(0, str(lewm_dir))
    os.environ.setdefault("STABLEWM_HOME", str(cache_dir))
    os.environ.setdefault("LOCAL_DATASET_DIR", str(cache_dir))

    import hydra
    import stable_pretraining as spt
    import stable_worldmodel as swm
    import torch
    from omegaconf import OmegaConf, open_dict

    from module import SIGReg
    from utils import get_column_normalizer, get_img_preprocessor

    if hasattr(torch.backends, "mha"):
        torch.backends.mha.set_fastpath_enabled(False)
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is false.")

    oracle_payload, per_state = load_oracle(oracle_json)

    os.chdir(lewm_dir)
    overrides = [
        "data=dmc",
        f"data.dataset.name={args.dataset_name}",
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

    train_count = args.train_states if args.train_states is not None else len(per_state) - args.heldout_states
    if train_count <= 0 or train_count > len(per_state):
        raise ValueError(f"Invalid train split: train_states={train_count}, oracle_states={len(per_state)}")
    heldout_count = int(args.heldout_states)
    if train_count + heldout_count > len(per_state):
        raise ValueError(
            f"train_states + heldout_states must be <= oracle states, got {train_count} + {heldout_count} > {len(per_state)}"
        )
    train_per_state = per_state[:train_count]
    heldout_per_state = per_state[train_count : train_count + heldout_count]

    train_set = OracleGeometryDataset(dataset, train_per_state)
    heldout_set = OracleGeometryDataset(dataset, heldout_per_state) if heldout_per_state else None
    generator = torch.Generator().manual_seed(args.seed)
    geom_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=0,
        pin_memory=args.device == "cuda",
        generator=generator,
    )
    heldout_loader = None
    if heldout_set is not None:
        heldout_loader = torch.utils.data.DataLoader(
            heldout_set,
            batch_size=args.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=args.device == "cuda",
        )
    pred_batch_size = args.pred_batch_size or args.batch_size
    full_pred_loader = None
    full_pred_iterator = None
    if args.full_prediction_batches:
        pred_generator = torch.Generator().manual_seed(args.seed + 1009)
        full_pred_loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=pred_batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=0,
            pin_memory=args.device == "cuda",
            generator=pred_generator,
        )

    pred_eval_loader = None
    if args.pred_eval_samples > 0:
        eval_generator = torch.Generator().manual_seed(args.seed + 2027)
        eval_count = min(int(args.pred_eval_samples), len(dataset))
        eval_indices = torch.randperm(len(dataset), generator=eval_generator)[:eval_count].tolist()
        pred_eval_set = torch.utils.data.Subset(dataset, eval_indices)
        pred_eval_loader = torch.utils.data.DataLoader(
            pred_eval_set,
            batch_size=pred_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=args.device == "cuda",
        )
    rollout_horizons = parse_horizons(args.rollout_horizons, args.frameskip)
    rollout_loader = None
    if args.rollout_eval_samples > 0:
        max_blocks = max(rollout_horizons) // args.frameskip
        rollout_dataset_cfg = dict(dataset_cfg)
        rollout_dataset_cfg["num_steps"] = args.history_size + max_blocks
        rollout_dataset = swm.data.load_dataset(
            dataset_name,
            transform=None,
            cache_dir=cache_dir,
            **rollout_dataset_cfg,
        )
        rollout_dataset.transform = dataset.transform
        rollout_generator = torch.Generator().manual_seed(args.seed + 3037)
        rollout_count = min(int(args.rollout_eval_samples), len(rollout_dataset))
        rollout_indices = torch.randperm(len(rollout_dataset), generator=rollout_generator)[:rollout_count].tolist()
        rollout_set = torch.utils.data.Subset(rollout_dataset, rollout_indices)
        rollout_loader = torch.utils.data.DataLoader(
            rollout_set,
            batch_size=pred_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=args.device == "cuda",
        )

    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    model = swm.wm.utils.load_pretrained(args.init_policy, cache_dir=cache_dir).to(device).train()
    model._freeze_batchnorm_stats = bool(args.freeze_batchnorm_stats)
    if args.freeze_batchnorm_stats:
        set_batchnorm_eval(model)
    anchor_model = None
    if args.anchor_weight > 0 or rollout_loader is not None:
        anchor_model = copy.deepcopy(model).eval()
        anchor_model.requires_grad_(False)
    trainable_summary = apply_trainable_scope(model, args.trainable_scope, args.freeze_encoder)
    sigreg = SIGReg().to(device)
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    if not trainable_params:
        raise SystemExit(f"No trainable parameters for scope {args.trainable_scope}")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=args.weight_decay)
    train_cfg = {
        "history_size": args.history_size,
        "num_preds": args.num_preds,
        "sigreg_weight": args.sigreg_weight,
    }

    eval_steps = parse_eval_steps(args.eval_steps, args.steps)
    run_dir = cache_dir / "checkpoints" / args.output_run
    snapshots = []

    def save_checkpoint_for_step(step: int) -> None:
        step_dir = run_dir / f"step_{step:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), step_dir / "weights.pt")
        shutil.copyfile(resolve_init_config(cache_dir, args.init_policy), step_dir / "config.json")

    def evaluate_snapshot(step: int):
        oracle_metrics = (
            evaluate_oracle_set(model, sigreg, heldout_loader, train_cfg, args.frameskip, device)
            if heldout_loader is not None
            else None
        )
        pred_metrics = (
            evaluate_prediction_set(model, sigreg, pred_eval_loader, train_cfg, device)
            if pred_eval_loader is not None
            else None
        )
        rollout_metrics = (
            evaluate_rollout_set(model, anchor_model, rollout_loader, train_cfg, args.frameskip, rollout_horizons, device)
            if rollout_loader is not None
            else None
        )
        row = {
            "step": step,
            "heldout_oracle": oracle_metrics,
            "full_prediction": pred_metrics,
            "rollout": rollout_metrics,
        }
        snapshots.append(row)
        print(json.dumps({"snapshot": row}, sort_keys=True), flush=True)
        if args.save_eval_checkpoints:
            save_checkpoint_for_step(step)
        return row

    heldout_before = evaluate_snapshot(0)["heldout_oracle"] if 0 in eval_steps else (
        evaluate_oracle_set(model, sigreg, heldout_loader, train_cfg, args.frameskip, device)
        if heldout_loader is not None
        else None
    )

    grad_probe = None
    geom_weight = float(args.geom_weight)
    if args.auto_geom_ratio is not None:
        geom_probe_batch = next(iter(geom_loader))
        geom_probe_batch = move_to_device(geom_probe_batch, device)
        if args.full_prediction_batches:
            pred_probe_batch = next(iter(full_pred_loader))
        else:
            pred_probe_batch = geom_probe_batch
        pred_probe_batch = move_to_device(pred_probe_batch, device)
        geom_weight, grad_probe = choose_geom_weight(
            model,
            sigreg,
            pred_probe_batch,
            geom_probe_batch,
            train_cfg,
            args.frameskip,
            args.auto_geom_ratio,
            trainable_params,
        )
        print(json.dumps({"grad_probe": grad_probe}, sort_keys=True), flush=True)

    history = []
    geom_iterator = iter(geom_loader)
    if full_pred_loader is not None:
        full_pred_iterator = iter(full_pred_loader)
    for step in range(1, args.steps + 1):
        try:
            geom_batch = next(geom_iterator)
        except StopIteration:
            geom_iterator = iter(geom_loader)
            geom_batch = next(geom_iterator)
        geom_batch = move_to_device(geom_batch, device)
        if args.full_prediction_batches:
            try:
                pred_batch = next(full_pred_iterator)
            except StopIteration:
                full_pred_iterator = iter(full_pred_loader)
                pred_batch = next(full_pred_iterator)
            pred_batch = move_to_device(pred_batch, device)
        else:
            pred_batch = geom_batch
        optimizer.zero_grad(set_to_none=True)
        if args.full_prediction_batches:
            loss, pred_loss, sigreg_loss, geom_loss, anchor_loss = two_stream_metrics(
                model,
                sigreg,
                pred_batch,
                geom_batch,
                train_cfg,
                geom_weight,
                args.frameskip,
                anchor_model=anchor_model,
                anchor_weight=args.anchor_weight,
            )
        else:
            loss, pred_loss, sigreg_loss, geom_loss, anchor_loss = finite_metrics(
                model, sigreg, geom_batch, train_cfg, geom_weight, args.frameskip
            )
        if not torch.isfinite(loss):
            raise SystemExit(f"Non-finite loss at step {step}: {float(loss.detach().cpu())}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clip)
        if not torch.isfinite(grad_norm):
            raise SystemExit(f"Non-finite grad norm at step {step}: {float(grad_norm)}")
        optimizer.step()

        row = {
            "step": step,
            "loss": float(loss.detach().cpu()),
            "pred_loss": float(pred_loss.detach().cpu()),
            "sigreg_loss": float(sigreg_loss.detach().cpu()),
            "geom_loss": float(geom_loss.detach().cpu()),
            "anchor_loss": float(anchor_loss.detach().cpu()),
            "grad_norm": float(grad_norm.detach().cpu() if torch.is_tensor(grad_norm) else grad_norm),
        }
        history.append(row)
        if step == 1 or step % args.log_every == 0 or step == args.steps:
            print(json.dumps(row, sort_keys=True), flush=True)
        if step in eval_steps:
            evaluate_snapshot(step)

    run_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), run_dir / "weights.pt")
    init_config = resolve_init_config(cache_dir, args.init_policy)
    shutil.copyfile(init_config, run_dir / "config.json")
    heldout_after = snapshots[-1]["heldout_oracle"] if snapshots and snapshots[-1]["step"] == args.steps else (
        evaluate_oracle_set(model, sigreg, heldout_loader, train_cfg, args.frameskip, device)
        if heldout_loader is not None
        else None
    )
    full_prediction_after = snapshots[-1]["full_prediction"] if snapshots and snapshots[-1]["step"] == args.steps else (
        evaluate_prediction_set(model, sigreg, pred_eval_loader, train_cfg, device)
        if pred_eval_loader is not None
        else None
    )
    rollout_after = snapshots[-1]["rollout"] if snapshots and snapshots[-1]["step"] == args.steps else (
        evaluate_rollout_set(model, anchor_model, rollout_loader, train_cfg, args.frameskip, rollout_horizons, device)
        if rollout_loader is not None
        else None
    )

    summary = {
        "ok": True,
        "output_run": args.output_run,
        "checkpoint_dir": str(run_dir),
        "init_policy": args.init_policy,
        "oracle_json": str(oracle_json),
        "oracle_num_states": len(per_state),
        "train_states": len(train_per_state),
        "heldout_states": len(heldout_per_state),
        "oracle_env_metric_coordinate": oracle_payload.get("env_metric_coordinate"),
        "dataset_name": dataset_name,
        "steps": args.steps,
        "batch_size": args.batch_size,
        "pred_batch_size": pred_batch_size,
        "full_prediction_batches": args.full_prediction_batches,
        "pred_eval_samples": args.pred_eval_samples,
        "rollout_eval_samples": args.rollout_eval_samples,
        "rollout_horizons": rollout_horizons,
        "eval_steps": sorted(eval_steps),
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "sigreg_weight": args.sigreg_weight,
        "anchor_weight": args.anchor_weight,
        "freeze_batchnorm_stats": args.freeze_batchnorm_stats,
        "freeze_encoder": args.freeze_encoder,
        "trainable_scope": args.trainable_scope,
        "trainable_summary": trainable_summary,
        "auto_geom_ratio": args.auto_geom_ratio,
        "geom_weight": geom_weight,
        "grad_probe": grad_probe,
        "device": args.device,
        "first": history[0],
        "last": history[-1],
        "heldout_before": heldout_before,
        "heldout_after": heldout_after,
        "full_prediction_after": full_prediction_after,
        "rollout_after": rollout_after,
        "snapshots": snapshots,
    }
    text = json.dumps({"summary": summary, "history": history}, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
