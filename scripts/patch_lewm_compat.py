#!/usr/bin/env python
"""Apply small compatibility patches needed by the GeoJEPA LeWM harness."""

from __future__ import annotations

import argparse
from pathlib import Path


PIXEL_PREPROCESSOR = '''\n\nclass PixelPreprocessor:\n    """Picklable image preprocessor for LeWM pixel sequences."""\n\n    def __init__(self, source: str, target: str, img_size: int = 224):\n        self.source = source\n        self.target = target\n        self.img_size = img_size\n        stats = dt.dataset_stats.ImageNet\n        self.mean = torch.tensor(stats["mean"], dtype=torch.float32).view(1, 3, 1, 1)\n        self.std = torch.tensor(stats["std"], dtype=torch.float32).view(1, 3, 1, 1)\n\n    def __call__(self, sample):\n        pixels = torch.as_tensor(sample[self.source]).float()\n        if pixels.ndim != 4:\n            raise ValueError(f"Expected {self.source} to have shape (T,H,W,C) or (T,C,H,W), got {tuple(pixels.shape)}")\n\n        if pixels.shape[-1] in (1, 3):\n            pixels = pixels.permute(0, 3, 1, 2)\n        if pixels.shape[1] == 1:\n            pixels = pixels.expand(-1, 3, -1, -1)\n\n        if pixels.max() > 2:\n            pixels = pixels / 255.0\n        if pixels.shape[-2:] != (self.img_size, self.img_size):\n            pixels = F.interpolate(\n                pixels,\n                size=(self.img_size, self.img_size),\n                mode="bilinear",\n                align_corners=False,\n                antialias=True,\n            )\n        sample[self.target] = (pixels - self.mean.to(pixels.device)) / self.std.to(pixels.device)\n        return sample\n'''

OLD_PREPROCESSOR = '''def get_img_preprocessor(source: str, target: str, img_size: int = 224):\n    imagenet_stats = dt.dataset_stats.ImageNet\n    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)\n    resize = dt.transforms.Resize(img_size, source=source, target=target)\n    return dt.transforms.Compose(to_image, resize)'''

NEW_PREPROCESSOR = '''def get_img_preprocessor(source: str, target: str, img_size: int = 224):\n    return PixelPreprocessor(source=source, target=target, img_size=img_size)'''

OLD_COLUMN_NORMALIZER = '''def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)'''

NEW_COLUMN_NORMALIZER = '''def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data)).float()
    if data.ndim == 1:
        data = data[torch.isfinite(data)]
        if data.numel() == 0:
            mean = torch.zeros(1)
            std = torch.ones(1)
        else:
            mean = data.mean().reshape(1).clone()
            std = data.std(unbiased=False).reshape(1).clamp_min(1e-6).clone()
    else:
        data = data[torch.isfinite(data).all(dim=1)]
        if data.numel() == 0:
            mean = torch.zeros((1, data.shape[-1]))
            std = torch.ones((1, data.shape[-1]))
        else:
            mean = data.mean(0, keepdim=True).clone()
            std = data.std(0, keepdim=True, unbiased=False).clamp_min(1e-6).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)'''

B1_EFFECT_ENCODER = '''\n\nclass StateConditionedEffectEncoder(nn.Module):\n    """State-conditioned action-effect encoder for prediction-only B1 ablations."""\n\n    def __init__(\n        self,\n        state_dim: int,\n        action_dim: int,\n        hidden_dim: int | None = None,\n        output_dim: int | None = None,\n    ):\n        super().__init__()\n        hidden_dim = hidden_dim or 4 * state_dim\n        output_dim = output_dim or action_dim\n        self.net = nn.Sequential(\n            nn.Linear(state_dim + action_dim, hidden_dim),\n            nn.SiLU(),\n            nn.Linear(hidden_dim, output_dim),\n        )\n\n    def forward(self, state_emb, action_emb):\n        return self.net(torch.cat([state_emb.detach(), action_emb], dim=-1))\n'''

B2_VALUE_HEAD = '''\n\nclass StateEffectValueHead(nn.Module):\n    """Q/value head whose gradient shapes the action-effect representation."""\n\n    def __init__(self, state_dim: int, effect_dim: int, hidden_dim: int | None = None):\n        super().__init__()\n        hidden_dim = hidden_dim or 4 * state_dim\n        self.net = nn.Sequential(\n            nn.Linear(state_dim + effect_dim, hidden_dim),\n            nn.SiLU(),\n            nn.Linear(hidden_dim, hidden_dim),\n            nn.SiLU(),\n            nn.Linear(hidden_dim, 1),\n        )\n\n    def forward(self, state_emb, effect_emb):\n        return self.net(torch.cat([state_emb.detach(), effect_emb], dim=-1))\n'''

B3_GOAL_VALUE_HEAD = '''\n\nclass GoalConditionedStateEffectValueHead(nn.Module):\n    """Goal-conditioned Q head Q(s, e, g) for value-aware planning."""\n\n    def __init__(\n        self,\n        state_dim: int,\n        effect_dim: int,\n        goal_dim: int | None = None,\n        hidden_dim: int | None = None,\n    ):\n        super().__init__()\n        goal_dim = goal_dim or state_dim\n        hidden_dim = hidden_dim or 4 * state_dim\n        self.net = nn.Sequential(\n            nn.Linear(state_dim + effect_dim + goal_dim, hidden_dim),\n            nn.SiLU(),\n            nn.Linear(hidden_dim, hidden_dim),\n            nn.SiLU(),\n            nn.Linear(hidden_dim, 1),\n        )\n\n    def forward(self, state_emb, effect_emb, goal_emb):\n        if goal_emb.ndim == state_emb.ndim - 1:\n            goal_emb = goal_emb.unsqueeze(1).expand_as(state_emb)\n        return self.net(torch.cat([state_emb.detach(), effect_emb, goal_emb.detach()], dim=-1))\n'''

STATE_SCALE_HEAD = '''\n\nclass StateScaleHead(nn.Module):\n    """Positive state-dependent scale head alpha_xi(sg(z)) for Resolution A."""\n\n    def __init__(self, state_dim: int, hidden_dim: int | None = None, alpha_min: float = 1e-3):\n        super().__init__()\n        hidden_dim = hidden_dim or state_dim\n        self.alpha_min = float(alpha_min)\n        self.net = nn.Sequential(\n            nn.Linear(state_dim, hidden_dim),\n            nn.SiLU(),\n            nn.Linear(hidden_dim, hidden_dim),\n            nn.SiLU(),\n            nn.Linear(hidden_dim, 1),\n        )\n\n    def forward(self, state_emb):\n        return F.softplus(self.net(state_emb.detach())) + self.alpha_min\n'''

B1_EFFECT_JEPA = '''\n\nclass EffectJEPA(JEPA):\n    """Prediction-only state-conditioned action-effect JEPA.\n\n    This implements the B1 ablation from the GeoJEPA draft: actions are first\n    embedded as u_t=A(a_t), then converted to effects e_t=H(sg(z_t),u_t), and\n    the existing predictor consumes e_t. No value head or geometric loss is\n    added in this ablation.\n    """\n\n    def __init__(\n        self,\n        encoder,\n        predictor,\n        action_encoder,\n        effect_encoder,\n        projector=None,\n        pred_proj=None,\n    ):\n        super().__init__(encoder, predictor, action_encoder, projector, pred_proj)\n        self.effect_encoder = effect_encoder\n\n    def action_condition(self, emb, action):\n        action_emb = self.action_encoder(action)\n        return self.effect_encoder(emb.detach(), action_emb)\n\n    def encode(self, info):\n        info = super().encode(info)\n        if "action" in info:\n            info["raw_act_emb"] = info["act_emb"]\n            info["act_emb"] = self.effect_encoder(info["emb"].detach(), info["raw_act_emb"])\n        return info\n\n    def rollout(self, info, action_sequence, history_size: int = 3):\n        """Rollout using state-conditioned action-effect embeddings."""\n        assert "pixels" in info, "pixels not in info_dict"\n        H = info["pixels"].size(2)\n        B, S, T = action_sequence.shape[:3]\n        act_0, act_future = torch.split(action_sequence, [H, T - H], dim=2)\n        info["action"] = act_0\n        n_steps = T - H\n\n        _init = {k: v[:, 0] for k, v in info.items() if torch.is_tensor(v)}\n        _init = self.encode(_init)\n        emb = info["emb"] = _init["emb"].unsqueeze(1).expand(B, S, -1, -1)\n        _init = {k: detach_clone(v) for k, v in _init.items()}\n\n        emb = rearrange(emb, "b s ... -> (b s) ...").clone()\n        act = rearrange(act_0, "b s ... -> (b s) ...")\n        act_future = rearrange(act_future, "b s ... -> (b s) ...")\n\n        HS = history_size\n        for t in range(n_steps):\n            emb_trunc = emb[:, -HS:]\n            act_trunc = act[:, -HS:]\n            effect_trunc = self.action_condition(emb_trunc, act_trunc)\n            pred_emb = self.predict(emb_trunc, effect_trunc)[:, -1:]\n            emb = torch.cat([emb, pred_emb], dim=1)\n\n            next_act = act_future[:, t : t + 1, :]\n            act = torch.cat([act, next_act], dim=1)\n\n        emb_trunc = emb[:, -HS:]\n        act_trunc = act[:, -HS:]\n        effect_trunc = self.action_condition(emb_trunc, act_trunc)\n        pred_emb = self.predict(emb_trunc, effect_trunc)[:, -1:]\n        emb = torch.cat([emb, pred_emb], dim=1)\n\n        info["predicted_emb"] = rearrange(emb, "(b s) ... -> b s ...", b=B, s=S)\n        return info\n'''

B2_EFFECT_VALUE_JEPA = '''\n\nclass ValueEffectJEPA(EffectJEPA):\n    """B2 ablation: effect representation plus value shaping, no geometry."""\n\n    def __init__(\n        self,\n        encoder,\n        predictor,\n        action_encoder,\n        effect_encoder,\n        value_head,\n        projector=None,\n        pred_proj=None,\n    ):\n        super().__init__(encoder, predictor, action_encoder, effect_encoder, projector, pred_proj)\n        self.value_head = value_head\n\n    def value(self, emb, effect_emb):\n        return self.value_head(emb.detach(), effect_emb)\n'''

B2_PRIME_EFFECT_VALUE_JEPA = '''\n\nclass ValuePlanningEffectJEPA(EffectJEPA):\n    """Effect JEPA with reward-trained value head used during CEM planning."""\n\n    def __init__(\n        self,\n        encoder,\n        predictor,\n        action_encoder,\n        effect_encoder,\n        value_head,\n        planning_value_weight: float = 0.0,\n        projector=None,\n        pred_proj=None,\n    ):\n        super().__init__(encoder, predictor, action_encoder, effect_encoder, projector, pred_proj)\n        self.value_head = value_head\n        self.planning_value_weight = float(planning_value_weight)\n\n    def value(self, emb, effect_emb):\n        return self.value_head(emb.detach(), effect_emb)\n\n    def rollout(self, info, action_sequence, history_size: int = 3):\n        info = super().rollout(info, action_sequence, history_size=history_size)\n        if self.planning_value_weight <= 0:\n            return info\n\n        B, S, T = action_sequence.shape[:3]\n        H = info[\"pixels\"].size(2)\n        state = info[\"predicted_emb\"][:, :, :T]\n        state = rearrange(state, \"b s t d -> (b s) t d\")\n        action = rearrange(action_sequence, \"b s t d -> (b s) t d\")\n        effect = self.action_condition(state, action)\n        value = self.value(state, effect).squeeze(-1)\n        future_value = value[:, H:] if value.size(1) > H else value[:, -1:]\n        plan_value = future_value.mean(dim=1)\n        info[\"plan_value\"] = rearrange(plan_value, \"(b s) -> b s\", b=B, s=S)\n        return info\n\n    def criterion(self, info_dict: dict):\n        cost = super().criterion(info_dict)\n        if self.planning_value_weight > 0 and \"plan_value\" in info_dict:\n            plan_value = torch.nan_to_num(info_dict[\"plan_value\"].float(), 0.0)\n            cost = cost - self.planning_value_weight * plan_value\n        return cost\n'''

B3_GOAL_VALUE_JEPA = '''\n\nclass GoalValuePlanningEffectJEPA(EffectJEPA):\n    """B3: B1 effects plus goal-conditioned Q used in CEM planning."""\n\n    value_goal_conditioned = True\n\n    def __init__(\n        self,\n        encoder,\n        predictor,\n        action_encoder,\n        effect_encoder,\n        value_head,\n        planning_value_weight: float = 0.0,\n        planning_value_gamma: float = 0.99,\n        log_cost_stats: bool = False,\n        cost_log_limit: int = 8,\n        projector=None,\n        pred_proj=None,\n    ):\n        super().__init__(encoder, predictor, action_encoder, effect_encoder, projector, pred_proj)\n        self.value_head = value_head\n        self.planning_value_weight = float(planning_value_weight)\n        self.planning_value_gamma = float(planning_value_gamma)\n        self.log_cost_stats = bool(log_cost_stats)\n        self.cost_log_limit = int(cost_log_limit)\n        self._cost_log_count = 0\n\n    def value(self, emb, effect_emb, goal_emb=None):\n        if goal_emb is None:\n            goal_emb = emb\n        return self.value_head(emb.detach(), effect_emb, goal_emb.detach())\n\n    def _expand_goal(self, goal_emb, B: int, S: int, T: int):\n        if goal_emb.ndim == 4:\n            goal = goal_emb[:, :1, -1:, :]\n        elif goal_emb.ndim == 3:\n            goal = goal_emb[:, None, -1:, :]\n        elif goal_emb.ndim == 2:\n            goal = goal_emb[:, None, None, :]\n        else:\n            raise ValueError(f\"Unsupported goal_emb shape: {tuple(goal_emb.shape)}\")\n        return goal.expand(B, S, T, -1)\n\n    def rollout(self, info, action_sequence, history_size: int = 3):\n        info = super().rollout(info, action_sequence, history_size=history_size)\n        if self.planning_value_weight <= 0 or \"goal_emb\" not in info:\n            return info\n\n        B, S, T = action_sequence.shape[:3]\n        H = info[\"pixels\"].size(2)\n        state = info[\"predicted_emb\"][:, :, :T]\n        goal = self._expand_goal(info[\"goal_emb\"], B, S, T)\n        state_f = rearrange(state, \"b s t d -> (b s) t d\")\n        goal_f = rearrange(goal, \"b s t d -> (b s) t d\")\n        action_f = rearrange(action_sequence, \"b s t d -> (b s) t d\")\n        effect = self.action_condition(state_f, action_f)\n        value = self.value(state_f, effect, goal_f).squeeze(-1)\n        future_value = value[:, H:] if value.size(1) > H else value[:, -1:]\n        discounts = torch.pow(\n            torch.full_like(future_value, self.planning_value_gamma),\n            torch.arange(future_value.size(1), device=future_value.device, dtype=future_value.dtype).unsqueeze(0),\n        )\n        plan_value = (future_value * discounts).sum(dim=1)\n        info[\"plan_value\"] = rearrange(plan_value, \"(b s) -> b s\", b=B, s=S)\n        return info\n\n    def criterion(self, info_dict: dict):\n        latent_cost = super().criterion(info_dict)\n        final_cost = latent_cost\n        if self.planning_value_weight > 0 and \"plan_value\" in info_dict:\n            plan_value = torch.nan_to_num(info_dict[\"plan_value\"].float(), 0.0)\n            final_cost = latent_cost - self.planning_value_weight * plan_value\n            if self.log_cost_stats and self._cost_log_count < self.cost_log_limit:\n                lc = latent_cost.detach().float()\n                pv = plan_value.detach().float()\n                fc = final_cost.detach().float()\n                print(\n                    \"COST_STATS \"\n                    f\"latent_mean={lc.mean().item():.6g} latent_std={lc.std(unbiased=False).item():.6g} \"\n                    f\"latent_min={lc.min().item():.6g} latent_max={lc.max().item():.6g} \"\n                    f\"plan_value_mean={pv.mean().item():.6g} plan_value_std={pv.std(unbiased=False).item():.6g} \"\n                    f\"plan_value_min={pv.min().item():.6g} plan_value_max={pv.max().item():.6g} \"\n                    f\"final_mean={fc.mean().item():.6g} final_std={fc.std(unbiased=False).item():.6g} \"\n                    f\"final_min={fc.min().item():.6g} final_max={fc.max().item():.6g}\",\n                    flush=True,\n                )\n                self._cost_log_count += 1\n        return final_cost\n'''

RESOLUTION_A_EFFECT_JEPA = '''\n\nimport copy as _geo_copy\n\nclass AdaptiveScaleEffectJEPA(EffectJEPA):\n    """B1 effects plus Resolution-A adaptive scale heads for geometry loss."""\n\n    def __init__(\n        self,\n        encoder,\n        predictor,\n        action_encoder,\n        effect_encoder,\n        alpha_head,\n        alpha_target=None,\n        projector=None,\n        pred_proj=None,\n    ):\n        super().__init__(encoder, predictor, action_encoder, effect_encoder, projector, pred_proj)\n        self.alpha_head = alpha_head\n        self.alpha_target = alpha_target or _geo_copy.deepcopy(alpha_head)\n        for param in self.alpha_target.parameters():\n            param.requires_grad_(False)\n\n    def geo_alpha(self, emb, target: bool = False):\n        head = self.alpha_target if target else self.alpha_head\n        return head(emb.detach())\n\n    @torch.no_grad()\n    def update_alpha_target(self, tau: float = 0.01):\n        tau = float(tau)\n        for target_param, fast_param in zip(self.alpha_target.parameters(), self.alpha_head.parameters()):\n            target_param.data.mul_(1.0 - tau).add_(fast_param.data, alpha=tau)\n        for target_buffer, fast_buffer in zip(self.alpha_target.buffers(), self.alpha_head.buffers()):\n            target_buffer.copy_(fast_buffer)\n'''

B1_MODEL_CONFIG = '''_target_: jepa.EffectJEPA

encoder:
  _target_: stable_pretraining.backbone.utils.vit_hf
  size: tiny
  patch_size: 14
  image_size: ${img_size}
  pretrained: false
  use_mask_token: false

predictor:
  _target_: module.ARPredictor
  num_frames: ${history_size}
  input_dim: ${embed_dim}
  hidden_dim: ${embed_dim}
  output_dim: ${embed_dim}
  depth: 6
  heads: 16
  mlp_dim: 2048
  dim_head: 64
  dropout: 0.1
  emb_dropout: 0.0

action_encoder:
  _target_: module.Embedder
  input_dim: ???
  emb_dim: ${embed_dim}

effect_encoder:
  _target_: module.StateConditionedEffectEncoder
  state_dim: ${embed_dim}
  action_dim: ${embed_dim}
  hidden_dim: 768
  output_dim: ${embed_dim}

projector:
  _target_: module.MLP
  input_dim: ${embed_dim}
  output_dim: ${embed_dim}
  hidden_dim: 2048
  norm_fn:
    _target_: torch.nn.BatchNorm1d
    _partial_: true

pred_proj:
  _target_: module.MLP
  input_dim: ${embed_dim}
  output_dim: ${embed_dim}
  hidden_dim: 2048
  norm_fn:
    _target_: torch.nn.BatchNorm1d
    _partial_: true
'''

RESOLUTION_A_MODEL_CONFIG = '''_target_: jepa.AdaptiveScaleEffectJEPA

encoder:
  _target_: stable_pretraining.backbone.utils.vit_hf
  size: tiny
  patch_size: 14
  image_size: ${img_size}
  pretrained: false
  use_mask_token: false

predictor:
  _target_: module.ARPredictor
  num_frames: ${history_size}
  input_dim: ${embed_dim}
  hidden_dim: ${embed_dim}
  output_dim: ${embed_dim}
  depth: 6
  heads: 16
  mlp_dim: 2048
  dim_head: 64
  dropout: 0.1
  emb_dropout: 0.0

action_encoder:
  _target_: module.Embedder
  input_dim: ???
  emb_dim: ${embed_dim}

effect_encoder:
  _target_: module.StateConditionedEffectEncoder
  state_dim: ${embed_dim}
  action_dim: ${embed_dim}
  hidden_dim: 768
  output_dim: ${embed_dim}

alpha_head:
  _target_: module.StateScaleHead
  state_dim: ${embed_dim}
  hidden_dim: 192
  alpha_min: 1e-3

projector:
  _target_: module.MLP
  input_dim: ${embed_dim}
  output_dim: ${embed_dim}
  hidden_dim: 2048
  norm_fn:
    _target_: torch.nn.BatchNorm1d
    _partial_: true

pred_proj:
  _target_: module.MLP
  input_dim: ${embed_dim}
  output_dim: ${embed_dim}
  hidden_dim: 2048
  norm_fn:
    _target_: torch.nn.BatchNorm1d
    _partial_: true
'''

B2_MODEL_CONFIG = '''_target_: jepa.ValueEffectJEPA

encoder:
  _target_: stable_pretraining.backbone.utils.vit_hf
  size: tiny
  patch_size: 14
  image_size: ${img_size}
  pretrained: false
  use_mask_token: false

predictor:
  _target_: module.ARPredictor
  num_frames: ${history_size}
  input_dim: ${embed_dim}
  hidden_dim: ${embed_dim}
  output_dim: ${embed_dim}
  depth: 6
  heads: 16
  mlp_dim: 2048
  dim_head: 64
  dropout: 0.1
  emb_dropout: 0.0

action_encoder:
  _target_: module.Embedder
  input_dim: ???
  emb_dim: ${embed_dim}

effect_encoder:
  _target_: module.StateConditionedEffectEncoder
  state_dim: ${embed_dim}
  action_dim: ${embed_dim}
  hidden_dim: 768
  output_dim: ${embed_dim}

value_head:
  _target_: module.StateEffectValueHead
  state_dim: ${embed_dim}
  effect_dim: ${embed_dim}
  hidden_dim: 768

projector:
  _target_: module.MLP
  input_dim: ${embed_dim}
  output_dim: ${embed_dim}
  hidden_dim: 2048
  norm_fn:
    _target_: torch.nn.BatchNorm1d
    _partial_: true

pred_proj:
  _target_: module.MLP
  input_dim: ${embed_dim}
  output_dim: ${embed_dim}
  hidden_dim: 2048
  norm_fn:
    _target_: torch.nn.BatchNorm1d
    _partial_: true
'''

B2_PRIME_MODEL_CONFIG = '''_target_: jepa.ValuePlanningEffectJEPA

encoder:
  _target_: stable_pretraining.backbone.utils.vit_hf
  size: tiny
  patch_size: 14
  image_size: ${img_size}
  pretrained: false
  use_mask_token: false

predictor:
  _target_: module.ARPredictor
  num_frames: ${history_size}
  input_dim: ${embed_dim}
  hidden_dim: ${embed_dim}
  output_dim: ${embed_dim}
  depth: 6
  heads: 16
  mlp_dim: 2048
  dim_head: 64
  dropout: 0.1
  emb_dropout: 0.0

action_encoder:
  _target_: module.Embedder
  input_dim: ???
  emb_dim: ${embed_dim}

effect_encoder:
  _target_: module.StateConditionedEffectEncoder
  state_dim: ${embed_dim}
  action_dim: ${embed_dim}
  hidden_dim: 768
  output_dim: ${embed_dim}

value_head:
  _target_: module.StateEffectValueHead
  state_dim: ${embed_dim}
  effect_dim: ${embed_dim}
  hidden_dim: 768

planning_value_weight: 0.1

projector:
  _target_: module.MLP
  input_dim: ${embed_dim}
  output_dim: ${embed_dim}
  hidden_dim: 2048
  norm_fn:
    _target_: torch.nn.BatchNorm1d
    _partial_: true

pred_proj:
  _target_: module.MLP
  input_dim: ${embed_dim}
  output_dim: ${embed_dim}
  hidden_dim: 2048
  norm_fn:
    _target_: torch.nn.BatchNorm1d
    _partial_: true
'''

B3_GOAL_VALUE_MODEL_CONFIG = '''_target_: jepa.GoalValuePlanningEffectJEPA

encoder:
  _target_: stable_pretraining.backbone.utils.vit_hf
  size: tiny
  patch_size: 14
  image_size: ${img_size}
  pretrained: false
  use_mask_token: false

predictor:
  _target_: module.ARPredictor
  num_frames: ${history_size}
  input_dim: ${embed_dim}
  hidden_dim: ${embed_dim}
  output_dim: ${embed_dim}
  depth: 6
  heads: 16
  mlp_dim: 2048
  dim_head: 64
  dropout: 0.1
  emb_dropout: 0.0

action_encoder:
  _target_: module.Embedder
  input_dim: ???
  emb_dim: ${embed_dim}

effect_encoder:
  _target_: module.StateConditionedEffectEncoder
  state_dim: ${embed_dim}
  action_dim: ${embed_dim}
  hidden_dim: 768
  output_dim: ${embed_dim}

value_head:
  _target_: module.GoalConditionedStateEffectValueHead
  state_dim: ${embed_dim}
  effect_dim: ${embed_dim}
  goal_dim: ${embed_dim}
  hidden_dim: 768

planning_value_weight: 0.1
planning_value_gamma: 0.99
log_cost_stats: false
cost_log_limit: 8

projector:
  _target_: module.MLP
  input_dim: ${embed_dim}
  output_dim: ${embed_dim}
  hidden_dim: 2048
  norm_fn:
    _target_: torch.nn.BatchNorm1d
    _partial_: true

pred_proj:
  _target_: module.MLP
  input_dim: ${embed_dim}
  output_dim: ${embed_dim}
  hidden_dim: 2048
  norm_fn:
    _target_: torch.nn.BatchNorm1d
    _partial_: true
'''

B2_VALUE_LOSS = '''    # LeWM + optional B2 value-shaping loss\n    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()\n    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))\n    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]\n\n    value_cfg = cfg.loss.get("value", {})\n    value_weight = float(value_cfg.get("weight", 0.0)) if value_cfg else 0.0\n    if value_weight > 0 and "reward" in batch and hasattr(self.model, "value"):\n        gamma = float(value_cfg.get("gamma", 0.99))\n        reward = torch.nan_to_num(batch["reward"].float(), 0.0)\n        if reward.ndim == 3 and reward.size(-1) == 1:\n            reward = reward.squeeze(-1)\n        if reward.ndim == 1:\n            reward = reward.unsqueeze(-1)\n        returns = torch.zeros_like(reward)\n        running = torch.zeros_like(reward[:, -1])\n        for t in range(reward.size(1) - 1, -1, -1):\n            running = reward[:, t] + gamma * running\n            returns[:, t] = running\n        value_target = returns[:, :ctx_len].unsqueeze(-1).detach()\n        value_pred = self.model.value(ctx_emb, ctx_act)\n        output["value_loss"] = (value_pred - value_target).pow(2).mean()\n        output["loss"] = output["loss"] + value_weight * output["value_loss"]\n'''

OLD_B2_VALUE_LOSS = B2_VALUE_LOSS
B2_VALUE_LOSS = '''    # LeWM + optional B2' value-shaping loss.
    # Q(s_t, a_t) is trained against cleaned transition rewards r_{t+1:}.
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    value_cfg = cfg.loss.get("value", {})
    value_weight = float(value_cfg.get("weight", 0.0)) if value_cfg else 0.0
    if value_weight > 0 and "reward" in batch and hasattr(self.model, "value"):
        gamma = float(value_cfg.get("gamma", 0.99))
        reward = batch["reward"].float()
        if reward.ndim == 3 and reward.size(-1) == 1:
            reward = reward.squeeze(-1)
        if reward.ndim == 1:
            reward = reward.unsqueeze(-1)
        valid_reward = torch.isfinite(reward) & (reward > -1e12)
        reward = torch.where(valid_reward, reward, torch.zeros_like(reward))
        transition_reward = reward[:, 1 : ctx_len + 1]
        if transition_reward.size(1) < ctx_len:
            pad = torch.zeros(
                transition_reward.size(0),
                ctx_len - transition_reward.size(1),
                device=transition_reward.device,
                dtype=transition_reward.dtype,
            )
            transition_reward = torch.cat([transition_reward, pad], dim=1)
        returns = torch.zeros_like(transition_reward)
        running = torch.zeros_like(transition_reward[:, -1])
        for t in range(transition_reward.size(1) - 1, -1, -1):
            running = transition_reward[:, t] + gamma * running
            returns[:, t] = running
        value_target = returns.unsqueeze(-1).detach()
        value_pred = self.model.value(ctx_emb, ctx_act)
        output["value_loss"] = (value_pred - value_target).pow(2).mean()
        output["loss"] = output["loss"] + value_weight * output["value_loss"]
	'''

OLD_B2_PRIME_VALUE_LOSS = B2_VALUE_LOSS
B2_VALUE_LOSS = '''    # LeWM + optional goal-conditioned rollout value loss.
    # Extra future frames may be loaded for value training, so keep prediction
    # targets aligned to the context-window predictor output.
    tgt_emb = tgt_emb[:, : pred_emb.size(1)]
    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()
    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))
    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]

    value_cfg = cfg.loss.get("value", {})
    value_weight = float(value_cfg.get("weight", 0.0)) if value_cfg else 0.0
    if value_weight > 0 and "reward" in batch and hasattr(self.model, "value"):
        gamma = float(value_cfg.get("gamma", 0.99))
        value_horizon = int(value_cfg.get("horizon", ctx_len))
        reward = batch["reward"].float()
        if reward.ndim == 3 and reward.size(-1) == 1:
            reward = reward.squeeze(-1)
        if reward.ndim == 1:
            reward = reward.unsqueeze(-1)
        valid_reward = torch.isfinite(reward) & (reward > -1e12)
        reward = torch.where(valid_reward, reward, torch.zeros_like(reward))

        available = min(
            max(0, int(batch["action"].size(1)) - ctx_len),
            max(0, int(reward.size(1)) - ctx_len - 1),
            max(0, int(emb.size(1)) - ctx_len - 1),
            value_horizon,
        )
        if available > 0:
            raw_action = torch.nan_to_num(batch["action"].float(), 0.0)
            roll_emb = ctx_emb
            roll_act = raw_action[:, :ctx_len]
            for t in range(available):
                emb_trunc = roll_emb[:, -ctx_len:]
                act_trunc = roll_act[:, -ctx_len:]
                if hasattr(self.model, "action_condition"):
                    effect_trunc = self.model.action_condition(emb_trunc, act_trunc)
                else:
                    effect_trunc = self.model.action_encoder(act_trunc)
                pred_next = self.model.predict(emb_trunc, effect_trunc)[:, -1:]
                roll_emb = torch.cat([roll_emb, pred_next], dim=1)
                next_act = raw_action[:, ctx_len + t : ctx_len + t + 1]
                roll_act = torch.cat([roll_act, next_act], dim=1)

            value_state = roll_emb[:, ctx_len : ctx_len + available]
            value_action = raw_action[:, ctx_len : ctx_len + available]
            if hasattr(self.model, "action_condition"):
                value_effect = self.model.action_condition(value_state, value_action)
            else:
                value_effect = self.model.action_encoder(value_action)
            goal_emb = emb[:, ctx_len + available : ctx_len + available + 1].detach()
            goal_emb = goal_emb.expand(-1, available, -1)

            transition_reward = reward[:, ctx_len + 1 : ctx_len + 1 + available]
            returns = torch.zeros_like(transition_reward)
            running = torch.zeros_like(transition_reward[:, -1])
            for t in range(transition_reward.size(1) - 1, -1, -1):
                running = transition_reward[:, t] + gamma * running
                returns[:, t] = running
            value_target = returns.unsqueeze(-1).detach()
            if getattr(self.model, "value_goal_conditioned", False):
                value_pred = self.model.value(value_state, value_effect, goal_emb)
            else:
                value_pred = self.model.value(value_state, value_effect)
            output["value_loss"] = (value_pred - value_target).pow(2).mean()
            output["loss"] = output["loss"] + value_weight * output["value_loss"]

    geo_cfg = cfg.loss.get("geo", {})
    geo_resolution = str(geo_cfg.get("resolution", "b")).lower() if geo_cfg else "b"
    aniso_weight = float(geo_cfg.get("aniso_weight", 0.0)) if geo_cfg else 0.0
    scale_weight = float(geo_cfg.get("scale_weight", 0.0)) if geo_cfg else 0.0
    geo_weight = float(geo_cfg.get("weight", 0.0)) if geo_cfg else 0.0
    alpha_weight = float(geo_cfg.get("alpha_weight", 0.0)) if geo_cfg else 0.0
    teacher_weight = float(geo_cfg.get("teacher_weight", geo_weight)) if geo_cfg else 0.0
    norm_eps = float(geo_cfg.get("norm_eps", 1e-6)) if geo_cfg else 1e-6
    geo_enabled = (
        (geo_resolution == "a" and (geo_weight > 0 or alpha_weight > 0))
        or (geo_resolution == "teacher" and teacher_weight > 0)
        or (geo_resolution == "dynmetric" and geo_weight > 0)
        or (geo_resolution not in ("a", "teacher", "dynmetric") and (aniso_weight > 0 or scale_weight > 0))
    )
    if geo_enabled and hasattr(self.model, "action_condition"):
        k_probes = int(geo_cfg.get("k", 4))
        if geo_resolution not in ("a", "teacher", "dynmetric") and k_probes < 2:
            raise ValueError("Resolution-B geometry needs k >= 2 for the T^2 U-statistic.")
        alpha0 = float(geo_cfg.get("alpha0", 1.0))
        alpha_tau = float(geo_cfg.get("alpha_tau", 0.01))
        max_points = int(geo_cfg.get("max_points", 0))
        geo_target = str(geo_cfg.get("target", "effect"))
        action_basis = str(geo_cfg.get("action_basis", "full"))
        frameskip = int(geo_cfg.get("frameskip", 5))
        if geo_target == "transition" or geo_resolution == "dynmetric":
            if hasattr(torch.backends, "mha"):
                torch.backends.mha.set_fastpath_enabled(False)
            if hasattr(torch.backends, "cuda"):
                torch.backends.cuda.enable_flash_sdp(False)
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_math_sdp(True)

        geo_state = ctx_emb.detach()
        raw_geo_action = torch.nan_to_num(batch["action"][:, :ctx_len].float(), 0.0)
        if max_points > 0 and geo_state.size(0) > max_points:
            perm = torch.randperm(geo_state.size(0), device=geo_state.device)[:max_points]
            geo_state = geo_state[perm]
            raw_geo_action = raw_geo_action[perm]

        def to_basis(action_full):
            if action_basis == "full":
                return action_full
            if action_basis == "first2":
                return action_full[..., :2]
            if action_basis == "repeat2":
                if action_full.size(-1) != 2 * frameskip:
                    raise ValueError(
                        f"repeat2 expects action_dim={2 * frameskip}, got {action_full.size(-1)}"
                    )
                return action_full.reshape(*action_full.shape[:-1], frameskip, 2).mean(dim=-2)
            raise ValueError(f"Unsupported geo action_basis={action_basis}")

        def from_basis(action_basis_value, action_template):
            if action_basis == "full":
                return action_basis_value
            if action_basis == "first2":
                action_full = action_template.clone()
                action_full[..., :2] = action_basis_value
                return action_full
            if action_basis == "repeat2":
                return action_basis_value.unsqueeze(-2).expand(
                    *action_basis_value.shape[:-1], frameskip, 2
                ).reshape(*action_basis_value.shape[:-1], 2 * frameskip)
            raise ValueError(f"Unsupported geo action_basis={action_basis}")

        if geo_resolution == "dynmetric":
            geo_action = to_basis(raw_geo_action[:, -1:, :])
        elif geo_target == "transition":
            geo_action = to_basis(raw_geo_action[:, -1:, :])
        elif geo_target == "effect":
            geo_action = to_basis(raw_geo_action)
        else:
            raise ValueError(f"Unsupported geo target={geo_target}")
        geo_action = geo_action.detach().requires_grad_(True)
        action_dim = geo_action.size(-1)

        p_vals = []
        q_vals = []
        geo_a_vals = []
        geo_teacher_vals = []
        geo_dyn_student_vals = []
        geo_dyn_target_vals = []
        dyn_p_vals = []

        alpha_state = geo_state[:, -1:, :] if geo_target == "transition" else geo_state
        alpha_target = None
        alpha_fast = None
        if geo_resolution == "a":
            if not hasattr(self.model, "geo_alpha"):
                raise ValueError("Resolution A requires model=lewm_b1_effect_resa or another model with geo_alpha().")
            alpha_target = self.model.geo_alpha(alpha_state, target=True).reshape(-1).detach()
            alpha_fast = self.model.geo_alpha(alpha_state, target=False).reshape(-1)
        teacher_model = None
        if geo_resolution == "teacher":
            teacher_model = getattr(self.model, "geo_teacher", None)
            if teacher_model is None:
                raise ValueError("Teacher-metric geometry requires cfg.loss.geo.teacher_model_path.")
            teacher_model = teacher_model.to(device=geo_state.device).eval()

        def effect_fn(action_in):
            if geo_resolution == "dynmetric":
                action_full = raw_geo_action.clone()
                action_full[:, -1:, :] = from_basis(action_in, raw_geo_action[:, -1:, :])
                return self.model.action_condition(geo_state, action_full)[:, -1:, :]
            if geo_target == "transition":
                action_full = raw_geo_action.clone()
                action_full[:, -1:, :] = from_basis(action_in, raw_geo_action[:, -1:, :])
                effect = self.model.action_condition(geo_state, action_full)
                return self.model.predict(geo_state, effect)[:, -1:, :]
            action_full = from_basis(action_in, raw_geo_action)
            return self.model.action_condition(geo_state, action_full)

        def dyn_metric_fn(action_in):
            action_full = raw_geo_action.clone()
            action_full[:, -1:, :] = from_basis(action_in, raw_geo_action[:, -1:, :])
            effect = self.model.action_condition(geo_state, action_full)
            return self.model.predict(geo_state, effect)[:, -1:, :]

        def teacher_effect_fn(action_in):
            if geo_target == "transition":
                action_full = raw_geo_action.clone()
                action_full[:, -1:, :] = from_basis(action_in, raw_geo_action[:, -1:, :])
                effect = teacher_model.action_condition(geo_state, action_full)
                return teacher_model.predict(geo_state, effect)[:, -1:, :]
            action_full = from_basis(action_in, raw_geo_action)
            return teacher_model.action_condition(geo_state, action_full)

        for _ in range(k_probes):
            eps = torch.empty_like(geo_action).bernoulli_(0.5).mul_(2.0).sub_(1.0)
            effect, j_eps = torch.autograd.functional.jvp(
                effect_fn,
                (geo_action,),
                (eps,),
                create_graph=True,
                strict=False,
            )
            jt_j_eps = torch.autograd.grad(
                effect,
                geo_action,
                grad_outputs=j_eps,
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]
            p_vals.append(j_eps.float().pow(2).sum(dim=-1).reshape(-1))
            q_vals.append(jt_j_eps.float().pow(2).sum(dim=-1).reshape(-1))
            if geo_resolution == "a":
                alpha_view = alpha_target.reshape(*jt_j_eps.shape[:-1], 1)
                geo_a_vals.append((jt_j_eps.float() - alpha_view * eps.float()).pow(2).sum(dim=-1).reshape(-1))
            elif geo_resolution == "teacher":
                with torch.enable_grad():
                    teacher_action = geo_action.detach().requires_grad_(True)
                    _, teacher_j_eps = torch.autograd.functional.jvp(
                        teacher_effect_fn,
                        (teacher_action,),
                        (eps.detach(),),
                        create_graph=False,
                        strict=False,
                    )
                    teacher_effect = teacher_effect_fn(teacher_action)
                    teacher_jt_j_eps = torch.autograd.grad(
                        teacher_effect,
                        teacher_action,
                        grad_outputs=teacher_j_eps,
                        create_graph=False,
                        retain_graph=False,
                        only_inputs=True,
                    )[0].detach()
                geo_teacher_vals.append(
                    (jt_j_eps.float() - teacher_jt_j_eps.float()).pow(2).sum(dim=-1).reshape(-1)
                )
            elif geo_resolution == "dynmetric":
                with torch.enable_grad():
                    dyn_action = geo_action.detach().requires_grad_(True)
                    _, dyn_j_eps = torch.autograd.functional.jvp(
                        dyn_metric_fn,
                        (dyn_action,),
                        (eps.detach(),),
                        create_graph=False,
                        strict=False,
                    )
                    dyn_effect = dyn_metric_fn(dyn_action)
                    dyn_jt_j_eps = torch.autograd.grad(
                        dyn_effect,
                        dyn_action,
                        grad_outputs=dyn_j_eps,
                        create_graph=False,
                        retain_graph=False,
                        only_inputs=True,
                    )[0].detach()
                dyn_p_vals.append(dyn_j_eps.float().pow(2).sum(dim=-1).reshape(-1))
                geo_dyn_student_vals.append(jt_j_eps.float())
                geo_dyn_target_vals.append(dyn_jt_j_eps.float())

        p = torch.stack(p_vals, dim=0)
        q = torch.stack(q_vals, dim=0)
        trace_hat = p.mean(dim=0)
        if geo_resolution == "a":
            geo_loss = torch.stack(geo_a_vals, dim=0).mean()
            trace_target = (trace_hat / action_dim).detach()
            alpha_loss = (alpha_fast - trace_target).pow(2).mean()
            weighted_geo_loss = geo_weight * geo_loss + alpha_weight * alpha_loss
            output["geo_loss"] = geo_loss
            output["geo_alpha_loss"] = alpha_loss
            output["geo_alpha_target_mean"] = alpha_target.mean().detach()
            output["geo_alpha_fast_mean"] = alpha_fast.mean().detach()
            output["geo_weighted_loss"] = weighted_geo_loss
            output["loss"] = output["loss"] + weighted_geo_loss
            if stage == "train" and hasattr(self.model, "update_alpha_target"):
                self.model.update_alpha_target(alpha_tau)
        elif geo_resolution == "teacher":
            geo_loss = torch.stack(geo_teacher_vals, dim=0).mean()
            weighted_geo_loss = teacher_weight * geo_loss
            output["geo_teacher_loss"] = geo_loss
            output["geo_weighted_loss"] = weighted_geo_loss
            output["loss"] = output["loss"] + weighted_geo_loss
        elif geo_resolution == "dynmetric":
            dyn_trace_hat = torch.stack(dyn_p_vals, dim=0).mean(dim=0).detach()
            effect_scale = (trace_hat / action_dim).clamp_min(norm_eps)
            dyn_scale = (dyn_trace_hat / action_dim).clamp_min(norm_eps)
            metric_losses = []
            for student_vec, target_vec in zip(geo_dyn_student_vals, geo_dyn_target_vals):
                effect_view = effect_scale.reshape(*student_vec.shape[:-1], 1)
                dyn_view = dyn_scale.reshape(*target_vec.shape[:-1], 1)
                metric_losses.append(
                    (student_vec / effect_view - target_vec / dyn_view)
                    .pow(2)
                    .sum(dim=-1)
                    .reshape(-1)
                )
            geo_loss = torch.stack(metric_losses, dim=0).mean()
            weighted_geo_loss = geo_weight * geo_loss
            output["geo_dynmetric_loss"] = geo_loss
            output["geo_dyn_trace_mean"] = dyn_trace_hat.mean().detach()
            output["geo_weighted_loss"] = weighted_geo_loss
            output["loss"] = output["loss"] + weighted_geo_loss
        else:
            trace_sq_u = (p.sum(dim=0).pow(2) - p.pow(2).sum(dim=0)) / (k_probes * (k_probes - 1))
            trace_g2_hat = q.mean(dim=0)
            aniso_loss = (trace_g2_hat - trace_sq_u / action_dim).mean()
            scale_loss = (
                trace_sq_u / (action_dim * action_dim)
                - 2.0 * alpha0 * trace_hat / action_dim
                + alpha0 * alpha0
            ).mean()
            weighted_geo_loss = aniso_weight * aniso_loss + scale_weight * scale_loss
            output["geo_aniso_loss"] = aniso_loss
            output["geo_scale_loss"] = scale_loss
            output["geo_weighted_loss"] = weighted_geo_loss
            output["loss"] = output["loss"] + weighted_geo_loss
        output["geo_trace_mean"] = trace_hat.mean().detach()
'''


def patch_utils(lewm_dir: Path) -> bool:
    path = lewm_dir / "utils.py"
    text = path.read_text(encoding="utf-8")
    changed = False
    if "import torch.nn.functional as F" not in text:
        text = text.replace("import torch\n", "import torch\nimport torch.nn.functional as F\n")
        changed = True
    if "class PixelPreprocessor" not in text:
        marker = "def get_img_preprocessor"
        text = text.replace(marker, PIXEL_PREPROCESSOR + "\n" + marker, 1)
        changed = True
    if OLD_PREPROCESSOR in text:
        text = text.replace(OLD_PREPROCESSOR, NEW_PREPROCESSOR)
        changed = True
    if OLD_COLUMN_NORMALIZER in text:
        text = text.replace(OLD_COLUMN_NORMALIZER, NEW_COLUMN_NORMALIZER)
        changed = True
    if changed:
        path.write_text(text, encoding="utf-8")
    return changed


def patch_module(lewm_dir: Path) -> bool:
    path = lewm_dir / "module.py"
    text = path.read_text(encoding="utf-8")
    changed = False
    if "class StateConditionedEffectEncoder" not in text:
        text = text.rstrip() + B1_EFFECT_ENCODER + "\n"
        changed = True
    if "class StateEffectValueHead" not in text:
        text = text.rstrip() + B2_VALUE_HEAD + "\n"
        changed = True
    if "class GoalConditionedStateEffectValueHead" not in text:
        text = text.rstrip() + B3_GOAL_VALUE_HEAD + "\n"
        changed = True
    if "class StateScaleHead" not in text:
        text = text.rstrip() + STATE_SCALE_HEAD + "\n"
        changed = True
    if changed:
        path.write_text(text, encoding="utf-8")
    return changed


def patch_jepa(lewm_dir: Path) -> bool:
    path = lewm_dir / "jepa.py"
    text = path.read_text(encoding="utf-8")
    changed = False
    if "class EffectJEPA" not in text:
        text = text.rstrip() + B1_EFFECT_JEPA + "\n"
        changed = True
    if "class ValueEffectJEPA" not in text:
        text = text.rstrip() + B2_EFFECT_VALUE_JEPA + "\n"
        changed = True
    if "class ValuePlanningEffectJEPA" not in text:
        text = text.rstrip() + B2_PRIME_EFFECT_VALUE_JEPA + "\n"
        changed = True
    if "class GoalValuePlanningEffectJEPA" not in text:
        text = text.rstrip() + B3_GOAL_VALUE_JEPA + "\n"
        changed = True
    if "class AdaptiveScaleEffectJEPA" not in text:
        text = text.rstrip() + RESOLUTION_A_EFFECT_JEPA + "\n"
        changed = True
    if changed:
        path.write_text(text, encoding="utf-8")
    return changed


def patch_b1_model_config(lewm_dir: Path) -> bool:
    path = lewm_dir / "config" / "train" / "model" / "lewm_b1_effect.yaml"
    if path.exists() and path.read_text(encoding="utf-8") == B1_MODEL_CONFIG:
        return False
    path.write_text(B1_MODEL_CONFIG, encoding="utf-8")
    return True


def patch_resolution_a_model_config(lewm_dir: Path) -> bool:
    path = lewm_dir / "config" / "train" / "model" / "lewm_b1_effect_resa.yaml"
    if path.exists() and path.read_text(encoding="utf-8") == RESOLUTION_A_MODEL_CONFIG:
        return False
    path.write_text(RESOLUTION_A_MODEL_CONFIG, encoding="utf-8")
    return True


def patch_b2_model_config(lewm_dir: Path) -> bool:
    path = lewm_dir / "config" / "train" / "model" / "lewm_b2_effect_value.yaml"
    if path.exists() and path.read_text(encoding="utf-8") == B2_MODEL_CONFIG:
        return False
    path.write_text(B2_MODEL_CONFIG, encoding="utf-8")
    return True


def patch_b2_prime_model_config(lewm_dir: Path) -> bool:
    path = lewm_dir / "config" / "train" / "model" / "lewm_b2_prime_effect_value.yaml"
    if path.exists() and path.read_text(encoding="utf-8") == B2_PRIME_MODEL_CONFIG:
        return False
    path.write_text(B2_PRIME_MODEL_CONFIG, encoding="utf-8")
    return True


def patch_b3_goal_value_model_config(lewm_dir: Path) -> bool:
    path = lewm_dir / "config" / "train" / "model" / "lewm_b3_goal_value.yaml"
    if path.exists() and path.read_text(encoding="utf-8") == B3_GOAL_VALUE_MODEL_CONFIG:
        return False
    path.write_text(B3_GOAL_VALUE_MODEL_CONFIG, encoding="utf-8")
    return True


def patch_train(lewm_dir: Path) -> bool:
    path = lewm_dir / "train.py"
    text = path.read_text(encoding="utf-8")
    changed = False
    if "import signal" not in text:
        text = text.replace("import os\n", "import os\nimport signal\n", 1)
        changed = True
    shim = '''\nfor _sig_name in ("SIGUSR1", "SIGUSR2", "SIGCONT"):\n    if not hasattr(signal, _sig_name):\n        setattr(signal, _sig_name, signal.SIGTERM)\n\n'''
    if "SIGUSR1" not in text.split("def lejepa_forward", 1)[0]:
        text = text.replace("\ndef lejepa_forward", shim + "\ndef lejepa_forward", 1)
        changed = True
    old_reward_normalizer_skip = '''            if col.startswith("pixels"):
                continue
'''
    new_reward_normalizer_skip = '''            if col.startswith("pixels") or col == "reward":
                continue
'''
    if old_reward_normalizer_skip in text:
        text = text.replace(old_reward_normalizer_skip, new_reward_normalizer_skip, 1)
        changed = True
    old_losses_dict = '''    losses_dict = {f"{stage}/{k}": v.detach() for k, v in output.items() if "loss" in k}
'''
    new_losses_dict = '''    losses_dict = {
        f"{stage}/{k}": v.detach()
        for k, v in output.items()
        if "loss" in k or k.startswith("geo_")
    }
'''
    if old_losses_dict in text:
        text = text.replace(old_losses_dict, new_losses_dict, 1)
        changed = True
    old_loss = '''    # LeWM loss\n    output["pred_loss"] = (pred_emb - tgt_emb).pow(2).mean()\n    output["sigreg_loss"]= self.sigreg(emb.transpose(0, 1))\n    output["loss"] = output["pred_loss"] + lambd * output["sigreg_loss"]  \n'''
    if OLD_B2_PRIME_VALUE_LOSS in text:
        text = text.replace(OLD_B2_PRIME_VALUE_LOSS, B2_VALUE_LOSS, 1)
        changed = True
    elif OLD_B2_VALUE_LOSS in text:
        text = text.replace(OLD_B2_VALUE_LOSS, B2_VALUE_LOSS, 1)
        changed = True
    elif "value_cfg = cfg.loss.get(\"value\", {})" not in text:
        if old_loss not in text:
            raise RuntimeError("Could not locate LeWM loss block for B2 value patch")
        text = text.replace(old_loss, B2_VALUE_LOSS, 1)
        changed = True
    elif (
        "geo_cfg = cfg.loss.get(\"geo\", {})" not in text
        or "geo_state = ctx_emb.detach().reshape" in text
        or "geo_target = str(geo_cfg.get" not in text
        or "enable_flash_sdp(False)" not in text
        or "geo_resolution = str(geo_cfg.get" not in text
        or "geo_alpha_loss" not in text
        or "geo_alpha\")" not in text
        or "teacher_weight" not in text
        or "geo_teacher_loss" not in text
        or "with torch.enable_grad()" not in text
        or "dynmetric" not in text
        or "geo_dynmetric_loss" not in text
    ):
        start = text.find("    # LeWM + optional")
        end = text.find("    losses_dict =", start)
        if start < 0 or end < 0:
            raise RuntimeError("Could not locate patched LeWM loss block for geometry patch")
        text = text[:start] + B2_VALUE_LOSS + "\n" + text[end:]
        changed = True
    elif "# LeWM + optional goal-conditioned rollout value loss." not in text:
        start = text.find("    # LeWM + optional")
        end = text.find("    losses_dict =", start)
        if start < 0 or end < 0:
            raise RuntimeError("Could not locate patched LeWM value-loss block")
        text = text[:start] + B2_VALUE_LOSS + "\n" + text[end:]
        changed = True
    old_world_model_init = '''    world_model = hydra.utils.instantiate(cfg.model)
'''
    new_world_model_init = '''    world_model = hydra.utils.instantiate(cfg.model)
    init_model_path = cfg.get("init_model_path")
    if init_model_path:
        pretrained = swm.wm.utils.load_pretrained(
            init_model_path,
            cache_dir=os.environ.get("STABLEWM_HOME", os.environ.get("LOCAL_DATASET_DIR")),
        )
        incompatible = world_model.load_state_dict(pretrained.state_dict(), strict=False)
        print(
            f"Initialized world_model from {init_model_path}; "
            f"missing={list(incompatible.missing_keys)}, unexpected={list(incompatible.unexpected_keys)}",
            flush=True,
        )
    geo_cfg = cfg.loss.get("geo", {})
    teacher_model_path = geo_cfg.get("teacher_model_path") if geo_cfg else None
    if teacher_model_path:
        geo_teacher = swm.wm.utils.load_pretrained(
            teacher_model_path,
            cache_dir=os.environ.get("STABLEWM_HOME", os.environ.get("LOCAL_DATASET_DIR")),
        )
        geo_teacher.eval().requires_grad_(False)
        object.__setattr__(world_model, "geo_teacher", geo_teacher)
        print(f"Loaded frozen geo teacher from {teacher_model_path}", flush=True)
'''
    if old_world_model_init in text and "init_model_path = cfg.get(\"init_model_path\")" not in text:
        text = text.replace(old_world_model_init, new_world_model_init, 1)
        changed = True
    elif "teacher_model_path = geo_cfg.get(\"teacher_model_path\")" not in text:
        insert_after = '''        print(
            f"Initialized world_model from {init_model_path}; "
            f"missing={list(incompatible.missing_keys)}, unexpected={list(incompatible.unexpected_keys)}",
            flush=True,
        )
'''
        teacher_load = '''        print(
            f"Initialized world_model from {init_model_path}; "
            f"missing={list(incompatible.missing_keys)}, unexpected={list(incompatible.unexpected_keys)}",
            flush=True,
        )
    geo_cfg = cfg.loss.get("geo", {})
    teacher_model_path = geo_cfg.get("teacher_model_path") if geo_cfg else None
    if teacher_model_path:
        geo_teacher = swm.wm.utils.load_pretrained(
            teacher_model_path,
            cache_dir=os.environ.get("STABLEWM_HOME", os.environ.get("LOCAL_DATASET_DIR")),
        )
        geo_teacher.eval().requires_grad_(False)
        object.__setattr__(world_model, "geo_teacher", geo_teacher)
        print(f"Loaded frozen geo teacher from {teacher_model_path}", flush=True)
'''
        if insert_after in text:
            text = text.replace(insert_after, teacher_load, 1)
            changed = True
    if changed:
        path.write_text(text, encoding="utf-8")
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lewm-dir", required=True)
    args = parser.parse_args()
    lewm_dir = Path(args.lewm_dir)
    if not (lewm_dir / "train.py").is_file() or not (lewm_dir / "utils.py").is_file():
        raise SystemExit(f"Not a LeWM source directory: {lewm_dir}")
    changed = []
    if patch_utils(lewm_dir):
        changed.append("utils.py")
    if patch_train(lewm_dir):
        changed.append("train.py")
    if patch_module(lewm_dir):
        changed.append("module.py")
    if patch_jepa(lewm_dir):
        changed.append("jepa.py")
    if patch_b1_model_config(lewm_dir):
        changed.append("config/train/model/lewm_b1_effect.yaml")
    if patch_resolution_a_model_config(lewm_dir):
        changed.append("config/train/model/lewm_b1_effect_resa.yaml")
    if patch_b2_model_config(lewm_dir):
        changed.append("config/train/model/lewm_b2_effect_value.yaml")
    if patch_b2_prime_model_config(lewm_dir):
        changed.append("config/train/model/lewm_b2_prime_effect_value.yaml")
    if patch_b3_goal_value_model_config(lewm_dir):
        changed.append("config/train/model/lewm_b3_goal_value.yaml")
    print("Patched " + ", ".join(changed) if changed else "LeWM compatibility patches already present")


if __name__ == "__main__":
    main()
