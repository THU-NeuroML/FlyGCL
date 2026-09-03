import math
from typing import List, Optional

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from .fly_method import FlyMethod


def _cfg_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y", "on"}


class _TextAdapterExpert(nn.Module):
    """Shared text-side expert operating on frozen CLIP text embeddings."""

    def __init__(self, embed_dim: int, down_dim: int, dropout: float = 0.0, scale: float = 1.0):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.down_dim = max(1, int(down_dim))
        self.scale = float(scale)

        self.net = nn.Sequential(
            nn.LayerNorm(self.embed_dim),
            nn.Linear(self.embed_dim, self.down_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.down_dim, self.embed_dim),
            nn.Dropout(float(dropout)),
        )
        # Keep text adapter as an identity residual at initialization.
        nn.init.zeros_(self.net[4].weight)
        nn.init.zeros_(self.net[4].bias)

    def set_scale(self, scale: float) -> None:
        self.scale = float(scale)

    def forward(self, base_text_features: torch.Tensor) -> torch.Tensor:
        delta = self.net(base_text_features)
        return F.normalize(base_text_features + self.scale * delta, dim=-1)


class _TextPromptExpert(nn.Module):
    """Prompt-like text expert using a learnable context vector in CLIP embedding space."""

    def __init__(self, embed_dim: int, prompt_len: int, scale: float = 1.0):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.prompt_len = max(1, int(prompt_len))
        self.scale = float(scale)

        self.prompt = nn.Parameter(torch.empty(self.prompt_len, self.embed_dim))
        nn.init.uniform_(self.prompt, -0.02, 0.02)

    def forward(self, base_text_features: torch.Tensor) -> torch.Tensor:
        delta = self.prompt.mean(dim=0, keepdim=True)
        return F.normalize(base_text_features + self.scale * delta, dim=-1)


class _TextLoRAExpert(nn.Module):
    """Low-rank text expert operating on frozen CLIP text embeddings."""

    def __init__(self, embed_dim: int, rank: int, alpha: float):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.rank = max(1, int(rank))
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)

        self.lora_a = nn.Linear(self.embed_dim, self.rank, bias=False)
        self.lora_b = nn.Linear(self.rank, self.embed_dim, bias=False)
        nn.init.kaiming_uniform_(self.lora_a.weight, a=math.sqrt(5.0))
        nn.init.zeros_(self.lora_b.weight)

    def forward(self, base_text_features: torch.Tensor) -> torch.Tensor:
        delta = self.lora_b(self.lora_a(base_text_features)) * self.scaling
        return F.normalize(base_text_features + delta, dim=-1)


class FlyCLIPMethod(FlyMethod):
    """Fly V1: vision experts + REAR router + shared text online/EMA experts."""

    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__(cfg, device)

        raw_text_expert_type = str(getattr(cfg, "text_expert_type", "match_vision")).lower()
        if raw_text_expert_type in {"match_vision", "same_as_vision", "auto"}:
            self.text_expert_type = str(self.fly_mode)
        else:
            self.text_expert_type = raw_text_expert_type
        if self.text_expert_type not in {"prompt", "adapter", "lora"}:
            raise ValueError("text_expert_type must be one of ['match_vision', 'prompt', 'adapter', 'lora']")

        self._text_embed_dim = int(self.clip_model.text_projection.shape[1])

        default_down = int(getattr(cfg, "fly_adapter_down_dim", 2 * int(getattr(cfg, "sdlora_rank", 4))))
        self._text_adapter_down_dim = int(getattr(cfg, "text_adapter_down_dim", default_down))
        self._text_adapter_dropout = float(
            getattr(cfg, "text_adapter_dropout", float(getattr(cfg, "fly_adapter_dropout", 0.0)))
        )
        self._text_adapter_scale = float(
            getattr(cfg, "text_adapter_scale", float(getattr(cfg, "fly_adapter_scale", 1.0)))
        )
        self._text_adapter_scale_target = float(self._text_adapter_scale)
        self._text_adapter_warmup_steps = max(
            0,
            int(getattr(cfg, "text_adapter_warmup_steps", int(getattr(cfg, "fly_adapter_warmup_steps", 0)))),
        )
        self._text_adapter_current_scale = (
            self._text_adapter_scale_target if self._text_adapter_warmup_steps <= 0 else 0.0
        )

        self._text_prompt_len = int(getattr(cfg, "text_prompt_len", int(getattr(cfg, "len_prompt", 20))))
        self._text_prompt_scale = float(getattr(cfg, "text_prompt_scale", 1.0))

        default_text_lora_rank = int(getattr(cfg, "fly_lora_rank", int(getattr(cfg, "sdlora_rank", 4))))
        self._text_lora_rank = int(getattr(cfg, "text_lora_rank", default_text_lora_rank))
        self._text_lora_alpha = float(getattr(cfg, "text_lora_alpha", float(getattr(cfg, "fly_lora_alpha", 16.0))))

        self.text_online_expert = self._build_text_expert(trainable=True)

        self.text_ema_experts = nn.ModuleList()
        self._base_text_features: Optional[torch.Tensor] = None
        self._use_online_expert_for_eval = _cfg_bool(
            getattr(cfg, "use_online_expert_for_eval", True)
        )

        self.tune_logit_scale = bool(getattr(cfg, "tune_logit_scale", False))
        if self.tune_logit_scale:
            self.logit_scale = nn.Parameter(self.clip_model.logit_scale.detach().float().clone())
        else:
            self.register_buffer("logit_scale", self.clip_model.logit_scale.detach().float().clone())

        self._aux_info.update({
            "fly_variant": "clip_text_ema_v1",
            "text_expert_type": self.text_expert_type,
            "text_online_shared": 1,
            "text_ema_per_vision_expert": 1,
            "use_online_expert_for_eval": int(self._use_online_expert_for_eval),
            "tune_logit_scale": int(self.tune_logit_scale),
        })
        if self.text_expert_type == "adapter":
            self._aux_info.update({
                "text_adapter_down_dim": int(self._text_adapter_down_dim),
                "text_adapter_dropout": float(self._text_adapter_dropout),
                "text_adapter_scale": float(self._text_adapter_current_scale),
                "text_adapter_scale_target": float(self._text_adapter_scale_target),
                "text_adapter_warmup_steps": int(self._text_adapter_warmup_steps),
                "optimizer_steps": int(self._optimizer_steps),
            })
        elif self.text_expert_type == "prompt":
            self._aux_info.update({
                "text_prompt_len": int(self._text_prompt_len),
                "text_prompt_scale": float(self._text_prompt_scale),
            })
        else:
            self._aux_info.update({
                "text_lora_rank": int(self._text_lora_rank),
                "text_lora_alpha": float(self._text_lora_alpha),
            })

    def _build_text_expert(self, trainable: bool) -> nn.Module:
        if self.text_expert_type == "adapter":
            expert = _TextAdapterExpert(
                embed_dim=self._text_embed_dim,
                down_dim=self._text_adapter_down_dim,
                dropout=self._text_adapter_dropout,
                scale=self._text_adapter_current_scale,
            )
        elif self.text_expert_type == "prompt":
            expert = _TextPromptExpert(
                embed_dim=self._text_embed_dim,
                prompt_len=self._text_prompt_len,
                scale=self._text_prompt_scale,
            )
        else:
            expert = _TextLoRAExpert(
                embed_dim=self._text_embed_dim,
                rank=self._text_lora_rank,
                alpha=self._text_lora_alpha,
            )

        expert = expert.to(self.device)
        for p in expert.parameters():
            p.requires_grad = bool(trainable)
        return expert

    def _new_text_ema_head(self) -> nn.Module:
        return self._build_text_expert(trainable=False)

    def _set_text_adapter_scale(self, scale: float) -> None:
        if self.text_expert_type != "adapter":
            return

        self._text_adapter_current_scale = float(scale)
        if isinstance(self.text_online_expert, _TextAdapterExpert):
            self.text_online_expert.set_scale(self._text_adapter_current_scale)

        for group in self.text_ema_experts:
            for expert in group:
                if isinstance(expert, _TextAdapterExpert):
                    expert.set_scale(self._text_adapter_current_scale)

    def _ensure_text_ema_group(self, expert_id: int) -> None:
        changed = False
        while len(self.text_ema_experts) <= int(expert_id):
            group = nn.ModuleList()
            for _ in range(self.num_ema):
                group.append(self._new_text_ema_head())
            self.text_ema_experts.append(group)
            changed = True

        if changed and self.text_expert_type == "adapter":
            self._set_text_adapter_scale(self._text_adapter_current_scale)

    @torch.no_grad()
    def init_text_ema(self, expert_id: Optional[int] = None) -> None:
        if self.num_ema <= 0:
            return
        expert_id = int(self.current_task if expert_id is None else expert_id)
        if expert_id < 0 or expert_id >= self.task_num:
            return

        self._ensure_text_ema_group(expert_id)
        group = self.text_ema_experts[expert_id]
        for ema_head in group:
            for p_ema, p_online in zip(ema_head.parameters(), self.text_online_expert.parameters()):
                p_ema.data.copy_(p_online.data)

    @torch.no_grad()
    def update_text_ema(self, expert_id: Optional[int] = None) -> None:
        if self.num_ema <= 0:
            return
        expert_id = int(self.current_task if expert_id is None else expert_id)
        expert_id = max(0, min(expert_id, self.task_num - 1))
        if expert_id < 0 or expert_id >= len(self.text_ema_experts):
            return

        group = self.text_ema_experts[expert_id]
        for ema_i, ratio in enumerate(self.ema_ratio):
            ema_head = group[ema_i]
            for p_ema, p_online in zip(ema_head.parameters(), self.text_online_expert.parameters()):
                p_ema.data.mul_(ratio).add_(p_online.data, alpha=1.0 - ratio)
        self._ema_updates += 1

    def on_optimizer_step(self) -> None:
        super().on_optimizer_step()

        if self.text_expert_type == "adapter":
            if self._text_adapter_warmup_steps > 0:
                ratio = min(1.0, float(self._optimizer_steps) / float(self._text_adapter_warmup_steps))
                self._set_text_adapter_scale(self._text_adapter_scale_target * ratio)
            else:
                self._set_text_adapter_scale(self._text_adapter_scale_target)

        self.update_text_ema()

    def adaptation(self, task_id: int, reset: bool = False) -> None:
        super().adaptation(task_id, reset=reset)
        active_expert = max(0, min(self.current_task, self.task_num - 1))
        self._ensure_text_ema_group(active_expert)
        self.init_text_ema(active_expert)
        self._aux_info.update({
            "fly_variant": "clip_text_ema_v1",
            "active_expert": int(active_expert),
            "text_ema_heads": int(self.num_ema),
            "use_online_expert_for_eval": int(self._use_online_expert_for_eval),
        })
        if self.text_expert_type == "adapter":
            self._aux_info.update({
                "text_adapter_scale": float(self._text_adapter_current_scale),
                "text_adapter_scale_target": float(self._text_adapter_scale_target),
                "text_adapter_warmup_steps": int(self._text_adapter_warmup_steps),
                "optimizer_steps": int(self._optimizer_steps),
            })

    @torch.no_grad()
    def _refresh_text_features(self) -> None:
        if not self.current_class_names:
            self._base_text_features = None
            self._text_features = None
            return

        raw_templates = getattr(self.cfg, "prompt_templates", None)
        if raw_templates is not None and len(raw_templates) > 0:
            templates = [str(t) for t in raw_templates]
        else:
            templates = [self.prompt_template]

        all_prompts: List[str] = [
            t.format(c.replace("_", " "))
            for c in self.current_class_names
            for t in templates
        ]
        tokens = clip.tokenize(all_prompts).to(self.device)
        text_features = self.clip_model.encode_text(tokens).float()
        text_features = F.normalize(text_features, dim=-1)

        k = len(self.current_class_names)
        text_features = text_features.view(k, len(templates), -1)
        base = F.normalize(text_features.mean(dim=1), dim=-1).detach()

        self._base_text_features = base
        # Keep parent forward() precondition unchanged.
        self._text_features = base

    def _apply_text_expert(self, expert: nn.Module) -> torch.Tensor:
        if self._base_text_features is None:
            raise RuntimeError("Text base features are unavailable. Call adaptation() before forward().")
        return expert(self._base_text_features)

    def _current_logit_scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)

    def _compute_logits_from_text_features(self, cls: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        image_features = F.normalize(self._project_visual(cls), dim=-1)
        return self._current_logit_scale() * image_features @ text_features.T

    def _compute_clip_logits(self, cls: torch.Tensor, train: Optional[bool] = None) -> torch.Tensor:
        text_features = self._apply_text_expert(self.text_online_expert)
        return self._compute_logits_from_text_features(cls, text_features)

    def _forward_with_ema_logits(
        self,
        image: torch.Tensor,
        q_features: torch.Tensor,
        expert_ids: torch.Tensor,
    ) -> List[torch.Tensor]:
        cls = self._extract_expert_features(image, q_features, expert_ids, train=False)

        logits_list: List[torch.Tensor] = []
        if self._use_online_expert_for_eval:
            online_text = self._apply_text_expert(self.text_online_expert)
            logits_list.append(self._compute_logits_from_text_features(cls, online_text))

        if self.num_ema <= 0 or len(self.text_ema_experts) == 0:
            if not logits_list:
                raise RuntimeError(
                    "No FlyCLIP text-feature eval logits are available. "
                    "Set use_online_expert_for_eval=true or enable text EMA experts."
                )
            return logits_list

        image_features = F.normalize(self._project_visual(cls), dim=-1)
        scale = self._current_logit_scale()

        for ema_i in range(self.num_ema):
            ema_logits = torch.zeros(
                image_features.size(0),
                self._num_seen_classes,
                device=image_features.device,
                dtype=image_features.dtype,
            )
            for eid in expert_ids.unique().tolist():
                idxs = (expert_ids == int(eid)).nonzero(as_tuple=True)[0]
                if int(idxs.numel()) == 0:
                    continue
                expert_idx = max(0, min(int(eid), len(self.text_ema_experts) - 1))
                text_features = self._apply_text_expert(self.text_ema_experts[expert_idx][ema_i])
                ema_logits[idxs] = scale * image_features[idxs] @ text_features.T
            logits_list.append(ema_logits)

        if not logits_list:
            raise RuntimeError(
                "No FlyCLIP text-feature eval logits are available. "
                "Set use_online_expert_for_eval=true or enable text EMA experts."
            )
        return logits_list

    def forward(self, image: torch.Tensor, test: bool = False, all_test: bool = False) -> torch.Tensor:
        logits = super().forward(image, test=test, all_test=all_test)
        self._aux_info.update({
            "method": "fly",
            "fly_variant": "clip_text_ema_v1",
            "text_ema_heads": int(self.num_ema),
            "text_online_shared": 1,
            "use_online_expert_for_eval": int(self._use_online_expert_for_eval),
        })
        if self.text_expert_type == "adapter":
            self._aux_info.update({
                "text_adapter_scale": float(self._text_adapter_current_scale),
                "text_adapter_scale_target": float(self._text_adapter_scale_target),
                "text_adapter_warmup_steps": int(self._text_adapter_warmup_steps),
                "optimizer_steps": int(self._optimizer_steps),
            })
        return logits
