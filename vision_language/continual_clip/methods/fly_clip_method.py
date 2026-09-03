import math
from typing import List, Optional

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from peft.lora.model import build_LoRA_model

from .fly_method import FlyMethod, _resolve_block_adapter_layers


def _cfg_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() not in {"0", "false", "no", "off"}
    return bool(value)


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


class _TextBlockAdapterExpert(nn.Module):
    """Adapter expert injected into selected CLIP text transformer blocks."""

    def __init__(
        self,
        clip_model: nn.Module,
        down_dim: int,
        selected_layers: List[int],
        dropout: float = 0.0,
        scale: float = 1.0,
        apply_mode: str = "all_tokens",
    ):
        super().__init__()
        object.__setattr__(self, "_clip_model", clip_model)
        self.embed_dim = int(clip_model.token_embedding.embedding_dim)
        self.down_dim = max(1, int(down_dim))
        self.selected_layers = [int(x) for x in selected_layers]
        self.layer_to_local = {layer_idx: i for i, layer_idx in enumerate(self.selected_layers)}
        self.scale = float(scale)
        self.apply_mode = str(apply_mode).lower()
        if self.apply_mode not in {"all_tokens", "eot"}:
            raise ValueError("text_adapter_apply must be one of ['all_tokens', 'eot']")

        self.adapters = nn.ModuleList()
        for _ in self.selected_layers:
            adapter = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, self.down_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.down_dim, self.embed_dim),
                nn.Dropout(float(dropout)),
            )
            nn.init.zeros_(adapter[4].weight)
            nn.init.zeros_(adapter[4].bias)
            self.adapters.append(adapter)

    def set_scale(self, scale: float) -> None:
        self.scale = float(scale)

    def _apply_adapter(self, x: torch.Tensor, layer_idx: int, text_tokens: torch.Tensor) -> torch.Tensor:
        local_idx = self.layer_to_local.get(int(layer_idx))
        if local_idx is None:
            return x

        adapter = self.adapters[local_idx]
        if self.apply_mode == "eot":
            x_batch = x.permute(1, 0, 2)
            eot_idx = text_tokens.argmax(dim=-1)
            batch_idx = torch.arange(x_batch.size(0), device=x_batch.device)
            eot = x_batch[batch_idx, eot_idx]
            x_batch = x_batch.clone()
            x_batch[batch_idx, eot_idx] = eot + self.scale * adapter(eot)
            return x_batch.permute(1, 0, 2)

        seq_len, batch_size, dim = x.shape
        flat = x.permute(1, 0, 2).reshape(batch_size * seq_len, dim)
        delta = adapter(flat).reshape(batch_size, seq_len, dim).permute(1, 0, 2)
        return x + self.scale * delta

    def forward(self, text_tokens: torch.Tensor, num_classes: int, num_templates: int) -> torch.Tensor:
        clip_model = self._clip_model
        x = clip_model.token_embedding(text_tokens).float()
        x = x + clip_model.positional_embedding.to(dtype=x.dtype, device=x.device)
        x = x.permute(1, 0, 2)

        for layer_idx, block in enumerate(clip_model.transformer.resblocks):
            x = block(x)
            x = self._apply_adapter(x, layer_idx, text_tokens)

        x = x.permute(1, 0, 2)
        x = clip_model.ln_final(x).float()
        text_features = x[torch.arange(x.shape[0], device=x.device), text_tokens.argmax(dim=-1)]
        text_features = text_features @ clip_model.text_projection.float()
        text_features = F.normalize(text_features, dim=-1)
        text_features = text_features.view(int(num_classes), int(num_templates), -1)
        return F.normalize(text_features.mean(dim=1), dim=-1)


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


class _SeqTextLoRAExpert(nn.Module):
    """Seq-LoRA-aligned text expert inside CLIP's text transformer."""

    def __init__(self, clip_model: nn.Module, rank: int, lora_mode: str, trainable: bool, device: torch.device):
        super().__init__()
        self.rank = max(1, int(rank))
        self.lora_mode = self._text_only_lora_mode(lora_mode)

        state_dict = {k: v.detach().cpu().clone() for k, v in clip_model.state_dict().items()}
        model = build_LoRA_model(state_dict=state_dict, r=self.rank, lora_mode=self.lora_mode)
        self.model = model.to(device).float()

        for name, param in self.model.named_parameters():
            is_trainable_lora = "lora_" in name and "lora_text_projection" not in name
            param.requires_grad_(bool(trainable and is_trainable_lora))

    @staticmethod
    def _text_only_lora_mode(lora_mode: str) -> str:
        parts = [p.strip().lower() for p in str(lora_mode).replace(",", "+").split("+") if p.strip()]
        flags = []
        if "only_kv" in parts:
            flags.append("only_kv")
        if "mlp" in parts:
            flags.append("mlp")
        return "+".join(["text"] + flags)

    def forward(self, text_tokens: torch.Tensor, num_classes: int, num_templates: int) -> torch.Tensor:
        text_features = self.model.encode_text(text_tokens).float()
        text_features = F.normalize(text_features, dim=-1)
        text_features = text_features.view(int(num_classes), int(num_templates), -1)
        return F.normalize(text_features.mean(dim=1), dim=-1)


class FlyCLIPMethod(FlyMethod):
    """Fly V1: vision experts + REAR router + shared text online/EMA experts."""

    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__(cfg, device)

        raw_text_expert_type = str(getattr(cfg, "text_expert_type", "adapter_block")).lower()
        if raw_text_expert_type in {"match_vision", "same_as_vision", "auto"}:
            self.text_expert_type = str(self.fly_mode)
        else:
            self.text_expert_type = raw_text_expert_type
        if self.text_expert_type in {"adapter_final", "feature_adapter", "feature"}:
            self.text_expert_type = "adapter"
        elif self.text_expert_type in {"block_adapter", "transformer_adapter", "transformer_block"}:
            self.text_expert_type = "adapter_block"
        if self.text_expert_type not in {"prompt", "adapter", "adapter_block", "lora"}:
            raise ValueError(
                "text_expert_type must be one of "
                "['adapter_block', 'adapter', 'adapter_final', 'match_vision', 'prompt', 'lora']"
            )

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
        self._text_adapter_apply = str(
            getattr(cfg, "text_adapter_apply", getattr(cfg, "fly_adapter_apply", "all_tokens"))
        ).lower()
        if self._text_adapter_apply == "cls":
            self._text_adapter_apply = "eot"
        raw_text_layers = getattr(cfg, "text_adapter_layers", None)
        if raw_text_layers is not None:
            num_text_layers = len(self.clip_model.transformer.resblocks)
            self._text_adapter_layers = [
                int(x) for x in sorted({int(v) for v in list(raw_text_layers)})
                if 0 <= int(x) < num_text_layers
            ]
        else:
            self._text_adapter_layers = _resolve_block_adapter_layers(
                cfg,
                len(self.clip_model.transformer.resblocks),
            )

        self._text_prompt_len = int(getattr(cfg, "text_prompt_len", int(getattr(cfg, "len_prompt", 20))))
        self._text_prompt_scale = float(getattr(cfg, "text_prompt_scale", 1.0))

        default_text_lora_rank = int(getattr(cfg, "fly_lora_rank", int(getattr(cfg, "sdlora_rank", 4))))
        self._text_lora_rank = int(getattr(cfg, "text_lora_rank", default_text_lora_rank))
        self._text_lora_alpha = float(getattr(cfg, "text_lora_alpha", float(getattr(cfg, "fly_lora_alpha", 16.0))))
        self._text_lora_impl = str(getattr(cfg, "text_lora_impl", "seq")).lower()
        self._seq_text_lora = self.text_expert_type == "lora" and self._text_lora_impl in {
            "seq",
            "seq_lora",
            "transformer",
            "clip",
        }
        self._text_lora_mode = str(getattr(cfg, "text_lora_mode", getattr(cfg, "lora_mode", "vision+only_kv+text")))
        self._use_text_ema = _cfg_bool(getattr(cfg, "use_text_ema", True))
        if self._seq_text_lora and str(getattr(cfg, "method", "")) == "fly_clip_text_linear_ema":
            self._use_text_ema = _cfg_bool(getattr(cfg, "use_text_ema", False))
        self._use_online_expert_for_eval = _cfg_bool(
            getattr(
                cfg,
                "use_online_expert_for_eval",
                getattr(cfg, "eval_use_online_expert", True),
            )
        )

        self.text_online_expert = self._build_text_expert(trainable=True)

        self.text_ema_experts = nn.ModuleList()
        self._base_text_features: Optional[torch.Tensor] = None
        self._text_tokens: Optional[torch.Tensor] = None
        self._num_text_templates = 1

        self.tune_logit_scale = bool(getattr(cfg, "tune_logit_scale", False))
        if self.tune_logit_scale:
            self.logit_scale = nn.Parameter(self.clip_model.logit_scale.detach().float().clone())
        else:
            self.register_buffer("logit_scale", self.clip_model.logit_scale.detach().float().clone())

        self._aux_info.update({
            "fly_variant": "clip_text_ema_v1",
            "text_expert_type": self.text_expert_type,
            "text_online_shared": 1,
            "text_ema_per_vision_expert": int(self._use_text_ema),
            "use_online_expert_for_eval": int(self._use_online_expert_for_eval),
            "tune_logit_scale": int(self.tune_logit_scale),
        })
        if self.text_expert_type in {"adapter", "adapter_block"}:
            self._aux_info.update({
                "text_adapter_down_dim": int(self._text_adapter_down_dim),
                "text_adapter_dropout": float(self._text_adapter_dropout),
                "text_adapter_scale": float(self._text_adapter_current_scale),
                "text_adapter_scale_target": float(self._text_adapter_scale_target),
                "text_adapter_warmup_steps": int(self._text_adapter_warmup_steps),
                "text_adapter_insert": "text_feature_residual"
                if self.text_expert_type == "adapter" else "text_transformer_block_post_mlp",
                "text_adapter_apply": str(self._text_adapter_apply)
                if self.text_expert_type == "adapter_block" else "feature",
                "text_adapter_layers": [int(x) for x in self._text_adapter_layers]
                if self.text_expert_type == "adapter_block" else [],
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
                "text_lora_impl": str(self._text_lora_impl),
                "text_lora_mode": str(_SeqTextLoRAExpert._text_only_lora_mode(self._text_lora_mode))
                if self._seq_text_lora else "feature_residual",
                "use_text_ema": int(self._use_text_ema),
            })

    def _build_text_expert(self, trainable: bool) -> nn.Module:
        if self.text_expert_type == "adapter":
            expert = _TextAdapterExpert(
                embed_dim=self._text_embed_dim,
                down_dim=self._text_adapter_down_dim,
                dropout=self._text_adapter_dropout,
                scale=self._text_adapter_current_scale,
            )
        elif self.text_expert_type == "adapter_block":
            expert = _TextBlockAdapterExpert(
                clip_model=self.clip_model,
                down_dim=self._text_adapter_down_dim,
                selected_layers=self._text_adapter_layers,
                dropout=self._text_adapter_dropout,
                scale=self._text_adapter_current_scale,
                apply_mode=self._text_adapter_apply,
            )
        elif self.text_expert_type == "prompt":
            expert = _TextPromptExpert(
                embed_dim=self._text_embed_dim,
                prompt_len=self._text_prompt_len,
                scale=self._text_prompt_scale,
            )
        elif self._seq_text_lora:
            expert = _SeqTextLoRAExpert(
                clip_model=self.clip_model,
                rank=self._text_lora_rank,
                lora_mode=self._text_lora_mode,
                trainable=trainable,
                device=self.device,
            )
        else:
            expert = _TextLoRAExpert(
                embed_dim=self._text_embed_dim,
                rank=self._text_lora_rank,
                alpha=self._text_lora_alpha,
            )

        expert = expert.to(self.device)
        if not isinstance(expert, _SeqTextLoRAExpert):
            for p in expert.parameters():
                p.requires_grad = bool(trainable)
        return expert

    def _new_text_ema_head(self) -> nn.Module:
        return self._build_text_expert(trainable=False)

    def _set_text_adapter_scale(self, scale: float) -> None:
        if self.text_expert_type not in {"adapter", "adapter_block"}:
            return

        self._text_adapter_current_scale = float(scale)
        if isinstance(self.text_online_expert, (_TextAdapterExpert, _TextBlockAdapterExpert)):
            self.text_online_expert.set_scale(self._text_adapter_current_scale)

        for group in self.text_ema_experts:
            for expert in group:
                if isinstance(expert, (_TextAdapterExpert, _TextBlockAdapterExpert)):
                    expert.set_scale(self._text_adapter_current_scale)

    def _ensure_text_ema_group(self, expert_id: int) -> None:
        if not self._use_text_ema:
            return
        changed = False
        while len(self.text_ema_experts) <= int(expert_id):
            group = nn.ModuleList()
            for _ in range(self.num_ema):
                group.append(self._new_text_ema_head())
            self.text_ema_experts.append(group)
            changed = True

        if changed and self.text_expert_type in {"adapter", "adapter_block"}:
            self._set_text_adapter_scale(self._text_adapter_current_scale)

    @torch.no_grad()
    def init_text_ema(self, expert_id: Optional[int] = None) -> None:
        if self.num_ema <= 0 or not self._use_text_ema:
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
        if self.num_ema <= 0 or not self._use_text_ema:
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

        if self.text_expert_type in {"adapter", "adapter_block"}:
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
        })
        if self.text_expert_type in {"adapter", "adapter_block"}:
            self._aux_info.update({
                "text_adapter_scale": float(self._text_adapter_current_scale),
                "text_adapter_scale_target": float(self._text_adapter_scale_target),
                "text_adapter_warmup_steps": int(self._text_adapter_warmup_steps),
                "text_adapter_insert": "text_feature_residual"
                if self.text_expert_type == "adapter" else "text_transformer_block_post_mlp",
                "text_adapter_apply": str(self._text_adapter_apply)
                if self.text_expert_type == "adapter_block" else "feature",
                "text_adapter_layers": [int(x) for x in self._text_adapter_layers]
                if self.text_expert_type == "adapter_block" else [],
                "optimizer_steps": int(self._optimizer_steps),
            })

    @torch.no_grad()
    def _refresh_text_features(self) -> None:
        if not self.current_class_names:
            self._base_text_features = None
            self._text_features = None
            self._text_tokens = None
            self._num_text_templates = 1
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
        self._text_tokens = tokens
        self._num_text_templates = len(templates)
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
        if isinstance(expert, (_SeqTextLoRAExpert, _TextBlockAdapterExpert)):
            if self._text_tokens is None:
                raise RuntimeError("Text tokens are unavailable. Call adaptation() before forward().")
            return expert(
                self._text_tokens,
                num_classes=len(self.current_class_names),
                num_templates=self._num_text_templates,
            )
        return expert(self._base_text_features)

    def _current_logit_scale(self) -> torch.Tensor:
        return self.logit_scale.exp().clamp(max=100.0)

    def _compute_logits_from_text_features(self, cls: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        image_features = F.normalize(self._project_visual(cls), dim=-1)
        return self._current_logit_scale() * image_features @ text_features.T

    def _compute_clip_logits(self, cls: torch.Tensor) -> torch.Tensor:
        text_features = self._apply_text_expert(self.text_online_expert)
        return self._compute_logits_from_text_features(cls, text_features)

    def _compute_routed_text_ema_logits(
        self,
        image_features: torch.Tensor,
        expert_ids: torch.Tensor,
    ) -> List[torch.Tensor]:
        logits_list: List[torch.Tensor] = []
        if self.num_ema <= 0 or len(self.text_ema_experts) == 0:
            return logits_list

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
        return logits_list

    def _forward_with_ema_logits(
        self,
        image: torch.Tensor,
        q_features: torch.Tensor,
        expert_ids: torch.Tensor,
    ) -> List[torch.Tensor]:
        cls = self._extract_expert_features(image, q_features, expert_ids, train=False)
        image_features = F.normalize(self._project_visual(cls), dim=-1)
        logits_list: List[torch.Tensor] = []

        if self._use_online_expert_for_eval:
            online_text = self._apply_text_expert(self.text_online_expert)
            logits_list.append(
                self._current_logit_scale() * image_features @ online_text.T
            )

        logits_list.extend(self._compute_routed_text_ema_logits(image_features, expert_ids))
        if not logits_list:
            raise RuntimeError(
                "No FlyCLIP eval logits are available. Set use_online_expert_for_eval=true "
                "or enable use_text_ema=true so EMA text experts are created."
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
        if self.text_expert_type in {"adapter", "adapter_block"}:
            self._aux_info.update({
                "text_adapter_scale": float(self._text_adapter_current_scale),
                "text_adapter_scale_target": float(self._text_adapter_scale_target),
                "text_adapter_warmup_steps": int(self._text_adapter_warmup_steps),
                "text_adapter_insert": "text_feature_residual"
                if self.text_expert_type == "adapter" else "text_transformer_block_post_mlp",
                "text_adapter_apply": str(self._text_adapter_apply)
                if self.text_expert_type == "adapter_block" else "feature",
                "text_adapter_layers": [int(x) for x in self._text_adapter_layers]
                if self.text_expert_type == "adapter_block" else [],
                "optimizer_steps": int(self._optimizer_steps),
            })
        return logits
