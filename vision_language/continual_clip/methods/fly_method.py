import math
from typing import Iterable, List, Optional

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from ..utils import get_class_ids_per_task, get_class_names
from .base import CLMethod
from .clip_text_adapter import CLIPTextTransformerAdapter
from .clip_vit_adapter import CLIPViTAdapter
from .prompt_utils import parse_prompt_modalities, resolve_prompt_layers


class _FlyPrompt(nn.Module):
    """Prompt bank with one prompt set per expert and selected ViT layers."""

    def __init__(
        self,
        num_experts: int,
        len_prompt: int,
        embed_dim: int,
        pos_prompt: Iterable[int],
        cls_pos_bias: torch.Tensor,
    ):
        super().__init__()
        self.num_experts = int(num_experts)
        self.len_prompt = int(len_prompt)
        self.embed_dim = int(embed_dim)
        pos_prompt = [int(x) for x in pos_prompt]
        if len(pos_prompt) == 0:
            pos_prompt = [0]

        self.register_buffer("pos_prompt", torch.tensor(pos_prompt, dtype=torch.int64))
        self.num_layers = int(self.pos_prompt.numel())
        self.prompts = nn.Parameter(
            torch.empty(self.num_layers, self.num_experts, self.len_prompt, self.embed_dim)
        )
        nn.init.uniform_(self.prompts)

        self.register_buffer("cls_pos_bias", cls_pos_bias.detach().float())
        self._active_expert_ids: Optional[torch.Tensor] = None
        self.task_count = 0

    def process_task_count(self) -> None:
        self.task_count += 1

    def set_active_experts(self, expert_ids: torch.Tensor) -> None:
        self._active_expert_ids = expert_ids.detach().long()

    def clear_active_experts(self) -> None:
        self._active_expert_ids = None

    @torch.no_grad()
    def init_new_expert(self, expert_id: int, warmup_mode: str = "mean_previous_experts") -> None:
        expert_id = int(expert_id)
        if expert_id <= 0 or expert_id >= self.num_experts:
            return
        if str(warmup_mode) == "previous_session_expert":
            self.prompts.data[:, expert_id] = self.prompts[:, expert_id - 1].clone()
            return
        prev = self.prompts[:, :expert_id].clone()
        self.prompts.data[:, expert_id] = prev.mean(dim=1)

    def _resolve_expert_ids(self, batch_size: int, device: torch.device, task_id: Optional[int]) -> torch.Tensor:
        if self._active_expert_ids is not None and int(self._active_expert_ids.numel()) == int(batch_size):
            expert_ids = self._active_expert_ids.to(device=device)
        else:
            default_task = int(task_id) if task_id is not None else int(self.task_count)
            default_task = max(default_task, 0)
            expert_ids = torch.full((batch_size,), default_task, device=device, dtype=torch.long)
        return expert_ids.clamp(min=0, max=self.num_experts - 1)

    def forward(self, x_querry, l, x_block, train: bool = False, task_id: Optional[int] = None):
        del x_querry, train
        layer_matches = (self.pos_prompt == int(l)).nonzero(as_tuple=False).flatten()
        if int(layer_matches.numel()) == 0:
            return None, x_block.new_zeros(1), x_block

        local_layer_idx = int(layer_matches[0].item())
        batch_size = int(x_block.shape[0])
        expert_ids = self._resolve_expert_ids(batch_size, x_block.device, task_id)

        p = self.prompts[local_layer_idx][expert_ids]
        p = p + self.cls_pos_bias.to(dtype=p.dtype, device=p.device).view(1, 1, -1)
        return [p, p], x_block.new_zeros(1), x_block


class _FlyAdapterBank(nn.Module):
    """Per-expert bottleneck adapters on CLS features."""

    def __init__(self, num_experts: int, embed_dim: int, down_dim: int, dropout: float = 0.0, scale: float = 1.0):
        super().__init__()
        self.num_experts = int(num_experts)
        self.embed_dim = int(embed_dim)
        self.down_dim = max(1, int(down_dim))
        self.scale = float(scale)

        self.adapters = nn.ModuleList()
        for _ in range(self.num_experts):
            adapter = nn.Sequential(
                nn.LayerNorm(self.embed_dim),
                nn.Linear(self.embed_dim, self.down_dim),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(self.down_dim, self.embed_dim),
                nn.Dropout(float(dropout)),
            )
            # Keep adapter as an identity residual at initialization.
            nn.init.zeros_(adapter[4].weight)
            nn.init.zeros_(adapter[4].bias)
            self.adapters.append(adapter)

    @torch.no_grad()
    def init_new_expert(self, expert_id: int, warmup_mode: str = "mean_previous_experts") -> None:
        expert_id = int(expert_id)
        if expert_id <= 0 or expert_id >= self.num_experts:
            return

        state = self.adapters[expert_id].state_dict()
        if str(warmup_mode) == "previous_session_expert":
            prev_states = [self.adapters[expert_id - 1].state_dict()]
        else:
            prev_states = [self.adapters[i].state_dict() for i in range(expert_id)]
        for key in state.keys():
            state[key].copy_(torch.stack([prev[key] for prev in prev_states], dim=0).mean(dim=0))
        self.adapters[expert_id].load_state_dict(state)

    def set_scale(self, scale: float) -> None:
        self.scale = float(scale)

    def forward(self, features: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        expert_ids = expert_ids.long().clamp(min=0, max=self.num_experts - 1)
        out = features.clone()
        for eid in expert_ids.unique().tolist():
            idxs = (expert_ids == int(eid)).nonzero(as_tuple=True)[0]
            if int(idxs.numel()) == 0:
                continue
            base = features[idxs]
            delta = self.adapters[int(eid)](base)
            out[idxs] = base + self.scale * delta
        return out


def _resolve_block_adapter_layers(cfg: DictConfig, num_layers: int) -> List[int]:
    raw_layers = getattr(cfg, "fly_adapter_layers", None)
    if raw_layers is not None:
        layers = sorted({int(x) for x in list(raw_layers)})
        return [idx for idx in layers if 0 <= idx < int(num_layers)]

    count = max(1, int(getattr(cfg, "fly_adapter_layer_count", 6)))
    end = int(num_layers)
    start = max(0, end - count)
    return list(range(start, end))


class _FlyBlockAdapterBank(nn.Module):
    """Per-expert bottleneck adapters injected into selected ViT blocks."""

    def __init__(
        self,
        num_experts: int,
        embed_dim: int,
        down_dim: int,
        selected_layers: Iterable[int],
        dropout: float = 0.0,
        scale: float = 1.0,
        apply_mode: str = "all_tokens",
    ):
        super().__init__()
        self.num_experts = int(num_experts)
        self.embed_dim = int(embed_dim)
        self.down_dim = max(1, int(down_dim))
        self.selected_layers = [int(x) for x in selected_layers]
        self.layer_to_local = {layer_idx: i for i, layer_idx in enumerate(self.selected_layers)}
        self.scale = float(scale)
        self.apply_mode = str(apply_mode).lower()
        if self.apply_mode not in {"all_tokens", "cls"}:
            raise ValueError("fly_adapter_apply must be one of ['all_tokens', 'cls']")

        self.experts = nn.ModuleList()
        for _ in range(self.num_experts):
            layers = nn.ModuleList()
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
                layers.append(adapter)
            self.experts.append(layers)

    @torch.no_grad()
    def init_new_expert(self, expert_id: int, warmup_mode: str = "mean_previous_experts") -> None:
        expert_id = int(expert_id)
        if expert_id <= 0 or expert_id >= self.num_experts:
            return

        if str(warmup_mode) == "previous_session_expert":
            prev_states = [self.experts[expert_id - 1].state_dict()]
        else:
            prev_states = [self.experts[i].state_dict() for i in range(expert_id)]

        state = self.experts[expert_id].state_dict()
        for key in state.keys():
            state[key].copy_(torch.stack([prev[key] for prev in prev_states], dim=0).mean(dim=0))
        self.experts[expert_id].load_state_dict(state)

    def set_scale(self, scale: float) -> None:
        self.scale = float(scale)

    def forward_block(self, x: torch.Tensor, layer_idx: int, expert_id: int) -> torch.Tensor:
        local_idx = self.layer_to_local.get(int(layer_idx))
        if local_idx is None:
            return x

        expert_id = max(0, min(int(expert_id), self.num_experts - 1))
        adapter = self.experts[expert_id][local_idx]

        if self.apply_mode == "cls":
            cls = x[0]
            x = x.clone()
            x[0] = cls + self.scale * adapter(cls)
            return x

        seq_len, batch_size, dim = x.shape
        flat = x.permute(1, 0, 2).reshape(batch_size * seq_len, dim)
        delta = adapter(flat).reshape(batch_size, seq_len, dim).permute(1, 0, 2)
        return x + self.scale * delta


class _FlyLoRABlock(nn.Module):
    """LoRA parameters for one ViT block and one expert."""

    def __init__(self, embed_dim: int, rank: int, alpha: float, only_kv: bool = True):
        super().__init__()
        self.embed_dim = int(embed_dim)
        self.rank = max(1, int(rank))
        self.alpha = float(alpha)
        self.scaling = self.alpha / float(self.rank)
        self.only_kv = bool(only_kv)

        self.k_a = nn.Linear(self.embed_dim, self.rank, bias=False)
        self.k_b = nn.Linear(self.rank, self.embed_dim, bias=False)
        self.v_a = nn.Linear(self.embed_dim, self.rank, bias=False)
        self.v_b = nn.Linear(self.rank, self.embed_dim, bias=False)
        if self.only_kv:
            self.q_a = None
            self.q_b = None
            self.out_a = None
            self.out_b = None
        else:
            self.q_a = nn.Linear(self.embed_dim, self.rank, bias=False)
            self.q_b = nn.Linear(self.rank, self.embed_dim, bias=False)
            self.out_a = nn.Linear(self.embed_dim, self.rank, bias=False)
            self.out_b = nn.Linear(self.rank, self.embed_dim, bias=False)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in [self.k_a, self.v_a, self.q_a, self.out_a]:
            if module is not None:
                nn.init.kaiming_uniform_(module.weight, a=math.sqrt(5.0))
        for module in [self.k_b, self.v_b, self.q_b, self.out_b]:
            if module is not None:
                nn.init.zeros_(module.weight)

    def _delta_weight(self, a: Optional[nn.Linear], b: Optional[nn.Linear]) -> Optional[torch.Tensor]:
        if a is None or b is None:
            return None
        return (b.weight @ a.weight) * self.scaling

    def q_delta(self) -> Optional[torch.Tensor]:
        return self._delta_weight(self.q_a, self.q_b)

    def k_delta(self) -> torch.Tensor:
        return self._delta_weight(self.k_a, self.k_b)

    def v_delta(self) -> torch.Tensor:
        return self._delta_weight(self.v_a, self.v_b)

    def out_delta(self) -> Optional[torch.Tensor]:
        return self._delta_weight(self.out_a, self.out_b)


class _FlyLoRABank(nn.Module):
    """Per-expert LoRA inserted into every visual transformer attention block."""

    def __init__(
        self,
        num_experts: int,
        num_layers: int,
        embed_dim: int,
        rank: int,
        alpha: float,
        only_kv: bool = True,
    ):
        super().__init__()
        self.num_experts = int(num_experts)
        self.num_layers = int(num_layers)
        self.embed_dim = int(embed_dim)
        self.rank = max(1, int(rank))
        self.alpha = float(alpha)
        self.only_kv = bool(only_kv)

        self.experts = nn.ModuleList()
        for _ in range(self.num_experts):
            blocks = nn.ModuleList()
            for _ in range(self.num_layers):
                blocks.append(
                    _FlyLoRABlock(
                        embed_dim=self.embed_dim,
                        rank=self.rank,
                        alpha=self.alpha,
                        only_kv=self.only_kv,
                    )
                )
            self.experts.append(blocks)

    @torch.no_grad()
    def init_new_expert(self, expert_id: int, warmup_mode: str = "mean_previous_experts") -> None:
        expert_id = int(expert_id)
        if expert_id <= 0 or expert_id >= self.num_experts:
            return

        if str(warmup_mode) == "previous_session_expert":
            prev_states = [self.experts[expert_id - 1].state_dict()]
        else:
            prev_states = [self.experts[i].state_dict() for i in range(expert_id)]

        state = self.experts[expert_id].state_dict()
        for key in state.keys():
            state[key].copy_(torch.stack([prev[key] for prev in prev_states], dim=0).mean(dim=0))
        self.experts[expert_id].load_state_dict(state)

    def forward_block(
        self,
        blk: nn.Module,
        x: torch.Tensor,
        layer_idx: int,
        expert_id: int,
        prompt=None,
    ) -> torch.Tensor:
        expert_id = max(0, min(int(expert_id), self.num_experts - 1))
        layer_idx = max(0, min(int(layer_idx), self.num_layers - 1))
        lora_block = self.experts[expert_id][layer_idx]

        attn_mask = blk.attn_mask
        if attn_mask is not None:
            attn_mask = attn_mask.to(dtype=x.dtype, device=x.device)

        x_ln = blk.ln_1(x)
        key = x_ln
        value = x_ln
        if prompt is not None:
            p_key, p_value = prompt
            p_key = p_key.permute(1, 0, 2)
            p_value = p_value.permute(1, 0, 2)
            key = torch.cat([p_key, x_ln], dim=0)
            value = torch.cat([p_value, x_ln], dim=0)

            if attn_mask is not None:
                n_prompt = p_key.shape[0]
                pad = attn_mask.new_zeros(attn_mask.shape[0], n_prompt)
                attn_mask = torch.cat([pad, attn_mask], dim=1)

        attn = blk.attn
        if getattr(attn, "in_proj_weight", None) is None:
            q_weight = attn.q_proj_weight
            k_weight = attn.k_proj_weight
            v_weight = attn.v_proj_weight
        else:
            q_weight, k_weight, v_weight = attn.in_proj_weight.chunk(3, dim=0)

        q_delta = lora_block.q_delta()
        out_delta = lora_block.out_delta()
        if q_delta is not None:
            q_weight = q_weight + q_delta.to(dtype=q_weight.dtype, device=q_weight.device)
        k_weight = k_weight + lora_block.k_delta().to(dtype=k_weight.dtype, device=k_weight.device)
        v_weight = v_weight + lora_block.v_delta().to(dtype=v_weight.dtype, device=v_weight.device)
        out_weight = attn.out_proj.weight
        if out_delta is not None:
            out_weight = out_weight + out_delta.to(dtype=out_weight.dtype, device=out_weight.device)

        attn_out = F.multi_head_attention_forward(
            query=x_ln,
            key=key,
            value=value,
            embed_dim_to_check=attn.embed_dim,
            num_heads=attn.num_heads,
            in_proj_weight=None,
            in_proj_bias=attn.in_proj_bias,
            bias_k=attn.bias_k,
            bias_v=attn.bias_v,
            add_zero_attn=attn.add_zero_attn,
            dropout_p=attn.dropout,
            out_proj_weight=out_weight,
            out_proj_bias=attn.out_proj.bias,
            training=attn.training,
            key_padding_mask=None,
            need_weights=False,
            attn_mask=attn_mask,
            use_separate_proj_weight=True,
            q_proj_weight=q_weight,
            k_proj_weight=k_weight,
            v_proj_weight=v_weight,
        )[0]

        x = x + attn_out
        x = x + blk.mlp(blk.ln_2(x))
        return x


class _RPFC(nn.Module):
    """Closed-form random projection classifier used as expert router."""

    def __init__(self, M: int, ridge: float, embed_dim: int, num_classes: int):
        super().__init__()
        self.ridge = float(ridge)
        self.embed_dim = int(embed_dim)
        self.num_classes = int(num_classes)

        if int(M) <= 0:
            self.M = self.embed_dim
            self.use_rp = False
            self.register_buffer("W_rand", torch.empty(0))
            self.register_buffer("Q", torch.zeros(self.embed_dim, self.num_classes))
            self.register_buffer("G", torch.zeros(self.embed_dim, self.embed_dim))
        else:
            self.M = int(M)
            self.use_rp = True
            self.register_buffer("W_rand", torch.randn(self.embed_dim, self.M))
            self.register_buffer("Q", torch.zeros(self.M, self.num_classes))
            self.register_buffer("G", torch.zeros(self.M, self.M))

        self.fc = nn.Linear(self.M, self.num_classes, bias=False)
        for param in self.parameters():
            param.requires_grad = False

    def _apply(self, fn):
        super()._apply(fn)
        # G/Q are accumulation statistics and are not needed in router forward.
        # Keep them on CPU to save GPU memory during expert/text training.
        self.Q = self.Q.cpu()
        self.G = self.G.cpu()
        return self

    def collect(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        features = features.detach()
        labels = labels.detach().long()
        features_h = F.relu(features @ self.W_rand) if self.use_rp else features
        targets = F.one_hot(labels, num_classes=self.num_classes).float()
        features_h_cpu = features_h.to(device=self.G.device, dtype=self.G.dtype)
        targets_cpu = targets.to(device=self.Q.device, dtype=self.Q.dtype)
        self.Q.add_(features_h_cpu.T @ targets_cpu)
        # Avoid materializing an extra M x M temporary. With rp_dim=10000 this
        # saves ~380MB peak CUDA memory while keeping the same accumulated G.
        for h in features_h_cpu:
            self.G.addr_(h, h)

    @torch.no_grad()
    def update(self) -> None:
        device = self.fc.weight.device
        G = self.G.to(device=device, non_blocking=True)
        Q = self.Q.to(device=device, non_blocking=True)
        eye = torch.eye(self.M, device=device, dtype=G.dtype)
        try:
            Wo = torch.linalg.solve(G + self.ridge * eye, Q).T
        except RuntimeError:
            inv = torch.linalg.pinv(G + self.ridge * eye)
            Wo = (inv @ Q).T
        self.fc.weight.data.copy_(Wo.to(device=device, dtype=self.fc.weight.dtype))
        del G, Q, eye, Wo
        if device.type == "cuda":
            torch.cuda.empty_cache()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_rp:
            x = F.relu(x @ self.W_rand)
        return self.fc(x)


class FlyMethod(CLMethod):
    """Fly-style experts + RP router with CLIP text-similarity classification."""

    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.classes_names = None
        self.prompt_template = str(getattr(cfg, "prompt_template", "a photo of a {}."))

        clip_model, self.transforms = clip.load(cfg.model_name, device=device)
        clip_model = clip_model.float()
        for p in clip_model.parameters():
            p.requires_grad = False

        self.clip_model = clip_model
        self.feat = CLIPViTAdapter(clip_model.visual)
        self.text_feat = CLIPTextTransformerAdapter(clip_model)
        self._feat_dim = int(clip_model.visual.transformer.width)
        self._text_feat_dim = int(clip_model.transformer.width)
        self.task_num = max(int(getattr(cfg, "task_num", 1)), 1)
        self.fly_mode = str(getattr(cfg, "fly_mode", "prompt")).lower()
        if self.fly_mode not in {"prompt", "adapter", "adapter_block", "lora"}:
            raise ValueError("fly_mode must be one of ['prompt', 'adapter', 'adapter_block', 'lora']")

        self.ema_ratio = [float(x) for x in list(getattr(cfg, "ema_ratio", [0.9, 0.99]))]
        self.num_ema = len(self.ema_ratio)
        self.ensemble_method = str(getattr(cfg, "ensemble_method", "softmax_max_prob")).lower()
        self._optimizer_steps = 0

        self._adapter_scale_target = float(getattr(cfg, "fly_adapter_scale", 1.0))
        self._adapter_warmup_steps = max(0, int(getattr(cfg, "fly_adapter_warmup_steps", 0)))
        adapter_init_scale = self._adapter_scale_target if self._adapter_warmup_steps <= 0 else 0.0
        self._expert_warmup_mode = self._resolve_expert_warmup_mode(
            getattr(cfg, "fly_expert_warmup_mode", "mean_previous_experts")
        )
        self.prompt_modalities = parse_prompt_modalities(cfg)
        self.use_vision_prompt = "vision" in self.prompt_modalities
        self.use_text_prompt = ("text" in self.prompt_modalities) and self.fly_mode == "prompt"

        self.prompt: Optional[_FlyPrompt] = None
        self.text_prompt: Optional[_FlyPrompt] = None
        self.adapter_bank: Optional[_FlyAdapterBank] = None
        self.block_adapter_bank: Optional[_FlyBlockAdapterBank] = None
        self.lora_bank: Optional[_FlyLoRABank] = None

        if self.fly_mode == "prompt":
            pos_prompt = resolve_prompt_layers(cfg, "pos_prompt", len(self.feat._blocks), default_layers=[0, 1, 2, 3, 4])
            self.prompt = _FlyPrompt(
                num_experts=self.task_num,
                len_prompt=int(getattr(cfg, "len_prompt", 20)),
                embed_dim=self._feat_dim,
                pos_prompt=pos_prompt,
                cls_pos_bias=clip_model.visual.positional_embedding[0],
            ).to(device)
            if self.use_text_prompt:
                text_pos_prompt = resolve_prompt_layers(
                    cfg,
                    "text_pos_prompt",
                    len(self.text_feat._blocks),
                    default_layers=pos_prompt,
                )
                text_bias = torch.zeros(self._text_feat_dim, device=device)
                self.text_prompt = _FlyPrompt(
                    num_experts=self.task_num,
                    len_prompt=int(getattr(cfg, "len_prompt", 20)),
                    embed_dim=self._text_feat_dim,
                    pos_prompt=text_pos_prompt,
                    cls_pos_bias=text_bias,
                ).to(device)
        elif self.fly_mode == "adapter":
            down_dim = int(
                getattr(cfg, "fly_adapter_down_dim", 2 * int(getattr(cfg, "sdlora_rank", 4)))
            )
            self.adapter_bank = _FlyAdapterBank(
                num_experts=self.task_num,
                embed_dim=self._feat_dim,
                down_dim=max(1, down_dim),
                dropout=float(getattr(cfg, "fly_adapter_dropout", 0.0)),
                scale=adapter_init_scale,
            ).to(device)
        elif self.fly_mode == "adapter_block":
            down_dim = int(
                getattr(cfg, "fly_adapter_down_dim", 2 * int(getattr(cfg, "sdlora_rank", 4)))
            )
            selected_layers = _resolve_block_adapter_layers(cfg, len(self.feat._blocks))
            self.block_adapter_bank = _FlyBlockAdapterBank(
                num_experts=self.task_num,
                embed_dim=self._feat_dim,
                down_dim=max(1, down_dim),
                selected_layers=selected_layers,
                dropout=float(getattr(cfg, "fly_adapter_dropout", 0.0)),
                scale=adapter_init_scale,
                apply_mode=str(getattr(cfg, "fly_adapter_apply", "all_tokens")),
            ).to(device)
        else:
            rank = int(getattr(cfg, "fly_lora_rank", int(getattr(cfg, "sdlora_rank", 4))))
            lora_mode = str(getattr(cfg, "lora_mode", "vision+only_kv+text")).lower()
            self.lora_bank = _FlyLoRABank(
                num_experts=self.task_num,
                num_layers=len(self.feat._blocks),
                embed_dim=self._feat_dim,
                rank=max(1, rank),
                alpha=float(getattr(cfg, "fly_lora_alpha", 16.0)),
                only_kv=("only_kv" in lora_mode),
            ).to(device)

        self.rp_head = _RPFC(
            M=int(getattr(cfg, "rp_dim", 10000)),
            ridge=float(getattr(cfg, "rp_ridge", 1e4)),
            embed_dim=self._feat_dim,
            num_classes=self.task_num,
        ).to(device)

        self.class_ids_per_task = list(get_class_ids_per_task(cfg))
        self.current_class_names: List[str] = []
        self.current_task = -1
        self._num_seen_classes = 0
        self._text_features: Optional[torch.Tensor] = None
        self._text_tokens: Optional[torch.Tensor] = None

        self._aux_info = {"method": "fly", "fly_mode": self.fly_mode}
        self._rp_collect_samples_total = 0
        self._ema_updates = 0
        self._rp_dirty = False  # True when G/Q have been updated but fc weights have not

    def _resolve_expert_warmup_mode(self, raw_mode) -> str:
        mode = str(raw_mode).strip().lower()
        aliases = {
            "mean": "mean_previous_experts",
            "avg": "mean_previous_experts",
            "average": "mean_previous_experts",
            "mean_previous_experts": "mean_previous_experts",
            "previous": "previous_session_expert",
            "prev": "previous_session_expert",
            "previous_session": "previous_session_expert",
            "previous_session_expert": "previous_session_expert",
            "last_session_expert": "previous_session_expert",
        }
        if mode not in aliases:
            raise ValueError(
                "fly_expert_warmup_mode must be one of "
                "['mean_previous_experts', 'previous_session_expert']"
            )
        return aliases[mode]

    def on_optimizer_step(self) -> None:
        self._optimizer_steps += 1
        if self.fly_mode == "adapter" and self.adapter_bank is not None:
            if self._adapter_warmup_steps > 0:
                ratio = min(1.0, float(self._optimizer_steps) / float(self._adapter_warmup_steps))
                self.adapter_bank.set_scale(self._adapter_scale_target * ratio)
            else:
                self.adapter_bank.set_scale(self._adapter_scale_target)
        if self.fly_mode == "adapter_block" and self.block_adapter_bank is not None:
            if self._adapter_warmup_steps > 0:
                ratio = min(1.0, float(self._optimizer_steps) / float(self._adapter_warmup_steps))
                self.block_adapter_bank.set_scale(self._adapter_scale_target * ratio)
            else:
                self.block_adapter_bank.set_scale(self._adapter_scale_target)
        return None

    def adaptation(self, task_id: int, reset: bool = False) -> None:
        del reset
        self.current_task += 1
        active_expert = max(0, min(self.current_task, self.task_num - 1))

        task_class_names = get_class_names(self.classes_names, self.class_ids_per_task[task_id])
        self.current_class_names += task_class_names
        self._num_seen_classes = len(self.current_class_names)

        if self.current_task > 0:
            if self.fly_mode == "prompt" and self.prompt is not None:
                self.prompt.process_task_count()
                self.prompt.init_new_expert(active_expert, warmup_mode=self._expert_warmup_mode)
                if self.text_prompt is not None:
                    self.text_prompt.process_task_count()
                    self.text_prompt.init_new_expert(active_expert, warmup_mode=self._expert_warmup_mode)
            elif self.fly_mode == "adapter" and self.adapter_bank is not None:
                self.adapter_bank.init_new_expert(active_expert, warmup_mode=self._expert_warmup_mode)
            elif self.fly_mode == "adapter_block" and self.block_adapter_bank is not None:
                self.block_adapter_bank.init_new_expert(active_expert, warmup_mode=self._expert_warmup_mode)
            elif self.fly_mode == "lora" and self.lora_bank is not None:
                self.lora_bank.init_new_expert(active_expert, warmup_mode=self._expert_warmup_mode)

        self._aux_info = {
            "method": "fly",
            "fly_mode": self.fly_mode,
            "task_id": int(self.current_task),
            "seen_classes": int(self._num_seen_classes),
            "seen_experts": int(min(self.current_task + 1, self.task_num)),
            "active_expert": int(active_expert),
            "ensemble_method": self.ensemble_method,
            "rp_dim": int(getattr(self.cfg, "rp_dim", 10000)),
            "rp_ridge": float(getattr(self.cfg, "rp_ridge", 1e4)),
            "fly_expert_warmup_mode": self._expert_warmup_mode,
            "prompt_modalities": "+".join(sorted(self.prompt_modalities)),
            "prompt_inject": "attention_kv_prefix",
            "prompt_visual_layers": int(len(self.feat._blocks)) if self.fly_mode == "prompt" and self.use_vision_prompt else 0,
            "prompt_text_layers": int(len(self.text_feat._blocks)) if self.use_text_prompt else 0,
        }
        if self.fly_mode == "adapter":
            self._aux_info.update({
                "fly_adapter_down_dim": int(getattr(self.cfg, "fly_adapter_down_dim", 2 * int(getattr(self.cfg, "sdlora_rank", 4)))),
                "fly_adapter_dropout": float(getattr(self.cfg, "fly_adapter_dropout", 0.0)),
                "fly_adapter_scale": float(self.adapter_bank.scale) if self.adapter_bank is not None else float(getattr(self.cfg, "fly_adapter_scale", 1.0)),
                "fly_adapter_scale_target": float(self._adapter_scale_target),
                "fly_adapter_warmup_steps": int(self._adapter_warmup_steps),
                "optimizer_steps": int(self._optimizer_steps),
            })
        if self.fly_mode == "adapter_block":
            selected_layers = self.block_adapter_bank.selected_layers if self.block_adapter_bank is not None else []
            self._aux_info.update({
                "fly_adapter_down_dim": int(getattr(self.cfg, "fly_adapter_down_dim", 2 * int(getattr(self.cfg, "sdlora_rank", 4)))),
                "fly_adapter_dropout": float(getattr(self.cfg, "fly_adapter_dropout", 0.0)),
                "fly_adapter_scale": float(self.block_adapter_bank.scale) if self.block_adapter_bank is not None else float(getattr(self.cfg, "fly_adapter_scale", 1.0)),
                "fly_adapter_scale_target": float(self._adapter_scale_target),
                "fly_adapter_warmup_steps": int(self._adapter_warmup_steps),
                "fly_adapter_insert": "visual_transformer_block_post_mlp",
                "fly_adapter_apply": str(getattr(self.cfg, "fly_adapter_apply", "all_tokens")),
                "fly_adapter_layers": [int(x) for x in selected_layers],
                "optimizer_steps": int(self._optimizer_steps),
            })
        if self.fly_mode == "lora":
            self._aux_info.update({
                "fly_lora_rank": int(getattr(self.cfg, "fly_lora_rank", int(getattr(self.cfg, "sdlora_rank", 4)))),
                "fly_lora_alpha": float(getattr(self.cfg, "fly_lora_alpha", 16.0)),
                "fly_lora_insert": "visual_transformer_attention",
                "fly_lora_layers": int(len(self.feat._blocks)),
                "fly_lora_only_kv": int("only_kv" in str(getattr(self.cfg, "lora_mode", "vision+only_kv+text")).lower()),
            })

        self._refresh_text_features()

    @torch.no_grad()
    def _refresh_text_features(self) -> None:
        if not self.current_class_names:
            self._text_features = None
            self._text_tokens = None
            return
        tokens = clip.tokenize(
            [self.prompt_template.format(c) for c in self.current_class_names]
        ).to(self.device)
        self._text_tokens = tokens
        if self.use_text_prompt:
            self._text_features = None
            return
        text_features = self.clip_model.encode_text(tokens).float()
        self._text_features = F.normalize(text_features, dim=-1).detach()

    def _project_text(self, text_tokens: torch.Tensor, token_ids: torch.Tensor) -> torch.Tensor:
        eot_idx = token_ids.argmax(dim=-1)
        text_features = text_tokens[torch.arange(text_tokens.shape[0], device=text_tokens.device), eot_idx]
        return text_features.float() @ self.clip_model.text_projection.float()

    def _encode_text_features_for_expert(self, train: bool, expert_id: int) -> torch.Tensor:
        if not self.use_text_prompt:
            if self._text_features is None:
                raise RuntimeError("Text features are unavailable. Call adaptation() before forward().")
            return self._text_features
        if self._text_tokens is None:
            raise RuntimeError("Text tokens are unavailable. Call adaptation() before forward().")
        if self.text_prompt is None:
            raise RuntimeError("Text prompt is enabled but no text_prompt module was created.")

        expert_id = max(0, min(int(expert_id), self.task_num - 1))
        text_experts = torch.full(
            (self._text_tokens.shape[0],),
            expert_id,
            device=self.device,
            dtype=torch.long,
        )
        with torch.no_grad():
            query_tokens, _ = self.text_feat(self._text_tokens)
            eot_idx = self._text_tokens.argmax(dim=-1)
            q = query_tokens[torch.arange(query_tokens.shape[0], device=query_tokens.device), eot_idx]

        self.text_prompt.set_active_experts(text_experts)
        try:
            with torch.set_grad_enabled(bool(train)):
                text_tokens, _ = self.text_feat(
                    self._text_tokens,
                    prompt=self.text_prompt,
                    q=q,
                    train=bool(train),
                    task_id=self.current_task,
                )
                text_features = self._project_text(text_tokens, self._text_tokens)
                return F.normalize(text_features, dim=-1)
        finally:
            self.text_prompt.clear_active_experts()

    def _project_visual(self, cls: torch.Tensor) -> torch.Tensor:
        if (
            hasattr(self.clip_model, "visual")
            and hasattr(self.clip_model.visual, "proj")
            and self.clip_model.visual.proj is not None
        ):
            return cls.float() @ self.clip_model.visual.proj.float()
        return cls.float()

    def _compute_clip_logits_for_text_features(self, cls: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        image_features = F.normalize(self._project_visual(cls), dim=-1)
        scale = self.clip_model.logit_scale.exp().clamp(max=100.0)
        return scale * image_features @ text_features.T

    def _compute_clip_logits(self, cls: torch.Tensor, train: bool = False, expert_id: int = None) -> torch.Tensor:
        if self.use_text_prompt:
            eid = self.current_task if expert_id is None else int(expert_id)
            text_features = self._encode_text_features_for_expert(train=bool(train), expert_id=eid)
        else:
            if self._text_features is None:
                raise RuntimeError("Text features are unavailable. Call adaptation() before forward().")
            text_features = self._text_features
        return self._compute_clip_logits_for_text_features(cls, text_features)

    def _extract_query_features(self, image: torch.Tensor) -> torch.Tensor:
        tokens, _ = self.feat(image)
        return tokens[:, 0, :]

    def _extract_prompted_features(
        self,
        image: torch.Tensor,
        q_features: torch.Tensor,
        expert_ids: torch.Tensor,
        train: bool,
    ) -> torch.Tensor:
        if self.prompt is None:
            raise RuntimeError("Prompt extractor is unavailable when fly_mode != 'prompt'.")
        if self.use_vision_prompt:
            self.prompt.set_active_experts(expert_ids)
        try:
            tokens, _ = self.feat(
                image,
                prompt=(self.prompt if self.use_vision_prompt else None),
                q=q_features,
                train=bool(train),
                task_id=self.current_task,
            )
        finally:
            if self.use_vision_prompt:
                self.prompt.clear_active_experts()
        return tokens[:, 0, :]

    def _extract_expert_features(
        self,
        image: torch.Tensor,
        q_features: torch.Tensor,
        expert_ids: torch.Tensor,
        train: bool,
    ) -> torch.Tensor:
        if self.fly_mode == "prompt":
            return self._extract_prompted_features(image, q_features, expert_ids, train=bool(train))
        if self.fly_mode == "adapter":
            if self.adapter_bank is None:
                raise RuntimeError("Adapter bank is unavailable for fly_mode='adapter'.")
            return self.adapter_bank(q_features, expert_ids)
        if self.fly_mode == "adapter_block":
            if self.block_adapter_bank is None:
                raise RuntimeError("Block adapter bank is unavailable for fly_mode='adapter_block'.")
            expert_ids = expert_ids.long().clamp(min=0, max=self.task_num - 1)
            out = q_features.clone()
            for eid in expert_ids.unique().tolist():
                idxs = (expert_ids == int(eid)).nonzero(as_tuple=True)[0]
                if int(idxs.numel()) == 0:
                    continue
                tokens, _ = self.feat(
                    image[idxs],
                    adapter_bank=self.block_adapter_bank,
                    expert_id=int(eid),
                )
                out[idxs] = tokens[:, 0, :]
            return out
        if self.fly_mode == "lora":
            if self.lora_bank is None:
                raise RuntimeError("LoRA bank is unavailable for fly_mode='lora'.")
            expert_ids = expert_ids.long().clamp(min=0, max=self.task_num - 1)
            out = q_features.clone()
            for eid in expert_ids.unique().tolist():
                idxs = (expert_ids == int(eid)).nonzero(as_tuple=True)[0]
                if int(idxs.numel()) == 0:
                    continue
                tokens, _ = self.feat(
                    image[idxs],
                    lora_bank=self.lora_bank,
                    expert_id=int(eid),
                )
                out[idxs] = tokens[:, 0, :]
            return out
        raise ValueError(f"Unsupported fly_mode: {self.fly_mode}")

    def _forward_with_expert(
        self,
        image: torch.Tensor,
        q_features: torch.Tensor,
        expert_ids: torch.Tensor,
        train: bool,
    ) -> torch.Tensor:
        cls = self._extract_expert_features(image, q_features, expert_ids, train=train)
        if not self.use_text_prompt:
            return self._compute_clip_logits(cls, train=train)
        logits = cls.new_empty((cls.shape[0], self._num_seen_classes))
        expert_ids = expert_ids.long().clamp(min=0, max=self.task_num - 1)
        for eid in expert_ids.unique().tolist():
            idxs = (expert_ids == int(eid)).nonzero(as_tuple=True)[0]
            if int(idxs.numel()) == 0:
                continue
            logits[idxs] = self._compute_clip_logits(cls[idxs], train=train, expert_id=int(eid))
        return logits

    def forward_with_rp(self, image: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            q = self._extract_query_features(image)
        return self.rp_head(q)

    def _forward_with_ema_logits(
        self,
        image: torch.Tensor,
        q_features: torch.Tensor,
        expert_ids: torch.Tensor,
    ) -> List[torch.Tensor]:
        cls = self._extract_expert_features(image, q_features, expert_ids, train=False)
        if not self.use_text_prompt:
            return [self._compute_clip_logits(cls, train=False)]
        logits = cls.new_empty((cls.shape[0], self._num_seen_classes))
        expert_ids = expert_ids.long().clamp(min=0, max=self.task_num - 1)
        for eid in expert_ids.unique().tolist():
            idxs = (expert_ids == int(eid)).nonzero(as_tuple=True)[0]
            if int(idxs.numel()) == 0:
                continue
            logits[idxs] = self._compute_clip_logits(cls[idxs], train=False, expert_id=int(eid))
        return [logits]

    @torch.no_grad()
    def infer_with_expert_ids(self, image: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        """Inference helper for analysis: bypass RP routing and force expert ids.

        Returns logits over currently seen classes using the same EMA ensemble
        path as the default test-time forward.
        """
        if self._text_features is None and self._text_tokens is None:
            raise RuntimeError("FlyMethod.adaptation() must be called before inference.")

        q = self._extract_query_features(image)
        logits_list = self._forward_with_ema_logits(image, q, expert_ids)
        logits = self._ensemble_logits(logits_list)
        return logits[:, : self._num_seen_classes]

    @torch.no_grad()
    def infer_with_expert_id(self, image: torch.Tensor, expert_id: int) -> torch.Tensor:
        """Inference helper for analysis: force a single expert for a whole batch."""
        eid = int(expert_id)
        expert_ids = torch.full((image.size(0),), eid, device=image.device, dtype=torch.long)
        return self.infer_with_expert_ids(image, expert_ids)

    def _ensemble_logits(self, logits_list: List[torch.Tensor]) -> torch.Tensor:
        if len(logits_list) == 1:
            return logits_list[0]

        method = self.ensemble_method
        to_stack = logits_list
        if "softmax" in method:
            to_stack = [torch.softmax(logits, dim=-1) for logits in to_stack]

        stacked = torch.stack(to_stack, dim=-1)
        if "mean" in method:
            return stacked.mean(dim=-1)
        if "max_prob" in method:
            return stacked.max(dim=-1)[0]
        if "min_entropy" in method:
            entropies = -torch.sum(stacked * torch.log(stacked + 1e-8), dim=1)
            pick = torch.argmin(entropies, dim=-1)
            batch_ids = torch.arange(stacked.size(0), device=stacked.device)
            return stacked[batch_ids, :, pick]
        raise ValueError(f"Unknown ensemble_method: {self.ensemble_method}")

    def _routing_stats(self, expert_ids: torch.Tensor, seen_experts: int):
        counts = torch.bincount(expert_ids.detach().cpu(), minlength=int(seen_experts)).float()
        total = float(counts.sum().item())
        if total <= 0:
            return 0.0, []
        probs = counts / total
        nz = probs[probs > 0]
        entropy = float((-(nz * torch.log2(nz))).sum().item())
        topk = min(3, int(counts.numel()))
        vals, ids = torch.topk(counts, k=topk, largest=True, sorted=True)
        top = []
        for idx, val in zip(ids.tolist(), vals.tolist()):
            if int(val) <= 0:
                continue
            top.append({"expert": int(idx), "count": int(val)})
        return entropy, top

    def forward(self, image: torch.Tensor, test: bool = False, all_test: bool = False) -> torch.Tensor:
        del all_test
        if self._text_features is None and self._text_tokens is None:
            raise RuntimeError("FlyMethod.adaptation() must be called before forward().")

        with torch.no_grad():
            q = self._extract_query_features(image)

        if test:
            with torch.no_grad():
                # Mirror official online_evaluate(): call rp_head.update() whenever
                # new G/Q data has been collected since the last solve.
                if self._rp_dirty:
                    self.rp_head.update()
                    self._rp_dirty = False
                seen_experts = max(1, min(self.current_task + 1, self.task_num))
                rp_logits = self.rp_head(q)[:, :seen_experts]
                expert_ids = torch.argmax(rp_logits, dim=-1)

                logits_list = self._forward_with_ema_logits(image, q, expert_ids)
                logits = self._ensemble_logits(logits_list)
                route_entropy, route_top = self._routing_stats(expert_ids, seen_experts)
                self._aux_info.update({
                    "method": "fly",
                    "fly_mode": self.fly_mode,
                    "task_id": int(self.current_task),
                    "seen_experts": int(seen_experts),
                    "route_entropy": route_entropy,
                    "routed_expert_top": route_top,
                    "rp_samples_accum": int(self._rp_collect_samples_total),
                    "ema_updates": int(self._ema_updates),
                })
                if self.fly_mode == "adapter" and self.adapter_bank is not None:
                    self._aux_info.update({
                        "fly_adapter_scale": float(self.adapter_bank.scale),
                        "fly_adapter_scale_target": float(self._adapter_scale_target),
                        "fly_adapter_warmup_steps": int(self._adapter_warmup_steps),
                        "optimizer_steps": int(self._optimizer_steps),
                    })
                if self.fly_mode == "adapter_block" and self.block_adapter_bank is not None:
                    self._aux_info.update({
                        "fly_adapter_scale": float(self.block_adapter_bank.scale),
                        "fly_adapter_scale_target": float(self._adapter_scale_target),
                        "fly_adapter_warmup_steps": int(self._adapter_warmup_steps),
                        "fly_adapter_layers": [int(x) for x in self.block_adapter_bank.selected_layers],
                        "optimizer_steps": int(self._optimizer_steps),
                    })
                return logits[:, :self._num_seen_classes]

        active_expert = max(0, min(self.current_task, self.task_num - 1))
        expert_ids = torch.full((image.size(0),), active_expert, device=image.device, dtype=torch.long)
        logits = self._forward_with_expert(image, q, expert_ids, train=True)

        with torch.no_grad():
            rp_labels = torch.full((q.size(0),), active_expert, device=q.device, dtype=torch.long)
            self.rp_head.collect(q, rp_labels)
            self._rp_collect_samples_total += int(q.size(0))
            self._rp_dirty = True

        self._aux_info.update({
            "method": "fly",
            "fly_mode": self.fly_mode,
            "task_id": int(self.current_task),
            "active_expert": int(active_expert),
            "rp_samples_accum": int(self._rp_collect_samples_total),
            "ema_updates": int(self._ema_updates),
        })
        if self.fly_mode == "adapter" and self.adapter_bank is not None:
            self._aux_info.update({
                "fly_adapter_scale": float(self.adapter_bank.scale),
                "fly_adapter_scale_target": float(self._adapter_scale_target),
                "fly_adapter_warmup_steps": int(self._adapter_warmup_steps),
                "optimizer_steps": int(self._optimizer_steps),
            })
        if self.fly_mode == "adapter_block" and self.block_adapter_bank is not None:
            self._aux_info.update({
                "fly_adapter_scale": float(self.block_adapter_bank.scale),
                "fly_adapter_scale_target": float(self._adapter_scale_target),
                "fly_adapter_warmup_steps": int(self._adapter_warmup_steps),
                "fly_adapter_layers": [int(x) for x in self.block_adapter_bank.selected_layers],
                "optimizer_steps": int(self._optimizer_steps),
            })
        return logits[:, :self._num_seen_classes]

    def collect(self, image: torch.Tensor, labels: torch.Tensor) -> None:
        del labels
        with torch.no_grad():
            q = self._extract_query_features(image)
            active_expert = max(0, min(self.current_task, self.task_num - 1))
            rp_labels = torch.full((q.size(0),), active_expert, device=q.device, dtype=torch.long)
            self.rp_head.collect(q, rp_labels)
            self._rp_collect_samples_total += int(q.size(0))
            self._rp_dirty = True

    def apply_batch_logit_mask(self, logits: torch.Tensor, local_labels: torch.Tensor) -> torch.Tensor:
        """Apply batch-level logit mask: suppress classes not in the current batch.

        Mirrors the official FlyGCL-main online_train() batch mask:
            logit_mask = torch.zeros_like(self.mask) - torch.inf
            for cc in torch.unique(y): logit_mask[cc] = 0
            logit += logit_mask
        Here local_labels are already remapped to [0, num_seen_classes).
        """
        mask = torch.full_like(logits, float("-inf"))
        for c in torch.unique(local_labels):
            mask[:, c.long()] = 0.0
        return logits + mask

    def update(self) -> None:
        self.rp_head.update()
        self._rp_dirty = False

    def after_task(self, train_loader=None) -> None:
        del train_loader
        self.rp_head.update()

    def auxiliary_loss(self):
        return None

    def auxiliary_info(self):
        return dict(self._aux_info)
