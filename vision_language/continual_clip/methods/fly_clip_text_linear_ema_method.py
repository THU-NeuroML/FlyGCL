import math
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig

from .fly_clip_method import FlyCLIPMethod


class FlyCLIPTextLinearEMAMethod(FlyCLIPMethod):
    """Fly-CLIP variant with online text head + per-expert linear EMA heads.

    - Keeps Fly routing/experts exactly as in FlyCLIPMethod.
    - Uses shared text expert branch as the online CLIP head.
    - Adds per-expert linear online heads and two EMA snapshots per expert.
    - Test-time ensemble combines text-online logits + linear EMA logits.
    """

    def __init__(self, cfg: DictConfig, device: torch.device):
        super().__init__(cfg, device)

        self._linear_head_bias = bool(getattr(cfg, "linear_head_bias", True))
        self._linear_ema_count = max(2, int(getattr(cfg, "linear_ema_count", 2)))
        self._text_online_train_weight = float(getattr(cfg, "text_online_train_weight", 0.7))
        self._text_online_train_weight = max(0.0, min(1.0, self._text_online_train_weight))

        # Reuse Fly's EMA ratios; if fewer than required, repeat the last one.
        ratios = [float(x) for x in list(self.ema_ratio)]
        if len(ratios) == 0:
            ratios = [0.9, 0.99]
        while len(ratios) < self._linear_ema_count:
            ratios.append(float(ratios[-1]))
        self._linear_ema_ratios = ratios[: self._linear_ema_count]

        self.linear_online_heads = nn.ModuleList()
        self.linear_ema_heads = nn.ModuleList()  # List[ModuleList[nn.Linear]]
        self._linear_out_dim = 0

        self._aux_info.update({
            "fly_variant": "clip_text_linear_ema_v1",
            "linear_ema_per_expert": int(self._linear_ema_count),
            "text_online_train_weight": float(self._text_online_train_weight),
            "linear_head_bias": int(self._linear_head_bias),
            "linear_ema_ratios": [float(x) for x in self._linear_ema_ratios],
            "use_online_expert_for_eval": int(self._use_online_expert_for_eval),
        })

    def _make_linear_head(self, out_dim: int, trainable: bool) -> nn.Linear:
        head = nn.Linear(self._text_embed_dim, int(out_dim), bias=self._linear_head_bias).to(self.device).float()
        for p in head.parameters():
            p.requires_grad_(bool(trainable))
        return head

    def _expand_linear_head(self, old_head: nn.Linear, new_out_dim: int, trainable: bool) -> nn.Linear:
        old_out_dim = int(old_head.out_features)
        if int(new_out_dim) <= old_out_dim:
            for p in old_head.parameters():
                p.requires_grad_(bool(trainable))
            return old_head

        new_head = self._make_linear_head(out_dim=int(new_out_dim), trainable=trainable)
        with torch.no_grad():
            new_head.weight[:old_out_dim].copy_(old_head.weight)
            if old_head.bias is not None and new_head.bias is not None:
                new_head.bias[:old_out_dim].copy_(old_head.bias)
        return new_head

    def _ensure_linear_capacity(self, required_out_dim: int) -> None:
        required_out_dim = max(1, int(required_out_dim))
        if required_out_dim <= int(self._linear_out_dim):
            return

        if int(self._linear_out_dim) == 0:
            self._linear_out_dim = required_out_dim
            return

        for expert_idx in range(len(self.linear_online_heads)):
            self.linear_online_heads[expert_idx] = self._expand_linear_head(
                self.linear_online_heads[expert_idx],
                new_out_dim=required_out_dim,
                trainable=any(p.requires_grad for p in self.linear_online_heads[expert_idx].parameters()),
            )

            group = self.linear_ema_heads[expert_idx]
            for ema_idx in range(len(group)):
                group[ema_idx] = self._expand_linear_head(
                    group[ema_idx],
                    new_out_dim=required_out_dim,
                    trainable=False,
                )

        self._linear_out_dim = required_out_dim

    def _ensure_linear_group(self, expert_id: int) -> None:
        while len(self.linear_online_heads) <= int(expert_id):
            new_expert_id = len(self.linear_online_heads)
            online_head = self._make_linear_head(out_dim=self._linear_out_dim, trainable=True)
            ema_group = nn.ModuleList()
            for _ in range(self._linear_ema_count):
                ema_group.append(self._make_linear_head(out_dim=self._linear_out_dim, trainable=False))
            self.linear_online_heads.append(online_head)
            self.linear_ema_heads.append(ema_group)
            self._warmup_linear_online_head(new_expert_id)

    @torch.no_grad()
    def _warmup_linear_online_head(self, expert_id: int) -> None:
        expert_id = int(expert_id)
        if expert_id <= 0 or expert_id >= len(self.linear_online_heads):
            return

        mode = str(getattr(self, "_expert_warmup_mode", "mean_previous_experts"))
        if mode == "previous_session_expert":
            src_heads = [self.linear_online_heads[expert_id - 1]]
        else:
            src_heads = [self.linear_online_heads[i] for i in range(expert_id)]

        dst = self.linear_online_heads[expert_id]
        src_params = [list(h.parameters()) for h in src_heads]
        dst_params = list(dst.parameters())
        for p_idx, p_dst in enumerate(dst_params):
            p_dst.data.copy_(torch.stack([sp[p_idx].data for sp in src_params], dim=0).mean(dim=0))

    @torch.no_grad()
    def _init_linear_ema(self, expert_id: int) -> None:
        if expert_id < 0 or expert_id >= len(self.linear_online_heads):
            return

        online_head = self.linear_online_heads[expert_id]
        group = self.linear_ema_heads[expert_id]
        for ema_head in group:
            for p_ema, p_online in zip(ema_head.parameters(), online_head.parameters()):
                p_ema.data.copy_(p_online.data)

    @torch.no_grad()
    def _update_linear_ema(self, expert_id: Optional[int] = None) -> None:
        if len(self.linear_online_heads) == 0:
            return

        if expert_id is None:
            expert_id = int(self.current_task)
        expert_id = max(0, min(int(expert_id), len(self.linear_online_heads) - 1))

        online_head = self.linear_online_heads[expert_id]
        group = self.linear_ema_heads[expert_id]
        for ema_idx, ema_head in enumerate(group):
            ratio = float(self._linear_ema_ratios[min(ema_idx, len(self._linear_ema_ratios) - 1)])
            for p_ema, p_online in zip(ema_head.parameters(), online_head.parameters()):
                p_ema.data.mul_(ratio).add_(p_online.data, alpha=1.0 - ratio)

    def on_optimizer_step(self) -> None:
        super().on_optimizer_step()
        self._update_linear_ema()

    def adaptation(self, task_id: int, reset: bool = False) -> None:
        super().adaptation(task_id, reset=reset)

        if self.classes_names is not None:
            required_out_dim = int(len(self.classes_names))
        else:
            required_out_dim = int(self._num_seen_classes)

        required_out_dim = max(required_out_dim, int(self._num_seen_classes), 1)
        self._ensure_linear_capacity(required_out_dim)

        active_expert = max(0, min(self.current_task, self.task_num - 1))
        self._ensure_linear_group(active_expert)

        # Keep only the active expert's linear online head trainable.
        for idx, head in enumerate(self.linear_online_heads):
            trainable = bool(idx == active_expert)
            for p in head.parameters():
                p.requires_grad_(trainable)

        self._init_linear_ema(active_expert)

        self._aux_info.update({
            "fly_variant": "clip_text_linear_ema_v1",
            "active_expert": int(active_expert),
            "linear_ema_per_expert": int(self._linear_ema_count),
            "linear_out_dim": int(self._linear_out_dim),
            "text_online_train_weight": float(self._text_online_train_weight),
            "linear_ema_ratios": [float(x) for x in self._linear_ema_ratios],
            "use_online_expert_for_eval": int(self._use_online_expert_for_eval),
        })

    def _compute_linear_logits(self, image_features: torch.Tensor, head: nn.Linear) -> torch.Tensor:
        logits = head(image_features.float())
        return logits[:, : self._num_seen_classes]

    def _forward_with_expert(
        self,
        image: torch.Tensor,
        q_features: torch.Tensor,
        expert_ids: torch.Tensor,
        train: bool,
    ) -> torch.Tensor:
        cls = self._extract_expert_features(image, q_features, expert_ids, train=train)
        image_features = F.normalize(self._project_visual(cls), dim=-1)

        text_logits = self._compute_clip_logits(cls)

        if len(self.linear_online_heads) == 0:
            return text_logits

        if expert_ids.numel() > 0:
            active_expert = int(expert_ids[0].item())
        else:
            active_expert = max(0, min(self.current_task, self.task_num - 1))
        active_expert = max(0, min(active_expert, len(self.linear_online_heads) - 1))

        linear_logits = self._compute_linear_logits(image_features, self.linear_online_heads[active_expert])
        w_text = float(self._text_online_train_weight)
        return w_text * text_logits + (1.0 - w_text) * linear_logits

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
            logits_list.append(self._current_logit_scale() * image_features @ online_text.T)

        logits_list.extend(self._compute_routed_text_ema_logits(image_features, expert_ids))

        if len(self.linear_ema_heads) > 0:
            for ema_idx in range(self._linear_ema_count):
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
                    expert_idx = max(0, min(int(eid), len(self.linear_ema_heads) - 1))
                    head = self.linear_ema_heads[expert_idx][ema_idx]
                    ema_logits[idxs] = self._compute_linear_logits(image_features[idxs], head)
                logits_list.append(ema_logits)

        if not logits_list:
            raise RuntimeError(
                "No FlyCLIP linear-EMA eval logits are available. Set "
                "use_online_expert_for_eval=true, or enable use_text_ema=true "
                "and/or create linear EMA heads before evaluation."
            )
        return logits_list

    def forward(self, image: torch.Tensor, test: bool = False, all_test: bool = False) -> torch.Tensor:
        logits = super().forward(image, test=test, all_test=all_test)
        self._aux_info.update({
            "method": "fly",
            "fly_variant": "clip_text_linear_ema_v1",
            "text_online_shared": 1,
            "linear_ema_per_expert": int(self._linear_ema_count),
            "text_online_train_weight": float(self._text_online_train_weight),
            "linear_ema_ratios": [float(x) for x in self._linear_ema_ratios],
            "use_online_expert_for_eval": int(self._use_online_expert_for_eval),
        })
        return logits
