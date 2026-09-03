import math
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import vit as custom_vit
from .experts import LoRAExpert


class HiDeLoRAModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        task_num: int,
        pretrained: bool = True,
        num_lora_layers: int = 5,
        sdlora_rank: int = 4,
        sdlora_alpha: float = 16.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.task_num = task_num
        assert hasattr(custom_vit, backbone_name), f"Unsupported backbone {backbone_name} for HiDe-LoRA"
        self.backbone = getattr(custom_vit, backbone_name)(pretrained=pretrained, num_classes=num_classes)
        # freeze all ViT backbone parameters; only LoRA and heads are trainable
        for _, p in self.backbone.named_parameters():
            p.requires_grad = False
        D = self.backbone.num_features
        self.feature_dim = D
        # per-task LoRA expert on first num_lora_layers blocks, LoRA config follows SDLoRA
        self.lora_expert = LoRAExpert(
            num_experts=task_num,
            embed_dim=D,
            num_lora_layers=num_lora_layers,
            lora_rank=sdlora_rank,
            lora_alpha=sdlora_alpha,
        )
        self.prompt_head = nn.Linear(D, num_classes)
        self.g_mlp = nn.Sequential(nn.Linear(D, D), nn.ReLU(inplace=True))
        self.g_head = nn.Linear(D, num_classes)
        # statistics for prompt branch (backbone features)
        self.register_buffer("p_count", torch.zeros(num_classes))
        self.register_buffer("p_sum", torch.zeros(num_classes, D))
        self.register_buffer("p_sum_xxT", torch.zeros(num_classes, D, D))
        # statistics for gating branch (g_mlp features)
        self.register_buffer("g_count", torch.zeros(num_classes))
        self.register_buffer("g_sum", torch.zeros(num_classes, D))
        self.register_buffer("g_sum_xxT", torch.zeros(num_classes, D, D))

    def _get_stats(self, branch: str):
        if branch == "prompt":
            return self.p_count, self.p_sum, self.p_sum_xxT
        elif branch == "gate":
            return self.g_count, self.g_sum, self.g_sum_xxT
        else:
            raise ValueError(f"Unknown branch {branch}")

    @torch.no_grad()
    def _update_stats(self, feat: torch.Tensor, labels: torch.Tensor, branch: str) -> None:
        counts, sums, sums_xxT = self._get_stats(branch)
        labels = labels.long()
        unique_labels = labels.unique()
        for c in unique_labels.tolist():
            mask = labels == c
            f_c = feat[mask]
            if f_c.numel() == 0:
                continue
            n_c = f_c.size(0)
            sums[c] += f_c.sum(dim=0)
            counts[c] += n_c
            sums_xxT[c] += f_c.t() @ f_c

    def _build_task_ids(self, B: int, task_id: Union[int, torch.Tensor], device) -> torch.Tensor:
        if isinstance(task_id, int):
            tids = torch.full((B,), task_id, device=device, dtype=torch.long)
        elif isinstance(task_id, torch.Tensor):
            if task_id.dim() == 0:
                tids = task_id.view(1).expand(B).to(device).long()
            else:
                assert task_id.shape[0] == B, "task_id tensor must have shape [B]"
                tids = task_id.to(device).long()
        else:
            raise TypeError("task_id must be int or 1D/0D tensor")
        return tids.clamp(0, self.task_num - 1)

    def forward_prompt(
        self,
        x: torch.Tensor,
        task_id: Union[int, torch.Tensor],
        labels: Optional[torch.Tensor] = None,
        update_stats: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = x.size(0)
        tids = self._build_task_ids(B, task_id, x.device)
        feat = self.lora_expert(self.backbone, x, tids)
        logit = self.prompt_head(feat)
        if update_stats and labels is not None:
            self._update_stats(feat.detach(), labels, branch="prompt")
        return logit, feat

    def forward_gate(
        self,
        x: torch.Tensor,
        detach_backbone: bool = True,
        labels: Optional[torch.Tensor] = None,
        update_stats: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # gate branch always uses frozen backbone without LoRA
        feats_seq = self.backbone.forward_features(x)
        feat = feats_seq[:, 0]
        if detach_backbone:
            feat = feat.detach()
        h = self.g_mlp(feat)
        logit = self.g_head(h)
        if update_stats and labels is not None:
            self._update_stats(h.detach(), labels, branch="gate")
        return logit, h

    def compute_orth_loss(
        self,
        feat: torch.Tensor,
        old_class_mask: torch.Tensor,
        branch: str,
    ) -> torch.Tensor:
        counts, sums, _ = self._get_stats(branch)
        device = feat.device
        counts = counts.to(device)
        sums = sums.to(device)
        old_class_mask = old_class_mask.to(device)
        valid = (counts > 0) & old_class_mask
        idx = valid.nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            return torch.tensor(0.0, device=device)
        proto = sums[idx] / counts[idx].unsqueeze(-1)
        proto = F.normalize(proto, dim=-1)
        proj = feat @ proto.t()
        loss = (proj ** 2).mean()
        return loss

    def sample_features_for_ca(
        self,
        branch: str,
        num_per_class: int = 10,
        device: Optional[torch.device] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        counts, sums, sums_xxT = self._get_stats(branch)
        if device is None:
            device = counts.device
        counts = counts.to(device)
        sums = sums.to(device)
        sums_xxT = sums_xxT.to(device)
        idx = (counts > 1).nonzero(as_tuple=False).flatten()
        if idx.numel() == 0:
            return None, None
        D = self.feature_dim
        feats = []
        labels = []
        for c in idx.tolist():
            n_c = counts[c]
            mu = sums[c] / n_c
            ExxT = sums_xxT[c] / n_c
            var = ExxT.diagonal(dim1=-2, dim2=-1) - mu * mu
            var = torch.clamp(var, min=1e-5)
            std = torch.sqrt(var)
            samples = mu.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
                num_per_class, D, device=device
            )
            feats.append(samples)
            labels.append(torch.full((num_per_class,), c, device=device, dtype=torch.long))
        feats = torch.cat(feats, dim=0)
        labels = torch.cat(labels, dim=0)
        return feats, labels

