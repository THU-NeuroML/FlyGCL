from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import vit as custom_vit
from .ranpac import Adapter


class AdapterExpert(nn.Module):
    def __init__(
        self,
        num_experts: int,
        embed_dim: int,
        num_adapter_layers: int,
        adapter_down_dim: int,
        adapter_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_experts = num_experts
        self.num_adapter_layers = num_adapter_layers
        self.adapters = nn.ModuleList(
            [
                nn.ModuleList([
                    Adapter(adapter_down_dim, embed_dim, dropout=adapter_dropout)
                    for _ in range(num_experts)
                ])
                for _ in range(num_adapter_layers)
            ]
        )

    def _forward_with_adapter_layers(self, backbone: nn.Module, x: torch.Tensor, expert_id: int) -> torch.Tensor:
        for idx, block in enumerate(backbone.blocks):
            if idx < self.num_adapter_layers:
                x_norm = block.norm1(x)
                attn_out = block.attn(x_norm)
                attn_out = block.ls1(attn_out)
                attn_out = block.drop_path1(attn_out)
                x = x + attn_out
                residual = x
                adapt_x = self.adapters[idx][expert_id](x)
                mlp_out = block.mlp(block.norm2(x))
                mlp_out = block.ls2(mlp_out)
                mlp_out = block.drop_path2(mlp_out)
                x = residual + adapt_x + mlp_out
            else:
                x = block(x)
        x = backbone.norm(x)
        return x[:, 0]

    def forward(self, backbone: nn.Module, inputs: torch.Tensor, expert_ids: torch.Tensor) -> torch.Tensor:
        x = backbone.patch_embed(inputs)
        B, N, D = x.size()
        cls_token = backbone.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = backbone.pos_drop(x + backbone.pos_embed)
        valid_mask = expert_ids >= 0
        if valid_mask.all():
            output = torch.zeros(B, D, device=x.device, dtype=x.dtype)
            for eid in expert_ids.unique().tolist():
                idxs = (expert_ids == eid).nonzero(as_tuple=True)[0]
                if idxs.numel() == 0:
                    continue
                cls_feats = self._forward_with_adapter_layers(backbone, x[idxs], int(eid))
                output[idxs] = cls_feats
            return output
        elif not valid_mask.any():
            x = backbone.blocks(x)
            x = backbone.norm(x)
            return x[:, 0]
        else:
            output = torch.zeros(B, D, device=x.device, dtype=x.dtype)
            invalid_idx = (~valid_mask).nonzero(as_tuple=True)[0]
            if invalid_idx.numel() > 0:
                xi = x[invalid_idx]
                xi = backbone.blocks(xi)
                xi = backbone.norm(xi)
                output[invalid_idx] = xi[:, 0]
            valid_idx = valid_mask.nonzero(as_tuple=True)[0]
            valid_ids = expert_ids[valid_idx]
            for eid in valid_ids.unique().tolist():
                idxs = valid_idx[valid_ids == eid]
                if idxs.numel() == 0:
                    continue
                cls_feats = self._forward_with_adapter_layers(backbone, x[idxs], int(eid))
                output[idxs] = cls_feats
            return output


class HiDeAdapterModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        task_num: int,
        pretrained: bool = True,
        num_adapter_layers: int = 5,
        sdlora_rank: int = 4,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.task_num = task_num
        assert hasattr(custom_vit, backbone_name), f"Unsupported backbone {backbone_name} for HiDe-Adapter"
        self.backbone = getattr(custom_vit, backbone_name)(pretrained=pretrained, num_classes=num_classes)
        for _, p in self.backbone.named_parameters():
            p.requires_grad = False
        D = self.backbone.num_features
        self.feature_dim = D
        adapter_down_dim = 2 * sdlora_rank
        self.adapter_expert = AdapterExpert(
            num_experts=task_num,
            embed_dim=D,
            num_adapter_layers=num_adapter_layers,
            adapter_down_dim=adapter_down_dim,
            adapter_dropout=0.0,
        )
        self.prompt_head = nn.Linear(D, num_classes)
        self.g_mlp = nn.Sequential(nn.Linear(D, D), nn.ReLU(inplace=True))
        self.g_head = nn.Linear(D, num_classes)
        self.register_buffer("p_count", torch.zeros(num_classes))
        self.register_buffer("p_sum", torch.zeros(num_classes, D))
        self.register_buffer("p_sum_xxT", torch.zeros(num_classes, D, D))
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
        feat = self.adapter_expert(self.backbone, x, tids)
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

