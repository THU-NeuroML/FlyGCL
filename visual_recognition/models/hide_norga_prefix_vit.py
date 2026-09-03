import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import vit as custom_vit
from .flyprompt import RPFC


class PrefixViTBackbone(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        task_num: int,
        prefix_len: int = 5,
        num_prefix_layers: int = 5,
        use_norga: bool = False,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        assert hasattr(custom_vit, backbone_name), f"Unsupported backbone {backbone_name} for prefix ViT"
        self.backbone = getattr(custom_vit, backbone_name)(pretrained=pretrained, num_classes=num_classes)
        self.task_num = task_num
        self.prefix_len = prefix_len
        self.num_prefix_layers = num_prefix_layers
        self.use_norga = use_norga

        self.embed_dim = self.backbone.num_features
        self.num_heads = self.backbone.blocks[0].attn.num_heads
        head_dim = self.embed_dim // self.num_heads
        k_shape = (num_prefix_layers, task_num, prefix_len, self.num_heads, head_dim)
        self.prefix_k = nn.Parameter(torch.zeros(k_shape))
        self.prefix_v = nn.Parameter(torch.zeros(k_shape))
        nn.init.trunc_normal_(self.prefix_k, std=0.02)
        nn.init.trunc_normal_(self.prefix_v, std=0.02)

        if self.use_norga:
            # act_scale: [num_layers, 2, 1, 1]
            self.act_scale = nn.Parameter(torch.ones(num_prefix_layers, 2, 1, 1))
            self.gate_act = nn.Sigmoid()
        else:
            self.act_scale = None
            self.gate_act = None

    @torch.no_grad()
    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def freeze_act_scale(self) -> None:
        if self.act_scale is not None:
            self.act_scale.requires_grad_(False)

    def _apply_prefix_attn(self, x_norm, block, layer_idx: int, task_ids: torch.Tensor) -> torch.Tensor:
        B, N, C = x_norm.shape
        attn = block.attn
        qkv = attn.qkv(x_norm).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        # prefix for current layer & task (per-sample)
        task_ids = task_ids.to(x_norm.device).long().clamp(0, self.task_num - 1)
        # prefix_k[layer_idx]: [T, L, H, D]
        k_layer = self.prefix_k[layer_idx]
        v_layer = self.prefix_v[layer_idx]
        k_pref = k_layer[task_ids]  # [B, L, H, D]
        v_pref = v_layer[task_ids]
        k_pref = k_pref.permute(0, 2, 1, 3)  # [B, H, L, D]
        v_pref = v_pref.permute(0, 2, 1, 3)
        k_cat = torch.cat([k_pref, k], dim=2)
        v_cat = torch.cat([v_pref, v], dim=2)
        scale = attn.scale
        attn_logits = (q @ k_cat.transpose(-2, -1)) * scale
        if self.use_norga and self.act_scale is not None:
            s = self.act_scale[layer_idx]
            # apply NoRGa gating only on prefix (first prefix_len keys) along last dim
            prompt_part = attn_logits[..., : self.prefix_len]
            base_part = attn_logits[..., self.prefix_len :]
            prompt_part = prompt_part + self.gate_act(prompt_part * s[0]) * s[1]
            attn_logits = torch.cat([prompt_part, base_part], dim=-1)
        attn_prob = attn_logits.softmax(dim=-1)
        attn_prob = attn.attn_drop(attn_prob)
        out = (attn_prob @ v_cat).transpose(1, 2).reshape(B, N, C)
        out = attn.proj(out)
        out = attn.proj_drop(out)
        return out

    def forward_features(self, x: torch.Tensor, use_prefix: bool, task_id: Optional[int] = None) -> torch.Tensor:
        b = self.backbone
        x = b.patch_embed(x)
        B, N, _ = x.shape
        cls_tokens = b.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + b.pos_embed
        x = b.pos_drop(x)
        task_ids_tensor = None
        if use_prefix and task_id is not None:
            if isinstance(task_id, int):
                task_ids_tensor = torch.full((B,), task_id, device=x.device, dtype=torch.long)
            elif isinstance(task_id, torch.Tensor):
                if task_id.dim() == 0:
                    task_ids_tensor = task_id.view(1).expand(B).to(x.device).long()
                else:
                    assert task_id.shape[0] == B, "task_id tensor must have shape [B]"
                    task_ids_tensor = task_id.to(x.device).long()
            else:
                raise TypeError("task_id must be int or 1D/0D tensor when use_prefix is True")
        for i, blk in enumerate(b.blocks):
            if use_prefix and i < self.num_prefix_layers and task_ids_tensor is not None:
                x_norm = blk.norm1(x)
                attn_out = self._apply_prefix_attn(x_norm, blk, i, task_ids_tensor)
                x = x + blk.drop_path1(blk.ls1(attn_out))
                x = x + blk.drop_path2(blk.ls2(blk.mlp(blk.norm2(x))))
            else:
                x = blk(x)
        x = b.norm(x)
        return x[:, 0]


class HiDePrefixModel(nn.Module):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        task_num: int,
        prefix_len: int = 5,
        num_prefix_layers: int = 5,
        pretrained: bool = True,
        **kwargs,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.task_num = task_num
        self.backbone = PrefixViTBackbone(
            backbone_name=backbone_name,
            num_classes=num_classes,
            task_num=task_num,
            prefix_len=prefix_len,
            num_prefix_layers=num_prefix_layers,
            use_norga=False,
            pretrained=pretrained,
        )
        # freeze all pre-trained ViT backbone parameters; only prefixes and heads are trainable
        if hasattr(self.backbone, "backbone"):
            for name, param in self.backbone.backbone.named_parameters():
                param.requires_grad = False
        D = self.backbone.embed_dim
        self.feature_dim = D
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

        # RPFC-based task gating (optional, FlyPrompt-style)
        self.use_rp_gate = kwargs.get("use_rp_gate", False)
        rp_dim = kwargs.get("rp_dim", 0)
        rp_ridge = kwargs.get("rp_ridge", 1e4)
        if self.use_rp_gate:
            self.rp_gate = RPFC(
                M=rp_dim,
                ridge=rp_ridge,
                embed_dim=D,
                num_classes=task_num,
            )
        else:
            self.rp_gate = None

        # EMA-based classifier head bank for prompt branch (optional, per-task experts)
        self.use_ema_head = kwargs.get("use_ema_head", False)
        ema_ratio = kwargs.get("ema_ratio", [0.9, 0.99])
        if isinstance(ema_ratio, (float, int)):
            ema_ratio = [float(ema_ratio)]
        self.ema_ratio = [float(r) for r in ema_ratio]
        self.num_ema = len(self.ema_ratio) if self.use_ema_head else 0

        if self.use_ema_head and self.num_ema > 0:
            self.experts_fc = nn.ModuleList(
                [
                    nn.ModuleList(
                        [nn.Linear(self.feature_dim, self.num_classes, bias=True) for _ in range(self.num_ema)]
                    )
                    for _ in range(self.task_num)
                ]
            )
            for expert_fc in self.experts_fc:
                for fc in expert_fc:
                    for param in fc.parameters():
                        param.requires_grad = False
            # initialize EMA heads for the first task from the online prompt head
            self.init_fc(expert_id=0)
        else:
            self.experts_fc = None


    def forward_prompt_with_ema(
        self,
        x: torch.Tensor,
        task_id: torch.Tensor,
    ) -> torch.Tensor:
        """Forward through prompt branch with online head + EMA heads.

        Returns a list of logits: [online, ema_1, ema_2, ...].
        """
        feat = self.backbone.forward_features(x, use_prefix=True, task_id=task_id)
        logit = self.prompt_head(feat)
        outputs_ls = [logit]

        if not self.use_ema_head or self.experts_fc is None or self.num_ema == 0:
            return outputs_ls

        # per-task EMA experts indexed by task_id
        task_ids_tensor = task_id.to(x.device).long().clamp(0, self.task_num - 1)
        for i in range(self.num_ema):
            outputs = []
            for feat_i, t_i in zip(feat, task_ids_tensor):
                e_idx = int(t_i.item())
                outputs.append(self.experts_fc[e_idx][i](feat_i))
            outputs_ls.append(torch.stack(outputs, dim=0))

        return outputs_ls

    @torch.no_grad()
    def init_fc(self, expert_id: int = 0) -> None:
        """Initialize EMA heads for a given expert from the prompt head.

        Expert id corresponds to task id.
        """
        if not self.use_ema_head or self.experts_fc is None or self.num_ema == 0:
            return
        if expert_id < 0 or expert_id >= self.task_num:
            return

        w = self.prompt_head.weight.data
        b = self.prompt_head.bias.data
        for i in range(self.num_ema):
            self.experts_fc[expert_id][i].weight.data.copy_(w)
            self.experts_fc[expert_id][i].bias.data.copy_(b)

    @torch.no_grad()
    def update_ema_fc(self, expert_id: int) -> None:
        """EMA update for classifier heads of a given expert (task).

        Should be called after optimizer.step() with the current task id.
        """
        if not self.use_ema_head or self.experts_fc is None or self.num_ema == 0:
            return
        if expert_id < 0 or expert_id >= self.task_num:
            return

        online_w = self.prompt_head.weight.data
        online_b = self.prompt_head.bias.data
        for i, ratio in enumerate(self.ema_ratio):
            ema_w = self.experts_fc[expert_id][i].weight.data
            ema_b = self.experts_fc[expert_id][i].bias.data
            ema_w.mul_(ratio).add_(online_w, alpha=1.0 - ratio)
            ema_b.mul_(ratio).add_(online_b, alpha=1.0 - ratio)

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

    def forward_prompt(
        self,
        x: torch.Tensor,
        task_id: int,
        labels: Optional[torch.Tensor] = None,
        update_stats: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone.forward_features(x, use_prefix=True, task_id=task_id)
        logit = self.prompt_head(feat)
        if update_stats and labels is not None:
            self._update_stats(feat.detach(), labels, branch="prompt")
        return logit, feat

    @torch.no_grad()
    def forward_with_rp(self, x: torch.Tensor) -> torch.Tensor:
        """Use RPFC head to predict task ids from gate features."""
        if self.rp_gate is None:
            raise RuntimeError("RPFC gating is disabled (use_rp_gate=False).")
        feat = self.backbone.forward_features(x, use_prefix=False, task_id=None)
        h = self.g_mlp(feat)
        logits = self.rp_gate(h)
        return logits

    @torch.no_grad()
    def collect_rp(self, x: torch.Tensor, task_labels: torch.Tensor) -> None:
        """Collect features for RPFC gating using gate branch features."""
        if self.rp_gate is None:
            return
        feat = self.backbone.forward_features(x, use_prefix=False, task_id=None)
        h = self.g_mlp(feat)
        self.rp_gate.collect(h, task_labels)

    @torch.no_grad()
    def update(self) -> None:
        """Update RPFC weights from collected statistics."""
        if self.rp_gate is not None:
            self.rp_gate.update()

    def forward_gate(
        self,
        x: torch.Tensor,
        detach_backbone: bool = True,
        labels: Optional[torch.Tensor] = None,
        update_stats: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        feat = self.backbone.forward_features(x, use_prefix=False, task_id=None)
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
            # use only diagonal covariance for numerical stability
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


class NoRGaPrefixModel(HiDePrefixModel):
    def __init__(
        self,
        backbone_name: str,
        num_classes: int,
        task_num: int,
        prefix_len: int = 5,
        num_prefix_layers: int = 5,
        pretrained: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            backbone_name=backbone_name,
            num_classes=num_classes,
            task_num=task_num,
            prefix_len=prefix_len,
            num_prefix_layers=num_prefix_layers,
            pretrained=pretrained,
            **kwargs,
        )
        # replace backbone with NoRGa-enabled one and keep ViT frozen
        self.backbone = PrefixViTBackbone(
            backbone_name=backbone_name,
            num_classes=num_classes,
            task_num=task_num,
            prefix_len=prefix_len,
            num_prefix_layers=num_prefix_layers,
            use_norga=True,
            pretrained=pretrained,
        )
        if hasattr(self.backbone, "backbone"):
            for name, param in self.backbone.backbone.named_parameters():
                param.requires_grad = False

    def freeze_act_scale(self) -> None:
        self.backbone.freeze_act_scale()

