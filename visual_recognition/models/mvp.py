import logging
from typing import Iterable

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.vit as vit

from models.flyprompt import RPFC


logger = logging.getLogger()


class MVP(nn.Module):
    def __init__(self,
                 pos_g_prompt   : Iterable[int] = (0,1),
                 len_g_prompt   : int   = 5 ,
                 pos_e_prompt   : Iterable[int] = (2,3,4),
                 len_e_prompt   : int   = 20,
                 g_pool         : int   = 1,
                 e_pool         : int   = 10,
                 selection_size : int   = 1,
                 prompt_func    : str   = 'prompt_tuning',
                 task_num       : int   = 10,
                 num_classes    : int   = 100,
                 lambd          : float = 1.0,
                 use_mask       : bool  = True,
                 use_contrastiv : bool  = True,
                 use_last_layer : bool  = False,
                 backbone_name  : str   = None,
                 **kwargs):

        super().__init__()

        self.lambd       = lambd
        self.kwargs      = kwargs
        self.task_num    = task_num
        self.use_mask    = use_mask
        self.num_classes = num_classes
        self.use_contrastiv  = use_contrastiv
        self.use_last_layer  = use_last_layer
        self.selection_size  = selection_size

        self.task_count = 0

        # Backbone
        assert backbone_name is not None, 'backbone_name must be specified'
        # Use custom ViT model from models.vit to support local .npz loading
        if hasattr(vit, backbone_name):
            logger.info(f'Using custom ViT model: {backbone_name}')
            self.add_module('backbone', getattr(vit, backbone_name)(pretrained=True, num_classes=num_classes))
        else:
            logger.info(f'Using timm model: {backbone_name}')
            self.add_module('backbone', timm.create_model(backbone_name, pretrained=True, num_classes=num_classes))
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
        self.backbone.fc.weight.requires_grad = True
        self.backbone.fc.bias.requires_grad = True

        # RPFC-based task gating (optional, FlyPrompt-style)
        self.use_rp_gate = kwargs.get("use_rp_gate", False)
        rp_dim = kwargs.get("rp_dim", 0)
        rp_ridge = kwargs.get("rp_ridge", 1e4)
        if self.use_rp_gate:
            self.rp_head = RPFC(
                M=rp_dim,
                ridge=rp_ridge,
                embed_dim=self.backbone.num_features,
                num_classes=task_num,
            )
        else:
            self.rp_head = None

        # EMA-based classifier head bank (optional, per-prompt-slot experts)
        self.use_ema_head = kwargs.get("use_ema_head", False)
        ema_ratio = kwargs.get("ema_ratio", [0.9, 0.99])
        if isinstance(ema_ratio, (float, int)):
            ema_ratio = [float(ema_ratio)]
        self.ema_ratio = [float(r) for r in ema_ratio]
        self.num_ema = len(self.ema_ratio) if self.use_ema_head else 0
        self.last_expert_ids = None
        self.experts_fc = None

        # Prompt
        self.g_pool = g_pool
        self.e_pool = e_pool = task_num
        self.len_g_prompt = len_g_prompt
        self.len_e_prompt = len_e_prompt
        self.g_length = len(pos_g_prompt) if pos_g_prompt else 0
        self.e_length = len(pos_e_prompt) if pos_e_prompt else 0

        self.register_buffer('pos_g_prompt', torch.tensor(pos_g_prompt, dtype=torch.int64))
        self.register_buffer('pos_e_prompt', torch.tensor(pos_e_prompt, dtype=torch.int64))
        self.register_buffer('similarity', torch.zeros(1))

        self.register_buffer('count', torch.zeros(e_pool))
        self.learnable_key  = nn.Parameter(torch.randn(e_pool, self.backbone.embed_dim))
        self.learnable_mask = nn.Parameter(torch.zeros(e_pool, self.num_classes) - 1)

        if prompt_func == 'prompt_tuning':
            self.prompt_func = self.prompt_tuning
            self.g_size = 1 * self.g_length * self.len_g_prompt
            self.e_size = 1 * self.e_length * self.len_e_prompt
            self.g_prompts = nn.Parameter(torch.randn(g_pool, self.g_size, self.backbone.embed_dim))
            self.e_prompts = nn.Parameter(torch.randn(e_pool, self.e_size, self.backbone.embed_dim))

        elif prompt_func == 'prefix_tuning':
            self.prompt_func = self.prefix_tuning
            self.g_size = 2 * self.g_length * self.len_g_prompt
            self.e_size = 2 * self.e_length * self.len_e_prompt
            self.g_prompts = nn.Parameter(torch.randn(g_pool, self.g_size, self.backbone.embed_dim))
            self.e_prompts = nn.Parameter(torch.randn(e_pool, self.e_size, self.backbone.embed_dim))

        # Initialize EMA heads after prompt pool is defined.
        if self.use_ema_head and self.num_ema > 0 and self.e_pool > 0:
            in_dim = getattr(self.backbone.fc, "in_features", self.backbone.num_features)
            self.experts_fc = nn.ModuleList(
                [
                    nn.ModuleList(
                        [nn.Linear(in_dim, self.num_classes, bias=True) for _ in range(self.num_ema)]
                    )
                    for _ in range(self.e_pool)
                ]
            )
            for expert_fc in self.experts_fc:
                for fc in expert_fc:
                    for param in fc.parameters():
                        param.requires_grad = False
            self.init_fc()

    def prompt_tuning(self,
                      x        : torch.Tensor,
                      g_prompt : torch.Tensor,
                      e_prompt : torch.Tensor,
                      **kwargs):

        B, N, C = x.size()
        g_prompt = g_prompt.contiguous().view(B, -1, self.len_g_prompt, C)
        e_prompt = e_prompt.contiguous().view(B, -1, self.len_e_prompt, C)

        for n, block in enumerate(self.backbone.blocks):
            pos_g = ((self.pos_g_prompt.eq(n)).nonzero()).squeeze()
            if pos_g.numel() != 0:
                x = torch.cat((x, g_prompt[:, pos_g]), dim = 1)

            pos_e = ((self.pos_e_prompt.eq(n)).nonzero()).squeeze()
            if pos_e.numel() != 0:
                x = torch.cat((x, e_prompt[:, pos_e]), dim = 1)
            x = block(x)
            x = x[:, :N, :]
        return x

    def prefix_tuning(self,
                      x        : torch.Tensor,
                      g_prompt : torch.Tensor,
                      e_prompt : torch.Tensor,
                      **kwargs):

        B, N, C = x.size()
        g_prompt = g_prompt.contiguous().view(B, -1, self.len_g_prompt, C)
        e_prompt = e_prompt.contiguous().view(B, -1, self.len_e_prompt, C)

        for n, block in enumerate(self.backbone.blocks):
            xq = block.norm1(x)
            xk = xq.clone()
            xv = xq.clone()

            pos_g = ((self.pos_g_prompt.eq(n)).nonzero()).squeeze()
            if pos_g.numel() != 0:
                xk = torch.cat((xk, g_prompt[:, pos_g * 2 + 0].clone()), dim = 1)
                xv = torch.cat((xv, g_prompt[:, pos_g * 2 + 1].clone()), dim = 1)

            pos_e = ((self.pos_e_prompt.eq(n)).nonzero()).squeeze()
            if pos_e.numel() != 0:
                xk = torch.cat((xk, e_prompt[:, pos_e * 2 + 0].clone()), dim = 1)
                xv = torch.cat((xv, e_prompt[:, pos_e * 2 + 1].clone()), dim = 1)

            attn   = block.attn
            weight = attn.qkv.weight
            bias   = attn.qkv.bias

            B, N, C = xq.shape
            xq = F.linear(xq, weight[:C   ,:], bias[:C   ]).reshape(B,  N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
            _B, _N, _C = xk.shape
            xk = F.linear(xk, weight[C:2*C,:], bias[C:2*C]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)
            _B, _N, _C = xv.shape
            xv = F.linear(xv, weight[2*C: ,:], bias[2*C: ]).reshape(B, _N, attn.num_heads, C // attn.num_heads).permute(0, 2, 1, 3)

            attention = (xq @ xk.transpose(-2, -1)) * attn.scale
            attention = attention.softmax(dim=-1)
            attention = attn.attn_drop(attention)

            attention = (attention @ xv).transpose(1, 2).reshape(B, N, C)
            attention = attn.proj(attention)
            attention = attn.proj_drop(attention)

            x = x + block.drop_path1(block.ls1(attention))
            x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
        return x

    def forward_features(self, inputs: torch.Tensor, **kwargs) -> torch.Tensor:
        self.backbone.eval()
        with torch.no_grad():
            x = self.backbone.patch_embed(inputs)
            B, N, D = x.size()

            cls_token = self.backbone.cls_token.expand(B, -1, -1)
            token_appended = torch.cat((cls_token, x), dim=1)
            x = self.backbone.pos_drop(token_appended + self.backbone.pos_embed)
            query = x.clone()
            for n, block in enumerate(self.backbone.blocks):
                if n == len(self.backbone.blocks) - 1 and not self.use_last_layer:
                    break
                query = block(query)
            query = query[:, 0]

        distance = 1 - F.cosine_similarity(query.unsqueeze(1), self.learnable_key, dim=-1)
        if self.use_contrastiv:
            mass = self.count + 1
        else:
            mass = 1.0
        scaled_distance = distance * mass

        # Gating: use RPFC at evaluation if enabled, otherwise use original MVP key routing.
        if getattr(self, "use_rp_gate", False) and (not self.training) and (self.rp_head is not None):
            # Only consider tasks that have been seen so far
            E = min(self.task_count + 1, self.task_num)
            logits = self.rp_head(query)  # [B, task_num]
            logits = logits[:, :E]
            topk = torch.topk(logits, self.selection_size, dim=1, largest=True)[1]
        else:
            topk = scaled_distance.topk(self.selection_size, dim=1, largest=False)[1]

        # Record last expert ids (top-1 prompt slot per sample) for EMA experts.
        if self.use_ema_head and self.num_ema > 0:
            self.last_expert_ids = topk[:, 0].detach()
        else:
            self.last_expert_ids = None

        distance_top = distance[
            torch.arange(topk.size(0), device=topk.device).unsqueeze(1).repeat(1, self.selection_size), topk
        ].squeeze().clone()
        e_prompts = self.e_prompts[topk].squeeze().clone()
        mask = self.learnable_mask[topk].mean(1).squeeze().clone()

        if self.use_contrastiv:
            key_wise_distance = 1 - F.cosine_similarity(self.learnable_key.unsqueeze(1), self.learnable_key, dim=-1)
            self.similarity_loss = -(
                (key_wise_distance[topk] / mass[topk]).exp().mean()
                / ((distance_top / mass[topk]).exp().mean() + (key_wise_distance[topk] / mass[topk]).exp().mean())
                + 1e-6
            ).log()
        else:
            self.similarity_loss = distance_top.mean()

        g_prompts = self.g_prompts[0].repeat(B, 1, 1)
        if self.training:
            with torch.no_grad():
                num = topk.view(-1).bincount(minlength=self.e_prompts.size(0))
                self.count += num

        x = self.prompt_func(self.backbone.pos_drop(token_appended + self.backbone.pos_embed), g_prompts, e_prompts)
        feature = self.backbone.norm(x)[:, 0]
        mask = torch.sigmoid(mask) * 2.0
        return feature, mask

    def forward_head(self, feature : torch.Tensor, **kwargs) -> torch.Tensor:
        x = self.backbone.fc_norm(feature)
        x = self.backbone.fc(x)
        return x

    def forward(self, inputs : torch.Tensor, **kwargs) -> torch.Tensor:
        x, mask = self.forward_features(inputs, **kwargs)
        x = self.forward_head(x, **kwargs)
        if self.use_mask:
            x = x * mask
        return x

    def loss_fn(self, output, target):
        return F.cross_entropy(output, target) + self.similarity_loss

    def get_similarity_loss(self):
        return self.similarity_loss

    def get_e_prompt_count(self):
        return self.count

    def forward_with_ema(self, inputs: torch.Tensor) -> list:
        """Forward with online head and EMA classifier heads.

        Returns a list of logits [online, ema_1, ema_2, ...] before adding
        the global class mask (self.mask). EMA heads are indexed by
        per-prompt-slot expert ids (top-1 from the prompt pool).
        """
        feature, mask = self.forward_features(inputs)

        # Online head
        feat_norm = self.backbone.fc_norm(feature)
        logit_online = self.backbone.fc(feat_norm)
        if self.use_mask:
            logit_online = logit_online * mask
        outputs_ls = [logit_online]

        if not self.use_ema_head or self.experts_fc is None or self.num_ema == 0:
            return outputs_ls

        expert_ids = getattr(self, "last_expert_ids", None)
        if expert_ids is None:
            return outputs_ls

        expert_ids = expert_ids.to(feature.device).long()

        for i in range(self.num_ema):
            outputs = []
            for feat_i, e_i in zip(feat_norm, expert_ids):
                e_idx = int(e_i.item())
                if e_idx < 0 or e_idx >= self.e_pool:
                    e_idx = max(0, min(self.e_pool - 1, e_idx))
                outputs.append(self.experts_fc[e_idx][i](feat_i))
            ema_logit = torch.stack(outputs, dim=0)
            if self.use_mask:
                ema_logit = ema_logit * mask
            outputs_ls.append(ema_logit)

        return outputs_ls

    @torch.no_grad()
    def init_fc(self, expert_ids: torch.Tensor = None) -> None:
        """Initialize EMA classifier heads from the online classifier.

        If expert_ids is None, initialize all experts; otherwise only
        initialize the EMA heads for the specified expert indices.
        """
        if not self.use_ema_head or self.experts_fc is None or self.num_ema == 0:
            return

        src_weight = self.backbone.fc.weight.data
        src_bias = self.backbone.fc.bias.data

        if expert_ids is None:
            indices = range(self.e_pool)
        else:
            if isinstance(expert_ids, torch.Tensor):
                indices = [int(i) for i in expert_ids.detach().cpu().tolist()]
            else:
                indices = [int(i) for i in expert_ids]

        for e_idx in indices:
            if e_idx < 0 or e_idx >= self.e_pool:
                continue
            expert_fc = self.experts_fc[e_idx]
            for fc in expert_fc:
                fc.weight.data.copy_(src_weight)
                fc.bias.data.copy_(src_bias)

    @torch.no_grad()
    def update_ema_fc(self, expert_ids: torch.Tensor) -> None:
        """EMA update for classifier heads of the given experts.

        Should be called after optimizer.step() with the expert ids of
        the current batch (top-1 prompt slot indices).
        """
        if not self.use_ema_head or self.experts_fc is None or self.num_ema == 0:
            return
        if expert_ids is None:
            return

        if isinstance(expert_ids, torch.Tensor):
            ids = expert_ids.detach().view(-1).cpu().tolist()
        else:
            ids = list(expert_ids)
        unique_ids = sorted(set(int(i) for i in ids))

        online_w = self.backbone.fc.weight.data
        online_b = self.backbone.fc.bias.data

        for e_idx in unique_ids:
            if e_idx < 0 or e_idx >= self.e_pool:
                continue
            expert_fc = self.experts_fc[e_idx]
            for i, ratio in enumerate(self.ema_ratio):
                ema_w = expert_fc[i].weight.data
                ema_b = expert_fc[i].bias.data
                ema_w.mul_(ratio).add_(online_w, alpha=1.0 - ratio)
                ema_b.mul_(ratio).add_(online_b, alpha=1.0 - ratio)


    def process_task_count(self):
        self.task_count += 1