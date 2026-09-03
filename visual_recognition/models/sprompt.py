import logging
from typing import Iterable

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

import models.vit as vit
from .flyprompt import Prompt, RPFC

logger = logging.getLogger()


class SPrompt(nn.Module):
    def __init__(
        self,
        task_num: int = 10,
        num_classes: int = 100,
        backbone_name: str = None,
        len_prompt: int = 20,
        pos_prompt: Iterable[int] = (0, 1, 2, 3, 4),
        **kwargs,
    ):
        super().__init__()

        self.kwargs = kwargs
        self.task_num = task_num
        self.num_classes = num_classes
        self.len_prompt = len_prompt
        self.pos_prompt = pos_prompt

        self.task_count = 0

        # Backbone (same as FlyPrompt)
        assert backbone_name is not None, "backbone_name must be specified"
        if hasattr(vit, backbone_name):
            logger.info(f"Using custom ViT model: {backbone_name}")
            self.add_module(
                "backbone",
                getattr(vit, backbone_name)(pretrained=True, num_classes=num_classes),
            )
        else:
            logger.info(f"Using timm model: {backbone_name}")
            self.add_module(
                "backbone",
                timm.create_model(backbone_name, pretrained=True, num_classes=num_classes),
            )

        self.embed_dim = self.backbone.num_features
        for name, param in self.backbone.named_parameters():
            param.requires_grad = False
        self.backbone.fc.weight.requires_grad = True
        self.backbone.fc.bias.requires_grad = True

        # Expert prompts (identical structure to FlyPrompt)
        self.experts = Prompt(
            num_experts=self.task_num,
            len_prompt=self.len_prompt,
            embed_dim=self.embed_dim,
            pos_prompt=self.pos_prompt,
        )

        # RPFC gating head (optional, FlyPrompt-style)
        self.use_rp_gate = kwargs.get("use_rp_gate", False)
        self.rp_dim = kwargs.get("rp_dim", 0)
        self.rp_ridge = kwargs.get("rp_ridge", 1e4)
        if self.use_rp_gate:
            self.rp_head = RPFC(
                M=self.rp_dim,
                ridge=self.rp_ridge,
                embed_dim=self.embed_dim,
                num_classes=self.task_num,
            )
        else:
            self.rp_head = None

        # EMA-based classifier head bank (optional, per-task experts)
        self.use_ema_head = self.kwargs.get("use_ema_head", False)
        ema_ratio = self.kwargs.get("ema_ratio", [0.9, 0.99])
        if isinstance(ema_ratio, (float, int)):
            ema_ratio = [float(ema_ratio)]
        self.ema_ratio = [float(r) for r in ema_ratio]
        self.num_ema = len(self.ema_ratio) if self.use_ema_head else 0

        if self.use_ema_head and self.num_ema > 0:
            # one EMA head bank per task (expert)
            self.experts_fc = nn.ModuleList(
                [
                    nn.ModuleList(
                        [nn.Linear(self.embed_dim, self.num_classes, bias=True) for _ in range(self.num_ema)]
                    )
                    for _ in range(self.task_num)
                ]
            )
            for expert_fc in self.experts_fc:
                for fc in expert_fc:
                    for param in fc.parameters():
                        param.requires_grad = False
            # initialize EMA heads for the first task from the online classifier
            self.init_fc(expert_id=0)
        else:
            self.experts_fc = None

    def forward_with_ema(self, inputs: torch.Tensor, expert_ids: torch.Tensor = None, **kwargs):
        """Forward with online head + EMA heads.

        Returns a list of logits: [online, ema_1, ema_2, ...].
        """
        if expert_ids is None:
            expert_ids = torch.full(
                (inputs.size(0),),
                self.task_count,
                device=inputs.device,
                dtype=torch.long,
            )
        x = self.experts(self.backbone, inputs, expert_ids)

        outputs_ls = []
        # online head
        outputs_ls.append(self.backbone.fc(x))

        # EMA heads (if enabled)
        if not self.use_ema_head or self.experts_fc is None or self.num_ema == 0:
            return outputs_ls

        for i in range(self.num_ema):
            outputs = []
            for x_i, e_i in zip(x, expert_ids):
                e_idx = int(e_i.item())
                if e_idx < 0 or e_idx >= self.task_num:
                    e_idx = max(0, min(self.task_num - 1, e_idx))
                outputs.append(self.experts_fc[e_idx][i](x_i))
            outputs_ls.append(torch.stack(outputs, dim=0))

        return outputs_ls

    @torch.no_grad()
    def init_fc(self, expert_id: int = None):
        """Initialize EMA heads for a given expert from the online classifier."""
        if not self.use_ema_head or self.experts_fc is None or self.num_ema == 0:
            return
        if expert_id is None:
            expert_id = self.task_count
        if expert_id < 0 or expert_id >= self.task_num:
            return
        w = self.backbone.fc.weight.data
        b = self.backbone.fc.bias.data
        for i in range(self.num_ema):
            self.experts_fc[expert_id][i].weight.data.copy_(w)
            self.experts_fc[expert_id][i].bias.data.copy_(b)

    @torch.no_grad()
    def update_ema_fc(self, expert_id: int = None):
        """EMA update for classifier heads of a given expert."""
        if not self.use_ema_head or self.experts_fc is None or self.num_ema == 0:
            return
        if expert_id is None:
            expert_id = self.task_count
        if expert_id < 0 or expert_id >= self.task_num:
            return

        online_w = self.backbone.fc.weight.data
        online_b = self.backbone.fc.bias.data
        for i, ratio in enumerate(self.ema_ratio):
            ema_w = self.experts_fc[expert_id][i].weight.data
            ema_b = self.experts_fc[expert_id][i].bias.data
            ema_w.mul_(ratio).add_(online_w, alpha=1.0 - ratio)
            ema_b.mul_(ratio).add_(online_b, alpha=1.0 - ratio)


    def forward(self, inputs: torch.Tensor, expert_ids: torch.Tensor = None, **kwargs):
        if expert_ids is None:
            expert_ids = torch.full(
                (inputs.size(0),),
                self.task_count,
                device=inputs.device,
                dtype=torch.long,
            )
        x = self.experts(self.backbone, inputs, expert_ids)
        x = self.backbone.fc(x)
        return x

    def loss_fn(self, output, target):
        return F.cross_entropy(output, target)

    @torch.no_grad()
    def forward_with_rp(self, inputs: torch.Tensor) -> torch.Tensor:
        """Use RPFC head to predict task ids from CLS features."""
        if self.rp_head is None:
            raise RuntimeError("RPFC gating is disabled (use_rp_gate=False).")
        features = self.backbone.forward_features(inputs)
        if isinstance(features, (list, tuple)):
            features = features[0]
        cls_feat = features[:, 0]
        logits = self.rp_head(cls_feat)
        return logits

    @torch.no_grad()
    def collect(self, inputs: torch.Tensor, labels: torch.Tensor):
        """Collect features for RPFC training.

        Labels are ignored and replaced with current task_count to learn task-id gating.
        """
        if self.rp_head is None:
            return
        features = self.backbone.forward_features(inputs)
        if isinstance(features, (list, tuple)):
            features = features[0]
        cls_feat = features[:, 0]
        task_labels = torch.full(
            (labels.size(0),),
            self.task_count,
            device=labels.device,
            dtype=torch.long,
        )
        self.rp_head.collect(cls_feat, task_labels)

    @torch.no_grad()
    def update(self):
        """Solve the closed-form ridge regression for RPFC based on collected stats."""
        if self.rp_head is not None:
            self.rp_head.update()

    @torch.no_grad()
    def process_task_count(self):
        # Update RPFC weights with all collected data so far before moving to next task
        if self.rp_head is not None:
            self.rp_head.update()
        self.task_count += 1
        self.experts.init_new_expert(self.task_count)
        # Initialize EMA heads for the new expert, if enabled
        if self.use_ema_head and self.experts_fc is not None and self.num_ema > 0:
            self.init_fc(expert_id=self.task_count)

