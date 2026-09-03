from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualAdapter(nn.Module):
    def __init__(self, dimension: int, bottleneck: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dimension),
            nn.Linear(dimension, bottleneck),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(bottleneck, dimension),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x):
        return x + self.net(x)


class FlyGCLAnticipation(nn.Module):
    """Shared temporal expert with view adapters, prototypes, and task prompts."""

    def __init__(self, input_dim: int, hidden_dim: int, classes: int, sessions: int, dropout: float):
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        layer = nn.TransformerEncoderLayer(
            hidden_dim,
            nhead=4,
            dim_feedforward=4 * hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        # norm_first disables PyTorch's nested-tensor fast path already; make
        # that explicit to avoid a misleading warning at every target launch.
        self.temporal = nn.TransformerEncoder(
            layer, num_layers=2, enable_nested_tensor=False
        )
        self.positional = nn.Parameter(torch.zeros(1, 32, hidden_dim))
        # ParameterList keeps old session prompts out of the current graph, so
        # AdamW cannot decay them while a later session is trained.
        self.prompts = nn.ParameterList(
            [nn.Parameter(torch.zeros(1, hidden_dim)) for _ in range(sessions)]
        )
        self.view_adapters = nn.ModuleList(
            [ResidualAdapter(hidden_dim, hidden_dim // 2, dropout) for _ in range(2)]
        )
        self.shared_head = nn.Linear(hidden_dim, classes)
        self.view_heads = nn.ModuleList([nn.Linear(hidden_dim, classes) for _ in range(2)])
        self.register_buffer("prototypes", torch.zeros(classes, hidden_dim))
        self.register_buffer("prototype_counts", torch.zeros(classes))

    def encode(self, feature: torch.Tensor, view: torch.Tensor, session: int):
        x = self.input_projection(feature)
        x = x + self.positional[:, : x.shape[1]] + self.prompts[session]
        x = self.temporal(x).mean(1)
        output = torch.empty_like(x)
        for view_id in (0, 1):
            indices = torch.nonzero(view == view_id, as_tuple=False).flatten()
            if indices.numel():
                output[indices] = self.view_adapters[view_id](x.index_select(0, indices))
        return F.normalize(output, dim=-1)

    def forward(
        self,
        feature: torch.Tensor,
        view: torch.Tensor,
        session: int,
        use_view_expert: bool = True,
    ):
        embedding = self.encode(feature, view, session)
        shared = self.shared_head(embedding)
        expert = torch.empty_like(shared)
        for view_id in (0, 1):
            indices = torch.nonzero(view == view_id, as_tuple=False).flatten()
            if indices.numel():
                expert[indices] = self.view_heads[view_id](embedding.index_select(0, indices))
        # A view expert is valid only when that view occurs in training. For
        # cross-view evaluation of ego-only/exo-only runs, the shared head is
        # the learned zero-shot predictor and the untouched random head must
        # not dilute it.
        logits = 0.5 * (shared + expert) if use_view_expert else shared
        prototype_logits = F.normalize(self.prototypes, dim=-1) @ embedding.transpose(0, 1)
        prototype_logits = prototype_logits.transpose(0, 1)
        valid = self.prototype_counts.gt(0)
        # Missing prototypes contribute no evidence; they must not implicitly
        # remove unseen classes from the full Top-5 label space.
        prototype_logits[:, ~valid] = 0.0
        return {
            "logits": logits,
            "shared_logits": shared,
            "expert_logits": expert,
            "prototype_logits": prototype_logits,
            "embedding": embedding,
        }

    @torch.no_grad()
    def update_prototypes(self, embedding: torch.Tensor, labels: torch.Tensor, momentum: float = 0.9):
        for class_id in torch.nonzero(labels.sum(0) > 0, as_tuple=False).flatten():
            selected = embedding[labels[:, class_id] > 0]
            centroid = selected.mean(0)
            index = int(class_id)
            if self.prototype_counts[index] == 0:
                self.prototypes[index] = centroid
            else:
                self.prototypes[index].mul_(momentum).add_(centroid, alpha=1.0 - momentum)
            self.prototype_counts[index] += selected.shape[0]
