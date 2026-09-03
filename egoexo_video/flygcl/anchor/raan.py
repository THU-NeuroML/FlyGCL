from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class RAANEncoder(nn.Module):
    """Relation-aware attentive aggregation over a variable-length feature sequence."""

    def __init__(self, input_dim: int = 1024, hidden_dim: int = 256, num_filters: int = 3):
        super().__init__()
        self.input_dim = int(input_dim)
        self.num_filters = int(num_filters)
        self.attention = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.input_dim, int(hidden_dim)),
                    nn.ReLU(),
                    nn.Linear(int(hidden_dim), 1),
                )
                for _ in range(self.num_filters)
            ]
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if features.ndim != 3 or features.shape[-1] != self.input_dim:
            raise ValueError(f"RAAN expects [B,T,{self.input_dim}], got {tuple(features.shape)}")
        weights = torch.stack(
            [torch.softmax(branch(features), dim=1) for branch in self.attention], dim=1
        )  # [B,F,T,1]
        pooled = (features.unsqueeze(1) * weights).sum(dim=2)  # [B,F,D]
        return pooled, weights


class RAAN(nn.Module):
    def __init__(
        self,
        output_dim: int,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        num_filters: int = 3,
    ):
        super().__init__()
        self.encoder = RAANEncoder(input_dim, hidden_dim, num_filters)
        self.head = nn.Linear(input_dim, int(output_dim))

    def forward(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        branch_features, attention = self.encoder(features)
        branch_outputs = self.head(branch_features)
        return {
            "output": branch_outputs.mean(dim=1),
            "embedding": branch_features.mean(dim=1),
            "branch_features": branch_features,
            "attention": attention,
        }


def ranking_loss(score_better: torch.Tensor, score_worse: torch.Tensor, margin: float = 1.0) -> torch.Tensor:
    target = torch.ones_like(score_better)
    return F.margin_ranking_loss(score_better, score_worse, target, margin=float(margin))


def exo_triplet_loss(
    better_embedding: torch.Tensor,
    worse_embedding: torch.Tensor,
    exo_embedding: torch.Tensor,
    margin: Optional[float] = None,
) -> torch.Tensor:
    if margin is not None:
        return F.triplet_margin_loss(exo_embedding, better_embedding, worse_embedding, margin=float(margin))
    positive_distance = torch.linalg.vector_norm(exo_embedding - better_embedding, dim=1)
    negative_distance = torch.linalg.vector_norm(exo_embedding - worse_embedding, dim=1)
    return F.soft_margin_loss(negative_distance - positive_distance, torch.ones_like(positive_distance))
