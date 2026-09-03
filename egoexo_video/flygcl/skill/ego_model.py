from __future__ import annotations

from copy import deepcopy
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalExpert(nn.Module):
    """One temporal scale in the ego-only mixture of experts."""

    def __init__(self, hidden_dim: int, kernel: int, dilation: int, dropout: float):
        super().__init__()
        padding = dilation * (kernel - 1) // 2
        self.temporal = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=kernel,
            padding=padding,
            dilation=dilation,
            groups=hidden_dim,
        )
        self.channel = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        temporal = self.temporal(tokens.transpose(1, 2)).transpose(1, 2)
        return self.norm(tokens + temporal + self.channel(tokens + temporal))


class EgoTemporalMoE(nn.Module):
    """Multi-scale temporal MoE with an antisymmetric pairwise score.

    Each video is scored independently, so swapping a skill pair reverses the
    margin exactly.  Four experts cover frame-local, short, medium and dilated
    long-range patterns.  A clip-dependent gate performs the MoE integration.
    """

    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 256,
        tokens: int = 10,
        dropout: float = 0.12,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.input = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.GELU(),
        )
        self.position = nn.Parameter(torch.zeros(1, int(tokens), self.hidden_dim))
        nn.init.trunc_normal_(self.position, std=0.02)
        self.experts = nn.ModuleList(
            [
                TemporalExpert(self.hidden_dim, 1, 1, dropout),
                TemporalExpert(self.hidden_dim, 3, 1, dropout),
                TemporalExpert(self.hidden_dim, 5, 1, dropout),
                TemporalExpert(self.hidden_dim, 3, 2, dropout),
            ]
        )
        self.gate = nn.Sequential(
            nn.LayerNorm(3 * self.hidden_dim),
            nn.Linear(3 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, len(self.experts)),
        )
        self.pool = nn.ModuleList(
            [nn.Linear(self.hidden_dim, 1) for _ in range(len(self.experts))]
        )
        self.score_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(4 * self.hidden_dim),
                    nn.Linear(4 * self.hidden_dim, self.hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(self.hidden_dim, 1),
                )
                for _ in range(len(self.experts))
            ]
        )
        self.residual_score = nn.Sequential(
            nn.LayerNorm(3 * self.hidden_dim),
            nn.Linear(3 * self.hidden_dim, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, 1),
        )

    def forward_video(self, features: torch.Tensor) -> Dict[str, torch.Tensor]:
        tokens = self.input(features) + self.position[:, : features.shape[1]]
        clip_statistics = torch.cat(
            (tokens.mean(1), tokens.std(1, unbiased=False), tokens[:, -1] - tokens[:, 0]),
            dim=-1,
        )
        gate = torch.softmax(self.gate(clip_statistics), dim=-1)
        summaries = []
        scores = []
        for expert, attention, head in zip(self.experts, self.pool, self.score_heads):
            encoded = expert(tokens)
            weights = torch.softmax(attention(encoded).squeeze(-1), dim=1)
            attended = (weights[..., None] * encoded).sum(1)
            summary = torch.cat(
                (
                    attended,
                    encoded.mean(1),
                    encoded.std(1, unbiased=False),
                    encoded[:, -1] - encoded[:, 0],
                ),
                dim=-1,
            )
            summaries.append(summary)
            scores.append(head(summary).squeeze(-1))
        expert_scores = torch.stack(scores, dim=-1)
        score = (gate * expert_scores).sum(-1) + 0.20 * self.residual_score(
            clip_statistics
        ).squeeze(-1)
        return {
            "score": score,
            "expert_scores": expert_scores,
            "gate": gate,
            "embedding": torch.stack(summaries, dim=1),
        }

    def forward_pair(
        self, better: torch.Tensor, worse: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        positive = self.forward_video(better)
        negative = self.forward_video(worse)
        return {
            "margin": positive["score"] - negative["score"],
            "score_better": positive["score"],
            "score_worse": negative["score"],
            "expert_margins": positive["expert_scores"] - negative["expert_scores"],
            "gate": 0.5 * (positive["gate"] + negative["gate"]),
        }


@torch.no_grad()
def update_ema(target: EgoTemporalMoE, source: EgoTemporalMoE, decay: float) -> None:
    target_state = target.state_dict()
    source_state = source.state_dict()
    for key, target_value in target_state.items():
        source_value = source_state[key]
        if torch.is_floating_point(target_value):
            target_value.mul_(float(decay)).add_(source_value, alpha=1.0 - float(decay))
        else:
            target_value.copy_(source_value)


def clone_frozen(model: EgoTemporalMoE) -> EgoTemporalMoE:
    result = deepcopy(model).eval()
    result.requires_grad_(False)
    return result


def ranking_objective(
    result: Dict[str, torch.Tensor],
    gap_target: torch.Tensor,
    margin: float = 0.30,
    temperature: float = 0.50,
) -> tuple[torch.Tensor, Dict[str, float]]:
    pair_margin = result["margin"]
    target = float(margin) + 0.10 * gap_target.clamp(0.0, 4.0)
    rank = F.softplus(-pair_margin / float(temperature)).mean()
    hinge = F.relu(target - pair_margin).mean()
    regression = F.smooth_l1_loss(pair_margin, gap_target.clamp(0.05, 4.0))
    expert_rank = F.softplus(
        -result["expert_margins"] / float(temperature)
    ).mean()
    mean_gate = result["gate"].mean(0)
    balance = ((mean_gate - 1.0 / result["gate"].shape[-1]) ** 2).mean()
    total = rank + 0.50 * hinge + 0.18 * regression + 0.20 * expert_rank + 0.02 * balance
    return total, {
        "loss": float(total.detach()),
        "rank": float(rank.detach()),
        "hinge": float(hinge.detach()),
        "regression": float(regression.detach()),
        "expert_rank": float(expert_rank.detach()),
        "balance": float(balance.detach()),
    }
